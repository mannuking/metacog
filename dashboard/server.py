"""
Metacog training dashboard — stdlib-only HTTP server.
Serves a real-time live view of Phase 1 RL training.

Endpoints:
  GET  /                -> HTML dashboard (auto-refreshes via EventSource)
  GET  /api/snapshot    -> JSON snapshot of current state
  GET  /api/stream      -> Server-Sent Events: pushes snapshots as log grows
  GET  /api/log         -> Full log file as plain text
  GET  /api/checkpoints -> List of saved LoRA checkpoints

Run:
  python dashboard/server.py --port 7860
  Then open http://127.0.0.1:7860 in a browser.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

# --- Path resolution --------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "phase1"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# --- Log parsing ------------------------------------------------------------
# Lines look like:
#   [17:19:46]   INIT  acc=0.250  ECE=0.3765  avg_conf=0.624  abst=0.000
#   [17:27:07]   rollouts: 200, reward mean=0.202, std=0.550
#   [17:27:07]   outcomes: {'humble_wrong': 153, 'overconfident_wrong': 8, 'correct': 39}
#   [17:27:35]   loss=0.0000, step_time=468.5s
#   [17:17:12] Creating LoRA training client...
RE_TS_LINE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s+(.*)$")
RE_STEP_HEADER_BARE = re.compile(r"^---\s+step\s+(\d+)/(\d+)\s+---")
RE_INIT = re.compile(
    r"INIT\s+acc=(?P<acc>[\d.]+)\s+ECE=(?P<ece>[\d.]+)\s+avg_conf=(?P<conf>[\d.]+)\s+abst=(?P<abst>[\d.]+)"
)
RE_ROLLOUTS = re.compile(
    r"rollouts:\s*(?P<n>\d+),\s*reward mean=(?P<mean>[\-\d.]+),\s*std=(?P<std>[\-\d.]+)"
)
RE_OUTCOMES = re.compile(r"outcomes:\s*(?P<dict>\{.*\})")
RE_STEP_TIME = re.compile(r"step_time=(?P<t>[\d.]+)s")
RE_LOSS = re.compile(r"loss=(?P<l>[\-\d.]+)")
RE_STEP_HEADER = re.compile(r"^---\s+step\s+(\d+)/(\d+)\s+---")
RE_DATUMS = re.compile(r"training data:\s*(\d+)\s+datums")


def _parse_outcomes(s: str) -> dict:
    # Outcomes come as a python dict repr; convert it safely.
    s = s.replace("'", '"')
    try:
        return json.loads(s)
    except Exception:
        return {}


def parse_log(log_path: Path) -> dict:
    """Parse a phase1 log file into a structured snapshot."""
    if not log_path.exists():
        return {
            "found": False,
            "path": str(log_path),
            "error": "log file not found",
        }

    raw = log_path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()

    snapshot = {
        "found": True,
        "path": str(log_path),
        "modified": log_path.stat().st_mtime,
        "raw_lines": len(lines),
        "config": {},
        "steps": [],
        "init": None,
        "current_step": None,
        "total_steps": None,
        "log_tail": lines[-25:],
    }

    for ln in lines:
        m = RE_TS_LINE.match(ln)
        if m:
            ts, body = m.group(1), m.group(2)
        else:
            # step headers don't carry a timestamp prefix
            mb = RE_STEP_HEADER_BARE.match(ln)
            if mb:
                snapshot["current_step"] = int(mb.group(1))
                snapshot["total_steps"] = int(mb.group(2))
                snapshot["steps"].append({
                    "step_index": int(mb.group(1)),
                })
            continue

        # config lines
        if body.startswith("base_model:"):
            snapshot["config"]["base_model"] = body.split(":", 1)[1].strip()
        elif body.startswith("lora_rank:"):
            parts = [p.strip() for p in body.split(",")]
            for p in parts:
                if ":" in p:
                    k, v = p.split(":", 1)
                    snapshot["config"][k.strip()] = v.strip()
        elif body.startswith("n_steps:"):
            parts = [p.strip() for p in body.split(",")]
            for p in parts:
                if ":" in p:
                    k, v = p.split(":", 1)
                    snapshot["config"][k.strip()] = v.strip()
        elif body.startswith("loss_fn:"):
            parts = [p.strip() for p in body.split(",")]
            for p in parts:
                if ":" in p:
                    k, v = p.split(":", 1)
                    snapshot["config"][k.strip()] = v.strip()

        elif (mi := RE_INIT.search(body)):
            snapshot["init"] = {
                "timestamp": ts,
                "accuracy": float(mi.group("acc")),
                "ece": float(mi.group("ece")),
                "avg_confidence": float(mi.group("conf")),
                "abstention": float(mi.group("abst")),
            }

        elif (mr := RE_ROLLOUTS.search(body)):
            if snapshot["steps"]:
                step = snapshot["steps"][-1]
                step["timestamp"] = ts
                step["rollouts"] = int(mr.group("n"))
                step["reward_mean"] = float(mr.group("mean"))
                step["reward_std"] = float(mr.group("std"))

        elif (mo := RE_OUTCOMES.search(body)):
            if snapshot["steps"]:
                snapshot["steps"][-1]["outcomes"] = _parse_outcomes(mo.group("dict"))

        elif (md := RE_DATUMS.search(body)):
            if snapshot["steps"]:
                snapshot["steps"][-1]["training_datums"] = int(md.group(1))

        elif (ml := RE_LOSS.search(body)):
            if snapshot["steps"]:
                step = snapshot["steps"][-1]
                step["loss"] = float(ml.group("l"))
                if (mt := RE_STEP_TIME.search(body)):
                    step["step_time_s"] = float(mt.group("t"))

    # Compute derived metrics
    if snapshot["steps"]:
        snapshot["latest_step"] = snapshot["steps"][-1]
        s = snapshot["latest_step"]
        out = s.get("outcomes") or {}
        total = sum(out.values()) or 1
        s["accuracy_pct"] = round(100.0 * out.get("correct", 0) / total, 1)
        s["overconfident_pct"] = round(100.0 * out.get("overconfident_wrong", 0) / total, 1)
        s["humble_pct"] = round(100.0 * out.get("humble_wrong", 0) / total, 1)

        # Trend
        rewards = [st.get("reward_mean") for st in snapshot["steps"] if "reward_mean" in st]
        if len(rewards) >= 2:
            snapshot["reward_trend"] = round(rewards[-1] - rewards[0], 4)
        if len(snapshot["steps"]) >= 2:
            snapshot["acc_trend"] = round(
                snapshot["steps"][-1].get("accuracy_pct", 0)
                - snapshot["steps"][0].get("accuracy_pct", 0),
                1,
            )

    return snapshot


def list_checkpoints() -> list[dict]:
    """Find all checkpoint files in results/phase1/."""
    if not RESULTS_DIR.exists():
        return []
    out = []
    for p in sorted(RESULTS_DIR.glob("*.pt")) + sorted(RESULTS_DIR.glob("*.json")):
        try:
            st = p.stat()
            out.append({
                "name": p.name,
                "size_bytes": st.st_size,
                "size_human": _human(st.st_size),
                "modified": st.st_mtime,
                "modified_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
            })
        except OSError:
            pass
    return out


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def find_latest_log() -> Path | None:
    if not RESULTS_DIR.exists():
        return None
    logs = sorted(RESULTS_DIR.glob("phase1_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


# --- HTTP handler ------------------------------------------------------------
class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence default access logs
        sys.stderr.write(f"[dashboard] {self.address_string()} {fmt % args}\n")

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, body: str, status=200, ctype="text/plain; charset=utf-8"):
        b = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            return self._serve_index()
        if path == "/api/snapshot":
            log = find_latest_log()
            snap = parse_log(log) if log else {"found": False, "error": "no log yet"}
            snap["checkpoints"] = list_checkpoints()
            snap["server_time"] = time.time()
            return self._send_json(snap)
        if path == "/api/stream":
            return self._serve_sse()
        if path == "/api/log":
            log = find_latest_log()
            if not log:
                return self._send_text("(no log file yet)\n", status=404)
            return self._send_text(log.read_text(encoding="utf-8", errors="replace"))
        if path == "/api/checkpoints":
            return self._send_json(list_checkpoints())
        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            return self._serve_static(rel)
        return self._send_text("Not Found", status=404)

    def _serve_index(self):
        idx = STATIC_DIR / "index.html"
        if not idx.exists():
            return self._send_text("dashboard/index.html missing", status=500)
        return self._send_text(
            idx.read_text(encoding="utf-8"), ctype="text/html; charset=utf-8"
        )

    def _serve_static(self, rel: str):
        # Block path traversal
        target = (STATIC_DIR / rel).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())):
            return self._send_text("Forbidden", status=403)
        if not target.exists() or not target.is_file():
            return self._send_text("Not Found", status=404)
        ctype = "text/css" if target.suffix == ".css" else (
            "application/javascript" if target.suffix == ".js" else "text/plain"
        )
        return self._send_text(target.read_text(encoding="utf-8"), ctype=ctype)

    def _serve_sse(self):
        # Server-Sent Events: stream snapshots as the log file grows
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        last_size = -1
        last_mtime = -1.0
        last_ckpt_count = -1
        keepalive_at = time.time()

        try:
            while True:
                log = find_latest_log()
                snapshot = parse_log(log) if log else {"found": False}
                snapshot["checkpoints"] = list_checkpoints()
                snapshot["server_time"] = time.time()

                size = (log.stat().st_size if log else 0)
                mtime = (log.stat().st_mtime if log else 0)
                ckpts = len(snapshot["checkpoints"])

                changed = (
                    size != last_size
                    or mtime != last_mtime
                    or ckpts != last_ckpt_count
                )

                if changed:
                    payload = json.dumps(snapshot, default=str)
                    self.wfile.write(b"event: snapshot\n")
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    last_size = size
                    last_mtime = mtime
                    last_ckpt_count = ckpts
                    keepalive_at = time.time()

                # Heartbeat every 15s so the connection doesn't die
                if time.time() - keepalive_at > 15:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    keepalive_at = time.time()

                time.sleep(1.0)
        except (BrokenPipeError, ConnectionResetError):
            return


# --- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    httpd = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"[dashboard] serving on http://{args.host}:{args.port}", flush=True)
    print(f"[dashboard] log dir: {RESULTS_DIR}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] shutting down", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
