"""Baseline runner for the Metacog project.

This is the most important file in the project. It measures how
calibrated the BASE Qwen3.6-35B-A3B model is BEFORE we do any
fine-tuning. This number is the baseline we have to beat.

Two modes:
  1. --k 1 (default):  fast, one sample per question. Confidence is
     estimated from the LENGTH of the thinking trace and the
     PRESENCE of self-correction language ("wait", "actually",
     "let me reconsider", "hmm"). Heuristic but free.
  2. --k 5:  slower, K samples per question. Confidence is the
     self-consistency rate (fraction of K that match the majority
     answer). More expensive but principled.

After the run, we save:
  - results/baseline_traces.jsonl      — one row per (question, sample)
  - results/baseline_summary.json      — aggregate metrics
  - logs/baseline_<timestamp>.log      — progress log
  - results/baseline_reliability.txt   — human-readable reliability table

The summary JSON includes accuracy, ECE, Brier, AUROC, over/under-
confidence rates, plus a per-source breakdown. This is what we
diff against after RL/training.

Usage:
  python -m scripts.run_baseline --k 1 --limit 20
  python -m scripts.run_baseline --k 5 --limit 50
  python -m scripts.run_baseline --dry-run  # just loads questions, no API calls
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

# Repo-root imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.load_datasets import load_gsm8k, load_mmlu_pro
from data.schemas import QuestionRecord, TraceRecord
from eval.metrics import calibration_report, format_report
from eval.normalize import extract_final_answer, is_correct, normalize_answer
from tink.client import MetacogConfig, TinkerClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Confidence heuristics for the K=1 baseline (no extra API cost)
# ─────────────────────────────────────────────────────────────────────────────

# Words/phrases that indicate the model is uncertain or self-correcting.
# Their PRESENCE in the thinking trace is a weak signal of low confidence.
_HEDGES = re.compile(
    r"\b("
    r"wait|actually|let me reconsider|hmm|maybe|perhaps|possibly|"
    r"i think|i believe|i'm not sure|i'm unsure|i'm not certain|"
    r"let me re-?check|let me re-?examine|on second thought|"
    r"this is tricky|this is confusing|i might be wrong|"
    r"could be|might be|probably|likely|presumably"
    r")\b",
    re.IGNORECASE,
)

# Phrases that indicate the model is confident.
_CONFIDENCE_MARKERS = re.compile(
    r"\b("
    r"definitely|certainly|clearly|obviously|undoubtedly|surely|"
    r"the answer is|therefore|thus|so the answer|we get|this gives|"
    r"i'm confident|i am confident|no doubt"
    r")\b",
    re.IGNORECASE,
)


def heuristic_confidence(thinking_text: str) -> float:
    """Heuristic confidence in [0, 1] from the thinking trace alone.

    Not a real probability. But it correlates with calibration in
    practice — a model that hedges a lot IS less confident — and
    gives us a signal to measure ECE against without K samples.

    Heuristic:
      start at 0.7 (slight optimism), subtract 0.05 per hedge hit,
      add 0.03 per confidence-marker hit, clamp to [0.05, 0.95].
    """
    if not thinking_text:
        return 0.5
    n_hedges = len(_HEDGES.findall(thinking_text))
    n_markers = len(_CONFIDENCE_MARKERS.findall(thinking_text))
    score = 0.7 - 0.05 * n_hedges + 0.03 * n_markers
    return max(0.05, min(0.95, score))


def logprob_to_confidence(lp_mean: float | None) -> float | None:
    """Convert mean per-token log-probability to a [0, 1] confidence score.

    The Tinker API returns per-token logprobs (one per generated token).
    `lp_mean` is the mean of those (one float, or None if unavailable).

    Logprobs are <= 0. Higher (closer to 0) = more confident.
    We map with a calibrated softmax-style transform:
        conf = exp(lp_mean / scale)   where scale is a temperature.
    Empirically, well-trained instruction models have lp_mean in [-0.5, 0]
    for high-confidence outputs and [-3, -1] for low-confidence ones.
    A scale of 1.0 gives a usable spread across that range.
    """
    if lp_mean is None:
        return None
    # Clip to a reasonable range so one very long low-probability streak
    # doesn't drag the score to 0. -10 nats is ~ exp(-10) ≈ 4.5e-5.
    clipped = max(min(lp_mean, 0.0), -10.0)
    return float(round(math.exp(clipped), 4))


# ─────────────────────────────────────────────────────────────────────────────
# Build a chat-style prompt with the Qwen3.6 template (simple version)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a precise reasoning assistant. Think step by step, then give "
    "your final answer. For multiple choice, respond with the letter only "
    "(e.g. A, B, C). For numerical problems, give the number. Wrap your "
    "final answer in <answer>...</answer> tags."
)


def build_prompt(question: QuestionRecord) -> str:
    """Build a prompt that elicits a thinking trace + final answer.

    We use a simple system + user format. Tinker's renderer will wrap
    this in the model's chat template.
    """
    return f"{SYSTEM_PROMPT}\n\nQuestion:\n{question.question}\n\nAnswer:"


# ─────────────────────────────────────────────────────────────────────────────
# Trace → confidence + correctness
# ─────────────────────────────────────────────────────────────────────────────

def parse_trace(raw: str, model: str, q: QuestionRecord) -> tuple[str, str, float]:
    """Parse a raw model output into (final_answer_text, thinking_text, confidence).

    The model emits its reasoning first (in <|think|> or unconstrained),
    then a final answer (after <|/think|> or wrapped in <answer>).

    For heuristic confidence: we use the THINKING portion's hedging signals.
    """
    # Extract just the thinking block (before <|/think|>)
    thinking = raw
    final_block = raw
    think_end = re.search(r"<\|/?think\|?>|</?think>", raw, re.IGNORECASE)
    if think_end:
        thinking = raw[: think_end.start()]
        final_block = raw[think_end.end():]
    # Also try to isolate content inside <answer>...</answer>
    ans_match = re.search(r"<answer>(.*?)</answer>", final_block, re.DOTALL | re.IGNORECASE)
    if ans_match:
        extracted = ans_match.group(1).strip()
    else:
        extracted = final_block.strip()

    # Normalize the extracted answer against the source
    normalized = normalize_answer(extracted, q.source)
    if normalized is None:
        # Fall back to the raw extracted text
        normalized = extracted

    conf = heuristic_confidence(thinking)
    return normalized, thinking, conf


# ─────────────────────────────────────────────────────────────────────────────
# Main baseline runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=1, help="Samples per question for self-consistency")
    p.add_argument("--limit", type=int, default=0, help="If > 0, cap total questions to this number (otherwise n-gsm8k + n-mmlu-pro controls size)")
    p.add_argument("--n-gsm8k", type=int, default=20, help="Number of GSM8K questions")
    p.add_argument("--n-mmlu-pro", type=int, default=20, help="Number of MMLU-Pro questions")
    p.add_argument("--max-tokens", type=int, default=1024, help="Max new tokens per sample")
    p.add_argument("--model", type=str, default="", help="Tinker base model id (overrides TINKER_BASE_MODEL env / client default)")
    p.add_argument("--temperature", type=float, default=-1.0, help="Sampling temperature (-1 = use default 0.0 for k=1, 0.7 for k>1)")
    p.add_argument("--use-logprob-confidence", action="store_true", default=True, help="Use real per-token logprobs as confidence signal (default: True). Falls back to heuristic only when no logprobs are available.")
    p.add_argument("--dry-run", action="store_true", help="Skip API calls, just verify the pipeline")
    p.add_argument("--out-prefix", type=str, default="baseline", help="Filename prefix for outputs")
    args = p.parse_args()

    # Apply model override BEFORE constructing the client so it picks up the new base model.
    if args.model:
        # Touch the env so the lazy MetacogConfig picks it up
        os.environ["TINKER_BASE_MODEL"] = args.model

    log_path = LOGS_DIR / f"{args.out_prefix}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line)
        log_lines.append(line)

    log(f"=== Metacog baseline runner ===")
    log(f"  k={args.k}, gsm8k={args.n_gsm8k}, mmlu_pro={args.n_mmlu_pro}")
    log(f"  max_tokens={args.max_tokens}, dry_run={args.dry_run}")

    # ── Load questions ─────────────────────────────────────────────────────
    questions: list[QuestionRecord] = []
    if args.n_gsm8k > 0:
        questions.extend(load_gsm8k(args.n_gsm8k))
    if args.n_mmlu_pro > 0:
        questions.extend(load_mmlu_pro(args.n_mmlu_pro))
    if args.limit > 0:
        questions = questions[: args.limit]
    log(f"  loaded {len(questions)} questions")

    if not questions:
        log("ERROR: no questions loaded")
        return

    # ── Initialize Tinker client (skip on dry-run) ─────────────────────────
    client = TinkerClient()
    if not args.dry_run and not os.environ.get("TINKER_API_KEY", "").strip():
        log("ERROR: TINKER_API_KEY not set in .env")
        log("  Add it to E:/Projects/Metacog/.env and re-run")
        return
    if args.dry_run:
        log("  DRY-RUN mode: no API calls will be made")
    else:
        log(f"  Tinker base model: {client.config.default_base_model}")

    # ── Run samples ────────────────────────────────────────────────────────
    traces: list[TraceRecord] = []
    per_q_samples: dict[str, list[TraceRecord]] = defaultdict(list)
    t0 = time.perf_counter()
    n_api_calls = 0
    n_errors = 0

    for qi, q in enumerate(questions, 1):
        prompt = build_prompt(q)
        # Pick temperature: explicit override > heuristic (0.0 for k=1, 0.7 for self-consistency)
        if args.temperature >= 0:
            temp = args.temperature
        else:
            temp = 0.0 if args.k == 1 else 0.7

        # === BATCH: get all K samples in one API call (more efficient) ===
        per_token_logprobs_list: list[list[float] | None] = [None] * args.k
        logprob_mean_list: list[float | None] = [None] * args.k
        if args.dry_run:
            raws = [_synthetic_response(q, si, args.k) for si in range(args.k)]
            latencies = [0.0] * args.k
            n_tokens = [0] * args.k
        else:
            raws: list[str] = []
            latencies: list[float] = []
            n_tokens: list[int] = []
            try:
                # Call with num_samples=k — Tinker returns K sequences in one request.
                results = client.sample(
                    prompt,
                    max_tokens=args.max_tokens,
                    temperature=temp,
                    num_samples=args.k,
                )
                if not results or len(results) < args.k:
                    raise RuntimeError(f"tinker.sample returned {len(results) if results else 0} results, expected {args.k}")
                for si, res in enumerate(results):
                    raws.append(res.text)
                    latencies.append(res.latency_s)
                    n_tokens.append(res.n_tokens)
                    per_token_logprobs_list[si] = res.logprobs
                    logprob_mean_list[si] = res.logprob_mean
                n_api_calls += 1
            except Exception as e:
                log(f"  ERROR on {q.id} (k={args.k}): {e}")
                n_errors += 1
                continue

        for sample_idx in range(args.k):
            raw = raws[sample_idx]
            latency = latencies[sample_idx]
            n_completion_tokens = n_tokens[sample_idx]
            per_token_logprobs = per_token_logprobs_list[sample_idx]
            logprob_mean = logprob_mean_list[sample_idx]

            trace = TraceRecord(
                id=f"{q.id}-s{sample_idx}",
                question_id=q.id,
                model=client.config.default_base_model if not args.dry_run else "DRY-RUN",
                prompt=prompt,
                raw_output=raw,
                thinking_steps=[],
                final_answer="",
                parse_ok=True,
                n_completion_tokens=n_completion_tokens,
                latency_s=latency,
                per_token_logprobs=per_token_logprobs,
                logprob_mean=logprob_mean,
            )
            traces.append(trace)
            per_q_samples[q.id].append(trace)

        if qi % 5 == 0 or qi == len(questions):
            log(f"  progress: {qi}/{len(questions)} questions, {n_api_calls} API calls, {n_errors} errors")

    elapsed = time.perf_counter() - t0
    log(f"  finished in {elapsed:.1f}s ({n_api_calls} calls, {n_errors} errors)")

    # ── Parse and score ────────────────────────────────────────────────────
    log("\n=== Parsing and scoring ===")

    # For K=1, we use heuristic confidence.
    # For K>1, we use self-consistency = fraction of K matching the majority answer.
    # In both cases, the "per question" confidence is then attached to every
    # trace from that question, and we evaluate ECE on the per-trace level.

    per_q_conf: dict[str, float] = {}
    per_q_correct: dict[str, bool] = {}

    for q in questions:
        q_traces = per_q_samples.get(q.id, [])
        if not q_traces:
            continue

        # Parse each trace
        parsed_answers: list[str | None] = []
        for tr in q_traces:
            final, thinking, _ = parse_trace(tr.raw_output, tr.model, q)
            tr.final_answer = final if final is not None else ""
            parsed_answers.append(tr.final_answer)

        # Majority answer (most common normalized answer)
        valid = [a for a in parsed_answers if a]
        if not valid:
            per_q_conf[q.id] = 0.0
            per_q_correct[q.id] = False
            continue
        majority, _ = Counter(valid).most_common(1)[0]
        correct_majority = is_correct(majority, q.gold_answer, q.source)

        if args.k == 1:
            # Use REAL logprob-based confidence if we have it, else fall back
            # to the trace-heuristic.
            tr0 = q_traces[0]
            lp_conf = logprob_to_confidence(tr0.logprob_mean) if args.use_logprob_confidence else None
            if lp_conf is not None:
                per_q_conf[q.id] = lp_conf
            else:
                _, thinking, conf = parse_trace(tr0.raw_output, tr0.model, q)
                per_q_conf[q.id] = conf
        else:
            # Self-consistency: fraction matching the majority
            matching = sum(1 for a in parsed_answers if a == majority)
            per_q_conf[q.id] = matching / len(parsed_answers) if parsed_answers else 0.0

        per_q_correct[q.id] = correct_majority

    # Build per-trace confidence/correctness for ECE
    confs: list[float] = []
    corrects: list[bool] = []
    for tr in traces:
        confs.append(per_q_conf.get(tr.question_id, 0.5))
        corrects.append(per_q_correct.get(tr.question_id, False))

    # ── Aggregate metrics ──────────────────────────────────────────────────
    overall = calibration_report(confs, corrects)

    # Per-source breakdown
    per_source: dict[str, dict] = {}
    by_source: dict[str, tuple[list[float], list[bool]]] = defaultdict(lambda: ([], []))
    q_by_id = {q.id: q for q in questions}
    for tr in traces:
        q = q_by_id.get(tr.question_id)
        if not q:
            continue
        c, ok = by_source[q.source]
        c.append(per_q_conf.get(tr.question_id, 0.5))
        ok.append(per_q_correct.get(tr.question_id, False))
    for src, (c, ok) in by_source.items():
        rep = calibration_report(c, ok)
        per_source[src] = {
            "n": rep.n,
            "accuracy": rep.accuracy,
            "ece": rep.ece,
            "brier": rep.brier,
            "auroc": rep.auroc,
        }

    # ── Save outputs ───────────────────────────────────────────────────────
    traces_path = RESULTS_DIR / f"{args.out_prefix}_traces.jsonl"
    with open(traces_path, "w", encoding="utf-8") as f:
        for tr in traces:
            f.write(tr.model_dump_json() + "\n")
    log(f"  wrote {traces_path}")

    summary = {
        "config": {
            "k": args.k,
            "n_gsm8k": args.n_gsm8k,
            "n_mmlu_pro": args.n_mmlu_pro,
            "max_tokens": args.max_tokens,
            "dry_run": args.dry_run,
            "model": client.config.default_base_model if not args.dry_run else "DRY-RUN",
        },
        "elapsed_s": elapsed,
        "n_api_calls": n_api_calls,
        "n_errors": n_errors,
        "overall": {
            "n": overall.n,
            "accuracy": overall.accuracy,
            "ece": overall.ece,
            "brier": overall.brier,
            "auroc": overall.auroc,
            "overconfidence_rate": overall.overconfidence_rate,
            "underconfidence_rate": overall.underconfidence_rate,
        },
        "per_source": per_source,
        "reliability_bins": [
            {"lo": lo, "hi": hi, "n": n, "mean_conf": mc, "actual_acc": acc}
            for lo, hi, n, mc, acc in overall.bin_report if n > 0
        ],
    }
    summary_path = RESULTS_DIR / f"{args.out_prefix}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"  wrote {summary_path}")

    # Pretty report
    report_path = RESULTS_DIR / f"{args.out_prefix}_reliability.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"=== Metacog baseline report ===\n")
        f.write(f"model: {client.config.default_base_model if not args.dry_run else 'DRY-RUN'}\n")
        f.write(f"k={args.k}, n={len(questions)} questions, {n_api_calls} API calls in {elapsed:.1f}s\n\n")
        f.write("OVERALL:\n")
        f.write(format_report(overall))
        f.write("\n\nPER SOURCE:\n")
        for src, m in per_source.items():
            f.write(f"\n  [{src}] n={m['n']} acc={m['accuracy']:.3f} "
                    f"ECE={m['ece']:.3f} Brier={m['brier']:.3f} AUROC={m['auroc']}\n")
    log(f"  wrote {report_path}")

    # Save log
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    # ── Print final summary ────────────────────────────────────────────────
    log("\n=== RESULTS ===")
    log(f"  overall accuracy:  {overall.accuracy:.3f}")
    log(f"  overall ECE:       {overall.ece:.3f}  (lower = better calibration)")
    log(f"  overall Brier:     {overall.brier:.3f}")
    log(f"  overall AUROC:     {overall.auroc}")
    log(f"  overconfident rate: {overall.overconfidence_rate:.3f}")
    log(f"  underconfident rate: {overall.underconfidence_rate:.3f}")
    log("")
    log("  Per-source:")
    for src, m in per_source.items():
        log(f"    {src}: n={m['n']}  acc={m['accuracy']:.3f}  ECE={m['ece']:.3f}  Brier={m['brier']:.3f}")
    log("")
    log(f"  outputs in: {RESULTS_DIR}/{args.out_prefix}_*")


def _synthetic_response(q: QuestionRecord, sample_idx: int, k: int) -> str:
    """Generate a synthetic model response for dry-runs.

    The synthetic response is INTENTIONALLY miscalibrated in a
    predictable way so the metrics pipeline can be verified.
    The pattern is a simple rotation over (right, wrong, right, wrong, ...)
    so multi-sample runs also test majority-vote.

    The dry-run "confidence" (from hedging language) is correlated
    with correctness: right answers have low hedge density, wrong
    answers have high hedge density. This produces a measurable but
    imperfect calibration — ECE in the 0.10-0.20 range, not 0.
    """
    # Alternate right/wrong across sample indices, offset by question
    # id so different questions see different starts.
    q_offset = sum(ord(c) for c in q.id) % 4
    is_correct_sample = ((sample_idx + q_offset) % 3) != 0  # 2/3 right
    if is_correct_sample:
        # Right, confident (few hedges)
        thinking = (
            "Let me solve this step by step.\n"
            "First, I identify the relevant facts.\n"
            "Then I apply the formula. The answer is clear.\n"
            "Therefore, the answer is " + str(q.gold_answer) + "."
        )
        final = f"<answer>{q.gold_answer}</answer>"
    else:
        # Wrong, with hedging (many hedges -> low confidence)
        wrong = "42" if str(q.gold_answer) != "42" else "0"
        thinking = (
            "Hmm, this is tricky. Maybe I should reconsider.\n"
            "Wait, let me re-check. Actually, I think " + wrong + " might be wrong.\n"
            "Let me think again. Perhaps " + wrong + " is the answer.\n"
            "I'm not sure, but I'll go with " + wrong + "."
        )
        final = f"<answer>{wrong}</answer>"
    return f"<|think|>{thinking}<|/think|>\n\n{final}"


if __name__ == "__main__":
    main()
