"""Reliability diagram for Phase 0 baseline (Qwen3.6-35B-A3B, K=1, n=50).

Outputs: paper/figures/phase0_reliability_diagram.png/.svg
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "phase0" / "baseline_k1_summary.json"
OUT_DIR = ROOT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with DATA.open() as f:
    data = json.load(f)

bins = data["reliability_bins"]
n_per_bin = [b["n"] for b in bins]
mean_conf = [b["mean_conf"] for b in bins]
actual_acc = [b["actual_acc"] for b in bins]
bin_centers = [(b["lo"] + b["hi"]) / 2 for b in bins]
gaps = [m - a for m, a in zip(mean_conf, actual_acc)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=140,
                                gridspec_kw={"width_ratios": [2, 1]})
fig.suptitle("Phase 0 reliability — Qwen3.6-35B-A3B baseline (n=50, K=1)",
             fontsize=12, fontweight="bold")

# Left: reliability diagram
ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
bar_widths = [b["hi"] - b["lo"] for b in bins]
ax1.bar(bin_centers, actual_acc, width=bar_widths, alpha=0.5,
        color="#60a5fa", edgecolor="#1e40af", label="actual accuracy")
ax1.plot(bin_centers, mean_conf, "o-", color="#ef4444",
         markersize=10, linewidth=2, label="mean confidence")
for i, (n, gap) in enumerate(zip(n_per_bin, gaps)):
    ax1.annotate(f"n={n}\nΔ={gap:+.2f}", (bin_centers[i], actual_acc[i]),
                 textcoords="offset points", xytext=(0, -22),
                 ha="center", fontsize=8, color="#1e40af")
ax1.set_xlabel("Confidence bin")
ax1.set_ylabel("Accuracy / Mean confidence")
ax1.set_xlim(0.5, 1.05)
ax1.set_ylim(0, 1.05)
ax1.set_title("Reliability diagram")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(alpha=0.3)

# Right: per-source comparison
ax2.bar(["GSM8K", "MMLU-Pro"],
        [data["per_source"]["gsm8k"]["accuracy"] * 100,
         data["per_source"]["mmlu_pro"]["accuracy"] * 100],
        color=["#22c55e", "#ef4444"], alpha=0.7)
for i, v in enumerate([data["per_source"]["gsm8k"]["accuracy"] * 100,
                        data["per_source"]["mmlu_pro"]["accuracy"] * 100]):
    ax2.text(i, v + 2, f"{v:.0f}%", ha="center", fontweight="bold")
ax2.set_ylabel("Accuracy %")
ax2.set_title("Accuracy by source")
ax2.set_ylim(0, 100)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
out_png = OUT_DIR / "phase0_reliability_diagram.png"
out_svg = OUT_DIR / "phase0_reliability_diagram.svg"
plt.savefig(out_png, bbox_inches="tight", facecolor="white")
plt.savefig(out_svg, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"wrote {out_png}")
print(f"wrote {out_svg}")
