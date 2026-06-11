"""Generate the Phase 1 training curve figure from per_step_metrics.csv.

Outputs: paper/figures/phase1_training_curves.png (and .svg)
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "phase1" / "per_step_metrics.csv"
OUT_DIR = ROOT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

steps, reward, correct, humble, overconf = [], [], [], [], []
with CSV.open() as f:
    for row in csv.DictReader(f):
        if row["step"] == "-1_init_eval":
            continue
        steps.append(int(row["step"]))
        reward.append(float(row["reward_mean"]))
        correct.append(int(row["correct"]))
        humble.append(int(row["humble_wrong"]))
        overconf.append(int(row["overconfident_wrong"]))

# Figure 1: 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=140)
fig.suptitle(
    "Phase 1 RL — Qwen3.6-35B-A3B + LoRA r=32, 4,000 rollouts over 20 steps",
    fontsize=12, fontweight="bold", y=0.995,
)

# Reward
ax = axes[0, 0]
ax.plot(steps, reward, "o-", color="#22c55e", linewidth=2, markersize=5)
ax.set_title("Mean reward per step")
ax.set_xlabel("Step")
ax.set_ylabel("Reward (mean)")
ax.set_ylim(-0.1, 1.05)
ax.grid(alpha=0.3)
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

# Outcomes stacked
ax = axes[0, 1]
ax.stackplot(steps, correct, humble, overconf,
             labels=["correct", "humble_wrong", "overconfident_wrong"],
             colors=["#22c55e", "#60a5fa", "#ef4444"], alpha=0.85)
ax.set_title("Outcome mix per step (200 rollouts)")
ax.set_xlabel("Step")
ax.set_ylabel("Count (out of 200)")
ax.legend(loc="upper left", fontsize=9)
ax.set_ylim(0, 200)
ax.set_xlim(1, 20)

# Accuracy
ax = axes[1, 0]
acc_pct = [100.0 * c / (c + h + o) for c, h, o in zip(correct, humble, overconf)]
ax.plot(steps, acc_pct, "o-", color="#22c55e", linewidth=2, markersize=5)
ax.set_title("Accuracy (% correct of 200 rollouts)")
ax.set_xlabel("Step")
ax.set_ylabel("Accuracy %")
ax.set_ylim(0, 100)
ax.grid(alpha=0.3)

# Overconfident rate
ax = axes[1, 1]
over_pct = [100.0 * o / (c + h + o) for c, h, o in zip(correct, humble, overconf)]
ax.plot(steps, over_pct, "o-", color="#ef4444", linewidth=2, markersize=5)
ax.set_title("Overconfident-wrong rate (% of 200 rollouts)")
ax.set_xlabel("Step")
ax.set_ylabel("Overconfident-wrong %")
ax.set_ylim(0, 20)
ax.grid(alpha=0.3)

plt.tight_layout()
out_png = OUT_DIR / "phase1_training_curves.png"
out_svg = OUT_DIR / "phase1_training_curves.svg"
plt.savefig(out_png, bbox_inches="tight", facecolor="white")
plt.savefig(out_svg, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"wrote {out_png}")
print(f"wrote {out_svg}")
