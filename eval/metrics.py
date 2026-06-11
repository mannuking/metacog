"""Calibration metrics for the Metacog project.

The point of metacognition is not "is the model right?" but "does the
model know WHEN it is right?". A perfectly metacognitive model would
say "I'm 90% sure" on the 90% of questions it gets right, and "I'm 10%
sure" on the 10% it gets wrong. The gap between stated confidence and
actual accuracy is what we measure.

Metrics:
    - accuracy              — fraction of correct answers
    - expected_calibration_error (ECE) — weighted |confidence - accuracy| per bin
    - brier_score           — mean squared error of probability predictions
    - auroc                 — can the confidence separate correct from wrong?
    - abstention_accuracy   — if we abstain on the bottom-X% confident, how
                              often is the abstention correct (we were right
                              NOT to answer)?
    - overconfidence_rate   — fraction of high-confidence (>0.8) that are wrong

All metrics take a list of (predicted_confidence, was_correct) pairs and
return a single float (or a small dict for binned reports).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class CalibrationReport:
    n: int
    accuracy: float
    ece: float
    brier: float
    auroc: float | None  # None if no variation in correctness
    overconfidence_rate: float
    underconfidence_rate: float
    # Bin-wise breakdown: (bin_lo, bin_hi, n, mean_conf, empirical_acc)
    bin_report: list[tuple[float, float, int, float, float]]


def expected_calibration_error(
    confs: list[float],
    corrects: list[bool],
    n_bins: int = 10,
) -> tuple[float, list[tuple[float, float, int, float, float]]]:
    """ECE with equal-width bins in [0, 1].

    ECE = sum over bins: (|bin| / N) * |mean_conf_in_bin - acc_in_bin|

    Lower is better. Perfect calibration = 0.
    """
    assert len(confs) == len(corrects)
    N = len(confs)
    if N == 0:
        return 0.0, []
    bins: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for c, ok in zip(confs, corrects):
        b = min(int(c * n_bins), n_bins - 1)
        bins[b].append((c, ok))
    ece = 0.0
    report: list[tuple[float, float, int, float, float]] = []
    for b in range(n_bins):
        items = bins.get(b, [])
        if not items:
            lo = b / n_bins
            hi = (b + 1) / n_bins
            report.append((lo, hi, 0, 0.0, 0.0))
            continue
        mean_conf = sum(c for c, _ in items) / len(items)
        acc = sum(1 for _, ok in items if ok) / len(items)
        ece += (len(items) / N) * abs(mean_conf - acc)
        lo = b / n_bins
        hi = (b + 1) / n_bins
        report.append((lo, hi, len(items), mean_conf, acc))
    return ece, report


def brier_score(confs: list[float], corrects: list[bool]) -> float:
    """Mean squared error of probability predictions.

    Brier = mean((p - y)^2) where y is 1 for correct, 0 for wrong.
    Perfect predictions: 0. Always-predicted-majority-class: ~0.25.
    """
    if not confs:
        return 0.0
    s = 0.0
    for c, ok in zip(confs, corrects):
        y = 1.0 if ok else 0.0
        s += (c - y) ** 2
    return s / len(confs)


def auroc(confs: list[float], corrects: list[bool]) -> float | None:
    """Area under the ROC curve. Measures how well confidence separates
    correct from wrong. 1.0 = perfect, 0.5 = random, 0.0 = inverted.

    Returns None if all-correct or all-wrong (degenerate).
    """
    if not confs:
        return None
    n_pos = sum(1 for c in corrects if c)
    n_neg = len(corrects) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    # Sort by confidence desc. For tied confidences, use average rank.
    pairs = sorted(
        zip(confs, corrects),
        key=lambda x: x[0],
        reverse=True,
    )
    # Compute rank sums
    ranks: list[float] = []
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        # Average rank (1-indexed) for this tied group
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks.append(avg_rank)
        i = j
    sum_rank_pos = sum(r for r, (_, ok) in zip(ranks, pairs) if ok)
    auc = (sum_rank_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def overconfidence_rate(confs: list[float], corrects: list[bool], threshold: float = 0.8) -> float:
    """Fraction of high-confidence predictions that were WRONG.

    Lower is better. A metacognitive model should be rare to be very
    confident AND wrong.
    """
    if not confs:
        return 0.0
    high = [(c, ok) for c, ok in zip(confs, corrects) if c >= threshold]
    if not high:
        return 0.0
    return sum(1 for _, ok in high if not ok) / len(high)


def underconfidence_rate(confs: list[float], corrects: list[bool], threshold: float = 0.2) -> float:
    """Fraction of low-confidence predictions that were RIGHT.

    Lower is better. A metacognitive model shouldn't be uncertain when
    it actually knows the answer.
    """
    if not confs:
        return 0.0
    low = [(c, ok) for c, ok in zip(confs, corrects) if c <= threshold]
    if not low:
        return 0.0
    return sum(1 for _, ok in low if ok) / len(low)


def calibration_report(
    confs: list[float],
    corrects: list[bool],
    n_bins: int = 10,
) -> CalibrationReport:
    """Compute all calibration metrics in one call."""
    n = len(confs)
    if n == 0:
        return CalibrationReport(
            n=0, accuracy=0.0, ece=0.0, brier=0.0, auroc=None,
            overconfidence_rate=0.0, underconfidence_rate=0.0, bin_report=[],
        )
    ece, bins = expected_calibration_error(confs, corrects, n_bins=n_bins)
    return CalibrationReport(
        n=n,
        accuracy=sum(corrects) / n,
        ece=ece,
        brier=brier_score(confs, corrects),
        auroc=auroc(confs, corrects),
        overconfidence_rate=overconfidence_rate(confs, corrects),
        underconfidence_rate=underconfidence_rate(confs, corrects),
        bin_report=bins,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Self-consistency confidence estimator (no extra model head needed)
# ─────────────────────────────────────────────────────────────────────────────

def self_consistency_confidence(
    answers: list[str | None], gold: str, source: str
) -> float:
    """Given K=5 (or however many) sampled answers from the same question,
    return the fraction that match the gold answer.

    This is the BASELINE confidence signal. It doesn't require a learned
    head — just multiple samples. Useful as:
      (a) The pre-training baseline for ECE
      (b) A pseudo-label target for the future confidence-head training
      (c) A ceiling on what any head-based method can achieve

    The intuition: if the model is consistently right, it's probably
    confident. If it's inconsistent, it's guessing.
    """
    from .normalize import is_correct

    if not answers:
        return 0.0
    valid = [a for a in answers if a is not None]
    if not valid:
        return 0.0
    correct_count = sum(1 for a in valid if is_correct(a, gold, source))
    return correct_count / len(valid)


# ─────────────────────────────────────────────────────────────────────────────
# Logit-entropy confidence estimator (per-step, when we have logprobs)
# ─────────────────────────────────────────────────────────────────────────────

def neg_avg_logprob_to_conf(neg_avg_logprob: float) -> float:
    """Convert a negative average log-probability to a confidence in [0, 1].

    This is a heuristic: the more "surprising" the model's output was to
    itself (low log-prob), the less confident we treat it.

    Calibration: avg_neg_logprob in [0, 0.1] -> high confidence; in
    [1.0, +inf] -> very low. We use an exponential decay.
    """
    if math.isnan(neg_avg_logprob):
        return 0.5
    return float(math.exp(-neg_avg_logprob))


def format_report(rep: CalibrationReport) -> str:
    """Pretty-print a CalibrationReport."""
    lines = [
        f"  N:                       {rep.n}",
        f"  accuracy:                {rep.accuracy:.3f}",
        f"  ECE (10-bin):            {rep.ece:.3f}  (lower = better calibrated)",
        f"  Brier score:             {rep.brier:.3f}  (lower = better)",
        f"  AUROC:                   {rep.auroc:.3f}  (1.0 = perfect separation)" if rep.auroc is not None else "  AUROC:                   N/A (degenerate)",
        f"  overconfident rate (>0.8): {rep.overconfidence_rate:.3f}  (lower = better)",
        f"  underconfident rate (<0.2): {rep.underconfidence_rate:.3f}  (lower = better)",
        "",
        "  Reliability diagram (10 bins):",
        "  bin          n    mean_conf   actual_acc   gap",
    ]
    for lo, hi, n, mc, acc in rep.bin_report:
        if n == 0:
            continue
        gap = mc - acc
        lines.append(
            f"  [{lo:.1f},{hi:.1f})    {n:>3}    {mc:.3f}      {acc:.3f}      {gap:+.3f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test with a known-bad and known-good distribution
    print("=== Perfectly calibrated ===")
    # 10 examples at each confidence level, with matching correctness rate
    confs = []
    corrects = []
    for true_p, n_at in [(0.1, 10), (0.3, 10), (0.5, 10), (0.7, 10), (0.9, 10)]:
        for i in range(n_at):
            confs.append(true_p)
            corrects.append(i < true_p * n_at)
    rep = calibration_report(confs, corrects)
    print(format_report(rep))

    print("\n=== Over-confident model ===")
    # Always says 0.9 confidence, but only 50% correct
    confs = [0.9] * 20
    corrects = [i < 10 for i in range(20)]
    rep = calibration_report(confs, corrects)
    print(format_report(rep))

    print("\n=== Self-consistency estimator ===")
    # 5 samples, 4 match the gold answer
    answers = ["18", "18", "18", "18", "17"]
    sc = self_consistency_confidence(answers, gold="18", source="gsm8k")
    print(f"  self-consistency: {sc:.3f}  (4/5 agree)")
