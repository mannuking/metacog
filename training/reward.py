"""Metacognitive reward function for Phase 1 RL.

The reward is the **only** thing the model is trained on. It must:

1. **Maximize correct final answers** (the floor: 100% if possible)
2. **Minimize confident wrong answers** (the failure mode of the baseline)
3. **Reward calibrated confidence** (when confident, be right; when unsure, say so)
4. **Reward appropriate abstention** (low confidence → abstain)
5. **Reward good metacognitive behaviors** (unit-check, sanity-bounds, verify
   without conflict)
6. **Penalize padding** (long thinking isn't a virtue)

Mathematically, for a single rollout:

    reward =
        +1.0  if correct AND not abstained
        +0.5  if correct AND not abstained AND confidence matches empirical accuracy
        +0.3  if abstained AND confidence < 0.4      (appropriate humility)
        -0.5  if wrong AND not abstained AND confidence > 0.8   (overconfident wrong)
         0.0  if wrong AND not abstained AND confidence ≤ 0.8  (humble wrong)
         0.0  if wrong AND abstained                (fine to abstain on hard)

    +0.05  if parse_ok                            (followed the format)
    +0.02  if unit_check_ok                       (did the back-check)
    +0.02  if sanity_bounds_ok                    (sanity-checked)
    -0.05  if verify_conflict                     (the model contradicted itself)

    + small calibration_bonus  (linear in |confidence - actual_accuracy| penalty)

    - length_penalty × (n_tokens / 1000)         (cap at -0.1)

The reward is computed from a `MetacogOutput` (parsed model text) plus
the question's gold answer and the source (`gsm8k` or `mmlu_pro`).

This file is **deterministic and pure** — no I/O, no Tinker calls. The
training loop in `training/rl_loop.py` calls it on every rollout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

from data.schemas import QuestionRecord
from eval.normalize import is_correct
from eval.structured_parse import MetacogOutput, parse_metacog_output


# ---------------------------------------------------------------------------
# Reward config (all tunable from the CLI)
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """Hyperparameters of the metacog reward function.

    Defaults chosen from the integration menu (items 42, 44) and the
    Anthropic "4× less likely to let flaws pass" framing.
    """
    # Main correctness signals
    r_correct: float = 1.0
    r_correct_calibrated: float = 0.5     # bonus if confidence is well-calibrated
    r_abstain_humble: float = 0.3         # abstain with low confidence
    r_overconfident_wrong: float = -0.5   # the BAD outcome to avoid
    r_humble_wrong: float = 0.0           # "I don't know" is fine

    # Format / behavior bonuses
    r_format_ok: float = 0.05
    r_unit_check: float = 0.02
    r_sanity_bounds: float = 0.02
    r_verify_conflict_penalty: float = -0.05

    # Calibration bonus
    # The model's confidence is rewarded for being close to its actual
    # accuracy AT that confidence level. We use a per-rollout proxy:
    # if it claims conf c and is correct, the bonus is +c (good).
    # if it claims conf c and is wrong, the bonus is -(1-c) (good, close to 0).
    # (This is equivalent to a Brier-style soft loss.)
    r_calibration_weight: float = 0.2

    # Length penalty (cap at -0.1)
    length_penalty_per_1k_tokens: float = 0.05
    max_length_penalty: float = 0.1

    # Abstention handling
    abstain_confidence_threshold: float = 0.4  # below this, abstain is "humble"

    # When the model abstains, we count it as "not wrong" but also not
    # "right". The empirical accuracy in abstention is undefined; we
    # assign a neutral reward but give a small bonus for humility
    # when confidence was genuinely low.
    r_abstain_wrong: float = 0.0
    r_abstain_correct: float = 0.0  # no bonus for correct abstention (we don't know)


# ---------------------------------------------------------------------------
# Reward record
# ---------------------------------------------------------------------------

@dataclass
class RewardBreakdown:
    """The components of a single reward computation, for logging."""

    total: float
    r_correct: float = 0.0
    r_correct_calibrated: float = 0.0
    r_abstain_humble: float = 0.0
    r_overconfident_wrong: float = 0.0
    r_humble_wrong: float = 0.0
    r_format: float = 0.0
    r_unit_check: float = 0.0
    r_sanity_bounds: float = 0.0
    r_verify_conflict: float = 0.0
    r_calibration: float = 0.0
    r_length: float = 0.0
    # Diagnostics
    correct: bool = False
    abstained: bool = False
    confidence: float | None = None
    n_tokens: int = 0
    parse_ok: bool = False
    # What flag fired
    primary_outcome: str = "neutral"  # correct | overconfident_wrong | humble_wrong | abstain | neutral

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Main reward function
# ---------------------------------------------------------------------------

def compute_reward(
    raw_output: str,
    question: QuestionRecord,
    n_completion_tokens: int = 0,
    cfg: RewardConfig | None = None,
) -> tuple[float, RewardBreakdown, MetacogOutput]:
    """Compute the metacog reward for one model rollout.

    Parameters
    ----------
    raw_output : str
        The raw text the model emitted.
    question : QuestionRecord
        The question (used for correctness check + source).
    n_completion_tokens : int
        How many tokens the model generated. For length penalty.
    cfg : RewardConfig | None
        Reward hyperparameters. Defaults to RewardConfig().

    Returns
    -------
    (total_reward, breakdown, parsed_output) : tuple
    """
    cfg = cfg or RewardConfig()
    parsed = parse_metacog_output(raw_output)

    # Determine correctness (special-case abstention)
    abstained = parsed.abstained
    if abstained:
        correct = False  # by definition, abstaining is not "correct" in the strict sense
    else:
        correct = bool(parsed.answer_text) and is_correct(
            parsed.answer_text, question.gold_answer, question.source
        )

    confidence = parsed.confidence

    # ── Primary outcome ─────────────────────────────────────────────────
    primary = "neutral"
    primary_r = 0.0
    calibration_r = 0.0

    if abstained and confidence is not None and confidence < cfg.abstain_confidence_threshold:
        primary = "abstain_humble"
        primary_r = cfg.r_abstain_humble
    elif abstained and confidence is not None and confidence >= cfg.abstain_confidence_threshold:
        # Abstained but with high confidence — model contradicted itself.
        # Treat as humble_wrong (no penalty, no big bonus).
        primary = "humble_wrong"
        primary_r = cfg.r_humble_wrong
    elif correct and not abstained:
        primary = "correct"
        primary_r = cfg.r_correct
        if confidence is not None:
            # Calibration bonus: high confidence + correct = good
            # (the model "knew" what it was doing)
            calibration_r = cfg.r_calibration_weight * confidence
            # Plus, if its confidence is in the well-calibrated zone (0.5-0.95),
            # we give an extra bonus
            if 0.5 <= confidence <= 0.95:
                primary_r += cfg.r_correct_calibrated
    elif (not correct) and (not abstained):
        if confidence is not None and confidence > 0.8:
            primary = "overconfident_wrong"
            primary_r = cfg.r_overconfident_wrong
            if confidence is not None:
                # Calibration penalty: high conf + wrong = bad
                calibration_r = -cfg.r_calibration_weight * (1.0 - (1.0 - confidence))
                # = -0.2 * confidence
        else:
            primary = "humble_wrong"
            primary_r = cfg.r_humble_wrong
            if confidence is not None:
                # Humble + wrong: small calibration penalty (claimed low, was wrong)
                # but not too bad
                calibration_r = -cfg.r_calibration_weight * 0.1
    elif correct and abstained:
        # Should not happen, but be safe
        primary = "neutral"
        primary_r = 0.0

    # ── Format and behavior bonuses ────────────────────────────────────
    r_format = cfg.r_format_ok if parsed.parse_ok else 0.0
    r_unit = cfg.r_unit_check if parsed.unit_check_ok else 0.0
    r_sanity = cfg.r_sanity_bounds if parsed.sanity_bounds_ok else 0.0
    r_verify_conflict = cfg.r_verify_conflict_penalty if parsed.verify_conflict else 0.0

    # ── Length penalty ──────────────────────────────────────────────────
    if n_completion_tokens > 0:
        r_length = -min(
            cfg.max_length_penalty,
            cfg.length_penalty_per_1k_tokens * (n_completion_tokens / 1000.0),
        )
    else:
        r_length = 0.0

    # ── Total ───────────────────────────────────────────────────────────
    total = (
        primary_r
        + calibration_r
        + r_format
        + r_unit
        + r_sanity
        + r_verify_conflict
        + r_length
    )

    breakdown = RewardBreakdown(
        total=round(total, 4),
        r_correct=round(primary_r, 4),
        r_correct_calibrated=round(cfg.r_correct_calibrated if primary == "correct" and 0.5 <= (confidence or 0) <= 0.95 else 0.0, 4),
        r_abstain_humble=round(cfg.r_abstain_humble if primary == "abstain_humble" else 0.0, 4),
        r_overconfident_wrong=round(cfg.r_overconfident_wrong if primary == "overconfident_wrong" else 0.0, 4),
        r_humble_wrong=round(cfg.r_humble_wrong if primary == "humble_wrong" else 0.0, 4),
        r_format=round(r_format, 4),
        r_unit_check=round(r_unit, 4),
        r_sanity_bounds=round(r_sanity, 4),
        r_verify_conflict=round(r_verify_conflict, 4),
        r_calibration=round(calibration_r, 4),
        r_length=round(r_length, 4),
        correct=correct,
        abstained=abstained,
        confidence=confidence,
        n_tokens=n_completion_tokens,
        parse_ok=parsed.parse_ok,
        primary_outcome=primary,
    )
    return total, breakdown, parsed


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data.schemas import QuestionRecord

    q = QuestionRecord(
        id="gsm8k-1",
        source="gsm8k",
        domain="arithmetic",
        question="Janet's ducks lay 16 eggs...",
        gold_answer="18",
    )

    cases = [
        # (description, raw, expected_primary)
        (
            "Correct + high confidence",
            "<think>solve: 18</think>\n<verify>ok</verify>\n<answer>18</answer>\n<confidence>0.95</confidence>",
            "correct",
        ),
        (
            "Correct + low confidence (underconfident)",
            "<think>solve: 18</think>\n<verify>ok</verify>\n<answer>18</answer>\n<confidence>0.4</confidence>",
            "correct",
        ),
        (
            "Wrong + high confidence (BAD)",
            "<think>solve: 36</think>\n<verify>ok</verify>\n<answer>36</answer>\n<confidence>0.9</confidence>",
            "overconfident_wrong",
        ),
        (
            "Wrong + low confidence (humble)",
            "<think>hmm maybe 36?</think>\n<verify>unsure</verify>\n<answer>36</answer>\n<confidence>0.3</confidence>",
            "humble_wrong",
        ),
        (
            "Abstain + low confidence (humble)",
            "<think>don't know</think>\n<verify>n/a</verify>\n<answer>abstain</answer>\n<confidence>0.2</confidence>",
            "abstain_humble",
        ),
        (
            "Abstain + high confidence (contradictory)",
            "<think>answer is 18</think>\n<verify>ok</verify>\n<answer>abstain</answer>\n<confidence>0.9</confidence>",
            "humble_wrong",  # counted as humble_wrong (no penalty, no big bonus)
        ),
    ]

    print(f"{'Case':<45} {'Outcome':<25} {'Reward':>8}")
    print("-" * 80)
    for desc, raw, expected in cases:
        total, br, _ = compute_reward(raw, q, n_completion_tokens=300)
        ok = "✓" if br.primary_outcome == expected else "✗"
        print(f"{desc:<45} {br.primary_outcome:<25} {total:>7.3f}  {ok}")
