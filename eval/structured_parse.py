"""Structured parser for Metacog model outputs.

The model emits a structured output (defined in `prompts/chat_template.py`):

    <think> ... </think>
    <verify> ... </verify>
    <answer>X</answer>
    <confidence>0.XX</confidence>

This module parses the raw text into a typed `MetacogOutput` record so
the reward function and the eval pipeline can use the individual fields.

It also extracts:
- sub-claims (the model's own decomposition of the problem)
- unit-check / sanity-bounds signals
- verify-block agreement with the first-pass answer
- confidence as a float
- whether the model abstained

Robust to malformed outputs: every field has a safe default. The
"parse_ok" flag indicates whether the output was well-formed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_RE_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_RE_VERIFY = re.compile(r"<verify>(.*?)</verify>", re.DOTALL | re.IGNORECASE)
_RE_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_RE_CONFIDENCE = re.compile(r"<confidence>\s*([0-9]*\.?[0-9]+)\s*</confidence>", re.IGNORECASE)
_RE_DIFFICULTY = re.compile(r"Difficulty:\s*(easy|hard)", re.IGNORECASE)
_RE_BUDGET = re.compile(r"Budget:\s*(\d+)\s*pass", re.IGNORECASE)
_RE_SUBCLAIMS = re.compile(
    r"Sub-claim\s+\d+:\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)
_RE_UNIT_CHECK_OK = re.compile(r"Unit-check:\s*.+?✓", re.IGNORECASE)
_RE_SANITY_OK = re.compile(r"Sanity-bounds:\s*.+?✓", re.IGNORECASE)
_RE_CONFLICT = re.compile(r"(conflict|disagree|mismatch|re-revise)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Output record
# ---------------------------------------------------------------------------

@dataclass
class MetacogOutput:
    """Parsed structured output of one Metacog model completion."""

    raw: str

    # Top-level blocks
    think_text: str = ""
    verify_text: str = ""
    answer_text: str = ""
    confidence: float | None = None

    # Sub-fields
    subclaims: list[str] = field(default_factory=list)
    difficulty: str | None = None       # "easy" | "hard" | None
    budget_passes: int | None = None
    unit_check_ok: bool = False
    sanity_bounds_ok: bool = False
    verify_conflict: bool = False       # True if the verify block flagged a conflict

    # Flags
    abstained: bool = False
    parse_ok: bool = False              # True if at least answer + confidence were found

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_metacog_output(raw: str) -> MetacogOutput:
    """Parse a raw model completion into a `MetacogOutput`.

    Robust to:
    - Missing blocks (defaults to empty string)
    - Out-of-order blocks (we extract each independently)
    - Malformed confidence (left as None)
    - Whitespace and case-insensitive tags
    """
    out = MetacogOutput(raw=raw)

    # <think>...</think>
    m = _RE_THINK.search(raw)
    if m:
        out.think_text = m.group(1).strip()
        out.subclaims = [s.strip() for s in _RE_SUBCLAIMS.findall(out.think_text)]
        m_diff = _RE_DIFFICULTY.search(out.think_text)
        if m_diff:
            out.difficulty = m_diff.group(1).lower()
        m_bud = _RE_BUDGET.search(out.think_text)
        if m_bud:
            out.budget_passes = int(m_bud.group(1))
        out.unit_check_ok = bool(_RE_UNIT_CHECK_OK.search(out.think_text))
        out.sanity_bounds_ok = bool(_RE_SANITY_OK.search(out.think_text))

    # <verify>...</verify>
    m = _RE_VERIFY.search(raw)
    if m:
        out.verify_text = m.group(1).strip()
        out.verify_conflict = bool(_RE_CONFLICT.search(out.verify_text))

    # <answer>...</answer>
    m = _RE_ANSWER.search(raw)
    if m:
        out.answer_text = m.group(1).strip()
        out.abstained = out.answer_text.lower().strip() == "abstain"

    # <confidence>0.XX</confidence>
    m = _RE_CONFIDENCE.search(raw)
    if m:
        try:
            out.confidence = float(m.group(1))
            # Clamp to [0, 1] in case the model went over
            out.confidence = max(0.0, min(1.0, out.confidence))
        except (ValueError, TypeError):
            out.confidence = None

    # parse_ok = we found the answer and either confidence or abstention
    out.parse_ok = bool(out.answer_text) and (out.confidence is not None or out.abstained)

    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = """<think>
Sub-claim 1: 16 eggs per day.
Sub-claim 2: 3 eaten, 4 baked.
Sub-claim 3: Sell the rest.
Difficulty: easy
Budget: 1 pass

Solve: 16 - 3 - 4 = 9. 9 × 2 = 18.
Unit-check: 18/9 = 2. ✓
Sanity-bounds: plausible. ✓
</think>

<verify>
Re-derive: 16 - 7 = 9. 9 × 2 = 18. Same.
Counterfactual: if 36, that would mean selling 18 eggs, not 9.
</verify>

<answer>18</answer>
<confidence>0.95</confidence>
"""
    out = parse_metacog_output(sample)
    import json
    print(json.dumps(out.to_dict(), indent=2))

    # Test abstention
    abstain_sample = """<think>hard</think>
<verify>not sure</verify>
<answer>abstain</answer>
<confidence>0.15</confidence>"""
    out2 = parse_metacog_output(abstain_sample)
    print("\n--- abstention ---")
    print(json.dumps(out2.to_dict(), indent=2))
