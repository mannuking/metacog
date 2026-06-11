"""Answer normalization and correctness checking.

This is the most fragile part of any LLM eval. Different models emit
final answers in wildly different formats: \\boxed{42}, **42**, "The
answer is 42.", "42.", "A", "(A)", etc. We need a single normalizer
that maps all of these to a canonical form per source.

For the BASELINE phase we don't have model outputs yet — this is
imported by both the baseline runner and the future confidence-head
training. Robustness matters.
"""

from __future__ import annotations

import re
from typing import Literal

# ─────────────────────────────────────────────────────────────────────────────
# Generic numeric extraction (works for GSM8K, math benchmarks)
# ─────────────────────────────────────────────────────────────────────────────

_NEGATIVE_NUMBER = r"-?\$?\d[\d,]*(?:\.\d+)?"
_BOXED = re.compile(r"\\boxed\s*\{\s*(" + _NEGATIVE_NUMBER + r")\s*\}")
_HASH_ANSWER = re.compile(r"####\s*(" + _NEGATIVE_NUMBER + r")")
_FINAL_NUMBER = re.compile(
    r"(?:answer\s*(?:is|=|:)\s*)?(" + _NEGATIVE_NUMBER + r")\s*\.?\s*$",
    re.IGNORECASE,
)


def normalize_numeric(text: str) -> str | None:
    """Extract the canonical numeric answer from a model output.

    Tries in order:
      1. \\boxed{...}      (LaTeX)
      2. #### N           (GSM8K convention)
      3. "answer is N"    (CoT-final phrasing)
      4. Last number in text

    Returns the number with commas stripped and trailing .0 removed.
    """
    if not text:
        return None
    for pat in (_BOXED, _HASH_ANSWER, _FINAL_NUMBER):
        m = pat.search(text)
        if m:
            return _clean_num(m.group(1))
    # Last resort: any number
    nums = re.findall(_NEGATIVE_NUMBER, text)
    if nums:
        return _clean_num(nums[-1])
    return None


def _clean_num(s: str) -> str:
    s = s.replace("$", "").replace(",", "").strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Letter answer (MMLU, ARC, multiple choice)
# ─────────────────────────────────────────────────────────────────────────────

_PARENS_LETTER = re.compile(r"\(([A-J])\)")
_BARE_LETTER = re.compile(r"^[^*]*?\b([A-J])\b\s*\.?\s*$", re.MULTILINE)
_FINAL_PHRASE = re.compile(
    r"(?:answer\s*(?:is|:)\s*\*?\*?)\s*\(?([A-J])\)?",
    re.IGNORECASE,
)


def normalize_letter(text: str, max_letter: str = "J") -> str | None:
    """Extract a multiple-choice letter answer from model output.

    Handles:
      "(A)", "A", "A.", "The answer is (B).", "**C**"
    """
    if not text:
        return None
    max_idx = ord(max_letter.upper())
    # Try in order of specificity
    for pat in (_PARENS_LETTER, _FINAL_PHRASE, _BARE_LETTER):
        for m in pat.finditer(text):
            letter = m.group(1).upper()
            if ord(letter) <= max_idx:
                return letter
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-source normalizer dispatch
# ─────────────────────────────────────────────────────────────────────────────

def normalize_answer(text: str, source: str, max_letter: str = "J") -> str | None:
    """Route to the right normalizer based on source."""
    if source in ("gsm8k",):
        return normalize_numeric(text)
    if source in ("mmlu_pro", "arc_challenge", "truthfulqa"):
        return normalize_letter(text, max_letter=max_letter)
    # Fallback: try numeric first, then letter
    n = normalize_numeric(text)
    if n is not None:
        return n
    return normalize_letter(text, max_letter=max_letter)


# ─────────────────────────────────────────────────────────────────────────────
# Correctness check
# ─────────────────────────────────────────────────────────────────────────────

def is_correct(predicted: str | None, gold: str, source: str) -> bool:
    """Compare predicted vs gold with source-appropriate tolerance."""
    if predicted is None:
        return False
    if source in ("gsm8k",):
        try:
            return abs(float(predicted) - float(gold)) < 1e-6
        except (ValueError, TypeError):
            return predicted.strip() == gold.strip()
    if source in ("mmlu_pro", "arc_challenge", "truthfulqa"):
        return predicted.strip().upper() == gold.strip().upper()
    return predicted.strip() == gold.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Extract final answer from raw model output with thinking block
# ─────────────────────────────────────────────────────────────────────────────

_THINK_END = re.compile(r"<\|/?think\|?>", re.IGNORECASE)
_THINK_END_ALT = re.compile(r"</?think>", re.IGNORECASE)


def extract_final_answer(raw: str) -> str:
    """Return the text AFTER the last <|/think|> (or </think>).

    If no think tags are present, the full text is returned.
    Used to isolate the model's final answer from its reasoning.
    """
    if not raw:
        return ""
    # Try the Qwen-specific tag first, then the generic
    for pat in (_THINK_END, _THINK_END_ALT):
        # Find the LAST closing tag
        matches = list(pat.finditer(raw))
        if matches:
            last = matches[-1]
            # If it's a closing tag, take everything after
            if last.group(0).startswith("</"):
                return raw[last.end():].strip()
    return raw.strip()


if __name__ == "__main__":
    # Smoke tests
    cases = [
        ("...#### 18", "gsm8k", "18", True),
        ("\\boxed{42}", "gsm8k", "42", True),
        ("Let me think...\nThe answer is 18.", "gsm8k", "18", True),
        ("She sells 18 eggs.\n#### 18", "gsm8k", "18", True),
        ("I think it's (B).", "mmlu_pro", "B", True),
        ("B", "mmlu_pro", "B", True),
        ("The answer is **C**.", "mmlu_pro", "C", True),
        ("blah blah", "mmlu_pro", "B", False),  # wrong answer
    ]
    for raw, src, gold, expected in cases:
        norm = normalize_answer(raw, src)
        ok = is_correct(norm, gold, src)
        flag = "OK" if ok == expected else "FAIL"
        print(f"  [{flag}] {src}: raw={raw[:40]!r} -> norm={norm!r}, gold={gold}, correct={ok}")
