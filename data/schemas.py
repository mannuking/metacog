"""Dataset schemas for Metacog.

We use Pydantic (v2) for typed JSONL records. Three record types:

  QuestionRecord        — raw input from GSM8K / MMLU-Pro / synthetic
  TraceRecord           — model-generated reasoning trace + final answer
  MetacogTrainingRecord — labeled (trace, step_confidence, correctness) for training

All records serialize to JSONL with one record per line. Schemas are
intentionally minimal so we can mix sources (HuggingFace datasets, hand
written, RL rollouts) without reshaping.

Why three types? Because we want the EVALUATION pipeline to operate on
TraceRecord (what the model produced) and the TRAINING pipeline to
operate on MetacogTrainingRecord (what the model should have done).
QuestionRecord is the universal input.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ─────────────────────────────────────────────────────────────────────────────
# Source / domain enums
# ─────────────────────────────────────────────────────────────────────────────

Source = Literal[
    "gsm8k",          # grade school math, gold answer in `#### <number>`
    "mmlu_pro",       # multi-choice graduate knowledge, gold letter A-J
    "arc_challenge",  # multi-choice science, gold letter A-E
    "truthfulqa",     # multi-choice truthfulness, gold letter
    "synthetic",      # hand-written or generated for the metacog study
    "rl_rollout",     # produced by the model's own sampling during RL
]


# ─────────────────────────────────────────────────────────────────────────────
# QuestionRecord — universal input
# ─────────────────────────────────────────────────────────────────────────────

class QuestionRecord(BaseModel):
    """A single evaluation question from any source."""

    id: str = Field(..., description="Stable unique id, e.g. gsm8k-0042")
    source: Source
    domain: str = Field("", description="Free-form tag, e.g. 'arithmetic' or 'physics'")
    question: str
    # Gold answer. Format depends on source:
    #   gsm8k:     numeric string e.g. "42"
    #   mmlu_pro:  letter "A".."J"
    #   arc:       letter "A".."E"
    gold_answer: str
    # Optional chain-of-thought ground truth for the few sources that have it
    # (e.g. GSM8K's `#### answer` block, or MMLU-Pro's CoT explanations).
    gold_rationale: str | None = None
    # Arbitrary metadata for filtering / stratification
    meta: dict = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _id_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("id must be non-empty")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# TraceRecord — what the model produced for one question
# ─────────────────────────────────────────────────────────────────────────────

class ReasoningStep(BaseModel):
    """A single step in the model's reasoning trace.

    The model emits a thinking block (between <|think|> and <|/think|>)
    then a final answer (after <|/think|>). We split the thinking block
    into steps by line boundaries or by 'Step N:' markers — heuristic
    for now, can be replaced with a structured parser later.
    """

    index: int
    text: str
    # Whether this step was followed by self-correction in the next step
    has_correction_next: bool = False


class TraceRecord(BaseModel):
    """A model's full output for a question — the trace we evaluate."""

    id: str = Field(..., description="Matches QuestionRecord.id")
    question_id: str
    model: str = Field(..., description="Model name + revision")
    prompt: str = Field(..., description="Full prompt that was sent")
    # Raw model output (thinking + final answer)
    raw_output: str
    # Parsed reasoning steps (from the <|think|>...<|/think|> block)
    thinking_steps: list[ReasoningStep] = Field(default_factory=list)
    # The model's final extracted answer. Free-form string; we compare to
    # gold_answer via a normalizer (see eval/normalize.py).
    final_answer: str
    # Whether parsing succeeded (i.e. we found a <|/think|> or extractable answer)
    parse_ok: bool = True
    # Cost / latency
    n_prompt_tokens: int = 0
    n_completion_tokens: int = 0
    latency_s: float = 0.0
    # Tinker request id (for tracing back to the exact API call)
    request_id: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# MetacogTrainingRecord — labeled data for confidence-head / RL training
# ─────────────────────────────────────────────────────────────────────────────

class StepLabel(BaseModel):
    """Per-step training signal.

    `confidence_target` is the model's P(correct | step), in [0, 1].
    We compute it from self-consistency: sample K=5 completions, see how
    often the model agrees with the step's conclusion; agreement rate
    becomes the target. Calibration target is 1.0 if the step was
    correct, 0.0 if it led to a wrong final answer, with a soft blend
    based on agreement.

    `was_correct` is whether THIS STEP was the right move. We use a
    simple proxy: correct iff the final answer was correct AND this
    step was retained in the successful trace. (Better labeling possible
    with process supervision, but out of scope for the baseline.)
    """

    index: int
    text: str
    confidence_target: float = Field(ge=0.0, le=1.0)
    was_correct: bool


class MetacogTrainingRecord(BaseModel):
    """One row in the training set. Combines a trace with its confidence
    targets so the model can be fine-tuned to produce calibrated
    self-assessments at every step.

    For SFT: train on (prompt, trace_text_with_<conf>X.XX</conf>_markers)
    For RL: use the final-answer correctness as the episode reward, with
    a KL penalty against the base model.
    """

    id: str
    question_id: str
    question: str
    gold_answer: str
    source: Source
    # The trace we want the model to learn
    model: str
    raw_output: str
    thinking_steps: list[StepLabel]
    final_answer: str
    final_correct: bool
    # Aggregated self-consistency (fraction of K=5 rollouts that agreed
    # with the final answer)
    self_consistency: float = Field(ge=0.0, le=1.0)
    # Reward signal for RL
    reward: float = 0.0
    # KL divergence of the training-policy logprobs vs. the base model
    # at each step (filled in by the training loop, not the data pipeline)
    kl_per_step: list[float] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str | Path, record_type: type[BaseModel]) -> list[BaseModel]:
    """Load a JSONL file into a list of validated pydantic records."""
    out: list[BaseModel] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(record_type.model_validate_json(line))
            except Exception as e:
                raise ValueError(f"{path}:{i} parse error: {e}") from e
    return out


def save_jsonl(records: list[BaseModel], path: str | Path) -> None:
    """Save records to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# GSM8K answer normalization
# ─────────────────────────────────────────────────────────────────────────────

_GSM8K_FINAL = re.compile(r"####\s*(-?[\d,\.]+)")
_NUMERIC = re.compile(r"-?[\d]+(?:[\.,]\d+)?")


def normalize_gsm8k_answer(text: str) -> str:
    """Extract the gold answer from a GSM8K completion.

    GSM8K gold answers are appended as '#### <number>'. We strip
    commas and trailing .0 to match model outputs that drop them.
    """
    m = _GSM8K_FINAL.search(text)
    if m:
        ans = m.group(1).replace(",", "").rstrip(".")
        # Drop trailing .0 for "18.0" -> "18"
        if ans.endswith(".0"):
            ans = ans[:-2]
        return ans
    # Fallback: last number in text
    nums = _NUMERIC.findall(text)
    if nums:
        return nums[-1].replace(",", "").rstrip(".")
    return text.strip()


if __name__ == "__main__":
    # Quick smoke test of the schemas
    q = QuestionRecord(
        id="gsm8k-test-1",
        source="gsm8k",
        domain="arithmetic",
        question="Janet's ducks lay 16 eggs per day...",
        gold_answer="18",
    )
    print("QuestionRecord OK:", q.id)
    t = TraceRecord(
        id="trace-1",
        question_id=q.id,
        model="Qwen/Qwen3.6-35B-A3B",
        prompt=q.question,
        raw_output="<|think|>16 + ...<|/think|>18",
        final_answer="18",
    )
    print("TraceRecord OK:", t.final_answer)
    m = MetacogTrainingRecord(
        id="metacog-1",
        question_id=q.id,
        question=q.question,
        gold_answer=q.gold_answer,
        source=q.source,
        model=t.model,
        raw_output=t.raw_output,
        thinking_steps=[StepLabel(index=0, text="16 + ...", confidence_target=0.92, was_correct=True)],
        final_answer="18",
        final_correct=True,
        self_consistency=0.8,
        reward=1.0,
    )
    print("MetacogTrainingRecord OK:", m.self_consistency)
    print("GSM8K normalize test:", normalize_gsm8k_answer("...#### 18"))
