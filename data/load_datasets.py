"""Real dataset loaders for GSM8K and MMLU-Pro.

Pulls from HuggingFace, normalizes to our QuestionRecord schema, and
saves as JSONL to data/ for the baseline runner and any downstream
training.

Usage:
    python -m data.load_datasets --n-gsm8k 50 --n-mmlu-pro 50
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .schemas import QuestionRecord, Source, save_jsonl

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / "data"


def load_gsm8k(n: int, split: str = "test") -> list[QuestionRecord]:
    """Load GSM8K from HuggingFace `openai/gsm8k` and return QuestionRecords.

    GSM8K's `test` split has 1319 examples. Gold answers are extracted
    from the `#### N` tail of the answer field.
    """
    from datasets import load_dataset

    print(f"[gsm8k] loading {split} split (capping at {n})...")
    ds = load_dataset("openai/gsm8k", "main", split=split)
    out: list[QuestionRecord] = []
    for i, row in enumerate(ds):
        if len(out) >= n:
            break
        ans_text = row["answer"]
        # GSM8K gold answer is everything after the last '####'
        if "####" not in ans_text:
            continue
        gold = ans_text.split("####")[-1].strip().replace(",", "")
        # Drop trailing .0
        if gold.endswith(".0"):
            gold = gold[:-2]
        out.append(
            QuestionRecord(
                id=f"gsm8k-{i:05d}",
                source="gsm8k",
                domain="arithmetic",
                question=row["question"],
                gold_answer=gold,
                gold_rationale=ans_text,
                meta={"split": split, "raw_index": i},
            )
        )
    print(f"[gsm8k] loaded {len(out)} records")
    return out


def load_mmlu_pro(n: int, split: str = "test") -> list[QuestionRecord]:
    """Load MMLU-Pro from HuggingFace `TIGER-Lab/MMLU-Pro`.

    MMLU-Pro has 10 options A-J (vs MMLU's 4). Gold answer is a letter.
    """
    from datasets import load_dataset

    print(f"[mmlu_pro] loading {split} split (capping at {n})...")
    try:
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    except Exception as e:
        print(f"[mmlu_pro] standard loader failed ({e}); trying test split fallback")
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
    out: list[QuestionRecord] = []
    for i, row in enumerate(ds):
        if len(out) >= n:
            break
        # MMLU-Pro fields: question, options (list), answer (int), category
        opts = row.get("options") or []
        ans_idx = row.get("answer")
        if ans_idx is None or not opts:
            continue
        # MMLU-Pro stores the answer as a letter directly (A-J)
        gold_letter = str(ans_idx).strip().upper()
        # Format question with options inline (the format the model sees)
        opts_text = "\n".join(f"{chr(ord('A')+j)}. {opt}" for j, opt in enumerate(opts))
        question_text = f"{row['question']}\n\n{opts_text}"
        out.append(
            QuestionRecord(
                id=f"mmlu_pro-{i:05d}",
                source="mmlu_pro",
                domain=str(row.get("category", "general")).lower(),
                question=question_text,
                gold_answer=gold_letter,
                gold_rationale=row.get("cot_content") or None,
                meta={"split": split, "raw_index": i, "num_options": len(opts)},
            )
        )
    print(f"[mmlu_pro] loaded {len(out)} records")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-gsm8k", type=int, default=50, help="Number of GSM8K records")
    p.add_argument("--n-mmlu-pro", type=int, default=50, help="Number of MMLU-Pro records")
    p.add_argument("--out", type=str, default="data/questions.jsonl")
    args = p.parse_args()

    questions: list[QuestionRecord] = []
    if args.n_gsm8k > 0:
        questions.extend(load_gsm8k(args.n_gsm8k))
    if args.n_mmlu_pro > 0:
        questions.extend(load_mmlu_pro(args.n_mmlu_pro))

    out_path = _PROJECT_ROOT / args.out
    save_jsonl(questions, out_path)
    print(f"[done] wrote {len(questions)} questions to {out_path}")


if __name__ == "__main__":
    main()
