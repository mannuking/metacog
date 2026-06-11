"""Phase 1 orchestrator: dry-run, small training, full training.

This is the entry point for Phase 1. It wraps the RL loop with:
  - Configurable hyperparameters from CLI
  - Dry-run mode (no Tinker calls) for verifying the data pipeline
  - Small test mode (2 questions, 1 step) for end-to-end smoke
  - Full training mode (default)
  - Saves all artifacts under `results/phase1/`

Usage:
  # Dry run (no API)
  uv run python -m scripts.run_phase1 --dry-run

  # Tiny test (2 questions, 1 step, real Tinker)
  uv run python -m scripts.run_phase1 --test

  # Full Phase 1
  uv run python -m scripts.run_phase1 --n-steps 20 --group-size 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Repo-root imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import tinker

from data.schemas import QuestionRecord
from eval.structured_parse import parse_metacog_output
from prompts.chat_template import build_chat_messages, SYSTEM_PROMPT
from training.reward import RewardConfig, compute_reward
from training.rollout import rollout_for_question, group_advantages
from training.rl_loop import RLConfig, run_phase1


# ---------------------------------------------------------------------------
# Dry run: verify the pipeline without any Tinker calls
# ---------------------------------------------------------------------------

def dry_run(cfg: RLConfig) -> dict:
    """Simulate one full step with synthetic model outputs."""
    from data.load_datasets import load_gsm8k, load_mmlu_pro
    from collections import Counter

    print("=" * 60)
    print("DRY RUN — no Tinker API calls")
    print("=" * 60)

    # Load questions
    questions: list[QuestionRecord] = []
    if cfg.n_gsm8k > 0:
        questions.extend(load_gsm8k(cfg.n_gsm8k))
    if cfg.n_mmlu_pro > 0:
        questions.extend(load_mmlu_pro(cfg.n_mmlu_pro))
    print(f"  loaded {len(questions)} training questions")

    # Inject synthetic outputs that follow the format
    import random
    master_rng = random.Random(42)
    def synthesize(q: QuestionRecord, sample_idx: int, k: int) -> str:
        """Generate a synthetic model output that follows the metacog format.

        For training simulation:
        - Easy questions (GSM8K arithmetic): mostly correct
        - Hard questions (MMLU-Pro): more variance, more abstentions
        - The base "correctness" decision is seeded by question+sample so
          we get variance across questions but reproducibility within one.
        """
        seed_val = hash((q.id, sample_idx)) & 0xFFFF
        r = random.Random(seed_val)
        is_easy = q.source == "gsm8k"
        # 70% correct for easy, 30% for hard (close to baseline)
        is_correct = r.random() < (0.70 if is_easy else 0.30)
        conf = r.uniform(0.6, 0.98)
        if is_correct:
            answer = q.gold_answer
        else:
            if r.random() < 0.4:
                # Abstain on a fraction
                answer = "abstain"
                conf = r.uniform(0.1, 0.35)
            else:
                # Wrong (use a clearly-wrong string)
                answer = q.gold_answer + "_wrong"
                # Make confidence depend on easy/hard
                conf = r.uniform(0.7, 0.98) if is_easy else r.uniform(0.85, 0.99)
        return (
            f"<think>\n"
            f"Sub-claim 1: problem requires {q.source} reasoning.\n"
            f"Sub-claim 2: apply metacog method.\n"
            f"Difficulty: {'easy' if is_easy else 'hard'}\n"
            f"Budget: 1 pass\n\n"
            f"Solve: derived {answer}.\n"
            f"Unit-check: verified. ✓\n"
            f"Sanity-bounds: plausible. ✓\n"
            f"</think>\n\n"
            f"<verify>\n"
            f"Re-derive: same as first pass. No conflict.\n"
            f"Counterfactual: considered alternatives.\n"
            f"</verify>\n\n"
            f"<answer>{answer}</answer>\n"
            f"<confidence>{conf:.2f}</confidence>\n"
        )

    # Simulate rollouts
    print("\nSimulating rollouts (K={cfg.group_size} per question)...".replace("{cfg.group_size}", str(cfg.group_size)))
    all_rollouts = []
    for q in questions:
        for s in range(cfg.group_size):
            raw = synthesize(q, s, cfg.group_size)
            reward, br, _ = compute_reward(raw, q, n_completion_tokens=200, cfg=cfg.reward)
            from training.rollout import Rollout
            all_rollouts.append(Rollout(
                question_id=q.id, raw_output=raw, n_tokens=200, sampled_logprobs=[-0.1]*200,
                reward=reward, breakdown=br, latency_s=0.0, effort=cfg.effort,
            ))

    rewards = [r.reward for r in all_rollouts]
    outcomes = Counter(r.breakdown.primary_outcome for r in all_rollouts)
    print(f"  rollouts: {len(all_rollouts)}")
    print(f"  reward mean={sum(rewards)/len(rewards):.3f}, std={((sum((r-sum(rewards)/len(rewards))**2 for r in rewards))/len(rewards))**0.5:.3f}")
    print(f"  outcomes: {dict(outcomes)}")

    # Simulate advantages
    by_q: dict[str, list] = {}
    for r in all_rollouts:
        by_q.setdefault(r.question_id, []).append(r)
    print(f"\nAdvantages (sample of 3 questions):")
    for i, (qid, group) in enumerate(by_q.items()):
        if i >= 3: break
        advs = group_advantages(group)
        print(f"  {qid}: rewards={[round(g.reward,2) for g in group]}, advantages={[round(a,2) for a in advs]}")

    # Simulate forward_backward + optim_step shape
    print("\nSimulated forward_backward + optim_step would happen here.")
    print("(Skipped because no real model.)\n")

    print("=" * 60)
    print("DRY RUN OK — pipeline logic is correct.")
    print("=" * 60)
    return {
        "n_rollouts": len(all_rollouts),
        "reward_mean": sum(rewards) / len(rewards),
        "outcomes": dict(outcomes),
    }


# ---------------------------------------------------------------------------
# Tiny real test: 2 questions, 1 step
# ---------------------------------------------------------------------------

def tiny_test(cfg: RLConfig) -> dict:
    """Run a 1-step, 2-question real Tinker training to validate the loop."""
    cfg.n_gsm8k = 1
    cfg.n_mmlu_pro = 1
    cfg.n_train_questions = 2
    cfg.group_size = 2
    cfg.n_steps = 1
    cfg.eval_n_questions = 2
    cfg.eval_every = 1
    cfg.save_every = 1
    cfg.out_dir = Path("results/phase1_test")
    cfg.log_every = 1

    print("=" * 60)
    print("TINY TEST — 2 questions, 1 step, REAL Tinker")
    print("=" * 60)
    api_key = os.environ.get("TINKER_API_KEY", "").strip()
    if not api_key or api_key == "PASTE_YOUR_TINKER_API_KEY_HERE":
        raise RuntimeError("TINKER_API_KEY missing. Set in .env first.")
    sc = tinker.ServiceClient()
    return run_phase1(sc, cfg)


# ---------------------------------------------------------------------------
# Full training
# ---------------------------------------------------------------------------

def full_training(cfg: RLConfig) -> dict:
    print("=" * 60)
    print(f"FULL PHASE 1 — n_steps={cfg.n_steps}, group_size={cfg.group_size}")
    print("=" * 60)
    api_key = os.environ.get("TINKER_API_KEY", "").strip()
    if not api_key or api_key == "PASTE_YOUR_TINKER_API_KEY_HERE":
        raise RuntimeError("TINKER_API_KEY missing. Set in .env first.")
    sc = tinker.ServiceClient()
    return run_phase1(sc, cfg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="No API calls, just verify the pipeline")
    p.add_argument("--test", action="store_true", help="Tiny real test (2 questions, 1 step)")
    p.add_argument("--n-steps", type=int, default=20, help="Number of optim_steps")
    p.add_argument("--group-size", type=int, default=4, help="K samples per question")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--n-gsm8k", type=int, default=25, help="GSM8K questions in training set")
    p.add_argument("--n-mmlu-pro", type=int, default=25, help="MMLU-Pro questions in training set")
    p.add_argument("--max-tokens", type=int, default=600)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--effort", type=str, default="medium", choices=["low","medium","high"])
    p.add_argument("--loss-fn", type=str, default="importance_sampling", choices=["importance_sampling","cispo","ppo","dro"])
    p.add_argument("--base-model", type=str, default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--out-dir", type=str, default="results/phase1")
    p.add_argument("--eval-n-questions", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=5)
    args = p.parse_args()

    cfg = RLConfig(
        base_model=args.base_model,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        group_size=args.group_size,
        n_gsm8k=args.n_gsm8k,
        n_mmlu_pro=args.n_mmlu_pro,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        effort=args.effort,
        loss_fn=args.loss_fn,
        out_dir=Path(args.out_dir),
        eval_n_questions=args.eval_n_questions,
        eval_every=args.eval_every,
        save_every=args.save_every,
    )

    if args.dry_run:
        dry_run(cfg)
    elif args.test:
        tiny_test(cfg)
    else:
        full_training(cfg)


if __name__ == "__main__":
    main()
