"""RL training loop for Metacog Phase 1.

This is the most complex file in the project. It implements a REINFORCE
with baseline + importance sampling loss, in the style of Tinker's
verifiable-reward RL recipes.

The loop is:

  for step in range(n_steps):
      1. Roll out: for each question in the batch, sample K completions
         from the current policy. Score each.
      2. Compute advantages: within-group normalized.
      3. Build training data: convert each (prompt, completion, sampled_logprobs, advantage)
         into a tinker.Datum (right-shifted model input, left-shifted targets).
      4. forward_backward: compute loss on the batch.
      5. optim_step: update the LoRA weights.
      6. (every N steps) save_state: snapshot the LoRA.
      7. (every M steps) evaluate: run the eval suite on the saved policy.

We deliberately don't use the full tinker_cookbook.rl.train pipeline because
it expects a specific Env interface that's overkill for our use case. We
inline what we need.

Loss function: importance_sampling with per-token advantages. The
reference is the policy that produced the rollouts (which is the same
as the current policy on the first iter, and `prev_iter` on subsequent
iters — Tinker's IS estimator handles off-policy correction via the
sampled logprobs).

Key implementation detail: every rollouts batch needs a corresponding
"save and reload" of the sampling client to ensure the sampling
matches the policy that produced the logprobs. Tinker's cookbook does
this with `save_weights_for_sampler` after each optim_step.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import tinker
from tinker import types as ttypes
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
    create_rightshifted_model_input_and_leftshifted_targets,
)

from data.load_datasets import load_gsm8k, load_mmlu_pro
from data.schemas import QuestionRecord
from eval.metrics import calibration_report, format_report
from eval.normalize import is_correct
from prompts.chat_template import build_chat_messages
from training.reward import RewardBreakdown, RewardConfig, compute_reward
from training.rollout import (
    Rollout,
    group_advantages,
    render_prompt_as_tokens,
    rollout_for_question,
)

logger = logging.getLogger("metacog.rl_loop")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RLConfig:
    """All hyperparameters for the RL loop."""

    # Model
    base_model: str = "Qwen/Qwen3.6-35B-A3B"
    lora_rank: int = 32
    lora_seed: int = 0

    # Optimization
    learning_rate: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    kl_penalty_coef: float = 0.0  # disabled for now; can add later

    # Rollout
    n_train_questions: int = 50  # per training step
    n_gsm8k: int = 25
    n_mmlu_pro: int = 25
    group_size: int = 4  # K samples per question
    max_tokens: int = 600
    temperature: float = 1.0
    effort: str = "medium"
    stop_sequences: list[str] = field(default_factory=lambda: ["<|im_end|>"])

    # Loop
    n_steps: int = 20  # number of optim_steps
    save_every: int = 5
    eval_every: int = 5
    eval_n_questions: int = 20  # small subset for in-loop eval
    eval_temperature: float = 0.0  # greedy for eval

    # Reward
    reward: RewardConfig = field(default_factory=RewardConfig)

    # Output
    out_dir: Path = field(default_factory=lambda: Path("results/phase1"))
    log_every: int = 1

    # Loss function type
    # importance_sampling = REINFORCE with IS ratio
    # cispo = Clipped IS (more stable, recommended for VR)
    # ppo = PPO-clip
    # dro = Distributionally Robust Optimization
    loss_fn: str = "importance_sampling"


# ---------------------------------------------------------------------------
# Per-step training data builder
# ---------------------------------------------------------------------------

def _strip_mask(datum: tinker.Datum) -> tinker.Datum:
    """Strip the `mask` field from a Datum.

    Tinker's `importance_sampling` and `cispo` loss functions don't accept
    a `mask` field — they derive the mask internally from the logprobs.
    The cookbook does this stripping before every `forward_backward` call.
    See `_remove_mask` in tinker_cookbook/rl/train.py.
    """
    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"},
    )


def build_datum_for_rollout(
    prompt_tokens: list[int],
    completion_tokens: list[int],
    sampled_logprobs: list[float],
    advantage: float,
) -> tinker.Datum:
    """Build a single tinker.Datum for the training step.

    The Datum is what `forward_backward` consumes. It contains:
      - model_input: right-shifted full sequence (prompt + completion)
      - loss_fn_inputs:
        - target_tokens: left-shifted target ids
        - logprobs: the logprob under the BEHAVIOR policy (sampling policy)
        - advantages: the per-token advantage
        - mask: 1.0 for completion tokens, 0.0 for prompt tokens

    Length accounting (matches tinker_cookbook's trajectory_to_data):
      - N = len(prompt_tokens) + len(completion_tokens)  (full sequence)
      - model_input has length N-1
      - target_tokens has length N-1
      - logprobs/advantages/mask must also be length N-1, shifted by 1
        (the prediction at position t uses the logprob at position t+1)
    """
    p_len = len(prompt_tokens)
    c_len = len(completion_tokens)
    N = p_len + c_len

    # Truncate logprobs to match completion length
    sampled_lp = list(sampled_logprobs[:c_len])
    if len(sampled_lp) < c_len:
        sampled_lp = sampled_lp + [0.0] * (c_len - len(sampled_lp))

    # Build the full sequence
    full = list(prompt_tokens) + list(completion_tokens)
    mi, target_tokens = create_rightshifted_model_input_and_leftshifted_targets(
        [ttypes.EncodedTextChunk(tokens=full)]
    )
    # target_tokens has length N-1

    # Build logprobs/advantages/mask of length N
    # (prompt positions get 0, completion positions get the actual value)
    full_logprobs = [0.0] * p_len + sampled_lp   # length N
    full_advantages = [0.0] * p_len + [float(advantage)] * c_len  # length N
    full_mask = [0.0] * p_len + [1.0] * c_len   # length N

    # Shift by 1: prediction at position t uses logprob at position t+1
    # After the shift, lengths are N-1 (matching target_tokens)
    logprobs_T = full_logprobs[1:]
    advantages_T = full_advantages[1:]
    mask_T = full_mask[1:]

    return tinker.Datum(
        model_input=mi,
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=target_tokens, dtype="int64"),
            "logprobs": tinker.TensorData(data=logprobs_T, dtype="float32"),
            "advantages": tinker.TensorData(data=advantages_T, dtype="float32"),
            "mask": tinker.TensorData(data=mask_T, dtype="float32"),
        },
    )


# ---------------------------------------------------------------------------
# Eval helper
# ---------------------------------------------------------------------------

def evaluate_policy(
    sampling_client: tinker.SamplingClient,
    tokenizer,
    questions: list[QuestionRecord],
    max_tokens: int = 600,
    temperature: float = 0.0,
    effort: str = "medium",
) -> dict:
    """Evaluate the current policy on a small question set. Greedy by default."""
    rollouts: list[Rollout] = []
    for q in questions:
        try:
            rs = rollout_for_question(
                sampling_client=sampling_client,
                tokenizer=tokenizer,
                question=q,
                k=1,
                max_tokens=max_tokens,
                temperature=temperature,
                effort=effort,
            )
            rollouts.extend(rs)
        except Exception as e:
            logger.warning(f"eval rollout failed for {q.id}: {e}")

    # Aggregate
    correct = [r.breakdown.correct for r in rollouts]
    confs = [r.breakdown.confidence or 0.5 for r in rollouts]
    n = len(rollouts)
    acc = sum(correct) / n if n else 0.0
    abstained = sum(1 for r in rollouts if r.breakdown.abstained)
    avg_conf = sum(confs) / n if n else 0.0
    if n:
        rep = calibration_report(confs, correct)
    else:
        rep = None

    return {
        "n": n,
        "accuracy": round(acc, 4),
        "abstain_rate": round(abstained / n, 4) if n else 0.0,
        "avg_confidence": round(avg_conf, 4),
        "ece": round(rep.ece, 4) if rep else None,
        "brier": round(rep.brier, 4) if rep else None,
        "auroc": round(rep.auroc, 4) if rep and rep.auroc is not None else None,
        "rollouts": [r.to_dict() for r in rollouts],
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_phase1(
    service_client: tinker.ServiceClient,
    cfg: RLConfig,
    on_step: Callable[[int, dict], None] | None = None,
) -> dict:
    """Run the full Phase 1 RL training loop.

    Parameters
    ----------
    service_client : tinker.ServiceClient
    cfg : RLConfig
    on_step : callable | None
        Optional callback called after each step with (step, summary).

    Returns
    -------
    dict
        Final training summary (all step summaries, plus the final eval).
    """
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"phase1_{time.strftime('%Y%m%d_%H%M%S')}.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"=== Metacog Phase 1 RL ===")
    log(f"  base_model: {cfg.base_model}")
    log(f"  lora_rank: {cfg.lora_rank}, group_size: {cfg.group_size}, lr: {cfg.learning_rate}")
    log(f"  n_steps: {cfg.n_steps}, save_every: {cfg.save_every}, eval_every: {cfg.eval_every}")
    log(f"  loss_fn: {cfg.loss_fn}, temperature: {cfg.temperature}")
    log(f"  out_dir: {out_dir}")

    # ── 1. Create training client (LoRA on the base model) ──────────────
    log("Creating LoRA training client...")
    training_client = service_client.create_lora_training_client(
        base_model=cfg.base_model,
        rank=cfg.lora_rank,
        seed=cfg.lora_seed,
    )
    tokenizer = training_client.get_tokenizer()
    log(f"  tokenizer: {type(tokenizer).__name__}, vocab={tokenizer.vocab_size if hasattr(tokenizer,'vocab_size') else '?'}")

    # ── 2. Load training questions ──────────────────────────────────────
    questions: list[QuestionRecord] = []
    if cfg.n_gsm8k > 0:
        questions.extend(load_gsm8k(cfg.n_gsm8k))
    if cfg.n_mmlu_pro > 0:
        questions.extend(load_mmlu_pro(cfg.n_mmlu_pro))
    log(f"  loaded {len(questions)} training questions")

    # Small eval set
    eval_questions: list[QuestionRecord] = []
    if cfg.eval_n_questions > 0:
        eval_questions.extend(load_gsm8k(min(10, cfg.eval_n_questions // 2)))
        eval_questions.extend(load_mmlu_pro(min(10, cfg.eval_n_questions // 2)))
    log(f"  loaded {len(eval_questions)} eval questions")

    # ── 3. Initial eval (baseline) ──────────────────────────────────────
    log("Initial eval (baseline of starting policy)...")
    # Save weights to get a sampling client for the base LoRA
    save_resp = training_client.save_weights_for_sampler(name="phase1_init").result()
    sampling_client = training_client.create_sampling_client(save_resp.path)
    init_eval = evaluate_policy(sampling_client, tokenizer, eval_questions, max_tokens=cfg.max_tokens, temperature=cfg.eval_temperature, effort=cfg.effort)
    log(f"  INIT  acc={init_eval['accuracy']:.3f}  ECE={init_eval['ece']}  avg_conf={init_eval['avg_confidence']:.3f}  abst={init_eval['abstain_rate']:.3f}")

    # ── 4. RL loop ──────────────────────────────────────────────────────
    step_summaries: list[dict] = [{"step": -1, "label": "init_eval", **init_eval}]
    checkpoint_paths: list[str] = []

    for step in range(cfg.n_steps):
        t_step = time.time()
        log(f"\n--- step {step+1}/{cfg.n_steps} ---")

        # 4a. Roll out K samples for each question
        all_rollouts: list[Rollout] = []
        for q in questions:
            try:
                rs = rollout_for_question(
                    sampling_client=sampling_client,
                    tokenizer=tokenizer,
                    question=q,
                    k=cfg.group_size,
                    max_tokens=cfg.max_tokens,
                    temperature=cfg.temperature,
                    effort=cfg.effort,
                    reward_cfg=cfg.reward,
                )
                all_rollouts.extend(rs)
            except Exception as e:
                logger.warning(f"rollout failed for {q.id}: {e}")
                continue
        if not all_rollouts:
            log(f"  no rollouts succeeded this step; skipping")
            continue

        rewards = [r.reward for r in all_rollouts]
        log(f"  rollouts: {len(all_rollouts)}, reward mean={np.mean(rewards):.3f}, std={np.std(rewards):.3f}")
        # Break down by primary outcome
        from collections import Counter
        outcomes = Counter(r.breakdown.primary_outcome for r in all_rollouts)
        log(f"  outcomes: {dict(outcomes)}")

        # 4b. Compute advantages, group by question
        by_q: dict[str, list[Rollout]] = {}
        for r in all_rollouts:
            by_q.setdefault(r.question_id, []).append(r)
        # Map rollout → advantage
        rollout_adv: dict[int, float] = {}
        for qid, group in by_q.items():
            advs = group_advantages(group)
            for r, a in zip(group, advs):
                rollout_adv[id(r)] = a

        # 4c. Build Datum for each rollout
        data: list[tinker.Datum] = []
        for r in all_rollouts:
            if id(r) not in rollout_adv:
                continue
            adv = rollout_adv[id(r)]
            # Re-render the prompt
            messages = build_chat_messages(r.question_id, effort=r.effort, include_fewshot=True)  # re-uses fewshot, ok
            # Actually, we don't have the question here. Re-load:
            # Better: stash the messages on the rollout
            # Simpler: just look up the question from the questions list
            q_obj = next((q for q in questions if q.id == r.question_id), None)
            if q_obj is None:
                continue
            messages = build_chat_messages(q_obj.question, effort=r.effort, include_fewshot=True)
            prompt_tokens, _ = render_prompt_as_tokens(messages, tokenizer)
            # The completion tokens: we don't have them stored; we stored
            # raw_output (text) and sampled_logprobs. Re-encode raw_output.
            completion_tokens = tokenizer.encode(r.raw_output, add_special_tokens=False)
            if not completion_tokens:
                continue
            # The sampled logprobs might have a length mismatch (one per
            # generated token). Trim to match.
            sampled_lp = r.sampled_logprobs[: len(completion_tokens)]
            if len(sampled_lp) < len(completion_tokens):
                # pad with zeros
                sampled_lp = sampled_lp + [0.0] * (len(completion_tokens) - len(sampled_lp))
            datum = build_datum_for_rollout(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                sampled_logprobs=sampled_lp,
                advantage=adv,
            )
            data.append(datum)

        log(f"  training data: {len(data)} datums")

        # 4d. forward_backward
        try:
            # loss_fn_config values must be floats (per tinker spec)
            # importance_sampling: no extra config needed
            loss_fn_config = None
            if cfg.loss_fn == "cispo":
                loss_fn_config = {"clip_ratio": 0.2}
            elif cfg.loss_fn == "ppo":
                loss_fn_config = {"clip_ratio": 0.2, "kl_penalty_coef": 0.0}
            fb = training_client.forward_backward(
                data=[_strip_mask(d) for d in data],  # importance_sampling doesn't accept `mask`
                loss_fn=cfg.loss_fn,
                loss_fn_config=loss_fn_config,
            )
            fb = fb.result()
            # `metrics` dict is the canonical place for scalar metrics like loss
            loss = float(fb.metrics.get("loss", 0.0)) if fb.metrics else 0.0
        except Exception as e:
            log(f"  forward_backward FAILED: {e}")
            continue

        # 4e. optim_step
        try:
            opt = training_client.optim_step(
                ttypes.AdamParams(
                    learning_rate=cfg.learning_rate,
                    beta1=cfg.adam_beta1,
                    beta2=cfg.adam_beta2,
                    eps=cfg.adam_eps,
                    weight_decay=cfg.weight_decay,
                    grad_clip_norm=cfg.grad_clip_norm,
                )
            )
            opt = opt.result()
        except Exception as e:
            log(f"  optim_step FAILED: {e}")
            continue

        # 4f. Save weights and update the sampling client
        try:
            save_resp = training_client.save_weights_for_sampler(name=f"phase1_step{step+1}").result()
            sampling_client = training_client.create_sampling_client(save_resp.path)
        except Exception as e:
            log(f"  save_weights_for_sampler FAILED: {e}")

        step_time = time.time() - t_step
        log(f"  loss={loss:.4f}, step_time={step_time:.1f}s")

        step_summary = {
            "step": step,
            "n_rollouts": len(all_rollouts),
            "reward_mean": round(float(np.mean(rewards)), 4),
            "reward_std": round(float(np.std(rewards)), 4),
            "outcomes": dict(outcomes),
            "loss": round(float(loss), 4),
            "step_time_s": round(step_time, 1),
        }
        step_summaries.append(step_summary)
        if on_step:
            on_step(step, step_summary)

        # 4g. Save checkpoint at intervals
        if (step + 1) % cfg.save_every == 0 or step == cfg.n_steps - 1:
            try:
                sv = training_client.save_state(name=f"phase1_step{step+1}").result()
                checkpoint_paths.append(sv.path)
                log(f"  saved checkpoint: {sv.path}")
            except Exception as e:
                log(f"  save_state FAILED: {e}")

        # 4h. Eval at intervals
        if (step + 1) % cfg.eval_every == 0 or step == cfg.n_steps - 1:
            log(f"  in-loop eval (n={len(eval_questions)})...")
            ev = evaluate_policy(sampling_client, tokenizer, eval_questions, max_tokens=cfg.max_tokens, temperature=cfg.eval_temperature, effort=cfg.effort)
            log(f"  EVAL  acc={ev['accuracy']:.3f}  ECE={ev['ece']}  avg_conf={ev['avg_confidence']:.3f}  abst={ev['abstain_rate']:.3f}")
            step_summaries.append({"step": step, "label": "eval", **ev})

    # ── 5. Final eval ───────────────────────────────────────────────────
    log("\n=== Final eval ===")
    final_eval = evaluate_policy(sampling_client, tokenizer, eval_questions, max_tokens=cfg.max_tokens, temperature=cfg.eval_temperature, effort=cfg.effort)
    log(f"  FINAL  acc={final_eval['accuracy']:.3f}  ECE={final_eval['ece']}  avg_conf={final_eval['avg_confidence']:.3f}  abst={final_eval['abstain_rate']:.3f}")
    step_summaries.append({"step": cfg.n_steps, "label": "final_eval", **final_eval})

    # ── 6. Persist summary ──────────────────────────────────────────────
    summary = {
        "config": {k: (str(v) if isinstance(v, Path) else (asdict(v) if hasattr(v, '__dataclass_fields__') else v)) for k, v in asdict(cfg).items()},
        "checkpoint_paths": checkpoint_paths,
        "step_summaries": step_summaries,
        "init_eval": init_eval,
        "final_eval": final_eval,
    }
    summary_path = out_dir / "phase1_summary.json"
    # Serialize safely: tensors, numpy → str
    def safe(o):
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=safe)
    log(f"\nSaved summary → {summary_path}")
    return summary
