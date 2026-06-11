"""Rollout module: sample K completions from the current policy and score them.

This is the *inner loop* of the RL training. For each question:
  1. Format the prompt via the chat template
  2. Sample K completions from the SamplingClient (which wraps the current LoRA)
  3. Decode each completion
  4. Compute the reward for each (raw_output, question, n_tokens) triple
  5. Return the list of (text, tokens, sampled_logprobs, reward) tuples

The SamplingClient is tinker-typed; the sampling is done via the API.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import tinker
from tinker import types as ttypes

from data.schemas import QuestionRecord
from prompts.chat_template import build_chat_messages
from training.reward import RewardBreakdown, RewardConfig, compute_reward


# ---------------------------------------------------------------------------
# Rollout record
# ---------------------------------------------------------------------------

@dataclass
class Rollout:
    """A single model rollout for one question."""

    question_id: str
    raw_output: str
    n_tokens: int
    sampled_logprobs: list[float]
    reward: float
    breakdown: RewardBreakdown
    latency_s: float = 0.0
    effort: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self.breakdown)
        d["question_id"] = self.question_id
        d["raw_output"] = self.raw_output
        d["n_tokens"] = self.n_tokens
        d["sampled_logprobs"] = self.sampled_logprobs
        d["reward"] = self.reward
        d["latency_s"] = self.latency_s
        d["effort"] = self.effort
        return d


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def render_prompt_as_tokens(
    messages: list[dict],
    tokenizer,
) -> tuple[list[int], list[int]]:
    """Render a list of chat messages into prompt-only tokens.

    The trick: we render the FULL conversation INCLUDING a placeholder
    assistant turn (just the `<|assistant|>` tag and a newline) — this is
    what the model sees as "input". The training loss is only on the
    assistant tokens that come AFTER.

    For the SAMPLING path (rollout), we want ONLY the prompt tokens (no
    placeholder assistant yet — the model generates everything).

    For the TRAINING path (forward_backward), we need both:
      - input_tokens = full sequence up to and including the assistant tag
      - target_tokens = the model's generated tokens

    Returns (prompt_tokens, prompt_only_tokens) for the sampling case.
    The "prompt_only" is what we feed to the sampler; the training
    data builder (in rl_loop.py) builds the proper right-shifted Datum.
    """
    # We use a simple approach: render messages up to the last user turn,
    # then add the assistant tag. For Qwen3.6, the chat template uses
    # <|im_start|>role\ncontent<|im_end|>\n. We approximate this if the
    # tokenizer doesn't have apply_chat_template.
    try:
        # Try the tokenizer's chat template
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback: manual Qwen-style template
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        # Add assistant turn opener
        parts.append("<|im_start|>assistant\n")
        prompt_text = "".join(parts)

    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    return prompt_tokens, prompt_text


# ---------------------------------------------------------------------------
# Main rollout
# ---------------------------------------------------------------------------

def rollout_for_question(
    sampling_client: tinker.SamplingClient,
    tokenizer,
    question: QuestionRecord,
    k: int = 4,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    effort: str = "medium",
    reward_cfg: RewardConfig | None = None,
    stop: list[str] | None = None,
) -> list[Rollout]:
    """Sample K completions for one question and score them.

    Parameters
    ----------
    sampling_client : tinker.SamplingClient
        Current LoRA policy (or base model for Phase 0).
    tokenizer
        HuggingFace tokenizer (from tinker.get_tokenizer()).
    question : QuestionRecord
    k : int
        Group size — K samples per question for variance reduction.
    max_tokens : int
    temperature : float
        Sampling temperature. ≥ 1.0 for exploration, 0.0 for greedy.
    effort : str
        "low" | "medium" | "high" — passed to the prompt.
    reward_cfg : RewardConfig | None
    stop : list[str] | None
        Stop sequences (e.g. ["<|im_end|>"]).

    Returns
    -------
    list[Rollout]
        Length k (or fewer if some samples errored).
    """
    reward_cfg = reward_cfg or RewardConfig()

    # Build prompt
    messages = build_chat_messages(question.question, effort=effort, include_fewshot=True)
    prompt_tokens, prompt_text = render_prompt_as_tokens(messages, tokenizer)
    mi = ttypes.ModelInput(chunks=[ttypes.EncodedTextChunk(tokens=prompt_tokens)])

    sp = ttypes.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        top_k=50,
        stop=stop or ["<|im_end|>"],
    )

    # Sample
    t0 = time.time()
    try:
        resp = sampling_client.sample(prompt=mi, num_samples=k, sampling_params=sp)
        if hasattr(resp, "result"):
            resp = resp.result()
    except Exception as e:
        # Surface the error so the training loop can decide
        raise RuntimeError(f"tinker.sample failed for {question.id}: {e}") from e
    latency = time.time() - t0

    # Decode + score each
    rollouts: list[Rollout] = []
    for seq in resp.sequences:
        token_ids = list(seq.tokens) if seq.tokens is not None else []
        logprobs = [float(lp) for lp in (seq.logprobs or []) if lp is not None]
        # Decode
        try:
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
        except Exception:
            text = ""

        # Score
        reward, br, _ = compute_reward(text, question, n_completion_tokens=len(token_ids), cfg=reward_cfg)

        rollouts.append(
            Rollout(
                question_id=question.id,
                raw_output=text,
                n_tokens=len(token_ids),
                sampled_logprobs=logprobs,
                reward=reward,
                breakdown=br,
                latency_s=round(latency / k, 4),
                effort=effort,
            )
        )
    return rollouts


# ---------------------------------------------------------------------------
# Group statistics
# ---------------------------------------------------------------------------

def group_advantages(rollouts: list[Rollout]) -> list[float]:
    """Compute within-group normalized advantages.

    For each rollout: advantage = (reward - group_mean) / (group_std + 1e-8)
    The training loss is then weighted by the advantage. Positive advantage
    → reinforce this trajectory. Negative advantage → suppress.

    This is the standard REINFORCE-with-baseline trick.
    """
    if not rollouts:
        return []
    rewards = np.array([r.reward for r in rollouts], dtype=np.float64)
    if rewards.std() < 1e-8:
        # All rewards are the same — no learning signal
        return [0.0] * len(rollouts)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    return advantages.tolist()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Smoke test: render prompt, don't actually call the API
    from data.load_datasets import load_gsm8k
    qs = load_gsm8k(1)
    q = qs[0]
    messages = build_chat_messages(q.question, effort="medium", include_fewshot=True)
    print(f"messages: {len(messages)}")
    # Try to find the actual tokenizer
    print("Need real tokenizer to test fully. Will test via run_phase1.py.")
