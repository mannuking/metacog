"""Snapshot the current Phase 1 config to paper/configs/phase1.yaml.

Run this BEFORE the run starts (per the paper directory rules). For Phase 1
it was run after the fact, but the config we used is the canonical one.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "results" / "phase1" / "phase1_summary.json"
OUT = ROOT / "paper" / "configs" / "phase1.yaml"
OUT.parent.mkdir(parents=True, exist_ok=True)

with SUMMARY.open() as f:
    data = json.load(f)

cfg = data["config"]

# Convert python repr keys to clean YAML
def fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, list):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f'"{k}": {fmt(val)}' for k, val in v.items()) + "}"
    return repr(v)

lines = [
    "# Phase 1 — RL on Tinker with verifiable metacognitive reward",
    "# Frozen at the start of the run (per paper directory rules)",
    f"# Source: {SUMMARY.relative_to(ROOT)}",
    "",
    "model:",
    f"  base: {fmt(cfg['base_model'])}",
    f"  lora_rank: {fmt(cfg['lora_rank'])}",
    f"  lora_seed: {fmt(cfg['lora_seed'])}",
    "  tokenizer_stop: ['<|im_end|>']",
    "",
    "data:",
    f"  n_gsm8k: {fmt(cfg['n_gsm8k'])}",
    f"  n_mmlu_pro: {fmt(cfg['n_mmlu_pro'])}",
    f"  n_train_total: {fmt(cfg['n_train_questions'])}",
    f"  max_tokens: {fmt(cfg['max_tokens'])}",
    f"  temperature: {fmt(cfg['temperature'])}",
    f"  effort: {fmt(cfg['effort'])}",
    "",
    "optimizer:",
    f"  name: adamw",
    f"  learning_rate: {fmt(cfg['learning_rate'])}",
    f"  adam_beta1: {fmt(cfg['adam_beta1'])}",
    f"  adam_beta2: {fmt(cfg['adam_beta2'])}",
    f"  adam_eps: {fmt(cfg['adam_eps'])}",
    f"  weight_decay: {fmt(cfg['weight_decay'])}",
    f"  grad_clip_norm: {fmt(cfg['grad_clip_norm'])}",
    "",
    "rl:",
    f"  method: tinker_verifiable_reward",
    f"  loss_fn: {fmt(cfg['loss_fn'])}",
    f"  group_size: {fmt(cfg['group_size'])}",
    f"  kl_penalty_coef: {fmt(cfg['kl_penalty_coef'])}",
    f"  n_steps: {fmt(cfg['n_steps'])}",
    f"  save_every: {fmt(cfg['save_every'])}",
    f"  eval_every: {fmt(cfg['eval_every'])}",
    f"  eval_n_questions: {fmt(cfg['eval_n_questions'])}",
    f"  eval_temperature: {fmt(cfg['eval_temperature'])}",
    "",
    "reward:",
]
for k, v in cfg["reward"].items():
    lines.append(f"  {k}: {fmt(v)}")

lines += [
    "",
    "checkpoints:",
]
for cp in data["checkpoint_paths"]:
    lines.append(f"  - {cp}")

lines.append("")
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {OUT}")
