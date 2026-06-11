# Baseline results

This file tracks the metric values we need to beat. Each line is one full
baseline run. Newer results at the bottom.

| model | k | n | accuracy | ECE  | Brier | AUROC | overconf_rate | elapsed_s | source |
|-------|---|---|----------|------|-------|-------|---------------|-----------|--------|
| Qwen/Qwen3.6-35B-A3B | 1 | 50 | 0.420 | 0.503 | 0.475 | 0.217 | 0.571 | 360.8 | real Tinker API |
| Qwen/Qwen3.6-35B-A3B | 1 | 25 gsm8k | 0.640 | 0.304 | 0.305 | 0.174 | — | — | real |
| Qwen/Qwen3.6-35B-A3B | 1 | 25 mmlu_pro | 0.200 | 0.702 | 0.645 | 0.260 | — | — | real |

## Reliability diagram of baseline (Qwen3.6-35B-A3B, K=1, n=50)

| bin          | n  | mean_conf | actual_acc | gap     |
|--------------|----|-----------|------------|---------|
| [0.6, 0.7)   | 1  | 0.698     | 0.000      | +0.698  |
| [0.8, 0.9)   | 15 | 0.870     | 0.067      | +0.803  |
| [0.9, 1.0)   | 34 | 0.953     | 0.588      | +0.365  |

**Read this as:** 34 of 50 questions landed in the top confidence bin
(model said 0.95+). Of those 34, only 20 (58.8%) were correct. The other
15 questions in the 0.87 confidence bin were correct only 6.7% of the time.
The model is **systematically overconfident on questions it gets wrong**.
Confidence is **anti-predictive** (AUROC < 0.5).

## What this means for training

- The base model has real domain competence (64% on GSM8K is good).
- It has zero self-knowledge — it doesn't know when it's guessing.
- **Phase 1 (RL with verifiable rewards)** should:
  - Reward **correct** final answers
  - Penalize **confident wrong** answers (large reward if right, smaller
    penalty if wrong AND confidence > 0.8)
  - Learn to say "I don't know" or output low confidence on questions
    in the model's blind spot
- **Target after Phase 1**: ECE < 0.20, AUROC > 0.65
