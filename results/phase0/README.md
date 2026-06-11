# Phase 0 (Baseline)

**Date:** 2026-06-11 (pre-Phase-1)
**Model:** Qwen/Qwen3.6-35B-A3B
**No LoRA, no fine-tuning. K=1 (single rollout per question).**

## Results (n=50)

| Metric              | Value     |
|---------------------|----------:|
| Accuracy            | **0.420** |
| ECE                 | **0.5029** |
| Brier               | 0.4751    |
| AUROC (conf→corr)  | 0.2167    |
| Overconfidence rate | **0.5714** |
| Underconfidence     | 0.0000    |
| Avg confidence      | 0.870     |

## Per-source breakdown

| Source     | n  | Acc   | ECE     | Brier   | AUROC  |
|------------|---:|------:|--------:|--------:|-------:|
| GSM8K      | 25 | 0.640 | 0.3039  | 0.3046  | 0.174  |
| MMLU-Pro   | 25 | 0.200 | 0.7019  | 0.6455  | 0.260  |

**Diagnosis:** The base model is *severely* overconfident. 57% of answers are wrong
*and* stated with high confidence. The ECE of 0.50 means its confidence is essentially
uncorrelated with correctness. The model is "always sure of itself, often wrong" — a
textbook example of a miscalibrated reasoning model.

## Reliability bins

| Confidence bin | n  | Mean conf | Actual acc |
|---------------:|---:|----------:|-----------:|
| 0.6 - 0.7      |  1 | 0.698     | 0.000      |
| 0.8 - 0.9      | 15 | 0.870     | 0.067      |
| 0.9 - 1.0      | 34 | 0.953     | 0.588      |

**The smoking gun:** in the 0.8-0.9 bin, the model is right 7% of the time despite
claiming 87% confidence. This is the failure mode Phase 1 is designed to fix.

## Files
- `results/phase0/baseline_k1_summary.json` — full JSON
- `paper/tables/phase0_baseline.json` — copy in paper dir
- `scripts/run_baseline.py` — generation script
- Raw outputs: `results/qwen36_35b_k1_*.jsonl` (one per question)

## Cost
- 50 API calls, ~600 tokens each
- Wall time: ~6 minutes
- Cost: <$1
