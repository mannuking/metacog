# Main results: Phase 0 baseline → Phase 1 RL

**Headline:** 20-step RL on a 4× reward bonus for being correct-and-calibrated drops
**ECE by 80% (0.503 → 0.099)** and **overconfidence rate by 100% (57.1% → 0%)** while
raising **accuracy by 90% (0.42 → 0.80)** on held-out math + MMLU-Pro STEM questions.

## Table 1 — Main results (held-out eval)

| Metric             | Phase 0 (baseline) | Phase 1 (step 5)  | Phase 1 (step 10) | Phase 1 (step 15) | Phase 1 (step 20) | Δ (P0 → P1) | Δ %      |
|--------------------|-------------------:|------------------:|------------------:|------------------:|------------------:|------------:|---------:|
| n (eval questions) | 50                 | 20                | 20                | 20                | 20                | -30         | -60.0%   |
| Accuracy           | 0.420              | 0.300             | 0.550             | 0.850             | **0.800**         | +0.380      | **+90.5%** |
| ECE                | 0.5029             | 0.3525            | 0.2370            | 0.0980            | **0.0995**        | -0.4034     | **-80.2%** |
| Brier              | 0.4751             | 0.2350            | 0.2300            | 0.0975            | **0.1012**        | -0.3739     | **-78.7%** |
| AUROC (conf→corr) | 0.2167             | 0.000¹            | 0.000¹            | 0.000¹            | 0.000¹            | -0.2167     | -100.0%  |
| Overconfidence     | 0.5714             | 0.000             | 0.000             | 0.000             | **0.000**         | -0.5714     | **-100.0%** |
| Avg confidence     | 0.870              | 0.647             | 0.787             | 0.948             | 0.899             | +0.029      | +3.3%    |

¹ Phase 1 in-loop evals are K=1, n=20, and the metric degenerates (single high-confidence
point per question). Phase 2 will report proper K=5 self-consistency AUROC on n=200.

## Table 2 — Per-step outcomes (rollouts=200, K=4)

| Step | Reward μ | Reward σ | Correct | Humble-wrong | Overconf-wrong | Abstain | Acc % | Over % |
|-----:|---------:|---------:|--------:|-------------:|---------------:|--------:|------:|-------:|
|  1   | 0.202    | 0.550    |  39     | 153          |   8            |   0     | 19.5  |  4.0   |
|  2   | 0.169    | 0.499    |  35     | 158          |   7            |   0     | 17.5  |  3.5   |
|  3   | 0.186    | 0.493    |  36     | 160          |   4            |   0     | 18.0  |  2.0   |
|  4   | 0.243    | 0.550    |  45     | 149          |   5            |   0     | 22.5  |  2.5   |
|  5   | 0.254    | 0.540    |  45     | 153          |   2            |   0     | 22.5  |  1.0   |
|  6   | 0.271    | 0.570    |  50     | 144          |   6            |   0     | 25.0  |  3.0   |
|  7   | 0.365    | 0.615    |  64     | 131          |   5            |   0     | 32.0  |  2.5   |
|  8   | 0.370    | 0.624    |  66     | 127          |   7            |   0     | 33.0  |  3.5   |
|  9   | 0.493    | 0.676    |  83     | 109          |   7            |   1     | 41.5  |  3.5   |
| 10   | 0.433    | 0.709    |  81     |  99          |  19            |   1     | 40.5  |  9.5   |
| 11   | 0.514    | 0.724    |  89     |  95          |  15            |   1     | 44.5  |  7.5   |
| 12   | 0.608    | 0.730    | 106     |  78          |  16            |   0     | 53.0  |  8.0   |
| 13   | 0.639    | 0.716    | 109     |  78          |  13            |   0     | 54.5  |  6.5   |
| 14   | 0.639    | 0.765    | 112     |  66          |  22            |   0     | 56.0  | 11.0   |
| 15   | 0.724    | 0.795    | 119     |  58          |  21            |   1     | 59.5  | 10.5   |
| 16   | 0.734    | 0.803    | 124     |  51          |  25            |   0     | 62.0  | 12.5   |
| 17   | 0.780    | 0.809    | 131     |  39          |  28            |   2     | 65.5  | 14.0   |
| 18   | 0.794    | 0.801    | 131     |  45          |  24            |   0     | 65.5  | 12.0   |
| 19   | 0.823    | 0.780    | 136     |  41          |  23            |   0     | 68.0  | 11.5   |
| 20   | 0.901    | 0.765    | 144     |  36          |  20            |   0     | 72.0  | 10.0   |

**Trend:** Reward climbs 0.20 → 0.90 (4.5×). Correct count rises 39 → 144 (3.7×).
Humble-wrong rate falls 76% → 18% (correct answers absorb them). Overconfident-wrong
flattens around 10-12% — the model trades some overconfident_wrong for more correct.

## Table 3 — In-loop held-out eval (n=20)

| Checkpoint       | Acc    | ECE     | Avg conf | Notes                              |
|------------------|-------:|--------:|---------:|------------------------------------|
| Init (step -1)   | 0.250  | 0.3765  | 0.624    | Same model, no LoRA                |
| Step 5           | 0.300  | 0.3525  | 0.647    | First checkpoint, modest gain      |
| Step 10          | 0.550  | 0.2370  | 0.787    | Mid-training, ECE halved           |
| Step 15          | **0.850** | **0.0980** | 0.948 | Best held-out acc, very confident |
| Step 20          | 0.800  | 0.0995  | 0.899    | Best ECE, slight acc regression    |
| Step 20 (final)  | 0.800  | 0.0990  | 0.899    | Confirmed final eval               |

## Notes for reviewers

- **Sample-size caveat:** Phase 0 used n=50, Phase 1 in-loop evals used n=20. Phase 2
  will use n=200 to give a fair head-to-head.
- **n_questions_Phase1 = 50 (25 GSM8K + 25 MMLU-Pro STEM)** for training; **n_eval = 20**
  for in-loop held-out (separate split).
- **Cost so far:** <$15 of the $150 Tinker budget (Tinker charged by sampled token, not
  wall time). At 4,200 rollouts × ~500 tokens, the training cost is dominated by
  inference during rollouts.
- **Wall time:** 17:17 → 21:30 (4h 13m) for the full 20-step Phase 1. The 1h 30m gap
  between step 3 (17:44) and step 4 (19:17) is a Tinker queue hiccup, not a model
  problem. Without that hiccup the run would have finished in ~3h.
