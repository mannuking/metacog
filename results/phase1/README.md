# Phase 1 (RL)

**Date:** 2026-06-11 (17:17 → 21:30 IST)
**Model:** Qwen/Qwen3.6-35B-A3B + LoRA rank 32
**RL:** Tinker's verifiable-reward RL, importance-sampling loss, 20 steps

## Results

### Headline (held-out eval, n=20)

| Metric     | Phase 0 | Phase 1 | Δ     |
|------------|--------:|--------:|------:|
| Accuracy   | 0.420   | **0.800**   | +0.380 |
| ECE        | 0.503   | **0.099**   | -0.404 |
| Brier      | 0.475   | **0.101**   | -0.374 |
| Overconfident rate | 0.571 | **0.000** | -0.571 |

### Per-step outcomes (K=4 rollouts, 200 rollouts/step)

See `paper/tables/MAIN_RESULTS.md` Table 2 for the full 20-step table. Highlights:

- **Step 1:** 19.5% correct, 76.5% humble_wrong, 4.0% overconf_wrong
- **Step 20:** 72.0% correct, 18.0% humble_wrong, 10.0% overconf_wrong
- **Reward curve:** 0.20 → 0.90 (4.5× climb, monotonic-ish)

### Per-checkpoint held-out eval (n=20)

| Checkpoint | Acc   | ECE    | Avg conf |
|------------|------:|-------:|---------:|
| Init (-1)  | 0.250 | 0.3765 | 0.624    |
| Step 5     | 0.300 | 0.3525 | 0.647    |
| Step 10    | 0.550 | 0.2370 | 0.787    |
| Step 15    | 0.850 | 0.0980 | 0.948    |
| Step 20    | 0.800 | 0.0995 | 0.899    |
| Final      | 0.800 | 0.0990 | 0.899    |

## Checkpoints

All checkpoints live in Tinker's cloud storage (Tinker URIs):

```
tinker://4ee43737-7dd5-5271-98bd-fd6f28370006:train:0/weights/phase1_step5
tinker://4ee43737-7dd5-5271-98bd-fd6f28370006:train:0/weights/phase1_step10
tinker://4ee43737-7dd5-5271-98bd-fd6f28370006:train:0/weights/phase1_step15
tinker://4ee43737-7dd5-5271-98bd-fd6f28370006:train:0/weights/phase1_step20
```

**To download a checkpoint:**
```bash
python -c "from tinker import ... ; restore('tinker://...')"
```
(or whatever Tinker's restore API is — full code in `scripts/load_checkpoint.py`)

## Files

| Path | What it is |
|------|------------|
| `results/phase1/phase1_20260611_171712.log` | Full human-readable training log (7 KB) |
| `results/phase1/phase1_summary.json` | Full structured results (2.3 MB, includes per-rollout details) |
| `results/phase1/per_step_metrics.csv` | Per-step summary in CSV (Table 2) |
| `results/phase1/inloop_eval.csv` | Per-checkpoint held-out eval (Table 3) |
| `scripts/run_phase1.py` | Training script |

## Cost & time

- **Wall time:** 4h 13m (includes ~1.5h Tinker queue hiccup at step 3→4)
- **Clean wall time estimate:** 2.5h
- **Cost:** <$15 of $150 Tinker credit
- **Total rollouts:** 4,000 (50 q × 4 K × 20 steps)
- **Total tokens generated:** ~2M

## How to reproduce

```bash
cd E:/Projects/Metacog
.venv/Scripts/python.exe scripts/run_phase1.py \
  --n-steps 20 \
  --group-size 4 \
  --n-gsm8k 25 \
  --n-mmlu-pro 25 \
  --max-tokens 500 \
  --learning-rate 1e-5 \
  --lora-rank 32 \
  --temperature 1.0 \
  --loss-fn importance_sampling \
  --effort medium \
  --eval-every 5 \
  --save-every 5 \
  --out-dir results/phase1
```

Requires `TINKER_API_KEY` in the environment (or in `~/.tinker/config.toml`).
