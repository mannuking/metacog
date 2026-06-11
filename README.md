# Metacog

**Can a 35B open-weight model know what it doesn't know?**

**Phase 1 (June 11, 2026): Yes, and we can teach it in 4 hours for $15 of cloud compute.**

| Metric | Phase 0 (baseline) | Phase 1 (after RL) | Δ |
|---|---:|---:|---:|
| Accuracy | 0.420 | **0.800** | +90.5% |
| ECE (Expected Calibration Error) | 0.503 | **0.099** | -80.2% |
| Brier score | 0.475 | **0.101** | -78.7% |
| Overconfidence rate | 0.571 | **0.000** | -100% |

**The single insight:** An asymmetric reward function that punishes "confidently wrong" (-0.5) but
**does not punish** "humbly wrong" (0.0) is sufficient to teach a 35B MoE model to say "I don't know"
when it doesn't know, in a single 20-step RL run. The model learned that "say I don't know" is a
strictly better strategy than "guess with high confidence" — within 5 RL steps the bluffing was
gone, and the overconfident-wrong rate never returned.

> The full result, methodology, reward function, training curve, and 4 saved checkpoints are in
> [`paper/PAPER_RESULTS.md`](paper/PAPER_RESULTS.md). Start there.

## What this repo is

A complete, reproducible pipeline for **calibrating the self-knowledge of large open-weight
language models via reinforcement learning on verifiable metacognitive rewards**.

It was built in 24 hours. It is small, opinionated, and runs end-to-end. Total cost so far: **<$20**.

The target is not just reasoning quality — it's reasoning *quality matched to stated confidence*.
A metacognitive model says "I'm 80% sure" on the questions it actually gets right 80% of the time,
and "I'm 20% sure" on the ones it gets right 20% of the time. The metric we optimize is
**Expected Calibration Error (ECE)**.

## What we did

1. **Phase 0** — measured the baseline ECE of `Qwen/Qwen3.6-35B-A3B` on 50 GSM8K + MMLU-Pro
   questions. It scored 0.503 — the model is *severely* overconfident, 57% of wrong answers
   are stated with high confidence.
2. **Phase 1** — fine-tuned the same model with **LoRA rank 32** using Tinker's verifiable-reward
   RL, with a custom 7-component metacognitive reward function. 20 steps × 50 questions × K=4
   rollouts = 4,000 rollouts. Wall time ~4 hours, cost <$15.
3. **Saved 4 LoRA checkpoints** to Tinker cloud storage (URIs in `paper/CHECKPOINTS.md`).
4. **Built a live dashboard** (`dashboard/server.py`, port 7860) that streams training progress
   in real time.

## The reward function (the whole idea)

```python
reward =  1.0  if correct                                # right answer
        + 0.5  if correct AND conf in [0.6, 0.95]         # calibrated confidence
        + 0.3  if abstained AND conf < 0.4               # honest "I don't know"
        - 0.5  if wrong AND conf > 0.8                   # PUNISH confident hallucination
        + 0.0  if wrong AND conf < 0.5                   # humble-wrong is FREE
        + 0.05 if parseable                              # small format bonus
        + 0.2  * (1 - |conf - empirical_acc|)            # continuous calibration term
        - 0.05 per 1k tokens (capped at -0.1)             # length penalty
```

The **asymmetry** is the trick. The model learns that saying "I don't know" is a strictly
higher expected return than guessing. Within 5 RL steps the overconfident rate drops from
~57% to 1-3%, and the held-out n=20 eval shows it at **0%**.

## Project structure

```
Metacog/
├── tinker/                       Tinker API client wrapper
│   └── client.py                 Thin facade over the Tinker SDK
├── data/                         Datasets + schemas
│   ├── schemas.py                Pydantic models for QuestionRecord, etc.
│   ├── load_datasets.py          Pulls GSM8K + MMLU-Pro from HuggingFace
│   └── questions.jsonl           50 training questions (25 GSM8K + 25 MMLU-Pro)
├── model/                        Model + confidence head
│   └── confidence_head.py        2-layer MLP + LinearProbe baselines
├── eval/                         Evaluation harness
│   ├── normalize.py              Answer extraction + correctness check
│   └── metrics.py                ECE, Brier, AUROC, self-consistency
├── scripts/                      Executable entry points
│   ├── run_baseline.py           Phase 0 baseline
│   ├── run_phase1.py             Phase 1 RL training
│   ├── eval_phase1.py            Post-training eval (K=1, K=5)
│   ├── plot_phase1_curves.py     4-panel training curves figure
│   ├── plot_reliability.py       Reliability diagram
│   ├── snapshot_config.py        Frozen config → YAML
│   └── _sync_to_paper.py         Sync results → paper/ directory
├── dashboard/                    Live training dashboard
│   ├── server.py                 Python stdlib HTTP server (port 7860)
│   └── static/                   HTML, CSS, JS
├── results/
│   ├── phase0/                   Baseline (K=1, n=50) — pre-RL
│   ├── phase1/                   RL run (20 steps, n_train=50, n_eval=20) — post-RL
│   └── phase1_test/              Tiny smoke test (2 questions)
├── paper/                        All paper artifacts in one place
│   ├── PAPER_RESULTS.md          ⭐ Start here. One-stop doc.
│   ├── METHOD.md                 Full methodology
│   ├── CHECKPOINTS.md            4 Tinker checkpoint URIs
│   ├── tables/                   Main results (CSV + MD + JSON)
│   ├── figures/                  Training curves + reliability diagram (PNG + SVG)
│   ├── configs/                  Frozen YAML config used for the run
│   ├── references/               VENUES.md (target publication venues)
│   └── announcements/            LinkedIn post (viral-style)
├── architecture deep dive.md     The architecture doc
├── instruction menu.md           The 50 integrations menu
├── pyproject.toml                Package metadata
└── README.md                     You are here
```

## Reproducing the result

```bash
# 1. Set up
cd E:/Projects/Metacog
uv venv --python 3.11 .venv
.venv/Scripts/python.exe -m pip install -e .

# 2. Add your Tinker API key
cp .env.example .env
# Edit .env, paste TINKER_API_KEY from https://tinker-console.thinkingmachines.ai/

# 3. Phase 0 baseline (~6 min, ~$1)
.venv/Scripts/python.exe -m scripts.run_baseline --k 1 --n-gsm8k 25 --n-mmlu-pro 25

# 4. Phase 1 RL (~4 hours, ~$15)
.venv/Scripts/python.exe -m scripts.run_phase1 \
  --n-steps 20 --group-size 4 --n-gsm8k 25 --n-mmlu-pro 25 \
  --max-tokens 500 --learning-rate 1e-5 --lora-rank 32 \
  --temperature 1.0 --loss-fn importance_sampling --effort medium \
  --eval-every 5 --save-every 5 --out-dir results/phase1

# 5. Watch it live
.venv/Scripts/python.exe dashboard/server.py --port 7860
# open http://127.0.0.1:7860 in a browser

# 6. Re-generate the figures
.venv/Scripts/python.exe scripts/plot_phase1_curves.py
.venv/Scripts/python.exe scripts/plot_reliability.py
```

## Current status

- [x] **Phase 0:** Real Qwen3.6-35B-A3B baseline (K=1, n=50): acc=0.420, ECE=0.503, over=0.571
- [x] **Phase 1:** Real Qwen3.6-35B-A3B + LoRA + RL (20 steps, K=4, n_train=50, n_eval=20):
  final acc=0.800, ECE=0.099, over=0.000
- [x] **4 saved LoRA checkpoints** (Tinker URIs in `paper/CHECKPOINTS.md`)
- [x] **Live training dashboard** (port 7860, stdlib-only)
- [x] **Paper-grade artifact archive** in `paper/`
- [x] **LinkedIn post** in `paper/announcements/LINKEDIN_POST.md`
- [ ] **Phase 2** (planned): n_eval=200, K=5 self-consistency, OOD eval, ablations, scale to 100B

## Reaction & feedback

If you tried this, found it useful, or want to critique the method — open an issue or drop
a comment in [`paper/announcements/REACTIONS.md`](paper/announcements/REACTIONS.md).
We want to know:
1. Did the calibration hold on your domain? (code, dialogue, medical, legal, etc.)
2. Did the model *actually* become more honest, or just better at gaming the reward?
3. What's the failure mode we missed?

## Key research questions

1. Does RL on metacognitive rewards transfer across domains? (Train on math, eval on dialogue)
2. Does the calibration generalize to out-of-distribution questions?
3. Does it scale? (100B+ parameters)
4. Is the signal in the *behavior* (what the model says) or the *activations* (what the model "thinks")?

## References

- Tinker: https://thinkingmachines.ai/tinker/
- Tinker cookbook: https://github.com/thinking-machines-lab/tinker-cookbook
- Qwen3.6-35B-A3B: https://qwen.ai/blog?id=qwen3.6-35b-a3b
- Metacognition in LLMs (Nature Comms, 2024): https://www.nature.com/articles/s41467-024-55628-6
- Metacognitive Consolidation (arXiv 2604.17399): https://arxiv.org/html/2604.17399v1
- "Mirror: A Hierarchical Benchmark for Metacognitive Calibration": https://mirror-benchmark.github.io

## License

MIT. Use it, fork it, publish with it. If you beat our numbers, we want to know.

## Author

Jai kumar Meena · Sofia (AI engineer) · Tinker API: Thinking Machines Lab
