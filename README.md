# Metacog

**A small, reproducible pipeline for teaching large language models to know what they don't know.**

The repo contains the production code: Tinker API client, dataset schemas, eval harness,
training script, and a live training dashboard. Results, paper drafts, and announcements
live in the author's private working dir and are intentionally **not** tracked here.

## What it does

Trains a 35B open-weight LLM (Qwen3.6-35B-A3B) with LoRA + reinforcement learning to
produce **calibrated self-assessments** alongside its answers. The metric it optimizes
is **Expected Calibration Error (ECE)** — a perfectly metacognitive model achieves ECE = 0.

The headline result from the author's reference run (not in this repo, see your own
output): ECE fell from 0.50 to 0.10, overconfidence rate from 57% to 0%, accuracy rose
from 0.42 to 0.80, in a single 20-step RL run that cost <$15 of cloud compute.

For the full result, see your own `results/phase1/` after running.

## Project structure

```
Metacog/
├── tinker/                       Tinker API client wrapper
│   └── client.py                 Thin facade over the Tinker SDK
├── data/                         Datasets + schemas
│   ├── schemas.py                Pydantic models
│   ├── load_datasets.py          Pulls GSM8K + MMLU-Pro from HuggingFace
│   └── questions.jsonl           50 training questions
├── model/                        Model + confidence head
│   └── confidence_head.py        2-layer MLP + LinearProbe baselines
├── eval/                         Evaluation harness
│   ├── normalize.py              Answer extraction + correctness check
│   └── metrics.py                ECE, Brier, AUROC, self-consistency
├── scripts/                      Executable entry points
│   ├── run_baseline.py           Phase 0: measure baseline ECE
│   ├── run_phase1.py             Phase 1: RL training
│   ├── eval_phase1.py            Post-training eval (K=1, K=5)
│   ├── plot_phase1_curves.py     Training-curve figure
│   ├── plot_reliability.py       Reliability-diagram figure
│   └── snapshot_config.py        Frozen config → YAML
├── dashboard/                    Live training dashboard
│   ├── server.py                 Python stdlib HTTP server (port 7860)
│   └── static/                   HTML, CSS, JS
├── results/                      (gitignored — your run outputs go here)
├── logs/                         (gitignored)
├── pyproject.toml                Package metadata
├── .env.example                  TINKER_API_KEY template (gitignored: .env)
└── README.md                     You are here
```

## Install + run

```bash
# 1. Set up
cd E:/Projects/Metacog
uv venv --python 3.11 .venv
.venv/Scripts/python.exe -m pip install -e .

# 2. Add your Tinker API key
cp .env.example .env
# Edit .env, paste TINKER_API_KEY from https://tinker-console.thinkingmachines.ai/

# 3. Phase 0: measure baseline ECE (~6 min, ~$1)
.venv/Scripts/python.exe -m scripts.run_baseline --k 1 --n-gsm8k 25 --n-mmlu-pro 25

# 4. Phase 1: RL training (~4 hours, ~$15)
.venv/Scripts/python.exe -m scripts.run_phase1 \
  --n-steps 20 --group-size 4 --n-gsm8k 25 --n-mmlu-pro 25 \
  --max-tokens 500 --learning-rate 1e-5 --lora-rank 32 \
  --temperature 1.0 --loss-fn importance_sampling --effort medium \
  --eval-every 5 --save-every 5 --out-dir results/phase1

# 5. Watch the training live (separate terminal)
.venv/Scripts/python.exe dashboard/server.py --port 7860
# open http://127.0.0.1:7860 in a browser

# 6. Generate figures from your results
.venv/Scripts/python.exe scripts/plot_phase1_curves.py
.venv/Scripts/python.exe scripts/plot_reliability.py
```

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

The **asymmetry** is the trick: the model learns that saying "I don't know" is a strictly
higher expected return than guessing. Tunable via the reward kwargs in
`scripts/run_phase1.py` — every term can be ablated independently.

## Requirements

- Python 3.11+
- Tinker API key ([tinker-console.thinkingmachines.ai](https://tinker-console.thinkingmachines.ai/))
- ~$20 of Tinker credit for the full pipeline (Phase 0 + Phase 1)
- A modern browser for the dashboard (Chrome/Firefox/Safari)

## License

MIT. Use it, fork it, publish with it.

## Author

Jai kumar Meena · Sofia (AI engineer) · Tinker API: Thinking Machines Lab
