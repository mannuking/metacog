# Metacog

**Can a small open-weight model know what it doesn't know?**

The Metacog project trains a model to produce **calibrated self-assessments** alongside its answers. The target is not just reasoning quality — it's reasoning *quality matched to stated confidence*. A metacognitive model says "I'm 80% sure" on the questions it actually gets right 80% of the time, and "I'm 20% sure" on the ones it gets right 20% of the time.

The metric we optimize is **Expected Calibration Error (ECE)**: a perfectly metacognitive model achieves ECE = 0. Frontier reasoning models typically sit at ECE = 0.10-0.20 out of the box. We want to do better — especially on a small edge-deployable model.

## Project structure

```
Metacog/
├── tinker/              Tinker API client wrapper
│   └── client.py        Thin facade over the Tinker SDK
├── data/                Datasets + schemas
│   ├── schemas.py       Pydantic models for QuestionRecord, TraceRecord, MetacogTrainingRecord
│   └── load_datasets.py Pulls GSM8K + MMLU-Pro from HuggingFace
├── model/               Model architecture
│   └── confidence_head.py  2-layer MLP + LinearProbe baselines
├── eval/                Evaluation harness
│   ├── normalize.py     Answer extraction + correctness check
│   └── metrics.py       ECE, Brier, AUROC, self-consistency confidence
├── scripts/             Executable entry points
│   └── run_baseline.py  The baseline runner (this is the most important file)
├── results/             Baseline outputs (gitignored)
├── logs/                Run logs (gitignored)
├── .env                 API key (gitignored)
├── .env.example         Template
├── pyproject.toml       Package metadata
└── README.md            You are here
```

## The research plan

### Phase 0: Baseline (current)

**Goal:** Measure how calibrated the base Qwen3.6-35B-A3B model is on GSM8K + MMLU-Pro BEFORE any fine-tuning.

**Why this matters:** You can't improve a metric you haven't measured. The baseline ECE is the number we have to beat. If base ECE is 0.05, the head has very little room to help. If it's 0.30, we have a real signal to chase.

**How it works:**
1. Pull 20-50 questions from GSM8K and MMLU-Pro
2. Sample K=1 (or K=5 for self-consistency) completions from the base model
3. Extract the final answer from each completion
4. Score correctness against gold
5. Compute a confidence signal (heuristic from hedging language for K=1, self-consistency for K=5)
6. Compute ECE, Brier, AUROC, over/under-confidence rates

**Output:** `results/baseline_summary.json` + `results/baseline_reliability.txt`

### Phase 1: Data generation (next)

**Goal:** Build a labeled dataset of `(reasoning_trace, step_confidence, final_correctness)` for training the confidence head.

**Approach:**
- For each of 10K-50K questions across GSM8K, MMLU-Pro, ARC-Challenge, generate K=5 completions from the base model at temperature 0.7
- Compute self-consistency per question: fraction of K matching the majority answer
- For each step in the trace, compute a soft confidence target = 0.6 × step_agreement + 0.4 × final_correct
- Save as JSONL

**Cost estimate:** 50K × 5 samples × ~1500 tokens = 375M tokens. At Qwen3.6-35B-A3B Tinker rates, ~$450-600. **Will likely be smaller (10-20K) to fit budget.**

### Phase 2: RL with verifiable rewards (the key experiment)

**Goal:** Fine-tune Qwen3.6-35B-A3B to be MORE metacognitive without SFT priming. Skip the SFT step and go straight to RL.

**Reward function (the metacog reward):**
```
reward = 1.0  if final_correct AND confidence matches outcome
       = 0.5  if final_correct AND confidence is off
       = 0.0  if final_wrong
       - 0.5  if final_wrong AND confidence is high (punish confident hallucination)
       + 0.5  if final_wrong AND confidence is low (reward appropriate humility)
```

This pushes the model toward: "if you're going to be wrong, at least be uncertain about it."

**Why this works:** The KL penalty against the base model (Tinker's default) prevents catastrophic forgetting. The reward signal is verifiable — every example has a known gold answer, so we can compute the reward exactly. No preference labels needed.

### Phase 3: Confidence head training (parallel track)

**Goal:** Train a separate small MLP on top of the frozen base model to predict P(correct) from the last hidden state.

**Architecture:** 2-layer MLP (2048 → 256 → 1) with residual gate, ~528K params. Trained on the labeled dataset from Phase 1. Provides a calibrated probability at every reasoning step.

**Why both RL and the head?** The RL change the model's behavior. The head reads the model's activations. They target different things. Best case: both work, head validates the behavioral change. Worst case: one works, we still have something publishable.

### Phase 4: Edge deployment (later)

**Goal:** Get the metacognitive Qwen3.6-35B-A3B (or its distilled version) running on:
- Jetson Orin Nano 8GB
- NVIDIA laptop with ≥8GB VRAM

**Quantization:** Q4_K_M for the 35B-A3B gives ~17GB — won't fit. Q3_K or Q2_K will, with quality loss. Alternative: distill the metacognitive behavior into a smaller model (Qwen3-4B or Llama-3.2-3B) using the RL'd Qwen3.6 as teacher. That's a future Phase 4b.

## Running the baseline

```bash
# 1. Set up the venv
cd E:/Projects/Metacog
uv venv --python 3.11 .venv
source .venv/Scripts/activate
uv pip install -e .

# 2. Add your Tinker API key
cp .env.example .env
# Edit .env and paste your key from https://tinker-console.thinkingmachines.ai/

# 3. Verify everything works (no API cost)
python -m scripts.run_baseline --dry-run --n-gsm8k 10 --n-mmlu-pro 10

# 4. Run the real baseline (K=1, fast)
python -m scripts.run_baseline --k 1 --n-gsm8k 25 --n-mmlu-pro 25

# 5. Run the real baseline with self-consistency (K=5, more expensive)
python -m scripts.run_baseline --k 5 --n-gsm8k 25 --n-mmlu-pro 25
```

## Current status

- [x] Tinker client wrapper (`tinker/client.py`)
- [x] Dataset schemas + GSM8K + MMLU-Pro loaders (`data/`)
- [x] Answer normalization + correctness (`eval/normalize.py`)
- [x] Calibration metrics — ECE, Brier, AUROC, over/under-confidence (`eval/metrics.py`)
- [x] Self-consistency confidence estimator (`eval/metrics.py`)
- [x] Confidence head architecture — 2-layer MLP + LinearProbe baseline (`model/confidence_head.py`)
- [x] Hidden-state pooling helper
- [x] Baseline runner with synthetic dry-run mode (`scripts/run_baseline.py`)
- [x] Smoke tests passing on all modules
- [ ] **Real baseline numbers from Qwen3.6-35B-A3B** — needs Tinker API key
- [ ] Data generation pipeline (Phase 1)
- [ ] RL training loop (Phase 2)
- [ ] Head training loop (Phase 3)
- [ ] Edge export (Phase 4)

## Key research questions

1. **Is Qwen3.6-35B-A3B already reasonably calibrated on math/science?** If yes, the head has diminishing returns and RL is where the gains are.
2. **Does RL with metacog rewards transfer across domains?** Train on GSM8K, eval on MMLU-Pro.
3. **Does the confidence head generalize?** Train on math traces, eval on science traces.
4. **Is the metacognitive signal in the LAST hidden state, or earlier layers?** Layer-wise probing is a natural ablation.

## References

- Tinker: https://thinkingmachines.ai/tinker/
- Tinker cookbook: https://github.com/thinking-machines-lab/tinker-cookbook
- Qwen3.6-35B-A3B blog: https://qwen.ai/blog?id=qwen3.6-35b-a3b
- Gemma 4 release: https://huggingface.co/blog/gemma4
- Metacognition in LLMs (Nature Comms, 2024): https://www.nature.com/articles/s41467-024-55628-6
- Metacognitive Consolidation (arXiv 2604.17399): https://arxiv.org/html/2604.17399v1
- On-device LLMs (Meta AI Research, 2026): https://v-chandra.github.io/on-device-llms/

## Author

Jai kumar Meena · Sofia (AI engineer) · Tinker API: Thinking Machines Lab
