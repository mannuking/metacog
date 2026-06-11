<div align="center">

# 🧠 Metacog

### *Teaching large language models to know what they don't know.*

<br/>

[![MIT License](https://img.shields.io/badge/License-MIT-22c55e.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](https://python.org)
[![Tinker](https://img.shields.io/badge/Powered_by-Tinker-6ee7b7.svg)](https://thinkingmachines.ai/tinker/)
[![Model: Qwen3.6-35B-A3B](https://img.shields.io/badge/Model-Qwen3.6--35B--A3B-60a5fa.svg)](https://qwen.ai)
[![ECE optimized](https://img.shields.io/badge/Metric-ECE_Optimized-ef4444.svg)](#the-reward-function)
[![Cost: <$20](https://img.shields.io/badge/Total_Cost-%3C%2420-f59e0b.svg)](#requirements)

<br/>

> **In 4 hours and $15 of cloud compute, a 35B open-weight model went from
> 57% overconfidence and ECE 0.50 to 0% overconfidence and ECE 0.10 —
> while accuracy rose from 42% to 80%.**
>
> **One reward function. No humans in the loop.**

<br/>

[**Quick start**](#-quick-start) · [**How it works**](#-how-it-works) · [**Future scope**](#-future-scope) · [**Roadmap**](#-roadmap) · [**Ecosystem**](#-ecosystem) · [**Cite this work**](#-citation)

</div>

---

## 📊 The headline result

```
                     Phase 0          Phase 1 (after 20 RL steps)
                     ──────────       ────────────────────────────
  Accuracy             0.420  ████████░░░░░░░░░░░░░░░░░░  → 0.800  ████████████████░░░░░░  +90.5%
  ECE                  0.503  ██████████████████░░░░░░░░  → 0.099  ████░░░░░░░░░░░░░░░░░░  -80.2%
  Brier                0.475  █████████████████░░░░░░░░░  → 0.101  ████░░░░░░░░░░░░░░░░░░  -78.7%
  Overconfidence       0.571  ████████████████████░░░░░░  → 0.000  ░░░░░░░░░░░░░░░░░░░░░░  -100%
  Avg confidence       0.870  ████████████████████░░░░░░  → 0.899  ████████████████████░░░  +3.3%
```

The reward curve over 20 RL steps climbs **4.5×** (0.20 → 0.90). The overconfident-wrong
rate drops from 4% to a held-out **0%** in 5 steps and never returns.

---

## 🎯 What this is

**Metacog** is a small, opinionated, end-to-end pipeline for **calibrating the
self-knowledge of large open-weight language models** using reinforcement learning
on *verifiable* metacognitive rewards.

The metric we optimize is **Expected Calibration Error (ECE)** — a perfectly
metacognitive model achieves ECE = 0. Frontier reasoning models typically
sit at ECE = 0.10-0.20 out of the box. We want to do better, **and** we want
to do it on a model you can download, fine-tune, and run yourself.

The trick: an **asymmetric reward function** that punishes *confidently wrong*
answers twice (once for being wrong, once via the calibration term) but does
**not** punish *humbly wrong* answers. Within 5 RL steps the bluffing is gone
and the model has learned that "say I don't know" is a strictly higher
expected return than "guess with high confidence."

---

## 🚀 Quick start

```bash
# 1. Set up
git clone https://github.com/mannuking/metacog.git
cd metacog
uv venv --python 3.11 .venv
.venv/Scripts/python.exe -m pip install -e .

# 2. Add your Tinker API key (get one at tinker-console.thinkingmachines.ai)
cp .env.example .env
# Edit .env and paste TINKER_API_KEY=...

# 3. Measure baseline ECE on Qwen3.6-35B-A3B  (~6 min, ~$1)
.venv/Scripts/python.exe -m scripts.run_baseline --k 1 --n-gsm8k 25 --n-mmlu-pro 25

# 4. Run RL training                        (~4 hours, ~$15)
.venv/Scripts/python.exe -m scripts.run_phase1 \
  --n-steps 20 --group-size 4 --n-gsm8k 25 --n-mmlu-pro 25 \
  --max-tokens 500 --learning-rate 1e-5 --lora-rank 32 \
  --temperature 1.0 --loss-fn importance_sampling --effort medium \
  --eval-every 5 --save-every 5 --out-dir results/phase1

# 5. Watch the training live in a browser (separate terminal)
.venv/Scripts/python.exe dashboard/server.py --port 7860
#  → open http://127.0.0.1:7860

# 6. Re-generate figures from your results
.venv/Scripts/python.exe scripts/plot_phase1_curves.py
.venv/Scripts/python.exe scripts/plot_reliability.py
```

Total cost: **<$20** for the full pipeline. No GPUs required — Tinker runs the
training in their cloud and bills by sampled token.

---

## 🧠 How it works

### The reward function (the whole idea)

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

The **asymmetry** is the trick. The model learns that "I don't know" is a strictly
higher expected return than guessing. Every term in the reward is independently
ablatable in `scripts/run_phase1.py` — try removing the calibration term, the
overconfident penalty, or the abstention bonus and see what changes.

### The architecture

```
       Question                    4 × K=4 rollouts
          │                              │
          ▼                              ▼
   ┌──────────────┐             ┌──────────────────┐
   │  Qwen3.6     │             │  Reward function │
   │  35B-A3B     │  ─────────► │  (7 components)  │
   │  + LoRA r32  │   per-step  │                  │
   └──────────────┘   gradients └────────┬─────────┘
          ▲                             │
          │                             ▼
          │                    ┌──────────────────┐
          │                    │  Importance-     │
          │                    │  sampling loss   │
          │                    │  (Tinker)        │
          │                    └────────┬─────────┘
          └─────────────────────────────┘
                   policy update
```

* **Base model:** Qwen3.6-35B-A3B (35B params, 3B active MoE)
* **Adaptation:** LoRA rank 32, applied to attention + MLP
* **RL method:** Tinker's verifiable-reward RL, importance-sampling loss
* **Group size:** K=4 rollouts per question
* **Optimizer:** AdamW, lr=1e-5, weight_decay=0, grad_clip=1.0
* **KL penalty:** 0 (let the policy drift freely — the reward is bounded)

### The training loop

```bash
# Per-step pseudo-code (see scripts/run_phase1.py for the real thing)
for step in range(n_steps):
    rollouts = []                                  # 50 questions × 4 = 200 rollouts
    for q in questions:
        samples = policy.sample(q, k=4, T=1.0)     # generate 4 candidate answers
        for s in samples:
            r = reward(s, gold_answer)              # compute 7-component reward
            s.reward = r
        rollouts.extend(samples)

    # importance-sampling loss: keep rollouts proportional to their advantage
    datums = build_datums(rollouts)                 # 200 datums
    policy.update(datums)                           # one gradient step

    if step % eval_every == 0:
        eval_acc, eval_ece = evaluate(policy)       # in-loop held-out
    if step % save_every == 0:
        save_lora_checkpoint(policy, step)          # → Tinker cloud storage
```

---

## 📁 Project structure

```
Metacog/
├── 📦 tink/                           # Tinker API client
│   └── client.py                      #   Thin facade over the Tinker SDK
│
├── 📚 data/                           # Datasets + schemas
│   ├── schemas.py                     #   Pydantic models: Question, Trace, MetacogRecord
│   ├── load_datasets.py               #   GSM8K + MMLU-Pro from HuggingFace
│   └── questions.jsonl                #   50 training questions (25 + 25)
│
├── 🏗️ model/                         # Model architecture
│   └── confidence_head.py             #   2-layer MLP + LinearProbe baselines
│
├── 🧪 eval/                           # Evaluation harness
│   ├── normalize.py                   #   Answer extraction + correctness check
│   ├── metrics.py                     #   ECE, Brier, AUROC, self-consistency
│   └── structured_parse.py            #   Reward-block parser
│
├── 🎬 scripts/                        # Executable entry points
│   ├── run_baseline.py                #   Phase 0: measure baseline ECE
│   ├── run_phase1.py                  #   Phase 1: RL training
│   ├── eval_phase1.py                 #   Post-training eval (K=1, K=5)
│   ├── plot_phase1_curves.py          #   Training-curve figure
│   ├── plot_reliability.py            #   Reliability-diagram figure
│   ├── snapshot_config.py             #   Frozen config → YAML
│   └── _sync_to_paper.py              #   Internal helper
│
├── 📺 dashboard/                      # Live training dashboard
│   ├── server.py                      #   Python stdlib HTTP server (port 7860)
│   └── static/                        #   HTML + CSS + JS (zero deps)
│
├── 🎓 training/                       # RL internals
│   ├── reward.py                      #   The 7-component reward function
│   ├── rl_loop.py                     #   Tinker's importance-sampling loop
│   └── rollout.py                     #   Sampling + tokenization
│
├── 💬 prompts/                        # Prompt engineering
│   └── chat_template.py               #   The 4-block output template
│
├── 🔒 results/                        # (gitignored) your run outputs
├── 🔒 logs/                           # (gitignored) training logs
├── 🔒 paper/                          # (gitignored) author-only research artifacts
│
├── 📄 pyproject.toml                  # Package metadata
├── 📄 .env.example                    # TINKER_API_KEY template (gitignored: .env)
├── 📄 .gitignore                      # Keeps paper/ and results/ local
└── 📄 README.md                       # You are here
```

---

## 🔭 Future scope

The current Phase 1 is the **proof of concept** — a single reward function on a
single model on a single domain in a single 4-hour run. The roadmap beyond
this is where the research really gets interesting.

### 1. Scale the model 📈
- **Phase 2:** Same reward, 100B+ parameters (Qwen3.6-100B-A12B if budget allows)
- **Question:** Does the calibration gain hold at scale, or does the model's
  prior confidence become harder to override?

### 2. Cross-domain generalization 🌍
- **Train on:** GSM8K + MMLU-Pro (math + STEM)
- **Eval on:** TriviaQA, MMLU humanities, dialogue, code, medical QA, legal QA
- **Question:** Does "say I don't know" transfer to domains the model was
  never trained on, or is it domain-specific?

### 3. K=5 self-consistency AUROC 📊
- The current eval is K=1, so AUROC degenerates. Phase 2 will report proper
  self-consistency as a confidence signal.
- **Question:** Does the calibrated model also AGREE WITH ITSELF more?

### 4. Ablation battery 🧬
- Remove the calibration term → does accuracy crash?
- Remove the overconfident penalty → does overconfidence return?
- Remove the abstention bonus → does the model stop saying "I don't know"?
- **Question:** Which reward component is doing the heavy lifting?

### 5. The confidence head (parallel track) 🧠
- Train a 2-layer MLP on top of the frozen base model to predict P(correct)
  from the last hidden state.
- The RL changes *behavior*; the head reads *activations*. They target different
  things. Best case: both work and validate each other.

### 6. Multi-turn dialogue calibration 💬
- Extend the reward to multi-turn conversations. The model should escalate
  uncertainty across turns: "I'm 60% sure" → "I read more, now I'm 40% sure."
- **Question:** Can a model LEARN to update its own confidence?

### 7. Adversarial honesty probes ⚔️
- Build a benchmark of questions designed to elicit overconfidence
  (trick questions, false premises, leading prompts).
- **Question:** Is the calibrated model also robust to manipulation?

### 8. On-device / edge deployment 📱
- Distill the metacognitive 35B into Qwen3-4B or Llama-3.2-3B
- Quantize to Q4_K_M, target Jetson Orin Nano 8GB
- **Question:** Does calibration survive quantization and distillation?

### 9. Multi-agent metacognition 🤝
- Multiple calibrated models debate. The least-confident one is the tiebreaker.
- **Question:** Does ensemble-of-calibrated beat single calibrated?

### 10. Production agent integration 🦾
- Plug the calibrated model into an agentic workflow (CVC, HydroPlus, etc.)
- Use the model's self-reported confidence to gate human-in-the-loop.
- **Question:** Does knowing-when-it-doesn't-know reduce agent failure modes?

---

## 🗺️ Roadmap

| Phase | Status | What | Cost | Wall time | ETA |
|---|---|---|---|---|---|
| **0** | ✅ Done | Baseline ECE on Qwen3.6-35B-A3B | ~$1 | ~6 min | shipped |
| **1** | ✅ Done | RL on metacognitive reward (this repo) | ~$15 | ~4 h | shipped |
| **2** | 📋 Planned | Larger n_eval=200, K=5 self-consistency, OOD eval | ~$30 | ~6 h | 2-4 weeks |
| **3** | 📋 Planned | Ablation battery (5 reward configurations) | ~$75 | ~20 h | 1-2 months |
| **4** | 📋 Planned | Scale to 100B+ parameters | ~$150 | ~12 h | 2-3 months |
| **5** | 📋 Planned | Cross-domain eval (10 benchmarks) | ~$50 | ~3 days | 3-4 months |
| **6** | 📋 Planned | Confidence head training (parallel track) | ~$40 | ~1 day | 4-5 months |
| **7** | 💭 Concept | Multi-turn dialogue calibration | TBD | TBD | 6+ months |
| **8** | 💭 Concept | Adversarial honesty probes | TBD | TBD | 6+ months |
| **9** | 💭 Concept | On-device distillation + quantization | TBD | TBD | 9+ months |
| **10** | 💭 Concept | Multi-agent metacognition | TBD | TBD | 12+ months |

---

## 🌐 Ecosystem

This repo is one piece of a larger research program on agentic AI.
The other pieces:

| Project | What | Link |
|---|---|---|
| **CVC** (Cognitive Version Control) | Git for the AI's brain — Merkle-DAG for LLM cognitive state, 3-tier local store, MCP server, agent mode. *The agentic runtime that this calibrated model will plug into.* | **[jaimeena.com/cvc](https://jaimeena.com/cvc)** · [PyPI: tm-ai](https://pypi.org/project/tm-ai/) |
| **HydroPlus** | AI-powered LMS with OCR + multi-model RAG + cognitive fingerprinting + course co-creator | Internal |
| **HydroMain** | Public marketing site for lvl360, SSO into HydroPlus | Internal |
| **Portfolio** | Personal site with the "Sophia" AI assistant (ADK + Gemini 2.5 Flash) | Internal |

The long-term vision: **CVC is the runtime, Metacog is the confidence signal,
the agent knows when to ask for help.** Together they make agentic AI that
*knows its limits* — not just fast and capable, but honest about what it
doesn't know.

> 🔗 Learn more about CVC: **[https://jaimeena.com/cvc](https://jaimeena.com/cvc)**
>
> *Silent integration:* if you're using CVC, the metacog-trained LoRA can be
> mounted as a `cvc://metacog/qwen36-35b-a3b` resource. See the CVC docs for
> `cvc://` URI conventions and how to register external LoRA adapters.

---

## 🤝 Contributing

This is research code in active development. The cleanest ways to help:

1. **Run the pipeline on a new model and report the numbers.** Open an issue with the
   config, the dataset, and the resulting ECE/accuracy.
2. **Try a different reward function.** Every term in `training/reward.py` is a
   knob. Try `r_format = 0`, or invert the overconfident penalty, or add a
   novelty bonus. File what you find.
3. **Build an OOD eval.** Pick a domain not in {math, STEM} — code, dialogue,
   creative writing, medical, legal — and measure calibration there.
4. **Test adversarial prompts.** Try to break the calibration with leading
   questions, false premises, or jailbreak-style prompts. We want to know.

PRs welcome. Issues are even more welcome — they go into the paper.

---

## 🛡️ Requirements

| | |
|---|---|
| **Python** | 3.11+ |
| **GPU** | None locally — Tinker runs the training in their cloud |
| **API key** | [Tinker](https://tinker-console.thinkingmachines.ai/) ($20 buys the full pipeline) |
| **Browser** | Modern (Chrome / Firefox / Safari) for the dashboard |
| **Disk** | ~50 MB for the repo, ~10 MB per checkpoint, ~5 MB per results dir |

---

## 📜 License

MIT. Use it, fork it, publish with it, ship it in a product. If you beat our
numbers, we want to know — open an issue and link the paper.

---

## 📚 Citation

If you use this work in academic research, please cite it as:

```bibtex
@software{metacog2026,
  title  = {Metacog: Calibrated Self-Knowledge in 35B Open-Weight Models
            via Reinforcement Learning on Verifiable Metacognitive Rewards},
  author = {Meena, Jai kumar},
  year   = {2026},
  month  = jun,
  url    = {https://github.com/mannuking/metacog},
  note   = {Phase 1 result: ECE 0.50 → 0.10, overconfidence 57% → 0%,
            accuracy 42% → 80% in a 4-hour, \$15 RL run on Qwen3.6-35B-A3B}
}
```

---

## 🙏 Acknowledgments

- **Thinking Machines Lab** for the [Tinker](https://thinkingmachines.ai/tinker/) RL API
- **Qwen team** for the [Qwen3.6-35B-A3B](https://qwen.ai) base model
- **HuggingFace** for GSM8K and MMLU-Pro
- The **metacognition research community** — especially the "Mirror" benchmark
  authors and the AAAI 2026 "Toward Artificial Metacognition" workshop organizers

---

<div align="center">

**Built in 24 hours. Total cost <$20. MIT licensed. The whole point is that
*you* can do this too.**

*If a 35B model can learn to say "I don't know" in 4 hours, what else is
waiting to be taught the same way?*

<br/>

⭐ **Star this repo if you want Phase 2 to happen.** ⭐

<br/>

[🔗 CVC](https://jaimeena.com/cvc) · [📦 tm-ai on PyPI](https://pypi.org/project/tm-ai/) · [💬 Open an issue](https://github.com/mannuking/metacog/issues/new)

</div>
