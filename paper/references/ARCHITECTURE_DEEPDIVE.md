# Architecture deep-dive: cognitive origin of frontier models

**Purpose:** For the Metacog paper, we need a clear understanding of how
frontier models implement (or attempt to implement) metacognition —
self-knowledge, calibrated confidence, deliberative self-checking — so
that our LoRA fine-tune of Qwen3.6-35B-A3B can distill the **cognitive
origins** (the *what they do*, not the *weights*) of these models into
a small, edge-deployable form.

This document covers the architectures of:
1. Anthropic Claude Opus 4.7 / 4.8 / Mythos Preview
2. Moonshot Kimi K2.6
3. Qwen 3.6 (35B-A3B — our base)
4. MiniMax M3
5. GLM-5.1

**Two model names from the user request that I could NOT verify and need
to confirm before quoting in the paper:**
- *"Entropic Vercel 5"* — no such model surfaces in search. Vercel is a
  deployment platform (Vercel.com) and Entropic is not a known AI lab.
  The user's quote in the Kimi testimonials: *"More than 50% improvement
  on Next.js benchmark… compelling option for agentic coding and front-end
  generation through AI Gateway"* refers to Vercel as a deployment
  partner, not a model family. **Likely a misremembered name.**
- *"Mathos preview from Entropic"* — the closest match is **Anthropic
  Claude Mythos Preview** (their most-aligned preview model, released
  alongside Opus 4.8). "Mathos" also exists as a math-tutoring startup
  with an LLM engineer intern from Stanford; not a frontier model lab.
  **Likely confused with Claude Mythos Preview.**

We will reference Mythos Preview (real) and skip "Entropic" / "Vercel 5"
pending user confirmation.

---

## 1. Anthropic Claude Opus 4.7 / 4.8 / Mythos Preview

**Source of architecture facts:** Anthropic docs + release notes (May 2026).

### Key mechanisms relevant to metacognition

| Mechanism | Where it shows up | What it does for calibration |
|---|---|---|
| **Adaptive thinking (`thinking: {type: "adaptive"}` + `effort` parameter)** | Opus 4.7/4.8 default; Fable 5, Mythos 5, Mythos Preview always-on | The model itself decides *how much* to think. Low-stakes → cheap, high-stakes → deep. Effort parameter lets the user set a floor/ceiling. **This is the closest production-grade analogue to the metacog loop we want to fine-tune.** |
| **Manual `budget_tokens` (deprecated on Opus 4.6+ but still works)** | Opus 4.6 and earlier | Explicit token budget for the thinking block. Minimum 1,024 tokens. **We can do this at fine-tune time via a system prompt.** |
| **Summarized / omitted thinking (`display: "summarized" \| "omitted"`)** | Mythos Preview and Opus 4.8 default to `omitted` | The encrypted `signature` field carries the full thinking, but the visible text is empty. **Useful for our model: keep the reasoning silent, the answer explicit, and a separate confidence token.** |
| **Interleaved thinking** (thinking between tool calls) | Sonnet 4.6+ | The model can think *after* a tool result. **This is the architectural pattern we need: think → tool call → think about result → tool call → final.** |
| **Constitutional self-critique** (in Claude Code) | Opus 4.8 in Claude Code | Opus 4.8 is "the only model to complete every case end-to-end" on Super-Agent benchmark, with "tendency to proactively flag issues with the inputs and outputs of an analysis". Anthropic: "**~4× less likely than its predecessor to allow flaws in code it has written to pass unremarked**". **This is honesty as a calibrated behavior.** |
| **Effort control** (claude.ai) | New in 4.8 launch | A single dial that biases the model toward more or less effort. Maps to: "I have $N more tokens to think with, use them wisely". |

### What this means for our fine-tune

- **We need a `thinking` channel and a `text` channel**, separated. The model
  should be free to think long inside `<|think|>` without paying for it
  in the answer's apparent confidence.
- **We need an `effort` control** at inference time — the robot brain
  can call `metacog.run(question, effort=low|medium|high)` and get
  more or fewer self-checks.
- **The reward signal "be more honest when uncertain"** is exactly what
  Anthropic's "4× less likely to let flaws pass" line is talking about.
  This is what we will train on.

---

## 2. Moonshot Kimi K2.6 (1T MoE, 32B active)

**Source of facts:** Moonshot blog + DeepInfra overview (April 2026).

### Architecture

- 1T total params, 32B active per token (matches our 35B/3B ratio)
- 384 experts, 8 routed + 1 shared
- 61 layers
- 256K context
- **Multi-Head Latent Attention (MLA)** — DeepSeek-style, low KV memory
- Native INT4 and FP4 quantization (important for our edge target)
- 400M-param MoonViT vision encoder (internal, not exposed)

### Mechanisms relevant to metacognition

| Mechanism | Description | Our take |
|---|---|---|
| **Agent Swarm** (up to 300 sub-agents, 4,000 coordinated steps) | Decomposes a task into parallel sub-tasks, then synthesizes | Not directly applicable to a 35B edge model — too much orchestration overhead. But the *principle* of "decompose → solve pieces in parallel → synthesize" maps to **self-decomposition**: the model breaks its own answer into sub-claims, verifies each, then assembles. |
| **Long-horizon execution (12+ hours, 1,000+ tool calls)** | Designed to sustain attention over a long task | Edge model can't do this. But the **failure mode we want to avoid** is "model loses track mid-task and starts hallucinating" — which is exactly the *opposite* of calibrated confidence. |
| **Persistent memory** of intermediate results | Maintains context across tool calls | The metacog head needs an analogous "remember the steps you've already verified" mechanism. We can train this in. |
| **"Catches its own mistakes"** (factory.ai quote: "less likely to make coding errors or use hacks") | Self-debug behavior | This is metacognition in action. We want the same: model that catches its own answer before final. |

### What this means for our fine-tune

- **Kimi's MLA → our 35B base has Gated DeltaNet + Gated Attention (hybrid)**. The base model is already efficient; we just need the head.
- **Kimi's INT4 native quantization** is the path we'll use to deploy on Jetson (Q4_K_M in llama.cpp).
- **Self-decomposition** is the metacog technique we'll train: "before you give the final answer, list the 3 sub-claims, check each, and only then commit".

---

## 3. Qwen 3.6 (35B-A3B) — our base model

**Source:** Spheron Network, Sebastian Raschka's gallery, NVIDIA docs, Facebook DeepNet.

### Architecture

- **35B total / 3B active per token** (MoE)
- **256 experts per MoE layer (8 routed + 1 shared)**
- **Hybrid attention: 3:1 mix of Gated DeltaNet (linear) + Gated Attention**
  - 75% of layers are Gated DeltaNet (cheaper sequence mixer)
  - 25% are full Gated Attention (where the "thinking" attention happens)
- 16 Q heads / 2 KV heads (Grouped Query Attention)
- 262K native context (1M extensible)
- SwiGLU activations, RMSNorm

### What this gives us for free

- **3B active params** is small enough for the head to be a meaningful percentage (~17% if we use 528K head)
- **Hybrid attention** = the linear layers are cheap, the attention layers are where the metacog loop should happen
- **Grouped query attention** = efficient long-context (we can keep full trace history in KV cache cheaply)

### What we need to add

- The base model has NO explicit confidence signal — it just emits text
- The base model has NO notion of "I'm not sure" — it speaks with one voice
- The base model has NO ability to abstain — there's no "<answer>abstain</answer>" pattern in its training

---

## 4. MiniMax M3

**Source:** ZeniteQ, LinkedIn, Instagram (June 1, 2026 release).

### Architecture highlights

- Frontier coding + 1M context + native multimodal (text + image + video)
- **MSA (Multi-Stream Attention)** — the architecture change called out in the release
- Native multimodal from the foundation
- Open weights

### Mechanisms relevant to metacognition

| Mechanism | Description |
|---|---|
| **MSA (Multi-Stream Attention)** | "The most important element in it" per the release. Likely a way to attend to multiple reasoning streams in parallel (different "modes" of thinking). **Maps to our self-decomposition**: different "streams" can check different sub-claims. |
| **1M context** | Allows the model to keep the entire problem + all intermediate reasoning in context. We can use this to keep the full trace of self-checks. |
| **Native multimodal** | The model sees images and video natively. For robotics, this is the future — our edge model should be able to see. |

### What this means for our fine-tune

- **MSA-inspired self-decomposition**: emit multiple "reasoning streams" in parallel (as different paragraphs in the think block), then synthesize. Train the model to do this.
- **1M context** = we can keep K=5 self-consistency samples + all their analyses in one prompt for cross-checking.
- **Multimodal later** — Phase 3 of the paper.

---

## 5. GLM-5.1 (Zhipu / Z.ai)

**Source:** zai-org GitHub, BigModel pricing, Instagram coverage.

### Architecture highlights

- 754B parameters (sparse activation)
- Designed for **8-hour autonomous execution**
- Long-horizon task specialization

### Mechanisms relevant to metacognition

| Mechanism | Description |
|---|---|
| **8-hour autonomous execution** | The model sustains a single task for 8 hours. This requires robust self-monitoring — knowing when to stop, when to ask the user, when to re-plan. |
| **Agentic engineering focus** | Optimized for full software engineering tasks. Uses tools heavily. |
| **"算力整合" (compute integration)** | The pricing page mentions explicit compute allocation per task. The model is told "you have N tokens to think with" and uses them. |

### What this means for our fine-tune

- **Budgeted reasoning**: the model should learn to spend more tokens on hard problems and fewer on easy ones. This is a meta-level RL signal.
- **Self-termination**: the model should learn to say "I have high enough confidence, here's the answer" rather than always running to max tokens. **This is the efficiency gain we need for the robot brain.**

---

## Summary: cognitive mechanisms to distill into our fine-tune

| From | Mechanism | How we train it |
|---|---|---|
| Anthropic Opus 4.7/4.8 | Adaptive thinking + effort | System prompt with `effort: low \| medium \| high`; train head to map effort → thinking depth |
| Anthropic Mythos Preview | Honest self-critique | Reward: penalize confident wrong answers; reward "I'm not sure" when wrong |
| Anthropic Opus 4.8 (Claude Code) | Proactive flagging of issues | Train the model to emit `<verify>` blocks before `<answer>` |
| Kimi K2.6 | Self-decomposition | Train the model to list sub-claims, check each, then synthesize |
| Kimi K2.6 | Persistent trace memory | Use Gated Attention layers to keep full trace in KV cache |
| Qwen 3.6 (our base) | Hybrid attention (linear + full) | Use the linear layers for cheap steps, attention for verification |
| MiniMax M3 | Multi-Stream Attention | Train parallel "reasoning streams" in the think block |
| GLM 5.1 | Budgeted reasoning | Head outputs a `budget_remaining` token; RL trains it to stop early when confident |
| All | 100% correct final answer | Hard floor: any reward function must include `+1 if correct, 0 if not` (not negative) |

---

## 2 questions I need you to confirm

1. **"Entropic Vercel 5"** — I cannot find this. Is it possible you meant something else (e.g. a private paper, a model you saw in a paper that isn't public, or a misremembered name)? If you can give me a single source URL I can verify, I'll add it. Otherwise, the paper will skip it.

2. **"Mathos preview from Entropic"** — the closest real thing is **Anthropic Claude Mythos Preview** (their most-aligned model, released alongside Opus 4.8 in May 2026). I propose using **Mythos Preview** as the reference for the "most cautious, most honest" model in the paper. Is that what you meant?

Once these are confirmed, the paper's Section 4 (architecture deep-dive) is ready to write.
