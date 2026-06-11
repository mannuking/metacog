# 50+ candidate integrations for the Metacog architecture

This is a menu. Pick what you want in. Each item is one *integration* — a
discrete piece of behavior we can train into the LoRA, evaluate, and
include in the paper as an ablation.

Items marked **[CORE]** are essential to the basic "calibrated
metacognition" claim. Items marked **[EXP]** are more speculative and
add an ablation arm. Items marked **[DEFER]** are real ideas but better
suited to a Phase 2 / Phase 3 of the project.

---

## A. Confidence output mechanisms (how the model SAYS what it knows)

| # | Integration | What it does | When to use it |
|---|---|---|---|
| 1 | **[CORE] Explicit `<confidence>X.XX</confidence>` tag at end of every answer** | Model emits its own scalar confidence in [0, 1] after `<answer>`. Trained via RL. | Always — this is the basic metacog signal. |
| 2 | **[CORE] `<answer>abstain</answer>` option** | Model can refuse to answer if confidence < threshold. | Always — enables accuracy@100% by abstaining on hard questions. |
| 3 | **Token-level confidence header** (e.g. Qwen's own `<|conf|>`) | Use a reserved token whose logit becomes the confidence. | Alternative to (1) — slightly cheaper to decode. |
| 4 | **Calibrated softmax over `yes / no / maybe` then 3-way classify** | Map to numeric via (yes=0.9, maybe=0.5, no=0.1) | Cheap and interpretable; worse for fine-grained calibration. |
| 5 | **Multiple confidence estimates at different depths** | Emit one conf after each self-check. | Tracks the trajectory of certainty. |
| 6 | **Confidence with explicit reasoning** (e.g. `<conf reason="I checked X and got Y">0.7</conf>`) | Self-explainable calibration. | Better for the paper's qualitative analysis. |

## B. Self-verification mechanisms (how the model CHECKS its answer)

| # | Integration | What it does | When to use it |
|---|---|---|---|
| 7 | **[CORE] `<verify>` block after `<answer>`: re-derive the answer from scratch and flag if mismatch** | Catches arithmetic and logical errors. | Always. |
| 8 | **Sub-claim enumeration + check** (Kimi-style) | List N sub-claims, mark each true/false, recompute final. | Maps well to MMLU-Pro and GSM8K. |
| 9 | **Reverse verification** | Given the answer, can the model reconstruct the question? If not, the answer is probably memorized, not derived. | Catches hallucinations. |
| 10 | **Counterfactual check** | "If the answer were X instead, what part of the reasoning would change?" | Tests if the model actually understands vs pattern-matches. |
| 11 | **Unit-test style check** (math) | Plug the answer back into the problem and verify the constraint holds. | Essential for math (GSM8K). |
| 12 | **Sanity-bounds check** | "Is this answer in a plausible range?" (e.g. population of a city can't be 10^12). | Quick pre-filter for nonsense. |
| 13 | **Self-consistency with K=5** | Generate K=5 samples at temperature 0.7, majority vote. | Gold standard but expensive. Use as the **upper bound** measurement. |
| 14 | **Best-of-N with confidence-weighted voting** | K samples, weight by confidence, take argmax weighted sum. | Cheaper than full K=5, still principled. |
| 15 | **Adversarial check** | Generate a "devil's advocate" argument for the wrong answer. If the model can be convinced, the original was weak. | Strong hallucination filter; expensive. |
| 16 | **Two-pass decoding** | First pass: full CoT. Second pass: just the final answer conditioned on first. | Often catches off-by-one errors in arithmetic. |

## C. Memory and context mechanisms

| # | Integration | What it does | When to use it |
|---|---|---|---|
| 17 | **Persistent scratchpad** | The model has an explicit `<scratchpad>` block that survives across turns. | Critical for the long-horizon agent case. |
| 18 | **Sliding-window self-summarization** | Every N tokens, model writes a summary of its own trace so far. | For very long robot reasoning sessions. |
| 19 | **Retrieval from past verifications** | The model can look up "did I already verify this?" | Reduces redundant self-checks. |
| 20 | **Cross-sample memory** | In a K-sample self-consistency batch, share verifications across samples. | 2-3× cheaper than K independent verifications. |

## D. Reasoning budget / effort control

| # | Integration | What it does | When to use it |
|---|---|---|---|
| 21 | **[CORE] `<effort>low \| medium \| high</effort>` from caller** | The robot brain can request "easy / medium / hard" mode. Low = 1 pass, high = verify + counterfactual. | The efficiency lever the paper hangs on. |
| 22 | **Self-set budget** | Model decides its own budget based on question difficulty. | Hard to train reliably, but elegant when it works. |
| 23 | **Token-remaining counter** | The model emits `<budget_remaining>N</budget_remaining>` and uses it to decide whether to do more checks. | Lets the model terminate early. |
| 24 | **Cost-aware thinking** | RL penalty for "I used 800 tokens to answer a 1-token question". | The robot's battery. |
| 25 | **Pause-and-resume** | Mid-reasoning, the model can emit `<pause>` and the caller decides to extend budget or finalize. | Maps to the GLM 5.1 / Kimi long-horizon pattern. |

## E. Hallucination mitigation

| # | Integration | What it does | When to use it |
|---|---|---|---|
| 26 | **"I don't know" prior** | RL trains a strong prior toward `<answer>abstain</answer>` when the question is out-of-distribution. | Critical for the safety story. |
| 27 | **Source-grounded answering** | If the question references a fact, model must emit `<cite source="...">...</cite>`. | Useful for retrieval-augmented settings. |
| 28 | **Factual consistency check** | After answering, model asks "does this contradict any obvious well-known fact?" | Catches confidently-wrong on common knowledge. |
| 29 | **Calibrated refusal thresholds** | `abstain` if confidence < 0.4 (highly conservative), else answer. | The 100%-correct contract: trade recall for precision. |
| 30 | **Adversarial input detection** | If the question contains prompt-injection patterns, emit `<unsafe>...</unsafe>` and refuse. | Necessary for the robot brain in the wild. |

## F. Self-improvement (meta-meta-cognition)

| # | Integration | What it does | When to use it |
|---|---|---|---|
| 31 | **Confidence calibration self-check** | After answering, model asks "given my confidence X, how often am I right at this level historically?" (looked up from a local stats file). | Trains the model to be aware of its own past miscalibration. |
| 32 | **Difficulty estimation** | Before answering, model emits `<difficulty>easy \| hard</difficulty>` and routes effort accordingly. | Maps to (21). |
| 33 | **Strategy selection** | "Should I do math, code, or recall for this question?" — model picks and self-monitors. | Multi-domain generalization. |
| 34 | **Self-distillation loop** | After training, the model generates its own training data for the next round. | Risky (can compound errors), but powerful. |

## G. Multi-agent / ensemble patterns

| # | Integration | What it does | When to use it |
|---|---|---|---|
| 35 | **Internal debate** | Two passes from the same model argue; the head picks the winner. | A 2x cost, but 2x sample size. |
| 36 | **Specialized heads** (one per domain) | Math-head, code-head, fact-head. Head chooses which to use. | Phase 3, not Phase 1. |
| 37 | **Cross-model check** | Use Qwen3.6 as the answerer, GLM-5.1 (or any other available) as the verifier. | Too expensive in practice, but interesting as a one-off experiment. |
| 38 | **Critic head** | A second head (trained jointly) scores every answer. | Pure ensemble; doubles inference cost. |

## H. Training-time mechanisms (what the RL can do beyond rewards)

| # | Integration | What it does | When to use it |
|---|---|---|---|
| 39 | **[CORE] Curriculum: easy → hard** | Start with GSM8K-easy, progress to MMLU-Pro-hard. | Standard but essential. |
| 40 | **Adversarial curriculum** | After baseline, generate hard examples the model fails on, retrain. | Phase 2. |
| 41 | **Self-play on verification** | Train two copies: one to answer, one to verify. Iterate. | Powerful but expensive. |
| 42 | **Reward shaping for calibration** | Don't just reward correctness, explicitly reward `|confidence - empirical_accuracy_at_confidence| → 0`. | This is the **metacog reward** we already specified. |
| 43 | **KL penalty against base model** | Stay close to Qwen3.6-35B-A3B so we don't drift. | Standard LoRA practice. |
| 44 | **Length normalization in reward** | Don't reward longer thinking just because it's longer. | Prevents the model from "padding to win". |

## I. Deployment / serving (for the Jetson and laptop)

| # | Integration | What it does | When to use it |
|---|---|---|---|
| 45 | **GGUF Q4_K_M export** | Standard llama.cpp format. | Phase 3 (deployment). |
| 46 | **Speculative decoding with confidence gating** | A tiny draft model proposes, the big model verifies, but only when confidence is low. | Halves the latency in easy cases. |
| 47 | **KV-cache compression for the trace** | Compress the think block in KV cache so the verification pass is cheap. | Edge optimization. |
| 48 | **INT4 confidence head** | Quantize the head separately from the base. | Edge optimization. |
| 49 | **Async self-check** | The model emits a candidate answer immediately, then runs verification in the background. Caller can return the candidate or wait. | Good for the robotics use case (act fast, check slow). |
| 50 | **Energy-aware thinking** | Phone's battery low → auto-effort=low. | The robot's "energy budget" abstraction. |

## J. Evaluation mechanisms (for the paper)

| # | Integration | What it does | When to use it |
|---|---|---|---|
| 51 | **Brier-skill-score vs base model** | Standard "did we improve?" metric. | Paper Section 7. |
| 52 | **Reliability diagram (10 bins)** | Visual: predicted conf vs actual acc per bin. | Paper Figure 2. |
| 53 | **AUROC of confidence as a predictor of correctness** | "Can the model's confidence be trusted as a filter?" | Paper Table 1. |
| 54 | **Selective accuracy at X% coverage** | "If we answer only the top 60% most confident, what's accuracy?" | The key 100%-correct curve. |
| 55 | **Expected Calibration Error (ECE) — 10-bin and 100-bin** | Both bin counts; show 100-bin is more sensitive. | Paper Table 1. |
| 56 | **Per-domain breakdown** | GSM8K vs MMLU-Pro vs ARC vs TruthfulQA. | Paper Section 7.4. |
| 57 | **Calibration at the final answer level vs the per-step level** | Two granularity views. | Paper Section 7.5. |
| 58 | **Token-level calibration** (per-token logprob vs eventual correctness) | Already partially captured; full version in appendix. | Paper Appendix B. |

---

## My recommended minimum for Phase 1 (the green-tick you asked for)

The absolute minimum to get a **publishable Phase 1** is:

- **CORE items: 1, 2, 7, 21, 39, 42, 43, 44** (8 items)
- **Eval items: 51, 52, 53, 54, 55, 56** (6 items)

That's a real, defensible paper.

**Strong Phase 1 (what I'd build if you said "all of it"):**

- All CORE items (1, 2, 7, 21, 39, 42, 43, 44)
- All eval items (51-58)
- Plus: 8 (sub-claim check), 11 (unit-test back-check), 12 (sanity-bounds), 26 (abstain prior), 29 (refusal threshold)
- Plus: 6 (confidence with reasoning), 16 (two-pass decode)

That's 20 items and a much richer paper.

---

## What I need from you

1. Confirm the 2 model names (Entropic Vercel 5, Mathos preview) — see ARCHITECTURE_DEEPDIVE.md
2. Tell me which integrations to include. Minimum phrase: "go core + eval". Max phrase: "go all". Or pick a custom set.
3. After that, I build Phase 1.
