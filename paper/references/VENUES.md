# Publication venue target list for the Metacog paper

**Compiled:** 2026-06-11 (Jai is at the gym; Phase 1 RL training is running)
**Status:** Live tracker. We will pick the target venue based on Phase 1 results.

The paper is about **calibrated self-knowledge in 35B-parameter open-weight
models via RL on verifiable metacognitive rewards**, with a target use case
of **edge robotics** (Jetson Orin Nano 8GB + 8GB-VRAM laptops).

This is squarely in the intersection of:
- LLM reasoning (RL fine-tuning, verifiable rewards)
- Calibration / uncertainty (ECE, Brier, AUROC)
- Test-time compute / metacognition
- Open-weight efficiency / edge deployment

So our target venues are the ones that explicitly welcome this blend.

---

## Tier 1: Main conference papers (highest prestige, longest prep)

| Venue | When | Where | Submission deadline | Fit | Notes |
|---|---|---|---|---|---|
| **NeurIPS 2026** | Dec 2026 | Sydney, AU | **~May 22, 2026** (passed for main track); position paper track Apr 9, 2026 | ✅ main track missed; position paper missed; **workshop track still open** | Main track is gone, but workshops are gold for this kind of work. |
| **ICML 2026** | Jul 13-19, 2026 | Vienna | **Jan 2026** (passed) | ❌ missed | The 2026 main conference is in progress. Position for ICML 2027. |
| **COLM 2026** (Conference on Language Modeling) | Oct 6-9, 2026 | San Francisco | **Apr 1, 2026** (passed) | ❌ main missed; **workshop track open** (Oct 9) | Workshop submission is the play. CfP is open. |
| **EMNLP 2026** | Nov 2026 | Suzhou, China | **May 27, 2026** (passed for main); ARR rolling | ⚠️ late | ARR (ACL Rolling Review) — possible to submit via ARR before final EMNLP cutoff. But the window is closing. |
| **AAAI 2027** | Feb 2027 | Philadelphia | **~Aug 2026** (~47 days from now) | ✅ **PRIMARY TARGET** | Perfect timing: Phase 1 done by mid-July, write paper in July, submit Aug 2026. Camera-ready Feb 2027. This is THE target. |
| **ICLR 2027** | Apr 2027 | TBA | **~Sep 2026** (likely) | ✅ secondary | A second top-tier option if AAAI slips. |

**Recommendation: AAAI 2027 as primary.** Timeline:
- Phase 1 RL done: ~mid-July 2026 (~5 weeks from now)
- Phase 2 ablations (if needed): late July
- Write paper: early August
- Submit: mid-August 2026 (before deadline)
- Reviews back: ~October
- Camera-ready: ~January
- Conference: February 2027

## Tier 2: Workshops (faster, more forgiving, on-topic)

These run alongside the main conferences and are often **the right venue**
for a focused, experimental paper like ours. Workshops are also where
the metacog community is most active.

| Workshop | Conf | When | Where | Submission | Fit | Notes |
|---|---|---|---|---|---|---|
| **Uncertainty-Aware NLP @ EMNLP 2026** (3rd edition) | EMNLP | Nov 2026 | Suzhou | rolling ARR | ✅✅✅ | **Perfect fit.** Calibration + LLMs. https://uncertainlp.github.io/ |
| **Reinforcement Learning Beyond Rewards (RLBREW) @ RLC 2026** | RLC | **Aug 15, 2026** | TBA | past; but workshop papers often accepted rolling | ✅✅ | Generalist agents, scalable RL. We do RL for calibration, not for reward itself. Good venue. |
| **ICML Workshop on Machine Learning for Cognition** | ICML | Jul 2026 | Vienna | (mid-2026) | ✅✅ | If it exists in 2026. |
| **NeurIPS Workshop on Self-Improving Agents** | NeurIPS | Dec 2026 | Sydney | TBA (typically Aug) | ✅✅ | Recent hot topic. |
| **COLM 2026 Workshop (any)** | COLM | Oct 9, 2026 | SF | rolling | ✅ | Lots of "Workshop on X" at COLM. Calibration of LMs is one. |
| **NeurIPS Workshop on Instruction Tuning and Following** | NeurIPS | Dec 2026 | Sydney | TBA | ⚠️ | Not quite calibration, but post-training focus. |
| **NeurIPS Workshop on Foundation Model Interventions** | NeurIPS | Dec 2026 | Sydney | TBA | ✅ | Intervention-style research fits. |

**Recommendation: Uncertainty-Aware NLP @ EMNLP 2026 as the workshop target.** If the timing aligns (we can have a workshop-ready paper by mid-October), this is a near-perfect fit for our paper.

## Tier 3: arXiv preprint (always, no deadline)

- **arXiv cs.CL / cs.AI / cs.LG**: Submit as soon as Phase 1 is done. Get a citable DOI. Most top venues accept arXiv submissions. This is also the **safety net** if we miss all deadlines.
- **Hugging Face Daily Papers**: 1-line submission via PR. Great for community reach. arXiv-only.

## Tier 4: Less traditional but interesting

| Venue | What | When | Why |
|---|---|---|---|
| **Tinker Cookbook / Tinker blog** | Thinking Machines has a developer blog | rolling | We use Tinker, they showcase users. **HIGH-VISIBILITY for the right audience.** Free to write. |
| **Latent Space podcast** | AI/ML podcast | rolling | If the paper has a story, this is the right audience. |
| **Import AI newsletter** (Jack Clark) | AI weekly | rolling | Free. Big audience. We could pitch. |
| **r/MachineLearning discussion thread** | Reddit | rolling | Free. Gets feedback from practitioners. |

## Recommended submission plan (concrete)

1. **This week (Jun 11-17, 2026)**:
   - Finish Phase 1 RL training
   - Run full eval suite
   - Start drafting paper Section 5 (Method) and Section 6 (Experiments) while results are fresh
2. **Late June**:
   - Phase 2 ablations (which loss fn? which K? which prompt format?)
   - Per-domain breakdown + reliability diagrams
3. **Early-mid July**:
   - Full paper draft
   - Internal review (Robin / Tina / Samantha characters)
4. **Mid-July**:
   - **arXiv preprint** (citable, no deadline pressure)
   - **Hugging Face Daily Papers** pitch
5. **August (by AAAI 2027 deadline ~Aug 15-20)**:
   - **AAAI 2027 main track submission** (PRIMARY TARGET)
6. **September-October**:
   - **Uncertainty-Aware NLP @ EMNLP 2026 workshop** (workshop track is more flexible timing)
   - **Tinker cookbook case study** (technical writeup for the Tinker community)
7. **November 2026**:
   - Present at EMNLP workshop (if accepted)
8. **February 2027**:
   - Present at AAAI 2027 (if accepted)

## What I need from you to start drafting

- A working title (working: "Calibrated Self-Knowledge in 35B-Parameter Open-Weight Models via Reinforcement Learning on Verifiable Metacognitive Rewards" — too long?)
- A 1-line elevator pitch (e.g. "We fine-tune a 35B open-weight model to know when it's wrong, and we show that the fine-tune costs $50 of compute")
- Your target venue preference (AAAI 2027 main track is my recommendation)

---

## Source citations

- AAAI 2027 deadline: aideadlines.org (47-day countdown)
- EMNLP 2026 main conf: myhuiban.com (May 27, 2026 deadline)
- EMNLP 2026 Uncertainty-Aware NLP workshop: uncertainlp.github.io
- COLM 2026: colmweb.org (Oct 6-9, 2026)
- RLC 2026: rl-conference.cc (August 15 workshop)
- "Mirror: A Hierarchical Benchmark for Metacognitive Calibration" — active metacog community
- "Toward Artificial Metacognition" AAAI-2026 talk — confirms the field is real and growing
