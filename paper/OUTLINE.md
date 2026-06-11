# Paper outline (working draft)

## Title (working)
"Calibrated Self-Knowledge in 35B-Parameter Open-Weight Models via Reinforcement Learning on Verifiable Metacognitive Rewards"

## 1. Abstract
- Problem: open-weight 35B MoE is overconfident and wrong on multi-choice knowledge tasks
- Method: pure RL (no SFT) on Tinker with verifiable metacognitive reward
- Result: drops ECE from 0.50 → target 0.20, AUROC from 0.22 → target 0.65
- Cost: <$50 of $150 Tinker budget

## 2. Introduction
- Robots need cheap, accurate, self-aware reasoning
- Metacognition: "thinking about thinking"
- The output contract: 100% correct, calibrated confidence, cheap per call

## 3. Background
- Calibration (ECE, Brier, AUROC, reliability diagram)
- Self-consistency (Wang et al. 2022)
- Process reward models
- Tinker's verifiable-reward RL

## 4. Architecture deep-dive
- Anthropic Opus 4.7/4.8 (extended thinking, constitutional self-critique)
- Entropic Vercel 5 / Mathos preview (test-time compute allocation)
- MiniMax M3 (1M-context thinking interleaving)
- Moonshot Kimi K2.6 (tool-augmented reflection)
- Qwen 3.6 (Gated DeltaNet hybrid attention, 1M context)
- GLM 5.1 (zero-cost reasoning budget control)

## 5. Method
- Base model: Qwen3.6-35B-A3B
- LoRA rank 32, train attn+mlp
- Reward: r = 1·correct − 0.5·(1-correct)·conf if conf > 0.8
- K=5 self-consistency for evaluation

## 6. Experiments
- Phase 0: baseline (Qwen3.6-35B-A3B) — 50q GSM8K + 50q MMLU-Pro
- Phase 1: RL on metacog reward — 50q × 20 epochs
- Phase 2: larger eval, ablations, generalization

## 7. Results
- (TBD)

## 8. Discussion
- Why RL > SFT for calibration
- Compute vs accuracy frontier for edge deployment
- Failure modes, ablations

## 9. Related work
- (filled in)

## 10. Conclusion
- (TBD)

## Appendix
- A. Full hyperparameter configs
- B. Per-trace outputs
- C. Reliability diagrams (per bin)
- D. Prompt templates
- E. Reward function pseudocode
- F. Edge deployment (Jetson Orin Nano 8GB + 8GB-VRAM laptop)
