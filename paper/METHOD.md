# Method

## Model
- **Base:** `Qwen/Qwen3.6-35B-A3B` (35B-param MoE, 3B active)
- **Adaptation:** LoRA, rank 32, applied to attention + MLP
- **Tokenizer:** Qwen3.6 (vocab=248,044), stop on `<|im_end|>`

## Data
- **Training (n=50):** 25 GSM8K + 25 MMLU-Pro STEM (mixed math + knowledge)
- **Eval (n=20):** Held-out split, sampled from the same distributions
- **Eval (Phase 0 baseline, n=50):** 25 GSM8K + 25 MMLU-Pro STEM (no LoRA, K=1)

## Prompt contract
Each rollout is given a strict output template:
```
<think>...</think>
<verify>...</verify>
<answer>...</answer>
<confidence>0.0-1.0</confidence>
```
The reward function parses these blocks and assigns partial credit. A response that
fails to match the format gets `r_format_ok = 0.05` and is otherwise uncounted.

## Reward function
```
r = r_correct                          # 1.0 if answer == ground truth
  + r_correct_calibrated               # +0.5 if correct AND conf in [0.6, 0.95]
  + r_abstain_humble                   # +0.3 if abstained AND conf < 0.4
  - r_overconfident_wrong              # -0.5 if wrong AND conf > 0.8
  + 0.0 * humble_wrong                 # wrong-and-humble is OK, no penalty
  + r_format_ok                        # +0.05 for parseable structure
  + r_unit_check                       # +0.02 if numeric answer has units
  + r_sanity_bounds                    # +0.02 if confidence in [0.05, 0.99]
  - r_verify_conflict_penalty          # -0.05 if <verify> disagrees with <answer>
  + r_calibration                      # 0.2 × (1 - |conf - empirical_acc|)
  - r_length                           # -0.05 per 1k tokens, capped at -0.1
```

The key asymmetry: **overconfident-wrong is punished twice** (the `-0.5` base penalty
and the calibration term drops), while **humble-wrong is not punished at all**. This
is the entire trick — the model learns that "say you don't know" has a strictly higher
expected return than "guess with high confidence."

## RL algorithm
- **Method:** Tinker's verifiable-reward RL with importance-sampling loss
- **Group size:** K=4 rollouts per question
- **Optimizer:** AdamW (β1=0.9, β2=0.95, eps=1e-8, weight_decay=0)
- **Learning rate:** 1e-5 (constant)
- **Grad clip:** 1.0
- **KL penalty:** 0.0 (no reference model pull — we let the policy drift freely)
- **Temperature:** 1.0 for rollouts, 0.0 for in-loop evals

## Training schedule
- **n_steps:** 20
- **save_every:** 5 (saved step 5, 10, 15, 20)
- **eval_every:** 5 (in-loop held-out eval)
- **final_eval:** at step 20, n=20

## Compute
- **Total rollouts:** 20 steps × 50 questions × 4 samples = **4,000**
- **Total tokens generated:** ~4,000 × ~500 tokens = **~2M tokens**
- **Wall time:** 4h 13m (with ~1.5h Tinker queue hiccup; clean estimate 2.5h)
- **Cost:** <$15 of $150 Tinker credit (Tinker bills by sampled token)
- **Hardware:** Tinker cloud (TPU/H100 pool, model not local)
