"""Confidence head architecture.

The head is a small MLP that maps the LAST hidden state of a reasoning
step to P(step is correct). It is trained on top of a frozen base model
(or a LoRA-adapted one) so we don't touch the base weights.

Why a separate head instead of fine-tuning the model to output
confidence? Three reasons:

  1. The base model is already strong. We don't need more knowledge,
     we need better SELF-KNOWLEDGE.
  2. The head is tiny (~5M params for E4B), so it can be trained in
     seconds and re-trained cheaply when the base model is updated.
  3. The head gives us a calibrated probability directly. We don't have
     to parse "I'm 73% sure" out of free-form text.

For the Qwen3.6-35B-A3B base, hidden_size = 2048 (per the model card).
The head is a 2-layer MLP with a residual gate — the residual makes
initial weights behave like a constant predictor at 0.5, which matters
for stable RL starting points.

The head outputs a single logit; we apply sigmoid to get a probability
during inference. The loss is binary cross-entropy with a small label
smoothing term so the head doesn't saturate at 0/1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ConfidenceHeadConfig:
    hidden_size: int = 2048        # Qwen3.6-35B-A3B hidden dim
    bottleneck: int = 256          # intermediate size
    dropout: float = 0.1
    label_smoothing: float = 0.05  # BCE target: y in [0.05, 0.95] not {0, 1}
    use_residual_gate: bool = True  # start as a constant predictor at 0.5


class ConfidenceHead(nn.Module):
    """2-layer MLP with optional residual gate, outputs a single logit.

    Forward input:  hidden state of shape (..., hidden_size)
    Forward output: logit of shape (...)  (apply sigmoid for probability)

    The residual gate: logit = gate * MLP(h) + (1 - gate) * 0.0,
    where gate is a learnable scalar initialized to 0. This means the
    head starts as a constant predictor at logit 0 (= probability 0.5).
    As training progresses, gate grows toward 1 and the MLP takes over.
    """

    def __init__(self, config: ConfidenceHeadConfig | None = None):
        super().__init__()
        cfg = config or ConfidenceHeadConfig()
        self.cfg = cfg

        self.norm = nn.LayerNorm(cfg.hidden_size)
        self.down = nn.Linear(cfg.hidden_size, cfg.bottleneck, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(cfg.dropout)
        self.up = nn.Linear(cfg.bottleneck, 1, bias=True)

        if cfg.use_residual_gate:
            # Initialize to 0 so initial output is 0 (= probability 0.5)
            self.gate = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("gate", None)

        # Initialize the up projection to small values so early gradients
        # don't blow up. down is left at default (Kaiming) since GELU
        # activations benefit from it.
        nn.init.normal_(self.up.weight, std=0.01)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """hidden_state: (..., hidden_size) -> logit (...)"""
        h = self.norm(hidden_state)
        h = self.down(h)
        h = self.act(h)
        h = self.drop(h)
        logit = self.up(h).squeeze(-1)  # remove last dim
        if self.gate is not None:
            logit = torch.sigmoid(self.gate) * logit
        return logit

    def predict_proba(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Sigmoid-wrapped, for inference."""
        return torch.sigmoid(self.forward(hidden_state))

    def loss(
        self,
        hidden_state: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Binary cross-entropy with label smoothing.

        target: (...), values in [0, 1] (1.0 = step was correct, 0.0 = wrong).
        Label smoothing maps {0, 1} to {eps, 1-eps} for stability.
        """
        logit = self.forward(hidden_state)
        target_smooth = target * (1 - 2 * self.cfg.label_smoothing) + self.cfg.label_smoothing
        loss = F.binary_cross_entropy_with_logits(logit, target_smooth)
        with torch.no_grad():
            proba = torch.sigmoid(logit)
            mse = F.mse_loss(proba, target).item()
        return loss, {"bce": loss.item(), "prob_mse": mse}


# ─────────────────────────────────────────────────────────────────────────────
# Calibration probe: a SIMPLER baseline head for comparison
# ─────────────────────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    """Single linear layer from hidden state to logit.

    Cheaper to train, easier to interpret, often a strong baseline for
    confidence estimation. Used as a sanity check — if the 2-layer
    head doesn't beat this, the architecture is overkill.
    """

    def __init__(self, hidden_size: int = 2048, label_smoothing: float = 0.05):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1, bias=True)
        self.label_smoothing = label_smoothing
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h).squeeze(-1)

    def loss(self, h: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        logit = self.forward(h)
        target_smooth = target * (1 - 2 * self.label_smoothing) + self.label_smoothing
        loss = F.binary_cross_entropy_with_logits(logit, target_smooth)
        return loss, {"bce": loss.item()}

    def predict_proba(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(h))


# ─────────────────────────────────────────────────────────────────────────────
# Hidden-state pooling: how to turn a sequence of tokens into ONE vector
# ─────────────────────────────────────────────────────────────────────────────

def pool_hidden_states(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool a (B, T, H) hidden-state tensor into (B, H) using the
    last-non-padded token's hidden state.

    This is the standard "take the EOS / last real token" pattern used
    in decoder-only LM probes.
    """
    # attention_mask: (B, T) with 1 for real, 0 for pad
    # Last non-pad position per row
    seq_lens = attention_mask.sum(dim=1) - 1  # (B,)
    seq_lens = seq_lens.clamp(min=0)
    idx = seq_lens.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
    last_h = hidden.gather(1, idx).squeeze(1)
    return last_h


# ─────────────────────────────────────────────────────────────────────────────
# Training loop skeleton (one step, no actual training data wired up)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HeadTrainerState:
    step: int = 0
    best_ece: float = 1.0
    history: list[dict] = None

    def __post_init__(self):
        if self.history is None:
            self.history = []


def head_train_step(
    head: nn.Module,
    hidden: torch.Tensor,
    target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> dict:
    """One optimizer step on a single batch of (hidden, target) pairs.

    `hidden` is the LAST hidden state per example (shape (B, H)).
    `target` is in [0, 1] (shape (B,)).
    """
    head.train()
    optimizer.zero_grad()
    loss, metrics = head.loss(hidden, target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
    optimizer.step()
    return metrics


if __name__ == "__main__":
    # Smoke test: random head, no training
    cfg = ConfidenceHeadConfig(hidden_size=2048)
    head = ConfidenceHead(cfg)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"ConfidenceHead: {n_params:,} params")
    h = torch.randn(4, 2048)
    logit = head(h)
    proba = head.predict_proba(h)
    print(f"  input shape:  {tuple(h.shape)}")
    print(f"  logit shape:  {tuple(logit.shape)}")
    print(f"  proba shape:  {tuple(proba.shape)}")
    print(f"  mean proba:   {proba.mean().item():.3f}  (should be ~0.5 at init due to gate=0)")
    # One training step
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    metrics = head_train_step(head, h, target, opt)
    print(f"  train step:   {metrics}")

    # Linear probe baseline
    print("\nLinearProbe:")
    probe = LinearProbe(hidden_size=2048)
    n_params = sum(p.numel() for p in probe.parameters())
    print(f"  {n_params:,} params")
    logit = probe(h)
    print(f"  initial proba mean: {probe.predict_proba(h).mean().item():.3f}  (should be ~0.5 at init)")
