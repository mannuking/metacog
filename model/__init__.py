"""Model subpackage for Metacog — confidence heads, future architecture work."""

from .confidence_head import (
    ConfidenceHead,
    ConfidenceHeadConfig,
    HeadTrainerState,
    LinearProbe,
    head_train_step,
    pool_hidden_states,
)

__all__ = [
    "ConfidenceHead",
    "ConfidenceHeadConfig",
    "HeadTrainerState",
    "LinearProbe",
    "head_train_step",
    "pool_hidden_states",
]
