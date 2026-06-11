"""Eval subpackage for Metacog."""

from .metrics import (
    CalibrationReport,
    auroc,
    brier_score,
    calibration_report,
    expected_calibration_error,
    format_report,
    neg_avg_logprob_to_conf,
    overconfidence_rate,
    self_consistency_confidence,
    underconfidence_rate,
)
from .normalize import (
    extract_final_answer,
    is_correct,
    normalize_answer,
    normalize_letter,
    normalize_numeric,
)

__all__ = [
    "CalibrationReport",
    "auroc",
    "brier_score",
    "calibration_report",
    "expected_calibration_error",
    "extract_final_answer",
    "format_report",
    "is_correct",
    "neg_avg_logprob_to_conf",
    "normalize_answer",
    "normalize_letter",
    "normalize_numeric",
    "overconfidence_rate",
    "self_consistency_confidence",
    "underconfidence_rate",
]
