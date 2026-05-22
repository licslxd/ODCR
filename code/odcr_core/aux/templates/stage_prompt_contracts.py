"""Stage-specific runtime prompt contracts."""

from __future__ import annotations


STAGE_PROMPT_CONTRACTS = {
    "step3": "bounded runtime validation only; no formal training",
    "step4": "E4/E5 evidence required for ranking; CPU preview is not tuning evidence",
    "rating_stability_control": "scorer-path validation only; do not change RatingStabilityControl objective",
    "step5_explanation": "explainer-path validation only; do not change Step5 explanation objective",
}
