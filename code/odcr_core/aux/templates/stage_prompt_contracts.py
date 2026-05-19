"""Stage-specific runtime prompt contracts."""

from __future__ import annotations


STAGE_PROMPT_CONTRACTS = {
    "step3": "bounded runtime validation only; no formal training",
    "step4": "E4/E5 evidence required for ranking; CPU preview is not tuning evidence",
    "step5A": "scorer-path validation only; do not change Step5A objective",
    "step5B": "explainer-path validation only; do not change Step5B objective",
}
