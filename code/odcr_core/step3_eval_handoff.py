"""Retired Step3 paper-eval handoff helpers.

Step3 downstream eligibility is now decided only by
``meta/readiness_audit.json`` through ``step3_upstream_readiness_gate``.  This
module remains as a fail-fast historical import boundary so old callers cannot
silently re-enable paper metrics as a Step3 gate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


EVAL_HANDOFF_SCHEMA_VERSION = "odcr_step3_eval_handoff/retired"
EVAL_HANDOFF_GATE_VERSION = "odcr_step3_eval_handoff_gate/retired"
PAPER_TARGET_ONLY_EVAL = "paper_target_only_eval"


class Step3EvalHandoffError(RuntimeError):
    """Raised when retired Step3 eval-handoff APIs are called."""


def _retired() -> Step3EvalHandoffError:
    return Step3EvalHandoffError(
        "Step3 eval_handoff is retired. Use meta/readiness_audit.json and "
        "step3_upstream_readiness_gate; paper metrics belong after Step3->Step4->Step5->eval/rerank."
    )


def default_eval_paths(run_root: str | Path) -> dict[str, Path]:
    root = Path(run_root).expanduser().resolve()
    return {
        "valid_eval_path": root / "meta" / "eval_only" / "retired" / "eval_paper_target_only_eval_valid" / "eval_summary.json",
        "test_eval_path": root / "meta" / "eval_only" / "retired" / "eval_paper_target_only_eval_test" / "eval_summary.json",
    }


def validate_step3_eval_handoff_evidence(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise _retired()


def build_eval_handoff_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise _retired()


def accept_step3_eval_handoff(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise _retired()


def load_eval_handoff(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise _retired()


def validate_accepted_eval_handoff(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise _retired()


def quality_audit_from_eval_handoff(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise _retired()


__all__ = [
    "EVAL_HANDOFF_GATE_VERSION",
    "EVAL_HANDOFF_SCHEMA_VERSION",
    "PAPER_TARGET_ONLY_EVAL",
    "Step3EvalHandoffError",
    "accept_step3_eval_handoff",
    "build_eval_handoff_payload",
    "default_eval_paths",
    "load_eval_handoff",
    "quality_audit_from_eval_handoff",
    "validate_accepted_eval_handoff",
    "validate_step3_eval_handoff_evidence",
]
