from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from odcr_core.config_schema import OneControlConfigError
from odcr_core.rating_source import RATING_SOURCE_TYPE, validate_rating_source

STEP5_EXPLANATION_HANDOFF_SCHEMA_VERSION = "odcr_step5_explanation_handoff/1"
STEP5_EXPLANATION_MODE = "explanation_only"


class Step5ExplanationHandoffError(OneControlConfigError):
    """Raised when a Step5 explanation handoff is incomplete."""


def build_step5_explanation_handoff(
    *,
    task: int,
    run_id: str,
    checkpoint: str,
    explanation_metrics: Mapping[str, Any],
    rating_source: Mapping[str, Any],
    generation_config: Mapping[str, Any] | None = None,
    ccv_fca_report: Mapping[str, Any] | None = None,
    route_explainer_stats: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": STEP5_EXPLANATION_HANDOFF_SCHEMA_VERSION,
        "stage": "step5",
        "mode": STEP5_EXPLANATION_MODE,
        "task_id": int(task),
        "run_id": str(run_id),
        "checkpoint": str(checkpoint),
        "explanation_metrics": dict(explanation_metrics),
        "generation_config": dict(generation_config or {}),
        "ccv_fca_report": dict(ccv_fca_report or {}),
        "route_explainer_stats": dict(route_explainer_stats or {}),
        "rating_source": dict(rating_source),
        "rating_source_type": str(rating_source.get("type") or RATING_SOURCE_TYPE),
        "no_rating_training_performed": True,
        "trains_rating": False,
        "status": "ok",
    }


def validate_step5_explanation_handoff(
    payload: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    require_checkpoint: bool = False,
) -> dict[str, Any]:
    if str(payload.get("schema_version") or "") != STEP5_EXPLANATION_HANDOFF_SCHEMA_VERSION:
        raise Step5ExplanationHandoffError("Step5 explanation handoff schema mismatch")
    if str(payload.get("stage") or "") != "step5":
        raise Step5ExplanationHandoffError("Step5 explanation handoff stage must be step5")
    if str(payload.get("mode") or "") != STEP5_EXPLANATION_MODE:
        raise Step5ExplanationHandoffError("Step5 handoff must be explanation_only")
    if bool(payload.get("trains_rating")) or not bool(payload.get("no_rating_training_performed")):
        raise Step5ExplanationHandoffError("Step5 explanation handoff must declare no rating training")
    rating_source = payload.get("rating_source")
    if not isinstance(rating_source, Mapping):
        raise Step5ExplanationHandoffError("Step5 explanation handoff missing rating_source")
    validated_source = validate_rating_source(rating_source, repo_root=repo_root)
    checkpoint = str(payload.get("checkpoint") or "").strip()
    if require_checkpoint:
        root = Path(repo_root).expanduser().resolve() if repo_root is not None else Path.cwd()
        path = Path(checkpoint)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise Step5ExplanationHandoffError(f"Step5 explanation checkpoint missing: {path}")
    return {
        **dict(payload),
        "rating_source": validated_source,
        "rating_source_status": "ok",
        "downstream_ready": True,
        "ready_for": ["eval", "rerank"],
    }
