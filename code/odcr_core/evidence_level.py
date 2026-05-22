"""Evidence-level policy for Step4 tuning and formal eligibility."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence


E0_STATIC_CONFIG = "E0_static_config"
E1_SCHEMA_PREVIEW = "E1_schema_preview"
E2_CPU_REAL_DATA_NO_MODEL = "E2_cpu_real_data_no_model"
E3_GPU_TRANSPORT = "E3_gpu_transport"
E4_GPU_SHARD_FORWARD_BOUNDED = "E4_gpu_shard_forward_bounded"
E4_GPU_SHARD_FORWARD_BOUNDED_FORMAL_ENTRY = "E4_gpu_shard_forward_bounded_formal_entry"
E4_GPU_SHARD_FORWARD_BOUNDED_FORMAL_ENTRY_WITH_VALIDATION = (
    "E4_gpu_shard_forward_bounded_formal_entry_with_validation"
)
E5_STEP5_EXPLANATION_POST_TRAIN_EVAL_LIFECYCLE = "E5_rating_stability_control_post_train_eval_lifecycle"
E5_FORMAL_FULL_RUN = "E5_formal_full_run"

MIN_TUNING_EVIDENCE_LEVEL = E4_GPU_SHARD_FORWARD_BOUNDED
MIN_FORMAL_PROMPT_EVIDENCE_LEVEL = E4_GPU_SHARD_FORWARD_BOUNDED
MIN_FORMAL_ARTIFACT_EVIDENCE_LEVEL = E5_FORMAL_FULL_RUN


@dataclass(frozen=True)
class EvidenceLevelSpec:
    level: str
    rank: int
    description: str
    tuning_eligible: bool
    formal_prompt_eligible: bool
    formal_artifact_eligible: bool


EVIDENCE_LEVEL_SPECS: tuple[EvidenceLevelSpec, ...] = (
    EvidenceLevelSpec(E0_STATIC_CONFIG, 0, "config parsing only; no data execution", False, False, False),
    EvidenceLevelSpec(
        E1_SCHEMA_PREVIEW,
        1,
        "CPU schema/contract preview with proxy diagnostics; never tuning evidence",
        False,
        False,
        False,
    ),
    EvidenceLevelSpec(E2_CPU_REAL_DATA_NO_MODEL, 2, "real data without model GPU-forward", False, False, False),
    EvidenceLevelSpec(E3_GPU_TRANSPORT, 3, "CUDA/tmux transport availability only", False, False, False),
    EvidenceLevelSpec(
        E4_GPU_SHARD_FORWARD_BOUNDED,
        4,
        "bounded real Step3 checkpoint + real task data + GPU forward/generate + posterior RCR",
        True,
        True,
        False,
    ),
    EvidenceLevelSpec(
        E4_GPU_SHARD_FORWARD_BOUNDED_FORMAL_ENTRY,
        4,
        "historical bounded Step5 formal-entry lifecycle without epoch-end validation; no longer sufficient for Step5 formal readiness",
        True,
        False,
        False,
    ),
    EvidenceLevelSpec(
        E4_GPU_SHARD_FORWARD_BOUNDED_FORMAL_ENTRY_WITH_VALIDATION,
        4,
        "bounded Step5 formal-entry lifecycle with first train step and epoch-end validation pass",
        True,
        True,
        False,
    ),
    EvidenceLevelSpec(
        E5_STEP5_EXPLANATION_POST_TRAIN_EVAL_LIFECYCLE,
        5,
        "bounded RatingStabilityControl post-train checkpoint save, training teardown, CPU-staged reload, and eval-forward lifecycle",
        True,
        True,
        False,
    ),
    EvidenceLevelSpec(E5_FORMAL_FULL_RUN, 5, "formal full Step4 export", True, True, True),
)

_SPEC_BY_LEVEL = {spec.level: spec for spec in EVIDENCE_LEVEL_SPECS}
_LEVEL_BY_PREFIX: dict[str, str] = {}
for _spec in EVIDENCE_LEVEL_SPECS:
    _LEVEL_BY_PREFIX.setdefault(_spec.level.split("_", 1)[0], _spec.level)


class EvidenceLevelError(ValueError):
    """Raised when an artifact tries to use evidence below the required level."""


def parse_evidence_level(value: Any) -> str:
    """Return a canonical E0-E5 evidence-level string.

    ``value`` may be a level string or a payload containing ``evidence_level``.
    Missing or unknown values fail closed.
    """
    if isinstance(value, Mapping):
        value = value.get("evidence_level")
    raw = str(value or "").strip()
    if not raw:
        raise EvidenceLevelError("missing evidence_level")
    if raw in _SPEC_BY_LEVEL:
        return raw
    prefix = raw.split("_", 1)[0]
    if prefix in _LEVEL_BY_PREFIX:
        return _LEVEL_BY_PREFIX[prefix]
    raise EvidenceLevelError(f"unknown evidence_level: {raw!r}")


def evidence_level_rank(level: Any) -> int:
    return _SPEC_BY_LEVEL[parse_evidence_level(level)].rank


def evidence_level_policy() -> dict[str, Any]:
    return {
        "schema_version": "odcr_step4_evidence_level_policy/1",
        "min_tuning_evidence_level": MIN_TUNING_EVIDENCE_LEVEL,
        "min_best_candidate_evidence_level": MIN_TUNING_EVIDENCE_LEVEL,
        "min_patch_suggestion_evidence_level": MIN_TUNING_EVIDENCE_LEVEL,
        "min_machine_verdict_a_evidence_level": MIN_TUNING_EVIDENCE_LEVEL,
        "min_formal_prompt_evidence_level": MIN_FORMAL_PROMPT_EVIDENCE_LEVEL,
        "min_formal_artifact_evidence_level": MIN_FORMAL_ARTIFACT_EVIDENCE_LEVEL,
        "levels": [
            {
                "level": spec.level,
                "rank": spec.rank,
                "description": spec.description,
                "tuning_eligible": spec.tuning_eligible,
                "formal_prompt_eligible": spec.formal_prompt_eligible,
                "formal_artifact_eligible": spec.formal_artifact_eligible,
            }
            for spec in EVIDENCE_LEVEL_SPECS
        ],
    }


def _bool_is_true(payload: Mapping[str, Any], keys: Sequence[str]) -> bool:
    return any(payload.get(key) is True for key in keys)


def tuning_blockers(payload: Mapping[str, Any], *, min_level: str = MIN_TUNING_EVIDENCE_LEVEL) -> list[str]:
    blockers: list[str] = []
    try:
        level = parse_evidence_level(payload)
        if evidence_level_rank(level) < evidence_level_rank(min_level):
            blockers.append(f"evidence_level {level} is below required {min_level}")
    except EvidenceLevelError as exc:
        blockers.append(str(exc))
    if payload.get("superseded") is True:
        blockers.append("artifact is superseded")
    if payload.get("schema_only") is True or payload.get("preview_only") is True:
        blockers.append("schema/preview-only evidence cannot be used for tuning")
    if payload.get("proxy_score_present") is True or payload.get("fake_score_present") is True:
        blockers.append("proxy/fake score evidence cannot be used for tuning")
    if payload.get("not_for_tuning") is True or payload.get("not_for_candidate_ranking") is True:
        blockers.append("artifact is explicitly marked not-for-tuning")
    if payload.get("candidate_source_is_cpu_preview") is True:
        blockers.append("candidate source is CPU preview")
    if payload.get("actual_gpu_forward_executed") is not True:
        blockers.append("actual_gpu_forward_executed is not true")
    if payload.get("gpu_runtime_evidence") is not True:
        blockers.append("gpu_runtime_evidence is not true")
    return blockers


def require_min_evidence_level(payload: Mapping[str, Any], min_level: str, context: str) -> None:
    try:
        level = parse_evidence_level(payload)
    except EvidenceLevelError as exc:
        raise EvidenceLevelError(f"{context}: {exc}") from exc
    if evidence_level_rank(level) < evidence_level_rank(min_level):
        raise EvidenceLevelError(f"{context}: evidence_level {level} is below required {min_level}")


def assert_not_schema_only_for_tuning(payload: Mapping[str, Any]) -> None:
    blockers = tuning_blockers(payload, min_level=MIN_TUNING_EVIDENCE_LEVEL)
    if blockers:
        raise EvidenceLevelError("Step4 tuning evidence rejected: " + "; ".join(blockers))


def is_tuning_eligible(payload: Mapping[str, Any]) -> bool:
    return not tuning_blockers(payload, min_level=MIN_TUNING_EVIDENCE_LEVEL)


def is_formal_prompt_eligible(payload: Mapping[str, Any]) -> bool:
    if tuning_blockers(payload, min_level=MIN_FORMAL_PROMPT_EVIDENCE_LEVEL):
        return False
    if _bool_is_true(
        payload,
        (
            "not_for_formal_prompt",
            "not_for_patch_suggestion",
            "not_for_best_candidate",
            "not_for_machine_verdict_a",
        ),
    ):
        return False
    return True


def mark_schema_preview(payload: MutableMapping[str, Any] | Mapping[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out.update(
        {
            "evidence_level": E1_SCHEMA_PREVIEW,
            "evidence_class": "schema_contract_only",
            "schema_only": True,
            "preview_only": True,
            "proxy_score_present": True,
            "fake_score_present": True,
            "not_for_tuning": True,
            "not_for_candidate_ranking": True,
            "not_for_best_candidate": True,
            "not_for_patch_suggestion": True,
            "not_for_machine_verdict_a": True,
            "not_for_formal_prompt": True,
            "gpu_runtime_evidence": False,
            "actual_gpu_forward_executed": False,
            "actual_model_loaded_on_gpu": False,
            "force_gpu_forward": False,
            "route_values_are_proxy_preview": True,
            "route_values_must_not_be_used_for_tuning": True,
        }
    )
    return out


def mark_gpu_shard_forward(payload: MutableMapping[str, Any] | Mapping[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    defaults = {
        "evidence_class": "posterior_gpu_forward_bounded",
        "schema_only": False,
        "preview_only": False,
        "proxy_score_present": False,
        "fake_score_present": False,
        "not_for_tuning": False,
        "not_for_candidate_ranking": False,
        "not_for_best_candidate": False,
        "not_for_patch_suggestion": False,
        "not_for_machine_verdict_a": False,
        "not_for_formal_prompt": False,
        "gpu_runtime_evidence": True,
        "actual_gpu_forward_executed": True,
        "actual_model_loaded_on_gpu": True,
        "force_gpu_forward": True,
        "real_checkpoint_used": True,
        "real_task_data_used": True,
        "posterior_rcr_evidence": True,
        "bounded_validation": True,
        "not_formal_full_run": True,
        "candidate_actual_gpu_confirmed": True,
        "candidate_source_is_real_gpu_forward": True,
        "candidate_source_is_cpu_preview": False,
        "no_proxy_score_used": True,
        "no_schema_only_evidence_used": True,
    }
    for key, value in defaults.items():
        out.setdefault(key, value)
    out["evidence_level"] = E4_GPU_SHARD_FORWARD_BOUNDED  # internal-only evidence metadata
    return out


__all__ = [
    "E0_STATIC_CONFIG",
    "E1_SCHEMA_PREVIEW",
    "E2_CPU_REAL_DATA_NO_MODEL",
    "E3_GPU_TRANSPORT",
    "E4_GPU_SHARD_FORWARD_BOUNDED",
    "E4_GPU_SHARD_FORWARD_BOUNDED_FORMAL_ENTRY",
    "E4_GPU_SHARD_FORWARD_BOUNDED_FORMAL_ENTRY_WITH_VALIDATION",
    "E5_STEP5_EXPLANATION_POST_TRAIN_EVAL_LIFECYCLE",
    "E5_FORMAL_FULL_RUN",
    "EvidenceLevelError",
    "evidence_level_policy",
    "evidence_level_rank",
    "parse_evidence_level",
    "require_min_evidence_level",
    "is_tuning_eligible",
    "is_formal_prompt_eligible",
    "assert_not_schema_only_for_tuning",
    "mark_schema_preview",
    "mark_gpu_shard_forward",
    "tuning_blockers",
]
