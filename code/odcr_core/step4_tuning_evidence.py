"""Hard gates for Step4 RCR candidate ranking artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from odcr_core.evidence_level import (
    E4_GPU_SHARD_FORWARD_BOUNDED,
    EvidenceLevelError,
    assert_not_schema_only_for_tuning,
    require_min_evidence_level,
)


def require_step4_candidate_ranking_evidence(payload: Mapping[str, Any], *, context: str = "Step4 candidate ranking") -> None:
    require_min_evidence_level(payload, E4_GPU_SHARD_FORWARD_BOUNDED, context)
    assert_not_schema_only_for_tuning(payload)


def rank_step4_candidates(candidates: Sequence[Mapping[str, Any]], *, score_key: str = "score") -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        require_step4_candidate_ranking_evidence(candidate, context=f"Step4 candidate ranking {candidate.get('candidate_id', '<unknown>')}")
        ranked.append(dict(candidate))
    return sorted(ranked, key=lambda item: float(item.get(score_key) or 0.0), reverse=True)


def build_best_candidate_record(candidate: Mapping[str, Any], *, source_artifacts: Sequence[str]) -> dict[str, Any]:
    require_step4_candidate_ranking_evidence(candidate, context="Step4 best_candidate")
    return {
        "schema_version": "odcr_step4_best_candidate_evidence_gated/1",
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "evidence_level": str(candidate["evidence_level"]),
        "candidate_actual_gpu_confirmed": True,
        "actual_gpu_forward_executed": True,
        "gpu_runtime_evidence": True,
        "source_artifacts": list(source_artifacts),
        "no_proxy_score_used": True,
        "no_schema_only_evidence_used": True,
        "not_formal_full_run": bool(candidate.get("not_formal_full_run", True)),
    }


def build_patch_suggestion_text(candidate: Mapping[str, Any], *, body: str) -> str:
    require_step4_candidate_ranking_evidence(candidate, context="Step4 patch_suggestion")
    return (
        "# Step4 RCR Patch Suggestion\n\n"
        f"Evidence level: `{candidate['evidence_level']}`.\n\n"
        "This suggestion is based only on real GPU-forward posterior evidence.\n\n"
        f"{body.rstrip()}\n"
    )


def reject_superseded_best_candidate(path: str | Path) -> None:
    sidecar = Path(str(path) + ".superseded_by_real_gpu_evidence.json")
    if sidecar.is_file():
        raise EvidenceLevelError(f"superseded CPU-preview best-candidate artifact rejected: {path}")


__all__ = [
    "require_step4_candidate_ranking_evidence",
    "rank_step4_candidates",
    "build_best_candidate_record",
    "build_patch_suggestion_text",
    "reject_superseded_best_candidate",
]
