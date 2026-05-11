"""Machine verdict builder for Step4 evidence-level tuning gates."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from odcr_core.evidence_level import (
    E4_GPU_SHARD_FORWARD_BOUNDED,
    EvidenceLevelError,
    evidence_level_rank,
    parse_evidence_level,
)


MACHINE_VERDICT_SCHEMA_VERSION = "odcr_machine_verdict/1"
TASK_SLUG = "step4_evidence_level_truth_antifake_rebuild"

STEP4_EVIDENCE_DEFAULTS: dict[str, Any] = {
    "evidence_level_min_required_for_a": E4_GPU_SHARD_FORWARD_BOUNDED,
    "candidate_ranking_evidence_level": "",
    "schema_only_evidence_used_for_tuning": False,
    "proxy_score_present": False,
    "proxy_score_used_for_tuning": False,
    "fake_score_used_for_tuning": False,
    "final_candidate_actual_gpu_confirmed": False,
    "actual_gpu_forward_executed": False,
    "gpu_runtime_evidence": False,
    "candidate_source_is_cpu_preview": False,
    "candidate_source_is_real_gpu_forward": False,
    "eligible_for_formal_prompt": False,
}


def _level_allows_a(payload: Mapping[str, Any]) -> bool:
    try:
        level = payload.get("candidate_ranking_evidence_level") or payload.get("evidence_level")
        return evidence_level_rank(parse_evidence_level(level)) >= evidence_level_rank(E4_GPU_SHARD_FORWARD_BOUNDED)
    except EvidenceLevelError:
        return False


def decide_step4_evidence_verdict(payload: Mapping[str, Any]) -> str:
    if int(payload.get("p0_count") or 0) > 0:
        return "C"
    if payload.get("schema_only_evidence_used_for_tuning") is True:
        return "C"
    if payload.get("fake_score_used_for_tuning") is True:
        return "C"
    if payload.get("proxy_score_present") is True and payload.get("proxy_score_used_for_tuning") is True:
        return "C"
    if str(payload.get("guardrail_r116_status") or "").lower() == "failed":
        return "C"
    required_for_a = (
        "final_candidate_actual_gpu_confirmed",
        "actual_gpu_forward_executed",
        "gpu_runtime_evidence",
        "candidate_source_is_real_gpu_forward",
    )
    if payload.get("candidate_source_is_cpu_preview") is True:
        return "B"
    if not _level_allows_a(payload):
        return "D" if not (payload.get("candidate_ranking_evidence_level") or payload.get("evidence_level")) else "B"
    if any(payload.get(key) is not True for key in required_for_a):
        return "B"
    if int(payload.get("p1_count") or 0) > 0:
        return "B"
    return "A"


def build_step4_evidence_machine_verdict(fields: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": MACHINE_VERDICT_SCHEMA_VERSION,
        "task_slug": TASK_SLUG,
        "verdict": "D",
        "p0_count": 0,
        "p1_count": 0,
        "p2_count": 0,
        **STEP4_EVIDENCE_DEFAULTS,
        "formal_step4_allowed": False,
        "blocks_step4_formal_preparation": True,
        "report_path": "AI_analysis/05_final_reports/step4_evidence_level_truth_antifake_rebuild_report.md",
    }
    payload.update(dict(fields))
    payload["verdict"] = decide_step4_evidence_verdict(payload)
    payload["formal_step4_allowed"] = False
    payload["blocks_step4_formal_preparation"] = True
    if payload["verdict"] != "A":
        payload["eligible_for_formal_prompt"] = False
    return payload


def write_step4_evidence_machine_verdict(path: str | Path, fields: Mapping[str, Any]) -> dict[str, Any]:
    payload = build_step4_evidence_machine_verdict(fields)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "MACHINE_VERDICT_SCHEMA_VERSION",
    "TASK_SLUG",
    "build_step4_evidence_machine_verdict",
    "decide_step4_evidence_verdict",
    "write_step4_evidence_machine_verdict",
]
