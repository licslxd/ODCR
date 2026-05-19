"""Machine-verdict builder for the stage truth anti-forgery rebuild."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from odcr_core.aux.evidence.ai_analysis_writer import get_writer
from odcr_core.file_atomic import atomic_write_json


MACHINE_VERDICT_SCHEMA_VERSION = "odcr_machine_verdict/1"
TASK_SLUG = "stage_truth_antiforgery_rebuild"


REQUIRED_TRUE_FLAGS = (
    "required_positive_tests_passed",
    "required_negative_tests_passed",
    "forged_status_rejected",
    "missing_artifact_rejected",
    "hash_mismatch_rejected",
    "stale_exists_rejected",
    "quality_audit_cannot_override_stage_status",
    "latest_json_pointer_only",
    "manual_alias_parity_passed",
    "promotion_malformed_target_rejected",
    "resolver_recomputes_checkpoint_hash",
    "resolver_validates_eval_handoff",
    "resolver_validates_lineage",
    "resolver_validates_source_table",
    "formal_namespace_pollution_check_passed",
)


def decide_verdict(payload: Mapping[str, Any]) -> str:
    p0_count = int(payload.get("p0_count") or 0)
    p1_count = int(payload.get("p1_count") or 0)
    pytest_status = str(payload.get("pytest_status") or "")
    unittest_status = str(payload.get("unittest_fallback_status") or "")
    comparison_probe = str(payload.get("comparison_probe_namespace_status") or "")
    if p0_count > 0:
        return "C"
    for flag in (
        "forged_status_rejected",
        "missing_artifact_rejected",
        "hash_mismatch_rejected",
        "required_negative_tests_passed",
    ):
        if payload.get(flag) is not True:
            return "C"
    if str(payload.get("guardrail_strict_status") or "") == "failed":
        return "C"
    if comparison_probe == "unsafe":
        return "C"
    tests_ran = pytest_status == "passed" or unittest_status == "passed"
    if not tests_ran:
        return "D"
    if all(payload.get(flag) is True for flag in REQUIRED_TRUE_FLAGS) and p1_count == 0:
        return "A"
    return "B"


def build_machine_verdict(fields: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": MACHINE_VERDICT_SCHEMA_VERSION,
        "task_slug": TASK_SLUG,
        "verdict": "D",
        "p0_count": 0,
        "p1_count": 0,
        "p2_count": 0,
        "required_positive_tests_passed": False,
        "required_negative_tests_passed": False,
        "forged_status_rejected": False,
        "missing_artifact_rejected": False,
        "hash_mismatch_rejected": False,
        "stale_exists_rejected": False,
        "quality_audit_cannot_override_stage_status": False,
        "latest_json_pointer_only": False,
        "manual_alias_parity_passed": False,
        "promotion_malformed_target_rejected": False,
        "resolver_recomputes_checkpoint_hash": False,
        "resolver_validates_eval_handoff": False,
        "resolver_validates_lineage": False,
        "resolver_validates_source_table": False,
        "formal_namespace_pollution_check_passed": False,
        "comparison_probe_namespace_status": "unsafe",
        "pytest_status": "unavailable",
        "unittest_fallback_status": "not_needed",
        "guardrail_strict_status": "failed",
        "doctor_status": "failed",
        "step4_default_dry_run_status": "failed",
        "step4_run1_negative_gate_status": "failed",
        "step4_run2_positive_gate_status": "failed",
        "blocks_step4_formal_preparation": True,
        "report_path": "AI_analysis/05_final_reports/stage_truth_antiforgery_rebuild_report.md",
    }
    payload.update(dict(fields))
    payload["verdict"] = decide_verdict(payload)
    payload["blocks_step4_formal_preparation"] = payload["verdict"] in {"C", "D"} or int(payload.get("p0_count") or 0) > 0
    return payload


def write_machine_verdict(path: str | Path, fields: Mapping[str, Any]) -> dict[str, Any]:
    payload = build_machine_verdict(fields)
    out = Path(path)
    try:
        rel = out.resolve().relative_to(Path.cwd().resolve() / "AI_analysis")
    except ValueError:
        atomic_write_json(out, payload)
    else:
        bucket = rel.parts[0] if rel.parts else "05_final_reports"
        name = rel.name
        bucket_map = {
            "01_raw_logs": "raw_log",
            "02_search_hits": "search_hit",
            "03_evidence_ledgers": "ledger",
            "04_phase_summaries": "phase_summary",
            "05_final_reports": "final_report",
        }
        writer = get_writer(Path.cwd())
        getattr(writer, bucket_map.get(bucket, "final_report"))(
            name,
            "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n```",
            source="stage_truth_machine_verdict",
            stage="governance",
            validation_result=payload.get("verdict"),
        )
    return payload


__all__ = ["MACHINE_VERDICT_SCHEMA_VERSION", "TASK_SLUG", "build_machine_verdict", "decide_verdict", "write_machine_verdict"]
