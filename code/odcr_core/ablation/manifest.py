"""Ablation manifest construction and validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from odcr_core.ablation.registry import (
    ABLATION_VARIANTS,
    AblationValidationError,
    SCENARIO,
    entry_key,
    normalize_variant,
    registry_entry,
)


BASE_PROTOCOL = "paper_greedy_25"
RATING_SOURCE = "task_local_step3_accepted_scorer"
STATUS_PLANNED = "planned_skeleton_no_training"
NOTES_PLANNED = "Skeleton only. No training or evaluation has been run for this ablation."


def manifest_path_for_run(repo_root: str | Path, task: int, variant: str) -> Path:
    entry = registry_entry(repo_root, task, variant)
    output_run = str(entry.get("output_run") or "")
    return Path(repo_root).expanduser().resolve() / output_run / "meta" / "ablation_manifest.json"


def expected_manifest(repo_root: str | Path, task: int, variant: str) -> dict[str, Any]:
    _ = repo_root
    variant_s = normalize_variant(variant)
    if variant_s not in ABLATION_VARIANTS:
        raise AblationValidationError("only ablation variants have ablation manifests")
    return {
        "is_ablation": True,
        "task": int(task),
        "variant": variant_s,
        "scenario": SCENARIO,
        "base_protocol": BASE_PROTOCOL,
        "base_full_run": f"runs/step5/task{int(task)}/1_19",
        "rating_source": RATING_SOURCE,
        "forbidden_to_promote_latest": True,
        "paper_table_allowed": False,
        "requires_manual_review": True,
        "status": STATUS_PLANNED,
        "notes": NOTES_PLANNED,
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AblationValidationError(f"ablation manifest missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AblationValidationError(f"ablation manifest invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise AblationValidationError(f"ablation manifest must be a JSON object: {path}")
    return payload


def validate_manifest_payload(payload: Mapping[str, Any], *, task: int, variant: str) -> dict[str, Any]:
    variant_s = normalize_variant(variant)
    required = {
        "is_ablation": True,
        "task": int(task),
        "variant": variant_s,
        "scenario": SCENARIO,
        "base_protocol": BASE_PROTOCOL,
        "base_full_run": f"runs/step5/task{int(task)}/1_19",
        "rating_source": RATING_SOURCE,
        "forbidden_to_promote_latest": True,
        "paper_table_allowed": False,
        "requires_manual_review": True,
    }
    for field, expected in required.items():
        if payload.get(field) != expected:
            raise AblationValidationError(f"{entry_key(task, variant_s)} manifest {field} must be {expected!r}")
    if str(payload.get("status") or "") != STATUS_PLANNED:
        raise AblationValidationError(f"{entry_key(task, variant_s)} manifest status must be {STATUS_PLANNED}")
    notes = str(payload.get("notes") or "")
    if "No training or evaluation" not in notes:
        raise AblationValidationError(f"{entry_key(task, variant_s)} manifest notes must state no training/eval")
    return {
        "task": int(task),
        "variant": variant_s,
        "status": "pass",
        "paper_table_allowed": False,
        "forbidden_to_promote_latest": True,
    }


def validate_manifest_file(repo_root: str | Path, task: int, variant: str) -> dict[str, Any]:
    return validate_manifest_payload(
        _load_json(manifest_path_for_run(repo_root, task, variant)),
        task=task,
        variant=variant,
    )


def validate_all_manifests(repo_root: str | Path) -> dict[str, Any]:
    results = []
    for task in (7, 8):
        for variant in ABLATION_VARIANTS:
            results.append(validate_manifest_file(repo_root, task, variant))
    return {
        "schema_version": "odcr_ablation_manifest_validation/1",
        "status": "pass",
        "manifests": results,
    }
