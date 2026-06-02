"""Variant-specific dry-run planning for weak cross-platform ablations."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from odcr_core.ablation.binding import load_ablation_binding
from odcr_core.ablation.guards import paper_table_gate
from odcr_core.ablation.manifest import manifest_path_for_run, validate_manifest_file
from odcr_core.ablation.registry import (
    AblationValidationError,
    entry_key,
    load_config_override,
    registry_entry,
    validate_config_override,
)


def _repo_root(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve()


def build_ablation_show(repo_root: str | Path, *, task: int, variant: str) -> dict[str, Any]:
    root = _repo_root(repo_root)
    key = entry_key(task, variant)
    entry = registry_entry(root, task, variant)
    override = load_config_override(root, task, variant)
    binding = load_ablation_binding(root, task=task, variant=variant)
    return {
        "schema_version": "odcr_ablation_show/1",
        "registry_key": key,
        "entry": entry,
        "config_override": override,
        "runtime_binding": binding.to_dict(),
        "manifest_path": manifest_path_for_run(root, task, variant).relative_to(root).as_posix(),
        "formal_training_started": False,
        "formal_eval_started": False,
    }


def build_ablation_dry_run_plan(repo_root: str | Path, *, task: int, variant: str) -> dict[str, Any]:
    root = _repo_root(repo_root)
    entry = registry_entry(root, task, variant)
    override = load_config_override(root, task, variant)
    validate_config_override(override, registry_entry=entry, key=entry_key(task, variant))
    manifest_result = validate_manifest_file(root, task, variant)
    safety = override.get("safety") if isinstance(override.get("safety"), dict) else {}
    if safety.get("dry_run_safe") is not True or safety.get("no_promote_latest") is not True:
        raise AblationValidationError("ablation dry-run requires dry_run_safe and no_promote_latest")
    snapshot_stub = {
        "valid_complete": False,
        "test_complete": False,
        "paper_greedy_25": True,
        "task_local_rating_source": True,
        "paper_table_allowed": False,
        "requires_manual_review": True,
    }
    output_run = str(entry.get("output_run"))
    binding = load_ablation_binding(root, task=task, variant=variant)
    return {
        "schema_version": "odcr_ablation_dry_run_plan/1",
        "dry_run_only": True,
        "would_start_training": False,
        "would_start_eval": False,
        "would_write_latest": False,
        "would_write_checkpoint": False,
        "registry_key": entry_key(task, variant),
        "task": int(task),
        "variant": str(variant),
        "scenario": entry.get("scenario"),
        "direction": entry.get("direction"),
        "output_run": output_run,
        "source_full_run": override.get("source_full_run"),
        "expected_step3_rating_source": override.get("expected_step3_rating_source"),
        "expected_step4_handoff_source": override.get("expected_step4_handoff_source"),
        "step5": override.get("step5"),
        "variant_controls": override.get("variant_controls"),
        "runtime_binding": binding.to_dict(),
        "safety": safety,
        "manifest_validation": manifest_result,
        "paper_table_gate": paper_table_gate(snapshot_stub),
        "future_formal_session_minimum_checks": [
            "./odcr ablation validate --task %d --variant %s" % (int(task), str(variant)),
            "confirm current GPU pane manually; do not run formal train/eval from this dry-run",
            "run formal ablation training only in the ablation namespace",
            "run formal valid/test eval only after training artifacts exist",
            "extract result snapshot and keep paper_table_allowed=false until manual review clears it",
        ],
    }
