#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.config_resolver import resolve_config  # noqa: E402


STEP3_TOKENIZE_CACHE_SCHEMA_VERSION = "odcr_step3_tokenizer_cache/2"
STEP3_TOKENIZE_CACHE_COMPLETED_MARKER = "completed.marker"
STEP3_TOKENIZE_CACHE_FAILED_MARKER = "failed.marker"


def _snapshot_for_task(task_id: int) -> dict[str, Any]:
    _cfg, _sources, snapshot = resolve_config(
        config_path=REPO_ROOT / "configs" / "odcr.yaml",
        command="step3",
        task_id=int(task_id),
        set_overrides=[],
        dry_run=True,
        run_id="auto",
        mode="full",
    )
    return dict(snapshot)


def _task_domains(snapshot: Mapping[str, Any], task_id: int) -> tuple[str, str]:
    task = snapshot.get("task") if isinstance(snapshot.get("task"), Mapping) else {}
    source = str(task.get("source") or "").strip()
    target = str(task.get("target") or "").strip()
    if source and target:
        return source, target
    raise RuntimeError(f"step3 cache-check could not resolve task{int(task_id)} source/target")


def _num_proc_report(snapshot: Mapping[str, Any], *, would_hit_cache: bool) -> dict[str, Any]:
    hw = snapshot.get("hardware") if isinstance(snapshot.get("hardware"), Mapping) else {}
    budget = hw.get("worker_budget_formula") if isinstance(hw.get("worker_budget_formula"), Mapping) else {}
    selected = int(hw.get("tokenization_num_proc") or hw.get("num_proc") or 0)
    reserved = int(hw.get("reserved_cpu") or budget.get("reserved_cpu") or 0)
    max_cpu = int(hw.get("max_parallel_cpu") or budget.get("max_parallel_cpu") or 0)
    formula = str(
        hw.get("tokenization_num_proc_formula")
        or budget.get("tokenization_num_proc_formula")
        or f"num_proc({selected}) + reserved_cpu({reserved}) <= max_parallel_cpu({max_cpu})"
    )
    return {
        "selected_num_proc_if_rebuild": selected,
        "selected_num_proc": 0 if would_hit_cache else selected,
        "formula": formula,
        "max_parallel_cpu": max_cpu,
        "reserved_cpu": reserved,
        "phase": "warm_cache_hit" if would_hit_cache else "cold_pre_ddp_cache_build",
        "reason": "warm cache hit; tokenization workers not used" if would_hit_cache else "cold cache build uses auto tokenization num_proc",
    }


def _manifest_candidates(cache_root: Path) -> list[Path]:
    if not cache_root.is_dir():
        return []
    return sorted(cache_root.glob("*/cache_manifest.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True)


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _fast_manifest_decision(manifest_path: Path, *, task_id: int, source: str, target: str) -> dict[str, Any]:
    cache_dir = manifest_path.parent.resolve()
    base = {
        "cache_status": "miss",
        "cache_dir": str(cache_dir),
        "tokenization_compat_hash": "",
        "run_lineage_hash": "",
        "manifest_exists": manifest_path.is_file(),
        "completed": False,
        "hard_gate_match": False,
        "miss_reason": "not_checked",
        "rejected_fields": [],
        "record_only_mismatches": [],
        "would_hit_cache": False,
    }
    if (cache_dir / STEP3_TOKENIZE_CACHE_FAILED_MARKER).exists():
        base["miss_reason"] = "failed_marker_present"
        return base
    if not (cache_dir / "dataset_dict.json").is_file():
        base["miss_reason"] = "missing_dataset"
        return base
    manifest = _read_manifest(manifest_path)
    if not manifest:
        base["miss_reason"] = "missing_manifest"
        return base
    base["completed"] = manifest.get("completed") is True
    base["tokenization_compat_hash"] = str(
        manifest.get("tokenization_compat_hash") or manifest.get("tokenizer_cache_compat_hash") or ""
    )
    base["run_lineage_hash"] = str(manifest.get("run_lineage_hash") or "")
    schema = str(manifest.get("manifest_schema_version") or manifest.get("schema_version") or "")
    if schema == "odcr_step3_tokenizer_cache/1":
        base["miss_reason"] = "retired_v1_schema_rebuild_required"
        return base
    expected = {
        "manifest_schema_version": STEP3_TOKENIZE_CACHE_SCHEMA_VERSION,
        "schema_version": STEP3_TOKENIZE_CACHE_SCHEMA_VERSION,
        "stage": "step3",
        "task_id": int(task_id),
        "source_domain": str(source),
        "target_domain": str(target),
        "mode": "train_valid",
    }
    rejected = [key for key, value in expected.items() if manifest.get(key) != value]
    if rejected:
        base["miss_reason"] = f"{rejected[0]}_mismatch"
        base["rejected_fields"] = rejected
        return base
    if manifest.get("completed") is not True:
        base["miss_reason"] = "completed_false"
        return base
    if not (cache_dir / STEP3_TOKENIZE_CACHE_COMPLETED_MARKER).is_file():
        base["miss_reason"] = "completed_marker_missing"
        return base
    base.update(
        {
            "cache_status": "hit",
            "hard_gate_match": True,
            "miss_reason": "",
            "would_hit_cache": True,
        }
    )
    return base


def run_cache_check(
    *,
    task_id: int,
    expected_profile: str | None = None,
    expect_cache_hit: bool = False,
    allow_cold_build: bool = False,
    expect_num_proc: int | None = None,
    resolved_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot = dict(resolved_snapshot or _snapshot_for_task(int(task_id)))
    formal_profile = str(
        (snapshot.get("task") or {}).get("task_profile_id")
        or (snapshot.get("step3_task_profile") or {}).get("profile_id")
        or ""
    )
    if expected_profile and formal_profile != str(expected_profile):
        raise SystemExit(f"expected {expected_profile} but resolved {formal_profile}")
    source, target = _task_domains(snapshot, int(task_id))
    cache_root = REPO_ROOT / "cache" / "step3" / "tokenizer" / f"task{int(task_id)}" / f"{source}_to_{target}"
    manifests = _manifest_candidates(cache_root)
    best_decision: dict[str, Any] | None = None
    reusable_old_cache_exists = False
    for manifest_path in manifests:
        manifest = _read_manifest(manifest_path)
        fingerprint = manifest.get("fingerprint") if isinstance(manifest.get("fingerprint"), Mapping) else {}
        if not fingerprint:
            fingerprint = {
                key: manifest.get(key)
                for key in (
                    "manifest_schema_version",
                    "cache_role",
                    "stage",
                    "cache_version",
                    "task_id",
                    "source_domain",
                    "target_domain",
                    "mode",
                    "writer_code_version",
                    "tokenizer_cache_compat_hash",
                    "tokenization_compat_hash",
                    "data_contract_hash",
                    "preprocessing_artifact_hash",
                    "formal_cache_namespace",
                    "full_run_config_hash",
                    "source_table_hash",
                    "train_runtime_config_hash",
                    "optimizer_config_hash",
                    "performance_profile_hash",
                    "run_lineage_hash",
                    "record_only_lineage",
                    "tokenizer_cache_compat_payload",
                    "step3_tokenizer_config",
                    "dataset_split_info",
                    "source_csv_fingerprints",
                    "preprocess_latest_run_ids",
                    "preprocess_manifest_fingerprints",
                    "preprocess_source_table_fingerprints",
                    "preprocess_metrics_verify_fingerprints",
                    "schema_contract",
                    "upstream_gate_hash",
                    "compatibility_key",
                    "fingerprint_hash",
                )
            }
        decision = _fast_manifest_decision(manifest_path, task_id=int(task_id), source=source, target=target)
        if bool(decision.get("would_hit_cache")):
            reusable_old_cache_exists = True
            record_only = manifest.get("record_only_lineage") if isinstance(manifest.get("record_only_lineage"), Mapping) else {}
            manifest_profile = str(manifest.get("task_profile_id") or record_only.get("task_profile_id") or "")
            mismatches = list(decision.get("record_only_mismatches") or [])
            if manifest_profile and manifest_profile != formal_profile and "task_profile_id" not in mismatches:
                mismatches.append("task_profile_id")
            if str(manifest.get("full_run_config_hash") or "") != str((fingerprint or {}).get("full_run_config_hash") or ""):
                if "full_run_config_hash" not in mismatches:
                    mismatches.append("full_run_config_hash")
            decision["record_only_mismatches"] = sorted(set(mismatches))
            best_decision = decision
            break
        if best_decision is None:
            best_decision = decision
    if best_decision is None:
        best_decision = {
            "cache_status": "miss",
            "cache_dir": str(cache_root),
            "tokenization_compat_hash": "",
            "run_lineage_hash": "",
            "manifest_exists": False,
            "completed": False,
            "hard_gate_match": False,
            "miss_reason": "missing_cache_namespace",
            "rejected_fields": [],
            "record_only_mismatches": [],
            "would_hit_cache": False,
        }
    would_hit = bool(best_decision.get("would_hit_cache"))
    num_proc = _num_proc_report(snapshot, would_hit_cache=would_hit)
    if expect_num_proc is not None and int(expect_num_proc) != int(num_proc["selected_num_proc_if_rebuild"]):
        raise SystemExit(
            f"expected selected_num_proc_if_rebuild={int(expect_num_proc)} but resolved {num_proc['selected_num_proc_if_rebuild']}"
        )
    if expect_cache_hit and not would_hit:
        raise SystemExit(f"expected cache hit but would_hit_cache=false miss_reason={best_decision.get('miss_reason')}")
    if not would_hit and not allow_cold_build:
        pass
    status = dict(best_decision)
    status.update(
        {
            "schema_version": "odcr_step3_cache_check/1",
            "task_id": int(task_id),
            "formal_profile": formal_profile,
            "expected_profile": str(expected_profile or ""),
            "reusable_old_cache_exists": reusable_old_cache_exists,
            "estimated_rebuild_cost": "none" if would_hit else "cold_tokenization_required",
            "selected_num_proc_if_rebuild": num_proc["selected_num_proc_if_rebuild"],
            "selected_num_proc": num_proc["selected_num_proc"],
            "num_proc_formula": num_proc["formula"],
            "max_parallel_cpu": num_proc["max_parallel_cpu"],
            "reserved_cpu": num_proc["reserved_cpu"],
            "num_proc_phase": num_proc["phase"],
            "num_proc_reason": num_proc["reason"],
            "read_only": True,
        }
    )
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--expect-profile", default=None)
    parser.add_argument("--expect-cache-hit", action="store_true")
    parser.add_argument("--allow-cold-build", action="store_true")
    parser.add_argument("--expect-num-proc", type=int, default=None)
    args = parser.parse_args(argv)
    result = run_cache_check(
        task_id=args.task,
        expected_profile=args.expect_profile,
        expect_cache_hit=args.expect_cache_hit,
        allow_cold_build=args.allow_cold_build,
        expect_num_proc=args.expect_num_proc,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
