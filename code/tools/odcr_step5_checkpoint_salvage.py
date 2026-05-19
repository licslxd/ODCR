#!/usr/bin/env python3
"""Validate a failed Step5 checkpoint as a recovery/eval-only candidate."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.file_atomic import atomic_write_json  # noqa: E402
from odcr_core.training_checkpoint import (  # noqa: E402
    CheckpointLineageError,
    checkpoint_event_ledger_path_for_weight,
    checkpoint_file_sha256,
    checkpoint_lineage_path_for_weight,
    read_checkpoint_lineage,
    stable_hash,
)


SALVAGE_SCHEMA_VERSION = "odcr_step5_checkpoint_salvage_validator/1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _repo_rel(repo_root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return str(path.resolve())


def _torch_cuda_probe() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "hostname": platform.node(),
        "torch_cuda_available": False,
        "device_count": 0,
        "cuda_visible_devices": "",
    }
    try:
        import os
        import torch

        payload["cuda_visible_devices"] = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "")
        payload["torch_cuda_available"] = bool(torch.cuda.is_available())
        payload["device_count"] = int(torch.cuda.device_count())
    except Exception as exc:
        payload["torch_error"] = repr(exc)
    return payload


def _event_ledger_coherence(ledger: Mapping[str, Any], sidecar: Mapping[str, Any], checkpoint: Path) -> dict[str, Any]:
    events = ledger.get("saved_checkpoint_events") if isinstance(ledger.get("saved_checkpoint_events"), list) else []
    latest = ledger.get("latest_event") if isinstance(ledger.get("latest_event"), Mapping) else {}
    expected_hash = str(sidecar.get("checkpoint_file_hash") or "")
    checkpoint_text = str(checkpoint.resolve())
    matched = [
        dict(event)
        for event in events
        if isinstance(event, Mapping)
        and str(event.get("checkpoint_file") or event.get("path") or "") == checkpoint_text
        and str(event.get("checkpoint_file_hash") or event.get("hash") or "") == expected_hash
    ]
    latest_matches = (
        bool(latest)
        and str(latest.get("checkpoint_file") or latest.get("path") or "") == checkpoint_text
        and str(latest.get("checkpoint_file_hash") or latest.get("hash") or "") == expected_hash
    )
    computed_ledger_hash = stable_hash({k: v for k, v in dict(ledger).items() if k != "checkpoint_event_ledger_hash"}) if ledger else ""
    stored_ledger_hash = str(ledger.get("checkpoint_event_ledger_hash") or "")
    return {
        "ledger_exists": bool(ledger),
        "ledger_schema_version": ledger.get("schema_version") if ledger else None,
        "matching_event_count": len(matched),
        "latest_event_matches_checkpoint": latest_matches,
        "stored_ledger_hash": stored_ledger_hash,
        "computed_ledger_hash": computed_ledger_hash,
        "ledger_hash_matches": bool(stored_ledger_hash and stored_ledger_hash == computed_ledger_hash),
    }


def validate_step5_checkpoint_salvage(
    *,
    repo_root: str | Path,
    task: int,
    run_id: str | None,
    checkpoint: str | Path,
    attempt_gpu_eval: bool = False,
    write_recovery_ledger: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    ckpt = Path(checkpoint).expanduser()
    if not ckpt.is_absolute():
        ckpt = (root / ckpt).resolve()
    inferred_run_root = ckpt.parent.parent.resolve() if ckpt.parent.name == "model" else ckpt.parent.resolve()
    rid = str(run_id or inferred_run_root.name)
    run_root = root / "runs" / "step5" / f"task{int(task)}" / rid
    if run_root.resolve() != inferred_run_root:
        run_root = inferred_run_root
    meta = run_root / "meta"
    state = run_root / "state"
    run_summary = _load_json(meta / "run_summary.json")
    stage_status = _load_json(meta / "stage_status.json")
    step4_latest = _load_json(root / "runs" / "step4" / f"task{int(task)}" / "latest.json")
    step4_run_id = str(run_summary.get("from_step4") or "1")
    step4_stage_status = _load_json(root / "runs" / "step4" / f"task{int(task)}" / step4_run_id / "meta" / "stage_status.json")
    latest = _load_json(root / "runs" / "step5" / f"task{int(task)}" / "latest.json")
    head_latest = _load_json(root / "runs" / "step5" / f"task{int(task)}" / "latest_step5A.json")
    errors: list[str] = []
    notes: list[str] = []
    actual_hash = ""
    sidecar: dict[str, Any] = {}
    lineage_path = checkpoint_lineage_path_for_weight(ckpt)
    ledger_path = checkpoint_event_ledger_path_for_weight(ckpt)
    ledger = _load_json(ledger_path)
    if not ckpt.is_file():
        errors.append(f"checkpoint_missing: {ckpt}")
    else:
        actual_hash = checkpoint_file_sha256(ckpt)
        try:
            sidecar = read_checkpoint_lineage(ckpt, expected_stage="step5")
        except CheckpointLineageError as exc:
            errors.append(f"checkpoint_lineage_invalid: {exc}")
        if sidecar:
            expected_hash = str(sidecar.get("checkpoint_file_hash") or "")
            if expected_hash != actual_hash:
                errors.append(f"checkpoint_hash_mismatch: sidecar={expected_hash} actual={actual_hash}")
    if str(run_summary.get("status") or "").lower() != "failed":
        errors.append(f"source_run_not_failed_namespace: status={run_summary.get('status')!r}")
    if str(run_summary.get("from_step4") or "") != str(step4_run_id):
        errors.append("run_summary_from_step4_missing")
    if str(step4_latest.get("latest_run_id") or step4_latest.get("run_id") or step4_run_id) != str(step4_run_id):
        notes.append("step4_latest_pointer_shape_unexpected")
    if step4_stage_status.get("downstream_ready") is not True:
        errors.append("upstream_step4_not_downstream_ready")
    if str(sidecar.get("step4_export_lineage_hash") or "") and str(step4_stage_status.get("export_readiness", {}).get("lineage_hash") or ""):
        if str(sidecar.get("step4_export_lineage_hash")) != str(step4_stage_status.get("export_readiness", {}).get("lineage_hash")):
            errors.append("sidecar_step4_lineage_hash_mismatch_stage_status")
    latest_points_here = str(latest.get("latest_run_id") or latest.get("run_id") or "") == rid
    head_latest_points_here = str(head_latest.get("latest_run_id") or head_latest.get("run_id") or "") == rid
    if latest_points_here or head_latest_points_here:
        errors.append("failed_run_is_latest_pointer_target")
    ledger_coherence = _event_ledger_coherence(ledger, sidecar, ckpt)
    if not ledger_coherence["ledger_exists"]:
        errors.append("state_checkpoint_lineage_missing")
    elif not ledger_coherence["latest_event_matches_checkpoint"]:
        errors.append("state_checkpoint_lineage_latest_event_mismatch")
    cuda = _torch_cuda_probe()
    if attempt_gpu_eval and errors:
        notes.append("gpu_eval_skipped_due_static_errors")
    elif attempt_gpu_eval:
        notes.append("gpu_eval_not_implemented_in_static_salvage_validator_use_step5_eval_only_fresh_process")
    elif bool(cuda.get("torch_cuda_available")) and int(cuda.get("device_count") or 0) >= 2:
        notes.append("static_salvage_passed_gpu_eval_not_attempted")
    else:
        notes.append("blocked_no_gpu_for_fresh_eval_handoff")
    static_usable = not errors
    if errors:
        salvage_status = "not_usable"
    elif not bool(cuda.get("torch_cuda_available")) or int(cuda.get("device_count") or 0) < 2:
        salvage_status = "blocked_no_gpu"
    else:
        salvage_status = "usable"
    result = {
        "schema_version": SALVAGE_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "task": int(task),
        "run_id": rid,
        "run_dir": _repo_rel(root, run_root),
        "checkpoint": _repo_rel(root, ckpt),
        "checkpoint_exists": ckpt.is_file(),
        "checkpoint_hash": actual_hash or None,
        "checkpoint_lineage_path": _repo_rel(root, lineage_path),
        "checkpoint_lineage_valid": bool(sidecar) and not any(err.startswith("checkpoint_lineage") for err in errors),
        "sidecar_checkpoint_hash": sidecar.get("checkpoint_file_hash") if sidecar else None,
        "state_checkpoint_lineage_path": _repo_rel(root, ledger_path),
        "state_checkpoint_lineage_coherent": bool(
            ledger_coherence.get("ledger_exists") and ledger_coherence.get("latest_event_matches_checkpoint")
        ),
        "state_checkpoint_lineage": ledger_coherence,
        "run_summary_status": run_summary.get("status"),
        "stage_status_final_status": stage_status.get("final_status"),
        "failed_namespace_preserved": str(run_summary.get("status") or "").lower() == "failed",
        "latest_points_to_failed_run": latest_points_here,
        "head_latest_points_to_failed_run": head_latest_points_here,
        "upstream_step4_run": step4_run_id,
        "upstream_step4_downstream_ready": step4_stage_status.get("downstream_ready"),
        "static_usable_for_recovery_eval_input": static_usable,
        "attempt_gpu_eval": bool(attempt_gpu_eval),
        "cpu_staged_checkpoint_load_required": True,
        "direct_cuda_checkpoint_load_allowed": False,
        "fresh_eval_handoff_required": True,
        "salvage_status": salvage_status,
        "recovery_status": (
            "static_usable_needs_fresh_eval_handoff"
            if static_usable and salvage_status != "blocked_no_gpu"
            else "blocked_no_gpu"
            if static_usable
            else "not_usable"
        ),
        "torch_cuda_probe": cuda,
        "errors": errors,
        "notes": notes,
    }
    if write_recovery_ledger:
        out = root / "AI_analysis" / "03_evidence_ledgers" / f"step5A_salvage_{rid}_static_ledger.json"
        atomic_write_json(out, result)
        result["recovery_ledger_path"] = _repo_rel(root, out)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--attempt-gpu-eval", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate_step5_checkpoint_salvage(
        repo_root=args.repo_root,
        task=int(args.task),
        run_id=args.run_id,
        checkpoint=args.checkpoint,
        attempt_gpu_eval=bool(args.attempt_gpu_eval),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0 if result.get("salvage_status") in {"usable", "blocked_no_gpu"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
