"""Step4 formal handoff metadata refresh helpers.

These helpers are CPU-only and do not rerun Step4 inference.  They refresh
post-export fingerprints, rebuild stage_status through the readiness validator,
and write the task-level latest pointer only after the ready claim revalidates.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from odcr_core import path_layout, run_naming
from odcr_core.file_atomic import atomic_write_json
from odcr_core.index_contract import (
    INDEX_CONTRACT_FILENAME,
    ODCR_ROUTING_TRAIN_CSV,
    load_index_contract,
    refresh_index_contract_train_csv_fingerprint,
    write_index_contract,
)
from odcr_core.manifests import write_latest_pointer_json
from odcr_core.stage_status import build_and_write_stage_status


def _repo_relative(repo_root: Path, path: str | Path | None) -> str | None:
    if path is None:
        return None
    raw = str(path).strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    p = (repo_root / p).resolve() if not p.is_absolute() else p.resolve()
    try:
        return p.relative_to(repo_root).as_posix()
    except ValueError:
        return p.as_posix()


def _repo_path(repo_root: Path, raw: Any) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return (repo_root / path).resolve() if not path.is_absolute() else path.resolve()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _optional_missing_record(repo_root: Path, raw: Any, *, reason: str) -> dict[str, Any]:
    return {
        "path": _repo_relative(repo_root, raw),
        "optional": True,
        "missing_ok": True,
        "reason": reason,
    }


def _normalize_step4_run_summary(repo_root: Path, summary_path: Path) -> dict[str, Any]:
    summary = _load_json(summary_path)
    if not summary:
        return {}
    optional = dict(summary.get("optional_artifacts") or {})
    for key, optional_key, reason in (
        ("source_table_verbose_path", "source_table_verbose", "verbose_source_table_not_requested"),
        ("metrics_path", "metrics", "metrics_not_produced_for_stage"),
        ("lineage_path", "lineage", "lineage_not_required_for_stage"),
        ("training_runtime_config_path", "training_runtime_config", "optional_missing_with_reason"),
    ):
        raw = summary.get(key)
        path = _repo_path(repo_root, raw)
        if path is not None and not path.is_file():
            summary[key] = None
            optional[optional_key] = _optional_missing_record(repo_root, path, reason=reason)
    summary["optional_artifacts"] = optional
    atomic_write_json(summary_path, summary)
    return summary


def refresh_step4_handoff_metadata(
    *,
    repo_root: str | Path,
    task: int,
    run_id: str,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    rid = run_naming.parse_run_id(str(run_id))
    run_dir = path_layout.get_stage_run_root(root, int(task), "v1", "step4", rid).resolve()
    meta = run_dir / "meta"
    status_path = meta / "stage_status.json"
    before_status = _load_json(status_path)
    before_readiness = before_status.get("export_readiness") if isinstance(before_status, Mapping) else {}

    csv_path = run_dir / ODCR_ROUTING_TRAIN_CSV
    contract_path = run_dir / INDEX_CONTRACT_FILENAME
    contract = load_index_contract(str(contract_path))
    refreshed_contract = refresh_index_contract_train_csv_fingerprint(contract, str(csv_path))
    write_index_contract(refreshed_contract, str(contract_path))

    summary_path = meta / "run_summary.json"
    run_summary = _normalize_step4_run_summary(root, summary_path)
    status = build_and_write_stage_status(repo_root=root, stage="step4", task=int(task), run_id=rid)
    latest_path = run_dir.parent / "latest.json"
    latest_written = False
    if status.get("downstream_ready") is True and "step5" in {str(x) for x in (status.get("ready_for") or [])}:
        write_latest_pointer_json(
            repo_root=root,
            stage_unit_dir=run_dir.parent,
            run_id=rid,
            run_dir=run_dir,
            summary_path=summary_path,
            status=str(run_summary.get("status") or "ok"),
        )
        latest_written = True

    return {
        "schema_version": "odcr_step4_handoff_refresh/1",
        "repo_root": str(root),
        "task": int(task),
        "run_id": rid,
        "run_dir": _repo_relative(root, run_dir),
        "refreshed_artifacts": {
            "index_contract": _repo_relative(root, contract_path),
            "stage_status": _repo_relative(root, status_path),
            "run_summary": _repo_relative(root, summary_path),
            "latest_pointer": _repo_relative(root, latest_path) if latest_written else None,
        },
        "before": {
            "final_status": before_status.get("final_status"),
            "downstream_ready": before_status.get("downstream_ready"),
            "ready_for": list(before_status.get("ready_for") or []),
            "export_readiness_ready": before_readiness.get("ready") if isinstance(before_readiness, Mapping) else None,
            "rejection_reasons": list(before_status.get("rejection_reasons") or []),
        },
        "after": {
            "final_status": status.get("final_status"),
            "downstream_ready": status.get("downstream_ready"),
            "ready_for": list(status.get("ready_for") or []),
            "export_readiness_ready": (status.get("export_readiness") or {}).get("ready")
            if isinstance(status.get("export_readiness"), Mapping)
            else None,
            "rejection_reasons": list(status.get("rejection_reasons") or []),
        },
        "latest_written": latest_written,
    }


__all__ = ["refresh_step4_handoff_metadata"]
