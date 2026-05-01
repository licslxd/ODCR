#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.file_atomic import atomic_write_json  # noqa: E402
from odcr_core.preprocess_metadata import (  # noqa: E402
    PREPROCESS_A_SPLIT_LOSS_POLICY,
    parse_dataset_shell_log,
    refresh_preprocess_a_stage_payload,
    refresh_source_table_payload,
    unit_current_headers,
    validate_header_collection,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} JSON root must be an object.")
    return payload


def _write_json(path: Path, payload: dict[str, Any], *, write: bool) -> None:
    if write:
        atomic_write_json(path, payload)


def _repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _normalize_optional_artifacts(summary: dict[str, Any], *, write_log: list[str]) -> dict[str, Any]:
    out = dict(summary)
    key_artifacts = dict(out.get("key_artifacts") or {})
    optional_artifacts = dict(out.get("optional_artifacts") or {})
    reasons = {
        "errors_log": "no_error" if not out.get("latest_error") else "error_log_not_materialized",
        "debug_log": "debug_disabled",
        "samples_log": "samples_not_requested",
    }
    for key, reason in reasons.items():
        raw_path = key_artifacts.get(key) or out.get(f"{key}_path")
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path.exists():
            continue
        optional_artifacts[key] = {
            "path": _repo_rel(path),
            "optional": True,
            "missing_ok": True,
            "reason": reason,
        }
        key_artifacts.pop(key, None)
        write_log.append(f"optional_artifacts[{key}] marked missing_ok reason={reason}")
    out["key_artifacts"] = key_artifacts
    out["optional_artifacts"] = optional_artifacts
    return out


def _refresh_unit_statuses(meta_dir: Path, *, write: bool, write_log: list[str]) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    dataset_statuses: dict[str, Any] = {}
    task_statuses: dict[str, Any] = {}
    issues: list[str] = []
    for path in sorted((meta_dir / "datasets").glob("*.status.json")):
        payload = _read_json(path)
        headers = unit_current_headers("dataset", payload.get("output_files") or [])
        payload["current_headers"] = headers
        payload["metadata"] = parse_dataset_shell_log(
            payload.get("shell_log_path") or "",
            str(payload.get("unit_name") or path.stem.replace(".status", "")),
        )
        if payload.get("status") == "ok":
            issues.extend(validate_header_collection(headers))
        _write_json(path, payload, write=write)
        dataset_statuses[str(payload.get("unit_name"))] = payload
        write_log.append(f"refreshed dataset status {path}")
    for path in sorted((meta_dir / "tasks").glob("*.status.json")):
        payload = _read_json(path)
        headers = unit_current_headers("task", payload.get("output_files") or [])
        payload["current_headers"] = headers
        payload["metadata"] = {
            "merge_policy": "concatenate auxiliary and target train/valid splits with domain transport labels"
        }
        if payload.get("status") == "ok":
            issues.extend(validate_header_collection(headers))
        _write_json(path, payload, write=write)
        task_statuses[str(payload.get("unit_name"))] = payload
        write_log.append(f"refreshed task status {path}")
    return dataset_statuses, task_statuses, issues


def _summary_markdown(
    *,
    generated_at: str,
    write: bool,
    issues: list[str],
    stage_status: dict[str, Any],
    summary: dict[str, Any],
) -> str:
    status = stage_status.get("status")
    metadata = stage_status.get("metadata") if isinstance(stage_status.get("metadata"), dict) else {}
    stage_specific = metadata.get("stage_specific") if isinstance(metadata.get("stage_specific"), dict) else {}
    split_summary = stage_specific.get("split_policy_summary") or {}
    cleaning_summary = stage_specific.get("cleaning_summary") or {}
    repro = metadata.get("reproducibility_evidence") or {}
    optional = summary.get("optional_artifacts") or {}
    datasets = stage_specific.get("dataset_inputs_outputs") or {}
    total_headers = 0
    true_headers = 0
    for item in datasets.values():
        for header in (item.get("current_headers") or {}).values():
            total_headers += 1
            true_headers += int(bool(header.get("exists")) and bool(header.get("header_match")))
    merged = stage_specific.get("merged_task_outputs") or {}
    for item in merged.values():
        for header in (item.get("current_headers") or {}).values():
            total_headers += 1
            true_headers += int(bool(header.get("exists")) and bool(header.get("header_match")))

    lines = [
        "# preprocess_a metadata lineage repair",
        "",
        f"- generated_at_utc: `{generated_at}`",
        f"- write_mode: `{write}`",
        f"- stage_status: `{status}`",
        f"- current_headers_ok: `{true_headers}/{total_headers}`",
        f"- issues: `{len(issues)}`",
        f"- optional_artifacts: `{', '.join(sorted(optional)) or 'none'}`",
        f"- split_loss_policy: `{PREPROCESS_A_SPLIT_LOSS_POLICY}`",
        f"- fingerprint_hash: `{repro.get('fingerprint_hash', '')}`",
        f"- git_dirty: `{repro.get('git_dirty', '')}`",
        f"- dirty_status_count: `{repro.get('dirty_status_count', '')}`",
        "",
        "## Split Policy Summary",
    ]
    for dataset, stats in sorted(split_summary.items()):
        lines.append(
            f"- {dataset}: processed={stats.get('processed_rows')} train={stats.get('train_rows')} "
            f"valid_before={stats.get('valid_rows_before_filter')} test_before={stats.get('test_rows_before_filter')} "
            f"valid_after={stats.get('valid_rows_after_filter')} test_after={stats.get('test_rows_after_filter')} "
            f"split_loss={stats.get('split_loss_rows')}"
        )
    lines.append("")
    lines.append("## Cleaning Summary")
    for dataset, stats in sorted(cleaning_summary.items()):
        if stats.get("cleaning_expected"):
            lines.append(
                f"- {dataset}: dropped_empty_review_rows={stats.get('dropped_empty_review_rows')} "
                f"k_core_removed_rows={stats.get('k_core_removed_rows')} "
                f"review_non_empty_validation_passed={stats.get('review_non_empty_validation_passed')}"
            )
    if not issues:
        lines.append("")
        lines.append("No remaining metadata/header consistency issues were detected.")
    else:
        lines.append("")
        lines.append("## Issues")
        lines.extend(f"- {issue}" for issue in issues)
    return "\n".join(lines) + "\n"


def refresh_run(run_dir: Path, *, write: bool, ai_analysis_dir: Path) -> int:
    meta_dir = run_dir / "meta"
    generated_at = _utc_now()
    raw_log: list[str] = [
        f"generated_at_utc={generated_at}",
        f"run_dir={run_dir}",
        f"write={write}",
    ]

    stage_status_path = meta_dir / "stage_status.json"
    stage_manifest_path = meta_dir / "stage_manifest.json"
    source_table_path = meta_dir / "source_table.json"
    run_summary_path = meta_dir / "run_summary.json"

    stage_status, stage_issues = refresh_preprocess_a_stage_payload(
        _read_json(stage_status_path),
        repo_root=REPO_ROOT,
        meta_root=meta_dir,
    )
    stage_manifest, manifest_issues = refresh_preprocess_a_stage_payload(
        _read_json(stage_manifest_path),
        repo_root=REPO_ROOT,
        meta_root=meta_dir,
    )
    unit_datasets, unit_tasks, unit_issues = _refresh_unit_statuses(meta_dir, write=write, write_log=raw_log)
    stage_status["dataset_statuses"] = unit_datasets
    stage_status["task_statuses"] = unit_tasks
    issues = stage_issues + manifest_issues + unit_issues
    if stage_status.get("status") == "ok" and issues:
        raise RuntimeError("run status is ok but declared output metadata is invalid: " + "; ".join(issues))

    source_table = refresh_source_table_payload(
        _read_json(source_table_path),
        stage_status.get("metadata", {}).get("stage_specific", {}),
        reproducibility_evidence=stage_status.get("metadata", {}).get("reproducibility_evidence", {}),
    )
    summary = _normalize_optional_artifacts(_read_json(run_summary_path), write_log=raw_log)
    summary["preprocess_metadata"] = dict(summary.get("preprocess_metadata") or {})
    metadata = stage_status.get("metadata") if isinstance(stage_status.get("metadata"), dict) else {}
    summary["preprocess_metadata"]["stage_specific_hash"] = metadata.get("stage_specific_hash")
    summary["preprocess_metadata"]["reproducibility_evidence"] = metadata.get("reproducibility_evidence")
    summary["metadata_refresh"] = {
        "schema_version": "odcr_preprocess_a_metadata_refresh/1",
        "generated_at_utc": generated_at,
        "refreshed_current_headers": True,
        "refreshed_optional_artifacts": True,
        "refreshed_split_policy_summary": True,
        "refreshed_cleaning_summary": True,
        "csv_data_modified": False,
    }

    _write_json(stage_status_path, stage_status, write=write)
    _write_json(stage_manifest_path, stage_manifest, write=write)
    _write_json(source_table_path, source_table, write=write)
    _write_json(run_summary_path, summary, write=write)
    raw_log.extend(
        [
            f"refreshed {stage_status_path}",
            f"refreshed {stage_manifest_path}",
            f"refreshed {source_table_path}",
            f"refreshed {run_summary_path}",
            "csv_data_modified=False",
            f"issues={len(issues)}",
        ]
    )

    markdown = _summary_markdown(
        generated_at=generated_at,
        write=write,
        issues=issues,
        stage_status=stage_status,
        summary=summary,
    )
    paths = {
        "raw_log": ai_analysis_dir / "01_raw_logs" / "fix_preprocess_a_metadata_lineage.log",
        "hits": ai_analysis_dir / "02_search_hits" / "fix_preprocess_a_metadata_lineage_hits.txt",
        "ledger": ai_analysis_dir / "03_evidence_ledgers" / "fix_preprocess_a_metadata_lineage_ledger.md",
        "phase_summary": ai_analysis_dir / "04_phase_summaries" / "fix_preprocess_a_metadata_lineage_summary.md",
        "final_report": ai_analysis_dir / "05_final_reports" / "fix_preprocess_a_metadata_lineage_report.md",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths["raw_log"].write_text("\n".join(raw_log) + "\n", encoding="utf-8")
    paths["hits"].write_text(
        "\n".join(
            [
                "current_headers: code/odcr_core/preprocess_runtime.py, code/odcr_core/preprocess_metadata.py",
                "optional_artifacts: code/odcr_core/manifests.py",
                "split_policy_stats: code/split_data.py, code/odcr_core/preprocess_metadata.py",
                "Yelp cleaning logs: code/preprocess_data.py, runs/preprocess/a/1/meta/shell_logs/preprocess_a__Yelp__cpu.log",
                "reproducibility_evidence: code/odcr_core/preprocess_metadata.py",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    paths["ledger"].write_text(markdown, encoding="utf-8")
    paths["phase_summary"].write_text(markdown, encoding="utf-8")
    paths["final_report"].write_text(markdown, encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh preprocess_a run metadata without modifying CSV data.")
    parser.add_argument("--run-dir", default="runs/preprocess/a/1")
    parser.add_argument("--ai-analysis-dir", default="AI_analysis")
    parser.add_argument("--write", action="store_true", help="write refreshed metadata; otherwise only reports")
    args = parser.parse_args()
    return refresh_run(
        Path(args.run_dir).expanduser().resolve(),
        write=bool(args.write),
        ai_analysis_dir=Path(args.ai_analysis_dir).expanduser().resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
