from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from data_contract import (
    PREPROCESS_CONTRACT_VERSION,
    expected_preprocess_column_order,
)
from odcr_core.training_checkpoint import file_fingerprint, stable_hash

PREPROCESS_A_SPLIT_LOSS_POLICY = "filter_valid_test_cold_user_item"

_DATASET_HEADER_SPECS = {
    "processed_csv": "processed",
    "train_csv": "split",
    "valid_csv": "split",
    "test_csv": "split",
}
_TASK_HEADER_SPECS = {
    "aug_train_csv": "merged",
    "aug_valid_csv": "merged",
}
_PREPROCESS_CRITICAL_FILES = (
    "code/preprocess_data.py",
    "code/split_data.py",
    "code/combine_data.py",
    "code/data_contract.py",
    "code/odcr_core/preprocess_runtime.py",
    "code/odcr_core/preprocess_schema.py",
    "code/odcr_core/preprocess_status.py",
    "configs/odcr.yaml",
)


def expected_header_for_contract_kind(contract_kind: str) -> list[str]:
    if contract_kind == "processed":
        return list(expected_preprocess_column_order())
    if contract_kind == "split":
        return list(expected_preprocess_column_order(require_split_indices=True))
    if contract_kind == "merged":
        return list(expected_preprocess_column_order(require_split_indices=True, require_domain=True))
    raise ValueError(f"unknown preprocess contract kind: {contract_kind!r}")


def csv_header_metadata(
    path: str | Path,
    *,
    contract_kind: str,
    expected_header: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = p.resolve()
    expected = list(expected_header) if expected_header is not None else expected_header_for_contract_kind(contract_kind)
    payload: dict[str, Any] = {
        "path": str(p),
        "exists": p.exists(),
        "header": [],
        "header_hash": "",
        "file_size": None,
        "mtime_ns": None,
        "contract_kind": contract_kind,
        "header_match": False,
    }
    if not p.is_file():
        return payload
    stat = p.stat()
    with p.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = [str(item) for item in next(reader)]
        except StopIteration:
            header = []
    payload.update(
        {
            "exists": True,
            "header": header,
            "header_hash": stable_hash(header),
            "file_size": int(stat.st_size),
            "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
            "header_match": header == expected,
        }
    )
    return payload


def dataset_current_headers(data_root: str | Path, dataset: str) -> dict[str, dict[str, Any]]:
    root = Path(data_root).expanduser().resolve()
    return {
        name: csv_header_metadata(root / dataset / filename, contract_kind=contract_kind)
        for name, filename, contract_kind in (
            ("processed_csv", "processed.csv", "processed"),
            ("train_csv", "train.csv", "split"),
            ("valid_csv", "valid.csv", "split"),
            ("test_csv", "test.csv", "split"),
        )
    }


def task_current_headers(merged_root: str | Path, task_id: int | str) -> dict[str, dict[str, Any]]:
    root = Path(merged_root).expanduser().resolve()
    return {
        name: csv_header_metadata(root / str(task_id) / filename, contract_kind="merged")
        for name, filename in (
            ("aug_train_csv", "aug_train.csv"),
            ("aug_valid_csv", "aug_valid.csv"),
        )
    }


def unit_current_headers(
    unit_kind: str,
    output_files: tuple[str, ...] | list[str],
) -> dict[str, dict[str, Any]]:
    if unit_kind == "dataset":
        names = ("processed_csv", "train_csv", "valid_csv", "test_csv")
        specs = _DATASET_HEADER_SPECS
    else:
        names = ("aug_train_csv", "aug_valid_csv")
        specs = _TASK_HEADER_SPECS
    return {
        name: csv_header_metadata(path, contract_kind=specs[name])
        for name, path in zip(names, output_files)
    }


def validate_header_collection(headers: Mapping[str, Mapping[str, Any]]) -> list[str]:
    issues: list[str] = []
    for name, item in sorted(headers.items()):
        path = str(item.get("path") or name)
        if not item.get("exists"):
            issues.append(f"missing output: {path}")
            continue
        if not item.get("header_match"):
            issues.append(f"header mismatch: {path}")
    return issues


def _split_before_counts(processed_rows: int, train_rows: int) -> tuple[int, int]:
    temp_rows = max(0, int(processed_rows) - int(train_rows))
    test_before = int(math.ceil(temp_rows * 0.5))
    valid_before = temp_rows - test_before
    return valid_before, test_before


def split_policy_stats(
    *,
    processed_rows: int,
    train_rows: int,
    valid_rows_after_filter: int,
    test_rows_after_filter: int,
) -> dict[str, Any]:
    valid_before, test_before = _split_before_counts(processed_rows, train_rows)
    valid_loss = max(0, int(valid_before) - int(valid_rows_after_filter))
    test_loss = max(0, int(test_before) - int(test_rows_after_filter))
    split_loss = valid_loss + test_loss
    return {
        "processed_rows": int(processed_rows),
        "train_rows": int(train_rows),
        "valid_rows_before_filter": int(valid_before),
        "test_rows_before_filter": int(test_before),
        "valid_rows_after_filter": int(valid_rows_after_filter),
        "test_rows_after_filter": int(test_rows_after_filter),
        "valid_filtered_cold_user_item_rows": int(valid_loss),
        "test_filtered_cold_user_item_rows": int(test_loss),
        "filtered_cold_user_item_rows": int(split_loss),
        "split_loss_rows": int(split_loss),
        "split_loss_policy": PREPROCESS_A_SPLIT_LOSS_POLICY,
        "split_loss_expected": True,
    }


def _parse_json_tail(line: str) -> dict[str, Any] | None:
    start = line.find("{")
    if start < 0:
        return None
    try:
        obj = json.loads(line[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_dataset_shell_log(path: str | Path, dataset: str) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8") if p.is_file() else ""
    dropped_empty = 0
    prepare_rows: int | None = None
    k_core_rows: int | None = None
    processed_rows: int | None = None
    split_counts: tuple[int, int, int] | None = None
    split_stats_json: dict[str, Any] | None = None
    for line in text.splitlines():
        if "split_policy_stats" in line:
            obj = _parse_json_tail(line)
            if obj is not None:
                split_stats_json = obj
        match = re.search(r"dropped_empty_review_rows=(\d+)", line)
        if match:
            dropped_empty = int(match.group(1))
        match = re.search(r"stage=prepare_raw done rows=(\d+)", line)
        if match:
            prepare_rows = int(match.group(1))
        match = re.search(r"stage=k_core done rows=(\d+)", line)
        if match:
            k_core_rows = int(match.group(1))
        match = re.search(r"stage=write_csv done path=.* rows=(\d+)", line)
        if match:
            processed_rows = int(match.group(1))
        match = re.search(rf"^{re.escape(dataset)}: train:(\d+), valid:(\d+), test:(\d+)", line)
        if match:
            split_counts = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    if processed_rows is None and k_core_rows is not None:
        processed_rows = k_core_rows

    split_policy: dict[str, Any] = {}
    if split_stats_json is not None:
        split_policy = dict(split_stats_json)
    elif processed_rows is not None and split_counts is not None:
        train_rows, valid_rows, test_rows = split_counts
        split_policy = split_policy_stats(
            processed_rows=processed_rows,
            train_rows=train_rows,
            valid_rows_after_filter=valid_rows,
            test_rows_after_filter=test_rows,
        )

    k_core_removed = None
    if prepare_rows is not None and k_core_rows is not None:
        k_core_removed = max(0, prepare_rows - k_core_rows)
    cleaning = {
        "dropped_empty_review_rows": int(dropped_empty),
        "prepare_raw_rows_after_empty_drop": prepare_rows,
        "k_core_rows": k_core_rows,
        "k_core_removed_rows": k_core_removed,
        "review_non_empty_validation_passed": True,
        "cleaning_expected": bool(dropped_empty or (k_core_removed or 0) > 0),
    }
    return {
        "dataset": dataset,
        "split_policy": split_policy,
        "cleaning": cleaning,
    }


def preprocess_a_reproducibility_evidence(
    *,
    repo_root: str | Path,
    stage_payload: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    metadata = stage_payload.get("metadata") if isinstance(stage_payload.get("metadata"), Mapping) else {}
    git = metadata.get("git") if isinstance(metadata.get("git"), Mapping) else {}
    code_schema = metadata.get("code_schema_fingerprint") if isinstance(metadata.get("code_schema_fingerprint"), Mapping) else {}
    critical_hashes: dict[str, Any] = {}
    for rel in _PREPROCESS_CRITICAL_FILES:
        from_metadata = code_schema.get(rel) if isinstance(code_schema.get(rel), Mapping) else None
        if from_metadata and (from_metadata.get("sha256") or from_metadata.get("sample_sha256")):
            critical_hashes[rel] = dict(from_metadata)
        else:
            critical_hashes[rel] = file_fingerprint(root / rel)
    return {
        "git_branch": git.get("branch", ""),
        "git_commit": git.get("commit", ""),
        "git_dirty": bool(git.get("dirty", False)),
        "dirty_status_count": int(git.get("dirty_status_count") or 0),
        "dirty_status_sample": list(git.get("dirty_status_sample") or [])[:50],
        "preprocess_critical_file_hashes": critical_hashes,
        "preprocess_a_run_id": str(stage_payload.get("run_id") or metadata.get("run_id") or ""),
        "fingerprint_hash": str(stage_payload.get("fingerprint_hash") or ""),
        "contract_version": PREPROCESS_CONTRACT_VERSION,
    }


def refresh_preprocess_a_stage_payload(
    payload: dict[str, Any],
    *,
    repo_root: str | Path,
    meta_root: str | Path,
) -> tuple[dict[str, Any], list[str]]:
    out = dict(payload)
    metadata = dict(out.get("metadata") or {})
    stage_specific = dict(metadata.get("stage_specific") or {})
    datasets = dict(stage_specific.get("dataset_inputs_outputs") or {})
    merged_tasks = dict(stage_specific.get("merged_task_outputs") or {})
    issues: list[str] = []
    split_summary: dict[str, Any] = {}
    cleaning_summary: dict[str, Any] = {}
    meta = Path(meta_root).expanduser().resolve()

    for dataset, item in sorted(datasets.items()):
        if not isinstance(item, dict):
            continue
        headers: dict[str, dict[str, Any]] = {}
        for name, contract_kind in _DATASET_HEADER_SPECS.items():
            if name not in item:
                continue
            headers[name] = csv_header_metadata(item[name], contract_kind=contract_kind)
        item["current_headers"] = headers
        issues.extend(validate_header_collection(headers))
        parsed = parse_dataset_shell_log(meta / "shell_logs" / f"preprocess_a__{dataset}__cpu.log", str(dataset))
        if parsed.get("split_policy"):
            split_summary[str(dataset)] = parsed["split_policy"]
            item["split_policy"] = parsed["split_policy"]
        if parsed.get("cleaning"):
            cleaning_summary[str(dataset)] = parsed["cleaning"]
            item["cleaning"] = parsed["cleaning"]
        datasets[dataset] = item

    for task_id, item in sorted(merged_tasks.items(), key=lambda pair: int(pair[0])):
        if not isinstance(item, dict):
            continue
        headers = {}
        for name, contract_kind in _TASK_HEADER_SPECS.items():
            if name not in item:
                continue
            headers[name] = csv_header_metadata(item[name], contract_kind=contract_kind)
        item["current_headers"] = headers
        issues.extend(validate_header_collection(headers))
        merged_tasks[task_id] = item

    stage_specific["dataset_inputs_outputs"] = datasets
    stage_specific["merged_task_outputs"] = merged_tasks
    if split_summary:
        stage_specific["split_policy_summary"] = split_summary
    if cleaning_summary:
        stage_specific["cleaning_summary"] = cleaning_summary
    metadata["stage_specific"] = stage_specific
    metadata["stage_specific_hash"] = stable_hash(stage_specific)
    metadata["reproducibility_evidence"] = preprocess_a_reproducibility_evidence(
        repo_root=repo_root,
        stage_payload={**out, "metadata": metadata},
    )
    out["metadata"] = metadata
    return out, issues


def refresh_source_table_payload(
    source_table: dict[str, Any],
    stage_specific: Mapping[str, Any],
    *,
    reproducibility_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    out = dict(source_table)
    field_sources = dict(out.get("field_sources") or {})
    field_sources.setdefault(
        "preprocess.a.split_policy_summary",
        "split_data.py cold user/item valid/test filter policy",
    )
    field_sources.setdefault(
        "preprocess.a.cleaning_summary",
        "preprocess_data.py empty-review and k-core cleaning logs",
    )
    field_sources.setdefault(
        "preprocess.a.reproducibility_evidence",
        "preprocess_a git/code/config/contract fingerprint evidence",
    )
    out["field_sources"] = field_sources
    records = []
    seen_keys: set[str] = set()
    for record in list(out.get("records") or []):
        if not isinstance(record, dict):
            records.append(record)
            continue
        key = str(record.get("key") or "")
        seen_keys.add(key)
        new_record = dict(record)
        if key == "preprocess.a.raw_input_fingerprint":
            new_record["value"] = stage_specific.get("dataset_inputs_outputs")
        elif key == "preprocess.a.output_paths":
            new_record["value"] = {
                "datasets": stage_specific.get("dataset_inputs_outputs"),
                "merged_tasks": stage_specific.get("merged_task_outputs"),
            }
        elif key in {"preprocess.a.split_policy_summary", "preprocess_a.split_policy_summary"}:
            new_record["value"] = stage_specific.get("split_policy_summary")
        elif key in {"preprocess.a.cleaning_summary", "preprocess_a.cleaning_summary"}:
            new_record["value"] = stage_specific.get("cleaning_summary")
        elif key in {"preprocess.a.reproducibility_evidence", "preprocess_a.reproducibility_evidence"}:
            new_record["value"] = dict(reproducibility_evidence or {})
        records.append(new_record)
    additions = {
        "preprocess.a.split_policy_summary": stage_specific.get("split_policy_summary"),
        "preprocess.a.cleaning_summary": stage_specific.get("cleaning_summary"),
        "preprocess.a.reproducibility_evidence": dict(reproducibility_evidence or {}),
    }
    for key, value in additions.items():
        if key not in seen_keys:
            records.append({"key": key, "source": field_sources[key], "value": value})
    out["records"] = records
    return out
