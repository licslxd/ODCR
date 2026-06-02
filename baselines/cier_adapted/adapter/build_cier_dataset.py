from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    BASELINE,
    FORBIDDEN_CIER_INPUT_FIELDS,
    REQUIRED_ODCR_FIELDS,
    die_if_missing_required_fields,
    ensure_run_layout,
    file_row_count,
    load_config,
    merged_context_path,
    normalize_mode,
    read_json,
    repo_path,
    run_dir,
    sha256_file,
    split_csv_path,
    task_domains,
    utc_now,
    write_json,
    write_jsonl,
    write_resolved_config,
)


def _head_columns(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def _read_odcr_split(path: Path, *, smoke_rows: int | None) -> pd.DataFrame:
    columns = _head_columns(path)
    die_if_missing_required_fields(columns, path)
    nrows = smoke_rows if smoke_rows and smoke_rows > 0 else None
    df = pd.read_csv(path, usecols=REQUIRED_ODCR_FIELDS, nrows=nrows)
    for col in ("user", "item", "review", "explanation"):
        df[col] = df[col].fillna("").astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="raise").astype(float)
    bad = df[(df["rating"] < 1.0) | (df["rating"] > 5.0)]
    if not bad.empty:
        raise ValueError(f"{path} has ratings outside [1, 5]: first bad index {bad.index[0]}")
    return df


def _iter_odcr_split(path: Path, *, smoke_rows: int | None, chunksize: int = 200_000):
    columns = _head_columns(path)
    die_if_missing_required_fields(columns, path)
    remaining = smoke_rows if smoke_rows and smoke_rows > 0 else None
    for chunk in pd.read_csv(path, usecols=REQUIRED_ODCR_FIELDS, chunksize=chunksize):
        if remaining is not None:
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
            remaining -= len(chunk)
        for col in ("user", "item", "review", "explanation"):
            chunk[col] = chunk[col].fillna("").astype(str)
        chunk["rating"] = pd.to_numeric(chunk["rating"], errors="raise").astype(float)
        bad = chunk[(chunk["rating"] < 1.0) | (chunk["rating"] > 5.0)]
        if not bad.empty:
            raise ValueError(f"{path} has ratings outside [1, 5]: first bad index {bad.index[0]}")
        yield chunk


def _keyword_from_allowed_text(explanation: str, review: str, *, max_words: int = 5) -> str:
    source = explanation.strip() or review.strip()
    words = re.findall(r"\S+", source)
    if not words:
        return "unknown"
    return " ".join(words[:max_words])


def _as_records(
    df: pd.DataFrame,
    *,
    task_id: int,
    domain: str,
    split: str,
    domain_role: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, row in df.reset_index(drop=True).iterrows():
        explanation = str(row["explanation"])
        review = str(row["review"])
        rows.append(
            {
                "schema_version": "odcr_cier_dataset_record_v1",
                "baseline": BASELINE,
                "task_id": int(task_id),
                "domain": domain,
                "domain_role": domain_role,
                "split": split,
                "row_index": int(idx),
                "record_id": f"{domain}:{split}:{idx}",
                "user_id": str(row["user"]),
                "item_id": str(row["item"]),
                "rating": float(row["rating"]),
                "rating_index": int(round(float(row["rating"]))) - 1,
                "review": review,
                "explanation": explanation,
                "keyword": _keyword_from_allowed_text(explanation, review),
                "keyword_words": _keyword_from_allowed_text(explanation, review),
            }
        )
    return rows


def _assign_cier_ids(rows: list[dict[str, Any]]) -> None:
    users = {str(row["user_id"]) for row in rows}
    items = {str(row["item_id"]) for row in rows}
    user_map = {value: idx for idx, value in enumerate(sorted(users))}
    item_map = {value: idx for idx, value in enumerate(sorted(items))}
    for row in rows:
        row["cier_user"] = int(user_map[str(row["user_id"])])
        row["cier_item"] = int(item_map[str(row["item_id"])])


def _collect_id_maps(inputs: list[tuple[Path, int | None]]) -> tuple[dict[str, int], dict[str, int]]:
    users: set[str] = set()
    items: set[str] = set()
    for path, smoke_rows in inputs:
        for chunk in _iter_odcr_split(path, smoke_rows=smoke_rows):
            users.update(chunk["user"].astype(str).tolist())
            items.update(chunk["item"].astype(str).tolist())
    return {value: idx for idx, value in enumerate(sorted(users))}, {value: idx for idx, value in enumerate(sorted(items))}


def _write_records(run_path: Path, name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    jsonl = run_path / "data" / "cier_records" / f"{name}.jsonl"
    count = write_jsonl(jsonl, rows)
    csv_path = run_path / "data" / "cier_records" / f"{name}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return {
        "name": name,
        "jsonl": str(jsonl),
        "csv": str(csv_path),
        "row_count": count,
        "sha256": sha256_file(jsonl),
    }


def _write_records_stream(
    run_path: Path,
    name: str,
    path: Path,
    *,
    task_id: int,
    domain: str,
    split: str,
    domain_role: str,
    user_map: dict[str, int],
    item_map: dict[str, int],
    smoke_rows: int | None,
) -> dict[str, Any]:
    jsonl = run_path / "data" / "cier_records" / f"{name}.jsonl"
    csv_path = run_path / "data" / "cier_records" / f"{name}.csv"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "schema_version",
        "baseline",
        "task_id",
        "domain",
        "domain_role",
        "split",
        "row_index",
        "record_id",
        "user_id",
        "item_id",
        "rating",
        "rating_index",
        "review",
        "explanation",
        "keyword",
        "keyword_words",
        "cier_user",
        "cier_item",
    ]
    count = 0
    with jsonl.open("w", encoding="utf-8") as jh, csv_path.open("w", encoding="utf-8", newline="") as ch:
        writer = csv.DictWriter(ch, fieldnames=fieldnames)
        writer.writeheader()
        for chunk in _iter_odcr_split(path, smoke_rows=smoke_rows):
            for row in _as_records(chunk, task_id=task_id, domain=domain, split=split, domain_role=domain_role):
                user_id = str(row["user_id"])
                item_id = str(row["item_id"])
                row["row_index"] = count
                row["record_id"] = f"{domain}:{split}:{count}"
                row["cier_user"] = int(user_map[user_id])
                row["cier_item"] = int(item_map[item_id])
                jh.write(json.dumps(row, sort_keys=True) + "\n")
                writer.writerow({key: row.get(key, "") for key in fieldnames})
                count += 1
    return {
        "name": name,
        "jsonl": str(jsonl),
        "csv": str(csv_path),
        "row_count": count,
        "sha256": sha256_file(jsonl),
    }


def _source_input_entry(
    domain: str,
    split: str,
    path: Path,
    *,
    consumed: bool,
    include_fingerprint: bool = False,
) -> dict[str, Any]:
    return {
        "domain": domain,
        "split": split,
        "path": str(path),
        "exists": path.is_file(),
        "consumed_by_cier": consumed,
        "row_count": file_row_count(path) if include_fingerprint and path.is_file() else None,
        "sha256": sha256_file(path) if include_fingerprint and path.is_file() else None,
        "fingerprint_policy": "sha256_when_full_build" if not include_fingerprint else "sha256",
        "required_fields": REQUIRED_ODCR_FIELDS,
        "excluded_fields": FORBIDDEN_CIER_INPUT_FIELDS,
    }


def _merged_context_entry(task_id: int, split: str, *, include_fingerprint: bool = False) -> dict[str, Any]:
    path = merged_context_path(task_id, split)
    return {
        "task_id": int(task_id),
        "split": split,
        "path": str(path),
        "exists": path.is_file(),
        "consumed_by_cier": False,
        "role": "context_only_not_cier_input",
        "row_count": file_row_count(path) if include_fingerprint and path.is_file() else None,
        "sha256": sha256_file(path) if include_fingerprint and path.is_file() else None,
        "fingerprint_policy": "sha256_when_full_build" if not include_fingerprint else "sha256",
    }


def _dry_run(task_id: int, mode: str, config: dict[str, Any]) -> dict[str, Any]:
    source_domain, target_domain = task_domains(config)
    consumed = []
    for domain, role in ((source_domain, "source"), (target_domain, "target")):
        for split in ("train", "valid", "test"):
            is_consumed = role == "target" or (mode == "source_to_target" and split == "train")
            path = split_csv_path(domain, split)
            if not path.is_file():
                raise FileNotFoundError(path)
            columns = _head_columns(path)
            die_if_missing_required_fields(columns, path)
            consumed.append(_source_input_entry(domain, split, path, consumed=is_consumed, include_fingerprint=False))
    return {
        "schema_version": "odcr_cier_dataset_build_dry_run_v1",
        "baseline": BASELINE,
        "task_id": int(task_id),
        "mode": mode,
        "source_domain": source_domain,
        "target_domain": target_domain,
        "planned_inputs": consumed,
        "merged_context": [
            _merged_context_entry(task_id, "train", include_fingerprint=False),
            _merged_context_entry(task_id, "valid", include_fingerprint=False),
        ],
        "will_use_step3_step4_routing": False,
    }


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.task)
    mode = normalize_mode(args.mode, config)
    run_id = args.run_id or ("smoke" if args.smoke else "dataset_build")
    source_domain, target_domain = task_domains(config)
    if args.dry_run:
        payload = _dry_run(args.task, mode, config)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload

    run_path = run_dir(args.task, run_id)
    ensure_run_layout(run_path)
    write_resolved_config(run_path / "meta" / "resolved_config.json", config, mode=mode, run_id=run_id)

    smoke_rows = int(args.smoke_rows) if args.smoke else None
    source_entries: list[dict[str, Any]] = []
    for split in ("train", "valid", "test"):
        source_path = split_csv_path(source_domain, split)
        target_path = split_csv_path(target_domain, split)
        source_entries.append(
            _source_input_entry(
                source_domain,
                split,
                source_path,
                consumed=(mode == "source_to_target" and split == "train"),
                include_fingerprint=not args.smoke,
            )
        )
        source_entries.append(_source_input_entry(target_domain, split, target_path, consumed=True, include_fingerprint=not args.smoke))

    active_inputs: list[tuple[Path, int | None]] = [
        (split_csv_path(target_domain, "train"), smoke_rows),
        (split_csv_path(target_domain, "valid"), smoke_rows),
        (split_csv_path(target_domain, "test"), smoke_rows),
    ]
    if mode == "source_to_target":
        active_inputs.insert(0, (split_csv_path(source_domain, "train"), smoke_rows))
    user_map, item_map = _collect_id_maps(active_inputs)

    artifacts: list[dict[str, Any]] = []
    if mode == "source_to_target":
        artifacts.append(
            _write_records_stream(
                run_path,
                "source_train",
                split_csv_path(source_domain, "train"),
                task_id=args.task,
                domain=source_domain,
                split="train",
                domain_role="source",
                user_map=user_map,
                item_map=item_map,
                smoke_rows=smoke_rows,
            )
        )
    else:
        artifacts.append(_write_records(run_path, "source_train", []))
    for split in ("train", "valid", "test"):
        artifacts.append(
            _write_records_stream(
                run_path,
                f"target_{split}",
                split_csv_path(target_domain, split),
                task_id=args.task,
                domain=target_domain,
                split=split,
                domain_role="target",
                user_map=user_map,
                item_map=item_map,
                smoke_rows=smoke_rows,
            )
        )

    source_table = {
        "schema_version": "odcr_cier_source_table_v1",
        "baseline": BASELINE,
        "task_id": int(args.task),
        "run_id": run_id,
        "mode": mode,
        "created_at": utc_now(),
        "source_domain": source_domain,
        "target_domain": target_domain,
        "consumed_inputs": source_entries,
        "merged_context_inputs": [
            _merged_context_entry(args.task, "train", include_fingerprint=not args.smoke),
            _merged_context_entry(args.task, "valid", include_fingerprint=not args.smoke),
        ],
        "field_mapping": {
            "user": "user_id",
            "item": "item_id",
            "rating": "rating",
            "review": "review",
            "explanation": "explanation",
            "keyword": "derived_from_explanation_or_review_first_words",
        },
        "not_used_as_cier_input": FORBIDDEN_CIER_INPUT_FIELDS,
        "artifacts": artifacts,
        "active_id_space": {
            "user_count": len(user_map),
            "item_count": len(item_map),
            "mapping_policy": "sorted_string_ids_over_consumed_source_train_and_target_train_valid_test",
        },
        "combined_active_artifact": "not_written_for_full_build_to_avoid_duplicate_large_payload",
        "smoke": bool(args.smoke),
        "smoke_rows_per_split": smoke_rows,
        "odcr_active_path_modified": False,
        "uses_step3_step4_evidence_routing": False,
    }
    write_json(run_path / "meta" / "source_table.json", source_table)
    stage_status = {
        "schema_version": "odcr_cier_stage_status_v1",
        "status": "dataset_built_smoke" if args.smoke else "dataset_built",
        "baseline": BASELINE,
        "task_id": int(args.task),
        "run_id": run_id,
        "mode": mode,
        "ready_for_training": True,
        "full_training_started": False,
        "updated_at": utc_now(),
    }
    write_json(run_path / "meta" / "stage_status.json", stage_status)
    run_summary = {
        "schema_version": "odcr_cier_run_summary_v1",
        "baseline": BASELINE,
        "task_id": int(args.task),
        "run_id": run_id,
        "mode": mode,
        "status": stage_status["status"],
        "artifacts": {
            "resolved_config": "meta/resolved_config.json",
            "source_table": "meta/source_table.json",
            "stage_status": "meta/stage_status.json",
            "data": [item["jsonl"] for item in artifacts],
        },
        "created_at": utc_now(),
    }
    write_json(run_path / "meta" / "run_summary.json", run_summary)
    print(json.dumps({"run_dir": str(run_path), "source_table": str(run_path / "meta/source_table.json")}, indent=2))
    return read_json(run_path / "meta" / "source_table.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build ODCR split data for the external CIER-adapted baseline.")
    parser.add_argument("--task", type=int, required=True, choices=sorted([2, 5, 7, 8]))
    parser.add_argument("--mode", choices=["source_to_target", "target_only", "source-to-target"], default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-rows", type=int, default=8)
    parser.add_argument("--repo-root", default=None, help="Accepted for runner symmetry; must be the current repo if set.")
    args = parser.parse_args(argv)
    if args.repo_root and repo_path(args.repo_root).resolve() != repo_path(".").resolve():
        raise ValueError("--repo-root must resolve to the ODCR repository root")
    build_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
