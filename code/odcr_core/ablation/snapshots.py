"""Result snapshot skeletons for weak cross-platform ablations."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from odcr_core.ablation.guards import paper_table_gate
from odcr_core.ablation.registry import ABLATION_VARIANTS, entry_key, load_registry, registry_entry


def _repo_root(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve()


def _load_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _official_profile_ok(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    profile = payload.get("official_eval_profile") or payload.get("eval_profile_name")
    if profile == "paper_greedy_25":
        return True
    official = payload.get("official_policy")
    if isinstance(official, dict) and official.get("profile") == "paper_greedy_25":
        return True
    return False


def _task_local_rating_ok(payloads: list[dict[str, Any] | None], *, task: int) -> bool:
    for payload in payloads:
        if not payload:
            continue
        rating = payload.get("rating_source")
        if isinstance(rating, dict):
            if rating.get("task") == int(task) and "step3" in str(rating.get("type") or ""):
                return True
        if payload.get("rating_source_task") == int(task):
            return True
    return False


def _nested_get(payload: dict[str, Any] | None, path: tuple[str, ...]) -> Any:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _extract_split_metrics(payload: dict[str, Any] | None) -> dict[str, Any]:
    metrics = payload.get("metrics") if isinstance(payload, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}
    paper = metrics.get("paper_metrics") if isinstance(metrics.get("paper_metrics"), dict) else {}
    if not paper and isinstance(metrics.get("explanation"), dict):
        paper = metrics["explanation"]
    recommendation = metrics.get("recommendation") if isinstance(metrics.get("recommendation"), dict) else {}
    distinct = paper.get("distinct_corpus") if isinstance(paper.get("distinct_corpus"), dict) else {}
    distinct_pct = distinct.get("scale_percent_0_100") if isinstance(distinct.get("scale_percent_0_100"), dict) else {}
    bleu = paper.get("bleu") if isinstance(paper.get("bleu"), dict) else {}
    rouge = paper.get("rouge") if isinstance(paper.get("rouge"), dict) else {}
    collapse = metrics.get("collapse_stats") if isinstance(metrics.get("collapse_stats"), dict) else {}
    return {
        "mae": _as_float(recommendation.get("mae") or recommendation.get("MAE")),
        "rmse": _as_float(recommendation.get("rmse") or recommendation.get("RMSE")),
        "bleu1": _as_float(bleu.get("1")),
        "bleu2": _as_float(bleu.get("2")),
        "bleu3": _as_float(bleu.get("3")),
        "bleu4": _as_float(bleu.get("4")),
        "rouge1": _as_float(rouge.get("rouge_1_f")),
        "rouge2": _as_float(rouge.get("rouge_2_f")),
        "rougeL": _as_float(rouge.get("rouge_l_f")),
        "meteor": _as_float(paper.get("meteor")),
        "dist1": _as_float(distinct_pct.get("1")),
        "dist2": _as_float(distinct_pct.get("2")),
        "n_samples": _as_float(collapse.get("n_samples")),
        "step5_rating_metrics_written": bool(metrics.get("step5_rating_metrics_written") is True),
        "rating_metrics_source": metrics.get("rating_metrics_source"),
    }


def _rating_source_task(payload: dict[str, Any] | None) -> int | None:
    task = _nested_get(payload, ("rating_source", "task"))
    if task is None:
        task = payload.get("rating_source_task") if isinstance(payload, dict) else None
    try:
        return int(task) if task is not None else None
    except (TypeError, ValueError):
        return None


def build_result_snapshot(repo_root: str | Path, *, task: int, variant: str) -> dict[str, Any]:
    root = _repo_root(repo_root)
    entry = registry_entry(root, task, variant)
    source_run = str(entry.get("output_run") or "")
    run_dir = root / source_run
    metric_files = {
        split: f"{source_run}/post_train_eval_no_ref/{split}/eval_metrics.json"
        for split in ("valid", "test")
    }
    report_files = {
        split: f"{source_run}/post_train_eval_no_ref/{split}/official_eval_report.json"
        for split in ("valid", "test")
    }
    valid_metrics = _load_json_if_present(root / metric_files["valid"])
    test_metrics = _load_json_if_present(root / metric_files["test"])
    valid_report = _load_json_if_present(root / report_files["valid"])
    test_report = _load_json_if_present(root / report_files["test"])
    valid_complete = valid_metrics is not None and valid_report is not None
    test_complete = test_metrics is not None and test_report is not None
    metrics_by_split = {
        "valid": _extract_split_metrics(valid_metrics),
        "test": _extract_split_metrics(test_metrics),
    }
    snapshot: dict[str, Any] = {
        "schema_version": "odcr_ablation_result_snapshot/1",
        "task": int(task),
        "variant": str(variant),
        "source_run": source_run,
        "metric_file": metric_files,
        "official_eval_report": report_files,
        "rating_source_task": int(task),
        "paper_greedy_25": _official_profile_ok(valid_metrics) and _official_profile_ok(test_metrics),
        "task_local_rating_source": _task_local_rating_ok([valid_metrics, test_metrics, valid_report, test_report], task=task),
        "valid_complete": bool(valid_complete),
        "test_complete": bool(test_complete),
        "paper_table_allowed": False,
        "requires_manual_review": True,
        "candidate_paper_metrics_available": bool(valid_complete and test_complete),
        "manual_review_required": True,
        "metrics": metrics_by_split,
        "rating_source_task_observed": {
            "valid": _rating_source_task(valid_metrics),
            "test": _rating_source_task(test_metrics),
        },
        "step5_rating_metrics_written": {
            "valid": metrics_by_split["valid"].get("step5_rating_metrics_written"),
            "test": metrics_by_split["test"].get("step5_rating_metrics_written"),
        },
        "status": "complete_pending_manual_review" if valid_complete and test_complete else "missing_artifact",
        "missing_artifacts": [
            rel
            for rel in (*metric_files.values(), *report_files.values())
            if not (root / rel).is_file()
        ],
        "run_dir_exists": run_dir.is_dir(),
    }
    snapshot["paper_table_gate"] = paper_table_gate(snapshot)
    return snapshot


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _table_row(
    *,
    task: int,
    direction: str,
    variant: str,
    status: str,
    metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = metrics or {}
    return {
        "task": int(task),
        "direction": direction,
        "variant": variant,
        "status": status,
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "bleu1": metrics.get("bleu1"),
        "bleu2": metrics.get("bleu2"),
        "bleu3": metrics.get("bleu3"),
        "bleu4": metrics.get("bleu4"),
        "rouge1": metrics.get("rouge1"),
        "rougeL": metrics.get("rougeL"),
        "meteor": metrics.get("meteor"),
        "dist1": metrics.get("dist1"),
        "dist2": metrics.get("dist2"),
        "manual_review_required": True,
    }


def _full_baseline_row(root: Path, *, task: int) -> dict[str, Any]:
    key = entry_key(int(task), "full_odcr")
    entry = load_registry(root).get(key, {})
    source_run = str(entry.get("source_run") or f"runs/step5/task{int(task)}/1_19")
    metrics = _extract_split_metrics(
        _load_json_if_present(root / source_run / "post_train_eval_no_ref" / "test" / "eval_metrics.json")
    )
    return _table_row(
        task=int(task),
        direction=str(entry.get("direction") or ""),
        variant="full_odcr",
        status=str(entry.get("status") or "existing_official_single_run"),
        metrics=metrics,
    )


def _write_candidate_paper_tables(root: Path, snapshots: list[dict[str, Any]]) -> dict[str, str]:
    table_dir = root / "ablations" / "paper_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for task in (8, 7):
        rows.append(_full_baseline_row(root, task=int(task)))
        for variant in ABLATION_VARIANTS:
            snap = next(item for item in snapshots if int(item["task"]) == int(task) and item["variant"] == variant)
            entry = registry_entry(root, int(task), variant)
            rows.append(
                _table_row(
                    task=int(task),
                    direction=str(entry.get("direction") or ""),
                    variant=variant,
                    status=str(snap.get("status") or ""),
                    metrics=(snap.get("metrics") or {}).get("test") if isinstance(snap.get("metrics"), dict) else {},
                )
            )
    csv_path = table_dir / "table_weak_cross_platform_ablation.csv"
    fieldnames = list(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(value) for key, value in row.items()})
    tex_path = table_dir / "table_weak_cross_platform_ablation.tex"
    caption = (
        "Single-run weak cross-platform ablations. These are not 5-seed mean\\(\\pm\\)std and not a full "
        "D4C paper-compatible main table. Results remain candidate paper values until manual review clears "
        "paper\\_table\\_allowed and requires\\_manual\\_review."
    )
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:weak-cross-platform-ablation}",
        "\\begin{tabular}{llllcccc}",
        "\\toprule",
        "Task & Direction & Variant & Status & MAE & RMSE & BLEU-4 & ROUGE-L \\\\",
        "\\midrule",
    ]
    for row in rows:
        direction = str(row["direction"]).replace("->", "$\\rightarrow$")
        lines.append(
            " & ".join(
                [
                    str(row["task"]),
                    direction,
                    str(row["variant"]).replace("_", "\\_"),
                    str(row["status"]).replace("_", "\\_"),
                    _fmt(row["mae"]),
                    _fmt(row["rmse"]),
                    _fmt(row["bleu4"]),
                    _fmt(row["rougeL"]),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    notes_path = table_dir / "table_weak_cross_platform_ablation_notes.md"
    notes_path.write_text(
        "# Weak Cross-Platform Ablation Candidate Notes\n\n"
        "Single-run weak cross-platform ablations. These are not 5-seed mean+/-std and not a full "
        "D4C paper-compatible main table. Results remain candidate paper values until manual review "
        "clears paper_table_allowed and requires_manual_review.\n\n"
        "The table uses only no-reference Step5 outputs under post_train_eval_no_ref. "
        "Old oracle-content post_train_eval outputs, probe artifacts, and dry-run artifacts are excluded.\n",
        encoding="utf-8",
    )
    return {
        "paper_table_csv": csv_path.relative_to(root).as_posix(),
        "paper_table_tex": tex_path.relative_to(root).as_posix(),
        "paper_table_notes": notes_path.relative_to(root).as_posix(),
    }


def write_result_snapshot_outputs(repo_root: str | Path) -> dict[str, Any]:
    root = _repo_root(repo_root)
    out_dir = root / "ablations" / "result_snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    snapshot_paths: list[str] = []
    for task in (8, 7):
        for variant in ABLATION_VARIANTS:
            snapshot = build_result_snapshot(root, task=task, variant=variant)
            path = out_dir / f"task{task}_{variant}_metrics.json"
            path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            snapshots.append(snapshot)
            snapshot_paths.append(path.relative_to(root).as_posix())
            valid = snapshot["metrics"]["valid"]
            test = snapshot["metrics"]["test"]
            rows.append(
                {
                    "task": int(task),
                    "variant": variant,
                    "status": snapshot["status"],
                    "source_run": snapshot["source_run"],
                    "valid_complete": snapshot["valid_complete"],
                    "test_complete": snapshot["test_complete"],
                    "paper_greedy_25": snapshot["paper_greedy_25"],
                    "task_local_rating_source": snapshot["task_local_rating_source"],
                    "paper_table_allowed": snapshot["paper_table_allowed"],
                    "paper_table_eligible": snapshot["paper_table_gate"]["eligible"],
                    "candidate_paper_metrics_available": snapshot["candidate_paper_metrics_available"],
                    "manual_review_required": snapshot["manual_review_required"],
                    "valid_mae": valid.get("mae"),
                    "valid_rmse": valid.get("rmse"),
                    "valid_bleu4": valid.get("bleu4"),
                    "valid_rougeL": valid.get("rougeL"),
                    "test_mae": test.get("mae"),
                    "test_rmse": test.get("rmse"),
                    "test_bleu4": test.get("bleu4"),
                    "test_rougeL": test.get("rougeL"),
                }
            )
    summary_path = out_dir / "ablation_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table_paths = _write_candidate_paper_tables(root, snapshots)
    return {
        "schema_version": "odcr_ablation_snapshot_write/1",
        "status": "pass",
        "snapshots": snapshot_paths,
        "summary_csv": summary_path.relative_to(root).as_posix(),
        **table_paths,
    }
