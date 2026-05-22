from __future__ import annotations

import json
import os
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


METRICS_LOG_SCHEMA_VERSION = "odcr_step5_post_train_eval_metrics_log/1"
LAYOUT_LOG_SCHEMA_VERSION = "odcr_step5_post_train_eval_compact_layout_log/1"
SPLITS = ("valid", "test")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_split_json(split_path: Path, relative_path: str) -> dict[str, Any]:
    for path in _split_json_candidates(split_path, relative_path):
        if path.is_file():
            return _load_json(path)
    archive = split_path / "evidence.tar.gz"
    if not archive.is_file():
        return {}
    try:
        with tarfile.open(archive, "r:gz") as tar:
            member = tar.getmember(relative_path)
            extracted = tar.extractfile(member)
            if extracted is None:
                return {}
            payload = json.loads(extracted.read().decode("utf-8"))
    except (KeyError, OSError, tarfile.TarError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _split_json_candidates(split_path: Path, relative_path: str) -> list[Path]:
    rel = Path(relative_path)
    candidates = [split_path / rel]
    evidence = split_path / "evidence"
    if rel.parts and rel.parts[0] == "meta":
        candidates.append(evidence / "meta" / Path(*rel.parts[1:]))
    elif rel.name in {
        "paper_metrics.json",
        "eval_metrics.json",
        "official_eval_report.json",
        "eval_handoff.json",
        "eval_checkpoint_sidecar.json",
    }:
        candidates.append(evidence / "metrics" / rel.name)
    elif rel.name in {"predictions.jsonl", "predictions.csv"}:
        candidates.append(evidence / "predictions" / rel.name)
    elif rel.name == "samples.jsonl":
        candidates.append(evidence / "samples" / rel.name)
    else:
        candidates.append(evidence / "other" / rel)
    return candidates


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content.rstrip() + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, *, digits: int = 4) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_paper(value: Any) -> str:
    parsed = _as_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:.2f}"


def _split_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if root.name in SPLITS and root.parent.name == "post_train_eval":
        return root.parent
    return root


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "bytes": int(path.stat().st_size) if path.is_file() else 0,
    }


def _split_artifact(split_path: Path, relative_path: str) -> dict[str, Any]:
    for path in _split_json_candidates(split_path, relative_path):
        if path.is_file():
            return _artifact(path)
    return _artifact(split_path / relative_path)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _infer_task_run(root: Path) -> tuple[str, str]:
    parts = root.resolve().parts
    for i, part in enumerate(parts):
        if part == "runs" and i + 4 < len(parts) and parts[i + 1] == "step5":
            return parts[i + 2], parts[i + 3]
    return "task_unknown", "run_unknown"


def _paper_block(report: Mapping[str, Any]) -> Mapping[str, Any]:
    explanation = report.get("explanation_metrics")
    if isinstance(explanation, Mapping):
        paper = explanation.get("paper_metrics")
        if isinstance(paper, Mapping):
            return paper
    paper = report.get("explanation_paper_metrics")
    return paper if isinstance(paper, Mapping) else {}


def _repo_explanation_block(report: Mapping[str, Any]) -> Mapping[str, Any]:
    explanation = report.get("explanation_metrics")
    if isinstance(explanation, Mapping):
        repo_metrics = explanation.get("explanation")
        if isinstance(repo_metrics, Mapping):
            return repo_metrics
    return {}


def _collapse_block(report: Mapping[str, Any]) -> Mapping[str, Any]:
    explanation = report.get("explanation_metrics")
    if isinstance(explanation, Mapping):
        collapse = explanation.get("collapse_stats")
        if isinstance(collapse, Mapping):
            return collapse
    return {}


def _sample_count(report: Mapping[str, Any]) -> int | None:
    explanation = report.get("explanation_metrics")
    if isinstance(explanation, Mapping):
        value = explanation.get("sample_count")
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _rating_for_split(report: Mapping[str, Any], split: str) -> dict[str, float | None]:
    rating_metrics = report.get("rating_metrics")
    if not isinstance(rating_metrics, Mapping):
        return {"mae": None, "rmse": None}
    split_metrics = rating_metrics.get(split)
    if not isinstance(split_metrics, Mapping):
        return {"mae": None, "rmse": None}
    return {
        "mae": _as_float(split_metrics.get("mae") if "mae" in split_metrics else split_metrics.get("MAE")),
        "rmse": _as_float(split_metrics.get("rmse") if "rmse" in split_metrics else split_metrics.get("RMSE")),
    }


def _paper_metrics_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    paper = _paper_block(report)
    bleu = paper.get("bleu") if isinstance(paper.get("bleu"), Mapping) else {}
    rouge = paper.get("rouge") if isinstance(paper.get("rouge"), Mapping) else {}
    distinct = paper.get("distinct_corpus") if isinstance(paper.get("distinct_corpus"), Mapping) else {}
    distinct_pct = (
        distinct.get("scale_percent_0_100")
        if isinstance(distinct.get("scale_percent_0_100"), Mapping)
        else {}
    )
    return {
        "scale": "percent_0_100",
        "bleu_1": _as_float(bleu.get("1")),
        "bleu_2": _as_float(bleu.get("2")),
        "bleu_3": _as_float(bleu.get("3")),
        "bleu_4": _as_float(bleu.get("4")),
        "rouge_1_f": _as_float(rouge.get("rouge_1_f")),
        "rouge_2_f": _as_float(rouge.get("rouge_2_f")),
        "rouge_l_f": _as_float(rouge.get("rouge_l_f")),
        "distinct_1": _as_float(distinct_pct.get("1")),
        "distinct_2": _as_float(distinct_pct.get("2")),
    }


def _repo_metrics_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    repo_metrics = _repo_explanation_block(report)
    bleu = repo_metrics.get("bleu") if isinstance(repo_metrics.get("bleu"), Mapping) else {}
    rouge = repo_metrics.get("rouge") if isinstance(repo_metrics.get("rouge"), Mapping) else {}
    dist = repo_metrics.get("dist") if isinstance(repo_metrics.get("dist"), Mapping) else {}
    return {
        "bleu_1": _as_float(bleu.get("1")),
        "bleu_2": _as_float(bleu.get("2")),
        "bleu_3": _as_float(bleu.get("3")),
        "bleu_4": _as_float(bleu.get("4")),
        "rouge_1": _as_float(rouge.get("1")),
        "rouge_2": _as_float(rouge.get("2")),
        "rouge_l": _as_float(rouge.get("l")),
        "dist_1": _as_float(dist.get("1")),
        "dist_2": _as_float(dist.get("2")),
        "meteor": _as_float(repo_metrics.get("meteor")),
    }


def build_split_metrics_only(split_dir: str | Path) -> dict[str, Any]:
    split_path = Path(split_dir).expanduser().resolve()
    split = split_path.name
    report = _load_split_json(split_path, "paper_metrics.json")
    run_summary = _load_split_json(split_path, "meta/run_summary.json")
    eval_metrics = _load_split_json(split_path, "eval_metrics.json")
    collapse = _collapse_block(report)
    paper_summary = _paper_metrics_summary(report)
    repo_summary = _repo_metrics_summary(report)
    return {
        "split": split,
        "status": run_summary.get("status"),
        "started_at": run_summary.get("started_at"),
        "finished_at": run_summary.get("finished_at"),
        "duration_sec": run_summary.get("duration_sec"),
        "sample_count": _sample_count(report),
        "rating_source_metrics": _rating_for_split(report, split),
        "explanation_paper_metrics": paper_summary,
        "explanation_repo_metrics": repo_summary,
        "collapse": {
            "top1_pred_text": collapse.get("top1_pred_text"),
            "top1_pred_count": collapse.get("top1_pred_count"),
            "top1_pred_ratio": _as_float(collapse.get("top1_pred_ratio")),
            "pred_unique_count": collapse.get("pred_unique_count"),
            "pred_unique_ratio": _as_float(collapse.get("pred_unique_ratio")),
            "mean_pred_len_tokens": _as_float(collapse.get("mean_pred_len_tokens")),
            "mean_ref_len_tokens": _as_float(collapse.get("mean_ref_len_tokens")),
            "warnings": list(collapse.get("collapse_warnings") or []),
        },
        "decode": eval_metrics.get("decode") if isinstance(eval_metrics.get("decode"), Mapping) else {},
        "artifacts": {
            "paper_metrics": _split_artifact(split_path, "paper_metrics.json"),
            "eval_metrics": _split_artifact(split_path, "eval_metrics.json"),
            "official_eval_report": _split_artifact(split_path, "official_eval_report.json"),
            "eval_handoff": _split_artifact(split_path, "eval_handoff.json"),
            "samples": _split_artifact(split_path, "samples.jsonl"),
            "predictions_jsonl": _split_artifact(split_path, "predictions.jsonl"),
            "predictions_csv": _split_artifact(split_path, "predictions.csv"),
            "run_summary": _split_artifact(split_path, "meta/run_summary.json"),
            "dataset_contract": _split_artifact(split_path, "meta/official_eval_dataset_contract.json"),
            "cache_reuse_decision": _split_artifact(split_path, f"meta/cache_reuse_decision_{split}.json"),
        },
    }


def build_post_train_eval_metrics_only(path: str | Path) -> dict[str, Any]:
    root = _split_root(path)
    split_payloads: dict[str, Any] = {}
    table: list[dict[str, Any]] = []
    rating_source_metrics: dict[str, Any] = {}
    step5_run_id = ""
    checkpoint = ""
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        payload = build_split_metrics_only(split_dir)
        split_payloads[split] = payload
        report = _load_split_json(split_dir, "paper_metrics.json")
        if not rating_source_metrics and isinstance(report.get("rating_metrics"), Mapping):
            rating_source_metrics = dict(report.get("rating_metrics") or {})
        handoff = _load_split_json(split_dir, "eval_handoff.json")
        if not step5_run_id:
            step5_run_id = str(handoff.get("run_id") or "")
        if not checkpoint:
            checkpoint = str(handoff.get("checkpoint") or "")
        paper = payload["explanation_paper_metrics"]
        collapse = payload["collapse"]
        rating = payload["rating_source_metrics"]
        table.append(
            {
                "split": split,
                "status": payload.get("status"),
                "sample_count": payload.get("sample_count"),
                "rating_mae": rating.get("mae"),
                "rating_rmse": rating.get("rmse"),
                "paper_bleu_4": paper.get("bleu_4"),
                "paper_rouge_l_f": paper.get("rouge_l_f"),
                "paper_distinct_2": paper.get("distinct_2"),
                "repo_meteor": payload["explanation_repo_metrics"].get("meteor"),
                "top1_pred_ratio": collapse.get("top1_pred_ratio"),
                "pred_unique_ratio": collapse.get("pred_unique_ratio"),
                "warnings": collapse.get("warnings"),
            }
        )
    return {
        "schema_version": METRICS_LOG_SCHEMA_VERSION,
        "role": "human_metrics_log",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "post_train_eval_dir": str(root),
        "stage": "step5",
        "mode": "explanation_only",
        "step5_run_id": step5_run_id,
        "checkpoint": checkpoint,
        "rating_metrics_source": "step3_eval_handoff",
        "step5_rating_metrics_overwritten": False,
        "rating_source_metrics": rating_source_metrics,
        "metrics_table": table,
        "splits": split_payloads,
        "note": (
            "This is the human metrics entrypoint. Split metrics logs keep per-split quick views; "
            "large evidence is compacted into split evidence archives, and cache evidence belongs under cache/."
        ),
    }


def render_post_train_eval_metrics_log(payload: Mapping[str, Any]) -> str:
    splits = payload.get("splits") if isinstance(payload.get("splits"), Mapping) else {}
    lines = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ]
    for split in SPLITS:
        detail = splits.get(split)
        if not isinstance(detail, Mapping):
            continue
        paper = detail.get("explanation_paper_metrics") if isinstance(detail.get("explanation_paper_metrics"), Mapping) else {}
        repo = detail.get("explanation_repo_metrics") if isinstance(detail.get("explanation_repo_metrics"), Mapping) else {}
        rating = detail.get("rating_source_metrics") if isinstance(detail.get("rating_source_metrics"), Mapping) else {}
        lines.extend(
            [
                f"[{split}]",
                f"  samples = {_fmt_int(detail.get('sample_count'))} | status = {_fmt(detail.get('status'))}",
                "[Recommendation]",
                f"\tMAE = {_fmt(rating.get('mae'))} | RMSE = {_fmt(rating.get('rmse'))} ",
                "[Explanation]",
                f"\tROUGE: {_fmt_paper(paper.get('rouge_1_f'))}, {_fmt_paper(paper.get('rouge_2_f'))}, {_fmt_paper(paper.get('rouge_l_f'))} ",
                f"\tBLEU: {_fmt_paper(paper.get('bleu_1'))}, {_fmt_paper(paper.get('bleu_2'))}, {_fmt_paper(paper.get('bleu_3'))}, {_fmt_paper(paper.get('bleu_4'))} ",
                f"\tDIST-1/DIST-2 (evaluate_text, paper-compatible): {_fmt_paper(paper.get('distinct_1'))}, {_fmt_paper(paper.get('distinct_2'))}",
                f"\tMETEOR: {_fmt_paper(repo.get('meteor'))} ",
            ]
        )
        lines.append("")
    return "\n".join(lines)


def render_split_metrics_log(payload: Mapping[str, Any]) -> str:
    split = payload.get("split")
    paper = payload.get("explanation_paper_metrics") if isinstance(payload.get("explanation_paper_metrics"), Mapping) else {}
    repo = payload.get("explanation_repo_metrics") if isinstance(payload.get("explanation_repo_metrics"), Mapping) else {}
    rating = payload.get("rating_source_metrics") if isinstance(payload.get("rating_source_metrics"), Mapping) else {}
    lines = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        f"[{split}]",
        f"\tsamples = {_fmt_int(payload.get('sample_count'))} | status = {_fmt(payload.get('status'))}",
        "[Recommendation]",
        f"\tMAE = {_fmt(rating.get('mae'))} | RMSE = {_fmt(rating.get('rmse'))} ",
        "[Explanation]",
        f"\tROUGE: {_fmt_paper(paper.get('rouge_1_f'))}, {_fmt_paper(paper.get('rouge_2_f'))}, {_fmt_paper(paper.get('rouge_l_f'))} ",
        f"\tBLEU: {_fmt_paper(paper.get('bleu_1'))}, {_fmt_paper(paper.get('bleu_2'))}, {_fmt_paper(paper.get('bleu_3'))}, {_fmt_paper(paper.get('bleu_4'))} ",
        f"\tDIST-1/DIST-2 (evaluate_text, paper-compatible): {_fmt_paper(paper.get('distinct_1'))}, {_fmt_paper(paper.get('distinct_2'))}",
        f"\tMETEOR: {_fmt_paper(repo.get('meteor'))} ",
    ]
    return "\n".join(lines)


def write_post_train_eval_metrics_log(path: str | Path) -> str:
    root = _split_root(path)
    output = root / "metrics.log"
    payload = build_post_train_eval_metrics_only(root)
    _atomic_write_text(output, render_post_train_eval_metrics_log(payload))
    return str(output)


def write_split_metrics_log(split_dir: str | Path) -> str:
    split_path = Path(split_dir).expanduser().resolve()
    output = split_path / "metrics.log"
    payload = build_split_metrics_only(split_path)
    _atomic_write_text(output, render_split_metrics_log(payload))
    return str(output)


def _move_cache_artifacts(split_dir: Path, destination: Path) -> list[dict[str, Any]]:
    meta_dir = split_dir / "meta"
    moved: list[dict[str, Any]] = []
    if not meta_dir.is_dir():
        if destination.is_dir():
            for path in sorted(destination.glob("cache_*")):
                if path.is_file():
                    moved.append({"from": "existing_cache_dir", "to": str(path), "bytes": int(path.stat().st_size)})
        return moved
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(meta_dir.glob("cache_*")):
        if not path.is_file():
            continue
        target = destination / path.name
        if target.exists():
            target.unlink()
        shutil.move(str(path), str(target))
        moved.append({"from": str(path), "to": str(target), "bytes": int(target.stat().st_size)})
    if not moved and destination.is_dir():
        for path in sorted(destination.glob("cache_*")):
            if path.is_file():
                moved.append({"from": "existing_cache_dir", "to": str(path), "bytes": int(path.stat().st_size)})
    return moved


def _evidence_bucket_for_relative_path(relative_path: Path) -> Path:
    if relative_path.parts and relative_path.parts[0] == "meta":
        name = relative_path.name
        if name in {"console.log", "debug.log", "errors.log", "full.log"}:
            return Path("logs") / name
        return Path("meta") / Path(*relative_path.parts[1:])
    name = relative_path.name
    if name in {
        "paper_metrics.json",
        "eval_metrics.json",
        "official_eval_report.json",
        "eval_handoff.json",
        "eval_checkpoint_sidecar.json",
    }:
        return Path("metrics") / name
    if name in {"predictions.csv", "predictions.jsonl"}:
        return Path("predictions") / name
    if name == "samples.jsonl":
        return Path("samples") / name
    if name.endswith(".log"):
        return Path("logs") / name
    return Path("other") / relative_path


def _unpack_legacy_evidence_archive(split_dir: Path) -> list[str]:
    archive = split_dir / "evidence.tar.gz"
    extracted: list[str] = []
    if not archive.is_file():
        return extracted
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isdir():
                continue
            rel = Path(member.name)
            if rel.is_absolute() or ".." in rel.parts:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            target = split_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(handle.read())
            extracted.append(str(target))
    archive.unlink()
    return extracted


def _categorize_split_evidence(split_dir: Path) -> dict[str, Any]:
    extracted = _unpack_legacy_evidence_archive(split_dir)
    evidence_dir = split_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    moved: list[dict[str, Any]] = []
    sources = [
        path
        for path in sorted(split_dir.iterdir())
        if path.name not in {"metrics.log", "evidence", "evidence.tar.gz"}
    ]
    for source in sources:
        if source.is_file():
            files = [source]
        else:
            files = [path for path in sorted(source.rglob("*")) if path.is_file()]
        for file_path in files:
            relative_path = file_path.relative_to(split_dir)
            target = evidence_dir / _evidence_bucket_for_relative_path(relative_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target.unlink()
            shutil.move(str(file_path), str(target))
            moved.append({"from": str(file_path), "to": str(target), "bytes": int(target.stat().st_size)})
        if source.is_dir() and source.exists():
            shutil.rmtree(source, ignore_errors=True)
    return {
        "path": str(evidence_dir),
        "exists": evidence_dir.is_dir(),
        "file_count": sum(1 for path in evidence_dir.rglob("*") if path.is_file()),
        "extracted_from_legacy_archive": extracted,
        "moved": moved,
    }


def render_post_train_eval_layout_log(summary: Mapping[str, Any]) -> str:
    split_sections = []
    for split in SPLITS:
        split_summary = summary.get("splits", {}).get(split, {}) if isinstance(summary.get("splits"), Mapping) else {}
        cache_files = split_summary.get("cache_files") if isinstance(split_summary.get("cache_files"), list) else []
        split_sections.extend(
            [
                f"[{split}]",
                f"metrics_log: {_fmt(split_summary.get('metrics_log'))}",
                f"evidence_dir: {_fmt(split_summary.get('evidence_dir'))}",
                f"cache_dir: {_fmt(split_summary.get('cache_dir'))}",
                f"cache_files_moved: {len(cache_files)}",
            ]
        )
        for item in cache_files:
            if isinstance(item, Mapping):
                split_sections.append(f"  - {item.get('to')}")
        split_sections.append("")
    lines = [
        "ODCR Step5 Post-Train Eval Compact Layout",
        "=" * 45,
        f"schema_version: {LAYOUT_LOG_SCHEMA_VERSION}",
        f"generated_at_utc: {summary.get('generated_at_utc')}",
        f"post_train_eval_dir: {summary.get('post_train_eval_dir')}",
        "",
        "HUMAN ENTRYPOINTS",
        "-----------------",
        f"metrics: {summary.get('metrics_log')}",
        f"layout: {summary.get('layout_log')}",
        "",
        "STRUCTURE",
        "---------",
        "Root keeps only latest.json, metrics.log, and layout.log.",
        "Each split keeps metrics.log plus an expanded evidence/ directory.",
        "The evidence directory is categorized into metrics/, predictions/, samples/, meta/, logs/, and other/.",
        "Cache files were moved to cache/ so run output no longer mixes reusable cache evidence with eval evidence.",
        "",
        "SPLITS",
        "------",
        *split_sections,
        "REMOVED DUPLICATES",
        "------------------",
        *[f"- {path}" for path in summary.get("removed_duplicates", [])],
    ]
    return "\n".join(lines)


def compact_post_train_eval_layout(path: str | Path, cache_root: str | Path | None = None) -> dict[str, Any]:
    root = _split_root(path)
    task_label, run_id = _infer_task_run(root)
    cache_base = Path(cache_root).expanduser().resolve() if cache_root is not None else _repo_root() / "cache"
    has_metrics_evidence = any(
        (root / split / "paper_metrics.json").is_file()
        or (root / split / "evidence.tar.gz").is_file()
        or (root / split / "evidence").is_dir()
        for split in SPLITS
    )
    metrics_log = (
        write_post_train_eval_metrics_log(root)
        if has_metrics_evidence or not (root / "metrics.log").is_file()
        else str(root / "metrics.log")
    )
    summary: dict[str, Any] = {
        "schema_version": LAYOUT_LOG_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "post_train_eval_dir": str(root),
        "metrics_log": metrics_log,
        "layout_log": str(root / "layout.log"),
        "cache_base": str(cache_base),
        "splits": {},
        "removed_duplicates": [],
    }
    for stale in (root / "metrics_only.json", root / "metrics_only.log"):
        if stale.exists():
            stale.unlink()
            summary["removed_duplicates"].append(str(stale))
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        split_metrics = (
            write_split_metrics_log(split_dir)
            if (split_dir / "paper_metrics.json").is_file()
            or (split_dir / "evidence.tar.gz").is_file()
            or (split_dir / "evidence").is_dir()
            or not (split_dir / "metrics.log").is_file()
            else str(split_dir / "metrics.log")
        )
        cache_dir = cache_base / "step5" / task_label / run_id / "post_train_eval" / split
        moved_cache = _move_cache_artifacts(split_dir, cache_dir)
        evidence = _categorize_split_evidence(split_dir)
        summary["splits"][split] = {
            "metrics_log": split_metrics,
            "evidence_dir": evidence.get("path"),
            "evidence_file_count": evidence.get("file_count"),
            "legacy_archive_removed": bool(evidence.get("extracted_from_legacy_archive")),
            "cache_dir": str(cache_dir),
            "cache_files": moved_cache,
        }
    _atomic_write_text(root / "layout.log", render_post_train_eval_layout_log(summary))
    return summary
