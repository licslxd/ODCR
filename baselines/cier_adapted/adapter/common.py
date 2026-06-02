from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


BASELINE = "cier_adapted"
PREDICTION_SCHEMA_VERSION = "odcr_baseline_prediction_v1"
CONFIG_SCHEMA_VERSION = "odcr_cier_adapted_config_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_ROOT = REPO_ROOT / "baselines" / "cier_adapted"
TASK_CONFIGS = {
    2: BASELINE_ROOT / "configs" / "task2.yaml",
    5: BASELINE_ROOT / "configs" / "task5.yaml",
    7: BASELINE_ROOT / "configs" / "task7.yaml",
    8: BASELINE_ROOT / "configs" / "task8.yaml",
}
RUN_ROOT = REPO_ROOT / "runs" / "baselines" / BASELINE
REQUIRED_ODCR_FIELDS = ["user", "item", "rating", "review", "explanation"]
FORBIDDEN_CIER_INPUT_FIELDS = [
    "confidence_bucket",
    "sample_weight_hint",
    "route_explainer",
    "route_scorer",
    "content_retention_score",
    "style_shift_score",
    "rating_stability_score",
    "cf_reliability_score",
    "uncertainty_score",
    "evidence_quality_prior",
    "preprocess_route_scorer_prior",
    "preprocess_route_explainer_prior",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_config(task_id: int) -> dict[str, Any]:
    path = TASK_CONFIGS.get(int(task_id))
    if path is None:
        raise ValueError(f"Unsupported CIER-adapted task: {task_id}")
    if not path.is_file():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if data.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"{path} schema_version must be {CONFIG_SCHEMA_VERSION}, got {data.get('schema_version')!r}"
        )
    if int(data.get("task_id")) != int(task_id):
        raise ValueError(f"{path} task_id mismatch: {data.get('task_id')} != {task_id}")
    return data


def default_run_id(dry_run: bool = False) -> str:
    if dry_run:
        return "dry_run"
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_dir(task_id: int, run_id: str) -> Path:
    return RUN_ROOT / f"task{int(task_id)}" / str(run_id)


def ensure_run_layout(path: Path) -> None:
    for rel in ("meta", "data", "model", "predictions", "eval/valid", "eval/test"):
        (path / rel).mkdir(parents=True, exist_ok=True)


def repo_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_row_count(path: Path) -> int:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            try:
                next(reader)
            except StopIteration:
                return 0
            return sum(1 for _ in reader)
    return -1


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def split_csv_path(domain: str, split: str) -> Path:
    return REPO_ROOT / "data" / domain / f"{split}.csv"


def merged_context_path(task_id: int, split: str) -> Path:
    return REPO_ROOT / "merged" / str(int(task_id)) / f"aug_{split}.csv"


def task_domains(config: Mapping[str, Any]) -> tuple[str, str]:
    return str(config["source_domain"]), str(config["target_domain"])


def die_if_missing_required_fields(columns: Iterable[str], path: Path) -> None:
    have = set(columns)
    missing = [field for field in REQUIRED_ODCR_FIELDS if field not in have]
    if missing:
        raise ValueError(f"{path} missing required ODCR fields for CIER baseline: {missing}")


def normalize_mode(mode: str | None, config: Mapping[str, Any]) -> str:
    value = str(mode or config.get("mode_default") or "source_to_target").strip()
    if value == "source-to-target":
        value = "source_to_target"
    if value not in {"source_to_target", "target_only"}:
        raise ValueError(f"Unsupported mode {value!r}; expected source_to_target or target_only")
    return value


def decode_profile(config: Mapping[str, Any]) -> dict[str, Any]:
    decode = dict(config.get("decode") or {})
    return {
        "profile": decode.get("profile", "paper_greedy_25"),
        "max_length": int(decode.get("max_length", 25)),
        "do_sample": bool(decode.get("do_sample", False)),
        "temperature": decode.get("temperature"),
        "top_p": decode.get("top_p"),
        "repetition_penalty": float(decode.get("repetition_penalty", 1.0)),
    }


def write_resolved_config(path: Path, config: Mapping[str, Any], *, mode: str, run_id: str) -> None:
    source_domain, target_domain = task_domains(config)
    payload = {
        "schema_version": "odcr_cier_adapted_resolved_config_v1",
        "baseline": BASELINE,
        "task_id": int(config["task_id"]),
        "run_id": run_id,
        "mode": mode,
        "source_domain": source_domain,
        "target_domain": target_domain,
        "created_at": utc_now(),
        "config": dict(config),
        "decode": decode_profile(config),
        "one_control_impact": "none_external_baseline_only",
    }
    write_json(path, payload)

