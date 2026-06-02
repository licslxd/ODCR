from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    BASELINE,
    PREDICTION_SCHEMA_VERSION,
    decode_profile,
    ensure_run_layout,
    load_config,
    normalize_mode,
    read_json,
    read_jsonl,
    run_dir,
    utc_now,
    write_json,
)

CODE_DIR = Path(__file__).resolve().parents[3] / "code"
sys.path.insert(0, str(CODE_DIR))
from base_utils import build_paper_metric_inputs, official_paper_metrics  # noqa: E402
from odcr_eval_metrics import code1_compatible_rating_metrics  # noqa: E402


class WhitespaceTokenizer:
    def __init__(self) -> None:
        self._last_words: list[str] = []

    def __call__(
        self,
        text: Any,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        verbose: bool = True,
    ) -> dict[str, list[int]]:
        del add_special_tokens, verbose
        self._last_words = str(text).split()
        ids = list(range(1, len(self._last_words) + 1))
        if truncation and max_length is not None:
            ids = ids[: int(max_length)]
        return {"input_ids": ids}

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        words = []
        for raw in ids:
            idx = int(raw) - 1
            if 0 <= idx < len(self._last_words):
                words.append(self._last_words[idx])
        return " ".join(words)


def _prediction_path(run_path: Path, split: str) -> Path:
    return run_path / "predictions" / f"{split}_predictions.jsonl"


def _validate_rows(rows: list[dict[str, Any]], *, task_id: int, split: str) -> None:
    required = {
        "schema_version",
        "baseline",
        "task_id",
        "split",
        "user_id",
        "item_id",
        "source_domain",
        "target_domain",
        "rating_gold",
        "rating_pred",
        "reference",
        "prediction",
    }
    for idx, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"prediction row {idx} missing {sorted(missing)}")
        if row["schema_version"] != PREDICTION_SCHEMA_VERSION:
            raise ValueError(f"prediction row {idx} schema mismatch: {row['schema_version']}")
        if row["baseline"] != BASELINE:
            raise ValueError(f"prediction row {idx} baseline mismatch: {row['baseline']}")
        if int(row["task_id"]) != int(task_id):
            raise ValueError(f"prediction row {idx} task mismatch: {row['task_id']} != {task_id}")
        if row["split"] != split:
            raise ValueError(f"prediction row {idx} split mismatch: {row['split']} != {split}")


def _rating_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return code1_compatible_rating_metrics(
        [float(row["rating_pred"]) for row in rows],
        [float(row["rating_gold"]) for row in rows],
    )


def _official_text_metrics(rows: list[dict[str, Any]], *, max_len: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tokenizer = WhitespaceTokenizer()
    metric_inputs = [
        build_paper_metric_inputs(row["prediction"], row["reference"], tokenizer, max_len=max_len)
        for row in rows
    ]
    preds = [item["metric_pred"] for item in metric_inputs]
    refs = [item["metric_ref"] for item in metric_inputs]
    return official_paper_metrics(preds, refs), metric_inputs


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _update_summary(run_path: Path, *, split: str, paper_path: Path, eval_path: Path) -> None:
    summary_path = run_path / "meta" / "run_summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else {"schema_version": "odcr_cier_run_summary_v1"}
    evals = dict(summary.get("eval") or {})
    evals[split] = {"paper_metrics": str(paper_path), "eval_metrics": str(eval_path)}
    summary.update({"eval": evals, "updated_at": utc_now()})
    write_json(summary_path, summary)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.task)
    mode = normalize_mode(args.mode, config)
    run_id = args.run_id or ("smoke" if not args.dry_run else "dry_run")
    run_path = run_dir(args.task, run_id)
    profile = decode_profile(config)
    pred_path = _prediction_path(run_path, args.split)
    dry_payload = {
        "schema_version": "odcr_cier_eval_dry_run_v1",
        "baseline": BASELINE,
        "task_id": int(args.task),
        "run_id": run_id,
        "mode": mode,
        "split": args.split,
        "prediction_input": str(pred_path),
        "paper_metrics_output": str(run_path / "eval" / args.split / "paper_metrics.json"),
        "eval_metrics_output": str(run_path / "eval" / args.split / "eval_metrics.json"),
        "official_metric_adapter": "code/base_utils.official_paper_metrics",
        "decode": profile,
    }
    if args.dry_run:
        print(json.dumps(dry_payload, indent=2, sort_keys=True))
        return dry_payload
    if not pred_path.is_file():
        raise FileNotFoundError(f"Prediction JSONL not found: {pred_path}")
    rows = read_jsonl(pred_path)
    if not rows:
        raise ValueError(f"Prediction JSONL is empty: {pred_path}")
    _validate_rows(rows, task_id=args.task, split=args.split)
    rating = _rating_metrics(rows)
    text_metrics, metric_inputs = _official_text_metrics(rows, max_len=int(profile["max_length"]))
    paper_metrics = {
        "schema_version": "odcr_cier_paper_metrics_v1",
        "baseline": BASELINE,
        "task_id": int(args.task),
        "run_id": run_id,
        "split": args.split,
        "sample_count": len(rows),
        "decode": profile,
        "recommendation": {
            "mae": _finite_or_none(rating.get("mae")),
            "rmse": _finite_or_none(rating.get("rmse")),
            "protocol": rating.get("metric_protocol"),
        },
        "explanation": text_metrics,
        "official_metric_adapter": "code/base_utils.official_paper_metrics",
        "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
    }
    eval_metrics = {
        "schema_version": "odcr_cier_eval_metrics_v1",
        "baseline": BASELINE,
        "task_id": int(args.task),
        "run_id": run_id,
        "split": args.split,
        "sample_count": len(rows),
        "metrics": {
            "recommendation": rating,
            "explanation": text_metrics,
        },
        "paper_metrics": paper_metrics,
        "metric_input_summary": {
            "schema_version": metric_inputs[0]["schema_version"],
            "max_len": int(profile["max_length"]),
            "prediction_truncated_count": sum(1 for item in metric_inputs if item["prediction_truncated"]),
            "reference_truncated_count": sum(1 for item in metric_inputs if item["reference_truncated"]),
        },
        "diagnostic_note": "CIER original evaluator is diagnostic only; this file is produced by the ODCR metric adapter.",
        "uses_step3_step4_evidence_routing": False,
    }
    ensure_run_layout(run_path)
    paper_path = run_path / "eval" / args.split / "paper_metrics.json"
    eval_path = run_path / "eval" / args.split / "eval_metrics.json"
    write_json(paper_path, paper_metrics)
    write_json(eval_path, eval_metrics)
    _update_summary(run_path, split=args.split, paper_path=paper_path, eval_path=eval_path)
    print(json.dumps({"paper_metrics": str(paper_path), "eval_metrics": str(eval_path)}, indent=2))
    return eval_metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate CIER-adapted predictions with the ODCR metric adapter.")
    parser.add_argument("--task", type=int, required=True, choices=[2, 5, 7, 8])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--mode", choices=["source_to_target", "target_only", "source-to-target"], default=None)
    parser.add_argument("--split", choices=["valid", "test"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
