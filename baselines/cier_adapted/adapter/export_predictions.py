from __future__ import annotations

import argparse
import json
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
    task_domains,
    utc_now,
    write_json,
    write_jsonl,
)


def _raw_path(run_path: Path, split: str) -> Path:
    return run_path / "model" / f"cier_raw_{split}_predictions.jsonl"


def _output_path(run_path: Path, split: str) -> Path:
    return run_path / "predictions" / f"{split}_predictions.jsonl"


def _convert_row(row: dict[str, Any], *, config: dict[str, Any], split: str) -> dict[str, Any]:
    source_domain, target_domain = task_domains(config)
    required = ["user_id", "item_id", "rating_gold", "rating_pred", "reference", "prediction"]
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"raw CIER prediction missing fields {missing}: {row}")
    return {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "baseline": BASELINE,
        "task_id": int(config["task_id"]),
        "split": split,
        "user_id": str(row["user_id"]),
        "item_id": str(row["item_id"]),
        "source_domain": str(row.get("source_domain") or source_domain),
        "target_domain": str(row.get("target_domain") or target_domain),
        "rating_gold": float(row["rating_gold"]),
        "rating_pred": float(row["rating_pred"]),
        "reference": str(row["reference"]),
        "prediction": str(row["prediction"]),
    }


def _update_summary(run_path: Path, *, split: str, output: Path, count: int) -> None:
    summary_path = run_path / "meta" / "run_summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else {"schema_version": "odcr_cier_run_summary_v1"}
    predictions = dict(summary.get("predictions") or {})
    predictions[split] = {"path": str(output), "row_count": count}
    summary.update({"predictions": predictions, "updated_at": utc_now()})
    write_json(summary_path, summary)


def export_predictions(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.task)
    mode = normalize_mode(args.mode, config)
    run_id = args.run_id or ("smoke" if not args.dry_run else "dry_run")
    run_path = run_dir(args.task, run_id)
    profile = decode_profile(config)
    dry_payload = {
        "schema_version": "odcr_cier_export_dry_run_v1",
        "baseline": BASELINE,
        "task_id": int(args.task),
        "run_id": run_id,
        "mode": mode,
        "splits": args.splits,
        "raw_inputs": {split: str(_raw_path(run_path, split)) for split in args.splits},
        "outputs": {split: str(_output_path(run_path, split)) for split in args.splits},
        "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
        "decode": profile,
    }
    if args.dry_run:
        print(json.dumps(dry_payload, indent=2, sort_keys=True))
        return dry_payload
    ensure_run_layout(run_path)
    results: dict[str, Any] = {"baseline": BASELINE, "task_id": int(args.task), "run_id": run_id, "splits": {}}
    for split in args.splits:
        raw = _raw_path(run_path, split)
        if not raw.is_file():
            raise FileNotFoundError(f"Missing raw CIER predictions for {split}: {raw}")
        rows = [_convert_row(row, config=config, split=split) for row in read_jsonl(raw)]
        out = _output_path(run_path, split)
        count = write_jsonl(out, rows)
        _update_summary(run_path, split=split, output=out, count=count)
        results["splits"][split] = {"path": str(out), "row_count": count}
    print(json.dumps(results, indent=2, sort_keys=True))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export CIER-adapted predictions to the ODCR baseline JSONL schema.")
    parser.add_argument("--task", type=int, required=True, choices=[2, 5, 7, 8])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--mode", choices=["source_to_target", "target_only", "source-to-target"], default=None)
    parser.add_argument("--split", choices=["valid", "test"], default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.splits = [args.split] if args.split else ["valid", "test"]
    export_predictions(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

