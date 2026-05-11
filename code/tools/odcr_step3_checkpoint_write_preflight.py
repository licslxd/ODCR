#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.file_atomic import atomic_write_json  # noqa: E402
from odcr_core.step3_quality import checkpoint_event_from_sidecar, utc_now  # noqa: E402
from odcr_core.training_checkpoint import (  # noqa: E402
    STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION,
    checkpoint_file_sha256,
    file_fingerprint,
    stable_hash,
    write_checkpoint_lineage,
)


REQUIRED_EVENT_FIELDS = (
    "event_id",
    "checkpoint_file",
    "checkpoint_file_hash",
    "checkpoint_epoch",
    "selection_scope",
    "selection_metric",
    "selection_metric_value",
    "selection_direction",
    "reason",
    "replaced_previous",
    "global_best_epoch",
    "global_best_metric",
    "after_min_epochs_best_epoch",
    "after_min_epochs_best_metric",
    "resolved_config_hash",
    "training_runtime_config_hash",
    "epoch_summary_hash",
    "metrics_jsonl_hash",
    "quality_status",
    "downstream_ready",
    "created_at",
)


def _evidence_dir(task_id: int, output_root: str | Path | None = None) -> Path:
    root = Path(output_root) if output_root else REPO_ROOT / "AI_analysis" / "06_probe_evidence"
    return root / "step3_formal_checkpoint_profile_cache_numproc_rebuild" / f"task{int(task_id)}" / "checkpoint_write_preflight"


def run_preflight(*, task_id: int, output_root: str | Path | None = None) -> dict[str, Any]:
    out_dir = _evidence_dir(int(task_id), output_root)
    ckpt = out_dir / "model" / "preflight_latest.pth"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"odcr-step3-checkpoint-preflight\n")
    created_at = utc_now()
    payload = {
        "sidecar_schema_version": STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION,
        "stage": "step3",
        "run_id": "checkpoint_write_preflight",
        "task_id": int(task_id),
        "source_domain": "preflight_source",
        "target_domain": "preflight_target",
        "task_profile_id": "task2_strong_forward_g1s" if int(task_id) == 2 else "preflight_profile",
        "profile_isolation_hash": stable_hash({"task_id": int(task_id), "profile": "preflight"}),
        "checkpoint_path": str(ckpt.resolve()),
        "checkpoint_file": file_fingerprint(ckpt),
        "checkpoint_file_hash": checkpoint_file_sha256(ckpt),
        "checkpoint_epoch": 1,
        "selection_metric": "valid_loss",
        "selection_metric_value": 1.25,
        "selection_direction": "min",
        "selection_scope": "latest",
        "reason": "latest_epoch_snapshot",
        "replaced_previous": False,
        "global_best_epoch": 1,
        "global_best_metric": 1.25,
        "after_min_epochs_best_epoch": 1,
        "after_min_epochs_best_metric": 1.25,
        "epoch_summary_hash": stable_hash({"epoch": 1, "valid_loss": 1.25}),
        "metrics_jsonl_hash": stable_hash({"split": "valid", "loss": 1.25}),
        "resolved_config_hash": stable_hash({"task_id": int(task_id), "config": "preflight"}),
        "training_runtime_config_hash": stable_hash({"task_id": int(task_id), "runtime": "preflight"}),
        "quality_status_at_save": "not_evaluated",
        "quality_status": "not_evaluated",
        "downstream_ready": False,
        "created_at": created_at,
    }
    sidecar_event = checkpoint_event_from_sidecar(
        payload,
        reason=str(payload["reason"]),
        replaced_previous=bool(payload["replaced_previous"]),
    )
    missing_sidecar_event = [field for field in REQUIRED_EVENT_FIELDS if field not in sidecar_event]
    if missing_sidecar_event:
        raise RuntimeError("checkpoint_event_from_sidecar missing fields: " + ", ".join(missing_sidecar_event))
    lineage_path = write_checkpoint_lineage(ckpt, payload)
    ledger_path = out_dir / "state" / "checkpoint_lineage.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    events = ledger.get("saved_checkpoint_events") or []
    if not events:
        raise RuntimeError("checkpoint preflight ledger did not record an event")
    latest = dict(events[-1])
    missing_lineage_event = [field for field in REQUIRED_EVENT_FIELDS if field not in latest]
    if missing_lineage_event:
        raise RuntimeError("checkpoint lineage event missing fields: " + ", ".join(missing_lineage_event))
    result = {
        "schema_version": "odcr_step3_checkpoint_write_preflight/1",
        "status": "pass",
        "task_id": int(task_id),
        "evidence_dir": str(out_dir.resolve()),
        "checkpoint_file": str(ckpt.resolve()),
        "checkpoint_file_hash": payload["checkpoint_file_hash"],
        "lineage_path": str(lineage_path),
        "ledger_path": str(ledger_path.resolve()),
        "reason": payload["reason"],
        "replaced_previous": payload["replaced_previous"],
        "selection_scope": payload["selection_scope"],
        "selection_metric": payload["selection_metric"],
        "selection_metric_value": payload["selection_metric_value"],
        "required_event_fields": list(REQUIRED_EVENT_FIELDS),
        "created_at": created_at,
    }
    atomic_write_json(out_dir / "preflight_result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)
    result = run_preflight(task_id=int(args.task), output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
