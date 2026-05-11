#!/usr/bin/env python3
"""Build Step3 paper-aware downstream candidate selection metadata.

This tool never runs eval and never overwrites checkpoints. It consumes existing
paper_target_only_eval summaries and writes ``meta/paper_candidate_selection.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.file_atomic import atomic_write_json  # noqa: E402
from odcr_core import path_layout  # noqa: E402
from odcr_core.step3_eval_handoff import default_eval_paths  # noqa: E402
from odcr_core.step3_v3_policy import flatten_paper_metrics, select_paper_aware_candidates  # noqa: E402
from odcr_core.training_checkpoint import checkpoint_file_sha256  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _candidate_from_existing_paper_eval(run_root: Path, *, scope: str, checkpoint: Path) -> dict[str, Any] | None:
    paths = default_eval_paths(run_root)
    valid_path = paths["valid_eval_path"]
    test_path = paths["test_eval_path"]
    if not valid_path.is_file() or not test_path.is_file():
        return None
    valid = _load_json(valid_path)
    test = _load_json(test_path)
    if str(valid.get("eval_protocol")) != "paper_target_only_eval" or str(test.get("eval_protocol")) != "paper_target_only_eval":
        return None
    metrics = flatten_paper_metrics(valid.get("metrics") if isinstance(valid.get("metrics"), Mapping) else {})
    test_metrics = flatten_paper_metrics(test.get("metrics") if isinstance(test.get("metrics"), Mapping) else {})
    return {
        "candidate_id": scope,
        "checkpoint_scope": scope,
        "checkpoint": _rel(checkpoint),
        "checkpoint_hash": checkpoint_file_sha256(checkpoint),
        "metrics": metrics,
        "test_metrics": test_metrics,
        "paper_eval_paths": {"valid": _rel(valid_path), "test": _rel(test_path)},
        "paper_eval_protocol": "paper_target_only_eval",
    }


def build_selection(*, task_id: int, run_id: str, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    run_root = path_layout.get_stage_run_root(REPO_ROOT, int(task_id), "v1", "step3", str(run_id)).resolve()
    best = run_root / "model" / "best_observed.pth"
    candidates: list[dict[str, Any]] = []
    if best.is_file():
        item = _candidate_from_existing_paper_eval(run_root, scope="best_observed", checkpoint=best)
        if item is not None:
            candidates.append(item)
    selection = select_paper_aware_candidates(candidates, config=config or {})
    selection.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "task_id": int(task_id),
            "run_id": str(run_id),
            "run_root": _rel(run_root),
            "candidate_pool_policy": [
                "best_observed",
                "best_after_min_epochs",
                "top_k_valid_loss",
                "milestone",
                "recovery",
                "ema",
                "checkpoint_average",
            ],
            "existing_candidates": candidates,
            "no_eval_rerun": True,
            "best_observed_not_overwritten": True,
            "latest_not_selected_unless_candidate_policy": True,
        }
    )
    return selection


def write_selection(*, task_id: int, run_id: str, config: Mapping[str, Any] | None = None, dry_run: bool = False) -> dict[str, Any]:
    selection = build_selection(task_id=task_id, run_id=run_id, config=config)
    run_root = path_layout.get_stage_run_root(REPO_ROOT, int(task_id), "v1", "step3", str(run_id)).resolve()
    meta = run_root / "meta"
    out = meta / "paper_candidate_selection.json"
    result = {
        "dry_run": bool(dry_run),
        "selection_path": _rel(out),
        "selection": selection,
    }
    if dry_run:
        return result
    atomic_write_json(out, selection)
    summary_path = meta / "run_summary.json"
    if summary_path.is_file():
        summary = _load_json(summary_path)
        summary["paper_candidate_selection"] = {
            "path": _rel(out),
            "schema_version": selection.get("schema_version"),
            "candidate_count": selection.get("candidate_count"),
            "scorer_downstream_checkpoint": selection.get("scorer_downstream_checkpoint"),
            "explainer_downstream_checkpoint": selection.get("explainer_downstream_checkpoint"),
            "dist_guard_active": True,
        }
        scorer = selection.get("scorer_downstream_checkpoint")
        if isinstance(scorer, Mapping) and scorer.get("checkpoint"):
            summary["scorer_downstream_checkpoint"] = scorer.get("checkpoint")
            summary["scorer_downstream_checkpoint_hash"] = scorer.get("checkpoint_hash")
        explainer = selection.get("explainer_downstream_checkpoint")
        if isinstance(explainer, Mapping) and explainer.get("checkpoint"):
            summary["explainer_downstream_checkpoint"] = explainer.get("checkpoint")
        else:
            summary["explainer_downstream_checkpoint"] = None
            summary["explainer_downstream_block_reason"] = "DIST guard did not pass for available paper-evaluated candidates"
        atomic_write_json(summary_path, summary)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, default=2)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = write_selection(task_id=int(args.task), run_id=str(args.run_id), dry_run=bool(args.dry_run))
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
