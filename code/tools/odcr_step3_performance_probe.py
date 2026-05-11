#!/usr/bin/env python3
"""Step3 bounded runtime performance probe.

This bridge/tooling entry runs a validation-namespace Step3 hot-path window.  It
does not start formal training, update formal latest pointers, write formal
checkpoints, or run Step4/Step5/eval/rerank.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.file_atomic import atomic_write_json  # noqa: E402
from odcr_core.step3_runtime_probe import (  # noqa: E402
    STEP3_RUNTIME_PROBE_TYPES,
    Step3ValidationNamespaceGuard,
    Step3ValidationWindowRequest,
    child_status_from_report,
    failure_report,
    run_step3_validation_window,
)


DEFAULT_SLUG = "step3_runtime_probe_truth_rebuild"


def _safe_component(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("component must be non-empty")
    return Step3ValidationNamespaceGuard(REPO_ROOT, 2, "tmp_slug", raw).run_id


def _append_log(path: str | Path | None, *parts: Any) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as handle:
        handle.write(" ".join(str(part) for part in parts) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Internal Step3 bounded runtime performance probe.")
    parser.add_argument("--probe-type", choices=STEP3_RUNTIME_PROBE_TYPES, required=True)
    parser.add_argument("--candidate-name", help="Optional validation-only Stage2 candidate name.")
    parser.add_argument("--namespace", choices=("validation",), required=True)
    parser.add_argument("--task", type=int, default=2)
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measured-steps", type=int, default=20)
    parser.add_argument("--max-seconds", type=int, default=170)
    parser.add_argument("--bridge-status-path", required=True)
    parser.add_argument("--bridge-log-path", required=True)
    parser.add_argument("--target-socket", required=True)
    parser.add_argument("--target-pane", required=True)
    parser.add_argument("--target-job-id", required=True)
    parser.add_argument("--target-node", required=True)
    return parser


def _target_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "socket": args.target_socket,
        "pane": args.target_pane,
        "job_id": args.target_job_id,
        "node": args.target_node,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.monotonic()
    run_id = _safe_component(args.run_id)
    _append_log(args.bridge_log_path, f"ODCR_BRIDGE_BEGIN_{run_id}")
    _append_log(
        args.bridge_log_path,
        "step3_runtime_probe",
        f"probe_type={args.probe_type}",
        "namespace=validation",
        f"task={int(args.task)}",
    )
    try:
        if int(args.task) != 2:
            raise ValueError("step3-performance-probe currently allows only task 2")
        if args.namespace != "validation":
            raise ValueError("step3-performance-probe requires validation namespace")
        report = run_step3_validation_window(
            task_id=int(args.task),
            validation_slug=str(args.slug),
            run_id=run_id,
            probe_type=str(args.probe_type),
            candidate_name=getattr(args, "candidate_name", None),
            warmup_steps=int(args.warmup_steps),
            measured_steps=int(args.measured_steps),
            max_wall_seconds=max(20, min(int(args.max_seconds), 170)),
            repo_root=REPO_ROOT,
            bridge_dispatched=True,
        )
    except Exception as exc:
        request = Step3ValidationWindowRequest(
            task_id=int(args.task),
            validation_slug=str(args.slug),
            run_id=run_id,
            probe_type=str(args.probe_type),
            candidate_name=getattr(args, "candidate_name", None),
            warmup_steps=int(args.warmup_steps),
            measured_steps=int(args.measured_steps),
            max_wall_seconds=max(20, min(int(args.max_seconds), 170)),
            bridge_dispatched=True,
        )
        guard = Step3ValidationNamespaceGuard(REPO_ROOT, int(args.task), str(args.slug), run_id)
        report = failure_report(
            request=request,
            guard=guard,
            failure_phase="probe_launcher",
            root_reason=repr(exc),
        )
    status = child_status_from_report(
        report,
        run_id=run_id,
        elapsed_s=time.monotonic() - started,
        max_seconds=int(args.max_seconds),
        target=_target_payload(args),
    )
    atomic_write_json(Path(args.bridge_status_path), status)
    report_json = str((report.get("paths") or {}).get("report_json") or "")
    _append_log(args.bridge_log_path, f"report_json={report_json}")
    _append_log(args.bridge_log_path, f"runtime_verified={str(bool(report.get('runtime_verified'))).lower()}")
    _append_log(args.bridge_log_path, f"evidence_complete={str(bool(report.get('evidence_complete'))).lower()}")
    _append_log(args.bridge_log_path, f"ODCR_BRIDGE_END_{run_id}")
    print(json.dumps(report, indent=2, sort_keys=True, default=str), flush=True)
    return int(status["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
