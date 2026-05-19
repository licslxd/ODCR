#!/usr/bin/env python3
"""Write Step5 formal-preparation preflight reports under AI_analysis only."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step5_formal_preparation import (  # noqa: E402
    build_formal_preparation_payloads,
    write_formal_preparation_reports,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, default=2)
    parser.add_argument("--from-step3", default="runs/step3/task2/2")
    parser.add_argument("--from-step4", default="runs/step4/task2/1")
    parser.add_argument("--step5a-result", required=True)
    parser.add_argument("--step5b-result", required=True)
    parser.add_argument("--compileall-pass", action="store_true")
    parser.add_argument("--doctor-pass", action="store_true")
    parser.add_argument("--guardrail-pass", action="store_true")
    parser.add_argument("--tests-pass", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payloads = build_formal_preparation_payloads(
        repo_root=REPO_ROOT,
        task_id=int(args.task),
        from_step3=str(args.from_step3),
        from_step4=str(args.from_step4),
        step5a_result_path=args.step5a_result,
        step5b_result_path=args.step5b_result,
        validation={
            "compileall_pass": bool(args.compileall_pass),
            "doctor_pass": bool(args.doctor_pass),
            "guardrail_pass": bool(args.guardrail_pass),
            "tests_pass": bool(args.tests_pass),
        },
    )
    write_formal_preparation_reports(repo_root=REPO_ROOT, payloads=payloads)
    return 0 if bool(payloads["machine"].get("allow_step5_formal_preparation")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
