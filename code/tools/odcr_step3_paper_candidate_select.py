#!/usr/bin/env python3
"""Retired Step3 paper-aware candidate selector.

Step3 no longer uses paper metrics for downstream readiness. Keep this command
as a fail-fast historical boundary; final paper metrics belong after
Step3 -> Step4 -> Step5 -> eval/rerank.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core import path_layout  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def build_selection(*, task_id: int, run_id: str, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    run_root = path_layout.get_stage_run_root(REPO_ROOT, int(task_id), "v1", "step3", str(run_id)).resolve()
    raise RuntimeError(
        f"Step3 paper candidate selection is retired for {run_root}. Use "
        "step3_upstream_readiness_gate for Step3->Step4 readiness; run paper metrics after Step5/eval."
    )


def write_selection(*, task_id: int, run_id: str, config: Mapping[str, Any] | None = None, dry_run: bool = False) -> dict[str, Any]:
    return build_selection(task_id=task_id, run_id=run_id, config=config)


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
