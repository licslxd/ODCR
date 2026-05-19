#!/usr/bin/env python3
"""Write the ODCR two-phase current GPU pane handoff."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.aux.runtime.gpu_pane_handoff import (  # noqa: E402
    HandoffError,
    STALE_HANDOFF_MESSAGE,
    admin_part_path,
    current_handoff_path,
    validate_current_gpu_pane_payload,
    load_current_handoff,
    write_admin_pre_srun,
    write_failed_handoff,
    write_gpu_post_srun,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("admin-pre-srun", "gpu-post-srun", "validate-only"),
        default="gpu-post-srun",
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--source", default="odcr-enter-gpu")
    parser.add_argument("--job-id", default="")
    parser.add_argument("--selected-node", default="")
    parser.add_argument("--json", action="store_true", help="print the full payload on success")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    try:
        if args.mode == "admin-pre-srun":
            payload = write_admin_pre_srun(
                repo_root=repo_root,
                source=str(args.source),
                job_id=str(args.job_id),
                selected_node=str(args.selected_node),
            )
            output_path = admin_part_path(repo_root)
        elif args.mode == "gpu-post-srun":
            payload = write_gpu_post_srun(repo_root=repo_root, source=str(args.source))
            output_path = current_handoff_path(repo_root)
        else:
            payload = load_current_handoff(repo_root)
            if payload is None:
                raise HandoffError(f"current GPU pane handoff is missing: {current_handoff_path(repo_root)}")
            validate_current_gpu_pane_payload(payload)
            output_path = current_handoff_path(repo_root)
    except HandoffError as exc:
        print(f"[write_current_gpu_pane] ERROR: {exc}", file=sys.stderr)
        if exc.details:
            print(json.dumps(exc.details, ensure_ascii=False, sort_keys=True, default=str), file=sys.stderr)
        if args.mode == "admin-pre-srun":
            write_failed_handoff(
                repo_root=repo_root,
                mode=str(args.mode),
                source=str(args.source),
                error=str(exc),
                details=exc.details,
            )
        print(STALE_HANDOFF_MESSAGE, file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[write_current_gpu_pane] ERROR: {exc}", file=sys.stderr)
        if args.mode == "admin-pre-srun":
            write_failed_handoff(
                repo_root=repo_root,
                mode=str(args.mode),
                source=str(args.source),
                error=str(exc),
                details={},
            )
        print(STALE_HANDOFF_MESSAGE, file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    else:
        print(f"[write_current_gpu_pane] wrote {output_path}")
        if args.mode == "admin-pre-srun":
            print("[write_current_gpu_pane] admin pre-srun captured tmux metadata only; CUDA is captured after srun.")
        elif args.mode == "gpu-post-srun":
            print("[write_current_gpu_pane] GPU post-srun captured CUDA metadata without tmux metadata probes.")
            print("[write_current_gpu_pane] selection uses tmux socket + pane; TMUX server PID is diagnostic only.")
        else:
            print("[write_current_gpu_pane] current_gpu_pane.json is valid and fresh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
