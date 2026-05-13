"""Runtime probe entry helpers."""

from __future__ import annotations

from typing import Sequence

from .stage_dispatch import runtime_probe_bridge_args


def build_probe_bridge_args(
    *,
    stage: str,
    task: int,
    profile: str | None,
    bounded: bool,
    dry_run: bool = False,
    no_send: bool = False,
    run_id: str | None = None,
) -> tuple[str, ...]:
    return runtime_probe_bridge_args(
        stage=stage,
        task=task,
        profile=profile,
        bounded=bounded,
        dry_run=dry_run,
        no_send=no_send,
        run_id=run_id,
    )


def run_probe_bridge(argv: Sequence[str]) -> int:
    from .gpu_bridge import main as bridge_main

    return bridge_main(tuple(argv))

