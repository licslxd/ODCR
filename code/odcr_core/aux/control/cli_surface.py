"""Runtime CLI surface for the single ODCR entrypoint."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from odcr_core.config_schema import OneControlConfigError
from odcr_core.aux.runtime.stage_dispatch import RUNTIME_STAGES, runtime_probe_bridge_args


BRIDGE_MODES = ("validate-only", "marker-probe", "cuda-probe")


def _add_bridge_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--timeout", default="auto")  # internal-only bridge sentinel, not One-Control config
    parser.add_argument("--socket")
    parser.add_argument("--target")
    parser.add_argument("--run-id")
    parser.add_argument("--no-send", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def add_runtime_parser(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    runtime = subparsers.add_parser("runtime", parents=[common], help="runtime probes and controlled tmux GPU bridge")
    runtime_sub = runtime.add_subparsers(dest="runtime_command", required=True)

    bridge = runtime_sub.add_parser("bridge", help="controlled tmux GPU bridge")
    bridge_sub = bridge.add_subparsers(dest="bridge_command", required=True)
    for mode in BRIDGE_MODES:
        child = bridge_sub.add_parser(mode, help=f"run bridge {mode}")
        _add_bridge_common(child)

    probe = runtime_sub.add_parser("probe", help="stage-dispatched runtime probe")
    probe.add_argument("--stage", choices=RUNTIME_STAGES, required=True)
    probe.add_argument("--task", type=int, required=True)
    probe.add_argument("--profile", default=None)
    probe.add_argument("--bounded", action="store_true")
    probe.add_argument("--probe-kind", default=None)
    probe.add_argument("--run-id", default=None)
    probe.add_argument("--no-send", action="store_true")
    probe.add_argument("--dry-run", action="store_true")


def _bridge_argv(args: argparse.Namespace) -> list[str]:
    argv = [str(args.bridge_command)]
    for name, flag in (("timeout", "--timeout"), ("socket", "--socket"), ("target", "--target"), ("run_id", "--run-id")):
        value = getattr(args, name, None)
        if value:
            argv.extend([flag, str(value)])
    if bool(getattr(args, "no_send", False)):
        argv.append("--no-send")
    if bool(getattr(args, "dry_run", False)):
        argv.append("--dry-run")
    return argv


def cmd_runtime(args: argparse.Namespace) -> None:
    if args.runtime_command == "bridge":
        from odcr_core.aux.runtime.gpu_bridge import main as bridge_main

        rc = bridge_main(_bridge_argv(args))
        if rc:
            raise SystemExit(rc)
        return

    if args.runtime_command == "probe":
        try:
            bridge_args = runtime_probe_bridge_args(
                stage=str(args.stage),
                task=int(args.task),
                profile=getattr(args, "profile", None),
                probe_kind=getattr(args, "probe_kind", None),
                bounded=bool(getattr(args, "bounded", False)),
                dry_run=bool(getattr(args, "dry_run", False)),
                no_send=bool(getattr(args, "no_send", False)),
                run_id=getattr(args, "run_id", None),
            )
        except ValueError as exc:
            raise OneControlConfigError(str(exc)) from exc
        from odcr_core.aux.runtime.gpu_bridge import main as bridge_main

        rc = bridge_main(bridge_args)
        if rc:
            raise SystemExit(rc)
        return

    raise OneControlConfigError(f"unknown runtime command: {args.runtime_command}")


def runtime_surface_summary() -> dict[str, object]:
    return {
        "runtime_commands": ["bridge", "probe"],
        "bridge_modes": list(BRIDGE_MODES),
        "stages": list(RUNTIME_STAGES),
        "arbitrary_shell": False,
    }
