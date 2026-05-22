"""Runtime CLI facade used by ./odcr."""

from __future__ import annotations

import argparse
from typing import Sequence

from odcr_core.aux.runtime.tmux_gpu_bridge import main as runtime_main


def build_runtime_parser() -> argparse.ArgumentParser:
    from odcr_core.aux.runtime.tmux_gpu_bridge import build_parser

    return build_parser()


def cmd_runtime(args: argparse.Namespace) -> int:
    argv: list[str] = []
    if args.runtime_command == "bridge":
        argv.extend(["bridge", args.bridge_command])
        for key in ("socket", "target"):
            value = getattr(args, key, None)
            if value:
                argv.extend([f"--{key}", str(value)])
        if bool(getattr(args, "global_discovery", False)):
            argv.append("--global")
        if bool(getattr(args, "all_sockets", False)):
            argv.append("--all-sockets")
        if bool(getattr(args, "all_panes", False)):
            argv.append("--all-panes")
        if bool(getattr(args, "json_output", False)):
            argv.append("--json")
        if bool(getattr(args, "dry_run", False)):
            argv.append("--dry-run")
        if bool(getattr(args, "no_send", False)):
            argv.append("--no-send")
        if getattr(args, "timeout", None) is not None:
            argv.extend(["--timeout", str(args.timeout)])
        if args.bridge_command == "exec":
            if bool(getattr(args, "background", False)):
                argv.append("--background")
            if not bool(getattr(args, "require_cuda", True)):
                argv.append("--no-require-cuda")
            for key, flag in (
                ("stdout_path", "--stdout"),
                ("stderr_path", "--stderr"),
                ("pid_file", "--pid-file"),
                ("status_path", "--status-path"),
            ):
                value = getattr(args, key, None)
                if value:
                    argv.extend([flag, str(value)])
            if not bool(getattr(args, "stderr_to_stdout", True)):
                argv.append("--split-stderr")
            exec_argv = list(getattr(args, "exec_argv", []) or [])
            if exec_argv:
                argv.append("--")
                argv.extend(str(item) for item in exec_argv if str(item) != "--")
        if args.bridge_command == "_handshake-child":
            for key in ("kind", "status_path", "log_path", "report_path", "repo_root", "stage", "task"):
                value = getattr(args, key, None)
                if value is not None:
                    argv.extend([f"--{key.replace('_', '-')}", str(value)])
            if bool(getattr(args, "require_cuda", False)):
                argv.append("--require-cuda")
    elif args.runtime_command == "probe":
        argv.extend(["probe", "--stage", str(args.stage), "--task", str(args.task)])
        if bool(getattr(args, "bounded", False)):
            argv.append("--bounded")
        if getattr(args, "config", None):
            argv.extend(["--config", str(args.config)])
        for item in list(getattr(args, "sets", []) or []):
            argv.extend(["--set", str(item)])
        if getattr(args, "candidate_id", None):
            argv.extend(["--candidate-id", str(args.candidate_id)])
        if getattr(args, "timeout", None) is not None:
            argv.extend(["--timeout", str(args.timeout)])
        if getattr(args, "from_step4", None):
            argv.extend(["--from-step4-run", str(args.from_step4)])
        if getattr(args, "evidence_level", None):
            argv.extend(["--evidence-level", str(args.evidence_level)])
        if bool(getattr(args, "scan", False)):
            argv.append("--scan")
        if bool(getattr(args, "global_discovery", False)):
            argv.append("--global")
        if bool(getattr(args, "probe_child", False)):
            argv.append("--probe-child")
        if getattr(args, "status_path", None):
            argv.extend(["--status-path", str(args.status_path)])
        for key in ("socket", "target"):
            value = getattr(args, key, None)
            if value:
                argv.extend([f"--{key}", str(value)])
    return runtime_main(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return runtime_main(argv)
