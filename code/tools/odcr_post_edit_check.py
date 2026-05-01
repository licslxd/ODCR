#!/usr/bin/env python3
"""Run ODCR post-edit validation before Codex handoff.

This is a lightweight final-response gate for AI-assisted edits. It does not
run real preprocess, training, Step4, Step5, eval, or rerank work.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCOPE = "governance"
SCOPES = (
    "governance-fast",
    "docs",
    "governance",
    "config",
    "logging",
    "preprocess",
    "step3",
    "step4",
    "step5",
    "eval",
    "all",
)
DEFAULT_MANUAL_MAX_SECONDS = 900
REAL_STAGE_SUBCOMMANDS = {"step3", "step4", "step5"}
FORBIDDEN_REAL_SUBCOMMANDS = {"preprocess", "eval", "rerank"}
LOGGING_SCOPE_PATH_HINTS = (
    "logging",
    "log_",
    "_log",
    "metrics",
    "cache",
    "report",
    "run_summary",
    "latest.json",
    "AI_analysis",
    "path_layout",
)


@dataclass(frozen=True)
class CheckCommand:
    label: str
    argv: tuple[str, ...]
    display_argv: tuple[str, ...] | None = None

    def display(self) -> str:
        return shlex.join(self.display_argv or self.argv)


@dataclass(frozen=True)
class CommandResult:
    command: CheckCommand
    status: str
    returncode: int | None
    elapsed_seconds: float
    detail: str = ""


def _python_command(label: str, *args: str, python_executable: str) -> CheckCommand:
    return CheckCommand(
        label=label,
        argv=(python_executable, *args),
        display_argv=("python", *args),
    )


def _odcr_command(label: str, *args: str) -> CheckCommand:
    return CheckCommand(label=label, argv=("./odcr", *args))


def _test_command(repo_root: Path, rel_path: str, *, python_executable: str) -> CheckCommand | None:
    if not (repo_root / rel_path).is_file():
        return None
    return _python_command(rel_path, rel_path, "-v", python_executable=python_executable)


def _existing_tests(repo_root: Path, paths: Iterable[str], *, python_executable: str) -> list[CheckCommand]:
    commands: list[CheckCommand] = []
    for rel_path in paths:
        command = _test_command(repo_root, rel_path, python_executable=python_executable)
        if command is not None:
            commands.append(command)
    return commands


def _base_commands(*, python_executable: str) -> list[CheckCommand]:
    return [
        _python_command("compileall", "-m", "compileall", "-q", "code", python_executable=python_executable),
        _python_command(
            "guardrail strict",
            "code/tools/check_one_control_guardrails.py",
            "--strict",
            python_executable=python_executable,
        ),
    ]


def _governance_fast_commands(*, python_executable: str) -> list[CheckCommand]:
    return [
        _python_command(
            "py_compile guardrail",
            "-m",
            "py_compile",
            "code/tools/check_one_control_guardrails.py",
            python_executable=python_executable,
        ),
        _python_command(
            "py_compile post-edit check",
            "-m",
            "py_compile",
            "code/tools/odcr_post_edit_check.py",
            python_executable=python_executable,
        ),
        _python_command(
            "guardrail strict",
            "code/tools/check_one_control_guardrails.py",
            "--strict",
            python_executable=python_executable,
        ),
    ]


def _governance_commands(repo_root: Path, *, python_executable: str) -> list[CheckCommand]:
    return [
        *_existing_tests(
            repo_root,
            (
                "code/tests/test_one_control_guardrails.py",
                "code/tests/test_post_edit_check.py",
            ),
            python_executable=python_executable,
        ),
        _odcr_command("doctor", "doctor"),
    ]


def _preprocess_commands(repo_root: Path, *, python_executable: str) -> list[CheckCommand]:
    return _existing_tests(
        repo_root,
        (
            "code/tests/test_preprocess_contract_cleanup.py",
            "code/tests/test_preprocess_b_precision.py",
            "code/tests/test_path_layout_boundaries.py",
        ),
        python_executable=python_executable,
    )


def _logging_commands(repo_root: Path, *, python_executable: str) -> list[CheckCommand]:
    return [
        *_existing_tests(
            repo_root,
            (
                "code/tests/test_run_summary_logging.py",
                "code/tests/test_logging_console_file.py",
                "code/tests/test_logging_tail.py",
                "code/tests/test_path_layout_boundaries.py",
            ),
            python_executable=python_executable,
        ),
        _odcr_command("doctor", "doctor"),
    ]


def _step3_commands(repo_root: Path, *, python_executable: str) -> list[CheckCommand]:
    return [
        _odcr_command("show step3", "show", "--stage", "step3", "--task", "4"),
        _odcr_command("step3 dry-run", "step3", "--task", "4", "--dry-run"),
        *_existing_tests(
            repo_root,
            (
                "code/tests/test_step3_control_plane.py",
                "code/tests/test_step3_structured_stability.py",
            ),
            python_executable=python_executable,
        ),
    ]


def _step4_commands(repo_root: Path, *, python_executable: str) -> list[CheckCommand]:
    return [
        _odcr_command("show step4", "show", "--stage", "step4", "--task", "4"),
        _odcr_command("step4 dry-run", "step4", "--task", "4", "--dry-run"),
        *_existing_tests(
            repo_root,
            (
                "code/tests/test_step4_rcr_routing.py",
                "code/tests/test_index_contract.py",
            ),
            python_executable=python_executable,
        ),
    ]


def _step5_commands(repo_root: Path, *, python_executable: str) -> list[CheckCommand]:
    return [
        _odcr_command("show step5", "show", "--stage", "step5", "--task", "4"),
        _odcr_command("step5 dry-run", "step5", "--task", "4", "--dry-run"),
        *_existing_tests(
            repo_root,
            (
                "code/tests/test_step5_lci.py",
                "code/tests/test_step5_ccv_fca.py",
                "code/tests/test_step5_graph_safety.py",
            ),
            python_executable=python_executable,
        ),
    ]


def _eval_commands(repo_root: Path, *, python_executable: str) -> list[CheckCommand]:
    return [
        _odcr_command("show step5 for eval lineage", "show", "--stage", "step5", "--task", "4"),
        *_existing_tests(
            repo_root,
            (
                "code/tests/test_index_contract.py",
                "code/tests/test_phase4a_checkpoint_lineage.py",
                "code/tests/test_full_bleu_monitor_decode.py",
            ),
            python_executable=python_executable,
        ),
    ]


def _dedupe(commands: Iterable[CheckCommand]) -> list[CheckCommand]:
    out: list[CheckCommand] = []
    seen: set[str] = set()
    for command in commands:
        key = command.display()
        if key in seen:
            continue
        seen.add(key)
        out.append(command)
    return out


def plan_safety_violations(commands: Iterable[CheckCommand]) -> list[str]:
    """Return commands that would run real ODCR stages instead of lightweight checks."""

    violations: list[str] = []
    for command in commands:
        argv = command.argv
        display = command.display()
        if len(argv) >= 2 and argv[0] == "./odcr":
            subcommand = argv[1]
            if subcommand in FORBIDDEN_REAL_SUBCOMMANDS:
                violations.append(f"real {subcommand} command is forbidden: {display}")
            elif subcommand in REAL_STAGE_SUBCOMMANDS and "--dry-run" not in argv:
                violations.append(f"real {subcommand} command must use --dry-run: {display}")
    return violations


def suggest_scope_for_paths(paths: Iterable[str]) -> str | None:
    """Return a lightweight scope hint for path-triggered post-edit checks."""

    normalized = [str(path).replace("\\", "/") for path in paths]
    if any(any(hint in path for hint in LOGGING_SCOPE_PATH_HINTS) for path in normalized):
        return "logging"
    return None


def build_plan(
    scope: str,
    *,
    repo_root: Path = REPO_ROOT,
    python_executable: str = sys.executable,
) -> list[CheckCommand]:
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}; expected one of: {', '.join(SCOPES)}")

    if scope in {"governance-fast", "docs"}:
        commands = _governance_fast_commands(python_executable=python_executable)
    else:
        commands = _base_commands(python_executable=python_executable)

    if scope in {"governance", "config"}:
        commands.extend(_governance_commands(repo_root, python_executable=python_executable))
    elif scope == "logging":
        commands.extend(_logging_commands(repo_root, python_executable=python_executable))
    elif scope == "preprocess":
        commands.extend(_preprocess_commands(repo_root, python_executable=python_executable))
    elif scope == "step3":
        commands.extend(_step3_commands(repo_root, python_executable=python_executable))
    elif scope == "step4":
        commands.extend(_step4_commands(repo_root, python_executable=python_executable))
    elif scope == "step5":
        commands.extend(_step5_commands(repo_root, python_executable=python_executable))
    elif scope == "eval":
        commands.extend(_eval_commands(repo_root, python_executable=python_executable))
    elif scope == "all":
        commands.extend(_governance_commands(repo_root, python_executable=python_executable))
        commands.extend(_logging_commands(repo_root, python_executable=python_executable))
        commands.extend(_preprocess_commands(repo_root, python_executable=python_executable))
        commands.extend(_step3_commands(repo_root, python_executable=python_executable))
        commands.extend(_step4_commands(repo_root, python_executable=python_executable))
        commands.extend(_step5_commands(repo_root, python_executable=python_executable))
        commands.extend(_eval_commands(repo_root, python_executable=python_executable))

    commands = _dedupe(commands)
    violations = plan_safety_violations(commands)
    if violations:
        raise RuntimeError(
            "post-edit validation plans must stay lightweight; forbidden commands: "
            + "; ".join(violations)
        )
    return commands



def _env(repo_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    code_dir = str(repo_root / "code")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = code_dir if not existing else code_dir + os.pathsep + existing
    return env


def print_command_list(commands: Sequence[CheckCommand], *, scope: str, dry_run: bool, max_seconds: int) -> None:
    print("ODCR post-edit validation")
    print(f"scope: {scope}")
    print(f"dry_run: {str(dry_run).lower()}")
    print(f"timeout_per_command_seconds: {max_seconds}")
    print("Commands:")
    for idx, command in enumerate(commands, start=1):
        print(f"  [{idx}] {command.display()}")


def run_commands(commands: Sequence[CheckCommand], *, repo_root: Path, max_seconds: int) -> list[CommandResult]:
    results: list[CommandResult] = []
    env = _env(repo_root)
    for idx, command in enumerate(commands, start=1):
        print(f"\n[{idx}/{len(commands)}] RUN {command.display()}", flush=True)
        started = time.monotonic()
        try:
            proc = subprocess.run(command.argv, cwd=repo_root, env=env, timeout=max_seconds)
            elapsed = time.monotonic() - started
            if proc.returncode == 0:
                print(f"[{idx}/{len(commands)}] PASS {command.label} ({elapsed:.1f}s)", flush=True)
                results.append(CommandResult(command, "PASS", proc.returncode, elapsed))
            else:
                print(
                    f"[{idx}/{len(commands)}] FAIL {command.label} "
                    f"(exit {proc.returncode}, {elapsed:.1f}s)",
                    flush=True,
                )
                results.append(CommandResult(command, "FAIL", proc.returncode, elapsed, f"exit {proc.returncode}"))
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - started
            print(f"[{idx}/{len(commands)}] FAIL {command.label} (timeout after {max_seconds}s)", flush=True)
            results.append(CommandResult(command, "FAIL", None, elapsed, f"timeout after {max_seconds}s"))
    return results


def print_summary(results: Sequence[CommandResult], *, dry_run: bool = False) -> None:
    print("\nSummary:")
    if dry_run:
        for result in results:
            print(f"  [DRY-RUN] {result.command.display()}")
        print("Result: DRY-RUN (0 executed)")
        return

    failures = 0
    for result in results:
        if result.status != "PASS":
            failures += 1
        suffix = f" - {result.detail}" if result.detail else ""
        print(f"  [{result.status}] {result.command.display()}{suffix}")
    print(f"Result: {'FAIL' if failures else 'PASS'} ({failures} failed, {len(results) - failures} passed)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run lightweight ODCR post-edit validation. This final-response gate never "
            "runs real preprocess, training, Step4, Step5, eval, or rerank work."
        )
    )
    parser.add_argument(
        "--scope",
        choices=SCOPES,
        default=DEFAULT_SCOPE,
        help=f"validation scope; defaults to {DEFAULT_SCOPE}",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the command list without executing it")
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=DEFAULT_MANUAL_MAX_SECONDS,
        help="per-command timeout in seconds",
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")

    repo_root = Path(args.repo_root).resolve()
    commands = build_plan(args.scope, repo_root=repo_root, python_executable=sys.executable)
    print_command_list(commands, scope=args.scope, dry_run=args.dry_run, max_seconds=args.max_seconds)
    if args.dry_run:
        dry_results = [CommandResult(command, "DRY-RUN", None, 0.0) for command in commands]
        print_summary(dry_results, dry_run=True)
        return 0

    results = run_commands(commands, repo_root=repo_root, max_seconds=args.max_seconds)
    print_summary(results)
    return 1 if any(result.status != "PASS" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
