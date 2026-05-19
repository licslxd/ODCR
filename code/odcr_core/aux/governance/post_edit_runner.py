#!/usr/bin/env python3
"""Run ODCR post-edit validation before Codex handoff.

This is a lightweight final-response gate for AI-assisted edits. It does not
run real preprocess, training, Step4, Step5, eval, or rerank work.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from odcr_core.aux.evidence.ai_analysis_writer import get_writer
from odcr_core.aux.governance.post_edit_registry import (
    DEFAULT_MANUAL_MAX_SECONDS,
    DEFAULT_SCOPE,
    LOGGING_SCOPE_PATH_HINTS,
    SCOPES,
    SCOPE_REASONS,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
REAL_STAGE_SUBCOMMANDS = {"step3", "step4", "step5"}
FORBIDDEN_REAL_SUBCOMMANDS = {"preprocess", "eval", "rerank"}


@dataclass(frozen=True)
class CheckCommand:
    label: str
    argv: tuple[str, ...]
    display_argv: tuple[str, ...] | None = None
    group: str = "general"

    def display(self) -> str:
        return shlex.join(self.display_argv or self.argv)


@dataclass(frozen=True)
class CommandResult:
    command: CheckCommand
    status: str
    returncode: int | None
    elapsed_seconds: float
    detail: str = ""
    classification: str = "not_run"
    blocks_gpu_probe: bool = False
    blocks_formal: bool = False
    killed_by_signal: int | None = None
    peak_rss_kb: int | None = None
    rerun_status: str | None = None
    rerun_returncode: int | None = None
    rerun_elapsed_seconds: float | None = None


def _python_command(label: str, *args: str, python_executable: str) -> CheckCommand:
    return CheckCommand(
        label=label,
        argv=(python_executable, *args),
        display_argv=("python", *args),
    )


def _cleanup_pycache_command(*, python_executable: str) -> CheckCommand:
    script = (
        "from pathlib import Path\n"
        "import shutil\n"
        "root=Path('code')\n"
        "for path in sorted(root.rglob('__pycache__'), key=lambda p: len(p.parts), reverse=True):\n"
        "    shutil.rmtree(path, ignore_errors=True)\n"
        "for path in list(root.rglob('*.pyc')) + list(root.rglob('*.pyo')):\n"
        "    path.unlink(missing_ok=True)\n"
    )
    return _python_command("cleanup pycache", "-c", script, python_executable=python_executable)


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
        _cleanup_pycache_command(python_executable=python_executable),
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
        _cleanup_pycache_command(python_executable=python_executable),
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
                "code/tests/test_stage_truth_antiforgery.py",
                "code/tests/test_stage_status_strict_validation.py",
                "code/tests/test_upstream_resolver_malformed_status.py",
                "code/tests/test_stage_promotion_strict.py",
                "code/tests/test_stage_truth_machine_verdict.py",
                "code/tests/test_no_accum_batch_semantics.py",
                "code/tests/test_grad_accum_removed.py",
                "code/tests/test_no_accum_guardrail.py",
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
        _odcr_command("show step3 default", "show", "--stage", "step3"),
        _odcr_command("show step3 task2", "show", "--stage", "step3", "--task", "2"),
        _odcr_command("show step3 task2 verbose", "show", "--stage", "step3", "--task", "2", "--verbose"),
        _odcr_command("show step3 task5", "show", "--stage", "step3", "--task", "5"),
        _odcr_command("show step3 task8", "show", "--stage", "step3", "--task", "8"),
        _odcr_command("show step3 task7", "show", "--stage", "step3", "--task", "7"),
        _odcr_command("step3 task2 dry-run", "step3", "--task", "2", "--dry-run"),
        _odcr_command("step3 task2 cache-check", "step3", "--task", "2", "--dry-run", "--cache-check", "--expect-profile", "task2_strong_forward_g1s", "--expect-num-proc", "8"),
        _odcr_command("step3 task2 checkpoint-write-preflight", "step3", "--task", "2", "--dry-run", "--checkpoint-write-preflight", "--expect-profile", "task2_strong_forward_g1s"),
        _odcr_command("step3 task5 dry-run", "step3", "--task", "5", "--dry-run"),
        _odcr_command("step3 task8 dry-run", "step3", "--task", "8", "--dry-run"),
        _odcr_command("step3 task7 dry-run", "step3", "--task", "7", "--dry-run"),
        *_existing_tests(
            repo_root,
            (
                "code/tests/test_step3_clean_baseline_no_legacy_params.py",
                "code/tests/test_no_accum_batch_semantics.py",
                "code/tests/test_grad_accum_removed.py",
                "code/tests/test_step3_no_accum_training_loop.py",
                "code/tests/test_no_accum_guardrail.py",
                "code/tests/test_step3_checkpoint_lineage_sidecar.py",
                "code/tests/test_step3_checkpoint_event_from_sidecar_signature.py",
                "code/tests/test_step3_formal_checkpoint_save_hot_path.py",
                "code/tests/test_step3_checkpoint_lineage_event_required_fields.py",
                "code/tests/test_step3_checkpoint_write_preflight.py",
                "code/tests/test_step3_checkpoint_global_best_regression.py",
                "code/tests/test_step3_task2_formal_default_is_g1s.py",
                "code/tests/test_step3_expect_profile_assertion.py",
                "code/tests/test_stage2_selected_candidate_bound_to_resolver.py",
                "code/tests/test_step3_g1_backup_not_default.py",
                "code/tests/test_step3_g2c_not_formal.py",
                "code/tests/test_step3_show_formal_only.py",
                "code/tests/test_step3_task_profile_isolation_display.py",
                "code/tests/test_step3_resolved_vs_runtime_config_contract.py",
                "code/tests/test_step3_logging_contract.py",
                "code/tests/test_step3_metrics_writers.py",
                "code/tests/test_step3_legacy_unknown_key_rejected.py",
                "code/tests/test_step3_control_plane.py",
                "code/tests/test_tmux_gpu_bridge.py",
                "code/tests/test_gpu_bridge_no_whitelist_hard_blocker.py",
                "code/tests/test_gpu_runtime_executor_namespace_guard.py",
                "code/tests/test_post_edit_not_gpu_gate.py",
                "code/tests/test_stage2_runtime_first_flow.py",
                "code/tests/test_stage2_candidate_selection_uses_runtime_evidence.py",
                "code/tests/test_step3_startup_validation_entry.py",
                "code/tests/test_step3_structured_stability.py",
                "code/tests/test_step3_pre_ddp_tokenizer_cache_startup.py",
                "code/tests/test_step3_tokenizer_cache_atomic_manifest.py",
                "code/tests/test_step3_failed_latest_gate.py",
                "code/tests/test_step3_runtime_config_failure_artifact.py",
                "code/tests/test_step3_cache_path_layout_one_control.py",
                "code/tests/test_step3_no_distributed_collective_in_cache_phase.py",
                "code/tests/test_step3_tokenizer_cache_manifest.py",
                "code/tests/test_step3_tokenizer_cache_reuses_across_g1_g1s.py",
                "code/tests/test_step3_tokenizer_cache_ignores_training_params.py",
                "code/tests/test_step3_tokenizer_cache_full_run_hash_record_only.py",
                "code/tests/test_step3_tokenizer_cache_miss_reason.py",
                "code/tests/test_step3_tokenizer_cache_hard_gate_fields.py",
                "code/tests/test_step3_cache_check_hit.py",
                "code/tests/test_step3_cache_check_miss_reason.py",
                "code/tests/test_step3_cache_check_no_write.py",
                "code/tests/test_step3_cache_check_expect_hit_fail.py",
                "code/tests/test_step3_tokenization_num_proc_auto_12core.py",
                "code/tests/test_step3_tokenization_num_proc_warm_cache_not_used.py",
                "code/tests/test_step3_num_proc_distinct_from_dataloader_workers.py",
                "code/tests/test_step3_num_proc_source_table.py",
                "code/tests/test_step3_cache_check_selected_num_proc.py",
                "code/tests/test_step3_failed_run2_latest_rejection.py",
                "code/tests/test_step3_eval_two_phase_no_barrier_after_cpu_metric.py",
                "code/tests/test_step3_eval_prediction_shards_have_sample_id.py",
                "code/tests/test_step3_eval_cpu_metric_after_destroy_pg.py",
                "code/tests/test_step3_eval_paper_target_only_protocol.py",
                "code/tests/test_step3_eval_no_bertscore_in_paper_protocol.py",
                "code/tests/test_step3_eval_25_token_paper_protocol.py",
                "code/tests/test_step3_eval_diagnostic_48_not_paper_comparable.py",
                "code/tests/test_step3_eval_batch_invariance.py",
                "code/tests/test_step3_eval_batch_scaling_no_metric_change.py",
                "code/tests/test_step3_eval_run2_checkpoint_eval_only.py",
                "code/tests/test_step3_eval_handoff.py",
                "code/tests/test_scheduler_pure_warmup_cosine_no_damping.py",
                "code/tests/test_scheduler_explicit_damping_semantics.py",
                "code/tests/test_scheduler_safe_damping_v2_semantics.py",
                "code/tests/test_scheduler_current_lr_floor_explained.py",
                "code/tests/test_training_effectiveness_gate_plateau.py",
                "code/tests/test_step3_v3_recovery_conflict_paper_selection.py",
                "code/tests/test_loss_component_dashboard_fields.py",
                "code/tests/test_run_status_train_eval_split.py",
                "code/tests/test_step3_quality_evidence_performance_rebuild.py",
                "code/tests/test_step3_performance_probe_requires_runtime_verified.py",
                "code/tests/test_step3_performance_probe_rejects_status_only.py",
                "code/tests/test_tmux_gpu_bridge_runtime_success_semantics.py",
                "code/tests/test_step3_performance_probe_metrics_required.py",
                "code/tests/test_step3_validation_namespace_guard.py",
                "code/tests/test_step3_bounded_hot_path_entry.py",
                "code/tests/test_stage2_collector_rejects_null_summaries.py",
                "code/tests/test_evidence_level_no_overclaim.py",
                "code/tests/test_latest_run_resolution.py",
                "code/tests/test_run_summary_logging.py",
                "code/tests/test_logging_console_file.py",
                "code/tests/test_path_layout_boundaries.py",
            ),
            python_executable=python_executable,
        ),
    ]


def _step4_commands(repo_root: Path, *, python_executable: str) -> list[CheckCommand]:
    return [
        _odcr_command("step4 help", "step4", "--help"),
        *_existing_tests(
            repo_root,
            (
                "code/tests/test_step4_rcr_routing.py",
                "code/tests/test_step4_export_validator.py",
                "code/tests/test_step4_readiness_requires_manifest.py",
                "code/tests/test_stage_truth_upstream_gate.py",
                "code/tests/test_index_contract.py",
            ),
            python_executable=python_executable,
        ),
    ]


def _step5_commands(repo_root: Path, *, python_executable: str) -> list[CheckCommand]:
    return [
        CheckCommand(
            label="step5 resolver dry-run",
            argv=(
                python_executable,
                "-c",
                (
                    "import sys; "
                    "from pathlib import Path; "
                    "import tempfile; "
                    "sys.path.insert(0, 'code'); "
                    "sys.path.insert(0, 'code/tests'); "
                    "import odcr_core.config_resolver as cr; "
                    "from odcr_core.config_resolver import resolve_config; "
                    "from helpers.fixtures import write_step4_upstream_fixture; "
                    "tmp=tempfile.TemporaryDirectory(); "
                    "repo=Path(tmp.name); "
                    "write_step4_upstream_fixture(repo, task_id=4, run_id='1'); "
                    "old=cr._REPO_ROOT\n"
                    "cr._REPO_ROOT=repo\n"
                    "try:\n"
                    "    resolve_config(config_path=Path.cwd() / 'configs' / 'odcr.yaml', command='step5', "
                    "task_id=4, set_overrides=[], dry_run=True, from_step4='1', "
                    "eval_profile='balanced_2gpu', mode='train_only')\n"
                    "finally:\n"
                    "    cr._REPO_ROOT=old\n"
                    "    tmp.cleanup()\n"
                ),
            ),
            display_argv=("python", "-c", "step5 resolver dry-run"),
            group="dry-run",
        ),
        *_existing_tests(
            repo_root,
            (
                "code/tests/test_step5_lci.py",
                "code/tests/test_step5_ccv_fca.py",
                "code/tests/test_step5_graph_safety.py",
                "code/tests/test_step5_cache_manifest.py",
                "code/tests/test_step5_token_cache_determinism.py",
                "code/tests/test_step5_index_contract_audit.py",
                "code/tests/test_step5_artifact_build_preflight.py",
                "code/tests/test_step5_auto_budget_tuning.py",
                "code/tests/test_step5_rating_only_eval_handoff.py",
            ),
            python_executable=python_executable,
        ),
    ]


def _eval_commands(repo_root: Path, *, python_executable: str) -> list[CheckCommand]:
    return [
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
            elif subcommand in REAL_STAGE_SUBCOMMANDS and "--dry-run" not in argv and "--help" not in argv:
                violations.append(f"real {subcommand} command must use --dry-run: {display}")
    return violations


def suggest_scope_for_paths(paths: Iterable[str]) -> str | None:
    """Return a lightweight scope hint for path-triggered post-edit checks."""

    normalized = [str(path).replace("\\", "/") for path in paths]
    if any(any(hint in path for hint in LOGGING_SCOPE_PATH_HINTS) for path in normalized):
        return "logging"
    return None


def command_group(command: CheckCommand) -> str:
    if command.group != "general":
        return command.group
    label = command.label.lower()
    display = command.display()
    if "compile" in label:
        return "compileall"
    if "guardrail" in label:
        return "guardrail"
    if display.startswith("./odcr doctor"):
        return "doctor"
    if display.startswith("./odcr show"):
        return "show"
    if "--dry-run" in display:
        return "dry-run"
    if "code/tests/" in display:
        return "tests"
    return "general"


def classify_exit(returncode: int | None, *, timed_out: bool = False) -> tuple[str, int | None]:
    if timed_out:
        return "timeout", None
    if returncode is None:
        return "unknown", None
    if returncode == 0:
        return "pass", None
    if returncode < 0:
        return "resource_kill", abs(int(returncode))
    if returncode in {128 + signal.SIGKILL, 128 + signal.SIGTERM}:
        return "resource_kill", int(returncode) - 128
    return "semantic_fail", None


def classification_blocks_gpu_probe(classification: str) -> bool:
    return False


def classification_blocks_formal(classification: str) -> bool:
    return classification in {"semantic_fail", "test_fail", "timeout", "resource_kill", "unknown", "P0_semantic_blocker"}


def _is_single_test_command(command: CheckCommand) -> bool:
    return any(part.startswith("code/tests/test_") and part.endswith(".py") for part in command.argv)


def post_edit_results_block_gpu_probe(results: Sequence[CommandResult]) -> bool:
    return any(result.blocks_gpu_probe for result in results)


def post_edit_results_block_formal(results: Sequence[CommandResult]) -> bool:
    return any(result.blocks_formal for result in results)


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
    env.setdefault("ODCR_TEST_ARTIFACT_ROOT", str((repo_root / "test_artifacts").resolve()))
    env.setdefault("PYTHONPYCACHEPREFIX", str(Path("/tmp") / f"odcr_post_edit_pycache_{os.getuid()}"))
    return env


def scope_reason(scope: str) -> str:
    return SCOPE_REASONS.get(scope, "manual scope selected by caller")


def print_command_list(commands: Sequence[CheckCommand], *, scope: str, dry_run: bool, max_seconds: int) -> None:
    print("ODCR post-edit validation")
    print(f"scope: {scope}")
    print(f"scope_reason: {scope_reason(scope)}")
    print(f"dry_run: {str(dry_run).lower()}")
    print(f"timeout_per_command_seconds: {max_seconds}")
    print("Commands:")
    for idx, command in enumerate(commands, start=1):
        print(f"  [{idx}] {command.display()}")


def run_commands(commands: Sequence[CheckCommand], *, repo_root: Path, max_seconds: int) -> list[CommandResult]:
    results: list[CommandResult] = []
    env = _env(repo_root)
    for idx, command in enumerate(commands, start=1):
        group = command_group(command)
        print(f"\n[{idx}/{len(commands)}] RUN {command.display()} [group={group}]", flush=True)
        started = time.monotonic()
        rss_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        try:
            proc = subprocess.run(command.argv, cwd=repo_root, env=env, timeout=max_seconds)
            elapsed = time.monotonic() - started
            rss_after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            peak_rss_kb = int(max(rss_before, rss_after))
            classification, killed_signal = classify_exit(proc.returncode)
            if proc.returncode == 0:
                print(f"[{idx}/{len(commands)}] PASS {command.label} ({elapsed:.1f}s)", flush=True)
                results.append(
                    CommandResult(
                        command,
                        "PASS",
                        proc.returncode,
                        elapsed,
                        classification="pass",
                        blocks_gpu_probe=False,
                        blocks_formal=False,
                        peak_rss_kb=peak_rss_kb,
                    )
                )
            else:
                rerun_status: str | None = None
                rerun_returncode: int | None = None
                rerun_elapsed: float | None = None
                final_classification = classification
                if classification == "resource_kill" and _is_single_test_command(command):
                    print(f"[{idx}/{len(commands)}] RESOURCE-KILL {command.label}; fail-close without PASS masking", flush=True)
                status = "FAIL"
                detail = f"exit {proc.returncode}; classification={final_classification}"
                if killed_signal:
                    detail += f"; signal={killed_signal}"
                if rerun_status:
                    detail += f"; rerun={rerun_status}"
                print(
                    f"[{idx}/{len(commands)}] {status} {command.label} "
                    f"(exit {proc.returncode}, {elapsed:.1f}s, classification={final_classification})",
                    flush=True,
                )
                results.append(
                    CommandResult(
                        command,
                        status,
                        proc.returncode,
                        elapsed,
                        detail,
                        classification=final_classification,
                        blocks_gpu_probe=classification_blocks_gpu_probe(final_classification),
                        blocks_formal=classification_blocks_formal(final_classification),
                        killed_by_signal=killed_signal,
                        peak_rss_kb=peak_rss_kb,
                        rerun_status=rerun_status,
                        rerun_returncode=rerun_returncode,
                        rerun_elapsed_seconds=rerun_elapsed,
                    )
                )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - started
            rss_after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            classification, killed_signal = classify_exit(None, timed_out=True)
            print(f"[{idx}/{len(commands)}] FAIL {command.label} (timeout after {max_seconds}s)", flush=True)
            results.append(
                CommandResult(
                    command,
                    "FAIL",
                    None,
                    elapsed,
                    f"timeout after {max_seconds}s; classification={classification}",
                    classification=classification,
                    blocks_gpu_probe=classification_blocks_gpu_probe(classification),
                    blocks_formal=classification_blocks_formal(classification),
                    killed_by_signal=killed_signal,
                    peak_rss_kb=int(max(rss_before, rss_after)),
                )
            )
    return results


def write_slow_tests_log(results: Sequence[CommandResult], *, repo_root: Path, scope: str, limit: int = 30) -> Path:
    rows = sorted(results, key=lambda item: item.elapsed_seconds, reverse=True)[:limit]
    lines = [
        "# ODCR post-edit slow test/command report",
        f"scope={scope}",
        "rank\tseconds\tstatus\tgroup\tlabel\tcommand",
    ]
    for idx, result in enumerate(rows, start=1):
        lines.append(
            "\t".join(
                (
                    str(idx),
                    f"{result.elapsed_seconds:.3f}",
                    result.status,
                    command_group(result.command),
                    result.command.label,
                    result.command.display(),
                )
            )
        )
    result = get_writer(repo_root).raw_log(
        "post_edit_slow_tests.log",
        "\n".join(lines),
        source="post_edit_runner",
        stage="post_edit",
        validation_result={"scope": scope, "result": "FAIL" if post_edit_results_block_formal(results) else "PASS"},
        outputs={"rows": len(rows)},
    )
    return result.path


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
        print(f"  [{result.status}] {result.command.display()} [{result.classification}]{suffix}")
    semantic_blockers = sum(1 for result in results if result.blocks_formal)
    print(f"Result: {'FAIL' if semantic_blockers else 'PASS'} ({failures} failed, {len(results) - failures} passed, {semantic_blockers} semantic blocker(s))")
    print(json.dumps(post_edit_summary_payload(results), indent=2, sort_keys=True))


def post_edit_summary_payload(results: Sequence[CommandResult]) -> dict[str, object]:
    return {
        "schema_version": "odcr_post_edit_diagnostic/1",
        "result": "FAIL" if post_edit_results_block_formal(results) else "PASS",
        "blocks_gpu_probe": post_edit_results_block_gpu_probe(results),
        "blocks_formal": post_edit_results_block_formal(results),
        "classifications": {
            name: sum(1 for result in results if result.classification == name)
            for name in sorted({result.classification for result in results})
        },
        "commands": [
            {
                "label": result.command.label,
                "group": command_group(result.command),
                "command": result.command.display(),
                "status": result.status,
                "returncode": result.returncode,
                "elapsed_seconds": round(result.elapsed_seconds, 3),
                "classification": result.classification,
                "blocks_gpu_probe": result.blocks_gpu_probe,
                "blocks_formal": result.blocks_formal,
                "killed_by_signal": result.killed_by_signal,
                "peak_rss_kb": result.peak_rss_kb,
                "rerun_status": result.rerun_status,
                "rerun_returncode": result.rerun_returncode,
                "rerun_elapsed_seconds": (
                    round(float(result.rerun_elapsed_seconds), 3)
                    if result.rerun_elapsed_seconds is not None
                    else None
                ),
                "detail": result.detail,
            }
            for result in results
        ],
    }


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
    slow_log = write_slow_tests_log(results, repo_root=repo_root, scope=args.scope)
    print(f"slow_tests_log: {slow_log.relative_to(repo_root).as_posix()}")
    print_summary(results)
    return 1 if post_edit_results_block_formal(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
