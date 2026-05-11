#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from odcr_core.config_resolver import (
    build_preprocess_config,
    load_yaml_config,
    resolve_config,
    write_resolved_config,
)
from odcr_core.config_schema import OneControlConfigError, SourceRecord
from odcr_core.manifests import (
    build_formal_source_table_snapshot,
    build_source_table_snapshot,
    formal_snapshot_view,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/odcr.yaml"
NO_ACCUM_REMOVED_MESSAGE = (
    "grad_accum has been removed in ODCR no-accum architecture; use per_gpu_batch_size "
    "and global_batch_size = per_gpu_batch_size * ddp_world_size."
)


class _RetiredAccumulationAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(NO_ACCUM_REMOVED_MESSAGE)


def _add_retired_accumulation_args(p: argparse.ArgumentParser) -> None:
    for opt in ("--grad-accum", "--gradient-accumulation-steps", "--accumulate-grad-batches"):
        p.add_argument(opt, nargs="?", action=_RetiredAccumulationAction, help=argparse.SUPPRESS)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _command_line() -> str:
    return shlex.join(sys.argv)


def _common_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--set", dest="sets", action="append", default=[])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--daemon", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--verbose", action="store_true", help="display-only: expand console detail")
    p.add_argument("--debug", action="store_true", help="display-only: show raw launcher/child output on console")
    _add_retired_accumulation_args(p)
    return p


def build_parser() -> argparse.ArgumentParser:
    common = _common_parser()
    p = argparse.ArgumentParser(
        prog="odcr",
        description="ODCR one-control entry: CLI --set > configs/odcr.yaml > resolver schema defaults.",
    )
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--set", dest="sets", action="append", default=[])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--daemon", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--verbose", action="store_true", help="display-only: expand console detail")
    p.add_argument("--debug", action="store_true", help="display-only: show raw launcher/child output on console")
    _add_retired_accumulation_args(p)
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("preprocess", parents=[common], help="run preprocess a/b/c")
    pp.add_argument("stage", choices=("a", "b", "c"))

    s3 = sub.add_parser("step3", parents=[common])
    s3.add_argument("--task", type=int, required=True)
    s3.add_argument("--run-id", default="auto")
    s3.add_argument("--mode", choices=("full", "train_only", "eval_only"), default="full")
    s3.add_argument("--profile", dest="profile", default=None)
    s3.add_argument("--expect-profile", dest="expect_profile", default=None)
    s3.add_argument("--cache-check", action="store_true")
    s3.add_argument("--checkpoint-write-preflight", action="store_true")
    s3.add_argument("--expect-cache-hit", action="store_true")
    s3.add_argument("--allow-cold-build", action="store_true")
    s3.add_argument("--expect-num-proc", type=int, default=None)
    s3.add_argument("--accept-eval-only", action="store_true")

    s4 = sub.add_parser("step4", parents=[common])
    s4.add_argument("--task", type=int, required=True)
    s4.add_argument("--from-step3", dest="from_step3", default=None)
    s4.add_argument("--from-step3-run", dest="from_step3_run", default=None)
    s4.add_argument("--run-id", default="auto")
    s4.add_argument("--profile", dest="eval_profile", default=None)
    s4.add_argument("--prepare-cache", action="store_true")
    s4.add_argument("--preflight", action="store_true")
    s4.add_argument("--preflight-mode", choices=("preview", "gpu-shard"), default="preview")
    s4.add_argument("--force-gpu-forward", action="store_true")
    s4.add_argument("--profile-utilization", action="store_true")
    s4.add_argument("--max-samples", type=int, default=None)
    s4.add_argument("--validation-namespace", default=None)
    s4.add_argument("--candidate-config", default=None)

    s5 = sub.add_parser("step5", parents=[common])
    s5.add_argument("--task", type=int, required=True)
    s5.add_argument("--from-step4", dest="from_step4", default=None)
    s5.add_argument("--from-step4-run", dest="from_step4_run", default=None)
    s5.add_argument("--run-id", default="auto")
    s5.add_argument("--profile", dest="eval_profile", default=None)

    ev = sub.add_parser("eval", parents=[common])
    ev.add_argument("--task", type=int, required=True)
    ev.add_argument("--from-step5", default="latest")
    ev.add_argument("--run-id", default="auto")
    ev.add_argument("--profile", dest="eval_profile", default=None)

    pl = sub.add_parser("pipeline", parents=[common])
    pl.add_argument("--task", type=int, required=True)
    pl.add_argument("--from", dest="from_stage", default="preprocess_a")
    pl.add_argument("--to", dest="to_stage", default="eval")
    pl.add_argument("--profile", dest="eval_profile", default=None)

    sh = sub.add_parser("show", parents=[common])
    sh.add_argument(
        "--stage",
        choices=("preprocess_a", "preprocess_b", "preprocess_c", "step3", "step4", "step5", "eval"),
        required=True,
    )
    sh.add_argument("--task", type=int, default=None)
    sh.add_argument("--profile", dest="eval_profile", default=None)

    sub.add_parser("doctor", parents=[common])

    pr = sub.add_parser("promote-upstream", parents=[common])
    pr.add_argument("--stage", choices=("step3", "step4", "step5"), required=True)
    pr.add_argument("--task", type=int, required=True)
    pr.add_argument("--run-id", required=True)

    tl = sub.add_parser("tail", parents=[common])
    tl.add_argument("--stage", choices=("step3", "step4", "step5", "eval"), required=True)
    tl.add_argument("--task", type=int, required=True)
    tl.add_argument("--lines", type=int, default=80)
    tail_log = tl.add_mutually_exclusive_group()
    tail_log.add_argument("--full", action="store_true", help="tail meta/full.log")
    tail_log.add_argument("--errors", action="store_true", help="tail meta/errors.log")
    return p


def _merged_sets(args: argparse.Namespace) -> list[str]:
    return list(getattr(args, "sets", []) or [])


def _config_path(args: argparse.Namespace) -> str:
    return str(getattr(args, "config", None) or DEFAULT_CONFIG)


def _dry_run(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "dry_run", False))


def _print_sources(records: list[SourceRecord]) -> None:
    print("Source table:")
    for record in records:
        if record.value is None:
            continue
        print(f"  {record.key}: {record.source}")


def _print_source_table_payload(payload: dict[str, Any]) -> None:
    print(f"Source table ({payload.get('view', 'verbose')}):")
    for record in payload.get("records") or []:
        if not isinstance(record, dict):
            continue
        key = record.get("key")
        source = record.get("source")
        if source is None:
            continue
        print(f"  {key}: {source}")


def _print_stage_summary(snapshot: dict[str, Any]) -> None:
    print(json.dumps(snapshot, ensure_ascii=False, indent=2, default=str))


def _display_snapshot(snapshot: dict[str, Any], *, verbose: bool) -> dict[str, Any]:
    if verbose:
        return snapshot
    return formal_snapshot_view(snapshot)


def _console_level(args: argparse.Namespace) -> str:
    from odcr_core.logging_meta import console_level_from_flags

    return console_level_from_flags(
        verbose=bool(getattr(args, "verbose", False)),
        debug=bool(getattr(args, "debug", False)),
    )


def _resolve_for_args(args: argparse.Namespace, command: str):
    from_step3 = getattr(args, "from_step3_run", None) or getattr(args, "from_step3", None)
    from_step4 = getattr(args, "from_step4_run", None) or getattr(args, "from_step4", None)
    return resolve_config(
        config_path=_config_path(args),
        command=command,
        task_id=getattr(args, "task", None),
        set_overrides=_merged_sets(args),
        dry_run=_dry_run(args) or command == "show",
        run_id=getattr(args, "run_id", None),
        from_step3=from_step3,
        from_step4=from_step4,
        from_step5=getattr(args, "from_step5", None),
        eval_profile=getattr(args, "eval_profile", None),
        mode=getattr(args, "mode", None),
    )


def _assert_step3_expected_profile(snapshot: dict[str, Any], expected: str | None) -> None:
    expected_text = str(expected or "").strip()
    if not expected_text:
        return
    actual = str(
        (snapshot.get("task") or {}).get("task_profile_id")
        or (snapshot.get("step3_task_profile") or {}).get("profile_id")
        or ""
    ).strip()
    if actual != expected_text:
        raise OneControlConfigError(f"expected {expected_text} but resolved {actual}")


def _step3_expected_profile_arg(args: argparse.Namespace) -> str | None:
    return str(getattr(args, "expect_profile", None) or getattr(args, "profile", None) or "").strip() or None


def _run_resolved(cfg, snapshot: dict[str, Any], *, dry_run: bool, console_level: str = "summary") -> None:
    write_resolved_config(cfg, snapshot, dry_run=dry_run)
    if dry_run:
        _print_stage_summary(_display_snapshot(snapshot, verbose=console_level != "summary"))
        return

    from odcr_core.logging_meta import (
        append_error_log,
        emit_console_lines,
        initialize_run_log_files,
        print_pre_run_banner,
        console_summary_lines,
    )
    from odcr_core.manifests import write_run_summary_for_config
    from odcr_core.runners import run_eval, run_eval_rerank, run_step3, run_step4, run_step5
    from odcr_core.validation import validate_resolved_config

    started_at = _utc_now()
    started_monotonic = time.monotonic()
    command_line = _command_line()
    initialize_run_log_files(
        cfg,
        snapshot,
        command_line=command_line,
        started_at=started_at,
        console_level=console_level,
    )
    write_run_summary_for_config(
        cfg,
        status="running",
        started_at=started_at,
        command=command_line,
        validation_status="pending",
    )
    print_pre_run_banner(cfg.command, cfg, console_level=console_level, started_at=started_at)
    try:
        validate_resolved_config(cfg)
        if cfg.command == "step3":
            run_step3(cfg, console_level=console_level)
        elif cfg.command == "step4":
            run_step4(cfg, console_level=console_level)
        elif cfg.command == "step5":
            run_step5(cfg, console_level=console_level)
        elif cfg.command == "eval-rerank":
            run_eval_rerank(cfg, console_level=console_level)
        elif cfg.command == "eval":
            run_eval(cfg, console_level=console_level)
        else:
            raise OneControlConfigError(f"unknown executable stage: {cfg.command}")
    except Exception as exc:
        finished_at = _utc_now()
        elapsed = time.monotonic() - started_monotonic
        tb = traceback.format_exc()
        append_error_log(
            cfg,
            [
                f"[ODCR exception] finished_at={finished_at}",
                tb.rstrip(),
            ],
        )
        write_run_summary_for_config(
            cfg,
            status="failed",
            started_at=started_at,
            finished_at=finished_at,
            command=command_line,
            latest_error=str(exc),
            validation_status="failed",
        )
        emit_console_lines(
            cfg,
            console_summary_lines(
                cfg,
                status="failed",
                started_at=started_at,
                finished_at=finished_at,
                elapsed_sec=elapsed,
                error=str(exc),
            ),
        )
        raise
    finished_at = _utc_now()
    elapsed = time.monotonic() - started_monotonic
    write_run_summary_for_config(
        cfg,
        status="ok",
        started_at=started_at,
        finished_at=finished_at,
        command=command_line,
        validation_status="ok",
    )
    emit_console_lines(
        cfg,
        console_summary_lines(
            cfg,
            status="ok",
            started_at=started_at,
            finished_at=finished_at,
            elapsed_sec=elapsed,
        ),
    )


def cmd_preprocess(args: argparse.Namespace) -> None:
    config = build_preprocess_config(
        config_path=_config_path(args),
        stage_letter=args.stage,
        set_overrides=_merged_sets(args),
        dry_run=_dry_run(args),
    )
    if _dry_run(args):
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2, default=str))
        return
    from odcr_core.preprocess_runtime import PreprocessRuntime

    PreprocessRuntime(config).run()


def cmd_stage(args: argparse.Namespace, command: str) -> None:
    if command == "step4" and getattr(args, "from_step3", None) and getattr(args, "from_step3_run", None):
        raise OneControlConfigError("use only one of --from-step3 or --from-step3-run")
    if command == "step4" and bool(getattr(args, "prepare_cache", False)) and bool(getattr(args, "preflight", False)):
        raise OneControlConfigError("use only one of --prepare-cache or --preflight")
    if command == "step5" and getattr(args, "from_step4", None) and getattr(args, "from_step4_run", None):
        raise OneControlConfigError("use only one of --from-step4 or --from-step4-run")
    if command == "step3" and bool(getattr(args, "accept_eval_only", False)):
        run_id = str(getattr(args, "run_id", "") or "").strip()
        if run_id in {"", "auto"}:
            raise OneControlConfigError("--accept-eval-only requires an explicit --run-id")
        from odcr_core.step3_eval_handoff import (
            Step3EvalHandoffError,
            accept_step3_eval_handoff,
        )

        try:
            result = accept_step3_eval_handoff(
                repo_root=REPO_ROOT,
                task_id=int(args.task),
                run_id=run_id,
                dry_run=_dry_run(args),
                require_test=True,
            )
        except Step3EvalHandoffError as exc:
            raise OneControlConfigError(str(exc)) from exc
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
        return
    resolve_args = args
    if (
        command == "step3"
        and (bool(getattr(args, "cache_check", False)) or bool(getattr(args, "checkpoint_write_preflight", False)))
    ) or (
        command == "step4"
        and (bool(getattr(args, "prepare_cache", False)) or bool(getattr(args, "preflight", False)))
    ):
        resolve_args = argparse.Namespace(**vars(args))
        resolve_args.dry_run = True
    cfg, sources, snapshot = _resolve_for_args(resolve_args, command)
    if command == "step3":
        _assert_step3_expected_profile(snapshot, _step3_expected_profile_arg(args))
        if bool(getattr(args, "cache_check", False)):
            from tools.odcr_step3_cache_check import run_cache_check

            result = run_cache_check(
                task_id=int(args.task),
                expected_profile=_step3_expected_profile_arg(args),
                expect_cache_hit=bool(getattr(args, "expect_cache_hit", False)),
                allow_cold_build=bool(getattr(args, "allow_cold_build", False)),
                expect_num_proc=getattr(args, "expect_num_proc", None),
                resolved_snapshot=snapshot,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
            return
        if bool(getattr(args, "checkpoint_write_preflight", False)):
            from tools.odcr_step3_checkpoint_write_preflight import run_preflight

            result = run_preflight(task_id=int(args.task))
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
            return
    if command == "show":
        verbose = bool(getattr(args, "verbose", False) or getattr(args, "debug", False))
        _print_stage_summary(_display_snapshot(snapshot, verbose=verbose))
        if verbose:
            _print_source_table_payload(build_source_table_snapshot(snapshot))
        else:
            _print_source_table_payload(build_formal_source_table_snapshot(snapshot))
        return
    if command == "step4" and bool(getattr(args, "prepare_cache", False)):
        from odcr_core.step4_runtime import prepare_step4_encoded_cache

        result = prepare_step4_encoded_cache(
            cfg,
            dry_run=_dry_run(args),
            build_allowed=not _dry_run(args),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
        return
    if command == "step4" and bool(getattr(args, "preflight", False)):
        from odcr_core.step4_runtime import run_step4_bounded_preflight

        result = run_step4_bounded_preflight(
            cfg,
            max_samples=getattr(args, "max_samples", None),
            validation_namespace=getattr(args, "validation_namespace", None),
            preflight_mode=getattr(args, "preflight_mode", "preview"),
            force_gpu_forward=bool(getattr(args, "force_gpu_forward", False)),
            profile_utilization=bool(getattr(args, "profile_utilization", False)),
            candidate_config=getattr(args, "candidate_config", None),
            dry_run=_dry_run(args),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
        return
    _run_resolved(cfg, snapshot, dry_run=_dry_run(args), console_level=_console_level(args))


def cmd_pipeline(args: argparse.Namespace) -> None:
    stages = ["preprocess_a", "preprocess_b", "preprocess_c", "step3", "step4", "step5", "eval"]
    if args.from_stage not in stages or args.to_stage not in stages:
        raise OneControlConfigError(f"pipeline stages must be in {stages}")
    start = stages.index(args.from_stage)
    end = stages.index(args.to_stage)
    if start > end:
        raise OneControlConfigError("--from must not come after --to")
    selected = stages[start : end + 1]
    if _dry_run(args):
        print(json.dumps({"pipeline": selected, "task": args.task}, ensure_ascii=False, indent=2))
        return
    step3_run = None
    step4_run = None
    step5_run = None
    for stage in selected:
        if stage.startswith("preprocess_"):
            ns = argparse.Namespace(**vars(args))
            ns.stage = stage[-1]
            cmd_preprocess(ns)
        elif stage == "step3":
            ns = argparse.Namespace(**vars(args))
            ns.run_id = "auto"
            ns.mode = "full"
            cfg, _, snapshot = _resolve_for_args(ns, "step3")
            step3_run = cfg.run_name
            _run_resolved(cfg, snapshot, dry_run=False, console_level=_console_level(args))
        elif stage == "step4":
            ns = argparse.Namespace(**vars(args))
            ns.from_step3 = step3_run or "latest"
            ns.run_id = "auto"
            cfg, _, snapshot = _resolve_for_args(ns, "step4")
            step4_run = cfg.step4_run
            _run_resolved(cfg, snapshot, dry_run=False, console_level=_console_level(args))
        elif stage == "step5":
            ns = argparse.Namespace(**vars(args))
            ns.from_step4 = step4_run or "latest"
            ns.run_id = "auto"
            cfg, _, snapshot = _resolve_for_args(ns, "step5")
            step5_run = cfg.step5_run
            _run_resolved(cfg, snapshot, dry_run=False, console_level=_console_level(args))
        elif stage == "eval":
            ns = argparse.Namespace(**vars(args))
            ns.from_step5 = step5_run or "latest"
            ns.run_id = "auto"
            cfg, _, snapshot = _resolve_for_args(ns, "eval")
            _run_resolved(cfg, snapshot, dry_run=False, console_level=_console_level(args))


def cmd_doctor(args: argparse.Namespace) -> None:
    cfg_path = _config_path(args)
    raw = load_yaml_config(cfg_path)
    top = list(raw.keys())
    checks: list[str] = []
    checks.append(f"config top-level blocks: {', '.join(top)}")
    doctor_snapshot: dict[str, Any] | None = None
    step3_doctor_snapshot: dict[str, Any] | None = None
    pending_upstreams: list[str] = []
    for command in ("step3", "step4", "step5", "eval"):
        ns = argparse.Namespace(**vars(args))
        ns.task = int(raw.get("project", {}).get("default_task", 2))
        ns.run_id = "auto"
        ns.from_step3 = None
        ns.from_step3_run = None
        ns.from_step4 = None
        ns.from_step4_run = None
        ns.from_step5 = "latest"
        ns.eval_profile = None
        ns.mode = "full"
        try:
            _cfg, _sources, _snapshot = _resolve_for_args(ns, command)
        except OneControlConfigError as exc:
            if command in {"step5", "eval"}:
                pending_upstreams.append(f"{command}: {exc}")
                continue
            raise
        if command == "step3":
            step3_doctor_snapshot = _snapshot
        if command == "step5":
            doctor_snapshot = _snapshot
    checks.append("step3/step4 resolve checks passed through unified upstream resolver")
    if pending_upstreams:
        checks.append("step5/eval resolver fail-fast pending upstream: " + " | ".join(pending_upstreams))
    if step3_doctor_snapshot:
        hw = step3_doctor_snapshot.get("hardware") or {}
        train = step3_doctor_snapshot.get("train") or {}
        sources = step3_doctor_snapshot.get("field_sources") or {}
        print("Step3 runtime controls:")
        print(f"  hardware_profile: {hw.get('profile')} (source: {sources.get('hardware')})")
        print(
            "  max_parallel_cpu: "
            f"{hw.get('max_parallel_cpu')} (source: {sources.get('hardware.max_parallel_cpu')})"
        )
        print(f"  num_proc: {hw.get('num_proc')} (source: {sources.get('hardware.num_proc')})")
        worker_budget = hw.get("worker_budget_formula") or {}
        if worker_budget:
            print(
                "  worker_formula: "
                "train=(workers_per_rank*ddp_world_size)+reserved_cpu="
                f"{worker_budget.get('train_active_processes')} <= max_parallel_cpu "
                f"{worker_budget.get('max_parallel_cpu')} "
                f"(reserved_cpu={worker_budget.get('reserved_cpu')})"
            )
            print(
                "  tokenization_formula: "
                f"num_proc+reserved_cpu={worker_budget.get('tokenization_active_processes')} "
                f"<= max_parallel_cpu {worker_budget.get('max_parallel_cpu')} "
                f"(reserved_cpu={worker_budget.get('reserved_cpu')})"
            )
        print(
            "  train_precision: "
            f"{train.get('precision')} (source: {sources.get('train_precision')})"
        )
        print(
            "  optimizer: "
            f"{(step3_doctor_snapshot.get('step3_optimizer') or {}).get('name')} "
            f"(source: {sources.get('step3_optimizer')})"
        )
        print(
            "  tokenizer/evidence: "
            f"{(step3_doctor_snapshot.get('step3_tokenizer') or {}).get('max_length')}/"
            f"{(step3_doctor_snapshot.get('step3_evidence') or {}).get('max_evidence_length')} "
            f"(source: {sources.get('step3_tokenizer')} / {sources.get('step3_evidence')})"
        )
        print(
            "  scheduler: "
            f"{(step3_doctor_snapshot.get('step3_scheduler') or {}).get('name')} "
            f"warmup_ratio={(step3_doctor_snapshot.get('step3_scheduler') or {}).get('warmup_ratio')} "
            f"min_lr_ratio={(step3_doctor_snapshot.get('step3_scheduler') or {}).get('min_lr_ratio')}"
        )
        print(
            "  grad_norm/valid_batch: "
            f"max_grad_norm={train.get('max_grad_norm')} "
            f"valid={step3_doctor_snapshot.get('step3_eval')}"
        )
        print(
            "  h2d: "
            f"pin_memory={hw.get('pin_memory')} persistent_workers={hw.get('persistent_workers')} "
            f"non_blocking_h2d={hw.get('non_blocking_h2d')}"
        )
        step3_ddp = step3_doctor_snapshot.get("step3_ddp") or {}
        print("Step3 DDP policy:")
        print(
            "  find_unused_parameters: "
            f"{step3_ddp.get('ddp_find_unused_parameters')} "
            "(source: step3.ddp.find_unused_parameters)"
        )
        print(
            "  static_graph: "
            f"{step3_ddp.get('ddp_static_graph')} "
            "(source: step3.ddp.static_graph)"
        )
        print(
            "  graph_safety_preflight: "
            f"{step3_ddp.get('ddp_graph_safety_preflight')} "
            "(source: step3.ddp.graph_safety_preflight)"
        )
        checks.append("step3 max_parallel_cpu, train_precision, and ddp policy resolve from configs/odcr.yaml")
    if doctor_snapshot:
        roots = doctor_snapshot.get("roots") or {}
        models = doctor_snapshot.get("models") or {}
        embed = doctor_snapshot.get("embed_dim") or {}
        step5_ddp = doctor_snapshot.get("step5_ddp") or {}
        print("One-Control roots/models/embed_dim:")
        print(f"  runs_dir: {roots.get('runs_dir')} (source: project.run_root)")
        print(f"  cache_dir: {roots.get('cache_dir')} (source: project.cache_dir)")
        print(f"  data_dir: {roots.get('data_dir')} (source: project.data_dir)")
        print(f"  merged_dir: {roots.get('merged_dir')} (source: project.merged_dir)")
        print(f"  models_dir: {roots.get('models_dir')} (source: env.models_dir)")
        print(f"  step5_text_model: {models.get('step5_text_model')} (source: env.step5_text_model)")
        print(f"  sentence_embed_model: {models.get('sentence_embed_model')} (source: env.sentence_embed_model)")
        print(f"  embed_dim: {embed.get('value')} (source: env.embed_dim)")
        print(f"  offline: {doctor_snapshot.get('offline', {}).get('value')} (source: env.offline)")
        print(
            "  local_files_only: "
            f"{doctor_snapshot.get('local_files_only', {}).get('value')} (source: env.local_files_only)"
        )
        print("Step5 DDP policy:")
        print(
            "  find_unused_parameters: "
            f"{step5_ddp.get('ddp_find_unused_parameters')} "
            "(source: step5.ddp.find_unused_parameters)"
        )
        print(
            "  find_unused_false_preflight: "
            f"{step5_ddp.get('ddp_find_unused_false_preflight')} "
            "(source: step5.ddp.find_unused_false_preflight)"
        )
        checks.append("roots/models/cache/offline/embed_dim resolve from configs/odcr.yaml")
        checks.append("step5 ddp find_unused policy resolve from configs/odcr.yaml")
    print("Preprocess CPU/GPU pipeline controls:")
    for letter in ("b", "c"):
        pp_cfg = build_preprocess_config(
            config_path=cfg_path,
            stage_letter=letter,
            set_overrides=_merged_sets(args),
            dry_run=True,
        )
        controls = {
            "workers": pp_cfg.runtime.workers,
            "gpu_ids": list(pp_cfg.hardware.gpu_ids),
            "tokenizer_parallelism_enabled": pp_cfg.tokenizer_parallelism_enabled,
            "tokenizer_threads_per_worker": pp_cfg.tokenizer_threads_per_worker,
            "tokenizer_total_threads": pp_cfg.tokenizer_total_threads,
            "prefetch_batches": pp_cfg.prefetch_batches,
            "pin_memory": pp_cfg.pin_memory,
            "non_blocking_h2d": pp_cfg.non_blocking_h2d,
            "async_prefetch_enabled": pp_cfg.async_prefetch_enabled,
            "cpu_cores_reserved": pp_cfg.cpu_cores_reserved,
            "cpu_cores_available": pp_cfg.cpu_cores_available,
        }
        if letter == "b":
            controls["token_aware_batching_enabled"] = pp_cfg.token_aware_batching_enabled
            controls["max_tokens_per_gpu_batch"] = pp_cfg.max_tokens_per_gpu_batch
        else:
            controls["scheduling_policy"] = pp_cfg.scheduling_policy
        print(f"  preprocess_{letter}: {json.dumps(controls, sort_keys=True)}")
    checks.append("preprocess_b/c CPU tokenizer, prefetch, H2D, and scheduling controls resolve from configs/odcr.yaml")
    retired = REPO_ROOT / "scripts" / "run_stage.sh"
    if retired.exists():
        raise OneControlConfigError("scripts/run_stage.sh must be absent; use ./odcr or python code/odcr.py")
    checks.append("legacy scripts/run_stage.sh absent")
    main_files = [REPO_ROOT / "odcr", REPO_ROOT / "code" / "odcr.py", REPO_ROOT / "code" / "odcr_core" / "config_resolver.py"]
    offenders = []
    legacy_marker = "presets" + "/"
    for path in main_files:
        if path.is_file() and legacy_marker in path.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    if offenders:
        raise OneControlConfigError(f"main control files still mention legacy preset paths: {offenders}")
    checks.append("main control files do not read legacy preset paths")
    from tools.check_one_control_guardrails import format_report, run_checks

    guardrail_report = run_checks(repo_root=REPO_ROOT, strict=True)
    print(format_report(guardrail_report))
    if not guardrail_report.ok or guardrail_report.warnings:
        raise OneControlConfigError("one-control guardrail lint failed or warned; see report above")
    checks.append("one-control guardrail passed")
    checks.append("no legacy preset mainline")
    checks.append("no scattered config")
    checks.append("no parameter drift")
    print("ODCR doctor: OK")
    for item in checks:
        print(f"  - {item}")


def _repo_path(raw: object, *, context: str) -> Path:
    value = str(raw or "").strip()
    if not value:
        raise OneControlConfigError(f"{context} is empty")
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _read_json_file(path: Path, *, context: str) -> dict[str, Any]:
    if not path.is_file():
        raise OneControlConfigError(f"{context} not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OneControlConfigError(f"{context} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise OneControlConfigError(f"{context} must be a JSON object: {path}")
    return payload


def _resolve_tail_log_path(args: argparse.Namespace) -> Path:
    parent = REPO_ROOT / "runs" / args.stage / f"task{int(args.task)}"
    latest = parent / "latest.json"
    if not latest.is_file():
        raise OneControlConfigError(
            f"missing {latest}; run the requested stage first so latest.json points to meta/run_summary.json"
        )

    latest_payload = _read_json_file(latest, context="latest.json")
    summary_path = _repo_path(latest_payload.get("latest_summary_path"), context="latest_summary_path")
    if not summary_path.is_file():
        raise OneControlConfigError(f"latest.json pointer is damaged; run_summary.json not found: {summary_path}")

    summary = _read_json_file(summary_path, context="run_summary.json")
    if bool(getattr(args, "errors", False)):
        key, filename = "errors_log_path", "errors.log"
    elif bool(getattr(args, "full", False)):
        key, filename = "full_log_path", "full.log"
    elif bool(getattr(args, "debug", False)):
        key, filename = "debug_log_path", "debug.log"
    else:
        key, filename = "console_log_path", "console.log"

    log_path = _repo_path(summary.get(key), context=f"run_summary.json {key}")
    meta = summary_path.parent.resolve()
    if log_path.parent.resolve() != meta or log_path.name != filename:
        raise OneControlConfigError(
            f"run_summary.json {key} must resolve to meta/{filename}; got {log_path}"
        )
    if not log_path.is_file():
        raise OneControlConfigError(f"new log layout did not generate target file: {log_path}")
    return log_path


def cmd_tail(args: argparse.Namespace) -> None:
    log_path = _resolve_tail_log_path(args)
    if _dry_run(args):
        print(f"ODCR tail dry-run: {log_path}")
        return
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"==> {log_path} <==")
    for line in lines[-max(1, int(args.lines)) :]:
        print(line)


def cmd_promote_upstream(args: argparse.Namespace) -> None:
    from odcr_core.stage_promotion import StagePromotionError, promote_upstream

    try:
        result = promote_upstream(
            repo_root=REPO_ROOT,
            stage=str(args.stage),
            task=int(args.task),
            run_id=str(args.run_id),
            dry_run=_dry_run(args),
        )
    except StagePromotionError as exc:
        raise OneControlConfigError(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def maybe_daemonize(args: argparse.Namespace) -> None:
    if not getattr(args, "daemon", False):
        return
    print(
        "ODCR --daemon is retired: run foreground through ./odcr or python code/odcr.py "
        "so logs stay under runs/<stage>/<unit>/<run_id>/meta.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    os.chdir(REPO_ROOT)
    maybe_daemonize(args)
    try:
        if args.command == "preprocess":
            cmd_preprocess(args)
        elif args.command in ("step3", "step4", "step5", "eval"):
            cmd_stage(args, args.command)
        elif args.command == "pipeline":
            cmd_pipeline(args)
        elif args.command == "show":
            if args.stage.startswith("preprocess_"):
                letter = args.stage[-1]
                config = build_preprocess_config(
                    config_path=_config_path(args),
                    stage_letter=letter,
                    set_overrides=_merged_sets(args),
                    dry_run=True,
                )
                _print_stage_summary(config.to_dict())
                from odcr_core.preprocess_runtime import PreprocessRuntime

                runtime = PreprocessRuntime(config)
                print("Source table:")
                for record in runtime._source_table_payload()["records"]:
                    if record.get("value") is None:
                        continue
                    print(f"  {record['key']}: {record['source']}")
                return 0
            cfg, sources, snapshot = resolve_config(
                config_path=_config_path(args),
                command=args.stage,
                task_id=args.task,
                set_overrides=_merged_sets(args),
                dry_run=True,
                eval_profile=getattr(args, "eval_profile", None),
                mode="full",
            )
            _ = cfg
            verbose = bool(getattr(args, "verbose", False) or getattr(args, "debug", False))
            _print_stage_summary(_display_snapshot(snapshot, verbose=verbose))
            if verbose:
                _print_source_table_payload(build_source_table_snapshot(snapshot))
            else:
                _print_source_table_payload(build_formal_source_table_snapshot(snapshot))
        elif args.command == "doctor":
            cmd_doctor(args)
        elif args.command == "promote-upstream":
            cmd_promote_upstream(args)
        elif args.command == "tail":
            cmd_tail(args)
        else:
            parser.error(f"unknown command: {args.command}")
    except OneControlConfigError as exc:
        print(f"ODCR config error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
