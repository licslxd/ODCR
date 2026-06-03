"""CLI helpers for `./odcr ablation ...`."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from odcr_core.ablation.binding import load_ablation_binding
from odcr_core.ablation.manifest import validate_all_manifests
from odcr_core.ablation.registry import load_config_override, validate_all
from odcr_core.ablation.schemas import validate_schema_files
from odcr_core.ablation.snapshots import build_result_snapshot, write_result_snapshot_outputs
from odcr_core.ablation.variants import build_ablation_dry_run_plan, build_ablation_show


def validate_ablation_infra(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    return {
        "schema_version": "odcr_ablation_validate/1",
        "status": "pass",
        "registry_and_overrides": validate_all(root),
        "schemas": validate_schema_files(root),
        "manifests": validate_all_manifests(root),
    }


def _latest_path(root: Path, task: int) -> Path:
    return root / "runs" / "step5" / f"task{int(task)}" / "latest.json"


def _sha256_if_present(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_cuda_context() -> None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError(f"formal ablation train/eval requires torch CUDA validation: {exc}") from exc
    if not bool(torch.cuda.is_available()):
        raise RuntimeError(
            "Current tmux does not expose CUDA. Please manually run `odcr-enter-gpu <JOBID>` "
            "in this same tmux to enter the GPU node, then rerun the probe."
        )


def _step4_source_from_override(root: Path, task: int, variant: str) -> str:
    override = load_config_override(root, int(task), str(variant))
    raw = str(override.get("expected_step4_handoff_source") or "latest").strip()
    if raw.endswith("/latest.json") or raw == "latest.json":
        return "latest"
    if raw.startswith("runs/step4/"):
        return Path(raw).name
    return raw or "latest"


def _run_child(root: Path, argv: list[str], *, dry_run: bool, task: int) -> dict[str, Any]:
    if not dry_run:
        _require_cuda_context()
    latest = _latest_path(root, int(task))
    before = _sha256_if_present(latest)
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    result = subprocess.run(argv, cwd=root, env=env)
    after = _sha256_if_present(latest)
    payload = {
        "command": argv,
        "returncode": int(result.returncode),
        "latest_path": latest.relative_to(root).as_posix(),
        "latest_before_sha256": before,
        "latest_after_sha256": after,
        "latest_changed": before != after,
    }
    if payload["latest_changed"]:
        raise RuntimeError(f"official latest changed during ablation command: {payload}")
    if result.returncode != 0:
        raise RuntimeError(f"ablation child command failed with returncode={result.returncode}: {' '.join(argv)}")
    return payload


def _append_sets(argv: list[str], extra_sets: list[str] | None) -> None:
    for item in extra_sets or []:
        argv.extend(["--set", str(item)])


def _cmd_train(
    root: Path,
    *,
    task: int,
    variant: str,
    dry_run: bool,
    extra_sets: list[str] | None = None,
) -> dict[str, Any]:
    binding = load_ablation_binding(root, task=int(task), variant=str(variant))
    step4_source = _step4_source_from_override(root, int(task), str(variant))
    argv = [
        sys.executable,
        "code/odcr.py",
        "step5",
        "--task",
        str(int(task)),
        "--from-step4",
        step4_source,
        "--run-id",
        binding.run_id,
        "--profile",
        "paper_greedy_25",
        "--train-only",
    ]
    if dry_run:
        argv.append("--dry-run")
    _append_sets(argv, extra_sets)
    result = _run_child(root, argv, dry_run=dry_run, task=int(task))
    return {
        "schema_version": "odcr_ablation_formal_train/1",
        "status": "pass",
        "task": int(task),
        "variant": str(variant),
        "run_namespace": binding.run_namespace,
        "dry_run": bool(dry_run),
        "child": result,
    }


def _cmd_eval(
    root: Path,
    *,
    task: int,
    variant: str,
    split: str,
    dry_run: bool,
    extra_sets: list[str] | None = None,
) -> dict[str, Any]:
    binding = load_ablation_binding(root, task=int(task), variant=str(variant))
    split_s = str(split or "").strip()
    if split_s not in {"valid", "test"}:
        raise ValueError(f"unsupported ablation eval split: {split!r}")
    argv = [
        sys.executable,
        "code/odcr.py",
        "eval",
        "--task",
        str(int(task)),
        "--from-step5",
        binding.run_id,
        "--profile",
        "paper_greedy_25",
        "--replay-step5-run-config",
        "--set",
        f"eval.split={split_s}",
    ]
    if dry_run:
        argv.append("--dry-run")
    _append_sets(argv, extra_sets)
    run_summary = root / "runs" / "step5" / f"task{int(task)}" / binding.run_id / "meta" / "run_summary.json"
    if dry_run and not run_summary.is_file():
        latest = _latest_path(root, int(task))
        latest_hash = _sha256_if_present(latest)
        return {
            "schema_version": "odcr_ablation_formal_eval/1",
            "status": "pass",
            "task": int(task),
            "variant": str(variant),
            "split": split_s,
            "run_namespace": binding.run_namespace,
            "dry_run": True,
            "child": {
                "command": argv,
                "returncode": None,
                "latest_path": latest.relative_to(root).as_posix(),
                "latest_before_sha256": latest_hash,
                "latest_after_sha256": latest_hash,
                "latest_changed": False,
                "planned_only": True,
                "reason": "step5 run_summary is created by formal train; eval dry-run remains non-executing before train",
                "required_before_formal_eval": run_summary.relative_to(root).as_posix(),
            },
        }
    result = _run_child(root, argv, dry_run=dry_run, task=int(task))
    return {
        "schema_version": "odcr_ablation_formal_eval/1",
        "status": "pass",
        "task": int(task),
        "variant": str(variant),
        "split": split_s,
        "run_namespace": binding.run_namespace,
        "dry_run": bool(dry_run),
        "child": result,
    }


def _cmd_run(
    root: Path,
    *,
    task: int,
    variant: str,
    eval_splits: str,
    dry_run: bool,
    extra_sets: list[str] | None = None,
) -> dict[str, Any]:
    splits = [item.strip() for item in str(eval_splits or "valid,test").split(",") if item.strip()]
    if not splits:
        splits = ["valid", "test"]
    if any(item not in {"valid", "test"} for item in splits):
        raise ValueError(f"unsupported ablation eval split list: {eval_splits!r}")
    train = _cmd_train(root, task=int(task), variant=str(variant), dry_run=bool(dry_run), extra_sets=extra_sets)
    evals = [
        _cmd_eval(root, task=int(task), variant=str(variant), split=split, dry_run=bool(dry_run), extra_sets=extra_sets)
        for split in splits
    ]
    return {
        "schema_version": "odcr_ablation_formal_run/1",
        "status": "pass",
        "task": int(task),
        "variant": str(variant),
        "dry_run": bool(dry_run),
        "train": train,
        "eval": evals,
    }


def cmd_ablation(args: argparse.Namespace, *, repo_root: str | Path) -> None:
    root = Path(repo_root).expanduser().resolve()
    action = str(getattr(args, "ablation_action", "") or "").strip()
    extra_sets = list(getattr(args, "sets", []) or [])
    if action == "show":
        payload = build_ablation_show(root, task=int(args.task), variant=str(args.variant))
    elif action == "validate":
        task_arg = getattr(args, "task", None)
        variant_arg = getattr(args, "variant", None)
        if task_arg is None and not variant_arg:
            payload = validate_ablation_infra(root)
        elif task_arg is None or not variant_arg:
            raise ValueError("ablation validate requires both --task and --variant, or neither")
        else:
            payload = {
                "schema_version": "odcr_ablation_validate_one/1",
                "status": "pass",
                "show": build_ablation_show(root, task=int(task_arg), variant=str(variant_arg)),
                "dry_run_plan": build_ablation_dry_run_plan(root, task=int(task_arg), variant=str(variant_arg)),
            }
    elif action == "dry-run":
        payload = build_ablation_dry_run_plan(root, task=int(args.task), variant=str(args.variant))
    elif action == "snapshot":
        if bool(getattr(args, "write", False)):
            payload = write_result_snapshot_outputs(root)
        else:
            payload = build_result_snapshot(root, task=int(args.task), variant=str(args.variant))
    elif action == "probe":
        raise RuntimeError("Old Step5 ablation probe code has been deleted with the generator-first path.")
    elif action == "train":
        payload = _cmd_train(
            root,
            task=int(args.task),
            variant=str(args.variant),
            dry_run=bool(getattr(args, "dry_run", False)),
            extra_sets=extra_sets,
        )
    elif action == "eval":
        payload = _cmd_eval(
            root,
            task=int(args.task),
            variant=str(args.variant),
            split=str(args.split),
            dry_run=bool(getattr(args, "dry_run", False)),
            extra_sets=extra_sets,
        )
    elif action == "run":
        payload = _cmd_run(
            root,
            task=int(args.task),
            variant=str(args.variant),
            eval_splits=str(getattr(args, "eval", "valid,test") or "valid,test"),
            dry_run=bool(getattr(args, "dry_run", False)),
            extra_sets=extra_sets,
        )
    else:
        raise ValueError(f"unknown ablation action: {action}")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def build_standalone_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m odcr_core.ablation.cli")
    sub = parser.add_subparsers(dest="ablation_action", required=True)
    for name in ("show", "dry-run"):
        item = sub.add_parser(name)
        item.add_argument("--task", type=int, required=True, choices=(7, 8))
        item.add_argument("--variant", required=True, choices=("wo_rcr", "wo_cf", "wo_ccv_fca"))
    validate = sub.add_parser("validate")
    validate.add_argument("--task", type=int, choices=(7, 8), default=None)
    validate.add_argument("--variant", choices=("wo_rcr", "wo_cf", "wo_ccv_fca"), default=None)
    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--task", type=int, required=True, choices=(7, 8))
    snapshot.add_argument("--variant", required=True, choices=("wo_rcr", "wo_cf", "wo_ccv_fca"))
    snapshot.add_argument("--write", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("--task", type=int, required=True, choices=(7, 8))
    train.add_argument("--variant", required=True, choices=("wo_rcr", "wo_cf", "wo_ccv_fca"))
    train.add_argument("--dry-run", action="store_true")
    ev = sub.add_parser("eval")
    ev.add_argument("--task", type=int, required=True, choices=(7, 8))
    ev.add_argument("--variant", required=True, choices=("wo_rcr", "wo_cf", "wo_ccv_fca"))
    ev.add_argument("--split", required=True, choices=("valid", "test"))
    ev.add_argument("--dry-run", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("--task", type=int, required=True, choices=(7, 8))
    run.add_argument("--variant", required=True, choices=("wo_rcr", "wo_cf", "wo_ccv_fca"))
    run.add_argument("--eval", default="valid,test")
    run.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_standalone_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[3]
    cmd_ablation(args, repo_root=repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
