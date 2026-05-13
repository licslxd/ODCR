"""Step3 torchrun 入口：argparse 与 train/eval 分发（重型逻辑在 step3_train_core）。"""
from __future__ import annotations

import argparse
import os
import sys

from executors import bootstrap
from executors import ddp_utils
from executors.startup_config_check import print_startup_config_check

_STEP3_RUNNER = "step3 runner（torchrun 内部入口）"


def print_step3_root_help() -> None:
    """仅标准库 + argparse，用于 ``step3_entry.py --help`` 快路径。"""
    epilog = (
        "用户日常（仓库根）: python code/odcr.py step3 …\n"
        "子命令完整参数: 在 code/ 下执行 executors/step3_entry.py train --help / eval --help。"
    )
    p = argparse.ArgumentParser(
        prog="step3-runner",
        description=(
            "Step3 shared/specific 结构化解耦 — torchrun 内部入口（executors/step3_entry.py）。"
            "请优先: python code/odcr.py step3 …"
        ),
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="DDP 训练（由 odcr.py step3 编排）")
    sub.add_parser("eval", help="DDP 评测")
    p.print_help()


def run_step3_cli() -> None:
    bootstrap.reject_legacy_gpus_argv(
        sys.argv,
        executor_label=_STEP3_RUNNER,
        torchrun_hint=(
            "推荐: python code/odcr.py step3 --task N …\n"
            "须自行 torchrun 时见 docs/ODCR_Scripts_and_Runtime_Guide.md 附录。\n"
        ),
    )
    from executors.step3_train_core import (
        _add_eval_args,
        _add_train_args,
        _dispatch_eval,
        _run_train_ddp,
    )

    epilog = (
        "用户日常（仓库根）: python code/odcr.py step3 …\n"
        "本入口仅在被 odcr.py / sh 以 torchrun 调用时使用。"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Step3 shared/specific 结构化解耦 — torchrun 内部入口。"
            "请优先: python code/odcr.py step3 …"
        ),
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    train_p = sub.add_parser("train", help="DDP 训练（由 odcr.py step3 编排）")
    _add_train_args(train_p)
    eval_p = sub.add_parser("eval", help="DDP 评测（内部可由 sh 或高级场景调用）")
    _add_eval_args(eval_p)
    args = parser.parse_args()
    print_startup_config_check(stage="step3", command=str(args.command))
    if args.command == "train":
        ddp_utils.exit_if_not_torchrun(
            executor_label=_STEP3_RUNNER,
            examples=(
                "推荐: python code/odcr.py step3 --task N …\n"
                "附录: 高级 torchrun 排障见 docs/ODCR_Scripts_and_Runtime_Guide.md。\n"
            ),
        )
        if os.environ.get("RANK", "0") == "0":
            print(
                "[step3 runner] train — 用户入口: python code/odcr.py step3 …",
                flush=True,
            )
        _run_train_ddp(args)
    elif args.command == "eval":
        if os.environ.get("RANK", "0") == "0":
            print(
                "[step3 runner] eval — 用户入口: python code/odcr.py step3 --eval-only …",
                flush=True,
            )
        _dispatch_eval(args)
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    run_step3_cli()
