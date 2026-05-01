"""Step5 INTERNAL EXECUTOR：argparse 与 _run_ddp 调用（逻辑在 step5_engine）。"""
from __future__ import annotations

import argparse
import os
import sys

from executors import bootstrap
from executors import ddp_utils
from executors.startup_config_check import print_startup_config_check

_STEP5_RUNNER = "step5 runner（torchrun 内部入口）"


def _add_common_run_odcr_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="日志文件；mainline 默认 runs/.../meta/full.log；内部直跑不应作为用户入口",
    )
    p.add_argument("--auxiliary", type=str, required=True)
    p.add_argument("--target", type=str, required=True)
    p.add_argument("--save_file", type=str, default=None)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--num-proc", type=int, default=None, help="datasets.map 进程数")
    p.add_argument("--nlayers", type=int, default=None)
    p.add_argument("--nhead", type=int, default=None)
    p.add_argument("--nhid", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    p.add_argument("--generate-temperature", type=float, default=0.8)
    p.add_argument("--generate-top-p", type=float, default=0.9)
    p.add_argument("--max-explanation-length", type=int, default=25)
    p.add_argument(
        "--decode-strategy",
        type=str,
        choices=["greedy", "nucleus", "uncertainty_low_temp_top_k"],
        default="greedy",
        help="torchrun 子进程参数；日常请用: python code/odcr.py … --decode-preset <stem>。"
        "greedy 消融；uncertainty_low_temp_top_k 为主线低温不确定采样；nucleus 可配合 --decode-seed",
    )
    p.add_argument("--decode-seed", type=int, default=None)
    p.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=None,
        dest="no_repeat_ngram_size",
        help="生成时 no_repeat_ngram_size；默认 null（记录于 metrics.decode / generation_semantic_resolved）",
    )
    p.add_argument(
        "--min-len",
        type=int,
        default=None,
        dest="min_len",
        help="生成最小长度约束；默认 null（记录于 metrics.decode / generation_semantic_resolved）",
    )
    p.add_argument("--eval-batch-size", type=int, default=None, help="覆盖 FinalTrainingConfig.eval_batch_size 解析链")
    p.add_argument(
        "--eval-single-process-safe",
        action="store_true",
        help="多卡时仅 rank0 顺序跑全量评测（避免 DDP 分片聚合差异；与 DDP 指标对照用）",
    )
    p.add_argument(
        "--sanity-compare-ddp-single",
        action="store_true",
        help="rank0 在 DDP 评测后再跑一遍单进程顺序，打印 MAE/RMSE/BLEU4 差值",
    )


def _add_train_cli_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="学习率；不传则由 build_resolved_training_config 解析",
    )
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--coef", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--gradient-accumulation-steps", type=int, default=None)
    p.add_argument("--per-device-batch-size", type=int, default=None)
    p.add_argument(
        "--train-only",
        action="store_true",
        help="训练结束后跳过 valid 收尾评测（训练中仍按 epoch 做 valid）",
    )
    p.add_argument("--min-epochs", type=int, default=None)
    p.add_argument("--early-stop-patience", type=int, default=None)
    p.add_argument("--early-stop-patience-full", type=int, default=None)
    p.add_argument("--early-stop-patience-loss", type=int, default=None)
    p.add_argument(
        "--checkpoint-metric",
        type=str,
        choices=["valid_loss", "loss"],
        default="valid_loss",
        help="与训练 preset 一致；主线仅 valid_loss 选模",
    )
    p.add_argument("--bleu4-max-samples", type=int, default=None)
    p.add_argument("--quick-eval-max-samples", type=int, default=None)
    p.add_argument("--scheduler-initial-lr", type=float, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--warmup-ratio", type=float, default=None)
    p.add_argument("--min-lr-ratio", type=float, default=None)


def print_step5_root_help() -> None:
    p = argparse.ArgumentParser(
        prog="step5-runner",
        description=(
            "Step5 主模型 train / eval / test / generate_samples — torchrun 内部入口（须 NCCL）。"
            "请优先: python code/odcr.py step5|eval|pipeline …"
        ),
        epilog="子命令完整参数: 在 code/ 下对 executors/step5_entry.py 执行 train --help / eval --help（将加载 PyTorch 等依赖）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="训练")
    sub.add_parser("eval", help="valid 评测")
    sub.add_parser("test", help="test.csv 评测")
    sub.add_parser("generate_samples", help="导出小批生成样例")
    p.print_help()


def run_step5_cli() -> None:
    bootstrap.reject_legacy_gpus_argv(
        sys.argv,
        executor_label=_STEP5_RUNNER,
        torchrun_hint=(
            "推荐: python code/odcr.py step5|eval …\n"
            "须自行 torchrun 时见 docs/ODCR_Scripts_and_Runtime_Guide.md 附录。\n"
        ),
    )
    epilog = (
        "用户日常（仓库根）:\n"
        "  python code/odcr.py step5 …   python code/odcr.py eval …   python code/odcr.py pipeline …\n"
        "本入口仅在被 odcr.py / sh 以 torchrun 调用时使用。"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Step5 主模型 train / eval / test / generate_samples — torchrun 内部入口。"
            "请优先: python code/odcr.py …"
        ),
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common_parent = argparse.ArgumentParser(add_help=False)
    _add_common_run_odcr_args(common_parent)

    train_p = sub.add_parser(
        "train",
        parents=[common_parent],
        help="训练（内部由 odcr.py step5 / sh 调用）",
    )
    _add_train_cli_args(train_p)

    sub.add_parser(
        "eval",
        parents=[common_parent],
        help="valid 评测（内部由 odcr.py eval / sh 调用）",
    )
    er = sub.add_parser(
        "eval-rerank",
        parents=[common_parent],
        help="valid 评测 + 多候选 rule rerank（内部由 odcr.py eval-rerank 调用）",
    )
    er.add_argument(
        "--num-return-sequences",
        type=int,
        default=None,
        dest="num_return_sequences",
        help="每样本生成候选数 K；省略则由父进程 configs/odcr.yaml 与默认 4 决定（与 odcr.py 一致）",
    )
    er.add_argument(
        "--rerank-method",
        type=str,
        default=None,
        dest="rerank_method",
        help="省略则由父进程 configs/odcr.yaml 决定（与 odcr.py 默认 None 一致）",
    )
    er.add_argument("--rerank-top-k", type=int, default=None, dest="rerank_top_k")
    er.add_argument(
        "--rerank-weight-logprob",
        type=float,
        default=None,
        dest="rerank_weight_logprob",
    )
    er.add_argument(
        "--rerank-weight-length",
        type=float,
        default=None,
        dest="rerank_weight_length",
    )
    er.add_argument(
        "--rerank-weight-repeat",
        type=float,
        default=None,
        dest="rerank_weight_repeat",
    )
    er.add_argument(
        "--rerank-weight-dirty",
        type=float,
        default=None,
        dest="rerank_weight_dirty",
    )
    er.add_argument(
        "--rerank-target-len-ratio",
        type=float,
        default=None,
        dest="rerank_target_len_ratio",
    )
    er.add_argument(
        "--export-examples-mode",
        type=str,
        default=None,
        choices=("changed_only", "head20", "head50", "full", "none"),
        dest="export_examples_mode",
        help="省略则由 configs/odcr.yaml 与默认 head50 决定（与 odcr.py 一致）",
    )
    er.add_argument(
        "--export-full-rerank-examples",
        action="store_true",
        dest="export_full_rerank_examples",
    )
    er.add_argument(
        "--rerank-malformed-tail-penalty",
        type=float,
        default=None,
        dest="rerank_malformed_tail_penalty",
    )
    er.add_argument(
        "--rerank-malformed-token-penalty",
        type=float,
        default=None,
        dest="rerank_malformed_token_penalty",
    )
    sub.add_parser(
        "test",
        parents=[common_parent],
        help="test.csv 评测（高级场景；日常优先 odcr.py）",
    )
    gen_p = sub.add_parser(
        "generate_samples",
        parents=[common_parent],
        help="导出小批生成样例（高级场景）",
    )
    gen_p.add_argument("--generate-max-samples", type=int, default=32)

    args = parser.parse_args()
    ddp_utils.exit_if_not_torchrun(
        executor_label=_STEP5_RUNNER,
        examples=(
            "推荐: python code/odcr.py step5|eval …\n"
            "附录: 高级 torchrun 排障见 docs/ODCR_Scripts_and_Runtime_Guide.md。\n"
        ),
    )
    print_startup_config_check(stage="step5", command=str(args.command))
    if os.environ.get("RANK", "0") == "0":
        print(
            f"[step5 runner] {args.command} — 用户入口: python code/odcr.py step5|eval|pipeline …",
            flush=True,
        )

    from executors.step5_engine import _run_ddp  # noqa: E402

    _run_ddp(args)


if __name__ == "__main__":
    run_step5_cli()
