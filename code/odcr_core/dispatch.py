"""torchrun 目标脚本名（仅子进程使用）与 MAINLINE 分发叙事。"""
from __future__ import annotations

import os

# torchrun 在 code/ 目录下加载的 runner 入口（实现仍在 executors/*_engine.py）
TORCHRUN_STEP3_SCRIPT = "executors/step3_entry.py"
TORCHRUN_STEP4_SCRIPT = "executors/step4_entry.py"
TORCHRUN_STEP5_SCRIPT = "executors/step5_entry.py"


def print_dispatch_routing(command: str) -> None:
    """用户可见主叙事：阶段 runner，不强调薄壳文件名。"""
    if command == "step3":
        print("[Dispatch] step3 -> step3 runner (torchrun)", flush=True)
    elif command == "step4":
        print("[Dispatch] step4 -> step4 runner (torchrun)", flush=True)
    elif command == "step5":
        print("[Dispatch] step5 -> step5 runner train (torchrun)", flush=True)
    elif command == "eval":
        print("[Dispatch] eval -> step5 runner eval (torchrun)", flush=True)
    elif command == "eval-matrix":
        print("[Dispatch] eval-matrix -> 多次 step5 runner eval (torchrun)", flush=True)
    elif command == "eval-rerank":
        print("[Dispatch] eval-rerank -> step5 runner eval-rerank (torchrun)", flush=True)
    elif command == "eval-rerank-matrix":
        print("[Dispatch] eval-rerank-matrix -> 多次 step5 runner eval-rerank (torchrun)", flush=True)
    elif command == "rerank-summary":
        print("[Dispatch] rerank-summary -> 扫描 rerank_summary.json 写 phase2_rerank_summary（无 torchrun）", flush=True)
    elif command == "eval-summary":
        print("[Dispatch] eval-summary -> 离线扫描 eval_metrics.json（无 torchrun）", flush=True)
    elif command == "pipeline":
        print(
            "[Dispatch] pipeline -> step3 runner -> step4 runner -> step5 runner (torchrun x3)",
            flush=True,
        )


def print_dispatch_script_detail(command: str) -> None:
    """排障用：打印实际 torchrun 加载的 runner 入口脚本。默认关闭，设 ODCR_DISPATCH_DETAIL=1 开启。"""
    flag = (os.environ.get("ODCR_DISPATCH_DETAIL") or "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return
    if command == "step3":
        print(f"[Dispatch][detail] torchrun script={TORCHRUN_STEP3_SCRIPT}", flush=True)
    elif command == "step4":
        print(f"[Dispatch][detail] torchrun script={TORCHRUN_STEP4_SCRIPT}", flush=True)
    elif command in ("step5", "eval", "eval-matrix", "eval-rerank", "eval-rerank-matrix"):
        print(f"[Dispatch][detail] torchrun script={TORCHRUN_STEP5_SCRIPT}", flush=True)
    elif command == "pipeline":
        print(
            "[Dispatch][detail] torchrun scripts="
            f"{TORCHRUN_STEP3_SCRIPT}, {TORCHRUN_STEP4_SCRIPT}, {TORCHRUN_STEP5_SCRIPT}",
            flush=True,
        )


def internal_executor_label(step: int) -> str:
    if step == 3:
        return TORCHRUN_STEP3_SCRIPT
    if step == 4:
        return TORCHRUN_STEP4_SCRIPT
    if step == 5:
        return TORCHRUN_STEP5_SCRIPT
    raise ValueError(f"unknown step: {step}")
