"""统一提示文案：bootstrap / argparse 可用，避免依赖 torch。"""

ODCR_MAINLINE_HINT = "用户日常推荐（仓库根）: python code/odcr.py …"


def legacy_gpus_removed(executor_label: str, *, torchrun_hint: str) -> str:
    return (
        f"{executor_label}: error: --gpus has been removed.\n"
        f"{ODCR_MAINLINE_HINT}\n"
        f"{torchrun_hint}\n"
    )


def torchrun_required(executor_label: str, *, examples: str) -> str:
    return (
        f"错误: {executor_label} 须由 torchrun 启动。\n"
        f"{ODCR_MAINLINE_HINT}\n"
        f"{examples}\n"
    )


def internal_executor_banner(executor_file: str, *, role: str) -> str:
    return (
        f"[Internal Executor] {executor_file} — {role}；"
        f"推荐入口: {ODCR_MAINLINE_HINT.strip()}"
    )
