"""DDP / torchrun 环境检查（仅标准库 + messages）。"""

from __future__ import annotations

import os
import sys

from executors import messages


def exit_if_not_torchrun(*, executor_label: str, examples: str) -> None:
    if "RANK" not in os.environ:
        sys.stderr.write(messages.torchrun_required(executor_label, examples=examples))
        raise SystemExit(1)
