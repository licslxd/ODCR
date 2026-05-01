"""仅标准库：argv 校验等，可在重型 import 之前调用。"""

from __future__ import annotations

import sys
from typing import Sequence

from executors import messages


def reject_legacy_gpus_argv(argv: Sequence[str], *, executor_label: str, torchrun_hint: str) -> None:
    for arg in argv[1:]:
        if arg == "--gpus" or arg.startswith("--gpus="):
            sys.stderr.write(
                messages.legacy_gpus_removed(executor_label, torchrun_hint=torchrun_hint)
            )
            raise SystemExit(2)
