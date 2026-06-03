"""Retired Step5 generator executor."""


RETIRED_STEP5_MESSAGE = (
    "Old Step5 generator/eval/rerank code has been deleted. "
    "Use `python code/odcr.py racer-c1 --task 2 --mode prepare|train_eval`."
)


def print_step5_root_help() -> None:
    print(RETIRED_STEP5_MESSAGE)


def run_step5_cli() -> None:
    raise RuntimeError(RETIRED_STEP5_MESSAGE)


run_step5_main = run_step5_cli

__all__ = ["RETIRED_STEP5_MESSAGE", "print_step5_root_help", "run_step5_cli", "run_step5_main"]
