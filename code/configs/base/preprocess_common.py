from __future__ import annotations

from odcr_core.preprocess_schema import (
    PreprocessPathsConfig,
    PreprocessRuntimeOptions,
    VALID_DATASETS,
)


def all_preprocess_datasets() -> tuple[str, ...]:
    return tuple(VALID_DATASETS)


def default_preprocess_paths(stage: str) -> PreprocessPathsConfig:
    meta_root = f"runs/{stage}/meta"
    return PreprocessPathsConfig(
        meta_root=meta_root,
        shell_log_dir=f"{meta_root}/shell_logs",
    )


def default_runtime_options(
    *,
    python_bin: str = "python",
    resume: bool = True,
    skip_completed: bool = True,
    verify_only: bool = False,
    dry_run: bool = False,
    workers: int | None = None,
) -> PreprocessRuntimeOptions:
    return PreprocessRuntimeOptions(
        python_bin=python_bin,
        resume=resume,
        skip_completed=skip_completed,
        verify_only=verify_only,
        dry_run=dry_run,
        workers=workers,
        force_datasets=(),
    )
