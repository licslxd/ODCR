"""Canonical run-meta log path policy."""

from __future__ import annotations

from pathlib import Path


def meta_log_paths(meta_dir: str | Path) -> dict[str, Path]:
    meta = Path(meta_dir)
    return {
        "console": meta / "console.log",
        "full": meta / "full.log",
        "debug": meta / "debug.log",
        "errors": meta / "errors.log",
    }

