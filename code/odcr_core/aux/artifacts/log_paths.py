"""Canonical log path helpers."""

from __future__ import annotations

from pathlib import Path


def run_meta_logs(run_dir: str | Path) -> dict[str, Path]:
    meta = Path(run_dir) / "meta"
    return {
        "console": meta / "console.log",
        "full": meta / "full.log",
        "errors": meta / "errors.log",
    }
