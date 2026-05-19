from __future__ import annotations

import shutil
from pathlib import Path


def remove_active_python_caches(repo_root: Path) -> None:
    for rel_root in ("code", "configs", "docs", ".codex"):
        root = repo_root / rel_root
        if not root.exists():
            continue
        for pyc in root.rglob("*.pyc"):
            pyc.unlink(missing_ok=True)
        for cache_dir in root.rglob("__pycache__"):
            shutil.rmtree(cache_dir, ignore_errors=True)
        for cache_dir in root.rglob(".pytest_cache"):
            shutil.rmtree(cache_dir, ignore_errors=True)
    for cache_name in ("__pycache__", ".pytest_cache"):
        shutil.rmtree(repo_root / cache_name, ignore_errors=True)
