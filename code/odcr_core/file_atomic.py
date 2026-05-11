from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(p))
    _fsync_parent(p)
    return p


def atomic_torch_save(path: str | Path, obj: Any) -> Path:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".tmp")
    with tmp.open("wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(p))
    _fsync_parent(p)
    return p


def atomic_save_numpy(path: str | Path, array: Any) -> Path:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, array)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(p))
    _fsync_parent(p)
    return p


def atomic_write_text(path: str | Path, text: str) -> Path:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(p))
    _fsync_parent(p)
    return p


def _fsync_parent(path: Path) -> None:
    try:
        dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
