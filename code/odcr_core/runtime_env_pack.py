"""父进程经 ODCR_*_JSON 注入、子进程 config_resolved 回写的 runtime / launcher 环境块。"""
from __future__ import annotations

import json
import os
from typing import Any, Dict


def runtime_env_dict_for_config_resolved() -> Dict[str, Any]:
    """从当前进程环境读取 torchrun 前注入的四段 JSON；无则返回空 dict。"""

    def _parse(name: str):
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            return None
        try:
            o = json.loads(raw)
            return o if isinstance(o, dict) else {"_invalid": raw}
        except json.JSONDecodeError:
            return {"_raw": raw}

    out: Dict[str, Any] = {}
    for label, key in (
        ("thread_env_requested", "ODCR_THREAD_ENV_REQUESTED_JSON"),
        ("thread_env_effective", "ODCR_THREAD_ENV_EFFECTIVE_JSON"),
        ("launcher_env_requested", "ODCR_LAUNCHER_ENV_REQUESTED_JSON"),
        ("launcher_env_effective", "ODCR_LAUNCHER_ENV_EFFECTIVE_JSON"),
    ):
        v = _parse(key)
        if v is not None:
            out[label] = v
    return out
