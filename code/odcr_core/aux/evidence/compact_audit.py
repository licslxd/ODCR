"""Small helpers for compact AI_analysis audit payloads."""

from __future__ import annotations

import json
from typing import Any, Mapping


def compact_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

