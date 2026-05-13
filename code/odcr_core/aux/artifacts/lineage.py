"""Small lineage helpers for reusable ODCR artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def lineage_fingerprint(payload: Mapping[str, Any]) -> str:
    text = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

