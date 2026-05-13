#!/usr/bin/env python3
"""Deprecated thin wrapper for the unified ODCR runtime GPU bridge.

Active use is through ``./odcr runtime ...``. This module remains only for
old unit imports and direct developer diagnostics; implementation lives in
``odcr_core.aux.runtime.gpu_bridge``.
"""

from __future__ import annotations

import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.aux.runtime import gpu_bridge as _impl  # noqa: E402


for _name in dir(_impl):
    if _name.startswith("__") and _name not in {"__all__", "__doc__"}:
        continue
    globals()[_name] = getattr(_impl, _name)


if __name__ == "__main__":
    raise SystemExit(_impl.main())

sys.modules[__name__] = _impl
