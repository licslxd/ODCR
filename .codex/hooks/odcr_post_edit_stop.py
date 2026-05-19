#!/usr/bin/env python3
"""Thin ODCR Codex Stop hook entrypoint.

Implementation lives in odcr_core.aux.governance.hook_scope.
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[2] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.aux.governance.hook_scope import *  # noqa: F401,F403,E402
from odcr_core.aux.governance.hook_scope import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
