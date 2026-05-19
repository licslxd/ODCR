#!/usr/bin/env python3
"""Thin ODCR post-edit validation entrypoint.

Implementation lives in odcr_core.aux.governance.post_edit_runner.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.aux.governance.post_edit_runner import *  # noqa: F401,F403,E402
from odcr_core.aux.governance.post_edit_runner import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
