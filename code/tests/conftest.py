from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
for rel in ("code", "code/tools", "code/tests"):
    path = str(REPO_ROOT / rel)
    if path not in sys.path:
        sys.path.insert(0, path)
