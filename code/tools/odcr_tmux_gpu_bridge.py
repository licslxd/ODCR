#!/usr/bin/env python3
"""Retired direct tmux bridge tool.

Use ./odcr runtime bridge ... instead.
"""

from __future__ import annotations

import sys


if __name__ == "__main__":
    _ = sys.argv[1:]
    print(
        "python code/tools/odcr_tmux_gpu_bridge.py is retired and fail-fast. "
        "Use ./odcr runtime bridge discover|validate-only|marker-probe|cuda-probe.",
        file=sys.stderr,
    )
    raise SystemExit(2)
