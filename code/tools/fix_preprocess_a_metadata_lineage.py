#!/usr/bin/env python3
"""Retired one-off repair tool.

This active CLI no longer mutates ODCR artifacts. Use the unified ./odcr
workflow and aux governance validation instead.
"""

from __future__ import annotations

import sys


def main() -> int:
    print("fix_preprocess_a_metadata_lineage.py is retired; use ./odcr doctor and post-edit validation.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
