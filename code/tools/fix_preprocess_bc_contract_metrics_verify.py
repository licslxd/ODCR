#!/usr/bin/env python3
"""Retired one-off preprocess_b/c repair tool."""

from __future__ import annotations

import sys


def main() -> int:
    print("fix_preprocess_bc_contract_metrics_verify.py is retired; use ./odcr doctor and post-edit validation.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
