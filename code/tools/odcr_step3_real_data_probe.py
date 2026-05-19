#!/usr/bin/env python3
"""Retired Step3 real-data probe entrypoint.

Bounded runtime validation is now allowlisted through ./odcr runtime probe.
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retired Step3 probe wrapper.")
    parser.add_argument("args", nargs="*")
    return parser


def main(argv: list[str] | None = None) -> int:
    _ = build_parser().parse_args(argv)
    print("odcr_step3_real_data_probe.py is retired; use ./odcr runtime probe --stage step3 --task 2 --bounded.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
