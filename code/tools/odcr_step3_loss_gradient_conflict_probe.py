#!/usr/bin/env python3
"""Retired Step3 loss-gradient conflict probe entrypoint.

Runtime probes now run only through ``./odcr runtime probe`` allowlist dispatch.
"""

from __future__ import annotations

import sys


MESSAGE = (
    "odcr_step3_loss_gradient_conflict_probe.py is retired. Use: "
    "./odcr runtime probe --stage step3 --task 2 --bounded"
)


def main(argv: list[str] | None = None) -> int:
    _ = argv
    print(MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
