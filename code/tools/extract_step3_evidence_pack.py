#!/usr/bin/env python3
"""Retired Step3 evidence-pack extractor.

Auxiliary runtime evidence is now produced through the registered
``./odcr runtime ...`` command surface and the unified AI_analysis writer.
"""

from __future__ import annotations

import sys


MESSAGE = (
    "extract_step3_evidence_pack.py is retired. Use registered aux runtime "
    "commands, for example: ./odcr runtime probe --stage step3 --task 2 --bounded"
)


def main(argv: list[str] | None = None) -> int:
    _ = argv
    print(MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
