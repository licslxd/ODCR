#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""主线 Gather/BLEU 协议轻量冒烟：DDP 行合并 + require_gathered_batch；无需 GPU。

用法（仓库根或 code 目录均可）::
    python code/smoke_bleu_gather_protocol.py
"""
from __future__ import annotations

import os
import sys

_CODE = os.path.dirname(os.path.abspath(__file__))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)


def main() -> None:
    from odcr_core.gather_schema import require_gathered_batch
    from odcr_core.bleu_runtime import smoke_bleu_protocol_merge_and_score

    try:
        require_gathered_batch((1, 2, 3))
    except TypeError:
        pass
    else:
        raise SystemExit("smoke: expected TypeError when gather returns tuple")

    smoke_bleu_protocol_merge_and_score()
    print("smoke_bleu_gather_protocol: ok", flush=True)


if __name__ == "__main__":
    main()
