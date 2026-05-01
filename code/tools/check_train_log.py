#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 train.log 做轻量自检（shell 行混入、Epoch 序号连续性）。用法（在项目根）:
  python code/tools/check_train_log.py runs/task4/v1/train/step5/2_1_1/logs/train.log
  python code/tools/check_train_log.py runs/task4/v1/train/step5/2_1_1/logs/train.log --json
  python code/tools/check_train_log.py train.log --strict   # 有问题时退出码 2

  legacy：旧仓库布局下的路径（如 log/.../train.log）仅作考古；主线请以 runs/task{T}/vN/train/.../logs/train.log 为准。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_TOOLS_DIR)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from train_logging import audit_train_log_file  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="检查 train.log 是否混入 shell 行或 Epoch 不连续（启发式）")
    p.add_argument("log_file", help="train.log 路径")
    p.add_argument("--json", action="store_true", help="输出 JSON")
    p.add_argument(
        "--strict",
        action="store_true",
        help="存在 shell_hits 或 epoch_sequence_gaps 时以退出码 2 结束",
    )
    args = p.parse_args()
    r = audit_train_log_file(args.log_file)
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        if r.get("error"):
            print(f"[check_train_log] {r['error']}: {args.log_file}", file=sys.stderr)
            return 1
        print(f"file: {r['path']}")
        print(f"epoch lines (Epoch N): {r.get('epoch_line_count', 0)}, max: {r.get('epoch_max')}")
        if r["shell_hits"]:
            print(f"shell-like lines: {len(r['shell_hits'])}")
            for h in r["shell_hits"][:20]:
                print(f"  L{h['line']}: {h['snippet']}")
            if len(r["shell_hits"]) > 20:
                print("  ...")
        else:
            print("shell-like lines: 0")
        if r["epoch_sequence_gaps"]:
            print(f"epoch sequence gaps: {len(r['epoch_sequence_gaps'])}")
            for g in r["epoch_sequence_gaps"][:20]:
                print(f"  after {g['after_epoch']} -> next {g['next_seen']}")
        else:
            print("epoch sequence gaps: 0")
    if r.get("error"):
        return 1
    if args.strict and (r.get("shell_hits") or r.get("epoch_sequence_gaps")):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
