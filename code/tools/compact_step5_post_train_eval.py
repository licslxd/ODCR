from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from odcr_core.step5_eval_summary import compact_post_train_eval_layout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compact a Step5 post_train_eval directory into human metrics logs, categorized evidence directories, and cache/ evidence."
    )
    parser.add_argument("post_train_eval_dir", help="Path to runs/step5/.../post_train_eval")
    parser.add_argument("--cache-root", default=None, help="Override cache root; defaults to repo cache/.")
    args = parser.parse_args(argv)
    summary = compact_post_train_eval_layout(args.post_train_eval_dir, cache_root=args.cache_root)
    print("Step5 post_train_eval layout compacted")
    print(f"metrics_log: {summary['metrics_log']}")
    print(f"layout_log: {summary['layout_log']}")
    for split, item in sorted(summary.get("splits", {}).items()):
        print(f"{split}_metrics_log: {item.get('metrics_log')}")
        print(f"{split}_evidence_dir: {item.get('evidence_dir')}")
        print(f"{split}_cache_dir: {item.get('cache_dir')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
