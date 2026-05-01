#!/usr/bin/env python3
"""Print a blank ODCR feature integration checklist.

This helper is intentionally side-effect free. It does not read ODCR data,
lineage, caches, checkpoints, exports, or run directories; it only prints a
copyable governance template for future change requests.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence


STAGE_DEFAULTS = {
    "preprocess": {
        "yaml": "preprocess.<new_key>",
        "producer": "code/preprocess_data.py or preprocess runtime owner",
        "consumer": "split/combine/downstream stage consumer",
        "dry_run": "./odcr preprocess <a|b|c> --dry-run when available",
    },
    "step3": {
        "yaml": "step3.<new_key>",
        "producer": "code/executors/step3_train_core.py or Step3 helper",
        "consumer": "Step3 checkpoint writer / Step4 loader",
        "dry_run": "./odcr step3 --task 4 --dry-run",
    },
    "step4": {
        "yaml": "step4.<new_key> or step4.rcr.<new_key>",
        "producer": "code/executors/step4_engine.py or routing export helper",
        "consumer": "Step5 routing export consumer",
        "dry_run": "./odcr step4 --task 4 --dry-run when allowed",
    },
    "step5": {
        "yaml": "step5.<new_key>",
        "producer": "code/executors/step5_engine.py or Step5 innovation helper",
        "consumer": "Step5 checkpoint / eval / rerank consumer",
        "dry_run": "./odcr step5 --task 4 --dry-run when allowed",
    },
    "eval": {
        "yaml": "eval.<new_key>",
        "producer": "eval/rerank output writer",
        "consumer": "metrics/report consumer",
        "dry_run": "./odcr eval --task 4 --dry-run when allowed",
    },
    "rerank": {
        "yaml": "eval.rerank.<new_key> or rerank owner path",
        "producer": "rerank output writer",
        "consumer": "rerank metrics/report consumer",
        "dry_run": "./odcr eval --task 4 --mode rerank --dry-run when allowed",
    },
    "tooling": {
        "yaml": "N/A - tooling-only unless public config is added",
        "producer": "tooling helper",
        "consumer": "developer/Codex workflow",
        "dry_run": "python code/tools/<tool>.py --help",
    },
}


CHANGE_TYPES = (
    "new parameter",
    "new field",
    "new artifact",
    "new entrypoint",
    "new model/loss/router/verbalizer",
    "modify configuration control plane",
    "modify cache/checkpoint/export",
    "modify eval/rerank",
    "delete or migrate old logic",
)


IMPACT_ROWS = (
    "One-Control",
    "YAML path",
    "schema path",
    "resolver path",
    "source table key",
    "producer",
    "consumer",
    "contract version",
    "manifest key",
    "index_contract key",
    "fingerprint key",
    "mismatch policy",
    "DDP risk",
    "eval/rerank risk",
    "legacy cleanup",
    "guardrail rule",
    "unit test",
    "dry-run command",
    "rerun decision",
)


def _stage_defaults(stage: str | None) -> dict[str, str]:
    if stage is None:
        return {
            "yaml": "",
            "producer": "",
            "consumer": "",
            "dry_run": "",
        }
    return STAGE_DEFAULTS[stage]


def build_checklist(stage: str | None = None) -> str:
    defaults = _stage_defaults(stage)
    stage_label = stage or ""
    lines = [
        "# ODCR Feature Integration Checklist Stub",
        "",
        "Generated template. Fill every blank row or write `N/A - reason`.",
        "",
        "## Classification",
        "",
        f"- Owning stage: {stage_label}",
        "- Goal:",
        "- Explicit non-goals:",
        "",
        "## Change Type Selection",
        "",
        "| Change type | yes/no | Details |",
        "| --- | --- | --- |",
    ]
    for change_type in CHANGE_TYPES:
        lines.append(f"| {change_type} |  |  |")
    lines.extend(
        [
            "",
            "## Required Impact Surface",
            "",
            "| Row | Value |",
            "| --- | --- |",
        ]
    )
    seeded = {
        "YAML path": defaults["yaml"],
        "producer": defaults["producer"],
        "consumer": defaults["consumer"],
        "dry-run command": defaults["dry_run"],
    }
    for row in IMPACT_ROWS:
        lines.append(f"| {row} | {seeded.get(row, '')} |")
    lines.extend(
        [
            "",
            "## Required Outputs",
            "",
            "- Modified files:",
            "- Old logic handling:",
            "- Rerun decision:",
            "- AI_analysis ledger path:",
            "- Lightweight verification result:",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print a blank ODCR feature integration checklist stub."
    )
    parser.add_argument(
        "--stage",
        choices=sorted(STAGE_DEFAULTS),
        help="Pre-fill stage-specific YAML, producer, consumer, and dry-run hints.",
    )
    args = parser.parse_args(argv)
    print(build_checklist(args.stage), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
