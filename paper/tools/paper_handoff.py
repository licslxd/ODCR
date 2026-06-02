#!/usr/bin/env python3
"""Generate the root paper.log handoff for ODCR paper tasks."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AI_PAPER_DIR = REPO_ROOT / "AI_analysis" / "00_paper"
PAPER_LOG = REPO_ROOT / "paper.log"

REQUIRED_NAMES = {
    "README.md",
    "writing_decision_from_chat.md",
    "paper_source_packet.md",
    "paper_review_packet.md",
    "paper_chat_handoff.md",
    "paper_diff_summary.md",
    "paper_build_report.md",
}

SOURCE_PACKET = AI_PAPER_DIR / "paper_source_packet.md"
REVIEW_PACKET = AI_PAPER_DIR / "paper_review_packet.md"
CHAT_HANDOFF = AI_PAPER_DIR / "paper_chat_handoff.md"
DIFF_SUMMARY = AI_PAPER_DIR / "paper_diff_summary.md"
BUILD_REPORT = AI_PAPER_DIR / "paper_build_report.md"


def read_file(path: Path) -> str:
    if not path.exists():
        return f"[MISSING: {path.relative_to(REPO_ROOT)}]\n"
    try:
        return path.read_text(encoding="utf-8", errors="replace").rstrip() + "\n"
    except OSError as exc:
        return f"[MISSING: {path.relative_to(REPO_ROOT)}; read error: {exc}]\n"


def extract_section(markdown: str, section: str) -> str:
    header = f"## {section}"
    lines = markdown.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == header:
            start = idx + 1
            break
    if start is None:
        return ""
    end = len(lines)
    for idx in range(start, len(lines)):
        line = lines[idx]
        if line.startswith("## ") and line.strip() != header:
            end = idx
            break
    return "\n".join(lines[start:end]).strip()


def infer_binary_recommendation(*texts: str) -> str:
    joined = "\n".join(texts)
    explicit_patterns = [
        r"Binary recommendation:\s*([A-Z0-9_]+)",
        r"binary recommendation:\s*([A-Z0-9_]+)",
        r"## Recommendation\s+([A-Z0-9_]+)",
        r"## Binary Recommendation\s+([A-Z0-9_]+)",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, joined, re.MULTILINE)
        if match:
            return match.group(1)
    known = [
        "READY_FOR_CHAT_FIGURE1_REWRITE",
        "READY_FOR_CHAT_REVIEW",
        "WORKFLOW_READY_BUILD_NEEDS_FIX",
        "NEEDS_CHAT_REVIEW",
        "BLOCKED",
    ]
    for value in known:
        if value in joined:
            return value
    return "NEEDS_CHAT_REVIEW"


def format_or_missing(content: str, fallback: str) -> str:
    stripped = content.strip()
    return stripped if stripped else fallback


def build_paper_log(task_name: str) -> str:
    handoff = read_file(CHAT_HANDOFF)
    source_packet = read_file(SOURCE_PACKET)
    review_packet = read_file(REVIEW_PACKET)
    diff_summary = read_file(DIFF_SUMMARY)
    build_report = read_file(BUILD_REPORT)
    binary = infer_binary_recommendation(handoff, build_report, diff_summary)
    timestamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")

    did = extract_section(handoff, "What Codex Did")
    did_not = extract_section(handoff, "What Codex Did Not Do")
    build_status = extract_section(handoff, "Build Status") or build_report
    chat_review = extract_section(handoff, "Chat Review Needed")
    next_action = extract_section(handoff, "Next Recommended Chat Action")
    if not next_action:
        next_action = "Chat should now rewrite paper/figures/figure1_dynamic_causal_refinement.tex."

    parts = [
        "# ODCR Paper Chat Handoff",
        f"- timestamp: {timestamp}",
        f"- task name: {task_name}",
        f"- working directory: {REPO_ROOT}",
        f"- binary recommendation: {binary}",
        "",
        "## What Codex Did",
        format_or_missing(did, "[MISSING_SECTION: paper_chat_handoff.md ## What Codex Did]"),
        "",
        "## What Codex Did Not Do",
        format_or_missing(
            did_not,
            "- No training/eval/runtime work was run for this paper-only task.\n"
            "- No requested incomplete item was recorded.",
        ),
        "",
        "## Build Status",
        format_or_missing(build_status, "[MISSING_SECTION: build status]"),
        "",
        "## Chat Review Needed",
        format_or_missing(chat_review, "[MISSING_SECTION: paper_chat_handoff.md ## Chat Review Needed]"),
        "",
        "## Paper Source Packet",
        source_packet.rstrip(),
        "",
        "## Paper Review Packet",
        review_packet.rstrip(),
        "",
        "## Diff Summary",
        diff_summary.rstrip(),
        "",
        "## Build Report",
        build_report.rstrip(),
        "",
        "## Next Recommended Chat Action",
        next_action.strip(),
        "",
    ]
    return "\n".join(parts)


def check_contract() -> int:
    problems: list[str] = []
    if not AI_PAPER_DIR.exists():
        problems.append(f"missing directory: {AI_PAPER_DIR.relative_to(REPO_ROOT)}")
    else:
        present = {p.name for p in AI_PAPER_DIR.iterdir()}
        missing = sorted(REQUIRED_NAMES - present)
        extra = sorted(p.name for p in AI_PAPER_DIR.iterdir() if p.name not in REQUIRED_NAMES)
        for name in missing:
            problems.append(f"missing required file: AI_analysis/00_paper/{name}")
        for name in extra:
            problems.append(f"extra entry violates handoff contract: AI_analysis/00_paper/{name}")
        for name in sorted(REQUIRED_NAMES & present):
            path = AI_PAPER_DIR / name
            if not path.is_file():
                problems.append(f"required entry is not a file: {path.relative_to(REPO_ROOT)}")
    try:
        PAPER_LOG.parent.mkdir(parents=True, exist_ok=True)
        with PAPER_LOG.open("a", encoding="utf-8"):
            pass
    except OSError as exc:
        problems.append(f"paper.log is not writable: {exc}")

    if problems:
        print("FAIL")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print("PASS")
    print(f"- handoff directory: {AI_PAPER_DIR}")
    print(f"- paper.log writable: {PAPER_LOG}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate the paper handoff contract only")
    parser.add_argument("--task-name", default=os.environ.get("ODCR_PAPER_TASK", "paper_workflow_cleanup"))
    args = parser.parse_args(argv)

    if args.check:
        return check_contract()

    PAPER_LOG.write_text(build_paper_log(args.task_name), encoding="utf-8")
    print(f"wrote {PAPER_LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
