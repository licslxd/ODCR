"""Codex runtime contract text for tmux GPU validation."""

from __future__ import annotations


CONTRACT = (
    "Codex must use ./odcr runtime bridge discover, validate-only, marker-probe, "
    "and cuda-probe against the user-prepared current tmux GPU pane. Codex must "
    "not allocate GPU resources or use arbitrary shell/repo-command modes."
)
