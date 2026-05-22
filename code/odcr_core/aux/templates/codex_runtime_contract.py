"""Codex runtime contract text for tmux GPU validation."""

from __future__ import annotations


CONTRACT = (
    "Codex must use ./odcr runtime bridge discover, validate-only, marker-probe, "
    "cuda-probe, or bridge exec against the user-prepared current tmux GPU pane. "
    "bridge exec may dispatch user-authorized GPU work after fresh CUDA validation."
)
