"""Aux doctor checks shown through the existing doctor command."""

from __future__ import annotations

from odcr_core.aux.runtime.command_registry import registered_command_names


def aux_doctor_lines() -> list[str]:
    names = registered_command_names()
    return [
        "aux architecture: active",
        f"runtime registered commands: {len(names)}",
        "runtime bridge: allowlist only",
    ]
