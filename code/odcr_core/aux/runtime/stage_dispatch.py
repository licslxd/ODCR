"""Dispatch runtime probe stages to registered specs."""

from __future__ import annotations

from .command_registry import require_command


def probe_command_name(stage: str, *, bounded: bool) -> str:
    if not bounded:
        raise ValueError("runtime probe must be explicitly --bounded")
    normalized = str(stage).strip()
    if normalized not in {"step3", "step4", "step5", "step5A", "step5B"}:
        raise ValueError(f"unsupported runtime probe stage: {stage}")
    name = f"probe.{normalized}.bounded"
    require_command(name)
    return name
