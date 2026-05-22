"""Registered runtime command templates."""

from __future__ import annotations


def bridge_command(name: str) -> tuple[str, ...]:
    if name not in {"discover", "validate-only", "marker-probe", "cuda-probe", "exec"}:
        raise ValueError(f"unregistered bridge command template: {name}")
    return ("./odcr", "runtime", "bridge", name)
