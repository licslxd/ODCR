"""Reusable tmux pane and GPU-capacity validation facade."""

from __future__ import annotations

from typing import Any


def validate_current_pane(*, socket: str | None = None, target: str | None = None) -> dict[str, Any]:
    from .gpu_bridge import BridgeOptions, TmuxGpuBridge

    pane, discovery, source = TmuxGpuBridge().resolve_target(BridgeOptions(mode="discover", socket=socket, target=target))
    return {"source": source, "target": pane.to_dict(), "discovery": discovery.to_dict()}


def validate_gpu_capacity(*, socket: str | None = None, target: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    from .gpu_bridge import BridgeOptions, TmuxGpuBridge

    return TmuxGpuBridge().run_mode(BridgeOptions(mode="validate-only", socket=socket, target=target, dry_run=dry_run))

