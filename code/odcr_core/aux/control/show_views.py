"""Aux show-view helpers."""

from __future__ import annotations

from odcr_core.aux.runtime.command_registry import get_registry


def runtime_registry_view() -> dict[str, object]:
    return {
        "schema_version": "odcr_aux_runtime_registry_view/1",
        "commands": [spec.__dict__ for spec in get_registry().specs() if not spec.internal_child],
    }
