"""Show-view helper namespace for the aux control plane."""

from __future__ import annotations

from typing import Any, Mapping


def compact_show_view(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return dict(snapshot)

