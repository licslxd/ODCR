"""Doctor checks shared by ODCR control-plane commands."""

from __future__ import annotations

from .cli_surface import runtime_surface_summary


def runtime_doctor_checks() -> list[str]:
    summary = runtime_surface_summary()
    bridge_modes = ", ".join(summary["bridge_modes"])
    stages = ", ".join(summary["stages"])
    return [
        f"runtime bridge modes registered: {bridge_modes}",
        f"runtime probe stages registered: {stages}",
        "runtime arbitrary shell dispatch disabled",
    ]

