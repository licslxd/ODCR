"""Controlled task7/task8 ablation infrastructure.

This package owns the registry, manifest validation, paper-table gate,
runtime binding, and bounded probe planning for weak cross-platform Step5
ablations.  It deliberately does not launch formal full training or eval work.
"""
from __future__ import annotations

from odcr_core.ablation.binding import load_ablation_binding
from odcr_core.ablation.registry import (
    AblationValidationError,
    entry_key,
    load_registry,
    validate_all,
    validate_registry,
)

__all__ = [
    "AblationValidationError",
    "entry_key",
    "load_ablation_binding",
    "load_registry",
    "validate_all",
    "validate_registry",
]
