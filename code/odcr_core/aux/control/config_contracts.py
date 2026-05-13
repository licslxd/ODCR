"""Config-contract helper namespace for control-plane checks."""

from __future__ import annotations

ONE_CONTROL_CONFIG = "configs/odcr.yaml"


def config_contract_summary() -> dict[str, str]:
    return {"primary_config": ONE_CONTROL_CONFIG, "preset_config": "retired"}

