from __future__ import annotations

import pytest

from odcr_core.aux.runtime.command_registry import LEGACY_FORBIDDEN_COMMANDS, RuntimeCommandError, get_registry


def test_required_runtime_commands_registered() -> None:
    names = set(get_registry().names())
    assert {
        "bridge.discover",
        "bridge.validate_only",
        "bridge.marker_probe",
        "bridge.cuda_probe",
        "probe.step3.bounded",
        "probe.step4.bounded",
        "probe.step5.bounded",
        "probe.step5A.bounded",
        "probe.step5B.bounded",
        "step5.admission.dry_run",
    } <= names


def test_step5_tuning_bounded_command_is_ai_analysis_only() -> None:
    spec = get_registry().require("probe.step5.bounded")
    assert spec.writes_formal_runs is False
    assert spec.output_policy == "AI_analysis"
    assert spec.formal_namespace_policy == "forbid"
    assert "--candidate-id" in spec.allowed_args
    assert "--set" in spec.allowed_args
    assert "--from-step4-run" in spec.allowed_args
    assert "--global" in spec.allowed_args
    assert "--socket" in spec.allowed_args
    assert "--target" in spec.allowed_args
    assert spec.timeout_s >= 900


def test_runtime_registry_rejects_allocation_and_tmux_session_control_tokens() -> None:
    spec = get_registry().require("probe.step5.bounded")
    for token in ("srun", "sbatch", "scancel", "odcr-enter-gpu"):
        with pytest.raises(RuntimeCommandError):
            spec.validate_args(("--stage", "step5", "--bounded", "--set", token))
    for token in ("tmux attach", "tmux switch-client", "tmux kill-session", "tmux new-session"):
        with pytest.raises(RuntimeCommandError):
            spec.validate_args(("--stage", "step5", "--bounded", "--set", token))


@pytest.mark.parametrize("name", LEGACY_FORBIDDEN_COMMANDS)
def test_legacy_runtime_commands_unregistered(name: str) -> None:
    assert get_registry().get(name) is None
    with pytest.raises(RuntimeCommandError):
        get_registry().require(name)
