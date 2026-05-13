from __future__ import annotations

from odcr_core.aux.runtime.stage_dispatch import runtime_probe_bridge_args
from odcr_core.step3_runtime_probe import STEP3_RUNTIME_PROBE_TYPES


def test_sidecar_gradient_firewall_bounded_probe_is_registered() -> None:
    assert "sidecar-gradient-firewall" in STEP3_RUNTIME_PROBE_TYPES
    args = runtime_probe_bridge_args(
        stage="step3",
        task=2,
        profile="csb_odcr_sidecar_stable",
        bounded=True,
        probe_kind="sidecar-gradient-firewall",
        no_send=True,
    )
    assert "step3-performance-probe" in args
    assert "--probe-type" in args
    assert "sidecar-gradient-firewall" in args
    assert "csb_odcr_sidecar_stable" in args

