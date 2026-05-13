from __future__ import annotations

import json
from pathlib import Path

from odcr_core.config_resolver import resolve_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_csb_odcr_sidecar_stable_profile_resolves_one_control_surface() -> None:
    cfg, _sources, snapshot = resolve_config(
        config_path=REPO_ROOT / "configs" / "odcr.yaml",
        command="step3",
        task_id=2,
        set_overrides=["experiment_profile=csb_odcr_sidecar_stable"],
        dry_run=True,
        run_id="auto",
        mode="full",
    )
    payload = json.loads(cfg.effective_training_payload_json)
    profile = payload["experiment_profile"]
    csb = payload["step3_csb_odcr"]

    assert cfg.per_gpu_batch_size == 704
    assert cfg.global_batch_size == 1408
    assert cfg.learning_rate == 0.0007
    assert cfg.max_grad_norm == 0.5
    assert snapshot["train"]["precision"] == "bf16"
    assert profile["name"] == "csb_odcr_sidecar_stable"
    assert csb["primary_training"] == "rating_only"
    assert csb["csb_mode"] == "sidecar"
    assert csb["gradient_firewall"] is True
    assert csb["controlled_injection_formal_train"] is False
    assert csb["light_explainer_step3_loss"] is False
    assert csb["paper_metric_gate"] is False
    assert snapshot["experiment_profiles"]["csb_odcr_full_safe"]["formal_allowed"] is False

