from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from odcr_core.config_resolver import resolve_config


def test_step5_resolves_as_explanation_only() -> None:
    cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="step5",
        task_id=2,
        set_overrides=[],
        dry_run=True,
        from_step4="1",
    )
    assert cfg.step5_head == "explanation"
    assert cfg.step5_mode == "explanation_only"
    assert snapshot["step5_task_decoupled_policy"]["rating_training"]["enabled"] is False
    assert snapshot["head_gated_loss_contract"]["active_losses"] == ["explainer_ce", "ccv", "fca"]
    assert "scorer_rating_mse" not in snapshot["step5_eval"]["valid_loss_components"]
    assert snapshot["rating_source"]["type"] == "step3_accepted_scorer"
    assert cfg.train_label_max_length == 128
    assert cfg.valid_loss_label_max_length == 128
    assert cfg.final_eval_prediction_max_length == 25
    assert cfg.final_eval_reference_max_length == 25
    assert snapshot["step5_final_eval"]["official_profile"] == "paper_greedy_25"
