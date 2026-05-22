from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from odcr_core.config_resolver import resolve_config
from odcr_core.step5_explanation_handoff import build_step5_explanation_handoff, validate_step5_explanation_handoff


def test_explanation_handoff_embeds_step3_rating_source() -> None:
    _cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="step5",
        task_id=2,
        set_overrides=[],
        dry_run=True,
        from_step4="1",
    )
    handoff = build_step5_explanation_handoff(
        task=2,
        run_id="1_1",
        checkpoint="runs/step5/task2/1_1/model/best_explanation.pth",
        explanation_metrics={"valid_explainer_loss": 1.23},
        generation_config={"max_explanation_length": 25},
        ccv_fca_report={"ccv": "enabled", "fca": "enabled"},
        route_explainer_stats={"route_explainer_rows": 1},
        rating_source=snapshot["rating_source"],
    )
    validated = validate_step5_explanation_handoff(handoff, repo_root=ROOT)
    assert validated["mode"] == "explanation_only"
    assert validated["rating_source"]["type"] == "step3_accepted_scorer"
    assert validated["no_rating_training_performed"] is True
