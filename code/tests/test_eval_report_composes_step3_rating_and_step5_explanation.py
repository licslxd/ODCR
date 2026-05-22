from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from odcr_core.config_resolver import resolve_config
from odcr_eval_metrics import compose_step3_rating_step5_explanation_report


def test_final_report_uses_step3_rating_and_step5_explanation() -> None:
    _cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="step5",
        task_id=2,
        set_overrides=[],
        dry_run=True,
        from_step4="1",
    )
    report = compose_step3_rating_step5_explanation_report(
        rating_source=snapshot["rating_source"],
        explanation_handoff={
            "mode": "explanation_only",
            "rating_source": snapshot["rating_source"],
            "explanation_metrics": {"valid_explainer_loss": 1.23},
        },
    )
    assert report["rating_source"] == "step3_accepted_scorer"
    assert report["rating_metrics_source"] == "step3_eval_handoff"
    assert report["rating_metrics"]["valid"] == {"mae": 0.575, "rmse": 0.8473}
    assert report["explanation_source"] == "step5_explanation_only"
    assert report["paper_rating_ready"] is True
    assert report["paper_explanation_ready"] is True
    assert report["step5_trains_rating"] is False
