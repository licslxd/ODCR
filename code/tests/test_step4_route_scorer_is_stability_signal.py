from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from odcr_core.config_resolver import resolve_config
from odcr_core.step4_dedicated_exports import FIELD_REASONS


def test_route_scorer_is_stability_audit_signal() -> None:
    _cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="step4",
        task_id=2,
        set_overrides=[],
        dry_run=True,
        from_step3="2",
    )
    route_cfg = snapshot["step4_rcr"]["route_scorer"]
    route_text = FIELD_REASONS["route_scorer"].lower()
    assert "rating-stability" in route_text
    assert "audit control" in route_text
    assert route_cfg["min_rating_stability"] >= 0.0
    assert snapshot["step4_rcr"]["route_explainer"]["min_style_shift"] >= 0.0


def test_route_explainer_remains_step5_primary_route() -> None:
    _cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="step5",
        task_id=2,
        set_overrides=[],
        dry_run=True,
        from_step4="1",
    )
    sampler = snapshot["step5_sampler"]
    assert sampler["route_primary"] == "route_explainer"
    assert sampler["components"]["cf"] == "enabled"
    assert snapshot["step5_sampler"]["task_decoupled_policy"]["mode"] == "explanation_only"
