from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from odcr_core.config_resolver import resolve_config
from odcr_core.rating_source import RATING_SOURCE_POLICY_TYPE, RATING_SOURCE_TYPE, validate_rating_source


def test_step3_accepted_rating_source_validates_from_one_control() -> None:
    _cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="step5",
        task_id=2,
        set_overrides=[],
        dry_run=True,
        from_step4="1",
    )
    rating_source = validate_rating_source(snapshot["rating_source"], repo_root=ROOT)
    assert rating_source["type"] == RATING_SOURCE_TYPE
    assert rating_source["status"] == "ok"
    assert rating_source["checkpoint_hash"] == "9089ac53b138c12ba1260370aed3d637b305f7f7f6a98a7bcbc7721eb5559017"
    assert rating_source["metrics"]["valid"] == {"mae": 0.575, "rmse": 0.8473}
    assert rating_source["metrics"]["test"] == {"mae": 0.5764, "rmse": 0.8494}
    assert snapshot["rating_source"]["policy_type"] == RATING_SOURCE_POLICY_TYPE
    assert snapshot["rating_source"]["source"] == "upstream_step3_eval_handoff"


def test_step5_rating_source_is_task_local_not_fixed_task2() -> None:
    _cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="step5",
        task_id=5,
        set_overrides=[],
        dry_run=True,
        from_step4="1",
    )
    rating_source = validate_rating_source(snapshot["rating_source"], repo_root=ROOT)
    assert rating_source["type"] == RATING_SOURCE_TYPE
    assert rating_source["task"] == 5
    assert rating_source["run"] == 1
    assert rating_source["checkpoint_hash"] == "60a1042dd9f0d7d9dfc3da5501419fd4d0a17f520bc4493d15f13bc6c441fe70"
    assert rating_source["metrics"]["valid"] == {"mae": 0.5789, "rmse": 0.8782}
    assert rating_source["metrics"]["test"] == {"mae": 0.575, "rmse": 0.8715}
