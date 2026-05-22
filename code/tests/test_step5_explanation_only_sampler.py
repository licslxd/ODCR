from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from odcr_core.config_resolver import resolve_config
from odcr_core.step5_pool_sampler import resolve_step5_pool_source, validate_step5_formal_sample_plan_for_source


def test_step5_sampler_uses_explanation_pools_and_route_explainer() -> None:
    _cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="step5",
        task_id=2,
        set_overrides=[],
        dry_run=True,
        from_step4="1",
    )
    source = resolve_step5_pool_source(step4_run_dir=ROOT / "runs" / "step4" / "task2" / "1", repo_root=ROOT)
    report = validate_step5_formal_sample_plan_for_source(
        source,
        sampler_config=snapshot["step5_sampler"],
        batch_candidates_config=snapshot["step5_batch_candidates"],
        tuning_config=snapshot["step5_tuning"],
        task_head="explanation",
        bounded_max_rows=128,
        no_write=True,
    )
    head = report["heads"]["explanation"]
    assert head["components"]["cf"]["route_filter"]["route_column"] == "route_explainer"
    assert head["task_decoupled_policy"]["rating_training"]["enabled"] is False
    assert report["formal_namespace_write"] is False
