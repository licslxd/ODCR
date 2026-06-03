from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from odcr import build_parser
from odcr_core.config_resolver import resolve_config
from odcr_core.racer_c1 import run_racer_c1


def _resolved_task2():
    return resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="step5",
        task_id=2,
        set_overrides=[],
        dry_run=True,
        from_step4="1",
        mode="train_only",
    )


def test_racer_c1_config_enters_resolved_payload() -> None:
    cfg, _sources, snapshot = _resolved_task2()
    racer = snapshot["step5_racer_c1"]
    assert racer["method_name"] == "RACER-C1"
    assert racer["paper_method_name"] == "RACER"
    assert racer["output_stage"] == "racer_c1"
    assert racer["legacy_generator_policy"] == "deleted_not_available"
    assert racer["train"]["global_batch_size"] == racer["train"]["per_gpu_batch_size"] * cfg.ddp_world_size
    assert racer["evidence_pool"]["schema_version"] == "odcr_racer_c1_evidence_pool/2"
    assert racer["evidence_pool"]["source_split"] == "train"
    assert racer["guardrails"]["forbid_big_model_generator_call"] is True
    assert racer["contrastive"]["positive_weight"]["causal_content"] > 0
    assert racer["contrastive"]["positive_weight"]["style_shortcut"] < 0
    assert racer["retrieval"]["top_k"] == 3
    assert racer["retrieval"]["diagnostic_prediction_policy"] == "top1_clean_explanation_25"
    assert racer["retrieval"]["official_prediction_policy"] == "composer_minimal_rewrite"
    assert racer["composer"]["enabled"] is True
    assert racer["composer"]["official_prediction_source"] == "composed"
    assert racer["composer"]["forbid_exact_copy"] is True
    assert "diagnostics/content_style_split_stats.json" in racer["logging"]["required_files"]
    assert "diagnostics/cf_anchor_stats.json" in racer["logging"]["required_files"]
    assert "diagnostics/contrastive_role_distribution.json" in racer["logging"]["required_files"]
    assert "diagnostics/copy_lcs_stats.json" in racer["logging"]["required_files"]
    assert "diagnostics/cross_domain_evidence_distribution.json" in racer["logging"]["required_files"]
    assert "RACER-C1" in cfg.step5_racer_c1_config_json


def test_racer_c1_prepare_dry_run_does_not_write_or_call_generator() -> None:
    cfg, _sources, snapshot = _resolved_task2()
    result = run_racer_c1(cfg, snapshot, mode="prepare", run_id="1", dry_run=True)
    assert result["status"] == "dry_run_ok"
    assert result["written"] is False
    assert result["paper_method_name"] == "RACER"
    assert result["method_name"] == "RACER-C1"
    assert result["paths"]["run_root"].endswith("runs/racer_c1/task2/1")
    assert result["legacy_cleanup_policy"]["formal_path_invokes_big_model"] is False
    assert "meta/epoch_metrics.jsonl" in result["planned_relative_outputs"]
    assert "predictions/valid_composed_predictions.jsonl" in result["planned_relative_outputs"]
    assert "metrics/valid_top1_diagnostic_paper_greedy_25.json" in result["planned_relative_outputs"]
    assert result["cache_identity"]["source_split"] == "train"


def test_racer_c1_cli_is_one_control_entrypoint() -> None:
    args = build_parser().parse_args(["--dry-run", "racer-c1", "--task", "2", "--mode", "prepare", "--from-step4", "1"])
    assert args.command == "racer-c1"
    assert args.task == 2
    assert args.mode == "prepare"
    assert args.from_step4 == "1"
