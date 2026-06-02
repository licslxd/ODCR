from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.config_resolver import resolve_config  # noqa: E402
from odcr_core.runners import _rerank_runner_cli_args, _step5_decode_cli_args  # noqa: E402


def test_step5_official_eval_uses_step5_eval_valid_batch() -> None:
    cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="eval",
        task_id=2,
        set_overrides=["eval.split=valid"],
        dry_run=True,
        from_step5="1_3",
    )
    assert cfg.old_eval_batch_2048_retired is True
    assert cfg.global_eval_batch_size == cfg.valid_global_batch_size == 2048
    assert cfg.eval_per_gpu_batch_size == cfg.valid_per_gpu_batch_size == 1024
    assert snapshot["eval"]["eval_batch_size"] == 2048
    assert snapshot["eval"]["eval_batch_size_role_for_step5_train_validation"] == (
        "step5_official_eval_uses_step5_eval_batch"
    )


def test_step5_official_eval_uses_step5_eval_test_batch() -> None:
    cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="eval",
        task_id=2,
        set_overrides=["eval.split=test"],
        dry_run=True,
        from_step5="1_3",
    )
    assert cfg.old_eval_batch_2048_retired is True
    assert cfg.global_eval_batch_size == cfg.test_per_gpu_batch_size * cfg.ddp_world_size
    assert cfg.eval_per_gpu_batch_size == cfg.test_per_gpu_batch_size == 1024
    assert snapshot["eval"]["eval_batch_size"] == 2048


def test_step5_eval_explicit_large_valid_batch_is_not_capped_by_train_batch() -> None:
    cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="eval",
        task_id=2,
        set_overrides=[
            "eval.split=valid",
            "step5.eval.valid_per_gpu_batch_size=1024",
            "step5.eval.valid_batch_size=2048",
            "step5.eval.valid_forward_micro_batch_size=1024",
        ],
        dry_run=True,
        from_step5="1_3",
    )
    assert cfg.per_gpu_batch_size == 64
    assert cfg.global_eval_batch_size == cfg.valid_global_batch_size == 2048
    assert cfg.eval_per_gpu_batch_size == cfg.valid_per_gpu_batch_size == 1024
    assert snapshot["eval"]["eval_batch_size"] == 2048


def test_step5_task2_1_18_flow_is_active_default() -> None:
    cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="step5",
        task_id=2,
        set_overrides=[],
        dry_run=True,
        from_step4="1",
    )
    assert cfg.train_batch_size == cfg.global_batch_size == 128
    assert cfg.per_gpu_batch_size == 64
    assert cfg.epochs == 1
    assert cfg.train_label_max_length == 128
    assert snapshot["step5_batch_candidates"]["selected_default"] == "B64"
    assert snapshot["step5_tuning"]["batch_candidate"] == "B64"
    assert snapshot["step5_eval"]["valid_global_batch_size"] == 2048
    assert snapshot["field_sources"]["step5_eval"].startswith("step5.tasks.2.eval")
    assert snapshot["eval"]["profile"] == "odcr_no_ref_k5_25"
    assert snapshot["eval"]["decode"]["decode_strategy"] == "nucleus"
    assert snapshot["eval"]["decode"]["repetition_penalty"] == 1.08
    assert snapshot["eval"]["decode"]["generate_temperature"] == 0.75
    assert snapshot["eval"]["decode"]["generate_top_p"] == 0.92
    assert snapshot["eval"]["decode"]["tail_temperature"] == 0.62
    assert snapshot["eval"]["decode"]["tail_top_p"] == 0.88


def test_step5_active_eval_config_exposes_only_official_profiles() -> None:
    text = (ROOT / "configs" / "odcr.yaml").read_text(encoding="utf-8")
    assert "paper_greedy_25:" in text
    assert "odcr_no_ref_k5_25:" in text
    assert "odcr_no_ref_fca_anchor:" in text
    assert "balanced_2gpu:" not in text
    assert "candidate_nucleus_2gpu:" not in text
    assert "rerank_quality_2gpu:" not in text
    assert "decode_strategy: uncertainty_low_temp_top_k" not in text
    assert "max_explanation_length: 36" not in text
    assert "generate_temperature: 0.75" in text
    assert "generate_top_p: 0.92" in text
    assert "tail_temperature: 0.62" in text
    assert "tail_top_p: 0.88" in text


def test_step5_official_greedy_cli_does_not_pass_sampling_args() -> None:
    cfg, _sources, _snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="eval",
        task_id=2,
        set_overrides=["eval.split=valid", "eval.profile=paper_greedy_25", "step5.final_eval.official_profile=paper_greedy_25"],
        dry_run=True,
        from_step5="1_18",
    )
    args = _step5_decode_cli_args(cfg)
    assert "--decode-strategy" in args
    assert args[args.index("--decode-strategy") + 1] == "greedy"
    assert "--generate-temperature" not in args
    assert "--generate-top-p" not in args


def test_step5_official_k5_cli_passes_reference_free_rerank_args() -> None:
    cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="eval",
        task_id=8,
        set_overrides=["eval.split=valid"],
        dry_run=True,
        from_step5="1_2",
    )
    args = _step5_decode_cli_args(cfg)
    assert cfg.command == "eval-rerank"
    assert cfg.eval_profile_id == "odcr_no_ref_k5_25"
    assert cfg.rerank_preset_id == "odcr_no_ref_fca_anchor"
    assert cfg.num_return_sequences == 5
    assert snapshot["step5_final_eval"]["official_profile"] == "odcr_no_ref_k5_25"
    assert snapshot["step5_final_eval"]["generation_input_policy"] == "history_conditioned_no_reference_evidence"
    assert snapshot["step5_final_eval"]["content_evidence_policy"] == "train_only_history"
    assert args[args.index("--decode-strategy") + 1] == "nucleus"
    assert args[args.index("--generate-temperature") + 1] == "0.75"
    assert args[args.index("--generate-top-p") + 1] == "0.92"


def test_step5_official_k5_rerank_test_passes_eval_split_transport() -> None:
    cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="eval",
        task_id=8,
        set_overrides=["eval.split=test"],
        dry_run=True,
        from_step5="1_2",
    )
    args = _rerank_runner_cli_args(cfg)
    assert cfg.command == "eval-rerank"
    assert cfg.eval_split == "test"
    assert snapshot["eval"]["split"] == "test"
    assert args[args.index("--eval-split") + 1] == "test"


def test_step5_train_cli_uses_training_dynamics_not_decode_profile() -> None:
    cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="step5",
        task_id=8,
        set_overrides=[],
        dry_run=True,
        from_step4="1",
    )
    payload = json.loads(cfg.effective_training_payload_json)
    row = payload["training_row"]
    assert row["label_smoothing"] == 0.01
    assert row["warmup_ratio"] == 0.03
    assert cfg.train_label_smoothing == 0.01
    assert cfg.train_warmup_ratio == 0.03
    assert snapshot["train"]["label_smoothing"] == 0.01
    assert snapshot["train"]["warmup_ratio"] == 0.03

    train_args = _step5_decode_cli_args(cfg, train=True)
    eval_args = _step5_decode_cli_args(cfg)
    assert train_args[train_args.index("--label-smoothing") + 1] == "0.01"
    assert eval_args[eval_args.index("--label-smoothing") + 1] == str(cfg.label_smoothing)


def test_step5_train_final_config_preserves_training_label_smoothing() -> None:
    text = (CODE_DIR / "executors" / "step5_engine.py").read_text(encoding="utf-8")
    base_final_src = text[text.index("base_final = replace(") : text.index("setattr(args, \"_odcr_eval_split_label\"")]
    assert 'label_smoothing=float(_dp_full["label_smoothing"]),' not in base_final_src
    assert 'label_smoothing=float(resolved.label_smoothing if not eval_only else _dp_full["label_smoothing"]),' in base_final_src


def test_step5_eval_only_has_no_final_nccl_barrier_after_cpu_metrics() -> None:
    text = (CODE_DIR / "executors" / "step5_engine.py").read_text(encoding="utf-8")
    cpu_tail = text[text.rindex("# Eval-only rank0 computes CPU text metrics") : text.rindex("finally:")]
    assert "if not eval_only:\n            dist.barrier()" in cpu_tail
    without_guarded_final = cpu_tail.replace("if not eval_only:\n            dist.barrier()", "")
    assert "\n        dist.barrier()\n" not in without_guarded_final


def test_step5_official_eval_generation_uses_forward_microbatch() -> None:
    text = (CODE_DIR / "executors" / "step5_engine.py").read_text(encoding="utf-8")
    eval_model_src = text[text.index("def evalModel(") : text.index("def _load_review_by_sample_id")]
    assert "eval_forward_micro_batch_size" in eval_model_src
    assert "for start in range(0, full_bsz, micro):" in eval_model_src
    assert "_slice_gathered_batch(gb, start, min(start + micro, full_bsz))" in eval_model_src
    assert "_odcr_eval_forward_micro_batch_size" in text
    assert "_odcr_eval_split_label" in text
    assert "eval_forward_micro_batch_size=int(getattr(args, \"_odcr_eval_forward_micro_batch_size\", 0) or 0) or None" in text
