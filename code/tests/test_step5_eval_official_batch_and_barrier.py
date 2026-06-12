from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.config_resolver import resolve_config  # noqa: E402
from odcr_core.runners import _step5_decode_cli_args  # noqa: E402


def test_step5_official_eval_uses_step5_eval_valid_batch() -> None:
    cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="eval",
        task_id=1,
        set_overrides=["eval.split=valid"],
        dry_run=True,
        from_step5="2_3",
    )
    assert cfg.global_eval_batch_size == cfg.valid_global_batch_size == 256
    assert cfg.eval_per_gpu_batch_size == cfg.valid_per_gpu_batch_size == 128
    assert snapshot["eval"]["eval_batch_size"] == 256
    assert snapshot["eval"]["eval_batch_size_role_for_step5_train_validation"] == (
        "step5_official_eval_uses_step5_eval_batch"
    )


def test_step5_official_eval_uses_step5_eval_test_batch() -> None:
    cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="eval",
        task_id=1,
        set_overrides=["eval.split=test"],
        dry_run=True,
        from_step5="2_3",
    )
    assert cfg.global_eval_batch_size == cfg.test_per_gpu_batch_size * cfg.ddp_world_size
    assert cfg.eval_per_gpu_batch_size == cfg.test_per_gpu_batch_size == 128
    assert snapshot["eval"]["eval_batch_size"] == 256


def test_step5_eval_explicit_large_valid_batch_is_not_capped_by_train_batch() -> None:
    cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="eval",
        task_id=1,
        set_overrides=[
            "eval.split=valid",
            "eval.profiles.paper_greedy_25.eval_batch_size=2048",
            "step5.tasks.1.eval.valid_per_gpu_batch_size=1024",
            "step5.tasks.1.eval.valid_batch_size=2048",
            "step5.tasks.1.eval.valid_forward_micro_batch_size=1024",
        ],
        dry_run=True,
        from_step5="2_3",
    )
    assert cfg.eval_per_gpu_batch_size > cfg.per_gpu_batch_size
    assert cfg.global_eval_batch_size == cfg.valid_global_batch_size == 2048
    assert cfg.eval_per_gpu_batch_size == cfg.valid_per_gpu_batch_size == 1024
    assert snapshot["eval"]["eval_batch_size"] == 2048


def test_step5_task1_b32_flow_is_active_default() -> None:
    cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="step5",
        task_id=1,
        set_overrides=[],
        dry_run=True,
        from_step4="2",
    )
    assert cfg.train_batch_size == cfg.global_batch_size == 64
    assert cfg.per_gpu_batch_size == 32
    assert cfg.epochs == 1
    assert cfg.train_label_max_length == 128
    assert snapshot["step5_batch_candidates"]["selected_default"] == "B64"
    assert snapshot["step5_tuning"]["batch_candidate"] == "B32"
    assert snapshot["step5_eval"]["valid_global_batch_size"] == 256
    assert snapshot["field_sources"]["step5_eval"].startswith("step5.tasks.1.eval")
    assert snapshot["eval"]["decode"]["decode_strategy"] == "greedy"
    assert snapshot["eval"]["decode"]["repetition_penalty"] == 1.0
    assert "generate_temperature" not in snapshot["eval"]["decode"]
    assert "generate_top_p" not in snapshot["eval"]["decode"]
    assert "tail_temperature" not in snapshot["eval"]["decode"]
    assert "tail_top_p" not in snapshot["eval"]["decode"]


def test_step5_active_eval_config_only_exposes_paper_greedy_profile() -> None:
    text = (ROOT / "configs" / "odcr.yaml").read_text(encoding="utf-8")
    assert "paper_greedy_25:" in text
    assert "balanced_2gpu:" not in text
    assert "candidate_nucleus_2gpu:" not in text
    assert "decode_strategy: uncertainty_low_temp_top_k" not in text
    assert "max_explanation_length: 36" not in text
    assert "generate_temperature:" not in text
    assert "generate_top_p:" not in text
    assert "tail_temperature:" not in text
    assert "tail_top_p:" not in text


def test_step5_official_greedy_cli_does_not_pass_sampling_args() -> None:
    cfg, _sources, _snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="eval",
        task_id=1,
        set_overrides=["eval.split=valid"],
        dry_run=True,
        from_step5="2_3",
    )
    args = _step5_decode_cli_args(cfg)
    assert "--decode-strategy" in args
    assert args[args.index("--decode-strategy") + 1] == "greedy"
    assert "--generate-temperature" not in args
    assert "--generate-top-p" not in args


def test_step5_eval_only_has_no_final_nccl_barrier_after_cpu_metrics() -> None:
    text = (CODE_DIR / "executors" / "step5_engine.py").read_text(encoding="utf-8")
    cpu_tail = text[text.rindex("# Eval-only rank0 computes CPU text metrics") : text.rindex("finally:")]
    assert "if not eval_only:\n            dist.barrier()" in cpu_tail
    without_guarded_final = cpu_tail.replace("if not eval_only:\n            dist.barrier()", "")
    assert "\n        dist.barrier()\n" not in without_guarded_final


def test_step5_eval_runner_has_no_pipeline_local_variable_prints() -> None:
    text = (CODE_DIR / "odcr_core" / "runners.py").read_text(encoding="utf-8")
    run_eval_src = text[text.index("def run_eval(") :]
    assert "cfg3" not in run_eval_src
    assert "cfg5" not in run_eval_src


def test_step5_eval_only_build_skips_step4_pool_train_loader() -> None:
    text = (CODE_DIR / "executors" / "step5_engine.py").read_text(encoding="utf-8")
    build_src = text[
        text.index("def build_odcr_ddp_artefacts(") : text.index("def _metrics_final_dict_from_rows(")
    ]
    assert (
        'if command == "train":\n'
        "        try:\n"
        "            train_table = load_step5_pool_train_table("
    ) in build_src
    eval_branch = build_src[build_src.index("    if eval_only:") : build_src.index("    else:", build_src.index("    if eval_only:"))]
    assert "load_step5_pool_train_table(" not in eval_branch
    assert (
        "_mm_train_file = (\n"
        "        train_table.raw_index_min_max or _idx_mm(train_table.audit_raw_df)\n"
        "        if train_table is not None\n"
        "        else None\n"
        "    )"
    ) in build_src


def test_step5_official_eval_generation_uses_forward_microbatch() -> None:
    text = (CODE_DIR / "executors" / "step5_engine.py").read_text(encoding="utf-8")
    eval_model_src = text[text.index("def evalModel(") : text.index("def _resolve_odcr_profile_paths(")]
    artifact_writer_src = text[
        text.index("def _rank0_write_eval_artifacts(") : text.index("explanation_metrics = {")
    ]
    assert "eval_forward_micro_batch_size" in eval_model_src
    assert "for start in range(0, full_bsz, micro):" in eval_model_src
    assert "_slice_gathered_batch(gb, start, min(start + micro, full_bsz))" in eval_model_src
    assert "_odcr_eval_forward_micro_batch_size" in text
    assert "_odcr_eval_split_label" in text
    assert "def _eval_rows_local(" in text
    assert 'setattr(args, "_odcr_eval_forward_micro_batch_size", int(_eval_forward_micro or 0))' in text
    assert 'eval_forward_micro_batch_size=int(getattr(args, "_odcr_eval_forward_micro_batch_size", 0) or 0) or None' in text
    assert '"current_row_review_policy": _eval_current_row_review_policy' in artifact_writer_src
    assert '"detail": _perf' in artifact_writer_src
    assert '"summary": _perf_summary' in artifact_writer_src
    assert 'if isinstance(_perf_value, (int, float)) and not isinstance(_perf_value, bool):' in artifact_writer_src
    assert '"current_row_review_policy",\n                    "tokenize_cache_time"' not in artifact_writer_src
    retired_loader_name = "".join(["_load", "_", "rev", "iew", "_by", "_", "sample", "_id"])
    retired_kw_name = "".join(["rev", "iew", "_", "key", "words"])
    assert retired_loader_name not in text
    assert retired_kw_name not in text
