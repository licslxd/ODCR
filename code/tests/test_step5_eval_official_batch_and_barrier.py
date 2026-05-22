from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.config_resolver import resolve_config  # noqa: E402


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
    assert cfg.per_gpu_batch_size == 192
    assert cfg.global_eval_batch_size == cfg.valid_global_batch_size == 2048
    assert cfg.eval_per_gpu_batch_size == cfg.valid_per_gpu_batch_size == 1024
    assert snapshot["eval"]["eval_batch_size"] == 2048


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
