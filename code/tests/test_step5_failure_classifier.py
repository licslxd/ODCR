from __future__ import annotations

from pathlib import Path

from odcr_core.manifests import _extract_failure_root_signature


def test_step5_without_grad_preflight_failure_classifies_as_ddp_preflight(tmp_path: Path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "full.log").write_text(
        "\n".join(
            [
                "Traceback (most recent call last):",
                "  File code/executors/step5_engine.py, line 2725, in run_step5_find_unused_parameters_preflight",
                "RuntimeError: Step5 find_unused_parameters=false preflight failed; trainable params without grad: "
                "recommender.linear1.weight, hidden2token.lora_A, domain_cross_attn.out_proj.lora_A",
                "[Tokenize] step5 cache key | fingerprint=abc | cache_dir=/tmp/cache",
            ]
        ),
        encoding="utf-8",
    )

    details = _extract_failure_root_signature(
        meta=meta,
        latest_error="tokenization_cache stale diagnostic should not win",
        repo_root=tmp_path,
        checkpoint_path=None,
    )

    assert details["failure_phase"] == "ddp_preflight"
    assert details["failure_type"] == "trainable_param_without_grad"
    assert details["root_cause"] == "step5_trainable_graph_mismatch"
    assert details["parameter_list"] == [
        "recommender.linear1.weight",
        "hidden2token.lora_A",
        "domain_cross_attn.out_proj.lora_A",
    ]


def test_step5_ema_deepcopy_failure_classifies_as_ema_init(tmp_path: Path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "errors.log").write_text(
        "\n".join(
            [
                "RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol",
                "  File code/executors/step5_engine.py, line 3120, in trainModel_ddp",
                "    ema_model = AveragedModel(_model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))",
                "AveragedModel deepcopy(model)",
                "[Tokenize] completed fingerprint=abc cache_dir=/tmp/cache",
            ]
        ),
        encoding="utf-8",
    )

    details = _extract_failure_root_signature(
        meta=meta,
        latest_error="tokenization_cache stale diagnostic should not win",
        repo_root=tmp_path,
        checkpoint_path=None,
    )

    assert details["failure_phase"] == "ema_init"
    assert details["failure_type"] == "model_deepcopy_non_leaf_tensor_after_preflight"
    assert details["root_cause"] == "step5_forward_cached_graph_tensors_persisted_before_ema_deepcopy"


def test_step5_epoch_end_validation_oom_beats_tokenization_cache_label(tmp_path: Path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "errors.log").write_text(
        "\n".join(
            [
                "[Tokenize] completed fingerprint=abc cache_dir=/tmp/cache",
                "[Step] global_step=400",
                "[Timing] valid_loss_forward start epoch=1",
                "Traceback (most recent call last):",
                "  File code/executors/step5_engine.py, line 4713, in validModel",
                "  File code/executors/step5_engine.py, line 2088, in forward",
                "    logits = out.logits.float()",
                "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.41 GiB",
                "validation forward failed",
            ]
        ),
        encoding="utf-8",
    )

    details = _extract_failure_root_signature(
        meta=meta,
        latest_error="tokenization_cache stale diagnostic should not win",
        repo_root=tmp_path,
        checkpoint_path=None,
    )

    assert details["failure_phase"] == "epoch_end_validation"
    assert details["failure_type"] == "validation_forward_oom"
    assert details["root_cause"] == "step5A_validation_materialized_explainer_logits_with_oversized_valid_batch"
    assert details["training_loop_started"] is True
