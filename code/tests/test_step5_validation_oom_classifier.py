from __future__ import annotations

from pathlib import Path

from odcr_core.manifests import _extract_failure_root_signature


def test_validation_oom_classifier_reports_epoch_end_validation(tmp_path: Path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "full.log").write_text(
        "\n".join(
            [
                "[Tokenize] completed fingerprint=abc cache_dir=/tmp/cache",
                "[Step] global_step=400",
                "[Timing] valid_loss_forward start epoch=1",
                "  File code/executors/step5_engine.py, line 4713, in validModel",
                "  File code/executors/step5_engine.py, line 2088, in forward",
                "    logits = out.logits.float()",
                "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.41 GiB",
                "validation",
            ]
        ),
        encoding="utf-8",
    )

    details = _extract_failure_root_signature(
        meta=meta,
        latest_error="tokenization_cache should not win",
        repo_root=tmp_path,
        checkpoint_path=None,
    )

    assert details["failure_phase"] == "epoch_end_validation"
    assert details["failure_type"] == "validation_forward_oom"
    assert details["root_cause"] == "step5A_validation_materialized_explainer_logits_with_oversized_valid_batch"
