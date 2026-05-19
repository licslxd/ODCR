from __future__ import annotations

from pathlib import Path

from odcr_core.config_resolver import resolve_config
from odcr_core.manifests import build_formal_source_table_snapshot


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_step5a():
    return resolve_config(
        config_path=REPO_ROOT / "configs" / "odcr.yaml",
        command="step5",
        task_id=2,
        set_overrides=[],
        dry_run=True,
        from_step4="1",
        run_id="auto",
        step5_head="step5A",
    )


def test_step5_validation_batch_and_microbatch_are_one_control() -> None:
    cfg, _sources, snapshot = _resolve_step5a()

    assert cfg.per_gpu_batch_size == 224
    assert cfg.valid_per_gpu_batch_size == 192
    assert cfg.valid_global_batch_size == 384
    assert cfg.valid_forward_micro_batch_size == 192
    assert cfg.valid_forward_micro_batch_size <= cfg.per_gpu_batch_size
    assert cfg.validation_microbatch_accumulation is True
    assert cfg.validation_memory_policy == "microbatch_accumulate"
    assert cfg.step5A_validation_mode == "scorer_only"
    assert cfg.formal_entry_E4_validation_required is True
    assert cfg.old_eval_batch_2048_retired is True
    assert snapshot["eval"]["eval_batch_size"] == 2048
    assert snapshot["eval"]["eval_batch_size_role_for_step5_train_validation"] == "not_active"
    assert snapshot["step5_eval"]["valid_loss_components"]["step5A"] == [
        "scorer_rating_mse",
        "lci_weighted",
        "orthogonal_shared_specific",
    ]


def test_step5_source_table_records_validation_memory_contract() -> None:
    _cfg, _sources, snapshot = _resolve_step5a()
    source = build_formal_source_table_snapshot(snapshot)
    records = {row["key"]: row["value"] for row in source["records"] if "value" in row}

    assert records["train_per_gpu_batch_size"] == 224
    assert records["valid_per_gpu_batch_size"] == 192
    assert records["valid_forward_micro_batch_size"] == 192
    assert records["validation_memory_policy"] == "microbatch_accumulate"
    assert records["step5A_validation_scorer_only"] is True
    assert records["validation_flans_logits_materialized"] is False
    assert records["validation_oom_guard_status"] == "pass"
    assert records["validation_e4_evidence_id"] == "pending_E4_gpu_shard_forward_bounded_formal_entry_with_validation"
