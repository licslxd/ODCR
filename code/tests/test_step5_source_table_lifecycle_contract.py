from __future__ import annotations

from pathlib import Path

from odcr_core.config_resolver import resolve_config
from odcr_core.manifests import build_formal_source_table_snapshot


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_step5_source_table_lifecycle_records_validation_e4_contract() -> None:
    _cfg, _sources, snapshot = resolve_config(
        config_path=REPO_ROOT / "configs" / "odcr.yaml",
        command="step5",
        task_id=2,
        set_overrides=[],
        dry_run=True,
        from_step4="1",
        run_id="auto",
        step5_head="step5A",
    )
    records = {row["key"]: row["value"] for row in build_formal_source_table_snapshot(snapshot)["records"] if "value" in row}

    assert records["step5A_validation_scorer_only"] is True
    assert records["valid_per_gpu_batch_size"] == 192
    assert records["valid_forward_micro_batch_size"] == 192
    assert records["validation_e4_evidence_id"] == "pending_E4_gpu_shard_forward_bounded_formal_entry_with_validation"
    assert records["validation_oom_guard_status"] == "pass"
