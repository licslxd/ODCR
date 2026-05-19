from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.config_schema import OneControlConfigError  # noqa: E402
from odcr_core.config_resolver import resolve_config  # noqa: E402
from odcr_core.manifests import build_formal_source_table_snapshot  # noqa: E402


SELECTED = "A_RATIO_0+B_RATIO_0+A_CF_MIX_FORMAL_HIGH_ONLY+B_CF_MIX_FORMAL_HIGH_MEDIUM+TG_MIX_0+AG_MIX_0+LR_1e-3+W0"


def _resolve(head: str = "step5A"):
    return resolve_config(
        config_path=REPO_ROOT / "configs" / "odcr.yaml",
        command="step5",
        task_id=2,
        set_overrides=[],
        dry_run=True,
        from_step4="1",
        run_id="auto",
        step5_head=head,
    )


def test_formal_step5_candidate_ids_and_mixes_are_one_control() -> None:
    cfg, _sources, snapshot = _resolve("step5A")
    active = snapshot["step5_formal_active_candidate"]
    assert cfg.step5_selected_tuning_candidate == SELECTED
    assert active["step5A_cf_mix_id"] == "A_CF_MIX_FORMAL_HIGH_ONLY"
    assert active["step5A_cf_mix"] == {"high": 1.0, "medium": 0.0, "low_weighted": 0.0}
    assert active["step5B_cf_mix_id"] == "B_CF_MIX_FORMAL_HIGH_MEDIUM"
    assert active["step5B_cf_mix"] == {"high": 0.588235, "medium": 0.411765, "low_weighted": 0.0}
    assert "A_CF_MIX_1" not in snapshot["selected_tuning_candidate"]
    assert active["low_weighted_policy"] == "disabled_for_mainline"


def test_retired_bounded_cf_mix_ids_cannot_be_formal_selected() -> None:
    old = "A_RATIO_0+B_RATIO_0+A_CF_MIX_1+B_CF_MIX_1+TG_MIX_0+AG_MIX_0+LR_1e-3+W0"
    with pytest.raises(OneControlConfigError):
        resolve_config(
            config_path=REPO_ROOT / "configs" / "odcr.yaml",
            command="step5",
            task_id=2,
            set_overrides=[f"step5.tuning.selected_tuning_candidate={old}"],
            dry_run=True,
            from_step4="1",
            run_id="auto",
            step5_head="step5A",
        )


def test_source_table_records_active_candidate_values_and_step4_lineage_role() -> None:
    _cfg, _sources, snapshot = _resolve("step5A")
    source_table = build_formal_source_table_snapshot(snapshot)
    records = {row["key"]: row for row in source_table["records"]}
    assert records["step5_formal_active_candidate.step5A_cf_mix_id"]["value"] == "A_CF_MIX_FORMAL_HIGH_ONLY"
    assert records["step5_formal_active_candidate.step5B_cf_mix_id"]["value"] == "B_CF_MIX_FORMAL_HIGH_MEDIUM"
    assert records["step5_formal_active_candidate.step4_sampling_contract_role"]["value"] == "pool_lineage_only"
    assert records["step5_formal_active_candidate.active_sampler_source"]["value"].startswith("configs/odcr.yaml")


def test_step4_sampling_contract_old_mix_is_lineage_only_not_active() -> None:
    _cfg, _sources, snapshot = _resolve("step5A")
    contract = json.loads(
        (REPO_ROOT / "runs" / "step4" / "task2" / "1" / "step5_pools" / "step5_sampling_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["step5A"]["cf_tier_mix"] == {"high": 0.7, "medium": 0.3, "low_weighted": 0.0}
    assert snapshot["step5_sampler"]["step5A"]["cf_tier_mix"] == {"high": 1.0, "medium": 0.0, "low_weighted": 0.0}
    assert snapshot["step5_formal_active_candidate"]["step4_sampling_contract_role"] == "pool_lineage_only"
