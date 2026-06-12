from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import sys
import time

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from odcr_core import step5_export_loader as loader
from odcr_core.step5_pool_sampler import Step5PoolSampleResult


class _PoolSource:
    def __init__(self, root: Path) -> None:
        self.pool_dir = root / "step5_pools"
        self.pool_dir.mkdir(parents=True)
        self.manifest_path = self.pool_dir / "step5_pool_manifest.json"
        self.sampling_contract_path = self.pool_dir / "step5_sampling_contract.json"
        self.manifest_path.write_text('{"schema_version":"pool"}', encoding="utf-8")
        self.sampling_contract_path.write_text('{"schema_version":"contract"}', encoding="utf-8")
        self.manifest = {"schema_version": "pool", "source_row_counts": {"total_rows": 2}}
        self.sampling_contract = {"schema_version": "contract"}
        self.source_full_export = root / "odcr_routing_train.csv"

    def to_summary(self) -> dict:
        return {
            "schema_version": "dummy_pool_source/1",
            "manifest_path": str(self.manifest_path),
            "sampling_contract_path": str(self.sampling_contract_path),
        }


def _dummy_source(root: Path) -> SimpleNamespace:
    export = root / "odcr_routing_train.csv"
    export.write_text("user_idx_global,item_idx_global,clean_text\n0,0,a\n", encoding="utf-8")
    return SimpleNamespace(
        export_path=export,
        index_contract={},
        required_columns=("user_idx_global", "item_idx_global", "clean_text"),
        header_columns=("user_idx_global", "item_idx_global", "clean_text"),
    )


def _sampler_config(
    *,
    target_gold: float,
    aux_gold: float,
    cf: float,
    seed: int = 3407,
    max_effective_epochs: int = 1,
) -> dict:
    return {
        "enabled": True,
        "contract_source": "step4_pool_manifest",
        "effective_epoch_enabled": True,
        "seed": seed,
        "rotate_across_epochs": True,
        "auto_budget": {
            "enabled": True,
            "capacity_basis": "balanced_capacity",
            "budget_multipliers": {"small": 0.6, "medium": 0.8, "full": 1.0, "large": 1.2},
            "min_steps_per_effective_epoch": 1,
            "preferred_steps_per_effective_epoch": [1, 4],
            "max_steps_per_effective_epoch": 4,
            "max_replacement_rate": 0.2,
        },
        "explanation": {
            "default_candidate": "medium",
            "target_gold_ratio": target_gold,
            "aux_gold_ratio": aux_gold,
            "cf_ratio": cf,
            "target_gold_tier_mix": {"high": 1.0, "medium": 0.0},
            "aux_gold_tier_mix": {"high": 1.0, "medium": 0.0},
            "cf_tier_mix": {"high": 1.0, "medium": 0.0, "low_weighted": 0.0},
            "aux_gold_weight": 0.5,
            "cf_high_weight": 1.2,
            "cf_medium_weight": 0.9,
            "cf_low_weight": 0.3,
        },
        "epochs": {"max_effective_epochs": max_effective_epochs},
    }


def _batch_config() -> dict:
    return {"selected_default": "B1", "candidates": [{"id": "B1", "per_gpu_batch_size": 1, "global_batch_size": 1}]}


def _tuning_config(
    *,
    ratio_id: str,
    target_gold: float,
    aux_gold: float,
    cf: float,
    effective_samples: int,
) -> dict:
    return {
        "selected_tuning_candidate": f"{ratio_id}+STEP5_CF_MIX_FORMAL_HIGH_MEDIUM+TG_MIX_0+AG_MIX_0+LR_1e-3+W0",
        "selected_budget_candidate": "medium",
        "batch_candidate": "B1",
        "effective_samples": {"explanation": effective_samples},
        "optimizer_steps": {"explanation": effective_samples},
        "ratio_candidates": {
            "explanation": [{"id": ratio_id, "target_gold": target_gold, "aux_gold": aux_gold, "cf": cf}]
        },
        "cf_tier_mix_candidates": {
            "explanation": [
                {"id": "STEP5_CF_MIX_FORMAL_HIGH_MEDIUM", "high": 1.0, "medium": 0.0, "low_weighted": 0.0}
            ]
        },
        "gold_tier_mix_candidates": {
            "target_gold": [{"id": "TG_MIX_0", "high": 1.0, "medium": 0.0}],
            "aux_gold": [{"id": "AG_MIX_0", "high": 1.0, "medium": 0.0}],
        },
        "innovation_weight_candidates": [
            {"id": "W0", "fca": 0.08, "explainer_loss_weight": 1.0, "ccv_numeric_control_weight": 1.0}
        ],
    }


def test_step5_pool_train_cache_ignores_optimizer_decode_lineage(monkeypatch, tmp_path: Path) -> None:
    calls = {"sample": 0}
    csv_source = _dummy_source(tmp_path)
    pool_source = _PoolSource(tmp_path)

    def fake_sample(*_args, **_kwargs) -> Step5PoolSampleResult:
        calls["sample"] += 1
        df = pd.DataFrame(
            {
                "user_idx_global": [0, 1],
                "item_idx_global": [2, 3],
                "clean_text": ["good text", "better text"],
                "sample_id": [0, 1],
                "sampler_component": ["target_gold", "cf"],
                "sampler_tier": ["high", "high"],
                "step5_prompt_template_id": ["a", "b"],
                "effective_epoch": [0, 0],
                "route_explainer": [1, 1],
            }
        )
        return Step5PoolSampleResult(
            train_df=df,
            audit_raw_df=df.head(1),
            source=pool_source,
            raw_row_count=2,
            filtered_row_count=2,
            stats={"mode": "formal_train", "sampler_plan_time_s": 1.0},
        )

    monkeypatch.setattr(loader, "validate_step5_export_source", lambda *a, **k: csv_source)
    monkeypatch.setattr(loader, "resolve_step5_pool_source", lambda *a, **k: pool_source)
    monkeypatch.setattr(loader, "validate_split_indices", lambda *a, **k: None)
    monkeypatch.setattr(loader, "sample_effective_epochs_from_pools", fake_sample)

    sampler_config = _sampler_config(target_gold=0.5, aux_gold=0.0, cf=0.5)
    batch_config = _batch_config()
    tuning_a = {
        **_tuning_config(
            ratio_id="STEP5_RATIO_MIXED",
            target_gold=0.5,
            aux_gold=0.0,
            cf=0.5,
            effective_samples=2,
        ),
        "lr_candidates": [0.001],
        "innovation_weight_candidates": [{"id": "W0"}],
    }
    tuning_b = {
        **tuning_a,
        "lr_candidates": [0.0002],
        "innovation_weight_candidates": [
            {"id": "W0", "fca": 0.08, "explainer_loss_weight": 9.0, "ccv_numeric_control_weight": 1.0}
        ],
    }

    first = loader.load_step5_pool_train_table(
        csv_source.export_path,
        cache_root=tmp_path / "cache",
        sampler_config=sampler_config,
        batch_candidates_config=batch_config,
        tuning_config=tuning_a,
        cache_enabled=True,
    )
    second = loader.load_step5_pool_train_table(
        csv_source.export_path,
        cache_root=tmp_path / "cache",
        sampler_config=sampler_config,
        batch_candidates_config=batch_config,
        tuning_config=tuning_b,
        cache_enabled=True,
    )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert calls["sample"] == 1
    assert second.cache_dir == first.cache_dir
    assert len(second.train_df) == 2


def test_step5_pool_train_cache_rebuilds_when_sample_identity_changes(monkeypatch, tmp_path: Path) -> None:
    calls = {"sample": 0}
    csv_source = _dummy_source(tmp_path)
    pool_source = _PoolSource(tmp_path)

    def fake_sample(*_args, **_kwargs) -> Step5PoolSampleResult:
        calls["sample"] += 1
        df = pd.DataFrame(
            {
                "user_idx_global": [calls["sample"]],
                "item_idx_global": [7],
                "clean_text": ["cached text"],
                "sample_id": [calls["sample"]],
                "sampler_component": ["target_gold"],
                "sampler_tier": ["high"],
                "step5_prompt_template_id": ["a"],
                "effective_epoch": [0],
                "route_explainer": [1],
            }
        )
        return Step5PoolSampleResult(
            train_df=df,
            audit_raw_df=df.head(1),
            source=pool_source,
            raw_row_count=1,
            filtered_row_count=1,
            stats={"mode": "formal_train"},
        )

    monkeypatch.setattr(loader, "validate_step5_export_source", lambda *a, **k: csv_source)
    monkeypatch.setattr(loader, "resolve_step5_pool_source", lambda *a, **k: pool_source)
    monkeypatch.setattr(loader, "validate_split_indices", lambda *a, **k: None)
    monkeypatch.setattr(loader, "sample_effective_epochs_from_pools", fake_sample)

    base = _sampler_config(target_gold=1.0, aux_gold=0.0, cf=0.0)
    changed = _sampler_config(target_gold=1.0, aux_gold=0.0, cf=0.0, seed=99)
    for sampler in (base, changed):
        loader.load_step5_pool_train_table(
            csv_source.export_path,
            cache_root=tmp_path / "cache",
            sampler_config=sampler,
            batch_candidates_config=_batch_config(),
            tuning_config=_tuning_config(
                ratio_id="STEP5_RATIO_TARGET_ONLY",
                target_gold=1.0,
                aux_gold=0.0,
                cf=0.0,
                effective_samples=1,
            ),
            cache_enabled=True,
        )

    assert calls["sample"] == 2


def test_step5_pool_loader_rejects_target_only_rows_when_resolved_intent_is_mixed(monkeypatch, tmp_path: Path) -> None:
    csv_source = _dummy_source(tmp_path)
    pool_source = _PoolSource(tmp_path)

    def fake_target_only_sample(*_args, **_kwargs) -> Step5PoolSampleResult:
        df = pd.DataFrame(
            {
                "user_idx_global": [0, 1],
                "item_idx_global": [2, 3],
                "clean_text": ["target text", "target text 2"],
                "sample_id": [0, 1],
                "sampler_component": ["target_gold", "target_gold"],
                "sampler_tier": ["high", "high"],
                "step5_prompt_template_id": ["a", "a"],
                "effective_epoch": [0, 0],
                "route_explainer": [1, 1],
            }
        )
        return Step5PoolSampleResult(
            train_df=df,
            audit_raw_df=df.head(1),
            source=pool_source,
            raw_row_count=2,
            filtered_row_count=2,
            stats={"mode": "formal_train", "max_effective_epochs": 1, "effective_samples_per_epoch": 2},
        )

    monkeypatch.setattr(loader, "validate_step5_export_source", lambda *a, **k: csv_source)
    monkeypatch.setattr(loader, "resolve_step5_pool_source", lambda *a, **k: pool_source)
    monkeypatch.setattr(loader, "validate_split_indices", lambda *a, **k: None)
    monkeypatch.setattr(loader, "sample_effective_epochs_from_pools", fake_target_only_sample)

    with pytest.raises(loader.Step5ExportLoaderError, match="component counts"):
        loader.load_step5_pool_train_table(
            csv_source.export_path,
            cache_root=tmp_path / "cache",
            sampler_config=_sampler_config(target_gold=0.5, aux_gold=0.0, cf=0.5),
            batch_candidates_config=_batch_config(),
            tuning_config=_tuning_config(
                ratio_id="STEP5_RATIO_MIXED",
                target_gold=0.5,
                aux_gold=0.0,
                cf=0.5,
                effective_samples=2,
            ),
            cache_enabled=True,
        )


def test_step5_pool_train_cache_concurrent_writers_use_disjoint_temp_dirs(monkeypatch, tmp_path: Path) -> None:
    csv_source = _dummy_source(tmp_path)
    pool_source = _PoolSource(tmp_path)

    def fake_sample(*_args, **_kwargs) -> Step5PoolSampleResult:
        time.sleep(0.05)
        df = pd.DataFrame(
            {
                "user_idx_global": [0],
                "item_idx_global": [1],
                "clean_text": ["cached text"],
                "sample_id": [0],
                "sampler_component": ["target_gold"],
                "sampler_tier": ["high"],
                "step5_prompt_template_id": ["a"],
                "effective_epoch": [0],
                "route_explainer": [1],
            }
        )
        return Step5PoolSampleResult(
            train_df=df,
            audit_raw_df=df.head(1),
            source=pool_source,
            raw_row_count=1,
            filtered_row_count=1,
            stats={"mode": "formal_train"},
        )

    monkeypatch.setattr(loader, "validate_step5_export_source", lambda *a, **k: csv_source)
    monkeypatch.setattr(loader, "resolve_step5_pool_source", lambda *a, **k: pool_source)
    monkeypatch.setattr(loader, "validate_split_indices", lambda *a, **k: None)
    monkeypatch.setattr(loader, "sample_effective_epochs_from_pools", fake_sample)

    kwargs = {
        "cache_root": tmp_path / "cache",
        "sampler_config": _sampler_config(target_gold=1.0, aux_gold=0.0, cf=0.0),
        "batch_candidates_config": _batch_config(),
        "tuning_config": _tuning_config(
            ratio_id="STEP5_RATIO_TARGET_ONLY",
            target_gold=1.0,
            aux_gold=0.0,
            cf=0.0,
            effective_samples=1,
        ),
        "cache_enabled": True,
    }
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: loader.load_step5_pool_train_table(csv_source.export_path, **kwargs), range(2)))

    assert all(len(result.train_df) == 1 for result in results)
    assert (tmp_path / "cache").is_dir()
    assert not list((tmp_path / "cache").glob("*.tmp.*"))
