from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import sys
import time

import pandas as pd


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
                "sampler_tier": ["high", "medium"],
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

    sampler_config = {"enabled": True, "seed": 3407, "explanation": {"target_gold_ratio": 1.0}}
    batch_config = {"selected_default": "B1", "candidates": [{"id": "B1", "global_batch_size": 1}]}
    tuning_a = {
        "selected_budget_candidate": "medium",
        "batch_candidate": "B1",
        "effective_samples": {"explanation": 2},
        "lr_candidates": [0.001],
        "innovation_weight_candidates": [{"id": "W0"}],
    }
    tuning_b = {
        **tuning_a,
        "lr_candidates": [0.0002],
        "innovation_weight_candidates": [{"id": "W9", "explainer_loss_weight": 9.0}],
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

    base = {"enabled": True, "seed": 3407, "explanation": {"target_gold_ratio": 1.0}}
    changed = {**base, "seed": 99}
    for sampler in (base, changed):
        loader.load_step5_pool_train_table(
            csv_source.export_path,
            cache_root=tmp_path / "cache",
            sampler_config=sampler,
            batch_candidates_config={"selected_default": "B1", "candidates": [{"id": "B1", "global_batch_size": 1}]},
            tuning_config={"selected_budget_candidate": "medium", "effective_samples": {"explanation": 1}},
            cache_enabled=True,
        )

    assert calls["sample"] == 2


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
        "sampler_config": {"enabled": True, "seed": 3407, "explanation": {"target_gold_ratio": 1.0}},
        "batch_candidates_config": {"selected_default": "B1", "candidates": [{"id": "B1", "global_batch_size": 1}]},
        "tuning_config": {"selected_budget_candidate": "medium", "effective_samples": {"explanation": 1}},
        "cache_enabled": True,
    }
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: loader.load_step5_pool_train_table(csv_source.export_path, **kwargs), range(2)))

    assert all(len(result.train_df) == 1 for result in results)
    assert (tmp_path / "cache").is_dir()
    assert not list((tmp_path / "cache").glob("*.tmp.*"))
