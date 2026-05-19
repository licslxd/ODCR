from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.index_contract import (  # noqa: E402
    INDEX_CONTRACT_FILENAME,
    INDEX_CONTRACT_SCHEMA_VERSION,
    ODCR_ROUTING_TRAIN_CSV,
    build_step4_export_lineage,
    refresh_index_contract_train_csv_fingerprint,
)
from odcr_core.step4_export_validator import STEP4_EXPORT_MANIFEST  # noqa: E402
from odcr_core.step4_pool_exports import export_step4_pool_exports  # noqa: E402
from odcr_core.step5_export_loader import Step5ExportLoaderError, load_step5_pool_train_table  # noqa: E402
from odcr_core.step5_pool_sampler import (  # noqa: E402
    STEP5_POOL_DISTRIBUTION_REPORT,
    STEP5_POOL_MANIFEST,
    STEP5_SAMPLE_PLAN_MANIFEST,
    STEP5_POOLS_DIRNAME,
    STEP5_SAMPLING_CONTRACT,
    read_step5_sample_plan_shard,
    resolve_step5_pool_source,
    sample_effective_epochs_from_pools,
    write_step5_sample_plan,
)
from odcr_core.step5_prompt_templates import default_prompt_registry  # noqa: E402


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sampler_config() -> dict:
    return {
        "enabled": True,
        "contract_source": "step4_pool_manifest",
        "effective_epoch_enabled": True,
        "seed": 3407,
        "rotate_across_epochs": True,
        "full_audit_default_allowed": False,
        "legacy_gold_heavy_exports_allowed": False,
        "auto_budget": {
            "enabled": True,
            "capacity_basis": "balanced_capacity",
            "budget_multipliers": {"small": 0.60, "medium": 0.80, "full": 1.00, "large": 1.20},
            "min_steps_per_effective_epoch": 1,
            "preferred_steps_per_effective_epoch": [1, 10],
            "max_steps_per_effective_epoch": 100,
            "max_replacement_rate": 0.20,
        },
        "step5A": {
            "default_candidate": "medium",
            "target_gold_ratio": 1.00,
            "aux_gold_ratio": 0.00,
            "cf_ratio": 0.00,
            "target_gold_tier_mix": {"high": 0.60, "medium": 0.40},
            "aux_gold_tier_mix": {"high": 0.70, "medium": 0.30},
            "cf_tier_mix": {"high": 0.00, "medium": 1.00, "low_weighted": 0.00},
            "aux_gold_weight": 0.5,
            "cf_high_weight": 1.2,
            "cf_medium_weight": 0.8,
            "cf_low_weight": 0.2,
        },
        "step5B": {
            "default_candidate": "medium",
            "target_gold_ratio": 0.40,
            "aux_gold_ratio": 0.20,
            "cf_ratio": 0.40,
            "target_gold_tier_mix": {"high": 0.60, "medium": 0.40},
            "aux_gold_tier_mix": {"high": 0.70, "medium": 0.30},
            "cf_tier_mix": {"high": 0.00, "medium": 0.00, "low_weighted": 1.00},
            "aux_gold_weight": 0.5,
            "cf_high_weight": 1.2,
            "cf_medium_weight": 0.9,
            "cf_low_weight": 0.3,
        },
        "task_decoupled_policy": {
            "schema_version": "odcr_step5_task_decoupled_policy/1",
            "enabled": True,
            "step5A": {
                "branch": "scorer_clean",
                "train_components": {"target_gold": 1.0, "aux_gold": 0.0, "cf": 0.0},
                "forbid_aux_cf_in_scorer_loss": True,
                "forbid_generation": True,
                "forbid_big_model": True,
                "scorer_init_required": True,
                "scorer_init_source": "step3_transplant",
                "distillation_enabled": False,
                "distillation_weight": 0.0,
            },
            "step5B": {
                "branch": "explainer_rich",
                "train_components": {"target_gold": "optional_anchor", "aux_gold": ">0", "cf": ">0"},
                "use_big_model": True,
                "allow_target_anchor": True,
                "target_anchor_role": "optional_target_explanation_anchor_not_rating_supervision",
            },
        },
        "epochs": {"max_effective_epochs": 2, "early_stopping_patience": 1, "pilot_fraction_candidates": [0.2, 0.5]},
    }


def _batch_candidates_config() -> dict:
    return {
        "ddp_world_size": 2,
        "fsdp_zero_policy": "not_introduced",
        "selected_default": "B2",
        "candidates": [{"id": "B2", "per_gpu_batch_size": 1, "global_batch_size": 2, "role": "unit"}],
    }


def _tuning_config() -> dict:
    return {"enabled": True, "batch_candidate": "B2", "selected_budget_candidate": "full"}


def _row(origin: str, i: int, **kw) -> dict:
    aux_cf = origin == "aux_cf"
    defaults = {
        "user": f"u{i}",
        "item": f"it{i}",
        "rating": 4.0,
        "review": "review text",
        "explanation": "clean useful explanation text",
        "content_evidence": "content evidence",
        "content_anchor_score": 0.8,
        "polarity_anchor": "positive",
        "domain_style_anchor": "target" if origin == "target_gold" else "auxiliary",
        "local_style_residual_hint": "hint",
        "style_evidence": "style evidence",
        "style_anchor_score": 0.7,
        "evidence_quality_prior": 0.8,
        "preprocess_route_scorer_prior": 0,
        "preprocess_route_explainer_prior": 0,
        "domain": "target" if origin == "target_gold" else "auxiliary",
        "sample_id": i,
        "entropy": 0.0,
        "rating_target": 4.0,
        "rating_counterfactual": 4.0,
        "rating_delta": 0.0,
        "rating_stability_score": 0.94,
        "shared_latent_similarity": 0.9,
        "specific_latent_shift": 0.4,
        "content_retention_score": 0.93,
        "style_shift_score": 0.20 if aux_cf else 0.0,
        "cf_reliability_score": 0.70 if aux_cf else 1.0,
        "uncertainty_score": 0.10 if aux_cf else 0.0,
        "entropy_score": 0.0,
        "text_quality_score": 0.95,
        "confidence_bucket": 2,
        "route_scorer": 1,
        "route_explainer": 1,
        "route_reason_scorer": "rcr_scorer_clean" if aux_cf else "gold",
        "route_reason_explainer": "rcr_explainer_rich" if aux_cf else "gold",
        "train_keep": 1,
        "sample_weight_hint": 1.0,
        "sample_origin": origin,
        "is_counterfactual": int(aux_cf),
        "clean_text": "clean useful explanation text",
        "clean_changed": 0,
        "html_entity_hit": 0,
        "bad_tail_hit": 0,
        "bad_tail_types": "",
        "template_hit": 0,
        "template_count": 0,
        "template_hard_drop_hit": 0,
        "template_downweighted": 0,
        "noisy_tail_downweighted": 0,
        "short_fragment_hit": 0,
        "repeat_tail_hit": 0,
        "train_drop_reason": "",
        "user_idx_global": i + 1,
        "item_idx_global": i + 11,
    }
    defaults.update(kw)
    return defaults


def _write_run(root: Path) -> Path:
    run = root / "runs" / "step4" / "task2" / "1"
    (run / "meta").mkdir(parents=True)
    rows = [
        _row("target_gold", 1),
        _row("target_gold", 2, template_hit=1, text_quality_score=0.7),
        _row("target_gold", 3, clean_text="", explanation="", bad_tail_hit=1),
        _row("aux_gold", 4),
        _row("aux_gold", 5, template_hit=1, text_quality_score=0.7),
        _row("aux_gold", 6, clean_text="", explanation="", bad_tail_hit=1),
        _row("aux_cf", 7),
        _row("aux_cf", 8, route_scorer=0, rating_stability_score=0.75, content_retention_score=0.76, uncertainty_score=0.40),
        _row("aux_cf", 9, route_scorer=0, route_explainer=0, rating_stability_score=0.70, content_retention_score=0.65, style_shift_score=0.12),
        _row("aux_cf", 10, clean_text="", explanation="", uncertainty_score=0.99, text_quality_score=0.0),
    ]
    df = pd.DataFrame(rows)
    df.to_csv(run / ODCR_ROUTING_TRAIN_CSV, index=False)
    lineage = build_step4_export_lineage(
        task_id=2,
        auxiliary_domain="AM_Movies",
        target_domain="AM_CDs",
        step3_checkpoint_lineage_hash="lineage",
        step4_rcr_config={"fixture": True},
        step4_run="1",
        frozen_step3_lineage={
            "upstream_step3_run_id": "2",
            "step3_checkpoint_path": "runs/step3/task2/2/model/best.pth",
            "step3_checkpoint_hash": "h",
            "step3_stage_status_hash": "s",
            "step3_eval_handoff_hash": "e",
        },
    )
    contract = refresh_index_contract_train_csv_fingerprint(
        {
            "schema_version": INDEX_CONTRACT_SCHEMA_VERSION,
            "embed_dim": 1024,
            "nuser_global": 100,
            "nitem_global": 100,
            "backbones": {
                "sentence_embed": {
                    "model_id": "m",
                    "local_dir": "/tmp/m",
                    "family": "bge_large_en",
                    "hidden_size": 1024,
                    "dual_channel": True,
                }
            },
            "step4_export_lineage": lineage,
        },
        str(run / ODCR_ROUTING_TRAIN_CSV),
    )
    (run / INDEX_CONTRACT_FILENAME).write_text(json.dumps(contract), encoding="utf-8")
    (run / STEP4_EXPORT_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": "odcr_step4_train_table/1.2",
                "row_counts": {"total_rows": len(df), "by_sample_origin": {"target_gold": 3, "aux_gold": 3, "aux_cf": 4}},
                "step4_export_lineage": lineage,
            }
        ),
        encoding="utf-8",
    )
    (run / "meta" / "stage_status.json").write_text(
        json.dumps({"schema_version": "odcr_stage_status/1", "stage": "step4", "task": 2, "task_id": 2, "run_id": "1"}),
        encoding="utf-8",
    )
    return run


class Step4Step5PoolSamplerTest(unittest.TestCase):
    def test_step4_pool_exports_gold_quality_cf_tiers_and_contract(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = _write_run(root)
            result = export_step4_pool_exports(
                repo_root=root,
                task=2,
                from_run="1",
                pool_config={"enabled": True, "output_dir_name": "step5_pools", "chunk_rows": 2},
                sampler_config=_sampler_config(),
            )
            self.assertTrue((run / STEP5_POOLS_DIRNAME / STEP5_POOL_MANIFEST).is_file())
            self.assertTrue((run / STEP5_POOLS_DIRNAME / STEP5_SAMPLING_CONTRACT).is_file())
            report = json.loads((run / STEP5_POOLS_DIRNAME / STEP5_POOL_DISTRIBUTION_REPORT).read_text(encoding="utf-8"))
            target_counts = report["gold_quality_counts"]["target_gold"]
            self.assertEqual(target_counts["high"] + target_counts["medium"], 2)
            self.assertGreaterEqual(target_counts["medium"], 1)
            self.assertGreaterEqual(report["cf_tier_counts"]["step5A"]["medium"], 1)
            self.assertGreaterEqual(report["cf_tier_counts"]["step5A"]["reject"], 1)
            self.assertTrue(result["status"]["step5_pool_exports_ready"])

    def test_step4_old_dedicated_exports_marked_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = _write_run(root)
            (run / "step5_exports").mkdir()
            for name in ("step5A_scorer_train.parquet", "step5B_explainer_train.parquet", "odcr_routing_full_audit.parquet"):
                (run / "step5_exports" / name).write_bytes(b"legacy")
            export_step4_pool_exports(
                repo_root=root,
                task=2,
                from_run="1",
                pool_config={"enabled": True, "output_dir_name": "step5_pools", "chunk_rows": 2},
                sampler_config=_sampler_config(),
            )
            legacy = json.loads((run / "step5_exports" / "legacy_old_filter_exports_status.json").read_text(encoding="utf-8"))
            self.assertTrue(legacy["not_default_step5_train"])
            self.assertTrue(legacy["gold_heavy_warning"])

    def test_step5_loader_requires_pool_manifest_and_rejects_legacy_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = _write_run(root)
            with self.assertRaisesRegex(Step5ExportLoaderError, "requires Step4 RCR pools"):
                load_step5_pool_train_table(
                    run / ODCR_ROUTING_TRAIN_CSV,
                    index_contract_path=run / INDEX_CONTRACT_FILENAME,
                    manifest_path=run / STEP4_EXPORT_MANIFEST,
                    sampler_config=_sampler_config(),
                    mode="bounded",
                    verify_sha256=False,
                )

    def test_step5_sampler_ratios_rotation_replacement_and_budget(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = _write_run(root)
            export_step4_pool_exports(
                repo_root=root,
                task=2,
                from_run="1",
                pool_config={"enabled": True, "output_dir_name": "step5_pools", "chunk_rows": 2},
                sampler_config=_sampler_config(),
            )
            source = resolve_step5_pool_source(step4_run_dir=run, repo_root=root)
            sampled_a = sample_effective_epochs_from_pools(
                source,
                sampler_config=_sampler_config(),
                batch_candidates_config=_batch_candidates_config(),
                tuning_config=_tuning_config(),
                mode="bounded",
                task_head="step5A",
                bounded_max_rows=4,
            )
            counts_a = sampled_a.stats["epoch_reports"][0]["heads"]["step5A"]["actual_counts"]
            self.assertEqual(counts_a["target_gold"], sampled_a.stats["effective_samples_per_epoch"])
            self.assertEqual(counts_a["aux_gold"], 0)
            self.assertEqual(counts_a["cf"], 0)
            repl = sampled_a.stats["epoch_reports"][0]["heads"]["step5A"]["components"]["target_gold"]["replacement_rate_by_tier"]
            self.assertIn("high", repl)
            self.assertEqual(sampled_a.stats["epoch_reports"][0]["heads"]["step5A"]["replacement_rate"], 0.0)
            self.assertEqual(sampled_a.stats["effective_samples_per_epoch"], len(sampled_a.train_df))
            sampled_b = sample_effective_epochs_from_pools(
                source,
                sampler_config=_sampler_config(),
                batch_candidates_config=_batch_candidates_config(),
                tuning_config=_tuning_config(),
                mode="formal_train",
                task_head="step5B",
            )
            epochs = sampled_b.train_df["effective_epoch"].value_counts().to_dict()
            self.assertEqual(set(epochs), {0, 1})
            e0 = list(sampled_b.train_df.loc[sampled_b.train_df["effective_epoch"] == 0, "sample_id"])
            e1 = list(sampled_b.train_df.loc[sampled_b.train_df["effective_epoch"] == 1, "sample_id"])
            self.assertNotEqual(e0, e1)

    def test_step5_sample_plan_prebuilt_deterministic_and_rank_sharded(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = _write_run(root)
            export_step4_pool_exports(
                repo_root=root,
                task=2,
                from_run="1",
                pool_config={"enabled": True, "output_dir_name": "step5_pools", "chunk_rows": 2},
                sampler_config=_sampler_config(),
            )
            source = resolve_step5_pool_source(step4_run_dir=run, repo_root=root)
            sampled1 = sample_effective_epochs_from_pools(
                source,
                sampler_config=_sampler_config(),
                batch_candidates_config=_batch_candidates_config(),
                tuning_config=_tuning_config(),
                mode="bounded",
                task_head="step5A",
                bounded_max_rows=8,
            )
            sampled2 = sample_effective_epochs_from_pools(
                source,
                sampler_config=_sampler_config(),
                batch_candidates_config=_batch_candidates_config(),
                tuning_config=_tuning_config(),
                mode="bounded",
                task_head="step5A",
                bounded_max_rows=8,
            )
            self.assertEqual(list(sampled1.train_df["sample_id"]), list(sampled2.train_df["sample_id"]))
            out_dir = root / "AI_analysis" / "test_artifacts" / "sample_plan"
            manifest = write_step5_sample_plan(
                out_dir,
                train_df=sampled1.train_df,
                stats=sampled1.stats,
                source_summary=sampled1.source.to_summary(),
                task_head="step5A",
                world_size=2,
                source_table={"scope": "unit"},
            )
            self.assertTrue((out_dir / "sample_plan" / STEP5_SAMPLE_PLAN_MANIFEST).is_file())
            shard0 = read_step5_sample_plan_shard(out_dir, rank=0)
            shard1 = read_step5_sample_plan_shard(out_dir, rank=1)
            self.assertEqual(len(shard0) + len(shard1), len(sampled1.train_df))
            self.assertEqual(set(shard0["step5_rank"]), {0})
            self.assertEqual(set(shard1["step5_rank"]), {1})
            for field in (
                "sample_id",
                "step5_plan_path",
                "step5_plan_row_group",
                "step5_sample_label",
                "step5_template_id",
                "sampler_tier",
                "sample_origin",
            ):
                self.assertIn(field, shard0.columns)
                self.assertIn(field, manifest["required_columns"])
            self.assertFalse((root / "runs" / "step5" / "task2" / "latest.json").exists())

    def test_step5_prompt_registry_minimal_deterministic_and_ccv_independent(self) -> None:
        reg = default_prompt_registry()
        manifest = reg.manifest()
        self.assertEqual(manifest["template_count"], 4)
        rendered1 = reg.render(sample={"sample_id": 7}, task_head="step5A", sample_origin="target_gold", seed=3407)
        rendered2 = reg.render(sample={"sample_id": 7}, task_head="step5A", sample_origin="target_gold", seed=3407)
        self.assertEqual(rendered1["step5_prompt_instance_id"], rendered2["step5_prompt_instance_id"])
        with self.assertRaises(KeyError):
            reg.render(sample={"sample_id": 7}, task_head="step5A", sample_origin="aux_cf", seed=3407)
        valid = reg.render(sample={"sample_id": 7}, task_head="step5B", sample_origin="target_gold", seed=1, split="valid")
        self.assertEqual(valid["step5_prompt_mode"], "fixed_canonical")
        self.assertIn("CCV", manifest["does_not_replace"])
        for tmpl in manifest["templates"]:
            self.assertNotIn("route_scorer", tmpl["text"])
            self.assertNotIn("sample_weight_hint", tmpl["text"])

    def test_step5_no_formal_namespace_pollution_from_pool_loader(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = _write_run(root)
            export_step4_pool_exports(
                repo_root=root,
                task=2,
                from_run="1",
                pool_config={"enabled": True, "output_dir_name": "step5_pools", "chunk_rows": 2},
                sampler_config=_sampler_config(),
            )
            load_step5_pool_train_table(
                run / ODCR_ROUTING_TRAIN_CSV,
                index_contract_path=run / INDEX_CONTRACT_FILENAME,
                manifest_path=run / STEP4_EXPORT_MANIFEST,
                sampler_config=_sampler_config(),
                batch_candidates_config=_batch_candidates_config(),
                tuning_config=_tuning_config(),
                mode="bounded",
                bounded_max_rows=8,
                verify_sha256=False,
            )
            self.assertFalse((root / "runs" / "step5" / "task2" / "latest.json").exists())


if __name__ == "__main__":
    unittest.main()
