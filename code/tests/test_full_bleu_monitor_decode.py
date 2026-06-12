"""训练期 full BLEU 监控 decode override：greedy 仅经 cfg_override 生效，不污染主 decode / eval 指纹。"""
import os
import sys
import unittest
from dataclasses import replace

import torch

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)

from config import FinalTrainingConfig, build_full_bleu_monitor_cfg_override  # noqa: E402
from config import parse_full_bleu_decode_strategy  # noqa: E402
from executors.decode_controller import (  # noqa: E402
    DECODE_BACKEND_KV_SAFE,
    GenerateConfig,
    coerce_generate_cfg_override,
    merge_generate_config_with_override,
)
from executors.step5_engine import (  # noqa: E402
    _step5_should_validate_epoch,
    _step5_train_monitor_generation_batch_size,
)
from odcr_core.bleu_runtime import build_explanation_bleu_rows_for_indices, mainline_monitor_full_valid_ddp  # noqa: E402
from odcr_core.gather_schema import GatheredBatch  # noqa: E402
from odcr_core.step5_innovation import for_test_default_step5_innovation_config  # noqa: E402


def _minimal_final_cfg(*, full_bleu_decode_strategy: str = "greedy") -> FinalTrainingConfig:
    from config import FullBleuEvalResolved

    return FinalTrainingConfig(
        task_idx=1,
        auxiliary="a",
        target="b",
        scenario="legacy_scenario",
        direction="test",
        task_profile_id="unit_profile",
        task_profile_key="unit_profile_key",
        profile_isolation_hash="unit-profile-hash",
        preset_name="step5",
        world_size=1,
        sources=(("x", "y"),),
        learning_rate=1e-3,
        scheduler_initial_lr=1e-3,
        initial_lr=1e-3,
        epochs=1,
        max_epochs=1,
        validate_every_epochs=1,
        max_grad_norm=0.5,
        tokenizer_max_length=48,
        evidence_max_length=48,
        valid_batch_size=8,
        valid_micro_batch_size=8,
        valid_per_gpu_batch_size=8,
        valid_global_batch_size=8,
        valid_forward_micro_batch_size=8,
        test_per_gpu_batch_size=8,
        test_forward_micro_batch_size=8,
        validation_microbatch_accumulation=False,
        validation_memory_policy="single_forward",
        step5_validation_mode="clean_memory32",
        formal_entry_E4_validation_required=False,
        valid_loss_components_json="{}",
        train_batch_size=8,
        global_batch_size=8,
        batch_size_global=8,
        batch_size=8,
        per_device_train_batch_size=8,
        per_gpu_batch_size=8,
        effective_global_batch_size=8,
        batch_semantics_version="odcr_no_accum/1",
        grad_accum_removed=True,
        num_proc=1,
        max_parallel_cpu=1,
        hardware_preset_name=None,
        dataloader_num_workers_train=0,
        dataloader_num_workers_valid=0,
        dataloader_num_workers_test=0,
        dataloader_prefetch_factor_train=None,
        dataloader_prefetch_factor_valid=None,
        dataloader_prefetch_factor_test=None,
        pin_memory=False,
        persistent_workers=False,
        non_blocking_h2d=False,
        min_lr_ratio=0.1,
        lr_scheduler="none",
        scheduler_type="none",
        warmup_epochs=0.0,
        odcr_warmup_steps=None,
        odcr_warmup_ratio=None,
        optimizer_config_json="{}",
        precision_config_json="{}",
        tokenizer_config_json="{}",
        evidence_config_json="{}",
        scheduler_config_json="{}",
        valid_batch_config_json="{}",
        scenario_profile_json="{}",
        task_profile_config_json="{}",
        backup_profiles_config_json="{}",
        exploration_profiles_config_json="{}",
        prefetcher_config_json="{}",
        checkpoint_policy_config_json="{}",
        quality_gate_config_json="{}",
        grad_finite_config_json="{}",
        diagnostic_eval_config_json="{}",
        cross_rank_structured_gather_config_json="{}",
        memory_config_json="{}",
        timing_config_json="{}",
        cache_policy_config_json="{}",
        objective_drift_config_json="{}",
        recovery_config_json="{}",
        phase_loss_schedule_config_json="{}",
        eval_batch_size=8,
        min_epochs=1,
        train_min_epochs=1,
        early_stop_patience=1,
        early_stop_patience_full=1,
        early_stop_patience_loss=1,
        full_bleu_eval_resolved=FullBleuEvalResolved(mode="off", every_epochs=None, enabled=False),
        checkpoint_metric="valid_loss",
        dual_bleu_eval=False,
        bleu4_max_samples=64,
        quick_eval_max_samples=64,
        coef=1.0,
        explainer_loss_weight=1e-3,
        full_bleu_decode_strategy=full_bleu_decode_strategy,
        decode_strategy="uncertainty_low_temp_top_k",
        generate_temperature=0.2,
        gap_threshold=0.35,
        prefix_greedy_steps=4,
        decode_top_k=5,
    )


class _OneRowDs:
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            "user_idx": torch.tensor(1, dtype=torch.long),
            "item_idx": torch.tensor(2, dtype=torch.long),
            "rating": torch.tensor(4.0, dtype=torch.float32),
            "explanation_idx": torch.tensor([11, 12], dtype=torch.long),
            "raw_ref_text": "reference text",
            "domain_idx": torch.tensor(0, dtype=torch.long),
            "sample_id": torch.tensor(0, dtype=torch.long),
            "exp_sample_weight": torch.tensor(1.0, dtype=torch.float32),
        }


class _RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_cfg_override = None
        self.decode_strategy = "uncertainty_low_temp_top_k"

    def _make_generate_config(self) -> GenerateConfig:
        return GenerateConfig(strategy=str(self.decode_strategy))

    def gather(self, batch, device, *, non_blocking_h2d=False):
        user_idx, item_idx, rating, tgt_output, domain_idx, sample_id, exp_sample_weight, raw_ref_text = batch
        bsz = int(user_idx.size(0))
        ones = torch.ones((bsz,), dtype=torch.float32, device=device)
        ids = torch.ones((bsz, 1), dtype=torch.long, device=device)
        return GatheredBatch(
            user_idx=user_idx.to(device),
            item_idx=item_idx.to(device),
            rating=rating.to(device),
            tgt_input=tgt_output.to(device),
            tgt_output=tgt_output.to(device),
            domain_idx=domain_idx.to(device),
            sample_id=sample_id.to(device),
            exp_sample_weight=exp_sample_weight.to(device),
            route_scorer_mask=ones,
            route_explainer_mask=ones,
            uncertainty_score=torch.zeros((bsz,), dtype=torch.float32, device=device),
            confidence_bucket=ones,
            content_anchor_score=ones,
            style_anchor_score=ones,
            evidence_features=torch.ones((bsz, 8), dtype=torch.float32, device=device),
            content_evidence_ids=ids,
            style_evidence_ids=ids,
            domain_style_anchor_ids=ids,
            local_style_hint_ids=ids,
            polarity_ids=ids,
            raw_ref_text=raw_ref_text,
        )

    def generate(self, user_idx, item_idx, domain_idx, *, cfg_override=None, ccv_control_packet=None):
        self.last_cfg_override = cfg_override
        bsz = int(user_idx.size(0))
        return (torch.full((bsz, 2), 7, dtype=torch.long, device=user_idx.device),)


class _Tok:
    def __call__(self, text, add_special_tokens=False, truncation=False, verbose=False, **kwargs):
        return {"input_ids": [1, 2]}

    def decode(self, ids, skip_special_tokens=True):
        return "x"

    def batch_decode(self, ids, skip_special_tokens=True):
        t = ids.detach().cpu()
        return ["x"] * int(t.size(0))


class TestFullBleuMonitorDecode(unittest.TestCase):
    def test_step5_validation_cadence_keeps_final_epoch(self) -> None:
        self.assertFalse(_step5_should_validate_epoch(1, 20, 5))
        self.assertTrue(_step5_should_validate_epoch(5, 20, 5))
        self.assertFalse(_step5_should_validate_epoch(19, 20, 5))
        self.assertTrue(_step5_should_validate_epoch(20, 20, 5))
        self.assertTrue(_step5_should_validate_epoch(3, 3, 20))
        self.assertTrue(_step5_should_validate_epoch(1, 3, 0))

    def test_train_monitor_uses_train_validation_generation_batch(self) -> None:
        cfg = replace(
            _minimal_final_cfg(),
            ddp_world_size=2,
            eval_batch_size=2048,
            valid_global_batch_size=64,
            valid_batch_size=64,
            valid_per_gpu_batch_size=32,
            valid_forward_micro_batch_size=16,
        )

        self.assertEqual(_step5_train_monitor_generation_batch_size(cfg), 16)
        self.assertEqual(
            _step5_train_monitor_generation_batch_size(replace(cfg, valid_forward_micro_batch_size=0)),
            32,
        )
        self.assertEqual(
            _step5_train_monitor_generation_batch_size(
                replace(cfg, valid_forward_micro_batch_size=0, valid_per_gpu_batch_size=0)
            ),
            32,
        )

    def test_build_full_bleu_monitor_override_greedy(self) -> None:
        cfg = _minimal_final_cfg(full_bleu_decode_strategy="greedy")
        ov = build_full_bleu_monitor_cfg_override(cfg)
        self.assertEqual(ov["strategy"], "greedy")
        self.assertEqual(float(ov["tail_temperature"]), -1.0)
        self.assertGreaterEqual(int(ov["no_repeat_ngram_size"]), 3)
        self.assertGreaterEqual(int(ov["min_len"]), 4)

    def test_build_full_bleu_monitor_override_inherit(self) -> None:
        cfg = _minimal_final_cfg(full_bleu_decode_strategy="inherit")
        ov = build_full_bleu_monitor_cfg_override(cfg)
        self.assertEqual(ov["strategy"], "uncertainty_low_temp_top_k")
        self.assertEqual(float(ov["tail_temperature"]), -1.0)
        self.assertEqual(int(ov["top_k"]), 5)
        self.assertEqual(ov["decode_backend"], DECODE_BACKEND_KV_SAFE)
        self.assertEqual(ov["decode_run_context"], "train_time_eval")

    def test_train_time_eval_backend_override_from_cfg(self) -> None:
        cfg = _minimal_final_cfg(full_bleu_decode_strategy="inherit")
        cfg = replace(cfg, train_time_eval_decode_backend="legacy_full_recompute")
        ov = build_full_bleu_monitor_cfg_override(cfg)
        self.assertEqual(ov["decode_backend"], "legacy_full_recompute")

    def test_merge_override_only_strategy(self) -> None:
        base = GenerateConfig(strategy="nucleus", temperature=0.5, top_p=0.95)
        m = merge_generate_config_with_override(base, {"strategy": "greedy"})
        self.assertEqual(m.strategy, "greedy")
        self.assertEqual(m.temperature, 0.5)
        self.assertEqual(m.top_p, 0.95)

    def test_coerce_generate_cfg_override_dict_no_pollution(self) -> None:
        m = _RecordingModel()
        gc1 = coerce_generate_cfg_override(m._make_generate_config(), {"strategy": "greedy"})
        self.assertIsNotNone(gc1)
        self.assertEqual(gc1.strategy, "greedy")
        gc2 = coerce_generate_cfg_override(m._make_generate_config(), None)
        self.assertIsNone(gc2)
        self.assertEqual(m.decode_strategy, "uncertainty_low_temp_top_k")

    def test_full_bleu_path_passes_override_quick_does_not(self) -> None:
        m = _RecordingModel()
        ds = _OneRowDs()
        tok = _Tok()
        step5_innov_cfg = for_test_default_step5_innovation_config()
        build_explanation_bleu_rows_for_indices(
            m,
            tok,
            torch.device("cpu"),
            ds,
            [0],
            batch_size=1,
            rank=0,
            logger=None,
            dataloader_num_workers=0,
            dataloader_prefetch_factor=None,
            cfg_override={"strategy": "greedy"},
            step5_innov_cfg=step5_innov_cfg,
            non_blocking_h2d=False,
        )
        self.assertIsNotNone(m.last_cfg_override)
        self.assertEqual(m.last_cfg_override.get("strategy"), "greedy")

        m2 = _RecordingModel()
        build_explanation_bleu_rows_for_indices(
            m2,
            tok,
            torch.device("cpu"),
            ds,
            [0],
            batch_size=1,
            rank=0,
            logger=None,
            dataloader_num_workers=0,
            dataloader_prefetch_factor=None,
            step5_innov_cfg=step5_innov_cfg,
            non_blocking_h2d=False,
        )
        self.assertIsNone(m2.last_cfg_override)

    def test_mainline_monitor_does_not_require_retired_recommend(self) -> None:
        m = _RecordingModel()
        ds = _OneRowDs()
        tok = _Tok()
        step5_innov_cfg = for_test_default_step5_innovation_config()

        score, bundle = mainline_monitor_full_valid_ddp(
            m,
            ds,
            tokenizer=tok,
            device=torch.device("cpu"),
            rank=0,
            world_size=1,
            batch_size=1,
            dataloader_num_workers=0,
            dataloader_prefetch_factor=None,
            cfg_override={"strategy": "greedy"},
            step5_innov_cfg=step5_innov_cfg,
            non_blocking_h2d=False,
        )

        self.assertIsInstance(score, float)
        self.assertIsNotNone(bundle)
        self.assertEqual(bundle.get("rmse_rating"), 0.0)
        self.assertEqual(bundle.get("mae_rating"), 0.0)

    def test_generation_fingerprint_chunk_excludes_monitor_decode_key(self) -> None:
        """generation_semantic_fingerprint 的 _fp_gen 构造段不得包含 full_bleu_decode_strategy。"""
        from pathlib import Path

        p = Path(__file__).resolve().parents[1] / "odcr_core" / "config_resolver.py"
        text = p.read_text(encoding="utf-8")
        i0 = text.find("gen_fp = fingerprint(")
        self.assertNotEqual(i0, -1)
        i1 = text.find("runtime_fp = fingerprint", i0)
        self.assertNotEqual(i1, -1)
        chunk = text[i0:i1]
        self.assertNotIn("full_bleu_decode_strategy", chunk)

    def test_parse_full_bleu_decode_strategy(self) -> None:
        self.assertEqual(parse_full_bleu_decode_strategy("Greedy"), "greedy")
        self.assertEqual(parse_full_bleu_decode_strategy("inherit"), "inherit")
        with self.assertRaises(ValueError):
            parse_full_bleu_decode_strategy("beam")


if __name__ == "__main__":
    unittest.main()
