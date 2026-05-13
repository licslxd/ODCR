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
from odcr_core.bleu_runtime import build_explanation_bleu_rows_for_indices  # noqa: E402
from odcr_core.gather_schema import GatheredBatch  # noqa: E402


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
        worker_profiles_config_json="{}",
        prefetcher_config_json="{}",
        checkpoint_policy_config_json="{}",
        quality_gate_config_json="{}",
        grad_finite_config_json="{}",
        numerical_stability_config_json="{}",
        diagnostic_eval_config_json="{}",
        cross_rank_structured_gather_config_json="{}",
        memory_config_json="{}",
        timing_config_json="{}",
        performance_candidates_config_json="{}",
        cache_policy_config_json="{}",
        objective_drift_config_json="{}",
        recovery_config_json="{}",
        phase_loss_schedule_config_json="{}",
        conflict_aware_config_json="{}",
        loss_gradient_conflict_probe_config_json="{}",
        adapter_gating_config_json="{}",
        paper_candidate_selection_config_json="{}",
        checkpoint_averaging_config_json="{}",
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

    def gather(self, batch, device):
        user_idx, item_idx, rating, tgt_output, domain_idx, sample_id, exp_sample_weight = batch
        return GatheredBatch(
            user_idx=user_idx.to(device),
            item_idx=item_idx.to(device),
            rating=rating.to(device),
            tgt_input=tgt_output.to(device),
            tgt_output=tgt_output.to(device),
            domain_idx=domain_idx.to(device),
            sample_id=sample_id.to(device),
            exp_sample_weight=exp_sample_weight.to(device),
        )

    def generate(self, user_idx, item_idx, domain_idx, *, cfg_override=None):
        self.last_cfg_override = cfg_override
        bsz = int(user_idx.size(0))
        return (torch.full((bsz, 2), 7, dtype=torch.long, device=user_idx.device),)


class _Tok:
    def batch_decode(self, ids, skip_special_tokens=True):
        t = ids.detach().cpu()
        return ["x"] * int(t.size(0))


class TestFullBleuMonitorDecode(unittest.TestCase):
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
        )
        self.assertIsNone(m2.last_cfg_override)

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
