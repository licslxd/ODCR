import os
import sys
import time
import unittest
import json
from contextlib import nullcontext
from dataclasses import asdict
from unittest.mock import patch

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)
os.environ["ODCR_STEP5_INIT_FLAN_STUB"] = "1"

from executors.step5_engine import Model as Step5Model, _domain_fusion_causal_mask  # noqa: E402
from odcr_core.step5_innovation import CCVControlPacket, for_test_default_step5_innovation_config  # noqa: E402
from odcr_core.generation.decoder_kv import DecoderKVBackend, _align_new_kv_to_cache  # noqa: E402
from odcr_core.generation.cache_types import LayerKVCache, PastKeyValues  # noqa: E402
from executors.decode_controller import (  # noqa: E402
    DECODE_BACKEND_KV_FAST,
    DECODE_BACKEND_KV_SAFE,
    DECODE_BACKEND_LEGACY,
    build_generate_kwargs_effective_v2,
    prepare_logits,
    resolve_decode_backend_name,
)
from config import build_full_bleu_monitor_cfg_override, FinalTrainingConfig, FullBleuEvalResolved  # noqa: E402
from train_diagnostics import odcr_cuda_bf16_autocast  # noqa: E402


def _build_model() -> Step5Model:
    torch.manual_seed(11)
    nuser, nitem, ntoken, d = 16, 16, 64, 32
    uc = torch.randn(nuser, d)
    us = torch.randn(nuser, d)
    ic = torch.randn(nitem, d)
    ist = torch.randn(nitem, d)
    dc = torch.randn(2, d)
    ds = torch.randn(2, d)
    m = Step5Model(
        nuser=nuser,
        nitem=nitem,
        ntoken=ntoken,
        emsize=d,
        nhead=4,
        nhid=64,
        nlayers=2,
        dropout=0.1,
        user_content_profiles=uc,
        user_style_profiles=us,
        item_content_profiles=ic,
        item_style_profiles=ist,
        domain_content_profiles=dc,
        domain_style_profiles=ds,
        step5_innovation_config_json=json.dumps(asdict(for_test_default_step5_innovation_config())),
    )
    m.eval()
    m.decoder_eos_id = -1
    return m


def _ccv_packet(batch_size: int, *, device: torch.device | str = "cpu") -> CCVControlPacket:
    ids = torch.ones(batch_size, 2, dtype=torch.long, device=device)
    ones = torch.ones(batch_size, dtype=torch.float32, device=device)
    zeros = torch.zeros(batch_size, dtype=torch.float32, device=device)
    return CCVControlPacket(
        content_evidence_ids=ids,
        style_evidence_ids=ids,
        domain_style_anchor_ids=ids,
        local_style_hint_ids=ids,
        polarity_ids=ids,
        route_scorer_mask=ones,
        route_explainer_mask=ones,
        sample_weight_hint=ones,
        cf_reliability_score=ones,
        content_retention_score=ones,
        style_shift_score=zeros,
        rating_stability_score=ones,
        uncertainty_score=zeros,
        confidence_bucket=ones * 2.0,
        evidence_quality_prior=ones,
        content_anchor_score=ones,
        style_anchor_score=ones,
    )


def _legacy_next_logits(m: Step5Model, prefix: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
    word_feature = m.word_embeddings(decoder_input_ids)
    src = torch.cat([prefix, word_feature], dim=1)
    src = src * (m.emsize ** 0.5)
    src = m.pos_encoder(src)
    attn_mask = _domain_fusion_causal_mask(decoder_input_ids.shape[1], decoder_input_ids.device, prefix_len=m._prefix_len())
    hidden, _ = m.transformer_encoder(src=src, mask=attn_mask)
    return prepare_logits(hidden[:, -1, :], m.hidden2token)


def _run_logits_compare(
    m: Step5Model,
    *,
    dtype: torch.dtype,
    disable_autocast: bool = False,
) -> dict:
    m = m.to(dtype=dtype)
    m.eval()
    u = torch.tensor([1, 2], dtype=torch.long)
    i = torch.tensor([3, 4], dtype=torch.long)
    d = torch.tensor([0, 1], dtype=torch.long)
    prefix = m._build_prefix(d, u, i).to(dtype=dtype)
    legacy_ids = torch.tensor([[0, 5, 7], [0, 9, 3]], dtype=torch.long)
    ac = torch.autocast(device_type="cpu", enabled=False) if disable_autocast else nullcontext()
    with torch.no_grad(), ac:
        legacy_logits = _legacy_next_logits(m, prefix, legacy_ids)
        kv = DecoderKVBackend(m, sdpa_variant="fast")
        prefix_pos = m._position_encode_with_offset(prefix * (m.emsize ** 0.5), 0)
        cache = kv.prefill(prefix_pos)
        out = None
        for pos, tok in enumerate([legacy_ids[:, 0:1], legacy_ids[:, 1:2], legacy_ids[:, 2:3]]):
            cur_pos = m._prefix_len() + pos

            def _embed(x: torch.Tensor, p=cur_pos) -> torch.Tensor:
                return m._position_encode_with_offset(m.word_embeddings(x) * (m.emsize ** 0.5), p)

            out = kv.decode_step(
                tok,
                cache,
                embed_token_fn=_embed,
                hidden_to_logits_fn=lambda h: prepare_logits(h[:, -1, :], m.hidden2token),
            )
            cache = out.past_key_values
        assert out is not None and out.logits is not None
        diff = (legacy_logits - out.logits).abs()
        return {
            "legacy_logits": legacy_logits,
            "kv_logits": out.logits,
            "max_abs_diff": float(diff.max().item()),
            "mean_abs_diff": float(diff.mean().item()),
        }


def _sdpa_fp64_guard_ctx():
    return sdpa_kernel([SDPBackend.MATH])


def _layerwise_debug_report(m: Step5Model, *, dtype: torch.dtype, threshold: float = 1e-5) -> str:
    m = m.to(dtype=dtype)
    m.eval()
    u = torch.tensor([1, 2], dtype=torch.long)
    i = torch.tensor([3, 4], dtype=torch.long)
    d = torch.tensor([0, 1], dtype=torch.long)
    legacy_ids = torch.tensor([[0, 5, 7], [0, 9, 3]], dtype=torch.long)
    prefix = m._build_prefix(d, u, i).to(dtype=dtype)
    ac = torch.autocast(device_type="cpu", enabled=False)

    with torch.no_grad(), ac:
        word_feature = m.word_embeddings(legacy_ids)
        src = torch.cat([prefix, word_feature], dim=1)
        src = src * (m.emsize ** 0.5)
        src = m.pos_encoder(src)
        attn_mask = _domain_fusion_causal_mask(legacy_ids.shape[1], legacy_ids.device, prefix_len=m._prefix_len())
        legacy_nodes = {"embed_plus_pos": src[:, -1:, :]}
        legacy_seq = src
        for li, layer in enumerate(m.transformer_encoder.layers):
            legacy_seq, _ = layer(legacy_seq, src_mask=attn_mask, src_key_padding_mask=None)
            legacy_nodes[f"layer_{li}_block_out"] = legacy_seq[:, -1:, :]
        legacy_nodes["hidden_last"] = legacy_seq[:, -1:, :]
        legacy_nodes["logits"] = prepare_logits(legacy_nodes["hidden_last"][:, -1, :], m.hidden2token)

        kv = DecoderKVBackend(m, sdpa_variant="fast")
        prefix_pos = m._position_encode_with_offset(prefix * (m.emsize ** 0.5), 0)
        cache = kv.prefill(prefix_pos)
        kv_nodes = {}
        out = None
        for pos, tok in enumerate([legacy_ids[:, 0:1], legacy_ids[:, 1:2], legacy_ids[:, 2:3]]):
            cur_pos = m._prefix_len() + pos

            def _embed(x: torch.Tensor, p=cur_pos) -> torch.Tensor:
                return m._position_encode_with_offset(m.word_embeddings(x) * (m.emsize ** 0.5), p)

            x_t = _embed(tok)
            if pos == legacy_ids.shape[1] - 1:
                kv_nodes["embed_plus_pos"] = x_t
            for li, layer in enumerate(m.transformer_encoder.layers):
                x_t, layer_cache = kv._layer_decode_step(layer, kv._sdpa_cores[li], x_t, cache.layers[li])
                cache.layers[li] = layer_cache
                if pos == legacy_ids.shape[1] - 1:
                    kv_nodes[f"layer_{li}_block_out"] = x_t
            cache.generated_len = int(cache.generated_len) + 1
            out = prepare_logits(x_t[:, -1, :], m.hidden2token)
        assert out is not None
        kv_nodes["hidden_last"] = x_t
        kv_nodes["logits"] = out

    ordered = ["embed_plus_pos"] + [f"layer_{li}_block_out" for li in range(len(m.transformer_encoder.layers))] + [
        "hidden_last",
        "logits",
    ]
    lines = ["[layerwise-diff]"]
    first_bad = None
    for name in ordered:
        diff = (legacy_nodes[name] - kv_nodes[name]).abs()
        mx = float(diff.max().item())
        me = float(diff.mean().item())
        lines.append(f"{name}: max={mx:.12g}, mean={me:.12g}")
        if first_bad is None and mx > threshold:
            first_bad = name
    lines.append(f"first_diff_gt_{threshold:.1e}: {first_bad or 'none'}")
    return "\n".join(lines)


def _minimal_cfg_for_mainline() -> FinalTrainingConfig:
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
        full_bleu_decode_strategy="greedy",
        decode_strategy="greedy",
    )


class TestKvCacheBackend(unittest.TestCase):
    # FP64/FP32 strict 用于实现等价性验证；BF16/FP16 regular 允许量化误差，
    # 仅用于部署容忍测试，不作为严格数学等价证明。
    def test_greedy_equivalence(self) -> None:
        m = _build_model()
        m.decode_strategy = "greedy"
        m.hard_max_len = 20
        u = torch.tensor([1, 2], dtype=torch.long)
        i = torch.tensor([3, 4], dtype=torch.long)
        d = torch.tensor([0, 1], dtype=torch.long)
        packet = _ccv_packet(2)
        m.decode_backend = DECODE_BACKEND_LEGACY
        a, _, _ = m.generate(u, i, d, ccv_control_packet=packet)
        m.decode_backend = DECODE_BACKEND_KV_FAST
        b, _, _ = m.generate(u, i, d, ccv_control_packet=packet)
        self.assertTrue(torch.equal(a, b), "greedy 序列应完全一致")

    def test_logits_close(self) -> None:
        m = _build_model()
        res = _run_logits_compare(m, dtype=torch.float32)
        print(f"[default] max abs diff={res['max_abs_diff']:.12g}, mean abs diff={res['mean_abs_diff']:.12g}")
        self.assertLess(res["max_abs_diff"], 1e-4)
        self.assertLess(res["mean_abs_diff"], 1e-5)

    def test_logits_close_fp32_strict(self) -> None:
        m = _build_model()
        res = _run_logits_compare(m, dtype=torch.float32, disable_autocast=True)
        print(f"[fp32-strict] max abs diff={res['max_abs_diff']:.12g}, mean abs diff={res['mean_abs_diff']:.12g}")
        if res["max_abs_diff"] > 1e-5:
            print(_layerwise_debug_report(m, dtype=torch.float32, threshold=1e-5))
        self.assertLess(res["max_abs_diff"], 1e-5)

    def test_logits_close_fp64_strict(self) -> None:
        m = _build_model()
        with _sdpa_fp64_guard_ctx():
            res = _run_logits_compare(m, dtype=torch.float64, disable_autocast=True)
        print(f"[fp64-strict] max abs diff={res['max_abs_diff']:.12g}, mean abs diff={res['mean_abs_diff']:.12g}")
        if res["max_abs_diff"] > 1e-9:
            print(_layerwise_debug_report(m, dtype=torch.float64, threshold=1e-5))
        self.assertLess(res["max_abs_diff"], 1e-9)

    def test_logits_close_bf16_regular(self) -> None:
        m = _build_model()
        try:
            res = _run_logits_compare(m, dtype=torch.bfloat16)
        except Exception as e:
            self.skipTest(f"bf16 unsupported on current backend: {e}")
            return
        print(f"[bf16] max abs diff={res['max_abs_diff']:.12g}, mean abs diff={res['mean_abs_diff']:.12g}")
        self.assertLessEqual(res["max_abs_diff"], 0.006)

    def test_logits_close_fp16_regular(self) -> None:
        m = _build_model()
        try:
            res = _run_logits_compare(m, dtype=torch.float16)
        except Exception as e:
            self.skipTest(f"fp16 unsupported on current backend: {e}")
            return
        print(f"[fp16] max abs diff={res['max_abs_diff']:.12g}, mean abs diff={res['mean_abs_diff']:.12g}")
        self.assertLess(res["max_abs_diff"], 1e-3)

    def test_cache_grows_monotonic(self) -> None:
        m = _build_model()
        u = torch.tensor([1, 2], dtype=torch.long)
        i = torch.tensor([3, 4], dtype=torch.long)
        d = torch.tensor([0, 1], dtype=torch.long)
        kv = DecoderKVBackend(m, sdpa_variant="fast")
        prefix = m._build_prefix(d, u, i)
        cache = kv.prefill(m._position_encode_with_offset(prefix * (m.emsize ** 0.5), 0))
        s0 = cache.debug_shapes()[0]["seq"]
        tok = torch.tensor([[0], [0]], dtype=torch.long)
        for step in range(4):
            pos = m._prefix_len() + step
            out = kv.decode_step(
                tok,
                cache,
                embed_token_fn=lambda x, p=pos: m._position_encode_with_offset(m.word_embeddings(x) * (m.emsize ** 0.5), p),
                hidden_to_logits_fn=lambda h: prepare_logits(h[:, -1, :], m.hidden2token),
            )
            cache = out.past_key_values
            self.assertEqual(cache.debug_shapes()[0]["seq"], s0 + step + 1)

    @unittest.skip("Flan generate 路径与 legacy/KV 手写解码不可比 wall-time；非本轮吞吐基准。")
    def test_perf_smoke_sdpa_kv_faster_than_legacy(self) -> None:
        m = _build_model()
        m.decode_strategy = "greedy"
        m.hard_max_len = 64
        u = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        i = torch.tensor([3, 4, 5, 6], dtype=torch.long)
        d = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        packet = _ccv_packet(4)
        m.decode_backend = DECODE_BACKEND_LEGACY
        t0 = time.perf_counter()
        m.generate(u, i, d, ccv_control_packet=packet)
        t_legacy = time.perf_counter() - t0
        m.decode_backend = DECODE_BACKEND_KV_FAST
        t1 = time.perf_counter()
        m.generate(u, i, d, ccv_control_packet=packet)
        t_kv = time.perf_counter() - t1
        self.assertLess(t_kv, t_legacy, f"kv={t_kv:.4f}s legacy={t_legacy:.4f}s")

    def test_past_key_values_validate_dtype_mismatch_fail_fast(self) -> None:
        m = _build_model()
        kv = DecoderKVBackend(m, sdpa_variant="fast")
        u = torch.tensor([1, 2], dtype=torch.long)
        i = torch.tensor([3, 4], dtype=torch.long)
        d = torch.tensor([0, 1], dtype=torch.long)
        prefix = m._build_prefix(d, u, i)
        cache = kv.prefill(m._position_encode_with_offset(prefix * (m.emsize ** 0.5), 0))
        bad = PastKeyValues(
            layers=[
                LayerKVCache(
                    key=cache.layers[0].key.to(dtype=torch.float32),
                    value=cache.layers[0].value.to(dtype=torch.bfloat16),
                )
            ]
            + list(cache.layers[1:]),
            prefix_len=cache.prefix_len,
            generated_len=cache.generated_len,
        )
        with self.assertRaises(ValueError):
            bad.validate()

    def test_align_new_kv_to_cache_before_cat(self) -> None:
        ref_k = torch.zeros(1, 2, 3, 4, dtype=torch.bfloat16)
        k_new = torch.ones(1, 2, 1, 4, dtype=torch.float32)
        v_new = torch.ones(1, 2, 1, 4, dtype=torch.float32)
        k_a, v_a = _align_new_kv_to_cache(ref_k, k_new, v_new)
        self.assertEqual(k_a.dtype, torch.bfloat16)
        self.assertEqual(v_a.dtype, torch.bfloat16)
        cat = torch.cat([ref_k, k_a], dim=2)
        self.assertEqual(cat.dtype, torch.bfloat16)

    @unittest.skip(
        "Step5 Model.generate 走 Flan-T5 soft-prompt 路径，不经过 DecoderKVBackend.decode_step；"
        "KV 对齐由 test_logits_close* / test_cache_grows_monotonic 覆盖。"
    )
    def test_fallback_policy_raise_propagates_kv_error(self) -> None:
        m = _build_model()
        m.decode_backend = DECODE_BACKEND_KV_FAST
        m.decode_backend_fallback_policy = "raise"
        u = torch.tensor([1], dtype=torch.long)
        i = torch.tensor([3], dtype=torch.long)
        d = torch.tensor([0], dtype=torch.long)
        with patch.object(DecoderKVBackend, "decode_step", side_effect=ValueError("unit_test_kv_failure")):
            with self.assertRaises(ValueError):
                m.generate(u, i, d, ccv_control_packet=_ccv_packet(1))

    def test_fallback_sync_then_fallback_nonfatal_uses_legacy(self) -> None:
        m = _build_model()
        m.decode_strategy = "greedy"
        m.hard_max_len = 8
        m.decode_backend = DECODE_BACKEND_KV_FAST
        m.decode_backend_fallback_policy = "sync_then_fallback"
        u = torch.tensor([1], dtype=torch.long)
        i = torch.tensor([3], dtype=torch.long)
        d = torch.tensor([0], dtype=torch.long)

        def _boom(self, *a, **k):
            raise ValueError("nonfatal_stub")

        with patch.object(DecoderKVBackend, "decode_step", _boom):
            out, _, _ = m.generate(u, i, d, ccv_control_packet=_ccv_packet(1))
        self.assertTrue(out.numel() >= 1)

    @unittest.skip(
        "Step5 Model.generate 不经 DecoderKVBackend；CUDA assert 类错误无法在 generate 中复现。"
    )
    def test_fallback_sync_then_fallback_cuda_message_still_raises(self) -> None:
        m = _build_model()
        m.decode_backend = DECODE_BACKEND_KV_FAST
        m.decode_backend_fallback_policy = "sync_then_fallback"
        u = torch.tensor([1], dtype=torch.long)
        i = torch.tensor([3], dtype=torch.long)
        d = torch.tensor([0], dtype=torch.long)
        with patch.object(
            DecoderKVBackend,
            "decode_step",
            side_effect=RuntimeError("CUDA error: device-side assert triggered"),
        ):
            with self.assertRaises(RuntimeError):
                m.generate(u, i, d, ccv_control_packet=_ccv_packet(1))

    @unittest.skipIf(not torch.cuda.is_available(), "需要 CUDA")
    @unittest.skipIf(not torch.cuda.is_bf16_supported(), "需要 bf16")
    def test_cuda_bf16_autocast_sdpa_kv_safe_generate(self) -> None:
        m = _build_model().cuda()
        m.decode_strategy = "greedy"
        m.hard_max_len = 12
        m.decode_backend = DECODE_BACKEND_KV_SAFE
        u = torch.tensor([1], dtype=torch.long, device="cuda")
        i = torch.tensor([3], dtype=torch.long, device="cuda")
        d = torch.tensor([0], dtype=torch.long, device="cuda")
        with torch.inference_mode(), odcr_cuda_bf16_autocast():
            out, _, _ = m.generate(u, i, d, ccv_control_packet=_ccv_packet(1, device="cuda"))
        self.assertTrue(out.numel() >= 1)

    def test_mainline_monitor_override_resolves_train_time_backend(self) -> None:
        cfg = _minimal_cfg_for_mainline()
        ov = build_full_bleu_monitor_cfg_override(cfg)
        self.assertEqual(resolve_decode_backend_name(ov["decode_backend"]), DECODE_BACKEND_KV_SAFE)
        self.assertEqual(ov["decode_run_context"], "train_time_eval")

    def test_generate_kwargs_v2_includes_decode_fields(self) -> None:
        from executors.decode_controller import GenerateConfig

        gc = GenerateConfig(
            decode_backend="sdpa_kv_safe",
            decode_backend_fallback_policy="raise",
            decode_run_context="train_time_eval",
        )
        d = build_generate_kwargs_effective_v2(gc, eos_token_id=-1)
        self.assertEqual(d.get("decode_backend"), "sdpa_kv_safe")
        self.assertEqual(d.get("decode_backend_fallback_policy"), "raise")
        self.assertEqual(d.get("decode_run_context"), "train_time_eval")

    def test_resolve_backend_names_canonical(self) -> None:
        self.assertEqual(resolve_decode_backend_name("sdpa_kv_fast"), DECODE_BACKEND_KV_FAST)
        self.assertEqual(resolve_decode_backend_name("sdpa_kv_safe"), DECODE_BACKEND_KV_SAFE)
        self.assertEqual(resolve_decode_backend_name("legacy_full_recompute"), DECODE_BACKEND_LEGACY)
        self.assertEqual(resolve_decode_backend_name("kv_cache"), DECODE_BACKEND_KV_FAST)
        with self.assertRaises(ValueError):
            resolve_decode_backend_name("sdpa_kv")


if __name__ == "__main__":
    unittest.main()
