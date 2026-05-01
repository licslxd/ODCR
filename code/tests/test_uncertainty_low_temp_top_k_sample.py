"""uncertainty_low_temp_top_k：top-k 槽位采样契约与 gather 安全（回归 #gather 越界）。"""
import os
import sys
import unittest

import torch

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)

from executors.decode_controller import (  # noqa: E402
    GenerateConfig,
    assert_topk_slot_indices_valid_for_gather,
    build_generate_kwargs_effective_v2,
    sample_next_token,
)


class TestUncertaintyLowTempTopKSample(unittest.TestCase):
    def _ctx(self) -> dict:
        return {
            "train_time_eval": "false",
            "backend": "sdpa_kv_fast",
            "policy": "raise",
        }

    def test_chosen_token_in_topk_cpu(self) -> None:
        """多次采样：输出 token 必须落在本步 top-k 候选 vocab id 集合内（隐含 sampled_inner ∈ [0,k-1]）。"""
        B, V, k = 6, 80, 5
        temperature = 0.8
        logits = torch.full((B, V), -12.0, dtype=torch.float32)
        for b in range(B):
            logits[b, 10 + b] = 0.0
            logits[b, 20 + b] = -0.2
            logits[b, 30] = -0.5
        gen = torch.Generator()
        gen.manual_seed(2026)
        for _ in range(40):
            out, _, _, diag = sample_next_token(
                logits,
                strategy="uncertainty_low_temp_top_k",
                temperature=temperature,
                top_p=0.9,
                generator=gen,
                gen_so_far=8,
                gap_threshold=0.35,
                prefix_greedy_steps=0,
                top_k=k,
                sampling_diag_context=self._ctx(),
            )
            self.assertIsNotNone(diag)
            assert diag is not None
            self.assertGreater(int(diag["trigger_count"]), 0)
            scaled = logits / max(temperature, 1e-8)
            _, inds = torch.topk(scaled, min(k, V), dim=-1)
            for b in range(B):
                tid = int(out[b, 0].item())
                self.assertIn(tid, inds[b].tolist())

    def test_prefix_greedy_no_topk_gather_path(self) -> None:
        """prefix 内全 greedy（argmax），不要求 top-k 分支；极端 logits 也不应触发非法 gather。"""
        B, V = 3, 50
        logits = torch.randn(B, V, dtype=torch.float32) * 5.0
        logits[0, 0] = float("nan")
        # 仍在 prefix：不应对全 logits 做 finite 检查（与训练首步一致）
        out, _, _, diag = sample_next_token(
            logits,
            strategy="uncertainty_low_temp_top_k",
            temperature=0.5,
            top_p=0.9,
            generator=None,
            gen_so_far=0,
            gap_threshold=0.01,
            prefix_greedy_steps=99,
            top_k=5,
            sampling_diag_context=self._ctx(),
        )
        self.assertIsNone(diag)
        self.assertEqual(out.tolist(), logits.argmax(dim=-1, keepdim=True).tolist())

    def test_effective_k_one_returns_top1(self) -> None:
        """effective_k==1 时直接取 top-1 vocab id，不经 multinomial/gather(topk>1) 路径。"""
        B, V = 4, 40
        logits = torch.full((B, V), -5.0, dtype=torch.float32)
        for b in range(B):
            logits[b, 7 + b] = 0.0
            logits[b, 1] = -0.01
        want = logits.argmax(dim=-1, keepdim=True)
        gen = torch.Generator()
        gen.manual_seed(1)
        out, _, _, diag = sample_next_token(
            logits,
            strategy="uncertainty_low_temp_top_k",
            temperature=0.3,
            top_p=0.9,
            generator=gen,
            gen_so_far=10,
            gap_threshold=1.0,
            prefix_greedy_steps=0,
            top_k=1,
            sampling_diag_context=self._ctx(),
        )
        self.assertIsNotNone(diag)
        torch.testing.assert_close(out, want)

    def test_effective_k_one_returns_top1_generator_none(self) -> None:
        B, V = 2, 15
        logits = torch.arange(float(V * B), dtype=torch.float32).view(B, V)
        want = logits.argmax(dim=-1, keepdim=True)
        out, _, _, _ = sample_next_token(
            logits,
            strategy="uncertainty_low_temp_top_k",
            temperature=1.0,
            top_p=0.9,
            generator=None,
            gen_so_far=3,
            gap_threshold=999.0,
            prefix_greedy_steps=0,
            top_k=1,
            sampling_diag_context=self._ctx(),
        )
        torch.testing.assert_close(out, want)

    def test_invalid_sampled_inner_fail_fast_message(self) -> None:
        inds = torch.tensor([[3, 1, 9], [2, 0, 4]], dtype=torch.long)
        bad = torch.tensor([[10], [10]], dtype=torch.long)
        with self.assertRaises(RuntimeError) as ar:
            assert_topk_slot_indices_valid_for_gather(
                bad,
                inds,
                decode_strategy="uncertainty_low_temp_top_k",
                decode_top_k=5,
                train_time_eval="false",
                backend="sdpa_kv_fast",
                policy="raise",
            )
        msg = str(ar.exception)
        self.assertIn("inds.shape=(2, 3)", msg)
        self.assertIn("sampled_inner.min/max=(10,10)", msg)

    def test_top_k_non_positive_value_error(self) -> None:
        logits = torch.zeros(1, 8)
        with self.assertRaises(ValueError) as ar:
            sample_next_token(
                logits,
                strategy="uncertainty_low_temp_top_k",
                temperature=1.0,
                top_p=0.9,
                generator=None,
                gen_so_far=5,
                prefix_greedy_steps=0,
                top_k=0,
                sampling_diag_context=self._ctx(),
            )
        self.assertIn("decode_top_k", str(ar.exception))

    def test_nan_logits_fail_fast_post_prefix(self) -> None:
        logits = torch.randn(2, 16, dtype=torch.float32)
        logits[0, 3] = float("nan")
        with self.assertRaises(RuntimeError) as ar:
            sample_next_token(
                logits,
                strategy="uncertainty_low_temp_top_k",
                temperature=1.0,
                top_p=0.9,
                generator=None,
                gen_so_far=10,
                prefix_greedy_steps=0,
                top_k=4,
                sampling_diag_context=self._ctx(),
            )
        self.assertIn("NaN/Inf", str(ar.exception))

    def test_smoke_cpu_single_row(self) -> None:
        logits = torch.tensor([[0.0, -0.5, -1.0, -1.0, -1.0]], dtype=torch.float32)
        gen = torch.Generator()
        gen.manual_seed(0)
        out, _, _, diag = sample_next_token(
            logits,
            strategy="uncertainty_low_temp_top_k",
            temperature=0.4,
            top_p=0.9,
            generator=gen,
            gen_so_far=2,
            gap_threshold=0.9,
            prefix_greedy_steps=0,
            top_k=3,
            sampling_diag_context=self._ctx(),
        )
        self.assertEqual(tuple(out.shape), (1, 1))
        self.assertIsNotNone(diag)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_smoke_cuda_topk_sample(self) -> None:
        B, V, k = 2, 64, 4
        logits = torch.full((B, V), -8.0, device="cuda", dtype=torch.float32)
        logits[:, 10:14] = torch.tensor([0.0, -0.1, -0.2, -0.3], device="cuda", dtype=torch.float32)
        gen = torch.Generator(device="cuda")
        gen.manual_seed(42)
        out, _, _, diag = sample_next_token(
            logits,
            strategy="uncertainty_low_temp_top_k",
            temperature=0.5,
            top_p=0.9,
            generator=gen,
            gen_so_far=6,
            gap_threshold=0.5,
            prefix_greedy_steps=0,
            top_k=k,
            sampling_diag_context={**self._ctx(), "backend": "sdpa_kv_fast"},
        )
        self.assertEqual(out.device.type, "cuda")
        self.assertIsNotNone(diag)

    def test_generate_kwargs_effective_mainline_strategy(self) -> None:
        """训练后 valid / manifest 用的 generate_kwargs_effective 仍暴露 uncertainty 参数。"""
        cfg = GenerateConfig(
            strategy="uncertainty_low_temp_top_k",
            temperature=0.2,
            top_k=5,
            gap_threshold=0.35,
            prefix_greedy_steps=4,
            decode_backend="sdpa_kv_fast",
            decode_backend_fallback_policy="raise",
            decode_run_context="valid_eval",
        )
        eff = build_generate_kwargs_effective_v2(cfg, eos_token_id=1)
        self.assertEqual(eff["strategy"], "uncertainty_low_temp_top_k")
        self.assertEqual(eff["top_k"], 5)
        self.assertEqual(eff["gap_threshold"], 0.35)
        self.assertEqual(eff["prefix_greedy_steps"], 4)
        self.assertEqual(eff["decode_backend"], "sdpa_kv_fast")
        self.assertEqual(eff["decode_backend_fallback_policy"], "raise")
        self.assertEqual(eff["decode_run_context"], "valid_eval")


if __name__ == "__main__":
    unittest.main()
