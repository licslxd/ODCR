"""sample_next_token：bf16 autocast 下诊断张量 fp32 与 index_put 安全（回归 dtype mismatch）。"""
import os
import sys
import unittest

import torch

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)

from executors.decode_controller import (  # noqa: E402
    _assign_diag_by_index1d,
    sample_next_token,
)


class TestSampleNextTokenDiagDtype(unittest.TestCase):
    def _ctx_eval_single_gpu(self) -> dict:
        return {
            "train_time_eval": "true",
            "backend": "sdpa_kv_fast",
            "policy": "raise",
        }

    def test_bf16_logits_uncertainty_entropy_write_no_error(self) -> None:
        """bf16 logits + 触发 top-k 采样路径时，ent / log_probs 写回不触发 index_put dtype 错误。"""
        B, V, k = 4, 48, 5
        temperature = 0.5
        logits = torch.full((B, V), -4.0, dtype=torch.bfloat16)
        for b in range(B):
            logits[b, 5 + b] = 0.0
            logits[b, 10 + b] = -0.05
        gen = torch.Generator()
        gen.manual_seed(9001)
        out, ent, log_probs, diag = sample_next_token(
            logits,
            strategy="uncertainty_low_temp_top_k",
            temperature=temperature,
            top_p=0.9,
            generator=gen,
            gen_so_far=10,
            gap_threshold=0.9,
            prefix_greedy_steps=0,
            top_k=k,
            sampling_diag_context=self._ctx_eval_single_gpu(),
        )
        self.assertIsNotNone(diag)
        self.assertEqual(ent.dtype, torch.float32)
        self.assertEqual(log_probs.dtype, torch.float32)
        self.assertEqual(tuple(out.shape), (B, 1))

    def test_bf16_logits_greedy_branch_diag_fp32(self) -> None:
        logits = torch.randn(3, 20, dtype=torch.bfloat16)
        out, ent, log_probs, diag = sample_next_token(
            logits,
            strategy="greedy",
            temperature=1.0,
            top_p=0.9,
            generator=None,
        )
        self.assertIsNone(diag)
        self.assertEqual(ent.dtype, torch.float32)
        self.assertEqual(log_probs.dtype, torch.float32)
        self.assertEqual(tuple(out.shape), (3, 1))

    def test_bf16_logits_nucleus_diag_fp32(self) -> None:
        logits = torch.tensor([[0.0, -1.0, -2.0], [-0.5, 0.0, -3.0]], dtype=torch.bfloat16)
        gen = torch.Generator()
        gen.manual_seed(3)
        out, ent, log_probs, diag = sample_next_token(
            logits,
            strategy="nucleus",
            temperature=0.7,
            top_p=0.95,
            generator=gen,
        )
        self.assertIsNone(diag)
        self.assertEqual(ent.dtype, torch.float32)
        self.assertEqual(log_probs.dtype, torch.float32)
        self.assertEqual(tuple(out.shape), (2, 1))

    def test_assign_diag_bf16_dst_float32_src_aligns(self) -> None:
        """模拟「目标 bf16、源 fp32」：经显式 .to 后应能安全写回。"""
        dst = torch.zeros(5, dtype=torch.bfloat16)
        idx = torch.tensor([0, 2, 4], dtype=torch.long)
        src = torch.tensor([1.25, 2.5, 3.75], dtype=torch.float32)
        _assign_diag_by_index1d(
            dst,
            idx,
            src,
            decode_strategy="uncertainty_low_temp_top_k",
            sampling_diag_context=self._ctx_eval_single_gpu(),
        )
        self.assertEqual(dst.dtype, torch.bfloat16)
        for i, j in enumerate([0, 2, 4]):
            self.assertAlmostEqual(float(dst[j].float().item()), float(src[i].item()), places=5)

    def test_assign_diag_index_put_error_includes_context(self) -> None:
        dst = torch.zeros(2, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as ar:
            _assign_diag_by_index1d(
                dst,
                torch.tensor([0], dtype=torch.long),
                torch.tensor([1.0, 2.0], dtype=torch.float32),
                decode_strategy="uncertainty_low_temp_top_k",
                sampling_diag_context=self._ctx_eval_single_gpu(),
            )
        msg = str(ar.exception)
        self.assertIn("decode_strategy", msg)
        self.assertIn("backend", msg)
        self.assertIn("train_time_eval", msg)
        self.assertIn("dst_dtype", msg)
        self.assertIn("src_dtype", msg)
        self.assertIn("dst_shape", msg)
        self.assertIn("src_shape", msg)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_smoke_cuda_bf16_autocast_uncertainty_sdpa_kv_fast_context(self) -> None:
        """单卡 eval 等价 smoke：bf16 autocast + uncertainty_low_temp_top_k + sdpa_kv_fast 上下文。"""
        device = "cuda"
        B, V, k = 2, 40, 4
        gen = torch.Generator(device=device)
        gen.manual_seed(404)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = torch.full((B, V), -3.0, device=device, dtype=torch.bfloat16)
            logits[:, 3:7] = torch.tensor([0.0, -0.1, -0.2, -0.3], device=device, dtype=torch.bfloat16)
            out, ent, log_probs, diag = sample_next_token(
                logits,
                strategy="uncertainty_low_temp_top_k",
                temperature=0.45,
                top_p=0.9,
                generator=gen,
                gen_so_far=6,
                gap_threshold=0.4,
                prefix_greedy_steps=0,
                top_k=k,
                sampling_diag_context=self._ctx_eval_single_gpu(),
            )
        self.assertEqual(out.device.type, "cuda")
        self.assertIsNotNone(diag)
        self.assertEqual(ent.dtype, torch.float32)
        self.assertEqual(log_probs.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
