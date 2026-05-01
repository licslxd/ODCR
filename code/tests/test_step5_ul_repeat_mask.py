"""Step5 repeat mask / UL / explanation CE(logp) 与参考慢实现数值对照（unittest）。"""
import os
import sys
import unittest

import torch
import torch.nn.functional as F

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)

from odcr_core.step5_word_losses import (  # noqa: E402
    odcr_anti_repeat_unlikelihood_loss_from_logp,
    odcr_repeat_mask_from_tgt_batched,
    per_sample_mean_ce_from_logp,
)


def _repeat_mask_reference_btt(tgt: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """物化 B×T×T 的参考 repeat_mask（与历史 step5_engine 逐行一致，仅测试用）。"""
    _B, T = tgt.shape
    device = tgt.device
    tg = tgt.long()
    same_token = tg.unsqueeze(2) == tg.unsqueeze(1)
    j_indices = torch.arange(T, device=device).unsqueeze(0)
    t_indices = torch.arange(T, device=device).unsqueeze(1)
    prev_mask = j_indices < t_indices
    repeat_full_mask = same_token & prev_mask
    has_repeat = repeat_full_mask.any(dim=1)
    not_pad = tg != pad_id
    return has_repeat & not_pad


def _ul_loss_reference(word_logits: torch.Tensor, tgt: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """旧路径：log_softmax + B×T×T mask（仅测试对照）。"""
    dtype = word_logits.dtype
    logp = F.log_softmax(word_logits, dim=-1)
    repeat_mask = _repeat_mask_reference_btt(tgt, pad_id)
    tg = tgt.long()
    logp_at = logp.gather(dim=-1, index=tg.unsqueeze(-1)).squeeze(-1)
    p = torch.exp(logp_at).clamp(max=1.0 - 1e-6)
    per_pos_loss = -torch.log(1.0 - p)
    valid_loss = per_pos_loss * repeat_mask.to(dtype=per_pos_loss.dtype)
    total = valid_loss.sum()
    count = repeat_mask.sum()
    safe_count = count.clamp(min=1).to(dtype)
    scaled = total / safe_count
    return scaled * (count > 0).to(dtype)


def _per_sample_mean_ce_reference_logits(
    logits_bt: torch.Tensor,
    tgt: torch.Tensor,
    *,
    ignore_index: int,
    label_smoothing: float,
) -> torch.Tensor:
    B, T, V = logits_bt.shape
    ce = F.cross_entropy(
        logits_bt.reshape(-1, V),
        tgt.reshape(-1).long(),
        ignore_index=ignore_index,
        label_smoothing=float(label_smoothing),
        reduction="none",
    ).view(B, T)
    mask = (tgt != ignore_index).to(dtype=ce.dtype)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (ce * mask).sum(dim=1) / denom


class TestStep5RepeatMaskAndUl(unittest.TestCase):
    def test_repeat_mask_matches_reference_small_grid(self) -> None:
        pad = 0
        # 与 legacy any(dim=1)：「后面还会出现同 token」的较早位置为 True；pad 恒 False
        patterns = [
            [[1, 2, 3, 2, 4]],
            [[1, 1, 1, 0, 0]],
            [[5, 3, 5, 3, 7]],
            [[0, 0, 0]],
            [[9]],
        ]
        for rows in patterns:
            tgt = torch.tensor(rows, dtype=torch.long)
            fast = odcr_repeat_mask_from_tgt_batched(tgt, pad_id=pad)
            slow = _repeat_mask_reference_btt(tgt, pad_id=pad)
            self.assertTrue(
                torch.equal(fast, slow),
                msg=f"tgt={rows}\nfast={fast}\nslow={slow}",
            )

    def test_repeat_mask_random_matches_reference(self) -> None:
        pad = 0
        torch.manual_seed(0)
        for _ in range(30):
            B, T = torch.randint(1, 5, (2,)).tolist()
            V = 40
            tgt = torch.randint(0, V, (B, T), dtype=torch.long)
            tgt = torch.where(torch.rand(B, T) < 0.15, torch.zeros_like(tgt), tgt)
            fast = odcr_repeat_mask_from_tgt_batched(tgt, pad_id=pad)
            slow = _repeat_mask_reference_btt(tgt, pad_id=pad)
            self.assertTrue(torch.equal(fast, slow), msg=f"B={B} T={T}")

    def test_ul_loss_close_to_reference(self) -> None:
        torch.manual_seed(1)
        for _ in range(20):
            B, T, V = 3, 12, 64
            logits = torch.randn(B, T, V, dtype=torch.float64) * 0.7
            tgt = torch.randint(1, V, (B, T), dtype=torch.long)
            tgt[:, :2] = 0
            logp = F.log_softmax(logits, dim=-1)
            ref = _ul_loss_reference(logits, tgt, pad_id=0)
            got = odcr_anti_repeat_unlikelihood_loss_from_logp(logp, tgt, pad_id=0)
            self.assertLessEqual(abs(float(ref - got)), 1e-6)

    def test_explanation_ce_from_logp_matches_cross_entropy(self) -> None:
        torch.manual_seed(2)
        for ls in (0.0, 0.05, 0.1, 0.2):
            for _ in range(15):
                B, T, V = 2, 9, 48
                logits = torch.randn(B, T, V, dtype=torch.float32) * 0.5
                tgt = torch.randint(0, V, (B, T), dtype=torch.long)
                tgt = torch.where(torch.rand(B, T) < 0.12, torch.zeros_like(tgt), tgt)
                logp = F.log_softmax(logits, dim=-1)
                ref = _per_sample_mean_ce_reference_logits(
                    logits, tgt, ignore_index=0, label_smoothing=ls
                )
                got = per_sample_mean_ce_from_logp(
                    logp, tgt, ignore_index=0, label_smoothing=ls
                )
                self.assertTrue(
                    torch.allclose(ref, got, atol=1e-5, rtol=1e-4),
                    msg=f"ls={ls} max_diff={(ref - got).abs().max()}",
                )


if __name__ == "__main__":
    unittest.main()
