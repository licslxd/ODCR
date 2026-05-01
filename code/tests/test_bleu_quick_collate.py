"""quick BLEU collate 冒烟：可变长 explanation_idx 不再触发 default_collate 崩溃。"""
import os
import sys
import unittest

import torch
from torch.utils.data import DataLoader

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)

from odcr_core.bleu_runtime import build_explanation_bleu_rows_for_indices  # noqa: E402
from odcr_core.gather_schema import GatheredBatch  # noqa: E402


class _VarLenDataset:
    def __init__(self):
        self._rows = [
            {
                "user_idx": torch.tensor(1, dtype=torch.long),
                "item_idx": torch.tensor(2, dtype=torch.long),
                "rating": torch.tensor(4.0, dtype=torch.float32),
                "explanation_idx": torch.tensor([11, 12, 13], dtype=torch.long),
                "domain_idx": torch.tensor(1, dtype=torch.long),
                "sample_id": torch.tensor(0, dtype=torch.long),
                "exp_sample_weight": torch.tensor(1.0, dtype=torch.float32),
                "metadata_tokens": [1, 2, 3],
            },
            {
                "user_idx": torch.tensor(3, dtype=torch.long),
                "item_idx": torch.tensor(4, dtype=torch.long),
                "rating": torch.tensor(5.0, dtype=torch.float32),
                "explanation_idx": torch.tensor([21, 22], dtype=torch.long),
                "domain_idx": torch.tensor(1, dtype=torch.long),
                "sample_id": torch.tensor(1, dtype=torch.long),
                "exp_sample_weight": torch.tensor(0.7, dtype=torch.float32),
                "metadata_tokens": [9],
            },
        ]

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        return self._rows[idx]


class _DummyModel(torch.nn.Module):
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
        bsz = int(user_idx.size(0))
        return (torch.full((bsz, 2), 7, dtype=torch.long, device=user_idx.device),)


class _DummyTokenizer:
    def batch_decode(self, ids, skip_special_tokens=True):
        t = ids.detach().cpu()
        return [" ".join(str(int(x)) for x in row.tolist()) for row in t]


class TestBleuQuickCollate(unittest.TestCase):
    def test_default_collate_on_varlen_batch_fails(self) -> None:
        ds = _VarLenDataset()
        dl = DataLoader(ds, batch_size=2, shuffle=False)
        with self.assertRaises(RuntimeError):
            _ = next(iter(dl))

    def test_build_rows_works_with_safe_collate(self) -> None:
        ds = _VarLenDataset()
        model = _DummyModel()
        tok = _DummyTokenizer()
        rows = build_explanation_bleu_rows_for_indices(
            model,
            tok,
            torch.device("cpu"),
            ds,
            indices=[0, 1],
            batch_size=2,
            rank=0,
            logger=None,
            dataloader_num_workers=0,
            dataloader_prefetch_factor=None,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(sorted(int(r["sample_id"]) for r in rows), [0, 1])


if __name__ == "__main__":
    unittest.main()
