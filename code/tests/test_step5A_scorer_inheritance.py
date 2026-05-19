from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from odcr_core.step5A_scorer_inheritance import (
    Step3InheritedRatingScorer,
    Step5AScorerInheritanceError,
    inheritance_report_for_model,
    transplant_step3_scorer_into_step5A,
)
from odcr_core.step5_task_decoupled_policy import (
    Step5TaskDecoupledPolicyError,
    assert_step5a_policy_clean,
    default_task_decoupled_policy,
)


class _TinyStep5AModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.user_embeddings = nn.Embedding(2, 3)
        self.item_embeddings = nn.Embedding(4, 3)
        self.odcr_scorer = Step3InheritedRatingScorer(3)


def _write_tiny_step3_checkpoint(path: Path) -> None:
    state = {
        "user_embeddings.weight": torch.arange(6, dtype=torch.float32).view(2, 3),
        "item_embeddings.weight": torch.arange(12, dtype=torch.float32).view(4, 3),
        "recommender.linear1.weight": torch.eye(3),
        "recommender.linear1.bias": torch.tensor([0.1, 0.2, 0.3]),
        "recommender.linear2.weight": torch.ones(1, 3),
        "recommender.linear2.bias": torch.tensor([0.4]),
    }
    torch.save(state, path)


class Step5AScorerInheritanceTest(unittest.TestCase):
    def test_step3_transplant_writes_machine_verifiable_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ckpt = root / "step3.pth"
            report_path = root / "inheritance.json"
            _write_tiny_step3_checkpoint(ckpt)
            model = _TinyStep5AModel()

            report = transplant_step3_scorer_into_step5A(
                model,
                repo_root=root,
                checkpoint_path=ckpt,
                expected_sha256="",
                strict_hash=False,
                report_path=report_path,
            )

            self.assertTrue(report["inheritance_or_distillation_pass"])
            self.assertEqual(report["scorer_init_source"], "step3_transplant")
            self.assertEqual(report["missing_keys"], [])
            self.assertTrue(report_path.is_file())
            self.assertTrue(torch.equal(model.user_embeddings.weight.detach().cpu(), torch.arange(6).view(2, 3).float()))
            self.assertTrue(inheritance_report_for_model(model)["inheritance_or_distillation_pass"])

    def test_missing_step3_checkpoint_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(Step5AScorerInheritanceError, "checkpoint missing"):
                transplant_step3_scorer_into_step5A(
                    _TinyStep5AModel(),
                    repo_root=td,
                    checkpoint_path="missing.pth",
                    expected_sha256="",
                    strict_hash=False,
                )

    def test_random_step5a_scorer_init_policy_fails_fast(self) -> None:
        policy = default_task_decoupled_policy()
        policy["step5A"]["scorer_init_source"] = "random"
        policy["step5A"]["distillation_enabled"] = False
        with self.assertRaisesRegex(Step5TaskDecoupledPolicyError, "random init is forbidden"):
            assert_step5a_policy_clean(policy)


if __name__ == "__main__":
    unittest.main()
