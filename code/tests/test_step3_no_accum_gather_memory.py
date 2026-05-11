from __future__ import annotations

import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn


CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE_DIR)

from executors import step3_train_core as step3  # noqa: E402


class Step3NoAccumGatherMemoryTest(unittest.TestCase):
    def test_cross_rank_gather_off_world_size_one_keeps_local_shape(self) -> None:
        shared = torch.randn(3, 4, requires_grad=True)
        specific = torch.randn(3, 4, requires_grad=True)
        domain = torch.tensor([0, 1, 0])
        ctx, summary = step3.gather_step3_structured_context_local_gradient(
            shared_repr=shared,
            specific_repr=specific,
            domain_ids=domain,
            world_size=1,
            rank=0,
        )
        self.assertEqual(tuple(ctx["shared_repr"].shape), (3, 4))
        self.assertEqual(summary["structured_effective_pool_size"], 3)
        self.assertFalse(summary["cross_rank_gather_enabled"])

    def test_cross_rank_gather_detaches_remote_context(self) -> None:
        shared = torch.ones(2, 3, requires_grad=True)
        specific = torch.ones(2, 3, requires_grad=True) * 2
        domain = torch.tensor([0, 1])

        def fake_all_gather(outputs, tensor):
            outputs[0].copy_(tensor)
            outputs[1].copy_(tensor + 10)

        with mock.patch.object(step3.dist, "all_gather", side_effect=fake_all_gather):
            ctx, summary = step3.gather_step3_structured_context_local_gradient(
                shared_repr=shared,
                specific_repr=specific,
                domain_ids=domain,
                quality_weights=torch.ones(2),
                world_size=2,
                rank=0,
            )
        self.assertEqual(tuple(ctx["shared_repr"].shape), (4, 3))
        self.assertEqual(summary["structured_effective_pool_size"], 4)
        self.assertTrue(summary["remote_tensors_detached"])
        remote_loss = ctx["shared_repr"][2:].sum()
        remote_loss.backward()
        self.assertTrue(shared.grad is None or torch.all(shared.grad == 0))

    def test_cross_rank_gather_forbids_raw_or_profile_tensors(self) -> None:
        shared = torch.randn(2, 3)
        specific = torch.randn(2, 3)
        domain = torch.tensor([0, 1])
        with self.assertRaisesRegex(RuntimeError, "forbids raw/profile tensors"):
            step3.gather_step3_structured_context_local_gradient(
                shared_repr=shared,
                specific_repr=specific,
                domain_ids=domain,
                world_size=1,
                rank=0,
                requested_keys=["shared_repr", "specific_repr", "domain_ids", "token_ids"],
            )

    def test_cross_rank_gather_shape_mismatch_fails_fast(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "shapes must match"):
            step3.gather_step3_structured_context_local_gradient(
                shared_repr=torch.randn(2, 3),
                specific_repr=torch.randn(2, 4),
                domain_ids=torch.tensor([0, 1]),
                world_size=1,
                rank=0,
            )

    def test_profile_buffer_policy_is_explicit_and_profiles_are_not_trainable(self) -> None:
        class Dummy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.ones(1))
                self.user_content_profiles = torch.ones(2, 3)
                self.user_style_profiles = torch.ones(2, 3)
                self.item_content_profiles = torch.ones(2, 3)
                self.item_style_profiles = torch.ones(2, 3)
                self.domain_content_profiles = torch.ones(2, 3)
                self.domain_style_profiles = torch.ones(2, 3)

        model = Dummy()
        cfg = SimpleNamespace(
            memory_config_json=json.dumps(
                {
                    "activation_checkpointing": {
                        "enabled": True,
                        "policy": "selective",
                        "modules": ["odcr_disentangler"],
                    },
                    "profile_buffer_policy": "cpu_pinned_batch_gather",
                },
                sort_keys=True,
            )
        )
        summary = step3.apply_step3_memory_controls(model, cfg)
        self.assertEqual(summary["profile_buffer_policy"], "cpu_pinned_batch_gather")
        self.assertFalse(summary["silent_fallback"])
        trainable_names = [name for name, _param in step3.step3_trainable_named_parameters(model)]
        self.assertEqual(trainable_names, ["weight"])

    def test_activation_checkpointing_default_is_not_silent_enabled(self) -> None:
        cfg = SimpleNamespace(memory_config_json=json.dumps({"profile_buffer_policy": "gpu_resident"}))
        parsed = step3._parse_step3_memory_config(cfg)
        self.assertFalse(parsed["activation_checkpointing"]["enabled"])
        self.assertEqual(parsed["profile_buffer_policy"], "gpu_resident")


if __name__ == "__main__":
    unittest.main(verbosity=2)
