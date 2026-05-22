from __future__ import annotations

import inspect
import os
import sys
import unittest

import torch
import yaml

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_CODE_DIR, ".."))
sys.path.insert(0, _CODE_DIR)

from executors.step5_engine import (  # noqa: E402
    Model,
    STEP5_FLAN_ENCODER_INPUT_CONTRACT_VERSION,
    _drop_step5_runtime_aux_state_keys,
    _step5_batch_diversity_ema_buffer,
    _step5_explainer_ce_route_mask,
    _step5_resolve_flan_vocab_size,
    odcr_terminal_cleanliness_loss,
)


class TestStep5ExplanationQualityContract(unittest.TestCase):
    def test_explainer_ce_is_primary_and_anti_collapse_is_enabled(self) -> None:
        with open(os.path.join(_REPO_ROOT, "configs/odcr.yaml"), "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        step5 = cfg["step5"]
        train = step5["train"]
        self.assertGreaterEqual(float(train["explainer_loss_weight"]), 0.5)
        self.assertGreater(float(train["backend"]["loss_weight_repeat_ul"]), 0.0)
        self.assertGreater(float(train["backend"]["loss_weight_terminal_clean"]), 0.0)
        self.assertGreaterEqual(int(train["backend"]["terminal_clean_span"]), 4)
        self.assertGreaterEqual(float(step5["explainer_gate"]["explainer_only_multiplier"]), 1.0)
        task2_backend = step5["tasks"][2]["backend"]
        self.assertGreater(float(task2_backend["loss_weight_batch_diversity"]), 0.0)
        self.assertEqual(int(task2_backend["batch_diversity_warmup_epochs"]), 0)
        candidates = {
            row["id"]: row for row in step5["tuning"]["innovation_weight_candidates"]
        }
        self.assertGreaterEqual(float(candidates["W0"]["explainer_loss_weight"]), 0.5)
        self.assertEqual(
            float(candidates["W0"]["explainer_loss_weight"]),
            float(train["explainer_loss_weight"]),
        )

    def test_flan_generate_forwards_quality_safety_kwargs(self) -> None:
        src = inspect.getsource(Model.generate)
        required = (
            "decoder_start_token_id",
            "pad_token_id",
            "eos_token_id",
            "min_new_tokens",
            "no_repeat_ngram_size",
            "bad_words_ids",
            "unk_token_id",
            "renormalize_logits",
            "generator",
            "initial_seed",
            "fork_rng",
            "_build_flan_encoder_inputs",
        )
        for token in required:
            self.assertIn(token, src)

    def test_flan_encoder_appends_content_evidence_tokens(self) -> None:
        class Flan(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = type("Cfg", (), {"vocab_size": 8, "d_model": 3, "pad_token_id": 0})()
                self.shared = torch.nn.Embedding(8, 3)

            def get_input_embeddings(self):
                return self.shared

        model = Model.__new__(Model)
        torch.nn.Module.__init__(model)
        model.flan_explainer = Flan()
        model.flan_vocab_size = 8
        model.ntoken = 8
        soft = torch.randn(2, 4, 3)
        packet = type(
            "Packet",
            (),
            {"content_evidence_ids": torch.tensor([[2, 3, 0], [0, 0, 0]], dtype=torch.long)},
        )()
        encoder_embeds, encoder_mask = model._build_flan_encoder_inputs(soft, packet)
        self.assertEqual(tuple(encoder_embeds.shape), (2, 7, 3))
        self.assertTrue(torch.allclose(encoder_embeds[:, :4, :], soft))
        self.assertEqual(encoder_mask.tolist(), [[1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0, 0]])

    def test_checkpoint_architecture_records_flan_encoder_input_contract(self) -> None:
        from executors import step5_engine as s5

        class Cfg:
            nuser = 1
            nitem = 1
            ntoken = 8
            emsize = 3
            nlayers = 1
            nhead = 1
            nhid = 4
            dropout = 0.0
            train_mode = "lora"
            step5_head = "explanation"
            lora_r = 1
            lora_alpha = 1.0
            lora_dropout = 0.0
            lora_target_modules = ()

        arch = s5._step5_model_architecture_lineage(Cfg(), None)
        self.assertEqual(
            arch["flan_encoder_input_contract"],
            STEP5_FLAN_ENCODER_INPUT_CONTRACT_VERSION,
        )

    def test_terminal_cleanliness_uses_real_nonpad_tail(self) -> None:
        logits = torch.zeros(1, 6, 8, dtype=torch.float32)
        tgt = torch.tensor([[4, 5, 1, 0, 0, 0]], dtype=torch.long)
        logits[0, 2, 2] = 8.0
        loss = odcr_terminal_cleanliness_loss(logits, tgt, bad_ids=[2], span=2, pad_id=0)
        self.assertGreater(float(loss.item()), 0.4)

    def test_explainer_ce_mask_uses_route_not_domain(self) -> None:
        per_sample_loss = torch.tensor([4.0, 6.0], dtype=torch.float32)
        route_explainer = torch.tensor([1.0, 1.0], dtype=torch.float32)
        domain_idx = torch.tensor([1, 0], dtype=torch.long)
        mask = _step5_explainer_ce_route_mask(route_explainer, like=per_sample_loss)
        self.assertEqual(mask.tolist(), [1.0, 1.0])
        self.assertNotEqual(mask.tolist(), (domain_idx == 0).to(dtype=torch.float32).tolist())

    def test_default_bad_terminal_tokens_include_unk_and_space_boundary(self) -> None:
        class Tok:
            unk_token_id = 2

            def encode(self, text: str, add_special_tokens: bool = False):
                return [4] if text == "(" else []

            def convert_tokens_to_ids(self, token: str):
                return 3 if token == "▁" else -1

        ids = set(Model._default_bad_terminal_token_ids(Tok()))
        self.assertIn(2, ids)
        self.assertIn(3, ids)

    def test_batch_diversity_ema_uses_generator_vocab_shape(self) -> None:
        class Flan(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = type("Cfg", (), {"vocab_size": 32128})()

        self.assertEqual(_step5_resolve_flan_vocab_size(Flan(), fallback=32100), 32128)
        module = torch.nn.Module()
        module.register_buffer("batch_diversity_ema_mean_probs", torch.zeros(32100))
        mean_probs = torch.zeros(32128, dtype=torch.float32)
        ema = _step5_batch_diversity_ema_buffer(module, mean_probs)
        self.assertEqual(tuple(ema.shape), (32128,))
        self.assertIn("batch_diversity_ema_mean_probs", module._non_persistent_buffers_set)

    def test_checkpoint_load_drops_batch_diversity_runtime_aux_buffer(self) -> None:
        state = {
            "word_embeddings.weight": torch.zeros(4, 2),
            "batch_diversity_ema_mean_probs": torch.zeros(32100),
        }
        dropped = _drop_step5_runtime_aux_state_keys(state)
        self.assertEqual(dropped["batch_diversity_ema_mean_probs"], (32100,))
        self.assertNotIn("batch_diversity_ema_mean_probs", state)
        self.assertIn("word_embeddings.weight", state)


if __name__ == "__main__":
    unittest.main()
