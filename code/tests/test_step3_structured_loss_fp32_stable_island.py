from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


REPO_ROOT = Path(__file__).resolve().parents[2]


class Step3StructuredLossFp32StableIslandTest(unittest.TestCase):
    def test_structured_loss_uses_fp32_stable_island_helpers(self) -> None:
        source = (REPO_ROOT / "code" / "executors" / "step3_train_core.py").read_text(encoding="utf-8")

        self.assertIn("def _fp32_stable_tensor", source)
        self.assertIn("def _stable_cosine_loss", source)
        self.assertIn("def _stable_variance_loss", source)
        self.assertIn("def _stable_projection", source)
        self.assertIn("def _stable_weighted_loss_sum", source)
        self.assertIn("with torch.cuda.amp.autocast(enabled=False):", source)
        self.assertNotIn("torch.cuda.amp.autocast(dtype=torch.float32)", source)
        self.assertIn("max(float(semantics.cosine_eps), 1e-6)", source)
        self.assertIn("_stable_weighted_loss_sum(", source)
        self.assertIn("STEP3_SIDECAR_LOSS_COMPONENT_KEYS", source)
        self.assertIn("ddp_firewall_loss.backward()", source)


if __name__ == "__main__":
    unittest.main()
