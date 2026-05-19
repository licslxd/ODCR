import sys
import unittest
from pathlib import Path

import torch


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from executors import step5_engine
from odcr_core.index_contract import IndexContractError


class Step5IndexContractAuditTest(unittest.TestCase):
    def _payload(self, head: str):
        return step5_engine._step5_profile_tensor_audit_payload(
            index_contract={"step4_run": "1"},
            user_profile_tensor=torch.zeros(3, 8, dtype=torch.float32),
            item_profile_tensor=torch.zeros(5, 8, dtype=torch.float16),
            contract_path="/tmp/index_contract.json",
            train_path="/tmp/odcr_routing_train.csv",
            eval_data_path="/tmp/valid.csv",
            step5_run_dir="/tmp/step5_run",
            task_idx=2,
            nuser=3,
            nitem=5,
            mm_train_file={"user_idx_global": [0, 2], "item_idx_global": [0, 4]},
            mm_train_after_filter={"user_idx_global": [0, 2], "item_idx_global": [0, 4]},
            mm_eval_split={"user_idx_global": [0, 2], "item_idx_global": [0, 4]},
            strict_collate_batch_audit=True,
            head=head,
            rank=0,
            profile_meta={"profile_mode": "dual_channel", "selected_paths": {"user_content": ["/tmp/uc.pt"], "item_content": ["/tmp/ic.pt"]}},
        )

    def test_no_retired_profile_variable_references_remain(self):
        source = Path(step5_engine.__file__).read_text(encoding="utf-8")
        self.assertNotIn("_prof_cpu_user", source)
        self.assertNotIn("_prof_cpu_item", source)

    def test_audit_payload_writes_current_profile_tensor_fields_for_each_head(self):
        for head in ("step5A", "step5B", "combined"):
            with self.subTest(head=head):
                payload = self._payload(head)
                self.assertEqual(payload["head"], head)
                self.assertEqual(payload["user_profile_count"], 3)
                self.assertEqual(payload["item_profile_count"], 5)
                self.assertEqual(payload["user_profile_shape"], [3, 8])
                self.assertEqual(payload["item_profile_shape"], [5, 8])
                self.assertEqual(payload["dtype"]["user"], "torch.float32")
                self.assertEqual(payload["dtype"]["item"], "torch.float16")
                self.assertTrue(payload["graph_safe_status"]["collate_batch_audit_completed"])
                self.assertFalse(payload["graph_safe_status"]["missing_profile_fallback"])

    def test_missing_profile_tensor_fails_fast(self):
        with self.assertRaisesRegex(IndexContractError, "requires user and item profile tensors"):
            step5_engine._step5_profile_tensor_audit_payload(
                index_contract={},
                user_profile_tensor=None,
                item_profile_tensor=torch.zeros(1, 1),
                contract_path="/tmp/index_contract.json",
                train_path="/tmp/train.csv",
                eval_data_path="/tmp/valid.csv",
                step5_run_dir="/tmp/run",
                task_idx=2,
                nuser=1,
                nitem=1,
                mm_train_file=None,
                mm_train_after_filter=None,
                mm_eval_split=None,
                strict_collate_batch_audit=False,
                head="step5A",
                rank=0,
            )


if __name__ == "__main__":
    unittest.main()
