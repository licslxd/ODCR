from __future__ import annotations

import json
import os
import sys
import unittest

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)

import config  # noqa: E402


class TestHardwareControlPlane(unittest.TestCase):
    def test_num_proc_requires_resolver_injected_hardware_json(self) -> None:
        old = {k: os.environ.get(k) for k in ("ODCR_HARDWARE_PROFILE_JSON", "MAX_PARALLEL_CPU", "ODCR_NUM_PROC")}
        try:
            os.environ.pop("ODCR_HARDWARE_PROFILE_JSON", None)
            os.environ["MAX_PARALLEL_CPU"] = "99"
            os.environ["ODCR_NUM_PROC"] = "77"
            with self.assertRaisesRegex(RuntimeError, "缺少 ODCR_HARDWARE_PROFILE_JSON"):
                config.get_num_proc()
            with self.assertRaisesRegex(RuntimeError, "缺少 ODCR_HARDWARE_PROFILE_JSON"):
                config.get_max_parallel_cpu()
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_num_proc_ignores_legacy_env_when_hardware_json_is_present(self) -> None:
        old = {k: os.environ.get(k) for k in ("ODCR_HARDWARE_PROFILE_JSON", "MAX_PARALLEL_CPU", "ODCR_NUM_PROC")}
        try:
            os.environ["ODCR_HARDWARE_PROFILE_JSON"] = json.dumps(
                {
                    "max_parallel_cpu": 6,
                    "num_proc": 2,
                    "dataloader_num_workers_train": 3,
                    "dataloader_num_workers_valid": 1,
                    "dataloader_num_workers_test": 1,
                    "dataloader_prefetch_factor_train": 2,
                    "dataloader_prefetch_factor_valid": 2,
                    "dataloader_prefetch_factor_test": 2,
                    "ddp_world_size": 1,
                },
                sort_keys=True,
            )
            os.environ["MAX_PARALLEL_CPU"] = "99"
            os.environ["ODCR_NUM_PROC"] = "77"
            self.assertEqual(config.get_max_parallel_cpu(), 6)
            self.assertEqual(config.get_num_proc(), 2)
            self.assertEqual(config.get_dataloader_num_workers("train"), 3)
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
