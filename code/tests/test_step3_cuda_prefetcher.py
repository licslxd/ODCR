from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from executors.step3_train_core import Step3CUDAPrefetcher  # noqa: E402


class Step3CUDAPrefetcherTest(unittest.TestCase):
    def test_cpu_without_explicit_diagnostic_fails_fast(self) -> None:
        loader = DataLoader([{"x": torch.tensor([1]), "meta": "row"}], batch_size=None)
        with self.assertRaisesRegex(RuntimeError, "diagnostic-only"):
            Step3CUDAPrefetcher(loader, device=torch.device("cpu"), enabled=True)

    def test_cpu_diagnostic_keeps_metadata_and_timing_fields(self) -> None:
        batch = {"x": torch.tensor([1]), "nested": {"flag": torch.tensor([True])}, "path": "row-1"}
        loader = DataLoader([batch], batch_size=None)
        prefetcher = Step3CUDAPrefetcher(
            loader,
            device=torch.device("cpu"),
            enabled=True,
            non_blocking=True,
            diagnostic_cpu_mode=True,
        )
        item = next(iter(prefetcher))
        self.assertEqual(item["path"], "row-1")
        self.assertTrue(torch.equal(item["x"], batch["x"]))
        self.assertIn("dataloader_next_wait", prefetcher.last_timing)
        self.assertIn("h2d_prefetch_time", prefetcher.last_timing)
        self.assertIn("optimizer_time", Step3CUDAPrefetcher.timing_fields)

    def test_cuda_path_records_stream_and_non_blocking_in_source(self) -> None:
        source = Path(sys.modules[Step3CUDAPrefetcher.__module__].__file__).read_text(encoding="utf-8")
        self.assertIn("non_blocking=self.non_blocking", source)
        self.assertIn("value.record_stream(stream)", source)
        self.assertIn("torch.cuda.Stream", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
