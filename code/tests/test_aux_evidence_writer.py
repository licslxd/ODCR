from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from odcr_core.aux.evidence.ai_analysis_writer import (
    ensure_ai_analysis_tree,
    write_final_report,
    write_index,
    write_ledger,
    write_phase_summary,
    write_raw_log,
    write_search_hit,
)


class AuxEvidenceWriterTest(unittest.TestCase):
    def test_writers_target_canonical_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = ensure_ai_analysis_tree(tmp)
            for key in ("index", "raw_logs", "search_hits", "evidence_ledgers", "phase_summaries", "final_reports"):
                self.assertTrue(paths[key].is_dir())
            self.assertTrue(write_index("idx", "index", base_dir=tmp).as_posix().endswith("00_index/idx.md"))
            self.assertTrue(write_raw_log("raw", "log", base_dir=tmp).as_posix().endswith("01_raw_logs/raw.log"))
            self.assertTrue(write_search_hit("hits", "hits", base_dir=tmp).as_posix().endswith("02_search_hits/hits.md"))
            self.assertTrue(write_ledger("ledger", "ledger", base_dir=tmp).as_posix().endswith("03_evidence_ledgers/ledger.md"))
            self.assertTrue(write_phase_summary("summary", "summary", base_dir=tmp).as_posix().endswith("04_phase_summaries/summary.md"))
            self.assertTrue(write_final_report("report", "report", base_dir=tmp).as_posix().endswith("05_final_reports/report.md"))

    def test_writer_rejects_path_traversal_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_raw_log("../bad", "x", base_dir=tmp)


if __name__ == "__main__":
    unittest.main()
