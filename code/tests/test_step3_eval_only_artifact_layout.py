from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.runners import _step3_eval_only_artifact_log_dir  # noqa: E402
from odcr_core.runners import _copy_step3_eval_only_lineage_artifacts  # noqa: E402


def test_step3_paper_eval_only_uses_handoff_artifact_label(tmp_path: Path) -> None:
    meta = tmp_path / "runs" / "step3" / "task1" / "4" / "meta"
    cfg = SimpleNamespace(
        step3_mode="eval_only",
        step3_eval_protocol="paper_target_only_eval",
        step3_eval_split="valid",
        valid_batch_size=6144,
        manifest_dir=str(meta),
    )

    assert _step3_eval_only_artifact_log_dir(cfg) == (
        meta.resolve() / "eval_only" / "paper_valid_b6144_full_detached"
    )


def test_step3_non_paper_eval_only_keeps_stage_meta_layout(tmp_path: Path) -> None:
    cfg = SimpleNamespace(
        step3_mode="eval_only",
        step3_eval_protocol="minimal_eval",
        step3_eval_split="valid",
        valid_batch_size=1024,
        manifest_dir=str(tmp_path),
    )

    assert _step3_eval_only_artifact_log_dir(cfg) is None


def test_step3_eval_only_lineage_artifacts_are_copied_to_label_dir(tmp_path: Path) -> None:
    parent = tmp_path / "runs" / "step3" / "task1" / "4" / "meta"
    child = parent / "eval_only" / "paper_test_b6144_full_detached"
    parent.mkdir(parents=True)
    (parent / "resolved_config.json").write_text('{"ok": true}\n', encoding="utf-8")
    (parent / "source_table.json").write_text('{"records": []}\n', encoding="utf-8")
    parent_cfg = SimpleNamespace(manifest_dir=str(parent))
    eval_cfg = SimpleNamespace(manifest_dir=str(child))

    _copy_step3_eval_only_lineage_artifacts(parent_cfg, eval_cfg)

    assert (child / "resolved_config.json").read_text(encoding="utf-8") == '{"ok": true}\n'
    assert (child / "source_table.json").read_text(encoding="utf-8") == '{"records": []}\n'
