#!/usr/bin/env python3
"""Run the Step5A frozen Step3 teacher parity gate without writing formal runs."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config import build_resolved_training_config  # noqa: E402
from odcr_core.file_atomic import atomic_write_json  # noqa: E402
from odcr_core.index_contract import load_index_contract, load_profile_tensors_from_contract  # noqa: E402
from odcr_core.step5A_frozen_teacher import (  # noqa: E402
    DEFAULT_STEP3_TEACHER_CHECKPOINT,
    DEFAULT_STEP3_TEACHER_SHA256,
    Step3FrozenTeacher,
)
from odcr_core.step5A_residual_calibration import ZeroInitResidualCalibrator  # noqa: E402
from odcr_core.step5A_teacher_parity import (  # noqa: E402
    build_step5a_teacher_parity_report,
    collect_step3_target_tokenized_rows,
)


def _default_cache_dir(repo: Path, task: int, step3_run: str) -> Path:
    startup = repo / "runs" / "step3" / f"task{int(task)}" / str(step3_run) / "meta" / "step3_tokenizer_cache_startup.json"
    payload = json.loads(startup.read_text(encoding="utf-8"))
    return Path(str(payload["cache_dir"]))


def _write_markdown(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Step5A Step3 Teacher Parity Report",
        "",
        f"- split: {report['split']}",
        f"- sample_count: {report['sample_count']}",
        f"- teacher_mae: {report['teacher_mae']}",
        f"- teacher_rmse: {report['teacher_rmse']}",
        f"- step5A_initial_mae: {report['step5A_initial_mae']}",
        f"- step5A_initial_rmse: {report['step5A_initial_rmse']}",
        f"- teacher_vs_step5A_max_abs_delta: {report['teacher_vs_step5A_max_abs_delta']}",
        f"- teacher_vs_step5A_rmse_delta: {report['teacher_vs_step5A_rmse_delta']}",
        f"- pearson: {report['pearson']}",
        f"- spearman_optional: {report['spearman_optional']}",
        f"- parity_pass: {report['parity_pass']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, default=2)
    parser.add_argument("--step3-run", default="2")
    parser.add_argument("--from-step4-run", default="1")
    parser.add_argument("--split", choices=("train", "valid"), default="valid")
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--checkpoint", default=DEFAULT_STEP3_TEACHER_CHECKPOINT)
    parser.add_argument("--expected-sha256", default=DEFAULT_STEP3_TEACHER_SHA256)
    parser.add_argument("--parity-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    out_dir = repo / "AI_analysis" / "05_final_reports"
    ledger_dir = repo / "AI_analysis" / "03_evidence_ledgers"
    index_contract = load_index_contract(str(repo / "runs" / "step4" / f"task{args.task}" / str(args.from_step4_run) / "index_contract.json"))
    profile_tensors = load_profile_tensors_from_contract(index_contract, "cpu")
    cache_dir = _default_cache_dir(repo, args.task, args.step3_run)
    local_ids = list(range(max(1, int(args.rows))))
    rows = collect_step3_target_tokenized_rows(cache_dir=cache_dir, split=args.split, target_local_ids=local_ids)

    batch = [rows[i] for i in local_ids if i in rows]
    device = torch.device("cpu")

    def _pad(name: str) -> torch.Tensor:
        seqs = [torch.tensor(row[name], dtype=torch.long) for row in batch]
        width = max(int(s.numel()) for s in seqs)
        out = torch.zeros((len(seqs), width), dtype=torch.long, device=device)
        for idx, seq in enumerate(seqs):
            out[idx, : int(seq.numel())] = seq
        return out

    user = torch.tensor([row["user_idx_global"] for row in batch], dtype=torch.long, device=device)
    item = torch.tensor([row["item_idx_global"] for row in batch], dtype=torch.long, device=device)
    domain = torch.ones(len(batch), dtype=torch.long, device=device)
    rating = torch.tensor([row["rating"] for row in batch], dtype=torch.float32, device=device)
    teacher = Step3FrozenTeacher(
        nuser=int(index_contract["nuser_global"]),
        nitem=int(index_contract["nitem_global"]),
        ntoken=1,
        emsize=int(index_contract["embed_dim"]),
        nhead=2,
        nhid=2048,
        nlayers=2,
        dropout=0.2,
        profile_tensors=profile_tensors,
        checkpoint_path=args.checkpoint,
        expected_sha256=args.expected_sha256,
        evidence_max_length=48,
        repo_root=repo,
        report_path=ledger_dir / "step5A_frozen_step3_teacher_load_report.json",
    )
    teacher.eval()
    out = teacher(
        user,
        item,
        domain,
        content_anchor=torch.tensor([row["content_anchor_score"] for row in batch], dtype=torch.float32),
        style_anchor=torch.tensor([row["style_anchor_score"] for row in batch], dtype=torch.float32),
        content_evidence_ids=_pad("content_evidence_ids"),
        style_evidence_ids=_pad("style_evidence_ids"),
        domain_style_anchor_ids=_pad("domain_style_anchor_ids"),
        local_style_hint_ids=_pad("local_style_hint_ids"),
        polarity_ids=_pad("polarity_ids").view(-1),
        evidence_quality_prior=torch.tensor([row["evidence_quality_prior"] for row in batch], dtype=torch.float32),
    )
    residual = ZeroInitResidualCalibrator(hidden_size=int(index_contract["embed_dim"]))
    step5_initial = out.pred_rating + residual(out.pred_rating, out.packet)
    report = build_step5a_teacher_parity_report(
        teacher_pred=out.pred_rating,
        step5a_initial_pred=step5_initial,
        gt_rating=rating,
        sample_ids=local_ids[: len(batch)],
        split=args.split,
        parity_tolerance=float(args.parity_tolerance),
    )
    report["step3_teacher_checkpoint"] = str(args.checkpoint)
    report["step3_teacher_checkpoint_hash"] = str(args.expected_sha256)
    report["formal_run_launched"] = False
    report["step5B_touched"] = False
    json_path = out_dir / "step5A_step3_teacher_parity_report.json"
    md_path = out_dir / "step5A_step3_teacher_parity_report.md"
    atomic_write_json(json_path, report)
    _write_markdown(md_path, report)
    print(json.dumps({"parity_pass": report["parity_pass"], "json_path": str(json_path)}, sort_keys=True))
    return 0 if report["parity_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
