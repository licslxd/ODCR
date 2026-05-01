"""将一次或多次运行的关键产物收敛到 runs/task{T}/vN/analysis/packNN/（单任务路径；禁止写入 runs/global/…/meta）。"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from odcr_core import path_layout, run_naming

# 超过此大小的文本样例不整文件拷贝，仅写 source_paths + 头部抽样
_MAX_EMBED_BYTES = 8 * 1024 * 1024
_HEAD_JSONL_LINES = 200


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _embed_jsonl_or_ref(
    src: Path,
    pack: Path,
    out_name: str,
    source_paths: Dict[str, str],
    *,
    key: str,
    head_lines: int = _HEAD_JSONL_LINES,
) -> None:
    """小文件 copy2；过大则记录全路径并写入前 head_lines 行到 out_name。"""
    if not src.is_file():
        return
    source_paths[key] = str(src.resolve())
    try:
        sz = src.stat().st_size
    except OSError:
        return
    if sz <= _MAX_EMBED_BYTES:
        shutil.copy2(src, pack / out_name)
        return
    lines: List[str] = []
    with src.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= head_lines:
                break
            lines.append(line.rstrip("\n"))
    (pack / out_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    source_paths[f"{key}_truncated"] = "true"


def export_analysis_pack(
    *,
    repo_root: Path,
    task_id: int,
    iteration_id: str,
    pack_id_req: Optional[str] = None,
    eval_run_dirs: Optional[List[Path]] = None,
    rerank_run_dirs: Optional[List[Path]] = None,
    matrix_run_dir: Optional[Path] = None,
    notes: str = "",
) -> Path:
    """
    创建 analysis/packNN/，写入短摘要、清单与可复制/可选的关键文件。
    eval_run_dirs / rerank_run_dirs: 本次希望纳入的绝对路径列表（通常各 1 个）。
    """
    it = run_naming.normalize_iteration_id(iteration_id)
    analysis_parent = path_layout.get_analysis_root(repo_root, task_id, it)
    analysis_parent.mkdir(parents=True, exist_ok=True)
    pack_id = run_naming.allocate_child_dir(analysis_parent, requested=pack_id_req, kind="pack")
    pack = path_layout.get_analysis_pack_root(repo_root, task_id, it, pack_id)
    pack.mkdir(parents=True, exist_ok=True)

    eval_run_dirs = eval_run_dirs or []
    rerank_run_dirs = rerank_run_dirs or []

    ai_manifest: Dict[str, Any] = {
        "schema": "odcr_ai_manifest_v2",
        "created_at_utc": _utc_now(),
        "task_id": task_id,
        "iteration_id": it,
        "pack_id": pack_id,
        "sources": {
            "eval_paths": [str(p.resolve()) for p in eval_run_dirs],
            "rerank_paths": [str(p.resolve()) for p in rerank_run_dirs],
            "matrix_path": str(matrix_run_dir.resolve()) if matrix_run_dir else None,
        },
    }

    key_metrics: Dict[str, Any] = {}

    def _read_metrics(p: Path, *, rerank: bool = False) -> Optional[Dict[str, Any]]:
        mp = path_layout.eval_metrics_path(p, rerank=rerank)
        if not mp.is_file():
            mp = p / "metrics.json"
        if not mp.is_file():
            return None
        try:
            return json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            return None

    primary_eval = eval_run_dirs[-1] if eval_run_dirs else None
    primary_rerank = rerank_run_dirs[-1] if rerank_run_dirs else None
    src_for_digest = primary_rerank or primary_eval

    if primary_eval:
        m = _read_metrics(primary_eval, rerank=False)
        if m:
            key_metrics["eval"] = {
                "path": str(primary_eval.resolve()),
                "repo_metrics": m.get("repo_metrics"),
                "paper_metrics": m.get("paper_metrics"),
                "generation_semantic_fingerprint": m.get("generation_semantic_fingerprint"),
                "training_semantic_fingerprint": m.get("training_semantic_fingerprint"),
            }
    if primary_rerank:
        m = _read_metrics(primary_rerank, rerank=True)
        if m:
            key_metrics["rerank"] = {
                "path": str(primary_rerank.resolve()),
                "repo_metrics": m.get("repo_metrics"),
                "paper_metrics": m.get("paper_metrics"),
                "rerank_summary": m.get("rerank_summary"),
            }

    (pack / "key_metrics.json").write_text(
        json.dumps(key_metrics, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    # eval_digest.log
    if src_for_digest:
        dig = src_for_digest / "eval_digest.log"
        if dig.is_file():
            shutil.copy2(dig, pack / "eval_digest.log")

    # phase summaries from matrix run dir
    if matrix_run_dir:
        for name in ("phase1_summary.csv", "phase1_summary.json", "phase2_rerank_summary.csv", "phase2_rerank_summary.json"):
            src = matrix_run_dir / name
            if src.is_file():
                shutil.copy2(src, pack / name)
        mm = matrix_run_dir / "matrix_manifest.json"
        if mm.is_file():
            shutil.copy2(mm, pack / "matrix_manifest.json")

    # predictions / rerank examples：大文件只抽样 + source_paths
    spaths: Dict[str, str] = {}
    if primary_rerank:
        co = primary_rerank / "rerank_examples_changed_only.jsonl"
        _embed_jsonl_or_ref(
            co,
            pack,
            "rerank_examples_changed_only.jsonl",
            spaths,
            key="rerank_examples_changed_only_jsonl",
        )
        h50 = primary_rerank / "rerank_examples_head50.json"
        if h50.is_file():
            try:
                if h50.stat().st_size <= _MAX_EMBED_BYTES:
                    shutil.copy2(h50, pack / "rerank_examples_head50.json")
                else:
                    spaths["rerank_examples_head50_json"] = str(h50.resolve())
            except OSError:
                pass

    if primary_rerank or primary_eval:
        pr = primary_rerank or primary_eval
        assert pr is not None
        pj = pr / "predictions.jsonl"
        if pj.is_file():
            spaths["predictions_jsonl"] = str(pj.resolve())
            head_lines: List[str] = []
            with pj.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 50:
                        break
                    head_lines.append(line.rstrip("\n"))
            (pack / "predictions_head50.jsonl").write_text("\n".join(head_lines) + "\n", encoding="utf-8")
    (pack / "source_paths.json").write_text(
        json.dumps(spaths, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # bad_cases.jsonl：若上游未单独产出则写空文件占位
    bad = pack / "bad_cases.jsonl"
    if not bad.is_file():
        bad.write_text("", encoding="utf-8")

    summary_lines = [
        f"# Analysis pack {pack_id}",
        "",
        f"- task: {task_id} | iteration: {it}",
        f"- created_at_utc: {ai_manifest['created_at_utc']}",
        "",
        "## Sources",
        "",
    ]
    for p in eval_run_dirs:
        summary_lines.append(f"- eval: `{p}`")
    for p in rerank_run_dirs:
        summary_lines.append(f"- rerank: `{p}`")
    if matrix_run_dir:
        summary_lines.append(f"- matrix: `{matrix_run_dir}`")
    summary_lines.extend(
        [
            "",
            "## Key files in this pack",
            "",
            "- analysis_summary.md（建议首先阅读）",
            "- key_metrics.json",
            "- ai_manifest.json",
            "- eval_digest.log（若有）",
            "- phase1_summary / phase2_rerank_summary（若提供 matrix_path）",
            "- rerank 样例与 predictions 抽样；超大 jsonl 见 source_paths.json",
            "",
        ]
    )
    (pack / "analysis_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    (pack / "notes.md").write_text(notes or "", encoding="utf-8")

    ai_manifest["pack_dir"] = str(pack.resolve())
    (pack / "ai_manifest.json").write_text(
        json.dumps(ai_manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    return pack
