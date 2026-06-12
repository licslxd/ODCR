#!/usr/bin/env python3
"""Blocking clean-memory audit gate for ODCR Step5 inputs.

The gate is intentionally read-only: it does not launch training/eval and does
not mutate caches or formal run artifacts. A nonzero exit means the requested
task still has active current-row-derived controls and should not be trained or
evaluated as ODCR-CleanMemory.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.config_resolver import resolve_config  # noqa: E402
from odcr_core.file_atomic import atomic_write_json  # noqa: E402
from odcr_core.step5_clean_memory import (  # noqa: E402
    STEP5_CLEAN_MEMORY_CONTROL_SCHEMA_VERSION,
    STEP5_CLEAN_MEMORY_MODE,
    STEP5_CLEAN_MEMORY_SOURCE,
    apply_step5_clean_memory_controls,
)
from odcr_core.training_checkpoint import stable_hash  # noqa: E402


TASK_NAME = "task2_clean_memory_gate"
SCHEMA_VERSION = "odcr_clean_memory_gate/1"

CURRENT_ROW_ANSWER_FIELDS = (
    "review",
    "explanation",
    "rating",
    "ref_text",
    "metric_ref_text",
    "clean_text",
)
CONTROL_FIELDS = (
    "content_evidence",
    "polarity_anchor",
    "content_anchor_score",
    "style_anchor_score",
    "evidence_quality_prior",
    "sample_weight_hint",
    "route_explainer",
    "route_scorer",
)
CLEAN_SOURCE_COLUMNS = (
    "step5_clean_control_source",
    "step5_memory_control_source",
    "train_memory_control_source",
)
LEAVE_ONE_OUT_COLUMNS = (
    "step5_leave_one_out_memory",
    "leave_one_out_train_memory",
)
CLEAN_SOURCE_VALUES = {
    "train_only_memory_controls",
    "leave_one_out_train_memory",
    "train_memory_controls",
    "step3_pred_step4_train_memory",
}


@dataclass(frozen=True)
class Finding:
    severity: str
    gate: str
    path: str
    message: str
    evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "gate": self.gate,
            "path": self.path,
            "message": self.message,
            "evidence": dict(self.evidence),
        }


def _repo_rel(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except Exception:
        return str(path)


def _read_csv_sample(path: Path, *, max_rows: int) -> pd.DataFrame:
    return pd.read_csv(path, nrows=max(1, int(max_rows)))


def _non_empty_values(df: pd.DataFrame, columns: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col]
        out[col] = int((~s.isna() & (s.astype(str).str.strip() != "")).sum())
    return out


def _has_clean_source(df: pd.DataFrame) -> bool:
    for col in CLEAN_SOURCE_COLUMNS:
        if col not in df.columns:
            continue
        values = {
            str(v).strip()
            for v in df[col].dropna().tolist()
            if str(v).strip()
        }
        if values and values.issubset(CLEAN_SOURCE_VALUES):
            return True
    return False


def _has_leave_one_out_marker(df: pd.DataFrame) -> bool:
    for col in LEAVE_ONE_OUT_COLUMNS:
        if col not in df.columns:
            continue
        vals = df[col].dropna().astype(str).str.strip().str.lower()
        if len(vals) and vals.isin({"1", "true", "yes", "y"}).all():
            return True
    return False


def analyze_frame_contract(
    df: pd.DataFrame,
    *,
    split: str,
    path: str,
    require_leave_one_out: bool,
) -> list[Finding]:
    findings: list[Finding] = []
    cols = set(str(c) for c in df.columns)
    present_answer = sorted(cols.intersection(CURRENT_ROW_ANSWER_FIELDS))
    present_controls = sorted(cols.intersection(CONTROL_FIELDS))
    clean_source = _has_clean_source(df)
    non_empty_answer = _non_empty_values(df, present_answer)
    non_empty_controls = _non_empty_values(df, present_controls)
    evidence = {
        "split": split,
        "rows_sampled": int(len(df)),
        "answer_side_columns": present_answer,
        "control_columns": present_controls,
        "non_empty_answer_side_counts": non_empty_answer,
        "non_empty_control_counts": non_empty_controls,
        "clean_source_columns_present": sorted(cols.intersection(CLEAN_SOURCE_COLUMNS)),
        "leave_one_out_columns_present": sorted(cols.intersection(LEAVE_ONE_OUT_COLUMNS)),
    }
    if present_controls and present_answer and not clean_source:
        findings.append(
            Finding(
                severity="P0",
                gate=f"{split}_current_row_control_source",
                path=path,
                message=(
                    f"{split} contains Step5 control fields together with current-row answer/label "
                    "fields but has no clean train-memory control source marker."
                ),
                evidence=evidence,
            )
        )
    if split in {"valid", "test"} and "rating" in cols and "polarity_anchor" in cols and not clean_source:
        findings.append(
            Finding(
                severity="P0",
                gate=f"{split}_gold_rating_polarity_risk",
                path=path,
                message=(
                    f"{split} exposes both gold rating and polarity_anchor without a clean source marker; "
                    "Step5 must use Step3 predicted polarity/rating bucket instead."
                ),
                evidence=evidence,
            )
        )
    if clean_source:
        contract_cols = {
            "step5_clean_control_contract_version": STEP5_CLEAN_MEMORY_CONTROL_SCHEMA_VERSION,
            "step5_control_contract_version": STEP5_CLEAN_MEMORY_CONTROL_SCHEMA_VERSION,
            "step5_control_source": STEP5_CLEAN_MEMORY_SOURCE,
            "step5_control_mode": STEP5_CLEAN_MEMORY_MODE,
        }
        stale: dict[str, dict[str, Any]] = {}
        for col, expected in contract_cols.items():
            if col not in cols:
                stale[col] = {"expected": expected, "actual": "missing"}
                continue
            actual_values = sorted(set(df[col].dropna().astype(str).str.strip().head(32)))
            if any(v != expected for v in actual_values) or not actual_values:
                stale[col] = {"expected": expected, "actual_sample": actual_values}
        if stale:
            findings.append(
                Finding(
                    severity="P0",
                    gate=f"{split}_clean_memory_contract_version",
                    path=path,
                    message=(
                        f"{split} has a clean-memory source marker but does not carry the active "
                        "ODCR-CleanMemory v2 control contract."
                    ),
                    evidence={**evidence, "stale_contract": stale},
                )
            )
    if require_leave_one_out and not _has_leave_one_out_marker(df):
        findings.append(
            Finding(
                severity="P0",
                gate=f"{split}_leave_one_out_memory",
                path=path,
                message=(
                    f"{split} does not carry a leave-one-out train-memory marker; training must not "
                    "consume current-row review/explanation/rating-derived controls."
                ),
                evidence=evidence,
            )
        )
    return findings


def analyze_step5_static_source(repo_root: Path) -> list[Finding]:
    path = repo_root / "code" / "executors" / "step5_engine.py"
    text = path.read_text(encoding="utf-8")
    findings: list[Finding] = []
    polarity_match = re.search(
        r'polarity_anchor"\]\s*=\s*np\.where\([^\\n]*rating', text, flags=re.DOTALL
    )
    if polarity_match:
        findings.append(
            Finding(
                severity="P0",
                gate="step5_eval_gold_rating_polarity_static",
                path=_repo_rel(repo_root, path),
                message=(
                    "Step5 clean-memory eval controls still derive polarity_anchor from current-row "
                    "gold rating."
                ),
                evidence={"matched": polarity_match.group(0)[:180]},
            )
        )
    return findings


def _analysis_paths(repo_root: Path) -> dict[str, Path]:
    base = repo_root / "AI_analysis"
    return {
        "log": base / "01_raw_logs" / f"audit_{TASK_NAME}.log",
        "hits": base / "02_search_hits" / f"audit_{TASK_NAME}_hits.txt",
        "ledger": base / "03_evidence_ledgers" / f"audit_{TASK_NAME}_ledger.md",
        "summary": base / "04_phase_summaries" / f"audit_{TASK_NAME}_summary.md",
        "report": base / "05_final_reports" / f"audit_{TASK_NAME}_report.md",
        "verdict": base / "05_final_reports" / f"audit_{TASK_NAME}_machine_verdict.json",
        "index": base / "00_index" / f"audit_{TASK_NAME}_index.md",
    }


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _render_hits(findings: Sequence[Finding]) -> str:
    if not findings:
        return "No P0/P1/P2 clean-memory gate findings.\n"
    lines: list[str] = []
    for idx, finding in enumerate(findings, start=1):
        lines.append(f"[{idx}] {finding.severity} {finding.gate} {finding.path}")
        lines.append(f"    {finding.message}")
        lines.append(f"    evidence={json.dumps(finding.evidence, ensure_ascii=False, sort_keys=True)}")
    return "\n".join(lines) + "\n"


def _render_summary(payload: Mapping[str, Any]) -> str:
    return (
        f"# audit_{TASK_NAME} Summary\n\n"
        f"- verdict: {payload['verdict']}\n"
        f"- task: {payload['task']}\n"
        f"- target: {payload['target_domain']}\n"
        f"- from_step4: {payload['from_step4']}\n"
        f"- p0_count: {payload['p0_count']}\n"
        f"- p1_count: {payload['p1_count']}\n"
        f"- p2_count: {payload['p2_count']}\n"
        f"- training_launched: false\n"
        f"- eval_launched: false\n\n"
        "The gate is read-only. A C/D verdict means task2 must not enter clean "
        "training/eval until current-row-derived controls are replaced by train-only memory controls.\n"
    )


def _render_report(payload: Mapping[str, Any], findings: Sequence[Finding]) -> str:
    verdict_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    findings_md = "\n".join(
        f"- {f.severity} `{f.gate}` at `{f.path}`: {f.message}" for f in findings
    ) or "- No blocking findings."
    return f"""# audit_{TASK_NAME} Report

## 0. 一句话结论

task2 clean-memory gate 当前状态属于：{"阻塞" if payload["p0_count"] else "已通过"}。

## 1. 标杆对象现行架构

CleanMemory 标杆要求 Step5 train/eval 输入只使用 train-only memory/profile、Step3 prediction、Step4 predicted/train-memory controls；当前 row review/explanation/gold rating 只能作为训练 target 或指标 reference，不能作为 generation controls。

## 2. 目标对象现行架构

本轮只审计 task2，读取 resolved Step5/Step4 路径和 target valid/test 样本，不启动训练或 eval。

## 3. 两者逐层对比表

架构层/组件 | CleanMemory 要求 | task2 当前状态 | 判定结论
--- | --- | --- | ---
Step5 eval controls | train-only memory controls + Step3 predicted polarity | current-row control risk detected when P0 > 0 | {"FAIL" if payload["p0_count"] else "PASS"}
Step5 train controls | leave-one-out train memory | Step4 export marker audited | {"FAIL" if payload["p0_count"] else "PASS"}
Step3 rating source | accepted Step3 scorer / prediction source | resolved from task-local handoff when available | audit-only
Step4 route controls | predicted/train-memory reliability | source export audited, no formal run launched | audit-only

## 4. 最大技术债缺口

{findings_md}

## 5. 最小重构边界

- 必改部分：Step5 train/eval control source construction for task2.
- 可复用部分：existing user/item/domain content/style profile tensors and Step3 rating_source handoff.
- 不该动的部分：GPU allocation, formal runs, historical run artifacts.

## 6. 风险评估

If training starts before this gate passes, ODCR-CleanMemory would still risk reproducing the old oracle-current-row behavior.

## 7. 最终建议

先插过渡重构阶段：实现 task2 train-only memory controls and leave-one-out train inputs, then rerun this gate.

## 8. Validation Summary

validation item | command | result | evidence path | notes
--- | --- | --- | --- | ---
clean-memory gate | code/tools/odcr_clean_memory_gate.py | {"FAIL" if payload["p0_count"] else "PASS"} | AI_analysis/05_final_reports/audit_{TASK_NAME}_machine_verdict.json | read-only

## 9. Modified Files / Diff Summary

This report is generated by the audit gate. Code modifications are summarized in the final Codex handoff.

## 10. Machine Verdict

```json
{verdict_json}
```
"""


def write_ai_analysis(repo_root: Path, payload: Mapping[str, Any], findings: Sequence[Finding]) -> None:
    paths = _analysis_paths(repo_root)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    _write_text(paths["log"], json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    _write_text(paths["hits"], _render_hits(findings))
    _write_text(paths["summary"], _render_summary(payload))
    _write_text(paths["report"], _render_report(payload, findings))
    _write_text(
        paths["index"],
        (
            f"# audit_{TASK_NAME} Index\n\n"
            f"- summary: `{_repo_rel(repo_root, paths['summary'])}`\n"
            f"- report: `{_repo_rel(repo_root, paths['report'])}`\n"
            f"- verdict: `{_repo_rel(repo_root, paths['verdict'])}`\n"
        ),
    )
    atomic_write_json(paths["verdict"], dict(payload))


def build_payload(
    *,
    repo_root: Path,
    task: int,
    from_step4: str,
    max_rows: int,
    findings: Sequence[Finding],
    resolved_context: Mapping[str, Any],
) -> dict[str, Any]:
    p0 = sum(1 for f in findings if f.severity == "P0")
    p1 = sum(1 for f in findings if f.severity == "P1")
    p2 = sum(1 for f in findings if f.severity == "P2")
    verdict = "A" if p0 == 0 else "C"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task_name": TASK_NAME,
        "stage": "step5_clean_memory_gate",
        "task": int(task),
        "from_step4": str(from_step4),
        "target_domain": str(resolved_context.get("target_domain") or ""),
        "auxiliary_domain": str(resolved_context.get("auxiliary_domain") or ""),
        "repo_root": str(repo_root),
        "max_rows": int(max_rows),
        "verdict": verdict,
        "p0_count": int(p0),
        "p1_count": int(p1),
        "p2_count": int(p2),
        "code_modified": False,
        "formal_run_launched": False,
        "training_launched": False,
        "eval_launched": False,
        "clean_protocol": "ODCR-CleanMemory/train_memory",
        "control_source_required": "train_only_memory_controls",
        "findings": [f.to_dict() for f in findings],
        "resolved_context": dict(resolved_context),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["verdict_hash"] = stable_hash(payload)
    return payload


def run_gate(
    *,
    repo_root: Path,
    task: int,
    from_step4: str,
    max_rows: int,
) -> tuple[dict[str, Any], list[Finding]]:
    cfg, _sources, snapshot = resolve_config(
        config_path=repo_root / "configs" / "odcr.yaml",
        command="step5",
        task_id=int(task),
        set_overrides=[],
        dry_run=True,
        from_step4=str(from_step4),
    )
    train_csv = Path(str(cfg.train_csv or ""))
    if not str(train_csv) or str(train_csv) == ".":
        train_csv = repo_root / "runs" / "step4" / f"task{int(task)}" / str(from_step4) / "odcr_routing_train.csv"
    elif not train_csv.is_absolute():
        train_csv = repo_root / train_csv
    target_valid = repo_root / "data" / str(cfg.target) / "valid.csv"
    target_test = repo_root / "data" / str(cfg.target) / "test.csv"
    index_contract_path = train_csv.parent / "index_contract.json"
    if index_contract_path.is_file():
        index_contract_obj = json.loads(index_contract_path.read_text(encoding="utf-8"))
    else:
        index_contract_obj = {}
    findings = analyze_step5_static_source(repo_root)
    raw_frames: dict[str, pd.DataFrame] = {}
    for split, path, require_loo in (
        ("train", train_csv, True),
        ("valid", target_valid, False),
        ("test", target_test, False),
    ):
        if not path.is_file():
            findings.append(
                Finding(
                    severity="P0",
                    gate=f"{split}_source_missing",
                    path=_repo_rel(repo_root, path),
                    message=f"{split} source file is missing; clean gate cannot prove input cleanliness.",
                    evidence={"split": split},
                )
            )
            continue
        sample = _read_csv_sample(path, max_rows=max_rows)
        if split in {"valid", "test"}:
            sample["domain"] = "target"
            sample = sample.reset_index(drop=True)
            sample["sample_id"] = range(len(sample))
            if "clean_text" not in sample.columns and "explanation" in sample.columns:
                sample["clean_text"] = sample["explanation"].fillna("").astype(str)
        sample = sample.rename(
            columns={
                "user_idx": "user_idx_global",
                "item_idx": "item_idx_global",
            }
        )
        sample["_gate_split"] = split
        raw_frames[split] = sample
    if raw_frames:
        combined = pd.concat(list(raw_frames.values()), ignore_index=True, sort=False)
        combined = apply_step5_clean_memory_controls(
            combined,
            repo_root=repo_root,
            target_domain=str(cfg.target),
            auxiliary_domain=str(cfg.auxiliary),
            index_contract=index_contract_obj,
            split_label="train",
            leave_one_out=True,
        )
    else:
        combined = pd.DataFrame()
    for split, _path, require_loo in (
        ("train", train_csv, True),
        ("valid", target_valid, False),
        ("test", target_test, False),
    ):
        if split not in raw_frames:
            continue
        sample = combined[combined["_gate_split"].astype(str) == split].copy()
        sample = sample.drop(columns=["_gate_split"], errors="ignore")
        findings.extend(
            analyze_frame_contract(
                sample,
                split=split,
                path=_repo_rel(repo_root, _path),
                require_leave_one_out=require_loo,
            )
        )
    rating_source = snapshot.get("rating_source") if isinstance(snapshot, Mapping) else {}
    resolved_context = {
        "train_csv": _repo_rel(repo_root, train_csv),
        "index_contract": _repo_rel(repo_root, index_contract_path),
        "target_valid": _repo_rel(repo_root, target_valid),
        "target_test": _repo_rel(repo_root, target_test),
        "target_domain": str(cfg.target),
        "auxiliary_domain": str(cfg.auxiliary),
        "rating_source_status": (rating_source or {}).get("status") if isinstance(rating_source, Mapping) else None,
        "rating_source_type": (rating_source or {}).get("type") if isinstance(rating_source, Mapping) else None,
    }
    payload = build_payload(
        repo_root=repo_root,
        task=task,
        from_step4=from_step4,
        max_rows=max_rows,
        findings=findings,
        resolved_context=resolved_context,
    )
    return payload, findings


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the ODCR CleanMemory blocking input gate.")
    p.add_argument("--task", type=int, default=2)
    p.add_argument("--from-step4", default="1")
    p.add_argument("--max-rows", type=int, default=2048)
    p.add_argument("--repo-root", default=str(Path.cwd()))
    p.add_argument("--write-ai-analysis", action="store_true")
    p.add_argument("--output-json", default="")
    p.add_argument("--no-fail", action="store_true", help="write verdict but return zero even when P0 findings exist")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    payload, findings = run_gate(
        repo_root=repo_root,
        task=int(args.task),
        from_step4=str(args.from_step4),
        max_rows=int(args.max_rows),
    )
    if args.output_json:
        atomic_write_json(Path(args.output_json), payload)
    if args.write_ai_analysis:
        write_ai_analysis(repo_root, payload, findings)
    print(json.dumps({k: payload[k] for k in ("verdict", "p0_count", "p1_count", "p2_count", "verdict_hash")}, sort_keys=True))
    if payload["p0_count"] and not args.no_fail:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
