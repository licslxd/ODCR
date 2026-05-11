#!/usr/bin/env python3
"""
ODCR Step3 evidence-pack extractor.

Purpose
-------
Create a deduplicated, high-signal audit.log from ODCR Step3 run artifacts.
It preserves useful raw evidence while removing obvious repetition: tqdm spam,
duplicate warnings, repeated config blocks, binary checkpoint contents, and long
low-value debug noise.

Typical usage
-------------
# Scan all runs under runs/step3 and write audit.log
python code/tools/extract_step3_evidence_pack.py \
  --root /public/home/zhangliml/lc/ODCR/ODCR-main/runs/step3 \
  --output /public/home/zhangliml/lc/ODCR/ODCR-main/audit.log \
  --target-bytes 850000 \
  --mode scan-all

# Extract one manually provided run directory
python code/tools/extract_step3_evidence_pack.py \
  --run-root /public/home/zhangliml/lc/ODCR/ODCR-main/runs/step3/task2/2 \
  --output /public/home/zhangliml/lc/ODCR/ODCR-main/audit.log \
  --target-bytes 850000 \
  --mode manual-run

Notes
-----
- Does not read binary checkpoint contents; only path/size/sha256 are recorded.
- Refuses live/running snapshots by default; use --allow-running-snapshot only
  when you explicitly want a partial live evidence pack.
- Uses only Python stdlib.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TEXT_EXTS = {
    ".log", ".json", ".jsonl", ".csv", ".txt", ".md", ".yaml", ".yml",
    ".generated_at", ".out", ".err",
}
BINARY_EXTS = {".pth", ".pt", ".bin", ".npy", ".npz", ".pkl", ".tmp"}
RUNNING_STATUSES = {"running", "pending", "started", "in_progress", "partial"}
FAILED_STATUSES = {"failed", "error", "interrupted", "aborted"}
OK_STATUSES = {"ok", "success", "completed", "done", "completed_validation"}

PAPER_REFERENCE = """Paper reference metrics (not necessarily Step3 diagnostic protocol):
Recommendation:
- MAE = 0.6601 ± 0.0144
- RMSE = 0.9179 ± 0.0118
Explanation:
- ROUGE = 13.41 ± 0.18, 10.49 ± 0.33
- BLEU = 12.84 ± 0.50, 4.85 ± 0.26, 2.29 ± 0.18, 1.26 ± 0.10
- DIST-1/DIST-2 = 0.98 ± 0.30, 4.07 ± 1.09
- METEOR = 9.82 ± 0.37
Caveat: Step3 diagnostic metrics may not equal final full-pipeline paper metrics unless Step4/Step5/eval/rerank protocol confirms comparability.
"""

KEY_LOG_PATTERNS = [
    "ERROR", "WARNING", "Traceback", "Exception", "TypeError", "RuntimeError", "ChildFailedError",
    "NCCL", "OOM", "out of memory", "NaN", "nan", "Inf", "inf", "nonfinite", "grad_norm",
    "quality_status", "downstream_ready", "checkpoint", "sidecar", "lineage", "best_observed",
    "best_after_min", "best.pth", "latest.pth", "topk", "epoch_time", "valid_loss", "train_loss",
    "Tokenize", "cache", "cache_status", "cache_dir", "cache_key", "manifest", "completed",
    "miss_reason", "full_run_config_hash", "num_proc", "profile", "candidate", "G1S", "G1-M", "G2-C",
    "early", "stop", "scheduler", "lr=", "eval", "BLEU", "ROUGE", "DIST", "METEOR", "MAE", "RMSE",
]

CONFIG_KEEP_KEYS = {
    "run_id", "status", "quality_status", "downstream_ready", "validation_status", "command",
    "task", "task_id", "source", "target", "source_domain", "target_domain",
    "profile", "profile_id", "active_profile", "candidate", "selected_candidate", "formal_allowed",
    "batch_size", "micro_batch_size", "ddp_world_size", "effective_pool", "lr", "learning_rate",
    "scheduler", "scheduler_type", "warmup_ratio", "min_lr_ratio", "optimizer", "train_precision",
    "precision", "bf16", "tf32", "max_length", "evidence_length", "max_grad_norm",
    "cache", "cache_dir", "cache_status", "cache_key", "tokenization_compat_hash", "run_lineage_hash",
    "num_proc", "selected_num_proc", "max_parallel_cpu", "reserved_cpu", "workers_per_rank",
    "pin_memory", "persistent_workers", "non_blocking_h2d", "prefetch", "prefetch_factor",
    "checkpoint", "checkpoint_policy", "best_observed", "best_after_min_epochs", "latest",
    "quality_gate", "started_at", "finished_at", "duration", "duration_s", "failure", "failure_reason",
    "fatal_signature", "latest_status", "latest_run_id", "latest_run_dir",
}

NUMERIC_KEY_HINTS = (
    "loss", "lr", "time", "ms", "sec", "step", "epoch", "grad", "norm", "allocated",
    "reserved", "util", "memory", "samples", "throughput", "bleu", "rouge", "dist", "meteor", "mae", "rmse",
)

@dataclass
class FileInfo:
    path: Path
    rel: str
    size: int
    line_count: Optional[int]
    sha256: str
    file_type: str
    parse_type: str
    priority: str
    reason: str
    section: str
    duplicate_group: str = ""
    max_bytes: int = 0

@dataclass
class RunInfo:
    run_root: Path
    task: str
    run_id: str
    role: str = "unknown"
    status: str = "unknown"
    quality_status: str = "unknown"
    downstream_ready: Any = "unknown"
    latest: bool = False
    files: List[FileInfo] = field(default_factory=list)


def sha256_file(path: Path, max_bytes: Optional[int] = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = max_bytes
        while True:
            if remaining is not None and remaining <= 0:
                break
            chunk_size = 1024 * 1024
            if remaining is not None:
                chunk_size = min(chunk_size, remaining)
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return h.hexdigest()


def is_binary_path(path: Path) -> bool:
    if path.suffix.lower() in BINARY_EXTS:
        return True
    try:
        with path.open("rb") as f:
            sample = f.read(4096)
        if b"\x00" in sample and path.suffix.lower() not in {".log", ".txt"}:
            return True
        # Heuristic: lots of non-text bytes.
        if sample:
            textish = sum(1 for b in sample if 32 <= b <= 126 or b in b"\n\r\t\0")
            return textish / len(sample) < 0.70
    except Exception:
        return False
    return False


def read_text(path: Path, limit_bytes: Optional[int] = None) -> Tuple[str, int]:
    """Read text, replace NUL and invalid chars; returns text and NUL count."""
    data: bytes
    with path.open("rb") as f:
        data = f.read(limit_bytes or path.stat().st_size)
    nul_count = data.count(b"\x00")
    text = data.replace(b"\x00", b"[NUL]").decode("utf-8", errors="replace")
    return text, nul_count


def count_lines(path: Path) -> Optional[int]:
    if is_binary_path(path):
        return None
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return None


def parse_json(path: Path) -> Optional[Any]:
    try:
        text, _ = read_text(path)
        return json.loads(text)
    except Exception:
        return None


def parse_jsonl(path: Path, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = []
    try:
        with path.open("rb") as f:
            for i, raw in enumerate(f):
                if max_rows is not None and i >= max_rows:
                    break
                line = raw.replace(b"\x00", b"[NUL]").decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        rows.append(obj)
                except Exception:
                    continue
    except Exception:
        pass
    return rows


def classify_parse_type(path: Path) -> str:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if is_binary_path(path):
        return "binary"
    if suffix == ".json" or name.endswith(".lineage.json"):
        return "json"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".csv":
        return "csv"
    if suffix == ".log" or suffix in {".txt", ".md"} or "generated_at" in name:
        return "log" if suffix == ".log" else "text"
    return "text"


def classify_file(rel: str, parse_type: str, size: int) -> Tuple[str, str, str, int]:
    r = rel.replace("\\", "/")
    if parse_type == "binary":
        return "hash_only", "binary checkpoint/optimizer; record path/size/hash only", "checkpoint_lineage_sidecars", 3000
    if r.endswith("epoch_summary.csv"):
        return "must_full", "compact epoch trend table; always keep", "epoch_metrics_loss", 50000
    if r.endswith("best_event.json") or r.endswith("checkpoint_lineage.json") or r.endswith("trainer_state.json"):
        return "must_key_fields", "high-value checkpoint/trainer state", "checkpoint_lineage_sidecars", 60000
    if r.endswith("run_summary.json") or r.endswith("manifest.json"):
        return "must_key_fields", "run status and artifact contract", "latest_or_manual_deep_summary", 60000
    if r.endswith("step3_tokenizer_cache_startup.json"):
        return "must_key_fields", "cache hit/miss/rebuild evidence", "cache_tokenizer_manifest", 50000
    if r.endswith("resolved_config.json") or r.endswith("training_runtime_config.json") or r.endswith("source_table.json"):
        return "must_key_fields", "active config; extract important fields only", "latest_or_manual_deep_summary", 50000
    if r.endswith("source_table_verbose.json"):
        return "must_key_fields", "candidate/role conflicts; extract compact candidate diff", "latest_or_manual_deep_summary", 30000
    if r.endswith("metrics.jsonl"):
        return "aggregate", "training/validation metrics; aggregate and sample rows", "epoch_metrics_loss", 60000
    if r.endswith("loss_breakdown.jsonl"):
        return "aggregate", "loss components; aggregate and anomalies", "epoch_metrics_loss", 60000
    if r.endswith("timing_profile.jsonl"):
        return "aggregate", "performance timings; aggregate and top anomalies", "timing_gpu_performance", 60000
    if r.endswith("gpu_profile.jsonl"):
        return "aggregate", "GPU/memory profile; aggregate and peaks", "timing_gpu_performance", 60000
    if r.endswith("samples.jsonl"):
        return "aggregate", "diagnostic samples; collapse stats and representative subset", "samples_eval_diagnostics", 80000
    if r.endswith("errors.log"):
        return "windows", "errors/tracebacks high value; keep mostly complete if small", "errors_traceback_log_windows", 80000
    if r.endswith("full.log"):
        return "windows", "authoritative log; extract high-signal windows", "errors_traceback_log_windows", 100000
    if r.endswith("console.log"):
        return "windows", "compact visible log; keep summaries", "errors_traceback_log_windows", 20000
    if r.endswith("debug.log"):
        return "windows", "debug duplicate-prone; only unique trace/rank evidence", "errors_traceback_log_windows", 40000
    if r.endswith(".lineage.json"):
        return "must_key_fields", "checkpoint sidecar/lineage JSON", "checkpoint_lineage_sidecars", 30000
    return "skip", "low-priority unknown text", "misc", 5000


def discover_runs(root: Path) -> List[Path]:
    runs = []
    if not root.exists():
        return runs
    # Expected layout: runs/step3/task2/1, runs/step3/task2/2, ...
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            if (run_dir / "meta").exists() or (run_dir / "state").exists() or (run_dir / "model").exists():
                runs.append(run_dir)
    return runs


def infer_task_run(run_root: Path) -> Tuple[str, str]:
    run_id = run_root.name
    task = run_root.parent.name if run_root.parent else "unknown_task"
    return task, run_id


def load_latest_map(step3_root: Path) -> Dict[str, str]:
    latest = {}
    if not step3_root.exists():
        return latest
    for task_dir in step3_root.iterdir():
        if not task_dir.is_dir():
            continue
        lp = task_dir / "latest.json"
        if not lp.exists():
            continue
        obj = parse_json(lp) or {}
        run_id = str(obj.get("latest_run_id") or obj.get("run_id") or "")
        if run_id:
            latest[task_dir.name] = run_id
    return latest


def inventory_run(run_root: Path) -> RunInfo:
    task, run_id = infer_task_run(run_root)
    info = RunInfo(run_root=run_root, task=task, run_id=run_id)
    for path in sorted(p for p in run_root.rglob("*") if p.is_file()):
        try:
            rel = str(path.relative_to(run_root))
            size = path.stat().st_size
            parse_type = classify_parse_type(path)
            line_count = count_lines(path) if parse_type != "binary" else None
            sha = sha256_file(path)
            priority, reason, section, max_bytes = classify_file(rel, parse_type, size)
            info.files.append(FileInfo(path, rel, size, line_count, sha, path.suffix.lower() or path.name, parse_type, priority, reason, section, max_bytes=max_bytes))
        except Exception:
            continue
    # Basic status from run_summary/latest-like files.
    run_summary = parse_json(run_root / "meta" / "run_summary.json") or {}
    manifest = parse_json(run_root / "meta" / "manifest.json") or {}
    for source in (run_summary, manifest):
        status = source.get("status") or source.get("latest_status")
        if status:
            info.status = str(status)
            break
    q = run_summary.get("quality_status") or manifest.get("quality_status")
    if q:
        info.quality_status = str(q)
    if "downstream_ready" in run_summary:
        info.downstream_ready = run_summary.get("downstream_ready")
    return info


def is_running(info: RunInfo) -> bool:
    s = (info.status or "").lower()
    if s in RUNNING_STATUSES:
        return True
    run_summary = parse_json(info.run_root / "meta" / "run_summary.json") or {}
    if run_summary.get("finished_at") is None and s not in FAILED_STATUSES and s not in OK_STATUSES:
        # If files are currently mutating or status unknown without finish, be conservative.
        return True
    return False


def truncate_text(text: str, max_bytes: int, label: str = "") -> str:
    if max_bytes <= 0:
        return ""
    data = text.encode("utf-8", errors="replace")
    if len(data) <= max_bytes:
        return text
    head = max_bytes // 2
    tail = max_bytes - head - 300
    if tail < 0:
        tail = 0
    return (
        data[:head].decode("utf-8", errors="replace")
        + f"\n\n[... TRUNCATED {label}: original_bytes={len(data)}, kept_bytes≈{max_bytes} ...]\n\n"
        + data[-tail:].decode("utf-8", errors="replace")
    )


def compact_json(obj: Any, max_depth: int = 4, keep_keys: Optional[set] = None) -> Any:
    if max_depth <= 0:
        if isinstance(obj, (dict, list)):
            return "<truncated>"
        return obj
    if isinstance(obj, dict):
        out = {}
        keys = list(obj.keys())
        for k in keys:
            v = obj[k]
            keep = keep_keys is None or k in keep_keys or any(h in str(k).lower() for h in ["status", "quality", "checkpoint", "cache", "profile", "metric", "loss", "epoch", "hash", "path", "failure", "reason", "run", "task", "source", "target", "num_proc", "worker", "lr", "scheduler"])
            if keep:
                out[k] = compact_json(v, max_depth - 1, keep_keys)
        if len(out) < len(obj):
            out["__omitted_keys_count__"] = len(obj) - len(out)
        return out
    if isinstance(obj, list):
        if len(obj) <= 20:
            return [compact_json(x, max_depth - 1, keep_keys) for x in obj]
        return [compact_json(x, max_depth - 1, keep_keys) for x in obj[:10]] + [f"<... {len(obj)-20} items omitted ...>"] + [compact_json(x, max_depth - 1, keep_keys) for x in obj[-10:]]
    return obj


def json_section(title: str, path: Path, budget: int) -> str:
    obj = parse_json(path)
    if obj is None:
        return f"### {title}\n{path}: NOT PARSEABLE or missing\n"
    compact = compact_json(obj, max_depth=5, keep_keys=CONFIG_KEEP_KEYS)
    text = json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True)
    return f"### {title}\nFILE: {path}\n{text}\n"


def csv_section(title: str, path: Path, budget: int) -> str:
    try:
        text, nul = read_text(path)
        note = f"NUL_COUNT={nul}\n" if nul else ""
        return f"### {title}\nFILE: {path}\n{note}{truncate_text(text, budget, title)}\n"
    except Exception as e:
        return f"### {title}\nFILE: {path}\nERROR reading csv: {e}\n"


def row_epoch(row: Dict[str, Any]) -> Any:
    for k in ("epoch", "epoch_idx", "epoch_index"):
        if k in row:
            return row[k]
    return None


def to_float(v: Any) -> Optional[float]:
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
        if math.isfinite(f):
            return f
        return f
    except Exception:
        return None


def summarize_jsonl(path: Path, title: str, budget: int) -> str:
    rows = parse_jsonl(path)
    out = [f"### {title}", f"FILE: {path}", f"row_count={len(rows)}"]
    if not rows:
        out.append("NO_PARSEABLE_ROWS or EMPTY")
        return "\n".join(out) + "\n"
    out.append("first_row=" + json.dumps(compact_json(rows[0], 3, CONFIG_KEEP_KEYS), ensure_ascii=False, sort_keys=True))
    out.append("last_row=" + json.dumps(compact_json(rows[-1], 3, CONFIG_KEEP_KEYS), ensure_ascii=False, sort_keys=True))

    # Key frequencies.
    key_counter = Counter(k for r in rows for k in r.keys())
    out.append("top_keys=" + json.dumps(key_counter.most_common(30), ensure_ascii=False))

    # Numeric summary for relevant keys.
    num_values: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        for k, v in r.items():
            lk = str(k).lower()
            if any(h in lk for h in NUMERIC_KEY_HINTS):
                f = to_float(v)
                if f is not None:
                    num_values[k].append(f)
    out.append("numeric_summary:")
    for k in sorted(num_values.keys())[:80]:
        vals = num_values[k]
        finite = [x for x in vals if math.isfinite(x)]
        if not finite:
            out.append(f"  {k}: count={len(vals)} nonfinite_only")
            continue
        try:
            p50 = statistics.median(finite)
            p90 = sorted(finite)[int(0.90 * (len(finite) - 1))]
            p99 = sorted(finite)[int(0.99 * (len(finite) - 1))]
            out.append(f"  {k}: count={len(vals)} mean={statistics.mean(finite):.6g} min={min(finite):.6g} p50={p50:.6g} p90={p90:.6g} p99={p99:.6g} max={max(finite):.6g}")
        except Exception:
            out.append(f"  {k}: count={len(vals)}")

    # Per-epoch final rows.
    per_epoch: Dict[Any, Dict[str, Any]] = {}
    for r in rows:
        ep = row_epoch(r)
        if ep is not None:
            per_epoch[ep] = r
    if per_epoch:
        out.append("per_epoch_last_rows:")
        for ep in sorted(per_epoch.keys(), key=lambda x: (str(type(x)), x))[:80]:
            out.append(f"  epoch={ep}: " + json.dumps(compact_json(per_epoch[ep], 2, CONFIG_KEEP_KEYS), ensure_ascii=False, sort_keys=True))

    # Anomaly rows.
    anomalies = []
    for i, r in enumerate(rows):
        s = json.dumps(r, ensure_ascii=False, sort_keys=True)
        lower = s.lower()
        if any(tok in lower for tok in ["nan", "inf", "error", "fail", "oom", "nonfinite", "traceback"]):
            anomalies.append((i, r))
        if len(anomalies) >= 50:
            break
    if anomalies:
        out.append("anomaly_rows_first_50:")
        for i, r in anomalies:
            out.append(f"  row_index={i}: " + json.dumps(compact_json(r, 3, CONFIG_KEEP_KEYS), ensure_ascii=False, sort_keys=True))

    return truncate_text("\n".join(out) + "\n", budget, title)


def normalize_log_line(line: str) -> str:
    s = line.strip()
    # remove timestamps
    s = re.sub(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?", "<TIME>", s)
    s = re.sub(r"\b\d{2}:\d{2}:\d{2}\b", "<TIME>", s)
    # remove tqdm percentages and sample counts
    s = re.sub(r"\b\d{1,3}%\|[^\r\n]*", "<TQDM>", s)
    s = re.sub(r"\b\d+\/\d+\b", "<N>/<N>", s)
    s = re.sub(r"\b\d+\.\d+it/s\b", "<IT/S>", s)
    s = re.sub(r"\bpid=\d+\b|\bPID=\d+\b", "pid=<PID>", s)
    s = re.sub(r"\brank\s*=?\s*\d+\b", "rank=<RANK>", s, flags=re.I)
    s = re.sub(r"/public/home/zhangliml/lc/ODCR/ODCR-main", "<REPO>", s)
    return s


def extract_log_windows(path: Path, title: str, budget: int, dedupe: bool = True) -> str:
    try:
        text, nul = read_text(path)
    except Exception as e:
        return f"### {title}\nFILE: {path}\nERROR reading log: {e}\n"
    lines = text.splitlines()
    out = [f"### {title}", f"FILE: {path}", f"line_count={len(lines)}", f"size_bytes={path.stat().st_size}"]
    if nul:
        out.append(f"NUL_SANITIZED_COUNT={nul}")

    # Begin/end windows.
    out.append("-- BEGIN FIRST 80 LINES --")
    out.extend(lines[:80])
    out.append("-- END FIRST 80 LINES --")

    # High-signal matching lines plus context.
    idxs = set()
    for i, line in enumerate(lines):
        if any(p.lower() in line.lower() for p in KEY_LOG_PATTERNS):
            for j in range(max(0, i - 3), min(len(lines), i + 4)):
                idxs.add(j)
    # Also include last 120 lines.
    for j in range(max(0, len(lines) - 120), len(lines)):
        idxs.add(j)
    ordered = sorted(idxs)

    if dedupe:
        seen = Counter()
        filtered = []
        for i in ordered:
            norm = normalize_log_line(lines[i])
            seen[norm] += 1
            if seen[norm] <= 2:  # keep first two examples
                filtered.append((i, lines[i]))
        repeated = [(k, v) for k, v in seen.items() if v > 2]
    else:
        filtered = [(i, lines[i]) for i in ordered]
        repeated = []

    out.append("-- HIGH SIGNAL WINDOWS (deduped) --")
    last_i = None
    for i, line in filtered[:3000]:
        if last_i is not None and i > last_i + 1:
            out.append(f"[... skipped lines {last_i+1}-{i-1} ...]")
        out.append(f"L{i+1}: {line}")
        last_i = i
    if repeated:
        out.append("-- REPEATED NORMALIZED LINES OMITTED --")
        for norm, count in sorted(repeated, key=lambda kv: kv[1], reverse=True)[:80]:
            out.append(f"[repeated {count}x] {norm[:240]}")

    return truncate_text("\n".join(out) + "\n", budget, title)


def summarize_samples(path: Path, title: str, budget: int, max_samples: int) -> str:
    rows = parse_jsonl(path)
    out = [f"### {title}", f"FILE: {path}", f"row_count={len(rows)}"]
    if not rows:
        out.append("samples.jsonl is EMPTY or has no parseable rows")
        return "\n".join(out) + "\n"

    def get_text(row: Dict[str, Any], keys: Sequence[str]) -> str:
        for k in keys:
            v = row.get(k)
            if isinstance(v, str):
                return v
        return ""

    pred_keys = ["prediction", "pred_text", "pred", "generated", "output", "decoded_pred"]
    tgt_keys = ["target", "target_text", "gold", "reference", "review", "decoded_target"]
    empty = 0
    pred_lens = []
    tgt_lens = []
    pred_counter = Counter()
    for r in rows:
        p = get_text(r, pred_keys)
        t = get_text(r, tgt_keys)
        if not p.strip():
            empty += 1
        pred_lens.append(len(p.split()))
        tgt_lens.append(len(t.split()))
        if p.strip():
            pred_counter[p.strip()] += 1
    out.append(f"empty_pred_count={empty} empty_pred_rate={empty/max(1,len(rows)):.4f}")
    out.append(f"avg_pred_len={statistics.mean(pred_lens):.3f} avg_target_len={statistics.mean(tgt_lens):.3f}")
    out.append("top_repeated_predictions=" + json.dumps(pred_counter.most_common(20), ensure_ascii=False))

    # Representative rows.
    random.seed(42)
    picks = []
    picks.extend(range(min(30, len(rows))))
    picks.extend(range(max(0, len(rows) - 30), len(rows)))
    if len(rows) > 60:
        picks.extend(random.sample(range(len(rows)), min(50, len(rows))))
    # empty preds first
    for i, r in enumerate(rows):
        if i in picks:
            continue
        if not get_text(r, pred_keys).strip():
            picks.append(i)
        if len(picks) >= max_samples:
            break
    picks = sorted(set(picks))[:max_samples]
    out.append(f"representative_rows_count={len(picks)}")
    for i in picks:
        out.append(f"row_index={i}: " + json.dumps(compact_json(rows[i], 3, CONFIG_KEEP_KEYS), ensure_ascii=False, sort_keys=True))
    return truncate_text("\n".join(out) + "\n", budget, title)


def binary_section(path: Path, rel: str) -> str:
    st = path.stat()
    try:
        sha = sha256_file(path)
    except Exception as e:
        sha = f"ERROR:{e}"
    return f"### BINARY ARTIFACT\nFILE: {path}\nrelative_path={rel}\nsize_bytes={st.st_size}\nsha256={sha}\ncontent=NOT_COPIED\n"


def summarize_file(info: FileInfo, run_root: Path, detail: str, max_samples: int, dedupe: bool) -> str:
    path = info.path
    title = f"{info.rel} ({info.priority})"
    budget = info.max_bytes
    if detail == "old" and info.section not in {"checkpoint_lineage_sidecars", "errors_traceback_log_windows", "epoch_metrics_loss", "cache_tokenizer_manifest"}:
        budget = max(5000, budget // 3)
    if info.parse_type == "binary":
        return binary_section(path, info.rel)
    if info.priority == "skip":
        return f"### SKIPPED LOW PRIORITY\nFILE: {path}\nsize_bytes={info.size}\nreason={info.reason}\n"
    if info.parse_type == "json":
        return truncate_text(json_section(title, path, budget), budget, title)
    if info.parse_type == "csv":
        return csv_section(title, path, budget)
    if info.parse_type == "jsonl":
        if info.rel.endswith("samples.jsonl"):
            return summarize_samples(path, title, budget, max_samples)
        return summarize_jsonl(path, title, budget)
    if info.parse_type in {"log", "text"}:
        return extract_log_windows(path, title, budget, dedupe=dedupe)
    return extract_log_windows(path, title, budget, dedupe=dedupe)


def run_header(info: RunInfo) -> str:
    rs = parse_json(info.run_root / "meta" / "run_summary.json") or {}
    manifest = parse_json(info.run_root / "meta" / "manifest.json") or {}
    latest_marker = "true" if info.latest else "false"
    obj = {
        "task": info.task,
        "run_id": info.run_id,
        "run_root": str(info.run_root),
        "role": info.role,
        "latest": latest_marker,
        "status": info.status,
        "quality_status": info.quality_status,
        "downstream_ready": info.downstream_ready,
        "run_summary_status": rs.get("status"),
        "validation_status": rs.get("validation_status"),
        "command": rs.get("command") or manifest.get("command"),
        "started_at": rs.get("started_at"),
        "finished_at": rs.get("finished_at"),
        "duration_s": rs.get("duration_s") or rs.get("duration"),
        "profile": rs.get("profile_id") or rs.get("active_profile") or manifest.get("profile_id"),
        "candidate": rs.get("candidate") or manifest.get("candidate"),
    }
    return "### RUN HEADER\n" + json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def determine_roles(runs: List[RunInfo], mode: str) -> None:
    if mode == "manual-run":
        for r in runs:
            r.role = "manual_detail"
        return
    for r in runs:
        status = (r.status or "").lower()
        quality = (r.quality_status or "").lower()
        if r.latest:
            r.role = "latest_detail"
        elif status in FAILED_STATUSES:
            r.role = "old_failed_delta"
        elif quality in {"blocked", "quality_blocked"}:
            r.role = "old_quality_blocked_delta"
        else:
            r.role = "old_delta"


def build_inventory_csv(runs: List[RunInfo]) -> str:
    import io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["run_id", "relative_path", "file_type", "size_bytes", "line_count", "sha256", "parse_type", "extraction_priority", "reason", "expected_section", "duplicate_group", "max_bytes_recommended"])
    for r in runs:
        for f in r.files:
            writer.writerow([r.run_id, f.rel, f.file_type, f.size, f.line_count if f.line_count is not None else "", f.sha256, f.parse_type, f.priority, f.reason, f.section, f.duplicate_group, f.max_bytes])
    return buf.getvalue()


def risk_matrix(runs: List[RunInfo]) -> str:
    out = ["## Risk matrix / extraction notes"]
    for r in runs:
        status = (r.status or "").lower()
        quality = (r.quality_status or "").lower()
        out.append(f"- task={r.task} run={r.run_id} role={r.role} status={r.status} quality_status={r.quality_status} downstream_ready={r.downstream_ready}")
        if status in FAILED_STATUSES:
            out.append("  - RISK: failed/interrupted run; keep root cause traceback and downstream rejection evidence.")
        if quality in {"blocked", "quality_blocked"}:
            out.append("  - RISK: quality-blocked run; keep best epoch mismatch, grad/nonfinite, sample collapse, and checkpoint evidence.")
        if is_running(r):
            out.append("  - LIVE SNAPSHOT: run appears running/pending/incomplete; final audit.log should normally refuse unless --allow-running-snapshot.")
    out.append("- Never compare Step3 diagnostic text metrics directly to paper final metrics unless protocol is confirmed comparable.")
    return "\n".join(out) + "\n"


def assemble_pack(runs: List[RunInfo], args: argparse.Namespace) -> str:
    sections: List[str] = []
    now = datetime.now(timezone.utc).astimezone().isoformat()
    sections.append("=" * 30 + "\nODCR_STEP3_EVIDENCE_PACK\n" + "=" * 30)
    sections.append(f"generated_at={now}")
    sections.append(f"mode={args.mode}")
    sections.append(f"target_bytes={args.target_bytes}")
    sections.append(f"dedupe={args.dedupe}")
    sections.append(f"run_count={len(runs)}")
    sections.append(PAPER_REFERENCE)

    sections.append("\n## Run index\n")
    for r in runs:
        sections.append(run_header(r))

    # Inventory compact summary.
    sections.append("\n## File inventory summary\n")
    for r in runs:
        total = sum(f.size for f in r.files)
        sections.append(f"task={r.task} run={r.run_id} files={len(r.files)} total_size_bytes={total} role={r.role}")
        for f in r.files[:200]:
            sections.append(f"  - {f.rel} size={f.size} parse={f.parse_type} priority={f.priority} section={f.section}")

    # Detailed content.
    for r in runs:
        sections.append("\n" + "=" * 30 + f"\nRUN task={r.task} run={r.run_id} role={r.role}\n" + "=" * 30 + "\n")
        detail = "latest" if r.role in {"latest_detail", "manual_detail"} else "old"
        # Prioritize files.
        ordered_files = sorted(r.files, key=lambda f: (
            0 if f.rel.endswith("run_summary.json") else
            1 if f.rel.endswith("epoch_summary.csv") else
            2 if f.rel.endswith("best_event.json") or f.rel.endswith("checkpoint_lineage.json") else
            3 if f.rel.endswith("errors.log") else
            4 if f.rel.endswith("full.log") else
            5 if f.rel.endswith("metrics.jsonl") or f.rel.endswith("loss_breakdown.jsonl") else
            6 if f.rel.endswith("timing_profile.jsonl") or f.rel.endswith("gpu_profile.jsonl") else
            7 if f.rel.endswith("samples.jsonl") else
            8 if f.parse_type == "binary" else 9,
            f.rel
        ))
        for f in ordered_files:
            # Old runs: include high value only.
            if detail == "old" and f.priority == "skip":
                continue
            if detail == "old" and f.rel.endswith("debug.log"):
                continue
            sections.append(summarize_file(f, r.run_root, detail, args.max_samples, args.dedupe))

    sections.append("\n" + risk_matrix(runs))
    sections.append("\n## Omitted content policy\n")
    sections.append("- Binary checkpoint/optimizer bodies are never copied; only size/hash/path are recorded.")
    sections.append("- Tqdm/progress spam, repeated warnings, duplicate debug lines, verbose source table repetition, and long old-run logs are deduplicated or truncated.")
    sections.append("- Latest/manual run gets detailed extraction; older runs get delta/root-cause evidence.")
    sections.append("- If this file is smaller than expected, check whether run files are absent, binary-only, empty, or still live/running.")

    pack = "\n".join(sections) + "\n"
    return enforce_global_budget(pack, args.target_bytes, args.hard_max_bytes)


def enforce_global_budget(text: str, target: int, hard_max: int) -> str:
    data = text.encode("utf-8", errors="replace")
    if len(data) <= hard_max:
        return text
    # Preserve header, run index, end risk/pruning notes. This is a last-resort truncation.
    keep = hard_max - 1000
    head = int(keep * 0.72)
    tail = keep - head
    return (
        data[:head].decode("utf-8", errors="replace")
        + f"\n\n[GLOBAL TRUNCATION: original_bytes={len(data)} hard_max={hard_max}. Increase --target-bytes/--hard-max-bytes or use --run-root for one run.]\n\n"
        + data[-tail:].decode("utf-8", errors="replace")
    )


def write_aux_files(out_dir: Path, runs: List[RunInfo], args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_file_inventory.csv").write_text(build_inventory_csv(runs), encoding="utf-8")
    spec = {
        "schema": "odcr_step3_evidence_extraction_spec/1",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "mode": args.mode,
        "target_bytes": args.target_bytes,
        "soft_range": [600000, 900000],
        "hard_max_bytes": args.hard_max_bytes,
        "manual_run_supported": True,
        "allow_running_snapshot_default": False,
        "dedupe_rules": [
            "normalize timestamps/pids/ranks/tqdm before line-level dedupe",
            "keep first examples of repeated warnings and record counts",
            "binary artifacts hash-only",
            "latest/manual run detailed; old runs delta/root-cause only",
        ],
        "never_drop": [
            "run status", "quality_status/downstream_ready", "epoch_summary",
            "checkpoint lineage key fields", "failure traceback", "cache hit/miss reason",
            "grad_inf/nonfinite evidence", "samples empty/collapse stats",
        ],
        "runs": [
            {"task": r.task, "run_id": r.run_id, "role": r.role, "status": r.status, "quality_status": r.quality_status, "file_count": len(r.files), "run_root": str(r.run_root)}
            for r in runs
        ],
    }
    (out_dir / "extraction_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "extraction_spec.md").write_text(
        "# ODCR Step3 Evidence Extraction Spec\n\n"
        "This spec was generated by extract_step3_evidence_pack.py. It supports scan-all and manual-run modes, refuses live/running runs by default, keeps binary artifacts hash-only, and targets an 800KB-class audit.log.\n\n"
        + json.dumps(spec, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "dedupe_plan.md").write_text(
        "# Dedupe plan\n\n"
        "- Remove tqdm/progress spam; keep start/mid/end/speed samples.\n"
        "- Normalize timestamps/pids/ranks/absolute paths.\n"
        "- Group repeated warnings by normalized line and count.\n"
        "- Merge overlapping log windows.\n"
        "- Keep errors.log traceback over duplicated debug/full tracebacks.\n"
        "- Old runs keep delta/root-cause, latest/manual run gets detail.\n",
        encoding="utf-8",
    )
    (out_dir / "size_budget_plan.md").write_text(
        "# Size budget plan\n\n"
        f"target_bytes={args.target_bytes}\n"
        "soft_min=600000\nsoft_max=900000\nhard_max=" + str(args.hard_max_bytes) + "\n\n"
        "Suggested sections: header 20KB, run index 40KB, latest/manual summary 120KB, epoch/metrics 140KB, checkpoint 120KB, cache 80KB, errors/log windows 100KB, timing/gpu 100KB, samples/eval 100KB, old-run deltas 80KB.\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract ODCR Step3 evidence pack into audit.log")
    p.add_argument("--root", type=Path, default=None, help="Step3 root, e.g. /.../runs/step3")
    p.add_argument("--run-root", type=Path, default=None, help="Manual run root, e.g. /.../runs/step3/task2/2")
    p.add_argument("--output", type=Path, required=True, help="Output audit.log path")
    p.add_argument("--target-bytes", type=int, default=850000)
    p.add_argument("--hard-max-bytes", type=int, default=1000000)
    p.add_argument("--mode", choices=["scan-all", "manual-run"], default=None)
    p.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow-running-snapshot", action="store_true")
    p.add_argument("--max-samples", type=int, default=150)
    p.add_argument("--dry-run-preview", action="store_true", help="Print summary and do not write audit.log")
    p.add_argument("--aux-dir", type=Path, default=None, help="Optional directory for inventory/spec outputs")
    return p.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if args.mode is None:
        args.mode = "manual-run" if args.run_root else "scan-all"
    if args.mode == "manual-run" and not args.run_root:
        print("ERROR: --mode manual-run requires --run-root", file=sys.stderr)
        return 2
    if args.mode == "scan-all" and not args.root:
        # Infer from output cwd if possible.
        args.root = Path.cwd() / "runs" / "step3"

    if args.mode == "manual-run":
        run_roots = [args.run_root.resolve()]
        step3_root = args.run_root.resolve().parents[1] if len(args.run_root.resolve().parents) >= 2 else args.run_root.resolve()
    else:
        step3_root = args.root.resolve()
        run_roots = discover_runs(step3_root)

    if not run_roots:
        print("ERROR: no run roots found", file=sys.stderr)
        return 2

    latest_map = load_latest_map(step3_root)
    runs = [inventory_run(rr) for rr in run_roots]
    for r in runs:
        r.latest = latest_map.get(r.task) == r.run_id
    determine_roles(runs, args.mode)

    running = [r for r in runs if is_running(r)]
    if running and not args.allow_running_snapshot:
        print("ERROR: one or more selected runs look live/running/incomplete; refusing to write final audit.log.", file=sys.stderr)
        for r in running:
            print(f"  task={r.task} run={r.run_id} status={r.status} root={r.run_root}", file=sys.stderr)
        print("Use --allow-running-snapshot only if you explicitly want a partial live evidence pack.", file=sys.stderr)
        return 3

    aux_dir = args.aux_dir
    if aux_dir is None:
        # Place next to output under AI_analysis if possible, otherwise output parent.
        aux_dir = args.output.parent / "AI_analysis" / "06_probe_evidence" / "step3_evidence_pack_extractor"
    try:
        write_aux_files(aux_dir, runs, args)
    except Exception as e:
        print(f"WARN: failed writing auxiliary spec files: {e}", file=sys.stderr)

    pack = assemble_pack(runs, args)
    if args.dry_run_preview:
        print("DRY RUN PREVIEW")
        print(f"runs={[(r.task, r.run_id, r.role, r.status) for r in runs]}")
        print(f"output_bytes={len(pack.encode('utf-8', errors='replace'))}")
        print(f"aux_dir={aux_dir}")
        print(pack[:5000])
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(pack, encoding="utf-8")
    print(f"WROTE {args.output} bytes={len(pack.encode('utf-8', errors='replace'))}")
    print(f"AUX {aux_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
