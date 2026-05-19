"""Step4-owned RCR pool exports for Step5 sampling.

This replaces the old gold-heavy dedicated train-table semantics as the
default Step5 handoff.  The old files may remain for legacy/ablation only.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import pandas as pd
import pyarrow.parquet as pq

from odcr_core.file_atomic import atomic_write_json
from odcr_core.gold_quality import (
    GOLD_PROXY_COLUMNS,
    add_step5_quality_columns,
    default_cf_tier_config,
    default_gold_quality_config,
)
from odcr_core.index_contract import INDEX_CONTRACT_FILENAME, ODCR_ROUTING_TRAIN_CSV
from odcr_core.step4_dedicated_exports import (
    FULL_AUDIT_PARQUET,
    STEP5A_SCORER_TRAIN_COLUMNS,
    STEP5B_EXPLAINER_TRAIN_COLUMNS,
    STEP5_DEDICATED_SOURCE_REQUIRED_COLUMNS,
    _ParquetSink,
    _file_sha256,
    _fsync_parent,
    _load_json,
    _normalize_chunk,
    _repo_relative,
    _validate_required_columns,
)
from odcr_core.step4_export_validator import STEP4_EXPORT_MANIFEST, validate_step4_export_ready
from odcr_core.step5_pool_sampler import (
    POOL_NAMES,
    POOL_PARQUET_NAMES,
    STEP5_POOL_DISTRIBUTION_REPORT,
    STEP5_POOL_EXPORTS_STATUS,
    STEP5_POOL_MANIFEST,
    STEP5_POOL_MANIFEST_SCHEMA_VERSION,
    STEP5_POOLS_DIRNAME,
    STEP5_SAMPLING_CONTRACT,
    STEP5_SAMPLING_CONTRACT_SCHEMA_VERSION,
)
from odcr_core.step5_prompt_templates import prompt_registry_manifest


STEP4_POOL_EXPORTS_SCHEMA_VERSION = STEP5_POOL_MANIFEST_SCHEMA_VERSION
STEP4_POOL_EXPORTS_STATUS_SCHEMA_VERSION = "odcr_step4_step5_pool_exports_status/1"
STEP4_LEGACY_DEDICATED_EXPORTS_STATUS = "legacy_old_filter_exports_status.json"

POOL_EXTRA_COLUMNS: tuple[str, ...] = (
    "gold_anchor_quality",
    "gold_quality_score",
    "gold_quality_tier",
    "gold_quality_reasons",
    "gold_quality_reject_reason",
    "hard_reject_flags",
    "coverage_bucket",
    "cf_tier",
    "cf_tier_reason",
    "cf_tier_step5A",
    "cf_tier_reason_step5A",
    "cf_tier_step5B",
    "cf_tier_reason_step5B",
    "cf_quality_score_step5A",
    "cf_quality_score_step5B",
    "recommended_sampling_weight",
    "recommended_sampling_weight_step5A",
    "recommended_sampling_weight_step5B",
    "rating_bucket",
    "length_bucket",
    "explanation_length_bucket",
    "explanation_length_words",
    "ngram_repeat_ratio",
    "user_coverage_bucket",
    "item_coverage_bucket",
    "domain_role",
    *GOLD_PROXY_COLUMNS,
    "step5_pool_contract_version",
)

POOL_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys((*STEP5A_SCORER_TRAIN_COLUMNS, *STEP5B_EXPLAINER_TRAIN_COLUMNS, *POOL_EXTRA_COLUMNS))
)


class Step4PoolExportError(RuntimeError):
    """Raised when Step4 cannot publish trusted Step5 pools."""


@dataclass
class Step4PoolExportValidationResult:
    ready: bool
    run_dir: Path
    status_path: Path
    manifest_path: Path
    sampling_contract_path: Path
    report_path: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_payload(self, repo_root: str | Path | None = None) -> dict[str, Any]:
        root = Path(repo_root).expanduser().resolve() if repo_root is not None else None

        def rel(path: Path | None) -> str | None:
            if path is None:
                return None
            p = path.resolve()
            if root is None:
                return p.as_posix()
            try:
                return p.relative_to(root).as_posix()
            except ValueError:
                return p.as_posix()

        return {
            "schema_version": STEP4_POOL_EXPORTS_STATUS_SCHEMA_VERSION,
            "ready": bool(self.ready),
            "run_dir": rel(self.run_dir),
            "status_path": rel(self.status_path),
            "manifest_path": rel(self.manifest_path),
            "sampling_contract_path": rel(self.sampling_contract_path),
            "report_path": rel(self.report_path),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "diagnostics": dict(self.diagnostics),
        }


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_pool_config(raw: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(raw or {})
    cfg.setdefault("enabled", True)
    cfg.setdefault("output_dir_name", STEP5_POOLS_DIRNAME)
    cfg.setdefault("chunk_rows", 100_000)
    cfg.setdefault("full_audit_role", "audit_only")
    cfg.setdefault("legacy_dedicated_exports_role", "legacy_old_filter_exports")
    return cfg


def _stats(series: pd.Series) -> dict[str, Any]:
    vals = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if vals.empty:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": int(len(vals)),
        "min": float(vals.min()),
        "mean": float(vals.mean()),
        "max": float(vals.max()),
    }


def _merge_stats(target: MutableMapping[str, Any], series: pd.Series) -> None:
    cur = _stats(series)
    target["count"] = int(target.get("count", 0)) + int(cur["count"])
    for stat in ("min", "max"):
        if cur[stat] is None:
            continue
        if target.get(stat) is None:
            target[stat] = cur[stat]
        elif stat == "min":
            target[stat] = min(float(target[stat]), float(cur[stat]))
        else:
            target[stat] = max(float(target[stat]), float(cur[stat]))
    target["_sum"] = float(target.get("_sum", 0.0)) + float(cur["mean"] or 0.0) * int(cur["count"])
    target["mean"] = float(target["_sum"] / max(int(target["count"]), 1)) if int(target["count"]) else None


def _strip_stat_sums(value: Any) -> None:
    if isinstance(value, MutableMapping):
        value.pop("_sum", None)
        for nested in value.values():
            _strip_stat_sums(nested)
    elif isinstance(value, list):
        for nested in value:
            _strip_stat_sums(nested)


def _update_counts(store: dict[str, int], series: pd.Series) -> None:
    for key, val in series.value_counts(dropna=False).items():
        store[str(key)] = int(store.get(str(key), 0)) + int(val)


def _gold_quality_sanity(counts: Mapping[str, Mapping[str, int]], cfg: Mapping[str, Any]) -> dict[str, Any]:
    sanity = cfg.get("sanity") if isinstance(cfg.get("sanity"), Mapping) else {}
    medium_min = float(sanity.get("medium_min_ratio", 0.05))
    high_max = float(sanity.get("high_max_ratio", 0.80))
    reject_warn = float(sanity.get("reject_warn_ratio", 0.40))
    out: dict[str, Any] = {
        "medium_min_ratio": medium_min,
        "high_max_ratio": high_max,
        "reject_warn_ratio": reject_warn,
        "by_origin": {},
        "p1_flags": [],
    }
    for origin, item in counts.items():
        total = max(1, sum(int(v) for v in item.values()))
        ratios = {tier: float(int(item.get(tier, 0)) / total) for tier in ("high", "medium", "reject")}
        flags: list[str] = []
        if ratios["medium"] < medium_min:
            flags.append("medium_below_sanity_min")
        if ratios["high"] > high_max:
            flags.append("high_above_sanity_max")
        if ratios["reject"] > reject_warn:
            flags.append("reject_above_warn_ratio")
        if flags:
            out["p1_flags"].extend([f"{origin}:{flag}" for flag in flags])
        out["by_origin"][origin] = {
            "total": int(total),
            "counts": {tier: int(item.get(tier, 0)) for tier in ("high", "medium", "reject")},
            "ratios": ratios,
            "flags": flags,
        }
    out["passes"] = not bool(out["p1_flags"])
    return out


def _pool_record(repo_root: Path, path: Path, *, role: str, tier: str, row_count: int, columns: Sequence[str]) -> dict[str, Any]:
    return {
        "path": _repo_relative(repo_root, path),
        "role": role,
        "tier": tier,
        "row_count": int(row_count),
        "column_count": int(len(columns)),
        "columns": list(columns),
        "format": "parquet",
        "size_bytes": int(path.stat().st_size),
        "sha256": _file_sha256(path),
    }


def _legacy_status(root: Path, run_dir: Path, legacy_dir: Path) -> dict[str, Any]:
    items: dict[str, Any] = {}
    for name in ("step5A_scorer_train.parquet", "step5B_explainer_train.parquet", "odcr_routing_full_audit.parquet"):
        path = legacy_dir / name
        items[name] = {
            "path": _repo_relative(root, path),
            "exists": path.is_file(),
            "role": "legacy_old_filter_exports" if name != "odcr_routing_full_audit.parquet" else "legacy_full_audit",
            "not_default_step5_train": True,
            "gold_heavy_warning": name != "odcr_routing_full_audit.parquet",
            "sha256": _file_sha256(path) if path.is_file() else None,
        }
    payload = {
        "schema_version": "odcr_step4_legacy_old_filter_exports/1",
        "generated_at_utc": _now(),
        "producer_stage": "step4",
        "run_dir": _repo_relative(root, run_dir),
        "legacy_export_policy": "ablation_or_history_only",
        "not_default_step5_train": True,
        "gold_heavy_warning": True,
        "exports": items,
    }
    out = legacy_dir / STEP4_LEGACY_DEDICATED_EXPORTS_STATUS
    legacy_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, payload)
    return {**payload, "path": _repo_relative(root, out), "sha256": _file_sha256(out)}


def export_step4_pool_exports(
    *,
    repo_root: str | Path,
    task: int,
    from_run: str,
    pool_config: Mapping[str, Any],
    gold_quality_config: Mapping[str, Any] | None = None,
    cf_tier_config: Mapping[str, Any] | None = None,
    sampler_config: Mapping[str, Any],
    dry_run: bool = False,
    update_stage_status: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    cfg = _default_pool_config(pool_config)
    gold_cfg = default_gold_quality_config(gold_quality_config)
    cf_cfg = default_cf_tier_config(cf_tier_config)
    if not bool(cfg.get("enabled", True)):
        raise Step4PoolExportError("step4.step5_pool_exports.enabled must be true")
    run_dir = root / "runs" / "step4" / f"task{int(task)}" / str(from_run)
    source_csv = run_dir / ODCR_ROUTING_TRAIN_CSV
    source_manifest_path = run_dir / STEP4_EXPORT_MANIFEST
    source_index_path = run_dir / INDEX_CONTRACT_FILENAME
    output_dir = run_dir / str(cfg.get("output_dir_name") or STEP5_POOLS_DIRNAME)
    header = tuple(str(c) for c in pd.read_csv(source_csv, nrows=0).columns)
    _validate_required_columns(header, STEP5_DEDICATED_SOURCE_REQUIRED_COLUMNS, context="Step4 full export")
    base_validation = validate_step4_export_ready(run_dir, repo_root=root)
    if not base_validation.ready:
        raise Step4PoolExportError("source Step4 export is not ready: " + "; ".join(base_validation.errors))
    source_sha = str(base_validation.diagnostics.get("export_sha256") or _file_sha256(source_csv))
    if dry_run:
        return {
            "schema_version": STEP4_POOL_EXPORTS_SCHEMA_VERSION,
            "dry_run": True,
            "task": int(task),
            "step4_run": str(from_run),
            "source_full_export": _repo_relative(root, source_csv),
            "source_full_export_sha256": source_sha,
            "output_dir": _repo_relative(root, output_dir),
            "planned_exports": {"full_audit": FULL_AUDIT_PARQUET, **POOL_PARQUET_NAMES},
            "sampling_contract_schema_version": STEP5_SAMPLING_CONTRACT_SCHEMA_VERSION,
            "gold_quality_schema_version": gold_cfg.get("schema_version"),
            "cf_tier_schema_version": cf_cfg.get("schema_version"),
            "legacy_old_filter_exports_marked": True,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    sinks: dict[str, _ParquetSink] = {"full_audit": _ParquetSink(output_dir / FULL_AUDIT_PARQUET, header)}
    for name in POOL_NAMES:
        sinks[name] = _ParquetSink(output_dir / POOL_PARQUET_NAMES[name], POOL_COLUMNS)

    chunk_rows = max(1, int(cfg.get("chunk_rows", 100_000)))
    total_rows = 0
    origin_counts: dict[str, int] = {}
    pool_counts: dict[str, int] = {name: 0 for name in POOL_NAMES}
    gold_quality_counts: dict[str, dict[str, int]] = {
        "target_gold": {"high": 0, "medium": 0, "reject": 0},
        "aux_gold": {"high": 0, "medium": 0, "reject": 0},
    }
    cf_tier_counts: dict[str, dict[str, int]] = {
        "step5A": {"high": 0, "medium": 0, "low_weighted": 0, "reject": 0},
        "step5B": {"high": 0, "medium": 0, "low_weighted": 0, "reject": 0},
    }
    distribution: dict[str, Any] = {
        "rating_bucket": {},
        "confidence": {},
        "uncertainty": {},
        "sample_weight": {},
        "text_quality": {},
        "score_summaries": {
            "gold_quality": {},
            "cf_quality_step5A": {},
            "cf_quality_step5B": {},
            "uncertainty": {},
            "sample_weight": {},
            "text_quality": {},
            "style_shift": {},
            "rating_stability": {},
            "content_retention": {},
            "cf_reliability": {},
        },
        "coverage": {"user_nunique_by_origin": {}, "item_nunique_by_origin": {}},
        "coverage_bucket": {},
        "length_bucket": {},
        "hard_reject_flags": {},
        "rejected_reasons": {},
        "by_pool": {
            name: {
                "row_count": 0,
                "sample_origin": {},
                "rating_bucket": {},
                "confidence": {},
                "metrics": {
                    "gold_quality": {},
                    "cf_quality_step5A": {},
                    "cf_quality_step5B": {},
                    "uncertainty": {},
                    "sample_weight": {},
                    "text_quality": {},
                    "style_shift": {},
                    "rating_stability": {},
                    "content_retention": {},
                    "cf_reliability": {},
                },
            }
            for name in POOL_NAMES
        },
    }
    metric_columns = {
        "uncertainty": "uncertainty_score",
        "sample_weight": "sample_weight_hint",
        "text_quality": "text_quality_score",
        "style_shift": "style_shift_score",
        "rating_stability": "rating_stability_score",
        "content_retention": "content_retention_score",
        "cf_reliability": "cf_reliability_score",
        "gold_quality": "gold_quality_score",
        "cf_quality_step5A": "cf_quality_score_step5A",
        "cf_quality_step5B": "cf_quality_score_step5B",
    }

    def _record_pool_distribution(name: str, frame: pd.DataFrame) -> None:
        bucket = distribution["by_pool"][name]
        bucket["row_count"] = int(bucket["row_count"]) + int(len(frame))
        if frame.empty:
            return
        _update_counts(bucket["sample_origin"], frame["sample_origin"].astype(str))
        _update_counts(bucket["rating_bucket"], frame["rating_bucket"].astype(str))
        _update_counts(bucket["confidence"], frame["confidence_bucket"].astype(str))
        for metric, col in metric_columns.items():
            _merge_stats(bucket["metrics"][metric], frame[col])

    dtype = {col: "string" for col in header if col in {"user", "item", "review", "explanation", "clean_text", "domain", "sample_origin"}}
    try:
        for chunk in pd.read_csv(source_csv, chunksize=chunk_rows, dtype=dtype):
            chunk = _normalize_chunk(chunk)
            chunk = add_step5_quality_columns(
                chunk,
                gold_quality_config=gold_cfg,
                cf_tier_config=cf_cfg,
                pool_contract_version=STEP4_POOL_EXPORTS_SCHEMA_VERSION,
            )
            total_rows += int(len(chunk))
            origin = chunk["sample_origin"].astype(str)
            target_gold = origin == "target_gold"
            aux_gold = origin == "aux_gold"
            aux_cf = origin == "aux_cf"
            _update_counts(origin_counts, origin)
            sinks["full_audit"].write(chunk.loc[:, header])

            for origin_name, mask in (("target_gold", target_gold), ("aux_gold", aux_gold)):
                for tier in ("high", "medium", "reject"):
                    count = int((mask & (chunk["gold_quality_tier"].astype(str) == tier)).sum())
                    gold_quality_counts[origin_name][tier] += count
            for head, col in (("step5A", "cf_tier_step5A"), ("step5B", "cf_tier_step5B")):
                for tier in ("high", "medium", "low_weighted", "reject"):
                    cf_tier_counts[head][tier] += int((aux_cf & (chunk[col].astype(str) == tier)).sum())

            for head in ("step5A", "step5B"):
                for origin_name, mask, stem in (
                    ("target_gold", target_gold, f"{head}_target_gold_anchor"),
                    ("aux_gold", aux_gold, f"{head}_aux_gold_anchor"),
                ):
                    for tier in ("high", "medium"):
                        m = mask & (chunk["gold_quality_tier"].astype(str) == tier)
                        name = f"{stem}_{tier}"
                        gold_out = chunk.loc[m, POOL_COLUMNS]
                        sinks[name].write(gold_out)
                        pool_counts[name] += int(m.sum())
                        _record_pool_distribution(name, gold_out)
                cf_col = "cf_tier_step5A" if head == "step5A" else "cf_tier_step5B"
                reason_col = "cf_tier_reason_step5A" if head == "step5A" else "cf_tier_reason_step5B"
                suffix = "cf_scorer" if head == "step5A" else "cf_explainer"
                for tier in ("high", "medium", "low_weighted", "reject"):
                    m = aux_cf & (chunk[cf_col].astype(str) == tier)
                    name = f"{head}_{suffix}_{tier}"
                    out = chunk.loc[m, POOL_COLUMNS].copy()
                    if not out.empty:
                        out["cf_tier"] = tier
                        out["cf_tier_reason"] = chunk.loc[m, reason_col].astype(str)
                    sinks[name].write(out)
                    pool_counts[name] += int(m.sum())
                    _record_pool_distribution(name, out)

            _update_counts(distribution["rating_bucket"], chunk["rating_bucket"])
            _update_counts(distribution["coverage_bucket"], chunk["coverage_bucket"])
            _update_counts(distribution["length_bucket"], chunk["length_bucket"])
            _update_counts(distribution["hard_reject_flags"], chunk["hard_reject_flags"])
            _update_counts(distribution["confidence"], chunk["confidence_bucket"].astype(str))
            for key, col in metric_columns.items():
                _merge_stats(distribution["score_summaries"][key], chunk[col])
            for key in ("uncertainty", "sample_weight", "text_quality"):
                distribution[key] = distribution["score_summaries"][key]
            for org, sub in chunk.groupby("sample_origin", dropna=False):
                distribution["coverage"]["user_nunique_by_origin"][str(org)] = int(
                    distribution["coverage"]["user_nunique_by_origin"].get(str(org), 0)
                ) + int(sub["user_idx_global"].nunique(dropna=True))
                distribution["coverage"]["item_nunique_by_origin"][str(org)] = int(
                    distribution["coverage"]["item_nunique_by_origin"].get(str(org), 0)
                ) + int(sub["item_idx_global"].nunique(dropna=True))
            rejects = chunk.loc[(target_gold | aux_gold) & (chunk["gold_quality_tier"].astype(str) == "reject"), "gold_quality_reject_reason"]
            _update_counts(distribution["rejected_reasons"], rejects)
    except Exception:
        for sink in sinks.values():
            sink.cleanup()
        raise

    sink_records: dict[str, dict[str, Any]] = {}
    try:
        for name, sink in sinks.items():
            sink_records[name] = sink.commit(require_non_empty=name == "full_audit")
    except Exception:
        for sink in sinks.values():
            sink.cleanup()
        raise

    _strip_stat_sums(distribution)

    source_manifest = _load_json(source_manifest_path, label="source Step4 manifest")
    lineage = source_manifest.get("step4_export_lineage") if isinstance(source_manifest.get("step4_export_lineage"), Mapping) else {}
    pools: dict[str, Any] = {}
    for name in POOL_NAMES:
        record = sink_records[name]
        tier = name.rsplit("_", 1)[-1]
        if name.endswith("low_weighted"):
            tier = "low_weighted"
        role = "cf_pool" if "_cf_" in name else "gold_anchor_pool"
        pools[name] = _pool_record(root, Path(record["path"]), role=role, tier=tier, row_count=int(record["row_count"]), columns=POOL_COLUMNS)

    legacy = _legacy_status(root, run_dir, run_dir / "step5_exports")
    gold_cfg_hash = hashlib.sha256(json.dumps(gold_cfg, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    cf_cfg_hash = hashlib.sha256(json.dumps(cf_cfg, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    sampling_contract = {
        "schema_version": STEP5_SAMPLING_CONTRACT_SCHEMA_VERSION,
        "producer_stage": "step4",
        "consumer_stage": "step5",
        "task_id": int(task),
        "step4_run": str(from_run),
        "source": "configs/odcr.yaml: step5.sampler",
        "gold_quality_schema_version": gold_cfg.get("schema_version"),
        "gold_quality_config_hash": gold_cfg_hash,
        "cf_tier_schema_version": cf_cfg.get("schema_version"),
        "cf_tier_config_hash": cf_cfg_hash,
        **dict(sampler_config),
    }
    sampling_contract_path = output_dir / STEP5_SAMPLING_CONTRACT
    atomic_write_json(sampling_contract_path, sampling_contract)

    report = {
        "schema_version": "odcr_step5_pool_distribution_report/1",
        "source_full_export": _repo_relative(root, source_csv),
        "source_full_export_sha256": source_sha,
        "gold_quality_schema_version": gold_cfg.get("schema_version"),
        "gold_quality_config": gold_cfg,
        "gold_quality_config_hash": gold_cfg_hash,
        "cf_tier_schema_version": cf_cfg.get("schema_version"),
        "cf_tier_config": cf_cfg,
        "cf_tier_config_hash": cf_cfg_hash,
        "total_rows": int(total_rows),
        "source_row_counts": {"total_rows": int(total_rows), "by_sample_origin": origin_counts},
        "pool_row_counts": {k: int(v) for k, v in pool_counts.items()},
        "gold_quality_counts": gold_quality_counts,
        "cf_tier_counts": cf_tier_counts,
        "gold_quality_sanity": _gold_quality_sanity(gold_quality_counts, gold_cfg),
        "distribution": distribution,
        "high_medium_sufficiency": {
            "step5A_cf_high_medium": int(cf_tier_counts["step5A"]["high"] + cf_tier_counts["step5A"]["medium"]),
            "step5B_cf_high_medium_low": int(
                cf_tier_counts["step5B"]["high"]
                + cf_tier_counts["step5B"]["medium"]
                + cf_tier_counts["step5B"]["low_weighted"]
            ),
            "recommend_rerun_step4_cf_if_insufficient": True,
        },
        "gold_sampling_guidance": {
            "target_gold_priority": "higher_than_aux_gold",
            "aux_gold_policy": "downsample_and_downweight_as_source_domain_migration_anchor",
            "do_not_select_high_only": True,
        },
    }
    report_path = output_dir / STEP5_POOL_DISTRIBUTION_REPORT
    atomic_write_json(report_path, report)

    manifest = {
        "schema_version": STEP5_POOL_MANIFEST_SCHEMA_VERSION,
        "repo_root": root.as_posix(),
        "producer_stage": "step4",
        "consumer_stage": "step5",
        "task_id": int(task),
        "step4_run": str(from_run),
        "source_full_export": _repo_relative(root, source_csv),
        "source_full_export_sha256": source_sha,
        "source_full_export_size_bytes": int(source_csv.stat().st_size),
        "source_manifest": _repo_relative(root, source_manifest_path),
        "source_manifest_sha256": _file_sha256(source_manifest_path),
        "source_index_contract": _repo_relative(root, source_index_path),
        "source_index_contract_sha256": _file_sha256(source_index_path),
        "source_row_counts": {"total_rows": int(total_rows), "by_sample_origin": origin_counts},
        "full_audit": _pool_record(
            root,
            Path(sink_records["full_audit"]["path"]),
            role="audit_only",
            tier="all",
            row_count=int(sink_records["full_audit"]["row_count"]),
            columns=header,
        ),
        "pools": pools,
        "sampling_contract": _repo_relative(root, sampling_contract_path),
        "sampling_contract_sha256": _file_sha256(sampling_contract_path),
        "distribution_report": _repo_relative(root, report_path),
        "distribution_report_sha256": _file_sha256(report_path),
        "prompt_registry": prompt_registry_manifest(),
        "gold_quality": {
            "schema_version": gold_cfg.get("schema_version"),
            "config_hash": gold_cfg_hash,
            "counts": gold_quality_counts,
            "sanity": _gold_quality_sanity(gold_quality_counts, gold_cfg),
        },
        "cf_tiers": {
            "schema_version": cf_cfg.get("schema_version"),
            "config_hash": cf_cfg_hash,
            "counts": cf_tier_counts,
        },
        "legacy_old_filter_exports": legacy,
        "full_audit_default_train_forbidden": True,
        "legacy_gold_heavy_exports_allowed_by_default": False,
        "gold_default_full_route_removed": True,
        "aux_cf_full_pool_preserved": int(origin_counts.get("aux_cf", 0)) == int(
            cf_tier_counts["step5A"]["high"]
            + cf_tier_counts["step5A"]["medium"]
            + cf_tier_counts["step5A"]["low_weighted"]
            + cf_tier_counts["step5A"]["reject"]
        ),
        "lineage": {
            "source_step4_export_lineage": lineage,
            "source_step4_export_lineage_hash": lineage.get("lineage_hash") if isinstance(lineage, Mapping) else None,
            "pool_contract_hash": hashlib.sha256(json.dumps(pools, sort_keys=True, default=str).encode("utf-8")).hexdigest(),
            "gold_quality_config_hash": gold_cfg_hash,
            "cf_tier_config_hash": cf_cfg_hash,
        },
        "created_at_utc": _now(),
    }
    manifest_path = output_dir / STEP5_POOL_MANIFEST
    atomic_write_json(manifest_path, manifest)
    validation = validate_step4_pool_exports(run_dir, repo_root=root, raise_on_error=True)
    status = write_step5_pool_exports_status(
        repo_root=root,
        run_dir=run_dir,
        validation=validation,
        update_stage_status=bool(update_stage_status),
    )
    return {
        "schema_version": STEP4_POOL_EXPORTS_SCHEMA_VERSION,
        "dry_run": False,
        "task": int(task),
        "step4_run": str(from_run),
        "output_dir": _repo_relative(root, output_dir),
        "source_full_export": _repo_relative(root, source_csv),
        "source_full_export_sha256": source_sha,
        "step5_pool_manifest": _repo_relative(root, manifest_path),
        "step5_sampling_contract": _repo_relative(root, sampling_contract_path),
        "step5_pool_distribution_report": _repo_relative(root, report_path),
        "status_sidecar": _repo_relative(root, run_dir / "meta" / STEP5_POOL_EXPORTS_STATUS),
        "pool_row_counts": {k: int(v) for k, v in pool_counts.items()},
        "gold_quality_counts": gold_quality_counts,
        "cf_tier_counts": cf_tier_counts,
        "status": status,
        "validation": validation.to_payload(root),
    }


def validate_step4_pool_exports(
    run_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    output_dir_name: str = STEP5_POOLS_DIRNAME,
    raise_on_error: bool = False,
) -> Step4PoolExportValidationResult:
    root = Path(repo_root).expanduser().resolve() if repo_root is not None else Path.cwd().resolve()
    run = Path(run_dir).expanduser()
    run = (root / run).resolve() if not run.is_absolute() else run.resolve()
    pool_dir = run / output_dir_name
    manifest_path = pool_dir / STEP5_POOL_MANIFEST
    contract_path = pool_dir / STEP5_SAMPLING_CONTRACT
    report_path = pool_dir / STEP5_POOL_DISTRIBUTION_REPORT
    status_path = run / "meta" / STEP5_POOL_EXPORTS_STATUS
    result = Step4PoolExportValidationResult(
        ready=False,
        run_dir=run,
        status_path=status_path,
        manifest_path=manifest_path,
        sampling_contract_path=contract_path,
        report_path=report_path,
    )
    try:
        manifest = _load_json(manifest_path, label="Step5 pool manifest")
        contract = _load_json(contract_path, label="Step5 sampling contract")
        report = _load_json(report_path, label="Step5 pool distribution report")
        if manifest.get("schema_version") != STEP5_POOL_MANIFEST_SCHEMA_VERSION:
            raise Step4PoolExportError("Step5 pool manifest schema_version mismatch")
        if contract.get("schema_version") != STEP5_SAMPLING_CONTRACT_SCHEMA_VERSION:
            raise Step4PoolExportError("Step5 sampling contract schema_version mismatch")
        if manifest.get("full_audit_default_train_forbidden") is not True:
            raise Step4PoolExportError("full audit must be forbidden as default train")
        if manifest.get("legacy_gold_heavy_exports_allowed_by_default") is not False:
            raise Step4PoolExportError("legacy gold-heavy exports must be disabled by default")
        pools = manifest.get("pools")
        if not isinstance(pools, Mapping):
            raise Step4PoolExportError("manifest missing pools")
        diagnostics_pools: dict[str, Any] = {}
        for name in POOL_NAMES:
            if name not in pools or not isinstance(pools[name], Mapping):
                raise Step4PoolExportError(f"manifest missing pool {name}")
            item = pools[name]
            path = Path(str(item.get("path") or ""))
            if not path.is_absolute():
                path = root / path
            path = path.resolve()
            if not path.is_file():
                raise Step4PoolExportError(f"pool parquet missing: {name}")
            if _file_sha256(path) != str(item.get("sha256") or ""):
                raise Step4PoolExportError(f"pool parquet sha256 mismatch: {name}")
            pf = pq.ParquetFile(str(path))
            expected_rows = item.get("row_count")
            if expected_rows is None:
                expected_rows = -1
            if int(pf.metadata.num_rows) != int(expected_rows):
                raise Step4PoolExportError(f"pool row_count mismatch: {name}")
            _validate_required_columns(tuple(pf.schema_arrow.names), POOL_COLUMNS, context=f"{name} pool")
            diagnostics_pools[name] = {
                "path": _repo_relative(root, path),
                "row_count": int(pf.metadata.num_rows),
                "role": str(item.get("role") or ""),
                "tier": str(item.get("tier") or ""),
            }
        report_counts = report.get("pool_row_counts") if isinstance(report.get("pool_row_counts"), Mapping) else {}
        for name in POOL_NAMES:
            report_rows = report_counts.get(name)
            pool_rows = pools[name].get("row_count")
            if report_rows is None:
                report_rows = -1
            if pool_rows is None:
                pool_rows = -2
            if int(report_rows) != int(pool_rows):
                raise Step4PoolExportError(f"distribution report row_count mismatch for {name}")
        result.ready = True
        result.diagnostics = {
            "step5_pool_manifest_sha256": _file_sha256(manifest_path),
            "step5_sampling_contract_sha256": _file_sha256(contract_path),
            "step5_pool_distribution_report_sha256": _file_sha256(report_path),
            "pools": diagnostics_pools,
            "gold_quality_counts": report.get("gold_quality_counts"),
            "cf_tier_counts": report.get("cf_tier_counts"),
            "full_audit_default_train_forbidden": True,
            "legacy_gold_heavy_exports_allowed_by_default": False,
        }
    except Exception as exc:
        result.errors.append(str(exc))
        if raise_on_error:
            raise
    return result


def step4_pool_stage_status_fields(
    *,
    repo_root: str | Path,
    run_dir: str | Path,
    validation: Step4PoolExportValidationResult | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    run = Path(run_dir).expanduser()
    run = (root / run).resolve() if not run.is_absolute() else run.resolve()
    validation = validation or validate_step4_pool_exports(run, repo_root=root)
    fields: dict[str, Any] = {
        "step5_pool_exports_ready": bool(validation.ready),
        "step5_train_input_role": "pool_manifest_sampling_contract" if validation.ready else None,
        "full_audit_default_train_forbidden": True,
        "legacy_gold_heavy_exports_allowed_by_default": False,
        "step5_pool_exports_status": _repo_relative(root, run / "meta" / STEP5_POOL_EXPORTS_STATUS),
        "step5_pool_manifest": _repo_relative(root, validation.manifest_path),
        "step5_sampling_contract": _repo_relative(root, validation.sampling_contract_path),
        "step5_pool_distribution_report": _repo_relative(root, validation.report_path),
        "pool_export_readiness": validation.to_payload(root),
    }
    return fields


def write_step5_pool_exports_status(
    *,
    repo_root: str | Path,
    run_dir: str | Path,
    validation: Step4PoolExportValidationResult,
    update_stage_status: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    run = Path(run_dir).expanduser()
    run = (root / run).resolve() if not run.is_absolute() else run.resolve()
    meta = run / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    stage_status_path = meta / "stage_status.json"
    previous_stage_status_sha = _file_sha256(stage_status_path) if stage_status_path.is_file() else None
    fields = step4_pool_stage_status_fields(repo_root=root, run_dir=run, validation=validation)
    payload = {
        "schema_version": STEP4_POOL_EXPORTS_STATUS_SCHEMA_VERSION,
        "generated_at_utc": _now(),
        "producer_stage": "step4",
        "task_id": int(run.parent.name.replace("task", "")) if run.parent.name.startswith("task") else 0,
        "step4_run": run.name,
        "run_dir": _repo_relative(root, run),
        "previous_stage_status_sha256": previous_stage_status_sha,
        "stage_status_update": "applied_after_sidecar_write" if update_stage_status and stage_status_path.is_file() else "not_requested_or_missing",
        **fields,
        "validation": validation.to_payload(root),
    }
    status_path = meta / STEP5_POOL_EXPORTS_STATUS
    atomic_write_json(status_path, payload)
    if update_stage_status and stage_status_path.is_file() and validation.ready:
        status = _load_json(stage_status_path, label="stage_status")
        status.update(fields)
        status.setdefault("artifacts", {})
        if isinstance(status["artifacts"], dict):
            for key in (
                "step5_pool_manifest",
                "step5_sampling_contract",
                "step5_pool_distribution_report",
                "step5_pool_exports_status",
            ):
                path_text = status.get(key)
                if path_text:
                    path = Path(path_text)
                    if not path.is_absolute():
                        path = root / path
                    status["artifacts"][key] = {
                        "path": _repo_relative(root, path),
                        "exists": path.is_file(),
                        "is_file": path.is_file(),
                        "sha256": _file_sha256(path) if path.is_file() else None,
                    }
        status["updated_at"] = _now()
        status["updated_at_utc"] = status["updated_at"]
        atomic_write_json(stage_status_path, status)
    _fsync_parent(status_path)
    return payload


__all__ = [
    "POOL_COLUMNS",
    "STEP4_LEGACY_DEDICATED_EXPORTS_STATUS",
    "STEP4_POOL_EXPORTS_SCHEMA_VERSION",
    "STEP4_POOL_EXPORTS_STATUS_SCHEMA_VERSION",
    "Step4PoolExportError",
    "Step4PoolExportValidationResult",
    "export_step4_pool_exports",
    "step4_pool_stage_status_fields",
    "validate_step4_pool_exports",
    "write_step5_pool_exports_status",
]
