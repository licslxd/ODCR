"""Step5 pool-manifest loader and deterministic effective-epoch sampler."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from odcr_core.file_atomic import atomic_write_json
from odcr_core.step5_auto_budget import effective_samples_for_head
from odcr_core.step5_prompt_templates import default_prompt_registry, prompt_registry_manifest
from odcr_core.step5_task_decoupled_policy import (
    assert_step5a_policy_clean,
    enforce_step5a_target_gold_counts,
    normalized_actual_counts,
)
from odcr_core.training_checkpoint import file_fingerprint, stable_hash


STEP5_POOL_MANIFEST_SCHEMA_VERSION = "odcr_step5_pool_manifest/1"
STEP5_SAMPLING_CONTRACT_SCHEMA_VERSION = "odcr_step5_sampling_contract/1"
STEP5_POOL_SAMPLER_SCHEMA_VERSION = "odcr_step5_pool_sampler/1"
STEP5_SAMPLE_PLAN_SCHEMA_VERSION = "odcr_step5_sample_plan/1"
STEP5_SAMPLE_PLAN_PREFLIGHT_SCHEMA_VERSION = "odcr_step5_route_compatible_sample_plan_preflight/1"
STEP5_SAMPLE_PLAN_MANIFEST = "sample_plan_manifest.json"
STEP5_POOLS_DIRNAME = "step5_pools"
STEP5_POOL_MANIFEST = "step5_pool_manifest.json"
STEP5_SAMPLING_CONTRACT = "step5_sampling_contract.json"
STEP5_POOL_DISTRIBUTION_REPORT = "step5_pool_distribution_report.json"
STEP5_POOL_EXPORTS_STATUS = "step5_pool_exports_status.json"

POOL_NAMES: tuple[str, ...] = (
    "step5A_target_gold_anchor_high",
    "step5A_target_gold_anchor_medium",
    "step5A_aux_gold_anchor_high",
    "step5A_aux_gold_anchor_medium",
    "step5A_cf_scorer_high",
    "step5A_cf_scorer_medium",
    "step5A_cf_scorer_low_weighted",
    "step5A_cf_scorer_reject",
    "step5B_target_gold_anchor_high",
    "step5B_target_gold_anchor_medium",
    "step5B_aux_gold_anchor_high",
    "step5B_aux_gold_anchor_medium",
    "step5B_cf_explainer_high",
    "step5B_cf_explainer_medium",
    "step5B_cf_explainer_low_weighted",
    "step5B_cf_explainer_reject",
)

POOL_PARQUET_NAMES: dict[str, str] = {name: f"{name}.parquet" for name in POOL_NAMES}


class Step5PoolSamplerError(RuntimeError):
    """Raised when Step5 pools or the sampling contract are missing/stale."""


@dataclass(frozen=True)
class Step5PoolSource:
    pool_dir: Path
    manifest_path: Path
    sampling_contract_path: Path
    manifest: Mapping[str, Any]
    sampling_contract: Mapping[str, Any]
    source_full_export: Path | None

    def to_summary(self) -> dict[str, Any]:
        return {
            "schema_version": STEP5_POOL_SAMPLER_SCHEMA_VERSION,
            "pool_dir": str(self.pool_dir),
            "manifest_path": str(self.manifest_path),
            "sampling_contract_path": str(self.sampling_contract_path),
            "manifest_sha256": file_fingerprint(self.manifest_path).get("sha256"),
            "sampling_contract_sha256": file_fingerprint(self.sampling_contract_path).get("sha256"),
            "source_full_export": str(self.source_full_export) if self.source_full_export else None,
            "pool_schema_version": self.manifest.get("schema_version"),
            "sampling_contract_schema_version": self.sampling_contract.get("schema_version"),
        }


@dataclass(frozen=True)
class Step5PoolSampleResult:
    train_df: pd.DataFrame
    audit_raw_df: pd.DataFrame
    source: Step5PoolSource
    raw_row_count: int
    filtered_row_count: int
    stats: Mapping[str, Any]

    def to_summary(self) -> dict[str, Any]:
        return {
            "schema_version": STEP5_POOL_SAMPLER_SCHEMA_VERSION,
            "raw_row_count": int(self.raw_row_count),
            "filtered_row_count": int(self.filtered_row_count),
            "stats": dict(self.stats),
            "source": self.source.to_summary(),
        }


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise Step5PoolSamplerError(f"{label} missing: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Step5PoolSamplerError(f"{label} invalid JSON: {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise Step5PoolSamplerError(f"{label} root must be an object: {path}")
    return obj


def _repo_path(repo_root: Path | None, raw: Any) -> Path | None:
    value = str(raw or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if repo_root is not None and not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def resolve_step5_pool_source(
    *,
    step4_run_dir: str | Path,
    repo_root: str | Path | None = None,
) -> Step5PoolSource:
    root = Path(repo_root).expanduser().resolve() if repo_root is not None else None
    run_dir = Path(step4_run_dir).expanduser()
    if root is not None and not run_dir.is_absolute():
        run_dir = root / run_dir
    run_dir = run_dir.resolve()
    pool_dir = run_dir / STEP5_POOLS_DIRNAME
    manifest_path = pool_dir / STEP5_POOL_MANIFEST
    sampling_contract_path = pool_dir / STEP5_SAMPLING_CONTRACT
    manifest = _load_json(manifest_path, label="Step5 pool manifest")
    contract = _load_json(sampling_contract_path, label="Step5 sampling contract")
    if root is None and str(manifest.get("repo_root") or "").strip():
        root = Path(str(manifest["repo_root"])).expanduser().resolve()
    if manifest.get("schema_version") != STEP5_POOL_MANIFEST_SCHEMA_VERSION:
        raise Step5PoolSamplerError("Step5 pool manifest schema_version mismatch")
    if contract.get("schema_version") != STEP5_SAMPLING_CONTRACT_SCHEMA_VERSION:
        raise Step5PoolSamplerError("Step5 sampling contract schema_version mismatch")
    if manifest.get("full_audit_default_train_forbidden") is not True:
        raise Step5PoolSamplerError("Step5 pool manifest must forbid full audit as default train input")
    if manifest.get("legacy_gold_heavy_exports_allowed_by_default") is not False:
        raise Step5PoolSamplerError("Step5 pool manifest must disable legacy gold-heavy exports by default")
    pools = manifest.get("pools")
    if not isinstance(pools, Mapping):
        raise Step5PoolSamplerError("Step5 pool manifest missing pools object")
    missing = [name for name in POOL_NAMES if name not in pools]
    if missing:
        raise Step5PoolSamplerError("Step5 pool manifest missing pools: " + ", ".join(missing))
    for name in POOL_NAMES:
        item = pools.get(name)
        if not isinstance(item, Mapping):
            raise Step5PoolSamplerError(f"Step5 pool manifest pool is not an object: {name}")
        path = _repo_path(root, item.get("path"))
        if path is None or not path.is_file():
            raise Step5PoolSamplerError(f"Step5 pool parquet missing for {name}: {path}")
    source_full_export = _repo_path(root, manifest.get("source_full_export"))
    return Step5PoolSource(
        pool_dir=pool_dir,
        manifest_path=manifest_path,
        sampling_contract_path=sampling_contract_path,
        manifest=manifest,
        sampling_contract=contract,
        source_full_export=source_full_export,
    )


def _seed32(*parts: Any) -> int:
    return int(stable_hash(list(parts), length=8), 16) & 0xFFFFFFFF


def _read_pool(
    source: Step5PoolSource,
    name: str,
    columns: Sequence[str] | None,
    timings: dict[str, float] | None = None,
) -> pd.DataFrame:
    item = source.manifest["pools"][name]
    path = _pool_parquet_path(source, name)
    use_columns = None
    if columns is not None:
        available = set(item.get("columns") or [])
        use_columns = [col for col in columns if col in available]
    t0 = time.perf_counter()
    df = pd.read_parquet(path, columns=use_columns)
    if timings is not None:
        timings["parquet_read_time_s"] = float(timings.get("parquet_read_time_s", 0.0) + (time.perf_counter() - t0))
    df["step5_pool_name"] = name
    df["step5_pool_path"] = str(path)
    df["step5_pool_row_group"] = 0
    df["step5_pool_tier"] = str(item.get("tier") or "")
    return df


def _pool_parquet_path(source: Step5PoolSource, name: str) -> Path:
    item = source.manifest["pools"][name]
    path = Path(item["path"])
    if not path.is_absolute():
        try:
            repo_root = Path(source.manifest.get("repo_root") or "").expanduser().resolve()
        except Exception:
            repo_root = source.pool_dir.parents[4]
        if not str(repo_root):
            repo_root = source.pool_dir.parents[4]
        path = repo_root / path
    return path.resolve()


def _sample_df(
    df: pd.DataFrame,
    *,
    n: int,
    seed: int,
    stratify: bool,
) -> tuple[pd.DataFrame, float]:
    n = int(n)
    if n <= 0:
        return df.head(0).copy(), 0.0
    if df.empty:
        return df.copy(), 1.0
    rng = np.random.default_rng(int(seed))
    replace = n > len(df)
    if not stratify:
        idx = rng.choice(len(df), size=n, replace=replace)
        out = df.iloc[idx].copy().reset_index(drop=True)
        return out, max(0.0, float(n - len(df)) / float(max(n, 1))) if replace else 0.0

    strata_cols = [
        col
        for col in ("rating_bucket", "length_bucket", "explanation_length_bucket", "coverage_bucket", "user_coverage_bucket", "item_coverage_bucket")
        if col in df.columns
    ]
    if not strata_cols:
        idx = rng.choice(len(df), size=n, replace=replace)
        out = df.iloc[idx].copy().reset_index(drop=True)
        return out, max(0.0, float(n - len(df)) / float(max(n, 1))) if replace else 0.0
    work = df.copy()
    work["_stratum"] = work[strata_cols].astype(str).agg("|".join, axis=1)
    groups = list(work.groupby("_stratum", sort=True, dropna=False).indices.items())
    if not groups:
        return df.head(0).copy(), 1.0
    sizes = np.array([len(v) for _, v in groups], dtype=float)
    raw = sizes / max(float(sizes.sum()), 1.0) * float(n)
    alloc = np.floor(raw).astype(int)
    remainder = n - int(alloc.sum())
    if remainder > 0:
        order = np.argsort(-(raw - alloc))
        for pos in order[:remainder]:
            alloc[pos] += 1
    parts: list[pd.DataFrame] = []
    replacement_extra = 0
    for (_, indices), take in zip(groups, alloc):
        if int(take) <= 0:
            continue
        local_replace = int(take) > len(indices)
        if local_replace:
            replacement_extra += int(take) - len(indices)
        chosen = rng.choice(indices, size=int(take), replace=local_replace)
        parts.append(work.loc[chosen].drop(columns=["_stratum"]).copy())
    out = pd.concat(parts, ignore_index=True) if parts else df.head(0).copy()
    if len(out) > n:
        out = out.iloc[:n].copy()
    return out.reset_index(drop=True), float(replacement_extra) / float(max(n, 1))


def _ratio_counts(total: int, ratios: Mapping[str, Any]) -> dict[str, int]:
    total = int(total)
    target = int(round(total * float(ratios.get("target_gold_ratio", 0.0))))
    aux = int(round(total * float(ratios.get("aux_gold_ratio", 0.0))))
    cf = total - target - aux
    if cf < 0:
        cf = 0
        overflow = target + aux - total
        aux = max(0, aux - overflow)
    return {"target_gold": target, "aux_gold": aux, "cf": cf}


def _task_decoupled_policy(sampler_config: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = sampler_config.get("task_decoupled_policy")
    return raw if isinstance(raw, Mapping) else {}


def _enforce_head_policy(
    sampler_config: Mapping[str, Any],
    *,
    head: str,
    counts: Mapping[str, Any],
    context: str,
    require_positive_target: bool = True,
) -> None:
    if head != "step5A":
        return
    policy = _task_decoupled_policy(sampler_config)
    if policy:
        assert_step5a_policy_clean(policy)
    enforce_step5a_target_gold_counts(
        counts,
        context=context,
        require_positive_target=require_positive_target,
    )


def _mix_counts(total: int, mix: Mapping[str, Any], keys: Sequence[str]) -> dict[str, int]:
    total = int(total)
    if total <= 0:
        return {str(key): 0 for key in keys}
    weights = {str(key): max(0.0, float(mix.get(key, 0.0))) for key in keys}
    weight_sum = float(sum(weights.values()))
    if weight_sum <= 0.0:
        raise Step5PoolSamplerError("Step5 sampler tier mix must contain at least one positive tier")
    raw = {key: total * (value / weight_sum) for key, value in weights.items()}
    out = {key: int(np.floor(value)) for key, value in raw.items()}
    remainder = int(total - sum(out.values()))
    if remainder > 0:
        positive_keys = [key for key in keys if weights[str(key)] > 0.0]
        order = sorted(
            (str(key) for key in positive_keys),
            key=lambda key: (-(raw[key] - out[key]), list(keys).index(key)),
        )
        for key in order[:remainder]:
            out[key] += 1
    return {str(key): int(out.get(str(key), 0)) for key in keys}


def _budget_for(
    source: Step5PoolSource,
    *,
    sampler_config: Mapping[str, Any],
    batch_candidates_config: Mapping[str, Any] | None,
    tuning_config: Mapping[str, Any] | None,
    head: str,
    head_cfg: Mapping[str, Any],
    mode: str,
    bounded_max_rows: int | None,
) -> tuple[int, dict[str, Any]]:
    auto_budget = sampler_config.get("auto_budget")
    if isinstance(auto_budget, Mapping) and auto_budget.get("enabled") is True:
        if not isinstance(batch_candidates_config, Mapping):
            raise Step5PoolSamplerError(
                "Step5 auto budget requires resolved step5.batch_candidates; "
                "fixed effective_samples_per_epoch_candidates are retired."
            )
        try:
            return effective_samples_for_head(
                source.manifest,
                sampler_config=sampler_config,
                batch_candidates_config=batch_candidates_config,
                tuning_config=tuning_config or {},
                head=head,
                bounded_max_rows=bounded_max_rows,
                mode=mode,
            )
        except Exception as exc:
            raise Step5PoolSamplerError(str(exc)) from exc
    raise Step5PoolSamplerError(
        "Step5 fixed sample-budget fallback is retired; enable step5.sampler.auto_budget "
        "and use resolved effective_samples from configs/odcr.yaml."
    )


def _pool_names_for(head: str, component: str, tier: str) -> str:
    if component == "target_gold":
        return f"{head}_target_gold_anchor_{tier}"
    if component == "aux_gold":
        return f"{head}_aux_gold_anchor_{tier}"
    if component == "cf":
        suffix = "cf_scorer" if head == "step5A" else "cf_explainer"
        return f"{head}_{suffix}_{tier}"
    raise KeyError(component)


def _route_column_for_head(head: str) -> str:
    if head == "step5A":
        return "route_scorer"
    if head == "step5B":
        return "route_explainer"
    raise Step5PoolSamplerError(f"unsupported Step5 head for route-compatible sampling: {head}")


def _route_compatible_pool_df(df: pd.DataFrame, *, head: str, pool_name: str) -> pd.DataFrame:
    route_col = _route_column_for_head(head)
    if route_col not in df.columns:
        raise Step5PoolSamplerError(f"Step5 pool {pool_name} missing required route column {route_col}")
    route = pd.to_numeric(df[route_col], errors="coerce").fillna(0).astype(int)
    if head == "step5B":
        return df.copy().reset_index(drop=True) if int((route == 1).sum()) > 0 else df.head(0).copy()
    return df.loc[route == 1].copy().reset_index(drop=True)


def _pool_row_count(source: Step5PoolSource, name: str) -> int:
    pools = source.manifest.get("pools")
    if not isinstance(pools, Mapping):
        raise Step5PoolSamplerError("Step5 pool manifest missing pools object")
    item = pools.get(name)
    if not isinstance(item, Mapping):
        raise Step5PoolSamplerError(f"Step5 pool manifest missing pool: {name}")
    try:
        return int(item.get("row_count") or 0)
    except Exception as exc:
        raise Step5PoolSamplerError(f"Step5 pool row_count must be integer for {name}") from exc


def _route_compatible_count(
    source: Step5PoolSource,
    *,
    head: str,
    pool_name: str,
    timings: dict[str, float] | None = None,
) -> int:
    route_col = _route_column_for_head(head)
    item = source.manifest["pools"][pool_name]
    if route_col not in set(item.get("columns") or []):
        raise Step5PoolSamplerError(f"Step5 pool {pool_name} missing required route column {route_col}")
    try:
        import pyarrow.parquet as pq  # type: ignore

        t0 = time.perf_counter()
        parquet = pq.ParquetFile(_pool_parquet_path(source, pool_name))
        if timings is not None:
            timings["parquet_metadata_time_s"] = float(
                timings.get("parquet_metadata_time_s", 0.0) + (time.perf_counter() - t0)
            )
        route_idx = parquet.schema_arrow.get_field_index(route_col)
        if route_idx >= 0:
            exact = 0
            unknown = False
            for row_group_idx in range(parquet.metadata.num_row_groups):
                rg = parquet.metadata.row_group(row_group_idx)
                col = rg.column(route_idx)
                stats = col.statistics
                if stats is None or stats.min is None or stats.max is None:
                    unknown = True
                    break
                try:
                    min_v = int(stats.min)
                    max_v = int(stats.max)
                except Exception:
                    unknown = True
                    break
                if min_v >= 1 and max_v >= 1:
                    exact += int(rg.num_rows)
                elif max_v <= 0:
                    continue
                else:
                    unknown = True
                    break
            if not unknown:
                return int(exact)
            if head == "step5B":
                return 1
    except Exception:
        pass
    df = _read_pool(source, pool_name, [route_col], timings=timings)
    route = pd.to_numeric(df[route_col], errors="coerce").fillna(0).astype(int)
    return int((route == 1).sum())


def _validate_sampler_guardrails(sampler_config: Mapping[str, Any], source: Step5PoolSource) -> None:
    if sampler_config.get("enabled") is not True:
        raise Step5PoolSamplerError("step5.sampler.enabled must be true for Step5 sample-plan preflight")
    if sampler_config.get("contract_source") != "step4_pool_manifest":
        raise Step5PoolSamplerError("step5.sampler.contract_source must be step4_pool_manifest")
    if sampler_config.get("full_audit_default_allowed") is not False:
        raise Step5PoolSamplerError("step5.sampler.full_audit_default_allowed must be false")
    if sampler_config.get("legacy_gold_heavy_exports_allowed") is not False:
        raise Step5PoolSamplerError("step5.sampler.legacy_gold_heavy_exports_allowed must be false")
    if source.manifest.get("full_audit_default_train_forbidden") is not True:
        raise Step5PoolSamplerError("Step5 pool manifest must forbid full audit as default train input")
    if source.manifest.get("legacy_gold_heavy_exports_allowed_by_default") is not False:
        raise Step5PoolSamplerError("Step5 pool manifest must disable legacy gold-heavy exports by default")


def _head_list(task_head: str) -> tuple[str, ...]:
    head = str(task_head or "combined").strip()
    if head == "combined":
        return ("step5A", "step5B")
    if head in {"step5A", "step5B"}:
        return (head,)
    raise Step5PoolSamplerError(f"unsupported Step5 head for sample-plan preflight: {task_head}")


def validate_step5_formal_sample_plan_for_source(
    source: Step5PoolSource,
    *,
    sampler_config: Mapping[str, Any],
    batch_candidates_config: Mapping[str, Any] | None = None,
    tuning_config: Mapping[str, Any] | None = None,
    task_head: str = "combined",
    mode: str = "formal_train",
    bounded_max_rows: int | None = None,
    fail_on_route_incompatible: bool = True,
    no_write: bool = True,
) -> dict[str, Any]:
    """Validate a no-write Step5 sample plan against route-compatible pool rows."""

    _validate_sampler_guardrails(sampler_config, source)
    timings: dict[str, float] = {"parquet_read_time_s": 0.0}
    heads: dict[str, Any] = {}
    errors: list[str] = []
    max_replacement_rate = float(
        ((sampler_config.get("auto_budget") or {}) if isinstance(sampler_config.get("auto_budget"), Mapping) else {}).get(
            "max_replacement_rate",
            0.0,
        )
        or 0.0
    )
    for head in _head_list(task_head):
        head_cfg = sampler_config.get(head)
        if not isinstance(head_cfg, Mapping):
            raise Step5PoolSamplerError(f"step5.sampler.{head} missing")
        head_bounded = bounded_max_rows
        if str(task_head) == "combined" and bounded_max_rows is not None:
            head_bounded = max(1, int(bounded_max_rows) // 2)
        budget, budget_report = _budget_for(
            source,
            sampler_config=sampler_config,
            batch_candidates_config=batch_candidates_config,
            tuning_config=tuning_config,
            head=head,
            head_cfg=head_cfg,
            mode=mode,
            bounded_max_rows=head_bounded,
        )
        component_counts = _ratio_counts(budget, head_cfg)
        _enforce_head_policy(
            sampler_config,
            head=head,
            counts=component_counts,
            context=f"Step5 sample-plan preflight {head}",
            require_positive_target=int(budget) > 0,
        )
        component_reports: dict[str, Any] = {}
        replacement_extra = 0
        pool_exhaustion = False
        for component, count in component_counts.items():
            tier_keys = ("high", "medium") if component in {"target_gold", "aux_gold"} else ("high", "medium", "low_weighted")
            component_mix = _component_mix(head_cfg, component)
            requested_counts = _mix_counts(int(count), component_mix, tier_keys)
            available_raw: dict[str, int] = {}
            available_route: dict[str, int] = {}
            sampling_capacity_by_tier: dict[str, int] = {}
            active_tiers: list[str] = []
            inactive_tiers: list[str] = []
            component_shortage = 0
            route_col = _route_column_for_head(head) if component == "cf" else None
            for tier in tier_keys:
                pool_name = _pool_names_for(head, component, tier)
                requested = int(requested_counts.get(tier, 0))
                active = float(component_mix.get(tier) or 0.0) > 0.0
                (active_tiers if active else inactive_tiers).append(str(tier))
                available_raw[tier] = _pool_row_count(source, pool_name)
                if component == "cf":
                    route_positive = _route_compatible_count(source, head=head, pool_name=pool_name, timings=timings)
                    available_route[tier] = int(route_positive)
                    if requested > 0 and int(route_positive) <= 0:
                        message = (
                            f"Step5 sampler active tier has no route-compatible rows for {head}/{component}/{tier}. "
                            "Set the tier mix to zero in configs/odcr.yaml or rerun Step4 pool export with matching route semantics."
                        )
                        errors.append(message)
                    sampling_capacity = int(route_positive) if head == "step5A" else (available_raw[tier] if int(route_positive) > 0 else 0)
                    sampling_capacity_by_tier[tier] = int(sampling_capacity)
                    shortage = max(0, requested - int(sampling_capacity))
                else:
                    sampling_capacity_by_tier[tier] = int(available_raw[tier])
                    shortage = max(0, requested - int(available_raw[tier]))
                if requested > 0 and shortage > 0:
                    pool_exhaustion = True
                    component_shortage += int(shortage)
            replacement_extra += int(component_shortage)
            component_reports[component] = {
                "requested": int(count),
                "requested_tier_counts": requested_counts,
                "actual_tier_counts": dict(requested_counts),
                "active_tiers": active_tiers,
                "inactive_tiers": inactive_tiers,
                "available_raw_by_tier": available_raw,
                "available_route_compatible_by_tier": available_route if component == "cf" else None,
                "sampling_capacity_by_tier": sampling_capacity_by_tier,
                "route_filter": {
                    "enabled": component == "cf",
                    "route_column": route_col,
                    "route_requirement": (
                        "route_scorer == 1" if route_col == "route_scorer" else ("route_explainer == 1" if route_col else None)
                    ),
                },
                "shortage": int(component_shortage),
                "graph_safe_zero_tiers": [
                    str(tier)
                    for tier in tier_keys
                    if int(requested_counts.get(tier, 0)) == 0 and float(component_mix.get(tier) or 0.0) == 0.0
                ],
            }
        replacement_rate = float(replacement_extra) / float(max(int(budget), 1))
        if replacement_rate > max_replacement_rate:
            errors.append(
                f"Step5 sample-plan replacement_rate={replacement_rate:.6f} exceeds max_replacement_rate={max_replacement_rate:.6f} for {head}"
            )
        if pool_exhaustion:
            errors.append(f"Step5 sample-plan pool exhaustion detected for {head}; formal preflight requires zero replacement.")
        heads[head] = {
            "head": head,
            "status": "pass",
            "effective_samples_per_epoch": int(budget),
            "auto_budget": budget_report,
            "requested_counts": normalized_actual_counts(component_counts),
            "components": component_reports,
            "replacement_extra": int(replacement_extra),
            "replacement_rate": float(replacement_rate),
            "max_replacement_rate": float(max_replacement_rate),
            "pool_exhaustion": bool(pool_exhaustion),
            "actual_tier_counts_match_active_mix": True,
            "low_weighted_policy": (
                "disabled_for_mainline"
                if float((_component_mix(head_cfg, "cf")).get("low_weighted") or 0.0) == 0.0
                else "active"
            ),
            "task_decoupled_policy": dict(_task_decoupled_policy(sampler_config)),
        }
    if errors and fail_on_route_incompatible:
        first = errors[0]
        raise Step5PoolSamplerError(first)
    status = "fail" if errors else "pass"
    for head_report in heads.values():
        head_report["status"] = status if errors else "pass"
    return {
        "schema_version": STEP5_SAMPLE_PLAN_PREFLIGHT_SCHEMA_VERSION,
        "status": status,
        "pass": not errors,
        "task_head": str(task_head),
        "heads": heads,
        "errors": errors,
        "timings": timings,
        "no_write": bool(no_write),
        "formal_namespace_write": False,
        "step4_sampling_contract_role": "pool_lineage_only",
        "active_sampler_source": "configs/odcr.yaml:step5.sampler + configs/odcr.yaml:step5.tuning.selected_tuning_candidate",
        "full_audit_default_forbidden": True,
        "old_dedicated_default_forbidden": True,
    }


def validate_step5_formal_sample_plan(
    cfg: Any,
    *,
    head: str | None = None,
    fail_on_route_incompatible: bool = True,
    no_write: bool = True,
) -> dict[str, Any]:
    repo_root = Path(getattr(cfg, "repo_root", ".")).expanduser().resolve()
    runs_dir = Path(str(getattr(cfg, "checkpoint_dir", "") or "")).expanduser()
    step4_run = str(getattr(cfg, "step4_run", "") or getattr(cfg, "from_run", "") or "").strip()
    if not step4_run:
        raise Step5PoolSamplerError("Step5 sample-plan preflight requires resolved Step4 run id")
    step4_run_dir = repo_root / "runs" / "step4" / f"task{int(getattr(cfg, 'task_id'))}" / step4_run
    source = resolve_step5_pool_source(step4_run_dir=step4_run_dir, repo_root=repo_root)
    try:
        sampler_config = json.loads(str(getattr(cfg, "step5_sampler_config_json", "{}") or "{}"))
        batch_candidates_config = json.loads(str(getattr(cfg, "step5_batch_candidates_config_json", "{}") or "{}"))
        tuning_config = json.loads(str(getattr(cfg, "step5_tuning_config_json", "{}") or "{}"))
    except json.JSONDecodeError as exc:
        raise Step5PoolSamplerError(f"Step5 resolved sampler JSON is invalid: {exc}") from exc
    task_head = str(head or getattr(cfg, "step5_head", "combined") or "combined")
    report = validate_step5_formal_sample_plan_for_source(
        source,
        sampler_config=sampler_config,
        batch_candidates_config=batch_candidates_config,
        tuning_config=tuning_config,
        task_head=task_head,
        mode="formal_train",
        bounded_max_rows=None,
        fail_on_route_incompatible=fail_on_route_incompatible,
        no_write=no_write,
    )
    report["resolved_step4_run"] = step4_run
    report["step4_pool_source"] = source.to_summary()
    report["run_namespace_checked"] = str(runs_dir) if str(runs_dir) else None
    return report


def _apply_prompt_and_weights(
    df: pd.DataFrame,
    *,
    task_head: str,
    component: str,
    tier: str,
    seed: int,
    weight: float,
    effective_epoch: int,
    timings: dict[str, float] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    registry = default_prompt_registry()
    out["task_head"] = task_head
    out["sampler_component"] = component
    out["sampler_tier"] = tier
    out["effective_epoch"] = int(effective_epoch)
    out["sampler_weight"] = float(weight)
    out["posterior_sample_weight_hint"] = pd.to_numeric(out["sample_weight_hint"], errors="coerce").fillna(0.0)
    out["sample_weight_hint"] = out["posterior_sample_weight_hint"].astype(float) * float(weight)
    t0 = time.perf_counter()
    prompts = [
        registry.render(
            sample=row,
            task_head=task_head,
            sample_origin=str(row.get("sample_origin") or component),
            seed=int(seed),
            split="train",
        )
        for row in out.to_dict("records")
    ]
    if timings is not None:
        timings["prompt_build_time_s"] = float(timings.get("prompt_build_time_s", 0.0) + (time.perf_counter() - t0))
    for key in (
        "step5_prompt_template_id",
        "step5_prompt_instance_id",
        "step5_prompt_family",
        "step5_prompt_version",
        "step5_prompt_mode",
        "step5_prompt_seed",
        "step5_prompt_text",
    ):
        out[key] = [p[key] for p in prompts]
    out["step5_prompt_input_role"] = "content_evidence_prefix"
    if "content_evidence" in out.columns:
        out["content_evidence"] = (
            out["step5_prompt_text"].astype(str)
            + "\n"
            + out["content_evidence"].fillna("").astype(str)
        )
    return out


def _sample_component(
    source: Step5PoolSource,
    *,
    head: str,
    component: str,
    count: int,
    component_mix: Mapping[str, Any],
    weights: Mapping[str, float],
    seed: int,
    epoch: int,
    columns: Sequence[str] | None,
    timings: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    tier_keys = ("high", "medium") if component in {"target_gold", "aux_gold"} else ("high", "medium", "low_weighted")
    requested_counts = _mix_counts(int(count), component_mix, tier_keys)
    counts = dict(requested_counts)
    shortage_reallocated: dict[str, Any] = {}
    pool_frames: dict[str, pd.DataFrame] = {}
    available_raw: dict[str, int] = {}
    available: dict[str, int] = {}
    for tier in tier_keys:
        pool_name = _pool_names_for(head, component, tier)
        pool_df = _read_pool(source, pool_name, columns, timings=timings)
        available_raw[tier] = int(len(pool_df))
        route_df = _route_compatible_pool_df(pool_df, head=head, pool_name=pool_name) if component == "cf" else pool_df
        pool_frames[tier] = route_df
        available[tier] = int(len(route_df))

    for tier in tier_keys:
        if component == "cf" and int(requested_counts.get(tier, 0)) > 0 and int(available.get(tier, 0)) <= 0:
            raise Step5PoolSamplerError(
                f"Step5 sampler active tier has no route-compatible rows for {head}/{component}/{tier}. "
                "Set the tier mix to zero in configs/odcr.yaml or rerun Step4 pool export with matching route semantics."
            )

    if component == "cf":
        for tier in tier_keys:
            take = int(counts.get(tier, 0))
            missing = max(0, take - int(available.get(tier, 0)))
            if missing > 0:
                shortage_reallocated[tier] = {"missing": int(missing), "to": {}, "replacement_within_tier": True}
    else:
        overflow: list[tuple[str, int]] = []
        for tier in tier_keys:
            take = int(counts.get(tier, 0))
            missing = max(0, take - int(available.get(tier, 0)))
            if missing <= 0:
                continue
            counts[tier] = int(available.get(tier, 0))
            overflow.append((tier, missing))
            shortage_reallocated[tier] = {"missing": int(missing), "to": {}}
        receiver_order = ("medium", "high")
        for original_tier, missing in overflow:
            remaining = int(missing)
            for receiver in receiver_order:
                if receiver == original_tier:
                    continue
                capacity = max(0, int(available.get(receiver, 0)) - int(counts.get(receiver, 0)))
                if capacity <= 0:
                    continue
                moved = min(remaining, capacity)
                counts[receiver] = int(counts.get(receiver, 0)) + int(moved)
                to_map = shortage_reallocated[original_tier]["to"]
                to_map[receiver] = int(to_map.get(receiver, 0)) + int(moved)
                remaining -= int(moved)
                if remaining <= 0:
                    break
            if remaining > 0:
                shortage_reallocated[original_tier]["replacement_within_component"] = True
                counts[original_tier] = int(counts.get(original_tier, 0)) + int(remaining)
    frames: list[pd.DataFrame] = []
    replacement: dict[str, float] = {tier: 0.0 for tier in tier_keys}
    for tier, take in counts.items():
        pool_df = pool_frames[tier]
        sampled, repl = _sample_df(
            pool_df,
            n=int(take),
            seed=_seed32(seed, head, component, tier, epoch),
            stratify=component in {"target_gold", "aux_gold"},
        )
        weight = float(weights.get(f"{component}_{tier}", weights.get(component, weights.get(tier, 1.0))))
        sampled = _apply_prompt_and_weights(
            sampled,
            task_head=head,
            component=component,
            tier=tier,
            seed=int(seed),
            weight=weight,
            effective_epoch=int(epoch),
            timings=timings,
        )
        frames.append(sampled)
        replacement[tier] = float(repl)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out, {
        "requested": int(count),
        "actual": int(len(out)),
        "requested_tier_counts": requested_counts,
        "tier_counts": counts,
        "route_filter": {
            "head": str(head),
            "enabled": component == "cf",
            "route_column": _route_column_for_head(head) if component == "cf" else None,
            "available_raw_by_tier": available_raw,
            "available_route_compatible_by_tier": available,
        },
        "replacement_rate_by_tier": replacement,
        "shortage_reallocated_by_tier": shortage_reallocated,
    }


def _head_weights(head_cfg: Mapping[str, Any]) -> dict[str, float]:
    return {
        "aux_gold": float(head_cfg.get("aux_gold_weight", 1.0)),
        "high": float(head_cfg.get("cf_high_weight", 1.0)),
        "medium": float(head_cfg.get("cf_medium_weight", 1.0)),
        "low_weighted": float(head_cfg.get("cf_low_weight", 1.0)),
        "target_gold": 1.0,
    }


def _component_mix(head_cfg: Mapping[str, Any], component: str) -> Mapping[str, Any]:
    if component == "target_gold":
        mix = head_cfg.get("target_gold_tier_mix")
    elif component == "aux_gold":
        mix = head_cfg.get("aux_gold_tier_mix")
    else:
        mix = head_cfg.get("cf_tier_mix")
    if not isinstance(mix, Mapping):
        raise Step5PoolSamplerError(f"step5.sampler component mix missing for {component}")
    return mix


def _coverage_stats(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    out: dict[str, Any] = {
        "unique_sample_count": int(df["sample_id"].nunique(dropna=True)) if "sample_id" in df.columns else int(len(df)),
        "rating_bucket": {},
        "coverage_bucket": {},
        "origin": {},
        "route": {},
    }
    for col, key in (
        ("rating_bucket", "rating_bucket"),
        ("coverage_bucket", "coverage_bucket"),
        ("sample_origin", "origin"),
        ("sampler_component", "component"),
        ("sampler_tier", "tier"),
    ):
        if col in df.columns:
            out[key] = {str(k): int(v) for k, v in df[col].astype(str).value_counts(dropna=False).items()}
    if {"route_scorer", "route_explainer"}.issubset(df.columns):
        rs = pd.to_numeric(df["route_scorer"], errors="coerce").fillna(0).astype(int)
        re = pd.to_numeric(df["route_explainer"], errors="coerce").fillna(0).astype(int)
        out["route"] = {
            "scorer": int((rs == 1).sum()),
            "explainer": int((re == 1).sum()),
            "both": int(((rs == 1) & (re == 1)).sum()),
            "neither": int(((rs != 1) & (re != 1)).sum()),
        }
    return out


def _weighted_replacement_rate(component_reports: Mapping[str, Any]) -> float:
    requested = 0
    replaced = 0.0
    for report in component_reports.values():
        if not isinstance(report, Mapping):
            continue
        actual = int(report.get("actual") or 0)
        requested += actual
        repl = report.get("replacement_rate_by_tier")
        if isinstance(repl, Mapping):
            for tier, rate in repl.items():
                tier_counts = report.get("tier_counts") if isinstance(report.get("tier_counts"), Mapping) else {}
                n = int(tier_counts.get(tier) or 0)
                replaced += float(rate) * float(n)
    return float(replaced / max(requested, 1))


def sample_effective_epochs_from_pools(
    source: Step5PoolSource,
    *,
    sampler_config: Mapping[str, Any],
    batch_candidates_config: Mapping[str, Any] | None = None,
    tuning_config: Mapping[str, Any] | None = None,
    mode: str,
    task_head: str = "combined",
    bounded_max_rows: int | None = None,
    columns: Sequence[str] | None = None,
) -> Step5PoolSampleResult:
    if sampler_config.get("enabled") is not True:
        raise Step5PoolSamplerError("step5.sampler.enabled must be true for Step5 pool sampling")
    if sampler_config.get("contract_source") != "step4_pool_manifest":
        raise Step5PoolSamplerError("step5.sampler.contract_source must be step4_pool_manifest")
    if sampler_config.get("full_audit_default_allowed") is not False:
        raise Step5PoolSamplerError("step5.sampler.full_audit_default_allowed must be false")
    if sampler_config.get("legacy_gold_heavy_exports_allowed") is not False:
        raise Step5PoolSamplerError("step5.sampler.legacy_gold_heavy_exports_allowed must be false")
    seed = int(sampler_config.get("seed", 3407))
    epochs_cfg = sampler_config.get("epochs") if isinstance(sampler_config.get("epochs"), Mapping) else {}
    max_epochs = max(1, int(epochs_cfg.get("max_effective_epochs", 1)))
    if str(mode).lower() == "bounded":
        max_epochs = 1
    heads = ("step5A", "step5B") if task_head == "combined" else (str(task_head),)
    plan_t0 = time.perf_counter()
    timings: dict[str, float] = {
        "parquet_read_time_s": 0.0,
        "prompt_build_time_s": 0.0,
    }
    frames: list[pd.DataFrame] = []
    epoch_reports: list[dict[str, Any]] = []
    per_epoch_budget = 0
    auto_budget_reports: dict[str, Any] = {}
    for epoch in range(max_epochs):
        epoch_report: dict[str, Any] = {"effective_epoch": int(epoch), "heads": {}}
        for head in heads:
            head_cfg = sampler_config.get(head)
            if not isinstance(head_cfg, Mapping):
                raise Step5PoolSamplerError(f"step5.sampler.{head} missing")
            head_bounded = bounded_max_rows
            if task_head == "combined" and bounded_max_rows is not None:
                head_bounded = max(1, int(bounded_max_rows) // len(heads))
            budget, budget_report = _budget_for(
                source,
                sampler_config=sampler_config,
                batch_candidates_config=batch_candidates_config,
                tuning_config=tuning_config,
                head=head,
                head_cfg=head_cfg,
                mode=mode,
                bounded_max_rows=head_bounded,
            )
            if epoch == 0:
                auto_budget_reports[head] = budget_report
            per_epoch_budget += budget if epoch == 0 else 0
            counts = _ratio_counts(budget, head_cfg)
            _enforce_head_policy(
                sampler_config,
                head=head,
                counts=counts,
                context=f"Step5 effective-epoch sampler {head}",
                require_positive_target=int(budget) > 0,
            )
            weights = _head_weights(head_cfg)
            head_frames: list[pd.DataFrame] = []
            component_reports: dict[str, Any] = {}
            for component, count in counts.items():
                comp_df, comp_report = _sample_component(
                    source,
                    head=head,
                    component=component,
                    count=int(count),
                    component_mix=_component_mix(head_cfg, component),
                    weights=weights,
                    seed=seed,
                    epoch=epoch,
                    columns=columns,
                    timings=timings,
                )
                head_frames.append(comp_df)
                component_reports[component] = comp_report
            head_df = pd.concat(head_frames, ignore_index=True) if head_frames else pd.DataFrame()
            if not head_df.empty:
                head_df = head_df.sample(frac=1.0, random_state=_seed32(seed, head, epoch, "shuffle")).reset_index(drop=True)
            frames.append(head_df)
            actual_tier_counts = (
                {
                    str(k): int(v)
                    for k, v in head_df.groupby(["sampler_component", "sampler_tier"], dropna=False).size().items()
                }
                if not head_df.empty
                else {}
            )
            actual_counts = (
                {
                    str(k): int(v)
                    for k, v in head_df["sampler_component"].value_counts().items()
                }
                if not head_df.empty
                else {}
            )
            actual_counts = normalized_actual_counts(actual_counts)
            _enforce_head_policy(
                sampler_config,
                head=head,
                counts=actual_counts,
                context=f"Step5 effective-epoch actual sample plan {head}",
                require_positive_target=int(len(head_df)) > 0,
            )
            epoch_report["heads"][head] = {
                "effective_samples_per_epoch": int(budget),
                "auto_budget": budget_report,
                "requested_counts": normalized_actual_counts(counts),
                "actual_counts": actual_counts,
                "actual_ratios": {
                    str(k): float(v) / float(max(len(head_df), 1))
                    for k, v in head_df["sampler_component"].value_counts().items()
                }
                if not head_df.empty
                else {},
                "components": component_reports,
                "gold_tier_counts": {
                    str(k): int(v)
                    for k, v in head_df.loc[head_df["sampler_component"].isin(["target_gold", "aux_gold"]), "sampler_tier"].value_counts().items()
                }
                if not head_df.empty
                else {},
                "cf_tier_counts": {
                    str(k): int(v)
                    for k, v in head_df.loc[head_df["sampler_component"].eq("cf"), "sampler_tier"].value_counts().items()
                }
                if not head_df.empty
                else {},
                "component_tier_counts": actual_tier_counts,
                "replacement_rate": _weighted_replacement_rate(component_reports),
                "unique_sample_count": int(head_df["sample_id"].nunique(dropna=True)) if (not head_df.empty and "sample_id" in head_df.columns) else int(len(head_df)),
                "coverage_stats": _coverage_stats(head_df),
            }
        epoch_reports.append(epoch_report)
    train_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    total_plan_time_s = time.perf_counter() - plan_t0
    sampler_compute_time_s = max(
        0.0,
        total_plan_time_s
        - float(timings.get("parquet_read_time_s", 0.0))
        - float(timings.get("prompt_build_time_s", 0.0)),
    )
    stats = {
        "schema_version": STEP5_POOL_SAMPLER_SCHEMA_VERSION,
        "mode": str(mode),
        "task_head": str(task_head),
        "seed": int(seed),
        "sampler_plan_time_s": float(total_plan_time_s),
        "parquet_read_time_s": float(timings.get("parquet_read_time_s", 0.0)),
        "prompt_build_time_s": float(timings.get("prompt_build_time_s", 0.0)),
        "sampler_compute_time_s": float(sampler_compute_time_s),
        "rotate_across_epochs": bool(sampler_config.get("rotate_across_epochs", True)),
        "effective_epoch_enabled": bool(sampler_config.get("effective_epoch_enabled", True)),
        "effective_samples_per_epoch": int(per_epoch_budget),
        "auto_budget_enabled": bool(
            isinstance(sampler_config.get("auto_budget"), Mapping)
            and sampler_config.get("auto_budget", {}).get("enabled") is True
        ),
        "auto_budget_reports": auto_budget_reports,
        "max_effective_epochs": int(max_epochs),
        "planned_total_rows": int(len(train_df)),
        "epoch_reports": epoch_reports,
        "prompt_registry": prompt_registry_manifest(),
        "full_audit_default_train_forbidden": True,
        "legacy_gold_heavy_exports_rejected_by_default": True,
    }
    audit_raw = train_df.head(min(len(train_df), 16)).copy()
    return Step5PoolSampleResult(
        train_df=train_df,
        audit_raw_df=audit_raw,
        source=source,
        raw_row_count=int((source.manifest.get("source_row_counts") or {}).get("total_rows") or len(train_df)),
        filtered_row_count=int(len(train_df)),
        stats=stats,
    )


def write_sampler_snapshot(path: str | Path, sampler_config: Mapping[str, Any]) -> None:
    atomic_write_json(Path(path), dict(sampler_config))


def shard_step5_sample_plan(df: pd.DataFrame, *, rank: int, world_size: int) -> pd.DataFrame:
    """Return the deterministic rank shard for a prebuilt Step5 sample plan."""

    if int(world_size) <= 0:
        raise Step5PoolSamplerError("world_size must be positive when sharding a Step5 sample plan")
    if int(rank) < 0 or int(rank) >= int(world_size):
        raise Step5PoolSamplerError(f"rank {rank} outside world_size {world_size}")
    plan = df.reset_index(drop=True).copy()
    if "step5_plan_index" not in plan.columns:
        plan["step5_plan_index"] = range(len(plan))
    shard = plan.iloc[int(rank) :: int(world_size)].copy().reset_index(drop=True)
    shard["step5_rank"] = int(rank)
    shard["step5_world_size"] = int(world_size)
    shard["step5_rank_plan_index"] = range(len(shard))
    return shard


def _sample_plan_compact_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "step5_plan_index" not in out.columns:
        out["step5_plan_index"] = range(len(out))
    if "step5_sample_label" not in out.columns and "rating" in out.columns:
        out["step5_sample_label"] = pd.to_numeric(out["rating"], errors="coerce")
    if "step5_template_id" not in out.columns and "step5_prompt_template_id" in out.columns:
        out["step5_template_id"] = out["step5_prompt_template_id"].astype(str)
    if "step5_plan_path" not in out.columns and "step5_pool_path" in out.columns:
        out["step5_plan_path"] = out["step5_pool_path"].astype(str)
    if "step5_plan_row_group" not in out.columns:
        out["step5_plan_row_group"] = out.get("step5_pool_row_group", 0)
    return out


def write_step5_sample_plan(
    output_dir: str | Path,
    *,
    train_df: pd.DataFrame,
    stats: Mapping[str, Any],
    source_summary: Mapping[str, Any],
    task_head: str,
    world_size: int,
    source_table: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a bounded/test sample plan and per-rank shards outside formal runs."""

    plan_dir = Path(output_dir) / "sample_plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan = _sample_plan_compact_columns(train_df)
    shard_rows: list[dict[str, Any]] = []
    for rank in range(int(world_size)):
        shard = shard_step5_sample_plan(plan, rank=rank, world_size=int(world_size))
        shard_path = plan_dir / f"sample_plan_rank{rank}.parquet"
        shard.to_parquet(shard_path, index=False)
        shard_rows.append(
            {
                "rank": int(rank),
                "path": str(shard_path),
                "row_count": int(len(shard)),
                "sha256": file_fingerprint(shard_path).get("sha256"),
            }
        )
    manifest = {
        "schema_version": STEP5_SAMPLE_PLAN_SCHEMA_VERSION,
        "task_head": str(task_head),
        "world_size": int(world_size),
        "row_count": int(len(plan)),
        "shards": shard_rows,
        "plan_hash": stable_hash(
            {
                "schema_version": STEP5_SAMPLE_PLAN_SCHEMA_VERSION,
                "task_head": str(task_head),
                "sample_ids": list(plan["sample_id"].astype(str)) if "sample_id" in plan.columns else list(range(len(plan))),
                "components": list(plan["sampler_component"].astype(str)) if "sampler_component" in plan.columns else [],
                "tiers": list(plan["sampler_tier"].astype(str)) if "sampler_tier" in plan.columns else [],
                "templates": list(plan["step5_template_id"].astype(str)) if "step5_template_id" in plan.columns else [],
            }
        ),
        "stats_hash": stable_hash(dict(stats)),
        "source_table_hash": stable_hash(dict(source_table or {})),
        "source": dict(source_summary),
        "stats": dict(stats),
        "required_columns": [
            "sample_id",
            "step5_plan_path",
            "step5_plan_row_group",
            "step5_sample_label",
            "step5_template_id",
            "sampler_tier",
            "sample_origin",
        ],
        "formal_namespace_write": False,
    }
    atomic_write_json(plan_dir / STEP5_SAMPLE_PLAN_MANIFEST, manifest)
    return manifest


def read_step5_sample_plan_shard(output_dir: str | Path, *, rank: int) -> pd.DataFrame:
    path = Path(output_dir) / "sample_plan" / f"sample_plan_rank{int(rank)}.parquet"
    if not path.is_file():
        raise Step5PoolSamplerError(f"Step5 sample plan shard missing: {path}")
    return pd.read_parquet(path)


__all__ = [
    "POOL_NAMES",
    "POOL_PARQUET_NAMES",
    "STEP5_POOL_DISTRIBUTION_REPORT",
    "STEP5_POOL_EXPORTS_STATUS",
    "STEP5_POOL_MANIFEST",
    "STEP5_POOL_MANIFEST_SCHEMA_VERSION",
    "STEP5_POOLS_DIRNAME",
    "STEP5_POOL_SAMPLER_SCHEMA_VERSION",
    "STEP5_SAMPLE_PLAN_MANIFEST",
    "STEP5_SAMPLE_PLAN_PREFLIGHT_SCHEMA_VERSION",
    "STEP5_SAMPLE_PLAN_SCHEMA_VERSION",
    "STEP5_SAMPLING_CONTRACT",
    "STEP5_SAMPLING_CONTRACT_SCHEMA_VERSION",
    "Step5PoolSampleResult",
    "Step5PoolSamplerError",
    "Step5PoolSource",
    "resolve_step5_pool_source",
    "read_step5_sample_plan_shard",
    "sample_effective_epochs_from_pools",
    "shard_step5_sample_plan",
    "validate_step5_formal_sample_plan",
    "validate_step5_formal_sample_plan_for_source",
    "write_step5_sample_plan",
    "write_sampler_snapshot",
]
