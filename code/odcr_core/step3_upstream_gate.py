"""Step3 preprocess upstream hard gate.

Step3 consumes canonical merged CSVs plus preprocess_b/c profile and domain
artifacts.  Those files are only admissible when they are backed by the current
preprocess latest/run_summary/manifest/status/metrics/verify lineage chain.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from odcr_core.training_checkpoint import file_fingerprint, stable_hash


STEP3_UPSTREAM_GATE_SCHEMA_VERSION = "odcr_step3_upstream_gate/1"
STEP3_UPSTREAM_CONTRACT_SCHEMA_VERSION = "odcr_step3_preprocess_upstream_contract/1"
PROFILE_ARTIFACT_CONTRACT_VERSION = "preprocess_b_profile_matrix/1"
DOMAIN_ARTIFACT_CONTRACT_VERSION = "preprocess_c_domain_vector/1"
STEP3_REJECTS_PREPROCESS_C_RANK2_DOMAIN_VECTOR = True

_FORMAL_PATH_FORBIDDEN_PARTS = {
    "AI_analysis",
    "history",
    "_archive",
    "archive",
}
_FORMAL_PATH_FORBIDDEN_MARKERS = (
    "dry-run",
    "dry_run",
    "probe",
)
_PROFILE_SPECS = {
    "user_content_profiles": ("user_content", "user"),
    "user_style_profiles": ("user_style", "user"),
    "item_content_profiles": ("item_content", "item"),
    "item_style_profiles": ("item_style", "item"),
}
_DOMAIN_SPECS = {
    "domain_content": "content",
    "domain_style": "style",
}


class Step3UpstreamGateError(RuntimeError):
    """Raised when Step3 preprocess upstream evidence is incomplete or stale."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Step3UpstreamGateError(message)


def _repo_path(repo_root: Path, raw: str | Path | None, *, context: str) -> Path:
    _require(raw not in (None, ""), f"{context} is required.")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _rel(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _same_path(a: Path, b: Path) -> bool:
    return a.expanduser().resolve() == b.expanduser().resolve()


def _reject_forbidden_formal_path(path: Path, *, context: str) -> None:
    parts = set(path.resolve().parts)
    if parts.intersection(_FORMAL_PATH_FORBIDDEN_PARTS):
        raise Step3UpstreamGateError(
            f"{context} points to forbidden non-formal evidence path: {path}. "
            "AI_analysis/history/archive material is audit-only and cannot satisfy Step3 admission."
        )
    low = path.as_posix().lower()
    if any(marker in low for marker in _FORMAL_PATH_FORBIDDEN_MARKERS) or path.name == "completed.stamp":
        raise Step3UpstreamGateError(
            f"{context} points to forbidden non-formal evidence path: {path}. "
            "dry-run/probe/completed.stamp artifacts cannot satisfy Step3 admission."
        )


def _assert_file(path: Path, *, context: str) -> Path:
    _reject_forbidden_formal_path(path, context=context)
    if not path.is_file():
        raise Step3UpstreamGateError(f"{context} missing or not a file: {path}")
    return path


def _load_json(path: Path, *, context: str) -> dict[str, Any]:
    _assert_file(path, context=context)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Step3UpstreamGateError(f"{context} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise Step3UpstreamGateError(f"{context} JSON root must be an object: {path}")
    return data


def _json_fingerprint(path: Path) -> dict[str, Any]:
    return file_fingerprint(path, sample_only=True)


def _compare_fingerprint(actual: Mapping[str, Any], expected: Mapping[str, Any], *, context: str) -> None:
    for key in ("exists", "is_file", "size", "mtime_ns", "sample_sha256"):
        if key in expected:
            _require(
                actual.get(key) == expected.get(key),
                f"{context} fingerprint mismatch for {key}: actual={actual.get(key)!r}, expected={expected.get(key)!r}",
            )
    if "path" in expected:
        actual_path = Path(str(actual.get("path", ""))).expanduser().resolve()
        expected_path = Path(str(expected.get("path", ""))).expanduser().resolve()
        _require(actual_path == expected_path, f"{context} fingerprint path mismatch: {actual_path} != {expected_path}")


def _artifact_verify_entry(
    verify_report: Mapping[str, Any],
    *,
    path: Path,
    dataset: str,
    spec: str | None = None,
    domain: str | None = None,
) -> dict[str, Any]:
    artifacts = verify_report.get("artifacts")
    _require(isinstance(artifacts, list), "verify_report.artifacts must be a list.")
    for item in artifacts:
        if not isinstance(item, Mapping):
            continue
        item_path = item.get("path")
        if not item_path:
            continue
        if not _same_path(Path(str(item_path)), path):
            continue
        if str(item.get("dataset", "")) != dataset:
            continue
        if spec is not None and str(item.get("spec", "")) != spec:
            continue
        if domain is not None and str(item.get("domain", "")) != domain:
            continue
        return dict(item)
    raise Step3UpstreamGateError(f"verify_report is missing artifact entry for {dataset}:{spec or domain}: {path}")


def _npy_contract(
    path: Path,
    *,
    expected_rank: int,
    expected_dtype: str,
    expected_embed_dim: int,
    expected_shape: list[int] | None,
    context: str,
) -> dict[str, Any]:
    _assert_file(path, context=context)
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:  # pragma: no cover - numpy raises version-specific subclasses
        raise Step3UpstreamGateError(f"{context} cannot be loaded as a safe npy artifact: {path}: {exc}") from exc

    shape = [int(dim) for dim in arr.shape]
    dtype = str(arr.dtype)
    rank = int(arr.ndim)
    if expected_rank == 1 and rank == 2:
        raise Step3UpstreamGateError(
            f"{context} rank=2 shape={shape} is the retired preprocess_c [row_count, embed_dim] domain vector form; "
            "expected rank=1 [env.embed_dim]."
        )
    _require(rank == expected_rank, f"{context} rank mismatch: actual={rank}, expected={expected_rank}, shape={shape}")
    _require(dtype == expected_dtype, f"{context} dtype mismatch: actual={dtype}, expected={expected_dtype}")
    _require(shape[-1] == int(expected_embed_dim), f"{context} embed_dim mismatch: shape={shape}, env.embed_dim={expected_embed_dim}")
    if expected_shape is not None:
        _require(shape == [int(x) for x in expected_shape], f"{context} shape mismatch: actual={shape}, expected={expected_shape}")

    sample_count = min(int(arr.size), 4096)
    _require(sample_count > 0, f"{context} artifact is empty: {path}")
    sample = np.asarray(arr.reshape(-1)[:sample_count])
    finite_count = int(np.isfinite(sample).sum())
    nonzero_count = int(np.count_nonzero(sample))
    _require(finite_count == sample_count, f"{context} sample contains non-finite values.")
    _require(nonzero_count > 0, f"{context} sample is all zero.")
    return {
        "path": str(path),
        "shape": shape,
        "rank": rank,
        "dtype": dtype,
        "finite_sample_count": finite_count,
        "nonzero_sample_count": nonzero_count,
        "sample_count": sample_count,
        "fingerprint": file_fingerprint(path, sample_only=True),
    }


def _validate_preprocess_unit(repo_root: Path, runs_dir: Path, unit: str) -> tuple[dict[str, Any], dict[str, Any]]:
    latest_path = (runs_dir / "preprocess" / unit / "latest.json").resolve()
    latest = _load_json(latest_path, context=f"preprocess_{unit} latest.json")
    _require(str(latest.get("latest_status", "")).lower() == "ok", f"preprocess_{unit} latest_status must be ok.")

    run_id = str(latest.get("latest_run_id", "")).strip()
    _require(run_id, f"preprocess_{unit} latest_run_id is required.")
    run_dir = _repo_path(repo_root, latest.get("latest_run_dir"), context=f"preprocess_{unit} latest_run_dir")
    _reject_forbidden_formal_path(run_dir, context=f"preprocess_{unit} latest_run_dir")
    expected_run_dir = (runs_dir / "preprocess" / unit / run_id).resolve()
    _require(_same_path(run_dir, expected_run_dir), f"preprocess_{unit} latest points outside its unique formal run: {run_dir}")

    summary_path = _repo_path(repo_root, latest.get("latest_summary_path"), context=f"preprocess_{unit} latest_summary_path")
    _reject_forbidden_formal_path(summary_path, context=f"preprocess_{unit} latest_summary_path")
    expected_summary_path = (run_dir / "meta" / "run_summary.json").resolve()
    _require(_same_path(summary_path, expected_summary_path), f"preprocess_{unit} latest_summary_path must point to meta/run_summary.json.")
    summary = _load_json(summary_path, context=f"preprocess_{unit} run_summary.json")

    _require(str(summary.get("run_id", "")) == run_id, f"preprocess_{unit} run_summary.run_id mismatch.")
    _require(str(summary.get("stage", "")) == "preprocess", f"preprocess_{unit} run_summary.stage must be preprocess.")
    _require(str(summary.get("unit", "")) == unit, f"preprocess_{unit} run_summary.unit mismatch.")
    _require(str(summary.get("status", "")).lower() == "ok", f"preprocess_{unit} run_summary.status must be ok.")
    _require(str(summary.get("validation_status", "")).lower() == "ok", f"preprocess_{unit} validation_status must be ok.")
    _require(_same_path(_repo_path(repo_root, summary.get("run_dir"), context=f"preprocess_{unit} run_summary.run_dir"), run_dir), f"preprocess_{unit} run_dir mismatch.")

    meta_dir = _repo_path(repo_root, summary.get("meta_dir"), context=f"preprocess_{unit} meta_dir")
    _require(_same_path(meta_dir, run_dir / "meta"), f"preprocess_{unit} meta_dir mismatch.")
    resolved_config_path = _repo_path(repo_root, summary.get("resolved_config_path"), context=f"preprocess_{unit} resolved_config_path")
    source_table_path = _repo_path(repo_root, summary.get("source_table_path"), context=f"preprocess_{unit} source_table_path")
    manifest_path = _repo_path(repo_root, summary.get("manifest_path"), context=f"preprocess_{unit} manifest_path")
    status_path = _repo_path(repo_root, summary.get("lineage_path") or summary.get("stage_status_path"), context=f"preprocess_{unit} stage_status_path")

    resolved_config = _load_json(resolved_config_path, context=f"preprocess_{unit} resolved_config.json")
    source_table = _load_json(source_table_path, context=f"preprocess_{unit} source_table.json")
    manifest = _load_json(manifest_path, context=f"preprocess_{unit} stage_manifest.json")
    stage_status = _load_json(status_path, context=f"preprocess_{unit} stage_status.json")

    metrics_path: Path | None = None
    verify_path: Path | None = None
    metrics: dict[str, Any] | None = None
    verify: dict[str, Any] | None = None
    if unit in {"b", "c"}:
        metrics_path = _repo_path(repo_root, summary.get("metrics_path"), context=f"preprocess_{unit} metrics_path")
        verify_path = _repo_path(repo_root, summary.get("verify_report_path"), context=f"preprocess_{unit} verify_report_path")
        metrics = _load_json(metrics_path, context=f"preprocess_{unit} metrics.json")
        verify = _load_json(verify_path, context=f"preprocess_{unit} verify_report.json")

    fingerprint_hash = str(summary.get("fingerprint_hash", "")).strip()
    _require(bool(fingerprint_hash), f"preprocess_{unit} run_summary.fingerprint_hash is required.")
    for obj_name, obj in (("stage_manifest", manifest), ("stage_status", stage_status)):
        _require(str(obj.get("fingerprint_hash", "")) == fingerprint_hash, f"preprocess_{unit} {obj_name}.fingerprint_hash mismatch.")
    _require(str(stage_status.get("status", "")).lower() == "ok", f"preprocess_{unit} stage_status.status must be ok.")

    if metrics is not None:
        if metrics.get("stage") is not None:
            _require(str(metrics.get("stage")) == f"preprocess_{unit}", f"preprocess_{unit} metrics.stage mismatch.")
        if metrics.get("run_meta_dir") is not None:
            got = _repo_path(repo_root, metrics.get("run_meta_dir"), context=f"preprocess_{unit} metrics.run_meta_dir")
            _require(_same_path(got, meta_dir), f"preprocess_{unit} metrics.run_meta_dir mismatch.")
        if metrics.get("run_id") is not None:
            _require(str(metrics.get("run_id")) == run_id, f"preprocess_{unit} metrics.run_id mismatch.")
        if metrics.get("unit") is not None:
            _require(str(metrics.get("unit")) == unit, f"preprocess_{unit} metrics.unit mismatch.")
    if verify is not None:
        if verify.get("stage") is not None:
            _require(str(verify.get("stage")) == f"preprocess_{unit}", f"preprocess_{unit} verify_report.stage mismatch.")
        if verify.get("status") is not None:
            _require(str(verify.get("status")).lower() == "pass", f"preprocess_{unit} verify_report.status must be pass.")
        if verify.get("run_meta_dir") is not None:
            got = _repo_path(repo_root, verify.get("run_meta_dir"), context=f"preprocess_{unit} verify_report.run_meta_dir")
            _require(_same_path(got, meta_dir), f"preprocess_{unit} verify_report.run_meta_dir mismatch.")
        if verify.get("run_id") is not None:
            _require(str(verify.get("run_id")) == run_id, f"preprocess_{unit} verify_report.run_id mismatch.")
        if verify.get("unit") is not None:
            _require(str(verify.get("unit")) == unit, f"preprocess_{unit} verify_report.unit mismatch.")

    metadata = manifest.get("metadata")
    _require(isinstance(metadata, Mapping), f"preprocess_{unit} stage_manifest.metadata must be an object.")
    _require(str(metadata.get("run_id", "")) == run_id, f"preprocess_{unit} manifest.metadata.run_id mismatch.")
    _require(str(metadata.get("stage_unit", "")) == unit, f"preprocess_{unit} manifest.metadata.stage_unit mismatch.")
    _require(str(metadata.get("stage", "")) == f"preprocess_{unit}", f"preprocess_{unit} manifest.metadata.stage mismatch.")

    manifest_paths = {
        "latest_path": latest_path,
        "run_summary_path": summary_path,
        "stage_manifest_path": manifest_path,
        "stage_status_path": status_path,
        "source_table_path": source_table_path,
        "resolved_config_path": resolved_config_path,
    }
    for key, expected in manifest_paths.items():
        got = _repo_path(repo_root, metadata.get(key), context=f"preprocess_{unit} manifest.metadata.{key}")
        _require(_same_path(got, expected), f"preprocess_{unit} manifest.metadata.{key} mismatch: {got} != {expected}")

    evidence = {
        "unit": unit,
        "run_id": run_id,
        "fingerprint_hash": fingerprint_hash,
        "latest_status": latest.get("latest_status"),
        "validation_status": summary.get("validation_status"),
        "paths": {
            "latest": _rel(repo_root, latest_path),
            "run_summary": _rel(repo_root, summary_path),
            "stage_status": _rel(repo_root, status_path),
            "stage_manifest": _rel(repo_root, manifest_path),
            "source_table": _rel(repo_root, source_table_path),
            "resolved_config": _rel(repo_root, resolved_config_path),
            "metrics": _rel(repo_root, metrics_path) if metrics_path else None,
            "verify_report": _rel(repo_root, verify_path) if verify_path else None,
        },
        "run_summary_fingerprint": _json_fingerprint(summary_path),
        "stage_status_fingerprint": _json_fingerprint(status_path),
        "stage_manifest_fingerprint": _json_fingerprint(manifest_path),
        "source_table_fingerprint": _json_fingerprint(source_table_path),
        "metrics_fingerprint": _json_fingerprint(metrics_path) if metrics_path else None,
        "verify_report_fingerprint": _json_fingerprint(verify_path) if verify_path else None,
    }
    return evidence, {
        "latest": latest,
        "run_summary": summary,
        "stage_status": stage_status,
        "stage_manifest": manifest,
        "manifest_metadata": dict(metadata),
        "resolved_config": resolved_config,
        "source_table": source_table,
        "metrics": metrics,
        "verify_report": verify,
    }


def _validate_source_csv_fingerprints(
    *,
    repo_root: Path,
    data_dir: Path,
    domains: tuple[str, str],
    b_manifest: Mapping[str, Any],
    c_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for domain in domains:
        source_path = (data_dir / domain / "train.csv").resolve()
        _assert_file(source_path, context=f"source CSV for {domain}")
        actual = file_fingerprint(source_path, sample_only=True)
        for unit, manifest in (("b", b_manifest), ("c", c_manifest)):
            expected = (
                manifest.get("metadata", {})
                .get("stage_specific", {})
                .get("source_csv_fingerprints", {})
                .get(domain)
            )
            _require(isinstance(expected, Mapping), f"preprocess_{unit} missing source_csv_fingerprints for {domain}.")
            _compare_fingerprint(actual, expected, context=f"preprocess_{unit} source_csv_fingerprints[{domain}]")
        out[domain] = actual
    return out


def _validate_merged_task_artifacts(
    *,
    repo_root: Path,
    merged_dir: Path,
    task_id: int,
    auxiliary_domain: str,
    target_domain: str,
    a_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    task_outputs = (
        a_manifest.get("metadata", {})
        .get("stage_specific", {})
        .get("merged_task_outputs", {})
        .get(str(task_id))
    )
    _require(isinstance(task_outputs, Mapping), f"preprocess_a manifest missing merged_task_outputs for task {task_id}.")
    _require(
        list(task_outputs.get("source_target") or []) == [auxiliary_domain, target_domain],
        f"preprocess_a task {task_id} source_target mismatch; expected {[auxiliary_domain, target_domain]}.",
    )
    out: dict[str, Any] = {}
    headers = task_outputs.get("current_headers")
    _require(isinstance(headers, Mapping), f"preprocess_a task {task_id} current_headers is required.")
    for key, filename in (("aug_train_csv", "aug_train.csv"), ("aug_valid_csv", "aug_valid.csv")):
        expected_path = (merged_dir / str(task_id) / filename).resolve()
        path = _repo_path(repo_root, task_outputs.get(key), context=f"preprocess_a task {task_id} {key}")
        _require(_same_path(path, expected_path), f"preprocess_a task {task_id} {key} path mismatch: {path} != {expected_path}")
        _assert_file(path, context=f"preprocess_a task {task_id} {key}")
        header = headers.get(key)
        _require(isinstance(header, Mapping), f"preprocess_a task {task_id} current_headers.{key} is required.")
        _require(bool(header.get("header_match")), f"preprocess_a task {task_id} current_headers.{key}.header_match must be true.")
        st = path.stat()
        _require(int(header.get("file_size", -1)) == int(st.st_size), f"preprocess_a task {task_id} {key} file_size mismatch.")
        _require(int(header.get("mtime_ns", -1)) == int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))), f"preprocess_a task {task_id} {key} mtime_ns mismatch.")
        out[key] = {
            "path": _rel(repo_root, path),
            "header_hash": header.get("header_hash"),
            "contract_kind": header.get("contract_kind"),
            "fingerprint": file_fingerprint(path, sample_only=True),
        }
    return out


def _verify_report_contract(entry: Mapping[str, Any], *, shape: list[int], dtype: str, context: str) -> None:
    _require(str(entry.get("status", "")).lower() == "pass", f"{context} verify_report status must be pass.")
    _require(bool(entry.get("exists")), f"{context} verify_report.exists must be true.")
    _require(str(entry.get("dtype", "")) == dtype, f"{context} verify_report dtype mismatch.")
    expected_shape = entry.get("expected_shape")
    if expected_shape is not None:
        _require(shape == [int(x) for x in expected_shape], f"{context} verify_report expected_shape mismatch.")
    reported_shape = entry.get("shape")
    if reported_shape is not None:
        _require(shape == [int(x) for x in reported_shape], f"{context} verify_report shape mismatch.")
    _require(int(entry.get("finite_sample_count", 0)) > 0, f"{context} verify_report finite_sample_count must be positive.")
    _require(int(entry.get("nonzero_sample_count", 0)) > 0, f"{context} verify_report nonzero_sample_count must be positive.")


def _validate_profile_artifacts(
    *,
    repo_root: Path,
    domains: tuple[str, str],
    embed_dim: int,
    b_manifest: Mapping[str, Any],
    b_verify: Mapping[str, Any],
    c_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    output_paths = b_manifest.get("metadata", {}).get("stage_specific", {}).get("profile_output_paths", {})
    source_profile_fps = c_manifest.get("metadata", {}).get("stage_specific", {}).get("source_profile_fingerprints", {})
    _require(isinstance(output_paths, Mapping), "preprocess_b profile_output_paths must be an object.")
    _require(isinstance(source_profile_fps, Mapping), "preprocess_c source_profile_fingerprints must be an object.")
    out: dict[str, Any] = {}
    for domain in domains:
        domain_paths = output_paths.get(domain)
        domain_lineage = source_profile_fps.get(domain)
        _require(isinstance(domain_paths, Mapping), f"preprocess_b missing profile_output_paths for {domain}.")
        _require(isinstance(domain_lineage, Mapping), f"preprocess_c missing source_profile_fingerprints for {domain}.")
        out[domain] = {}
        for key, (verify_spec, entity_kind) in _PROFILE_SPECS.items():
            path = _repo_path(repo_root, domain_paths.get(key), context=f"preprocess_b {domain} {key}")
            verify_entry = _artifact_verify_entry(b_verify, path=path, dataset=domain, spec=verify_spec)
            expected_shape = verify_entry.get("expected_shape")
            contract = _npy_contract(
                path,
                expected_rank=2,
                expected_dtype="float32",
                expected_embed_dim=embed_dim,
                expected_shape=[int(x) for x in expected_shape] if isinstance(expected_shape, list) else None,
                context=f"preprocess_b {domain} {key}",
            )
            _verify_report_contract(verify_entry, shape=contract["shape"], dtype=contract["dtype"], context=f"preprocess_b {domain} {key}")
            _require(str(verify_entry.get("entity_kind", "")) == entity_kind, f"preprocess_b {domain} {key} entity_kind mismatch.")
            expected_fp = domain_lineage.get(key)
            _require(isinstance(expected_fp, Mapping), f"preprocess_c missing source_profile_fingerprints[{domain}][{key}].")
            _compare_fingerprint(contract["fingerprint"], expected_fp, context=f"preprocess_c source_profile_fingerprints[{domain}][{key}]")
            out[domain][key] = {
                "contract_version": PROFILE_ARTIFACT_CONTRACT_VERSION,
                **contract,
                "verify_report": {
                    "status": verify_entry.get("status"),
                    "finite_sample_count": verify_entry.get("finite_sample_count"),
                    "nonzero_sample_count": verify_entry.get("nonzero_sample_count"),
                    "expected_shape_label": verify_entry.get("expected_shape_label"),
                },
            }
    return out


def _validate_domain_artifacts(
    *,
    repo_root: Path,
    domains: tuple[str, str],
    embed_dim: int,
    c_manifest: Mapping[str, Any],
    c_verify: Mapping[str, Any],
) -> dict[str, Any]:
    output_paths = c_manifest.get("metadata", {}).get("stage_specific", {}).get("domain_output_paths", {})
    _require(isinstance(output_paths, Mapping), "preprocess_c domain_output_paths must be an object.")
    out: dict[str, Any] = {}
    for domain in domains:
        domain_paths = output_paths.get(domain)
        _require(isinstance(domain_paths, Mapping), f"preprocess_c missing domain_output_paths for {domain}.")
        out[domain] = {}
        for key, verify_domain in _DOMAIN_SPECS.items():
            path = _repo_path(repo_root, domain_paths.get(key), context=f"preprocess_c {domain} {key}")
            verify_entry = _artifact_verify_entry(c_verify, path=path, dataset=domain, domain=verify_domain)
            expected_shape = verify_entry.get("expected_shape")
            contract = _npy_contract(
                path,
                expected_rank=1,
                expected_dtype="float32",
                expected_embed_dim=embed_dim,
                expected_shape=[int(x) for x in expected_shape] if isinstance(expected_shape, list) else [int(embed_dim)],
                context=f"preprocess_c {domain} {key}",
            )
            _verify_report_contract(verify_entry, shape=contract["shape"], dtype=contract["dtype"], context=f"preprocess_c {domain} {key}")
            _require(
                str(verify_entry.get("domain_shape_contract_version", "")) == DOMAIN_ARTIFACT_CONTRACT_VERSION,
                f"preprocess_c {domain} {key} domain_shape_contract_version mismatch.",
            )
            out[domain][key] = {
                "contract_version": DOMAIN_ARTIFACT_CONTRACT_VERSION,
                **contract,
                "verify_report": {
                    "status": verify_entry.get("status"),
                    "finite_sample_count": verify_entry.get("finite_sample_count"),
                    "nonzero_sample_count": verify_entry.get("nonzero_sample_count"),
                    "expected_shape_label": verify_entry.get("expected_shape_label"),
                    "domain_shape_contract_version": verify_entry.get("domain_shape_contract_version"),
                },
            }
    return out


def _fingerprint_table(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for domain, domain_items in artifacts.items():
        table[domain] = {}
        if isinstance(domain_items, Mapping):
            for key, item in domain_items.items():
                if isinstance(item, Mapping) and isinstance(item.get("fingerprint"), Mapping):
                    table[domain][key] = item["fingerprint"]
    return table


def validate_step3_preprocess_upstream_gate(
    *,
    repo_root: str | Path,
    task_id: int,
    auxiliary_domain: str,
    target_domain: str,
    data_dir: str | Path,
    merged_dir: str | Path,
    runs_dir: str | Path,
    embed_dim: int,
) -> dict[str, Any]:
    """Validate Step3's preprocess upstream evidence before training loads data."""

    root = Path(repo_root).expanduser().resolve()
    data_root = _repo_path(root, data_dir, context="Step3 data_dir")
    merged_root = _repo_path(root, merged_dir, context="Step3 merged_dir")
    runs_root = _repo_path(root, runs_dir, context="Step3 runs_dir")
    _require(int(embed_dim) > 0, "env.embed_dim must be positive.")
    domains = (str(auxiliary_domain), str(target_domain))

    unit_evidence: dict[str, Any] = {}
    unit_docs: dict[str, dict[str, Any]] = {}
    for unit in ("a", "b", "c"):
        evidence, docs = _validate_preprocess_unit(root, runs_root, unit)
        unit_evidence[unit] = evidence
        unit_docs[unit] = docs

    merged_artifacts = _validate_merged_task_artifacts(
        repo_root=root,
        merged_dir=merged_root,
        task_id=int(task_id),
        auxiliary_domain=domains[0],
        target_domain=domains[1],
        a_manifest=unit_docs["a"]["stage_manifest"],
    )
    source_csv_artifacts = _validate_source_csv_fingerprints(
        repo_root=root,
        data_dir=data_root,
        domains=domains,
        b_manifest=unit_docs["b"]["stage_manifest"],
        c_manifest=unit_docs["c"]["stage_manifest"],
    )
    profile_artifacts = _validate_profile_artifacts(
        repo_root=root,
        domains=domains,
        embed_dim=int(embed_dim),
        b_manifest=unit_docs["b"]["stage_manifest"],
        b_verify=unit_docs["b"]["verify_report"] or {},
        c_manifest=unit_docs["c"]["stage_manifest"],
    )
    domain_artifacts = _validate_domain_artifacts(
        repo_root=root,
        domains=domains,
        embed_dim=int(embed_dim),
        c_manifest=unit_docs["c"]["stage_manifest"],
        c_verify=unit_docs["c"]["verify_report"] or {},
    )

    payload: dict[str, Any] = {
        "schema_version": STEP3_UPSTREAM_GATE_SCHEMA_VERSION,
        "contract_schema_version": STEP3_UPSTREAM_CONTRACT_SCHEMA_VERSION,
        "status": "ok",
        "task": {
            "task_id": int(task_id),
            "auxiliary_domain": domains[0],
            "target_domain": domains[1],
        },
        "env": {
            "embed_dim": int(embed_dim),
        },
        "preprocess": unit_evidence,
        "merged_artifacts": merged_artifacts,
        "source_csv_artifacts": source_csv_artifacts,
        "profile_artifacts": profile_artifacts,
        "domain_artifacts": domain_artifacts,
        "profile_artifact_fingerprints": _fingerprint_table(profile_artifacts),
        "domain_artifact_fingerprints": _fingerprint_table(domain_artifacts),
    }
    payload["fingerprint_hash"] = stable_hash(payload)
    return payload


validate_step3_upstream_preflight = validate_step3_preprocess_upstream_gate
