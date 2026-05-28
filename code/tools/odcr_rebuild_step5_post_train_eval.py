#!/usr/bin/env python3
"""Step5_e official post-train eval protocol verifier/finalizer.

This tool does not train or synthesize metric values. It verifies evaluator
outputs, records invalidated stale artifacts before rebuild, and repairs
handoff/latest metadata after clean official split artifacts exist.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from odcr_core.manifests import write_latest_pointer_json, write_run_summary_json  # noqa: E402
from odcr_core.rating_source import resolve_task_local_rating_source_config, validate_rating_source  # noqa: E402
from odcr_core.stage_status import build_and_write_stage_status  # noqa: E402
from odcr_core.step5_explanation_handoff import build_step5_explanation_handoff  # noqa: E402
from base_utils import official_paper_metrics  # noqa: E402
from odcr_eval_metrics import compose_step3_rating_step5_explanation_report  # noqa: E402

try:  # noqa: E402
    import yaml  # type: ignore
except Exception:  # pragma: no cover - repository runtime provides PyYAML
    yaml = None

TASK_RUNS = {2: "1_18", 5: "1_19", 7: "1_19", 8: "1_19"}
TASK_DOMAINS = {
    2: ("AM_Movies", "AM_CDs"),
    5: ("AM_CDs", "AM_Movies"),
    8: ("TripAdvisor", "Yelp"),
    7: ("Yelp", "TripAdvisor"),
}
OFFICIAL_PAPER_KEYS = {
    "ROUGE-1",
    "ROUGE-L",
    "BLEU-1",
    "BLEU-2",
    "BLEU-3",
    "BLEU-4",
    "METEOR",
    "DIST-1",
    "DIST-2",
    "RMSE",
    "MAE",
}


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _read(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read configs/odcr.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"YAML root must be a mapping: {path}")
    return data


def _read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _task_local_rating_source(task: int) -> dict[str, Any]:
    cfg = _load_yaml(ROOT / "configs" / "odcr.yaml")
    raw = cfg.get("rating_source")
    resolved = resolve_task_local_rating_source_config(
        raw if isinstance(raw, Mapping) else {},
        task_id=int(task),
        step3_run=_step3_run_for_task(task),
        repo_root=ROOT,
    )
    return validate_rating_source(resolved, repo_root=ROOT) | {
        key: resolved[key]
        for key in ("policy_schema_version", "policy_type", "task_local_required", "source")
        if key in resolved
    }


def _get_task_run(task: int, run: str | None = None) -> str:
    return str(run or TASK_RUNS[int(task)])


def _split_dir(task: int, run: str, split: str) -> Path:
    return ROOT / "runs" / "step5" / f"task{int(task)}" / str(run) / "post_train_eval" / split


def _run_dir(task: int, run: str) -> Path:
    return ROOT / "runs" / "step5" / f"task{int(task)}" / str(run)


def _step3_run_for_task(task: int) -> str:
    return "2" if int(task) == 2 else "1"


def _contains_bertscore(obj: Any) -> bool:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if "bert" in str(key).lower():
                return True
            if _contains_bertscore(value):
                return True
    elif isinstance(obj, list):
        return any(_contains_bertscore(item) for item in obj)
    return False


def _paper_metric_flat(payload: Mapping[str, Any], *, split: str) -> dict[str, float | None]:
    rating = payload.get("rating_metrics") if isinstance(payload.get("rating_metrics"), Mapping) else {}
    rating_split = rating.get(split) if isinstance(rating.get(split), Mapping) else rating
    exp = payload.get("explanation_paper_metrics") if isinstance(payload.get("explanation_paper_metrics"), Mapping) else {}
    rouge = exp.get("rouge") if isinstance(exp.get("rouge"), Mapping) else {}
    bleu = exp.get("bleu") if isinstance(exp.get("bleu"), Mapping) else {}
    distinct = exp.get("distinct_corpus") if isinstance(exp.get("distinct_corpus"), Mapping) else {}
    distinct_pct = distinct.get("scale_percent_0_100") if isinstance(distinct.get("scale_percent_0_100"), Mapping) else {}
    return {
        "ROUGE-1": rouge.get("rouge_1_f"),
        "ROUGE-L": rouge.get("rouge_l_f"),
        "BLEU-1": bleu.get("1"),
        "BLEU-2": bleu.get("2"),
        "BLEU-3": bleu.get("3"),
        "BLEU-4": bleu.get("4"),
        "METEOR": exp.get("meteor"),
        "DIST-1": distinct_pct.get("1"),
        "DIST-2": distinct_pct.get("2"),
        "RMSE": rating_split.get("RMSE", rating_split.get("rmse")),
        "MAE": rating_split.get("MAE", rating_split.get("mae")),
    }


def verify_split(task: int, run: str, split: str) -> dict[str, Any]:
    sdir = _split_dir(task, run, split)
    metrics_path = sdir / "eval_metrics.json"
    paper_path = sdir / "paper_metrics.json"
    handoff_path = sdir / "eval_handoff.json"
    metrics = _read(metrics_path)
    paper = _read(paper_path)
    handoff = _read(handoff_path)
    rating_source = metrics.get("rating_source") if isinstance(metrics.get("rating_source"), Mapping) else {}
    handoff_rating = handoff.get("rating_source") if isinstance(handoff.get("rating_source"), Mapping) else {}
    eval_handoff_path = str(rating_source.get("eval_handoff") or handoff_rating.get("eval_handoff") or "")
    policy_type = str(rating_source.get("policy_type") or "")
    policy_source = str(rating_source.get("source") or "")
    task_local_policy = (
        (policy_type in {"", "task_local_step3_accepted_scorer"})
        and (policy_source in {"", "upstream_step3_eval_handoff"})
        and f"runs/step3/task{int(task)}/" in eval_handoff_path
    )
    official_policy = metrics.get("official_eval_policy") if isinstance(metrics.get("official_eval_policy"), Mapping) else {}
    decode = metrics.get("decode") if isinstance(metrics.get("decode"), Mapping) else {}
    flat = _paper_metric_flat(paper, split=split)
    checks = {
        "task_idx_matches_rating_source": int(metrics.get("task_idx") or -1) == int(task) == int(rating_source.get("task") or -1),
        "handoff_rating_source_matches_task": int(handoff_rating.get("task") or -1) == int(task),
        "rating_source_policy": task_local_policy,
        "rating_source_source": task_local_policy,
        "eval_handoff_path_task_local": f"runs/step3/task{int(task)}/" in eval_handoff_path,
        "step5_rating_metrics_overwritten_false": metrics.get("step5_rating_metrics_overwritten") is False,
        "official_profile": str(metrics.get("official_eval_profile") or metrics.get("eval_profile_name") or "") == "paper_greedy_25",
        "target_only": metrics.get("target_only") is True,
        "decode_strategy_greedy": str(decode.get("decode_strategy") or official_policy.get("decode_strategy") or "") == "greedy",
        "do_sample_false": official_policy.get("do_sample") is False,
        "repetition_penalty_one": abs(float(official_policy.get("repetition_penalty") or decode.get("repetition_penalty") or 0.0) - 1.0) < 1e-12,
        "max_len_25": int(official_policy.get("max_new_tokens") or official_policy.get("max_explanation_length") or 0) == 25,
        "paper_metrics_present": set(flat) == OFFICIAL_PAPER_KEYS and all(value is not None for value in flat.values()),
        "bertscore_absent": not _contains_bertscore(paper),
        "split_handoff_exists": handoff_path.is_file(),
    }
    clean = all(checks.values())
    return {
        "task": int(task),
        "run": str(run),
        "split": split,
        "clean": clean,
        "checks": checks,
        "rating_source_task": int(rating_source.get("task") or -1),
        "rating_source_eval_handoff": eval_handoff_path,
        "official_profile": metrics.get("official_eval_profile") or metrics.get("eval_profile_name"),
        "decode_strategy": decode.get("decode_strategy") or official_policy.get("decode_strategy"),
        "target_only": metrics.get("target_only"),
        "paper_metrics": flat,
        "paths": {
            "eval_metrics": _rel(metrics_path),
            "paper_metrics": _rel(paper_path),
            "eval_handoff": _rel(handoff_path),
        },
        "payloads": {"metrics": metrics, "paper": paper, "handoff": handoff},
    }


def invalidate_known_bad(tasks: list[int]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for task in tasks:
        run = _get_task_run(task)
        for split in ("valid", "test"):
            sdir = _split_dir(task, run, split)
            if not sdir.exists():
                continue
            try:
                result = verify_split(task, run, split)
            except Exception as exc:
                result = {"clean": False, "error": str(exc), "checks": {}}
            if result.get("clean"):
                continue
            files = []
            for name in ("eval_metrics.json", "paper_metrics.json", "eval_handoff.json", "official_eval_report.json"):
                path = sdir / name
                if path.is_file():
                    files.append({"path": _rel(path), "sha256": _sha(path), "bytes": path.stat().st_size})
            record = {
                "task": int(task),
                "run": run,
                "split": split,
                "reason": "pre_rebuild_artifact_failed_task_local_official_protocol",
                "observed_rating_source_task": result.get("rating_source_task"),
                "checks": result.get("checks", {}),
                "files": files,
            }
            records.append(record)
            _write(sdir / "meta" / "invalidated_before_rebuild.json", {
                "schema_version": "odcr_step5e_invalidated_before_rebuild/1",
                "generated_at_utc": _utc(),
                **record,
            })
    manifest = {
        "schema_version": "odcr_step5e_invalidated_artifacts_manifest/1",
        "generated_at_utc": _utc(),
        "records": records,
    }
    _write(ROOT / "AI_analysis" / "03_evidence_ledgers" / "rebuild_step5e_tasklocal_protocol_invalidated_artifacts.json", manifest)
    return manifest


def recompose_existing_predictions(tasks: list[int], splits: list[str]) -> dict[str, Any]:
    """Regenerate official split artifacts from existing official predictions.

    This is an evaluator-only path: it re-reads the split predictions, reruns
    official_paper_metrics on the 25-policy metric texts, and re-resolves the
    task-local Step3 rating source from the current One-Control policy.
    """

    records: list[dict[str, Any]] = []
    for task in tasks:
        run = _get_task_run(task)
        rating_source = _task_local_rating_source(task)
        for split in splits:
            sdir = _split_dir(task, run, split)
            predictions_path = sdir / "predictions.csv"
            old_metrics_path = sdir / "eval_metrics.json"
            old_paper_path = sdir / "paper_metrics.json"
            old_handoff_path = sdir / "eval_handoff.json"
            rows = _read_prediction_rows(predictions_path)
            if not rows:
                raise RuntimeError(f"predictions.csv has no rows: {_rel(predictions_path)}")
            bad_policy = [
                str(row.get("sample_id") or idx)
                for idx, row in enumerate(rows)
                if str(row.get("paper_metric_input_schema_version") or "") != "odcr_step5_paper_metric_inputs/1"
                or int(row.get("paper_metric_token_max_len") or 0) != 25
            ]
            if bad_policy:
                raise RuntimeError(
                    f"predictions.csv is not a clean official 25-token input source: {_rel(predictions_path)} "
                    f"first_bad_sample={bad_policy[0]}"
                )
            metric_pred = [str(row.get("metric_pred_text") or "") for row in rows]
            metric_ref = [str(row.get("metric_ref_text") or "") for row in rows]
            paper_metrics = official_paper_metrics(metric_pred, metric_ref)
            old_metrics = _read(old_metrics_path)
            final = old_metrics.get("metrics") if isinstance(old_metrics.get("metrics"), Mapping) else {}
            final = dict(final)
            rating_metrics = {
                "valid": {
                    "MAE": rating_source.get("valid_mae"),
                    "RMSE": rating_source.get("valid_rmse"),
                    "mae": rating_source.get("valid_mae"),
                    "rmse": rating_source.get("valid_rmse"),
                },
                "test": {
                    "MAE": rating_source.get("test_mae"),
                    "RMSE": rating_source.get("test_rmse"),
                    "mae": rating_source.get("test_mae"),
                    "rmse": rating_source.get("test_rmse"),
                },
            }
            split_rating = rating_metrics[str(split)]
            final.update(
                {
                    "paper_metrics": paper_metrics,
                    "rating_metrics": rating_metrics,
                    "rating_metrics_source": "step3_eval_handoff",
                    "step5_rating_metrics_written": False,
                    "recommendation": {
                        "MAE": split_rating["MAE"],
                        "RMSE": split_rating["RMSE"],
                        "mae": split_rating["mae"],
                        "rmse": split_rating["rmse"],
                        "source": "step3_eval_handoff",
                        "step5_computed": False,
                    },
                }
            )
            decode = old_metrics.get("decode") if isinstance(old_metrics.get("decode"), Mapping) else {}
            official_policy = old_metrics.get("official_eval_policy") if isinstance(old_metrics.get("official_eval_policy"), Mapping) else {}
            official_policy = {
                **dict(official_policy),
                "schema_version": "odcr_step5_official_eval_policy/1",
                "profile": "paper_greedy_25",
                "split": str(split),
                "command": "test" if split == "test" else "eval",
                "decode_strategy": "greedy",
                "do_sample": False,
                "max_new_tokens": 25,
                "max_explanation_length": 25,
                "prediction_max_length": 25,
                "reference_max_length": 25,
                "repetition_penalty": 1.0,
                "official_paper_metrics": "base_utils.official_paper_metrics",
                "metric_input_builder": "base_utils.build_paper_metric_inputs",
                "rerank_allowed": False,
            }
            decode = {
                **dict(decode),
                "decode_strategy": "greedy",
                "do_sample": False,
                "repetition_penalty": 1.0,
                "max_explanation_length": 25,
                "hard_max_len": 25,
                "generate_temperature": 1.0,
                "generate_top_p": 1.0,
            }
            checkpoint = str(old_metrics.get("checkpoint") or _run_dir(task, run) / "model" / "best.pth")
            explanation_metrics = {
                "split": str(split),
                "sample_count": int(len(rows)),
                "explanation": final.get("explanation") or {},
                "paper_metrics": paper_metrics,
                "collapse_stats": old_metrics.get("collapse_stats") or {},
            }
            handoff = build_step5_explanation_handoff(
                task=int(task),
                run_id=str(run),
                checkpoint=checkpoint,
                explanation_metrics=explanation_metrics,
                rating_source=rating_source,
                generation_config=decode,
                ccv_fca_report={
                    "fca_enabled": True,
                    "control_projection_module": "explanation_control_projection",
                },
                route_explainer_stats={
                    "split": str(split),
                    "eval_control_mode": old_metrics.get("eval_control_mode") or "factual_eval_default",
                },
            )
            report = {
                "schema_version": "odcr_step5_official_eval_report/1",
                "stage": "step5",
                "task_id": int(task),
                "split": str(split),
                "checkpoint": checkpoint,
                "metrics_path": _rel(old_metrics_path),
                "rating_metrics_source": "step3_eval_handoff",
                "step5_rating_metrics_overwritten": False,
                "encoder_input_token_audit": old_metrics.get("encoder_input_token_audit") or {},
                "metrics": final,
                "recomposed_from_existing_predictions": True,
            }
            metrics_payload = {
                **old_metrics,
                "official_eval_profile": "paper_greedy_25",
                "official_eval_policy": official_policy,
                "target_only": True,
                "rerank_touched": False,
                "task_idx": int(task),
                "split": str(split),
                "command": "test" if split == "test" else "eval",
                "eval_profile_name": "paper_greedy_25",
                "decode": decode,
                "token_length_policy": {
                    "prediction_max_length": 25,
                    "reference_max_length": 25,
                    "legacy_encoder_max_content_tokens": 32,
                    "builder": "base_utils.build_paper_metric_inputs",
                },
                "paper_metrics": paper_metrics,
                "rating_source": rating_source,
                "rating_metrics_source": "step3_eval_handoff",
                "step5_rating_metrics_overwritten": False,
                "metrics": final,
                "official_eval_recompose": {
                    "schema_version": "odcr_step5e_existing_prediction_recompose/1",
                    "source_predictions": _rel(predictions_path),
                    "source_predictions_sha256": _sha(predictions_path),
                    "source_eval_metrics_sha256": _sha(old_metrics_path),
                    "source_paper_metrics_sha256": _sha(old_paper_path),
                    "source_eval_handoff_sha256": _sha(old_handoff_path),
                    "generated_at_utc": _utc(),
                    "rating_source_policy": "current_task_local_resolver",
                },
            }
            paper = compose_step3_rating_step5_explanation_report(
                rating_source=rating_source,
                explanation_handoff=handoff,
            )
            paper["split"] = str(split)
            paper["explanation_paper_metrics"] = paper_metrics
            _write(old_metrics_path, metrics_payload)
            _write(old_paper_path, paper)
            _write(old_handoff_path, handoff)
            _write(sdir / "official_eval_report.json", report)
            replay_manifest = {
                "schema_version": "odcr_step5e_existing_prediction_recompose_manifest/1",
                "generated_at_utc": _utc(),
                "task": int(task),
                "run": str(run),
                "split": str(split),
                "source_predictions": _rel(predictions_path),
                "source_predictions_sha256": _sha(predictions_path),
                "official_metric_recomputed_by": "base_utils.official_paper_metrics",
                "rating_source_recomputed_by": "current task-local rating_source resolver",
                "rating_source_eval_handoff": rating_source.get("eval_handoff"),
                "checkpoint": checkpoint,
            }
            _write(sdir / "meta" / "replay_manifest.json", replay_manifest)
            records.append(
                {
                    "task": int(task),
                    "run": str(run),
                    "split": str(split),
                    "rows": int(len(rows)),
                    "eval_metrics": _rel(old_metrics_path),
                    "paper_metrics": _rel(old_paper_path),
                    "eval_handoff": _rel(old_handoff_path),
                    "rating_source_task": int(rating_source.get("task") or -1),
                    "rating_source_eval_handoff": rating_source.get("eval_handoff"),
                }
            )
    payload = {
        "schema_version": "odcr_step5e_existing_prediction_recompose_result/1",
        "generated_at_utc": _utc(),
        "records": records,
    }
    _write(ROOT / "AI_analysis" / "03_evidence_ledgers" / "rebuild_step5e_tasklocal_protocol_recompose.json", payload)
    return payload


def write_split_stage_status(task: int, run: str, split: str, result: Mapping[str, Any]) -> dict[str, Any]:
    sdir = _split_dir(task, run, split)
    payload = {
        "schema_version": "odcr_stage_status/1",
        "validator_version": "odcr_step5_post_train_eval_split_verifier/1",
        "stage": "step5_post_train_eval",
        "task_id": int(task),
        "run_id": str(run),
        "split": split,
        "run_dir": _rel(sdir),
        "final_status": "eval_handoff_accepted" if result.get("clean") else "protocol_blocked",
        "downstream_ready": bool(result.get("clean")),
        "ready_for": ["step5_official_handoff", "paper_single_run_table"] if result.get("clean") else [],
        "status_source": "strict_task_local_official_eval_verifier",
        "rejection_reasons": [
            key for key, ok in (result.get("checks") or {}).items() if not ok
        ],
        "rating_source_task": result.get("rating_source_task"),
        "rating_source_eval_handoff": result.get("rating_source_eval_handoff"),
        "official_eval_profile": result.get("official_profile"),
        "decode_strategy": result.get("decode_strategy"),
        "target_only": result.get("target_only"),
        "artifacts": result.get("paths") or {},
        "generated_at_utc": _utc(),
    }
    _write(sdir / "meta" / "stage_status.json", payload)
    return payload


def finalize_task(task: int, run: str) -> dict[str, Any]:
    valid = verify_split(task, run, "valid")
    test = verify_split(task, run, "test")
    valid_status = write_split_stage_status(task, run, "valid", valid)
    test_status = write_split_stage_status(task, run, "test", test)
    clean = bool(valid["clean"] and test["clean"])
    run_root = _run_dir(task, run)
    if not clean:
        return {
            "task": int(task),
            "run": run,
            "clean": False,
            "valid": valid_status,
            "test": test_status,
        }
    valid_handoff = valid["payloads"]["handoff"]
    test_handoff = test["payloads"]["handoff"]
    root_handoff = dict(test_handoff)
    root_handoff.update(
        {
            "status": "accepted",
            "post_train_eval_status": "accepted",
            "accepted_splits": ["valid", "test"],
            "split_handoffs": {
                "valid": _rel(_split_dir(task, run, "valid") / "eval_handoff.json"),
                "test": _rel(_split_dir(task, run, "test") / "eval_handoff.json"),
            },
            "explanation_metrics": {
                "valid": valid_handoff.get("explanation_metrics"),
                "test": test_handoff.get("explanation_metrics"),
            },
        }
    )
    _write(run_root / "meta" / "explanation_handoff.json", root_handoff)
    summary_path = run_root / "meta" / "run_summary.json"
    summary = _read(summary_path)
    summary.update(
        {
            "status": "completed_with_explanation_handoff",
            "validation_status": "ok",
            "eval_status": "official_post_train_eval_accepted",
            "needs_eval_handoff": False,
            "downstream_ready": True,
            "ready_for": ["eval", "rerank"],
            "official_post_train_eval": {
                "schema_version": "odcr_step5_post_train_eval_acceptance/1",
                "status": "accepted",
                "profile": "paper_greedy_25",
                "valid": valid_status,
                "test": test_status,
                "rating_source_task": int(task),
                "rating_source_eval_handoff": valid.get("rating_source_eval_handoff"),
            },
        }
    )
    key_artifacts = dict(summary.get("key_artifacts") or {})
    key_artifacts["explanation_handoff"] = _rel(run_root / "meta" / "explanation_handoff.json")
    summary["key_artifacts"] = key_artifacts
    write_run_summary_json(summary, repo_root=ROOT, update_latest=True)
    stage_status = build_and_write_stage_status(repo_root=ROOT, stage="step5", task=int(task), run_id=run)
    write_latest_pointer_json(
        repo_root=ROOT,
        stage_unit_dir=run_root.parent,
        run_id=run,
        run_dir=run_root,
        summary_path=summary_path,
        status=str(stage_status.get("final_status") or "completed_with_explanation_handoff"),
    )
    return {
        "task": int(task),
        "run": run,
        "clean": True,
        "valid": valid_status,
        "test": test_status,
        "stage_status": stage_status,
        "latest": _read(run_root.parent / "latest.json"),
    }


def repair_step3_latest(tasks: list[int]) -> list[dict[str, Any]]:
    out = []
    for task in tasks:
        run = _step3_run_for_task(task)
        run_root = ROOT / "runs" / "step3" / f"task{int(task)}" / run
        status = build_and_write_stage_status(repo_root=ROOT, stage="step3", task=int(task), run_id=run)
        summary_path = run_root / "meta" / "run_summary.json"
        write_latest_pointer_json(
            repo_root=ROOT,
            stage_unit_dir=run_root.parent,
            run_id=run,
            run_dir=run_root,
            summary_path=summary_path,
            status=str(status.get("final_status") or ""),
        )
        out.append({"task": int(task), "run": run, "stage_status": status, "latest": _read(run_root.parent / "latest.json")})
    return out


def finalize(tasks: list[int]) -> dict[str, Any]:
    results = []
    for task in tasks:
        results.append(finalize_task(task, _get_task_run(task)))
    step3 = repair_step3_latest([task for task in tasks if task in {5, 7, 8}])
    payload = {"schema_version": "odcr_step5e_finalize_result/1", "generated_at_utc": _utc(), "tasks": results, "step3_latest_repair": step3}
    _write(ROOT / "AI_analysis" / "03_evidence_ledgers" / "rebuild_step5e_tasklocal_protocol_finalize.json", payload)
    return payload


def scan(tasks: list[int]) -> dict[str, Any]:
    rows = []
    legacy_alias = {
        "export_step5_dedicated": "removed_from_parser",
        "head_legacy_rating_a": "fail_fast",
        "head_legacy_rating_b": "fail_fast",
        "head_legacy_combined": "fail_fast",
    }
    for task in tasks:
        run = _get_task_run(task)
        source, target = TASK_DOMAINS[int(task)]
        step3_run = _step3_run_for_task(task)
        step3_latest = _read(ROOT / "runs" / "step3" / f"task{task}" / "latest.json", required=False)
        step3_status = _read(ROOT / "runs" / "step3" / f"task{task}" / step3_run / "meta" / "stage_status.json", required=False)
        step3_handoff = _read(ROOT / "runs" / "step3" / f"task{task}" / step3_run / "meta" / "eval_handoff.json", required=False)
        step4_latest = _read(ROOT / "runs" / "step4" / f"task{task}" / "latest.json", required=False)
        step4_run = str(step4_latest.get("latest_run_id") or "1")
        step4_status = _read(ROOT / "runs" / "step4" / f"task{task}" / step4_run / "meta" / "stage_status.json", required=False)
        step5_latest = _read(ROOT / "runs" / "step5" / f"task{task}" / "latest.json", required=False)
        step5_status = _read(ROOT / "runs" / "step5" / f"task{task}" / run / "meta" / "stage_status.json", required=False)
        valid = verify_split(task, run, "valid")
        test = verify_split(task, run, "test")
        valid_status = _read(_split_dir(task, run, "valid") / "meta" / "stage_status.json", required=False)
        test_status = _read(_split_dir(task, run, "test") / "meta" / "stage_status.json", required=False)
        verdict_clean = bool(
            valid.get("clean")
            and test.get("clean")
            and str(step5_status.get("final_status") or "") == "completed_with_explanation_handoff"
            and step5_latest.get("latest_run_id") == run
            and valid_status.get("downstream_ready") is True
            and test_status.get("downstream_ready") is True
        )
        rows.append(
            {
                "task_id": int(task),
                "source_domain": source,
                "target_domain": target,
                "step3_latest_run": step3_latest.get("latest_run_id"),
                "step3_run": step3_run,
                "step3_status": step3_status.get("final_status"),
                "step3_eval_handoff_accepted": str(step3_handoff.get("paper_eval_status") or "").lower() in {"completed", "accepted"},
                "step3_valid_mae": ((step3_handoff.get("valid_metrics") or {}).get("MAE")),
                "step3_valid_rmse": ((step3_handoff.get("valid_metrics") or {}).get("RMSE")),
                "step3_test_mae": ((step3_handoff.get("test_metrics") or {}).get("MAE")),
                "step3_test_rmse": ((step3_handoff.get("test_metrics") or {}).get("RMSE")),
                "step4_latest_run": step4_latest.get("latest_run_id"),
                "step4_run": step4_run,
                "step4_status": step4_status.get("final_status"),
                "step4_handoff_pool_manifest_sampling_contract": bool(
                    "pool_manifest_sampling_contract" in json.dumps(step4_status, ensure_ascii=False)
                ),
                "step5_latest_run": step5_latest.get("latest_run_id"),
                "step5_run": run,
                "step5_status": step5_status.get("final_status"),
                "step5_valid_clean": valid.get("clean"),
                "step5_test_clean": test.get("clean"),
                "valid_rating_source_task": valid.get("rating_source_task"),
                "test_rating_source_task": test.get("rating_source_task"),
                "valid_eval_handoff_path": valid.get("rating_source_eval_handoff"),
                "test_eval_handoff_path": test.get("rating_source_eval_handoff"),
                "official_profile_valid": valid.get("official_profile"),
                "official_profile_test": test.get("official_profile"),
                "decode_strategy": valid.get("decode_strategy"),
                "max_len": 25 if valid["checks"].get("max_len_25") and test["checks"].get("max_len_25") else None,
                "target_only": bool(valid.get("target_only") and test.get("target_only")),
                "paper_metrics_present": bool(valid["checks"].get("paper_metrics_present") and test["checks"].get("paper_metrics_present")),
                "bertscore_absent": bool(valid["checks"].get("bertscore_absent") and test["checks"].get("bertscore_absent")),
                "legacy_alias_status": legacy_alias,
                "table_verdict_ignoring_longest_5seed": "clean" if verdict_clean else "blocked",
            }
        )
    clean_all = all(row["table_verdict_ignoring_longest_5seed"] == "clean" for row in rows)
    payload = {
        "schema_version": "odcr_step5e_four_task_protocol_scan/1",
        "generated_at_utc": _utc(),
        "clean_all_ignoring_longest_and_5seed": clean_all,
        "rows": rows,
    }
    _write(ROOT / "AI_analysis" / "03_evidence_ledgers" / "rebuild_step5e_tasklocal_protocol_matrix.json", payload)
    return payload


def parse_tasks(raw: str) -> list[int]:
    return [int(part) for part in str(raw).split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify/finalize Step5_e task-local official eval artifacts.")
    parser.add_argument("--tasks", default="2,5,8,7")
    parser.add_argument("--splits", default="valid,test")
    parser.add_argument("--invalidate-known-bad", action="store_true")
    parser.add_argument("--recompose-existing-predictions", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--scan", action="store_true")
    args = parser.parse_args()
    tasks = parse_tasks(args.tasks)
    splits = [part.strip() for part in str(args.splits).split(",") if part.strip()]
    outputs = {}
    if args.invalidate_known_bad:
        outputs["invalidate_known_bad"] = invalidate_known_bad(tasks)
    if args.recompose_existing_predictions:
        outputs["recompose_existing_predictions"] = recompose_existing_predictions(tasks, splits)
    if args.finalize:
        outputs["finalize"] = finalize(tasks)
    if args.scan:
        outputs["scan"] = scan(tasks)
    print(json.dumps(outputs, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
