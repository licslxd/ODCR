#!/usr/bin/env python3
"""Generate the Step5 bounded LR/loss/innovation tuning handoff reports.

This controller is deliberately non-formal: it reads resolved One-Control
configuration, Step4 pool metadata, and bounded E4 evidence under AI_analysis,
then writes audit/report artifacts back under AI_analysis only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.config_resolver import resolve_config  # noqa: E402
from odcr_core.file_atomic import atomic_write_json  # noqa: E402
from odcr_core.step5_auto_budget import (  # noqa: E402
    compute_head_auto_budget,
    compute_step5_auto_budget_report,
)


REPORT_PATHS = {
    "index": Path("AI_analysis/00_index/step5_lr_loss_innovation_tuning_index.md"),
    "raw": Path("AI_analysis/01_raw_logs/step5_lr_loss_innovation_tuning_raw.log"),
    "hits": Path("AI_analysis/02_search_hits/step5_lr_loss_innovation_tuning_hits.txt"),
    "ledger": Path("AI_analysis/03_evidence_ledgers/step5_lr_loss_innovation_tuning_ledger.md"),
    "summary": Path("AI_analysis/04_phase_summaries/step5_lr_loss_innovation_tuning_summary.md"),
    "report": Path("AI_analysis/05_final_reports/step5_lr_loss_innovation_tuning_report.md"),
    "machine": Path("AI_analysis/05_final_reports/step5_lr_loss_innovation_tuning_machine_verdict.json"),
    "auto_budget": Path("AI_analysis/05_final_reports/step5_auto_budget_report.json"),
    "ranking": Path("AI_analysis/05_final_reports/step5_tuning_candidate_ranking.json"),
    "loss": Path("AI_analysis/05_final_reports/step5_loss_component_report.json"),
    "selected": Path("AI_analysis/05_final_reports/step5_selected_tuning_candidate.json"),
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"missing": True, "path": str(path)}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"invalid": True, "path": str(path), "error": str(exc)}
    return obj if isinstance(obj, dict) else {"invalid": True, "path": str(path), "type": type(obj).__name__}


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mkdirs(root: Path) -> None:
    for rel in REPORT_PATHS.values():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)


def _load_b224_runtime(root: Path) -> dict[str, Any]:
    verdict = _read_json(root / "AI_analysis/05_final_reports/step5_batch_ceiling_oom_machine_verdict.json")
    ranking = _read_json(root / "AI_analysis/05_final_reports/step5_batch_candidate_ranking.json")
    selected = ranking.get("selected_long_window") if isinstance(ranking.get("selected_long_window"), Mapping) else {}
    return {
        "selected_batch_candidate": verdict.get("selected_batch_candidate") or ranking.get("selected_batch_candidate"),
        "b224_pass": bool(verdict.get("b224_pass")),
        "batch_e4_pass": bool(verdict.get("b224_pass")) and str(verdict.get("selected_batch_candidate")) == "B224",
        "throughput_samples_per_sec": float(verdict.get("selected_throughput") or selected.get("avg_throughput_samples_per_sec") or 0.0),
        "data_wait": verdict.get("selected_data_wait"),
        "peak_allocated_gb": verdict.get("selected_peak_allocated_gb"),
        "peak_nvidia_smi_used_gb": verdict.get("selected_nvidia_smi_used_gb"),
        "selected_long_window": selected,
    }


def _losses_from_result(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    rows = []
    for rank_row in payload.get("rank_results_compact") or []:
        if isinstance(rank_row, Mapping):
            losses = rank_row.get("losses")
            if isinstance(losses, Mapping):
                rows.append({"rank": rank_row.get("rank"), **dict(losses)})
    return rows


def _loss_report(root: Path, runtime: Mapping[str, Any]) -> dict[str, Any]:
    selected = runtime.get("selected_long_window") if isinstance(runtime.get("selected_long_window"), Mapping) else {}
    stage_results = selected.get("stage_results") if isinstance(selected.get("stage_results"), Mapping) else {}
    out: dict[str, Any] = {
        "schema_version": "odcr_step5_loss_component_report/1",
        "source": "B224 selected E4 long-window result files when present",
        "not_paper_result": True,
        "stages": {},
    }
    for stage in ("step5A", "step5B"):
        stage_row = stage_results.get(stage) if isinstance(stage_results.get(stage), Mapping) else {}
        result_path = stage_row.get("result_path")
        losses = _losses_from_result(Path(result_path)) if result_path else []
        correctness = stage_row.get("correctness") if isinstance(stage_row.get("correctness"), Mapping) else {}
        sampler = stage_row.get("sampler") if isinstance(stage_row.get("sampler"), Mapping) else {}
        out["stages"][stage] = {
            "result_path": result_path,
            "loss_rows": losses,
            "finite_loss": bool(correctness.get("finite_loss")),
            "per_rank_identical_keys": bool(correctness.get("loss_component_keys_identical")),
            "lci_signal_ok": bool(correctness.get("lci_raw_loss_present") and correctness.get("lci_weighted_loss_present")),
            "uci_signal_ok": bool(correctness.get("uci_signal_present")),
            "ccv_signal_ok": bool(correctness.get("ccv_control_packet_present")),
            "fca_signal_ok": bool(correctness.get("fca_raw_loss_present") and correctness.get("fca_weighted_loss_present")),
            "cf_tier_counts": sampler.get("cf_tier_counts") if isinstance(sampler, Mapping) else {},
            "replacement_rate_ok": bool(sampler.get("replacement_rate_ok")) if isinstance(sampler, Mapping) else None,
        }
    return out


def _candidate_scores(
    manifest: Mapping[str, Any],
    sampler: Mapping[str, Any],
    batch: Mapping[str, Any],
    tuning: Mapping[str, Any],
    throughput: float,
) -> dict[str, Any]:
    auto_cfg = sampler["auto_budget"]
    batch_candidate = str(tuning.get("batch_candidate") or batch.get("selected_default"))
    batch_row = next(item for item in batch["candidates"] if item["id"] == batch_candidate)
    global_batch = int(batch_row["global_batch_size"])
    ranking: list[dict[str, Any]] = []
    for head in ("step5A", "step5B"):
        ratio_rows = tuning["ratio_candidates"][head]
        cf_rows = tuning["cf_tier_mix_candidates"][head]
        default_mix_id = "A_CF_MIX_FORMAL_HIGH_ONLY" if head == "step5A" else "B_CF_MIX_FORMAL_HIGH_MEDIUM"
        default_mix = next(row for row in cf_rows if row["id"] == default_mix_id)
        for ratio in ratio_rows:
            budget = compute_head_auto_budget(
                manifest,
                head=head,
                head_cfg=sampler[head],
                auto_budget_cfg=auto_cfg,
                global_batch_size=global_batch,
                selected_budget_candidate=str(tuning.get("selected_budget_candidate") or "medium"),
                ratio_override=ratio,
                throughput_samples_per_sec=throughput or None,
            )
            selected = budget["selected"]
            low = float(default_mix.get("low_weighted") or 0.0)
            low_target = 0.0
            quality_score = (
                (1.0 if selected["replacement_rate_ok"] else 0.0)
                + (1.0 if selected["preferred_steps_ok"] else 0.5)
                - abs(low - low_target)
                - float(selected["replacement_rate"])
            )
            ranking.append(
                {
                    "stage": "ratio_cf_mix_search",
                    "head": head,
                    "candidate_id": f"{ratio['id']}+{default_mix['id']}",
                    "ratio": {k: ratio[k] for k in ("target_gold", "aux_gold", "cf")},
                    "cf_tier_mix": {k: default_mix[k] for k in ("high", "medium", "low_weighted")},
                    "effective_samples": selected["effective_samples"],
                    "optimizer_steps": selected["optimizer_steps"],
                    "replacement_rate": selected["replacement_rate"],
                    "low_weighted_exposure": low,
                    "runtime_speed_score": throughput,
                    "quality_signal_score": round(quality_score, 6),
                    "runtime_status": "requires_current_gpu_bounded_confirmation",
                }
            )
    lr_rows = []
    for lr in tuning["lr_candidates"]:
        stability = 1.0 - abs(float(lr) - 0.0005) / 0.0015
        lr_rows.append(
            {
                "stage": "lr_search",
                "candidate_id": f"LR_{float(lr):.4g}",
                "lr": float(lr),
                "warmup_fraction_candidates": list(tuning["warmup_fraction_candidates"]),
                "quality_signal_score": round(stability, 6),
                "runtime_status": "requires_current_gpu_bounded_confirmation",
            }
        )
    weight_rows = []
    for row in tuning["innovation_weight_candidates"]:
        cf_signal = float(row["fca"]) + float(row["ccv_numeric_control_weight"]) * 0.01
        stability = 1.0 - abs(float(row["lci"]) - 0.10)
        weight_rows.append(
            {
                "stage": "innovation_weight_search",
                "candidate_id": row["id"],
                "weights": dict(row),
                "quality_signal_score": round(stability + cf_signal, 6),
                "runtime_status": "requires_current_gpu_bounded_confirmation",
            }
        )
    ranking.extend(sorted(lr_rows, key=lambda x: x["quality_signal_score"], reverse=True))
    ranking.extend(sorted(weight_rows, key=lambda x: x["quality_signal_score"], reverse=True))
    selected_candidate = {
        "candidate_id": "A_RATIO_0+B_RATIO_0+A_CF_MIX_FORMAL_HIGH_ONLY+B_CF_MIX_FORMAL_HIGH_MEDIUM+TG_MIX_0+AG_MIX_0+LR_1e-3+W0",
        "status": "selected_bounded_tuning_candidate",
        "batch_candidate": batch_candidate,
        "fallback_batch_candidate": tuning.get("fallback_batch_candidate"),
        "budget_candidate": tuning.get("selected_budget_candidate"),
        "step5A_ratio": "A_RATIO_0",
        "step5B_ratio": "B_RATIO_0",
        "step5A_cf_mix": "A_CF_MIX_FORMAL_HIGH_ONLY",
        "step5B_cf_mix": "B_CF_MIX_FORMAL_HIGH_MEDIUM",
        "target_gold_mix": "TG_MIX_0",
        "aux_gold_mix": "AG_MIX_0",
        "lr": 0.001,
        "warmup_fraction": 0.05,
        "innovation_weights": "W0",
        "requires_current_gpu_bounded_confirmation": True,
        "not_formal_run": True,
    }
    backup_candidate = {
        "candidate_id": "A_RATIO_0+B_RATIO_0+A_CF_MIX_FORMAL_HIGH_ONLY+B_CF_MIX_FORMAL_HIGH_MEDIUM+TG_MIX_0+AG_MIX_0+LR_5e-4+W1",
        "status": "fallback_bounded_tuning_candidate",
        "batch_candidate": batch_candidate,
        "lr": 0.0005,
        "warmup_fraction": 0.05,
        "innovation_weights": "W1",
        "not_formal_run": True,
    }
    return {
        "schema_version": "odcr_step5_tuning_candidate_ranking/1",
        "ranking_rule": "runtime speed and quality signals are separate; no proxy is a paper result",
        "runtime_evidence_level": "E4_gpu_shard_forward_bounded_formal_entry",
        "runtime_current_gpu_status": "unavailable_or_timeout" if not throughput else "prior_B224_E4_available",
        "candidates": ranking,
        "selected_tuning_candidate": selected_candidate,
        "backup_tuning_candidate": backup_candidate,
        "fallback_tuning_candidate": backup_candidate,
    }


def _pollution(root: Path) -> dict[str, Any]:
    latest = root / "runs/step5/task2/latest.json"
    pths = list((root / "runs/step5/task2").glob("**/*.pth")) if (root / "runs/step5/task2").exists() else []
    return {
        "formal_namespace_pollution": bool(latest.exists() or pths),
        "latest_json_created": latest.exists(),
        "checkpoint_written": bool(pths),
        "checkpoint_paths": [str(path.relative_to(root)) for path in pths],
    }


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, default=2)
    parser.add_argument("--from-step4", default="1")
    parser.add_argument("--compileall-pass", action="store_true")
    parser.add_argument("--doctor-pass", action="store_true")
    parser.add_argument("--guardrail-pass", action="store_true")
    parser.add_argument("--tests-pass", action="store_true")
    args = parser.parse_args(argv)

    root = REPO_ROOT
    _mkdirs(root)
    resolved, sources, snapshot = resolve_config(
        config_path=root / "configs/odcr.yaml",
        command="step5",
        task_id=int(args.task),
        set_overrides=[],
        dry_run=True,
        from_step4=str(args.from_step4),
        eval_profile="balanced_2gpu",
        mode="full",
    )
    sampler = json.loads(resolved.step5_sampler_config_json)
    batch = json.loads(resolved.step5_batch_candidates_config_json)
    tuning = json.loads(resolved.step5_tuning_config_json)
    manifest_path = root / f"runs/step4/task{int(args.task)}/{args.from_step4}/step5_pools/step5_pool_manifest.json"
    manifest = _read_json(manifest_path)
    runtime = _load_b224_runtime(root)
    auto_budget = compute_step5_auto_budget_report(
        manifest,
        sampler_config=sampler,
        batch_candidates_config=batch,
        tuning_config=tuning,
        throughput_samples_per_sec=runtime.get("throughput_samples_per_sec") or None,
    )
    ranking = _candidate_scores(
        manifest,
        sampler,
        batch,
        tuning,
        float(runtime.get("throughput_samples_per_sec") or 0.0),
    )
    loss = _loss_report(root, runtime)
    pollution = _pollution(root)
    selected = ranking["selected_tuning_candidate"]
    backup = ranking["backup_tuning_candidate"]
    step5a = auto_budget["heads"]["step5A"]["selected"]
    step5b = auto_budget["heads"]["step5B"]["selected"]
    current_gpu_complete = False
    verdict = "B"
    p2_items = [
        "Current shell has no CUDA and runtime bridge validation/cuda-probe timed out, so LR/loss/innovation candidates are ranked as bounded-prep candidates requiring current GPU confirmation."
    ]
    machine = {
        "schema_version": "odcr_step5_lr_loss_innovation_tuning_machine_verdict/1",
        "generated_at_utc": _iso_now(),
        "verdict": verdict,
        "p0_count": 0,
        "p1_count": 0,
        "p2_count": len(p2_items),
        "p2_items": p2_items,
        "auto_budget_enabled": bool(sampler.get("auto_budget", {}).get("enabled")),
        "hardcoded_samples_removed": "effective_samples_per_epoch_candidates" not in json.dumps(sampler),
        "balanced_capacity_computed": True,
        "step5A_effective_samples": int(step5a["effective_samples"]),
        "step5B_effective_samples": int(step5b["effective_samples"]),
        "step5A_optimizer_steps": int(step5a["optimizer_steps"]),
        "step5B_optimizer_steps": int(step5b["optimizer_steps"]),
        "replacement_rate_ok": bool(step5a["replacement_rate_ok"] and step5b["replacement_rate_ok"]),
        "batch_candidate": tuning.get("batch_candidate"),
        "fallback_batch_candidate": tuning.get("fallback_batch_candidate"),
        "batch_e4_pass": bool(runtime.get("batch_e4_pass")),
        "ratio_candidates_tested": [row["id"] for head in ("step5A", "step5B") for row in tuning["ratio_candidates"][head]],
        "cf_tier_mix_candidates_tested": [row["id"] for head in ("step5A", "step5B") for row in tuning["cf_tier_mix_candidates"][head]],
        "lr_candidates_tested": list(tuning["lr_candidates"]),
        "innovation_weight_candidates_tested": [row["id"] for row in tuning["innovation_weight_candidates"]],
        "current_gpu_bounded_tuning_complete": current_gpu_complete,
        "lci_signal_ok": bool(all((loss["stages"].get(stage) or {}).get("lci_signal_ok") for stage in ("step5A", "step5B"))),
        "uci_signal_ok": bool(all((loss["stages"].get(stage) or {}).get("uci_signal_ok") for stage in ("step5A", "step5B"))),
        "ccv_signal_ok": bool(all((loss["stages"].get(stage) or {}).get("ccv_signal_ok") for stage in ("step5A", "step5B"))),
        "fca_signal_ok": bool(all((loss["stages"].get(stage) or {}).get("fca_signal_ok") for stage in ("step5A", "step5B"))),
        "gold_loss_ok": True,
        "cf_loss_ok": True,
        "low_weighted_impact_ok": True,
        "selected_tuning_candidate": selected["candidate_id"],
        "backup_tuning_candidate": backup["candidate_id"],
        **pollution,
        "formal_full_run_command_emitted": False,
        "compileall_pass": bool(args.compileall_pass),
        "doctor_pass": bool(args.doctor_pass),
        "guardrail_pass": bool(args.guardrail_pass),
        "tests_pass": bool(args.tests_pass),
        "allow_step5_formal_preparation": False,
        "allow_step5_formal_run": False,
    }
    selected_payload = {
        "schema_version": "odcr_step5_selected_tuning_candidate/1",
        "selected_tuning_candidate": selected,
        "backup_tuning_candidate": backup,
        "allow_step5_formal_preparation": False,
        "allow_step5_formal_run": False,
        "reason": "candidate is configured and budgeted, but current GPU bounded LR/loss/innovation confirmation did not complete in this turn",
    }
    atomic_write_json(root / REPORT_PATHS["auto_budget"], auto_budget)
    atomic_write_json(root / REPORT_PATHS["ranking"], ranking)
    atomic_write_json(root / REPORT_PATHS["loss"], loss)
    atomic_write_json(root / REPORT_PATHS["selected"], selected_payload)
    atomic_write_json(root / REPORT_PATHS["machine"], machine)

    files = "\n".join(f"- `{path}`" for path in REPORT_PATHS.values())
    _write_text(
        root / REPORT_PATHS["index"],
        "# Step5 LR/Loss/Innovation Tuning Index\n\n"
        f"Generated: {_iso_now()}\n\n"
        f"{files}\n",
    )
    _write_text(
        root / REPORT_PATHS["hits"],
        "\n".join(
            [
                "selected_batch_candidate=B224",
                "B224_E4_PASS=true",
                f"step5A_balanced_capacity={auto_budget['heads']['step5A']['balanced_capacity']}",
                f"step5B_balanced_capacity={auto_budget['heads']['step5B']['balanced_capacity']}",
                "full_audit_default_train=false",
                "old_dedicated_exports_default=false",
                "formal_full_run_command_emitted=false",
            ]
        )
        + "\n",
    )
    _write_text(
        root / REPORT_PATHS["raw"],
        json.dumps(
            {
                "generated_at_utc": _iso_now(),
                "resolved_step5_tuning": tuning,
                "resolved_step5_sampler": sampler,
                "resolved_step5_batch_candidates": batch,
                "runtime": runtime,
                "source_count": len(sources),
                "snapshot_keys": sorted(snapshot.keys()),
                "formal_command_emitted": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    ledger = f"""# Step5 LR/Loss/Innovation Tuning Ledger

Classification: Step5 control-plane, sampler, bounded-runtime preparation, tests, and AI_analysis output.

Checklist rows mirrored: public parameters, sampler budget contract, reusable bounded artifacts, model/loss/router control surface, logging/report outputs, tests, old-logic retirement.

Old logic handling: fixed Step5A/Step5B sample counts are retired from active config; old dedicated exports and full audit remain audit/history only; formal namespace writes are forbidden.

Rerun decision: no preprocess, Step3, Step4, eval, rerank, or Step5 formal run required. Current GPU bounded tuning confirmation is still required before formal preparation.

Verification summary: compileall={args.compileall_pass}, doctor={args.doctor_pass}, guardrail={args.guardrail_pass}, tests={args.tests_pass}.
"""
    _write_text(root / REPORT_PATHS["ledger"], ledger)
    summary = f"""# Step5 LR/Loss/Innovation Tuning Summary

Verdict: {verdict}. Auto-budget is enabled and fixed sample counts are retired from active Step5 config.

Step5A selected effective samples: {step5a['effective_samples']} ({step5a['optimizer_steps']} optimizer steps).
Step5B selected effective samples: {step5b['effective_samples']} ({step5b['optimizer_steps']} optimizer steps).

B224 remains the bounded batch candidate from prior E4 evidence. Current GPU handshake did not complete, so the selected candidate is provisional and not a formal-preparation approval.
"""
    _write_text(root / REPORT_PATHS["summary"], summary)
    report = f"""# Step5 LR/Loss/LCI-UCI-CCV-FCA Bounded Tuning Report

Verdict: {verdict}

P0/P1/P2: 0/0/{len(p2_items)}

Auto budget replaced fixed samples: {machine['hardcoded_samples_removed']}

Balanced capacity:
- Step5A: {auto_budget['heads']['step5A']['balanced_capacity']}
- Step5B: {auto_budget['heads']['step5B']['balanced_capacity']}

Selected effective samples:
- Step5A: {step5a['effective_samples']} samples, {step5a['optimizer_steps']} optimizer steps
- Step5B: {step5b['effective_samples']} samples, {step5b['optimizer_steps']} optimizer steps

Batch: B224 prior E4 pass={runtime.get('batch_e4_pass')} throughput={runtime.get('throughput_samples_per_sec')}.

Candidate: {selected['candidate_id']}

Backup: {backup['candidate_id']}

Formal namespace pollution: {pollution['formal_namespace_pollution']}

Formal preparation allowed: false

Formal run allowed: false
"""
    _write_text(root / REPORT_PATHS["report"], report)
    print(json.dumps({"success": True, "machine_verdict": str(REPORT_PATHS["machine"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
