"""Contracts for the RACER-C1 retrieval-first path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


RACER_C1_RUN_SCHEMA_VERSION = "odcr_racer_c1_run/1"
RACER_C1_SOURCE_TABLE_SCHEMA_VERSION = "odcr_racer_c1_source_table/1"
RACER_C1_STAGE_STATUS_SCHEMA_VERSION = "odcr_racer_c1_stage_status/1"
RACER_C1_LEAKAGE_SCHEMA_VERSION = "odcr_racer_c1_leakage_check/1"
RACER_C1_BOTTLENECK_SCHEMA_VERSION = "odcr_racer_c1_bottleneck_analysis/1"

REQUIRED_RUN_FILES: tuple[str, ...] = (
    "meta/console.log",
    "meta/full.log",
    "meta/errors.log",
    "meta/epoch_metrics.jsonl",
    "meta/resource_utilization.jsonl",
    "meta/throughput.jsonl",
    "diagnostics/bottleneck_analysis.json",
    "diagnostics/token_length_stats.json",
    "diagnostics/leakage_check.json",
    "diagnostics/innovation_alignment.json",
    "diagnostics/content_style_split_stats.json",
    "diagnostics/cf_anchor_stats.json",
    "diagnostics/contrastive_role_distribution.json",
    "diagnostics/copy_lcs_stats.json",
    "diagnostics/cross_domain_evidence_distribution.json",
    "diagnostics/retrieval_source_distribution.json",
    "diagnostics/rcr_bucket_distribution.json",
)

PLANNED_ARTIFACTS: tuple[str, ...] = (
    "evidence/train_evidence_pool.jsonl",
    "evidence/evidence_embeddings.npy",
    "evidence/query_features.npy",
    "train/contrastive_pairs_manifest.json",
    "train/best_checkpoint.pt",
    "predictions/valid_top1_diagnostic.jsonl",
    "predictions/test_top1_diagnostic.jsonl",
    "predictions/valid_composed_predictions.jsonl",
    "predictions/test_composed_predictions.jsonl",
    "metrics/valid_top1_diagnostic_paper_greedy_25.json",
    "metrics/test_top1_diagnostic_paper_greedy_25.json",
    "metrics/valid_official_paper_greedy_25.json",
    "metrics/test_official_paper_greedy_25.json",
)


@dataclass(frozen=True)
class RacerC1Paths:
    run_root: Path
    meta_dir: Path
    evidence_dir: Path
    train_dir: Path
    predictions_dir: Path
    metrics_dir: Path
    diagnostics_dir: Path

    @classmethod
    def from_root(cls, run_root: Path) -> "RacerC1Paths":
        return cls(
            run_root=run_root,
            meta_dir=run_root / "meta",
            evidence_dir=run_root / "evidence",
            train_dir=run_root / "train",
            predictions_dir=run_root / "predictions",
            metrics_dir=run_root / "metrics",
            diagnostics_dir=run_root / "diagnostics",
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "run_root": str(self.run_root),
            "meta_dir": str(self.meta_dir),
            "evidence_dir": str(self.evidence_dir),
            "train_dir": str(self.train_dir),
            "predictions_dir": str(self.predictions_dir),
            "metrics_dir": str(self.metrics_dir),
            "diagnostics_dir": str(self.diagnostics_dir),
        }


def planned_relative_outputs(config: dict[str, Any]) -> list[str]:
    configured = ((config.get("logging") or {}).get("required_files") or [])
    out = list(dict.fromkeys([*configured, *PLANNED_ARTIFACTS]))
    return [str(item) for item in out]
