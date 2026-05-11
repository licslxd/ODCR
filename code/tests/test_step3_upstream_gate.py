from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_upstream_gate import (  # noqa: E402
    DOMAIN_ARTIFACT_CONTRACT_VERSION,
    Step3UpstreamGateError,
    validate_step3_preprocess_upstream_gate,
)
from odcr_core.training_checkpoint import file_fingerprint  # noqa: E402


TASK_ID = 4
AUX = "AM_Movies"
TARGET = "AM_Electronics"
EMBED_DIM = 4
CSV_HEADER = (
    "user,item,rating,review,explanation,content_evidence,content_anchor_score,"
    "polarity_anchor,domain_style_anchor,local_style_residual_hint,style_evidence,"
    "style_anchor_score,evidence_quality_prior,preprocess_route_scorer_prior,"
    "preprocess_route_explainer_prior,user_idx,item_idx,domain\n"
)


class Step3UpstreamGateTests(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")

    def _write_csv(self, path: Path, domain: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(CSV_HEADER + f"u,i,5,review,explanation,ev,1,pos,style,hint,sev,1,1,1,1,0,0,{domain}\n", encoding="utf-8")

    def _fingerprint_after_write(self, path: Path) -> dict:
        return file_fingerprint(path, sample_only=True)

    def _header_meta(self, path: Path) -> dict:
        st = path.stat()
        return {
            "path": str(path.resolve()),
            "exists": True,
            "header": CSV_HEADER.rstrip("\n").split(","),
            "header_hash": "unit-test-header-hash",
            "file_size": int(st.st_size),
            "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
            "contract_kind": "merged",
            "header_match": True,
        }

    def _write_npy(self, path: Path, array: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)

    def _valid_repo(self, root: Path) -> dict[str, Path]:
        data_dir = root / "data"
        merged_dir = root / "merged"
        runs_dir = root / "runs"
        for domain in (AUX, TARGET):
            self._write_csv(data_dir / domain / "train.csv", domain)

        aug_train = merged_dir / str(TASK_ID) / "aug_train.csv"
        aug_valid = merged_dir / str(TASK_ID) / "aug_valid.csv"
        self._write_csv(aug_train, AUX)
        self._write_csv(aug_valid, TARGET)

        profile_paths: dict[str, dict[str, Path]] = {}
        domain_paths: dict[str, dict[str, Path]] = {}
        for domain in (AUX, TARGET):
            profile_paths[domain] = {}
            for key in ("user_content_profiles", "user_style_profiles", "item_content_profiles", "item_style_profiles"):
                path = data_dir / domain / f"{key}.npy"
                self._write_npy(path, np.ones((2, EMBED_DIM), dtype=np.float32))
                profile_paths[domain][key] = path
            domain_paths[domain] = {}
            for key in ("domain_content", "domain_style"):
                path = data_dir / domain / f"{key}.npy"
                self._write_npy(path, np.ones((EMBED_DIM,), dtype=np.float32))
                domain_paths[domain][key] = path

        source_csv_fps = {
            domain: self._fingerprint_after_write(data_dir / domain / "train.csv")
            for domain in (AUX, TARGET)
        }
        profile_fps = {
            domain: {
                key: self._fingerprint_after_write(path)
                for key, path in paths.items()
            }
            for domain, paths in profile_paths.items()
        }

        for unit in ("a", "b", "c"):
            run_dir = runs_dir / "preprocess" / unit / "1"
            meta_dir = run_dir / "meta"
            latest_path = runs_dir / "preprocess" / unit / "latest.json"
            summary_path = meta_dir / "run_summary.json"
            manifest_path = meta_dir / "stage_manifest.json"
            status_path = meta_dir / "stage_status.json"
            source_table_path = meta_dir / "source_table.json"
            resolved_config_path = meta_dir / "resolved_config.json"
            metrics_path = meta_dir / "metrics.json"
            verify_path = meta_dir / "verify_report.json"

            self._write_json(
                latest_path,
                {
                    "latest_run_id": "1",
                    "latest_run_dir": str(run_dir.relative_to(root)),
                    "latest_summary_path": str(summary_path.relative_to(root)),
                    "latest_status": "ok",
                },
            )
            self._write_json(resolved_config_path, {"unit": unit, "embed_dim": EMBED_DIM})
            self._write_json(source_table_path, {"unit": unit, "sources": [AUX, TARGET]})

            stage_specific: dict = {}
            if unit == "a":
                stage_specific = {
                    "merged_task_outputs": {
                        str(TASK_ID): {
                            "source_target": [AUX, TARGET],
                            "aug_train_csv": str(aug_train.resolve()),
                            "aug_valid_csv": str(aug_valid.resolve()),
                            "current_headers": {
                                "aug_train_csv": self._header_meta(aug_train),
                                "aug_valid_csv": self._header_meta(aug_valid),
                            },
                        }
                    }
                }
            elif unit == "b":
                stage_specific = {
                    "profile_output_paths": {
                        domain: {key: str(path.resolve()) for key, path in paths.items()}
                        for domain, paths in profile_paths.items()
                    },
                    "source_csv_fingerprints": source_csv_fps,
                    "expected_shape_dtype": {"rank": 2, "dtype": "float32", "shape_label": "[entity_count, env.embed_dim]"},
                }
            else:
                stage_specific = {
                    "domain_output_paths": {
                        domain: {key: str(path.resolve()) for key, path in paths.items()}
                        for domain, paths in domain_paths.items()
                    },
                    "source_profile_fingerprints": profile_fps,
                    "source_csv_fingerprints": source_csv_fps,
                    "expected_shape_dtype": {"rank": 1, "dtype": "float32", "shape_label": "[env.embed_dim]"},
                }

            fingerprint = f"fp-{unit}"
            manifest = {
                "fingerprint_hash": fingerprint,
                "metadata": {
                    "metadata_schema_version": "odcr_preprocess_metadata/1.0",
                    "contract_version": "odcr_preprocess_contract/3.1",
                    "run_id": "1",
                    "stage": f"preprocess_{unit}",
                    "stage_unit": unit,
                    "latest_path": str(latest_path.resolve()),
                    "run_summary_path": str(summary_path.resolve()),
                    "stage_manifest_path": str(manifest_path.resolve()),
                    "stage_status_path": str(status_path.resolve()),
                    "source_table_path": str(source_table_path.resolve()),
                    "resolved_config_path": str(resolved_config_path.resolve()),
                    "stage_specific": stage_specific,
                },
            }
            self._write_json(manifest_path, manifest)
            self._write_json(status_path, {"stage": f"preprocess_{unit}", "status": "ok", "fingerprint_hash": fingerprint})

            metrics_payload = {"unit": unit, "status": "ok"}
            verify_payload: dict = {"unit": unit, "artifacts": []}
            if unit == "b":
                for domain, paths in profile_paths.items():
                    for key, path in paths.items():
                        spec = key.removesuffix("_profiles")
                        entity_kind = "user" if key.startswith("user_") else "item"
                        verify_payload["artifacts"].append(
                            {
                                "path": str(path.resolve()),
                                "exists": True,
                                "shape": [2, EMBED_DIM],
                                "expected_shape": [2, EMBED_DIM],
                                "dtype": "float32",
                                "expected_dtype": "float32",
                                "finite_sample_count": EMBED_DIM,
                                "nonzero_sample_count": EMBED_DIM,
                                "verify_sample_count": EMBED_DIM,
                                "status": "pass",
                                "dataset": domain,
                                "spec": spec,
                                "entity_kind": entity_kind,
                                "expected_shape_label": "[entity_count, env.embed_dim]",
                            }
                        )
            elif unit == "c":
                for domain, paths in domain_paths.items():
                    for key, path in paths.items():
                        verify_payload["artifacts"].append(
                            {
                                "path": str(path.resolve()),
                                "exists": True,
                                "shape": [EMBED_DIM],
                                "expected_shape": [EMBED_DIM],
                                "dtype": "float32",
                                "expected_dtype": "float32",
                                "finite_sample_count": EMBED_DIM,
                                "nonzero_sample_count": EMBED_DIM,
                                "verify_sample_count": EMBED_DIM,
                                "status": "pass",
                                "dataset": domain,
                                "domain": "content" if key == "domain_content" else "style",
                                "expected_shape_label": "[env.embed_dim]",
                                "domain_shape_contract_version": DOMAIN_ARTIFACT_CONTRACT_VERSION,
                            }
                        )
            if unit in {"b", "c"}:
                self._write_json(metrics_path, metrics_payload)
                self._write_json(verify_path, verify_payload)

            summary = {
                "run_id": "1",
                "stage": "preprocess",
                "unit": unit,
                "status": "ok",
                "run_dir": str(run_dir.resolve()),
                "meta_dir": str(meta_dir.resolve()),
                "resolved_config_path": str(resolved_config_path.resolve()),
                "source_table_path": str(source_table_path.resolve()),
                "manifest_path": str(manifest_path.resolve()),
                "lineage_path": str(status_path.resolve()),
                "metrics_path": str(metrics_path.resolve()) if unit in {"b", "c"} else None,
                "verify_report_path": str(verify_path.resolve()) if unit in {"b", "c"} else None,
                "validation_status": "ok",
                "fingerprint_hash": fingerprint,
            }
            self._write_json(summary_path, summary)

        return {"data": data_dir, "merged": merged_dir, "runs": runs_dir}

    def _validate_fixture(self, root: Path) -> dict:
        paths = self._valid_repo(root)
        return validate_step3_preprocess_upstream_gate(
            repo_root=root,
            task_id=TASK_ID,
            auxiliary_domain=AUX,
            target_domain=TARGET,
            data_dir=paths["data"],
            merged_dir=paths["merged"],
            runs_dir=paths["runs"],
            embed_dim=EMBED_DIM,
        )

    def test_current_preprocess_latest_artifacts_pass_gate(self) -> None:
        summary = validate_step3_preprocess_upstream_gate(
            repo_root=REPO_ROOT,
            task_id=TASK_ID,
            auxiliary_domain=AUX,
            target_domain=TARGET,
            data_dir=REPO_ROOT / "data",
            merged_dir=REPO_ROOT / "merged",
            runs_dir=REPO_ROOT / "runs",
            embed_dim=1024,
        )
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["preprocess"]["a"]["run_id"], "1")
        self.assertEqual(summary["env"]["embed_dim"], 1024)
        self.assertIn(AUX, summary["profile_artifact_fingerprints"])
        self.assertIn(TARGET, summary["domain_artifact_fingerprints"])

    def test_valid_fixture_passes_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._validate_fixture(Path(tmp))
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["preprocess"]["b"]["run_id"], "1")
        self.assertEqual(summary["domain_artifacts"][AUX]["domain_content"]["shape"], [EMBED_DIM])

    def _expect_failure_after_mutation(self, mutate, pattern: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._valid_repo(root)
            mutate(root, paths)
            with self.assertRaisesRegex(Step3UpstreamGateError, pattern):
                validate_step3_preprocess_upstream_gate(
                    repo_root=root,
                    task_id=TASK_ID,
                    auxiliary_domain=AUX,
                    target_domain=TARGET,
                    data_dir=paths["data"],
                    merged_dir=paths["merged"],
                    runs_dir=paths["runs"],
                    embed_dim=EMBED_DIM,
                )

    def test_missing_latest_fails(self) -> None:
        self._expect_failure_after_mutation(
            lambda root, paths: (paths["runs"] / "preprocess" / "b" / "latest.json").unlink(),
            "latest.json.*missing",
        )

    def test_latest_points_to_missing_run_fails(self) -> None:
        def mutate(root: Path, paths: dict[str, Path]) -> None:
            latest = paths["runs"] / "preprocess" / "b" / "latest.json"
            payload = json.loads(latest.read_text(encoding="utf-8"))
            payload["latest_run_id"] = "999"
            payload["latest_run_dir"] = "runs/preprocess/b/999"
            payload["latest_summary_path"] = "runs/preprocess/b/999/meta/run_summary.json"
            self._write_json(latest, payload)

        self._expect_failure_after_mutation(mutate, "run_summary.json.*missing")

    def test_run_summary_status_inconsistent_fails(self) -> None:
        def mutate(root: Path, paths: dict[str, Path]) -> None:
            summary = paths["runs"] / "preprocess" / "b" / "1" / "meta" / "run_summary.json"
            payload = json.loads(summary.read_text(encoding="utf-8"))
            payload["status"] = "failed"
            self._write_json(summary, payload)

        self._expect_failure_after_mutation(mutate, "status must be ok")

    def test_missing_manifest_fails(self) -> None:
        self._expect_failure_after_mutation(
            lambda root, paths: (paths["runs"] / "preprocess" / "b" / "1" / "meta" / "stage_manifest.json").unlink(),
            "stage_manifest.json.*missing",
        )

    def test_missing_metrics_or_verify_fails(self) -> None:
        self._expect_failure_after_mutation(
            lambda root, paths: (paths["runs"] / "preprocess" / "c" / "1" / "meta" / "verify_report.json").unlink(),
            "verify_report.json.*missing",
        )

    def test_preprocess_c_domain_vector_rank2_fails(self) -> None:
        def mutate(root: Path, paths: dict[str, Path]) -> None:
            self._write_npy(paths["data"] / AUX / "domain_content.npy", np.ones((2, EMBED_DIM), dtype=np.float32))

        self._expect_failure_after_mutation(mutate, "rank=2.*retired preprocess_c")

    def test_profile_dtype_mismatch_fails(self) -> None:
        def mutate(root: Path, paths: dict[str, Path]) -> None:
            self._write_npy(paths["data"] / AUX / "user_content_profiles.npy", np.ones((2, EMBED_DIM), dtype=np.float64))

        self._expect_failure_after_mutation(mutate, "dtype mismatch")

    def test_embed_dim_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._valid_repo(root)
            with self.assertRaisesRegex(Step3UpstreamGateError, "embed_dim mismatch"):
                validate_step3_preprocess_upstream_gate(
                    repo_root=root,
                    task_id=TASK_ID,
                    auxiliary_domain=AUX,
                    target_domain=TARGET,
                    data_dir=paths["data"],
                    merged_dir=paths["merged"],
                    runs_dir=paths["runs"],
                    embed_dim=EMBED_DIM + 1,
                )

    def test_fingerprint_mismatch_fails(self) -> None:
        def mutate(root: Path, paths: dict[str, Path]) -> None:
            manifest_path = paths["runs"] / "preprocess" / "c" / "1" / "meta" / "stage_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            fp = copy.deepcopy(payload["metadata"]["stage_specific"]["source_profile_fingerprints"][AUX]["user_content_profiles"])
            fp["sample_sha256"] = "bad"
            payload["metadata"]["stage_specific"]["source_profile_fingerprints"][AUX]["user_content_profiles"] = fp
            self._write_json(manifest_path, payload)

        self._expect_failure_after_mutation(mutate, "fingerprint mismatch")

    def test_history_ai_analysis_completed_stamp_cannot_satisfy_gate(self) -> None:
        def mutate(root: Path, paths: dict[str, Path]) -> None:
            stamp = root / "AI_analysis" / "history" / "completed.stamp"
            stamp.parent.mkdir(parents=True, exist_ok=True)
            stamp.write_text("done\n", encoding="utf-8")
            latest = paths["runs"] / "preprocess" / "b" / "latest.json"
            payload = json.loads(latest.read_text(encoding="utf-8"))
            payload["latest_summary_path"] = str(stamp)
            self._write_json(latest, payload)

        self._expect_failure_after_mutation(mutate, "forbidden non-formal evidence path")

if __name__ == "__main__":
    unittest.main()
