"""Single writer API for ODCR AI_analysis artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "odcr_ai_analysis_writer/1"
BUCKETS: dict[str, str] = {
    "index": "00_index",
    "raw_log": "01_raw_logs",
    "search_hit": "02_search_hits",
    "ledger": "03_evidence_ledgers",
    "phase_summary": "04_phase_summaries",
    "final_report": "05_final_reports",
}


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[4]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_name(name: str) -> str:
    raw = str(name or "").strip().replace("\\", "/")
    if not raw or raw.startswith("/") or ".." in Path(raw).parts:
        raise ValueError(f"unsafe AI_analysis artifact name: {name!r}")
    return raw


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AIAnalysisWriteResult:
    path: Path
    sha256: str
    bucket: str

    def to_dict(self, *, repo_root: Path | None = None) -> dict[str, str]:
        root = repo_root or _repo_root_from_here()
        try:
            rel = self.path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            rel = str(self.path)
        return {"path": rel, "sha256": self.sha256, "bucket": self.bucket}


class AIAnalysisWriter:
    """Write compact AI_analysis artifacts with consistent metadata."""

    def __init__(self, repo_root: str | Path | None = None) -> None:
        self.repo_root = Path(repo_root).resolve() if repo_root else _repo_root_from_here()
        self.root = self.repo_root / "AI_analysis"

    def bucket_dir(self, bucket: str) -> Path:
        key = str(bucket)
        if key not in BUCKETS:
            raise ValueError(f"unknown AI_analysis bucket {bucket!r}")
        path = self.root / BUCKETS[key]
        path.mkdir(parents=True, exist_ok=True)
        return path

    def metadata(
        self,
        *,
        source: str,
        stage: str | None = None,
        task: int | str | None = None,
        run_id: str | None = None,
        inputs: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
        validation_result: Mapping[str, Any] | str | None = None,
        warnings: Sequence[str] | None = None,
        errors: Sequence[str] | None = None,
        command: Sequence[str] | str | None = None,
    ) -> dict[str, Any]:
        if command is None:
            command_text = shlex.join(sys.argv)
        elif isinstance(command, str):
            command_text = command
        else:
            command_text = shlex.join([str(part) for part in command])
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": _utc_now(),
            "repo_root": str(self.repo_root),
            "cwd": os.getcwd(),
            "command": command_text,
            "source": str(source),
            "stage": None if stage is None else str(stage),
            "task": None if task is None else str(task),
            "run_id": run_id,
            "inputs": dict(inputs or {}),
            "outputs": dict(outputs or {}),
            "validation_result": validation_result,
            "warnings": list(warnings or []),
            "errors": list(errors or []),
        }

    def _write_text(self, bucket: str, name: str, text: str) -> AIAnalysisWriteResult:
        path = self.bucket_dir(bucket) / _safe_name(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
        return AIAnalysisWriteResult(path=path, sha256=_sha256_text(text), bucket=bucket)

    def write_text(
        self,
        bucket: str,
        name: str,
        body: str,
        *,
        source: str,
        stage: str | None = None,
        task: int | str | None = None,
        run_id: str | None = None,
        inputs: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
        validation_result: Mapping[str, Any] | str | None = None,
        warnings: Sequence[str] | None = None,
        errors: Sequence[str] | None = None,
        command: Sequence[str] | str | None = None,
        include_header: bool = True,
    ) -> AIAnalysisWriteResult:
        meta = self.metadata(
            source=source,
            stage=stage,
            task=task,
            run_id=run_id,
            inputs=inputs,
            outputs=outputs,
            validation_result=validation_result,
            warnings=warnings,
            errors=errors,
            command=command,
        )
        if include_header:
            text = "<!-- " + json.dumps(meta, ensure_ascii=False, sort_keys=True, default=str) + " -->\n" + body.rstrip() + "\n"
        else:
            text = body.rstrip() + "\n"
        result = self._write_text(bucket, name, text)
        index_payload = {**meta, "artifact": result.to_dict(repo_root=self.repo_root), "sha256": result.sha256}
        index_name = Path(name).with_suffix(".json").name
        self._write_text("index", index_name, json.dumps(index_payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")
        return result

    def write_json(
        self,
        bucket: str,
        name: str,
        payload: Mapping[str, Any],
        *,
        source: str,
        stage: str | None = None,
        task: int | str | None = None,
        run_id: str | None = None,
        inputs: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
        validation_result: Mapping[str, Any] | str | None = None,
        warnings: Sequence[str] | None = None,
        errors: Sequence[str] | None = None,
        command: Sequence[str] | str | None = None,
    ) -> AIAnalysisWriteResult:
        meta = self.metadata(
            source=source,
            stage=stage,
            task=task,
            run_id=run_id,
            inputs=inputs,
            outputs=outputs,
            validation_result=validation_result,
            warnings=warnings,
            errors=errors,
            command=command,
        )
        data = {**meta, "payload": dict(payload)}
        text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
        result = self._write_text(bucket, name, text)
        self._write_text("index", Path(name).with_suffix(".json").name, text)
        return result

    def raw_log(self, name: str, body: str, **kwargs: Any) -> AIAnalysisWriteResult:
        return self.write_text("raw_log", name, body, include_header=True, **kwargs)

    def search_hit(self, name: str, body: str, **kwargs: Any) -> AIAnalysisWriteResult:
        return self.write_text("search_hit", name, body, **kwargs)

    def ledger(self, name: str, body: str, **kwargs: Any) -> AIAnalysisWriteResult:
        return self.write_text("ledger", name, body, **kwargs)

    def phase_summary(self, name: str, body: str, **kwargs: Any) -> AIAnalysisWriteResult:
        return self.write_text("phase_summary", name, body, **kwargs)

    def final_report(self, name: str, body: str, **kwargs: Any) -> AIAnalysisWriteResult:
        return self.write_text("final_report", name, body, **kwargs)

    def runtime_diagnostic(self, name: str, payload: Mapping[str, Any], **kwargs: Any) -> AIAnalysisWriteResult:
        meta = self.metadata(**kwargs)
        diagnostic_payload = dict(payload)
        data = {**diagnostic_payload, **meta, "payload": diagnostic_payload}
        if "schema_version" in diagnostic_payload:
            data["payload_schema_version"] = diagnostic_payload["schema_version"]
        if "generated_at" in diagnostic_payload:
            data["payload_generated_at"] = diagnostic_payload["generated_at"]
        text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
        result = self._write_text("raw_log", name, text)
        self._write_text("index", Path(name).with_suffix(".json").name, text)
        return result

    def compact_audit(self, name: str, body: str, **kwargs: Any) -> AIAnalysisWriteResult:
        return self.write_text("ledger", name, body, **kwargs)


def get_writer(repo_root: str | Path | None = None) -> AIAnalysisWriter:
    return AIAnalysisWriter(repo_root=repo_root)
