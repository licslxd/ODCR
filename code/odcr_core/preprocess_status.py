from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from odcr_core.file_atomic import atomic_write_json, atomic_write_text

UnitKind = Literal["dataset", "task"]
UnitState = Literal["pending", "running", "ok", "failed", "skipped"]
StageState = Literal["pending", "running", "ok", "failed"]


@dataclass(frozen=True)
class ResolvedPreprocessStagePaths:
    repo_root: str
    meta_root: str
    shell_log_dir: str
    stage_log_path: str
    console_log_path: str
    full_log_path: str
    errors_log_path: str
    stage_manifest_path: str
    stage_status_path: str
    completed_stamp_path: str
    datasets_status_dir: str
    tasks_status_dir: str


@dataclass(frozen=True)
class PreprocessUnitStatus:
    stage: str
    preset: str
    unit_kind: UnitKind
    unit_name: str
    status: UnitState
    started_at: str | None
    finished_at: str | None
    shell_log_path: str | None
    output_files: tuple[str, ...]
    reason: str | None = None
    error_message: str | None = None
    worker_id: int | None = None
    gpu_id: int | None = None
    command: tuple[str, ...] | None = None
    fingerprint: dict[str, Any] | None = None
    fingerprint_hash: str | None = None
    current_headers: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreprocessWorkerResult:
    worker_id: int
    gpu_id: int | None
    exit_code: int
    handled_units: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreprocessStageManifest:
    stage: str
    preset: str
    description: str
    stage_label: str
    started_at: str
    datasets: tuple[str, ...]
    combine_task_ids: tuple[int, ...]
    config_snapshot: dict[str, Any]
    contract_snapshot: dict[str, Any]
    fingerprint: dict[str, Any]
    fingerprint_hash: str
    paths: dict[str, str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreprocessStageStatus:
    stage: str
    preset: str
    status: StageState
    started_at: str
    finished_at: str | None
    description: str
    stage_label: str
    datasets: tuple[str, ...]
    combine_task_ids: tuple[int, ...]
    dataset_statuses: dict[str, dict[str, Any]]
    task_statuses: dict[str, dict[str, Any]]
    worker_results: tuple[dict[str, Any], ...]
    config_snapshot: dict[str, Any]
    contract_snapshot: dict[str, Any]
    fingerprint: dict[str, Any]
    fingerprint_hash: str
    paths: dict[str, str]
    metadata: dict[str, Any]
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_preprocess_stage_paths(
    *,
    repo_root: str,
    meta_root: str,
    shell_log_dir: str,
    stage: str,
    timestamp_tag: str,
) -> ResolvedPreprocessStagePaths:
    meta_root_path = Path(meta_root if Path(meta_root).is_absolute() else Path(repo_root) / meta_root)
    shell_log_dir_path = Path(
        shell_log_dir if Path(shell_log_dir).is_absolute() else Path(repo_root) / shell_log_dir
    )
    return ResolvedPreprocessStagePaths(
        repo_root=str(Path(repo_root).resolve()),
        meta_root=str(meta_root_path.resolve()),
        shell_log_dir=str(shell_log_dir_path.resolve()),
        stage_log_path=str((meta_root_path / "full.log").resolve()),
        console_log_path=str((meta_root_path / "console.log").resolve()),
        full_log_path=str((meta_root_path / "full.log").resolve()),
        errors_log_path=str((meta_root_path / "errors.log").resolve()),
        stage_manifest_path=str((meta_root_path / "stage_manifest.json").resolve()),
        stage_status_path=str((meta_root_path / "stage_status.json").resolve()),
        completed_stamp_path=str((meta_root_path / "completed.stamp").resolve()),
        datasets_status_dir=str((meta_root_path / "datasets").resolve()),
        tasks_status_dir=str((meta_root_path / "tasks").resolve()),
    )


def ensure_preprocess_stage_dirs(paths: ResolvedPreprocessStagePaths) -> None:
    Path(paths.meta_root).mkdir(parents=True, exist_ok=True)
    Path(paths.shell_log_dir).mkdir(parents=True, exist_ok=True)
    Path(paths.datasets_status_dir).mkdir(parents=True, exist_ok=True)
    Path(paths.tasks_status_dir).mkdir(parents=True, exist_ok=True)


def preprocess_unit_status_path(
    paths: ResolvedPreprocessStagePaths,
    *,
    unit_kind: UnitKind,
    unit_name: str,
) -> str:
    root = paths.datasets_status_dir if unit_kind == "dataset" else paths.tasks_status_dir
    safe_name = str(unit_name)
    return str((Path(root) / f"{safe_name}.status.json").resolve())


def write_preprocess_unit_status(path: str | Path, payload: PreprocessUnitStatus) -> None:
    atomic_write_json(path, payload.to_dict())


def read_preprocess_unit_status(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.is_file():
        return None
    import json

    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_preprocess_stage_manifest(path: str | Path, payload: PreprocessStageManifest) -> None:
    atomic_write_json(path, payload.to_dict())


def write_preprocess_stage_status(path: str | Path, payload: PreprocessStageStatus) -> None:
    atomic_write_json(path, payload.to_dict())


def write_preprocess_completed_stamp(
    path: str | Path,
    *,
    stage: str,
    preset: str,
    started_at: str,
    finished_at: str,
) -> None:
    atomic_write_text(
        path,
        "\n".join(
            (
                f"stage={stage}",
                f"preset={preset}",
                f"started_at={started_at}",
                f"finished_at={finished_at}",
            )
        ),
    )
