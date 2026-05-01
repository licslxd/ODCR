from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from odcr_core import run_naming
from odcr_core.config_resolver import load_yaml_config

Step3Mode = Literal["full", "train_only", "eval_only"]

VALID_STEP3_MODES: tuple[Step3Mode, ...] = ("full", "train_only", "eval_only")
STEP3_ALLOWED_CLI_OVERRIDES: tuple[str, ...] = (
    "task",
    "mode",
    "run_id",
    "run_name",
    "iter",
    "hardware_preset",
    "epochs",
    "num_proc",
    "ddp_world_size",
    "seed",
    "omp_num_threads",
    "mkl_num_threads",
    "tokenizers_parallelism",
    "cuda_visible_devices",
)
STEP3_BLOCKED_SHARED_FLAGS: tuple[str, ...] = (
    "--from-run",
    "--step4-run",
    "--step5-run",
    "--train-csv",
    "--model-path",
    "--eta",
)

_BLOCKED_FLAG_ATTRS: dict[str, str] = {
    "from_run": "--from-run",
    "step4_run": "--step4-run",
    "step5_run": "--step5-run",
    "train_csv": "--train-csv",
    "model_path": "--model-path",
    "eta": "--eta",
}


@dataclass(frozen=True)
class Step3TaskBinding:
    task_id: int
    auxiliary: str
    target: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Step3RunIdentity:
    iteration_id: str = "v1"
    requested_run_id: str | None = None
    requested_run_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Step3SourceRefs:
    upstream_contract: str = "canonical_preprocess_assets"
    from_run: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Step3HardwareBinding:
    preset_name: str = "default"
    omp_num_threads: int | None = None
    mkl_num_threads: int | None = None
    tokenizers_parallelism: bool | None = None
    cuda_visible_devices: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Step3ExecutionOverrides:
    epochs: int | None = None
    num_proc: int | None = None
    ddp_world_size: int | None = None
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Step3BackendBridge:
    training_preset_stem: str
    hardware_preset_stem: str
    uses_shared_task_table: bool = True
    uses_shared_training_loader: bool = True
    uses_shared_hardware_loader: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Step3Config:
    preset_name: str
    description: str
    task: Step3TaskBinding
    mode: Step3Mode
    run: Step3RunIdentity
    source_refs: Step3SourceRefs
    hardware: Step3HardwareBinding
    overrides: Step3ExecutionOverrides
    bridge: Step3BackendBridge
    stage: Literal["step3"] = "step3"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_step3_task_binding(task_id: int) -> Step3TaskBinding:
    raw_config = load_yaml_config("configs/odcr.yaml")
    task_block = raw_config.get("tasks", {})
    if not isinstance(task_block, dict):
        raise ValueError("configs/odcr.yaml tasks block must be a mapping.")
    task_table: dict[int, dict[str, Any]] = {}
    for raw_key, raw_row in task_block.items():
        if not isinstance(raw_row, dict):
            continue
        tid = int(raw_key)
        task_table[tid] = {
            "auxiliary": raw_row.get("source", raw_row.get("auxiliary")),
            "target": raw_row.get("target"),
        }
    try:
        row = task_table[int(task_id)]
    except KeyError as exc:
        raise KeyError(f"Unknown Step3 task_id={task_id!r}; expected one of 1..8.") from exc
    return Step3TaskBinding(
        task_id=int(task_id),
        auxiliary=str(row["auxiliary"]),
        target=str(row["target"]),
    )


def _normalize_optional_slug(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    return run_naming.parse_run_id(s)


def _normalize_optional_positive_int(raw: int | str | None, *, field_name: str) -> int | None:
    if raw is None:
        return None
    value = int(raw)
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1; got {value!r}")
    return value


def _normalize_optional_tokenizers_parallelism(raw: bool | str | None) -> bool | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    if s not in ("true", "false"):
        raise ValueError("tokenizers_parallelism must be true/false when specified.")
    return s == "true"


def _normalize_optional_cuda_visible_devices(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    return s or None


def _reject_blocked_shared_cli_args(args: Any) -> None:
    blocked: list[str] = []
    for attr, flag in _BLOCKED_FLAG_ATTRS.items():
        value = getattr(args, attr, None)
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                blocked.append(flag)
            continue
        if str(value).strip() != "":
            blocked.append(flag)
    if blocked:
        raise ValueError(
            "Step3 typed control plane does not expose shared-only flags: "
            f"{', '.join(sorted(blocked))}. "
            "Use `odcr.py step4/step5/eval` for lineage-bearing downstream stages."
        )


def resolve_step3_mode_from_args(args: Any, *, default: Step3Mode = "full") -> Step3Mode:
    cli_mode = getattr(args, "mode", None)
    eval_only = bool(getattr(args, "eval_only", False))
    train_only = bool(getattr(args, "train_only", False))
    compat_mode: Step3Mode | None = None
    if eval_only and train_only:
        raise ValueError("Step3 cannot use --eval-only and --train-only together.")
    if eval_only:
        compat_mode = "eval_only"
    elif train_only:
        compat_mode = "train_only"
    if cli_mode is not None:
        s = str(cli_mode).strip().lower()
        if s not in VALID_STEP3_MODES:
            raise ValueError(f"Unknown Step3 mode {cli_mode!r}; expected one of {list(VALID_STEP3_MODES)}")
        typed_mode = s  # type: ignore[assignment]
        if compat_mode is not None and typed_mode != compat_mode:
            raise ValueError(
                f"Step3 mode conflict: --mode={typed_mode!r} but compatibility flag requested {compat_mode!r}."
            )
        return typed_mode
    if compat_mode is not None:
        return compat_mode
    return default


def validate_step3_config(config: Step3Config) -> Step3Config:
    if not str(config.preset_name).strip():
        raise ValueError("Step3 preset_name must be non-empty.")
    if not str(config.description).strip():
        raise ValueError("Step3 description must be non-empty.")
    if config.mode not in VALID_STEP3_MODES:
        raise ValueError(f"Invalid Step3 mode {config.mode!r}; expected one of {list(VALID_STEP3_MODES)}")

    expected_task = resolve_step3_task_binding(int(config.task.task_id))
    if config.task != expected_task:
        raise ValueError(
            "Step3 task binding drifted from shared task table; "
            f"expected {expected_task.to_dict()} but got {config.task.to_dict()}."
        )

    normalized_iteration = run_naming.normalize_iteration_id(config.run.iteration_id)
    normalized_run_id = _normalize_optional_slug(config.run.requested_run_id)
    normalized_run_name = _normalize_optional_slug(config.run.requested_run_name)
    if normalized_run_id and normalized_run_name and normalized_run_id != normalized_run_name:
        raise ValueError(
            "Step3 --run-id and --run-name must agree when both are provided; "
            f"got {normalized_run_id!r} vs {normalized_run_name!r}."
        )

    if config.source_refs.from_run is not None:
        raise ValueError(
            "Step3 canonical typed control plane does not currently consume from_run; "
            "upstream lineage begins at canonical preprocess assets."
        )
    if not str(config.source_refs.upstream_contract).strip():
        raise ValueError("Step3 source_refs.upstream_contract must be non-empty.")

    hardware_preset = str(config.hardware.preset_name).strip()
    if not hardware_preset:
        raise ValueError("Step3 hardware preset must be non-empty.")
    omp_threads = _normalize_optional_positive_int(
        config.hardware.omp_num_threads,
        field_name="omp_num_threads",
    )
    mkl_threads = _normalize_optional_positive_int(
        config.hardware.mkl_num_threads,
        field_name="mkl_num_threads",
    )
    tok_parallel = _normalize_optional_tokenizers_parallelism(config.hardware.tokenizers_parallelism)
    cuda_visible_devices = _normalize_optional_cuda_visible_devices(config.hardware.cuda_visible_devices)

    epochs = _normalize_optional_positive_int(config.overrides.epochs, field_name="epochs")
    num_proc = _normalize_optional_positive_int(config.overrides.num_proc, field_name="num_proc")
    ddp_world_size = _normalize_optional_positive_int(
        config.overrides.ddp_world_size,
        field_name="ddp_world_size",
    )
    seed = _normalize_optional_positive_int(config.overrides.seed, field_name="seed")

    training_preset_stem = str(config.bridge.training_preset_stem).strip()
    if not training_preset_stem:
        raise ValueError("Step3 backend bridge must declare training_preset_stem.")
    if str(config.bridge.hardware_preset_stem).strip() != hardware_preset:
        raise ValueError(
            "Step3 backend bridge hardware_preset_stem must match the effective hardware binding."
        )

    return replace(
        config,
        run=replace(
            config.run,
            iteration_id=normalized_iteration,
            requested_run_id=normalized_run_id,
            requested_run_name=normalized_run_name,
        ),
        hardware=replace(
            config.hardware,
            preset_name=hardware_preset,
            omp_num_threads=omp_threads,
            mkl_num_threads=mkl_threads,
            tokenizers_parallelism=tok_parallel,
            cuda_visible_devices=cuda_visible_devices,
        ),
        overrides=replace(
            config.overrides,
            epochs=epochs,
            num_proc=num_proc,
            ddp_world_size=ddp_world_size,
            seed=seed,
        ),
        bridge=replace(
            config.bridge,
            training_preset_stem=training_preset_stem,
            hardware_preset_stem=hardware_preset,
        ),
    )


def apply_step3_cli_overrides(config: Step3Config, args: Any) -> Step3Config:
    _reject_blocked_shared_cli_args(args)

    updated = config
    cli_task = getattr(args, "task", None)
    if cli_task is not None and int(cli_task) != config.task.task_id:
        updated = replace(updated, task=resolve_step3_task_binding(int(cli_task)))

    mode = resolve_step3_mode_from_args(args, default=updated.mode)
    updated = replace(updated, mode=mode)

    cli_iteration_id = getattr(args, "iteration_id", None)
    cli_run_id = getattr(args, "run_id", None)
    cli_run_name = getattr(args, "run_name", None)
    updated = replace(
        updated,
        run=replace(
            updated.run,
            iteration_id=str(cli_iteration_id).strip() if cli_iteration_id is not None else updated.run.iteration_id,
            requested_run_id=(
                None
                if cli_run_id is None or str(cli_run_id).strip().lower() in ("", "auto")
                else str(cli_run_id).strip()
            ),
            requested_run_name=(
                None
                if cli_run_name is None or str(cli_run_name).strip() == ""
                else str(cli_run_name).strip()
            ),
        ),
    )

    hardware_preset = (
        str(getattr(args, "hardware_preset", "")).strip()
        if getattr(args, "hardware_preset", None) is not None
        else updated.hardware.preset_name
    )
    tokenizers_parallelism = (
        _normalize_optional_tokenizers_parallelism(getattr(args, "tokenizers_parallelism", None))
        if getattr(args, "tokenizers_parallelism", None) is not None
        else updated.hardware.tokenizers_parallelism
    )
    updated = replace(
        updated,
        hardware=replace(
            updated.hardware,
            preset_name=hardware_preset,
            omp_num_threads=getattr(args, "omp_num_threads", None),
            mkl_num_threads=getattr(args, "mkl_num_threads", None),
            tokenizers_parallelism=tokenizers_parallelism,
            cuda_visible_devices=getattr(args, "cuda_visible_devices", None),
        ),
        overrides=replace(
            updated.overrides,
            epochs=getattr(args, "epochs", None),
            num_proc=getattr(args, "num_proc", None),
            ddp_world_size=getattr(args, "ddp_world_size", None),
            seed=getattr(args, "seed", None),
        ),
        bridge=replace(
            updated.bridge,
            hardware_preset_stem=hardware_preset,
        ),
    )
    return validate_step3_config(updated)
