from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from data_contract import (
    CONTENT_PROFILE_TEXT_COLUMNS,
    DOMAIN_CONTENT_TEXT_COLUMNS,
    DOMAIN_STYLE_TEXT_COLUMNS,
    PREPROCESS_CONTRACT_VERSION,
    STYLE_PROFILE_TEXT_COLUMNS,
    render_preprocess_contract_snapshot,
)

PreprocessStage = Literal["preprocess_a", "preprocess_b", "preprocess_c"]

VALID_PREPROCESS_STAGES: tuple[PreprocessStage, ...] = (
    "preprocess_a",
    "preprocess_b",
    "preprocess_c",
)
VALID_DATASETS: tuple[str, ...] = (
    "AM_Movies",
    "AM_Electronics",
    "AM_CDs",
    "TripAdvisor",
    "Yelp",
)
COMBINE_TASK_MAP: tuple[tuple[int, str, str], ...] = (
    (1, "AM_Electronics", "AM_CDs"),
    (2, "AM_Movies", "AM_CDs"),
    (3, "AM_CDs", "AM_Electronics"),
    (4, "AM_Movies", "AM_Electronics"),
    (5, "AM_CDs", "AM_Movies"),
    (6, "AM_Electronics", "AM_Movies"),
    (7, "Yelp", "TripAdvisor"),
    (8, "TripAdvisor", "Yelp"),
)


@dataclass(frozen=True)
class PreprocessStageContract:
    stage: PreprocessStage
    producer_scripts: tuple[str, ...]
    output_files: tuple[str, ...]
    consumers: tuple[str, ...]
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PREPROCESS_STAGE_CONTRACTS: dict[PreprocessStage, PreprocessStageContract] = {
    "preprocess_a": PreprocessStageContract(
        stage="preprocess_a",
        producer_scripts=("code/preprocess_data.py", "code/split_data.py", "code/combine_data.py"),
        output_files=("processed.csv", "train.csv", "valid.csv", "test.csv", "aug_train.csv", "aug_valid.csv"),
        consumers=("preprocess_b", "preprocess_c", "step3", "step4", "step5"),
        description="Materialize the canonical preprocess CSV contract and keep split/combine transport lossless.",
    ),
    "preprocess_b": PreprocessStageContract(
        stage="preprocess_b",
        producer_scripts=("code/compute_embeddings.py",),
        output_files=(
            "user_content_profiles.npy",
            "user_style_profiles.npy",
            "item_content_profiles.npy",
            "item_style_profiles.npy",
        ),
        consumers=("step3", "step4", "step5"),
        description="Build dual-channel user/item profiles directly from canonical content/style preprocess assets.",
    ),
    "preprocess_c": PreprocessStageContract(
        stage="preprocess_c",
        producer_scripts=("code/infer_domain_semantics.py",),
        output_files=("domain_content.npy", "domain_style.npy"),
        consumers=("step3", "step4", "step5"),
        description="Build dual-channel domain semantics directly from canonical content/style preprocess assets.",
    ),
}


def render_preprocess_stage_contract(stage: PreprocessStage) -> dict[str, Any]:
    contract = PREPROCESS_STAGE_CONTRACTS[stage].to_dict()
    contract["preprocess_contract_version"] = PREPROCESS_CONTRACT_VERSION
    if stage == "preprocess_a":
        contract["processed_csv_contract"] = render_preprocess_contract_snapshot()
        contract["split_csv_contract"] = render_preprocess_contract_snapshot(require_split_indices=True)
        contract["merged_csv_contract"] = render_preprocess_contract_snapshot(
            require_split_indices=True,
            require_domain=True,
        )
    elif stage == "preprocess_b":
        contract["content_channel_text_sources"] = list(CONTENT_PROFILE_TEXT_COLUMNS)
        contract["style_channel_text_sources"] = list(STYLE_PROFILE_TEXT_COLUMNS)
    else:
        contract["content_channel_text_sources"] = list(DOMAIN_CONTENT_TEXT_COLUMNS)
        contract["style_channel_text_sources"] = list(DOMAIN_STYLE_TEXT_COLUMNS)
    return contract


def _dedupe_preserve_order(items: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)


def normalize_dataset_names(raw: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if raw is None:
        return tuple(VALID_DATASETS)
    if isinstance(raw, str):
        tokens = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        tokens = [str(item).strip() for item in raw if str(item).strip()]
    if not tokens:
        return tuple(VALID_DATASETS)
    unknown = [item for item in tokens if item not in VALID_DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}; expected one of {list(VALID_DATASETS)}")
    return _dedupe_preserve_order(tokens)


def normalize_force_datasets(
    raw: str | list[str] | tuple[str, ...] | None,
    *,
    datasets: tuple[str, ...],
) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        tokens = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        tokens = [str(item).strip() for item in raw if str(item).strip()]
    if not tokens:
        return ()
    unknown = [item for item in tokens if item not in VALID_DATASETS]
    if unknown:
        raise ValueError(f"Unknown force_datasets entries: {unknown}")
    invalid = [item for item in tokens if item not in datasets]
    if invalid:
        raise ValueError(f"force_datasets must be a subset of datasets; invalid entries: {invalid}")
    return _dedupe_preserve_order(tokens)


def normalize_gpu_ids(raw: str | list[int] | tuple[int, ...] | None) -> tuple[int, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        tokens = [item.strip() for item in raw.split(",") if item.strip()]
        values: list[int] = []
        for item in tokens:
            if not item.isdigit():
                raise ValueError(f"Invalid GPU id: {item!r}")
            values.append(int(item))
    else:
        values = [int(item) for item in raw]
    if any(item < 0 for item in values):
        raise ValueError(f"GPU ids must be non-negative: {values}")
    out: list[int] = []
    seen: set[int] = set()
    for item in values:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)


def normalize_optional_indices(raw: str | list[int] | tuple[int, ...] | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        tokens = [item.strip() for item in raw.split(",") if item.strip()]
        if not tokens:
            return None
        values: list[int] = []
        for item in tokens:
            if not item.isdigit():
                raise ValueError(f"Expected non-negative integer list, got {raw!r}")
            values.append(int(item))
    else:
        values = [int(item) for item in raw]
    if any(item < 0 for item in values):
        raise ValueError(f"Indices must be non-negative: {values}")
    out: list[int] = []
    seen: set[int] = set()
    for item in values:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out) if out else None


def resolve_combine_task_ids(datasets: tuple[str, ...]) -> tuple[int, ...]:
    selected = set(datasets)
    out: list[int] = []
    for task_id, source, target in COMBINE_TASK_MAP:
        if source in selected and target in selected:
            out.append(task_id)
    if not out:
        raise ValueError(
            "Selected datasets do not cover any source-target pair in the current combine task map."
        )
    return tuple(out)


@dataclass(frozen=True)
class PreprocessPathsConfig:
    meta_root: str
    shell_log_dir: str


@dataclass(frozen=True)
class PreprocessRuntimeOptions:
    python_bin: str = "python"
    resume: bool = True
    skip_completed: bool = True
    verify_only: bool = False
    dry_run: bool = False
    workers: int | None = None
    force_datasets: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreprocessHardwareConfig:
    gpu_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class PreprocessResolvedPayload:
    data_dir: str = ""
    merged_dir: str = ""
    runs_dir: str = ""
    cache_dir: str = ""
    models_dir: str = ""
    step5_text_model: str = ""
    sentence_embed_model: str = ""
    sentence_embed_model_path: str = ""
    embed_dim: int = 0
    offline: bool = True
    local_files_only: bool = True
    gpu_ids: tuple[int, ...] = ()
    bf16: bool = False
    tf32: bool = False
    sources: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PreprocessBaseConfig:
    preset_name: str
    description: str
    datasets: tuple[str, ...]
    paths: PreprocessPathsConfig
    runtime: PreprocessRuntimeOptions
    stage: PreprocessStage
    run_id: str = "1"
    resolved: PreprocessResolvedPayload = PreprocessResolvedPayload()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreprocessAConfig(PreprocessBaseConfig):
    stage: Literal["preprocess_a"] = "preprocess_a"


@dataclass(frozen=True)
class PreprocessBConfig(PreprocessBaseConfig):
    hardware: PreprocessHardwareConfig = PreprocessHardwareConfig()
    embed_batch_size: int = 512
    read_chunk_rows: int = 100_000
    group_shard_size: int = 4_096
    grouped_text_cache_enabled: bool = True
    grouped_text_cache_dir: str = "cache/preprocess_b"
    grouped_text_cache_version: str = "preprocess_b_grouped_text_cache_v1"
    bf16_enabled: bool = True
    tf32_enabled: bool = True
    verify_sample_size: int = 8
    verify_seed: int = 7
    verify_user_indices: tuple[int, ...] | None = None
    verify_item_indices: tuple[int, ...] | None = None
    stage: Literal["preprocess_b"] = "preprocess_b"


@dataclass(frozen=True)
class PreprocessCConfig(PreprocessBaseConfig):
    hardware: PreprocessHardwareConfig = PreprocessHardwareConfig()
    chunk_batch_size: int = 256
    bf16_enabled: bool = True
    tf32_enabled: bool = True
    tokenizer_hotpath_enabled: bool = True
    token_window_cache_enabled: bool = True
    token_window_cache_dir: str = "cache/preprocess_c"
    token_window_cache_version: str = "preprocess_c_token_windows_v3"
    token_window_cache_shard_size: int = 2048
    stage: Literal["preprocess_c"] = "preprocess_c"


PreprocessConfig = PreprocessAConfig | PreprocessBConfig | PreprocessCConfig


def _validate_paths(paths: PreprocessPathsConfig, *, stage: str) -> None:
    if not str(paths.meta_root).strip():
        raise ValueError(f"{stage} requires a non-empty meta_root")
    if not str(paths.shell_log_dir).strip():
        raise ValueError(f"{stage} requires a non-empty shell_log_dir")


def _validate_runtime(
    runtime: PreprocessRuntimeOptions,
    *,
    stage: str,
    datasets: tuple[str, ...],
) -> None:
    if not str(runtime.python_bin).strip():
        raise ValueError(f"{stage} requires a non-empty python_bin")
    if runtime.workers is not None and runtime.workers <= 0:
        raise ValueError(f"{stage} workers must be a positive integer when specified")
    invalid_force = [item for item in runtime.force_datasets if item not in datasets]
    if invalid_force:
        raise ValueError(f"{stage} force_datasets must be a subset of datasets: {invalid_force}")


def validate_preprocess_config(config: PreprocessConfig) -> PreprocessConfig:
    datasets = normalize_dataset_names(config.datasets)
    force_datasets = normalize_force_datasets(config.runtime.force_datasets, datasets=datasets)
    runtime = replace(config.runtime, force_datasets=force_datasets)
    _validate_paths(config.paths, stage=config.stage)
    _validate_runtime(runtime, stage=config.stage, datasets=datasets)

    if config.stage == "preprocess_a":
        if runtime.verify_only:
            raise ValueError("preprocess_a does not support verify_only")
        if runtime.workers not in (None, 1):
            raise ValueError("preprocess_a is currently serial and only supports workers=1")
        resolved = replace(config.resolved, gpu_ids=(), bf16=False, tf32=False)
        return replace(config, datasets=datasets, runtime=replace(runtime, workers=1), resolved=resolved)

    if config.stage == "preprocess_b":
        gpu_ids = normalize_gpu_ids(config.hardware.gpu_ids)
        if not gpu_ids:
            raise ValueError("preprocess_b requires at least one GPU id")
        workers = runtime.workers if runtime.workers is not None else len(gpu_ids)
        if workers > len(gpu_ids):
            raise ValueError("preprocess_b workers cannot exceed the number of gpu_ids")
        if config.embed_batch_size <= 0:
            raise ValueError("preprocess_b embed_batch_size must be positive")
        if config.read_chunk_rows <= 0:
            raise ValueError("preprocess_b read_chunk_rows must be positive")
        if config.group_shard_size <= 0:
            raise ValueError("preprocess_b group_shard_size must be positive")
        if config.grouped_text_cache_enabled and not str(config.grouped_text_cache_dir).strip():
            raise ValueError("preprocess_b grouped_text_cache_dir must be non-empty when cache is enabled")
        if config.grouped_text_cache_enabled and not str(config.grouped_text_cache_version).strip():
            raise ValueError("preprocess_b grouped_text_cache_version must be non-empty when cache is enabled")
        if config.verify_sample_size < 0:
            raise ValueError("preprocess_b verify_sample_size must be non-negative")
        if config.verify_seed < 0:
            raise ValueError("preprocess_b verify_seed must be non-negative")
        return replace(
            config,
            datasets=datasets,
            runtime=replace(runtime, workers=workers),
            hardware=PreprocessHardwareConfig(gpu_ids=gpu_ids),
            resolved=replace(
                config.resolved,
                gpu_ids=gpu_ids,
                bf16=bool(config.bf16_enabled),
                tf32=bool(config.tf32_enabled),
            ),
            verify_user_indices=normalize_optional_indices(config.verify_user_indices),
            verify_item_indices=normalize_optional_indices(config.verify_item_indices),
        )

    if config.stage == "preprocess_c":
        gpu_ids = normalize_gpu_ids(config.hardware.gpu_ids)
        if not gpu_ids:
            raise ValueError("preprocess_c requires at least one GPU id")
        workers = runtime.workers if runtime.workers is not None else len(gpu_ids)
        if workers > len(gpu_ids):
            raise ValueError("preprocess_c workers cannot exceed the number of gpu_ids")
        if config.chunk_batch_size <= 0:
            raise ValueError("preprocess_c chunk_batch_size must be positive")
        if not str(config.token_window_cache_dir).strip():
            raise ValueError("preprocess_c token_window_cache_dir must be non-empty")
        if not str(config.token_window_cache_version).strip():
            raise ValueError("preprocess_c token_window_cache_version must be non-empty")
        if config.token_window_cache_shard_size <= 0:
            raise ValueError("preprocess_c token_window_cache_shard_size must be positive")
        return replace(
            config,
            datasets=datasets,
            runtime=replace(runtime, workers=workers),
            hardware=PreprocessHardwareConfig(gpu_ids=gpu_ids),
            resolved=replace(
                config.resolved,
                gpu_ids=gpu_ids,
                bf16=bool(config.bf16_enabled),
                tf32=bool(config.tf32_enabled),
            ),
        )

    raise ValueError(f"Unknown preprocess stage: {config.stage}")


def apply_preprocess_cli_overrides(config: PreprocessConfig, args: Any) -> PreprocessConfig:
    datasets = normalize_dataset_names(getattr(args, "datasets", None) or config.datasets)
    runtime = config.runtime
    if getattr(args, "resume", None) is not None:
        runtime = replace(runtime, resume=bool(args.resume))
    if getattr(args, "skip_completed", None) is not None:
        runtime = replace(runtime, skip_completed=bool(args.skip_completed))
    if getattr(args, "verify_only", False):
        runtime = replace(runtime, verify_only=True)
    if getattr(args, "dry_run", False):
        runtime = replace(runtime, dry_run=True)
    if getattr(args, "workers", None) is not None:
        runtime = replace(runtime, workers=int(args.workers))
    force_raw = getattr(args, "force_datasets", None)
    if force_raw is not None:
        runtime = replace(
            runtime,
            force_datasets=normalize_force_datasets(force_raw, datasets=datasets),
        )

    if config.stage == "preprocess_a":
        invalid = []
        for flag_name in (
            "gpu_ids",
            "embed_batch_size",
            "read_chunk_rows",
            "group_shard_size",
            "grouped_text_cache_enabled",
            "grouped_text_cache_dir",
            "grouped_text_cache_version",
            "chunk_batch_size",
            "bf16_enabled",
            "tf32_enabled",
            "tokenizer_hotpath_enabled",
            "token_window_cache_enabled",
            "token_window_cache_dir",
            "token_window_cache_version",
            "token_window_cache_shard_size",
            "verify_sample_size",
            "verify_seed",
            "verify_user_indices",
            "verify_item_indices",
        ):
            if getattr(args, flag_name, None) is not None:
                invalid.append(flag_name)
        if invalid:
            raise ValueError(f"preprocess_a does not accept overrides: {invalid}")
        return validate_preprocess_config(replace(config, datasets=datasets, runtime=runtime))

    gpu_ids = normalize_gpu_ids(getattr(args, "gpu_ids", None) or getattr(config.hardware, "gpu_ids", ()))
    if config.stage == "preprocess_b":
        invalid = []
        for flag_name in (
            "chunk_batch_size",
            "tokenizer_hotpath_enabled",
            "token_window_cache_enabled",
            "token_window_cache_dir",
            "token_window_cache_version",
            "token_window_cache_shard_size",
        ):
            if getattr(args, flag_name, None) is not None:
                invalid.append(flag_name)
        if invalid:
            raise ValueError(f"preprocess_b does not accept overrides: {invalid}")
        embed_batch_size = (
            int(args.embed_batch_size) if getattr(args, "embed_batch_size", None) is not None else config.embed_batch_size
        )
        read_chunk_rows = (
            int(args.read_chunk_rows) if getattr(args, "read_chunk_rows", None) is not None else config.read_chunk_rows
        )
        group_shard_size = (
            int(args.group_shard_size) if getattr(args, "group_shard_size", None) is not None else config.group_shard_size
        )
        grouped_text_cache_enabled = (
            bool(args.grouped_text_cache_enabled)
            if getattr(args, "grouped_text_cache_enabled", None) is not None
            else config.grouped_text_cache_enabled
        )
        grouped_text_cache_dir = (
            str(args.grouped_text_cache_dir)
            if getattr(args, "grouped_text_cache_dir", None) is not None
            else config.grouped_text_cache_dir
        )
        grouped_text_cache_version = (
            str(args.grouped_text_cache_version)
            if getattr(args, "grouped_text_cache_version", None) is not None
            else config.grouped_text_cache_version
        )
        bf16_enabled = (
            bool(args.bf16_enabled) if getattr(args, "bf16_enabled", None) is not None else config.bf16_enabled
        )
        tf32_enabled = (
            bool(args.tf32_enabled) if getattr(args, "tf32_enabled", None) is not None else config.tf32_enabled
        )
        verify_sample_size = (
            int(args.verify_sample_size)
            if getattr(args, "verify_sample_size", None) is not None
            else config.verify_sample_size
        )
        verify_seed = (
            int(args.verify_seed) if getattr(args, "verify_seed", None) is not None else config.verify_seed
        )
        updated = replace(
            config,
            datasets=datasets,
            runtime=runtime,
            hardware=PreprocessHardwareConfig(gpu_ids=gpu_ids),
            embed_batch_size=embed_batch_size,
            read_chunk_rows=read_chunk_rows,
            group_shard_size=group_shard_size,
            grouped_text_cache_enabled=grouped_text_cache_enabled,
            grouped_text_cache_dir=grouped_text_cache_dir,
            grouped_text_cache_version=grouped_text_cache_version,
            bf16_enabled=bf16_enabled,
            tf32_enabled=tf32_enabled,
            verify_sample_size=verify_sample_size,
            verify_seed=verify_seed,
            verify_user_indices=normalize_optional_indices(
                getattr(args, "verify_user_indices", None) or config.verify_user_indices
            ),
            verify_item_indices=normalize_optional_indices(
                getattr(args, "verify_item_indices", None) or config.verify_item_indices
            ),
        )
        return validate_preprocess_config(updated)

    invalid = []
    for flag_name in (
        "embed_batch_size",
        "read_chunk_rows",
        "group_shard_size",
        "grouped_text_cache_enabled",
        "grouped_text_cache_dir",
        "grouped_text_cache_version",
        "verify_sample_size",
        "verify_seed",
        "verify_user_indices",
        "verify_item_indices",
    ):
        if getattr(args, flag_name, None) is not None:
            invalid.append(flag_name)
    if invalid:
        raise ValueError(f"preprocess_c does not accept overrides: {invalid}")
    chunk_batch_size = (
        int(args.chunk_batch_size) if getattr(args, "chunk_batch_size", None) is not None else config.chunk_batch_size
    )
    bf16_enabled = (
        bool(args.bf16_enabled) if getattr(args, "bf16_enabled", None) is not None else config.bf16_enabled
    )
    tf32_enabled = (
        bool(args.tf32_enabled) if getattr(args, "tf32_enabled", None) is not None else config.tf32_enabled
    )
    tokenizer_hotpath_enabled = (
        bool(args.tokenizer_hotpath_enabled)
        if getattr(args, "tokenizer_hotpath_enabled", None) is not None
        else config.tokenizer_hotpath_enabled
    )
    token_window_cache_enabled = (
        bool(args.token_window_cache_enabled)
        if getattr(args, "token_window_cache_enabled", None) is not None
        else config.token_window_cache_enabled
    )
    token_window_cache_dir = (
        str(args.token_window_cache_dir)
        if getattr(args, "token_window_cache_dir", None) is not None
        else config.token_window_cache_dir
    )
    token_window_cache_version = (
        str(args.token_window_cache_version)
        if getattr(args, "token_window_cache_version", None) is not None
        else config.token_window_cache_version
    )
    token_window_cache_shard_size = (
        int(args.token_window_cache_shard_size)
        if getattr(args, "token_window_cache_shard_size", None) is not None
        else config.token_window_cache_shard_size
    )
    updated = replace(
        config,
        datasets=datasets,
        runtime=runtime,
        hardware=PreprocessHardwareConfig(gpu_ids=gpu_ids),
        chunk_batch_size=chunk_batch_size,
        bf16_enabled=bf16_enabled,
        tf32_enabled=tf32_enabled,
        tokenizer_hotpath_enabled=tokenizer_hotpath_enabled,
        token_window_cache_enabled=token_window_cache_enabled,
        token_window_cache_dir=token_window_cache_dir,
        token_window_cache_version=token_window_cache_version,
        token_window_cache_shard_size=token_window_cache_shard_size,
    )
    return validate_preprocess_config(updated)
