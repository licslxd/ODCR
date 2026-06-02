"""Registry and override validation for controlled ODCR ablations."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping


ALLOWED_TASKS = (7, 8)
FULL_VARIANT = "full_odcr"
ABLATION_VARIANTS = ("wo_rcr", "wo_cf", "wo_ccv_fca")
ALLOWED_VARIANTS = (FULL_VARIANT, *ABLATION_VARIANTS)
SCENARIO = "weak_cross_platform"
EXPECTED_FULL_RUN = "1_19"

_OUTPUT_RE = re.compile(r"^runs/step5/task([78])/ablation_[A-Za-z0-9_]+$")


class AblationValidationError(RuntimeError):
    """Raised when ablation registry, overrides, or manifests are invalid."""


def repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[3]


def _repo_root(repo_root: str | Path | None = None) -> Path:
    return Path(repo_root).expanduser().resolve() if repo_root is not None else repo_root_from_file()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is an ODCR dependency.
        raise AblationValidationError("PyYAML is required for ablation YAML files") from exc
    if not path.is_file():
        raise AblationValidationError(f"ablation YAML file missing: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise AblationValidationError(f"{path} must contain a top-level mapping")
    return raw


def entry_key(task: int, variant: str) -> str:
    task_i = int(task)
    variant_s = normalize_variant(variant)
    return f"task{task_i}_{'full' if variant_s == FULL_VARIANT else variant_s}"


def normalize_variant(variant: str) -> str:
    value = str(variant or "").strip()
    if value == "full":
        value = FULL_VARIANT
    if value not in ALLOWED_VARIANTS:
        raise AblationValidationError(
            f"unsupported ablation variant {variant!r}; expected one of {', '.join(ALLOWED_VARIANTS)}"
        )
    return value


def registry_path(repo_root: str | Path | None = None) -> Path:
    return _repo_root(repo_root) / "ablations" / "registry.yaml"


def override_path(repo_root: str | Path | None, task: int, variant: str) -> Path:
    variant_s = normalize_variant(variant)
    if variant_s == FULL_VARIANT:
        raise AblationValidationError("full_odcr does not have an ablation config override")
    return _repo_root(repo_root) / "ablations" / "config_overrides" / f"task{int(task)}_{variant_s}.yaml"


def load_registry(repo_root: str | Path | None = None) -> dict[str, dict[str, Any]]:
    raw = _load_yaml(registry_path(repo_root))
    return {str(k): dict(v) for k, v in raw.items() if isinstance(v, Mapping)}


def load_config_override(repo_root: str | Path | None, task: int, variant: str) -> dict[str, Any]:
    return _load_yaml(override_path(repo_root, task, variant))


def expected_registry_keys() -> set[str]:
    keys: set[str] = set()
    for task in (8, 7):
        keys.add(entry_key(task, FULL_VARIANT))
        for variant in ABLATION_VARIANTS:
            keys.add(entry_key(task, variant))
    return keys


def _require(entry: Mapping[str, Any], field: str, *, key: str) -> Any:
    if field not in entry:
        raise AblationValidationError(f"{key} missing required field {field}")
    return entry[field]


def _validate_output_run(path: str, *, task: int, key: str) -> None:
    match = _OUTPUT_RE.fullmatch(str(path or ""))
    if not match:
        raise AblationValidationError(
            f"{key}.output_run must match runs/step5/task7|task8/ablation_*; got {path!r}"
        )
    if int(match.group(1)) != int(task):
        raise AblationValidationError(f"{key}.output_run task does not match entry task {task}: {path}")


def validate_registry(registry: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    registry_obj = dict(registry or load_registry())
    keys = set(registry_obj)
    expected = expected_registry_keys()
    if keys != expected:
        raise AblationValidationError(
            "ablations/registry.yaml must contain exactly task7/task8 full + 3 variants; "
            f"missing={sorted(expected - keys)} extra={sorted(keys - expected)}"
        )
    entries: list[dict[str, Any]] = []
    for key in sorted(keys):
        item = registry_obj[key]
        if not isinstance(item, Mapping):
            raise AblationValidationError(f"{key} must be a mapping")
        task = int(_require(item, "task", key=key))
        if task not in ALLOWED_TASKS:
            raise AblationValidationError(f"{key}.task must be 7 or 8")
        variant = normalize_variant(str(_require(item, "variant", key=key)))
        if entry_key(task, variant) != key:
            raise AblationValidationError(f"{key} does not match task={task} variant={variant}")
        if _require(item, "scenario", key=key) != SCENARIO:
            raise AblationValidationError(f"{key}.scenario must be {SCENARIO}")
        if "direction" not in item:
            raise AblationValidationError(f"{key} missing required field direction")
        if "paper_role" not in item:
            raise AblationValidationError(f"{key} missing required field paper_role")
        is_ablation = bool(_require(item, "is_ablation", key=key))
        allowed = bool(_require(item, "allowed_for_paper_table", key=key))
        if variant == FULL_VARIANT:
            if is_ablation:
                raise AblationValidationError(f"{key} full_odcr entry must set is_ablation=false")
            if not allowed:
                raise AblationValidationError(f"{key} full_odcr entry must be paper-table eligible as baseline")
            source_run = str(_require(item, "source_run", key=key))
            expected_source = f"runs/step5/task{task}/{EXPECTED_FULL_RUN}"
            if source_run != expected_source:
                raise AblationValidationError(f"{key}.source_run must be {expected_source}")
        else:
            if not is_ablation:
                raise AblationValidationError(f"{key} variant entry must set is_ablation=true")
            if allowed:
                raise AblationValidationError(f"{key} planned ablation must not be paper-table eligible")
            _validate_output_run(str(_require(item, "output_run", key=key)), task=task, key=key)
        entries.append({"key": key, "task": task, "variant": variant, "is_ablation": is_ablation})
    return {
        "schema_version": "odcr_ablation_registry_validation/1",
        "status": "pass",
        "entry_count": len(entries),
        "entries": entries,
    }


def validate_config_override(
    payload: Mapping[str, Any],
    *,
    registry_entry: Mapping[str, Any],
    key: str,
) -> dict[str, Any]:
    task = int(_require(payload, "task", key=key))
    variant = normalize_variant(str(_require(payload, "variant", key=key)))
    if task != int(registry_entry.get("task")) or variant != str(registry_entry.get("variant")):
        raise AblationValidationError(f"{key} override task/variant mismatch registry entry")
    if _require(payload, "scenario", key=key) != SCENARIO:
        raise AblationValidationError(f"{key}.scenario must be {SCENARIO}")
    output_run = str(_require(payload, "output_run", key=key))
    if output_run != str(registry_entry.get("output_run")):
        raise AblationValidationError(f"{key}.output_run mismatch registry output_run")
    _validate_output_run(output_run, task=task, key=key)
    source_full_run = str(_require(payload, "source_full_run", key=key))
    if source_full_run != f"runs/step5/task{task}/{EXPECTED_FULL_RUN}":
        raise AblationValidationError(f"{key}.source_full_run must target the task-local full run")
    required_scalar = {
        "expected_step3_rating_source": "task_local_step3_accepted_scorer",
        "base_protocol": "paper_greedy_25",
    }
    for field, expected in required_scalar.items():
        if str(_require(payload, field, key=key)) != expected:
            raise AblationValidationError(f"{key}.{field} must be {expected}")
    step5 = _require(payload, "step5", key=key)
    safety = _require(payload, "safety", key=key)
    controls = _require(payload, "variant_controls", key=key)
    if not isinstance(step5, Mapping) or step5.get("explanation_only") is not True:
        raise AblationValidationError(f"{key}.step5.explanation_only must be true")
    if step5.get("official_eval_profile") != "paper_greedy_25":
        raise AblationValidationError(f"{key}.step5.official_eval_profile must be paper_greedy_25")
    if not isinstance(safety, Mapping):
        raise AblationValidationError(f"{key}.safety must be a mapping")
    if safety.get("no_promote_latest") is not True or safety.get("dry_run_safe") is not True:
        raise AblationValidationError(f"{key}.safety must set no_promote_latest=true and dry_run_safe=true")
    if not isinstance(controls, Mapping):
        raise AblationValidationError(f"{key}.variant_controls must be a mapping")
    return {
        "key": key,
        "task": task,
        "variant": variant,
        "status": "pass",
        "output_run": output_run,
    }


def registry_entry(repo_root: str | Path | None, task: int, variant: str) -> dict[str, Any]:
    registry = load_registry(repo_root)
    key = entry_key(task, variant)
    if key not in registry:
        raise AblationValidationError(f"registry entry missing: {key}")
    return registry[key]


def validate_all(repo_root: str | Path | None = None) -> dict[str, Any]:
    root = _repo_root(repo_root)
    registry = load_registry(root)
    reg_result = validate_registry(registry)
    override_results: list[dict[str, Any]] = []
    for task in ALLOWED_TASKS:
        for variant in ABLATION_VARIANTS:
            key = entry_key(task, variant)
            override_results.append(
                validate_config_override(
                    load_config_override(root, task, variant),
                    registry_entry=registry[key],
                    key=key,
                )
            )
    return {
        "schema_version": "odcr_ablation_infra_validation/1",
        "status": "pass",
        "registry": reg_result,
        "config_overrides": override_results,
    }
