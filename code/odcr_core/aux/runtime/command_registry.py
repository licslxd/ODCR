"""Runtime command metadata for ODCR auxiliary work."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


LEGACY_FORBIDDEN_COMMANDS = (
    "repo-command",
    "repo-script",
    "repo-module",
    "command-file",
    "long-run",
    "bash-c",
    "arbitrary-shell",
)

FORBIDDEN_TOKENS = (
    "bash",
    "sh",
    "-c",
    "python -c",
    "nohup",
    "&",
    "srun",
    "sbatch",
    "scancel",
    "odcr-enter-gpu",
    "rm",
    "rm -rf",
    "tmux send-keys",
    "tmux attach",
    "tmux attach-session",
    "tmux switch-client",
    "tmux kill-session",
    "tmux new-session",
    "attach-session",
    "switch-client",
    "kill-session",
    "new-session",
)


@dataclass(frozen=True)
class RuntimeCommandSpec:
    name: str
    stage: str | None
    substage: str | None
    mode: str
    requires_gpu: bool
    requires_tmux: bool
    writes_formal_runs: bool
    allowed_args: tuple[str, ...] = ()
    forbidden_args: tuple[str, ...] = FORBIDDEN_TOKENS
    timeout_s: int = 60
    output_policy: str = "AI_analysis"
    ai_analysis_policy: str = "compact_evidence"
    formal_namespace_policy: str = "forbid"
    test_artifact_policy: str = "test_artifacts_only"
    internal_child: bool = False
    metadata: Mapping[str, str] = field(default_factory=dict)

    def validate_args(self, args: Sequence[str]) -> None:
        joined = " ".join(str(arg) for arg in args)
        for token in self.forbidden_args:
            if token and token in joined:
                raise RuntimeCommandError(f"{self.name} rejects forbidden token: {token}")
        if self.allowed_args:
            for arg in args:
                if str(arg).startswith("--") and str(arg) not in self.allowed_args:
                    raise RuntimeCommandError(f"{self.name} rejects unregistered arg: {arg}")


class RuntimeCommandError(RuntimeError):
    pass


FORMAL_TRAIN_DETECTOR_VERSION = "odcr_bridge_formal_train_audit/2"


def _clean_argv(argv: Sequence[str]) -> list[str]:
    return [str(item) for item in argv if str(item).strip()]


def formal_training_command_reason(argv: Sequence[str]) -> str | None:
    """Historical audit hook; bridge exec no longer blocks ODCR training.

    GPU safety now lives in fresh pane discovery, CUDA probe, audit-only
    compute-app evidence, run namespace validation, and stage-level One-Control
    checks. User-authorized fixed-run/reclosure training must be dispatched to
    the validated GPU pane instead of being stopped by a string detector.
    """

    args = _clean_argv(argv)
    if not args:
        return "empty command"
    return None


def assert_not_formal_training(argv: Sequence[str]) -> None:
    del argv
    return None


class RuntimeCommandRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, RuntimeCommandSpec] = {}

    def register(self, spec: RuntimeCommandSpec) -> None:
        if spec.name in LEGACY_FORBIDDEN_COMMANDS:
            raise RuntimeCommandError(f"legacy runtime command is forbidden: {spec.name}")
        if spec.name in self._specs:
            raise RuntimeCommandError(f"duplicate runtime command: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> RuntimeCommandSpec | None:
        return self._specs.get(name)

    def require(self, name: str) -> RuntimeCommandSpec:
        spec = self.get(name)
        if spec is None:
            raise RuntimeCommandError(f"unregistered runtime command: {name}")
        return spec

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def specs(self) -> tuple[RuntimeCommandSpec, ...]:
        return tuple(self._specs[name] for name in self.names())


def _build_registry() -> RuntimeCommandRegistry:
    registry = RuntimeCommandRegistry()
    for spec in (
        RuntimeCommandSpec("bridge.discover", None, "bridge", "discover", False, False, False, timeout_s=20),
        RuntimeCommandSpec("bridge.validate_only", None, "bridge", "validate-only", False, True, False, timeout_s=60),
        RuntimeCommandSpec("bridge.marker_probe", None, "bridge", "marker-probe", False, True, False, timeout_s=30),
        RuntimeCommandSpec("bridge.cuda_probe", None, "bridge", "cuda-probe", True, True, False, timeout_s=120),
        RuntimeCommandSpec("probe.step3.bounded", "step3", "bounded", "probe", True, True, False, allowed_args=("--stage", "--task", "--bounded"), timeout_s=180),
        RuntimeCommandSpec("probe.step4.bounded", "step4", "bounded", "probe", True, True, False, allowed_args=("--stage", "--task", "--bounded"), timeout_s=240),
        RuntimeCommandSpec("probe.step5.bounded", "step5", "bounded", "probe", True, True, False, allowed_args=("--stage", "--task", "--bounded", "--set", "--candidate-id", "--timeout", "--from-step4-run", "--evidence-level", "--scan", "--global", "--socket", "--target"), timeout_s=900),
        RuntimeCommandSpec("probe.rating_stability_control.bounded", "rating_stability_control", "bounded", "probe", True, True, False, allowed_args=("--stage", "--task", "--bounded", "--set", "--candidate-id", "--timeout", "--from-step4-run", "--evidence-level", "--global", "--socket", "--target"), timeout_s=900),
        RuntimeCommandSpec("probe.step5_explanation.bounded", "step5_explanation", "bounded", "probe", True, True, False, allowed_args=("--stage", "--task", "--bounded", "--set", "--candidate-id", "--timeout", "--from-step4-run", "--evidence-level", "--global", "--socket", "--target"), timeout_s=900),
        RuntimeCommandSpec("bridge.probe_child", None, "bridge", "internal", True, False, False, allowed_args=("--stage", "--task", "--bounded", "--probe-child", "--status-path", "--config", "--set", "--candidate-id", "--timeout", "--from-step4-run", "--evidence-level"), timeout_s=900, internal_child=True),
        RuntimeCommandSpec("step5.admission.dry_run", "step5", "admission", "dry-run", False, False, False, allowed_args=("--task",), timeout_s=60),
        RuntimeCommandSpec("bridge.handshake_child", None, "bridge", "internal", False, False, False, allowed_args=("--kind", "--stage", "--task", "--run-id", "--status-path", "--log-path", "--report-path", "--repo-root", "--require-cuda"), timeout_s=120, internal_child=True),
    ):
        registry.register(spec)
    return registry


_REGISTRY = _build_registry()


def get_registry() -> RuntimeCommandRegistry:
    return _REGISTRY


def registered_command_names() -> tuple[str, ...]:
    return _REGISTRY.names()


def require_command(name: str) -> RuntimeCommandSpec:
    return _REGISTRY.require(name)
