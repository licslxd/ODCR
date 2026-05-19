"""Static guardrail runner backed by aux governance registries."""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from odcr_core.aux.governance.rule_registry import (
    AUX_RULE_DESCRIPTIONS,
    GUARDRAIL_GROUPS,
    RULE_GROUP_BY_ID,
    all_rule_ids,
)


REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    snippet: str
    message: str


@dataclass
class RuleResult:
    rule_id: str
    status: str
    summary: str
    findings: list[Finding] = field(default_factory=list)


@dataclass
class GuardrailReport:
    results: list[RuleResult]

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def failures(self) -> list[RuleResult]:
        return [result for result in self.results if result.status == "FAIL"]

    @property
    def warnings(self) -> list[RuleResult]:
        return [result for result in self.results if result.status == "WARN"]


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _pass(rule_id: str, summary: str | None = None) -> RuleResult:
    return RuleResult(rule_id, "PASS", summary or AUX_RULE_DESCRIPTIONS.get(rule_id, "guardrail passed"))


def _fail(rule_id: str, summary: str, findings: Iterable[Finding] = ()) -> RuleResult:
    return RuleResult(rule_id, "FAIL", summary, list(findings))


def _line_of(text: str, needle: str) -> int:
    idx = text.find(needle)
    if idx < 0:
        return 1
    return text[:idx].count("\n") + 1


def scan_evolution_snippet(rule_id: str, snippet: str, *, path: str = "code/example.py") -> list[Finding]:
    if path.startswith(("docs/", "AI_analysis/", "_archive/", "code/tests/")):
        return []
    banned: dict[str, str] = {
        "R042": "--new-public-param",
        "R043": "os.environ",
        "R044": "add_argument",
        "R045": "hardcoded",
        "R046": "to_csv",
        "R047": "torch.save",
        "R048": "loss",
        "R049": "mask.any",
        "R050": "fallback",
    }
    needle = banned.get(rule_id)
    if needle and re.search(re.escape(needle), snippet, re.IGNORECASE):
        return [Finding(path, _line_of(snippet, needle), needle, f"{rule_id} evolution guardrail hit")]
    return []


def scan_run_artifact_snippet(rule_id: str, snippet: str, *, path: str = "code/example.py") -> list[Finding]:
    if path.startswith(("docs/", "AI_analysis/", "code/tests/")):
        return []
    if "runs/" in snippet and "meta/run_summary.json" not in snippet:
        return [Finding(path, _line_of(snippet, "runs/"), "runs/", f"{rule_id} run artifact policy hit")]
    return []


def scan_logging_artifact_snippet(rule_id: str, snippet: str, *, path: str = "code/example.py") -> list[Finding]:
    if path.startswith(("docs/", "AI_analysis/", "code/tests/")):
        return []
    if "AI_analysis" in snippet and "full.log" in snippet:
        return [Finding(path, _line_of(snippet, "AI_analysis"), "AI_analysis", f"{rule_id} AI_analysis full-log mirror hit")]
    if rule_id == "R071" and "ODCR One-Control Guardrails: PASS " in snippet:
        return [Finding(path, _line_of(snippet, "ODCR One-Control"), "verbose guardrail stdout", "guardrail console must stay summary-level")]
    return []


def scan_old_layout_log_snippet(rule_id: str, snippet: str, *, path: str = "code/example.py") -> list[Finding]:
    if path.startswith(("docs/", "AI_analysis/", "code/tests/")):
        return []
    banned = ("logs/", "code/log.out", "nohup")
    for needle in banned:
        if needle in snippet:
            return [Finding(path, _line_of(snippet, needle), needle, f"{rule_id} old layout log hit")]
    return []


def _check_active_aux(repo_root: Path) -> RuleResult:
    required = (
        "runtime/tmux_gpu_bridge.py",
        "runtime/command_registry.py",
        "governance/rule_registry.py",
        "governance/post_edit_registry.py",
        "evidence/ai_analysis_writer.py",
        "artifacts/path_policy.py",
        "control/cli_runtime.py",
        "templates/codex_runtime_contract.py",
    )
    missing = [rel for rel in required if not (repo_root / "code" / "odcr_core" / "aux" / rel).is_file()]
    if missing:
        return _fail("R119", "active aux modules missing", [Finding("code/odcr_core/aux", 1, item, "required aux module missing") for item in missing])
    return _pass("R119")


def _check_runtime_registry(repo_root: Path) -> RuleResult:
    sys.path.insert(0, str(repo_root / "code"))
    from odcr_core.aux.runtime.command_registry import registered_command_names

    required = {
        "bridge.discover",
        "bridge.validate_only",
        "bridge.marker_probe",
        "bridge.cuda_probe",
        "probe.step3.bounded",
        "probe.step4.bounded",
        "probe.step5.bounded",
        "probe.step5A.bounded",
        "probe.step5B.bounded",
        "step5.admission.dry_run",
    }
    names = set(registered_command_names())
    missing = sorted(required - names)
    if missing:
        return _fail("R120", "runtime command allowlist missing commands", [Finding("code/odcr_core/aux/runtime/command_registry.py", 1, name, "register command") for name in missing])
    return _pass("R120")


def _check_legacy_bridge_modes(repo_root: Path) -> RuleResult:
    sys.path.insert(0, str(repo_root / "code"))
    from odcr_core.aux.runtime.command_registry import LEGACY_FORBIDDEN_COMMANDS, get_registry

    registry = get_registry()
    leaked = [name for name in LEGACY_FORBIDDEN_COMMANDS if registry.get(name) is not None]
    if leaked:
        return _fail("R121", "legacy bridge commands must not be registered", [Finding("code/odcr_core/aux/runtime/command_registry.py", 1, name, "remove legacy mode") for name in leaked])
    return _pass("R121")


def _check_writer(repo_root: Path) -> RuleResult:
    writer = repo_root / "code" / "odcr_core" / "aux" / "evidence" / "ai_analysis_writer.py"
    text = _read(writer)
    terms = ("raw_log", "search_hit", "ledger", "phase_summary", "final_report", "runtime_diagnostic", "compact_audit")
    missing = [term for term in terms if f"def {term}" not in text]
    if missing:
        return _fail("R122", "unified AI_analysis writer missing APIs", [Finding(str(writer.relative_to(repo_root)), 1, term, "missing writer method") for term in missing])
    return _pass("R122")


def _check_direct_ai_analysis_writes(repo_root: Path) -> RuleResult:
    findings: list[Finding] = []
    # .codex/hooks/*.sh may write the minimal bootstrap/runtime JSON needed
    # before Python starts; active Python code must use the unified writer.
    allowed_prefixes = (
        "code/odcr_core/aux/evidence/",
        "code/odcr_core/aux/governance/hook_scope.py",
        "code/tests/",
        "docs/",
        "AI_analysis/",
        "_archive/",
    )
    for path in sorted((repo_root / "code").rglob("*.py")) + sorted((repo_root / ".codex").rglob("*.py")):
        rel = path.relative_to(repo_root).as_posix()
        if rel.startswith(allowed_prefixes):
            continue
        text = _read(path)
        if "AI_analysis" not in text:
            continue
        direct_write = re.search(r"AI_analysis[\s\S]{0,240}(?:write_text|open|_atomic_write_text)\s*\(", text)
        direct_write = direct_write or re.search(r"(?:write_text|open|_atomic_write_text)\s*\([\s\S]{0,240}AI_analysis", text)
        if direct_write:
            findings.append(Finding(rel, _line_of(text, "AI_analysis"), "AI_analysis direct write", "route AI_analysis writes through odcr_core.aux.evidence.ai_analysis_writer"))
    if findings:
        return _fail("R123", "direct AI_analysis writes outside aux evidence writer", findings[:20])
    return _pass("R123")


def _check_registry_single_source(repo_root: Path) -> RuleResult:
    tool = _read(repo_root / "code" / "tools" / "odcr_post_edit_check.py")
    hook = _read(repo_root / ".codex" / "hooks" / "odcr_post_edit_stop.py")
    guard = _read(repo_root / "code" / "tools" / "check_one_control_guardrails.py")
    required = (
        "odcr_core.aux.governance.post_edit_runner",
        "odcr_core.aux.governance.hook_scope",
        "odcr_core.aux.governance.guardrail_runner",
    )
    haystack = "\n".join((tool, hook, guard))
    missing = [term for term in required if term not in haystack]
    if missing:
        return _fail("R124", "wrappers must delegate to aux governance", [Finding("code/tools", 1, term, "missing thin wrapper import") for term in missing])
    return _pass("R124")


def _check_gpu_handshake(repo_root: Path) -> RuleResult:
    text = _read(repo_root / "code" / "odcr_core" / "aux" / "runtime" / "gpu_handshake.py")
    required = (
        "hostname",
        "TMUX",
        "SLURM_JOB_ID",
        "CUDA_VISIBLE_DEVICES",
        "nvidia-smi",
        "torch.cuda.is_available",
        "torch.cuda.device_count",
        "torch.cuda.current_device",
        "torch.cuda.get_device_name",
    )
    missing = [term for term in required if term not in text]
    if missing:
        return _fail("R125", "gpu handshake missing current-pane evidence fields", [Finding("code/odcr_core/aux/runtime/gpu_handshake.py", 1, term, "required evidence") for term in missing])
    return _pass("R125")


def _check_no_stale_aux_pycache(repo_root: Path) -> RuleResult:
    cache_paths: list[str] = []
    scan_paths: list[Path] = [
        path
        for path in repo_root.iterdir()
        if path.is_file() or path.name in {"__pycache__", ".pytest_cache"}
    ]
    for rel_root in ("code", "configs", "docs", ".codex"):
        root = repo_root / rel_root
        if root.exists():
            scan_paths.extend(sorted(root.rglob("*")))
    for path in scan_paths:
        rel = path.relative_to(repo_root).as_posix()
        if path.name == "__pycache__" or path.name == ".pytest_cache" or path.suffix in {".pyc", ".pyo", ".pyd"}:
            cache_paths.append(rel)
    if cache_paths:
        return _fail(
            "R126",
            "active-tree Python bytecode/cache artifacts exist",
            [Finding(path, 1, "pycache", "delete bytecode/cache artifacts") for path in cache_paths[:20]],
        )
    return _pass("R126")


def _allowed_retired_text_context(line: str) -> bool:
    lowered = line.lower()
    return any(term in lowered for term in ("retired", "forbidden", "negative", "legacy", "historical", "不得", "禁止"))


def _check_preprocess_user_wording(repo_root: Path) -> RuleResult:
    findings: list[Finding] = []
    scan_paths = [
        repo_root / "README.md",
        repo_root / "AGENTS.md",
        repo_root / "code" / "odcr.py",
        repo_root / "code" / "odcr_core" / "preprocess_runtime.py",
    ]
    scan_paths.extend(sorted((repo_root / "docs").glob("*.md")))
    patterns = (
        re.compile(r"preprocess[^\n]*--stage[^\n]*--preset", re.IGNORECASE),
        re.compile(r"preprocess\s+--stage", re.IGNORECASE),
        re.compile(r"preprocess\s+preset", re.IGNORECASE),
    )
    for path in scan_paths:
        text = _read(path)
        rel = path.relative_to(repo_root).as_posix()
        for line_no, line in enumerate(text.splitlines(), start=1):
            if _allowed_retired_text_context(line):
                continue
            if any(pattern.search(line) for pattern in patterns):
                findings.append(
                    Finding(
                        rel,
                        line_no,
                        line.strip()[:160],
                        "use ./odcr preprocess a/b/c for active preprocess wording",
                    )
                )
    if findings:
        return _fail("R129", "retired preprocess CLI/preset wording found in active help/docs", findings[:20])
    return _pass("R129")


def _check_gitignore_secret_policy(repo_root: Path) -> RuleResult:
    required_patterns = (
        "__pycache__/",
        "*.py[cod]",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
        ".coverage",
        "htmlcov/",
        ".env",
        ".env.*",
        "*.pem",
        "*.key",
        "*.crt",
        "*.p12",
        "*.pfx",
        ".netrc",
        "id_rsa",
        "id_ed25519",
    )
    text = _read(repo_root / ".gitignore")
    lines = {line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")}
    missing = [pattern for pattern in required_patterns if pattern not in lines]
    findings = [Finding(".gitignore", 1, pattern, "required local cache/secret ignore pattern missing") for pattern in missing]
    secret_patterns = (".env", ".env.*", "*.pem", "*.key", "*.crt", "*.p12", "*.pfx", ".netrc", "id_rsa", "id_ed25519")
    scan_paths: list[Path] = [path for path in repo_root.iterdir() if path.is_file()]
    for rel_root in ("code", "configs", "docs", ".codex"):
        root = repo_root / rel_root
        if root.exists():
            scan_paths.extend(sorted(root.rglob("*")))
    for path in scan_paths:
        if not path.is_file():
            continue
        name = path.name
        if any(fnmatch.fnmatch(name, pattern) for pattern in secret_patterns):
            rel = path.relative_to(repo_root).as_posix()
            findings.append(Finding(rel, 1, name, "high-confidence local secret filename must not be in active repo"))
    if findings:
        return _fail("R130", "local secret/cache ignore policy violation", findings[:20])
    return _pass("R130")


def _check_reference_only_boundaries(repo_root: Path) -> RuleResult:
    findings: list[Finding] = []
    for rel in ("code1", "code2"):
        path = repo_root / rel
        if path.exists():
            findings.append(Finding(rel, 1, rel, "retired reference source tree must be deleted from the active workspace"))
    marker = repo_root / "_archive/legacy_presets_20260424/REFERENCE_ONLY.md"
    text = _read(marker)
    lowered = text.lower()
    if not marker.is_file() or "reference-only" not in lowered or "active" not in lowered or "fallback" not in lowered:
        findings.append(
            Finding(
                "_archive/legacy_presets_20260424/REFERENCE_ONLY.md",
                1,
                "REFERENCE_ONLY.md",
                "kept historical archive marker must declare no active import/execution/fallback",
            )
        )
    import_patterns = (
        re.compile(r"^\s*(?:from|import)\s+(?:code1|code2)\b", re.MULTILINE),
        re.compile(r"['\"](?:code1|code2)/", re.MULTILINE),
        re.compile(r"['\"]_archive/legacy_presets_20260424", re.MULTILINE),
    )
    for path in sorted((repo_root / "code").rglob("*.py")):
        rel = path.relative_to(repo_root).as_posix()
        if rel.startswith("code/tests/") or rel == "code/odcr_core/aux/governance/guardrail_runner.py":
            continue
        text = _read(path)
        for pattern in import_patterns:
            match = pattern.search(text)
            if match:
                findings.append(
                    Finding(rel, _line_of(text, match.group(0)), match.group(0), "active code must not import/read reference-only trees")
                )
                break
    if findings:
        return _fail("R131", "reference-only tree boundary violation", findings[:20])
    return _pass("R131")


def _check_step5_runtime_probes(repo_root: Path) -> RuleResult:
    sys.path.insert(0, str(repo_root / "code"))
    from odcr_core.aux.runtime.command_registry import get_registry

    registry = get_registry()
    missing = [name for name in ("probe.step5A.bounded", "probe.step5B.bounded") if registry.get(name) is None]
    if missing:
        return _fail("R127", "Step5A/Step5B runtime probes must be registered", [Finding("code/odcr_core/aux/runtime/command_registry.py", 1, name, "missing Step5 probe") for name in missing])
    forbidden_patterns = (
        re.compile(r"gpu_memory_peak_exceeds_36gb_safety_margin", re.IGNORECASE),
        re.compile(r"reserved_memory_safety_margin", re.IGNORECASE),
        re.compile(r"peak_reserved_safe", re.IGNORECASE),
        re.compile(r"(?:reserved_peak|max_memory_reserved|memory_reserved)[^\n]{0,80}>[^\n]{0,20}(?:36|32)", re.IGNORECASE),
        re.compile(r"(?:skip|reject)[^\n]{0,80}A4[^\n]{0,120}reserved", re.IGNORECASE),
        re.compile(r"A3[^\n]{0,120}reserved[^\n]{0,120}(?:skip|reject)[^\n]{0,80}A4", re.IGNORECASE),
    )
    findings: list[Finding] = []
    for path in sorted((repo_root / "code").rglob("*.py")):
        rel = path.relative_to(repo_root).as_posix()
        if rel.startswith("code/tests/") or rel == "code/odcr_core/aux/governance/guardrail_runner.py":
            continue
        text = _read(path)
        for pattern in forbidden_patterns:
            match = pattern.search(text)
            if match:
                findings.append(
                    Finding(
                        rel,
                        _line_of(text, match.group(0)),
                        match.group(0)[:160],
                        "reserved CUDA memory must remain diagnostic-only and must not reject/skip/select candidates",
                    )
                )
                break
    if findings:
        return _fail("R127", "reserved-memory hard gates found in active runtime code", findings[:20])
    return _pass("R127")


def _check_old_bridge_wrapper(repo_root: Path) -> RuleResult:
    text = _read(repo_root / "code" / "tools" / "odcr_tmux_gpu_bridge.py")
    if "retired and fail-fast" not in text:
        return _fail("R128", "old bridge wrapper must be fail-fast retired shim", [Finding("code/tools/odcr_tmux_gpu_bridge.py", 1, "missing fail-fast text", "retire direct wrapper")])
    if "odcr_core.aux.runtime.tmux_gpu_bridge" in text:
        return _fail("R128", "old bridge wrapper must not import active runtime bridge", [Finding("code/tools/odcr_tmux_gpu_bridge.py", 1, "active bridge import", "remove legacy wrapper execution")])
    tree = ast.parse(text or "")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"run", "Popen"}:
            return _fail("R128", "old bridge wrapper must not execute subprocesses", [Finding("code/tools/odcr_tmux_gpu_bridge.py", getattr(node, "lineno", 1), node.func.attr, "remove legacy execution")])
    return _pass("R128")


def _active_python_and_config_paths(repo_root: Path) -> list[Path]:
    paths: list[Path] = []
    for rel_root in ("code", "configs", ".codex"):
        root = repo_root / rel_root
        if root.exists():
            paths.extend(path for path in sorted(root.rglob("*")) if path.is_file())
    paths.extend(path for path in (repo_root / "odcr", repo_root / "README.md", repo_root / "AGENTS.md") if path.is_file())
    return paths


def _check_synthetic_formal_ban(repo_root: Path) -> RuleResult:
    findings: list[Finding] = []
    banned = (
        "synthetic_one_batch",
        "_step5_synthetic_preflight_batch",
        "find_unused_false_preflight.*synthetic",
        "synthetic_preflight_role",
    )
    patterns = [re.compile(item) for item in banned]
    for path in _active_python_and_config_paths(repo_root):
        rel = path.relative_to(repo_root).as_posix()
        if rel.startswith(("code/tests/", "code/tests/helpers/")) or rel == "code/odcr_core/aux/governance/guardrail_runner.py":
            continue
        text = _read(path)
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                findings.append(Finding(rel, _line_of(text, match.group(0)), match.group(0), "synthetic Step5 formal path must be absent from production"))
                break
    if findings:
        return _fail("R132", "synthetic formal/admission/tuning/preflight production path found", findings[:20])
    return _pass("R132")


def _check_evidence_level_gate(repo_root: Path) -> RuleResult:
    findings: list[Finding] = []
    bounded = _read(repo_root / "code" / "odcr_core" / "aux" / "runtime" / "bounded_probe.py")
    for needle in (
        'probe_result.get("forward_executed") is True',
        'probe_result.get("loss_backward_executed") is True',
        'probe_result.get("optimizer_step_executed") is True',
        'probe_result.get("real_ccv_packet_used") is True',
        'probe_result.get("synthetic_batch_used_for_formal_gate") is not True',
    ):
        if needle not in bounded:
            findings.append(Finding("code/odcr_core/aux/runtime/bounded_probe.py", 1, needle, "E4 gate must require real forward/backward/optimizer/CCV and no synthetic"))
    runtime_probe = _read(repo_root / "code" / "odcr_core" / "step5_runtime_probe.py")
    artifact_idx = runtime_probe.find('"artifact_build_only": True')
    if artifact_idx >= 0:
        artifact_block = runtime_probe[artifact_idx : artifact_idx + 1200]
        if '"evidence_level": E3_GPU_TRANSPORT' not in artifact_block:
            findings.append(Finding("code/odcr_core/step5_runtime_probe.py", _line_of(runtime_probe, '"artifact_build_only": True'), "artifact_build_only", "artifact-only evidence must not declare E4"))
    if findings:
        return _fail("R133", "evidence-level gate overclaim risk found", findings[:20])
    return _pass("R133")


def _check_gpu_handoff_v2_only(repo_root: Path) -> RuleResult:
    findings: list[Finding] = []
    handoff = _read(repo_root / "code" / "odcr_core" / "aux" / "runtime" / "gpu_pane_handoff.py")
    if "COMPATIBLE_SCHEMA_VERSIONS = {SCHEMA_VERSION}" not in handoff:
        findings.append(Finding("code/odcr_core/aux/runtime/gpu_pane_handoff.py", 1, "COMPATIBLE_SCHEMA_VERSIONS", "only handoff schema v2 may be accepted"))
    if "handoff/1" in handoff:
        findings.append(Finding("code/odcr_core/aux/runtime/gpu_pane_handoff.py", _line_of(handoff, "handoff/1"), "handoff/1", "handoff v1 must not be accepted in production"))
    bridge = _read(repo_root / "code" / "odcr_core" / "aux" / "runtime" / "tmux_gpu_bridge.py")
    forbidden = (
        "_run_global_bridge_mode",
        "_default_socket_path",
        "_read_state_hint",
        "_state_hint_socket_target",
        "TARGET_SOURCE_ENV",
        "TARGET_SOURCE_CURRENT_TMUX",
        "TARGET_SOURCE_DEFAULT",
    )
    for needle in forbidden:
        if needle in bridge:
            findings.append(Finding("code/odcr_core/aux/runtime/tmux_gpu_bridge.py", _line_of(bridge, needle), needle, "retired bridge target selector must be deleted"))
    if "global_target_selection_retired" not in bridge:
        findings.append(Finding("code/odcr_core/aux/runtime/tmux_gpu_bridge.py", 1, "global_target_selection_retired", "global/default execution target selection must fail fast"))
    if findings:
        return _fail("R134", "GPU handoff v2-only bridge policy violation", findings[:20])
    return _pass("R134")


def _check_legacy_code_branch_ban(repo_root: Path) -> RuleResult:
    findings: list[Finding] = []
    for rel in ("code1", "code2"):
        if (repo_root / rel).exists():
            findings.append(Finding(rel, 1, rel, "retired source tree must be deleted"))
    import_patterns = (
        re.compile(r"^\s*(?:from|import)\s+(?:code1|code2)\b", re.MULTILINE),
        re.compile(r"['\"](?:code1|code2)/", re.MULTILINE),
    )
    for path in sorted((repo_root / "code").rglob("*.py")):
        rel = path.relative_to(repo_root).as_posix()
        if rel.startswith("code/tests/") or rel == "code/odcr_core/aux/governance/guardrail_runner.py":
            continue
        text = _read(path)
        for pattern in import_patterns:
            match = pattern.search(text)
            if match:
                findings.append(Finding(rel, _line_of(text, match.group(0)), match.group(0), "active runtime must not import/read code1/code2"))
                break
    pool_sampler = _read(repo_root / "code" / "odcr_core" / "step5_pool_sampler.py")
    if "legacy_fixed_budget" in pool_sampler:
        findings.append(Finding("code/odcr_core/step5_pool_sampler.py", _line_of(pool_sampler, "legacy_fixed_budget"), "legacy_fixed_budget", "fixed sample-budget fallback must be deleted"))
    if findings:
        return _fail("R135", "legacy code/bridge/sample branch found in active runtime", findings[:20])
    return _pass("R135")


def _check_step5a_formal_contract(repo_root: Path) -> RuleResult:
    findings: list[Finding] = []
    config = _read(repo_root / "configs" / "odcr.yaml")
    if "A_TARGET_ONLY" not in config:
        findings.append(Finding("configs/odcr.yaml", 1, "A_TARGET_ONLY", "Step5A target-only ratio candidate missing"))
    if "A_NO_CF" not in config:
        findings.append(Finding("configs/odcr.yaml", 1, "A_NO_CF", "Step5A no-CF mix candidate missing"))
    if "task_decoupled_policy:" not in config or "branch: scorer_clean" not in config:
        findings.append(Finding("configs/odcr.yaml", 1, "step5.task_decoupled_policy.step5A.scorer_clean", "Step5A scorer-clean policy missing"))
    if "A_CF_MIX_1" in config or "A_CF_MIX_2" in config or "A_CF_MIX_3" in config:
        findings.append(Finding("configs/odcr.yaml", _line_of(config, "A_CF_MIX_"), "A_CF_MIX_*", "Step5A medium/low CF candidates must not remain active"))
    selected = "A_TARGET_ONLY+B_RATIO_0+A_NO_CF+B_CF_MIX_FORMAL_HIGH_MEDIUM+TG_MIX_0+AG_MIX_0+LR_1e-3+W0"
    for needle in (
        selected,
        "step5A: 190646",
        "batch_candidate: B224",
        "per_gpu_batch_size: 224",
    ):
        if needle not in config:
            findings.append(Finding("configs/odcr.yaml", 1, needle, "Step5A formal B224/high-only/sample/LR contract missing"))
    if findings:
        return _fail("R136", "Step5A formal contract drift found", findings[:20])
    return _pass("R136")


def _check_step5_legacy_module_deletion(repo_root: Path) -> RuleResult:
    engine_rel = "code/executors/step5_engine.py"
    engine = _read(repo_root / engine_rel)
    findings: list[Finding] = []
    for needle in (
        "self.recommender",
        "PETER_MLP",
        "self.hidden2token",
        "self.flan_soft_prompt_stack",
    ):
        if needle in engine:
            findings.append(
                Finding(
                    engine_rel,
                    _line_of(engine, needle),
                    needle,
                    "retired Step5 module must be deleted from the production model, not frozen or kept as fallback",
                )
            )
    if findings:
        return _fail("R137", "retired Step5 production modules still present", findings)
    return _pass("R137")


def _check_step5_lora_allowlist_only(repo_root: Path) -> RuleResult:
    findings: list[Finding] = []
    config_rel = "configs/odcr.yaml"
    lora_rel = "code/odcr_core/step5_native_lora.py"
    resolver_rel = "code/odcr_core/config_resolver.py"
    config = _read(repo_root / config_rel)
    lora = _read(repo_root / lora_rel)
    resolver = _read(repo_root / resolver_rel)
    if "target_modules: []" in config:
        findings.append(Finding(config_rel, _line_of(config, "target_modules: []"), "target_modules: []", "empty target_modules auto-discovery is retired"))
    required = (
        "HEAD_AWARE_LORA_TARGET_SENTINEL",
        "head_aware_step5_lora_targets",
        "resolve_step5_lora_targets",
        "target_modules=[] is retired",
        "target is outside the head-aware allowlist",
    )
    for needle in required:
        if needle not in lora and needle not in resolver:
            findings.append(Finding(lora_rel, 1, needle, "head-aware allowlist-only LoRA policy missing"))
    if "discover_step5_text_linear_targets(model)" in lora or "targets = list(discovered)" in lora:
        findings.append(Finding(lora_rel, 1, "auto-discovery", "native LoRA must not auto-scan the full model"))
    if findings:
        return _fail("R138", "Step5 LoRA allowlist-only policy violation", findings[:20])
    return _pass("R138")


def _check_step5_mha_out_proj_lora_ban(repo_root: Path) -> RuleResult:
    findings: list[Finding] = []
    lora_rel = "code/odcr_core/step5_native_lora.py"
    lora = _read(repo_root / lora_rel)
    required = (
        "nn.MultiheadAttention",
        'child == "out_proj"',
        "must not be LoRA-wrapped",
    )
    for needle in required:
        if needle not in lora:
            findings.append(Finding(lora_rel, 1, needle, "MHA out_proj LoRA ban missing"))
    for rel in ("code/executors/step5_engine.py", "code/odcr_core/step5_runtime_probe.py"):
        text = _read(repo_root / rel)
        if "out_proj.lora_A" in text or "out_proj.lora_B" in text:
            findings.append(Finding(rel, _line_of(text, "out_proj.lora_"), "out_proj.lora_*", "MHA out_proj LoRA must not be trainable"))
    if findings:
        return _fail("R139", "MHA out_proj LoRA ban missing or violated", findings[:20])
    return _pass("R139")


def _check_step5_all_trainable_grad_gate(repo_root: Path) -> RuleResult:
    findings: list[Finding] = []
    for rel in ("code/executors/step5_engine.py", "code/odcr_core/step5_runtime_probe.py"):
        text = _read(repo_root / rel)
        if "validate_all_trainable_params_receive_grad" not in text:
            findings.append(Finding(rel, 1, "validate_all_trainable_params_receive_grad", "formal/E4 must use the shared all-trainable-grad gate"))
        if "trainable_param_count" not in text or "lora_grad_present_count" not in text:
            findings.append(Finding(rel, 1, "trainable/lora grad counts", "all-grad evidence counts must be emitted"))
    bounded_rel = "code/odcr_core/aux/runtime/bounded_probe.py"
    bounded = _read(repo_root / bounded_rel)
    for needle in (
        'probe_result.get("all_trainable_grad_status") == "pass"',
        'probe_result.get("trainable_param_count")',
        'probe_result.get("grad_present_count")',
        'probe_result.get("lora_trainable_count")',
        'probe_result.get("lora_grad_present_count")',
    ):
        if needle not in bounded:
            findings.append(Finding(bounded_rel, 1, needle, "E4 admission must require all-grad PASS"))
    if findings:
        return _fail("R140", "Step5 all-trainable-grad gate is not unified", findings[:20])
    return _pass("R140")


def _check_step5_combined_formal_ban(repo_root: Path) -> RuleResult:
    findings: list[Finding] = []
    engine_rel = "code/executors/step5_engine.py"
    engine = _read(repo_root / engine_rel)
    if "Step5 combined formal training is disabled until a combined-specific all-trainable-grad E4 passes" not in engine:
        findings.append(Finding(engine_rel, 1, "combined formal disabled", "combined formal train must fail fast until audited"))
    for rel in ("code/odcr_core/step5_grad_contract.py", "code/odcr_core/config_resolver.py"):
        text = _read(repo_root / rel)
        if "combined_formal_enabled" not in text:
            findings.append(Finding(rel, 1, "combined_formal_enabled", "combined formal ban must be recorded in contracts"))
    if findings:
        return _fail("R141", "Step5 combined formal ban missing", findings[:20])
    return _pass("R141")


def _check_step5_validation_scorer_only_gate(repo_root: Path) -> RuleResult:
    findings: list[Finding] = []
    config_rel = "configs/odcr.yaml"
    config = _read(repo_root / config_rel)
    for needle in (
        "valid_per_gpu_batch_size: 192",
        "valid_forward_micro_batch_size: 192",
        "validation_memory_policy: microbatch_accumulate",
        "step5A_validation_mode: scorer_only",
        "formal_entry_E4_validation_required: true",
        "E4_gpu_shard_forward_bounded_formal_entry_with_validation",
    ):
        if needle not in config:
            findings.append(Finding(config_rel, 1, needle, "Step5 validation One-Control gate missing"))
    engine_rel = "code/executors/step5_engine.py"
    engine = _read(repo_root / engine_rel)
    for needle in (
        "return_explainer_logits=False",
        "scorer_only=scorer_only_validation",
        "validation_mode=True",
        "P0: Step5A scorer-only validation returned word_dist logits",
        "valid_forward_micro_batch_size",
        "evalRatingOnlyModel",
        "CODE1_COMPATIBLE_RATING_PROTOCOL_ID",
    ):
        if needle not in engine:
            findings.append(Finding(engine_rel, 1, needle, "Step5A validation must be scorer-only and microbatched"))
    odcr_rel = "code/odcr.py"
    odcr = _read(repo_root / odcr_rel)
    for needle in (
        "bool(getattr(args, \"recovery_eval\", False))",
        "_step5_eval_only_head_guard",
        "Step5 eval-only blocked",
    ):
        if needle not in odcr:
            findings.append(Finding(odcr_rel, 1, needle, "Step5A eval-only must not be hijacked by static salvage"))
    handoff_rel = "code/odcr_core/step5_rating_handoff.py"
    handoff = _read(repo_root / handoff_rel)
    for needle in (
        "STEP5A_RATING_HANDOFF_SCHEMA_VERSION",
        "CODE1_COMPATIBLE_RATING_PROTOCOL_ID",
        "batch_invariance_gate_removed",
        "batch_invariance_required",
        "rating_metric_compatibility_report.json",
    ):
        if needle not in handoff:
            findings.append(Finding(handoff_rel, 1, needle, "Step5A rating eval_handoff schema/gate-removal contract missing"))
    probe_rel = "code/odcr_core/step5_runtime_probe.py"
    probe = _read(repo_root / probe_rel)
    for needle in (
        "validation_pass_executed",
        "validation_forward_pass",
        "validation_loss_finite",
        "E4_GPU_SHARD_FORWARD_BOUNDED_FORMAL_ENTRY_WITH_VALIDATION",
        "out_logits_materialized_in_step5A_validation",
    ):
        if needle not in probe:
            findings.append(Finding(probe_rel, 1, needle, "formal-entry E4 must include validation"))
    bounded_rel = "code/odcr_core/aux/runtime/bounded_probe.py"
    bounded = _read(repo_root / bounded_rel)
    for needle in (
        "validation_pass_executed",
        "validation_forward_pass",
        "validation_loss_finite",
        "E4_GPU_SHARD_FORWARD_BOUNDED_FORMAL_ENTRY_WITH_VALIDATION",
    ):
        if needle not in bounded:
            findings.append(Finding(bounded_rel, 1, needle, "bounded admission must require validation E4"))
    if findings:
        return _fail("R142", "Step5 validation scorer-only/E4-with-validation gate missing", findings[:20])
    return _pass("R142")


CHECKS: dict[str, Callable[[Path], RuleResult]] = {
    "R119": _check_active_aux,
    "R120": _check_runtime_registry,
    "R121": _check_legacy_bridge_modes,
    "R122": _check_writer,
    "R123": _check_direct_ai_analysis_writes,
    "R124": _check_registry_single_source,
    "R125": _check_gpu_handshake,
    "R126": _check_no_stale_aux_pycache,
    "R127": _check_step5_runtime_probes,
    "R128": _check_old_bridge_wrapper,
    "R129": _check_preprocess_user_wording,
    "R130": _check_gitignore_secret_policy,
    "R131": _check_reference_only_boundaries,
    "R132": _check_synthetic_formal_ban,
    "R133": _check_evidence_level_gate,
    "R134": _check_gpu_handoff_v2_only,
    "R135": _check_legacy_code_branch_ban,
    "R136": _check_step5a_formal_contract,
    "R137": _check_step5_legacy_module_deletion,
    "R138": _check_step5_lora_allowlist_only,
    "R139": _check_step5_mha_out_proj_lora_ban,
    "R140": _check_step5_all_trainable_grad_gate,
    "R141": _check_step5_combined_formal_ban,
    "R142": _check_step5_validation_scorer_only_gate,
}


def _default_pass_results() -> list[RuleResult]:
    return [_pass(rule_id) for rule_id in all_rule_ids() if rule_id not in CHECKS]


def run_checks(*, repo_root: Path = REPO_ROOT, strict: bool = False) -> GuardrailReport:
    del strict
    root = Path(repo_root).resolve()
    results = _default_pass_results()
    for rule_id in sorted(CHECKS):
        try:
            results.append(CHECKS[rule_id](root))
        except Exception as exc:
            results.append(_fail(rule_id, f"guardrail check raised: {exc}", [Finding("guardrail", 1, repr(exc), "fix guardrail")]))
    results.sort(key=lambda item: int(item.rule_id[1:]) if item.rule_id[1:].isdigit() else 9999)
    return GuardrailReport(results=results)


def format_report(report: GuardrailReport) -> str:
    lines = [
        f"ODCR One-Control Guardrails: {'FAIL' if report.failures else 'PASS'} ({len(report.failures)} fail, {len(report.warnings)} warn)"
    ]
    for result in report.results:
        if result.status == "FAIL":
            lines.append(f"[{result.status}] {result.rule_id} {RULE_GROUP_BY_ID.get(result.rule_id, 'unknown')}: {result.summary}")
            for finding in result.findings[:10]:
                lines.append(f"  - {finding.path}:{finding.line}: {finding.message} ({finding.snippet})")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ODCR One-Control static guardrails.")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_checks(repo_root=Path(args.repo_root), strict=bool(args.strict))
    if args.as_json:
        print(json.dumps([result.__dict__ for result in report.results], default=lambda obj: obj.__dict__, indent=2))
    else:
        print(format_report(report))
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
