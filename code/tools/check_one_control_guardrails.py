#!/usr/bin/env python3
"""Static guardrails for the ODCR One-Control architecture.

The checker is intentionally lightweight: it reads repository text files,
validates the canonical config shape, and reports architecture drift before a
future change can reintroduce preset/env/shell sprawl.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK_WRAPPER_ABS = "/public/home/zhangliml/lc/ODCR/ODCR-main/.codex/hooks/odcr_post_edit_stop.sh"
HOOK_STOP_COMMAND = f"/usr/bin/env bash {HOOK_WRAPPER_ABS}"
D4C_PYTHON_ABS = "/public/home/zhangliml/miniconda3/envs/D4C/bin/python"
HOOK_DIAGNOSTICS_REL = "AI_analysis/01_raw_logs/codex_hooks"

TOP_LEVEL_BLOCKS = (
    "project",
    "env",
    "hardware",
    "tasks",
    "preprocess",
    "step3",
    "step4",
    "step5",
    "eval",
)

GUARDRAIL_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "control-plane",
        ("R002", "R003", "R005", "R006", "R009", "R025", "R026", "R027", "R028"),
    ),
    ("data-contract", ("R008", "R024")),
    ("lineage-cache", ("R031", "R032", "R033", "R034", "R035", "R041")),
    ("ddp-loss", ("R036", "R037", "R038", "R040")),
    ("legacy-cleanup", ("R001", "R004", "R095")),
    (
        "step3-mainline",
        (
            "R010",
            "R011",
            "R029",
            "R097",
            "R099",
            "R100",
            "R101",
            "R102",
            "R103",
            "R104",
            "R105",
            "R106",
            "R107",
            "R108",
            "R109",
            "R110",
            "R111",
        ),
    ),
    ("stage-truth-upstream", ("R112", "R113")),
    ("step4-runtime-preflight", ("R114", "R115")),
    ("step4-evidence-level", ("R116",)),
    ("no-accum-architecture", ("R117",)),
    ("step4-rcr", ("R012", "R013", "R014", "R015")),
    ("step5-innovation", ("R016", "R017", "R018", "R019", "R020", "R021", "R022", "R023", "R030", "R039")),
    ("code-hygiene", ("R007",)),
    ("evolution-protocol", ("R042", "R043", "R044", "R045", "R046", "R047", "R048", "R049", "R050", "R096")),
    ("post-edit-workflow", ("R051", "R052", "R053", "R054", "R055", "R056", "R089")),
    ("run-summary-logging", ("R057", "R058", "R059", "R090", "R091")),
    ("p0-cache-hard-gates", ("R092", "R093", "R094", "R098")),
    ("logging-console-file", ("R060", "R061", "R062")),
    ("logging-artifact-evolution", ("R068", "R069", "R070", "R071", "R072")),
    ("logging-directory-boundaries", ("R078", "R079", "R080", "R081", "R082", "R083")),
    ("logging-old-layout-tail", ("R084", "R085", "R086", "R087", "R088")),
    (
        "post-edit-fast-path",
        (
            "R063",
            "R064",
            "R065",
            "R066",
            "R067",
            "R073",
            "R074",
            "R075",
            "R076",
            "R077",
        ),
    ),
)

RULE_GROUP_BY_ID = {rule_id: group for group, rule_ids in GUARDRAIL_GROUPS for rule_id in rule_ids}

MAINLINE_FILES = (
    "odcr",
    "code/odcr.py",
    "code/odcr_core/config_schema.py",
    "code/odcr_core/config_resolver.py",
    "code/odcr_core/runners.py",
    "code/odcr_core/dispatch.py",
    "code/odcr_core/preprocess_schema.py",
    "code/odcr_core/preprocess_registry.py",
    "code/odcr_core/preprocess_runtime.py",
    "code/odcr_core/step5_native_lora.py",
)

LEGACY_PRESET_READER_ALLOWLIST = {
    "code/config.py": "legacy defaults module; One-Control entry uses config_resolver",
}
LEGACY_PRESET_READER_MARKER = "ODCR_DEPRECATED_INTERNAL_LEGACY_PRESET_READER = True"

ALLOWED_YAML_ENV = {
    "configs/odcr.yaml",
}

BANNED_SHELL_NAMES = {
    "step1.sh",
    "step2.sh",
    "step3.sh",
    "step4.sh",
    "step5.sh",
    "train.sh",
    "train_ddp.sh",
    "eval.sh",
    "eval_ddp.sh",
    "smoke_ddp.sh",
}

PREPROCESS_RETIRED_DETAIL_COLUMNS = (
    "content_keywords",
    "content_aspects",
    "content_entities",
    "style_markers",
    "template_family",
    "length_style_bucket",
)

TRAINING_ARG_RE = re.compile(
    r"add_argument\(\s*['\"]--(?:lr|learning[_-]rate|epochs|batch[_-]size|"
    r"micro[_-]batch[_-]size|grad[_-]accum|gradient[_-]accumulation[_-]steps|"
    r"top[_-]p|rerank(?:[_-][a-z0-9]+)?)['\"]"
)

PRESET_READ_RE = re.compile(
    r"presets/|/presets|Path\([^)\n]*['\"]presets['\"]|"
    r"_REPO_ROOT\s*/\s*['\"]presets['\"]|(?<![A-Za-z0-9_])_PRESETS\s*=|presets_dir\s*/",
    re.IGNORECASE,
)

STEP3_RETIRED_SEMANTIC_RE = re.compile(
    r"AdvTrain|weak_adv|domain adversarial|adversarial training|域对抗",
    re.IGNORECASE,
)

STEP3_TYPED_BRIDGE_RE = re.compile(
    r"config_loader|load_resolved_config|instantiate_step3_preset|"
    r"bridge_step3_config_to_resolved_config|run_step3_with_typed_control_plane"
)

STEP3_RUNTIME_RECONNECT_RE = re.compile(
    r"step3_runtime|bridge_step3_config_to_resolved_config|"
    r"run_step3_with_typed_control_plane|resolve_step3_cli_config"
)

EVOLUTION_EXEMPT_PREFIXES = ("AI_analysis", "_archive", "docs", "code/tests", "code/tools", "code1", ".codex")
EVOLUTION_EXEMPT_FILES = {
    "code/tools/check_one_control_guardrails.py",
    "scripts/generate_light_ppt.py",
}

EVOLUTION_ALLOWED_ARG_DEFAULT_FLAGS = {
    "--a",
    "--b",
    "--c",
    "--config",
    "--decode-policy",
    "--device",
    "--from",
    "--from-step3",
    "--from-step3-run",
    "--from-step4",
    "--from-step4-run",
    "--from-step5",
    "--generate-max-samples",
    "--generate-temperature",
    "--generate-top-p",
    "--grouped-text-cache",
    "--grouped-text-cache-enabled",
    "--label-smoothing",
    "--lines",
    "--max-explanation-length",
    "--max-samples",
    "--mode",
    "--nlayers",
    "--poll-profile",
    "--repetition-penalty",
    "--run-id",
    "--save_file",
    "--seed",
    "--to",
    "--verify-sample-size",
    "--verify-seed",
    "--prepare-cache",
    "--preflight",
    "--preflight-mode",
    "--validation-namespace",
    "--verbose",
    "--debug",
    "--decode-strategy",
    "--checkpoint-metric",
    "--ddp-find-unused-parameters",
}

EVOLUTION_ALLOWED_ENV_READS = {
    "CUDA_VISIBLE_DEVICES",
    "ODCR_CONSOLE_LEVEL",
    "ODCR_CONFIG_FIELD_SOURCES_JSON",
    "ODCR_DAEMON_CHILD",
    "ODCR_DDP_EPOCH_END_BARRIER",
    "ODCR_DDP_FAST",
    "ODCR_DEBUG_GRAD_DIFF",
    "ODCR_DECODE_PRESET_STEM",
    "ODCR_DECODE_PROFILE_JSON",
    "ODCR_DISPATCH_DETAIL",
    "ODCR_DUAL_TRAIN_LOG",
    "ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON",
    "ODCR_EMBED_DIM",
    "ODCR_EVAL_PROFILE_NAME",
    "ODCR_EVAL_RUN_DIR",
    "ODCR_EVAL_SUMMARY",
    "ODCR_EVAL_SUMMARY_GLOBAL_DIR",
    "ODCR_FILE_LEVEL",
    "ODCR_FINITE_CHECK_MODE",
    "ODCR_GENERATION_SEMANTIC_FINGERPRINT",
    "ODCR_GLOBAL_EVAL_BATCH_SIZE",
    "ODCR_GRAD_TOPK",
    "ODCR_GRAD_WARN_NORM",
    "ODCR_HARDWARE_PRESET",
    "ODCR_HARDWARE_PROFILE_JSON",
    "ODCR_ITER",
    "ODCR_ITERATION_META_DIR",
    "ODCR_LOG_CONSOLE",
    "ODCR_LOG_DIR",
    "ODCR_LOG_GATHER_SCHEMA",
    "ODCR_LOG_GRAD_INTERVAL",
    "ODCR_LOG_PRETTY",
    "ODCR_LOG_SILENT_STDIO_WARN",
    "ODCR_LOG_STEP_JSON",
    "ODCR_LOG_STEP_LOSS_PARTS",
    "ODCR_LOG_STRUCTURED_CONSOLE",
    "LOCAL_RANK",
    "ODCR_MANIFEST_CLI_INVOCATION",
    "ODCR_MANIFEST_DIR",
    "ODCR_MATRIX_CONTEXT_JSON",
    "ODCR_MIRROR_LOG",
    "ODCR_NLTK_DATA",
    "ODCR_NONFINITE_LOSS_ABORT_AFTER",
    "ODCR_PROFILE_CONSUME_FUSED_LEGACY",
    "ODCR_PROFILE_STEP_COMPONENTS",
    "ODCR_RERANK_DEBUG",
    "ODCR_RERANK_PRESET_STEM",
    "ODCR_RERANK_PROFILE_JSON",
    "ODCR_RESOLVED_EMBED_DIM",
    "ODCR_ROOT",
    "ODCR_RUNTIME_DIAGNOSTICS_FINGERPRINT",
    "ODCR_RUNTIME_ALLOW_TF32",
    "ODCR_RUNTIME_AMP_AUTOCAST",
    "ODCR_RUNTIME_GRAD_SCALER",
    "ODCR_RUNTIME_PRECISION_MODE",
    "ODCR_THREAD_ENV_EFFECTIVE_JSON",
    "ODCR_STEP3_TOKENIZER_MAX_LENGTH",
    "ODCR_STEP3_EVIDENCE_MAX_LENGTH",
    "ODCR_STEP3_TOKENIZER_CACHE_STARTUP_JSON",
    "ODCR_STAGE_RUN_DIR",
    "ODCR_STEP3_RUN_DIR",
    "ODCR_STEP4_DECODE_CHUNK",
    "ODCR_STEP4_DECODE_CHUNK_LOG_EVERY",
    "ODCR_STEP4_DECODE_THREADS",
    "ODCR_STEP4_PARTIAL_FORMAT",
    "ODCR_STEP4_PERF_LOG_INTERVAL",
    "ODCR_STEP4_RCR_CONFIG_JSON",
    "ODCR_STEP4_RUNTIME_CONFIG_JSON",
    "ODCR_STEP4_MODE",
    "ODCR_STEP5_EMBEDDED_EVAL_LOG",
    "ODCR_STEP5_INIT_FLAN_STUB",
    "ODCR_SUMMARY_LOG",
    "ODCR_TRAINING_SEMANTIC_FINGERPRINT",
    "ODCR_TRAINING_STAGE",
    "ODCR_UPSTREAM_RESOLUTION_JSON",
    "RANK",
    "RUNNING_CPU_COUNT",
    "TOKENIZERS_PARALLELISM",
}

EVOLUTION_UNUSED_LOSS_ALLOWLIST = {
    "fca_cosine_alignment_loss",
    "flan_teacher_forcing_loss",
    "lci_score_consistency_loss",
}

ENV_READ_CALL_RE = re.compile(
    r"(?:os\.environ\.get|os\.getenv|getenv)\(\s*['\"](?P<name>[A-Z][A-Z0-9_]*)['\"]"
)

MASK_ANY_BRANCH_RE = re.compile(
    r"^\s*if\s+(?:not\s+)?(?:bool\()?[^:\n]*(?:mask|gate)[A-Za-z0-9_.\[\]]*\.any\([^)]*\)",
    re.IGNORECASE,
)

FIELD_WRITE_RE = re.compile(
    r"(?:df|frame|table|out|export|row|record)\s*\[\s*['\"](?P<field>[a-z][a-z0-9_]+)['\"]\s*\]\s*=",
)

ARTIFACT_WRITE_RE = re.compile(r"\b(torch\.save|to_csv|write_text|json\.dump|open\()", re.IGNORECASE)
ARTIFACT_READ_RE = re.compile(r"\b(torch\.load|read_csv|json\.load|open\()", re.IGNORECASE)
ARTIFACT_LINEAGE_RE = re.compile(
    r"lineage|fingerprint|schema_version|contract_version|validate_|manifest|metadata|payload_budget|invalid shard",
    re.IGNORECASE,
)
DEPRECATED_CONFIG_SNAPSHOT_FILENAMES = (
    "config_resolved.json",
    "resolved_config_snapshot.json",
    "config_snapshot.json",
)
DEPRECATED_CONFIG_SNAPSHOT_RE = re.compile(
    r"config_resolved\.json|resolved_config_snapshot\.json|config_snapshot\.json"
)
RUN_LOGGING_ACTIVE_PREFIXES = ("code/odcr.py", "code/odcr_core", "code/executors", "odcr")
OLD_LAYOUT_LOG_ACTIVE_FILES = (
    "odcr",
    "code/odcr.py",
    "code/odcr_core/logging_meta.py",
    "code/odcr_core/run_naming.py",
    "code/odcr_core/path_layout.py",
    "code/odcr_core/manifests.py",
    "code/odcr_core/runners.py",
    "code/odcr_core/preprocess_runtime.py",
    "code/odcr_core/preprocess_status.py",
    "code/odcr_core/config_resolver.py",
    "code/train_logging.py",
    "code/paths_config.py",
)

LEGACY_KILL_ABSENT_PATHS = (
    "code/odcr_core/config_loader.py",
    "code/odcr_core/training_preset_resolve.py",
    "code/odcr_core/stage_context.py",
    "code/odcr_core/step3_runtime.py",
    "code/odcr_core/step3_registry.py",
    "code/tools/async_eval_daemon.py",
    "scripts/run_stage.sh",
)
LEGACY_KILL_MODULE_NAMES = (
    "config_loader",
    "training_preset_resolve",
    "stage_context",
    "step3_runtime",
    "step3_registry",
)
LEGACY_KILL_ACTIVE_IMPORT_RE = re.compile(
    r"^\s*(?:"
    r"from\s+odcr_core\.(?:config_loader|training_preset_resolve|stage_context|step3_runtime|step3_registry)\b|"
    r"import\s+odcr_core\.(?:config_loader|training_preset_resolve|stage_context|step3_runtime|step3_registry)\b|"
    r"from\s+odcr_core\s+import\s+.*\b(?:config_loader|training_preset_resolve|stage_context|step3_runtime|step3_registry)\b|"
    r"from\s+(?:tools|code\.tools)\.async_eval_daemon\b|"
    r"import\s+(?:tools|code\.tools)\.async_eval_daemon\b|"
    r".*import_module\(\s*['\"](?:odcr_core\.(?:config_loader|training_preset_resolve|stage_context|step3_runtime|step3_registry)|(?:tools|code\.tools)\.async_eval_daemon)['\"]|"
    r"def\s+run_smoke_ddp\s*\(|"
    r".*\brun_smoke_ddp\s*\("
    r")"
)
LEGACY_KILL_DOC_RE = re.compile(
    r"config_loader\.py|training_preset_resolve\.py|stage_context\.py|"
    r"step3_runtime\.py|step3_registry\.py|async_eval_daemon\.py|"
    r"scripts/run_stage\.sh|run_smoke_ddp",
    re.IGNORECASE,
)
LEGACY_KILL_DOC_CONTEXT_RE = re.compile(
    r"legacy|retired|deleted|absent|forbidden|do not|must not|history|negative|removed|no longer|archived",
    re.IGNORECASE,
)

LEGACY_ACTIVE_RE = re.compile(
    r"presets/|shared[_-]?yaml|"
    r"AdvTrain|weak_adv|domain[_ -]?adv|adversarial training|"
    r"get\(\s*['\"](?:adv|eta|lambda_lci|lambda_fca)['\"]|"
    r"lambda_lci|lambda_fca|content_preserve_score|evidence_quality(?!_prior)|"
    r"entropy_filter|control_prompt|prompt\s*\+\s*|"
    r"add_argument\(\s*['\"]--lora[-_](?:r|alpha|dropout|target_modules)",
    re.IGNORECASE,
)
CONFIG_LOADER_ACTIVE_RE = re.compile(r"config_loader|load_resolved_config|instantiate_step3_preset")

EVOLUTION_INTERNAL_FIELD_ALLOWLIST = {
    "rerank_score",
    "v3_score_breakdown",
    "rule_v2_score_breakdown",
    "template_downweighted",
    "noisy_tail_downweighted",
    "training_runtime_config_fingerprint",
}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    text: str
    suggestion: str


@dataclass
class RuleResult:
    rule_id: str
    title: str
    status: str = "PASS"
    summary: str = ""
    findings: list[Finding] = field(default_factory=list)
    group: str = ""

    def __post_init__(self) -> None:
        if not self.group:
            self.group = RULE_GROUP_BY_ID.get(self.rule_id, "uncategorized")

    def fail(self, summary: str, findings: Iterable[Finding] = ()) -> None:
        self.status = "FAIL"
        self.summary = summary
        self.findings.extend(findings)

    def warn(self, summary: str, findings: Iterable[Finding] = ()) -> None:
        if self.status != "FAIL":
            self.status = "WARN"
            self.summary = summary
        self.findings.extend(findings)


@dataclass
class GuardrailReport:
    results: list[RuleResult]

    @property
    def ok(self) -> bool:
        return all(item.status != "FAIL" for item in self.results)

    @property
    def warnings(self) -> int:
        return sum(1 for item in self.results if item.status == "WARN")

    @property
    def failures(self) -> int:
        return sum(1 for item in self.results if item.status == "FAIL")


def _rel(path: Path, repo_root: Path = REPO_ROOT) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _is_under(rel: str, prefixes: Sequence[str]) -> bool:
    return any(rel == p.rstrip("/") or rel.startswith(p.rstrip("/") + "/") for p in prefixes)


def _is_legacy_preset_archive(rel: str) -> bool:
    return rel.startswith("_archive/legacy_presets_")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _json_load(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _text_lines(path: Path) -> list[str]:
    return _read(path).splitlines()


def _iter_repo_files(repo_root: Path, suffixes: tuple[str, ...] | None = None) -> Iterable[Path]:
    skip_dirs = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "data",
        "merged",
        "runs",
        "cache",
        "AI_analysis",
        "artifacts",
    }
    stack = [repo_root]
    while stack:
        cur = stack.pop()
        try:
            children = sorted(cur.iterdir(), reverse=True)
        except OSError:
            continue
        for child in children:
            rel = _rel(child, repo_root)
            if child.is_dir():
                if child.name in skip_dirs:
                    continue
                stack.append(child)
                continue
            if suffixes is not None and child.suffix not in suffixes:
                continue
            yield child


def _grep_file(
    path: Path,
    pattern: re.Pattern[str],
    repo_root: Path,
    *,
    ignore_comment_lines: bool = False,
) -> list[Finding]:
    hits: list[Finding] = []
    for idx, line in enumerate(_text_lines(path), start=1):
        if ignore_comment_lines:
            stripped = line.strip()
            if stripped.startswith(("#", '"""', "'''")):
                continue
        if pattern.search(line):
            hits.append(
                Finding(
                    path=_rel(path, repo_root),
                    line=idx,
                    text=line.strip()[:240],
                    suggestion="Route configuration through configs/odcr.yaml and code/odcr_core/config_resolver.py.",
                )
            )
    return hits


def _is_evolution_exempt_path(rel: str) -> bool:
    return rel in EVOLUTION_EXEMPT_FILES or _is_under(rel, EVOLUTION_EXEMPT_PREFIXES)


def _has_allow_marker(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "constant",
            "test-only",
            "test only",
            "for_test",
            "internal-only",
            "internal only",
            "no-op",
            "noop",
            "fail-fast",
            "fail fast",
            "retired",
            "deprecated",
            "history",
            "guardrail-only",
            "negative test",
        )
    )


def _iter_argparse_calls(text: str) -> Iterable[tuple[int, str]]:
    lines = text.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if "add_argument(" not in line:
            idx += 1
            continue
        start = idx
        chunk = [line]
        balance = line.count("(") - line.count(")")
        idx += 1
        while balance > 0 and idx < len(lines):
            chunk.append(lines[idx])
            balance += lines[idx].count("(") - lines[idx].count(")")
            idx += 1
        yield start + 1, "\n".join(chunk)


def _iter_python_function_blocks(text: str) -> Iterable[tuple[int, str, str]]:
    matches = list(re.finditer(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text, flags=re.MULTILINE))
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        yield text.count("\n", 0, match.start()) + 1, match.group(1), text[match.start() : end]


def _line_window(lines: Sequence[str], line_no: int, radius: int = 2) -> str:
    start = max(0, line_no - radius - 1)
    end = min(len(lines), line_no + radius)
    return "\n".join(lines[start:end])


def _scan_r042_text(rel: str, text: str) -> list[Finding]:
    if _is_evolution_exempt_path(rel):
        return []
    findings: list[Finding] = []
    for line_no, call in _iter_argparse_calls(text):
        flag_match = re.search(r"['\"](?P<flag>--[a-zA-Z0-9][a-zA-Z0-9_-]*)['\"]", call)
        if not flag_match:
            continue
        flag = flag_match.group("flag")
        default_match = re.search(r"default\s*=\s*(?P<default>[^,\)\n]+)", call)
        if not default_match:
            continue
        default_value = default_match.group("default").strip()
        if default_value in {"None", "[]"} or default_value.startswith("DEFAULT_"):
            continue
        if flag in EVOLUTION_ALLOWED_ARG_DEFAULT_FLAGS or _has_allow_marker(call):
            continue
        findings.append(
            Finding(
                rel,
                line_no,
                call.strip().replace("\n", " ")[:240],
                "Route new active parameters through configs/odcr.yaml, schema, resolver, source table, show/doctor, or mark constant/test-only.",
            )
        )
    return findings


def _scan_r043_text(rel: str, text: str, contract_text: str = "") -> list[Finding]:
    if _is_evolution_exempt_path(rel) or rel.endswith("data_contract.py") or rel.endswith("index_contract.py"):
        return []
    findings: list[Finding] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        match = FIELD_WRITE_RE.search(line)
        if not match:
            continue
        field = match.group("field")
        if field in EVOLUTION_INTERNAL_FIELD_ALLOWLIST:
            continue
        if field in contract_text or field.startswith("_") or _has_allow_marker(line):
            continue
        if not re.search(r"(score|route|bucket|weight|field|flag|contract|lineage|fingerprint|evidence|quality|stability|retention|shift|reliability)", field):
            continue
        findings.append(
            Finding(
                rel,
                idx,
                line.strip()[:240],
                "Add active CSV/export fields to data_contract or index_contract, producer, consumer, manifest/fingerprint, or mark internal-only.",
            )
        )
    return findings


def _scan_r044_text(rel: str, text: str) -> list[Finding]:
    if _is_evolution_exempt_path(rel):
        return []
    findings: list[Finding] = []
    for line_no, name, body in _iter_python_function_blocks(text):
        lowered_name = name.lower()
        if not re.search(r"cache|checkpoint|export", lowered_name):
            continue
        if _has_allow_marker(body) or ARTIFACT_LINEAGE_RE.search(body):
            continue
        if ARTIFACT_WRITE_RE.search(body):
            findings.append(
                Finding(
                    rel,
                    line_no,
                    f"def {name}(...)",
                    "Artifact writers for cache/checkpoint/export must write lineage/fingerprint/schema metadata.",
                )
            )
        elif ARTIFACT_READ_RE.search(body) and re.search(r"load|read|consume|reuse", lowered_name):
            findings.append(
                Finding(
                    rel,
                    line_no,
                    f"def {name}(...)",
                    "Artifact consumers for cache/checkpoint/export must validate lineage before reuse.",
                )
            )
    return findings


def _scan_r045_text(rel: str, text: str) -> list[Finding]:
    if _is_evolution_exempt_path(rel):
        return []
    findings: list[Finding] = []
    if rel.endswith(".sh") and rel != "odcr" and "has been retired" not in text:
        if re.search(r"\b(torchrun|python(?:3)?\s+code/(?:executors|tools|preprocess|step|train|eval))", text):
            findings.append(
                Finding(
                    rel,
                    1,
                    "shell dispatch to ODCR implementation",
                    "Route active scripts through ./odcr or python code/odcr.py; keep old shell entries retired.",
                )
            )
    return findings


def _scan_r046_text(rel: str, text: str) -> list[Finding]:
    if _is_evolution_exempt_path(rel):
        return []
    findings: list[Finding] = []
    lines = text.splitlines()
    for idx, line in enumerate(lines, start=1):
        for match in ENV_READ_CALL_RE.finditer(line):
            name = match.group("name")
            window = _line_window(lines, idx, radius=3)
            if name in EVOLUTION_ALLOWED_ENV_READS or name.startswith("ODCR_RESOLVED_"):
                continue
            if name in {
                "ODCR_DATA_DIR",
                "ODCR_MERGED_DATA_DIR",
                "ODCR_MODELS_DIR",
                "ODCR_STEP5_TEXT_MODEL",
                "ODCR_SENTENCE_EMBED_MODEL",
                "ODCR_EMBED_DIM",
            } and re.search(r"legacy|conflict|不得覆盖|cannot override|fail", window, re.IGNORECASE):
                continue
            findings.append(
                Finding(
                    rel,
                    idx,
                    line.strip()[:240],
                    "Do not add bare env config sources; use resolver-injected transport or explicit fail-fast conflict checks.",
                )
            )
    return findings


def _scan_r047_text(rel: str, text: str, all_active_text: str = "") -> list[Finding]:
    if _is_evolution_exempt_path(rel):
        return []
    findings: list[Finding] = []
    corpus = all_active_text or text
    for line_no, name, body in _iter_python_function_blocks(text):
        if "loss" not in name.lower() or name in EVOLUTION_UNUSED_LOSS_ALLOWLIST:
            continue
        if name.startswith(("check_", "_validate_", "_loss_weight", "odcr_log_")) or _has_allow_marker(body):
            continue
        occurrences = len(re.findall(r"\b" + re.escape(name) + r"\b", corpus))
        if occurrences <= 1:
            findings.append(
                Finding(
                    rel,
                    line_no,
                    f"def {name}(...)",
                    "New active losses must be called through the stage total-loss composer, or documented no-op/test-only.",
                )
            )
    return findings


def _scan_r048_text(rel: str, text: str) -> list[Finding]:
    if _is_evolution_exempt_path(rel):
        return []
    findings: list[Finding] = []
    lines = text.splitlines()
    for idx, line in enumerate(lines, start=1):
        if not MASK_ANY_BRANCH_RE.search(line):
            continue
        window = _line_window(lines, idx, radius=4)
        if _has_allow_marker(window) or "graph_tied_zero" in window:
            continue
        findings.append(
            Finding(
                rel,
                idx,
                line.strip()[:240],
                "Do not branch active DDP loss graphs on rank-local mask.any(); use graph-tied zero for empty masks.",
            )
        )
    return findings


def _scan_r049_text(rel: str, text: str) -> list[Finding]:
    if _is_evolution_exempt_path(rel):
        return []
    findings: list[Finding] = []
    lines = text.splitlines()
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith(("#", '"""', "'''")) or "``" in line:
            continue
        window = _line_window(lines, idx, radius=6)
        step5_legacy_input = rel in {
            "code/executors/step5_engine.py",
            "code/odcr_core/step5_innovation.py",
        } and re.search(r"style_markers|domain_style_id", line, re.IGNORECASE)
        config_loader_hit = False
        if CONFIG_LOADER_ACTIVE_RE.search(line) and rel in MAINLINE_FILES:
            config_loader_hit = bool(
                re.search(r"\b(from|import)\b|load_resolved_config\s*\(|config_loader\.", line)
            )
        if not LEGACY_ACTIVE_RE.search(line) and not step5_legacy_input and not config_loader_hit:
            continue
        if _has_allow_marker(window):
            continue
        findings.append(
            Finding(
                rel,
                idx,
                line.strip()[:240],
                "Delete, migrate, retire/fail-fast, or move legacy aliases/fields to docs/history; never add silent active fallbacks.",
            )
        )
    return findings


def scan_evolution_snippet(rule_id: str, snippet: str, *, path: str = "code/executors/new_feature.py") -> list[Finding]:
    """Return findings for a synthetic snippet under one evolution rule.

    Tests use this to prove the R042-R049 detectors catch representative future
    drift without needing to mutate the real repository.
    """

    scanners = {
        "R042": lambda: _scan_r042_text(path, snippet),
        "R043": lambda: _scan_r043_text(path, snippet, contract_text=""),
        "R044": lambda: _scan_r044_text(path, snippet),
        "R045": lambda: _scan_r045_text(path, snippet),
        "R046": lambda: _scan_r046_text(path, snippet),
        "R047": lambda: _scan_r047_text(path, snippet, all_active_text=snippet),
        "R048": lambda: _scan_r048_text(path, snippet),
        "R049": lambda: _scan_r049_text(path, snippet),
    }
    if rule_id not in scanners:
        raise KeyError(f"unsupported evolution snippet rule: {rule_id}")
    return scanners[rule_id]()


def _scan_r058_text(path: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        if not DEPRECATED_CONFIG_SNAPSHOT_RE.search(line):
            continue
        findings.append(
            Finding(
                path,
                idx,
                line.strip()[:240],
                "New run config snapshots must write meta/resolved_config.json and meta/source_table.json.",
            )
        )
    return findings


def scan_run_artifact_snippet(rule_id: str, snippet: str, *, path: str = "code/odcr_core/new_logging.py") -> list[Finding]:
    """Return findings for synthetic run-summary/logging drift snippets."""
    if rule_id != "R058":
        raise KeyError(f"unsupported run artifact snippet rule: {rule_id}")
    return _scan_r058_text(path, snippet)


LOGGING_OUTPUT_WRITE_RE = re.compile(
    r"\b(?:open|write_text|json\.dump|to_csv|torch\.save)\s*\(|"
    r"\.(?:log|jsonl|json|csv|txt)['\"]|"
    r"\b(?:metrics|cache|report|log(?:ging)?)\b",
    re.IGNORECASE,
)
LOGGING_ROLE_DECL_RE = re.compile(
    r"artifact[_ ]role|artifact_role|role\s*=|ArtifactRole|"
    r"path_layout\.|run_log_paths|metrics_filename|build_run_summary|"
    r"producer|consumer|retention",
    re.IGNORECASE,
)
RUN_FACING_OUTPUT_RE = re.compile(
    r"runs/|Path\(['\"]runs['\"]\)|manifest_dir|meta_dir|run_dir|"
    r"console\.log|full\.log|errors\.log|metrics\.jsonl|report",
    re.IGNORECASE,
)
RUN_SUMMARY_OR_LATEST_RE = re.compile(r"run_summary|latest\.json|latest_summary_path", re.IGNORECASE)
AI_ANALYSIS_FULL_LOG_RE = re.compile(
    r"AI_analysis[/\\][^'\"]*(?:full[_-]?train|train[_-]?full|full\.log|training\.log)|"
    r"(?:full_log|full\.log)[^'\"]*AI_analysis|"
    r"copy(?:file)?\([^)\n]*(?:full\.log|training\.log)[^)\n]*AI_analysis",
    re.IGNORECASE,
)
CONSOLE_DUMP_RE = re.compile(
    r"print\([^)\n]*(?:resolved_config|source_table|field_sources|config_snapshot|"
    r"ODCR One-Control Guardrails: PASS|\[PASS\]\s*R\d{3})",
    re.IGNORECASE,
)
BANNED_LOG_DEST_RE = re.compile(
    r"(?:^|[\"'/])(?:data|merged)/[^\"'\n]*(?:\.log|log\.|logs?/)|"
    r"(?:Path\(['\"](?:data|merged)['\"]\)[^\n]*(?:\.log|log))|"
    r"(?:^|[\"'/])logs/[^\"'\n]*(?:\.log|log\.|logs?/)|"
    r"code/log\.out|CODE_DIR,\s*['\"]log\.out['\"]",
    re.IGNORECASE,
)
OLD_DEFAULT_LOG_WRITE_RE = re.compile(
    r"(?:open|FileHandler|basicConfig|write_text|touch|mkdir|makedirs)\s*\([^)\n]*"
    r"(?:[\"']logs[\"']|[\"']/logs/|[\"']logs/|code/log\.out|CODE_DIR\s*,\s*[\"']log\.out[\"'])",
    re.IGNORECASE,
)
TAIL_OLD_FALLBACK_RE = re.compile(
    r"legacy_parent|no numeric run directories|candidates\s*=|iterdir\(\)|"
    r"[\"'](?:train|eval|step4)\.log[\"']|[\"']shell_logs[\"']|"
    r"code/log\.out|[\"']logs/|nohup[^\"'\n]*\.log|fallback\.log|mirror\.log|timestamp_timestamp\.log",
    re.IGNORECASE,
)
OLD_FALLBACK_LOG_RE = re.compile(
    r"nohup[^\"'\n]*\.log|fallback\.log|mirror\.log|timestamp_timestamp\.log|"
    r"_launcher_logs|_adhoc_logs|_legacy_logs",
    re.IGNORECASE,
)
AI_ANALYSIS_ACTIVE_MIRROR_RE = re.compile(
    r"AI_analysis[/\\][^\"'\n]*(?:full\.log|full[_-]?train|training\.log)|"
    r"(?:copyfile|copy2|shutil\.copy)[^)\n]*(?:full\.log|training\.log)[^)\n]*AI_analysis",
    re.IGNORECASE,
)
DATA_MERGED_LOG_WRITE_RE = re.compile(
    r"(?:open|FileHandler|basicConfig|write_text|touch|mkdir|makedirs)\s*\([^)\n]*"
    r"(?:[\"']data/[^\"'\n]*\.log|[\"']merged/[^\"'\n]*\.log|"
    r"Path\([\"']data[\"']\)[^)\n]*\.log|Path\([\"']merged[\"']\)[^)\n]*\.log)",
    re.IGNORECASE,
)


def _logging_artifact_exempt_path(rel: str) -> bool:
    return _is_under(rel, ("AI_analysis", "_archive", "docs/history", "code/tests"))


def _scan_logging_artifact_text(rule_id: str, text: str, *, path: str) -> list[Finding]:
    if _logging_artifact_exempt_path(path):
        return []
    findings: list[Finding] = []
    lines = text.splitlines()
    if rule_id == "R068":
        for idx, line in enumerate(lines, start=1):
            if not LOGGING_OUTPUT_WRITE_RE.search(line):
                continue
            window = _line_window(lines, idx, radius=4)
            if LOGGING_ROLE_DECL_RE.search(window) or _has_allow_marker(window):
                continue
            findings.append(
                Finding(
                    path,
                    idx,
                    line.strip()[:240],
                    "Declare artifact role, output directory, producer, consumer, retention policy, and AI_analysis copy policy.",
                )
            )
    elif rule_id == "R069":
        for idx, line in enumerate(lines, start=1):
            if not RUN_FACING_OUTPUT_RE.search(line):
                continue
            window = _line_window(lines, idx, radius=6)
            if RUN_SUMMARY_OR_LATEST_RE.search(window) or _has_allow_marker(window):
                continue
            findings.append(
                Finding(
                    path,
                    idx,
                    line.strip()[:240],
                    "Run-facing outputs must update meta/run_summary.json/latest.json or explicitly declare why not.",
                )
            )
    elif rule_id == "R070":
        for idx, line in enumerate(lines, start=1):
            if AI_ANALYSIS_FULL_LOG_RE.search(line):
                findings.append(
                    Finding(
                        path,
                        idx,
                        line.strip()[:240],
                        "AI_analysis may hold audit evidence and digests, but must not mirror full training logs.",
                    )
                )
    elif rule_id == "R071":
        for idx, line in enumerate(lines, start=1):
            if not CONSOLE_DUMP_RE.search(line):
                continue
            window = _line_window(lines, idx, radius=4)
            if re.search(r"verbose|debug|CONSOLE_LEVEL_VERBOSE|CONSOLE_LEVEL_DEBUG", window, re.IGNORECASE):
                continue
            findings.append(
                Finding(
                    path,
                    idx,
                    line.strip()[:240],
                    "Keep default console summary-level; full config/source/per-rule PASS detail belongs in files or verbose/debug.",
                )
            )
    elif rule_id == "R072":
        for idx, line in enumerate(lines, start=1):
            if BANNED_LOG_DEST_RE.search(line):
                findings.append(
                    Finding(
                        path,
                        idx,
                        line.strip()[:240],
                        "New log paths must not target data/, merged/, top-level logs/, or code/log.out.",
                    )
                )
    else:
        raise KeyError(f"unsupported logging artifact snippet rule: {rule_id}")
    return findings


def scan_logging_artifact_snippet(
    rule_id: str,
    snippet: str,
    *,
    path: str = "code/odcr_core/new_logging.py",
) -> list[Finding]:
    """Return findings for synthetic logging/output governance snippets."""

    return _scan_logging_artifact_text(rule_id, snippet, path=path)


def _old_layout_log_exempt_path(rel: str) -> bool:
    return _is_under(rel, ("AI_analysis", "_archive", "docs", "code/tests")) or rel in {"README.md", "AGENTS.md"}


def _scan_old_layout_log_text(rule_id: str, text: str, *, path: str) -> list[Finding]:
    if _old_layout_log_exempt_path(path):
        return []
    if rule_id == "R084":
        pattern = OLD_DEFAULT_LOG_WRITE_RE
        suggestion = "Active log writers must use runs/<stage>/<unit>/<run_id>/meta, not top-level logs/ or code/log.out."
    elif rule_id == "R085":
        pattern = TAIL_OLD_FALLBACK_RE
        suggestion = "odcr tail must resolve only latest.json -> run_summary.json -> meta/{console.log,full.log,errors.log}."
    elif rule_id == "R086":
        pattern = OLD_FALLBACK_LOG_RE
        suggestion = "Retire active nohup/fallback/mirror/timestamp log fallback paths."
    elif rule_id == "R087":
        pattern = AI_ANALYSIS_ACTIVE_MIRROR_RE
        suggestion = "AI_analysis may hold audit digests, not active full-log mirrors."
    elif rule_id == "R088":
        pattern = DATA_MERGED_LOG_WRITE_RE
        suggestion = "Logs must stay under runs/.../meta, never data/ or merged/."
    else:
        raise KeyError(f"unsupported old-layout log snippet rule: {rule_id}")
    findings: list[Finding] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        if pattern.search(line):
            findings.append(Finding(path, idx, line.strip()[:240], suggestion))
    return findings


def scan_old_layout_log_snippet(
    rule_id: str,
    snippet: str,
    *,
    path: str = "code/odcr_core/new_logging.py",
) -> list[Finding]:
    """Return findings for old-layout log/cache cleanup governance snippets."""

    return _scan_old_layout_log_text(rule_id, snippet, path=path)


def _active_run_logging_paths(repo_root: Path) -> list[Path]:
    paths: list[Path] = []
    for prefix in RUN_LOGGING_ACTIVE_PREFIXES:
        base = repo_root / prefix
        if base.is_file():
            paths.append(base)
        elif base.is_dir():
            paths.extend(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)
    skip = {
        repo_root / "code" / "tools" / "check_one_control_guardrails.py",
    }
    return sorted({p.resolve() for p in paths if p.resolve() not in {s.resolve() for s in skip}})


def _check_mainline_preset_reads(repo_root: Path) -> RuleResult:
    result = RuleResult("R001", "mainline must not read presets/")
    findings: list[Finding] = []
    for rel in MAINLINE_FILES:
        path = repo_root / rel
        if path.is_file():
            findings.extend(_grep_file(path, PRESET_READ_RE, repo_root, ignore_comment_lines=True))
    if findings:
        result.fail("Mainline files contain preset path reads or references.", findings)
        return result

    legacy_hits: list[Finding] = []
    unmarked_legacy: list[Finding] = []
    for rel in LEGACY_PRESET_READER_ALLOWLIST:
        path = repo_root / rel
        if path.is_file():
            hits = _grep_file(path, PRESET_READ_RE, repo_root, ignore_comment_lines=True)[:5]
            legacy_hits.extend(hits)
            if hits and LEGACY_PRESET_READER_MARKER not in _read(path):
                unmarked_legacy.append(
                    Finding(
                        rel,
                        1,
                        "missing deprecated/internal marker",
                        "Mark legacy-only preset readers explicitly or remove preset reads.",
                    )
                )
    if unmarked_legacy:
        result.fail(
            "Allowlisted legacy preset readers must be explicitly marked deprecated/internal.",
            unmarked_legacy,
        )
    elif legacy_hits:
        result.summary = (
            "No mainline preset readers found; deprecated/internal preset readers are explicitly isolated."
        )
    else:
        result.summary = "No mainline preset readers found."
    return result


def _check_scattered_yaml_env(repo_root: Path) -> RuleResult:
    result = RuleResult("R002", "no scattered yaml/env config")
    offenders: list[Finding] = []
    for path in _iter_repo_files(repo_root):
        rel = _rel(path, repo_root)
        if _is_under(rel, ("AI_analysis",)) or _is_legacy_preset_archive(rel):
            continue
        if rel in ALLOWED_YAML_ENV:
            continue
        name = path.name
        if path.suffix in (".yaml", ".yml", ".env") or name == ".env" or name.endswith(".env"):
            offenders.append(
                Finding(
                    rel,
                    1,
                    name,
                    "Move main configuration into configs/odcr.yaml or add an explicit allowlist entry with justification.",
                )
            )
    if offenders:
        result.fail("Found yaml/env files outside the One-Control allowlist.", offenders)
    else:
        result.summary = "Only configs/odcr.yaml plus archived/analysis materials are present."
    return result


def _check_shell_entrypoints(repo_root: Path) -> RuleResult:
    result = RuleResult("R003", "./odcr is the only shell entry")
    offenders: list[Finding] = []
    extra_shell: list[Finding] = []
    allowed_hook_helpers = {".codex/hooks/odcr_post_edit_stop.sh"}
    for path in _iter_repo_files(repo_root, suffixes=(".sh",)):
        rel = _rel(path, repo_root)
        if _is_under(rel, ("AI_analysis", "_archive")):
            continue
        if rel in allowed_hook_helpers:
            continue
        if rel.startswith("scripts/entrypoints/") or path.name in BANNED_SHELL_NAMES:
            offenders.append(
                Finding(rel, 1, path.name, "Use ./odcr and code/odcr.py; do not add stage shell entrypoints.")
            )
        else:
            extra_shell.append(
                Finding(rel, 1, path.name, "Confirm this shell script is not a user-visible ODCR entrypoint.")
            )
    if offenders:
        result.fail("Found banned old-style shell entrypoints.", offenders)
    elif extra_shell:
        result.warn("Non-entry shell scripts exist and should remain non-user-facing.", extra_shell)
    else:
        result.summary = "No active step/train/eval shell entrypoints found."
    return result


def _check_retired_run_stage(repo_root: Path) -> RuleResult:
    result = RuleResult("R004", "legacy scripts/run_stage.sh must be absent")
    path = repo_root / "scripts" / "run_stage.sh"
    if path.exists():
        result.fail(
            "scripts/run_stage.sh must not remain as a compatibility shim.",
            [Finding("scripts/run_stage.sh", 1, "present", "Delete the legacy shell entrypoint; use ./odcr.")],
        )
    else:
        result.summary = "Legacy run_stage shell entrypoint is absent."
    return result


def _check_legacy_kill_pass_absence(repo_root: Path) -> RuleResult:
    result = RuleResult("R095", "legacy kill-pass files and imports must stay absent")
    findings: list[Finding] = []

    for rel in LEGACY_KILL_ABSENT_PATHS:
        if (repo_root / rel).exists():
            findings.append(
                Finding(
                    rel,
                    1,
                    "present",
                    "Delete the retired file; do not keep an empty shell, compatibility shim, or fail-fast stub.",
                )
            )

    for path in _iter_repo_files(repo_root, suffixes=(".py", ".sh")):
        rel = _rel(path, repo_root)
        if _is_under(rel, ("AI_analysis", "_archive", "code/tests")):
            continue
        if rel == "code/tools/check_one_control_guardrails.py":
            continue
        if not (rel == "odcr" or rel.startswith("code/") or rel.startswith("scripts/")):
            continue
        for idx, line in enumerate(_text_lines(path), start=1):
            if LEGACY_KILL_ACTIVE_IMPORT_RE.search(line):
                findings.append(
                    Finding(
                        rel,
                        idx,
                        line.strip()[:240],
                        "Remove the active legacy import/call and route through config_resolver or the current ODCR command path.",
                    )
                )

    doc_paths = [repo_root / "README.md", repo_root / "AGENTS.md"]
    docs_dir = repo_root / "docs"
    if docs_dir.is_dir():
        doc_paths.extend(sorted(docs_dir.rglob("*.md")))
    for path in doc_paths:
        if not path.is_file():
            continue
        rel = _rel(path, repo_root)
        lines = _text_lines(path)
        for idx, line in enumerate(lines, start=1):
            context = _line_window(lines, idx, radius=3)
            if LEGACY_KILL_DOC_RE.search(line) and not LEGACY_KILL_DOC_CONTEXT_RE.search(context):
                findings.append(
                    Finding(
                        rel,
                        idx,
                        line.strip()[:240],
                        "Docs may mention deleted legacy names only as history, negative guidance, or absence checks.",
                    )
                )

    if findings:
        result.fail("Legacy kill-pass residue is still present or active.", findings)
    else:
        result.summary = (
            "Deleted legacy modules/tools are absent, active imports are clear, and docs mention them only negatively."
        )
    return result


def _check_config_top_level(repo_root: Path) -> RuleResult:
    result = RuleResult("R005", "configs/odcr.yaml top-level contract")
    path = repo_root / "configs" / "odcr.yaml"
    if not path.is_file():
        result.fail("configs/odcr.yaml is missing.", [Finding("configs/odcr.yaml", 1, "missing", "Restore the canonical config.")])
        return result
    try:
        import yaml
    except ImportError as exc:
        result.fail(
            "PyYAML is required for guardrail config validation.",
            [Finding("configs/odcr.yaml", 1, repr(exc), "Install PyYAML in the ODCR environment.")],
        )
        return result
    raw = yaml.safe_load(_read(path))
    if not isinstance(raw, dict):
        result.fail("configs/odcr.yaml must be a mapping.", [Finding("configs/odcr.yaml", 1, "not mapping", "Use mapping top-level blocks.")])
        return result
    missing = [key for key in TOP_LEVEL_BLOCKS if key not in raw]
    extra = sorted(set(raw) - set(TOP_LEVEL_BLOCKS))
    if missing or extra:
        result.fail(
            f"Top-level config blocks mismatch; missing={missing}, extra={extra}.",
            [Finding("configs/odcr.yaml", 1, ", ".join(raw.keys()), "Keep exactly the One-Control top-level blocks.")],
        )
    else:
        result.summary = "Required top-level blocks are present and no extras were found."
    return result


def _check_batch_formula_validation(repo_root: Path) -> RuleResult:
    result = RuleResult("R006", "batch formula must be schema/resolver validated")
    paths = [
        repo_root / "code" / "odcr_core" / "config_schema.py",
        repo_root / "code" / "odcr_core" / "config_resolver.py",
    ]
    text = "\n".join(_read(path) for path in paths if path.is_file())
    required_terms = ("batch_size", "per_gpu_batch_size", "ddp_world_size", "odcr_no_accum/1")
    has_terms = all(term in text for term in required_terms)
    has_formula = (
        "_validate_train_batch" in text
        and "per_gpu * int(ddp_world_size)" in text
        and "global_batch_size = per_gpu_batch_size * ddp_world_size" in text
        and "_validate_config_shape" in text
    )
    has_failfast = "OneControlConfigError" in text and "batch formula failed" in text
    if not (has_terms and has_formula and has_failfast):
        result.fail(
            "Batch formula validation is missing or incomplete.",
            [
                Finding(
                    "code/odcr_core/config_resolver.py",
                    1,
                    "global_batch_size = per_gpu_batch_size * ddp_world_size",
                    "Validate no-accum formula in resolver/schema and fail with OneControlConfigError.",
                )
            ],
        )
    else:
        result.summary = "Resolver validates ODCR no-accum per-GPU/global batch semantics."
    return result


def _check_logs_not_data(repo_root: Path) -> RuleResult:
    result = RuleResult("R007", "logs must not target data/ or merged/")
    pattern = re.compile(
        r"(log_dir|log_file|output_log|shell_logs|train\.log|eval\.log).*(data/|merged/)|"
        r"(data/|merged/).*(log_dir|log_file|output_log|shell_logs|\.log)",
        re.IGNORECASE,
    )
    findings: list[Finding] = []
    for path in _iter_repo_files(repo_root, suffixes=(".py", ".sh")):
        rel = _rel(path, repo_root)
        if _is_under(rel, ("AI_analysis", "_archive", "code/tests")) or rel == "code/tools/check_one_control_guardrails.py":
            continue
        findings.extend(_grep_file(path, pattern, repo_root))
    if findings:
        result.fail("Found log paths pointing at data/ or merged/.", findings)
    else:
        result.summary = "No obvious log targets under data/ or merged/."
    return result


def _check_data_not_runs(repo_root: Path) -> RuleResult:
    result = RuleResult("R008", "data artifacts must not be written to runs/")
    pattern = re.compile(
        r"runs/[^'\"]*(aug_|processed|train\.csv|valid\.csv|test\.csv|embedding|domain_.*\.npy|"
        r"user_.*profiles|item_.*profiles)",
        re.IGNORECASE,
    )
    findings: list[Finding] = []
    for path in _iter_repo_files(repo_root, suffixes=(".py", ".yaml", ".yml", ".sh")):
        rel = _rel(path, repo_root)
        if _is_under(rel, ("AI_analysis", "_archive")):
            continue
        for finding in _grep_file(path, pattern, repo_root):
            lowered = finding.text.lower()
            if "cache" in lowered or "/meta" in lowered or "shell_logs" in lowered:
                continue
            findings.append(finding)
    if findings:
        result.fail("Found likely dataset artifacts under runs/.", findings)
    else:
        result.summary = "No obvious data artifact writes to runs/."
    return result


def _check_parameter_drift(repo_root: Path) -> RuleResult:
    result = RuleResult("R009", "new parameters must flow through One-Control")
    config_text = _read(repo_root / "configs" / "odcr.yaml") if (repo_root / "configs" / "odcr.yaml").is_file() else ""
    resolver_text = _read(repo_root / "code" / "odcr_core" / "config_resolver.py")
    expected = ("batch_size", "lr", "epochs", "generate_top_p", "rerank")
    missing = [term for term in expected if term not in config_text or term not in resolver_text]
    findings: list[Finding] = []
    odcr_path = repo_root / "code" / "odcr.py"
    if odcr_path.is_file():
        findings.extend(_grep_file(odcr_path, TRAINING_ARG_RE, repo_root))
    if missing:
        result.fail(
            f"Common config fields are not represented in both configs/odcr.yaml and config_resolver.py: {missing}.",
            [
                Finding(
                    "configs/odcr.yaml",
                    1,
                    ", ".join(missing),
                    "Add new public parameters to configs/odcr.yaml, schema/resolver, show, doctor, and tests.",
                )
            ],
        )
    elif findings:
        result.fail(
            "code/odcr.py exposes direct training/generation argparse knobs; use --set and resolver instead.",
            findings,
        )
    else:
        legacy_env_hits: list[Finding] = []
        legacy_patterns = re.compile(r"ODCR_(TRAIN_BATCH_SIZE|OPT_BATCH_SIZE|EPOCHS)|generate_top_p\s*=\s*0\.9")
        for rel in ("code/config.py", "code/executors/step3_train_core.py", "code/executors/decode_controller.py"):
            path = repo_root / rel
            if path.is_file():
                legacy_env_hits.extend(_grep_file(path, legacy_patterns, repo_root)[:3])
        if legacy_env_hits:
            result.warn(
                "Legacy/internal defaults still exist; public One-Control entry remains configs/odcr.yaml -> resolver.",
                legacy_env_hits[:8],
            )
        else:
            result.summary = "Common fields are represented in config and resolver, with no direct public argparse bypass."
    return result


def _check_step3_live_semantics(repo_root: Path) -> RuleResult:
    result = RuleResult("R010", "step3 live semantics must stay structured")
    scan = (
        "code/odcr_core/logging_meta.py",
        "code/executors/step3_train_core.py",
        "docs/AI_PROJECT_CANONICAL.md",
        "docs/ODCR_ARCHITECTURE_CONTRACT.md",
        "AGENTS.md",
    )
    findings: list[Finding] = []
    for rel in scan:
        path = repo_root / rel
        if path.is_file():
            findings.extend(_grep_file(path, STEP3_RETIRED_SEMANTIC_RE, repo_root, ignore_comment_lines=False))
    if findings:
        result.fail(
            "Step3 live files still contain retired adversarial/AdvTrain semantics.",
            findings,
        )
    else:
        result.summary = "Step3 live labels describe structured shared/specific training."
    return result


def _check_step3_typed_bridge_retired(repo_root: Path) -> RuleResult:
    result = RuleResult("R011", "step3 typed bridge must stay deleted")
    runtime = repo_root / "code" / "odcr_core" / "step3_runtime.py"
    test_file = repo_root / "code" / "tests" / "test_step3_control_plane.py"
    runners = repo_root / "code" / "odcr_core" / "runners.py"
    findings: list[Finding] = []

    if runtime.exists():
        findings.append(
            Finding(
                "code/odcr_core/step3_runtime.py",
                1,
                "present",
                "Delete the retired typed bridge from active code; do not keep compatibility shims.",
            )
        )

    if test_file.is_file():
        for hit in _grep_file(test_file, STEP3_TYPED_BRIDGE_RE, repo_root):
            findings.append(
                Finding(
                    hit.path,
                    hit.line,
                    hit.text,
                    "Resolve Step3 tests through config_resolver/odcr.py only.",
                )
            )
    if runners.is_file():
        for hit in _grep_file(runners, STEP3_RUNTIME_RECONNECT_RE, repo_root):
            findings.append(
                Finding(
                    hit.path,
                    hit.line,
                    hit.text,
                    "Do not reconnect the retired Step3 typed runtime from active runners.",
                )
            )

    if findings:
        result.fail("Found Step3 typed bridge or presets-era loader residue in active surfaces.", findings)
    else:
        result.summary = "Step3 typed bridge file is absent and active tests/runners use One-Control."
    return result


def _check_step4_legacy_fields_not_primary(repo_root: Path) -> RuleResult:
    result = RuleResult("R012", "step4 must not read legacy routing fields as primary")
    active_files = (
        "code/odcr_core/odcr_cf_routing.py",
        "code/odcr_core/step4_training_export.py",
        "code/executors/step4_engine.py",
        "code/executors/step5_engine.py",
    )
    pattern = re.compile(
        r"content_preserve_score|evidence_quality(?!_prior)|"
        r"content_keywords|content_aspects|content_entities|style_markers|"
        r"sentiment_style|domain_style_id|template_family|length_style_bucket|"
        r"route_hint|old_route",
        re.IGNORECASE,
    )
    findings: list[Finding] = []
    for rel in active_files:
        path = repo_root / rel
        if path.is_file():
            findings.extend(_grep_file(path, pattern, repo_root))
    if findings:
        result.fail(
            "Step4/Step5 active path still references legacy Step4/preprocess detail fields as active inputs.",
            findings,
        )
    else:
        result.summary = "Step4/Step5 active path uses canonical RCR/posterior fields only."
    return result


def _check_step4_rcr_one_control(repo_root: Path) -> RuleResult:
    result = RuleResult("R013", "step4 rcr parameters must flow through One-Control")
    config = _read(repo_root / "configs" / "odcr.yaml") if (repo_root / "configs" / "odcr.yaml").is_file() else ""
    resolver = _read(repo_root / "code" / "odcr_core" / "config_resolver.py")
    runners = _read(repo_root / "code" / "odcr_core" / "runners.py")
    engine = _read(repo_root / "code" / "executors" / "step4_engine.py")
    required = {
        "configs/odcr.yaml": ("step4:", "rcr:", "cf_reliability_weights", "route_scorer", "sample_weight_hint"),
        "code/odcr_core/config_resolver.py": ("_resolve_step4_rcr_config", "step4_rcr_config_json", "step4.rcr"),
        "code/odcr_core/runners.py": ("ODCR_STEP4_RCR_CONFIG_JSON", "step4_rcr_config_json"),
        "code/executors/step4_engine.py": ("ODCFRoutingConfig.from_env(require=True)", "rcr_config=rcr_config"),
    }
    texts = {
        "configs/odcr.yaml": config,
        "code/odcr_core/config_resolver.py": resolver,
        "code/odcr_core/runners.py": runners,
        "code/executors/step4_engine.py": engine,
    }
    findings: list[Finding] = []
    for rel, terms in required.items():
        text = texts[rel]
        missing = [term for term in terms if term not in text]
        if missing:
            findings.append(
                Finding(
                    rel,
                    1,
                    "missing: " + ", ".join(missing),
                    "Route Step4 RCR values through configs/odcr.yaml -> config_resolver -> resolved env.",
                )
            )
    if findings:
        result.fail("Step4 RCR config is not fully wired through One-Control.", findings)
    else:
        result.summary = "Step4 RCR weights, thresholds, buckets, train_keep, sample weights, and export fields are One-Control wired."
    return result


def _check_step4_route_contract_posterior(repo_root: Path) -> RuleResult:
    result = RuleResult("R014", "step4 route fields must be documented as posterior")
    path = repo_root / "code" / "odcr_core" / "index_contract.py"
    text = _read(path) if path.is_file() else ""
    required = (
        "preprocess_route_scorer_prior",
        "preprocess_route_explainer_prior",
        '"route_scorer": (',
        "Step4 posterior binary scorer-clean path decision",
        "Step4 posterior binary explainer-rich path decision",
        "prior_posterior_boundary",
    )
    missing = [term for term in required if term not in text]
    if missing:
        result.fail(
            "Step4 route prior/posterior boundary is missing from the export contract.",
            [
                Finding(
                    "code/odcr_core/index_contract.py",
                    1,
                    "missing: " + ", ".join(missing),
                    "Document preprocess priors separately from Step4 posterior route decisions.",
                )
            ],
        )
    else:
        result.summary = "index_contract documents route_scorer/route_explainer as Step4 posterior decisions."
    return result


def _check_step4_entropy_auxiliary_only(repo_root: Path) -> RuleResult:
    result = RuleResult("R015", "entropy/text quality must stay auxiliary")
    active_files = (
        "code/odcr_core/odcr_cf_routing.py",
        "code/odcr_core/step4_training_export.py",
        "code/odcr_core/index_contract.py",
        "docs/ODCR_ARCHITECTURE_CONTRACT.md",
        "docs/AI_PROJECT_CANONICAL.md",
        "AGENTS.md",
    )
    banned = re.compile(r"entropy_filter|entropy_only", re.IGNORECASE)
    findings: list[Finding] = []
    for rel in active_files:
        path = repo_root / rel
        if path.is_file():
            findings.extend(_grep_file(path, banned, repo_root))
    contract = _read(repo_root / "code" / "odcr_core" / "index_contract.py")
    if "Auxiliary generation uncertainty" not in contract or "never the" not in contract:
        findings.append(
            Finding(
                "code/odcr_core/index_contract.py",
                1,
                "entropy_score definition missing auxiliary wording",
                "State that entropy_score is auxiliary and never the primary RCR decision signal.",
            )
        )
    docs = _read(repo_root / "docs" / "ODCR_ARCHITECTURE_CONTRACT.md")
    if "entropy/text hygiene" not in docs or "auxiliary" not in docs:
        findings.append(
            Finding(
                "docs/ODCR_ARCHITECTURE_CONTRACT.md",
                1,
                "missing entropy/text auxiliary contract",
                "Document entropy/text hygiene as auxiliary only.",
            )
        )
    if findings:
        result.fail("Found entropy-only/filter semantics or missing auxiliary-only contract.", findings)
    else:
        result.summary = "Entropy/text quality are documented and guarded as auxiliary Step4 signals."
    return result


def _check_step5_rcr_posterior_consumption(repo_root: Path) -> RuleResult:
    result = RuleResult("R016", "step5 must consume Step4 RCR posterior fields")
    engine = _read(repo_root / "code" / "executors" / "step5_engine.py")
    innov = _read(repo_root / "code" / "odcr_core" / "step5_innovation.py")
    config = _read(repo_root / "configs" / "odcr.yaml")
    required = (
        "route_scorer",
        "route_explainer",
        "cf_reliability_score",
        "content_retention_score",
        "rating_stability_score",
        "uncertainty_score",
        "confidence_bucket",
        "sample_weight_hint",
    )
    missing = [term for term in required if term not in engine or term not in innov]
    cfg_missing = [term for term in ("lci:", "uci:", "ccv:", "fca:") if term not in config]
    if missing or cfg_missing:
        result.fail(
            "Step5 RCR posterior consumption or One-Control blocks are incomplete.",
            [
                Finding(
                    "code/executors/step5_engine.py",
                    1,
                    f"missing={missing}, cfg_missing={cfg_missing}",
                    "Consume Step4 RCR posterior fields through Step5A/B gates and configure them under step5.*.",
                )
            ],
        )
    else:
        result.summary = "Step5A/B consume Step4 RCR posterior fields through explicit gates."
    return result


def _check_step5_lci_called(repo_root: Path) -> RuleResult:
    result = RuleResult("R017", "LCI must enter Step5A total loss")
    engine = _read(repo_root / "code" / "executors" / "step5_engine.py")
    innov = _read(repo_root / "code" / "odcr_core" / "step5_innovation.py")
    required = (
        "build_step5a_scorer_gate",
        "lci_score_invariance_loss",
        "lci_bundle.lci_weighted_loss",
        "uci_weight",
        "loss_lci_weighted",
    )
    missing = [term for term in required if term not in engine and term not in innov]
    if missing:
        result.fail(
            "LCI/UCI Step5A active path is missing required calls/logging.",
            [
                Finding(
                    "code/executors/step5_engine.py",
                    1,
                    "missing: " + ", ".join(missing),
                    "Call LCI loss and add its weighted value to Step5A total loss.",
                )
            ],
        )
    else:
        result.summary = "LCI is built from Step5A gate weights and contributes lci_weighted_loss to total loss."
    return result


def _check_step5_ccv_packet(repo_root: Path) -> RuleResult:
    result = RuleResult("R018", "CCV must use explicit control packet")
    engine = _read(repo_root / "code" / "executors" / "step5_engine.py")
    innov = _read(repo_root / "code" / "odcr_core" / "step5_innovation.py")
    required = (
        "CCVControlPacket",
        "build_ccv_control_packet",
        "ccv_control_adapter",
        "numeric_controls",
        "content_evidence_ids",
        "style_evidence_ids",
    )
    missing = [term for term in required if term not in engine and term not in innov]
    banned = re.compile(r"control_prompt|prompt\s*\+\s*|soft prompt 拼", re.IGNORECASE)
    findings = []
    for rel in ("code/executors/step5_engine.py", "code/odcr_core/step5_innovation.py"):
        path = repo_root / rel
        if path.is_file():
            findings.extend(_grep_file(path, banned, repo_root, ignore_comment_lines=True))
    if missing or findings:
        findings = findings or [
            Finding(
                "code/odcr_core/step5_innovation.py",
                1,
                "missing: " + ", ".join(missing),
                "Build a structured CCVControlPacket and consume it through the verbalizer control adapter.",
            )
        ]
        result.fail("CCV is not clearly implemented as a structured control packet.", findings)
    else:
        result.summary = "CCV uses structured control tensors and a verbalizer control adapter, not prompt concatenation."
    return result


def _check_step5_fca_evidence_basis(repo_root: Path) -> RuleResult:
    result = RuleResult("R019", "FCA must align scorer/explainer evidence bases")
    innov = _read(repo_root / "code" / "odcr_core" / "step5_innovation.py")
    engine = _read(repo_root / "code" / "executors" / "step5_engine.py")
    required = (
        "evidence_basis_fca_loss",
        "scorer_evidence_basis",
        "explainer_evidence_basis",
        "content_evidence_latent",
        "fca_bundle.fca_weighted_loss",
    )
    missing = [term for term in required if term not in innov and term not in engine]
    if missing:
        result.fail(
            "FCA evidence-basis alignment is missing or not wired into total loss.",
            [
                Finding(
                    "code/odcr_core/step5_innovation.py",
                    1,
                    "missing: " + ", ".join(missing),
                    "Align scorer evidence basis with explainer evidence basis and add weighted FCA loss.",
                )
            ],
        )
    else:
        result.summary = "FCA aligns explicit scorer/explainer evidence bases and contributes weighted loss."
    return result


def _check_step5_legacy_fields_not_primary(repo_root: Path) -> RuleResult:
    result = RuleResult("R020", "step5 must not use legacy fields as primary inputs")
    active_files = (
        "code/executors/step5_engine.py",
        "code/odcr_core/step5_innovation.py",
    )
    pattern = re.compile(
        r"content_preserve_score|evidence_quality(?!_prior)|entropy_filter|"
        r"entropy_only|route_hint|old_route|content_keywords|content_aspects|"
        r"content_entities|style_markers|sentiment_style|domain_style_id|"
        r"template_family|length_style_bucket",
        re.IGNORECASE,
    )
    findings: list[Finding] = []
    for rel in active_files:
        path = repo_root / rel
        if path.is_file():
            findings.extend(_grep_file(path, pattern, repo_root, ignore_comment_lines=True))
    if findings:
        result.fail(
            "Step5 active path references legacy fields as primary inputs.",
            findings,
        )
    else:
        result.summary = "Step5 active path uses RCR posterior and canonical evidence fields only."
    return result


def _check_step5_innovation_one_control(repo_root: Path) -> RuleResult:
    result = RuleResult("R021", "Step5 innovation parameters must flow through One-Control")
    config = _read(repo_root / "configs" / "odcr.yaml")
    resolver = _read(repo_root / "code" / "odcr_core" / "config_resolver.py")
    cfg_py = _read(repo_root / "code" / "config.py")
    engine = _read(repo_root / "code" / "executors" / "step5_engine.py")
    required_config_terms = (
        "lci:",
        "confidence_schedule:",
        "perturb_std:",
        "uci:",
        "bucket_weights:",
        "ccv:",
        "control_packet_field_policy:",
        "verbalizer_adapter_policy:",
        "fca:",
        "evidence_alignment_mode:",
    )
    required_flow_terms = (
        "_resolve_step5_innovation_config",
        "step5_innovation_config_json",
        "step5.lci.weight",
        "step5.fca.weight",
    )
    missing_config = [term for term in required_config_terms if term not in config]
    missing_flow = [term for term in required_flow_terms if term not in resolver + cfg_py + engine]
    findings: list[Finding] = []
    active_files = (
        "configs/odcr.yaml",
        "code/executors/step5_engine.py",
        "code/odcr_core/config_resolver.py",
        "code/config.py",
    )
    retired = re.compile(r"lambda_lci|lambda_fca", re.IGNORECASE)
    for rel in active_files:
        path = repo_root / rel
        if not path.is_file():
            continue
        for idx, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            if not retired.search(line):
                continue
            if "retired" in line.lower() or "contains retired" in line.lower() or "effective training_row contains retired" in line.lower():
                continue
            if rel == "code/odcr_core/config_resolver.py" and line.strip().strip(",").strip("\"'") in (
                "lambda_lci",
                "lambda_fca",
            ):
                continue
            findings.append(
                Finding(
                    rel,
                    idx,
                    line.strip(),
                    "Remove active lambda_lci/lambda_fca names; use step5.lci.weight / step5.fca.weight.",
                )
            )
    if missing_config or missing_flow or findings:
        if not findings:
            findings = [
                Finding(
                    "configs/odcr.yaml",
                    1,
                    f"missing_config={missing_config}, missing_flow={missing_flow}",
                    "Expose and resolve Step5 LCI/UCI/CCV/FCA active parameters through configs/odcr.yaml.",
                )
            ]
        result.fail("Step5 innovation config is not fully One-Control-owned.", findings)
    else:
        result.summary = "Step5 LCI/UCI/CCV/FCA parameters resolve from configs/odcr.yaml with retired weights fail-fast."
    return result


def _check_step5_lora_flan_one_control(repo_root: Path) -> RuleResult:
    result = RuleResult("R022", "Step5 native LoRA/Flan controls must not bypass One-Control")
    config = _read(repo_root / "configs" / "odcr.yaml")
    resolver = _read(repo_root / "code" / "odcr_core" / "config_resolver.py")
    engine = _read(repo_root / "code" / "executors" / "step5_engine.py")
    lora = _read(repo_root / "code" / "odcr_core" / "step5_native_lora.py")
    required = (
        "native_lora:",
        "soft_prompt_len:",
        "numeric_control_dim:",
        "control_adapter_input_blocks:",
        "step5.ccv.native_lora",
        "_apply_step5_native_lora_row",
        "ccv_numeric_control_dim",
        "ccv_control_adapter_input_blocks",
        "discover_step5_text_linear_targets",
    )
    haystack = "\n".join([config, resolver, engine, lora])
    missing = [term for term in required if term not in haystack]
    findings: list[Finding] = []
    backend_block_match = re.search(r"step5:\n(?P<body>.*?)(?:\neval:|\Z)", config, re.DOTALL)
    backend_body = backend_block_match.group("body") if backend_block_match else ""
    legacy_backend = re.findall(r"^\s+lora_(?:r|alpha|dropout|target_modules):", backend_body, flags=re.MULTILINE)
    if legacy_backend:
        findings.append(
            Finding(
                "configs/odcr.yaml",
                1,
                "legacy step5.train.backend LoRA keys remain: " + ", ".join(sorted(set(legacy_backend))),
                "Move native LoRA controls under step5.ccv.native_lora.",
            )
        )
    if missing or findings:
        findings = findings or [
            Finding(
                "configs/odcr.yaml",
                1,
                "missing: " + ", ".join(missing),
                "Route Step5 native LoRA and CCV adapter dimensions through One-Control.",
            )
        ]
        result.fail("Step5 native LoRA/Flan controls are not fully closed under One-Control.", findings)
    else:
        result.summary = "Step5 native LoRA and CCV adapter dimensions are One-Control-owned."
    return result


def _check_step5_positive_tests_no_legacy_fields(repo_root: Path) -> RuleResult:
    result = RuleResult("R023", "Step5 positive-path tests must use canonical RCR fields")
    test_files = (
        "code/tests/test_step5_lci.py",
        "code/tests/test_step5_ccv_fca.py",
        "code/tests/test_step5_eval_default_control.py",
        "code/tests/test_index_contract.py",
    )
    pattern = re.compile(
        r"content_preserve_score|evidence_quality(?!_prior)|entropy_filter|entropy_only|"
        r"route_hint|old_route|content_keywords|content_aspects|content_entities|"
        r"style_markers|sentiment_style|domain_style_id|template_family|length_style_bucket",
        re.IGNORECASE,
    )
    findings: list[Finding] = []
    required_positive_terms = (
        "route_scorer_mask",
        "route_explainer_mask",
        "cf_reliability",
        "content_retention",
        "rating_stability",
        "uncertainty_score",
        "confidence_bucket",
    )
    combined = ""
    for rel in test_files:
        path = repo_root / rel
        if path.is_file():
            txt = path.read_text(encoding="utf-8", errors="ignore")
            combined += "\n" + txt
            findings.extend(_grep_file(path, pattern, repo_root, ignore_comment_lines=True))
    missing = [term for term in required_positive_terms if term not in combined]
    if missing:
        findings.append(
            Finding(
                "code/tests/test_step5_lci.py",
                1,
                "missing canonical positive fixture terms: " + ", ".join(missing),
                "Positive Step5 fixtures must use Step4 RCR posterior fields.",
            )
        )
    if findings:
        result.fail("Step5 positive tests still include legacy primary fields or miss RCR posterior fixtures.", findings)
    else:
        result.summary = "Step5 positive tests use canonical RCR posterior fixtures; legacy fields are guardrail-only."
    return result


def _check_preprocess_detail_fields_retired(repo_root: Path) -> RuleResult:
    result = RuleResult("R024", "preprocess retired detail fields must stay off the main contract")
    findings: list[Finding] = []
    contract_path = repo_root / "code" / "data_contract.py"
    step4_contract_path = repo_root / "code" / "odcr_core" / "index_contract.py"
    if not contract_path.is_file():
        findings.append(
            Finding(
                "code/data_contract.py",
                1,
                "missing",
                "Keep the preprocess CSV contract centralized in code/data_contract.py.",
            )
        )
    else:
        text = _read(contract_path)
        for column in PREPROCESS_RETIRED_DETAIL_COLUMNS:
            spec_re = re.compile(
                r"PreprocessFieldSpec\(\s*name\s*=\s*['\"]" + re.escape(column) + r"['\"]",
                re.DOTALL,
            )
            if spec_re.search(text):
                findings.append(
                    Finding(
                        "code/data_contract.py",
                        1,
                        f"retired detail field remains in PREPROCESS_FIELD_SPECS: {column}",
                        "Remove retired detail fields from PREPROCESS_FIELD_SPECS and derived CSV column orders.",
                    )
                )
        required_terms = (
            "DEPRECATED_PREPROCESS_DETAIL_COLUMNS",
            "STEP4_POSTERIOR_ROUTE_COLUMNS",
            "assert_no_deprecated_preprocess_detail_columns",
            "assert_no_step4_posterior_route_columns",
            "content_evidence",
            "style_evidence",
            "evidence_quality_prior",
            "preprocess_route_scorer_prior",
            "preprocess_route_explainer_prior",
        )
        missing = [term for term in required_terms if term not in text]
        if missing:
            findings.append(
                Finding(
                    "code/data_contract.py",
                    1,
                    "missing canonical/retirement terms: " + ", ".join(missing),
                    "Keep the canonical preprocess contract and fail-fast retired-column check together.",
                )
            )
        for posterior_column in ("route_scorer", "route_explainer"):
            spec_re = re.compile(
                r"PreprocessFieldSpec\(\s*name\s*=\s*['\"]" + re.escape(posterior_column) + r"['\"]",
                re.DOTALL,
            )
            if spec_re.search(text):
                findings.append(
                    Finding(
                        "code/data_contract.py",
                        1,
                        f"Step4 posterior field remains in PREPROCESS_FIELD_SPECS: {posterior_column}",
                        "Preprocess CSVs must emit preprocess_route_*_prior only; Step4 owns route_scorer/route_explainer.",
                    )
                )
    step4_text = _read(step4_contract_path) if step4_contract_path.is_file() else ""
    for required_step4_field in ("route_scorer", "route_explainer"):
        if required_step4_field not in step4_text:
            findings.append(
                Finding(
                    "code/odcr_core/index_contract.py",
                    1,
                    f"missing Step4 posterior field: {required_step4_field}",
                    "Step4 may and must own route_scorer/route_explainer as posterior route decisions.",
                )
            )

    active_consumers = (
        "code/split_data.py",
        "code/combine_data.py",
        "code/compute_embeddings.py",
        "code/infer_domain_semantics.py",
        "code/odcr_core/preprocess_schema.py",
        "code/odcr_core/preprocess_runtime.py",
    )
    retired_pattern = re.compile("|".join(re.escape(item) for item in PREPROCESS_RETIRED_DETAIL_COLUMNS))
    for rel in active_consumers:
        path = repo_root / rel
        if not path.is_file():
            continue
        for hit in _grep_file(path, retired_pattern, repo_root, ignore_comment_lines=True):
            findings.append(
                Finding(
                    hit.path,
                    hit.line,
                    hit.text,
                    "Do not consume retired preprocess detail fields outside fail-fast contract checks.",
                )
            )

    if findings:
        result.fail("Retired preprocess detail fields still appear on active contract/consumer surfaces.", findings)
    else:
        result.summary = "Preprocess contract is canonical evidence only; retired detail fields are fail-fast only."
    return result


def _check_roots_embed_dim_one_control(repo_root: Path) -> RuleResult:
    result = RuleResult("R025", "global roots/models/embed_dim must be One-Control-owned")
    config = _read(repo_root / "configs" / "odcr.yaml")
    resolver = _read(repo_root / "code" / "odcr_core" / "config_resolver.py")
    runners = _read(repo_root / "code" / "odcr_core" / "runners.py")
    paths = _read(repo_root / "code" / "paths_config.py")
    cfg_py = _read(repo_root / "code" / "config.py")
    required = {
        "configs/odcr.yaml": (
            "run_root:",
            "cache_dir:",
            "data_dir:",
            "merged_dir:",
            "offline:",
            "local_files_only:",
            "models_dir:",
            "step5_text_model:",
            "sentence_embed_model:",
            "embed_dim:",
        ),
        "code/odcr_core/config_resolver.py": (
            "_resolve_global_runtime_roots",
            "PreprocessResolvedPayload",
            "project.cache_dir",
            "env.local_files_only",
            "ODCR_MODELS_DIR",
            "ODCR_EMBED_DIM",
            "Legacy ODCR_* root/model/embed_dim",
            "runtime_roots",
        ),
        "code/odcr_core/preprocess_runtime.py": (
            "self.config.resolved.data_dir",
            "ODCR_RESOLVED_DATA_DIR",
            "ODCR_RESOLVED_SENTENCE_EMBED_MODEL",
            "_assert_resolved_payload",
            "_assert_gpu_admission",
            "tmux -L odcr_gpu new-session -A -s odcr",
        ),
        "code/odcr_core/runners.py": (
            "ODCR_RESOLVED_DATA_DIR",
            "ODCR_RESOLVED_MERGED_DIR",
            "ODCR_RESOLVED_MODELS_DIR",
            "ODCR_RESOLVED_STEP5_TEXT_MODEL",
            "ODCR_RESOLVED_SENTENCE_EMBED_MODEL",
            "ODCR_RESOLVED_EMBED_DIM",
        ),
        "code/paths_config.py": (
            "ODCR_RESOLVED_DATA_DIR",
            "ODCR_RESOLVED_MODELS_DIR",
            "旧 ODCR_* 环境变量不得覆盖",
            "configs/odcr.yaml",
        ),
        "code/config.py": (
            "ODCR_RESOLVED_EMBED_DIM",
            "runtime_roots",
            "legacy env cannot override configs/odcr.yaml",
        ),
    }
    texts = {
        "configs/odcr.yaml": config,
        "code/odcr_core/config_resolver.py": resolver,
        "code/odcr_core/preprocess_runtime.py": _read(repo_root / "code" / "odcr_core" / "preprocess_runtime.py"),
        "code/odcr_core/runners.py": runners,
        "code/paths_config.py": paths,
        "code/config.py": cfg_py,
        "code/compute_embeddings.py": _read(repo_root / "code" / "compute_embeddings.py"),
        "code/infer_domain_semantics.py": _read(repo_root / "code" / "infer_domain_semantics.py"),
    }
    findings: list[Finding] = []
    for rel, terms in required.items():
        missing = [term for term in terms if term not in texts[rel]]
        if missing:
            findings.append(
                Finding(
                    rel,
                    1,
                    "missing: " + ", ".join(missing),
                    "Resolve roots/model paths/embed_dim from configs/odcr.yaml and inject only ODCR_RESOLVED_* to children.",
                )
            )
    for rel in ("code/compute_embeddings.py", "code/infer_domain_semantics.py"):
        text = texts[rel]
        required_gpu_terms = (
            "requires CUDA before loading BGE-large",
            "allow_cpu_debug",
            "tmux -L odcr_gpu new-session -A -s odcr",
        )
        missing_gpu_terms = [term for term in required_gpu_terms if term not in text]
        if missing_gpu_terms:
            findings.append(
                Finding(
                    rel,
                    1,
                    "missing: " + ", ".join(missing_gpu_terms),
                    "preprocess_b/c GPU children must fail fast before BGE-large and mention the tmux GPU session.",
                )
            )
    if findings:
        result.fail("Global roots/model/embed_dim are not fully closed under One-Control.", findings)
    else:
        result.summary = "Roots, model paths, and embed_dim resolve from configs/odcr.yaml; legacy ODCR_* conflicts fail-fast."
    return result


def _check_manifest_embed_dim_no_bare_env(repo_root: Path) -> RuleResult:
    result = RuleResult("R041", "manifest embed_dim metadata must not read bare env")
    path = repo_root / "code" / "odcr_core" / "manifests.py"
    text = _read(path)
    findings: list[Finding] = []
    banned = re.compile(r"ODCR_EMBED_DIM|os\.environ\.get\([^)\n]*EMBED_DIM|getenv\([^)\n]*EMBED_DIM")
    findings.extend(_grep_file(path, banned, repo_root, ignore_comment_lines=False))
    required = (
        'getattr(cfg, "embed_dim", None)',
        "manifest backbones hidden_size requires resolved cfg.embed_dim",
    )
    missing = [term for term in required if term not in text]
    if missing:
        findings.append(
            Finding(
                "code/odcr_core/manifests.py",
                1,
                "missing: " + ", ".join(missing),
                "Manifest backbone metadata must use resolved cfg.embed_dim and fail-fast when it is unavailable.",
            )
        )
    if "ODCR_WRITE_RUN_MANIFEST" in text:
        findings.append(
            Finding(
                "code/odcr_core/manifests.py",
                1,
                "ODCR_WRITE_RUN_MANIFEST",
                "Manifest writing is mandatory; do not control it with a bare env side channel.",
            )
        )
    if "Run manifests are mandatory One-Control handoff artifacts" not in text:
        findings.append(
            Finding(
                "code/odcr_core/manifests.py",
                1,
                "missing mandatory manifest policy",
                "Manifest writing must be always-on unless a future One-Control parameter explicitly governs it.",
            )
        )
    if findings:
        result.fail(
            "Manifest backbone hidden_size/embed_dim metadata still has a bare env/default fallback risk.",
            findings,
        )
    else:
        result.summary = "Manifest backbone hidden_size uses resolved cfg.embed_dim and has no bare embed_dim env fallback."
    return result


def _check_child_argparse_payload_transport(repo_root: Path) -> RuleResult:
    result = RuleResult("R026", "child argparse must not override effective payload")
    cfg_py = _read(repo_root / "code" / "config.py")
    runners = _read(repo_root / "code" / "odcr_core" / "runners.py")
    resolver = _read(repo_root / "code" / "odcr_core" / "config_resolver.py")
    one_control_yaml = _read(repo_root / "configs" / "odcr.yaml")
    step4_entry = _read(repo_root / "code" / "executors" / "step4_entry.py")
    step3_core = _read(repo_root / "code" / "executors" / "step3_train_core.py")
    compute_embeddings = _read(repo_root / "code" / "compute_embeddings.py")
    infer_domain_semantics = _read(repo_root / "code" / "infer_domain_semantics.py")
    findings: list[Finding] = []
    required = (
        "internal child argparse conflict",
        "validated_equal_to_effective_payload",
        "Only public ./odcr --set may override configs/odcr.yaml",
        "缺少 ODCR_HARDWARE_PROFILE_JSON",
        "MAX_PARALLEL_CPU、ODCR_NUM_PROC 和 child --num-proc 不再作为 active fallback",
    )
    missing = [term for term in required if term not in cfg_py]
    if missing:
        findings.append(
            Finding(
                "code/config.py",
                1,
                "missing: " + ", ".join(missing),
                "Compare child CLI training args against ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON and fail on conflict.",
            )
        )
    hardware_required = (
        "max_parallel_cpu",
        "HARDWARE_PROFILE_REQUIRED_KEYS",
        "hardware.profiles.{stem}.max_parallel_cpu",
        "train_precision_source",
        "runtime_precision_mode",
    )
    missing_resolver = [term for term in hardware_required if term not in resolver]
    if missing_resolver:
        findings.append(
            Finding(
                "code/odcr_core/config_resolver.py",
                1,
                "missing: " + ", ".join(missing_resolver),
                "Step3 child hardware and precision must resolve through YAML/schema/resolver/source table.",
            )
        )
    if "max_parallel_cpu:" not in one_control_yaml or "train_precision: bf16" not in one_control_yaml:
        findings.append(
            Finding(
                "configs/odcr.yaml",
                1,
                "missing max_parallel_cpu or step3 train_precision",
                "Default Step3 hardware and precision controls must be explicit One-Control YAML values.",
            )
        )
    default_hw_required = (
        "max_parallel_cpu: 12",
        "dataloader_num_workers_train: 4",
        "dataloader_num_workers_valid: 2",
        "dataloader_num_workers_test: 2",
        "pin_memory: true",
        "persistent_workers: true",
        "non_blocking_h2d: true",
    )
    for term in default_hw_required:
        if term not in one_control_yaml:
            findings.append(
                Finding(
                    "configs/odcr.yaml",
                    1,
                    "missing: " + term,
                    "Current default hardware profile must match the user-confirmed 12 CPU core Slurm allocation.",
                )
            )
    if "max_parallel_cpu: 16" in one_control_yaml:
        findings.append(
            Finding(
                "configs/odcr.yaml",
                1,
                "max_parallel_cpu: 16",
                "Current default hardware truth is 12 cores, not 16.",
            )
        )
    for term in (
        "Step3 dataloader workers are per rank",
        "tokenization_active_processes",
        "reserved_cpu",
        "worker_budget_formula",
    ):
        if term not in resolver:
            findings.append(
                Finding(
                    "code/odcr_core/config_resolver.py",
                    1,
                    "missing: " + term,
                    "Resolver must reject Step3 dataloader/tokenizer CPU worker oversubscription.",
                )
            )
    hardcoded_precision_terms = (
        'out["ODCR_RUNTIME_PRECISION_MODE"] = "fp32"',
        '"ODCR_RUNTIME_PRECISION_MODE": "bf16"',
    )
    for term in hardcoded_precision_terms:
        if term in runners:
            findings.append(
                Finding(
                    "code/odcr_core/runners.py",
                    1,
                    term,
                    "Runtime precision transport must use ResolvedConfig.train_precision, not runner/hardware constants.",
                )
            )
    for banned in ('"--epochs"', '"--learning_rate"', '"--coef"', '"--eta"'):
        if banned in runners:
            findings.append(
                Finding(
                    "code/odcr_core/runners.py",
                    1,
                    banned,
                    "Do not pass train semantic knobs as child CLI overrides; use ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON.",
                )
            )
    if "ODCR_GLOBAL_EVAL_BATCH_SIZE" not in step4_entry or "step4 child argparse conflict" not in step4_entry:
        findings.append(
            Finding(
                "code/executors/step4_entry.py",
                1,
                "missing Step4 child batch conflict check",
                "Compare --batch-size with resolver-injected ODCR_GLOBAL_EVAL_BATCH_SIZE.",
            )
        )
    if "step4 child argparse conflict: --num-proc" not in step4_entry:
        findings.append(
            Finding(
                "code/executors/step4_entry.py",
                1,
                "missing Step4 child num_proc conflict check",
                "Compare --num-proc with resolver-injected ODCR_HARDWARE_PROFILE_JSON.num_proc.",
            )
        )
    if "step3 eval child argparse conflict: --num-proc" not in step3_core:
        findings.append(
            Finding(
                "code/executors/step3_train_core.py",
                1,
                "missing Step3 child num_proc conflict check",
                "Compare --num-proc with resolver-injected ODCR_HARDWARE_PROFILE_JSON.num_proc.",
            )
        )
    child_preprocess_required = {
        "code/compute_embeddings.py": (
            "must be launched with the resolved preprocess payload from ./odcr",
            "--embed-batch-size is required from the resolved preprocess_b payload.",
            "--read-chunk-rows is required from the resolved preprocess_b payload.",
            "--group-shard-size is required from the resolved preprocess_b payload.",
            "--tokenizer-parallelism/--no-tokenizer-parallelism",
            "--tokenizer-threads-per-worker",
            "--tokenizer-total-threads",
            "--prefetch-batches",
            "--pin-memory/--no-pin-memory",
            "--non-blocking-h2d/--no-non-blocking-h2d",
            "--async-prefetch/--no-async-prefetch",
            "--token-aware-batching/--no-token-aware-batching",
            "--cpu-cores-reserved",
            "--cpu-cores-available",
            "--grouped-text-cache/--no-grouped-text-cache is required from the resolved preprocess_b payload.",
            "--grouped-text-cache-dir is required from the resolved preprocess_b payload.",
            "--grouped-text-cache-version is required from the resolved preprocess_b payload.",
            "--bf16/--no-bf16",
            "--tf32/--no-tf32",
            "ODCR_RESOLVED_SENTENCE_EMBED_MODEL",
            "ODCR_RESOLVED_EMBED_DIM",
        ),
        "code/infer_domain_semantics.py": (
            "must be launched with the resolved preprocess payload from ./odcr",
            "--chunk-batch-size is required from the resolved preprocess_c payload.",
            "--tokenizer-parallelism/--no-tokenizer-parallelism",
            "--tokenizer-threads-per-worker",
            "--tokenizer-total-threads",
            "--prefetch-batches",
            "--pin-memory/--no-pin-memory",
            "--non-blocking-h2d/--no-non-blocking-h2d",
            "--async-prefetch/--no-async-prefetch",
            "--cpu-cores-reserved",
            "--cpu-cores-available",
            "--token-window-cache/--no-token-window-cache",
            "--token-window-cache-dir is required from the resolved preprocess_c payload.",
            "--token-window-cache-version is required from the resolved preprocess_c payload.",
            "--token-window-cache-shard-size is required from the resolved preprocess_c payload.",
            "--tokenizer-hotpath/--no-tokenizer-hotpath",
            "--bf16/--no-bf16",
            "--tf32/--no-tf32",
            "ODCR_RESOLVED_SENTENCE_EMBED_MODEL",
            "ODCR_RESOLVED_EMBED_DIM",
        ),
    }
    for rel, required_terms in child_preprocess_required.items():
        text = compute_embeddings if rel.endswith("compute_embeddings.py") else infer_domain_semantics
        missing = [term for term in required_terms if term not in text]
        if missing:
            findings.append(
                Finding(
                    rel,
                    1,
                    "missing: " + ", ".join(missing),
                    "preprocess_b/c child scripts must require resolved runtime transport instead of child defaults.",
                )
            )
    forbidden_preprocess_side_channels = {
        "code/compute_embeddings.py": (
            "EMBED_BATCH_SIZE",
            "default=DEFAULT_READ_CHUNK_ROWS",
            "default=DEFAULT_GROUP_SHARD_SIZE",
            "default=DEFAULT_GROUPED_TEXT_CACHE_DIR",
            "default=DEFAULT_GROUPED_TEXT_CACHE_VERSION",
            "DEFAULT_PREPROCESS_B_BF16_ENABLED",
            "DEFAULT_PREPROCESS_B_TF32_ENABLED",
        ),
        "code/infer_domain_semantics.py": (
            "DOMAIN_CHUNK_BATCH_SIZE",
            "DEFAULT_PREPROCESS_C_BF16_ENABLED",
            "DEFAULT_PREPROCESS_C_TF32_ENABLED",
            "DEFAULT_TOKENIZER_HOTPATH_ENABLED",
            "DEFAULT_TOKEN_WINDOW_CACHE_ENABLED",
        ),
    }
    for rel, forbidden_terms in forbidden_preprocess_side_channels.items():
        text = compute_embeddings if rel.endswith("compute_embeddings.py") else infer_domain_semantics
        for term in forbidden_terms:
            if term in text:
                findings.append(
                    Finding(
                        rel,
                        1,
                        term,
                        "Remove bare env/default fallback side channels from formal preprocess child scripts.",
                    )
                )
    forbidden_cfg_terms = (
        'os.environ["MAX_PARALLEL_CPU"]',
        "os.environ['MAX_PARALLEL_CPU']",
        'os.environ["ODCR_NUM_PROC"]',
        "os.environ['ODCR_NUM_PROC']",
        "num_proc_cli is not None",
        "max_parallel_cli is not None",
    )
    for term in forbidden_cfg_terms:
        if term in cfg_py:
            findings.append(
                Finding(
                    "code/config.py",
                    1,
                    term,
                    "Hardware child values must fail fast without ODCR_HARDWARE_PROFILE_JSON; no env or CLI fallback.",
                )
            )
    if findings:
        result.fail("Found child argparse surfaces that can bypass the resolved payload.", findings)
    else:
        result.summary = "Child train/hardware/preprocess args are transport-only; missing resolved payloads fail fast."
    return result


def _check_step4_helper_defaults_strict(repo_root: Path) -> RuleResult:
    result = RuleResult("R027", "Step4 helper defaults must be explicit test-only")
    routing = _read(repo_root / "code" / "odcr_core" / "odcr_cf_routing.py")
    export = _read(repo_root / "code" / "odcr_core" / "step4_training_export.py")
    findings: list[Finding] = []
    required = (
        "allow_test_defaults",
        "for_test_default",
        "Step4 active RCR config mapping is required",
        "attach_odcr_cf_routing requires resolved step4.rcr config",
        "_require_live_rcr_diagnostics(merged)",
        'pd.to_numeric(merged["shared_latent_similarity"]',
        'pd.to_numeric(merged["specific_latent_shift"]',
        'pd.to_numeric(merged["rating_delta"]',
    )
    missing = [term for term in required if term not in routing]
    if missing:
        findings.append(
            Finding(
                "code/odcr_core/odcr_cf_routing.py",
                1,
                "missing: " + ", ".join(missing),
                "Make ODCFRoutingConfig defaults available only via explicit test helpers.",
            )
        )
    for rel, text in (
        ("code/odcr_core/odcr_cf_routing.py", routing),
        ("code/odcr_core/step4_training_export.py", export),
    ):
        for idx, line in enumerate(text.splitlines(), start=1):
            if "ODCFRoutingConfig()" not in line:
                continue
            findings.append(
                Finding(
                    rel,
                    idx,
                    line.strip(),
                    "Use ODCFRoutingConfig.from_env(require=True) or ODCFRoutingConfig.for_test_default().",
                )
            )
    if "assemble_step4_training_table requires resolved step4.rcr" not in export:
        findings.append(
            Finding(
                "code/odcr_core/step4_training_export.py",
                1,
                "missing active rcr_config requirement",
                "Do not let Step4 table assembly silently construct fallback RCR defaults.",
            )
        )
    forbidden_proxy_terms = (
        'shared_latent_similarity", np.nan',
        'specific_latent_shift", np.nan',
        'rating_delta", np.nan',
        "content_text_alignment = _jaccard",
    )
    for term in forbidden_proxy_terms:
        if term in routing and term != "content_text_alignment = _jaccard":
            findings.append(
                Finding(
                    "code/odcr_core/odcr_cf_routing.py",
                    1,
                    term,
                    "Active Step4 RCR must fail fast on missing latent diagnostics and must not proxy them.",
                )
            )
    if findings:
        result.fail("Step4 RCR helper fallback can still be used from active paths.", findings)
    else:
        result.summary = "Step4 RCR defaults are explicit test-only; active paths require resolver config."
    return result


def _check_step5_parser_defaults_strict(repo_root: Path) -> RuleResult:
    result = RuleResult("R028", "Step5 innovation parser defaults must be explicit test-only")
    innov = _read(repo_root / "code" / "odcr_core" / "step5_innovation.py")
    engine = _read(repo_root / "code" / "executors" / "step5_engine.py")
    findings: list[Finding] = []
    required = (
        "allow_test_defaults",
        "for_test_default_step5_innovation_config",
        "Step5 active config JSON is required",
        "Step5 active config JSON must not be {}",
    )
    missing = [term for term in required if term not in innov]
    if missing:
        findings.append(
            Finding(
                "code/odcr_core/step5_innovation.py",
                1,
                "missing: " + ", ".join(missing),
                "Make parser None/{} defaults test-only and require resolver JSON in active Step5.",
            )
        )
    bad_engine_terms = ('step5_innovation_config_json", "{}"', "parse_step5_innovation_config_json(None)")
    for term in bad_engine_terms:
        if term in engine:
            findings.append(
                Finding(
                    "code/executors/step5_engine.py",
                    1,
                    term,
                    "Active Step5 must require resolved step5_innovation_config_json.",
                )
            )
    tests_text = "\n".join(_read(p) for p in (repo_root / "code" / "tests").glob("test_step5_*.py"))
    if "parse_step5_innovation_config_json(None)" in tests_text:
        findings.append(
            Finding(
                "code/tests",
                1,
                "parse_step5_innovation_config_json(None)",
                "Use for_test_default_step5_innovation_config() for unit defaults.",
            )
        )
    if findings:
        result.fail("Step5 innovation parser can still return active defaults without resolver JSON.", findings)
    else:
        result.summary = "Step5 parser None/{} fallback is test-only; active Step5 requires resolver JSON."
    return result


def _check_step3_structured_losses_one_control(repo_root: Path) -> RuleResult:
    result = RuleResult("R029", "Step3 structured loss weights must be One-Control-owned")
    config = _read(repo_root / "configs" / "odcr.yaml")
    resolver = _read(repo_root / "code" / "odcr_core" / "config_resolver.py")
    cfg_py = _read(repo_root / "code" / "config.py")
    engine = _read(repo_root / "code" / "executors" / "step3_train_core.py")
    probe = _read(repo_root / "code" / "tools" / "odcr_step3_real_data_probe.py")
    required = {
        "configs/odcr.yaml": (
            "structured_losses:",
            "loss_semantics:",
            "ddp:",
            "orthogonal:",
            "content_alignment_weight:",
            "shared_prototype_weight:",
            "prototype_separation_weight:",
            "light_explainer_weight:",
        ),
        "code/odcr_core/config_resolver.py": (
            "_resolve_step3_structured_losses_config",
            "_resolve_step3_loss_semantics_config",
            "_resolve_step3_ddp_config",
            "step3_structured_losses",
            "step3_loss_semantics",
            "step3_ddp",
            "step3.structured_losses",
        ),
        "code/config.py": (
            "step3_structured_loss_weights_json",
            "step3_loss_semantics_json",
            "ddp_graph_safety_preflight",
            "step3.structured_losses",
        ),
        "code/executors/step3_train_core.py": (
            "Step3StructuredLossWeights",
            "step3_structured_loss_weights_from_config",
            "Step3ForwardOutput",
            "Step3LossBundle",
            "compose_step3_loss_from_forward_output",
        ),
        "code/tools/odcr_step3_real_data_probe.py": (
            "compose_step3_loss_from_forward_output",
            "validate_step3_graph_safety_preflight",
            "profile_domain_artifacts",
        ),
    }
    texts = {
        "configs/odcr.yaml": config,
        "code/odcr_core/config_resolver.py": resolver,
        "code/config.py": cfg_py,
        "code/executors/step3_train_core.py": engine,
        "code/tools/odcr_step3_real_data_probe.py": probe,
    }
    findings: list[Finding] = []
    for rel, terms in required.items():
        missing = [term for term in terms if term not in texts[rel]]
        if missing:
            findings.append(
                Finding(
                    rel,
                    1,
                    "missing: " + ", ".join(missing),
                    "Route Step3 structured loss weights through step3.structured_losses.",
                )
            )
    banned_engine = (
        "lambda_var = 0.10",
        "lambda_shared_inv = 0.18",
        "lambda_specific_sep = 0.16",
        "lambda_content_align = 0.12",
        "lambda_proto = 0.04",
        "0.2 * light_explainer",
    )
    for term in banned_engine:
        if term in engine:
            findings.append(
                Finding(
                    "code/executors/step3_train_core.py",
                    1,
                    term,
                    "Do not keep active Step3 structured loss literals in the train loop.",
                )
            )
    banned_side_channels = {
        "code/executors/step3_train_core.py": (
            "_model.last_odcr_latents",
            "ddp_model.module",
            "domain_style_proto.weight",
            "shared_global_proto",
            "nn.Parameter(user_content_profiles",
            "nn.Parameter(domain_content_profiles",
        ),
        "code/tools/odcr_step3_real_data_probe.py": (
            "underlying.last_odcr_latents",
            "ddp_model.module",
            "domain_style_proto.weight",
            "shared_global_proto",
        ),
    }
    for rel, terms in banned_side_channels.items():
        text = texts[rel]
        for term in terms:
            if term in text:
                findings.append(
                    Finding(
                        rel,
                        1,
                        term,
                        "Step3 loss/probe must use Step3ForwardOutput and frozen buffers, not side-channel or direct parameter loss paths.",
                    )
                )
    if findings:
        result.fail("Step3 structured loss weights are not fully One-Control-wired.", findings)
    else:
        result.summary = "Step3 structured loss weights resolve from step3.structured_losses and the shared Step3ForwardOutput loss builder reads resolved JSON only."
    return result


def _check_step3_upstream_preprocess_hard_gate(repo_root: Path) -> RuleResult:
    result = RuleResult("R097", "Step3 active loader must require preprocess upstream hard gate")
    core = _read(repo_root / "code" / "executors" / "step3_train_core.py")
    gate = _read(repo_root / "code" / "odcr_core" / "step3_upstream_gate.py")
    index_contract = _read(repo_root / "code" / "odcr_core" / "index_contract.py")
    cfg_py = _read(repo_root / "code" / "config.py")
    findings: list[Finding] = []

    required_gate_terms = (
        "validate_step3_preprocess_upstream_gate",
        "latest_summary_path",
        "run_summary.json",
        "stage_status.json",
        "stage_manifest.json",
        "source_table.json",
        "metrics_path",
        "verify_report_path",
        "source_profile_fingerprints",
        "preprocess_c_domain_vector/1",
        "STEP3_REJECTS_PREPROCESS_C_RANK2_DOMAIN_VECTOR",
        "completed.stamp",
        "AI_analysis",
        "history",
        "fingerprint mismatch",
    )
    missing_gate_terms = [term for term in required_gate_terms if term not in gate]
    if missing_gate_terms:
        findings.append(
            Finding(
                "code/odcr_core/step3_upstream_gate.py",
                1,
                "missing: " + ", ".join(missing_gate_terms),
                "Step3 upstream gate must validate formal latest/run_summary/manifest/verify/fingerprint evidence and reject old domain vector forms.",
            )
        )

    required_core_terms = (
        "validate_step3_preprocess_upstream_gate",
        "step3_upstream_evidence_json",
        "step3_upstream_preflight_summary=upstream_evidence",
        "step3_upstream_preprocess_gate",
        "load_profile_tensors_dual_first",
    )
    missing_core_terms = [term for term in required_core_terms if term not in core]
    if missing_core_terms:
        findings.append(
            Finding(
                "code/executors/step3_train_core.py",
                1,
                "missing: " + ", ".join(missing_core_terms),
                "Step3 train loader must run the upstream gate and carry its summary into checkpoint lineage.",
            )
        )
    gate_pos = core.find("validate_step3_preprocess_upstream_gate(")
    read_pos = core.find("pd.read_csv(train_path)")
    profile_pos = core.find("load_profile_tensors_dual_first(")
    model_pos = core.find("model = Model(")
    if gate_pos < 0 or read_pos < 0 or gate_pos > read_pos:
        findings.append(
            Finding(
                "code/executors/step3_train_core.py",
                1,
                "upstream gate is not before pd.read_csv(train_path)",
                "Run the preprocess upstream hard gate before dataset construction.",
            )
        )
    if gate_pos < 0 or profile_pos < 0 or gate_pos > profile_pos:
        findings.append(
            Finding(
                "code/executors/step3_train_core.py",
                1,
                "upstream gate is not before load_profile_tensors_dual_first",
                "Do not path-only consume profile/domain artifacts before contract validation.",
            )
        )
    if gate_pos < 0 or model_pos < 0 or gate_pos > model_pos:
        findings.append(
            Finding(
                "code/executors/step3_train_core.py",
                1,
                "upstream gate is not before Model construction",
                "Run the preprocess upstream hard gate before model/loss construction.",
            )
        )
    if "step3_upstream_evidence_json" not in cfg_py:
        findings.append(
            Finding(
                "code/config.py",
                1,
                "step3_upstream_evidence_json missing from FinalTrainingConfig",
                "Carry upstream evidence through the resolved Step3 config for checkpoint sidecars.",
            )
        )
    missing_index_terms = [
        term
        for term in (
            "step3_upstream_preflight_summary",
            "_require_step3_preflight_profile_paths",
            "Step3 profile loader path mismatch",
            "Step3 domain loader path mismatch",
        )
        if term not in index_contract
    ]
    if missing_index_terms:
        findings.append(
            Finding(
                "code/odcr_core/index_contract.py",
                1,
                "missing: " + ", ".join(missing_index_terms),
                "Step3 profile/domain tensor loader must bind to the preflight-validated artifact summary when Step3 calls it.",
            )
        )

    if findings:
        result.fail("Step3 can still bypass preprocess upstream admission evidence.", findings)
    else:
        result.summary = "Step3 train loader runs the preprocess upstream hard gate before data/profile/model construction and records its summary."
    return result


def _check_step5_gate_arch_adv_eta_one_control(repo_root: Path) -> RuleResult:
    result = RuleResult("R030", "Step5 gate/model/loss semantics must be One-Control-owned")
    config = _read(repo_root / "configs" / "odcr.yaml")
    resolver = _read(repo_root / "code" / "odcr_core" / "config_resolver.py")
    cfg_py = _read(repo_root / "code" / "config.py")
    entry = _read(repo_root / "code" / "executors" / "step5_entry.py")
    engine = _read(repo_root / "code" / "executors" / "step5_engine.py")
    innov = _read(repo_root / "code" / "odcr_core" / "step5_innovation.py")
    haystack = "\n".join([config, resolver, cfg_py, entry, engine, innov])
    required = (
        "explainer_gate:",
        "uncertainty_exponent:",
        "style_shift_diversity_boost:",
        "explainer_only_multiplier:",
        "model:",
        "nlayers:",
        "nhead:",
        "nhid:",
        "dropout:",
        "explainer_loss_weight:",
        "_resolve_step5_model_config",
        "_reject_step5_retired_controls",
        "Step5ExplainerGateConfig",
        "_load_step5_checkpoint_fail_fast",
    )
    findings: list[Finding] = []
    missing = [term for term in required if term not in haystack]
    if missing:
        findings.append(
            Finding(
                "configs/odcr.yaml",
                1,
                "missing: " + ", ".join(missing),
                "Expose Step5 gate/model/loss semantics through configs/odcr.yaml and resolver payloads.",
            )
        )
    step5_block = re.search(r"step5:\n(?P<body>.*?)(?:\neval:|\Z)", config, re.DOTALL)
    step5_body = step5_block.group("body") if step5_block else ""
    legacy_yaml = re.findall(r"^\s+(?:adv|eta):", step5_body, flags=re.MULTILINE)
    if legacy_yaml:
        findings.append(
            Finding(
                "configs/odcr.yaml",
                1,
                "retired Step5 adv/eta keys remain: " + ", ".join(sorted(set(legacy_yaml))),
                "Use step5.train.explainer_loss_weight instead.",
            )
        )
    banned_terms = (
        'p.add_argument("--eta"',
        'train.get("eta", train.get("adv"',
        'task.get("adv"',
        "* 0.7",
        'default=4)',
        'default=2048)',
        'default=0.15)',
    )
    for rel, text in (
        ("code/executors/step5_entry.py", entry),
        ("code/odcr_core/config_resolver.py", resolver),
        ("code/executors/step5_engine.py", engine),
    ):
        for term in banned_terms:
            if term in text:
                findings.append(
                    Finding(
                        rel,
                        1,
                        term,
                        "Remove active Step5 legacy aliases or architecture/gate literals.",
                    )
                )
    if findings:
        result.fail("Step5 gate/model/loss semantics are not fully One-Control-owned.", findings)
    else:
        result.summary = "Step5B gate, explainer-only multiplier, model architecture, and explainer loss weight are One-Control-owned; adv/eta are fail-fast retired names."
    return result


def _missing_terms_finding(rel: str, missing: Sequence[str], suggestion: str) -> Finding:
    return Finding(rel, 1, "missing: " + ", ".join(missing), suggestion)


def _check_preprocess_skip_completed_fingerprint_gate(repo_root: Path) -> RuleResult:
    result = RuleResult("R031", "preprocess skip_completed must fingerprint gate")
    files = {
        "code/odcr_core/preprocess_runtime.py": (
            "PREPROCESS_CONTRACT_VERSION",
            "canonical_column_hash",
            "source_datasets",
            "path_roots",
            "input_output_roots",
            "model_fingerprints",
            "embed_dim",
            "config_snapshot_hash",
            "schema_code_fingerprint",
            "unit_fingerprint_hash",
            "_source_table_payload",
            "metadata_schema_version",
            "cache_key_fields",
            "lineage_stale_policy",
            "CANONICAL_PREPROCESS_CHUNK_SIZE",
            "skip_completed refused",
            "_assert_unit_outputs_current_contract",
        ),
        "code/odcr_core/preprocess_status.py": (
            "fingerprint",
            "fingerprint_hash",
        ),
        "code/compute_embeddings.py": (
            "canonical_text_source_contract",
            "canonical_column_hash",
            "source_file",
            "selected_columns",
            "model_artifact_fingerprint",
            "odcr_embed_dim",
            "read_chunk_rows",
            "group_shard_size",
            "cache_version",
        ),
        "code/infer_domain_semantics.py": (
            "preprocess_contract_version",
            "canonical_column_hash",
            "canonical_text_source_contract",
            "selected_columns",
            "tokenizer",
            "model_artifact_fingerprint",
            "odcr_embed_dim",
            "max_total_tokens",
            "payload_budget",
            "token_window_cache_version",
        ),
    }
    findings: list[Finding] = []
    for rel, terms in files.items():
        path = repo_root / rel
        text = _read(path) if path.is_file() else ""
        missing = [term for term in terms if term not in text]
        if missing:
            findings.append(
                _missing_terms_finding(
                    rel,
                    missing,
                    "Bind preprocess status/cache reuse to current contract/config/source/model/embed fingerprints.",
                )
            )
    if findings:
        result.fail("Preprocess cache/skip_completed hard gate is incomplete.", findings)
    else:
        result.summary = "skip_completed and preprocess_b/c caches compare Phase 4A fingerprints before reuse."
    return result


def _check_step3_checkpoint_step4_lineage_gate(repo_root: Path) -> RuleResult:
    result = RuleResult("R032", "Step3 checkpoint -> Step4 lineage must hard gate")
    files = {
        "code/executors/step3_train_core.py": (
            "_build_step3_checkpoint_lineage",
            "STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION",
            "sidecar_schema_version",
            "checkpoint_file_hash",
            "git_code_fingerprint",
            "resolved_config_compatibility_hash",
            "source_table_compatibility_hash",
            "one_control_resolved_config_hash",
            "preprocess_contract_version",
            "preprocess_latest_run_ids",
            "preprocess_stage_manifest_fingerprints",
            "preprocess_verify_report_fingerprints",
            "profile_artifact_fingerprints_hash",
            "domain_artifact_fingerprints_hash",
            "step3_tokenizer_cache_manifest_hash",
            "batch_semantics",
            "ddp_config_hash",
            "precision_config_hash",
            "data_merged_artifact_fingerprint",
            "step3_structured_losses_config_hash",
            "step3_loss_semantics_config_hash",
            "model_architecture_config_hash",
            "source_task",
            "write_checkpoint_lineage",
        ),
        "code/executors/step4_engine.py": (
            "_validate_step3_checkpoint_lineage_for_step4",
            "validate_step3_checkpoint_lineage",
            "CheckpointLineageError",
            "checkpoint_file_hash",
            "Step4 refused Step3 checkpoint",
        ),
        "code/odcr_core/training_checkpoint.py": (
            "LINEAGE_GATE_SCHEMA_VERSION",
            "STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION",
            "CHECKPOINT_LINEAGE_FILENAME",
            "STEP3_CHECKPOINT_REQUIRED_FIELDS",
            "read_checkpoint_lineage",
            "write_checkpoint_lineage",
            "validate_step3_checkpoint_lineage",
        ),
    }
    findings: list[Finding] = []
    for rel, terms in files.items():
        path = repo_root / rel
        text = _read(path) if path.is_file() else ""
        missing = [term for term in terms if term not in text]
        if missing:
            findings.append(
                _missing_terms_finding(
                    rel,
                    missing,
                    "Write Step3 checkpoint lineage and make Step4 validate it before loading weights.",
                )
            )
    if findings:
        result.fail("Step3->Step4 checkpoint lineage hard gate is incomplete.", findings)
    else:
        result.summary = "Step4 reads Step3 checkpoint lineage and fails before load on config/data/schema mismatch."
    return result


def _check_step4_export_step5_lineage_gate(repo_root: Path) -> RuleResult:
    result = RuleResult("R033", "Step4 export -> Step5 lineage must hard gate")
    files = {
        "code/odcr_core/index_contract.py": (
            "STEP4_EXPORT_LINEAGE_SCHEMA_VERSION",
            "STEP4_ROUTE_POSTERIOR_CONTRACT_VERSION",
            "step4_rcr_required_fields_hash",
            "build_step4_export_lineage",
            "validate_step4_export_lineage",
            "Step5 refused Step4 export",
        ),
        "code/executors/step4_engine.py": (
            "build_step4_export_lineage",
            "step4_export_lineage",
            "step4_export_lineage_hash",
        ),
        "code/odcr_core/step4_training_export.py": (
            "step4_export_lineage",
            "posterior_contract_version",
            "required_fields_hash",
        ),
        "code/executors/step5_engine.py": (
            "validate_step4_export_lineage",
            "step4_export_lineage_json",
        ),
    }
    findings: list[Finding] = []
    for rel, terms in files.items():
        path = repo_root / rel
        text = _read(path) if path.is_file() else ""
        missing = [term for term in terms if term not in text]
        if missing:
            findings.append(
                _missing_terms_finding(
                    rel,
                    missing,
                    "Attach Step4 export lineage to index_contract/manifest and validate it in Step5.",
                )
            )
    if findings:
        result.fail("Step4->Step5 export lineage hard gate is incomplete.", findings)
    else:
        result.summary = "Step5 validates Step4 export/index_contract lineage, RCR config, route contract, and task domains."
    return result


def _check_step5_checkpoint_eval_rerank_lineage_gate(repo_root: Path) -> RuleResult:
    result = RuleResult("R034", "Step5 checkpoint -> eval/rerank compatibility must hard gate")
    engine = _read(repo_root / "code" / "executors" / "step5_engine.py")
    ckpt = _read(repo_root / "code" / "odcr_core" / "training_checkpoint.py")
    required_engine = (
        "_build_step5_checkpoint_lineage",
        "_current_step5_checkpoint_expectation",
        "_load_step5_checkpoint_fail_fast",
        "read_checkpoint_lineage(checkpoint_path, expected_stage=\"step5\")",
        "STEP5_CHECKPOINT_COMPAT_SCHEMA_VERSION",
        "STEP5_TRAIN_SCHEMA_VERSION",
        "tokenizer_model_artifact_fingerprint",
        "step5_config_hashes",
        "Step5 eval/rerank refused checkpoint",
    )
    required_ckpt = (
        "STEP5_CHECKPOINT_COMPAT_SCHEMA_VERSION",
        "STEP5_EVAL_OUTPUT_SCHEMA_VERSION",
        "STEP5_TRAIN_SCHEMA_VERSION",
    )
    findings: list[Finding] = []
    missing_engine = [term for term in required_engine if term not in engine]
    missing_ckpt = [term for term in required_ckpt if term not in ckpt]
    if missing_engine:
        findings.append(
            _missing_terms_finding(
                "code/executors/step5_engine.py",
                missing_engine,
                "Write Step5 checkpoint lineage and validate it before eval/rerank load.",
            )
        )
    if missing_ckpt:
        findings.append(
            _missing_terms_finding(
                "code/odcr_core/training_checkpoint.py",
                missing_ckpt,
                "Centralize Step5 checkpoint/eval compatibility schema constants.",
            )
        )
    if findings:
        result.fail("Step5 checkpoint eval/rerank compatibility hard gate is incomplete.", findings)
    else:
        result.summary = "eval/rerank load Step5 checkpoints only after lineage/schema/config/model compatibility passes."
    return result


def _check_eval_rerank_resolved_config_no_fallback(repo_root: Path) -> RuleResult:
    result = RuleResult("R035", "eval/rerank must consume resolved Step5 config")
    engine = _read(repo_root / "code" / "executors" / "step5_engine.py")
    runners = _read(repo_root / "code" / "odcr_core" / "runners.py")
    daemon_path = repo_root / "code" / "tools" / "async_eval_daemon.py"
    findings: list[Finding] = []
    required_engine = (
        "eval requires resolved Step5 LCI/UCI/CCV/FCA config JSON",
        "eval-rerank requires resolver-owned One-Control rerank config",
        "ODCR_RERANK_PROFILE_JSON must be a non-empty object",
        "rerank_source_table",
        "eval/rerank refused old output schema",
        "STEP5_EVAL_OUTPUT_SCHEMA_VERSION",
        "STEP5_FACTUAL_EVAL_CONTROL_SCHEMA_VERSION",
        "factual_eval_default",
        "eval_control_contract",
        "_apply_step5_factual_eval_default_controls",
        "_require_step5_rcr_posterior_controls",
    )
    missing_engine = [term for term in required_engine if term not in engine]
    if missing_engine:
        findings.append(
            _missing_terms_finding(
                "code/executors/step5_engine.py",
                missing_engine,
                "Require resolver-injected Step5 innovation/rerank config and write only current eval schema.",
            )
        )
    banned_engine = (
        "num_ret = 4 if ns is None",
        "rm_s = \"rule_v3\" if rm is None",
        "top_k = 1 if mtk is None",
        "w_lp = 0.45 if",
        "ex_mode = \"head50\" if",
        'ev_df["route_scorer"] = 1',
        'valid_df["route_scorer"] = 1',
        'ev_df["route_explainer"] = 1',
        'valid_df["route_explainer"] = 1',
    )
    for term in banned_engine:
        if term in engine:
            findings.append(
                Finding(
                    "code/executors/step5_engine.py",
                    1,
                    term,
                    "Remove eval-rerank fallback literals; only One-Control transport values may be accepted.",
                )
            )
    required_runner = (
        "ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON",
        "ODCR_RERANK_PROFILE_JSON",
        "_rerank_runner_cli_args",
    )
    missing_runner = [term for term in required_runner if term not in runners]
    if missing_runner:
        findings.append(
            _missing_terms_finding(
                "code/odcr_core/runners.py",
                missing_runner,
                "Parent runner must inject effective payload and rerank profile/CLI transport.",
            )
        )
    if daemon_path.exists():
        findings.append(
            Finding(
                "code/tools/async_eval_daemon.py",
                1,
                "present",
                "Delete helper eval/rerank paths that can bypass resolver and checkpoint compatibility gates.",
            )
        )
    if findings:
        result.fail("eval/rerank resolved config hard gate is incomplete.", findings)
    else:
        result.summary = "eval/rerank require resolver Step5 config and label factual eval defaults separately from Step4 RCR posterior."
    return result


def _check_step3_finite_loss_global_sync(repo_root: Path) -> RuleResult:
    result = RuleResult("R036", "Step3 finite-loss skip must be globally synchronized")
    core = _read(repo_root / "code" / "executors" / "step3_train_core.py")
    required = (
        "step3_sync_loss_bundle_finite_status",
        "dist.all_reduce(local_vec, op=dist.ReduceOp.MIN)",
        "single_all_reduce_min_vector",
        "global_loss_finite",
        "non-finite loss synchronized skip",
    )
    missing = [term for term in required if term not in core]
    bad = "if not bool(torch.isfinite(loss).all().item()):" in core
    if missing or bad:
        result.fail(
            "Step3 finite-loss path is not a DDP-wide decision before backward.",
            [
                Finding(
                    "code/executors/step3_train_core.py",
                    1,
                    f"missing={missing}, rank_local_skip={bad}",
                    "All ranks must all-reduce local finite flags and either all backward or all skip.",
                )
            ],
        )
    else:
        result.summary = "Step3 all-reduces local finite flags before backward and skips each optimizer step globally."
    return result


def _check_graph_tied_zero_losses(repo_root: Path) -> RuleResult:
    result = RuleResult("R037", "Step3/Step5 train zero losses must be graph-tied")
    losses = _read(repo_root / "code" / "odcr_core" / "odcr_losses.py")
    step5 = _read(repo_root / "code" / "executors" / "step5_engine.py")
    word = _read(repo_root / "code" / "odcr_core" / "step5_word_losses.py")
    required = (
        "def graph_tied_zero",
        "return graph_tied_zero(shared_proj)",
        "return graph_tied_zero(specific_proj)",
        "return graph_tied_zero(domain_prototypes)",
        "graph_tied_zero(word_dist)",
        "graph_tied_zero_like(pred_rating)",
        "values.sum() * 0.0",
    )
    haystack = "\n".join([losses, step5, word])
    missing = [term for term in required if term not in haystack]
    banned = (
        "return shared_proj.new_zeros(())",
        "return specific_proj.new_zeros(())",
        "return domain_prototypes.new_zeros(())",
        "loss_ul = word_dist.new_zeros(())",
        "loss_tc = word_dist.new_zeros(())",
        "loss_bd = word_dist.new_zeros(())",
    )
    present_banned = [term for term in banned if term in haystack]
    if missing or present_banned:
        result.fail(
            "Graphless train zero losses remain in Step3/Step5 paths.",
            [
                Finding(
                    "code/odcr_core/odcr_losses.py",
                    1,
                    f"missing={missing}, banned={present_banned}",
                    "Use ref.sum() * 0.0 or ref * 0.0 for train zero losses.",
                )
            ],
        )
    else:
        result.summary = "Step3 auxiliary and Step5 route/regularizer zero losses stay attached to graph tensors."
    return result


def _check_step5_find_unused_preflight_policy(repo_root: Path) -> RuleResult:
    result = RuleResult("R038", "Step5 find_unused_parameters=false requires preflight")
    config = _read(repo_root / "configs" / "odcr.yaml")
    resolver = _read(repo_root / "code" / "odcr_core" / "config_resolver.py")
    engine = _read(repo_root / "code" / "executors" / "step5_engine.py")
    required = (
        "find_unused_parameters:",
        "find_unused_false_preflight:",
        "_resolve_step5_ddp_config",
        "synthetic_one_batch",
        "run_step5_find_unused_parameters_preflight",
        "Step5 find_unused_parameters=false preflight failed",
    )
    haystack = "\n".join([config, resolver, engine])
    missing = [term for term in required if term not in haystack]
    bad_literal = "find_unused_parameters=False" in engine
    if missing or bad_literal:
        result.fail(
            "Step5 DDP find_unused false path lacks an explicit preflight/fail-fast policy.",
            [
                Finding(
                    "configs/odcr.yaml",
                    1,
                    f"missing={missing}, bad_literal={bad_literal}",
                    "Keep find_unused true by default and require synthetic preflight before false.",
                )
            ],
        )
    else:
        result.summary = "Step5 find_unused false is resolver-gated and runtime-gated by synthetic graph preflight."
    return result


def _check_step5_no_hf_labels_ce_once(repo_root: Path) -> RuleResult:
    result = RuleResult("R039", "Step5 Flan forward must not pass HF labels")
    engine = _read(repo_root / "code" / "executors" / "step5_engine.py")
    labels_hits = [
        Finding("code/executors/step5_engine.py", idx, line.strip(), "Do not pass labels into HF T5/Flan forward.")
        for idx, line in enumerate(engine.splitlines(), start=1)
        if re.search(r"\blabels\s*=", line)
    ]
    required = (
        "per_sample_decoder_ce_from_logits",
        "target_tokens",
        "不传 labels",
    )
    missing = [term for term in required if term not in engine]
    if labels_hits or missing:
        findings = labels_hits or [
            Finding(
                "code/executors/step5_engine.py",
                1,
                "missing: " + ", ".join(missing),
                "HF forward must emit logits only; outer per-sample CE is the sole token CE.",
            )
        ]
        result.fail("Step5 may reintroduce HF internal CE or lose outer CE guard.", findings)
    else:
        result.summary = "Step5 Flan forward emits logits without labels; outer per-sample CE remains the token CE."
    return result


def _check_step5_weighted_lci_fca_once(repo_root: Path) -> RuleResult:
    result = RuleResult("R040", "LCI/FCA weighted losses must enter total once")
    engine = _read(repo_root / "code" / "executors" / "step5_engine.py")
    required = (
        "def compose_step5_total_loss",
        "lci_weighted_loss",
        "fca_weighted_loss",
        "compose_step5_total_loss(",
    )
    missing = [term for term in required if term not in engine]
    banned = (
        "+ lci_bundle.lci_loss",
        "+ fca_bundle.fca_loss",
        "+ l_lci",
        "+ l_fca",
    )
    present_banned = [term for term in banned if term in engine]
    helper_match = re.search(
        r"def compose_step5_total_loss\(.*?return loss",
        engine,
        flags=re.DOTALL,
    )
    helper = helper_match.group(0) if helper_match else ""
    count_lci = helper.count("+ lci_weighted_loss")
    count_fca = helper.count("+ fca_weighted_loss")
    if missing or present_banned or count_lci != 1 or count_fca != 1:
        result.fail(
            "Step5 total loss does not clearly add weighted LCI/FCA exactly once.",
            [
                Finding(
                    "code/executors/step5_engine.py",
                    1,
                    f"missing={missing}, banned={present_banned}, counts={(count_lci, count_fca)}",
                    "Use compose_step5_total_loss and add only weighted LCI/FCA terms once.",
                )
            ],
        )
    else:
        result.summary = "Step5 total loss composition adds weighted LCI and FCA once; raw losses are logging-only."
    return result


def _iter_evolution_active_files(repo_root: Path, suffixes: tuple[str, ...] = (".py", ".sh")) -> Iterable[Path]:
    for path in _iter_repo_files(repo_root, suffixes=suffixes):
        rel = _rel(path, repo_root)
        if _is_evolution_exempt_path(rel):
            continue
        yield path


def _evolution_active_texts(repo_root: Path) -> list[tuple[str, str]]:
    texts: list[tuple[str, str]] = []
    for path in _iter_evolution_active_files(repo_root, suffixes=(".py",)):
        rel = _rel(path, repo_root)
        texts.append((rel, _read(path)))
    return texts


def _check_evolution_active_parameters(repo_root: Path) -> RuleResult:
    result = RuleResult("R042", "new active parameters must join One-Control")
    findings: list[Finding] = []
    for rel, text in _evolution_active_texts(repo_root):
        findings.extend(_scan_r042_text(rel, text))
    required = {
        "configs/odcr.yaml": ("project:", "env:", "step3:", "step4:", "step5:", "eval:"),
        "code/odcr_core/config_schema.py": ("ResolvedConfig", "SourceRecord"),
        "code/odcr_core/config_resolver.py": ("field_sources", "SourceRecord", "resolved_snapshot"),
        "code/odcr.py": ("_print_sources", "doctor", "show"),
    }
    for rel, terms in required.items():
        path = repo_root / rel
        text = _read(path) if path.is_file() else ""
        missing = [term for term in terms if term not in text]
        if missing:
            findings.append(
                Finding(
                    rel,
                    1,
                    "missing: " + ", ".join(missing),
                    "New active parameters must be visible in YAML, schema/resolver, source table, show, and doctor.",
                )
            )
    if findings:
        result.fail("Found active parameter surfaces outside the One-Control parameter chain.", findings)
    else:
        result.summary = "Active parameter additions are guarded for YAML/schema/resolver/source-table ownership or explicit constant/test-only status."
    return result


def _check_evolution_contract_fields(repo_root: Path) -> RuleResult:
    result = RuleResult("R043", "new active CSV/export fields must join contracts")
    contract_parts = []
    for rel in (
        "code/data_contract.py",
        "code/odcr_core/index_contract.py",
        "code/odcr_core/manifests.py",
        "docs/ODCR_EVOLUTION_PROTOCOL.md",
        "docs/ODCR_FEATURE_INTEGRATION_CHECKLIST.md",
    ):
        path = repo_root / rel
        if path.is_file():
            contract_parts.append(_read(path))
    contract_text = "\n".join(contract_parts)
    findings: list[Finding] = []
    for rel, text in _evolution_active_texts(repo_root):
        findings.extend(_scan_r043_text(rel, text, contract_text=contract_text))
    required_terms = (
        "data_contract/schema",
        "producer",
        "consumer",
        "manifest/index_contract",
        "fingerprint",
        "internal-only",
    )
    protocol = _read(repo_root / "docs" / "ODCR_EVOLUTION_PROTOCOL.md")
    missing = [term for term in required_terms if term not in protocol]
    if missing:
        findings.append(
            Finding(
                "docs/ODCR_EVOLUTION_PROTOCOL.md",
                1,
                "missing: " + ", ".join(missing),
                "Document the field integration path and internal-only escape hatch.",
            )
        )
    if findings:
        result.fail("Found active data/export fields without contract or internal-only ownership.", findings)
    else:
        result.summary = "CSV/export field additions are guarded for contract, producer, consumer, manifest/fingerprint, or internal-only status."
    return result


def _check_evolution_artifact_lineage(repo_root: Path) -> RuleResult:
    result = RuleResult("R044", "new cache/checkpoint/export artifacts must carry lineage")
    findings: list[Finding] = []
    for rel, text in _evolution_active_texts(repo_root):
        findings.extend(_scan_r044_text(rel, text))
    required_terms = (
        "schema version",
        "config hash",
        "data contract/export contract version",
        "input artifact fingerprint",
        "model path fingerprint",
        "consumer validation",
        "mismatch fail-fast",
    )
    protocol = _read(repo_root / "docs" / "ODCR_EVOLUTION_PROTOCOL.md")
    missing = [term for term in required_terms if term not in protocol]
    if missing:
        findings.append(
            Finding(
                "docs/ODCR_EVOLUTION_PROTOCOL.md",
                1,
                "missing: " + ", ".join(missing),
                "Keep the cache/checkpoint/export lineage protocol explicit.",
            )
        )
    if findings:
        result.fail("Found cache/checkpoint/export writers or consumers without lineage/fingerprint gates.", findings)
    else:
        result.summary = "Artifact writers/consumers are guarded for lineage, fingerprints, schema versions, and fail-fast validation."
    return result


def _check_evolution_entrypoints(repo_root: Path) -> RuleResult:
    result = RuleResult("R045", "new active scripts must not bypass ./odcr")
    findings: list[Finding] = []
    for path in _iter_evolution_active_files(repo_root, suffixes=(".sh",)):
        rel = _rel(path, repo_root)
        findings.extend(_scan_r045_text(rel, _read(path)))
    if (repo_root / "scripts" / "entrypoints").exists():
        findings.append(
            Finding(
                "scripts/entrypoints",
                1,
                "directory exists",
                "Do not add active ODCR shell entrypoints; route through ./odcr/code/odcr.py.",
            )
        )
    if findings:
        result.fail("Found script entry surfaces that can bypass the canonical ODCR entrypoints.", findings)
    else:
        result.summary = "Future shell/script entry additions are guarded to route through ./odcr or code/odcr.py."
    return result


def _check_evolution_env_reads(repo_root: Path) -> RuleResult:
    result = RuleResult("R046", "new active env reads must not become config sources")
    findings: list[Finding] = []
    for rel, text in _evolution_active_texts(repo_root):
        findings.extend(_scan_r046_text(rel, text))
    if findings:
        result.fail("Found bare env reads outside resolver transport or fail-fast conflict checks.", findings)
    else:
        result.summary = "Active env reads are limited to known resolver transport, diagnostics, or conflict checks; new env config sources fail guardrail."
    return result


def _check_evolution_loss_wiring(repo_root: Path) -> RuleResult:
    result = RuleResult("R047", "new active losses must be wired through total-loss composer")
    texts = _evolution_active_texts(repo_root)
    corpus = "\n".join(text for _, text in texts)
    findings: list[Finding] = []
    for rel, text in texts:
        findings.extend(_scan_r047_text(rel, text, all_active_text=corpus))
    required = (
        "compose_step3_loss_from_forward_output",
        "compose_step5_total_loss",
        "total loss single insertion point",
    )
    docs = _read(repo_root / "docs" / "ODCR_EVOLUTION_PROTOCOL.md")
    haystack = corpus + "\n" + docs
    missing = [term for term in required if term not in haystack]
    if missing:
        findings.append(
            Finding(
                "docs/ODCR_EVOLUTION_PROTOCOL.md",
                1,
                "missing: " + ", ".join(missing),
                "Document and preserve total-loss single insertion points for new losses.",
            )
        )
    if findings:
        result.fail("Found loss definitions that are not clearly wired into the active total-loss path.", findings)
    else:
        result.summary = "Future loss additions are guarded for total-loss insertion or explicit no-op/test-only documentation."
    return result


def _check_evolution_mask_gate_ddp(repo_root: Path) -> RuleResult:
    result = RuleResult("R048", "new mask/gate branches must stay DDP graph-safe")
    findings: list[Finding] = []
    for rel, text in _evolution_active_texts(repo_root):
        findings.extend(_scan_r048_text(rel, text))
    if findings:
        result.fail("Found rank-local mask/gate any() branches without graph-tied zero protection.", findings)
    else:
        result.summary = "Future mask/gate additions are guarded against rank-local any() graph divergence."
    return result


def _check_evolution_legacy_aliases(repo_root: Path) -> RuleResult:
    result = RuleResult("R049", "new legacy aliases must not silently fallback")
    findings: list[Finding] = []
    for rel, text in _evolution_active_texts(repo_root):
        findings.extend(_scan_r049_text(rel, text))
    required_docs = (
        "Delete: remove",
        "Migrate:",
        "Retired/fail-fast:",
        "Docs/history only:",
        "Silent fallback",
    )
    protocol = _read(repo_root / "docs" / "ODCR_EVOLUTION_PROTOCOL.md")
    missing = [term for term in required_docs if term not in protocol]
    if missing:
        findings.append(
            Finding(
                "docs/ODCR_EVOLUTION_PROTOCOL.md",
                1,
                "missing: " + ", ".join(missing),
                "Keep legacy handling choices explicit for every replacement feature.",
            )
        )
    if findings:
        result.fail("Found legacy names or fallback paths without delete/migrate/retired-fail-fast handling.", findings)
    else:
        result.summary = "Legacy aliases and retired fields are guarded against active silent fallback."
    return result


def _check_evolution_checklist_or_ledger(repo_root: Path) -> RuleResult:
    result = RuleResult("R050", "new feature must declare integration path")
    required_files = (
        "AGENTS.md",
        "docs/ODCR_EVOLUTION_PROTOCOL.md",
        "docs/ODCR_FEATURE_INTEGRATION_CHECKLIST.md",
        "docs/CODEX_CHANGE_REQUEST_TEMPLATE.md",
        "code/tools/odcr_post_edit_check.py",
    )
    findings: list[Finding] = []
    texts: dict[str, str] = {}
    for rel in required_files:
        path = repo_root / rel
        if not path.is_file():
            findings.append(
                Finding(
                    rel,
                    1,
                    "missing",
                    "Keep Codex workflow governance files present so future changes classify first and fill the checklist.",
                )
            )
            texts[rel] = ""
        else:
            texts[rel] = _read(path)

    required_terms_by_file = {
        "docs/CODEX_CHANGE_REQUEST_TEMPLATE.md": (
            "Task Objective",
            "Forbidden Actions",
            "Change Type Selection",
            "New parameter",
            "New field",
            "New artifact",
            "New entrypoint",
            "New model/loss/router/verbalizer",
            "Modify configuration control plane",
            "Modify cache/checkpoint/export",
            "Modify logging/metrics/cache/report output",
            "Modify eval/rerank",
            "Delete or migrate old logic",
            "Required Impact Surface",
            "One-Control",
            "YAML/config_schema/config_resolver/source table",
            "Data contract",
            "Manifest/index_contract",
            "Lineage/fingerprint",
            "DDP/loss graph",
            "Eval/rerank",
            "Logging/metrics/cache/report output",
            "Guardrail/tests/docs",
            "Logging / Artifact Output Impact",
            "Output role",
            "Directory rationale",
            "run_summary indexing",
            "latest.json update",
            "Default visibility",
            "Post-Edit Validation:",
            "chosen scope",
            "guardrail strict",
            "failures fixed",
            "final status",
            "python code/tools/odcr_post_edit_check.py --scope <scope>",
            "Required Outputs",
            "Modified files",
            "Old logic handling",
            "Rerun decision",
            "AI_analysis file",
            "Lightweight verification result",
        ),
        "docs/ODCR_FEATURE_INTEGRATION_CHECKLIST.md": (
            "YAML path",
            "schema path",
            "resolver path",
            "source table key",
            "producer",
            "consumer",
            "contract version",
            "manifest key",
            "fingerprint key",
            "mismatch policy",
            "DDP risk",
            "eval/rerank risk",
            "console_output_changed",
            "file_log_added",
            "metrics_file_added",
            "cache_file_added",
            "report_file_added",
            "run_summary_updated",
            "latest_pointer_updated",
            "AI_analysis_output_added",
            "artifact_role",
            "output_directory",
            "retention_policy",
            "verbose_or_default",
            "post_edit_logging_scope",
            "legacy cleanup",
            "guardrail rule",
            "unit test",
            "dry-run command",
            "post_edit_scope",
            "post_edit_check_command",
            "post_edit_check_result",
            "validation_block_in_final_response",
            "rerun decision",
            "Evolution Guardrail Coverage",
            "R042",
            "R043",
            "R044",
            "R045",
            "R046",
            "R047",
            "R048",
            "R049",
            "R050",
            "R051",
            "R052",
            "R053",
            "R068",
            "R069",
            "R070",
            "R071",
            "R072",
            "AI_analysis evidence path",
        ),
        "docs/ODCR_EVOLUTION_PROTOCOL.md": (
            "CODEX_CHANGE_REQUEST_TEMPLATE.md",
            "Classify the change type",
            "Fill or mirror",
            "AI_analysis",
            "Future changes must not skip checklist coverage",
            "Logging And Artifact Evolution",
            "Artifact role",
            "run_summary",
            "latest.json",
            "R051",
            "R052",
            "R053",
            "R068",
            "R069",
            "R070",
            "R071",
            "R072",
        ),
        "AGENTS.md": (
            "CODEX_CHANGE_REQUEST_TEMPLATE.md",
            "ODCR_EVOLUTION_PROTOCOL.md",
            "ODCR_FEATURE_INTEGRATION_CHECKLIST.md",
            "odcr_post_edit_check.py",
            "must not skip checklist",
            "AI_analysis ledger",
            "at most one interim status update",
            "python code/tools/odcr_post_edit_check.py --scope <scope>",
            "must not wait for git commit",
            "must not leave validation to the user",
            "logging/metrics/cache/report output",
            "R068",
            "R072",
        ),
        "code/tools/odcr_post_edit_check.py": (
            "DEFAULT_SCOPE",
            "governance",
            "SCOPES",
            "plan_safety_violations",
            "compileall",
            "check_one_control_guardrails.py",
            "--dry-run",
            "--max-seconds",
        ),
    }
    for rel, terms in required_terms_by_file.items():
        text = texts.get(rel, "")
        normalized = " ".join(text.split())
        missing = [term for term in terms if term not in text and term not in normalized]
        if missing:
            findings.append(
                Finding(
                    rel,
                    1,
                    "missing: " + ", ".join(missing),
                    "Every feature must classify first, fill or mirror the checklist, write an AI_analysis ledger, and name affected guardrails.",
                )
            )

    if findings:
        result.fail(
            "Codex workflow governance files do not fully enforce classify-first/checklist-first handoff.",
            findings,
        )
    else:
        result.summary = (
            "Codex workflow governance files exist and require classify-first, checklist-first, "
            "AI_analysis-ledger handoff before future code changes."
        )
    return result


def _check_post_edit_script_exists(repo_root: Path) -> RuleResult:
    result = RuleResult("R051", "post-edit validation script must exist")
    path = repo_root / "code" / "tools" / "odcr_post_edit_check.py"
    if not path.is_file():
        result.fail(
            "The unified post-edit validation script is missing.",
            [
                Finding(
                    "code/tools/odcr_post_edit_check.py",
                    1,
                    "missing",
                    "Restore the post-edit gate so Codex can validate before final response.",
                )
            ],
        )
    else:
        result.summary = "code/tools/odcr_post_edit_check.py is present."
    return result


def _check_codex_workflow_requires_post_edit(repo_root: Path) -> RuleResult:
    result = RuleResult("R052", "Codex workflow must require post-edit validation")
    required_files = (
        "AGENTS.md",
        "docs/CODEX_CHANGE_REQUEST_TEMPLATE.md",
    )
    required_terms = (
        "post-edit validation suite",
        "python code/tools/odcr_post_edit_check.py --scope <scope>",
        "must not wait for git commit",
        "must not leave validation to the user",
        "fix and rerun",
        "Validation block",
        "tmux -L odcr_gpu new-session -A -s odcr",
        "preprocess_b/c",
    )
    findings: list[Finding] = []
    for rel in required_files:
        path = repo_root / rel
        if not path.is_file():
            findings.append(
                Finding(
                    rel,
                    1,
                    "missing",
                    "Codex workflow docs must require post-edit validation before final response.",
                )
            )
            continue
        text = _read(path)
        normalized = " ".join(text.split())
        missing = [term for term in required_terms if term not in text and term not in normalized]
        if missing:
            findings.append(
                Finding(
                    rel,
                    1,
                    "missing: " + ", ".join(missing),
                    "State that Codex runs post-edit validation after edits, before final response, without waiting for commit.",
                )
            )
    if findings:
        result.fail(
            "Codex workflow docs do not fully require post-edit validation before final response.",
            findings,
        )
    else:
        result.summary = (
            "AGENTS.md and CODEX_CHANGE_REQUEST_TEMPLATE.md require post-edit validation, "
            "failure fix/rerun, and the final Validation block before handoff."
        )
    return result


def _check_gpu_tmux_policy_docs(repo_root: Path) -> RuleResult:
    result = RuleResult("R096", "GPU runtime-first policy must not restore whitelist or post-edit GPU gates")
    checks = {
        "AGENTS.md": (
            "tmux -L odcr_gpu new-session -A -s odcr",
            "odcr-enter-gpu <JOBID>",
            "Codex must not execute `odcr-enter-gpu`, `srun`, `sbatch`, or `scancel`",
            "must not create, kill, or switch tmux",
            "current tmux session's real-time CUDA",
            "GPU use is allowed by default",
            "repo-local validation, probe, and bounded runtime",
            "post-edit full is not a GPU prerequisite",
            "runtime evidence takes priority over static full-suite instability",
            "user-created, already-entered, uniquely validated GPU pane",
            "not arbitrary send-keys",
            "formal namespace guard",
            "AI_analysis",
            "normal admin shell",
            "old `AI_analysis` probe output",
        ),
        "docs/CODEX_CHANGE_REQUEST_TEMPLATE.md": (
            "tmux itself as a GPU",
            "odcr-enter-gpu <JOBID>",
            "Codex must not execute `odcr-enter-gpu`, `srun`, `sbatch`, or `scancel`",
            "current tmux session's real-time CUDA",
            "GPU use is allowed by default",
            "repo-local validation, probe, and bounded runtime",
            "post-edit full is not a GPU prerequisite",
            "user-created, already-entered, uniquely validated GPU pane",
            "not arbitrary send-keys",
            "formal namespace guard",
            "AI_analysis",
            "old `AI_analysis` probe",
        ),
        "docs/ODCR_ACTIVE_ARCHITECTURE.md": (
            "not itself a GPU allocation",
            "odcr-enter-gpu <JOBID>",
            "Codex does not manage GPU allocation",
            "current tmux session's real-time CUDA",
            "GPU use is allowed by default",
            "repo-local validation, probe, and bounded runtime",
            "post-edit full is not a GPU prerequisite",
            "user-created, already-entered, uniquely validated GPU pane",
            "not arbitrary send-keys",
            "formal namespace guard",
            "AI_analysis",
            "old `AI_analysis` probe output",
        ),
        "docs/ODCR_EVOLUTION_PROTOCOL.md": (
            "tmux is only the shared session boundary",
            "odcr-enter-gpu <JOBID>",
            "Codex must not execute `odcr-enter-gpu`, `srun`, `sbatch`, or `scancel`",
            "current tmux session's real-time CUDA",
            "GPU use is allowed by default",
            "repo-local validation, probe, and bounded runtime",
            "post-edit full is not a GPU prerequisite",
            "user-created, already-entered, uniquely validated GPU pane",
            "not arbitrary send-keys",
            "formal namespace guard",
            "AI_analysis",
            "old `AI_analysis` probe output",
            "R096",
        ),
        "docs/ODCR_ARCHITECTURE_CONTRACT.md": (
            "does not by itself prove GPU availability",
            "odcr-enter-gpu <JOBID>",
            "Codex must not execute `odcr-enter-gpu`, `srun`, `sbatch`, or `scancel`",
            "current tmux session's real-time CUDA",
            "GPU use is allowed by default",
            "repo-local validation, probe, and bounded runtime",
            "post-edit full is not a GPU prerequisite",
            "user-created, already-entered, uniquely validated GPU pane",
            "not arbitrary send-keys",
            "formal namespace guard",
            "AI_analysis",
            "old `AI_analysis` probe output",
        ),
        "docs/AI_PROJECT_CANONICAL.md": (
            "tmux itself is not a GPU allocation",
            "odcr-enter-gpu <JOBID>",
            "Codex must not execute `odcr-enter-gpu`, `srun`, `sbatch`, or `scancel`",
            "current tmux session's real-time CUDA",
            "GPU use is allowed by default",
            "repo-local validation, probe, and bounded runtime",
            "post-edit full is not a GPU prerequisite",
            "user-created, already-entered, uniquely validated GPU pane",
            "not arbitrary send-keys",
            "formal namespace guard",
            "AI_analysis",
            "old `AI_analysis` probe output",
        ),
        "docs/ODCR_GPU_RUNTIME_FIRST_EXECUTION_CONTRACT.md": (
            "GPU use is allowed by default",
            "repo-local validation, probe, and bounded runtime",
            "No GPU whitelist hard blocker",
            "post-edit full is not a GPU prerequisite",
            "fast sanity",
            "formal namespace guard",
            "formal full train still requires user confirmation",
            "runtime evidence takes priority over static full-suite instability",
            "Stage2 candidate selection uses real runtime probes",
        ),
        "docs/ODCR_TMUX_GPU_BRIDGE_STEP3_VALIDATION.md": (
            "Codex/admin shell is not the GPU shell",
            "step3-startup-validation",
            "fresh discover",
            "fresh validate",
            "odcr-enter-gpu <JOBID>",
            "Codex must not execute `odcr-enter-gpu`, `srun`, `sbatch`, or `scancel`",
            "does not create, kill, switch, or attach tmux",
            "not formal",
            "not short-pilot",
            "not a parameter experiment",
            "AI_analysis/06_probe_evidence",
            "runs/step3_validation",
        ),
    }
    forbidden_patterns = (
        re.compile(r"tmux session exists\s+(?:means|equals|=|is equivalent to)\s+GPU", re.IGNORECASE),
        re.compile(r"tmux session exists\s+(?:means|equals|=|is equivalent to)\s+CUDA", re.IGNORECASE),
        re.compile(r"--scope all\s+is\s+(?:forbidden|disabled|removed)", re.IGNORECASE),
        re.compile(r"--scope all.{0,40}(?:permanently\s+(?:forbidden|disabled|removed))", re.IGNORECASE),
        re.compile(r"(?:permanently\s+(?:forbid|disable|remove)).{0,40}--scope all", re.IGNORECASE),
        re.compile(r"post-edit.{0,80}only.{0,20}config/preprocess", re.IGNORECASE),
        re.compile(r"only.{0,20}config/preprocess.{0,80}post-edit", re.IGNORECASE),
        re.compile(r"Codex\s+(?:should|must|may|can)\s+(?:run|execute|start).{0,40}odcr-enter-gpu", re.IGNORECASE),
        re.compile(r"Codex\s+(?:should|must|may|can)\s+(?:run|execute|start).{0,40}srun", re.IGNORECASE),
        re.compile(r"Codex\s+(?:should|must|may|can)\s+(?:run|execute|start).{0,40}sbatch", re.IGNORECASE),
        re.compile(r"arbitrary\s+(?:tmux\s+)?send-keys\s+(?:is|are)\s+(?:allowed|permitted)", re.IGNORECASE),
        re.compile(r"whitelist short validation scripts", re.IGNORECASE),
        re.compile(r"closed-choice whitelist", re.IGNORECASE),
        re.compile(r"post-edit full.{0,40}(?:must|required|pass).{0,80}(?:GPU prerequisite|before GPU|GPU gate)", re.IGNORECASE),
    )
    findings: list[Finding] = []
    for rel, terms in checks.items():
        path = repo_root / rel
        if not path.is_file():
            findings.append(Finding(rel, 1, "missing", "Keep GPU/tmux governance docs present."))
            continue
        text = _read(path)
        normalized = " ".join(text.split())
        missing = [term for term in terms if term not in text and term not in normalized]
        if missing:
            findings.append(
                Finding(
                    rel,
                    1,
                    "missing: " + ", ".join(missing),
                    "Document runtime-first GPU use, user-managed allocation, current-tmux CUDA evidence, bridge executor boundaries, and formal namespace guards.",
                )
            )
        for pattern in forbidden_patterns:
            match = pattern.search(normalized)
            if match:
                findings.append(
                    Finding(
                        rel,
                        1,
                        match.group(0)[:240],
                        "Do not imply tmux equals GPU, Codex manages GPU allocation, arbitrary send-keys is allowed, --scope all is permanently banned, or whitelist/post-edit gates are required for GPU.",
                    )
                )
    bridge = _read(repo_root / "code" / "tools" / "odcr_tmux_gpu_bridge.py")
    post_edit = _read(repo_root / "code" / "tools" / "odcr_post_edit_check.py")
    startup = _read(repo_root / "code" / "tools" / "odcr_step3_startup_validation.py")
    bridge_required = (
        '"step3-startup-validation"',
        '"repo-command"',
        '"repo-script"',
        '"repo-module"',
        '"command-file"',
        "OPERATION_SPECS",
        "build_repo_runtime_executor_script",
        "validate_runtime_command_safety",
        "resolve_runtime_output_dir",
        "user_confirmed_formal",
        "formal_namespace_blocked",
        "gpu_transport_ok",
        "runtime_evidence_ok",
        "formal_pollution",
        "build_step3_startup_validation_script",
        "code/tools/odcr_step3_startup_validation.py",
        "--mode startup-only",
        "--namespace validation",
        "target_unique_slurm_gpu_repo_nvidia_smi_and_torch_cuda_2plus_valid",
        "state file",
        "Retired",
        "generated bridge script contains forbidden token",
        "formal step3",
    )
    for term in bridge_required:
        if term not in bridge:
            findings.append(
                Finding(
                    "code/tools/odcr_tmux_gpu_bridge.py",
                    1,
                    "missing: " + term,
                    "Bridge must support runtime-first repo-local execution while preserving formal namespace guards.",
                )
            )
    post_edit_required = (
        "resource_kill",
        "flaky_resource_kill",
        "classification_blocks_gpu_probe",
        "post_edit_results_block_gpu_probe",
        "post_edit_results_block_formal",
        "odcr_post_edit_diagnostic/1",
    )
    for term in post_edit_required:
        if term not in post_edit:
            findings.append(
                Finding(
                    "code/tools/odcr_post_edit_check.py",
                    1,
                    "missing: " + term,
                    "Post-edit diagnostics must classify resource kills and must not be a GPU probe gate.",
                )
            )
    if "MODE_SPECS" in bridge:
        findings.append(
            Finding(
                "code/tools/odcr_tmux_gpu_bridge.py",
                1,
                "MODE_SPECS",
                "Do not restore MODE_SPECS as a whitelist hard blocker; use operation metadata plus runtime namespace guards.",
            )
        )
    specs_start = bridge.find("OPERATION_SPECS")
    specs_end = bridge.find("GLOBAL_MAX_TIMEOUT_S", specs_start)
    specs_block = bridge[specs_start:specs_end] if specs_start >= 0 and specs_end > specs_start else ""
    parser_start = bridge.find("def build_parser")
    parser_end = bridge.find("def options_from_args", parser_start)
    parser_block = bridge[parser_start:parser_end] if parser_start >= 0 and parser_end > parser_start else ""
    for retired in ('"step3-ddp-smoke"', '"step3-short-pilot"'):
        if retired in specs_block or retired in parser_block:
            findings.append(
                Finding(
                    "code/tools/odcr_tmux_gpu_bridge.py",
                    1,
                    retired,
                    "Retired Step3 bridge probes must not be exposed by MODE_SPECS or argparse.",
                )
            )
    for term in (
        '"step3-performance-probe"',
        "STEP3_PERFORMANCE_PROBE_TYPES",
        "build_step3_performance_probe_script",
        "code/tools/odcr_step3_performance_probe.py",
        "--namespace validation",
    ):
        if term not in bridge:
            findings.append(
                Finding(
                    "code/tools/odcr_tmux_gpu_bridge.py",
                    1,
                    "missing: " + term,
                    "Step3 performance probe must stay validation-only and non-formal while GPU execution is runtime-first.",
                )
            )
    startup_required = (
        "SCHEMA_VERSION = \"odcr_step3_startup_validation/1\"",
        "validation_mode",
        "startup-only",
        "validation_namespace",
        "write_run_summary_json",
        "update_latest=False",
        "distributed_collective_calls_in_cache_phase",
        "nccl_init_after_cache_ready",
        "formal_namespace_polluted",
        "formal_latest_updated",
        "checkpoint_created",
        "training_loop_full_epoch_started",
        "TOKENIZERS_PARALLELISM",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "num_proc",
        "max_parallel_cpu",
        "reserved_cpu",
        "tokenization_formula",
        "worker_formula",
        "Current tmux does not expose CUDA.",
    )
    for term in startup_required:
        if term not in startup:
            findings.append(
                Finding(
                    "code/tools/odcr_step3_startup_validation.py",
                    1,
                    "missing: " + term,
                    "Step3 startup validation must be AI_analysis/runs step3_validation-only, root-signature capable, and no-checkpoint.",
                )
            )
    if findings:
        result.fail("GPU/tmux policy docs are missing required boundaries or contain stale claims.", findings)
    else:
        result.summary = (
            "GPU/tmux docs require runtime-first repo-local GPU validation, user-managed GPU-node entry, "
            "current-tmux CUDA probes, AI_analysis output, and formal namespace protection."
        )
    return result


def _load_post_edit_module(repo_root: Path):
    path = repo_root / "code" / "tools" / "odcr_post_edit_check.py"
    spec = importlib.util.spec_from_file_location("_odcr_post_edit_check_guardrail", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_bridge_module(repo_root: Path):
    path = repo_root / "code" / "tools" / "odcr_tmux_gpu_bridge.py"
    spec = importlib.util.spec_from_file_location("_odcr_tmux_gpu_bridge_guardrail", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _check_step3_runtime_probe_truth_contract(repo_root: Path) -> RuleResult:
    result = RuleResult("R110", "Step3 performance probe must require runtime truth evidence")
    files = {
        "code/tools/odcr_step3_performance_probe.py": _read(repo_root / "code" / "tools" / "odcr_step3_performance_probe.py"),
        "code/tools/odcr_tmux_gpu_bridge.py": _read(repo_root / "code" / "tools" / "odcr_tmux_gpu_bridge.py"),
        "code/odcr_core/step3_runtime_probe.py": _read(repo_root / "code" / "odcr_core" / "step3_runtime_probe.py"),
        "code/odcr_core/stage2_runtime_first.py": _read(repo_root / "code" / "odcr_core" / "stage2_runtime_first.py"),
        "code/executors/step3_train_core.py": _read(repo_root / "code" / "executors" / "step3_train_core.py"),
        "code/tools/odcr_post_edit_check.py": _read(repo_root / "code" / "tools" / "odcr_post_edit_check.py"),
        "docs/ODCR_STEP3_RUNTIME_PROBE_TRUTH_CONTRACT.md": _read(repo_root / "docs" / "ODCR_STEP3_RUNTIME_PROBE_TRUTH_CONTRACT.md"),
    }
    required = {
        "code/tools/odcr_step3_performance_probe.py": (
            "run_step3_validation_window",
            "child_status_from_report",
            "runtime_verified",
            "evidence_complete",
            "return int(status[\"exit_code\"])",
        ),
        "code/tools/odcr_tmux_gpu_bridge.py": (
            "bridge_transport_ok",
            "child_process_ok",
            "runtime_probe_ok",
            "evidence_complete",
            "normalize_bridge_runtime_success",
            "bounded_step3_hot_path_runtime_verified_and_evidence_complete",
        ),
        "code/odcr_core/step3_runtime_probe.py": (
            "Step3ValidationNamespaceGuard",
            "Step3RuntimeEvidenceSink",
            "run_step3_validation_window",
            "evaluate_stage2_probe_evidence",
            "runtime_verified",
            "evidence_complete",
            "timing rows count for rank0 < measured_steps",
            "plan_only",
        ),
        "code/odcr_core/stage2_runtime_first.py": (
            "runtime_first_flow_order",
            "post_edit_diagnostic_blocks_gpu_probe",
            "select_stage2_candidate",
            "PREREQUISITE_RUNTIME_PROBES",
        ),
        "code/executors/step3_train_core.py": (
            "build_step3_training_components",
            "run_step3_measured_steps",
            "run_step3_validation_window",
        ),
        "code/tools/odcr_post_edit_check.py": (
            "test_gpu_bridge_no_whitelist_hard_blocker.py",
            "test_gpu_runtime_executor_namespace_guard.py",
            "test_post_edit_not_gpu_gate.py",
            "test_stage2_runtime_first_flow.py",
            "test_stage2_candidate_selection_uses_runtime_evidence.py",
            "test_step3_performance_probe_requires_runtime_verified.py",
            "test_step3_performance_probe_rejects_status_only.py",
            "test_tmux_gpu_bridge_runtime_success_semantics.py",
            "test_step3_performance_probe_metrics_required.py",
            "test_step3_validation_namespace_guard.py",
            "test_step3_bounded_hot_path_entry.py",
            "test_stage2_collector_rejects_null_summaries.py",
            "test_evidence_level_no_overclaim.py",
        ),
        "docs/ODCR_STEP3_RUNTIME_PROBE_TRUTH_CONTRACT.md": (
            "bridge_transport_ok is not runtime_probe_ok",
            "runtime_verified=false must fail",
            "metrics all null must fail",
            "status-only/plan-only is not performance-probe",
        ),
    }
    findings: list[Finding] = []
    for rel, terms in required.items():
        if not files[rel]:
            findings.append(Finding(rel, 1, "missing file", "Keep Step3 runtime probe truth contract active."))
            continue
        missing = [term for term in terms if term not in files[rel]]
        if missing:
            findings.append(
                Finding(
                    rel,
                    1,
                    "missing: " + ", ".join(missing),
                    "Step3 performance probe truth semantics must not regress to status-only or transport-only success.",
                )
            )
    probe_text = files["code/tools/odcr_step3_performance_probe.py"].lower()
    if "build_probe_plan" in probe_text or "status writer" in probe_text:
        findings.append(
            Finding(
                "code/tools/odcr_step3_performance_probe.py",
                1,
                "plan/status writer token",
                "Do not let status-only or plan-only code masquerade as step3-performance-probe.",
            )
        )
    if findings:
        result.fail("Step3 runtime probe truth guardrail failed.", findings)
    else:
        result.summary = (
            "Step3 performance probe requires bounded hot-path runtime, complete evidence rows, "
            "validation namespace, bridge success split, and Stage2 null-summary rejection."
        )
    return result


def _check_step3_paper_eval_explicit_damping_contract(repo_root: Path) -> RuleResult:
    result = RuleResult("R111", "Step3 paper eval and V3 recovery contract must stay active")
    files = {
        "configs/odcr.yaml": _read(repo_root / "configs" / "odcr.yaml"),
        "code/odcr_core/step3_eval_protocol.py": _read(repo_root / "code" / "odcr_core" / "step3_eval_protocol.py"),
        "code/executors/step3_train_core.py": _read(repo_root / "code" / "executors" / "step3_train_core.py"),
        "code/odcr_core/config_resolver.py": _read(repo_root / "code" / "odcr_core" / "config_resolver.py"),
        "code/odcr_core/manifests.py": _read(repo_root / "code" / "odcr_core" / "manifests.py"),
        "code/tools/odcr_post_edit_check.py": _read(repo_root / "code" / "tools" / "odcr_post_edit_check.py"),
        "docs/ODCR_STEP3_EVAL_AND_EFFECTIVENESS_CONTRACT.md": _read(repo_root / "docs" / "ODCR_STEP3_EVAL_AND_EFFECTIVENESS_CONTRACT.md"),
        "docs/ODCR_STEP3_V3_TRAINING_POLICY.md": _read(repo_root / "docs" / "ODCR_STEP3_V3_TRAINING_POLICY.md"),
    }
    required = {
        "configs/odcr.yaml": (
            "name: warmup_cosine",
            "safe_damping_v2",
            "objective_drift:",
            "recovery:",
            "phase_loss_schedule:",
            "paper_candidate_selection:",
            "paper_target_only_eval",
            "max_ref_len: 25",
            "berts_score_enabled: false",
        ),
        "code/odcr_core/step3_eval_protocol.py": (
            "PAPER_TARGET_ONLY_EVAL",
            "PREDICTION_SHARD_REQUIRED_FIELDS",
            "sample_integrity_report",
            "compare_eval_batch_outputs",
            "scheduler_semantics",
            "build_training_effectiveness_record",
            "bertscore_enabled",
        ),
        "code/executors/step3_train_core.py": (
            "prediction_shards",
            "metrics_from_prediction_rows",
            "dist.destroy_process_group()",
            "safe_damping_v2",
            "detect_objective_drift",
            "build_recovery_plan",
            "resolve_phase_for_epoch",
            "append_step3_training_effectiveness_jsonl",
            "write_step3_loss_component_trends_json",
        ),
        "code/odcr_core/config_resolver.py": (
            "hidden Step3 LR damping is forbidden",
            "_resolve_step3_objective_drift_config",
            "_resolve_step3_recovery_config",
            "_resolve_step3_phase_loss_schedule_config",
            "_resolve_step3_paper_candidate_selection_config",
            "paper_target_only_eval must use max_ref_len=max_decode_len=25",
            "step3_eval_protocol",
        ),
        "code/odcr_core/manifests.py": (
            "train_status",
            "eval_status",
            "post_train_eval",
            "step3_eval_status.json",
        ),
        "code/tools/odcr_post_edit_check.py": (
            "test_step3_eval_two_phase_no_barrier_after_cpu_metric.py",
            "test_scheduler_safe_damping_v2_semantics.py",
            "test_step3_v3_recovery_conflict_paper_selection.py",
            "test_training_effectiveness_gate_plateau.py",
            "test_run_status_train_eval_split.py",
        ),
        "docs/ODCR_STEP3_EVAL_AND_EFFECTIVENESS_CONTRACT.md": (
            "paper_target_only_eval",
            "BERTScore",
            "Two-Phase Runtime",
            "Batch Invariance",
            "Safe Damping",
            "Training Effectiveness",
        ),
        "docs/ODCR_STEP3_V3_TRAINING_POLICY.md": (
            "Objective Drift",
            "Recovery",
            "Phase-Wise",
            "Paper-Aware",
            "DIST",
        ),
    }
    findings: list[Finding] = []
    for rel, terms in required.items():
        if not files[rel]:
            findings.append(Finding(rel, 1, "missing file", "Keep the Step3 eval/effectiveness contract active."))
            continue
        missing = [term for term in terms if term not in files[rel]]
        if missing:
            findings.append(
                Finding(
                    rel,
                    1,
                    "missing: " + ", ".join(missing),
                    "Preserve paper target-only eval, no-BERTScore metrics, two-phase eval, V3 recovery, and status split.",
                )
            )
    eval_block = files["code/executors/step3_train_core.py"]
    if "def _run_eval_ddp" in eval_block and "metrics_from_prediction_rows(" in eval_block:
        block = eval_block[eval_block.index("def _run_eval_ddp") :]
        metric_idx = block.index("metrics_from_prediction_rows(")
        if "dist.barrier()" in block[metric_idx:]:
            findings.append(
                Finding(
                    "code/executors/step3_train_core.py",
                    1,
                    "dist.barrier() after CPU metric path",
                    "Destroy the process group before CPU-heavy metrics and do not re-enter NCCL barriers afterward.",
                )
            )
    if findings:
        result.fail("Step3 paper eval/effectiveness/V3 guardrail failed.", findings)
    else:
        result.summary = "Paper target-only eval, shard sample_id integrity, no-BERTScore metrics, safe V3 recovery policy, and train/eval status split are present."
    return result


def _post_edit_plan_violations(module, repo_root: Path) -> list[str]:
    violations: list[str] = []
    scopes = getattr(module, "SCOPES", ())
    build_plan = getattr(module, "build_plan", None)
    safety = getattr(module, "plan_safety_violations", None)
    if not scopes or build_plan is None:
        return ["odcr_post_edit_check.py does not expose SCOPES/build_plan"]
    for scope in scopes:
        commands = build_plan(scope, repo_root=repo_root, python_executable=sys.executable)
        if safety is not None:
            violations.extend(f"{scope}: {item}" for item in safety(commands))
            continue
        for command in commands:
            argv = tuple(getattr(command, "argv", ()))
            display = command.display() if hasattr(command, "display") else " ".join(argv)
            if len(argv) >= 2 and argv[0] == "./odcr":
                subcommand = argv[1]
                if subcommand == "preprocess":
                    violations.append(f"{scope}: real preprocess command: {display}")
                elif subcommand in {"step3", "step4", "step5"} and "--dry-run" not in argv:
                    violations.append(f"{scope}: real {subcommand} command: {display}")
                elif subcommand in {"eval", "rerank"}:
                    violations.append(f"{scope}: real {subcommand} command: {display}")
    return violations


def _load_stop_hook_module(repo_root: Path):
    path = repo_root / ".codex" / "hooks" / "odcr_post_edit_stop.py"
    spec = importlib.util.spec_from_file_location("_odcr_stop_hook_guardrail", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_guardrail_transcript(path: Path, touched_files: Sequence[str]) -> None:
    rows = []
    for touched in touched_files:
        rows.append(
            {
                "type": "tool_use",
                "name": "functions.apply_patch",
                "arguments": (
                    "*** Begin Patch\n"
                    f"*** Update File: {touched}\n"
                    "@@\n"
                    " unchanged\n"
                    "*** End Patch\n"
                ),
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def _check_post_edit_no_real_training(repo_root: Path) -> RuleResult:
    result = RuleResult("R053", "post-edit validation must not run real training by default")
    script = repo_root / "code" / "tools" / "odcr_post_edit_check.py"
    if not script.is_file():
        result.fail(
            "Cannot validate post-edit command safety because the script is missing.",
            [
                Finding(
                    "code/tools/odcr_post_edit_check.py",
                    1,
                    "missing",
                    "Restore the post-edit gate and keep default scope plans lightweight.",
                )
            ],
        )
        return result
    try:
        module = _load_post_edit_module(repo_root)
        violations = _post_edit_plan_violations(module, repo_root)
    except Exception as exc:  # pragma: no cover - surfaced as guardrail evidence.
        result.fail(
            "Could not inspect post-edit validation dry-run plans.",
            [
                Finding(
                    "code/tools/odcr_post_edit_check.py",
                    1,
                    repr(exc),
                    "Keep odcr_post_edit_check.py import-safe and expose build_plan/SCOPES.",
                )
            ],
        )
        return result
    if violations:
        result.fail(
            "Post-edit validation scopes include real preprocess/training/eval/rerank commands.",
            [
                Finding(
                    "code/tools/odcr_post_edit_check.py",
                    1,
                    violation,
                    "Use show/dry-run and lightweight tests only; real runs require explicit user authorization.",
                )
                for violation in violations
            ],
        )
    else:
        result.summary = "All post-edit scopes use compile/static/show/dry-run/tests and avoid real stage execution."
    return result


def _check_codex_stop_hook_exists(repo_root: Path) -> RuleResult:
    result = RuleResult("R054", "Codex Stop hook must run post-edit validation")
    config = repo_root / ".codex" / "hooks.json"
    wrapper = repo_root / ".codex" / "hooks" / "odcr_post_edit_stop.sh"
    script = repo_root / ".codex" / "hooks" / "odcr_post_edit_stop.py"
    findings: list[Finding] = []
    if not config.is_file():
        findings.append(
            Finding(
                ".codex/hooks.json",
                1,
                "missing",
                "Add repo-local Codex Hooks config with a Stop command for ODCR post-edit validation.",
            )
        )
    if not wrapper.is_file():
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.sh",
                1,
                "missing",
                "Add the stable Stop hook wrapper and call it from .codex/hooks.json.",
            )
        )
    if not script.is_file():
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.py",
                1,
                "missing",
                "Add the Stop hook script that invokes code/tools/odcr_post_edit_check.py.",
            )
        )
    if findings:
        result.fail("Repo-local Codex Stop hook files are missing.", findings)
        return result

    try:
        raw = json.loads(_read(config))
    except json.JSONDecodeError as exc:
        result.fail(
            ".codex/hooks.json is not valid JSON.",
            [Finding(".codex/hooks.json", 1, str(exc), "Keep the Codex Hooks config parseable JSON.")],
        )
        return result

    stop_entries = raw.get("hooks", {}).get("Stop", []) if isinstance(raw, dict) else []
    command_entries: list[dict[str, object]] = []
    if isinstance(stop_entries, list):
        for entry in stop_entries:
            if not isinstance(entry, dict):
                continue
            hooks = entry.get("hooks", [])
            if isinstance(hooks, list):
                command_entries.extend(item for item in hooks if isinstance(item, dict))
    matching = [
        item
        for item in command_entries
        if item.get("type") == "command"
        and str(item.get("command", "")) == HOOK_STOP_COMMAND
    ]
    if not matching:
        findings.append(
            Finding(
                ".codex/hooks.json",
                1,
                "Stop hook command missing absolute odcr_post_edit_stop.sh wrapper",
                f"Wire Stop to the cwd-independent command: {HOOK_STOP_COMMAND}",
            )
        )
    for item in command_entries:
        command = str(item.get("command", ""))
        if re.search(r"(^|\s)\.codex/hooks/odcr_post_edit_stop\.sh(?:\s|$)", command):
            findings.append(
                Finding(
                    ".codex/hooks.json",
                    1,
                    command,
                    "Do not use a cwd-sensitive .codex/hooks/... relative path in hooks.json.",
                )
            )
        if "/usr/bin/python3" in command:
            findings.append(
                Finding(
                    ".codex/hooks.json",
                    1,
                    command,
                    "Do not hardcode /usr/bin/python3 in hooks.json; delegate to the wrapper so it can discover Python.",
                )
            )
        if "$(git rev-parse" in command:
            findings.append(
                Finding(
                    ".codex/hooks.json",
                    1,
                    command,
                    "Do not use shell git substitution in hooks.json; the wrapper/Python hook locates the repo root.",
                )
            )
        if command and command != HOOK_STOP_COMMAND:
            findings.append(
                Finding(
                    ".codex/hooks.json",
                    1,
                    command,
                    "Stop command must be the fixed absolute bash wrapper command.",
                )
            )
        status_message = item.get("statusMessage")
        if status_message != "Running ODCR post-edit validation":
            findings.append(
                Finding(
                    ".codex/hooks.json",
                    1,
                    f"statusMessage={status_message!r}",
                    "Keep the Stop hook statusMessage stable for Codex UI diagnostics.",
                )
            )
    for item in matching:
        timeout = item.get("timeout")
        if not isinstance(timeout, int) or timeout > 180:
            findings.append(
                Finding(
                    ".codex/hooks.json",
                    1,
                    f"timeout={timeout!r}",
                    "Use an automatic Stop hook timeout of 180 seconds or less; manual deep checks can use 900 seconds.",
                )
            )
    if findings:
        result.fail("Codex Stop hook config does not match the ODCR post-edit workflow.", findings)
    else:
        result.summary = "Repo-local .codex/hooks.json wires Stop to the absolute stable wrapper with a fast automatic timeout."
    return result


def _check_codex_hook_script_safe(repo_root: Path) -> RuleResult:
    result = RuleResult("R055", "Codex hook script must delegate safely")
    wrapper = repo_root / ".codex" / "hooks" / "odcr_post_edit_stop.sh"
    script = repo_root / ".codex" / "hooks" / "odcr_post_edit_stop.py"
    findings: list[Finding] = []
    if not wrapper.is_file():
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.sh",
                1,
                "missing",
                "Add the shell wrapper that finds the repo root and Python interpreter.",
            )
        )
    if not script.is_file():
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.py",
                1,
                "missing",
                "Add the Python Stop hook and delegate to odcr_post_edit_check.py.",
            )
        )
    if findings:
        result.fail(
            "Cannot inspect Codex Stop hook launcher because required files are missing.",
            findings,
        )
        return result

    wrapper_text = _read(wrapper)
    text = _read(script)
    required = (
        "code",
        "tools",
        "odcr_post_edit_check.py",
        "--scope",
        "--max-seconds",
        "skip",
        "governance-fast",
        "AI_analysis",
        "01_raw_logs",
        "codex_hooks",
        "runtime_last.json",
        "runtime_",
        "schema_version",
        "odcr_codex_hook_runtime/2.2",
        "post_edit_returncode",
        "failure_stage",
        "timed_out",
        "started_at",
        "post_edit_started_at",
        "finished_at",
        "post_edit_stdout",
        "post_edit_stderr",
        "DEFAULT_WRAPPER_TIMEOUT_SECONDS = 180",
        "DEFAULT_HOOK_CHILD_MAX_SECONDS = 120",
        "ODCR_HOOK_MAX_SECONDS",
        "ODCR_HOOK_CHILD_MAX_SECONDS",
        "child_timeout_seconds",
        "wrapper_timeout_seconds",
        "apply_automatic_stop_scope_policy",
        "auto_all_scope_degraded_to_governance_fast",
        "MANUAL_ALL_FOLLOWUP_COMMAND",
        "MAX_TOUCHED_FILES_SAMPLE = 50",
        "IGNORED_EXACT_PATHS",
        "IGNORED_DIR_PREFIXES",
        "IGNORED_FILE_PATTERNS",
        "audit_runtime_only",
        "ignored_only",
        "transcript_path",
        "infer_scope_for_payload",
        "transcript_parse_failed",
        "multi_business_stage_session_touched_files",
        "session_touched_files_count",
        "session_touched_files_sample",
        "ignored_files_count",
        "effective_scope_files_count",
        "effective_scope_files_sample",
        "ignored_files_sample",
        "workspace_dirty_detected",
        "workspace_changed_files_count",
        "workspace_git_status_used_for_scope",
        "selected_scope",
        "scope_candidates",
        "multi_stage_detected",
        'json.dumps({"continue": True}',
        "sys.stdout.write",
    )
    missing = [term for term in required if term not in text]
    if missing:
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.py",
                1,
                "missing: " + ", ".join(missing),
                "The hook must infer scope, invoke odcr_post_edit_check.py, and write full logs under AI_analysis.",
            )
        )

    wrapper_required = (
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "BASH_SOURCE",
        "EXPECTED_REPO_ROOT",
        "../..",
        "CONDA_PREFIX",
        "command -v python3",
        "command -v python",
        D4C_PYTHON_ABS,
        "VERSION_CHECK",
        "(v[0], v[1]) >= (3, 8)",
        "python_discovery",
        "selected none",
        "ODCR_HOOK_SELECTED_PYTHON",
        HOOK_DIAGNOSTICS_REL,
        "exec \"$PYTHON_BIN\" \"$REPO_ROOT/.codex/hooks/odcr_post_edit_stop.py\"",
    )
    wrapper_missing = [term for term in wrapper_required if term not in wrapper_text]
    if wrapper_missing:
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.sh",
                1,
                "missing: " + ", ".join(wrapper_missing),
                "The wrapper must locate the repo root, discover Python, and finally exec the Python Stop hook.",
            )
        )

    ordering_terms = (
        D4C_PYTHON_ABS,
        "$CONDA_PREFIX/bin/python",
        "command -v python3",
        "command -v python >/dev/null",
    )
    ordering_positions = [wrapper_text.find(term) for term in ordering_terms]
    if any(pos < 0 for pos in ordering_positions) or ordering_positions != sorted(ordering_positions):
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.sh",
                1,
                "Python candidate order is not D4C -> CONDA_PREFIX -> python3 -> python",
                "Keep Python discovery deterministic and check every candidate for Python >= 3.8.",
            )
        )

    if "print(" in text:
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.py",
                1,
                "print(",
                "Stop hook stdout must be JSON-only; write human-readable messages to stderr or files.",
            )
        )

    if '"changed_files":' in text or "'changed_files':" in text:
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.py",
                1,
                "full changed_files diagnostics key",
                "Hook diagnostics must record only counts and bounded samples, never full changed file lists.",
            )
        )

    if not all(term in text for term in ("AI_analysis", "01_raw_logs", "codex_hooks", "runtime_last.json")):
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.py",
                1,
                "missing runtime diagnostics path",
                "Runtime diagnostics must be written under AI_analysis/01_raw_logs/codex_hooks.",
            )
        )

    try:
        hook_module = _load_stop_hook_module(repo_root)
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "transcript.jsonl"
            _write_guardrail_transcript(transcript, ["code/odcr_core/step5_innovation.py"])
            transcript_case = hook_module.infer_scope_for_payload(
                {"transcript_path": str(transcript)},
                repo_root=repo_root,
                cwd=repo_root,
                workspace_changed_files_func=lambda _root: ["code/executors/step4_engine.py"],
            )
            if transcript_case.selected_scope != "step5" or transcript_case.inference_source != "transcript":
                findings.append(
                    Finding(
                        ".codex/hooks/odcr_post_edit_stop.py",
                        1,
                        f"transcript priority returned {transcript_case}",
                        "Stop hook scope inference must use transcript_path session files without consulting workspace status.",
                    )
                )

            multi = Path(tmp) / "multi.jsonl"
            _write_guardrail_transcript(
                multi,
                ["code/executors/step4_engine.py", "code/executors/step5_engine.py"],
            )
            multi_case = hook_module.infer_scope_for_payload(
                {"transcript_path": str(multi)},
                repo_root=repo_root,
                cwd=repo_root,
                workspace_changed_files_func=lambda _root: [],
            )
            if multi_case.selected_scope != "all" or multi_case.inference_reason != "multi_business_stage_session_touched_files":
                findings.append(
                    Finding(
                        ".codex/hooks/odcr_post_edit_stop.py",
                        1,
                        f"multi-stage returned {multi_case}",
                        "The hook may select all only when multiple business stages are explicitly touched.",
                    )
                )

            bad = Path(tmp) / "bad.jsonl"
            bad.write_text("{bad-json", encoding="utf-8")
            bad_case = hook_module.infer_scope_for_payload(
                {"transcript_path": str(bad)},
                repo_root=repo_root,
                cwd=repo_root,
                workspace_changed_files_func=lambda _root: ["code/executors/step5_engine.py"],
            )
            if bad_case.selected_scope != "skip" or bad_case.skip_reason != "transcript_parse_failed":
                findings.append(
                    Finding(
                        ".codex/hooks/odcr_post_edit_stop.py",
                        1,
                        f"parse-failed returned {bad_case}",
                        "Parse-failed transcript inference must skip and must not fall back to git status.",
                    )
                )

        dirty_case = hook_module.infer_scope_for_payload(
            {},
            repo_root=repo_root,
            cwd=repo_root,
            workspace_changed_files_func=lambda _root: [f"code/file_{idx}.py" for idx in range(201)],
        )
        if dirty_case.selected_scope != "skip" or dirty_case.skip_reason != "no_session_touched_files":
            findings.append(
                Finding(
                    ".codex/hooks/odcr_post_edit_stop.py",
                    1,
                    f"dirty workspace returned {dirty_case}",
                    "Dirty workspace alone must select skip and must not trigger validation.",
                )
            )

        summary = hook_module._inference_summary(
            hook_module.ScopeInference(
                selected_scope="governance-fast",
                inference_source="payload",
                inference_reason="payload_session_touched_files",
                session_touched_files=tuple(f"docs/file_{idx}.md" for idx in range(300)),
                effective_scope_files=tuple(f"docs/file_{idx}.md" for idx in range(300)),
                scope_candidates=("governance-fast",),
            )
        )
        if (
            len(summary.get("session_touched_files_sample", [])) > 50
            or "touched_files" in summary
            or "raw_touched_files_count" in summary
            or "effective_touched_files_count" in summary
            or "git_changed_files_count" in summary
            or "git_status_truncated" in summary
            or "changed_files_total" in summary
            or "changed_files_sample" in summary
        ):
            findings.append(
                Finding(
                    ".codex/hooks/odcr_post_edit_stop.py",
                    1,
                    f"diagnostics summary keys={sorted(summary)}",
                    "Runtime diagnostics must keep touched files bounded and avoid legacy changed_files_* keys.",
                )
            )
    except Exception as exc:
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.py",
                1,
                repr(exc),
                "Keep Stop hook scope inference import-safe and dynamically testable by the guardrail.",
            )
        )

    forbidden_patterns = (
        r"(?:\./odcr|code/odcr\.py)\s+preprocess\b",
        r"(?:\./odcr|code/odcr\.py)\s+(?:eval|rerank)\b",
        r"(?:\./odcr|code/odcr\.py)\s+(?:step3|step4|step5)\b(?![^\n]*--dry-run)",
        r"\bpython(?:3)?\s+code/(?:preprocess|train|eval|rerank|executors/step[345])",
    )
    for rel, scan_text in (
        (".codex/hooks/odcr_post_edit_stop.sh", wrapper_text),
        (".codex/hooks/odcr_post_edit_stop.py", text),
    ):
        for pattern in forbidden_patterns:
            match = re.search(pattern, scan_text)
            if match:
                line = scan_text[: match.start()].count("\n") + 1
                findings.append(
                    Finding(
                        rel,
                        line,
                        match.group(0),
                        "The Codex Stop hook launcher must delegate only to odcr_post_edit_check.py; real stage commands require explicit user authorization.",
                    )
                )

    if findings:
        result.fail("Codex Stop hook launcher is missing delegation/safety requirements.", findings)
    else:
        result.summary = (
            "Codex Stop hook wrapper rejects Python 2, prioritizes D4C Python, delegates to the Python hook, "
            "validates current-session transcript/payload touched files only, skips dirty/unknown/no-session cases, "
            "keeps stdout JSON-only, writes bounded AI_analysis runtime diagnostics, and contains no real stage commands."
        )
    return result


def _check_codex_hooks_or_manual_primary_docs(repo_root: Path) -> RuleResult:
    result = RuleResult("R056", "Codex Hooks or manual post-edit check must be primary")
    docs = (
        "AGENTS.md",
        "README.md",
        "docs/CODEX_CHANGE_REQUEST_TEMPLATE.md",
        "docs/ODCR_EVOLUTION_PROTOCOL.md",
    )
    required_terms = (
        "Codex Hooks",
        "Stop hook",
        ".codex",
        "does not require git commit",
        "git hook",
        "CI",
        "optional",
        "manual",
        "python code/tools/odcr_post_edit_check.py --scope <scope>",
        "Real training",
        "explicit",
    )
    findings: list[Finding] = []
    for rel in docs:
        path = repo_root / rel
        if not path.is_file():
            findings.append(Finding(rel, 1, "missing", "Keep workflow docs present."))
            continue
        text = _read(path)
        normalized = " ".join(text.split())
        missing = [term for term in required_terms if term not in text and term not in normalized]
        if missing:
            findings.append(
                Finding(
                    rel,
                    1,
                    "missing: " + ", ".join(missing),
                    "Document Codex Hooks Stop hook as the primary single-user gate, with manual fallback and optional git hook/CI.",
                )
            )
    if findings:
        result.fail(
            "Workflow docs do not fully describe Codex Hooks/manual post-edit validation as the primary path.",
            findings,
        )
    else:
        result.summary = "Workflow docs state Codex Hooks Stop hook or manual post-edit check is primary; git hook/CI are optional."
    return result


def _check_stop_hook_ignored_path_rules(repo_root: Path) -> RuleResult:
    result = RuleResult("R063", "Stop hook must ignore runtime and audit artifacts")
    findings: list[Finding] = []
    try:
        hook_module = _load_stop_hook_module(repo_root)
    except Exception as exc:
        result.fail(
            "Could not import Stop hook for ignored-path inspection.",
            [Finding(".codex/hooks/odcr_post_edit_stop.py", 1, repr(exc), "Keep the hook import-safe.")],
        )
        return result

    ignored_examples = (
        "audit.log",
        "./audit.log",
        "/public/home/zhangliml/lc/ODCR/ODCR-main/audit.log",
        "AI_analysis/03_evidence_ledgers/ledger.md",
        "AI_analysis/history/old_report.md",
        "AI_analysis/01_raw_logs/codex_hooks/runtime_last.json",
        "runs/task4/meta/run_summary.json",
        "cache/foo.bin",
        "data/raw.csv",
        "merged/task4.csv",
        "artifacts/models/model.bin",
        "code/__pycache__/x.pyc",
        ".pytest_cache/v/cache/nodeids",
        ".mypy_cache/meta.json",
        ".ruff_cache/0.1/cache",
        "tmp/session.log",
        "tmp/session.pyc",
    )
    missed = [path for path in ignored_examples if not hook_module._is_ignored_path(path)]
    if missed:
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.py",
                1,
                "not ignored: " + ", ".join(missed),
                "Ignore audit.log, AI_analysis, runs/cache/data/merged/artifacts, cache dirs, *.log, and *.pyc before scope inference.",
            )
        )
    text = _read(repo_root / ".codex" / "hooks" / "odcr_post_edit_stop.py")
    required_terms = (
        "IGNORED_EXACT_PATHS",
        "audit.log",
        "IGNORED_DIR_PREFIXES",
        "AI_analysis/",
        "runs/",
        "cache/",
        "data/",
        "merged/",
        "artifacts/",
        "__pycache__",
        ".pytest_cache",
        "IGNORED_FILE_PATTERNS",
        "*.log",
        "*.pyc",
    )
    missing = [term for term in required_terms if term not in text]
    if missing:
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.py",
                1,
                "missing: " + ", ".join(missing),
                "Keep ignored-path rules visible and centralized in the Stop hook.",
            )
        )
    if findings:
        result.fail("Stop hook ignored-path rules are incomplete.", findings)
    else:
        result.summary = "Stop hook filters audit/runtime/data/cache artifacts before scope inference."
    return result


def _check_stop_hook_ignored_only_noop(repo_root: Path) -> RuleResult:
    result = RuleResult("R064", "Stop hook must no-op when only ignored files changed")
    findings: list[Finding] = []
    try:
        hook_module = _load_stop_hook_module(repo_root)
        inference = hook_module.infer_scope_for_payload(
            {"touched_files": ["audit.log", "AI_analysis/01_raw_logs/codex_hooks/runtime_last.json"]},
            repo_root=repo_root,
            cwd=repo_root,
            workspace_changed_files_func=lambda _root: ["code/executors/step5_engine.py"],
        )
        payload = hook_module._runtime_payload(
            repo_root=repo_root,
            cwd=repo_root,
            hook_event_name="Stop",
            command=None,
            returncode=0,
            failure_stage=None,
            stdout_path=repo_root / "AI_analysis/01_raw_logs/codex_hooks/stdout.log",
            stderr_path=repo_root / "AI_analysis/01_raw_logs/codex_hooks/stderr.log",
            inference=inference,
            max_seconds=120,
            child_timeout_seconds=120,
            wrapper_timeout_seconds=180,
            started_at="2026-05-02T00:00:00+00:00",
            finished_at="2026-05-02T00:00:01+00:00",
        )
    except Exception as exc:
        result.fail(
            "Could not exercise ignored-only Stop hook inference.",
            [Finding(".codex/hooks/odcr_post_edit_stop.py", 1, repr(exc), "Keep ignored-only inference import-safe.")],
        )
        return result
    if (
        inference.selected_scope != "skip"
        or not inference.skipped
        or inference.skip_reason not in {"audit_runtime_only", "ignored_only"}
        or inference.inference_reason not in {"audit_runtime_only", "ignored_only"}
    ):
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.py",
                1,
                repr(inference),
                "Only ignored files changed must select skip, mark skipped=true, and avoid post-edit check execution.",
            )
        )
    summary = hook_module._inference_summary(inference)
    if summary.get("effective_scope_files_count") != 0 or summary.get("ignored_files_count", 0) < 2:
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.py",
                1,
                json.dumps(summary, ensure_ascii=False, sort_keys=True),
                "Ignored-only diagnostics must record raw/ignored/effective counts.",
            )
        )
    if payload.get("post_edit_command") is not None:
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.py",
                1,
                json.dumps(payload, ensure_ascii=False, sort_keys=True)[:240],
                "Ignored-only Stop hook runtime must keep post_edit_command=null and skip odcr_post_edit_check.py.",
            )
        )
    if findings:
        result.fail("Ignored-only edits do not take the required Stop hook no-op path.", findings)
    else:
        result.summary = "Ignored-only changes select skip and record no-op diagnostics."
    return result


def _check_post_edit_governance_fast_scope(repo_root: Path) -> RuleResult:
    result = RuleResult("R065", "post-edit check must support governance-fast")
    findings: list[Finding] = []
    try:
        module = _load_post_edit_module(repo_root)
        commands = module.build_plan("governance-fast", repo_root=repo_root, python_executable=sys.executable)
    except Exception as exc:
        result.fail(
            "Could not build governance-fast post-edit plan.",
            [Finding("code/tools/odcr_post_edit_check.py", 1, repr(exc), "Expose governance-fast in SCOPES and build_plan.")],
        )
        return result
    displays = [command.display() for command in commands]
    joined = "\n".join(displays)
    if "governance-fast" not in getattr(module, "SCOPES", ()):
        findings.append(
            Finding("code/tools/odcr_post_edit_check.py", 1, "SCOPES missing governance-fast", "Add governance-fast to SCOPES.")
        )
    if "python code/tools/check_one_control_guardrails.py --strict" not in joined:
        findings.append(
            Finding("code/tools/odcr_post_edit_check.py", 1, joined, "governance-fast must run strict guardrail.")
        )
    forbidden = ("compileall", "./odcr doctor", "./odcr show", "./odcr step3", "./odcr step4", "./odcr step5", "./odcr eval", "./odcr rerank")
    bad = [term for term in forbidden if term in joined]
    if bad:
        findings.append(
            Finding(
                "code/tools/odcr_post_edit_check.py",
                1,
                "forbidden: " + ", ".join(bad),
                "governance-fast must avoid compileall, doctor, stage dry-runs, tests, and real runs.",
            )
        )
    if findings:
        result.fail("governance-fast scope is missing or too heavy.", findings)
    else:
        result.summary = "governance-fast runs py_compile for governance tools plus strict guardrail only."
    return result


def _check_stop_hook_uncertain_cases_skip(repo_root: Path) -> RuleResult:
    result = RuleResult("R066", "Stop hook uncertain cases must skip")
    findings: list[Finding] = []
    try:
        hook_module = _load_stop_hook_module(repo_root)
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.jsonl"
            bad.write_text("{bad-json", encoding="utf-8")
            cases = {
                "transcript_parse_failed": hook_module.infer_scope_for_payload(
                    {"transcript_path": str(bad)},
                    repo_root=repo_root,
                    cwd=repo_root,
                    workspace_changed_files_func=lambda _root: ["code/executors/step5_engine.py"],
                ),
                "dirty_workspace_only": hook_module.infer_scope_for_payload(
                    {},
                    repo_root=repo_root,
                    cwd=repo_root,
                    workspace_changed_files_func=lambda _root: [f"code/file_{idx}.py" for idx in range(201)],
                ),
                "unknown_session_touched_files": hook_module.infer_scope_for_payload(
                    {"touched_files": ["misc/unknown.note"]},
                    repo_root=repo_root,
                    cwd=repo_root,
                    workspace_changed_files_func=lambda _root: [],
                ),
                "no_session_touched_files": hook_module.infer_scope_for_payload(
                    {},
                    repo_root=repo_root,
                    cwd=repo_root,
                    workspace_changed_files_func=lambda _root: [],
                ),
            }
    except Exception as exc:
        result.fail(
            "Could not exercise uncertain Stop hook inference cases.",
            [Finding(".codex/hooks/odcr_post_edit_stop.py", 1, repr(exc), "Keep inference import-safe.")],
        )
        return result
    for name, inference in cases.items():
        if inference.selected_scope != "skip" or not inference.skipped or not inference.skip_reason:
            findings.append(
                Finding(
                    ".codex/hooks/odcr_post_edit_stop.py",
                    1,
                    f"{name}: {inference}",
                    "Parse-failed, dirty-only, unknown, and empty session cases must skip validation.",
                )
            )
    if findings:
        result.fail("Stop hook uncertain fallback cases do not all skip.", findings)
    else:
        result.summary = "Parse-failed, dirty-only, unknown, and empty session cases select skip."
    return result


def _check_stop_hook_auto_timeout_fast(repo_root: Path) -> RuleResult:
    result = RuleResult("R067", "Stop hook automatic timeout must be fast")
    findings: list[Finding] = []
    try:
        hook_module = _load_stop_hook_module(repo_root)
        post_module = _load_post_edit_module(repo_root)
    except Exception as exc:
        result.fail(
            "Could not inspect Stop hook timeout settings.",
            [Finding(".codex/hooks/odcr_post_edit_stop.py", 1, repr(exc), "Keep timeout constants import-safe.")],
        )
        return result
    child_timeout = getattr(hook_module, "DEFAULT_HOOK_CHILD_MAX_SECONDS", None)
    wrapper_timeout = getattr(hook_module, "DEFAULT_WRAPPER_TIMEOUT_SECONDS", None)
    manual_timeout = getattr(post_module, "DEFAULT_MANUAL_MAX_SECONDS", None)
    if not isinstance(wrapper_timeout, int) or wrapper_timeout != 180:
        findings.append(
            Finding(".codex/hooks/odcr_post_edit_stop.py", 1, f"DEFAULT_WRAPPER_TIMEOUT_SECONDS={wrapper_timeout!r}", "Codex UI wrapper timeout must remain recorded as 180 seconds.")
        )
    if (
        not isinstance(child_timeout, int)
        or not isinstance(wrapper_timeout, int)
        or child_timeout >= wrapper_timeout
        or child_timeout > 150
    ):
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.py",
                1,
                f"DEFAULT_HOOK_CHILD_MAX_SECONDS={child_timeout!r}, wrapper={wrapper_timeout!r}",
                "Automatic child post-edit timeout must be shorter than the 180-second wrapper timeout.",
            )
        )
    if not isinstance(manual_timeout, int) or manual_timeout < 900:
        findings.append(
            Finding("code/tools/odcr_post_edit_check.py", 1, f"DEFAULT_MANUAL_MAX_SECONDS={manual_timeout!r}", "Manual deep-check timeout should remain 900 seconds.")
        )
    hooks = json.loads(_read(repo_root / ".codex" / "hooks.json"))
    timeouts = [
        hook.get("timeout")
        for entry in hooks.get("hooks", {}).get("Stop", [])
        for hook in entry.get("hooks", [])
        if isinstance(hook, dict)
    ]
    if not timeouts or any(not isinstance(timeout, int) or timeout != 180 for timeout in timeouts):
        findings.append(
            Finding(".codex/hooks.json", 1, f"timeouts={timeouts!r}", "Codex Stop hook config timeout must remain 180 seconds.")
        )
    hook_text = _read(repo_root / ".codex" / "hooks" / "odcr_post_edit_stop.py")
    if "ODCR_HOOK_MAX_SECONDS" not in hook_text or "ODCR_HOOK_CHILD_MAX_SECONDS" not in hook_text:
        findings.append(
            Finding(".codex/hooks/odcr_post_edit_stop.py", 1, "hook child timeout env missing", "Allow explicit hook child timeout override but clamp it below the wrapper timeout.")
        )
    try:
        inferred_all = hook_module.infer_scope_for_payload(
            {"touched_files": ["code/executors/step4_engine.py", "code/executors/step5_engine.py"]},
            repo_root=repo_root,
            cwd=repo_root,
            workspace_changed_files_func=lambda _root: [],
        )
        automatic = hook_module.apply_automatic_stop_scope_policy(inferred_all)
        command = hook_module._build_post_edit_command(
            post_edit_path=repo_root / "code" / "tools" / "odcr_post_edit_check.py",
            scope=automatic.selected_scope,
            max_seconds=hook_module._child_timeout_seconds(hook_module._wrapper_timeout_seconds()),
            dry_run=False,
        )
    except Exception as exc:
        findings.append(
            Finding(".codex/hooks/odcr_post_edit_stop.py", 1, repr(exc), "Auto-all degradation must be import-safe.")
        )
    else:
        if inferred_all.selected_scope != "all":
            findings.append(
                Finding(
                    ".codex/hooks/odcr_post_edit_stop.py",
                    1,
                    repr(inferred_all),
                    "Synthetic multi-stage inference should first identify all before automatic policy degradation.",
                )
            )
        if (
            automatic.selected_scope == "all"
            or automatic.original_inferred_scope != "all"
            or not automatic.manual_followup_required
            or "odcr_post_edit_check.py --scope all --max-seconds 900" not in (automatic.manual_followup_command or "")
        ):
            findings.append(
                Finding(
                    ".codex/hooks/odcr_post_edit_stop.py",
                    1,
                    repr(automatic),
                    "Automatic Stop hook must degrade all-scope inference and record the manual all follow-up command.",
                )
            )
        if hook_module._command_scope(command) == "all":
            findings.append(
                Finding(
                    ".codex/hooks/odcr_post_edit_stop.py",
                    1,
                    " ".join(command),
                    "Automatic Stop hook must not execute odcr_post_edit_check.py with --scope all.",
                )
            )
    if findings:
        result.fail("Automatic Stop hook timeout is not bounded below manual deep-check timeout.", findings)
    else:
        result.summary = "Stop hook wrapper stays 180s, child validation is shorter, auto all is degraded, and manual all remains 900s."
    return result


def _check_logging_outputs_declare_artifact_role(repo_root: Path) -> RuleResult:
    result = RuleResult("R068", "new log/report/metrics/cache outputs must declare artifact role")
    checks = {
        "docs/ODCR_FEATURE_INTEGRATION_CHECKLIST.md": (
            "console_output_changed",
            "file_log_added",
            "metrics_file_added",
            "cache_file_added",
            "report_file_added",
            "artifact_role",
            "output_directory",
            "producer",
            "consumer",
            "retention_policy",
            "verbose_or_default",
            "post_edit_logging_scope",
        ),
        "docs/CODEX_CHANGE_REQUEST_TEMPLATE.md": (
            "Logging / Artifact Output Impact",
            "Output role",
            "Directory rationale",
            "Duplicate or replacement",
            "Default visibility",
            "Guardrail/test needed",
        ),
        "docs/ODCR_EVOLUTION_PROTOCOL.md": (
            "Logging And Artifact Evolution",
            "Artifact role",
            "Output directory",
            "Producer and consumer",
            "Retention policy",
            "post-edit scope",
        ),
        "code/odcr_core/path_layout.py": (
            "ArtifactRoleSpec",
            "artifact_role_registry",
            "default_directory",
            "filename_convention",
            "producer",
            "consumer",
            "retention_note",
            "ai_analysis_may_copy",
        ),
    }
    findings: list[Finding] = []
    for rel, terms in checks.items():
        text = _read(repo_root / rel) if (repo_root / rel).is_file() else ""
        missing = [term for term in terms if term not in text]
        if missing:
            findings.append(
                Finding(
                    rel,
                    1,
                    "missing: " + ", ".join(missing),
                    "Future output files must declare role, directory, producer, consumer, retention, visibility, and AI_analysis copy policy.",
                )
            )
    bad = scan_logging_artifact_snippet(
        "R068",
        'open("runs/step3/task4/1/meta/new_report.json", "w")',
        path="code/odcr_core/new_logging.py",
    )
    if not bad:
        findings.append(
            Finding(
                "code/tools/check_one_control_guardrails.py",
                1,
                "R068 synthetic writer not detected",
                "R068 must flag log/report/metrics/cache output writers that lack artifact role declarations.",
            )
        )
    if findings:
        result.fail("Logging artifact role governance is incomplete.", findings)
    else:
        result.summary = "Future log/report/metrics/cache outputs must declare artifact role, owner, retention, visibility, and tests."
    return result


def _check_run_facing_outputs_update_summary_latest(repo_root: Path) -> RuleResult:
    result = RuleResult("R069", "run-facing outputs must update run_summary/latest decisions")
    checks = {
        "docs/ODCR_FEATURE_INTEGRATION_CHECKLIST.md": (
            "run_summary_updated",
            "latest_pointer_updated",
        ),
        "docs/CODEX_CHANGE_REQUEST_TEMPLATE.md": (
            "run_summary indexing",
            "latest.json update",
            "meta/run_summary.json",
            "latest.json",
        ),
        "docs/ODCR_EVOLUTION_PROTOCOL.md": (
            "meta/run_summary.json",
            "latest.json",
            "run-facing outputs",
        ),
        "code/odcr_core/manifests.py": (
            "build_run_summary",
            "write_run_summary_json",
            "latest_summary_path",
        ),
        "code/tests/test_run_summary_logging.py": (
            "latest_summary_path",
            "run_summary.json",
        ),
    }
    findings: list[Finding] = []
    for rel, terms in checks.items():
        text = _read(repo_root / rel) if (repo_root / rel).is_file() else ""
        missing = [term for term in terms if term not in text]
        if missing:
            findings.append(
                Finding(
                    rel,
                    1,
                    "missing: " + ", ".join(missing),
                    "Run-facing outputs must be indexed by run_summary/latest or explicitly declare why not.",
                )
            )
    bad = scan_logging_artifact_snippet(
        "R069",
        'meta_dir = Path("runs/step3/task4/1/meta")\n(meta_dir / "debug_report.json").write_text("{}")',
        path="code/odcr_core/new_logging.py",
    )
    if not bad:
        findings.append(
            Finding(
                "code/tools/check_one_control_guardrails.py",
                1,
                "R069 synthetic run-facing writer not detected",
                "R069 must flag run-facing output writers that omit run_summary/latest declaration.",
            )
        )
    if findings:
        result.fail("Run-facing output handoff indexing governance is incomplete.", findings)
    else:
        result.summary = "Run-facing outputs must declare run_summary/latest indexing behavior."
    return result


def _check_ai_analysis_not_training_full_log_mirror_evolution(repo_root: Path) -> RuleResult:
    result = RuleResult("R070", "AI_analysis must not become a full training log mirror")
    checks = {
        "docs/ODCR_EVOLUTION_PROTOCOL.md": (
            "AI_analysis/` must not become a full training log mirror",
            "audit",
            "evidence ledgers",
            "handoff digests",
        ),
        "docs/CODEX_CHANGE_REQUEST_TEMPLATE.md": (
            "AI_analysis/` as a full training log mirror",
        ),
        "code/odcr_core/path_layout.py": (
            "do not mirror full training logs",
            "ai_analysis_may_copy=False",
        ),
    }
    findings: list[Finding] = []
    for rel, terms in checks.items():
        text = _read(repo_root / rel) if (repo_root / rel).is_file() else ""
        missing = [term for term in terms if term not in text]
        if missing:
            findings.append(
                Finding(
                    rel,
                    1,
                    "missing: " + ", ".join(missing),
                    "AI_analysis may store governance evidence, not copied full training logs.",
                )
            )
    bad = scan_logging_artifact_snippet(
        "R070",
        'shutil.copyfile(run_dir / "meta" / "full.log", "AI_analysis/full_train.log")',
        path="code/odcr_core/new_logging.py",
    )
    if not bad:
        findings.append(
            Finding(
                "code/tools/check_one_control_guardrails.py",
                1,
                "R070 synthetic AI_analysis mirror not detected",
                "R070 must flag AI_analysis full training log mirrors.",
            )
        )
    if findings:
        result.fail("AI_analysis full-log mirror prevention is incomplete.", findings)
    else:
        result.summary = "AI_analysis remains evidence/digest/report storage, not a full training log mirror."
    return result


def _check_console_default_no_full_dump_evolution(repo_root: Path) -> RuleResult:
    result = RuleResult("R071", "default console must not dump full config/source/guardrail detail")
    checks = {
        "docs/ODCR_EVOLUTION_PROTOCOL.md": (
            "Default console output must remain summary-level",
            "verbose/debug display",
            "per-rule guardrail PASS",
        ),
        "docs/CODEX_CHANGE_REQUEST_TEMPLATE.md": (
            "full resolved config",
            "full source table",
            "per-rule guardrail",
            "default console output",
        ),
        "code/odcr_core/logging_meta.py": (
            "CONSOLE_LEVEL_SUMMARY",
            "CONSOLE_LEVEL_VERBOSE",
            "CONSOLE_LEVEL_DEBUG",
        ),
        "code/tests/test_logging_console_file.py": (
            "test_default_console_summary_omits_full_source_table",
            "test_verbose_debug_flags_do_not_change_resolved_training_payload",
        ),
    }
    findings: list[Finding] = []
    for rel, terms in checks.items():
        text = _read(repo_root / rel) if (repo_root / rel).is_file() else ""
        missing = [term for term in terms if term not in text]
        if missing:
            findings.append(
                Finding(
                    rel,
                    1,
                    "missing: " + ", ".join(missing),
                    "Default console must be summary-level; detailed dumps belong in files or verbose/debug display.",
                )
            )
    bad = scan_logging_artifact_snippet(
        "R071",
        'print("ODCR One-Control Guardrails: PASS " + "\\n".join(per_rule_pass_lines))',
        path="code/odcr_core/new_logging.py",
    )
    if not bad:
        findings.append(
            Finding(
                "code/tools/check_one_control_guardrails.py",
                1,
                "R071 synthetic console dump not detected",
                "R071 must flag default console full config/source/per-rule PASS dumps.",
            )
        )
    if findings:
        result.fail("Default console summary governance is incomplete.", findings)
    else:
        result.summary = "Default console stays summary-level; full detail belongs in files or verbose/debug."
    return result


def _check_new_log_paths_not_forbidden_destinations(repo_root: Path) -> RuleResult:
    result = RuleResult("R072", "new log paths must not target data/merged/top-level logs/code/log.out")
    checks = {
        "docs/ODCR_EVOLUTION_PROTOCOL.md": (
            "data/",
            "merged/",
            "top-level `logs/`",
            "code/log.out",
        ),
        "docs/CODEX_CHANGE_REQUEST_TEMPLATE.md": (
            "data/` or `merged/`",
            "top-level `logs/`",
            "code/log.out",
        ),
        "code/tests/test_path_layout_boundaries.py": (
            "test_run_log_paths_are_run_meta_not_cache_data_or_merged",
            "test_ai_analysis_is_not_active_full_log_sink",
        ),
    }
    findings: list[Finding] = []
    for rel, terms in checks.items():
        text = _read(repo_root / rel) if (repo_root / rel).is_file() else ""
        missing = [term for term in terms if term not in text]
        if missing:
            findings.append(
                Finding(
                    rel,
                    1,
                    "missing: " + ", ".join(missing),
                    "New log paths must avoid data/, merged/, top-level logs/, and code/log.out.",
                )
            )
    bad_snippets = (
        'open("data/foo.log", "w")',
        'open("merged/foo.log", "w")',
        'open("code/log.out", "a")',
    )
    for snippet in bad_snippets:
        if not scan_logging_artifact_snippet("R072", snippet, path="code/odcr_core/new_logging.py"):
            findings.append(
                Finding(
                    "code/tools/check_one_control_guardrails.py",
                    1,
                    f"R072 synthetic forbidden destination not detected: {snippet}",
                    "R072 must flag data/merged/code/log.out log destinations.",
                )
            )
    if findings:
        result.fail("Forbidden log destination governance is incomplete.", findings)
    else:
        result.summary = "New log destinations are guarded away from data/, merged/, top-level logs/, and code/log.out."
    return result


def _check_stop_hook_unknown_session_files_skip(repo_root: Path) -> RuleResult:
    result = RuleResult("R073", "Unknown session touched files must skip unless overridden")
    findings: list[Finding] = []
    try:
        hook_module = _load_stop_hook_module(repo_root)
        inference = hook_module.infer_scope_for_payload(
            {"touched_files": ["misc/unknown.note"]},
            repo_root=repo_root,
            cwd=repo_root,
            workspace_changed_files_func=lambda _root: [],
        )
        old_value = os.environ.get("ODCR_HOOK_SCOPE")
        os.environ["ODCR_HOOK_SCOPE"] = "all"
        try:
            override = hook_module.infer_scope_for_payload(
                {"touched_files": ["misc/unknown.note"]},
                repo_root=repo_root,
                cwd=repo_root,
                workspace_changed_files_func=lambda _root: [],
            )
        finally:
            if old_value is None:
                os.environ.pop("ODCR_HOOK_SCOPE", None)
            else:
                os.environ["ODCR_HOOK_SCOPE"] = old_value
    except Exception as exc:
        result.fail(
            "Could not exercise unknown-session Stop hook inference.",
            [Finding(".codex/hooks/odcr_post_edit_stop.py", 1, repr(exc), "Keep unknown-session inference import-safe.")],
        )
        return result
    if inference.selected_scope != "skip" or inference.skip_reason != "unknown_session_touched_files":
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.py",
                1,
                repr(inference),
                "Unknown session touched files must select skip by default.",
            )
        )
    if override.selected_scope != "all" or override.override_source != "env":
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.py",
                1,
                repr(override),
                "ODCR_HOOK_SCOPE must be able to explicitly override unknown touched-file skip.",
            )
        )
    if findings:
        result.fail("Unknown session files are not guarded by skip/explicit override semantics.", findings)
    else:
        result.summary = "Unknown session files skip by default and ODCR_HOOK_SCOPE can explicitly override."
    return result


def _check_runtime_diagnostics_schema_v22(repo_root: Path) -> RuleResult:
    result = RuleResult("R074", "Runtime diagnostics must use schema v2.2")
    findings: list[Finding] = []
    try:
        hook_module = _load_stop_hook_module(repo_root)
        inference = hook_module.infer_scope_for_payload(
            {"touched_files": ["docs/ODCR_EVOLUTION_PROTOCOL.md"]},
            repo_root=repo_root,
            cwd=repo_root,
            workspace_changed_files_func=lambda _root: [],
        )
        payload = hook_module._runtime_payload(
            repo_root=repo_root,
            cwd=repo_root,
            hook_event_name="Stop",
            command=hook_module._build_post_edit_command(
                post_edit_path=repo_root / "code" / "tools" / "odcr_post_edit_check.py",
                scope=inference.selected_scope,
                max_seconds=120,
                dry_run=False,
            ),
            returncode=None,
            failure_stage="post_edit_running",
            stdout_path=repo_root / "AI_analysis/01_raw_logs/codex_hooks/stdout.log",
            stderr_path=repo_root / "AI_analysis/01_raw_logs/codex_hooks/stderr.log",
            inference=inference,
            max_seconds=120,
            child_timeout_seconds=120,
            wrapper_timeout_seconds=180,
            started_at="2026-05-02T00:00:00+00:00",
            post_edit_started_at="2026-05-02T00:00:01+00:00",
        )
    except Exception as exc:
        result.fail(
            "Could not build runtime diagnostics payload for schema v2.2.",
            [Finding(".codex/hooks/odcr_post_edit_stop.py", 1, repr(exc), "Keep runtime payload import-safe.")],
        )
        return result
    if payload.get("schema_version") != "odcr_codex_hook_runtime/2.2":
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.py",
                1,
                f"schema_version={payload.get('schema_version')!r}",
                "Runtime diagnostics schema_version must be odcr_codex_hook_runtime/2.2.",
            )
        )
    required_runtime_fields = (
        "selected_scope",
        "inference_reason",
        "inference_source",
        "session_touched_files_count",
        "ignored_files_count",
        "effective_scope_files_count",
        "ignored_files_sample",
        "effective_scope_files_sample",
        "workspace_dirty_detected",
        "workspace_changed_files_count",
        "workspace_git_status_used_for_scope",
        "post_edit_command",
        "stdout_path",
        "stderr_path",
        "started_at",
        "post_edit_started_at",
        "child_timeout_seconds",
        "wrapper_timeout_seconds",
        "failure_stage",
    )
    missing_runtime_fields = [field for field in required_runtime_fields if field not in payload]
    if missing_runtime_fields or payload.get("inference_reason") == "initializing":
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.py",
                1,
                "missing: " + ", ".join(missing_runtime_fields),
                "Runtime diagnostics must be finalized after scope inference and before child execution.",
            )
        )
    hook_text = _read(repo_root / ".codex" / "hooks" / "odcr_post_edit_stop.py")
    prelaunch_pos = hook_text.find('failure_stage="post_edit_running"')
    child_pos = hook_text.find("child_result = _run_post_edit_child(")
    if prelaunch_pos < 0 or child_pos < 0 or prelaunch_pos > child_pos:
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.py",
                1,
                "pre-child runtime write ordering",
                "The hook must write selected_scope/command runtime diagnostics before launching the child process.",
            )
        )
    if findings:
        result.fail("Runtime diagnostics schema_version is not v2.2.", findings)
    else:
        result.summary = "Runtime diagnostics schema v2.2 records finalized scope/command fields before child execution."
    return result


def _check_runtime_diagnostics_workspace_scope_flag(repo_root: Path) -> RuleResult:
    result = RuleResult("R075", "Runtime diagnostics must record git-status-not-used flag")
    findings: list[Finding] = []
    try:
        hook_module = _load_stop_hook_module(repo_root)
        inference = hook_module.infer_scope_for_payload(
            {},
            repo_root=repo_root,
            cwd=repo_root,
            workspace_changed_files_func=lambda _root: ["code/executors/step5_engine.py"],
        )
        payload = hook_module._runtime_payload(
            repo_root=repo_root,
            cwd=repo_root,
            hook_event_name="Stop",
            command=None,
            returncode=0,
            failure_stage=None,
            stdout_path=repo_root / "AI_analysis/01_raw_logs/codex_hooks/stdout.log",
            stderr_path=repo_root / "AI_analysis/01_raw_logs/codex_hooks/stderr.log",
            inference=inference,
            max_seconds=120,
            child_timeout_seconds=120,
            wrapper_timeout_seconds=180,
            started_at="2026-05-02T00:00:00+00:00",
        )
    except Exception as exc:
        result.fail(
            "Could not build runtime diagnostics payload for workspace flag.",
            [Finding(".codex/hooks/odcr_post_edit_stop.py", 1, repr(exc), "Keep runtime payload import-safe.")],
        )
        return result
    if payload.get("workspace_git_status_used_for_scope") is not False:
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.py",
                1,
                json.dumps(payload, ensure_ascii=False, sort_keys=True)[:240],
                "Runtime diagnostics must include workspace_git_status_used_for_scope=false.",
            )
        )
    if payload.get("child_timeout_seconds", 999) >= payload.get("wrapper_timeout_seconds", 0):
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.py",
                1,
                json.dumps(payload, ensure_ascii=False, sort_keys=True)[:240],
                "Runtime diagnostics must prove child_timeout_seconds < wrapper_timeout_seconds.",
            )
        )
    if findings:
        result.fail("Runtime diagnostics do not prove git status was unused for scope.", findings)
    else:
        result.summary = "Runtime diagnostics include workspace_git_status_used_for_scope=false."
    return result


def _check_runtime_diagnostics_no_legacy_fields(repo_root: Path) -> RuleResult:
    result = RuleResult("R076", "Runtime diagnostics must not write legacy touched/git fields")
    findings: list[Finding] = []
    try:
        hook_module = _load_stop_hook_module(repo_root)
        inference = hook_module.infer_scope_for_payload(
            {"touched_files": ["audit.log"]},
            repo_root=repo_root,
            cwd=repo_root,
            workspace_changed_files_func=lambda _root: [],
        )
        payload = hook_module._runtime_payload(
            repo_root=repo_root,
            cwd=repo_root,
            hook_event_name="Stop",
            command=None,
            returncode=0,
            failure_stage=None,
            stdout_path=repo_root / "AI_analysis/01_raw_logs/codex_hooks/stdout.log",
            stderr_path=repo_root / "AI_analysis/01_raw_logs/codex_hooks/stderr.log",
            inference=inference,
            max_seconds=180,
        )
    except Exception as exc:
        result.fail(
            "Could not build runtime diagnostics payload for legacy-field inspection.",
            [Finding(".codex/hooks/odcr_post_edit_stop.py", 1, repr(exc), "Keep runtime payload import-safe.")],
        )
        return result
    forbidden = (
        "raw_touched_files_count",
        "effective_touched_files_count",
        "touched_files_sample",
        "git_changed_files_count",
        "git_status_truncated",
        "changed_files_total",
        "changed_files_sample",
        "changed_files_truncated",
        "workspace_changed_files_sample",
    )
    present = [key for key in forbidden if key in payload]
    if present:
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.py",
                1,
                "legacy fields: " + ", ".join(present),
                "Runtime diagnostics v2.2 must not write legacy touched/git-status/changed-files keys.",
            )
        )
    wrapper_text = _read(repo_root / ".codex" / "hooks" / "odcr_post_edit_stop.sh")
    wrapper_present = [
        key
        for key in forbidden
        if re.search(r'"' + re.escape(key) + r'"\s*:', wrapper_text)
    ]
    if wrapper_present:
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.sh",
                1,
                "legacy launcher fields: " + ", ".join(wrapper_present),
                "Shell launcher fallback diagnostics must be v2.2-shaped and must not write legacy fields.",
            )
        )
    if findings:
        result.fail("Runtime diagnostics still write legacy fields.", findings)
    else:
        result.summary = "Runtime diagnostics omit legacy touched/git-status/changed-files keys."
    return result


def _check_skip_scope_has_null_command(repo_root: Path) -> RuleResult:
    result = RuleResult("R077", "selected_scope=skip must have post_edit_command=null")
    findings: list[Finding] = []
    try:
        hook_module = _load_stop_hook_module(repo_root)
        inference = hook_module.infer_scope_for_payload(
            {},
            repo_root=repo_root,
            cwd=repo_root,
            workspace_changed_files_func=lambda _root: [],
        )
        payload = hook_module._runtime_payload(
            repo_root=repo_root,
            cwd=repo_root,
            hook_event_name="Stop",
            command=None,
            returncode=0,
            failure_stage=None,
            stdout_path=repo_root / "AI_analysis/01_raw_logs/codex_hooks/stdout.log",
            stderr_path=repo_root / "AI_analysis/01_raw_logs/codex_hooks/stderr.log",
            inference=inference,
            max_seconds=180,
        )
    except Exception as exc:
        result.fail(
            "Could not build skip runtime payload.",
            [Finding(".codex/hooks/odcr_post_edit_stop.py", 1, repr(exc), "Keep skip runtime diagnostics import-safe.")],
        )
        return result
    if payload.get("selected_scope") != "skip" or payload.get("post_edit_command") is not None:
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.py",
                1,
                json.dumps(payload, ensure_ascii=False, sort_keys=True)[:240],
                "When selected_scope=skip, post_edit_command must be null.",
            )
        )
    if findings:
        result.fail("Skip runtime payload can still include a post-edit command.", findings)
    else:
        result.summary = "Skip runtime payload writes post_edit_command=null."
    return result


def _check_run_summary_entrypoint(repo_root: Path) -> RuleResult:
    result = RuleResult("R057", "new runs must expose meta/run_summary.json")
    required = {
        "code/odcr_core/manifests.py": (
            "RUN_SUMMARY_FILENAME",
            "run_summary.json",
            "build_run_summary",
            "write_run_summary_for_config",
        ),
        "code/odcr.py": ("write_run_summary_for_config", 'status="running"', 'status="ok"', 'status="failed"'),
        "code/odcr_core/preprocess_runtime.py": ("write_run_summary_json", 'status="running"', 'status="ok"', 'status="failed"'),
    }
    findings: list[Finding] = []
    for rel, terms in required.items():
        path = repo_root / rel
        if not path.is_file():
            findings.append(Finding(rel, 1, "missing", "Run-summary logging requires this file."))
            continue
        text = _read(path)
        missing = [term for term in terms if term not in text]
        if missing:
            findings.append(
                Finding(
                    rel,
                    1,
                    "missing: " + ", ".join(missing),
                    "Every formal run must write and update meta/run_summary.json.",
                )
            )
    if findings:
        result.fail("Run-summary entrypoint coverage is incomplete.", findings)
    else:
        result.summary = "Stage and preprocess runs write meta/run_summary.json through odcr_core.manifests."
    return result


def _check_resolved_config_filename_canonical(repo_root: Path) -> RuleResult:
    result = RuleResult("R058", "new resolved config filename must be meta/resolved_config.json")
    findings: list[Finding] = []
    for path in _active_run_logging_paths(repo_root):
        rel = _rel(path, repo_root)
        text = _read(path)
        findings.extend(_scan_r058_text(rel, text))
    manifests = _read(repo_root / "code" / "odcr_core" / "manifests.py")
    for term in ("RESOLVED_CONFIG_FILENAME", "SOURCE_TABLE_FILENAME", "resolved_config.json", "source_table.json"):
        if term not in manifests:
            findings.append(
                Finding(
                    "code/odcr_core/manifests.py",
                    1,
                    f"missing {term}",
                    "Keep the canonical config/source-table filenames centralized.",
                )
            )
    if findings:
        result.fail("Deprecated config snapshot filename is present in active run-writing code.", findings)
    else:
        result.summary = "Active run-writing code uses meta/resolved_config.json and meta/source_table.json."
    return result


def _check_latest_points_to_run_summary(repo_root: Path) -> RuleResult:
    result = RuleResult("R059", "stage/task/unit latest.json must point to run_summary")
    manifests_path = repo_root / "code" / "odcr_core" / "manifests.py"
    resolver_path = repo_root / "code" / "odcr_core" / "config_resolver.py"
    findings: list[Finding] = []
    manifests = _read(manifests_path) if manifests_path.is_file() else ""
    resolver = _read(resolver_path) if resolver_path.is_file() else ""
    upstream_path = repo_root / "code" / "odcr_core" / "upstream_resolver.py"
    upstream = _read(upstream_path) if upstream_path.is_file() else ""
    required_manifest_terms = (
        "LATEST_FILENAME",
        "latest.json",
        "write_latest_pointer_json",
        "latest_summary_path",
        "latest_run_id",
        "latest_stage_status_path",
        "status_claim_source",
        "write_run_summary_json",
    )
    missing_manifest = [term for term in required_manifest_terms if term not in manifests]
    if missing_manifest:
        findings.append(
            Finding(
                "code/odcr_core/manifests.py",
                1,
                "missing: " + ", ".join(missing_manifest),
                "latest.json must be written from the run_summary helper and include latest_summary_path.",
            )
        )
    required_resolver_terms = ("latest.json", "latest_run_id", "latest_summary_path", "run_summary.json", "get_stage_task_root")
    missing_resolver = [term for term in required_resolver_terms if term not in (resolver + "\n" + upstream)]
    if missing_resolver:
        findings.append(
            Finding(
                "code/odcr_core/config_resolver.py",
                1,
                "missing: " + ", ".join(missing_resolver),
                "Resolving latest must read the stage/task latest.json pointer and validate run_summary.",
            )
        )
    latest_block = ""
    for _line, name, block in _iter_python_function_blocks(resolver):
        if name == "_latest_run":
            latest_block = block
            break
    if not latest_block:
        findings.append(
            Finding(
                "code/odcr_core/config_resolver.py",
                1,
                "_latest_run block not found",
                "Keep latest resolution explicit and latest.json-only.",
            )
        )
    else:
        forbidden = (
            ".iterdir(",
            'repo_root / "runs" / f"task',
            "legacy_parent",
            "candidates =",
        )
        for term in forbidden:
            if term in latest_block:
                findings.append(
                    Finding(
                        "code/odcr_core/config_resolver.py",
                        1,
                        term,
                        "Remove directory-scanning or old-layout fallback from _latest_run.",
                    )
                )
    if findings:
        result.fail("latest.json pointer coverage is incomplete.", findings)
    else:
        result.summary = "latest.json includes latest_summary_path and resolver has no directory fallback."
    return result


def _latest_run_block(resolver_text: str) -> str:
    for _line, name, block in _iter_python_function_blocks(resolver_text):
        if name == "_latest_run":
            return block
    return ""


def _check_latest_lookup_no_scan_or_legacy_layout(repo_root: Path) -> RuleResult:
    result = RuleResult("R090", "latest lookup must not scan directories or old runs/task layout")
    resolver_path = repo_root / "code" / "odcr_core" / "config_resolver.py"
    resolver = _read(resolver_path) if resolver_path.is_file() else ""
    block = _latest_run_block(resolver)
    findings: list[Finding] = []
    if not block:
        findings.append(
            Finding(
                "code/odcr_core/config_resolver.py",
                1,
                "_latest_run block not found",
                "Keep latest resolution explicit and latest.json-only.",
            )
        )
    forbidden = (
        ".iterdir(",
        ".glob(",
        ".rglob(",
        "os.listdir(",
        "os.scandir(",
        'repo_root / "runs" / f"task',
        "runs/task",
        "legacy_parent",
        "candidates =",
        'return "latest"',
    )
    for term in forbidden:
        if term in block:
            findings.append(
                Finding(
                    "code/odcr_core/config_resolver.py",
                    1,
                    term,
                    "Remove directory scans, legacy layout probes, and dry-run latest sentinels from _latest_run.",
                )
            )
    if findings:
        result.fail("_latest_run can still bypass latest.json.", findings)
    else:
        result.summary = "_latest_run is latest.json-only and has no directory scan or old-layout fallback."
    return result


def _check_latest_lookup_requires_summary_hard_gate(repo_root: Path) -> RuleResult:
    result = RuleResult("R091", "_latest_run must require latest.json -> meta/run_summary.json")
    resolver_path = repo_root / "code" / "odcr_core" / "config_resolver.py"
    resolver = _read(resolver_path) if resolver_path.is_file() else ""
    block = _latest_run_block(resolver)
    upstream_path = repo_root / "code" / "odcr_core" / "upstream_resolver.py"
    upstream = _read(upstream_path) if upstream_path.is_file() else ""
    combined = block + "\n" + upstream
    findings: list[Finding] = []
    required = (
        "latest.json",
        "latest_run_id",
        "latest_summary_path",
        "latest_stage_status_path",
        "run_summary.json",
        "expected_summary",
        "missing run_summary.json",
        "resolve_latest",
        "stage_status.json",
        "validate_upstream_eligibility",
        "OneControlConfigError",
    )
    for term in required:
        if term not in combined:
            findings.append(
                Finding(
                    "code/odcr_core/config_resolver.py",
                    1,
                    f"missing: {term}",
                    "_latest_run must fail-fast on missing/damaged latest.json and run_summary.json pointers.",
                )
            )
    if findings:
        result.fail("_latest_run hard gate is incomplete.", findings)
    else:
        result.summary = "_latest_run requires latest.json -> meta/run_summary.json and validates the pointed summary."
    return result


def _check_step3_tokenizer_cache_manifest_hard_gate(repo_root: Path) -> RuleResult:
    result = RuleResult("R098", "Step3 tokenizer cache must use v2 tokenizer-only compatibility hash split")
    path = repo_root / "code" / "executors" / "step3_train_core.py"
    text = _read(path) if path.is_file() else ""
    findings: list[Finding] = []
    required = (
        "STEP3_TOKENIZE_CACHE_SCHEMA_VERSION",
        "STEP3_TOKENIZE_CACHE_MANIFEST",
        "STEP3_TOKENIZE_CACHE_COMPLETED_MARKER",
        "STEP3_TOKENIZE_CACHE_FAILED_MARKER",
        "STEP3_TOKENIZE_CACHE_ROLE",
        "manifest_schema_version",
        "cache_role",
        "tokenizer_cache_compat_hash",
        "tokenization_compat_hash",
        "run_lineage_hash",
        "data_contract_hash",
        "preprocessing_artifact_hash",
        "full_run_config_hash",
        "train_runtime_config_hash",
        "optimizer_config_hash",
        "performance_profile_hash",
        "source_table_hash",
        "step3_tokenizer_config",
        "source_csv_fingerprints",
        "preprocess_latest_run_ids",
        "preprocess_manifest_fingerprints",
        "preprocess_metrics_verify_fingerprints",
        "profile_artifact_fingerprints",
        "domain_artifact_fingerprints",
        "schema_contract_hash",
        "cache_content_hash",
        "dataset_dict_fingerprint",
        "retired_v1_schema_rebuild_required",
        "compatibility_key",
        "_step3_tokenize_cache_manifest_matches",
        "_write_step3_tokenize_cache_manifest",
        "_step3_tokenize_cache_manifest_decision",
        "ensure_step3_tokenizer_cache_ready_pre_ddp",
        "build_or_reuse_step3_tokenizer_cache_atomic",
        "wait_for_completed_cache_manifest_file_polling",
        "validate_completed_step3_tokenizer_cache",
        "load_completed_step3_tokenizer_cache_for_rank",
        "init_step3_ddp_after_cache_ready",
        "step3_tokenizer_cache_entry_dir",
        "formal_cache_namespace",
        "completed",
        "_fingerprint_step3_hf_cache_content",
        "cache_content_fingerprint_mismatch",
    )
    for term in required:
        if term not in text:
            findings.append(
                Finding(
                    "code/executors/step3_train_core.py",
                    1,
                    f"missing: {term}",
                    "Step3 tokenizer cache reuse must be gated by cache_manifest.json v2 tokenizer/data/preprocess compatibility hashes.",
                )
            )
    gate_block = ""
    for _line, name, block in _iter_python_function_blocks(text):
        if name == "_step3_tokenize_cache_manifest_gate_fields":
            gate_block = block
            break
    for forbidden in (
        "full_run_config_hash",
        "resolved_config",
        "source_table_hash",
        "train_runtime_config_hash",
        "optimizer_config_hash",
        "performance_profile_hash",
        "profile_artifact_fingerprints",
        "domain_artifact_fingerprints",
        "task_profile_id",
    ):
        if forbidden in gate_block:
            findings.append(
                Finding(
                    "code/executors/step3_train_core.py",
                    1,
                    forbidden,
                    "Tokenizer cache hard gate must exclude record-only run/profile/training lineage fields.",
                )
            )
    if not (repo_root / "code" / "tools" / "odcr_step3_cache_check.py").is_file():
        findings.append(
            Finding(
                "code/tools/odcr_step3_cache_check.py",
                1,
                "missing cache-check",
                "Step3 cache-check must exist as a read-only preflight.",
            )
        )
    gate_start = text.find("def _step3_tokenize_cache_manifest_gate_fields")
    gate_end = text.find("def _step3_tokenize_cache_manifest_sections", gate_start)
    gate_text = text[gate_start:gate_end] if gate_start >= 0 and gate_end > gate_start else ""
    for forbidden in ("one_control_resolved_config_hash", '"resolved_config"', "env_embed_dim"):
        if forbidden in gate_text:
            findings.append(
                Finding(
                    "code/executors/step3_train_core.py",
                    1,
                    forbidden,
                    "Tokenizer cache v2 compatibility gate must not include full resolved config/source_table/runtime-only fields.",
                )
            )
    if "load_from_disk(cache_dir)" in text and "_step3_tokenize_cache_manifest_matches" not in text:
        findings.append(
            Finding(
                "code/executors/step3_train_core.py",
                1,
                "load_from_disk(cache_dir)",
                "Step3 must only load tokenizer cache after _step3_tokenize_cache_manifest_matches succeeds.",
            )
        )
    cache_start = text.find("def build_or_reuse_step3_tokenizer_cache_atomic")
    cache_end = text.find("def _load_step3_artefacts", cache_start)
    cache_block = text[cache_start:cache_end] if cache_start >= 0 and cache_end > cache_start else ""
    for forbidden in ("dist.barrier", "dist.all_reduce", "broadcast_object_list", "save_to_disk(cache_dir)"):
        if forbidden in cache_block:
            findings.append(
                Finding(
                    "code/executors/step3_train_core.py",
                    1,
                    forbidden,
                    "Step3 tokenizer/cache readiness must use pre-DDP atomic manifest/file polling, not distributed collectives or direct final writes.",
                )
            )
    for forbidden in ("get_hf_cache_root(task_idx)", "cache/task{"):
        if forbidden in text:
            findings.append(
                Finding(
                    "code/executors/step3_train_core.py",
                    1,
                    forbidden,
                    "Step3 formal tokenizer cache path must come from One-Control path_layout namespace.",
                )
            )
    if "os.path.getmtime(" in text:
        findings.append(
            Finding(
                "code/executors/step3_train_core.py",
                1,
                "os.path.getmtime(",
                "Step3 tokenizer cache identity must not use path/mtime-only reuse.",
            )
        )
    cache_test = repo_root / "code" / "tests" / "test_step3_cache_path_layout_one_control.py"
    if not cache_test.is_file():
        findings.append(
            Finding(
                "code/tests/test_step3_cache_path_layout_one_control.py",
                1,
                "missing",
                "Independent Step3 cache path_layout One-Control regression test must exist.",
            )
        )
    else:
        test_text = _read(cache_test)
        for term in (
            "step3_tokenizer_cache_entry_dir",
            "step3_validation_tokenizer_cache_entry_dir",
            "cache/step3/tokenizer",
            "cache/task2/hf",
            "get_hf_cache_root(task_idx)",
            "validation namespace",
        ):
            if term not in test_text:
                findings.append(
                    Finding(
                        "code/tests/test_step3_cache_path_layout_one_control.py",
                        1,
                        "missing: " + term,
                        "Dedicated cache path layout test must cover formal/validation separation and old path rejection.",
                    )
                )
    if findings:
        result.fail("Step3 tokenizer cache v2 hard gate is incomplete.", findings)
    else:
        result.summary = "Step3 tokenizer cache v2 gates on tokenizer/data/preprocess compatibility and records full config separately."
    return result


def _check_step4_encoded_cache_manifest_hard_gate(repo_root: Path) -> RuleResult:
    result = RuleResult("R092", "Step4 encoded cache must require manifest/schema/config/source/tokenizer/lineage hash gate")
    path = repo_root / "code" / "executors" / "step4_engine.py"
    text = _read(path) if path.is_file() else ""
    findings: list[Finding] = []
    required = (
        "_STEP4_ENCODE_CACHE_SCHEMA_VERSION",
        "_STEP4_ENCODE_CACHE_MANIFEST",
        "cache_schema_version",
        "source_data_path",
        "source_data_sha256",
        "source_data_size",
        "source_data_mtime_ns",
        "tokenizer_path_or_id",
        "tokenizer_fingerprint",
        "max_length",
        "resolved_config_hash",
        "step3_checkpoint_lineage_hash",
        "index_contract_or_required_fields_hash",
        "producer_code_version",
        "created_at",
        "_step4_encoded_cache_manifest_matches",
        "_write_step4_encoded_cache_manifest",
        "fingerprint_mismatch",
    )
    for term in required:
        if term not in text:
            findings.append(
                Finding(
                    "code/executors/step4_engine.py",
                    1,
                    f"missing: {term}",
                    "Step4 encoded cache reuse must be gated by manifest, schema, config, source, tokenizer, and Step3 lineage.",
                )
            )
    if "load_from_disk(cache_dir)" in text and "if cache_valid:" not in text:
        findings.append(
            Finding(
                "code/executors/step4_engine.py",
                1,
                "load_from_disk(cache_dir)",
                "Step4 must only load encoded cache after _step4_encoded_cache_manifest_matches succeeds.",
            )
        )
    if findings:
        result.fail("Step4 encoded cache hard gate is incomplete.", findings)
    else:
        result.summary = "Step4 encoded cache requires cache_manifest.json with schema/config/source/tokenizer/lineage fields."
    return result


def _check_step5_tokenize_cache_manifest_hard_gate(repo_root: Path) -> RuleResult:
    result = RuleResult("R093", "Step5 tokenization cache must require manifest/schema/config/source/tokenizer/lineage hash gate")
    path = repo_root / "code" / "executors" / "step5_engine.py"
    text = _read(path) if path.is_file() else ""
    findings: list[Finding] = []
    required = (
        "STEP5_TOKENIZE_CACHE_SCHEMA_VERSION",
        "STEP5_TOKENIZE_CACHE_MANIFEST",
        "cache_schema_version",
        "source_step4_export_path",
        "source_step4_export_sha256",
        "step4_export_lineage_hash",
        "index_contract_hash",
        "tokenizer_path_or_id",
        "tokenizer_fingerprint",
        "max_length",
        "resolved_step5_config_hash",
        "step5_innovation_config_hash",
        "required_fields_hash",
        "producer_code_version",
        "created_at",
        "_step5_tokenize_cache_manifest_matches",
        "_write_step5_tokenize_cache_manifest",
        "fingerprint_mismatch",
    )
    for term in required:
        if term not in text:
            findings.append(
                Finding(
                    "code/executors/step5_engine.py",
                    1,
                    f"missing: {term}",
                    "Step5 tokenize cache reuse must be gated by manifest, schema, config, Step4 source/export lineage, index contract, and tokenizer.",
                )
            )
    if "load_from_disk(cache_dir)" in text and "_step5_tokenize_cache_manifest_matches" not in text:
        findings.append(
            Finding(
                "code/executors/step5_engine.py",
                1,
                "load_from_disk(cache_dir)",
                "Step5 must only load tokenize cache after _step5_tokenize_cache_manifest_matches succeeds.",
            )
        )
    if findings:
        result.fail("Step5 tokenize cache hard gate is incomplete.", findings)
    else:
        result.summary = "Step5 tokenize cache requires cache_manifest.json with schema/config/source/tokenizer/lineage fields."
    return result


def _check_active_cache_reuse_not_path_mtime_only(repo_root: Path) -> RuleResult:
    result = RuleResult("R094", "dataset_info/load_from_disk alone and path/mtime-only cache keys are forbidden")
    findings: list[Finding] = []
    checks = {
        "code/executors/step3_train_core.py": (
            "_step3_tokenize_cache_manifest_matches",
            "_write_step3_tokenize_cache_manifest",
            "cache_content_hash",
            "source_csv_fingerprints",
            "file_fingerprint(",
            "dataset_dict.json",
        ),
        "code/executors/step4_engine.py": (
            "_step4_encoded_cache_manifest_matches",
            "_write_step4_encoded_cache_manifest",
            "source_data_sha256",
            "file_fingerprint(",
            "dataset_info.json",
        ),
        "code/executors/step5_engine.py": (
            "_step5_tokenize_cache_manifest_matches",
            "_write_step5_tokenize_cache_manifest",
            "source_step4_export_sha256",
            "file_fingerprint(",
            "dataset_dict.json",
        ),
        "code/odcr_core/index_contract.py": (
            "_fingerprint_for_path",
            "file_fingerprint(path)",
            "fingerprint_version",
            "size",
            "mtime_ns",
            "sha256",
            "sample_sha256",
        ),
    }
    for rel, required in checks.items():
        text = _read(repo_root / rel)
        for term in required:
            if term not in text:
                findings.append(
                    Finding(
                        rel,
                        1,
                        f"missing: {term}",
                        "Active Step4/Step5 cache reuse must use content fingerprints and cache manifests.",
                    )
                )
        forbidden = ("os.path.getmtime(", "Path.stat().st_mtime", "mtime_only", "path_mtime")
        for term in forbidden:
            if term in text:
                findings.append(
                    Finding(
                        rel,
                        1,
                        term,
                        "Path/mtime-only cache identity is forbidden for active Step4/Step5 cache reuse.",
                    )
                )
    if findings:
        result.fail("Step3/Step4/Step5 cache reuse can still be path/mtime-only or dataset-only.", findings)
    else:
        result.summary = "Step3/Step4/Step5 cache reuse is manifest-gated and content-fingerprint based."
    return result


def _check_console_default_summary_policy(repo_root: Path) -> RuleResult:
    result = RuleResult("R060", "default console must stay summary-level")
    required = {
        "code/odcr.py": (
            "--verbose",
            "--debug",
            "_console_level",
            "print_pre_run_banner",
            "console_level=console_level",
        ),
        "code/odcr_core/logging_meta.py": (
            "CONSOLE_LEVEL_SUMMARY",
            "CONSOLE_POLICY_SUMMARY",
            "console_summary_lines",
            "resolved_config_path",
            "source_table_path",
            "full.log",
            "console.log",
            "errors.log",
        ),
    }
    findings: list[Finding] = []
    for rel, terms in required.items():
        path = repo_root / rel
        text = _read(path) if path.is_file() else ""
        missing = [term for term in terms if term not in text]
        if missing:
            findings.append(
                Finding(
                    rel,
                    1,
                    "missing: " + ", ".join(missing),
                    "Default run console must emit summary lines while full config/source table live in run-meta files.",
                )
            )
    if findings:
        result.fail("Console summary policy coverage is incomplete.", findings)
    else:
        result.summary = "Default executable-stage console uses summary policy; detailed config/source table stay in files."
    return result


def _check_active_logs_not_legacy_defaults(repo_root: Path) -> RuleResult:
    result = RuleResult("R061", "active logs must not default to code/log.out or top-level logs/")
    findings: list[Finding] = []
    runners = repo_root / "code" / "odcr_core" / "runners.py"
    manifests = repo_root / "code" / "odcr_core" / "manifests.py"
    train_logging = repo_root / "code" / "train_logging.py"
    paths_config = repo_root / "code" / "paths_config.py"
    checks = {
        "code/odcr_core/runners.py": (
            "_full_log_file",
            "run_log_paths(cfg)[\"full\"]",
            "ODCR_SUMMARY_LOG",
            "ODCR_DUAL_TRAIN_LOG",
        ),
        "code/odcr_core/manifests.py": (
            "CONSOLE_LOG_FILENAME",
            "FULL_LOG_FILENAME",
            "DEBUG_LOG_FILENAME",
            "SAMPLES_LOG_FILENAME",
        ),
        "code/train_logging.py": (
            "runs\", \"internal",
            "full.log",
            "fallback mirror log",
        ),
        "code/paths_config.py": (
            "DEFAULT_MIRROR_LOG = \"\"",
            "fallback mirror logs are retired",
        ),
    }
    for rel, terms in checks.items():
        path = repo_root / rel
        text = _read(path) if path.is_file() else ""
        missing = [term for term in terms if term not in text]
        if missing:
            findings.append(
                Finding(
                    rel,
                    1,
                    "missing: " + ", ".join(missing),
                    "Mainline logging defaults must point to runs/.../meta or a run-meta internal path, not code/log.out or top-level logs/.",
                )
            )
    for path in (runners, manifests):
        text = _read(path) if path.is_file() else ""
        for bad in (" / \"train.log\"", " / \"eval.log\"", " / \"step4.log\"", "code/log.out"):
            if bad in text:
                findings.append(
                    Finding(
                        _rel(path, repo_root),
                        1,
                        bad,
                        "Active mainline log defaults must use meta/full.log, meta/console.log, and meta/errors.log.",
                    )
                )
    text_tl = _read(train_logging) if train_logging.is_file() else ""
    if "os.path.join(get_odcr_root(), \"logs\")" in text_tl:
        findings.append(
            Finding(
                "code/train_logging.py",
                1,
                "top-level logs default",
                "Internal fallback log directory must be runs/internal/<unit>/<run_id>/meta/full.log.",
            )
        )
    text_pc = _read(paths_config) if paths_config.is_file() else ""
    if "CODE_DIR, \"log.out\"" in text_pc:
        findings.append(
            Finding(
                "code/paths_config.py",
                1,
                "code/log.out default",
                "Legacy mirror defaults must be retired, not redirected to code/log.out.",
            )
        )
    if findings:
        result.fail("Legacy active log defaults remain.", findings)
    else:
        result.summary = "Active logs default to run-meta files; code/log.out, top-level logs/, and mirror fallbacks are not defaults."
    return result


def _check_verbose_debug_display_only(repo_root: Path) -> RuleResult:
    result = RuleResult("R062", "verbose/debug must not change training semantics")
    findings: list[Finding] = []
    odcr = repo_root / "code" / "odcr.py"
    text = _read(odcr) if odcr.is_file() else ""
    required = (
        "console_level_from_flags",
        "verbose=bool(getattr(args, \"verbose\", False))",
        "debug=bool(getattr(args, \"debug\", False))",
        "resolve_config(",
    )
    missing = [term for term in required if term not in text]
    if missing:
        findings.append(
            Finding(
                "code/odcr.py",
                1,
                "missing: " + ", ".join(missing),
                "--verbose/--debug must be consumed only by console helpers, not resolver payload construction.",
            )
        )
    for bad in (
        "set_overrides=_merged_sets(args) +",
        "sets.append(\"logging.",
        "logging.console_level",
    ):
        if bad in text:
            findings.append(
                Finding(
                    "code/odcr.py",
                    1,
                    bad,
                    "Display flags must not enter One-Control training payload or CLI --set overrides.",
                )
            )
    logging_test = repo_root / "code" / "tests" / "test_logging_console_file.py"
    test_text = _read(logging_test) if logging_test.is_file() else ""
    if "test_verbose_debug_flags_do_not_change_resolved_training_payload" not in test_text:
        findings.append(
            Finding(
                "code/tests/test_logging_console_file.py",
                1,
                "missing payload invariance test",
                "Add a test proving display-only flags leave resolved training payload unchanged.",
            )
        )
    if findings:
        result.fail("Verbose/debug display-only contract is incomplete.", findings)
    else:
        result.summary = "--verbose/--debug are local display controls and do not alter resolved training payload."
    return result


def _check_run_logs_target_run_meta(repo_root: Path) -> RuleResult:
    result = RuleResult("R078", "run logs must target runs/<stage>/<unit>/<run_id>/meta")
    findings: list[Finding] = []
    required = {
        "code/odcr_core/path_layout.py": (
            "runs/<stage>/<unit>/<run_id>/meta",
            "console.log",
            "full.log",
            "errors.log",
        ),
        "code/odcr_core/logging_meta.py": (
            "run_log_paths",
            "Path(cfg.manifest_dir)",
            "CONSOLE_LOG_FILENAME",
            "FULL_LOG_FILENAME",
            "ERRORS_LOG_FILENAME",
        ),
        "code/odcr_core/preprocess_status.py": (
            "console_log_path",
            "full_log_path",
            "errors_log_path",
            "meta_root_path / \"console.log\"",
            "meta_root_path / \"full.log\"",
        ),
    }
    for rel, terms in required.items():
        text = _read(repo_root / rel) if (repo_root / rel).is_file() else ""
        missing = [term for term in terms if term not in text]
        if missing:
            findings.append(
                Finding(
                    rel,
                    1,
                    "missing: " + ", ".join(missing),
                    "Formal run logs must be computed from the run meta directory.",
                )
            )
    for rel in ("odcr", "code/odcr.py"):
        text = _read(repo_root / rel) if (repo_root / rel).is_file() else ""
        for bad in ("_launcher_logs", "nohup "):
            if bad in text:
                findings.append(
                    Finding(
                        rel,
                        1,
                        bad,
                        "Launcher sidecar logs are retired; foreground One-Control runs own run-meta logs.",
                    )
                )
    odcr_py = _read(repo_root / "code" / "odcr.py")
    if "--daemon is retired" not in odcr_py:
        findings.append(
            Finding(
                "code/odcr.py",
                1,
                "daemon not retired",
                "--daemon must fail fast instead of writing launcher logs outside run meta.",
            )
        )
    if findings:
        result.fail("Run log path boundary coverage is incomplete.", findings)
    else:
        result.summary = "Formal run logs resolve to meta/console.log, meta/full.log, and meta/errors.log."
    return result


def _check_cache_artifacts_not_runs_meta(repo_root: Path) -> RuleResult:
    result = RuleResult("R079", "cache artifacts must not be stored under runs/meta")
    findings: list[Finding] = []
    checks = {
        "configs/odcr.yaml": ("grouped_text_cache_dir: cache/preprocess_b", "token_window_cache_dir: cache/preprocess_c"),
        "code/odcr_core/preprocess_schema.py": ("cache/preprocess_b", "cache/preprocess_c"),
        "code/odcr_core/config_resolver.py": ("cache/preprocess_b", "cache/preprocess_c"),
        "code/compute_embeddings.py": ("entry_dir = cache_root / cache_key",),
        "code/infer_domain_semantics.py": ("return base_dir / _cache_digest(fingerprint)",),
        "code/tests/test_path_layout_boundaries.py": ("test_cache_paths_are_not_under_runs_meta",),
    }
    for rel, terms in checks.items():
        text = _read(repo_root / rel) if (repo_root / rel).is_file() else ""
        missing = [term for term in terms if term not in text]
        if missing:
            findings.append(
                Finding(
                    rel,
                    1,
                    "missing: " + ", ".join(missing),
                    "Preprocess B/C caches must live under cache/preprocess_b|c/<cache_key>.",
                )
            )
    banned_terms = (
        "runs/preprocess_b/grouped_text_cache",
        "runs/preprocess_c/token_window_cache",
        "entry_dir = cache_root / dataset / spec.name / cache_key",
        "return base_dir / dataset / f\"{spec.name}__{_cache_digest(fingerprint)}\"",
    )
    for rel in (
        "configs/odcr.yaml",
        "code/odcr_core/preprocess_schema.py",
        "code/odcr_core/config_resolver.py",
        "code/compute_embeddings.py",
        "code/infer_domain_semantics.py",
    ):
        text = _read(repo_root / rel) if (repo_root / rel).is_file() else ""
        for bad in banned_terms:
            if bad in text:
                findings.append(
                    Finding(
                        rel,
                        1,
                        bad,
                        "Active cache defaults and writers must target cache/, not runs/.",
                    )
                )
    if findings:
        result.fail("Cache/run separation is incomplete.", findings)
    else:
        result.summary = "Preprocess cache defaults and cache-key directories target cache/preprocess_b|c, outside runs/meta."
    return result


def _check_ai_analysis_not_full_log_mirror(repo_root: Path) -> RuleResult:
    result = RuleResult("R080", "AI_analysis must not be an active full-log mirror")
    findings: list[Finding] = []
    path_layout = _read(repo_root / "code" / "odcr_core" / "path_layout.py")
    required = (
        "do not mirror full training logs",
        "full_log",
        "ai_analysis_may_copy=False",
        "AI_analysis/<bucket>",
    )
    missing = [term for term in required if term not in path_layout]
    if missing:
        findings.append(
            Finding(
                "code/odcr_core/path_layout.py",
                1,
                "missing: " + ", ".join(missing),
                "Artifact role registry must state AI_analysis is for digests/ledgers, not full-log mirrors.",
            )
        )
    hook = _read(repo_root / ".codex" / "hooks" / "odcr_post_edit_stop.py")
    if "AI_analysis" not in hook or "01_raw_logs" not in hook or "codex_hooks" not in hook:
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.py",
                1,
                "missing AI_analysis codex_hooks log root",
                "Post-edit diagnostics must stay under AI_analysis/01_raw_logs/codex_hooks.",
            )
        )
    if "Path(\"runs\")" in hook or "_launcher_logs" in hook:
        findings.append(
            Finding(
                ".codex/hooks/odcr_post_edit_stop.py",
                1,
                "runs hook diagnostics sink",
                "Codex hook diagnostics must not write into formal runs/.",
            )
        )
    test_text = _read(repo_root / "code" / "tests" / "test_path_layout_boundaries.py")
    if "test_ai_analysis_is_not_active_full_log_sink" not in test_text:
        findings.append(
            Finding(
                "code/tests/test_path_layout_boundaries.py",
                1,
                "missing AI_analysis boundary test",
                "Add a test proving full_log/console_log are not copyable AI_analysis sinks.",
            )
        )
    if findings:
        result.fail("AI_analysis full-log mirror boundary is incomplete.", findings)
    else:
        result.summary = "AI_analysis is limited to audit/search/evidence/summary/report material and hook diagnostics."
    return result


def _check_data_merged_do_not_receive_logs(repo_root: Path) -> RuleResult:
    result = RuleResult("R081", "data/merged must not receive logs")
    findings: list[Finding] = []
    path_layout = _read(repo_root / "code" / "odcr_core" / "path_layout.py")
    for term in ("data_artifact", "merged_artifact", "canonical data contract outputs only; no logs"):
        if term not in path_layout:
            findings.append(
                Finding(
                    "code/odcr_core/path_layout.py",
                    1,
                    "missing: " + term,
                    "Artifact roles for data/merged must state that logs do not belong there.",
                )
            )
    for rel in ("code/odcr_core/preprocess_runtime.py", "code/odcr_core/logging_meta.py", "code/train_logging.py"):
        text = _read(repo_root / rel) if (repo_root / rel).is_file() else ""
        for bad in (
            "data_root / \"full.log\"",
            "data_root / \"console.log\"",
            "merged_root / \"full.log\"",
            "merged_root / \"console.log\"",
            "data/\" + \"log",
            "merged/\" + \"log",
        ):
            if bad in text:
                findings.append(
                    Finding(
                        rel,
                        1,
                        bad,
                        "Logs must stay under runs/.../meta, not data/ or merged/.",
                    )
                )
    if findings:
        result.fail("data/merged log exclusion is incomplete.", findings)
    else:
        result.summary = "data/ and merged/ roles are data-contract-only, with no active log writers."
    return result


def _check_top_level_logs_and_fallbacks_retired(repo_root: Path) -> RuleResult:
    result = RuleResult("R082", "top-level logs/code/log.out/fallback logs are retired")
    findings: list[Finding] = []
    active_files = (
        "odcr",
        "code/odcr.py",
        "code/train_logging.py",
        "code/paths_config.py",
        "code/executors/step4_entry.py",
    )
    banned = ("_adhoc_logs", "_legacy_logs", "_launcher_logs", "nohup ", "code/log.out")
    for rel in active_files:
        text = _read(repo_root / rel) if (repo_root / rel).is_file() else ""
        for bad in banned:
            if bad in text:
                findings.append(
                    Finding(
                        rel,
                        1,
                        bad,
                        "Active defaults must not use retired top-level, launcher, mirror, or log.out paths.",
                    )
                )
    paths_config = _read(repo_root / "code" / "paths_config.py")
    if "DEFAULT_MIRROR_LOG = \"\"" not in paths_config or "return False" not in paths_config:
        findings.append(
            Finding(
                "code/paths_config.py",
                1,
                "mirror fallback not retired",
                "append_log_dual must write only the primary run-meta log.",
            )
        )
    train_logging = _read(repo_root / "code" / "train_logging.py")
    if "runs\", \"internal" not in train_logging or "meta\", \"full.log\"" not in train_logging:
        findings.append(
            Finding(
                "code/train_logging.py",
                1,
                "internal fallback is not run-meta shaped",
                "Internal fallback must be runs/internal/task{T}/{run_id}/meta/full.log.",
            )
        )
    if findings:
        result.fail("Retired log fallback defaults remain active.", findings)
    else:
        result.summary = "Daemon sidecar, mirror fallback, top-level logs, and code/log.out defaults are retired."
    return result


def _check_old_layout_default_log_writes_retired(repo_root: Path) -> RuleResult:
    result = RuleResult("R084", "active code must not write top-level logs or code/log.out")
    findings: list[Finding] = []
    for rel in OLD_LAYOUT_LOG_ACTIVE_FILES:
        path = repo_root / rel
        if path.is_file():
            findings.extend(_scan_old_layout_log_text("R084", _read(path), path=rel))
    if not scan_old_layout_log_snippet("R084", 'open("logs/run.log", "w")\nopen("code/log.out", "a")'):
        findings.append(
            Finding(
                "code/tools/check_one_control_guardrails.py",
                1,
                "R084 synthetic forbidden write not detected",
                "R084 must catch active writes to top-level logs/ and code/log.out.",
            )
        )
    if findings:
        result.fail("Active old-layout log write defaults remain.", findings)
    else:
        result.summary = "Active writers avoid top-level logs/ and code/log.out defaults."
    return result


def _check_tail_only_latest_run_summary_meta_logs(repo_root: Path) -> RuleResult:
    result = RuleResult("R085", "odcr tail must use latest.json -> run_summary.json -> meta logs only")
    path = repo_root / "code" / "odcr.py"
    text = _read(path) if path.is_file() else ""
    findings: list[Finding] = []
    required = (
        "_resolve_tail_log_path",
        "latest.json",
        "latest_summary_path",
        "run_summary.json",
        "console_log_path",
        "full_log_path",
        "errors_log_path",
        "meta/{filename}",
    )
    missing = [term for term in required if term not in text]
    if missing:
        findings.append(
            Finding(
                "code/odcr.py",
                1,
                "missing: " + ", ".join(missing),
                "Keep odcr tail anchored on latest.json -> run_summary.json -> meta logs.",
            )
        )
    tail_blocks: list[tuple[int, str, str]] = []
    for line_no, name, block in _iter_python_function_blocks(text):
        if name in {"cmd_tail", "_resolve_tail_log_path"}:
            tail_blocks.append((line_no, name, block))
    if len(tail_blocks) != 2:
        findings.append(
            Finding(
                "code/odcr.py",
                1,
                "tail helper blocks not found",
                "Keep tail resolution small and explicit in cmd_tail/_resolve_tail_log_path.",
            )
        )
    for line_no, name, block in tail_blocks:
        for finding in _scan_old_layout_log_text("R085", block, path="code/odcr.py"):
            findings.append(
                Finding(
                    finding.path,
                    line_no + finding.line - 1,
                    f"{name}: {finding.text}",
                    finding.suggestion,
                )
            )
    bad = 'candidates = [meta / "console.log", meta / "train.log"]\nlegacy_parent = Path("runs/task4/step3")'
    if not scan_old_layout_log_snippet("R085", bad, path="code/odcr.py"):
        findings.append(
            Finding(
                "code/tools/check_one_control_guardrails.py",
                1,
                "R085 synthetic tail fallback not detected",
                "R085 must catch legacy tail fallback chains.",
            )
        )
    if findings:
        result.fail("odcr tail old-layout fallback risk remains.", findings)
    else:
        result.summary = "odcr tail resolves latest.json -> run_summary.json -> meta/console|full|errors.log only."
    return result


def _check_no_old_fallback_log_paths(repo_root: Path) -> RuleResult:
    result = RuleResult("R086", "no active nohup/fallback/mirror/timestamp log fallback paths")
    findings: list[Finding] = []
    for rel in OLD_LAYOUT_LOG_ACTIVE_FILES:
        path = repo_root / rel
        if path.is_file():
            findings.extend(_scan_old_layout_log_text("R086", _read(path), path=rel))
    bad = 'open("nohup_task.log", "a")\nopen("fallback.log", "a")\nopen("mirror.log", "a")'
    if not scan_old_layout_log_snippet("R086", bad):
        findings.append(
            Finding(
                "code/tools/check_one_control_guardrails.py",
                1,
                "R086 synthetic fallback logs not detected",
                "R086 must catch nohup/fallback/mirror/timestamp log fallbacks.",
            )
        )
    if findings:
        result.fail("Active fallback log paths remain.", findings)
    else:
        result.summary = "Active code has no nohup/fallback/mirror/timestamp log fallback paths."
    return result


def _check_ai_analysis_not_active_training_mirror_old_layout(repo_root: Path) -> RuleResult:
    result = RuleResult("R087", "AI_analysis must not be an active training full-log mirror")
    findings: list[Finding] = []
    for rel in OLD_LAYOUT_LOG_ACTIVE_FILES:
        path = repo_root / rel
        if path.is_file():
            findings.extend(_scan_old_layout_log_text("R087", _read(path), path=rel))
    bad = 'shutil.copyfile(run_dir / "meta" / "full.log", "AI_analysis/full_train.log")'
    if not scan_old_layout_log_snippet("R087", bad):
        findings.append(
            Finding(
                "code/tools/check_one_control_guardrails.py",
                1,
                "R087 synthetic AI_analysis mirror not detected",
                "R087 must catch active full-log mirrors into AI_analysis.",
            )
        )
    if findings:
        result.fail("AI_analysis active full-log mirror remains.", findings)
    else:
        result.summary = "AI_analysis remains audit/ledger/report storage, not an active full-log mirror."
    return result


def _check_data_merged_no_log_files_old_layout(repo_root: Path) -> RuleResult:
    result = RuleResult("R088", "data/merged directories must not receive log files")
    findings: list[Finding] = []
    for rel in OLD_LAYOUT_LOG_ACTIVE_FILES:
        path = repo_root / rel
        if path.is_file():
            findings.extend(_scan_old_layout_log_text("R088", _read(path), path=rel))
    bad = 'open("data/foo.log", "w")\nopen("merged/foo.log", "w")'
    if not scan_old_layout_log_snippet("R088", bad):
        findings.append(
            Finding(
                "code/tools/check_one_control_guardrails.py",
                1,
                "R088 synthetic data/merged logs not detected",
                "R088 must catch log files written under data/ or merged/.",
            )
        )
    if findings:
        result.fail("Active data/merged log writes remain.", findings)
    else:
        result.summary = "Active code does not write log files under data/ or merged/."
    return result


def _check_agents_post_edit_scope_not_fixed_step3(repo_root: Path) -> RuleResult:
    result = RuleResult("R089", "AGENTS.md must require narrowest post-edit scope")
    path = repo_root / "AGENTS.md"
    if not path.is_file():
        result.fail(
            "AGENTS.md is missing.",
            [Finding("AGENTS.md", 1, "missing", "Restore AGENTS.md with post-edit validation instructions.")],
        )
        return result
    text = _read(path)
    normalized = " ".join(text.split())
    findings: list[Finding] = []
    required_terms = (
        "narrowest applicable post-edit validation scope",
        "python code/tools/odcr_post_edit_check.py --scope <scope>",
        "ignored-only, dirty-workspace-only",
        "not fixed defaults for every user-facing change",
    )
    for term in required_terms:
        if term not in text and term not in normalized:
            findings.append(
                Finding(
                    "AGENTS.md",
                    1,
                    "missing: " + term,
                    "Document scope-first post-edit validation and avoid universal Step3 checks.",
                )
            )
    header = "## Required Checks Before Finishing"
    section = text.split(header, 1)[1] if header in text else text
    bad_fixed_patterns = (
        "For user-facing changes, also run",
        "For user-facing changes also run",
    )
    for pattern in bad_fixed_patterns:
        if pattern in section:
            findings.append(
                Finding(
                    "AGENTS.md",
                    text[: text.find(pattern)].count("\n") + 1,
                    pattern,
                    "Replace fixed user-facing Step3 checks with narrowest applicable scope guidance.",
                )
            )
    if "./odcr step3 --task 2 --dry-run" in section and "not fixed defaults for every user-facing change" not in section:
        line = text[: text.find("./odcr step3 --task 2 --dry-run")].count("\n") + 1
        findings.append(
            Finding(
                "AGENTS.md",
                line,
                "./odcr step3 --task 2 --dry-run",
                "Mention Step3 dry-run only as a Step3/config/all scope-owned check.",
            )
        )
    if findings:
        result.fail("AGENTS.md still risks prescribing fixed Step3 validation for all changes.", findings)
    else:
        result.summary = "AGENTS.md requires narrowest-scope post-edit validation and scopes Step3 dry-run narrowly."
    return result


def _check_metrics_filename_canonical(repo_root: Path) -> RuleResult:
    result = RuleResult("R083", "metrics and audit filenames must be canonical")
    findings: list[Finding] = []
    path_layout = _read(repo_root / "code" / "odcr_core" / "path_layout.py")
    required_names = (
        "metrics.jsonl",
        "epoch_summary.csv",
        "loss_breakdown.jsonl",
        "gpu_profile.jsonl",
        "rcr_distribution.json",
        "eval_metrics.json",
        "rerank_summary.json",
        "data_audit_summary.csv",
    )
    missing = [name for name in required_names if name not in path_layout]
    if missing:
        findings.append(
            Finding(
                "code/odcr_core/path_layout.py",
                1,
                "missing: " + ", ".join(missing),
                "Canonical metrics/audit filenames must be registered in one helper.",
            )
        )
    checks = {
        "code/train_logging.py": ("path_layout.metrics_filename(\"metrics\")",),
        "code/executors/step5_engine.py": (
            "path_layout.metrics_filename(\"data_audit\")",
            "path_layout.metrics_filename(\"data_audit_summary\")",
            "path_layout.eval_metrics_filename",
        ),
        "code/tests/test_path_layout_boundaries.py": ("test_metrics_filename_helper",),
    }
    for rel, terms in checks.items():
        text = _read(repo_root / rel) if (repo_root / rel).is_file() else ""
        missing_terms = [term for term in terms if term not in text]
        if missing_terms:
            findings.append(
                Finding(
                    rel,
                    1,
                    "missing: " + ", ".join(missing_terms),
                    "Active writers/tests must use canonical metrics filename helpers.",
                )
            )
    for rel in ("code/train_logging.py", "code/executors/step5_engine.py"):
        text = _read(repo_root / rel) if (repo_root / rel).is_file() else ""
        for bad in ("train_epoch_metrics.jsonl", "step5_train_data_audit_summary.csv"):
            if bad in text:
                findings.append(
                    Finding(
                        rel,
                        1,
                        bad,
                        "Retired metrics/audit filenames must not be active write targets.",
                    )
                )
    if findings:
        result.fail("Metrics/audit filename canonicalization is incomplete.", findings)
    else:
        result.summary = "Metrics/audit writers use canonical helper names; retired filenames are not write targets."
    return result


def _check_step3_v0_parameter_surface(repo_root: Path) -> RuleResult:
    result = RuleResult("R099", "Step3 v0 parameter surface must be One-Control and legacy-free")
    findings: list[Finding] = []
    cfg_text = _read(repo_root / "configs" / "odcr.yaml")
    resolver = _read(repo_root / "code" / "odcr_core" / "config_resolver.py")
    schema = _read(repo_root / "code" / "odcr_core" / "config_schema.py")
    config_py = _read(repo_root / "code" / "config.py")
    step3_core = _read(repo_root / "code" / "executors" / "step3_train_core.py")
    required_yaml = (
        "default_task: 2",
        "optimizer:",
        "name: adamw",
        "train_precision: bf16",
        "allow_tf32: true",
        "amp_autocast: true",
        "grad_scaler: false",
        "max_grad_norm: 0.5",
        "max_epochs: 40",
        "warmup_ratio: 0.06",
        "min_lr_ratio: 0.05",
        "valid_batch_size: null",
        "derive_from_train: true",
        "strong_related:",
        "weak_cross_platform:",
        "cache:",
        "tokenizer_schema_version: odcr_step3_tokenizer_cache/2",
        "prefetcher:",
        "cross_rank_structured_gather:",
        "enabled: true",
        "diagnostic_allow_disabled: false",
        "task_profiles:",
        "task2_am_movies_to_cds:",
        "profile_id: task2_strong_forward_g1s",
        "candidate: G1S",
        "task5_cds_to_movies:",
        "profile_id: task5_strong_reverse_g1_init",
        "task8_tripadvisor_to_yelp:",
        "profile_id: task8_weak_forward_init",
        "task7_yelp_to_tripadvisor:",
        "profile_id: task7_weak_reverse_init",
        "backup_profiles:",
        "task2_g1_backup:",
        "task2_g0_backup:",
        "exploration_profiles:",
        "task2_g2_effective_pool_2048:",
        "formal_allowed: false",
        "probe_only: true",
        "backup_only: true",
        "manual_selection_required: true",
        "exploration_only: true",
        "replacement_gate_status: failed_or_not_passed",
        "activation_checkpointing:",
        "profile_buffer_policy: gpu_resident",
        "worker_profiles:",
        "pin_memory: true",
        "persistent_workers: true",
        "non_blocking_h2d: true",
    )
    for term in required_yaml:
        if term not in cfg_text:
            findings.append(Finding("configs/odcr.yaml", 1, "missing: " + term, "Step3 v0 defaults must be explicit One-Control YAML."))
    for term in (
        "2: {source: AM_Movies, target: AM_CDs, scenario: strong_related, direction: forward}",
        "5: {source: AM_CDs, target: AM_Movies, scenario: strong_related, direction: reverse}",
        "7: {source: Yelp, target: TripAdvisor, scenario: weak_cross_platform, direction: reverse}",
        "8: {source: TripAdvisor, target: Yelp, scenario: weak_cross_platform, direction: forward}",
    ):
        if term not in cfg_text:
            findings.append(Finding("configs/odcr.yaml", 1, "missing scenario row: " + term, "Paper-mainline task metadata is required."))
    banned_yaml = (
        "step3:\n  train:\n    epochs:",
        "step3:\n  train:\n    coef:",
        "train_precision: fp32",
        "optim.Adam",
    )
    for term in banned_yaml:
        if term in cfg_text:
            findings.append(Finding("configs/odcr.yaml", 1, term, "Retired Step3 v0 controls must not be active YAML defaults."))
    try:
        import yaml

        raw_cfg = yaml.safe_load(cfg_text) or {}
    except Exception:
        raw_cfg = {}
    step3_cfg = raw_cfg.get("step3") if isinstance(raw_cfg, dict) else {}
    step3_train = (step3_cfg or {}).get("train") if isinstance(step3_cfg, dict) else {}
    if isinstance(step3_train, dict) and (
        int(step3_train.get("batch_size") or 0) != 1536
        or int(step3_train.get("per_gpu_batch_size") or 0) != 768
    ):
        findings.append(Finding("configs/odcr.yaml", 1, "step3.train", "Active Step3 formal default must be task2 G1S 1536/768 with per_gpu_batch_size."))
    if isinstance(raw_cfg, dict) and (raw_cfg.get("project") or {}).get("default_task") != 2:
        findings.append(Finding("configs/odcr.yaml", 1, "project.default_task", "Paper-mainline default task must remain task2."))
    gather_cfg = (step3_cfg or {}).get("cross_rank_structured_gather") if isinstance(step3_cfg, dict) else {}
    if not (isinstance(gather_cfg, dict) and gather_cfg.get("enabled") is True and gather_cfg.get("mode") == "local_gradient_context"):
        findings.append(Finding("configs/odcr.yaml", 1, "step3.cross_rank_structured_gather", "Formal Step3 gather must default to true/local_gradient_context."))
    profiles = (step3_cfg or {}).get("task_profiles") if isinstance(step3_cfg, dict) else {}
    expected_profiles = {
        "task2_am_movies_to_cds": (2, "task2_strong_forward_g1s"),
        "task5_cds_to_movies": (5, "task5_strong_reverse_g1_init"),
        "task8_tripadvisor_to_yelp": (8, "task8_weak_forward_init"),
        "task7_yelp_to_tripadvisor": (7, "task7_weak_reverse_init"),
    }
    if not isinstance(profiles, dict):
        findings.append(Finding("configs/odcr.yaml", 1, "step3.task_profiles", "Four isolated Step3 task profiles are required."))
    else:
        seen_profile_ids: set[str] = set()
        for key, (task_id, profile_id) in expected_profiles.items():
            item = profiles.get(key)
            if not isinstance(item, dict) or int(item.get("task_id") or -1) != task_id or item.get("profile_id") != profile_id:
                findings.append(Finding("configs/odcr.yaml", 1, key, "Missing or mismatched isolated paper-task profile."))
            elif profile_id in seen_profile_ids:
                findings.append(Finding("configs/odcr.yaml", 1, profile_id, "Step3 task profile ids must be unique."))
            else:
                seen_profile_ids.add(profile_id)
    for removed in ("grad_accum", "gradient_accumulation_steps", "accumulate_grad_batches"):
        if isinstance(step3_train, dict) and removed in step3_train:
            findings.append(Finding("configs/odcr.yaml", 1, removed, "Step3 no-accum architecture forbids active accumulation controls."))
    if isinstance(step3_cfg, dict):
        for removed_block in ("smoke_ladder", "performance_ladder", "performance_probe", "short_pilot"):
            if removed_block in step3_cfg:
                findings.append(
                    Finding(
                        "configs/odcr.yaml",
                        1,
                        f"step3.{removed_block}",
                        "Step3 clean baseline keeps backup/exploration in isolated profile blocks, not active ladder/probe/pilot blocks.",
                    )
                )
    required_code_terms = {
        "code/odcr_core/config_resolver.py": (
            "_validate_config_shape",
            "_reject_unknown_keys",
            "_resolve_step3_optimizer_config",
            "_resolve_step3_backend_config",
            "_resolve_step3_tokenizer_evidence_config",
            "_resolve_step3_scheduler_config",
            "_resolve_step3_eval_config",
            "_select_step3_task_profile",
            "_resolve_step3_task_profile_config",
            "_resolve_step3_exploration_profiles_config",
            "_resolve_step3_worker_profiles_config",
            "_resolve_step3_prefetcher_config",
            "_resolve_step3_cross_rank_gather_config",
            "_resolve_step3_memory_config",
            "_resolve_step3_cache_policy_config",
            "step3.optimizer.name must be 'adamw'",
        ),
        "code/odcr_core/config_schema.py": (
            "optimizer_config_json",
            "precision_config_json",
            "tokenizer_max_length",
            "evidence_max_length",
            "max_grad_norm",
            "pin_memory",
            "non_blocking_h2d",
            "prefetcher_config_json",
            "cross_rank_structured_gather_config_json",
            "memory_config_json",
            "task_profile_config_json",
            "exploration_profiles_config_json",
            "task_profile_id",
            "profile_isolation_hash",
        ),
        "code/config.py": (
            "Step3 optimizer must be AdamW",
            "effective Step3 train_precision must be bf16",
            "ODCR_HARDWARE_PROFILE_JSON missing",
        ),
        "code/executors/step3_train_core.py": (
            "build_step3_optimizer",
            "optim.AdamW",
            "float(final_cfg.max_grad_norm)",
            "resolved.tokenizer_max_length",
            "resolved.evidence_max_length",
            "bool(resolved.pin_memory)",
            "bool(final_cfg.non_blocking_h2d)",
            "Step3CUDAPrefetcher",
            "startup_steady_state_timing",
            "gather_step3_structured_context_local_gradient",
            "apply_step3_memory_controls",
        ),
    }
    texts = {
        "code/odcr_core/config_resolver.py": resolver,
        "code/odcr_core/config_schema.py": schema,
        "code/config.py": config_py,
        "code/executors/step3_train_core.py": step3_core,
    }
    for rel, terms in required_code_terms.items():
        missing = [term for term in terms if term not in texts[rel]]
        if missing:
            findings.append(Finding(rel, 1, "missing: " + ", ".join(missing), "Step3 v0 controls must flow through resolved payload."))
    banned_code_terms = {
        "code/executors/step3_train_core.py": (
            "optim.Adam(",
            "weight_decay=1e-5",
            "clip_grad_norm_(step3_trainable_parameters(model), 1.0)",
            "self.max_length = 25",
            "self.evidence_length = 24",
            "pin_memory = torch.cuda.is_available()",
        ),
        "code/odcr_core/runners.py": (
            '"ODCR_RUNTIME_PRECISION_MODE": "bf16"',
        ),
    }
    for rel, terms in banned_code_terms.items():
        text = _read(repo_root / rel)
        for term in terms:
            if term in text:
                findings.append(Finding(rel, 1, term, "Retired Step3 active hardcode/fallback is prohibited."))
    if findings:
        result.fail("Step3 v0 parameter surface still has legacy drift or missing One-Control fields.", findings)
    else:
        result.summary = "Step3 no-accum optimizer/precision/length/scheduler/grad-norm/hardware/scenario controls are One-Control and legacy-free."
    return result


def _check_step3_s2r_perf_cache_downstream_guardrails(repo_root: Path) -> RuleResult:
    result = RuleResult("R100", "Step3 clean cache/downstream/profile split must stay active")
    findings: list[Finding] = []
    cfg_text = _read(repo_root / "configs" / "odcr.yaml")
    step3_core = _read(repo_root / "code" / "executors" / "step3_train_core.py")
    checkpoint = _read(repo_root / "code" / "odcr_core" / "training_checkpoint.py")
    step4 = _read(repo_root / "code" / "executors" / "step4_engine.py")
    required = {
        "configs/odcr.yaml": (
            "task_profiles:",
            "task2_am_movies_to_cds:",
            "candidate: G1S",
            "task2_g1_backup:",
            "candidate: G1",
            "backup_profiles:",
            "task2_g0_backup:",
            "backup_only: true",
            "manual_selection_required: true",
            "exploration_profiles:",
            "task2_g2_effective_pool_2048:",
            "profile_buffer_policy: cpu_pinned_batch_gather",
            "exploration_only: true",
            "replacement_gate_status: failed_or_not_passed",
            "worker_profiles:",
            "W3:",
            "train_workers_per_rank: 5",
            "prefetch_factor: 4",
        ),
        "code/executors/step3_train_core.py": (
            "odcr_step3_tokenizer_cache/2",
            "tokenizer_cache_compat_hash",
            "record_only_lineage",
            "retired_v1_schema_rebuild_required",
            "Step3CUDAPrefetcher",
            "compute_wait_for_prefetch",
            "optimizer_time",
            "scheduler_time",
        ),
        "code/odcr_core/training_checkpoint.py": (
            "odcr_step3_checkpoint_compat/2",
            "semantic_model_compat_hash",
            "train_runtime_config_hash",
            "optimizer_config_hash",
            "record_only_fields",
        ),
        "code/executors/step4_engine.py": (
            "semantic_model_compat_hash",
            "data_contract_hash",
            "artifact_lineage_hash",
            "validate_step3_checkpoint_lineage",
        ),
    }
    texts = {
        "configs/odcr.yaml": cfg_text,
        "code/executors/step3_train_core.py": step3_core,
        "code/odcr_core/training_checkpoint.py": checkpoint,
        "code/executors/step4_engine.py": step4,
    }
    for rel, terms in required.items():
        missing = [term for term in terms if term not in texts[rel]]
        if missing:
            findings.append(Finding(rel, 1, "missing: " + ", ".join(missing), "S2-R cache/downstream/performance contracts must be explicit."))
    step4_func = step4[step4.find("def _validate_step3_checkpoint_lineage_for_step4"): step4.find("def ", step4.find("def _validate_step3_checkpoint_lineage_for_step4") + 10)]
    for forbidden in ("step3_optimizer_config_hash", "optimizer_config_hash", "train_runtime_config_hash", "batch_semantics_hash", "ddp_config_hash"):
        if forbidden in step4_func:
            findings.append(
                Finding(
                    "code/executors/step4_engine.py",
                    1,
                    forbidden,
                    "Step4 hard gate must not reject semantic checkpoints for record-only optimizer/batch/runtime metadata.",
                )
            )
    try:
        import yaml

        raw_cfg = yaml.safe_load(cfg_text) or {}
    except Exception:
        raw_cfg = {}
    step3 = raw_cfg.get("step3") if isinstance(raw_cfg, dict) else {}
    step3_train = (step3 or {}).get("train") if isinstance(step3, dict) else {}
    if not (
        isinstance(step3_train, dict)
        and int(step3_train.get("batch_size") or 0) == 1536
        and int(step3_train.get("per_gpu_batch_size") or 0) == 768
        and all(key not in step3_train for key in ("grad_accum", "gradient_accumulation_steps", "accumulate_grad_batches"))
    ):
        findings.append(Finding("configs/odcr.yaml", 1, "step3.train", "Active Step3 default must be task2 G1 no-accum 1536/768 with per_gpu_batch_size and no grad_accum field."))
    exploration = (step3 or {}).get("exploration_profiles") if isinstance(step3, dict) else {}
    g2 = exploration.get("task2_g2_effective_pool_2048") if isinstance(exploration, dict) else {}
    if not (isinstance(g2, dict) and g2.get("probe_only") is True and g2.get("formal_allowed") is False):
        findings.append(Finding("configs/odcr.yaml", 1, "step3.exploration_profiles.task2_g2_effective_pool_2048", "G2 exploration must be probe_only and formal disallowed."))
    if "_ddp_no_sync_model" in step3_core or ".no_sync(" in step3_core:
        findings.append(Finding("code/executors/step3_train_core.py", 1, "no_sync", "Step3 no-accum train loop must not keep active no_sync accumulation paths."))
    if findings:
        result.fail("Step3 clean cache/downstream/profile guardrails failed.", findings)
    else:
        result.summary = "Step3 split remains active with no-accum batch semantics, cache v2 tokenizer compat, semantic downstream gate, prefetcher, and isolated backup/exploration profiles."
    return result


def _check_step3_formal_logging_param_surface(repo_root: Path) -> RuleResult:
    result = RuleResult("R101", "Step3 formal logging and parameter surface must remain closed")
    findings: list[Finding] = []
    runners = _read(repo_root / "code" / "odcr_core" / "runners.py")
    logging_meta = _read(repo_root / "code" / "odcr_core" / "logging_meta.py")
    train_logging = _read(repo_root / "code" / "train_logging.py")
    manifests = _read(repo_root / "code" / "odcr_core" / "manifests.py")
    odcr_py = _read(repo_root / "code" / "odcr.py")
    step3_core = _read(repo_root / "code" / "executors" / "step3_train_core.py")

    checks = {
        "code/odcr_core/runners.py": (
            "append_full_log",
            "[raw child]",
            "========== LAUNCHER COMMAND ==========",
        ),
        "code/odcr_core/logging_meta.py": (
            "odcr_step3_logging/2",
            "authoritative_full_log=true",
            "training_runtime_config_path",
            "debug.log is only a transport mirror",
        ),
        "code/train_logging.py": (
            "log_run_header",
            "ROUTE_DETAIL",
            "append_step3_loss_breakdown_jsonl",
            "append_step3_timing_profile_jsonl",
            "append_step3_gpu_profile_jsonl",
            "append_step3_epoch_summary_csv",
        ),
        "code/odcr_core/manifests.py": (
            "authoritative_full_log_path",
            "training_runtime_config_path",
            "formal_snapshot_view",
            "build_formal_source_table_snapshot",
        ),
        "code/odcr.py": (
            "formal_snapshot_view",
            "build_formal_source_table_snapshot",
        ),
        "code/executors/step3_train_core.py": (
            "write_training_runtime_config_artifact",
            "append_step3_loss_breakdown_jsonl",
            "append_step3_timing_profile_jsonl",
            "append_step3_gpu_profile_jsonl",
            "append_step3_epoch_summary_csv",
        ),
    }
    for rel, terms in checks.items():
        text = {
            "code/odcr_core/runners.py": runners,
            "code/odcr_core/logging_meta.py": logging_meta,
            "code/train_logging.py": train_logging,
            "code/odcr_core/manifests.py": manifests,
            "code/odcr.py": odcr_py,
            "code/executors/step3_train_core.py": step3_core,
        }[rel]
        missing = [term for term in terms if term not in text]
        if missing:
            findings.append(Finding(rel, 1, "missing: " + ", ".join(missing), "Keep Step3 formal logging closure active."))
    if 'os.path.dirname(log_path), "resolved_config.json"' in step3_core or "[Config resolved] wrote" in step3_core:
        findings.append(
            Finding(
                "code/executors/step3_train_core.py",
                1,
                "child resolved_config writer",
                "Step3 child must write training_runtime_config.json, never parent canonical resolved_config.json.",
            )
        )
    if "ROUTE_SUMMARY" in train_logging[train_logging.find("def log_config_snapshot"): train_logging.find("def log_run_snapshot")]:
        findings.append(
            Finding(
                "code/train_logging.py",
                1,
                "RUN_CONFIG route summary",
                "RUN_CONFIG must route to detail/full.log, not default console.",
            )
        )
    if findings:
        result.fail("Step3 logging/config artifact closure regressed.", findings)
    else:
        result.summary = "full.log is authoritative, console is compact, child runtime config is split, and structured Step3 metrics are active."
    return result


def _check_step3_formal_view_dynamic(repo_root: Path) -> RuleResult:
    result = RuleResult("R102", "Step3 formal view must hide backup/exploration/probe/pilot by default")
    findings: list[Finding] = []
    code_dir = repo_root / "code"
    inserted = False
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
        inserted = True
    try:
        from odcr_core.config_resolver import resolve_config
        from odcr_core.manifests import build_formal_source_table_snapshot, formal_snapshot_view

        _cfg, _sources, snapshot = resolve_config(
            config_path=repo_root / "configs" / "odcr.yaml",
            command="step3",
            task_id=2,
            set_overrides=[],
            dry_run=True,
            run_id="auto",
            mode="full",
        )
        formal = formal_snapshot_view(snapshot)
        source_table = build_formal_source_table_snapshot(snapshot)
        formal_text = json.dumps(formal, ensure_ascii=False)
        source_text = json.dumps(source_table, ensure_ascii=False)
        forbidden = (
            "step3_backup_profiles",
            "step3_exploration_profiles",
            "step3_performance_ladder",
            "step3_performance_probe",
            "step3_short_pilot",
            "task2_g2_effective_pool_2048",
            "probe_only",
        )
        for term in forbidden:
            if term in formal_text:
                findings.append(Finding("code/odcr_core/manifests.py", 1, term, "Default formal Step3 view must hide backup/exploration/probe/pilot."))
        for term in ("backup", "exploration", "performance_probe", "short_pilot", "step5"):
            if term in source_text:
                findings.append(Finding("code/odcr_core/manifests.py", 1, term, "Default formal source_table must exclude non-formal rows."))
        g2 = snapshot["step3_exploration_profiles"]["task2_g2_effective_pool_2048"]
        if not bool(g2.get("probe_only")) or bool(g2.get("formal_allowed")):
            findings.append(Finding("configs/odcr.yaml", 1, repr(g2), "G2 must remain probe_only=true and formal_allowed=false."))
        for task_id in (5, 8, 7):
            _cfg_i, _sources_i, snap_i = resolve_config(
                config_path=repo_root / "configs" / "odcr.yaml",
                command="step3",
                task_id=task_id,
                set_overrides=[],
                dry_run=True,
                run_id="auto",
                mode="full",
            )
            role = str((snap_i.get("train") or {}).get("step3_batch_candidate_role") or "")
            if "task2_" in role:
                findings.append(
                    Finding("code/odcr_core/config_resolver.py", 1, f"task{task_id}: {role}", "Non-task2 profiles must not inherit task2 ladder roles.")
                )
    except Exception as exc:
        findings.append(Finding("code/odcr_core/config_resolver.py", 1, repr(exc), "Formal view guardrail must remain import-safe."))
    finally:
        if inserted:
            try:
                sys.path.remove(str(code_dir))
            except ValueError:
                pass
    if findings:
        result.fail("Step3 formal view or task-profile role isolation regressed.", findings)
    else:
        result.summary = "Default Step3 view/source_table are formal-only, G2 stays probe-only, and task5/8/7 do not leak task2 ladder roles."
    return result


def _check_step3_config_artifact_split(repo_root: Path) -> RuleResult:
    result = RuleResult("R103", "Step3 child runtime config must not overwrite parent resolved_config")
    findings: list[Finding] = []
    step3_core = _read(repo_root / "code" / "executors" / "step3_train_core.py")
    manifests = _read(repo_root / "code" / "odcr_core" / "manifests.py")
    checkpoint = _read(repo_root / "code" / "odcr_core" / "training_checkpoint.py")
    if 'resolved_config.json"' in step3_core and "write_training_runtime_config_artifact" not in step3_core:
        findings.append(Finding("code/executors/step3_train_core.py", 1, "resolved_config.json", "Child runtime snapshots must use training_runtime_config.json."))
    for rel, text, terms in (
        ("code/odcr_core/manifests.py", manifests, ("TRAINING_RUNTIME_CONFIG_FILENAME", "write_training_runtime_config_artifact", "training_runtime_config_path")),
        ("code/odcr_core/training_checkpoint.py", checkpoint, ("current_training_runtime_config_lineage", "training_runtime_config_hash", "training_runtime_config_path")),
    ):
        missing = [term for term in terms if term not in text]
        if missing:
            findings.append(Finding(rel, 1, "missing: " + ", ".join(missing), "Config artifact split must be lineage-visible."))
    if findings:
        result.fail("Step3 child/parent config artifact split is incomplete.", findings)
    else:
        result.summary = "Parent resolved_config.json and child training_runtime_config.json are split and lineage-indexed."
    return result


def _check_step3_structured_metric_files_active(repo_root: Path) -> RuleResult:
    result = RuleResult("R104", "Step3 structured metrics files must be active")
    findings: list[Finding] = []
    path_layout = _read(repo_root / "code" / "odcr_core" / "path_layout.py")
    train_logging = _read(repo_root / "code" / "train_logging.py")
    step3_core = _read(repo_root / "code" / "executors" / "step3_train_core.py")
    required = {
        "code/odcr_core/path_layout.py": (path_layout, ("timing_profile", "loss_breakdown", "gpu_profile", "epoch_summary")),
        "code/train_logging.py": (
            train_logging,
            (
                "append_step3_loss_breakdown_jsonl",
                "append_step3_timing_profile_jsonl",
                "append_step3_gpu_profile_jsonl",
                "append_step3_epoch_summary_csv",
            ),
        ),
        "code/executors/step3_train_core.py": (
            step3_core,
            (
                "_step3_loss_breakdown_row",
                "_step3_timing_profile_row",
                "_step3_gpu_profile_row",
                "append_step3_epoch_summary_csv",
            ),
        ),
    }
    for rel, (text, terms) in required.items():
        missing = [term for term in terms if term not in text]
        if missing:
            findings.append(Finding(rel, 1, "missing: " + ", ".join(missing), "Step3 structured metrics must not regress to full.log parsing."))
    if findings:
        result.fail("Step3 structured metric files are not fully active.", findings)
    else:
        result.summary = "metrics/loss_breakdown/timing_profile/gpu_profile/epoch_summary writers are active."
    return result


def _check_step3_run_summary_log_indexes(repo_root: Path) -> RuleResult:
    result = RuleResult("R105", "run_summary must index authoritative full/debug/errors/config artifacts")
    manifests = _read(repo_root / "code" / "odcr_core" / "manifests.py")
    terms = (
        "authoritative_full_log_path",
        "debug_log_path",
        "training_runtime_config_path",
        "optional_missing_with_reason",
        "failure_root_signature",
        "failure_phase",
        "fatal_signature",
        "training_loop_started",
        "checkpoint_created",
        "resolved_config_path",
        "loss_breakdown",
        "timing_profile",
        "gpu_profile",
        "epoch_summary",
    )
    missing = [term for term in terms if term not in manifests]
    if missing:
        result.fail(
            "run_summary artifact indexes are incomplete.",
            [Finding("code/odcr_core/manifests.py", 1, "missing: " + ", ".join(missing), "Index formal log/config/metric artifacts in run_summary.")],
        )
    else:
        result.summary = "run_summary indexes authoritative full.log, debug.log, config split, and Step3 structured metric files."
    return result


def _check_step3_formal_source_table_static(repo_root: Path) -> RuleResult:
    result = RuleResult("R106", "Step3 source_table default must be formal-only")
    findings: list[Finding] = []
    manifests = _read(repo_root / "code" / "odcr_core" / "manifests.py")
    resolver = _read(repo_root / "code" / "odcr_core" / "config_resolver.py")
    odcr_py = _read(repo_root / "code" / "odcr.py")
    terms = (
        "_STEP3_FORMAL_SOURCE_EXCLUDE_PARTS",
        "build_formal_source_table_snapshot",
        "formal_only_source_table",
    )
    combined = "\n".join((manifests, resolver, odcr_py))
    for term in terms:
        if term not in combined:
            findings.append(Finding("code/odcr_core/manifests.py", 1, term, "Formal source_table must stay the default for Step3."))
    for term in ("backup", "exploration", "step5"):
        if term not in manifests:
            findings.append(Finding("code/odcr_core/manifests.py", 1, term, "Formal source_table exclusion list is missing a non-formal surface."))
    if findings:
        result.fail("Step3 formal source_table closure is incomplete.", findings)
    else:
        result.summary = "Step3 source_table defaults to formal-only with verbose/history rows excluded."
    return result


def _check_step3_non_task2_roles_and_g2_dynamic(repo_root: Path) -> RuleResult:
    result = RuleResult("R107", "Non-task2 Step3 views must not leak task2 ladder role and G2 stays non-formal")
    findings: list[Finding] = []
    code_dir = repo_root / "code"
    inserted = False
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
        inserted = True
    try:
        from odcr_core.config_resolver import resolve_config

        for task_id in (5, 8, 7):
            _cfg, _sources, snap = resolve_config(
                config_path=repo_root / "configs" / "odcr.yaml",
                command="step3",
                task_id=task_id,
                set_overrides=[],
                dry_run=True,
                run_id="auto",
                mode="full",
            )
            role = str((snap.get("train") or {}).get("step3_batch_candidate_role") or "")
            profile_id = str((snap.get("task") or {}).get("task_profile_id") or "")
            if "task2_" in role or (profile_id and profile_id not in role):
                findings.append(Finding("code/odcr_core/config_resolver.py", 1, f"task{task_id}: {role}", "Batch candidate role must be task-profile isolated."))
        _cfg2, _sources2, snap2 = resolve_config(
            config_path=repo_root / "configs" / "odcr.yaml",
            command="step3",
            task_id=2,
            set_overrides=[],
            dry_run=True,
            run_id="auto",
            mode="full",
        )
        g2 = snap2["step3_exploration_profiles"]["task2_g2_effective_pool_2048"]
        if not bool(g2.get("probe_only")) or bool(g2.get("formal_allowed")):
            findings.append(Finding("configs/odcr.yaml", 1, repr(g2), "G2 must stay probe_only=true/formal_allowed=false."))
    except Exception as exc:
        findings.append(Finding("code/odcr_core/config_resolver.py", 1, repr(exc), "Role/G2 guardrail must remain import-safe."))
    finally:
        if inserted:
            try:
                sys.path.remove(str(code_dir))
            except ValueError:
                pass
    if findings:
        result.fail("Step3 task-profile role isolation or G2 non-formal status regressed.", findings)
    else:
        result.summary = "task5/8/7 roles are profile-local and G2 remains probe_only/formal disallowed."
    return result


def _check_step3_no_old_param_regressions_static(repo_root: Path) -> RuleResult:
    result = RuleResult("R108", "Step3 old params must be absent from active clean baseline")
    findings: list[Finding] = []
    cfg = _read(repo_root / "configs" / "odcr.yaml")
    resolver = _read(repo_root / "code" / "odcr_core" / "config_resolver.py")
    schema = _read(repo_root / "code" / "odcr_core" / "config_schema.py")
    step3_core = _read(repo_root / "code" / "executors" / "step3_train_core.py")
    required = (
        "_validate_config_shape",
        "_reject_unknown_keys",
        "tokenizer_schema_version must be odcr_step3_tokenizer_cache/2",
    )
    for term in required:
        if term not in resolver:
            findings.append(Finding("code/odcr_core/config_resolver.py", 1, term, "Step3 unknown keys must be rejected by generic strict schema."))
    active_forbidden = (
        "step3:\n  train:\n    grad_accum:",
        "step3:\n  train:\n    coef:",
        "optimizer: adam\n",
        "train_precision: fp32",
        "max_length: 25",
        "max_evidence_length: 24",
        "performance_ladder:",
        "smoke_ladder:",
        "performance_probe:",
        "short_pilot:",
    )
    for term in active_forbidden:
        if term in cfg:
            findings.append(Finding("configs/odcr.yaml", 1, term, "Do not reintroduce retired Step3 formal defaults."))
    active_code_forbidden = (
        "_STEP3_RETIRED_CONTROL_FIELDS",
        "_STEP3_REMOVED_GRAD_ACCUM_FIELDS",
        "STEP3_GRAD_ACCUM_REMOVED_MESSAGE",
        "Step3 child --coef is retired",
        "_resolve_step3_ladder_config",
        "_resolve_step3_performance_probe_config",
        "_resolve_step3_short_pilot_config",
        "performance_ladder_config_json",
        "smoke_ladder_config_json",
        "performance_probe_config_json",
        "short_pilot_config_json",
    )
    for term in active_code_forbidden:
        if term in resolver or term in schema or term in step3_core:
            findings.append(Finding("code", 1, term, "Step3 clean baseline must not keep old active parser/resolver/schema branches."))
    if ".no_sync(" in step3_core or "_ddp_no_sync_model" in step3_core:
        findings.append(Finding("code/executors/step3_train_core.py", 1, "no_sync", "Step3 no-accum path must not restore accumulation."))
    if findings:
        result.fail("Old Step3 parameters or no-accum bypasses are active again.", findings)
    else:
        result.summary = "Step3 old parser/resolver/schema fields are absent; generic strict schema owns unknown-key rejection."
    return result


def _check_step3_quality_evidence_performance_rebuild(repo_root: Path) -> RuleResult:
    result = RuleResult("R109", "Step3 quality/evidence/performance rebuild contracts must stay active")
    findings: list[Finding] = []
    cfg = _read(repo_root / "configs" / "odcr.yaml")
    resolver = _read(repo_root / "code" / "odcr_core" / "config_resolver.py")
    schema = _read(repo_root / "code" / "odcr_core" / "config_schema.py")
    config_py = _read(repo_root / "code" / "config.py")
    core = _read(repo_root / "code" / "executors" / "step3_train_core.py")
    quality = _read(repo_root / "code" / "odcr_core" / "step3_quality.py")
    checkpoint = _read(repo_root / "code" / "odcr_core" / "training_checkpoint.py")
    bridge = _read(repo_root / "code" / "tools" / "odcr_tmux_gpu_bridge.py")
    post_edit = _read(repo_root / "code" / "tools" / "odcr_post_edit_check.py")
    docs = _read(repo_root / "docs" / "ODCR_STEP3_QUALITY_EVIDENCE_PERFORMANCE_CONTRACT.md")
    tests = _read(repo_root / "code" / "tests" / "test_step3_quality_evidence_performance_rebuild.py")
    required = {
        "configs/odcr.yaml": (
            "checkpoint_policy:",
            "quality_gate:",
            "grad_finite:",
            "diagnostic_eval:",
            "performance_candidates:",
            "validation_aware_lr_damping:",
        ),
        "code/odcr_core/config_resolver.py": (
            "_resolve_step3_checkpoint_policy_config",
            "_resolve_step3_quality_gate_config",
            "_resolve_step3_grad_finite_config",
            "_resolve_step3_diagnostic_eval_config",
            "_resolve_step3_performance_candidates_config",
        ),
        "code/odcr_core/stage_status.py": (
            "do_not_use_quality_audit_as_final_truth",
            "quality_audit.json.superseded_by.json",
            "completed_with_eval_handoff",
        ),
        "code/odcr_core/config_schema.py": (
            "checkpoint_policy_config_json",
            "quality_gate_config_json",
            "grad_finite_config_json",
            "diagnostic_eval_config_json",
            "performance_candidates_config_json",
        ),
        "code/config.py": (
            "checkpoint_policy_config_json",
            "quality_gate_config_json",
            "grad_finite_config_json",
            "diagnostic_eval_config_json",
            "performance_candidates_config_json",
        ),
        "code/odcr_core/step3_quality.py": (
            "TIMING_REQUIRED_FIELDS",
            "MEMORY_REQUIRED_FIELDS",
            "PREFETCH_EVIDENCE_FIELDS",
            "validate_step3_downstream_quality_gate",
            "Evidence Level",
        ),
        "code/executors/step3_train_core.py": (
            "best_observed.pth",
            "best_after_min_epochs.pth",
            "latest.pth",
            "grad_check_ms",
            "optimizer_step_executed",
            "structured_gather_total_bytes",
            "collapse_stats_from_predictions",
            "Step3CUDAPrefetcher",
        ),
        "code/odcr_core/training_checkpoint.py": (
            "checkpoint_epoch",
            "selection_scope",
            "CHECKPOINT_EVENT_LEDGER_SCHEMA_VERSION",
            "never_silently_overwrite_global_best",
        ),
        "code/tools/odcr_tmux_gpu_bridge.py": (
            "step3-performance-probe",
            "STEP3_PERFORMANCE_PROBE_TYPES",
            "build_step3_performance_probe_script",
        ),
        "code/tools/odcr_post_edit_check.py": (
            "test_step3_quality_evidence_performance_rebuild.py",
        ),
        "docs/ODCR_STEP3_QUALITY_EVIDENCE_PERFORMANCE_CONTRACT.md": (
            "Evidence Level 1",
            "run1",
            "checkpoint policy",
            "performance-probe bridge",
        ),
        "code/tests/test_step3_quality_evidence_performance_rebuild.py": (
            "this test proves",
            "this test does not prove",
            "whether formal hot path is covered",
            "whether runtime evidence is required",
            "regression bug it prevents",
        ),
    }
    texts = {
        "configs/odcr.yaml": cfg,
        "code/odcr_core/config_resolver.py": resolver,
        "code/odcr_core/stage_status.py": _read(repo_root / "code" / "odcr_core" / "stage_status.py"),
        "code/odcr_core/config_schema.py": schema,
        "code/config.py": config_py,
        "code/odcr_core/step3_quality.py": quality,
        "code/executors/step3_train_core.py": core,
        "code/odcr_core/training_checkpoint.py": checkpoint,
        "code/tools/odcr_tmux_gpu_bridge.py": bridge,
        "code/tools/odcr_post_edit_check.py": post_edit,
        "docs/ODCR_STEP3_QUALITY_EVIDENCE_PERFORMANCE_CONTRACT.md": docs,
        "code/tests/test_step3_quality_evidence_performance_rebuild.py": tests,
    }
    for rel, terms in required.items():
        missing = [term for term in terms if term not in texts[rel]]
        if missing:
            findings.append(Finding(rel, 1, "missing: " + ", ".join(missing), "Step3 quality/evidence/performance rebuild contract regressed."))
    for forbidden in (
        "current_valid_loss <= prev_valid_loss",
        "current_valid_loss < prev_valid_loss",
    ):
        if forbidden in core:
            findings.append(Finding("code/executors/step3_train_core.py", 1, forbidden, "Checkpoint best selection must compare against global best, not previous epoch."))
    if findings:
        result.fail("Step3 quality/evidence/performance rebuild guardrail failed.", findings)
    else:
        result.summary = "Step3 global-best checkpoints, quality gate, grad finite gate, timing/memory/prefetch evidence, diagnostics, and bridge probe contracts are active."
    return result


def _check_stage_truth_upstream_resolver(repo_root: Path) -> RuleResult:
    result = RuleResult("R112", "stage truth and unified upstream resolver")
    required_files = (
        "code/odcr_core/stage_status.py",
        "code/odcr_core/upstream_resolver.py",
        "code/odcr_core/stage_promotion.py",
        "docs/CURRENT_PROJECT_STATE.md",
    )
    findings: list[Finding] = []
    for rel in required_files:
        if not (repo_root / rel).is_file():
            findings.append(Finding(rel, 1, "missing", "Add the stage truth/upstream resolver surface."))
    text_by_file: dict[str, str] = {}
    for rel in required_files:
        path = repo_root / rel
        if path.is_file():
            text_by_file[rel] = _read(path)
    required_terms = {
        "code/odcr_core/stage_status.py": (
            "STAGE_STATUS_SCHEMA_VERSION",
            "quality_audit.json.superseded_by.json",
            "do_not_use_quality_audit_as_final_truth",
        ),
        "code/odcr_core/upstream_resolver.py": (
            "resolve_upstream",
            "validate_upstream_eligibility",
            "validate_stage_status_evidence",
            "non_latest_eligible_run_requires_promote",
        ),
        "code/odcr_core/stage_promotion.py": (
            "promote_upstream",
            "historical_stage_status_immutable",
            "promotion_history.jsonl",
        ),
        "docs/CURRENT_PROJECT_STATE.md": (
            "stage_status.json",
            "latest.json",
            "Historical docs and AI_analysis are not live truth",
            "machine_verdict.json",
        ),
    }
    for rel, terms in required_terms.items():
        text = text_by_file.get(rel, "")
        for term in terms:
            if term not in text:
                findings.append(Finding(rel, 1, f"missing term: {term}", "Keep the stage truth contract explicit."))
    resolver = _read(repo_root / "code" / "odcr_core" / "config_resolver.py")
    for term in ("resolve_upstream", "upstream_resolution_json", "from_step3"):
        if term not in resolver:
            findings.append(Finding("code/odcr_core/config_resolver.py", 1, f"missing term: {term}", "Route Step4/Step5 through upstream_resolver."))
    odcr = _read(repo_root / "code" / "odcr.py")
    for term in ("--from-step3-run", "promote-upstream"):
        if term not in odcr:
            findings.append(Finding("code/odcr.py", 1, f"missing term: {term}", "Expose the governed upstream run selector/promoter."))
    active_doc_re = re.compile(
        r"(run2|run 2|run_id\s*=?\s*2).{0,160}(failed|downstream_ready\s*=\s*false|must not be .*downstream|blocked)",
        re.IGNORECASE,
    )
    docs_dir = repo_root / "docs"
    for path in sorted(docs_dir.glob("*.md")) if docs_dir.is_dir() else []:
        rel = _rel(path, repo_root)
        if rel == "docs/CURRENT_PROJECT_STATE.md" or "/history/" in rel:
            continue
        text = _read(path)
        first_lines = "\n".join(text.splitlines()[:6])
        if "SUPERSEDED / HISTORICAL ONLY" in first_lines:
            continue
        for idx, line in enumerate(text.splitlines(), start=1):
            if active_doc_re.search(line):
                findings.append(
                    Finding(
                        rel,
                        idx,
                        line.strip()[:240],
                        "Mark historical-only or route active state through docs/CURRENT_PROJECT_STATE.md.",
                    )
                )
    if findings:
        result.fail("Stage truth/upstream resolver guardrail failed.", findings)
    else:
        result.summary = "Stage status, active latest pointers, promotion, resolver, and active-doc truth boundaries are present."
    return result


def _check_stage_status_strict_antiforgery_guardrail(repo_root: Path) -> RuleResult:
    result = RuleResult("R113", "stage_status strict anti-forgery guardrail")
    findings: list[Finding] = []
    code_dir = str(repo_root / "code")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    try:
        from odcr_core.stage_truth_antiforgery import run_antiforgery_selftest

        payload = run_antiforgery_selftest()
    except Exception as exc:
        result.fail(
            "R113 selftest could not execute.",
            [Finding("code/odcr_core/stage_truth_antiforgery.py", 1, repr(exc), "Keep R113 import-safe and lightweight.")],
        )
        return result
    required = {
        "forged_status_rejected": "forged/minimal status must be rejected",
        "missing_artifact_rejected": "missing artifact must be rejected",
        "hash_mismatch_rejected": "hash mismatch must be rejected",
        "stale_exists_rejected": "stale exists=true must be rejected from disk",
        "promotion_malformed_target_rejected": "promotion dry-run must reject malformed target",
        "alias_run1_rejected": "manual alias must use the same strict resolver",
        "latest_pointer_only_passed": "deprecated latest_status must not be final truth",
    }
    for key, message in required.items():
        if payload.get(key) is not True:
            findings.append(
                Finding(
                    "code/odcr_core/stage_truth_antiforgery.py",
                    1,
                    f"{key}={payload.get(key)!r}",
                    message,
                )
            )
    if findings:
        result.fail("Stage status anti-forgery negative selftest failed.", findings)
    else:
        result.summary = "R113 negative cases reject forged status, missing artifacts, hash mismatch, stale exists, malformed promotion, and alias bypass."
    return result


def _check_step4_runtime_preflight_twophase_guardrail(repo_root: Path) -> RuleResult:
    result = RuleResult("R114", "Step4 pre-DDP cache, preflight, two-phase export, and frozen lineage")
    findings: list[Finding] = []
    required_files = {
        "code/odcr.py": (
            "--prepare-cache",
            "--preflight",
            "--validation-namespace",
        ),
        "code/odcr_core/step4_runtime.py": (
            "prepare_step4_encoded_cache",
            "run_step4_bounded_preflight",
            "reject_step4_formal_env_overrides",
            "runs\" / \"step4_preflight\"",
            "formal_latest_write",
            "formal_export_write",
        ),
        "code/executors/step4_engine.py": (
            "pre_ddp_cache_load_start",
            "cold_build_allowed=False",
            "destroy_process_group_before_cpu_export=True",
            "non_rank0_gpu_released_before_cpu_tail=True",
            "_step4_wait_for_partial_manifests",
            "partial_artifacts_retained_for_readiness_validator=True",
        ),
        "code/odcr_core/step4_export_validator.py": (
            "validate_step4_export_ready",
            "index_contract",
            "frozen Step3 lineage",
            "step5_required_fields_precheck",
        ),
        "code/odcr_core/index_contract.py": (
            "frozen_step3_lineage",
            "Step5 refused Step4 export: missing frozen Step3 lineage",
        ),
        "configs/odcr.yaml": (
            "step4:",
            "runtime:",
            "partial_wait_timeout_seconds",
        ),
    }
    for rel, terms in required_files.items():
        path = repo_root / rel
        if not path.is_file():
            findings.append(Finding(rel, 1, "missing", "Required Step4 runtime/preflight surface is absent."))
            continue
        text = _read(path)
        for term in terms:
            if term not in text:
                findings.append(Finding(rel, 1, f"missing term: {term}", "Keep the Step4 runtime rebuild contract active."))
    engine = _read(repo_root / "code" / "executors" / "step4_engine.py")
    cold_pattern = re.compile(r"target_dataset\.map\(.{0,240}dist\.barrier\(", re.DOTALL)
    if cold_pattern.search(engine):
        findings.append(
            Finding(
                "code/executors/step4_engine.py",
                1,
                "target_dataset.map before dist.barrier",
                "Cold tokenization/cache must not occur inside an active DDP/NCCL section.",
            )
        )
    if "dist.gather_object" in engine or "all_gather_object" in engine:
        findings.append(
            Finding(
                "code/executors/step4_engine.py",
                1,
                "distributed object gather present",
                "Step4 export must use partial manifests, not text/object gather payloads.",
            )
        )
    if "while not os.path.exists" in engine:
        findings.append(
            Finding(
                "code/executors/step4_engine.py",
                1,
                "endless file polling present",
                "Step4 file waiting must have timeout and failed-marker checks.",
            )
        )
    if findings:
        result.fail("Step4 runtime/preflight/two-phase export guardrail failed.", findings)
    else:
        result.summary = "Step4 cache is pre-DDP, preflight is non-formal, CPU export follows PG destroy, readiness is strict, and frozen lineage is required."
    return result


def _check_gpu_bridge_step4_bounded_preflight_admission(repo_root: Path) -> RuleResult:
    result = RuleResult("R115", "GPU bridge Step4 bounded preflight admission")
    findings: list[Finding] = []
    bridge_path = repo_root / "code" / "tools" / "odcr_tmux_gpu_bridge.py"
    bridge_text = _read(bridge_path)
    required_terms = (
        "BridgeCommandPolicy",
        "classify_repo_command",
        "is_allowed_step4_validation_command",
        "STEP4_BRIDGE_MAX_BOUNDED_LIMIT",
        "runs\" / \"step4_preflight\"",
        "runs\" / \"step4_validation\"",
        "snapshot_formal_namespace",
        "formal_namespace_polluted",
        "parse_step4_preflight_evidence",
        "runtime_evidence_split_present",
        "command_allowed_by_policy",
        "formal_pollution_check_passed",
    )
    for term in required_terms:
        if term not in bridge_text:
            findings.append(Finding("code/tools/odcr_tmux_gpu_bridge.py", 1, f"missing term: {term}", "Keep Step4 bridge admission policy explicit and testable."))
    retired_blankets = (
        '("step4", "step4")',
        'generated bridge script contains forbidden token: step4',
        'Step4 launch from validation',
        r"(?:\./odcr|python\s+code/odcr\.py)\s+step4\b\", \"step4",
    )
    for term in retired_blankets:
        if term in bridge_text:
            findings.append(Finding("code/tools/odcr_tmux_gpu_bridge.py", 1, term, "Blanket Step4 forbidden logic must stay retired; command semantics decide."))
    if "MODE_SPECS" in bridge_text or "closed-choice whitelist" in bridge_text:
        findings.append(Finding("code/tools/odcr_tmux_gpu_bridge.py", 1, "MODE_SPECS/closed-choice whitelist", "Old whitelist or closed-choice hard blocker must not return."))
    try:
        bridge = _load_bridge_module(repo_root)
        safe_preflight = (
            "./odcr",
            "step4",
            "--task",
            "2",
            "--preflight",
            "--max-samples",
            "128",
            "--validation-namespace",
            "step4_preflight_smoke",
        )
        safe_prepare = (
            "./odcr",
            "step4",
            "--task",
            "2",
            "--prepare-cache",
            "--max-samples",
            "128",
            "--validation-namespace",
            "step4_preflight_smoke",
        )
        positive = {
            "safe Step4 preflight": safe_preflight,
            "safe Step4 prepare-cache": safe_prepare,
        }
        for label, command in positive.items():
            classification = bridge.BridgeCommandPolicy.classify_repo_command(command)
            if not classification.allowed:
                findings.append(Finding("code/tools/odcr_tmux_gpu_bridge.py", 1, label, f"Expected allow, got: {classification.reason}"))
        negative = {
            "formal Step4": ("./odcr", "step4", "--task", "2"),
            "Step4 without bounded limit": ("./odcr", "step4", "--task", "2", "--preflight", "--validation-namespace", "step4_preflight_smoke"),
            "Step4 without namespace": ("./odcr", "step4", "--task", "2", "--preflight", "--max-samples", "128"),
            "bad namespace": ("./odcr", "step4", "--task", "2", "--preflight", "--max-samples", "128", "--validation-namespace", "../bad"),
            "formal output": (
                "./odcr",
                "step4",
                "--task",
                "2",
                "--preflight",
                "--max-samples",
                "128",
                "--validation-namespace",
                "step4_preflight_smoke",
                "--output",
                "runs/step4/task2/latest.json",
            ),
            "Step5": ("./odcr", "step5", "--task", "2"),
            "eval": ("./odcr", "eval", "--task", "2"),
            "rerank": ("./odcr", "rerank", "--task", "2"),
            "nohup": ("nohup", "./odcr", "step4"),
            "allocation": ("srun", "--pty", "bash"),
            "destructive": ("rm", "-rf", "runs/step4/task2"),
        }
        for label, command in negative.items():
            classification = bridge.BridgeCommandPolicy.classify_repo_command(command)
            if classification.allowed:
                findings.append(Finding("code/tools/odcr_tmux_gpu_bridge.py", 1, label, "Expected rejection, got allowed."))
        out_dir = bridge.resolve_runtime_output_dir("runs/step4_preflight/task2/step4_preflight_smoke", "guardrail_r115")
        if "runs/step4_preflight/task2/step4_preflight_smoke" not in str(out_dir):
            findings.append(Finding("code/tools/odcr_tmux_gpu_bridge.py", 1, str(out_dir), "Step4 preflight output root must be allowed."))
        evidence = bridge.parse_step4_preflight_evidence(output_dir=repo_root / "AI_analysis" / "nonexistent_r115")
        if evidence.get("runtime_evidence_ok") is not False or evidence.get("evidence_complete") is not False:
            findings.append(Finding("code/tools/odcr_tmux_gpu_bridge.py", 1, "cuda/probe-only evidence", "Missing Step4 artifacts must not pass runtime evidence."))
    except Exception as exc:
        findings.append(Finding("code/tools/odcr_tmux_gpu_bridge.py", 1, repr(exc), "R115 dynamic policy checks must import and execute."))
    test_text = _read(repo_root / "code" / "tests" / "test_gpu_bridge_step4_preflight_admission.py")
    for term in ("SAFE_PREFLIGHT", "test_step4_prepare_cache_repo_command_allowed", "test_cuda_probe_only_is_not_step4_runtime_evidence", "test_formal_pollution_snapshot_change_fails"):
        if term not in test_text:
            findings.append(Finding("code/tests/test_gpu_bridge_step4_preflight_admission.py", 1, f"missing term: {term}", "Step4 bridge policy tests must cover positive, negative, evidence, and pollution cases."))
    if findings:
        result.fail("GPU bridge Step4 bounded preflight admission guardrail failed.", findings)
    else:
        result.summary = "Bridge allows bounded Step4 preflight/prepare-cache validation, rejects formal/destructive/allocation/arbitrary paths, watches formal namespaces, and parses Step4 evidence."
    return result


def _check_step4_evidence_level_antifake_tuning_guardrail(repo_root: Path) -> RuleResult:
    result = RuleResult("R116", "Step4 evidence-level anti-fake-tuning guardrail")
    findings: list[Finding] = []
    required_terms = {
        "code/odcr_core/evidence_level.py": (
            "E0_static_config",
            "E1_schema_preview",
            "E2_cpu_real_data_no_model",
            "E3_gpu_transport",
            "E4_gpu_shard_forward_bounded",
            "E5_formal_full_run",
            "require_min_evidence_level",
            "assert_not_schema_only_for_tuning",
            "mark_schema_preview",
            "mark_gpu_shard_forward",
        ),
        "code/odcr_core/step4_runtime.py": (
            "mark_schema_preview",
            "cpu_preview_proxy_fields",
            "not_step4_runtime_evidence",
        ),
        "code/odcr_core/step4_gpu_preflight_runner.py": (
            "mark_gpu_shard_forward",
            "actual_gpu_forward_executed",
            "actual_model_loaded_on_gpu",
            "force_gpu_forward",
        ),
        "code/odcr_core/step4_tuning_evidence.py": (
            "rank_step4_candidates",
            "build_best_candidate_record",
            "build_patch_suggestion_text",
            "E4_GPU_SHARD_FORWARD_BOUNDED",
        ),
        "code/odcr_core/step4_evidence_machine_verdict.py": (
            "evidence_level_min_required_for_a",
            "candidate_ranking_evidence_level",
            "schema_only_evidence_used_for_tuning",
            "fake_score_used_for_tuning",
            "candidate_source_is_cpu_preview",
        ),
        "code/tools/odcr_tmux_gpu_bridge.py": (
            "_step4_runtime_evidence_ok",
            "actual_gpu_forward_executed",
            "actual_model_loaded_on_gpu",
            "force_gpu_forward",
            "cuda_probe_alone_is_not_step4_runtime_evidence",
            "E3_GPU_TRANSPORT",
            "E4_GPU_SHARD_FORWARD_BOUNDED",
        ),
        "AI_analysis/06_step4_tuning_c9_neighborhood/run_c9_neighborhood_validation.py": (
            "historical-invalid",
            "--allow-schema-preview-report-only",
            "best_candidate_written",
            "patch_suggestion_written",
            "candidate_ranking_requires_e4_or_e5",
        ),
    }
    for rel, terms in required_terms.items():
        path = repo_root / rel
        if not path.is_file():
            findings.append(Finding(rel, 1, "missing", "Required Step4 evidence-level surface is absent."))
            continue
        text = _read(path)
        for term in terms:
            if term not in text:
                findings.append(Finding(rel, 1, f"missing term: {term}", "Keep Step4 evidence-level anti-fake tuning hard gates active."))
    docs_terms = (
        "E1_schema_preview",
        "CPU preview",
        "not tuning evidence",
        "E4_gpu_shard_forward_bounded",
        "formal Step4 remains blocked",
        "C9_bucket_balanced",
    )
    for rel in (
        "docs/CURRENT_PROJECT_STATE.md",
        "docs/ODCR_ACTIVE_ARCHITECTURE.md",
        "docs/AI_PROJECT_CANONICAL.md",
        "docs/ODCR_ARCHITECTURE_CONTRACT.md",
        "AGENTS.md",
    ):
        text = _read(repo_root / rel)
        for term in docs_terms:
            if term not in text:
                findings.append(Finding(rel, 1, f"missing term: {term}", "Active docs must warn CPU preview is E1/schema-only and cannot tune Step4."))
    superseded_targets = (
        "AI_analysis/06_step4_tuning/best_candidate.yaml",
        "AI_analysis/06_step4_tuning/best_candidate_patch_suggestion.md",
        "AI_analysis/06_step4_tuning_c9_neighborhood/best_candidate.yaml",
        "AI_analysis/06_step4_tuning_c9_neighborhood/best_candidate_patch_suggestion.md",
        "AI_analysis/05_final_reports/step4_rcr_param_tuning_bounded_gpu_machine_verdict.json",
        "AI_analysis/05_final_reports/step4_c9_neighborhood_throughput_validation_machine_verdict.json",
    )
    for rel in superseded_targets:
        sidecar = repo_root / f"{rel}.superseded_by_real_gpu_evidence.json"
        if not sidecar.is_file():
            findings.append(Finding(f"{rel}.superseded_by_real_gpu_evidence.json", 1, "missing", "Old CPU-preview best/verdict artifacts must have superseded sidecars."))
            continue
        payload = _json_load(sidecar)
        if payload.get("superseded") is not True or payload.get("not_valid_for_formal_tuning") is not True:
            findings.append(Finding(str(sidecar.relative_to(repo_root)), 1, "invalid superseded payload", "Sidecar must fail closed for formal tuning/prompt use."))
    try:
        code_dir = str(repo_root / "code")
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        from odcr_core.evidence_level import mark_gpu_shard_forward, mark_schema_preview
        from odcr_core.step4_evidence_machine_verdict import build_step4_evidence_machine_verdict
        from odcr_core.step4_tuning_evidence import rank_step4_candidates

        cpu_preview = mark_schema_preview({"candidate_id": "cpu", "score": 1.0})
        try:
            rank_step4_candidates([cpu_preview])
            findings.append(Finding("code/odcr_core/step4_tuning_evidence.py", 1, "CPU preview ranked", "E1 schema preview must raise before ranking."))
        except Exception:
            pass
        verdict = build_step4_evidence_machine_verdict(
            {
                "candidate_ranking_evidence_level": "E1_schema_preview",
                "schema_only_evidence_used_for_tuning": True,
                "proxy_score_present": True,
                "fake_score_used_for_tuning": True,
                "guardrail_r116_status": "passed",
            }
        )
        if verdict.get("verdict") == "A":
            findings.append(Finding("code/odcr_core/step4_evidence_machine_verdict.py", 1, "E1 verdict A", "Machine verdict A must require E4/E5."))
        e4 = mark_gpu_shard_forward({"candidate_id": "g1", "score": 1.0})
        if not rank_step4_candidates([e4]):
            findings.append(Finding("code/odcr_core/step4_tuning_evidence.py", 1, "E4 rejected", "E4 gpu-shard evidence must remain rankable for bounded tuning."))
        bridge = _load_bridge_module(repo_root)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def w(name: str, payload: dict[str, object]) -> None:
                (root / name).write_text(json.dumps(payload), encoding="utf-8")

            base_summary = mark_schema_preview(
                {
                    "validation_namespace": "guardrail_r116",
                    "sample_count": 4,
                    "max_samples": 8,
                    "formal_latest_write": False,
                    "formal_export_write": False,
                    "upstream_step3_run_id": "2",
                }
            )
            w("preflight_summary.json", base_summary)
            w(
                "rcr_distribution.json",
                mark_schema_preview(
                    {
                        "sample_count": 4,
                        "route_scorer_count": 2,
                        "route_explainer_count": 2,
                        "train_keep_count": 2,
                        "confidence_bucket_distribution": {"1": 4},
                        "sample_weight_hint": {},
                    }
                ),
            )
            w("required_fields_check.json", mark_schema_preview({"passed": True, "missing": []}))
            w("manifest_preview.json", mark_schema_preview({"schema_version": "x"}))
            w("index_contract_preview.json", mark_schema_preview({"schema_version": "x"}))
            w("lineage_preview.json", {"lineage_hash": "abc"})
            w("cpu_gpu_utilization_snapshot.json", mark_schema_preview({"cuda_available": True, "gpu_runtime_evidence": False}))
            evidence = bridge.parse_step4_preflight_evidence(output_dir=root)
            if evidence.get("runtime_evidence_ok") is not False:
                findings.append(Finding("code/tools/odcr_tmux_gpu_bridge.py", 1, "cuda_available accepted", "CUDA transport alone must not be Step4 runtime evidence."))
    except Exception as exc:
        findings.append(Finding("code/tools/check_one_control_guardrails.py", 1, repr(exc), "R116 dynamic checks must import and execute."))
    if findings:
        result.fail("Step4 evidence-level anti-fake-tuning guardrail failed.", findings)
    else:
        result.summary = "Step4 evidence levels gate CPU preview, candidate ranking, best candidates, patch suggestions, verdict A, bridge runtime evidence, docs, and superseded C9 artifacts."
    return result


def _check_no_accum_architecture_guardrail(repo_root: Path) -> RuleResult:
    result = RuleResult("R117", "ODCR no-accum architecture guardrail")
    findings: list[Finding] = []
    removed_fields = {"grad_accum", "gradient_accumulation_steps", "accumulate_grad_batches", "accum_steps", "accumulation_steps"}
    removed_env = {"ODCR_GRAD_ACCUM", "ODCR_GRADIENT_ACCUMULATION_STEPS", "ODCR_ACCUMULATE_GRAD_BATCHES"}
    message_fragment = "grad_accum has been removed in ODCR no-accum architecture"
    try:
        import yaml

        cfg = yaml.safe_load(_read(repo_root / "configs" / "odcr.yaml")) or {}
    except Exception as exc:
        result.fail("R117 could not read configs/odcr.yaml.", [Finding("configs/odcr.yaml", 1, repr(exc), "Config must be YAML-readable.")])
        return result

    def _walk_config(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_s = str(key)
                child_path = f"{path}.{key_s}" if path else key_s
                if key_s in removed_fields:
                    findings.append(Finding("configs/odcr.yaml", 1, child_path, "Retired accumulation field must not be active config."))
                _walk_config(child, child_path)
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                _walk_config(child, f"{path}[{idx}]")

    _walk_config(cfg, "")
    try:
        ddp_world = int((((cfg.get("hardware") or {}).get("profiles") or {}).get("default") or {}).get("ddp_world_size") or 1)
    except Exception:
        ddp_world = 1

    def _check_train_row(row: object, ctx: str, *, world: int | None = None) -> None:
        if not isinstance(row, Mapping):
            findings.append(Finding("configs/odcr.yaml", 1, ctx, "Train row must be a mapping."))
            return
        for key in removed_fields:
            if key in row:
                findings.append(Finding("configs/odcr.yaml", 1, f"{ctx}.{key}", "Retired accumulation field must be absent."))
        if "micro_batch_size" in row:
            findings.append(Finding("configs/odcr.yaml", 1, f"{ctx}.micro_batch_size", "Use per_gpu_batch_size as active source of truth."))
        if "batch_size" in row and "per_gpu_batch_size" in row:
            try:
                g = int(row["batch_size"])
                p = int(row["per_gpu_batch_size"])
                w = int(world or row.get("ddp_world_size") or ddp_world)
                if g != p * w:
                    findings.append(Finding("configs/odcr.yaml", 1, ctx, "No-accum formula failed: global_batch_size = per_gpu_batch_size * ddp_world_size."))
            except Exception as exc:
                findings.append(Finding("configs/odcr.yaml", 1, ctx, f"Batch formula fields must be integers: {exc!r}."))

    for stage in ("step3", "step4", "step5"):
        _check_train_row(((cfg.get(stage) or {}).get("train") if isinstance(cfg.get(stage), Mapping) else None), f"{stage}.train")
    step3 = cfg.get("step3") if isinstance(cfg.get("step3"), Mapping) else {}
    for block_name in ("task_profiles", "backup_profiles", "exploration_profiles"):
        block = step3.get(block_name) if isinstance(step3, Mapping) else {}
        if isinstance(block, Mapping):
            for name, item in block.items():
                row = (item.get("train") if block_name == "task_profiles" and isinstance(item, Mapping) else item)
                world = int((item or {}).get("ddp_world_size") or ddp_world) if isinstance(item, Mapping) else ddp_world
                _check_train_row(row, f"step3.{block_name}.{name}.train" if block_name == "task_profiles" else f"step3.{block_name}.{name}", world=world)
    perf = ((step3.get("performance_candidates") or {}).get("batch_ladder") if isinstance(step3, Mapping) else {})
    if isinstance(perf, Mapping):
        for name, item in perf.items():
            _check_train_row(item, f"step3.performance_candidates.batch_ladder.{name}", world=int((item or {}).get("ddp_world_size") or ddp_world) if isinstance(item, Mapping) else ddp_world)

    schema = _read(repo_root / "code" / "odcr_core" / "config_schema.py")
    resolver = _read(repo_root / "code" / "odcr_core" / "config_resolver.py")
    config_py = _read(repo_root / "code" / "config.py")
    odcr_py = _read(repo_root / "code" / "odcr.py")
    step3_core = _read(repo_root / "code" / "executors" / "step3_train_core.py")
    step5_engine = _read(repo_root / "code" / "executors" / "step5_engine.py")
    manifests = _read(repo_root / "code" / "odcr_core" / "manifests.py")

    active_field_patterns = {
        "code/odcr_core/config_schema.py": (schema, (r"^\s*gradient_accumulation_steps\s*:", r"^\s*grad_accum\s*:")),
        "code/config.py": (config_py, (r"^\s*gradient_accumulation_steps\s*:", r"gradient_accumulation_steps\s*=")),
        "code/executors/step5_engine.py": (step5_engine, (r"gradient_accumulation_steps", r"\binv_accum\b", r"micro_step_count", r"_ddp_no_sync_model", r"\.no_sync\(")),
        "code/executors/step3_train_core.py": (step3_core, (r"_ddp_no_sync_model", r"\.no_sync\(", r"odcr_step3_no_accum/1", r"micro_steps?", r"micro_batches_per_epoch")),
    }
    for rel, (text, patterns) in active_field_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text, flags=re.MULTILINE):
                findings.append(Finding(rel, 1, pattern, "Active code must not keep accumulation or old micro-step semantics."))

    required_terms = {
        "code/odcr_core/config_resolver.py": (resolver, ("_reject_retired_accum_env", "_reject_retired_accum_keys", "per_gpu_batch_size", "odcr_no_accum/1", message_fragment)),
        "code/config.py": (config_py, ("_reject_removed_accumulation_env", "resolve_train_batch_layout", "per_gpu_batch_size", "odcr_no_accum/1", message_fragment)),
        "code/odcr.py": (odcr_py, ("_RetiredAccumulationAction", "--grad-accum", "--gradient-accumulation-steps", "--accumulate-grad-batches", message_fragment)),
        "code/odcr_core/manifests.py": (manifests, ("batch_semantics_version", "grad_accum_removed", "per_gpu_batch_size")),
    }
    for rel, (text, terms) in required_terms.items():
        missing = [term for term in terms if term not in text]
        if missing:
            findings.append(Finding(rel, 1, "missing: " + ", ".join(missing), "No-accum fail-fast and runtime metadata must remain explicit."))

    for env_name in removed_env:
        if env_name not in resolver or env_name not in config_py:
            findings.append(Finding("code/odcr_core/config_resolver.py", 1, env_name, "Retired accumulation env variables must fail fast."))

    active_docs = (
        "AGENTS.md",
        "README.md",
        "docs/ODCR_ARCHITECTURE_CONTRACT.md",
        "docs/ODCR_ACTIVE_ARCHITECTURE.md",
        "docs/AI_PROJECT_CANONICAL.md",
        "docs/ODCR_EVOLUTION_PROTOCOL.md",
        "docs/ODCR_STEP3_CLEAN_BASELINE.md",
        "docs/CURRENT_PROJECT_STATE.md",
    )
    forbidden_doc_re = re.compile(
        r"batch_size\s*==\s*micro_batch_size\s*\*\s*ddp_world_size|"
        r"batch_size\s*==\s*micro_batch_size\s*\*\s*ddp_world_size\s*\*\s*grad_accum|"
        r"Non-Step3.*grad[-_]accum",
        re.IGNORECASE,
    )
    for rel in active_docs:
        path = repo_root / rel
        if not path.exists():
            continue
        text = _read(path)
        match = forbidden_doc_re.search(text)
        if match:
            findings.append(Finding(rel, 1, match.group(0), "Active docs must state ODCR no-accum per-GPU/global formula."))

    tests_text = "\n".join(_read(path) for path in (repo_root / "code" / "tests").glob("test_*.py"))
    for forbidden in ("grad_accum=2", "grad_accum=4", "gradient_accumulation_steps=2", "accumulate_grad_batches=2"):
        if forbidden in tests_text:
            findings.append(Finding("code/tests", 1, forbidden, "Tests must not expect active accumulation candidates."))

    code_dir = repo_root / "code"
    inserted = False
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
        inserted = True
    old_env = {name: os.environ.get(name) for name in removed_env}
    try:
        for name in removed_env:
            os.environ.pop(name, None)
        from odcr_core.config_resolver import resolve_config

        _cfg, _sources, snapshot = resolve_config(
            config_path=repo_root / "configs" / "odcr.yaml",
            command="step3",
            task_id=2,
            set_overrides=[],
            dry_run=True,
            run_id="1",
            mode="full",
        )
        train = snapshot.get("train") or {}
        for key in removed_fields:
            if key in train:
                findings.append(Finding("code/odcr_core/config_resolver.py", 1, key, "Resolved train snapshot must not expose retired accumulation fields."))
        if (
            train.get("batch_semantics_version") != "odcr_no_accum/1"
            or train.get("batch_formula") != "global_batch_size = per_gpu_batch_size * ddp_world_size"
            or train.get("grad_accum_removed") is not True
            or int(train.get("global_batch_size") or train.get("batch_size") or 0)
            != int(train.get("per_gpu_batch_size") or 0) * int(train.get("ddp_world_size") or 1)
        ):
            findings.append(Finding("code/odcr_core/config_resolver.py", 1, repr(train), "Resolved train snapshot must expose no-accum formula and flags."))
    except Exception as exc:
        findings.append(Finding("code/odcr_core/config_resolver.py", 1, repr(exc), "No-accum dynamic resolver check must pass."))
    finally:
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if inserted:
            try:
                sys.path.remove(str(code_dir))
            except ValueError:
                pass

    if findings:
        result.fail("ODCR no-accum architecture has active drift.", findings)
    else:
        result.summary = "No active accumulation parser/config/resolver/training/runtime/doc path remains; per-GPU/global no-accum semantics are enforced."
    return result


def run_checks(*, repo_root: str | Path = REPO_ROOT, strict: bool = False) -> GuardrailReport:
    root = Path(repo_root).resolve()
    checks = [
        _check_mainline_preset_reads,
        _check_scattered_yaml_env,
        _check_shell_entrypoints,
        _check_retired_run_stage,
        _check_legacy_kill_pass_absence,
        _check_config_top_level,
        _check_batch_formula_validation,
        _check_logs_not_data,
        _check_data_not_runs,
        _check_parameter_drift,
        _check_step3_live_semantics,
        _check_step3_typed_bridge_retired,
        _check_step4_legacy_fields_not_primary,
        _check_step4_rcr_one_control,
        _check_step4_route_contract_posterior,
        _check_step4_entropy_auxiliary_only,
        _check_step5_rcr_posterior_consumption,
        _check_step5_lci_called,
        _check_step5_ccv_packet,
        _check_step5_fca_evidence_basis,
        _check_step5_legacy_fields_not_primary,
        _check_step5_innovation_one_control,
        _check_step5_lora_flan_one_control,
        _check_step5_positive_tests_no_legacy_fields,
        _check_preprocess_detail_fields_retired,
        _check_roots_embed_dim_one_control,
        _check_child_argparse_payload_transport,
        _check_step4_helper_defaults_strict,
        _check_step5_parser_defaults_strict,
        _check_step3_structured_losses_one_control,
        _check_step3_upstream_preprocess_hard_gate,
        _check_step3_v0_parameter_surface,
        _check_step3_s2r_perf_cache_downstream_guardrails,
        _check_step3_formal_logging_param_surface,
        _check_step3_formal_view_dynamic,
        _check_step3_config_artifact_split,
        _check_step3_structured_metric_files_active,
        _check_step3_run_summary_log_indexes,
        _check_step3_formal_source_table_static,
        _check_step3_non_task2_roles_and_g2_dynamic,
        _check_step3_no_old_param_regressions_static,
        _check_step3_quality_evidence_performance_rebuild,
        _check_stage_truth_upstream_resolver,
        _check_stage_status_strict_antiforgery_guardrail,
        _check_step4_runtime_preflight_twophase_guardrail,
        _check_gpu_bridge_step4_bounded_preflight_admission,
        _check_step4_evidence_level_antifake_tuning_guardrail,
        _check_no_accum_architecture_guardrail,
        _check_step3_runtime_probe_truth_contract,
        _check_step3_paper_eval_explicit_damping_contract,
        _check_step5_gate_arch_adv_eta_one_control,
        _check_preprocess_skip_completed_fingerprint_gate,
        _check_step3_checkpoint_step4_lineage_gate,
        _check_step4_export_step5_lineage_gate,
        _check_step5_checkpoint_eval_rerank_lineage_gate,
        _check_eval_rerank_resolved_config_no_fallback,
        _check_step3_finite_loss_global_sync,
        _check_graph_tied_zero_losses,
        _check_step5_find_unused_preflight_policy,
        _check_step5_no_hf_labels_ce_once,
        _check_step5_weighted_lci_fca_once,
        _check_manifest_embed_dim_no_bare_env,
        _check_evolution_active_parameters,
        _check_evolution_contract_fields,
        _check_evolution_artifact_lineage,
        _check_evolution_entrypoints,
        _check_evolution_env_reads,
        _check_evolution_loss_wiring,
        _check_evolution_mask_gate_ddp,
        _check_evolution_legacy_aliases,
        _check_evolution_checklist_or_ledger,
        _check_gpu_tmux_policy_docs,
        _check_post_edit_script_exists,
        _check_codex_workflow_requires_post_edit,
        _check_post_edit_no_real_training,
        _check_codex_stop_hook_exists,
        _check_codex_hook_script_safe,
        _check_codex_hooks_or_manual_primary_docs,
        _check_run_summary_entrypoint,
        _check_resolved_config_filename_canonical,
        _check_latest_points_to_run_summary,
        _check_latest_lookup_no_scan_or_legacy_layout,
        _check_latest_lookup_requires_summary_hard_gate,
        _check_step3_tokenizer_cache_manifest_hard_gate,
        _check_step4_encoded_cache_manifest_hard_gate,
        _check_step5_tokenize_cache_manifest_hard_gate,
        _check_active_cache_reuse_not_path_mtime_only,
        _check_console_default_summary_policy,
        _check_active_logs_not_legacy_defaults,
        _check_verbose_debug_display_only,
        _check_stop_hook_ignored_path_rules,
        _check_stop_hook_ignored_only_noop,
        _check_post_edit_governance_fast_scope,
        _check_stop_hook_uncertain_cases_skip,
        _check_stop_hook_auto_timeout_fast,
        _check_logging_outputs_declare_artifact_role,
        _check_run_facing_outputs_update_summary_latest,
        _check_ai_analysis_not_training_full_log_mirror_evolution,
        _check_console_default_no_full_dump_evolution,
        _check_new_log_paths_not_forbidden_destinations,
        _check_stop_hook_unknown_session_files_skip,
        _check_runtime_diagnostics_schema_v22,
        _check_runtime_diagnostics_workspace_scope_flag,
        _check_runtime_diagnostics_no_legacy_fields,
        _check_skip_scope_has_null_command,
        _check_run_logs_target_run_meta,
        _check_cache_artifacts_not_runs_meta,
        _check_ai_analysis_not_full_log_mirror,
        _check_data_merged_do_not_receive_logs,
        _check_top_level_logs_and_fallbacks_retired,
        _check_metrics_filename_canonical,
        _check_old_layout_default_log_writes_retired,
        _check_tail_only_latest_run_summary_meta_logs,
        _check_no_old_fallback_log_paths,
        _check_ai_analysis_not_active_training_mirror_old_layout,
        _check_data_merged_no_log_files_old_layout,
        _check_agents_post_edit_scope_not_fixed_step3,
    ]
    results = [check(root) for check in checks]
    if strict:
        pass
    return GuardrailReport(results)


def format_report(report: GuardrailReport) -> str:
    lines = [
        f"ODCR One-Control Guardrails: {'PASS' if report.ok else 'FAIL'} "
        f"({report.failures} fail, {report.warnings} warn)"
    ]
    lines.append("Guardrail groups:")
    for group, rule_ids in GUARDRAIL_GROUPS:
        visible = [rule_id for rule_id in rule_ids if any(item.rule_id == rule_id for item in report.results)]
        if visible:
            lines.append(f" - {group}: {', '.join(visible)}")
    for item in report.results:
        summary = f" - {item.summary}" if item.summary else ""
        lines.append(f"[{item.status}] {item.rule_id} ({item.group}) {item.title}{summary}")
        for finding in item.findings[:12]:
            lines.append(f"  {finding.path}:{finding.line}: {finding.text}")
            lines.append(f"    fix: {finding.suggestion}")
        if len(item.findings) > 12:
            lines.append(f"  ... {len(item.findings) - 12} more finding(s) omitted")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check ODCR One-Control architecture guardrails.")
    parser.add_argument("--strict", action="store_true", help="Fail on architecture violations.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    args = parser.parse_args(argv)
    report = run_checks(repo_root=args.repo_root, strict=args.strict)
    print(format_report(report))
    if not report.ok or (args.strict and report.warnings):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
