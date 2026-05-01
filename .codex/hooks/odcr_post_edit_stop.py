#!/usr/bin/env python3
"""Repo-local Codex Stop hook for ODCR post-edit validation."""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import traceback
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Iterable


EXPECTED_REPO_ROOT = Path("/public/home/zhangliml/lc/ODCR/ODCR-main")
POST_EDIT_REL = Path("code") / "tools" / "odcr_post_edit_check.py"
HOOK_REL = Path(".codex") / "hooks" / "odcr_post_edit_stop.py"
LOG_DIR_REL = Path("AI_analysis") / "01_raw_logs" / "codex_hooks"
RUNTIME_SCHEMA_VERSION = "odcr_codex_hook_runtime/2.2"
DEFAULT_HOOK_MAX_SECONDS = 180
MANUAL_DEEP_CHECK_MAX_SECONDS = 900
MAX_TOUCHED_FILES_SAMPLE = 50
BUSINESS_STAGE_SCOPES = ("preprocess", "step3", "step4", "step5", "eval")
VALID_HOOK_SCOPES = ("governance-fast", "governance", "config", "logging", *BUSINESS_STAGE_SCOPES, "all")
SCOPE_ORDER = ("governance-fast", "governance", "config", "logging", *BUSINESS_STAGE_SCOPES, "all")
CROSS_STAGE_SCOPE_FILES = {
    "code/data_contract.py",
    "code/odcr_core/index_contract.py",
    "code/odcr_core/manifests.py",
    "code/odcr_core/training_checkpoint.py",
}
IGNORED_EXACT_PATHS = {
    "audit.log",
    EXPECTED_REPO_ROOT.joinpath("audit.log").as_posix(),
}
IGNORED_DIR_PREFIXES = (
    "AI_analysis/",
    "AI_analysis/01_raw_logs/codex_hooks/",
    "runs/",
    "cache/",
    "artifacts/",
    "data/",
    "merged/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
)
IGNORED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
IGNORED_FILE_PATTERNS = (
    "*.log",
    "*.out",
    "*.err",
    "*.tmp",
    "*.pyc",
    "*.pyo",
    "*.swp",
    "*.swo",
    "*.bak",
)


@dataclass(frozen=True)
class ScopeInference:
    selected_scope: str
    inference_source: str
    inference_reason: str
    session_touched_files: tuple[str, ...] = ()
    ignored_files: tuple[str, ...] = ()
    effective_scope_files: tuple[str, ...] = ()
    scope_candidates: tuple[str, ...] = ()
    multi_stage_detected: bool = False
    workspace_dirty_detected: bool | None = None
    workspace_changed_files_count: int | None = None
    workspace_git_status_used_for_scope: bool = False
    skipped: bool = False
    skip_reason: str | None = None
    override_source: str | None = None


def _read_payload() -> dict[str, object]:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("Codex hook payload must be a JSON object")
    return payload


def _payload_cwd(payload: dict[str, object]) -> Path:
    value = os.environ.get("ODCR_HOOK_LAUNCH_CWD") or payload.get("cwd") or os.getcwd()
    return Path(str(value)).resolve()


def _hook_event_name(payload: dict[str, object]) -> str:
    for key in ("hook_event_name", "hook_event", "event"):
        value = payload.get(key)
        if value:
            return str(value)
    return os.environ.get("ODCR_HOOK_EVENT_NAME") or os.environ.get("CODEX_HOOK_EVENT_NAME") or "Stop"


def _git_stdout(args: list[str], *, cwd: Path) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout.strip()


def _looks_like_repo_root(path: Path) -> bool:
    return (path / HOOK_REL).is_file() and (path / "code").is_dir()


def _candidate_roots(*starts: Path) -> Iterable[Path]:
    seen: set[Path] = set()
    script_root = Path(__file__).resolve().parents[2]
    for start in (script_root, *starts):
        for candidate in (start, *start.parents):
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield resolved


def _find_repo_root(cwd: Path) -> Path:
    env_root = os.environ.get("ODCR_HOOK_REPO_ROOT")
    if env_root:
        candidate = Path(env_root).resolve()
        if _looks_like_repo_root(candidate):
            return candidate

    try:
        git_root = Path(_git_stdout(["rev-parse", "--show-toplevel"], cwd=cwd)).resolve()
        if _looks_like_repo_root(git_root):
            return git_root
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    for candidate in _candidate_roots(cwd):
        if _looks_like_repo_root(candidate):
            return candidate

    raise RuntimeError(
        "Could not locate ODCR repo root from hook script path, cwd, or git. "
        f"script={Path(__file__).resolve()} cwd={cwd}"
    )


def _fallback_repo_root() -> Path:
    env_root = os.environ.get("ODCR_HOOK_REPO_ROOT")
    if env_root:
        return Path(env_root).resolve()
    script_root = Path(__file__).resolve().parents[2]
    if _looks_like_repo_root(script_root):
        return script_root
    return EXPECTED_REPO_ROOT


def _decode_status_path(raw_path: str) -> str:
    raw_path = raw_path.strip()
    if raw_path.startswith('"'):
        parts = shlex.split(raw_path)
        if parts:
            return parts[0]
    return raw_path


def _changed_files(repo_root: Path) -> list[str]:
    output = _git_stdout(["status", "--porcelain", "--untracked-files=all"], cwd=repo_root)
    changed: list[str] = []
    for line in output.splitlines():
        if len(line) < 4:
            continue
        path_part = line[3:]
        pieces = path_part.split(" -> ") if " -> " in path_part else [path_part]
        changed.extend(_decode_status_path(piece) for piece in pieces if piece.strip())
    return sorted(set(changed))


def _unique_sorted(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({item for item in items if item}))


def _clean_path_token(value: str) -> str:
    token = value.strip().strip("'\"`")
    token = token.replace("\\", "/")
    token = re.sub(r"(?::\d+){1,2}$", "", token)
    return token.strip().rstrip(",;")


def _normalize_rel_path(raw_path: str, *, repo_root: Path, cwd: Path | None = None) -> str | None:
    token = _clean_path_token(raw_path)
    if not token or "\0" in token or "://" in token:
        return None
    if token in {"/dev/null", "dev/null"} or token.startswith(("-", "{", "[")):
        return None
    if token.startswith(("a/", "b/")) and len(token) > 2:
        token = token[2:]
    while token.startswith("./"):
        token = token[2:]

    root = repo_root.resolve()
    if token.startswith(root.as_posix() + "/"):
        return token[len(root.as_posix()) + 1 :]

    path = Path(token)
    try:
        if path.is_absolute():
            resolved = path.resolve()
            try:
                return resolved.relative_to(root).as_posix()
            except ValueError:
                return None
        if cwd is not None:
            resolved_cwd = cwd.resolve()
            if resolved_cwd == root or root in resolved_cwd.parents:
                candidate = (resolved_cwd / token).resolve()
                try:
                    return candidate.relative_to(root).as_posix()
                except ValueError:
                    pass
    except OSError:
        return None

    return token


def _normalize_paths(paths: Iterable[str], *, repo_root: Path, cwd: Path | None = None) -> tuple[str, ...]:
    return _unique_sorted(
        rel
        for raw_path in paths
        if (rel := _normalize_rel_path(str(raw_path), repo_root=repo_root, cwd=cwd))
    )


def _is_ignored_path(rel_path: str) -> bool:
    rel = _clean_path_token(str(rel_path)).replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]
    if rel in IGNORED_EXACT_PATHS:
        return True
    root_prefix = EXPECTED_REPO_ROOT.as_posix() + "/"
    if rel.startswith(root_prefix):
        rel = rel[len(root_prefix) :]
        if rel in IGNORED_EXACT_PATHS:
            return True
    if rel == "audit.log":
        return True
    if any(rel == prefix.rstrip("/") or rel.startswith(prefix) for prefix in IGNORED_DIR_PREFIXES):
        return True
    parts = set(Path(rel).parts)
    if parts & IGNORED_PARTS:
        return True
    name = Path(rel).name
    return any(fnmatch(name, pattern) for pattern in IGNORED_FILE_PATTERNS)


def _split_ignored_paths(paths: Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    raw = _unique_sorted(paths)
    ignored = tuple(path for path in raw if _is_ignored_path(path))
    effective = tuple(path for path in raw if not _is_ignored_path(path))
    return raw, ignored, effective


def _patch_paths(text: str, *, repo_root: Path, cwd: Path | None = None) -> tuple[str, ...]:
    paths: list[str] = []
    for match in re.finditer(
        r"^\*\*\* (?:Add|Update|Delete) File:\s*(?P<file>.+?)\s*$|"
        r"^\*\*\* Move to:\s*(?P<move>.+?)\s*$",
        text,
        flags=re.MULTILINE,
    ):
        paths.append(match.group("file") or match.group("move") or "")
    return _normalize_paths(paths, repo_root=repo_root, cwd=cwd)


def _patch_result_paths(text: str, *, repo_root: Path, cwd: Path | None = None) -> tuple[str, ...]:
    if "Success. Updated the following files" not in text:
        return ()
    paths: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\s*(?:[AMDRCU?!]{1,2}\s+)?(?P<path>[\w./ -]+\.[A-Za-z0-9_]+)\s*$", line)
        if match:
            paths.append(match.group("path"))
    return _normalize_paths(paths, repo_root=repo_root, cwd=cwd)


def _paths_from_shell_command(command: str, *, repo_root: Path, cwd: Path | None = None) -> tuple[str, ...]:
    paths: list[str] = []
    if "*** Begin Patch" in command:
        paths.extend(_patch_paths(command, repo_root=repo_root, cwd=cwd))

    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    if tokens and Path(tokens[0]).name in {"bash", "sh"}:
        for idx, token in enumerate(tokens[:-1]):
            if token in {"-c", "-lc"}:
                paths.extend(_paths_from_shell_command(tokens[idx + 1], repo_root=repo_root, cwd=cwd))

    for match in re.finditer(r"(?:^|\s)(?:\d?>|>>)\s*(?P<path>(?!&)[^ \t\n;|]+)", command):
        paths.append(match.group("path"))

    if tokens:
        tool = Path(tokens[0]).name
        if tool == "tee":
            paths.extend(token for token in tokens[1:] if not token.startswith("-"))
        elif tool in {"touch", "rm"}:
            paths.extend(token for token in tokens[1:] if not token.startswith("-"))
        elif tool in {"mv", "cp"} and len(tokens) >= 3:
            paths.extend(token for token in tokens[1:] if not token.startswith("-"))
        elif tool == "sed" and any(token.startswith("-i") for token in tokens[1:]):
            paths.extend(token for token in tokens[1:] if not token.startswith("-") and "/" in token)

    return _normalize_paths(paths, repo_root=repo_root, cwd=cwd)


def _string_values(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _string_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _string_values(item)


def _values_for_keys(value: object, keys: set[str]) -> Iterable[object]:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in keys:
                yield item
            yield from _values_for_keys(item, keys)
    elif isinstance(value, list):
        for item in value:
            yield from _values_for_keys(item, keys)


def _tool_name(record: dict[str, object]) -> str:
    for key in ("name", "tool_name", "recipient", "recipient_name"):
        value = record.get(key)
        if value:
            return str(value)
    function = record.get("function")
    if isinstance(function, dict) and function.get("name"):
        return str(function["name"])
    return ""


def _extract_touched_from_transcript_record(
    record: object,
    *,
    repo_root: Path,
    cwd: Path | None = None,
) -> tuple[str, ...]:
    paths: list[str] = []
    if isinstance(record, dict):
        name = _tool_name(record).lower()
        if "apply_patch" in name:
            for text in _string_values(record):
                paths.extend(_patch_paths(text, repo_root=repo_root, cwd=cwd))
                paths.extend(_patch_result_paths(text, repo_root=repo_root, cwd=cwd))
        elif "exec_command" in name:
            for value in _values_for_keys(record, {"cmd", "command"}):
                if isinstance(value, str):
                    paths.extend(_paths_from_shell_command(value, repo_root=repo_root, cwd=cwd))
        else:
            record_type = str(record.get("type", "")).lower()
            if any(marker in record_type for marker in ("tool", "function", "result")):
                for text in _string_values(record):
                    if "*** Begin Patch" in text:
                        paths.extend(_patch_paths(text, repo_root=repo_root, cwd=cwd))
                    paths.extend(_patch_result_paths(text, repo_root=repo_root, cwd=cwd))
        for item in record.values():
            paths.extend(_extract_touched_from_transcript_record(item, repo_root=repo_root, cwd=cwd))
    elif isinstance(record, list):
        for item in record:
            paths.extend(_extract_touched_from_transcript_record(item, repo_root=repo_root, cwd=cwd))
    elif isinstance(record, str):
        if "*** Begin Patch" in record:
            paths.extend(_patch_paths(record, repo_root=repo_root, cwd=cwd))
        paths.extend(_patch_result_paths(record, repo_root=repo_root, cwd=cwd))
    return _unique_sorted(paths)


def _json_records_from_transcript(text: str) -> list[object]:
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, list) else [parsed]
    if stripped.startswith("{") and "\n" not in stripped:
        return [json.loads(stripped)]

    records: list[object] = []
    failures = 0
    json_lines = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith(("{", "[")):
            json_lines += 1
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                failures += 1
    if json_lines and failures == json_lines:
        raise ValueError("transcript JSON/JSONL parse failed")
    return records


def _transcript_touched_files(
    transcript_path: Path,
    *,
    repo_root: Path,
    cwd: Path | None = None,
) -> tuple[str, ...]:
    text = transcript_path.read_text(encoding="utf-8", errors="replace")
    records = _json_records_from_transcript(text)
    paths: list[str] = []
    if records:
        for record in records:
            paths.extend(_extract_touched_from_transcript_record(record, repo_root=repo_root, cwd=cwd))
    elif text.strip():
        paths.extend(_extract_touched_from_transcript_record(text, repo_root=repo_root, cwd=cwd))
    return _unique_sorted(paths)


TOUCHED_PAYLOAD_KEYS = {
    "touched_files",
    "changed_files",
    "modified_files",
    "edited_files",
    "created_files",
    "deleted_files",
    "file_changes",
    "file_paths",
    "paths",
}


def _payload_touched_files(payload: dict[str, object], *, repo_root: Path, cwd: Path | None = None) -> tuple[str, ...]:
    paths: list[str] = []
    for value in _values_for_keys(payload, TOUCHED_PAYLOAD_KEYS):
        if isinstance(value, str):
            paths.append(value)
        elif isinstance(value, list):
            paths.extend(str(item) for item in value if isinstance(item, (str, Path)))
        elif isinstance(value, dict):
            paths.extend(str(item) for item in value.values() if isinstance(item, (str, Path)))
    for value in _values_for_keys(payload, {"tool_outputs", "tool_output", "output"}):
        for text in _string_values(value):
            if "*** Begin Patch" in text:
                paths.extend(_patch_paths(text, repo_root=repo_root, cwd=cwd))
            paths.extend(_patch_result_paths(text, repo_root=repo_root, cwd=cwd))
    return _normalize_paths(paths, repo_root=repo_root, cwd=cwd)


def _scope_sort_key(scope: str) -> int:
    try:
        return SCOPE_ORDER.index(scope)
    except ValueError:
        return len(SCOPE_ORDER)


def _inference_summary(inference: ScopeInference) -> dict[str, object]:
    session = list(inference.session_touched_files)
    ignored = list(inference.ignored_files)
    effective = list(inference.effective_scope_files)
    return {
        "inference_source": inference.inference_source,
        "inference_reason": inference.inference_reason,
        "selected_scope": inference.selected_scope,
        "scope_candidates": list(inference.scope_candidates),
        "multi_stage_detected": inference.multi_stage_detected,
        "session_touched_files_count": len(session),
        "session_touched_files_sample": session[:MAX_TOUCHED_FILES_SAMPLE],
        "ignored_files_count": len(ignored),
        "ignored_files_sample": ignored[:MAX_TOUCHED_FILES_SAMPLE],
        "effective_scope_files_count": len(effective),
        "effective_scope_files_sample": effective[:MAX_TOUCHED_FILES_SAMPLE],
        "workspace_dirty_detected": inference.workspace_dirty_detected,
        "workspace_changed_files_count": inference.workspace_changed_files_count,
        "workspace_git_status_used_for_scope": inference.workspace_git_status_used_for_scope,
        "skipped": inference.skipped,
        "skip_reason": inference.skip_reason,
        "override_source": inference.override_source,
    }


def _looks_like(rel_path: str, needles: Iterable[str]) -> bool:
    lowered = rel_path.lower()
    return any(needle in lowered for needle in needles)


def _scope_for_path(rel_path: str) -> str | None:
    rel = rel_path.replace("\\", "/")
    name = Path(rel).name

    if _is_ignored_path(rel) or rel.startswith("_archive/"):
        return None
    if rel in {"README.md", "AGENTS.md"} or rel.startswith("docs/"):
        return "governance-fast"
    if rel.startswith(".codex/"):
        return "governance-fast"
    if rel in CROSS_STAGE_SCOPE_FILES or _looks_like(
        rel,
        ("lineage", "cache_manifest", "checkpoint_lineage", "export_contract"),
    ):
        return "all"
    if rel in {
        "code/tools/check_one_control_guardrails.py",
        "code/tools/odcr_post_edit_check.py",
        "code/tests/test_one_control_guardrails.py",
        "code/tests/test_post_edit_check.py",
    }:
        return "governance-fast"
    if rel in {
        "code/odcr_core/logging_meta.py",
        "code/tests/test_run_summary_logging.py",
        "code/tests/test_logging_console_file.py",
        "code/train_logging.py",
    }:
        return "logging"
    if rel.startswith("configs/") or name in {
        "config_schema.py",
        "config_resolver.py",
        "config.py",
        "paths_config.py",
        "runners.py",
        "path_layout.py",
    }:
        return "config"
    if name in {
        "preprocess_data.py",
        "split_data.py",
        "combine_data.py",
        "compute_embeddings.py",
        "infer_domain_semantics.py",
        "preprocess_schema.py",
        "preprocess_runtime.py",
        "preprocess_status.py",
        "preprocess_registry.py",
    } or name.startswith("preprocess_") or _looks_like(rel, ("preprocess",)):
        return "preprocess"
    if name in {
        "step3_train_core.py",
        "step3_entry.py",
        "odcr_representation.py",
        "odcr_losses.py",
    } or _looks_like(rel, ("step3", "test_step3")):
        return "step3"
    if name in {
        "step4_engine.py",
        "step4_entry.py",
        "odcr_cf_routing.py",
        "step4_training_export.py",
    } or _looks_like(rel, ("step4", "test_step4", "test_index_contract")):
        return "step4"
    if name in {
        "step5_engine.py",
        "step5_entry.py",
        "step5_innovation.py",
        "step5_native_lora.py",
        "step5_word_losses.py",
    } or _looks_like(rel, ("step5", "test_step5")):
        return "step5"
    if _looks_like(rel, ("eval", "rerank", "bleu", "bert_score", "bertscore", "decode")):
        return "eval"
    return None


def _scope_candidates(changed_files: Iterable[str]) -> tuple[str, ...]:
    scopes = {_scope_for_path(path) for path in changed_files}
    return tuple(sorted((scope for scope in scopes if scope), key=_scope_sort_key))


def _scope_override() -> tuple[str, str] | None:
    value = os.environ.get("ODCR_HOOK_SCOPE", "").strip()
    if not value:
        return None
    if value in VALID_HOOK_SCOPES:
        return value, "env"
    return None


def _workspace_state(
    repo_root: Path,
    workspace_changed_files_func: Callable[[Path], list[str]] | None = None,
) -> tuple[bool | None, int | None]:
    try:
        files = (workspace_changed_files_func or _changed_files)(repo_root)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None
    count = len(_unique_sorted(files))
    return count > 0, count


def _override_inference(
    *,
    workspace_dirty_detected: bool | None = None,
    workspace_changed_files_count: int | None = None,
) -> ScopeInference | None:
    override = _scope_override()
    if override is None:
        return None
    selected_scope, override_source = override
    return ScopeInference(
        selected_scope=selected_scope,
        inference_source="explicit_override",
        inference_reason="env_override",
        workspace_dirty_detected=workspace_dirty_detected,
        workspace_changed_files_count=workspace_changed_files_count,
        skipped=False,
        skip_reason=None,
        override_source=override_source,
    )


def _skip_inference(
    *,
    source: str,
    reason: str,
    session_touched_files: Iterable[str] = (),
    ignored_files: Iterable[str] = (),
    effective_scope_files: Iterable[str] = (),
    scope_candidates: Iterable[str] = (),
    workspace_dirty_detected: bool | None = None,
    workspace_changed_files_count: int | None = None,
) -> ScopeInference:
    return ScopeInference(
        selected_scope="skip",
        inference_source=source,
        inference_reason=reason,
        session_touched_files=_unique_sorted(session_touched_files),
        ignored_files=_unique_sorted(ignored_files),
        effective_scope_files=_unique_sorted(effective_scope_files),
        scope_candidates=tuple(sorted(set(scope_candidates), key=_scope_sort_key)),
        workspace_dirty_detected=workspace_dirty_detected,
        workspace_changed_files_count=workspace_changed_files_count,
        skipped=True,
        skip_reason=reason,
    )


def _scope_from_session_touched_files(
    touched_files: Iterable[str],
    *,
    source: str,
    default_reason: str,
    empty_reason: str,
    workspace_dirty_detected: bool | None = None,
    workspace_changed_files_count: int | None = None,
) -> ScopeInference:
    raw_files, ignored_files, files = _split_ignored_paths(touched_files)
    if raw_files and not files:
        return _skip_inference(
            source=source,
            reason="only_ignored_files_changed",
            session_touched_files=raw_files,
            ignored_files=ignored_files,
            workspace_dirty_detected=workspace_dirty_detected,
            workspace_changed_files_count=workspace_changed_files_count,
        )
    if not files:
        return _skip_inference(
            source=source,
            reason=empty_reason,
            ignored_files=ignored_files,
            workspace_dirty_detected=workspace_dirty_detected,
            workspace_changed_files_count=workspace_changed_files_count,
        )

    candidates = _scope_candidates(files)
    unknown_files = tuple(path for path in files if _scope_for_path(path) is None)
    if unknown_files:
        return _skip_inference(
            source=source,
            reason="unknown_session_touched_files",
            session_touched_files=raw_files,
            ignored_files=ignored_files,
            effective_scope_files=files,
            scope_candidates=candidates,
            workspace_dirty_detected=workspace_dirty_detected,
            workspace_changed_files_count=workspace_changed_files_count,
        )

    if "all" in candidates:
        return ScopeInference(
            selected_scope="all",
            inference_source=source,
            inference_reason="cross_stage_session_touched_files",
            session_touched_files=raw_files,
            ignored_files=ignored_files,
            effective_scope_files=files,
            scope_candidates=candidates,
            multi_stage_detected=True,
            workspace_dirty_detected=workspace_dirty_detected,
            workspace_changed_files_count=workspace_changed_files_count,
        )

    business = tuple(scope for scope in candidates if scope in BUSINESS_STAGE_SCOPES)
    if len(business) >= 2:
        return ScopeInference(
            selected_scope="all",
            inference_source=source,
            inference_reason="multi_business_stage_session_touched_files",
            session_touched_files=raw_files,
            ignored_files=ignored_files,
            effective_scope_files=files,
            scope_candidates=candidates,
            multi_stage_detected=True,
            workspace_dirty_detected=workspace_dirty_detected,
            workspace_changed_files_count=workspace_changed_files_count,
        )
    if len(business) == 1:
        return ScopeInference(
            selected_scope=business[0],
            inference_source=source,
            inference_reason=default_reason,
            session_touched_files=raw_files,
            ignored_files=ignored_files,
            effective_scope_files=files,
            scope_candidates=candidates,
            workspace_dirty_detected=workspace_dirty_detected,
            workspace_changed_files_count=workspace_changed_files_count,
        )
    if "config" in candidates:
        return ScopeInference(
            selected_scope="config",
            inference_source=source,
            inference_reason=default_reason,
            session_touched_files=raw_files,
            ignored_files=ignored_files,
            effective_scope_files=files,
            scope_candidates=candidates,
            workspace_dirty_detected=workspace_dirty_detected,
            workspace_changed_files_count=workspace_changed_files_count,
        )
    if "governance-fast" in candidates:
        return ScopeInference(
            selected_scope="governance-fast",
            inference_source=source,
            inference_reason=default_reason,
            session_touched_files=raw_files,
            ignored_files=ignored_files,
            effective_scope_files=files,
            scope_candidates=candidates,
            workspace_dirty_detected=workspace_dirty_detected,
            workspace_changed_files_count=workspace_changed_files_count,
        )
    return _skip_inference(
        source=source,
        reason="unknown_session_touched_files",
        session_touched_files=raw_files,
        ignored_files=ignored_files,
        effective_scope_files=files,
        scope_candidates=candidates,
        workspace_dirty_detected=workspace_dirty_detected,
        workspace_changed_files_count=workspace_changed_files_count,
    )


def _payload_transcript_path(payload: dict[str, object], *, cwd: Path) -> Path | None:
    value = payload.get("transcript_path")
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()


def infer_scope_for_payload(
    payload: dict[str, object],
    *,
    repo_root: Path,
    cwd: Path | None = None,
    workspace_changed_files_func: Callable[[Path], list[str]] | None = None,
) -> ScopeInference:
    resolved_cwd = cwd or repo_root
    workspace_dirty_detected, workspace_changed_files_count = _workspace_state(
        repo_root,
        workspace_changed_files_func,
    )
    override = _override_inference(
        workspace_dirty_detected=workspace_dirty_detected,
        workspace_changed_files_count=workspace_changed_files_count,
    )
    if override is not None:
        return override

    transcript_path = _payload_transcript_path(payload, cwd=resolved_cwd)
    if transcript_path is not None and transcript_path.is_file():
        try:
            touched = _transcript_touched_files(transcript_path, repo_root=repo_root, cwd=resolved_cwd)
        except Exception:
            return _skip_inference(
                source="transcript",
                reason="transcript_parse_failed",
                workspace_dirty_detected=workspace_dirty_detected,
                workspace_changed_files_count=workspace_changed_files_count,
            )
        if touched:
            return _scope_from_session_touched_files(
                touched,
                source="transcript",
                default_reason="transcript_session_touched_files",
                empty_reason="transcript_no_touched_files",
                workspace_dirty_detected=workspace_dirty_detected,
                workspace_changed_files_count=workspace_changed_files_count,
            )
        return _skip_inference(
            source="transcript",
            reason="transcript_no_touched_files",
            workspace_dirty_detected=workspace_dirty_detected,
            workspace_changed_files_count=workspace_changed_files_count,
        )

    payload_touched = _payload_touched_files(payload, repo_root=repo_root, cwd=resolved_cwd)
    if payload_touched:
        return _scope_from_session_touched_files(
            payload_touched,
            source="payload",
            default_reason="payload_session_touched_files",
            empty_reason="no_session_touched_files",
            workspace_dirty_detected=workspace_dirty_detected,
            workspace_changed_files_count=workspace_changed_files_count,
        )

    return _skip_inference(
        source="none",
        reason="no_session_touched_files",
        workspace_dirty_detected=workspace_dirty_detected,
        workspace_changed_files_count=workspace_changed_files_count,
    )


def _fallback_inference(reason: str = "initializing") -> ScopeInference:
    return ScopeInference(
        selected_scope="skip",
        inference_source="none",
        inference_reason=reason,
        skipped=True,
        skip_reason=reason,
    )


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _timestamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _log_dir(repo_root: Path) -> Path:
    path = repo_root / LOG_DIR_REL
    path.mkdir(parents=True, exist_ok=True)
    return path


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).resolve() if value else default


def _runtime_paths(repo_root: Path, stamp: str) -> tuple[Path, Path]:
    log_dir = _log_dir(repo_root)
    runtime_path = _path_from_env("ODCR_HOOK_RUNTIME_PATH", log_dir / f"runtime_{stamp}.json")
    runtime_last = _path_from_env("ODCR_HOOK_RUNTIME_LAST_PATH", log_dir / "runtime_last.json")
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_last.parent.mkdir(parents=True, exist_ok=True)
    return runtime_path, runtime_last


def _stream_paths(repo_root: Path, stamp: str) -> tuple[Path, Path]:
    log_dir = _log_dir(repo_root)
    stdout_path = _path_from_env("ODCR_HOOK_STDOUT_PATH", log_dir / f"post_edit_stdout_{stamp}.log")
    stderr_path = _path_from_env("ODCR_HOOK_STDERR_PATH", log_dir / f"post_edit_stderr_{stamp}.log")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.touch(exist_ok=True)
    stderr_path.touch(exist_ok=True)
    return stdout_path, stderr_path


def _selected_python() -> str:
    return os.environ.get("ODCR_HOOK_SELECTED_PYTHON") or sys.executable


def _selected_python_version() -> str:
    return os.environ.get("ODCR_HOOK_SELECTED_PYTHON_VERSION") or platform.python_version()


def _hook_max_seconds() -> int:
    value = os.environ.get("ODCR_HOOK_MAX_SECONDS", "").strip()
    if not value:
        return DEFAULT_HOOK_MAX_SECONDS
    try:
        parsed = int(value)
    except ValueError:
        return DEFAULT_HOOK_MAX_SECONDS
    return parsed if parsed > 0 else DEFAULT_HOOK_MAX_SECONDS


def _build_post_edit_command(
    *,
    post_edit_path: Path,
    scope: str,
    max_seconds: int,
    dry_run: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(post_edit_path),
        "--scope",
        scope,
        "--max-seconds",
        str(max_seconds),
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def _command_scope(command: list[str] | None) -> str | None:
    if not command or "--scope" not in command:
        return None
    idx = command.index("--scope")
    try:
        return command[idx + 1]
    except IndexError:
        return None


def _runtime_payload(
    *,
    repo_root: Path,
    cwd: Path,
    hook_event_name: str,
    command: list[str] | None,
    returncode: int | None,
    failure_stage: str | None,
    stdout_path: Path,
    stderr_path: Path,
    inference: ScopeInference,
    max_seconds: int,
) -> dict[str, object]:
    inference_summary = _inference_summary(inference)
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "cwd": str(cwd),
        "repo_root": str(repo_root),
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "CODEX_HOME": os.environ.get("CODEX_HOME", ""),
        "SHELL": os.environ.get("SHELL", ""),
        "CONDA_PREFIX": os.environ.get("CONDA_PREFIX", ""),
        "selected_python": _selected_python(),
        "selected_python_version": _selected_python_version(),
        "hook_event_name": hook_event_name,
        "stop_hook_active": hook_event_name == "Stop",
        "post_edit_command": command,
        "post_edit_returncode": returncode,
        "max_seconds": max_seconds,
        "failure_stage": failure_stage,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        **inference_summary,
    }


def _write_runtime(
    *,
    repo_root: Path,
    stamp: str,
    cwd: Path,
    hook_event_name: str,
    command: list[str] | None,
    returncode: int | None,
    failure_stage: str | None,
    stdout_path: Path,
    stderr_path: Path,
    inference: ScopeInference,
    max_seconds: int,
) -> Path:
    runtime_path, runtime_last = _runtime_paths(repo_root, stamp)
    payload = _runtime_payload(
        repo_root=repo_root,
        cwd=cwd,
        hook_event_name=hook_event_name,
        command=command,
        returncode=returncode,
        failure_stage=failure_stage,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        inference=inference,
        max_seconds=max_seconds,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    runtime_path.write_text(text, encoding="utf-8")
    runtime_last.write_text(text, encoding="utf-8")
    return runtime_path


def _read_for_log(path: Path, *, max_chars: int = 20000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"<could not read {path}: {exc!r}>"
    if len(text) > max_chars:
        return text[:max_chars] + "\n<output truncated>\n"
    return text


def _write_log(
    *,
    repo_root: Path,
    payload: dict[str, object],
    inference: ScopeInference,
    command: list[str] | None,
    returncode: int,
    stdout_path: Path,
    stderr_path: Path,
    output: str,
) -> Path:
    log_dir = _log_dir(repo_root)
    log_path = log_dir / f"post_edit_stop_{_timestamp()}.log"
    inference_summary = _inference_summary(inference)
    transcript_path = payload.get("transcript_path")
    lines = [
        "ODCR Codex Stop hook",
        f"repo_root: {repo_root}",
        f"payload_keys: {json.dumps(sorted(payload.keys()), ensure_ascii=False)}",
        f"transcript_path: {transcript_path or 'none'}",
        f"inference_source: {inference.inference_source}",
        f"inference_reason: {inference.inference_reason}",
        f"session_touched_files_count: {inference_summary['session_touched_files_count']}",
        f"session_touched_files_sample: {json.dumps(inference_summary['session_touched_files_sample'], ensure_ascii=False)}",
        f"ignored_files_count: {inference_summary['ignored_files_count']}",
        f"ignored_files_sample: {json.dumps(inference_summary['ignored_files_sample'], ensure_ascii=False)}",
        f"effective_scope_files_count: {inference_summary['effective_scope_files_count']}",
        f"effective_scope_files_sample: {json.dumps(inference_summary['effective_scope_files_sample'], ensure_ascii=False)}",
        f"workspace_dirty_detected: {inference.workspace_dirty_detected}",
        f"workspace_changed_files_count: {inference.workspace_changed_files_count}",
        f"workspace_git_status_used_for_scope: {inference.workspace_git_status_used_for_scope}",
        f"scope_candidates: {json.dumps(list(inference.scope_candidates), ensure_ascii=False)}",
        f"multi_stage_detected: {inference.multi_stage_detected}",
        f"inferred_scope: {inference.selected_scope}",
        f"skipped: {inference.skipped}",
        f"skip_reason: {inference.skip_reason}",
        f"override_source: {inference.override_source}",
        f"command: {shlex.join(command) if command else 'none'}",
        f"returncode: {returncode}",
        f"stdout_path: {stdout_path}",
        f"stderr_path: {stderr_path}",
        "",
        output.rstrip(),
        "",
    ]
    log_path.write_text("\n".join(lines), encoding="utf-8")
    return log_path


def _stderr(message: str) -> None:
    sys.stderr.write(message.rstrip() + "\n")


def _emit_stop_json() -> None:
    sys.stdout.write(json.dumps({"continue": True}) + "\n")
    sys.stdout.flush()


def _append_text(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def main() -> int:
    stamp = _timestamp()
    payload: dict[str, object] = {}
    cwd = Path(os.environ.get("ODCR_HOOK_LAUNCH_CWD") or os.getcwd()).resolve()
    repo_root = _fallback_repo_root()
    stdout_path, stderr_path = _stream_paths(repo_root, stamp)
    hook_event_name = os.environ.get("ODCR_HOOK_EVENT_NAME") or "Stop"
    inference = _fallback_inference("initializing")
    command: list[str] | None = None
    max_seconds = _hook_max_seconds()

    try:
        payload = _read_payload()
        cwd = _payload_cwd(payload)
        hook_event_name = _hook_event_name(payload)
        repo_root = _find_repo_root(cwd)
        if repo_root != EXPECTED_REPO_ROOT:
            raise RuntimeError(f"Unexpected ODCR repo root: expected={EXPECTED_REPO_ROOT} resolved={repo_root}")
        stdout_path, stderr_path = _stream_paths(repo_root, stamp)
        _write_runtime(
            repo_root=repo_root,
            stamp=stamp,
            cwd=cwd,
            hook_event_name=hook_event_name,
            command=None,
            returncode=None,
            failure_stage=None,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            inference=inference,
            max_seconds=max_seconds,
        )

        inference = infer_scope_for_payload(payload, repo_root=repo_root, cwd=cwd)

        if inference.selected_scope == "skip":
            message = f"ODCR post-edit validation skipped: {inference.skip_reason or inference.inference_reason}."
            _append_text(stderr_path, message)
            log_path = _write_log(
                repo_root=repo_root,
                payload=payload,
                inference=inference,
                command=None,
                returncode=0,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                output=message,
            )
            _write_runtime(
                repo_root=repo_root,
                stamp=stamp,
                cwd=cwd,
                hook_event_name=hook_event_name,
                command=None,
                returncode=0,
                failure_stage=None,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                inference=inference,
                max_seconds=max_seconds,
            )
            _stderr(message)
            _stderr(f"log={log_path.relative_to(repo_root)}")
            _emit_stop_json()
            return 0

        post_edit_path = repo_root / POST_EDIT_REL
        if not post_edit_path.is_file():
            message = f"Required ODCR post-edit check is missing: {post_edit_path}"
            _append_text(stderr_path, message)
            log_path = _write_log(
                repo_root=repo_root,
                payload=payload,
                inference=inference,
                command=None,
                returncode=127,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                output=message,
            )
            _write_runtime(
                repo_root=repo_root,
                stamp=stamp,
                cwd=cwd,
                hook_event_name=hook_event_name,
                command=None,
                returncode=127,
                failure_stage="post_edit_check",
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                inference=inference,
                max_seconds=max_seconds,
            )
            _stderr(f"{message} log={log_path.relative_to(repo_root)}")
            _emit_stop_json()
            return 127

        command = _build_post_edit_command(
            post_edit_path=post_edit_path,
            scope=inference.selected_scope,
            max_seconds=max_seconds,
            dry_run=_truthy_env("ODCR_HOOK_DRY_RUN"),
        )

        with stdout_path.open("w", encoding="utf-8") as out_handle, stderr_path.open("a", encoding="utf-8") as err_handle:
            proc = subprocess.run(
                command,
                cwd=repo_root,
                text=True,
                stdout=out_handle,
                stderr=err_handle,
            )

        combined_output = (
            "SCOPE INFERENCE:\n"
            + json.dumps(_inference_summary(inference), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n\n"
        )
        combined_output += "STDOUT:\n" + _read_for_log(stdout_path)
        combined_output += "\nSTDERR:\n" + _read_for_log(stderr_path)
        log_path = _write_log(
            repo_root=repo_root,
            payload=payload,
            inference=inference,
            command=command,
            returncode=proc.returncode,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            output=combined_output,
        )
        failure_stage = "post_edit_check" if proc.returncode != 0 else None
        _write_runtime(
            repo_root=repo_root,
            stamp=stamp,
            cwd=cwd,
            hook_event_name=hook_event_name,
            command=command,
            returncode=proc.returncode,
            failure_stage=failure_stage,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            inference=inference,
            max_seconds=max_seconds,
        )
        status = "PASS" if proc.returncode == 0 else "FAIL"
        _stderr(
            "ODCR post-edit validation "
            f"{status} scope={inference.selected_scope} reason={inference.inference_reason} "
            f"log={log_path.relative_to(repo_root)}"
        )
        _stderr(f"stdout={stdout_path.relative_to(repo_root)} stderr={stderr_path.relative_to(repo_root)}")
        if proc.returncode != 0:
            _stderr(_read_for_log(stderr_path).rstrip())
        _emit_stop_json()
        return proc.returncode
    except Exception:
        trace = traceback.format_exc()
        try:
            stdout_path, stderr_path = _stream_paths(repo_root, stamp)
            _append_text(stderr_path, trace)
            log_path = _write_log(
                repo_root=repo_root,
                payload=payload,
                inference=inference,
                command=command,
                returncode=1,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                output=trace,
            )
            _write_runtime(
                repo_root=repo_root,
                stamp=stamp,
                cwd=cwd,
                hook_event_name=hook_event_name,
                command=command,
                returncode=1,
                failure_stage="python_hook",
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                inference=inference,
                max_seconds=max_seconds,
            )
            _stderr(f"ODCR Stop hook failed before validation; log={log_path.relative_to(repo_root)}")
            _stderr(f"stdout={stdout_path.relative_to(repo_root)} stderr={stderr_path.relative_to(repo_root)}")
        except Exception:
            _stderr("ODCR Stop hook failed before validation and could not write AI_analysis diagnostics.")
            _stderr(trace)
        _emit_stop_json()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
