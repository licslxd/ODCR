#!/usr/bin/env bash
set -euo pipefail

EXPECTED_REPO_ROOT="/public/home/zhangliml/lc/ODCR/ODCR-main"
D4C_PYTHON="/public/home/zhangliml/miniconda3/envs/D4C/bin/python"
LOG_DIR="$EXPECTED_REPO_ROOT/AI_analysis/01_raw_logs/codex_hooks"
RUNTIME_SCHEMA_VERSION="odcr_codex_hook_runtime/2.2"
DEFAULT_HOOK_MAX_SECONDS="180"
LAUNCH_CWD="$(pwd -P 2>/dev/null || pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
TIMESTAMP="$(date -u +"%Y%m%dT%H%M%S")_$$"
RUNTIME_PATH="$LOG_DIR/runtime_${TIMESTAMP}.json"
RUNTIME_LAST_PATH="$LOG_DIR/runtime_last.json"
STDOUT_PATH="$LOG_DIR/post_edit_stdout_${TIMESTAMP}.log"
STDERR_PATH="$LOG_DIR/post_edit_stderr_${TIMESTAMP}.log"
PYTHON_BIN=""
PYTHON_VERSION=""

json_escape() {
  local value="${1-}"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//$'\n'/\\n}"
  value="${value//$'\r'/\\r}"
  value="${value//$'\t'/\\t}"
  printf '%s' "$value"
}

json_string_or_null() {
  local value="${1-}"
  if [[ -z "$value" ]]; then
    printf 'null'
  else
    printf '"%s"' "$(json_escape "$value")"
  fi
}

write_launcher_runtime() {
  local failure_stage="${1:-unknown}"
  local post_edit_returncode="${2:-null}"
  local selected_python="${3:-}"
  local selected_version="${4:-}"
  local post_edit_command="${5:-}"
  local max_seconds="${ODCR_HOOK_MAX_SECONDS:-$DEFAULT_HOOK_MAX_SECONDS}"
  if ! [[ "$max_seconds" =~ ^[0-9]+$ ]]; then
    max_seconds="$DEFAULT_HOOK_MAX_SECONDS"
  fi
  mkdir -p "$LOG_DIR"
  {
    printf '{\n'
    printf '  "schema_version": "%s",\n' "$(json_escape "$RUNTIME_SCHEMA_VERSION")"
    printf '  "cwd": "%s",\n' "$(json_escape "$LAUNCH_CWD")"
    printf '  "repo_root": "%s",\n' "$(json_escape "$REPO_ROOT")"
    printf '  "PATH": "%s",\n' "$(json_escape "${PATH:-}")"
    printf '  "HOME": "%s",\n' "$(json_escape "${HOME:-}")"
    printf '  "CODEX_HOME": "%s",\n' "$(json_escape "${CODEX_HOME:-}")"
    printf '  "SHELL": "%s",\n' "$(json_escape "${SHELL:-}")"
    printf '  "CONDA_PREFIX": "%s",\n' "$(json_escape "${CONDA_PREFIX:-}")"
    printf '  "selected_python": %s,\n' "$(json_string_or_null "$selected_python")"
    printf '  "selected_python_version": %s,\n' "$(json_string_or_null "$selected_version")"
    printf '  "hook_event_name": "%s",\n' "$(json_escape "${CODEX_HOOK_EVENT_NAME:-Stop}")"
    printf '  "stop_hook_active": true,\n'
    printf '  "inference_source": "launcher",\n'
    printf '  "inference_reason": "%s",\n' "$(json_escape "$failure_stage")"
    printf '  "selected_scope": "skip",\n'
    printf '  "scope_candidates": [],\n'
    printf '  "multi_stage_detected": false,\n'
    printf '  "session_touched_files_count": 0,\n'
    printf '  "session_touched_files_sample": [],\n'
    printf '  "ignored_files_count": 0,\n'
    printf '  "ignored_files_sample": [],\n'
    printf '  "effective_scope_files_count": 0,\n'
    printf '  "effective_scope_files_sample": [],\n'
    printf '  "workspace_git_status_used_for_scope": false,\n'
    printf '  "workspace_dirty_detected": null,\n'
    printf '  "workspace_changed_files_count": null,\n'
    printf '  "skipped": true,\n'
    printf '  "skip_reason": "%s",\n' "$(json_escape "$failure_stage")"
    printf '  "override_source": null,\n'
    printf '  "post_edit_command": null,\n'
    printf '  "post_edit_returncode": %s,\n' "$post_edit_returncode"
    printf '  "max_seconds": %s,\n' "$max_seconds"
    printf '  "failure_stage": "%s",\n' "$(json_escape "$failure_stage")"
    printf '  "stdout_path": "%s",\n' "$(json_escape "$STDOUT_PATH")"
    printf '  "stderr_path": "%s",\n' "$(json_escape "$STDERR_PATH")"
    printf '  "timestamp": "%s",\n' "$(json_escape "$TIMESTAMP")"
    printf '  "candidates": "%s"\n' "$(json_escape "${CANDIDATES_TEXT:-}")"
    printf '}\n'
  } >"$RUNTIME_PATH"
  cp "$RUNTIME_PATH" "$RUNTIME_LAST_PATH"
}

if [[ "$REPO_ROOT" != "$EXPECTED_REPO_ROOT" ]]; then
  CANDIDATES_TEXT=""
  write_launcher_runtime "launcher" "127" "" "" ""
  {
    echo "ODCR Codex Stop hook: launcher path resolved unexpected repo root."
    echo "expected=$EXPECTED_REPO_ROOT"
    echo "resolved=$REPO_ROOT"
    echo "cwd=$LAUNCH_CWD"
  } >&2
  exit 127
fi

declare -a CANDIDATES=()
declare -a CANDIDATE_LABELS=()

add_candidate() {
  local candidate="${1:-}"
  local label="${2:-$candidate}"
  if [[ -n "$candidate" ]]; then
    CANDIDATES+=("$candidate")
    CANDIDATE_LABELS+=("$label")
  fi
}

add_candidate "$D4C_PYTHON" "$D4C_PYTHON"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  add_candidate "$CONDA_PREFIX/bin/python" "\$CONDA_PREFIX/bin/python"
fi
if command -v python3 >/dev/null 2>&1; then
  add_candidate "$(command -v python3)" "python3"
fi
if command -v python >/dev/null 2>&1; then
  add_candidate "$(command -v python)" "python"
fi

CANDIDATES_TEXT="$(IFS='; '; printf '%s' "${CANDIDATE_LABELS[*]-}")"
VERSION_CHECK='import sys
v = sys.version_info
sys.stdout.write("%s.%s.%s" % (v[0], v[1], v[2]))
raise SystemExit(0 if (v[0], v[1]) >= (3, 8) else 42)'

for candidate in "${CANDIDATES[@]}"; do
  if [[ ! -x "$candidate" ]]; then
    continue
  fi
  version_output="$("$candidate" -c "$VERSION_CHECK" 2>/dev/null)" || continue
  PYTHON_BIN="$candidate"
  PYTHON_VERSION="$version_output"
  break
done

if [[ -z "$PYTHON_BIN" ]]; then
  write_launcher_runtime "python_discovery" "127" "" "" ""
  {
    echo "ODCR Codex Stop hook: no Python >= 3.8 interpreter found."
    echo "PATH=${PATH:-}"
    echo "CONDA_PREFIX=${CONDA_PREFIX:-}"
    echo "candidates=${CANDIDATES_TEXT:-none}"
    echo "selected none"
  } >&2
  exit 127
fi

export ODCR_HOOK_LAUNCH_CWD="$LAUNCH_CWD"
export ODCR_HOOK_REPO_ROOT="$REPO_ROOT"
export ODCR_HOOK_SELECTED_PYTHON="$PYTHON_BIN"
export ODCR_HOOK_SELECTED_PYTHON_VERSION="$PYTHON_VERSION"
export ODCR_HOOK_RUNTIME_PATH="$RUNTIME_PATH"
export ODCR_HOOK_RUNTIME_LAST_PATH="$RUNTIME_LAST_PATH"
export ODCR_HOOK_STDOUT_PATH="$STDOUT_PATH"
export ODCR_HOOK_STDERR_PATH="$STDERR_PATH"
export ODCR_HOOK_EVENT_NAME="${CODEX_HOOK_EVENT_NAME:-Stop}"
export ODCR_HOOK_CANDIDATES="$CANDIDATES_TEXT"

cd "$REPO_ROOT"
exec "$PYTHON_BIN" "$REPO_ROOT/.codex/hooks/odcr_post_edit_stop.py"
