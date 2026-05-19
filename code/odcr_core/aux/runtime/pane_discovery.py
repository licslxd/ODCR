"""Tmux pane discovery for ODCR GPU runtime."""

from __future__ import annotations

import os
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


TMUX_PANE_FORMAT = "\t".join(
    (
        "#{session_name}",
        "#{window_index}",
        "#{window_name}",
        "#{pane_index}",
        "#{pane_id}",
        "#{pane_pid}",
        "#{pane_current_command}",
        "#{pane_current_path}",
        "#{pane_active}",
        "#{pane_dead}",
        "#{pane_in_mode}",
        "#{pane_title}",
        "#{pane_tty}",
        "#{pane_start_command}",
    )
)

TMUX_SESSION_FORMAT = "\t".join(("#{session_name}", "#{session_id}", "#{session_windows}", "#{session_attached}"))
TMUX_WINDOW_FORMAT = "\t".join(
    (
        "#{session_name}",
        "#{window_index}",
        "#{window_name}",
        "#{window_active}",
        "#{window_panes}",
    )
)


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class SubprocessRunner:
    def run(self, args: Sequence[str], *, timeout: float | None = None) -> CommandResult:
        proc = subprocess.run(list(args), text=True, capture_output=True, check=False, timeout=timeout)
        return CommandResult(tuple(str(part) for part in args), proc.returncode, proc.stdout, proc.stderr)


@dataclass(frozen=True)
class PaneCandidate:
    socket: str
    session: str
    target: str
    window_index: str
    window_name: str
    pane_index: str
    pane_id: str
    pane_pid: int
    pane_command: str
    cwd: str
    active: bool
    dead: bool
    in_mode: bool
    pane_title: str = ""
    pane_tty: str = ""
    pane_start_command: str = ""
    cwd_match_repo: bool = False
    command_class: str = "unknown"
    last_visible_line_hash: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "socket": self.socket,
            "session": self.session,
            "target": self.target,
            "window_index": self.window_index,
            "window_name": self.window_name,
            "pane_index": self.pane_index,
            "pane_id": self.pane_id,
            "pane_pid": self.pane_pid,
            "pane_command": self.pane_command,
            "cwd": self.cwd,
            "active": self.active,
            "dead": self.dead,
            "in_mode": self.in_mode,
            "pane_title": self.pane_title,
            "pane_tty": self.pane_tty,
            "pane_start_command": self.pane_start_command,
            "cwd_match_repo": self.cwd_match_repo,
            "command_class": self.command_class,
            "last_visible_line_hash": self.last_visible_line_hash,
        }


@dataclass(frozen=True)
class DiscoveryResult:
    candidates: tuple[PaneCandidate, ...]
    invalid: tuple[dict[str, Any], ...]
    sockets_considered: tuple[str, ...]
    sockets: tuple[dict[str, Any], ...] = ()
    sessions: tuple[dict[str, Any], ...] = ()
    windows: tuple[dict[str, Any], ...] = ()
    panes: tuple[PaneCandidate, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "candidates": [item.to_dict() for item in self.candidates],
            "invalid_candidates": list(self.invalid),
            "sockets_considered": list(self.sockets_considered),
            "sockets": list(self.sockets),
            "sessions": list(self.sessions),
            "windows": list(self.windows),
            "panes": [item.to_dict() for item in self.panes],
            "total_sockets_seen": len(self.sockets_considered),
            "total_sessions_seen": len(self.sessions),
            "total_windows_seen": len(self.windows),
            "total_panes_seen": len(self.panes),
        }


def _tmux_socket_from_env(value: str | None) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    first = text.split(",", 1)[0].strip()
    return Path(first) if first else None


def _append_socket_dir(paths: list[Path], root: Path) -> None:
    paths.extend([root / "odcr_gpu", root / "default"])
    if root.is_dir():
        paths.extend(_safe_socket_dir_entries(root))


def _codex_tmux_roots(uid: int) -> tuple[Path, ...]:
    roots: list[Path] = []
    try:
        home = Path.home()
        roots.append(home / "tmp" / "codex" / f"tmux-{uid}")
    except RuntimeError:
        pass
    roots.append(Path(f"/run/user/{uid}/codex-tmp/tmux-{uid}"))
    roots.append(Path(f"/run/user/{uid}/tmux-{uid}"))
    return tuple(roots)


def candidate_socket_paths(uid: int | None = None, *, all_sockets: bool = False) -> tuple[Path, ...]:
    uid = os.getuid() if uid is None else int(uid)
    paths: list[Path] = []
    explicit_socket = str(os.environ.get("ODCR_GPU_TMUX_SOCKET", "") or "").strip()
    if explicit_socket:
        paths.append(Path(explicit_socket))
    tmux_socket = _tmux_socket_from_env(os.environ.get("TMUX"))
    if tmux_socket is not None:
        paths.append(tmux_socket)
    for root in _codex_tmux_roots(uid):
        _append_socket_dir(paths, root)
    base = Path(f"/tmp/tmux-{uid}")
    _append_socket_dir(paths, base)
    if all_sockets:
        tmp = Path("/tmp")
        for root in _safe_glob(tmp, "tmux-*"):
            _append_socket_dir(paths, root)
        for root in _codex_tmux_roots(uid):
            parent = root.parent
            if parent.is_dir():
                for sibling in _safe_glob(parent, "tmux-*"):
                    _append_socket_dir(paths, sibling)
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return tuple(out)


def _safe_glob(root: Path, pattern: str) -> tuple[Path, ...]:
    try:
        return tuple(sorted(root.glob(pattern)))
    except OSError:
        return ()


def _safe_socket_dir_entries(root: Path) -> tuple[Path, ...]:
    out: list[Path] = []
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return ()
    for path in entries:
        try:
            if path.is_socket():
                out.append(path)
        except OSError:
            continue
    return tuple(out)


def _parse_bool(value: str) -> bool:
    return str(value).strip() == "1"


def _cwd_matches_repo(cwd: str, repo_root: Path) -> bool:
    if not str(cwd or "").strip():
        return False
    try:
        Path(cwd).resolve().relative_to(repo_root.resolve())
        return True
    except (OSError, ValueError):
        return False


def classify_command(command: str) -> str:
    normalized = str(command or "").strip().lower()
    base = normalized.rsplit("/", 1)[-1].lstrip("-")
    if base in {"bash", "zsh", "sh", "fish", "tcsh", "csh"}:
        return "shell"
    if base == "srun":
        return "srun"
    if base in {"python", "python3", "torchrun", "accelerate", "deepspeed"}:
        return "active_compute_app"
    if not base:
        return "unknown"
    return "other"


def _last_visible_line_hash(runner: SubprocessRunner, socket: Path, pane_id: str) -> str | None:
    result = runner.run(("tmux", "-S", str(socket), "capture-pane", "-p", "-t", pane_id, "-S", "-5"), timeout=3)
    if result.returncode != 0:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    return hashlib.sha256(lines[-1].encode("utf-8", errors="replace")).hexdigest()


def _candidate_from_row(
    socket: Path,
    row: str,
    *,
    repo_root: Path,
    runner: SubprocessRunner | None = None,
    capture_hash: bool = False,
) -> PaneCandidate | dict[str, Any]:
    parts = row.rstrip("\n").split("\t")
    if len(parts) == 10:
        session, window, pane, pane_id, pid, command, cwd, active, dead, in_mode = parts
        window_name = ""
        pane_title = ""
        pane_tty = ""
        pane_start_command = ""
    elif len(parts) == 14:
        (
            session,
            window,
            window_name,
            pane,
            pane_id,
            pid,
            command,
            cwd,
            active,
            dead,
            in_mode,
            pane_title,
            pane_tty,
            pane_start_command,
        ) = parts
    else:
        return {"socket": str(socket), "reason": "bad_tmux_row", "row": row}
    try:
        pane_pid = int(pid)
    except ValueError:
        return {"socket": str(socket), "target": f"{session}:{window}.{pane}", "reason": "bad_pane_pid"}
    line_hash = _last_visible_line_hash(runner, socket, pane_id) if runner is not None and capture_hash else None
    return PaneCandidate(
        socket=str(socket),
        session=session,
        target=f"{session}:{window}.{pane}",
        window_index=window,
        window_name=window_name,
        pane_index=pane,
        pane_id=pane_id,
        pane_pid=pane_pid,
        pane_command=command,
        cwd=cwd,
        active=_parse_bool(active),
        dead=_parse_bool(dead),
        in_mode=_parse_bool(in_mode),
        pane_title=pane_title,
        pane_tty=pane_tty,
        pane_start_command=pane_start_command,
        cwd_match_repo=_cwd_matches_repo(cwd, repo_root),
        command_class=classify_command(command),
        last_visible_line_hash=line_hash,
    )


def _invalid_candidate(candidate: PaneCandidate, reason: str) -> dict[str, Any]:
    payload = candidate.to_dict()
    payload["reason"] = reason
    return payload


def _parse_sessions(socket: Path, stdout: str) -> tuple[dict[str, Any], ...]:
    out: list[dict[str, Any]] = []
    for row in stdout.splitlines():
        parts = row.rstrip("\n").split("\t")
        if len(parts) != 4:
            out.append({"socket": str(socket), "reason": "bad_tmux_session_row", "row": row})
            continue
        name, session_id, windows, attached = parts
        out.append(
            {
                "socket": str(socket),
                "session": name,
                "session_id": session_id,
                "session_windows": windows,
                "session_attached": _parse_bool(attached),
            }
        )
    return tuple(out)


def _parse_windows(socket: Path, stdout: str) -> tuple[dict[str, Any], ...]:
    out: list[dict[str, Any]] = []
    for row in stdout.splitlines():
        parts = row.rstrip("\n").split("\t")
        if len(parts) != 5:
            out.append({"socket": str(socket), "reason": "bad_tmux_window_row", "row": row})
            continue
        session, index, name, active, panes = parts
        out.append(
            {
                "socket": str(socket),
                "session": session,
                "window_index": index,
                "window_name": name,
                "window_active": _parse_bool(active),
                "window_panes": panes,
            }
        )
    return tuple(out)


def _socket_exists(path: Path, exists: Callable[[Path], bool] | None) -> tuple[bool, str | None]:
    if exists is not None:
        try:
            return bool(exists(path)), None
        except PermissionError:
            return False, "permission_denied"
        except OSError as exc:
            return False, f"os_error:{exc.__class__.__name__}"
    try:
        if not path.exists():
            return False, "socket_missing"
        if not path.is_socket():
            return False, "not_socket"
        return True, None
    except PermissionError:
        return False, "permission_denied"
    except OSError as exc:
        return False, f"os_error:{exc.__class__.__name__}"


def discover_panes(
    *,
    runner: SubprocessRunner | None = None,
    socket_paths: Sequence[Path] | None = None,
    socket_exists: Callable[[Path], bool] | None = None,
    all_sockets: bool = False,
    include_filtered: bool = False,
    capture_hash: bool = False,
    repo_root: Path | None = None,
) -> DiscoveryResult:
    runner = runner or SubprocessRunner()
    repo = (repo_root or Path.cwd()).resolve()
    sockets = tuple(socket_paths) if socket_paths is not None else candidate_socket_paths(all_sockets=all_sockets)
    candidates: list[PaneCandidate] = []
    invalid: list[dict[str, Any]] = []
    socket_inventory: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    panes: list[PaneCandidate] = []
    for socket in sockets:
        socket_record: dict[str, Any] = {"socket": str(socket), "exists": False, "list_errors": []}
        exists_ok, exists_reason = _socket_exists(socket, socket_exists)
        if not exists_ok:
            reason = exists_reason or "socket_missing"
            socket_record["reason"] = reason
            socket_inventory.append(socket_record)
            invalid.append({"socket": str(socket), "reason": reason})
            continue
        socket_record["exists"] = True
        session_result = runner.run(("tmux", "-S", str(socket), "list-sessions", "-F", TMUX_SESSION_FORMAT), timeout=5)
        if session_result.returncode != 0:
            socket_record["list_errors"].append(
                {"command": "list-sessions", "reason": "list_failed", "stderr": session_result.stderr.strip()}
            )
        else:
            parsed_sessions = _parse_sessions(socket, session_result.stdout)
            sessions.extend(item for item in parsed_sessions if "reason" not in item)
            invalid.extend(item for item in parsed_sessions if "reason" in item)
        window_result = runner.run(("tmux", "-S", str(socket), "list-windows", "-a", "-F", TMUX_WINDOW_FORMAT), timeout=5)
        if window_result.returncode != 0:
            socket_record["list_errors"].append(
                {"command": "list-windows", "reason": "list_failed", "stderr": window_result.stderr.strip()}
            )
        else:
            parsed_windows = _parse_windows(socket, window_result.stdout)
            windows.extend(item for item in parsed_windows if "reason" not in item)
            invalid.extend(item for item in parsed_windows if "reason" in item)
        result = runner.run(("tmux", "-S", str(socket), "list-panes", "-a", "-F", TMUX_PANE_FORMAT), timeout=5)
        if result.returncode != 0:
            socket_record["list_errors"].append(
                {"command": "list-panes", "reason": "list_failed", "stderr": result.stderr.strip()}
            )
            socket_record["reason"] = "list_failed"
            socket_inventory.append(socket_record)
            invalid.append({"socket": str(socket), "reason": "tmux_list_panes_failed", "stderr": result.stderr.strip()})
            continue
        for row in result.stdout.splitlines():
            parsed = _candidate_from_row(socket, row, repo_root=repo, runner=runner, capture_hash=capture_hash)
            if isinstance(parsed, PaneCandidate):
                panes.append(parsed)
                if parsed.dead:
                    invalid.append(_invalid_candidate(parsed, "pane_dead"))
                elif parsed.in_mode:
                    invalid.append(_invalid_candidate(parsed, "pane_in_mode"))
                else:
                    candidates.append(parsed)
                if include_filtered and (parsed.dead or parsed.in_mode):
                    continue
            else:
                invalid.append(parsed)
        socket_record["session_count"] = len([item for item in sessions if item.get("socket") == str(socket)])
        socket_record["window_count"] = len([item for item in windows if item.get("socket") == str(socket)])
        socket_record["pane_count"] = len([item for item in panes if item.socket == str(socket)])
        socket_inventory.append(socket_record)
    return DiscoveryResult(
        tuple(candidates),
        tuple(invalid),
        tuple(str(path) for path in sockets),
        tuple(socket_inventory),
        tuple(sessions),
        tuple(windows),
        tuple(panes),
    )


def select_unique_pane(
    *,
    socket: str | None = None,
    target: str | None = None,
    runner: SubprocessRunner | None = None,
    socket_paths: Sequence[Path] | None = None,
    socket_exists: Callable[[Path], bool] | None = None,
) -> tuple[PaneCandidate, DiscoveryResult]:
    paths = (Path(socket),) if socket else socket_paths
    discovery = discover_panes(runner=runner, socket_paths=paths, socket_exists=socket_exists)
    candidates = list(discovery.candidates)
    if target:
        candidates = [candidate for candidate in candidates if candidate.target == target or candidate.pane_id == target]
    if len(candidates) != 1:
        raise RuntimeError(
            "Fresh tmux GPU pane discovery did not resolve exactly one runnable pane "
            f"(candidates={len(candidates)}, sockets={len(discovery.sockets_considered)}). "
            "This is a bridge/pane discovery ambiguity, not proof that CUDA is unavailable."
        )
    return candidates[0], discovery
