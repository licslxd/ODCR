#!/usr/bin/env python3
"""Small helper commands for the ODCR Chat/Codex paper workflow."""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

PAPER_DIR = REPO_ROOT / "paper"
AI_PAPER_DIR = REPO_ROOT / "AI_analysis" / "00_paper"
DECISION_PATH = AI_PAPER_DIR / "writing_decision_from_chat.md"
HANDOFF_PATH = AI_PAPER_DIR / "paper_chat_handoff.md"
MAIN_TEX = PAPER_DIR / "main.tex"
MAIN_LOG = PAPER_DIR / "main.log"
MAIN_PDF = PAPER_DIR / "main.pdf"
TEX_ENV_TECTONIC = Path("/public/home/zhangliml/miniconda3/envs/tex/bin/tectonic")

CITE_RE = re.compile(
    r"\\(?:cite|citep|citet|citealp|citeauthor|citeyear|parencite|textcite)"
    r"(?:\[[^\]]*\])*\{([^}]*)\}"
)
INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]*)\}")
TODO_RE = re.compile(r"\b(?:TODO|FIXME|XXX)\b", re.IGNORECASE)
SKELETON_RE = re.compile(r"\b(?:TBD|skeleton|placeholder)\b", re.IGNORECASE)
BIB_KEY_RE = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)


def run_command(cmd: Sequence[str], cwd: Path | None = None) -> Tuple[int, str]:
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError as exc:
        return 127, str(exc)
    return proc.returncode, proc.stdout


def git_head() -> str:
    code, out = run_command(["git", "rev-parse", "HEAD"], REPO_ROOT)
    if code != 0:
        return f"unavailable: {out.strip()}"
    return out.strip()


def iter_paper_text_files() -> Iterable[Path]:
    if not PAPER_DIR.exists():
        return []
    suffixes = {".tex", ".bib", ".md", ".sty", ".cls"}
    return sorted(
        p
        for p in PAPER_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in suffixes
    )


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def write_paper_handoff_file(name: str, text: str) -> None:
    AI_PAPER_DIR.mkdir(parents=True, exist_ok=True)
    (AI_PAPER_DIR / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def paper_files() -> List[str]:
    if not PAPER_DIR.exists():
        return []
    keep = {".tex", ".bib", ".md", ".pdf"}
    return [
        str(p.relative_to(REPO_ROOT))
        for p in sorted(PAPER_DIR.rglob("*"))
        if p.is_file() and p.suffix.lower() in keep
    ]


def cite_keys_from_text(text: str) -> List[str]:
    keys: List[str] = []
    for match in CITE_RE.finditer(text):
        keys.extend(k.strip() for k in match.group(1).split(",") if k.strip())
    return keys


def refs_keys() -> List[str]:
    refs = PAPER_DIR / "refs.bib"
    return sorted(set(BIB_KEY_RE.findall(read_text(refs))))


def collect_status() -> dict:
    text_files = list(iter_paper_text_files())
    all_text = "\n".join(read_text(p) for p in text_files)
    main_text = read_text(MAIN_TEX)
    cite_keys = cite_keys_from_text(all_text)
    refs = set(refs_keys())
    table_files = []
    table_dir = PAPER_DIR / "tables"
    if table_dir.exists():
        table_files = [
            str(p.relative_to(REPO_ROOT))
            for p in sorted(table_dir.glob("*.tex"))
        ]
    decision_text = read_text(DECISION_PATH)
    decision_stripped = decision_text.strip()
    return {
        "repo_path": str(REPO_ROOT),
        "git_head": git_head(),
        "paper_files": paper_files(),
        "pdf_exists": MAIN_PDF.exists(),
        "pdf_path": str(MAIN_PDF),
        "decision_exists": DECISION_PATH.exists(),
        "decision_empty": not bool(decision_stripped),
        "decision_summary": summarize_text(decision_text, max_chars=1200),
        "active_cite_commands_count": len(CITE_RE.findall(all_text)),
        "active_cite_keys_count": len(cite_keys),
        "missing_cite_keys": sorted(set(cite_keys) - refs),
        "todo_count": len(TODO_RE.findall(all_text)),
        "skeleton_tbd_count": len(SKELETON_RE.findall(all_text)),
        "main_inputs": INPUT_RE.findall(main_text),
        "table_files": table_files,
    }


def summarize_text(text: str, max_chars: int = 1200) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "(empty)"
    summary = "\n".join(lines[:20])
    if len(summary) > max_chars:
        return summary[: max_chars - 3].rstrip() + "..."
    return summary


def print_status() -> int:
    status = collect_status()
    print(f"repo path: {status['repo_path']}")
    print(f"git head: {status['git_head']}")
    print("paper file list:")
    for path in status["paper_files"]:
        print(f"  - {path}")
    print(f"PDF exists: {status['pdf_exists']} ({status['pdf_path']})")
    print(
        "writing_decision_from_chat.md: "
        f"exists={status['decision_exists']} empty={status['decision_empty']}"
    )
    print(f"active cite commands count: {status['active_cite_commands_count']}")
    print(f"active cite keys count: {status['active_cite_keys_count']}")
    print(f"missing cite keys count: {len(status['missing_cite_keys'])}")
    if status["missing_cite_keys"]:
        print("missing cite keys:")
        for key in status["missing_cite_keys"][:40]:
            print(f"  - {key}")
    print(f"TODO count: {status['todo_count']}")
    print(f"skeleton/TBD count: {status['skeleton_tbd_count']}")
    print("current main.tex inputs:")
    for item in status["main_inputs"]:
        print(f"  - {item}")
    print("current table files:")
    for item in status["table_files"]:
        print(f"  - {item}")
    return 0


def grep_log(pattern: str) -> List[Tuple[int, str]]:
    if not MAIN_LOG.exists():
        return []
    rx = re.compile(pattern)
    hits: List[Tuple[int, str]] = []
    for line_no, line in enumerate(read_text(MAIN_LOG).splitlines(), start=1):
        if rx.search(line):
            hits.append((line_no, line.rstrip()))
    return hits


def build_with_tectonic() -> Tuple[str, int, str]:
    tectonic = shutil.which("tectonic")
    if not tectonic and TEX_ENV_TECTONIC.exists():
        tectonic = str(TEX_ENV_TECTONIC)
    if not tectonic:
        return "tectonic unavailable", 127, ""
    cmd = [tectonic, "--keep-logs", "--keep-intermediates", "main.tex"]
    code, out = run_command(cmd, PAPER_DIR)
    return "tectonic", code, out


def build_with_pdflatex() -> Tuple[str, int, str]:
    pdflatex = shutil.which("pdflatex") or "/usr/bin/pdflatex"
    bibtex = shutil.which("bibtex") or "/usr/bin/bibtex"
    if not Path(pdflatex).exists() and shutil.which("pdflatex") is None:
        return "pdflatex unavailable", 127, ""

    outputs: List[str] = []
    final_code = 0
    commands = [
        [pdflatex, "main.tex"],
        [bibtex, "main"],
        [pdflatex, "main.tex"],
        [pdflatex, "main.tex"],
    ]
    for cmd in commands:
        if cmd[0] == bibtex and not (Path(bibtex).exists() or shutil.which("bibtex")):
            outputs.append("bibtex unavailable; skipped")
            continue
        code, out = run_command(cmd, PAPER_DIR)
        outputs.append(f"$ {' '.join(cmd)}\n{out}")
        if code != 0 and final_code == 0:
            final_code = code
    return "pdflatex/bibtex", final_code, "\n".join(outputs)


def collect_diagnostics() -> List[Tuple[str, List[Tuple[int, str]]]]:
    checks = [
        ("Overfull \\\\hbox", r"Overfull \\hbox"),
        ("Undefined control sequence", r"Undefined control sequence"),
        ("LaTeX Error", r"LaTeX Error"),
        ("Citation.*undefined", r"Citation.*undefined"),
        ("Reference.*undefined", r"Reference.*undefined"),
    ]
    return [(label, grep_log(pattern)) for label, pattern in checks]


def print_diagnostics() -> List[Tuple[str, List[Tuple[int, str]]]]:
    diagnostics = collect_diagnostics()
    print(f"main.pdf exists: {MAIN_PDF.exists()} ({MAIN_PDF})")
    print(f"main.log exists: {MAIN_LOG.exists()} ({MAIN_LOG})")
    for label, hits in diagnostics:
        print(f"{label}: {len(hits)}")
        for line_no, line in hits[:20]:
            print(f"  {line_no}: {line}")
        if len(hits) > 20:
            print(f"  ... {len(hits) - 20} more")
    return diagnostics


def update_handoff_build_section(
    tool: str, code: int, diagnostics: List[Tuple[str, List[Tuple[int, str]]]]
) -> None:
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    diagnostic_lines = [f"- {label}: {len(hits)}" for label, hits in diagnostics]
    body_lines = [
        f"- Updated: {now}",
        f"- Build tool: {tool}",
        f"- Build exit code: {code}",
        f"- PDF exists: {MAIN_PDF.exists()}",
        f"- PDF path: `{MAIN_PDF}`",
        f"- Log path: `{MAIN_LOG}`",
        *diagnostic_lines,
        "- Training/eval/rerank: not run",
    ]
    body = "\n".join(body_lines)
    heading = "## Build Check"
    if HANDOFF_PATH.exists():
        current = read_text(HANDOFF_PATH).rstrip()
        if heading in current:
            current = current.split(heading, 1)[0].rstrip()
        text = f"{current}\n\n{heading}\n\n{body}\n"
    else:
        text = f"# ODCR Paper Chat Handoff\n\n{heading}\n\n{body}\n"
    write_paper_handoff_file("paper_chat_handoff.md", text)


def build_check() -> int:
    if not MAIN_TEX.exists():
        print(f"main.tex missing: {MAIN_TEX}")
        return 1
    tool, code, out = build_with_tectonic()
    if code == 127:
        print(tool)
        tool, code, out = build_with_pdflatex()
    print(f"build tool: {tool}")
    print(f"build exit code: {code}")
    if out:
        print("build output tail:")
        lines = out.splitlines()
        for line in lines[-80:]:
            print(line)
    diagnostics = print_diagnostics()
    update_handoff_build_section(tool, code, diagnostics)
    return 0


def handoff() -> int:
    status = collect_status()
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    old_handoff = REPO_ROOT / "AI_analysis" / "05_final_reports" / "paper_chat_handoff.md"
    lines = [
        "# ODCR Paper Chat Handoff",
        "",
        f"Updated: {now}",
        "",
        "## This Round",
        "",
        "- Workflow status refreshed by `python paper/tools/paper_ops.py handoff`.",
        "- No training, eval, rerank, 5-seed, longest-reference rebuild, "
        "or baseline adaptation was run.",
        "",
        "## Paper Files",
        "",
        f"- Paper file count: {len(status['paper_files'])}",
        f"- Main file: `paper/main.tex` exists={MAIN_TEX.exists()}",
        f"- Table file count: {len(status['table_files'])}",
        "",
        "## PDF Status",
        "",
        "- Compilation status: not freshly run by `handoff`; run "
        "`paper_ops.py build-check` for a fresh build check.",
        f"- PDF exists: {status['pdf_exists']}",
        f"- PDF path: `{status['pdf_path']}`",
        "",
        "## Chat Decision Status",
        "",
        f"- Decision file exists: {status['decision_exists']}",
        f"- Decision file empty: {status['decision_empty']}",
        "- Completed Chat decisions this round: none recorded by `paper_ops.py`.",
        "- Unfinished Chat decisions: provide a non-empty "
        "`writing_decision_from_chat.md` before manuscript rewriting.",
        "",
        "### Decision Summary",
        "",
        status["decision_summary"],
        "",
        "## Current Paper Story Summary",
        "",
        "- Codex did not rewrite or reinterpret the manuscript story in this handoff.",
        "- Chat remains responsible for the innovation story, causal framing, "
        "and academic wording decisions.",
        "",
        "## Evidence Status",
        "",
        "- Evidence was not recomputed.",
        "- Existing paper files and citation/table status were inspected only "
        "as static workflow state.",
        "",
        "## Citation Status",
        "",
        f"- Active cite commands: {status['active_cite_commands_count']}",
        f"- Active cite keys: {status['active_cite_keys_count']}",
        f"- Missing cite keys in refs.bib: {len(status['missing_cite_keys'])}",
    ]
    if status["missing_cite_keys"]:
        lines.append("- Missing keys: " + ", ".join(status["missing_cite_keys"][:30]))
    lines.extend(
        [
            "",
            "## TODO And Skeleton Status",
            "",
            f"- TODO/FIXME count: {status['todo_count']}",
            f"- TBD/skeleton/placeholder count: {status['skeleton_tbd_count']}",
            "",
            "## Tables",
            "",
        ]
    )
    if status["table_files"]:
        lines.extend(f"- `{path}`" for path in status["table_files"])
    else:
        lines.append("- No table files found.")
    lines.extend(
        [
            "",
            "## Risks",
            "",
            "- Empty Chat decision means Codex must not perform manuscript rewriting.",
            "- Citation verification still requires trusted metadata sources "
            "for any new BibTeX entries.",
            "- Build diagnostics should be checked after each paper edit.",
            "",
            "## Questions For Chat",
            "",
            "- What is the next approved innovation-first writing decision?",
            "- Which sections, tables, or citations should Codex update next?",
            "",
            "## Next Step",
            "",
            "- Put Chat's next decision in "
            "`AI_analysis/00_paper/writing_decision_from_chat.md`, then let "
            "Codex execute that decision in LaTeX.",
            "",
            "## Migration Note",
            "",
            f"- Old handoff path exists: {old_handoff.exists()}",
            "- Active handoff path is `AI_analysis/00_paper/paper_chat_handoff.md`.",
        ]
    )
    write_paper_handoff_file("paper_chat_handoff.md", "\n".join(lines))
    print(f"wrote {HANDOFF_PATH}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["status", "handoff", "build-check"])
    args = parser.parse_args(argv)

    if args.command == "status":
        return print_status()
    if args.command == "handoff":
        return handoff()
    if args.command == "build-check":
        return build_check()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
