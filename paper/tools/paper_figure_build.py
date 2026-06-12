#!/usr/bin/env python3
"""Build ODCR paper figure PDFs from editable SVG sources.

This helper only checks and converts figure files. It never edits SVG content
or creates replacement artwork.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIGURES_DIR = REPO_ROOT / "paper" / "figures"

FIGURES = (
    "figure1_dynamic_causal_refinement",
    "figure2_odcr_architecture",
    "figure3_reliability_routing",
)


@dataclass(frozen=True)
class ToolStatus:
    name: str
    available: bool
    command: str
    version: str
    converter: bool
    notes: str = ""


@dataclass(frozen=True)
class FigureStatus:
    name: str
    svg_path: Path
    pdf_path: Path
    svg_exists: bool
    pdf_exists_before: bool
    pdf_exists_after: bool
    converted: bool
    converter: str
    status: str
    message: str


def run_command(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError as exc:
        return 127, str(exc)
    return proc.returncode, proc.stdout.strip()


def summarize_version(name: str, output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "(no version output)"
    preferred_prefixes = {
        "inkscape": ("Inkscape",),
        "rsvg-convert": ("rsvg-convert",),
        "convert": ("Version:",),
        "cairosvg": ("CairoSVG", "cairosvg"),
        "python -m cairosvg": ("CairoSVG", "cairosvg"),
    }
    for prefix in preferred_prefixes.get(name, ()):
        for line in lines:
            if line.startswith(prefix):
                return line
    return lines[0]


def version_for(name: str, cmd: list[str]) -> tuple[bool, str]:
    code, output = run_command(cmd)
    if code != 0:
        return False, output or f"exit code {code}"
    return True, summarize_version(name, output)


def cairosvg_status() -> ToolStatus:
    path = shutil.which("cairosvg")
    if path:
        ok, version = version_for("cairosvg", [path, "--version"])
        return ToolStatus("cairosvg", ok, path, version, ok)

    ok, version = version_for(
        "python -m cairosvg",
        [sys.executable, "-m", "cairosvg", "--version"],
    )
    if ok:
        return ToolStatus("cairosvg", True, f"{sys.executable} -m cairosvg", version, True)
    return ToolStatus("cairosvg", False, "cairosvg", version, False)


def collect_tools() -> list[ToolStatus]:
    tools: list[ToolStatus] = []

    path = shutil.which("inkscape")
    if path:
        ok, version = version_for("inkscape", [path, "--version"])
        tools.append(ToolStatus("inkscape", ok, path, version, ok))
    else:
        tools.append(ToolStatus("inkscape", False, "inkscape", "not found", False))

    path = shutil.which("rsvg-convert")
    if path:
        ok, version = version_for("rsvg-convert", [path, "--version"])
        tools.append(ToolStatus("rsvg-convert", ok, path, version, ok))
    else:
        tools.append(ToolStatus("rsvg-convert", False, "rsvg-convert", "not found", False))

    tools.append(cairosvg_status())

    path = shutil.which("convert")
    if path:
        ok, version = version_for("convert", [path, "--version"])
        tools.append(
            ToolStatus(
                "convert",
                ok,
                path,
                version,
                False,
                "checked for availability only; not used by default for vector PDF export",
            )
        )
    else:
        tools.append(ToolStatus("convert", False, "convert", "not found", False))

    return tools


def rsvg_convert(tool: ToolStatus, svg_path: Path, pdf_path: Path) -> tuple[int, str]:
    return run_command([tool.command, "-f", "pdf", "-o", str(pdf_path), str(svg_path)])


def inkscape_convert(tool: ToolStatus, svg_path: Path, pdf_path: Path) -> tuple[int, str]:
    code, output = run_command(
        [
            tool.command,
            str(svg_path),
            "--export-type=pdf",
            f"--export-filename={pdf_path}",
        ]
    )
    if code == 0:
        return code, output
    old_code, old_output = run_command(
        [tool.command, str(svg_path), f"--export-pdf={pdf_path}"]
    )
    if old_code == 0:
        return old_code, old_output
    return code, output + ("\n" if output and old_output else "") + old_output


def cairosvg_convert(tool: ToolStatus, svg_path: Path, pdf_path: Path) -> tuple[int, str]:
    parts = tool.command.split()
    return run_command([*parts, str(svg_path), "-o", str(pdf_path)])


def imagemagick_convert(tool: ToolStatus, svg_path: Path, pdf_path: Path) -> tuple[int, str]:
    return run_command([tool.command, str(svg_path), str(pdf_path)])


Converter = Callable[[ToolStatus, Path, Path], tuple[int, str]]


def converter_order(
    tools: Iterable[ToolStatus], allow_imagemagick: bool
) -> list[tuple[ToolStatus, Converter]]:
    by_name = {tool.name: tool for tool in tools if tool.available}
    order: list[tuple[ToolStatus, Converter]] = []
    if "rsvg-convert" in by_name:
        order.append((by_name["rsvg-convert"], rsvg_convert))
    if "inkscape" in by_name:
        order.append((by_name["inkscape"], inkscape_convert))
    if "cairosvg" in by_name:
        order.append((by_name["cairosvg"], cairosvg_convert))
    if allow_imagemagick and "convert" in by_name:
        order.append((by_name["convert"], imagemagick_convert))
    return order


def build_one(
    name: str,
    figures_dir: Path,
    converters: list[tuple[ToolStatus, Converter]],
    dry_run: bool,
) -> FigureStatus:
    svg_path = figures_dir / f"{name}.svg"
    pdf_path = figures_dir / f"{name}.pdf"
    svg_exists = svg_path.is_file()
    pdf_before = pdf_path.is_file()

    if not svg_exists:
        return FigureStatus(
            name=name,
            svg_path=svg_path,
            pdf_path=pdf_path,
            svg_exists=False,
            pdf_exists_before=pdf_before,
            pdf_exists_after=pdf_before,
            converted=False,
            converter="",
            status="missing_svg",
            message="SVG source is missing; PDF was not generated.",
        )

    if dry_run:
        return FigureStatus(
            name=name,
            svg_path=svg_path,
            pdf_path=pdf_path,
            svg_exists=True,
            pdf_exists_before=pdf_before,
            pdf_exists_after=pdf_before,
            converted=False,
            converter="dry-run",
            status="dry_run",
            message="SVG exists; conversion skipped by --dry-run.",
        )

    if not converters:
        return FigureStatus(
            name=name,
            svg_path=svg_path,
            pdf_path=pdf_path,
            svg_exists=True,
            pdf_exists_before=pdf_before,
            pdf_exists_after=pdf_before,
            converted=False,
            converter="",
            status="no_converter",
            message="No SVG-to-PDF converter is available.",
        )

    messages: list[str] = []
    for tool, converter in converters:
        code, output = converter(tool, svg_path, pdf_path)
        if code == 0 and pdf_path.is_file():
            return FigureStatus(
                name=name,
                svg_path=svg_path,
                pdf_path=pdf_path,
                svg_exists=True,
                pdf_exists_before=pdf_before,
                pdf_exists_after=True,
                converted=True,
                converter=tool.name,
                status="converted",
                message="PDF generated from SVG.",
            )
        detail = output.strip() or f"exit code {code}"
        messages.append(f"{tool.name}: {detail}")

    return FigureStatus(
        name=name,
        svg_path=svg_path,
        pdf_path=pdf_path,
        svg_exists=True,
        pdf_exists_before=pdf_before,
        pdf_exists_after=pdf_path.is_file(),
        converted=False,
        converter="",
        status="conversion_failed",
        message="; ".join(messages),
    )


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def render_report(
    root: Path,
    figures_dir: Path,
    tools: list[ToolStatus],
    statuses: list[FigureStatus],
    dry_run: bool,
    allow_imagemagick: bool,
) -> str:
    now = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    lines = [
        "# ODCR Paper Figure Build Report",
        f"- timestamp: {now}",
        f"- repository: {root}",
        f"- figures directory: {rel(figures_dir, root)}",
        f"- dry run: {str(dry_run).lower()}",
        f"- imagemagick fallback enabled: {str(allow_imagemagick).lower()}",
        "",
        "## Converter Tools",
        "| tool | available | command | version | notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for tool in tools:
        lines.append(
            f"| {tool.name} | {'yes' if tool.available else 'no'} | "
            f"`{tool.command}` | {tool.version.replace('|', '/')} | {tool.notes or ' '} |"
        )

    lines.extend(
        [
            "",
            "## Figure Status",
            "| figure | SVG | PDF before | PDF after | status | converter | message |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for status in statuses:
        lines.append(
            f"| {status.name} | {'present' if status.svg_exists else 'missing'} | "
            f"{'present' if status.pdf_exists_before else 'missing'} | "
            f"{'present' if status.pdf_exists_after else 'missing'} | "
            f"{status.status} | {status.converter or ' '} | "
            f"{status.message.replace('|', '/')} |"
        )

    missing_svg = [status.name for status in statuses if not status.svg_exists]
    missing_pdf = [status.name for status in statuses if not status.pdf_exists_after]
    failed = [status.name for status in statuses if status.status == "conversion_failed"]
    lines.extend(
        [
            "",
            "## Summary",
            f"- SVG files present: {len(statuses) - len(missing_svg)}/{len(statuses)}",
            f"- PDF files present after run: {len(statuses) - len(missing_pdf)}/{len(statuses)}",
            f"- missing SVG: {', '.join(missing_svg) if missing_svg else 'none'}",
            f"- missing PDF after run: {', '.join(missing_pdf) if missing_pdf else 'none'}",
            f"- conversion failures: {', '.join(failed) if failed else 'none'}",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument(
        "--figure",
        action="append",
        choices=FIGURES,
        help="Build one figure; may be repeated. Defaults to all required figures.",
    )
    parser.add_argument("--dry-run", action="store_true", help="check only; do not convert")
    parser.add_argument(
        "--allow-imagemagick",
        action="store_true",
        help="allow ImageMagick convert as a last-resort PDF fallback",
    )
    parser.add_argument("--report", type=Path, help="optional Markdown report path")
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="exit nonzero if any required SVG/PDF is missing after the run",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    figures_dir = args.figures_dir
    if not figures_dir.is_absolute():
        figures_dir = root / figures_dir
    figures_dir = figures_dir.resolve()

    names = list(args.figure) if args.figure else list(FIGURES)
    tools = collect_tools()
    converters = converter_order(tools, allow_imagemagick=args.allow_imagemagick)
    statuses = [
        build_one(name, figures_dir, converters, dry_run=args.dry_run) for name in names
    ]
    report = render_report(
        root=root,
        figures_dir=figures_dir,
        tools=tools,
        statuses=statuses,
        dry_run=args.dry_run,
        allow_imagemagick=args.allow_imagemagick,
    )

    if args.report:
        report_path = args.report
        if not report_path.is_absolute():
            report_path = root / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
    else:
        print(report, end="")

    if any(status.status == "conversion_failed" for status in statuses):
        return 1
    if args.require_all and any(
        (not status.svg_exists) or (not status.pdf_exists_after) for status in statuses
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
