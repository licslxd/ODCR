---
name: odcr-paper-build-check
description: Build the ODCR paper PDF and check LaTeX diagnostics, including fatal errors, undefined citations, undefined references, and overfull boxes. Use this after paper edits.
---

# ODCR Paper Build Check

This skill is instruction-only. It must not include or call local scripts.

## Role

Build `paper/main.tex`, inspect diagnostics, and summarize the result in the
Chat handoff. Do not change the paper story.

## Default Build Command

Use the TeX environment when available:

```bash
conda activate /public/home/zhangliml/miniconda3/envs/tex
cd /public/home/zhangliml/lc/ODCR/ODCR-main/paper
tectonic --keep-logs --keep-intermediates main.tex
```

## Fallback

If `tectonic` is unavailable, fall back to local TeX tools when present:

```bash
/usr/bin/pdflatex main.tex
/usr/bin/bibtex main
/usr/bin/pdflatex main.tex
/usr/bin/pdflatex main.tex
```

## Required Diagnostics

After a build attempt, inspect `paper/main.log` with:

```bash
grep -n "Overfull \\hbox" main.log || true
grep -n "Undefined control sequence" main.log || true
grep -n "LaTeX Error" main.log || true
grep -n "Citation.*undefined" main.log || true
grep -n "Reference.*undefined" main.log || true
```

## Hard Rules

- Report PDF status and diagnostics.
- Do not modify the manuscript story.
- Do not add citations or claims while building.
- Do not run ODCR training, eval, rerank, or data-stage commands.
- Build failure is a paper validation finding, not permission to rewrite the
  paper narrative.
