# ODCR_PAPER_CHAT_CODEX_SKILL

## 1. Role Split

Chat is the paper author and figure designer.

Codex is the repository operator, LaTeX applier, build checker, source-packet
collector, and handoff generator.

Codex must not freely rewrite paper prose, redesign figures, invent method
framing, or polish language unless the user explicitly asks Codex to do so.

## 2. Allowed Codex Work

- collect `AI_analysis/00_paper/paper_source_packet.md`
- apply exact Chat-authored LaTeX/TikZ content
- fix compile errors with minimal syntax-only edits
- update include paths
- run paper build/check
- report undefined citations, undefined references, and figure errors
- update claim ledger mechanically from Chat-approved claims
- generate `paper_review_packet.md`
- generate `paper_build_report.md`
- generate `paper_diff_summary.md`
- generate `paper_chat_handoff.md`
- generate root `paper.log`

## 3. Disallowed Codex Work

- free Method rewriting
- free Abstract/Introduction rewriting
- free figure design
- changing claims without Chat decision
- adding unsupported experiment numbers
- turning paper into an engineering report
- restoring old figures, tables, snapshots, or archives
- preserving unused files as archive/snapshot content inside active paper
  handoff
- running training, eval, preprocess, Step3, Step4, Step5, rerank, CUDA probes,
  or GPU jobs for paper-writing tasks

## 4. Active Paper Handoff Contract

`AI_analysis/00_paper/` must contain only:

- `README.md`
- `writing_decision_from_chat.md`
- `paper_source_packet.md`
- `paper_review_packet.md`
- `paper_chat_handoff.md`
- `paper_diff_summary.md`
- `paper_build_report.md`

No `archive/` or `snapshots/` directories are allowed in active paper handoff.

## 5. Figure Contract

Final paper figures use editable SVG sources and exported PDF vector figures.
The paper text should include the exported PDF files, not editable source files:

- `paper/figures/figure1_dynamic_causal_refinement.svg` ->
  `paper/figures/figure1_dynamic_causal_refinement.pdf`
- `paper/figures/figure2_odcr_architecture.svg` ->
  `paper/figures/figure2_odcr_architecture.pdf`
- `paper/figures/figure3_reliability_routing.svg` ->
  `paper/figures/figure3_reliability_routing.pdf`

Codex may check that SVG files exist, convert SVG to PDF, update LaTeX include
paths, build the paper, and report missing or visually unapproved figures.
Codex must not freely design, redraw, or reinterpret the figures.

Paper-only helper scripts belong under `paper/tools/`; `code/` is reserved for
ODCR runtime/project code.

TikZ coordinate tweaking is draft-only and is not the final figure workflow.
Legacy TikZ figure sources belong under `paper/figures/draft/` once their
replacement SVG/PDF assets exist and the paper has been switched to PDFs.

Every figure requires visual approval before moving to the next major figure.
If a required SVG or PDF is missing, Codex must report the missing asset and
must not force the paper body to reference a nonexistent PDF.

## 6. Review Flow

Default flow:

1. Codex collects source packet.
2. Chat writes one target `.tex` file.
3. Codex applies exactly.
4. Codex builds PDF and reports.
5. Chat reviews `paper.log`, not the full PDF by default.

PDF upload is required only for visual figure review, template review, complex
table review, or final submission check.

## 7. Mandatory paper.log Chat Handoff

At the end of every ODCR paper-writing task, Codex must generate or overwrite:

`/public/home/zhangliml/lc/ODCR/ODCR-main/paper.log`

`paper.log` is the single copy-paste handoff from Codex to Chat.

Always overwrite `paper.log` for the latest paper task. Do not append
endlessly. Do not ask the user to upload multiple `AI_analysis/00_paper` files
when `paper.log` can be produced.

`paper.log` must contain these sections in this order:

```text
# ODCR Paper Chat Handoff
- timestamp
- task name
- working directory
- binary recommendation

## What Codex Did
- completed edits/checks
- modified files
- created files
- deleted files

## What Codex Did Not Do
- explicitly state no training/eval/runtime work was run for paper-only tasks
- list requested items not completed

## Build Status
- build command
- PDF path
- pass/fail
- undefined citations
- undefined references
- missing figures
- serious overfull/underfull warnings
- page count if available

## Chat Review Needed
- exact questions Chat should answer next
- which file Chat should rewrite next
- whether PDF upload is needed

## Paper Source Packet
- paste or summarize AI_analysis/00_paper/paper_source_packet.md
- when next Chat task is a target .tex rewrite, include enough current source
  context for that file

## Paper Review Packet
- paste title, abstract, contribution list, target section text, figure
  captions, and any text Chat must review
- do not paste unnecessary full-paper text unless required

## Diff Summary
- paste or summarize paper_diff_summary.md

## Build Report
- paste or summarize paper_build_report.md

## Next Recommended Chat Action
- one clear action, e.g. "Chat should now rewrite
  paper/figures/figure1_dynamic_causal_refinement.tex."
```

Prohibited in `paper.log`:

- no huge raw git diff
- no unrelated training logs
- no old archive/snapshot content
- no stale paper status from previous rounds
- no missing build status
- no silent skip of `paper.log`

Use:

```bash
python paper/tools/paper_handoff.py --check
python paper/tools/paper_handoff.py
```
