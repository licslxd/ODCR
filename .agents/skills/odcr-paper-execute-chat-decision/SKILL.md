---
name: odcr-paper-execute-chat-decision
description: Execute Chat-provided ODCR paper writing decisions from AI_analysis/00_paper/writing_decision_from_chat.md. Use this skill to apply approved narrative, section, table, figure, and citation changes to paper files without inventing new claims.
---

# ODCR Paper Execute Chat Decision

This skill is instruction-only. It must not include or call local scripts.

## Role

Codex implements paper decisions that Chat has already made. The required input
is:

`AI_analysis/00_paper/writing_decision_from_chat.md`

The required output handoff is:

`AI_analysis/00_paper/paper_chat_handoff.md`

## Hard Rules

- Chat decision file is the highest-priority writing source.
- Codex is not the paper storyteller.
- Do not invent innovation story.
- Do not independently choose the paper narrative, causal framing, or main
  contribution language.
- Do not convert implementation stages into claimed contributions unless Chat
  explicitly says so.
- Do not promote Step3, Step4, or Step5 into paper contributions unless the
  Chat decision file explicitly asks for it.
- Do not add unsupported claims.
- Do not add fake citations or fake BibTeX.
- Do not touch `runs/`, `configs/`, checkpoints, latest pointers, formal
  run summaries, or training artifacts.
- Do not run training, eval, rerank, 5-seed experiments, longest-reference
  rebuilds, or baseline adaptation.

## Workflow

1. Read `AI_analysis/00_paper/writing_decision_from_chat.md` first.
2. If the file is missing or empty, stop manuscript rewriting and produce only a
   status/handoff report.
3. Map every requested change to specific `paper/` files before editing.
4. Apply only the approved narrative, section, table, figure, and citation
   changes.
5. If Chat decisions conflict with the current manuscript, follow Chat and
   record the conflict in the handoff.
6. Keep evidence boundaries explicit when a requested change depends on
   metrics, citations, or table values.
7. Update `AI_analysis/00_paper/paper_chat_handoff.md` before handoff.

## Safe Output

Codex may produce LaTeX edits, concise evidence summaries, citation TODOs,
build diagnostics, and Chat-facing handoffs. It must not redesign the paper
story or create new results.
