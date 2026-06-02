---
name: odcr-paper-academic-polish-operator
description: Apply Chat-approved academic wording to the ODCR paper. Use this to polish manuscript text, reduce engineering-report language, and enforce the approved innovation story without inventing new claims.
---

# ODCR Paper Academic Polish Operator

This skill is instruction-only. It must not include or call local scripts.

## Role

Apply academic wording that Chat has already approved. This skill can polish
LaTeX prose and reduce engineering-report language, but it cannot invent the
paper story.

## Hard Rules

- Do not invent the story.
- Do not add new theoretical claims.
- Do not independently decide innovation points or causal framing.
- Replace excessive engineering terms only according to Chat decision.
- Keep single-run caveat in experiments/limitations, not as the main abstract
  contribution.
- Step3, Step4, and Step5 may be mentioned as implementation mapping, not as
  primary paper contribution, unless Chat says otherwise.
- Do not add unsupported claims, fake citations, fake BibTeX, or fabricated
  results.
- Do not run training, eval, rerank, 5-seed experiments, longest-reference
  rebuilds, or baseline adaptation.

## Workflow

1. Read `AI_analysis/00_paper/writing_decision_from_chat.md`.
2. Identify the exact approved wording or direction.
3. Apply local LaTeX edits that implement that direction.
4. Preserve evidence boundaries and citation TODOs.
5. Update the Chat handoff with changed files, unresolved risks, and questions.

## Output Boundary

Academic polish may improve clarity, flow, and terminology. It must not
upgrade evidence, create a new contribution, or turn implementation artifacts
into the paper's central claim.
