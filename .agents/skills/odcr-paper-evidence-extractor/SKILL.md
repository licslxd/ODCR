---
name: odcr-paper-evidence-extractor
description: Extract concise ODCR paper evidence from runs, AI_analysis, and paper files for Chat review. Use this skill to summarize metrics, artifact sources, citation status, figure status, and claim risks without rewriting the manuscript.
---

# ODCR Paper Evidence Extractor

This skill is instruction-only. It must not include or call local scripts.

## Role

Collect short, Chat-facing evidence from existing `runs/`, `AI_analysis/`, and
`paper/` materials. This skill supports Chat decisions; it does not write the
paper story.

## Hard Rules

- Prefer short Chat-facing summaries over long audit dumps.
- Output must be readable by Chat.
- Evidence extraction must separate facts from recommendations.
- Never treat `AI_analysis` as a substitute for live artifacts when live
  artifacts are required.
- Do not rewrite manuscript prose.
- Do not modify the paper main text.
- Do not run training, eval, rerank, 5-seed experiments, longest-reference
  rebuilds, or baseline adaptation.
- Do not write or modify `runs/` formal artifacts.

## Evidence Categories

For each extraction task, report only the categories that are relevant:

- Metric source: exact source file or run artifact and whether it is live,
  archived, or paper-only.
- Artifact source: table, figure, PDF, ledger, or run metadata location.
- Citation TODO: missing or uncertain cite keys and the claim they support.
- Figure/table status: present, missing, stale, placeholder, or needs Chat
  decision.
- Claim risk: unsupported, over-broad, stale, citation-missing, or safe.

## Output Shape

Use a compact structure:

1. Facts found.
2. Evidence source paths.
3. Risks.
4. Recommendations for Chat.

Recommendations must be labeled as recommendations, not facts.
