---
name: odcr-paper-chat-handoff
description: Generate the concise Chat-facing ODCR paper handoff after a Codex paper task. The handoff must summarize what changed, PDF status, evidence used, unresolved risks, and questions for Chat.
---

# ODCR Paper Chat Handoff

This skill is instruction-only. It must not include or call local scripts.

## Required Output

Every Codex paper task must overwrite:

`AI_analysis/00_paper/paper_chat_handoff.md`

## Required Contents

The handoff must contain:

1. What happened this round.
2. Which paper files changed.
3. Whether compilation succeeded.
4. PDF path.
5. Completed Chat decisions.
6. Unfinished Chat decisions and why.
7. Current paper story summary, summarized only and not rewritten.
8. Current evidence status.
9. Current citation status.
10. Current risks.
11. Questions needing Chat judgment.
12. Suggested next paper-writing step.

## Style Rules

- Keep it short, clear, and Chat-facing.
- Do not paste long logs.
- Do not copy full `AI_analysis` content.
- Do not write training recommendations.
- Do not invent paper narrative, claims, citations, or results.
