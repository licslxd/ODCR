---
name: odcr-task-adaptive-sampler-cache-audit
description: Use when reviewing, designing, or modifying ODCR task-specific sampling, Gold/CF quality tiers, Step5 pool/sample-plan/token-cache reuse, sequence-length caps, or ablation protocols; decisions must adapt to each task's data distribution instead of reusing a fixed Task8 recipe.
---

# ODCR Task-Adaptive Sampler/Cache Audit

This skill is instruction-only. It must not call local scripts by itself.

## Required Reading

Read `docs/skills/ODCR_TASK_ADAPTIVE_SAMPLER_CACHE_SKILL.md` first.
For cache identity details, also read `docs/skills/ODCR_CACHE_SKILL.md`.

## Operating Rules

- Treat Task8 weak cross-platform CF as an example, not a universal rule.
- Decide from current task evidence: pool counts, quality tiers, posterior route
  fields, sample weights, token-length stats, and cache artifact boundaries.
- For Step5 sampling, enforce total-budget order: `effective_samples` first,
  then `target_gold / aux_gold / cf` component budgets, then tier fallback inside
  each component. CF fallback is High -> Medium -> capped Low_weighted; do not
  let tier scarcity collapse the CF component ratio.
- Check Step5 train/eval interface consistency when sampler or evidence changes:
  training input format, evidence source, and generation target must match final
  eval semantics, and token/sample-plan caches must rebuild when content changes.
- Prefer keep-with-weight or cap strategies for reliable Gold/CF evidence unless
  contamination, leakage, or a named diagnostic ablation justifies deletion.
- Separate ratio, Gold filtering, CF weighting, route policy, length cap, and
  cache identity into independently testable variables.
- If changing runtime behavior, follow One-Control and post-edit validation.
- Do not run training, eval, rerank, or GPU formal work unless the user
  separately authorizes it.
- For RACER-C1/retrieval-first Step5 resets, FLAN/T5/LoRA generation is deleted
  rather than kept as a baseline. The active path must be train-only evidence
  retrieval with RCR-aware contrastive alignment, provenance, cache identity,
  throughput/resource logs, and no hidden generator fallback.
- If old Step5 generator, multi-candidate rerank, or legacy alias code is found
  on the new active path, stop feature work and clean that path first.
