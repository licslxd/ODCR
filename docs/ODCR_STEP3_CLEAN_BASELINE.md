# ODCR Step3 Clean Baseline

This document is the only active tombstone for Step3 pre-formal legacy controls. The live control plane is `configs/odcr.yaml` plus `code/odcr_core/config_schema.py` and `code/odcr_core/config_resolver.py`; old Step3 keys are not compatibility aliases.

## Active Task2 G1 Baseline

- task: `2`, `AM_Movies -> AM_CDs`
- profile: `task2_strong_forward_g1`
- candidate: `G1`
- batch: `global_batch_size=1536`, `per_gpu_batch_size=768`, `ddp_world_size=2`
- semantics: `global_batch_size = per_gpu_batch_size * ddp_world_size`
- batch semantics version: `odcr_no_accum/1`
- gather: enabled, `local_gradient_context`
- optimizer: `AdamW`
- lr: `7e-4`
- precision: `bf16` with TF32 and AMP
- lengths: tokenizer/evidence `48/48`
- max grad norm: `0.5`
- scheduler: `warmup_cosine`
- tokenizer cache schema: `odcr_step3_tokenizer_cache/2`
- checkpoint sidecar schema: `odcr_step3_checkpoint_compat/2`

`task2_g0_backup` remains backup-only and manual-only. `task2_g2_effective_pool_2048` remains exploration-only, probe-only, and formal-forbidden. Task5, task8, and task7 keep isolated profile-ready entries; they do not inherit task2 labels or task2 performance evidence.

## Legacy Tombstone

| Old name | Why removed | Current replacement | Code remains |
|---|---|---|---|
| `grad_accum`, `gradient_accumulation_steps`, `accumulate_grad_batches`, accumulation aliases | Step3 uses structured representation losses; accumulation does not enlarge the same-forward sample pool and confuses optimizer/scheduler/logging/checkpoints | `global_batch_size = per_gpu_batch_size * ddp_world_size` | No active schema/resolver/parser/env path; retired names fail fast |
| `adv`, `eta`, `coef` as Step3 controls | Step3 loss is structured shared/specific disentanglement | `step3.structured_losses` and `step3.loss_semantics` | No Step3 active path |
| `Adam` | Formal baseline uses decoupled weight decay | `AdamW` | No Step3 optimizer branch |
| `fp32`, `fp16` Step3 precision | Formal baseline uses bf16 + TF32 + AMP | `bf16` backend block | No Step3 active fallback |
| `max_length=25`, `evidence_length=24`, 24/25 defaults | Old tokenizer/evidence contract | `48/48` for task2 G1 | No Step3 active default |
| `max_grad_norm=1.0`, `weight_decay=1e-5` | Old helper defaults | YAML-owned `0.5` and optimizer param groups | No Step3 active helper fallback |
| Old S/M/N/C/S1-C ladder | Pre-baseline candidate table | Isolated `task_profiles`, `backup_profiles`, `exploration_profiles` | No active ladder block |
| Non-gather formal mode | Formal Step3 requires cross-rank structured context | `cross_rank_structured_gather.enabled=true` | Diagnostic disable only requires explicit flag |
| G2 as formal replacement | G2 has not passed replacement gate | `exploration_only=true`, `formal_allowed=false` | Only isolated exploration profile |
| Exploration `formal_allowed=true` or `probe_only=false` | Exploration cannot become formal by config drift | `formal_allowed=false`, `probe_only=true` | Resolver invariant |
| Old preset/env/CLI Step3 controls | One-Control must be reproducible | `configs/odcr.yaml` plus resolver payload | Generic strict unknown-key rejection |

Old keys should appear only here, in negative tests, and in `AI_analysis` handoff material. Active show, dry-run, source table, schema, and Step3 child help must stay clean.
