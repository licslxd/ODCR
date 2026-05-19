# AI Project Canonical

This is the current canonical operating note for ODCR AI agents and internal
maintainers. Future Codex/AI code changes must read `AGENTS.md` first, then this
file, then `docs/ODCR_ARCHITECTURE_CONTRACT.md`.

All future Codex code-change requests should start from
`docs/CODEX_CHANGE_REQUEST_TEMPLATE.md`; do not ask Codex to freely edit ODCR
without first classifying the change and filling or mirroring the integration
checklist.

Future development flow:

1. Use `docs/CODEX_CHANGE_REQUEST_TEMPLATE.md`.
2. Codex classifies the change and fills or mirrors
   `docs/ODCR_FEATURE_INTEGRATION_CHECKLIST.md` before edits.
3. After edits, Codex automatically runs
   `python code/tools/odcr_post_edit_check.py --scope <scope>`.
4. Real preprocess, training, Step4, Step5, eval, and rerank runs require
   explicit user authorization.

After Codex modifies `code/`, `configs/`, `docs/`, or `tools`, Codex must run
post-edit validation before the final response. Git commit hooks are optional
insurance only; the primary gate is the final-response-before-delivery
validation block. Real preprocess, training, Step4, Step5, eval, and rerank
runs require explicit user authorization.

## Governance Documents

The active governance set is:

- `AGENTS.md`: non-negotiable agent and developer rules.
- `docs/CURRENT_PROJECT_STATE.md`: the only active human-readable project
  state entry, generated from stage status/latest truth rather than historical
  notes.
- `docs/ODCR_ACTIVE_ARCHITECTURE.md`: current active mainline only.
- `docs/ODCR_ARCHITECTURE_CONTRACT.md`: binding One-Control architecture
  contract.
- `docs/ODCR_EVOLUTION_PROTOCOL.md`: required flow for future changes.
- `docs/ODCR_FEATURE_INTEGRATION_CHECKLIST.md`: template to fill before adding
  features, fields, artifacts, entries, losses, routers, or verbalizers.
- `docs/CODEX_CHANGE_REQUEST_TEMPLATE.md`: copyable request template that
  forces classify-first, checklist-first Codex execution.

The static guardrail is a mandatory gate for architecture/config changes:

```bash
python code/tools/check_one_control_guardrails.py --strict
```

Post-edit validation records the chosen scope, commands, PASS/FAIL results,
real-training decision, and rerun requirement in the Codex final response and
the relevant `AI_analysis` evidence.

Real run validation and code governance are separate lines of work. Governance
docs and static checks do not imply running preprocess, training, Step4, Step5,
eval, or rerank. Real data validation must be requested and tracked separately.

## Current Mainline

- Unique shell entrypoint: `./odcr`
- Unique Python entrypoint: `code/odcr.py`
- Unique primary config: `configs/odcr.yaml`
- Static guardrail: `python code/tools/check_one_control_guardrails.py --strict`
- Runtime doctor: `./odcr doctor`

Legacy `scripts/run_stage.sh` is deleted. `presets/` is no longer a live
mainline config tree; historical presets live only under
`_archive/legacy_presets_20260424/`.

Run handoff metadata is canonicalized around `meta/run_summary.json`. The
summary points to `meta/resolved_config.json`, `meta/source_table.json`, logs,
metrics, lineage, manifest, and key artifacts; it must not inline the full
config. Each `runs/{stage}/task{T}/` or `runs/preprocess/{unit}/` directory
keeps `latest.json` with `latest_summary_path`. New code must not write
`config_resolved.json`, `resolved_config_snapshot.json`, or
`config_snapshot.json` as mainline outputs. Latest lookup is pointer-only:
missing or damaged `latest.json -> meta/run_summary.json` fails fast; code must
not scan run directories, probe old `runs/task<T>/...` layouts, or synthesize
dry-run latest values.
Stage truth is finalized per run at `meta/stage_status.json`, and formal
downstream handoff must pass through `odcr_core.upstream_resolver`. Historical
docs and `AI_analysis/` reports are not live state sources.

Current code-state governance after the run2 repair is:

- Active runtime tree: `code/`
- Reference baseline only: `code2/`
- Paper-original reference only: `code1/`
- Current Step3 task2 truth source: `runs/step3/task2/2`
- Step4 upstream checkpoint binding:
  `stage_status.selected_checkpoint`

`code2/` must not be used as a direct overwrite source or runtime fallback.
`code1/` must not be used as a recovery target. For Step4 admission,
`best.pth` and `latest.pth` are secondary consistency aliases only; the primary
checkpoint is the selected checkpoint in stage status or accepted eval handoff.
Run2 frozen training config may differ from the current live Step3 config for
future runs. This is recorded as live-vs-frozen drift and must not cause current
config hashes to masquerade as run2 checkpoint architecture truth.

Default run console output is intentionally compact and mirrored to
`meta/console.log`. Detailed launcher and training diagnostics live in
`meta/full.log`, warnings/errors in `meta/errors.log`, raw captured child output
in `meta/debug.log`, and sample-level text in `meta/samples.jsonl` when
emitted. Full config/source ownership stays in `meta/resolved_config.json` and
`meta/source_table.json`. `--verbose` and `--debug` expand display only; they
do not change the resolved training payload.
`./odcr tail` follows only the formal handoff chain:
`latest.json -> meta/run_summary.json -> meta/{console.log,full.log,errors.log}`.
Default tail reads `console.log`; `--full` reads `full.log`; `--errors` reads
`errors.log`. The retired layouts `logs/`, `code/log.out`, `nohup*.log`,
fallback/mirror logs, timestamp logs, and legacy shell logs are not compatibility
fallbacks. If a true new run exposes a tail error, fix the new run-summary/meta
path that failed.

Run/cache/audit/data roots are separate. `runs/` stores formal run logs,
metrics, manifests, lineage, and summaries. `cache/` stores reusable cache
payloads such as `cache/preprocess_b/<cache_key>/` and
`cache/preprocess_c/<cache_key>/`. Step4 encoded cache and Step5 tokenization
cache hits require `cache_manifest.json` with schema, source content hash,
resolved config hash, tokenizer fingerprint, upstream lineage hash, max length,
required-field hash, and producer code version; dataset markers, path-only
keys, or mtime-only keys are not enough. `AI_analysis/` stores Codex raw audit
logs, search hits, evidence ledgers, phase summaries, final reports, and
handoff digests, but not copied full training logs. `data/` and `merged/`
store only their data-contract artifacts. New artifact kinds must be registered
by role, directory, filename convention, producer, consumer, retention policy,
and AI_analysis copy policy.

Future logging/output additions are allowed, but they must declare whether they
are console, file log, metrics, manifest, lineage, cache, AI_analysis, data
artifact, report, or another explicit role. They must also declare whether the
output is default, verbose-only, debug-only, or file-only, whether
`meta/run_summary.json` indexes it, whether `latest.json` changes, and which
guardrail/test protects the boundary.

## Configuration Rules

All public parameters must flow through:

```text
CLI --set > configs/odcr.yaml > code/odcr_core/config_resolver.py safe defaults
```

Do not add loose YAML files, runtime `.env` files, hidden argparse training
knobs, or shell wrappers as configuration surfaces. Global data roots, model
roots, Step5 text model path, sentence embed model path, and `embed_dim` are
also One-Control values in `configs/odcr.yaml`; legacy `ODCR_*` variables may
only appear as resolver-injected transport and must fail-fast on conflict.
Manifest metadata follows the same rule: backbone `embed_dim` / `hidden_size`
must be read from the resolved config, lineage, manifest snapshot, or an
explicit resolved value, never from a bare user `ODCR_*` environment fallback.

New parameters must update:

- `configs/odcr.yaml`
- `code/odcr_core/config_schema.py`
- `code/odcr_core/config_resolver.py`
- `./odcr show`
- `./odcr doctor`
- tests

Batch terms are fixed:

```text
global_batch_size / batch_size = effective optimizer-step train batch
per_gpu_batch_size = per-GPU forward/backward train batch
micro_batch_size = display alias only for per_gpu_batch_size
All active ODCR train stages: global_batch_size = per_gpu_batch_size * ddp_world_size
batch_semantics_version = odcr_no_accum/1
```

`grad_accum`, `gradient_accumulation_steps`, and
`accumulate_grad_batches` are retired historical concepts and are rejected on
the active config/CLI/env path.

## Path Rules

- Data artifacts: `data/`
- Merged task artifacts: `merged/`
- Logs, manifests, resolved configs, status: `runs/.../meta`
- AI audit and intermediate materials: `AI_analysis/`
- Historical preset material: `_archive/legacy_presets_*`

The mainline must not read `_archive/legacy_presets_*` or any restored
`presets/` tree.

## Stage Rules

Stages are exposed only through `./odcr` / `code/odcr.py`:

- `./odcr preprocess a`
- `./odcr preprocess b`
- `./odcr preprocess c`
- `./odcr step3 --task <id>`
- `./odcr step4 --task <id>`
- `./odcr step5 --task <id>`
- `./odcr eval --task <id>`

Do not use `presets/`. Do not use `scripts/run_stage.sh`. Do not add
`step*.sh`, `train*.sh`, `eval*.sh`, or `scripts/entrypoints/*.sh`.
The deleted legacy modules `code/odcr_core/config_loader.py`,
`code/odcr_core/training_preset_resolve.py`,
`code/odcr_core/stage_context.py`, `code/odcr_core/step3_runtime.py`,
`code/odcr_core/step3_registry.py`, and `code/tools/async_eval_daemon.py`
are negative/history names only and must not be restored as active files,
empty shells, compatibility shims, or call-time fail-fast stubs.

Preprocess live semantics are canonical evidence only. Processed, split, and
merged CSV contracts contain the core review/rating fields plus
`content_evidence`, `style_evidence`, `polarity_anchor`,
`domain_style_anchor`, `local_style_residual_hint`, `content_anchor_score`,
`style_anchor_score`, `evidence_quality_prior`,
`preprocess_route_scorer_prior`, and `preprocess_route_explainer_prior`;
split/merged transport adds `user_idx` and `item_idx`, and
merged transport adds `domain`. Retired detail columns `content_keywords`,
`content_aspects`, `content_entities`, `style_markers`, `template_family`, and
`length_style_bucket` may exist only inside preprocess construction logic or
negative/history notes. They are not cross-stage contract columns.

Step3 live semantics are structured shared/specific disentanglement with
evidence/prototype geometry. The historical typed bridge is retired; active
Step3 tests and orchestration must resolve through `configs/odcr.yaml` and
`code/odcr_core/config_resolver.py`. Structured shared/specific loss weights,
including orthogonal, evidence alignment, prototype, invariance/separation,
residual, and light explainer weights, are One-Control values under
`step3.structured_losses`; the Step3 train loop must not carry active literal
weights.

Step4 live semantics are RCR routing semantics. `odcr_routing_train.csv` and
same-directory `index_contract.json` are the Step4 -> Step5 contract. Step4
must preserve `evidence_quality_prior` as the preprocess-side prior, and must
write posterior RCR fields separately: `content_retention_score`,
`style_shift_score`, `rating_stability_score`, `cf_reliability_score`,
`uncertainty_score`, `confidence_bucket`, `route_scorer`, `route_explainer`,
`train_keep`, and `sample_weight_hint`. Decoder entropy and text hygiene are
auxiliary signals only; they must not become the primary Step4 router again.
Preprocess route hints are prior-only; Step4 may retain them as
`preprocess_route_scorer_prior` and `preprocess_route_explainer_prior`, while
`route_scorer` / `route_explainer` in the Step4 export always mean posterior
route decisions. Step4 RCR weights, thresholds, confidence buckets, train-keep
policy, sample-weight policy, and export required fields are owned by
`configs/odcr.yaml` under `step4.rcr`.

Step4 evidence levels are mandatory. CPU preview is `E1_schema_preview`: it
uses proxy diagnostics for schema/contract preview and is not tuning evidence.
CPU preview cannot rank RCR candidates, write `best_candidate.yaml`, produce a
patch suggestion, support machine verdict A, or support a formal Step4 prompt.
CUDA/tmux probe availability is only `E3_gpu_transport`; it is not Step4
posterior runtime evidence. Only `E4_gpu_shard_forward_bounded` or
`E5_formal_full_run` evidence may support Step4 RCR candidate ranking,
best-candidate selection, patch suggestions, or verdict-A eligibility. The old
`C9_balanced_quantile` and `C9_bucket_balanced` CPU-preview candidates are
superseded, and formal Step4 remains blocked until real GPU E4 validation
completes.

Current canonical status after the preprocess P0/P1 closure work is code and
metadata readiness only. It does not mean formal preprocess_a/b/c has already
been rerun. Fresh preprocess artifacts are still required before Step3/Step4/
Step5 admission.

preprocess_b/c are GPU-only formal stages, but tmux itself is not a GPU
allocation. The shared session is created or entered on admin with
`tmux -L odcr_gpu new-session -A -s odcr`; the user manually runs
`odcr-enter-gpu <JOBID>` inside that same tmux to enter the GPU node. Codex must
not execute `odcr-enter-gpu`, `srun`, `sbatch`, or `scancel`; must not create,
kill, or switch tmux sessions; and must not manage GPU allocation. Codex trusts
only the current tmux session's real-time CUDA environment. The `odcr-enter-gpu`
handoff is automatic and two-phase: admin-side tmux metadata is captured before
`srun`, GPU-side CUDA metadata is captured after `srun`, and the GPU-side phase
does not call tmux. The active bridge handoff is only
`AI_analysis/runtime/current_gpu_pane.json` with schema
`odcr_current_gpu_pane_handoff/2`; stale or invalid handoff state fails fast
instead of falling back to admin, and `AI_analysis/runtime/gpu_pane.json` is
historical hint material only. GPU use is allowed by default for repo-local
validation, probe, and bounded runtime after fast sanity and current-pane
validation. The controlled tmux GPU bridge at
`python code/tools/odcr_tmux_gpu_bridge.py` may target only a user-created,
already-entered, uniquely validated GPU pane and may send one bridge-generated
command file. It is not arbitrary send-keys and is no longer limited by a GPU
whitelist hard blocker. Bridge output stays under
`AI_analysis/01_raw_logs` or `AI_analysis/05_final_reports` by default, with a
mandatory formal namespace guard. post-edit full is not a GPU prerequisite, and
runtime evidence takes priority over static full-suite instability. A normal
admin shell without `nvidia-smi`, a
tmux still on admin, or old `AI_analysis` probe output must not be treated as
proof that the cluster has no GPU. If current tmux CUDA is unavailable, Codex
fails fast and asks the user to manually enter the GPU node in the same tmux,
then rerun the probe. Formal full train, complete preprocess_b/c,
Step4/Step5/eval/rerank, and downstream paper metrics still require explicit
user authorization; full
preprocess_b/c, complete stage experiments, Step3/Step4/Step5, eval/rerank, and
long benchmarks are outside Codex's GPU work boundary unless explicitly
authorized.

Step5 is split into two active innovation paths. Step5A is the scorer stability
path: it consumes Step4 posterior `route_scorer` rows through a scorer-clean
gate and applies LCI under UCI weights from `cf_reliability_score`,
`rating_stability_score`, `content_retention_score`, `uncertainty_score`,
`confidence_bucket`, and `sample_weight_hint`. Step5B is the explainer
verbalization path: it consumes `route_explainer` rows through a structured CCV
control packet built from content/style evidence, route decisions, reliability,
uncertainty, confidence, and sample weights. FCA aligns the scorer evidence
basis with the explainer evidence basis. Step5 parameters live under
`configs/odcr.yaml`: `step5.lci`, `step5.uci`, `step5.explainer_gate`,
`step5.ccv`, `step5.fca`, `step5.model`, and
`step5.train.explainer_loss_weight`. Step4 `sample_weight_hint` remains the
posterior base sample weight; `step5.explainer_gate.explainer_only_multiplier`
is only a Step5B training scheduling multiplier for explainer-only rows. CCV
control adapter dimensions and native LoRA controls are owned by `step5.ccv`,
including `step5.ccv.native_lora`. Retired `adv`, `eta`, `lambda_lci` /
`lambda_fca`, and prompt-concat controls are fail-fast legacy names, not active
Step5 configuration.

Step5 valid/test factual controls are separately labeled
`mode=factual_eval_default` under
`odcr_step5_factual_eval_control/1.0`. They only let factual target eval rows
build a neutral control packet; they are not RCR posterior, not train routes,
and not Step4 export posterior. Missing posterior controls on
`odcr_routing_train.csv` remain a fail-fast Step5 train error.

## Lineage And Cache Gates

ODCR does not silently reuse cross-stage artifacts. Preprocess
`skip_completed`, preprocess_b/c caches, Step3 checkpoints, Step4 exports,
Step5 checkpoints, and eval/rerank outputs must carry Phase 4A lineage:
One-Control resolved config fingerprints, preprocess/export schema versions,
source artifact hashes, local model/tokenizer fingerprints, embed/profile
dimensions, and task/domain identity. Consumers validate lineage before loading
the artifact; old v2.x preprocess outputs, missing checkpoint sidecars, stale
Step4 index contracts, and old eval/rerank schemas are rejected with a rerun
requirement.

## Future Evolution Protocol

Future changes must follow `docs/ODCR_EVOLUTION_PROTOCOL.md`.

- New public parameters must update YAML, schema, resolver, resolved payload,
  source table, tests, and guardrail.
- New data fields must update the data contract/schema, producer, transport,
  consumer, manifest or index contract, fingerprints, tests, and guardrail.
- New active scripts or entries must route through `./odcr` or `code/odcr.py`.
- New losses, routers, or verbalizers must have a config block, schema,
  resolved payload, total-loss insertion point, logging, DDP graph-safe zero
  handling, tests, and guardrail coverage.
- New caches, checkpoints, and exports must carry schema version, config hash,
  contract version, input artifact fingerprints, model path fingerprints,
  consumer validation, and mismatch fail-fast behavior.
- `R042`-`R053` are the long-term static evolution guardrails for future
  parameters, fields, artifact writers/consumers, entries, env reads, losses,
  mask/gate branches, legacy cleanup, checklist or `AI_analysis` ledgers, and
  post-edit validation workflow safety.
- `R068`-`R072` are the logging artifact evolution guardrails for future
  logs, reports, metrics, caches, AI_analysis outputs, run_summary/latest
  indexing decisions, console summary boundaries, and banned log destinations.

Large refactors are allowed when the change explicitly states whether old logic
is deleted, migrated, retired/fail-fast, or moved to docs/history only. Silent
fallback and long-term dual active paths are not allowed.

Before adding a feature, fill or mirror
`docs/ODCR_FEATURE_INTEGRATION_CHECKLIST.md`.

## Required Verification

For architecture/config changes, run the narrowest applicable post-edit
validation scope:

```bash
python code/tools/odcr_post_edit_check.py --scope <scope>
```

Use `governance-fast` or `governance` for docs/governance changes, `logging`
for logging/path/tail/AI_analysis policy, `config` for schema/resolver/runner
changes, owning stage scopes for preprocess/Step3/Step4/Step5/eval changes,
and related scopes or `all` for cross-stage contracts, manifests, lineage,
cache/checkpoint hard gates, and eval-rerank gates. `--scope all` is also valid
for explicit multi-business-stage changes, final reclosure/release gates, or
manual deep validation; it is not permanently banned, only inappropriate when a
narrow task maps cleanly to a smaller scope. Ignored-only, dirty-workspace-only,
no-session, and unknown current-session hook cases may select `skip`. Step3
show/dry-run commands are required only for Step3 scope, config changes that
affect Step3, or `all`, not for every user-facing change. Real training still
requires explicit user authorization.

Also confirm legacy shell entrypoints remain absent.
## Active Aux Runtime

`code/odcr_core/aux/` is the active auxiliary infrastructure tree. It owns
runtime/tmux/GPU bridge dispatch, governance registries, AI_analysis writing,
artifact path policy, and runtime CLI facades. `./odcr` remains the only user
entrypoint; `./odcr runtime ...` is a subcommand, not a second control plane.

Codex/tmux/GPU validation must use `./odcr runtime bridge discover`,
`validate-only`, `marker-probe`, `cuda-probe`, or registered bounded probes
such as `./odcr runtime probe --stage step5A --task 2 --bounded`. The bridge
records current-pane hostname, TMUX, SLURM_JOB_ID, CUDA_VISIBLE_DEVICES,
nvidia-smi, and torch CUDA evidence under AI_analysis. Tests write run-like
artifacts under `test_artifacts/`, not formal `runs/`.
