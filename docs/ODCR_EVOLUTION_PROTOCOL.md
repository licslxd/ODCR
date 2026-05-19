# ODCR Evolution Protocol

ODCR is allowed to evolve. The rule is not to freeze the current
implementation; the rule is that every new control, field, artifact, and entry
must join the same control plane, contracts, lineage, and guardrails.

This protocol applies to every future Codex, AI, and human code change.

## 0. Required Codex Workflow

Future Codex requests should start from
`docs/CODEX_CHANGE_REQUEST_TEMPLATE.md`. Before editing code, Codex must:

1. Classify the change type.
2. Fill or mirror `docs/ODCR_FEATURE_INTEGRATION_CHECKLIST.md`.
3. Declare new parameters, fields, artifacts, entrypoints,
   model/loss/router/verbalizer paths, cache/checkpoint/export changes,
   eval/rerank changes, and legacy cleanup.
4. State whether old logic is deleted, migrated, retired/fail-fast, or moved to
   docs/history only.
5. State whether preprocess, Step3, Step4, Step5, eval, or rerank must be
   rerun.
6. Run post-edit validation after any `code/`, `configs/`, `docs/`, or `tools`
   change and before final delivery.
7. Write an `AI_analysis` ledger/summary/report before handoff.

The checklist is not optional. Future changes must not skip checklist coverage;
they may mirror the filled checklist in an `AI_analysis` ledger only when the
request explicitly avoids editing the checklist document itself.

## 1. New Public Parameters

Every new public parameter must enter the One-Control chain in one coherent
change:

```text
configs/odcr.yaml
-> code/odcr_core/config_schema.py
-> code/odcr_core/config_resolver.py
-> resolved payload
-> source table
-> tests
-> guardrail
```

Required implementation details:

- Define the YAML path first.
- Add schema validation and type normalization.
- Add resolver logic and resolved payload serialization.
- Add field-source reporting so `./odcr show` can identify where the value came
  from.
- Add `./odcr doctor` validation when the value affects architecture, paths,
  data contracts, DDP, or artifact compatibility.
- Add tests that prove the parameter is effective.
- Add or update static guardrail checks when the parameter closes a class of
  architecture risk.

Forbidden behavior:

- Hidden defaults that change active behavior without appearing in the resolved
  payload.
- Bare user environment variables as active sources.
- Child argparse flags that override resolved config.
- Loose YAML files or runtime `.env` files as active control surfaces.

For preprocess children, direct script execution is not a control plane. Batch,
chunk, shard, cache, precision, model, root, tokenizer, and GPU values must be
runtime transport from the One-Control resolved payload. Bare fallbacks such as
`EMBED_BATCH_SIZE` and `DOMAIN_CHUNK_BATCH_SIZE` are forbidden in formal paths.

## 2. New Data Fields

Every new cross-stage data field must enter the contract chain:

```text
data_contract/schema
-> producer
-> transport
-> consumer
-> manifest/index_contract
-> fingerprint
-> tests
-> guardrail
```

Required implementation details:

- Add the field to the canonical schema or data contract version.
- Identify the producing function and every consumer.
- Preserve transport through processed, split, merged, export, or manifest
  files as appropriate.
- Update `index_contract.json` or equivalent manifest metadata when the field
  crosses a stage boundary.
- Add the field or schema version to lineage fingerprints.
- Add mismatch handling that fails fast when a consumer sees an incompatible
  field set.
- Add positive tests for valid data and negative tests for missing, stale, or
  retired fields.

Forbidden behavior:

- Adding CSV columns that are not represented in the data contract.
- Reading a producer-private construction detail as a downstream primary input.
- Treating an internal-only construction field as a cross-stage CSV/export
  contract field.
- Silently accepting old field names as active fallbacks.

Preprocess field changes have an additional prior/posterior boundary: fields
produced before Step4 must be named and documented as priors. The unprefixed
`route_scorer` and `route_explainer` names are Step4 posterior route decisions
only; preprocess producers may emit only `preprocess_route_scorer_prior` and
`preprocess_route_explainer_prior`.

GPU stage additions must include an admission gate and child-side fail-fast
before model load. Silent CPU fallback is not allowed for formal GPU stages.
For ODCR preprocess_b/c and any BGE-large/CUDA probe, tmux is only the shared
session boundary, not a GPU allocation. The user creates or enters the admin
tmux with `tmux -L odcr_gpu new-session -A -s odcr`, then manually runs
`odcr-enter-gpu <JOBID>` inside that same tmux to enter the GPU node. Codex must
not execute `odcr-enter-gpu`, `srun`, `sbatch`, or `scancel`; must not create,
kill, or switch tmux sessions; and must not manage GPU allocation. Codex may
trust only the current tmux session's real-time CUDA environment. GPU use is
allowed by default for repo-local validation, probe, and bounded runtime after
fast sanity and current-pane validation. The controlled tmux GPU bridge
`python code/tools/odcr_tmux_gpu_bridge.py` targets a user-created,
already-entered, uniquely validated GPU pane with one bridge-generated command
file; this is not arbitrary send-keys and is no longer limited by a GPU
whitelist hard blocker. Bridge outputs stay under
`AI_analysis/01_raw_logs` or `AI_analysis/05_final_reports` by default. The
formal namespace guard remains mandatory. post-edit full is not a GPU
prerequisite, and runtime evidence takes priority over static full-suite
instability. If current tmux CUDA is not visible, fail fast and ask the user to
enter the GPU node in the same tmux, then rerun the probe. Do not infer cluster
GPU absence from a normal admin shell, a tmux still on admin, or old
`AI_analysis` probe output. Formal full train, complete preprocess_b/c,
Step4/Step5/eval/rerank, and downstream paper metrics still require explicit
user authorization.

## 3. New Scripts Or Entrypoints

Every active stage or workflow entry must enter through:

```text
./odcr
-> code/odcr.py
-> resolved config
-> executor/worker
```

New active `step*.sh`, `train*.sh`, `eval*.sh`, or
`scripts/entrypoints/*.sh` launchers are not allowed. A helper script may exist
only when it is not a user-visible ODCR entrypoint and cannot bypass the
resolved payload.

Required implementation details:

- Add the CLI command or subcommand in `code/odcr.py`.
- Resolve config before executing work.
- Write logs and metadata under `runs/.../meta`.
- Keep dataset artifacts under `data/` or `merged/`.
- Add post-edit validation through the narrowest applicable scope. Doctor and
  stage show/dry-run coverage are scope-owned checks, not universal defaults
  for every user-facing workflow.

## 4. New Losses, Routers, Or Verbalizers

Every new loss, router, verbalizer, or training-side control must enter the
training control chain:

```text
config block
-> schema
-> resolved payload
-> total loss single insertion point
-> logging
-> DDP graph-safe zero
-> tests
-> guardrail
```

Required implementation details:

- Define a named config block under the owning stage.
- Resolve all weights, thresholds, gates, schedules, and dimensions.
- Feed the executor from the resolved payload only.
- Insert the objective through the stage's single total-loss assembly point.
- Log the resolved value and active contribution.
- Use graph-safe zero tensors when masks are empty.
- Avoid rank-local `mask.any()` control flow that makes DDP graphs uneven.
- Test enabled, disabled, empty-mask, and DDP-relevant paths.

Forbidden behavior:

- New active losses that are computed but not included in total loss.
- New routers that bypass RCR or equivalent resolved route config.
- New verbalizer controls hidden in prompt text when an explicit control packet
  is the active contract.
- New hardcoded weights in executors.

## 5. New Cache, Checkpoint, Or Export Artifacts

Every reusable artifact must carry compatibility lineage:

```text
schema version
-> config hash
-> data contract/export contract version
-> input artifact fingerprint
-> model path fingerprint
-> consumer validation
-> mismatch fail-fast
```

Required implementation details:

- Define the artifact schema version.
- Record the resolved config hash or stage semantic fingerprint.
- Record the data contract/export contract version.
- Record input artifact hashes and source identities.
- Record local model/tokenizer/profile fingerprints when model artifacts affect
  outputs.
- Record task, source domain, target domain, and relevant dimensions.
- Make every consumer validate lineage before loading.
- Reject missing, stale, or mismatched lineage with an explicit rerun message.

Forbidden behavior:

- Best-effort cache reuse when lineage is missing.
- Loading old checkpoints or exports through silent compatibility shims.
- Writing artifacts without enough metadata to prove compatibility.

## 6. Logging And Artifact Evolution

ODCR does not prohibit new logs, new reports, metrics, caches, or AI-assisted
evidence. The governance rule is that every new output has a declared artifact
role, directory boundary, producer, consumer, retention policy, and display
level before it becomes active.

Required declaration for any new console output, file log, metrics file,
manifest, lineage file, cache, AI_analysis output, data artifact, or report:

- Artifact role: console, file log, metrics, manifest, lineage, cache,
  AI_analysis, data artifact, report, or another explicit role.
- Output directory and filename convention.
- Producer and consumer.
- Retention policy and whether `AI_analysis` may contain a digest/copy.
- Whether the output is default, verbose-only, debug-only, or file-only.
- Whether `meta/run_summary.json` must index the output.
- Whether the parent `latest.json` must be updated.
- Which guardrail and tests protect the boundary.
- Which post-edit scope validates the change, normally `logging` when the
  output touches logging/path/cache/AI_analysis rules.

Allowed evolution:

- New logs are allowed when they have an explicit role and run-meta or
  otherwise declared directory ownership.
- New reports are allowed when their producer, consumer, retention, and
  indexing behavior are declared.
- New metrics and caches are allowed when they keep canonical metric/cache
  boundaries and carry lineage/fingerprint metadata when reusable.

Step3 formal logging is governed by `odcr_step3_logging/2`. Future changes
must preserve the split: default `console.log` is compact human status,
`full.log` is the authoritative full log and contains launcher/raw child/detail
streams, `debug.log` is an auxiliary transport mirror, and `errors.log` carries
warning/error context with rank, local rank, pid, hostname, and run id where
available. Child runtime snapshots must write
`meta/training_runtime_config.json`; they must not overwrite parent
`meta/resolved_config.json`. `run_summary.json` must index both config files
and the authoritative full log.

Step3 structured metrics are first-class run-meta artifacts:
`metrics.jsonl`, `loss_breakdown.jsonl`, `timing_profile.jsonl`,
`gpu_profile.jsonl`, and `epoch_summary.csv`. Adding, renaming, or changing
these files requires a metrics artifact declaration, run_summary decision,
guardrail coverage, and targeted tests. AI_analysis may keep an audit digest,
but it must not become the metrics store or full-log mirror.

Default Step3 `show`, `dry-run`, and `source_table.json` are formal-only.
Backup, exploration, performance-probe, short-pilot, and historical rows must
stay hidden unless the user requests verbose/history detail. G2 must remain
`probe_only=true` and `formal_allowed=false` until a future One-Control change
explicitly promotes it; non-task2 task profiles must not inherit task2 ladder
roles.

Forbidden behavior:

- Real run logs must not be written into `data/` or `merged/`.
- `AI_analysis/` must not become a full training log mirror. It may hold audit
  raw logs for Codex/tooling, search hits, evidence ledgers, summaries, final
  reports, and handoff digests.
- Cache payloads must not be mixed into ordinary full logs.
- Default console output must remain summary-level; detailed resolved config,
  source table, per-batch/per-rank diagnostics, and per-rule guardrail PASS
  detail belong in files or verbose/debug display.
- Formal run handoff starts at `meta/run_summary.json`, and the parent
  `latest.json` points to that summary. New run-facing outputs must either be
  indexed there or explicitly declare why they are not.
- New log paths must not target top-level `logs/`, `code/log.out`, `data/`, or
  `merged/`.
- `odcr tail` must remain a new-layout reader only:
  `latest.json -> meta/run_summary.json -> meta/{console.log,full.log,errors.log}`.
  Do not restore compatibility fallback to top-level logs, `code/log.out`,
  `nohup*.log`, fallback/mirror logs, timestamp logs, or legacy shell logs. If a
  real new run breaks tail, repair the new run-summary/meta path that failed.

## 7. Old Logic Handling

Every change that replaces existing logic must choose one of four explicit
paths:

- Delete: remove the old logic and tests in the same change.
- Migrate: convert old state to the new contract once, then use only the new
  path.
- Retired/fail-fast: keep a compatibility stub that rejects use with a clear
  message.
- Docs/history only: move explanation to docs or archive with no execution
  path.

Forbidden behavior:

- Silent fallback from new logic to old logic.
- Long-term dual active mainlines.
- Keeping retired parameters as aliases for active controls.
- Reconnecting archived code, presets, shared YAML, or shell entrypoints to the
  main execution chain.

## 8. Required Design Note Before Large Changes

Large refactors are allowed. Before implementation, the change description must
state:

- What current logic is being replaced.
- Whether old logic will be deleted, migrated, retired/fail-fast, or moved to
  docs/history only.
- Which config paths, data fields, artifacts, lineage keys, and guardrails are
  affected.
- Which stages or outputs must be rerun after merge.

The design note can live in the PR, issue, `AI_analysis/`, or the feature
integration checklist, but it must exist before major code movement.

## 9. Static Evolution Guardrail IDs

The static guardrail has a long-term evolution block. Future features must not
weaken or bypass these checks:

- `R042`: active parameters must join YAML, schema, resolver, source table, and
  show/doctor, or be explicit constants/test-only controls.
- `R043`: active CSV/export fields must join the data contract, producer,
  consumer, manifest/index contract, and fingerprints, or be internal-only.
- `R044`: cache/checkpoint/export writers must write lineage/fingerprints and
  consumers must validate lineage before reuse.
- `R045`: active script entries must route through `./odcr` or
  `python code/odcr.py`.
- `R046`: new env reads must not become active config sources; only
  resolver-injected transport or fail-fast conflict checks are allowed.
- `R047`: active losses must be wired through the stage total-loss composer, or
  be documented no-op/test-only code.
- `R048`: active mask/gate branches must avoid rank-local `mask.any()` DDP graph
  divergence and use graph-tied zero for empty masks.
- `R049`: legacy aliases and old fields must be deleted, migrated,
  retired/fail-fast, or moved to docs/history, never used as silent fallback.
- `R050`: each new feature must fill or mirror the integration checklist or an
  `AI_analysis` ledger before handoff.
- `R051`: the unified post-edit validation script must exist at
  `code/tools/odcr_post_edit_check.py`.
- `R052`: `AGENTS.md` and `docs/CODEX_CHANGE_REQUEST_TEMPLATE.md` must require
  Codex to run the post-edit validation suite before final response, fix and
  rerun failures, avoid waiting for git commit, and include the Validation
  block.
- `R053`: post-edit validation scopes and dry-runs must not include real
  preprocess, training, Step4, Step5, eval, or rerank execution by default.
- `R054`: repo-local Codex Stop hook must exist and invoke the absolute stable
  wrapper path, not a cwd-sensitive `.codex/hooks/...` command, hardcoded
  `/usr/bin/python3`, or shell `git rev-parse` substitution.
- `R055`: the Codex hook wrapper must reject Python 2, prefer the D4C Python
  interpreter, delegate to the Python hook, keep Stop stdout JSON-only, and
  write bounded runtime diagnostics under
  `AI_analysis/01_raw_logs/codex_hooks`. Scope inference must use only
  current-session `transcript_path` or payload touched files; dirty workspace
  is not a post-edit signal, and `git status` must not be used for scope
  inference. Ignored runtime/audit artifacts such as `audit.log`,
  `AI_analysis/`, `runs/`, `cache/`, `artifacts/`, `data/`, `merged/`,
  `__pycache__/`, `.pytest_cache/`, `*.log`, and `*.pyc` must be filtered
  before scope inference. Missing/parse-failed/empty session evidence,
  dirty-workspace-only state, ignored-only files, and unknown session files
  must select `skip`; automatic `all` inference is degraded to
  `governance-fast` with a manual all follow-up instead of being executed by the
  Stop hook.
  Neither file may
  contain real preprocess, training, Step4, Step5, eval, or rerank commands.
- `R056`: docs must state that Codex Hooks or the manual post-edit check are
  the primary single-user workflow, while git hook / CI are optional.
- `R096`: GPU/tmux governance docs must state the admin tmux -> user manual
  `odcr-enter-gpu <JOBID>` -> current tmux real-time CUDA flow, forbid Codex
  GPU allocation management, allow runtime-first repo-local GPU validation,
  probe, and bounded runtime through the controlled tmux GPU bridge rather than
  arbitrary send-keys, remove GPU whitelist hard blockers and post-edit full
  GPU prerequisites, require AI_analysis/step3_validation validation outputs
  with a formal namespace guard, reject old admin probes as blockers, and keep
  formal full train / complete preprocess_b/c / Step4/Step5/eval/rerank behind
  explicit user confirmation.
- `R089`: `AGENTS.md` must require the narrowest applicable post-edit
  validation scope and must not prescribe fixed Step3 show/dry-run commands
  for all user-facing changes.
- `R060`: default executable-stage console output must remain summary-level;
  full resolved config and source-table detail belong in run-meta files or
  verbose/debug display.
- `R061`: active mainline logs must not default to `code/log.out` or top-level
  `logs/`; run logs belong under `runs/.../meta`.
- `R062`: verbose/debug display controls must not change resolved training
  payloads, fingerprints, losses, routers, or data-stage semantics.
- `R068`: new log, report, metrics, and cache outputs must declare an artifact
  role, output directory, producer, consumer, retention policy, and
  AI_analysis copy policy.
- `R069`: run-facing outputs must state whether `meta/run_summary.json` and
  the parent `latest.json` are updated, and must update them when they are the
  formal handoff entry.
- `R070`: `AI_analysis` must not become a full training log mirror.
- `R071`: default console output must not dump full resolved config, full
  source table, per-batch/per-rank detail, or per-rule guardrail PASS lists;
  those details belong in files or verbose/debug display.
- `R072`: new log paths must not target `data/`, `merged/`, top-level `logs/`,
  or `code/log.out`.
- `R078`: run logs must resolve to formal run meta directories:
  `runs/<stage>/<unit>/<run_id>/meta`.
- `R079`: reusable cache artifacts must target `cache/<producer>/<cache_key>`
  and must not be stored under run meta.
- `R080`: `AI_analysis` must not be an active full-log mirror; it stores audit
  material, ledgers, summaries, reports, and digests.
- `R081`: `data/` and `merged/` must not receive logs.
- `R082`: top-level logs, `code/log.out`, daemon launcher logs, nohup logs, and
  fallback mirror logs are retired as active defaults.
- `R083`: metrics/audit writers must use canonical filenames such as
  `metrics.jsonl`, `eval_metrics.json`, `rerank_summary.json`, and
  `data_audit_summary.csv`.
- `R084`: active code must not write default logs to top-level `logs/` or
  `code/log.out`.
- `R085`: `odcr tail` must resolve only through
  `latest.json -> run_summary.json -> meta/{console.log,full.log,errors.log}`.
- `R086`: active code must not fallback to `nohup*.log`, `fallback.log`,
  `mirror.log`, old timestamp logs, or legacy launcher log paths.
- `R087`: `AI_analysis` must not be used as an active training full-log mirror.
- `R088`: `data/` and `merged/` directories must not receive log files.
- `R063`-`R067` and `R073`-`R077`: the Stop hook fast path must no-op ignored-only changes,
  support `governance-fast`, skip no-session/dirty-only/unknown cases, keep
  automatic timeout at 180 seconds or less while manual deep checks may use 900
  seconds, write runtime diagnostics schema
  `odcr_codex_hook_runtime/2.2`, record
  `workspace_git_status_used_for_scope=false`, avoid legacy changed/raw/git
  fields, and keep `post_edit_command=null` whenever `selected_scope=skip`.

## 10. Verification Boundary

Architecture and governance changes must run static and lightweight checks.
Real data validation is a separate line of work.

Post-edit validation is mandatory after every Codex code/config/docs/tool
change. The recommended single-user automation is the repo-local Codex Hooks
Stop hook: when the project `.codex` layer is trusted, `.codex/hooks.json`
uses the absolute wrapper path
`/public/home/zhangliml/lc/ODCR/ODCR-main/.codex/hooks/odcr_post_edit_stop.sh`
at the Stop event. The wrapper verifies the repository root, rejects Python 2,
prefers the D4C Python interpreter, invokes
`.codex/hooks/odcr_post_edit_stop.py`, infers the scope from the Codex
`transcript_path` first, then payload touched files/tool outputs, and calls
`code/tools/odcr_post_edit_check.py` only when current-session evidence maps to
a known scope. Dirty workspace is not a post-edit signal; `git status` is not
used for scope inference and may only be recorded as an optional count. The
hook filters
`audit.log`, `AI_analysis/`, `AI_analysis/01_raw_logs/codex_hooks/`, `runs/`,
`cache/`, `artifacts/`, `data/`, `merged/`, Python cache directories, and
temporary/log/bytecode files before inference. Ignored-only changes select
`skip` with `post_edit_command=null` and do not call the checker.
Missing/parse-failed/empty transcripts, no payload touched files,
dirty-workspace-only state, and unknown session files also select `skip`.
Docs/governance hook changes select `governance-fast`; automatic Stop hook
inference must not execute `scope: all` inside the 180-second wrapper path.
If automatic inference or hook override resolves to `all`, the hook degrades to
`governance-fast` and records `manual_followup_required=true` with
`python code/tools/odcr_post_edit_check.py --scope all --max-seconds 900`.
Manual users may still run that explicit `all` deep validation command.
Automatic hook child checks default to `--max-seconds 120`; manual deep checks
may use `--max-seconds 900`.
Successful Stop hook stdout is JSON-only, and human-readable logs plus
`runtime_last.json` are written under
`AI_analysis/01_raw_logs/codex_hooks`. This does not require git commit. A git
hook / CI are optional insurance, not the primary gate. The primary gate is
Codex final-response-before-delivery validation: Codex must run the
task-scoped validation suite, fix failures,
rerun failed checks, and report the fixed Validation block before handing the
change to the user.

If Codex Hooks are unavailable or the project `.codex` layer is not trusted,
Codex must run the manual fallback before final delivery:

```bash
python code/tools/odcr_post_edit_check.py --scope <scope>
```

Future development flow is fixed: start from
`docs/CODEX_CHANGE_REQUEST_TEMPLATE.md`, classify the change, fill or mirror
`docs/ODCR_FEATURE_INTEGRATION_CHECKLIST.md`, then run the task-scoped
post-edit check immediately after edits. Real training or real data-stage work
still requires explicit user authorization.

Unified lightweight gate:

```bash
python code/tools/odcr_post_edit_check.py --scope <scope>
```

Available scopes are `governance-fast`, `docs`, `governance`, `config`,
`logging`, `preprocess`, `step3`, `step4`, `step5`, `eval`, and `all`.
`governance-fast` runs only governance-tool py_compile checks plus the strict
One-Control guardrail; it intentionally skips compileall, doctor, show,
stage dry-runs, full tests, and real training. The fuller scopes remain
lightweight and stage scopes use show/dry-run commands and tests only. Choose
the narrowest scope that matches the current-session touched files:
docs/governance changes use `governance-fast` or `governance`; logging,
path, tail, or AI_analysis policy uses `logging`; config/schema/resolver or
runner changes use `config`; stage changes use their owning stage scope; and
cross-stage contracts, manifests, lineage, cache/checkpoint hard gates, and
eval-rerank gates use the related stage scopes or `all` when one scope cannot
represent the impact. `--scope all` is a valid high-cost full-chain lightweight
check for final reclosure/release gates or manual deep validation; it is not
permanently banned. Automatic Stop hook all-scope inference is degraded to
`governance-fast` and records the manual all follow-up command instead of
executing all inside the 180-second wrapper path. It is inappropriate when a
narrow single-stage or docs-only change is misclassified into `all`.
Ignored-only, dirty-workspace-only, no-session, and unknown current-session
cases may select `skip` by hook. Step3 show/dry-run commands are required only
when the selected scope touches Step3, config changes that affect Step3, or
`all`.

Minimum manual governance check:

```bash
python code/tools/odcr_post_edit_check.py --scope governance-fast
```

Run stage dry-runs only when the current task allows them. Do not run
preprocess, training, Step4, Step5, eval, or rerank during documentation-only
governance work unless the task explicitly asks for it.

## Step3 S2-R Evolution Rules

Step3 cache, checkpoint, downstream compatibility, and performance-probe
changes must preserve the S2-R split:

- Tokenizer cache compatibility must use
  `odcr_step3_tokenizer_cache/2` and `tokenizer_cache_compat_hash`.
  Full resolved config, full source table, optimizer, batch, per-GPU batch,
  scheduler, logging, checkpoint cadence, no-accum batch semantics, and
  probe/pilot candidate names are record-only lineage and forbidden as cache
  reuse hard gates. Active `grad_accum`, `gradient_accumulation_steps`, and
  `accumulate_grad_batches` are retired fail-fast.
- Step3 checkpoint compatibility must use
  `odcr_step3_checkpoint_compat/2` with separate
  `semantic_model_compat_hash`, `data_contract_hash`,
  `artifact_lineage_hash`, `tokenizer_cache_compat_hash`,
  `train_runtime_config_hash`, `optimizer_config_hash`,
  `performance_profile_hash`, and `full_run_config_hash`.
  Step4/Step5/eval/rerank may reject semantic mismatches but must not reject a
  checkpoint solely because optimizer, learning rate, batch, per-GPU batch,
  scheduler, no-accum runtime metadata, or performance profile changed.
- Active Step3 defaults must remain no-accum, cross-rank structured-gather
  semantics. Task2 default is `1536/768` with
  `global_batch_size = per_gpu_batch_size * ddp_world_size`; future changes must update
  YAML, schema, resolver, show/doctor, tests, guardrail, docs, and the
  AI_analysis ledger together.
- Step3 paper tasks must use isolated `step3.task_profiles`; task2 remains the
  first paper task and must not be remapped to task1. Active performance
  candidates are G0/G1 only. G2 2048-pool lives under
  `step3.exploration_profiles` with `probe_only=true` and
  `formal_allowed=false`. Historical S1/S2/M*/N*/C* candidates must not return
  as active Step3 probe or formal candidates. Worker profiles W0-W4 remain CPU
  worker candidate surfaces.
- Governed `step3-performance-probe` and `step3-short-pilot` modes must keep a
  fixed command shape, AI_analysis-only outputs, no formal latest/checkpoint or
  formal cache writes, and no downstream-consumable pilot checkpoints.
## Aux Infrastructure Evolution

Auxiliary changes must extend the active aux registries instead of adding new
side tables. Static guardrail rules live in
`code/odcr_core/aux/governance/rule_registry.py`; post-edit scopes and reasons
live in `code/odcr_core/aux/governance/post_edit_registry.py`; runtime/tmux/GPU
allowlist entries live in `code/odcr_core/aux/runtime/command_registry.py`.

Any new runtime command needs a `RuntimeCommandSpec`, tests, AI_analysis output
through `code/odcr_core/aux/evidence/ai_analysis_writer.py`, and an explicit
formal namespace policy. It must not reintroduce arbitrary shell, repo-command,
repo-script, repo-module, command-file, or allocation paths.
