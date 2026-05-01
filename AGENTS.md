# ODCR Agent Rules

This file is mandatory reading for every Codex/GPT/AI coding agent and every
developer before changing ODCR. These are architecture rules, not suggestions.

## Non-Negotiable One-Control Rules

1. User-visible entrypoints are only:
   - `./odcr`
   - `python code/odcr.py`
2. The only primary configuration file is `configs/odcr.yaml`.
3. Do not add or restore `presets/` as a main configuration source.
4. Do not add runtime `.env` files as a main configuration source.
5. Do not add `step*.sh`, `train*.sh`, `eval*.sh`, or `scripts/entrypoints/*.sh`
   as user-visible ODCR entrypoints.
6. Any new public parameter must update all of these in the same change:
   - `configs/odcr.yaml`
   - `code/odcr_core/config_schema.py`
   - `code/odcr_core/config_resolver.py`
   - `./odcr show`
   - `./odcr doctor`
   - relevant tests
6a. Global roots/model paths/embed dimension are One-Control values:
   `project.data_dir`, `project.merged_dir`, `env.models_dir`,
   `env.step5_text_model`, `env.sentence_embed_model`, and `env.embed_dim`.
   Legacy `ODCR_DATA_DIR`, `ODCR_MERGED_DATA_DIR`, `ODCR_MODELS_DIR`,
   `ODCR_STEP5_TEXT_MODEL`, `ODCR_SENTENCE_EMBED_MODEL`, and
   `ODCR_EMBED_DIM` must not override `configs/odcr.yaml`; child processes may
   only receive resolver-injected `ODCR_RESOLVED_*` transport values.
6b. Manifest metadata is also part of the One-Control surface. Backbone
   metadata such as `embed_dim` / `hidden_size` must come from resolved config,
   checkpoint lineage, manifest snapshots, or explicit resolved values; it must
   not read bare user `ODCR_*` environment variables.
7. Batch semantics are fixed:
   - `batch_size` = effective global train batch
   - `micro_batch_size` = per-GPU train batch
   - `grad_accum` = gradient accumulation steps
   - `batch_size == micro_batch_size * ddp_world_size * grad_accum`
8. Logs and run metadata belong under `runs/.../meta`.
9. Dataset artifacts belong under `data/` and `merged/`.
10. AI-assisted notes, audits, and generated analysis belong under `AI_analysis/`.
11. Never reconnect `_archive/legacy_presets_*` to the main execution chain.
12. If a future feature needs new configuration, extend `configs/odcr.yaml`
    first. Do not create loose YAML files, loose env files, or hidden side
    channels.
13. Preprocess processed/split/merged CSVs are canonical evidence contracts.
    Retired detail columns `content_keywords`, `content_aspects`,
    `content_entities`, `style_markers`, `template_family`, and
    `length_style_bucket` may only be preprocess-internal construction details
    or negative/history references. They must not be CSV output columns,
    contract fields, or downstream primary inputs.
14. Step3 live semantics are structured shared/specific disentanglement. Do not
    reconnect the retired typed bridge or old domain-adv labels to the mainline.
    Step3 structured loss weights are One-Control parameters under
    `configs/odcr.yaml: step3.structured_losses`; do not hardcode active
    evidence/prototype/HSS/orthogonal/shared-specific weights in the executor.
15. Step4 live semantics are RCR routing semantics. Do not collapse
    `content_retention_score`, `style_shift_score`, `rating_stability_score`,
    `cf_reliability_score`, `uncertainty_score`, `route_scorer`, or
    `route_explainer` back into a text entropy/filter exporter. Preserve
    `evidence_quality_prior` as a preprocess prior, not a Step4 posterior.
    Step4 export `route_scorer` / `route_explainer` are posterior decisions;
    preprocess hints may only survive as `preprocess_route_scorer_prior` /
    `preprocess_route_explainer_prior`.
16. Step4 RCR weights, thresholds, confidence buckets, train-keep policy,
    sample-weight policy, and export required fields are One-Control parameters
    under `configs/odcr.yaml: step4.rcr`; do not hardcode active values in the
    Step4 execution path.
17. Step5 live semantics are Step5A/Step5B dual paths: Step5A uses
    LCI/UCI for scorer stability on `route_scorer` samples; Step5B uses
    CCV/FCA for controlled explanation on `route_explainer` samples. Step5
    `lci`, `uci`, `explainer_gate`, `ccv`, `fca`, `model`, and
    `train.explainer_loss_weight` parameters are One-Control parameters under
    `configs/odcr.yaml: step5`; CCV adapter dimensions and Step5 native LoRA
    controls belong under `step5.ccv` / `step5.ccv.native_lora`. Step4
    `sample_weight_hint` is the posterior base sample weight; Step5
    `explainer_gate.explainer_only_multiplier` is only a Step5B training
    scheduling multiplier. Do not bypass these with hidden active defaults or
    retired `adv` / `eta` / `lambda_lci` / `lambda_fca` aliases.
18. Cross-stage reuse is gated by lineage fingerprints. Preprocess
    `skip_completed`, preprocess_b/c caches, Step3 checkpoints, Step4 exports,
    Step5 checkpoints, and eval/rerank outputs must record and validate current
    One-Control config, schema/contract, source artifacts, model artifacts, and
    task/domain lineage before reuse. Missing or mismatched lineage must
    fail-fast and require rerun; do not silently accept old v2.x preprocess
    artifacts, old Step3/Step4/Step5 checkpoints, or old eval/rerank schemas.
19. Formal run handoff starts at `meta/run_summary.json`, with the parent
    stage/task or preprocess/unit `latest.json` pointing to it. New resolved
    config and source-table outputs are only `meta/resolved_config.json` and
    `meta/source_table.json`; retired names `config_resolved.json`,
    `resolved_config_snapshot.json`, and `config_snapshot.json` are historical
    reads only and must not be newly written.
20. Preprocess route hints are prior-only. `preprocess_a` may output
    `preprocess_route_scorer_prior` and
    `preprocess_route_explainer_prior`; it must not output
    `route_scorer` or `route_explainer`. The unprefixed route fields belong
    only to Step4 posterior exports.
21. `preprocess_b/c` are GPU stages and must fail fast before BGE-large model
    load when CUDA is unavailable. Do not add default CPU fallback. The tmux
    session is not itself a GPU allocation: it is created or entered on the
    admin node with `tmux -L odcr_gpu new-session -A -s odcr`, then the user
    manually runs `odcr-enter-gpu <JOBID>` inside that same tmux to enter the
    GPU node. Codex must not execute `odcr-enter-gpu`, `srun`, `sbatch`, or
    `scancel`; must not create, kill, or switch tmux sessions; and must not
    manage GPU allocation. The narrow exception is the controlled tmux GPU
    bridge at `python code/tools/odcr_tmux_gpu_bridge.py`: Codex may use that
    tool to send one bridge-generated command to a user-created,
    already-entered, uniquely validated GPU pane, and only for whitelist short
    validation scripts. This is not arbitrary send-keys. Bridge output must
    live under `AI_analysis`, use mode-specific adaptive timeout, and default
    to `stop_after_first_valid_result`. Codex may only use the current tmux
    session's real-time CUDA environment. If CUDA is not visible in the current
    tmux, fail fast with: "Current tmux does not expose CUDA. Please manually
    run `odcr-enter-gpu <JOBID>` in this same tmux to enter the GPU node, then
    rerun the probe." Do not use a normal admin shell, a tmux session still
    sitting on admin, or old `AI_analysis` probe output as proof that the
    cluster has no GPU. User-authorized Codex GPU work is limited to
    <= 3 minutes of short probe, short benchmark, command smoke, or quick
    parameter comparison; do not run formal preprocess_b/c, complete
    preprocess stages, Step3/Step4/Step5, eval/rerank, or long benchmarks.

## Codex Long-Term Governance Constraints

These constraints apply to every future Codex/AI-assisted change, including
large refactors. ODCR may evolve, but changes must enter the unified control
plane, data contract, lineage contract, and guardrail system.

1. Do not bypass One-Control. Active behavior must flow through
   `configs/odcr.yaml`, `code/odcr_core/config_schema.py`, and
   `code/odcr_core/config_resolver.py`.
2. Do not add bare user environment variables as active configuration sources.
   Environment transport to children must use resolver-injected
   `ODCR_RESOLVED_*` values only.
3. Do not add active argparse parameters that override the resolved payload.
   New user-facing knobs must be YAML/schema/resolver/source-table values.
4. Do not add hidden hardcodes for active weights, thresholds, paths, model
   dimensions, routing decisions, loss switches, or artifact compatibility.
5. Do not add CSV/export fields without updating the data contract, producer,
   transport, consumers, manifest or index contract, fingerprints, tests, and
   guardrail.
6. Do not add cache, checkpoint, export, eval, or rerank artifacts without
   schema version, config hash, data/export contract version, input artifact
   fingerprint, model artifact fingerprint, consumer validation, and mismatch
   fail-fast behavior.
7. Do not add a new loss/router/verbalizer path without a config block, schema,
   resolved payload, total-loss single insertion point, logging, graph-safe
   zero behavior, tests, and guardrail.
8. Do not use rank-local `mask.any()` branches or optional-loss branches that
   can create uneven DDP graphs. Empty-mask paths must still participate with
   graph-safe zero tensors when needed.
9. Do not keep old fields, aliases, checkpoints, exports, or schemas as silent
   fallbacks. Retired names must fail fast, be migrated, be deleted, or appear
   only in docs/history/negative tests.
10. Large rewrites are allowed only after the change states how old logic will
    be deleted, migrated, retired/fail-fast, or moved to docs/history. Long-term
    dual active paths are not allowed.
11. Static evolution rules `R042`-`R056` are long-term gates, not one-off phase
    checks. New parameters, fields, scripts, losses, routers, verbalizers,
    caches, checkpoints, exports, env reads, DDP mask/gate paths, and legacy
    cleanup must pass those rules and fill or mirror the feature checklist or
    an `AI_analysis` ledger; post-edit validation workflow rules must also
    remain enforced. Logging governance rules `R060`-`R062` preserve
    summary-level default console output, run-meta file detail, and
    display-only verbose/debug semantics. Logging artifact evolution rules
    `R068`-`R072` require new logs, reports, metrics, caches, and AI_analysis
    outputs to declare role, directory, producer, consumer, retention,
    run_summary/latest impact, visibility, and guardrail/test coverage.
12. `odcr tail` is new-layout only:
    `latest.json -> run_summary.json -> meta/{console.log,full.log,errors.log}`.
    Do not restore fallback to top-level `logs/`, `code/log.out`,
    `nohup*.log`, fallback/mirror logs, old timestamp logs, or legacy shell
    logs. If a real new run breaks tail, fix the new-layout handoff.
13. `AI_analysis` is audit/handoff material, not a training full-log mirror;
    reusable cache payloads belong under `cache/`, and `data/` / `merged/` must
    not receive log files.

Before introducing a new feature, fill or mirror the checklist in
`docs/ODCR_FEATURE_INTEGRATION_CHECKLIST.md`. For protocol details, read
`docs/ODCR_ACTIVE_ARCHITECTURE.md` and
`docs/ODCR_EVOLUTION_PROTOCOL.md`.

## Mandatory Codex Change Workflow

Future Codex/AI-assisted code changes must use
`docs/CODEX_CHANGE_REQUEST_TEMPLATE.md` as the request shape. Codex must not
start writing code directly after receiving a feature/refactor/change request.
It must first classify the change under `docs/ODCR_EVOLUTION_PROTOCOL.md`, then
fill or mirror `docs/ODCR_FEATURE_INTEGRATION_CHECKLIST.md`; future changes
must not skip checklist coverage.

Before edits, Codex must declare whether the change introduces or modifies any
public parameter, data/export field, reusable artifact, user-visible entrypoint,
model/loss/router/verbalizer path, configuration control-plane surface,
cache/checkpoint/export behavior, logging/metrics/cache/report output,
AI_analysis output, eval/rerank behavior, or old-logic deletion/migration. New
or changed surfaces must name their One-Control, contract,
lineage/fingerprint, DDP/loss-graph, eval/rerank, logging/output role and
directory, guardrail, test, and documentation impacts.

Every replacement must state how old logic is handled: deleted, migrated,
retired/fail-fast, or moved to docs/history only. Silent fallback and long-term
dual active paths are not allowed. Every handoff must state whether any
preprocess, Step3, Step4, Step5, eval, or rerank rerun is required.

Every Codex change must write an AI_analysis ledger or summary that records
the classification, affected checklist rows, files changed, old-logic handling,
rerun decision, and lightweight verification results. During execution, Codex
may provide at most one interim status update unless the user explicitly asks
for more; final delivery should be one complete response.

## Mandatory Post-Edit Validation

After Codex modifies any `code/`, `configs/`, `docs/`, or `tools` surface, it
must immediately run the ODCR post-edit validation suite before the final
response. This is the primary handoff gate and does not wait for a git commit.
The recommended single-user automation is the repo-local Codex Hooks Stop hook:
when the project `.codex` layer is trusted, `.codex/hooks.json` runs
the absolute wrapper path
`/public/home/zhangliml/lc/ODCR/ODCR-main/.codex/hooks/odcr_post_edit_stop.sh`
at the Stop event. The wrapper verifies the repository root, rejects Python
2, prefers `/public/home/zhangliml/miniconda3/envs/D4C/bin/python`, then
invokes `.codex/hooks/odcr_post_edit_stop.py`, which calls
`code/tools/odcr_post_edit_check.py` with the inferred scope. The Python hook
must validate only current-session evidence: Codex `transcript_path` touched
files first, then payload touched files/tool outputs. Dirty workspace is not a
post-edit signal, and `git status` must not be used for scope inference; it may
only be recorded as an optional workspace dirty count in diagnostics. Ignored
runtime/audit artifacts such as
`audit.log`, `AI_analysis/`, `AI_analysis/01_raw_logs/codex_hooks/`, `runs/`,
`cache/`, `artifacts/`, `data/`, `merged/`, `__pycache__/`,
`.pytest_cache/`, and `*.log` / `*.pyc` files are filtered before scope
inference. If only ignored files changed, the hook no-ops and does not call the
post-edit checker. If no current-session touched files are available, if the
transcript is missing/parse-failed/empty, if only historical dirty workspace
state exists, or if current-session files are unknown, the hook selects `skip`
and does not call `odcr_post_edit_check.py`. Docs/governance hook changes use
`governance-fast`; `all` is allowed only for current-session touched files that
explicitly hit multiple business stages or an explicit `ODCR_HOOK_SCOPE=all`
override. Users may set `ODCR_HOOK_SCOPE=<scope>` to force a check, or manually
run `python code/tools/odcr_post_edit_check.py --scope governance` or
`python code/tools/odcr_post_edit_check.py --scope all` for deeper validation.
Automatic Stop hook checks pass `--max-seconds 180` by default; a manual deep
check may use `--max-seconds 900`. Successful Stop hook
stdout must remain JSON-only, for example `{"continue":true}`; human logs
and post-edit stdout/stderr go under
`AI_analysis/01_raw_logs/codex_hooks`, including `runtime_last.json` for the
latest launcher/runtime diagnosis. This does not require git commit, and git
hook / CI are optional insurance, not the primary workflow.

Codex must not leave post-edit validation for the user to run manually. If any
required check fails, Codex must continue fixing the change and rerun the
affected validation commands until the task-scoped suite passes or a true
blocker is recorded in `AI_analysis` and the final response.

Use the unified post-edit gate with the narrowest applicable scope:

```bash
python code/tools/odcr_post_edit_check.py --scope <scope>
```

Hard requirement after any modification:

```bash
python code/tools/odcr_post_edit_check.py --scope <scope>
```

Scope must match the changed area. If the script fails, Codex must fix and
rerun it. Codex must not wait for git commit. Codex must not leave validation
to the user.

If Codex Hooks are unavailable or the project `.codex` layer is not trusted,
Codex must manually run the same gate before final delivery:

```bash
python code/tools/odcr_post_edit_check.py --scope <scope>
```

Choose the narrowest relevant scope from `governance-fast`, `docs`,
`governance`, `config`, `logging`, `preprocess`, `step3`, `step4`, `step5`,
`eval`, or `all`. The tool defaults to `governance`, supports `--dry-run` and
`--max-seconds`, and must not run real preprocess, training, Step4, Step5,
eval, or rerank work. `governance-fast` runs only py_compile on the governance
tools plus the strict One-Control guardrail; the fuller `governance` scope may
run compileall, guardrail tests, and doctor.
Real training and real data-stage execution still require explicit user
authorization.

Every final response after such a change must include this fixed Validation
block:

```text
【Validation】
- compileall: PASS/FAIL
- guardrail strict: PASS/FAIL
- doctor: PASS/FAIL or not applicable
- show/dry-run: PASS/FAIL or not applicable
- tests: PASS/FAIL or not applicable
- real training: not run
```

## Required Checks Before Finishing

After any `code/`, `configs/`, `docs/`, or `tools` change, run the narrowest
applicable post-edit validation scope:

```bash
python code/tools/odcr_post_edit_check.py --scope <scope>
```

Scope selection:

- docs-only or governance docs: `governance-fast` or `governance`
- logging/path/tail/AI_analysis policy: `logging`
- config/schema/resolver/runners: `config`
- preprocess contract/runtime: `preprocess`
- Step3: `step3`
- Step4: `step4`
- Step5: `step5`
- eval/rerank: `eval`
- cross-stage contract, manifest, lineage, cache/checkpoint hard gate,
  eval-rerank gate, multiple business stages, release/reclosure, or explicit
  manual deep validation: `all`
- ignored-only, dirty-workspace-only, no session touched files, or unknown
  current-session files: `skip` by hook

Run the static guardrail directly when the selected scope or task requires it:

```bash
python code/tools/check_one_control_guardrails.py --strict
```

`./odcr doctor` is owned by the selected post-edit scope or by manual deep
validation. `./odcr show --stage step3 --task 4` and
`./odcr step3 --task 4 --dry-run` are required only when the selected scope
actually touches Step3, config changes that affect Step3, or `all`; they are
not fixed defaults for every user-facing change.

`--scope all` is a high-cost full-chain lightweight validation scope, not a
disabled or permanently banned option. It is inappropriate for narrow
single-stage or docs-only tasks when touched files map cleanly to a smaller
scope, but it remains valid for cross-stage contracts, final reclosure/release
gates, and explicit manual deep validation.

Do not change Step3/Step4/Step5 model, loss, routing, training logic, data
contracts, or generated data while performing architecture-only guardrail work.
