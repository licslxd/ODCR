# ODCR Active Architecture

This document records the current active ODCR mainline only. Historical
material belongs in history notes or archive directories, not here.

## Active Entrypoints

The only user-visible entrypoints are:

- `./odcr`
- `python code/odcr.py`

All stage commands, dry runs, doctor checks, and show commands must pass through
these entrypoints. New shell launchers may exist only as historical or
developer-local helpers that fail fast when used as ODCR mainline entrypoints.

## Active Configuration

The primary configuration surface is:

- `configs/odcr.yaml`
- `code/odcr_core/config_schema.py`
- `code/odcr_core/config_resolver.py`

The resolver owns the resolved payload, source table, child-process transport,
run metadata, and semantic fingerprints. New public control values must be
visible in the YAML, schema, resolver, `./odcr show`, `./odcr doctor`, tests,
and the static guardrail.

Global roots, model paths, and embedding dimension are One-Control values:

- `project.data_dir`
- `project.merged_dir`
- `env.models_dir`
- `env.step5_text_model`
- `env.sentence_embed_model`
- `env.embed_dim`

Bare user `ODCR_*` environment variables are not active configuration sources.
Only resolver-injected `ODCR_RESOLVED_*` values may be passed to child
processes as transport.

## Active Data Contract

The active preprocess contract version is:

- `odcr_preprocess_contract/3.1`

Older 3.0 artifacts are stale for fresh preprocess admission.

The contract is defined by `code/data_contract.py` and enforced across
processed, split, and merged CSVs.

## Active Logging And Tail

Formal run handoff starts at `meta/run_summary.json`, with the parent
`latest.json` pointing to it through `latest_summary_path`. Logs for a run live
only under that run's `meta/` directory: `console.log`, `full.log`, and
`errors.log` are the supported `odcr tail` targets. `./odcr tail` reads
`latest.json -> run_summary.json -> meta/{console.log,full.log,errors.log}` and
does not scan run directories, use old `runs/task<T>/...` layouts, synthesize a
dry-run `latest` value, or fall back to retired top-level `logs/`,
`code/log.out`, `nohup*.log`, fallback/mirror logs, timestamp logs, or legacy
shell logs.

`AI_analysis/` stores audit/search/ledger/summary/report material only; it is
not a training full-log mirror. Reusable cache payloads belong under `cache/`,
not under run metadata or data roots. Step4 encoded cache and Step5 tokenize
cache reuse requires `cache_manifest.json` with schema, source content hash,
resolved config hash, tokenizer fingerprint, upstream lineage hash, max length,
required-field hash, and producer code version; dataset existence markers or
path/mtime-only keys are insufficient. `data/` and `merged/` receive only
data-contract artifacts, never logs.

## Active Preprocess

Preprocess produces canonical evidence fields for downstream stages:

- Core fields: `user`, `item`, `rating`, `review`, `explanation`
- Evidence fields: `content_evidence`, `style_evidence`
- Anchor fields: `content_anchor_score`, `style_anchor_score`,
  `polarity_anchor`, `domain_style_anchor`, `local_style_residual_hint`
- Prior/routing hints: `evidence_quality_prior`,
  `preprocess_route_scorer_prior`, `preprocess_route_explainer_prior`
- Split/merged transport: `user_idx`, `item_idx`
- Merged transport: `domain`

The main CSV and contract surface must not output retired detail fields:

- `content_keywords`
- `content_aspects`
- `content_entities`
- `style_markers`
- `template_family`
- `length_style_bucket`

Those names may appear only in preprocess-internal construction logic,
fail-fast checks, negative tests, or history notes.

`route_scorer` and `route_explainer` are not preprocess fields. They are Step4
posterior route decisions only, and stale preprocess CSVs containing those
unprefixed names must fail fast instead of being reused.

Preprocess execution receives roots, cache root, model paths, `embed_dim`,
offline/local mode, GPU ids, and precision flags from the One-Control resolved
payload. Child scripts may receive those values only through explicit runtime
CLI transport and resolver-injected `ODCR_RESOLVED_*` environment variables;
they must not re-read YAML or accept bare fallback variables such as
`EMBED_BATCH_SIZE` or `DOMAIN_CHUNK_BATCH_SIZE`.

`preprocess_b` and `preprocess_c` are GPU-only formal stages. The tmux session
is not itself a GPU allocation: it is created or entered on the admin node with
`tmux -L odcr_gpu new-session -A -s odcr`, then the user manually runs
`odcr-enter-gpu <JOBID>` inside that same tmux to enter the GPU node. Codex must
not execute `odcr-enter-gpu`, `srun`, `sbatch`, or `scancel`; must not create,
kill, or switch tmux sessions. Codex does not manage GPU allocation.
The narrow admin-to-GPU-pane exception is the controlled tmux GPU bridge,
`python code/tools/odcr_tmux_gpu_bridge.py`. It may target only a
user-created, already-entered, uniquely validated GPU pane and may send only
bridge-generated whitelist short validation scripts. It is not arbitrary
send-keys. Bridge output is AI_analysis-only, every mode has a mode-specific
adaptive timeout, and `stop_after_first_valid_result` is the default completion
rule.

Runtime admission and child scripts must fail fast before BGE-large model load
if CUDA is not visible in the current tmux session's real-time CUDA
environment. Silent CPU fallback is test-only via explicit debug flags and is
not an admitted formal path. A normal admin shell without `nvidia-smi`, a tmux
session still sitting on admin, or old `AI_analysis` probe output must not be
treated as proof that the cluster has no GPU; after the user manually enters the
GPU node in the same tmux, Codex must rerun the current-environment probe.
Codex GPU validation is limited to <= 3 minutes of short probe, short
benchmark, command smoke, or quick parameter comparison and must not start full
preprocess_b/c, complete stage experiments, Step3/Step4/Step5, eval/rerank, or
long benchmarks.

## Active Step3

Step3 is structured shared/specific disentanglement.

Active Step3 semantics:

- Shared/specific representation separation
- Structured evidence and prototype geometry
- HSS/local residual style semantics when present
- Structured loss weights resolved from `step3.structured_losses`
- DDP-safe loss execution, including globally synchronized finite-loss
  decisions before backward

Inactive Step3 semantics:

- No active domain-adversarial training path
- No active `adv` / `eta` controls
- No active retired adversarial-training mainline
- No retired typed bridge reconnected to execution

## Active Step4

Step4 is the RCR posterior routing stage.

Active Step4 controls live under:

- `configs/odcr.yaml: step4.rcr`
- `code/odcr_core/config_schema.py`
- `code/odcr_core/config_resolver.py`

Active Step4 posterior fields include:

- `content_retention_score`
- `style_shift_score`
- `rating_stability_score`
- `cf_reliability_score`
- `uncertainty_score`
- `confidence_bucket`
- `route_scorer`
- `route_explainer`
- `train_keep`
- `sample_weight_hint`

`evidence_quality_prior` remains a preprocess prior. Preprocess route hints may
survive only as `preprocess_route_scorer_prior` and
`preprocess_route_explainer_prior`. Step4 export `route_scorer` and
`route_explainer` always mean posterior decisions.

The active Step4 export contract is:

- `odcr_routing_train.csv`
- same-directory `index_contract.json`

The index contract documents route semantics, required fields, lineage, and
consumer compatibility for Step5.

## Active Step5

Step5 has two active paths.

Step5A is the scorer stability path:

- Consumes Step4 posterior `route_scorer` samples
- Uses LCI for scorer stability
- Uses UCI weights derived from Step4 reliability, uncertainty, confidence,
  and sample-weight fields

Step5B is the controlled explanation path:

- Consumes Step4 posterior `route_explainer` samples
- Uses CCV through an explicit control packet
- Uses FCA to align scorer evidence and explainer evidence bases

Step5 valid/test target factual rows are not Step4 exports. When Step5 builds
eval-only control packets for these rows, the control contract is explicitly
`mode=factual_eval_default` with schema
`odcr_step5_factual_eval_control/1.0`. These defaults are neutral eval controls
for factual target rows only: they are not RCR posterior decisions, not train
routes, and not Step4 export posterior fields. Any input path that is supposed
to consume `odcr_routing_train.csv` must fail fast if the Step4 posterior
route/control columns are missing.

Active Step5 controls live under:

- `step5.lci`
- `step5.uci`
- `step5.explainer_gate`
- `step5.ccv`
- `step5.ccv.native_lora`
- `step5.fca`
- `step5.model`
- `step5.train.explainer_loss_weight`

The active Step5B verbalizer receives explicit structured control, not prompt
concatenation. Step4 `sample_weight_hint` remains the posterior base sample
weight; `step5.explainer_gate.explainer_only_multiplier` is only a Step5B
training scheduling multiplier.

## Active Lineage, Cache, And DDP Guards

Cross-stage reuse is gated by lineage. Active artifacts must record and validate
the current control and source identity before reuse.

Required lineage/cache gates:

- Preprocess `skip_completed` status validates config, contract, source
  artifacts, model artifacts, schema code, and unit fingerprints.
- Preprocess B/C caches validate cache fingerprints before reuse.
- Step3 checkpoints write lineage and Step4 validates it before loading.
- Step4 exports and `index_contract.json` write export lineage and Step5
  validates it before loading.
- Step5 checkpoints write lineage and eval/rerank validate it before loading.
- Eval/rerank outputs validate resolved Step5 config and schema compatibility.

DDP graph guards:

- Finite-loss skip decisions must be synchronized across ranks before backward.
- Mask-based optional losses must use graph-safe zero tensors when no samples
  participate.
- Branches must avoid rank-local `mask.any()` decisions that make DDP graphs
  uneven.

Missing, stale, or mismatched lineage is a hard fail-fast condition. It must not
silently fall back to old artifacts.

## Active Guardrail Groups

`code/tools/check_one_control_guardrails.py` reports rules by governance group:
`control-plane`, `data-contract`, `lineage-cache`, `ddp-loss`,
`legacy-cleanup`, `step3-mainline`, `step4-rcr`, `step5-innovation`,
`code-hygiene`, `evolution-protocol`, `post-edit-workflow`,
`logging-console-file`, `logging-artifact-evolution`,
`logging-directory-boundaries`, `logging-old-layout-tail`, and
`post-edit-fast-path`, plus P0 hard-blocker gates for latest/cache reuse. The
`evolution-protocol` group contains `R042`-`R050` and is the long-term gate for
future parameters, fields, scripts, losses, routers, verbalizers, caches,
checkpoints, exports, env sources, DDP mask/gate paths, and legacy cleanup.
The logging groups keep future logs, reports, metrics, caches, and
AI_analysis outputs extensible but declared by role, directory, producer,
consumer, retention, visibility, run_summary/latest impact, and guardrail/test
coverage.

## Active Codex Change Workflow

Future Codex code-change requests should use
`docs/CODEX_CHANGE_REQUEST_TEMPLATE.md`. Codex must classify the change under
`docs/ODCR_EVOLUTION_PROTOCOL.md`, fill or mirror
`docs/ODCR_FEATURE_INTEGRATION_CHECKLIST.md`, write an `AI_analysis` ledger, and
state the rerun decision before handoff.

After Codex modifies `code/`, `configs/`, `docs/`, or `tools`, Codex must run
post-edit validation before the final response. This validation is the primary
handoff gate; git commit hooks are optional insurance. Failed required checks
must be fixed and rerun before delivery, and the final response must include
the fixed Validation block from `AGENTS.md`.

## Retired Surfaces

The following are retired and must not become active control surfaces:

- `code1`
- `presets`
- old shell entrypoints
- shared YAML side channels
- old argparse semantics that override the resolved payload
- `adv`
- `eta`
- `lambda_lci`
- `lambda_fca`
- prompt concatenation as Step5B control
- `content_preserve_score`
- old public LoRA params outside `step5.ccv.native_lora`
- top-level `logs/`, `code/log.out`, `nohup*.log`, fallback/mirror logs,
  timestamp logs, and legacy shell log fallback chains

Retired surfaces may be deleted, migrated, or kept as fail-fast/history-only
references. They must not provide silent fallback behavior or long-term dual
mainlines.
