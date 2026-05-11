# ODCR One-Control Architecture Contract

ODCR uses a One-Control architecture: one shell wrapper, one Python CLI, and one
primary YAML configuration. This contract exists to prevent configuration drift
and old entrypoint sprawl from returning.

The current active surface is summarized in
`docs/ODCR_ACTIVE_ARCHITECTURE.md`. Future changes are governed by
`docs/ODCR_EVOLUTION_PROTOCOL.md`, and new features should use
`docs/CODEX_CHANGE_REQUEST_TEMPLATE.md` plus
`docs/ODCR_FEATURE_INTEGRATION_CHECKLIST.md` before implementation.

## Current Architecture

- Shell entrypoint: `./odcr`
- Python entrypoint: `code/odcr.py`
- Primary config: `configs/odcr.yaml`
- Config resolver: `code/odcr_core/config_resolver.py`
- Config schema/types: `code/odcr_core/config_schema.py`
- Guardrail lint: `code/tools/check_one_control_guardrails.py`

Legacy `scripts/run_stage.sh` is deleted. `presets/` has been archived under
`_archive/legacy_presets_20260424/` and must not be reattached to the mainline.

Global runtime roots and model identifiers are part of the same One-Control
surface: `project.data_dir`, `project.merged_dir`, `env.models_dir`,
`env.step5_text_model`, `env.sentence_embed_model`, and `env.embed_dim`.
Legacy user-side `ODCR_DATA_DIR`, `ODCR_MERGED_DATA_DIR`, `ODCR_MODELS_DIR`,
`ODCR_STEP5_TEXT_MODEL`, `ODCR_SENTENCE_EMBED_MODEL`, and `ODCR_EMBED_DIM`
must not override `configs/odcr.yaml`; torchrun children receive only
resolver-injected `ODCR_RESOLVED_*` transport values and fail-fast on conflict.
Run manifests and other metadata snapshots follow the same contract: backbone
`embed_dim` / `hidden_size` must come from resolved config, lineage, manifest
snapshot, or an explicitly passed resolved value, not from a bare user
`ODCR_*` environment fallback.

Formal run metadata has one first entry: `meta/run_summary.json`. The summary
is an index, not a full config dump; detailed resolved configuration lives in
`meta/resolved_config.json`, and the source table lives in
`meta/source_table.json`. Each stage/task or preprocess/unit directory keeps a
`latest.json` pointer whose `latest_summary_path` targets that run summary.
New task-stage runs use `runs/{stage}/task{T}/{run_id}/meta/`; new preprocess
runs use `runs/preprocess/{a|b|c}/{run_id}/meta/`. Deprecated config snapshot
filenames such as `config_resolved.json`, `resolved_config_snapshot.json`, and
`config_snapshot.json` are historical-read names only and must not be new
mainline outputs.

Console/file logging is layered. Default executable-stage console output is a
human summary: stage, task, source-target, run id, status, timing, key train
batch configuration, device summary, epoch/final metrics when emitted, error
summary, and the `run_summary.json` path. Detailed resolved config, source
table, launcher command, child stdout/stderr, per-batch/per-rank diagnostics,
and sample-level text belong under run meta as `console.log`, `full.log`,
`errors.log`, `debug.log`, `samples.jsonl`, `resolved_config.json`, and
`source_table.json`. `--verbose` and `--debug` are display-only controls and
must not alter resolved training payloads or fingerprints. Active mainline log
defaults must not write `code/log.out` or top-level `logs/`.
`./odcr tail` is intentionally narrow: it reads
`runs/<stage>/task<T>/latest.json`, follows `latest_summary_path` to
`meta/run_summary.json`, and tails only `meta/console.log` by default,
`meta/full.log` with `--full`, or `meta/errors.log` with `--errors`. It must
fail fast when any part of that new-layout chain is missing. Active latest
lookup must not scan run directories, probe old `runs/task<T>/...` layouts, or
synthesize a dry-run `latest` sentinel. It must not fall back to top-level
`logs/`, `code/log.out`, `nohup*.log`, fallback/mirror logs, timestamp logs,
or legacy shell log files; real-run tail defects should be fixed in the new
layout rather than by restoring old fallback chains.

Directory boundaries are explicit artifact roles. Formal run metadata and logs
belong under `runs/{stage}/task{T}/{run_id}/meta/` or
`runs/preprocess/{a|b|c}/{run_id}/meta/`. Reusable caches belong under
`cache/<producer>/<cache_key>/`; preprocess B/C defaults are
`cache/preprocess_b/<cache_key>/` and `cache/preprocess_c/<cache_key>/`.
Step4 encoded cache and Step5 tokenization cache are reusable only through
`cache_manifest.json` gates. A cache hit requires matching schema version,
source content hash, resolved config hash, tokenizer fingerprint, upstream
lineage hash, max length, required-field contract hash, and producer code
version. `dataset_info.json`, `dataset_dict.json`, path-only keys, or mtime-only
keys are never sufficient for active cache reuse.
`AI_analysis/` is for Codex audit logs, search hits, evidence ledgers, phase
summaries, final reports, and handoff digests; it must not mirror full
training logs. `data/` and `merged/` are data-contract roots only and must not
receive logs. New artifact roles must register their role, default directory,
filename convention, producer, consumer, retention note, and whether
`AI_analysis` may copy them before becoming active writers.
`cache/` is separate from `runs/`: reusable cache payloads belong in cache roots,
while run logs and run metadata stay in run meta.

Canonical metric/audit filenames are `metrics.jsonl`, `epoch_summary.csv`,
`loss_breakdown.jsonl`, `gpu_profile.jsonl`, `rcr_distribution.json`,
`eval_metrics.json`, `rerank_summary.json`, and `data_audit_summary.csv`.
Retired names such as `train_epoch_metrics.jsonl` and
`step5_train_data_audit_summary.csv` are historical references only.

Step3 is currently the structured shared/specific training stage. The old typed
Step3 bridge is a retired compatibility shim and must not become a second
control surface.

Preprocess is the only producer of the CSV data contract consumed by split,
combine, profile embedding, domain semantics, and later stages. Its current
contract is canonical evidence only: core `user` / `item` / `rating` /
`review` / `explanation`, canonical `content_evidence` and `style_evidence`,
canonical style anchors, anchor scores, `evidence_quality_prior`, and the
preprocess route priors. Split and merged CSVs add the required transport
indices; merged CSVs add `domain` as the auxiliary/target label. Retired detail
columns `content_keywords`, `content_aspects`, `content_entities`,
`style_markers`, `template_family`, and `length_style_bucket` must not appear
as processed/split/merged columns or downstream primary inputs.

The preprocess route priors are named `preprocess_route_scorer_prior` and
`preprocess_route_explainer_prior`. `route_scorer` and `route_explainer` are
not valid preprocess CSV fields; they are Step4 posterior fields only. Any
processed/split/merged CSV, preprocess_b grouped-text cache, or preprocess_c
token-window cache built against the old unprefixed route names is stale and
must rebuild or fail fast.

Step4 is the RCR routing center. Its stable export contract is
`odcr_routing_train.csv` plus same-directory `index_contract.json`. The
preprocess field `evidence_quality_prior` remains a prior and must not be
rewritten as Step4 reliability. Preprocess `route_scorer` / `route_explainer`
hints may be preserved only as `preprocess_route_scorer_prior` /
`preprocess_route_explainer_prior`; Step4 export `route_scorer` /
`route_explainer` are posterior route decisions. Step4 posterior routing must
use the explicit RCR fields `content_retention_score`, `style_shift_score`,
`rating_stability_score`, `cf_reliability_score`, `uncertainty_score`,
`confidence_bucket`, `route_scorer`, and `route_explainer`; entropy/text hygiene
may only act as auxiliary quality signals.

All Step4 RCR weights, thresholds, confidence-bucket boundaries, train-keep
policy, sample-weight policy, and export required fields live under
`configs/odcr.yaml` at `step4.rcr` and are resolved by
`code/odcr_core/config_resolver.py`. Live Step4 receives them from One-Control
as resolved config; code defaults are fallback-only for isolated unit tests.

Step4 evidence-level rules are hard contract. CPU preview is
`E1_schema_preview`; it uses proxy diagnostics and is not tuning evidence. CPU
preview cannot be used for RCR candidate ranking, `best_candidate.yaml`, formal
patch suggestion, machine verdict A, or a formal Step4 prompt. CUDA/tmux probe
availability is only `E3_gpu_transport`, not Step4 posterior runtime evidence.
Only `E4_gpu_shard_forward_bounded` or `E5_formal_full_run` may support Step4
RCR candidate ranking and verdict-A eligibility. The old `C9_balanced_quantile`
and `C9_bucket_balanced` CPU-preview candidates are superseded, and formal
Step4 remains blocked until a real GPU E4 candidate completes required
validation.

In short: formal Step4 remains blocked whenever the only available Step4
tuning evidence is CPU preview or CUDA transport evidence.

Step5 has two active paths. Step5A is scorer-only stability optimization:
`route_scorer` gates scorer-clean samples, UCI weights LCI with Step4 posterior
reliability/uncertainty/confidence fields, and LCI contributes its own weighted
loss. Step5B is explainer-only verbalization: `route_explainer` gates
explainer-rich samples, CCV uses an explicit control packet instead of prompt
concatenation, and FCA aligns scorer and explainer evidence bases. The public
Step5 controls are One-Control blocks: `step5.lci`, `step5.uci`, `step5.ccv`,
and `step5.fca`. CCV adapter dimensions and Step5 native LoRA controls are
resolved from `step5.ccv` / `step5.ccv.native_lora`; retired `lambda_lci`,
`lambda_fca`, prompt-concat controls, and backend LoRA keys must not form a
parallel active path.

Step5 eval/validation factual target rows may receive default control values
only under the explicit eval contract
`odcr_step5_factual_eval_control/1.0` with `mode=factual_eval_default`. Those
values are not Step4 RCR posterior, not train routes, and not Step4 export
posterior fields. Step5 train inputs that are expected to come from
`odcr_routing_train.csv` must reject missing posterior route/control columns
instead of substituting factual eval defaults.

## Evolution Protocol

ODCR may be refactored or extended, but every new feature must enter the
unified governance chain:

- Future Codex requests: use `docs/CODEX_CHANGE_REQUEST_TEMPLATE.md`, classify
  first, fill or mirror `docs/ODCR_FEATURE_INTEGRATION_CHECKLIST.md`, and write
  an `AI_analysis` ledger before handoff.
- Future development flow: use the template, classify first, fill or mirror
  the checklist, run the task-scoped post-edit check after edits, and reserve
  real preprocess/training/Step4/Step5/eval/rerank runs for explicit user
  authorization.
- Post-edit validation: after any Codex `code/`, `configs/`, `docs/`, or
  `tools` change, run task-scoped validation before the final response. Git
  commit hooks are optional insurance, not the primary gate.
- New parameters: `configs/odcr.yaml` -> schema -> resolver -> resolved payload
  -> source table -> tests -> guardrail.
- New data fields: data contract/schema -> producer -> transport -> consumer
  -> manifest/index contract -> fingerprint -> tests -> guardrail.
- New artifacts: schema version -> config hash -> contract version -> input and
  model fingerprints -> consumer validation -> mismatch fail-fast.
- New logs, reports, metrics, caches, and AI_analysis outputs: artifact role
  -> directory/filename convention -> producer -> consumer -> retention
  policy -> default/verbose/debug visibility -> run_summary/latest decision ->
  tests -> guardrail.
- New losses, routers, and verbalizers: config block -> schema -> resolved
  payload -> total-loss insertion point -> logging -> DDP graph-safe zero ->
  tests -> guardrail.
- New active entries: `./odcr` or `python code/odcr.py`; active shell stage
  entrypoints remain forbidden.
- Static evolution gate: `R042`-`R059` and logging artifact rules
  `R068`-`R072` must remain active so future features
  cannot bypass One-Control, contracts, lineage/fingerprints, DDP graph safety,
  legacy cleanup, source ownership, checklist/AI_analysis declaration, or
  post-edit validation workflow safety, canonical run-summary logging, and
  declared logging/output artifact boundaries.

Old logic must be deleted, migrated, retired/fail-fast, or moved to
docs/history only. Silent fallback and long-term dual active mainlines are not
allowed.

## Governance Line Vs Run-Validation Line

Code governance and real run validation are separate work lines. Governance
changes may update docs, contracts, guardrails, and lightweight static checks
without running preprocess, training, Step4, Step5, eval, or rerank. Real
data-running tasks should explicitly request the relevant stage execution and
must still obey the same One-Control, contract, lineage, and guardrail rules.

The static guardrail is a required merge/handoff gate for architecture and
configuration changes:

```bash
python code/tools/check_one_control_guardrails.py --strict
```

Codex post-edit validation uses the narrowest applicable scope:

```bash
python code/tools/odcr_post_edit_check.py --scope <scope>
```

Use `governance-fast` or `governance` for docs/governance work, `logging` for
logging/path/tail/AI_analysis policy, `config` for schema/resolver/runner
work, the owning stage scope for stage work, and `all` only for explicit
multi-business-stage changes or manual deep validation. Ignored-only,
dirty-workspace-only, no-session, and unknown current-session cases may select
`skip` by hook. Step3 show/dry-run commands are not universal user-facing
defaults; they run only for Step3 scope, config changes that affect Step3, or
`all`.

## User Entrypoints

Users should run ODCR commands through `./odcr` or `python code/odcr.py`.
Examples:

```bash
./odcr doctor
./odcr show --stage step3 --task 4
./odcr step3 --task 4 --dry-run
./odcr preprocess a
```

Direct Python use is allowed only through:

```bash
python code/odcr.py ...
```

No `step*.sh`, `train*.sh`, `eval*.sh`, or `scripts/entrypoints/*.sh` may become
main entrypoints.

## Configuration Entrypoint

All primary configuration lives in `configs/odcr.yaml`. One-off changes use
`--set key=value` and must still resolve through `code/odcr_core/config_resolver.py`.

Parameter precedence is:

1. CLI `--set`
2. `configs/odcr.yaml`
3. resolver schema defaults only

Runtime `.env` files and loose YAML files are not main configuration sources.

Step3 structured loss weights are public One-Control parameters under
`step3.structured_losses`. Step5 model architecture lives under `step5.model`;
Step5B gate and explainer-only scheduling live under `step5.explainer_gate`;
the Step5B explainer objective multiplier lives under
`step5.train.explainer_loss_weight`. Retired `adv` and `eta` names must not be
accepted as active Step5 aliases.

## Batch Semantics

Training batch terms have fixed meaning:

- `global_batch_size` / `batch_size`: effective optimizer-step train batch
- `per_gpu_batch_size`: per-GPU forward/backward train batch
- `micro_batch_size`: display alias only, meaning `per_gpu_batch_size`
- `ddp_world_size`: DDP process count from the active hardware profile
- `batch_semantics_version`: `odcr_no_accum/1`

All active ODCR train stages use the no-accum architecture. The resolver must
enforce:

```text
global_batch_size = per_gpu_batch_size * ddp_world_size
```

`grad_accum`, `gradient_accumulation_steps`, and `accumulate_grad_batches` are
retired historical names. They are rejected instead of being migrated,
defaulted, or used.

## Path Ownership

- `data/`: raw and processed dataset artifacts
- `merged/`: merged task CSV artifacts consumed by training stages
- `runs/.../meta`: logs, resolved config snapshots, status, manifests
- `AI_analysis/`: AI audit notes, search hits, intermediate evidence, reports
- `_archive/legacy_presets_*`: historical material only

Do not write logs into `data/` or `merged/`. Do not write dataset artifacts into
`runs/`, except explicit caches/status under the current architecture.

## Lineage Gates

Artifact reuse is a hard compatibility decision, not a best-effort cache hit.
Preprocess status/manifests, preprocess_b/c caches, Step3 checkpoint sidecars,
Step4 `index_contract.json` / train manifests, Step5 checkpoint sidecars, and
eval/rerank metrics must carry current One-Control config/schema/source/model
fingerprints. Downstream stages must validate those fingerprints before loading
artifacts; missing, old, or mismatched lineage must fail-fast and require the
producing stage to be rerun through `./odcr`.

Preprocess_b grouped-text cache keys must bind the preprocess contract version,
canonical column hash, selected text columns, source CSV fingerprint, sentence
model fingerprint, `embed_dim`, `read_chunk_rows`, `group_shard_size`, and cache
version. Preprocess_c token-window cache keys must bind the preprocess contract
version, canonical column hash, selected columns, tokenizer identity, sentence
model fingerprint, `embed_dim`, token-window parameters, hotpath setting, source
fingerprints, and cache version. Contract/schema/data/model/config mismatches
are stale by definition; silent reuse is forbidden.

Preprocess_b/c are formal GPU stages. The tmux session is created or entered on
the admin node with `tmux -L odcr_gpu new-session -A -s odcr`; it does not by
itself prove GPU availability. The user manually runs
`odcr-enter-gpu <JOBID>` inside the same tmux to enter the GPU node. Codex must
not execute `odcr-enter-gpu`, `srun`, `sbatch`, or `scancel`; must not create,
kill, or switch tmux sessions; and must not manage GPU allocation.
GPU use is allowed by default for repo-local validation, probe, and bounded
runtime after fast sanity and current-pane validation. The controlled tmux GPU
bridge at `python code/tools/odcr_tmux_gpu_bridge.py` can target only a
user-created, already-entered, uniquely validated GPU pane and can send one
bridge-generated command file. It is not arbitrary send-keys and is no longer
limited by a GPU whitelist hard blocker. Its validation outputs stay under
`AI_analysis/06_probe_evidence` or `runs/step3_validation` by default, with a
mandatory formal namespace guard. post-edit full is not a GPU prerequisite, and
runtime evidence takes priority over static full-suite instability.

CUDA admission checks and BGE-large use trust only the current tmux session's
real-time CUDA environment. A normal admin shell reporting no CUDA, a tmux still
on admin, or old `AI_analysis` probe output is not proof that the cluster has
no GPU and must not block a later probe after the user manually enters the GPU
node. Formal execution must fail fast before model load instead of silently
falling back to CPU. Codex GPU validation is limited to <= 3 minutes of short
probe, short benchmark, command smoke, or quick parameter comparison and must
not run complete preprocess_b/c, full stage experiments, Step3/Step4/Step5,
eval/rerank, or long benchmarks.

## Adding A Parameter

Every new public parameter must be added in one coherent change:

1. Add it to `configs/odcr.yaml`.
2. Add or validate it in `code/odcr_core/config_schema.py`.
3. Resolve it in `code/odcr_core/config_resolver.py`.
4. Ensure `./odcr show` displays the resolved value when relevant.
5. Ensure `./odcr doctor` validates the architecture and config shape.
6. Add tests for the resolved behavior.
7. Run `python code/tools/check_one_control_guardrails.py --strict`.

Do not add a new loose YAML/env file for a parameter.

## Adding A Stage

Every new stage must enter through `./odcr` and `code/odcr.py`.

Required flow:

1. Add the user command in `code/odcr.py`.
2. Add the stage config block to `configs/odcr.yaml` only if it needs public
   configuration.
3. Add schema/resolver support.
4. Add `show`, `doctor`, and dry-run coverage.
5. Add tests.
6. Keep logs in `runs/.../meta`.
7. Keep data artifacts in `data/` or `merged/`.

Do not create a parallel shell launcher.

## Forbidden Actions

- Recreate `presets/` as a live main config tree.
- Read `_archive/legacy_presets_*` from the main execution chain.
- Use runtime `.env` files as the public config interface.
- Add old-style shell stage entrypoints.
- Add hidden argparse knobs that bypass `configs/odcr.yaml` and the resolver.
- Change Step3/Step4/Step5 business logic during architecture-only guardrail
  work.

## Developer Checklist

Before merging or handing off a change:

- Read `AGENTS.md`.
- Confirm the change uses `./odcr` and `code/odcr.py`.
- Confirm new config goes through `configs/odcr.yaml`.
- Confirm resolver/source reporting is updated.
- Confirm logs/data paths remain in the right roots.
- Run `./odcr doctor`.

## Codex/AI Checklist

Before editing code:

- Read `AGENTS.md` and this contract.
- Identify whether the task changes architecture, configuration, or business
  logic.
- For architecture-only work, do not touch Step3 model/loss, Step4 routing, or
  Step5 training logic.
- Save audit notes under `AI_analysis/`.
- Run post-edit validation, including the guardrail lint when applicable,
  before final response. Select the scope from current-session touched files:
  config/One-Control/resolver changes use `config`; preprocess uses
  `preprocess`; Step3/Step4/Step5/eval use their owning scopes; docs and
  governance use `governance-fast` or `governance`; cross-stage contracts,
  manifests, lineage, cache/checkpoint hard gates, eval-rerank gates, final
  reclosure/release gates, and manual deep validation may use `all`. `--scope
  all` is not permanently banned; it is only inappropriate for narrow changes
  that map cleanly to a smaller scope.

## Verification

The lightweight architecture verification command is:

```bash
./odcr doctor
```

It includes static One-Control guardrail checks and must not require GPU or
start training.
