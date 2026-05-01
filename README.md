# ODCR Mainline

ODCR is now operated through a One-Control architecture.

- Unique shell entrypoint: `./odcr`
- Unique Python entrypoint: `python code/odcr.py`
- Unique primary config: `configs/odcr.yaml`

Before future Codex/AI or developer code changes, read `AGENTS.md` and
`docs/ODCR_ARCHITECTURE_CONTRACT.md`.

For future Codex code-change tasks, start from
`docs/CODEX_CHANGE_REQUEST_TEMPLATE.md`; do not ask Codex to freely modify ODCR
before the change is classified and the integration checklist is filled or
mirrored.

Future development flow:

1. Use `docs/CODEX_CHANGE_REQUEST_TEMPLATE.md`.
2. Codex classifies the change and fills or mirrors
   `docs/ODCR_FEATURE_INTEGRATION_CHECKLIST.md` before edits.
3. After edits, Codex automatically runs the task-scoped post-edit check.
   The recommended single-user automation is Codex Hooks: after the project
   `.codex` layer is trusted, the Stop hook runs
   `/public/home/zhangliml/lc/ODCR/ODCR-main/.codex/hooks/odcr_post_edit_stop.sh`,
   which verifies the repo root, rejects Python 2, prefers the D4C Python
   interpreter, and delegates to `.codex/hooks/odcr_post_edit_stop.py`.
   The hook validates only this-session touched files from the Codex
   `transcript_path`, then payload touched files/tool outputs. Dirty workspace
   is not a post-edit signal: `git status` is not used for scope inference and
   may only appear as an optional dirty-file count in diagnostics. It ignores
   runtime/audit artifacts such as `audit.log`, `AI_analysis/`, hook runtime
   logs, `runs/`, `cache/`, `artifacts/`, `data/`, `merged/`, Python cache
   directories, `*.log`, and `*.pyc`; ignored-only changes skip validation.
   Missing, empty, or parse-failed transcripts, no payload touched files,
   dirty-workspace-only state, and unknown current-session touched files select
   `skip`. Docs/governance hook changes select `governance-fast`. Cross-stage
   contracts, manifests, lineage, cache/checkpoint hard gates, eval-rerank
   gates, current-session multi-business-stage changes, final
   reclosure/release gates, manual deep validation, or `ODCR_HOOK_SCOPE=all`
   may use `all`. Users can set `ODCR_HOOK_SCOPE=<scope>` to force a check.
   Automatic hook checks default to `--max-seconds 180`; manual deep checks may
   use `--max-seconds 900`.
   Successful Stop stdout is JSON-only; `runtime_last.json` and stdout/stderr
   logs are written under `AI_analysis/01_raw_logs/codex_hooks`.
4. Real preprocess, training, Step4, Step5, eval, or rerank runs still require
   explicit user authorization.

After Codex modifies code, config, docs, or tools, Codex must automatically run
post-edit validation before the final response. This does not require git
commit. A git hook / CI are optional insurance, not the primary workflow, and
the hook does not run real preprocess, training, Step4, Step5, eval, or rerank
unless the request explicitly authorizes those stages. Real training still
requires explicit user authorization. If Codex Hooks are not
available, run the manual fallback:

Unified lightweight post-edit gate:

```bash
python code/tools/odcr_post_edit_check.py --scope <scope>
```

Use `--scope governance-fast|docs|governance|config|logging|preprocess|step3|step4|step5|eval|all`.
Use `--dry-run` to inspect the exact commands without executing them.
Use `--scope governance-fast` for docs/governance hook checks; it runs only
governance-tool py_compile checks plus strict guardrail, not compileall,
doctor, show, stage dry-runs, full tests, or training. Manual deep checks are
still available with `python code/tools/odcr_post_edit_check.py --scope
governance` or `python code/tools/odcr_post_edit_check.py --scope all`.
Choose the narrowest scope for the current change. `all` is not permanently
banned; it is for cross-stage contracts, multi-business-stage changes, final
reclosure/release gates, or manual deep validation, and ignored-only or
dirty-workspace-only hook cases can select `skip`. Step3 show/dry-run commands
are required only for Step3 scope, config changes that affect Step3, or `all`;
they are not defaults for every user-facing change.

## Quick Commands

Post-edit validation:

```bash
python code/tools/odcr_post_edit_check.py --scope <scope>
```

Common ODCR runtime commands:

```bash
./odcr doctor
./odcr show --stage step3 --task 4
./odcr step3 --task 4 --dry-run
./odcr preprocess a
./odcr preprocess b
./odcr preprocess c
```

Use `--set key=value` for one-off overrides. Do not create or use `presets/`,
runtime `.env` files, or old shell stage entrypoints.

## Controlled Tmux GPU Bridge

Codex still must not manage GPU allocation: it must not execute
`odcr-enter-gpu`, `srun`, `sbatch`, or `scancel`, and must not create, kill,
switch, or attach tmux sessions. The user creates or enters the admin tmux with
`tmux -L odcr_gpu new-session -A -s odcr`, then manually runs
`odcr-enter-gpu <JOBID>` inside that same tmux to enter the GPU node.

The only allowed admin-to-GPU-pane automation is the controlled tmux GPU bridge:

```bash
python code/tools/odcr_tmux_gpu_bridge.py discover
python code/tools/odcr_tmux_gpu_bridge.py validate-only
python code/tools/odcr_tmux_gpu_bridge.py cuda-probe
```

The bridge validates socket path, session, pane, child Slurm step, node, and
GPU TRES before any send-keys. It can target only a user-created,
already-entered, uniquely validated GPU pane, and it sends only a
bridge-generated whitelist short validation script. This is not arbitrary
send-keys. Bridge logs, status JSON, summaries, and reports are AI_analysis
outputs only. Every mode uses mode-specific adaptive timeout and
`stop_after_first_valid_result`; it must not run complete preprocess_b/c, full
preprocess stages, Step3/Step4/Step5, eval/rerank, or long benchmarks.

## Data And Log Roots

- `data/`: raw and processed dataset artifacts
- `merged/`: merged task CSV artifacts
- `runs/.../meta`: logs, resolved configs, manifests, status
- `AI_analysis/`: AI-assisted analysis and intermediate reports
- `_archive/legacy_presets_20260424/`: historical presets only

New logs, reports, metrics, caches, and AI_analysis outputs are allowed only
after declaring their role, directory, filename convention, producer, consumer,
retention policy, default/verbose/debug visibility, run_summary/latest impact,
and guardrail/test coverage. Real run logs do not belong in `data/`,
`merged/`, top-level `logs/`, or `code/log.out`, and `AI_analysis/` must not be
used as a full training log mirror.

`./odcr tail` supports only the current run-meta layout. It reads
`runs/<stage>/task<T>/latest.json`, follows `latest_summary_path` to
`meta/run_summary.json`, then tails `meta/console.log` by default,
`meta/full.log` with `--full`, or `meta/errors.log` with `--errors`. Retired
`logs/`, `code/log.out`, `nohup*.log`, fallback/mirror, timestamp, and legacy
shell log layouts are not fallback sources; future tail bugs should repair the
new layout path that failed. Reusable cache payloads belong under `cache/`, kept
separate from `runs/.../meta`.

Legacy `scripts/run_stage.sh` is deleted; use only `./odcr` or `python code/odcr.py`.
The retired legacy modules `code/odcr_core/config_loader.py`,
`code/odcr_core/training_preset_resolve.py`,
`code/odcr_core/stage_context.py`, `code/odcr_core/step3_runtime.py`,
`code/odcr_core/step3_registry.py`, and `code/tools/async_eval_daemon.py`
must remain absent. Do not restore them as empty shells, compatibility shims,
or call-time fail-fast files.

## Adding Configuration

Every new public parameter must go through the One-Control flow:

1. Add it to `configs/odcr.yaml`.
2. Add schema/resolver support in `code/odcr_core/config_schema.py` and
   `code/odcr_core/config_resolver.py`.
3. Ensure `./odcr show` and `./odcr doctor` cover it.
4. Add relevant tests.
5. Run `python code/tools/check_one_control_guardrails.py --strict`.

Batch semantics are fixed:

```text
batch_size == micro_batch_size * ddp_world_size * grad_accum
```

where `batch_size` is the effective global train batch.
