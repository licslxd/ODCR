# ODCR Codex Change Request Template

Copy this whole template into future Codex code-change requests. Fill the
classification and impact rows before asking Codex to edit files.

## 1. Task Objective

- Goal:
- Owning stage or area:
- Expected behavior after change:
- Explicit non-goals:
- Real data/training allowed: no, unless the task explicitly says yes.

## 2. Forbidden Actions

- During execution, emit zero interim status updates unless the user explicitly
  asks for them. Final delivery should be one complete response, with
  intermediate analysis saved under `AI_analysis/` when requested.
- Do not start code edits before classifying the change under
  `docs/ODCR_EVOLUTION_PROTOCOL.md`.
- Do not skip
  `docs/ODCR_FEATURE_INTEGRATION_CHECKLIST.md`; mirror it in the request or in
  an `AI_analysis` ledger when the full file is not edited.
- Do not bypass One-Control, data contracts, lineage/fingerprints, guardrails,
  or tests.
- Do not add bare user `ODCR_*` environment variables as active configuration.
- Do not add loose YAML, runtime `.env`, or active shell entrypoint side
  channels.
- Do not reconnect retired logic, aliases, archived presets, old fields, or old
  checkpoints as silent fallback paths.
- Do not run preprocess, training, Step4, Step5, eval, or rerank unless this
  request explicitly allows that run.
- Do not treat tmux itself as a GPU. The user creates or enters the admin-side
  tmux with `tmux -L odcr_gpu new-session -A -s odcr`, then manually runs
  `odcr-enter-gpu <JOBID>` inside the same tmux to enter the GPU node.
- Codex must not execute `odcr-enter-gpu`, `srun`, `sbatch`, or `scancel`;
  must not create, kill, or switch tmux sessions; and must not manage GPU
  allocation.
- The only exception is the controlled tmux GPU bridge
  `python code/tools/odcr_tmux_gpu_bridge.py`. It may send exactly one
  bridge-generated command to a user-created, already-entered, uniquely
  validated GPU pane, and only for whitelist short validation scripts. This is
  not arbitrary send-keys.
- Controlled bridge outputs must be AI_analysis-only and must use
  mode-specific adaptive timeout plus `stop_after_first_valid_result`.
- For `nvidia-smi`, `torch.cuda.is_available()`, preprocess_b, preprocess_c,
  BGE-large, short-window GPU probes, or CUDA admission checks, Codex may trust
  only the current tmux session's real-time CUDA environment. A normal admin
  shell, a tmux still on admin, or old `AI_analysis` probe output must not
  block a later user-entered GPU-node probe.
- If current tmux CUDA is not visible, fail fast and ask the user to manually
  run `odcr-enter-gpu <JOBID>` in that same tmux, then rerun the probe.
- Codex GPU validation is limited to <= 3 minutes of short probe, short
  benchmark, command smoke, or quick parameter comparison. Do not run complete
  preprocess_b/c, full stages, Step3/Step4/Step5, eval/rerank, or long
  benchmarks unless a future request explicitly authorizes that real run.
- Do not allow preprocess_b/c to fall back silently to CPU. Formal GPU stages
  must fail fast before BGE-large load when CUDA is not visible.

## 3. Change Type Selection

Mark every applicable type with `yes`; write `no` for the rest.

| Change type | yes/no | Details |
| --- | --- | --- |
| New parameter |  |  |
| New field |  |  |
| New artifact |  |  |
| New entrypoint |  |  |
| New model/loss/router/verbalizer |  |  |
| Modify configuration control plane |  |  |
| Modify cache/checkpoint/export |  |  |
| Modify logging/metrics/cache/report output |  |  |
| Modify eval/rerank |  |  |
| Delete or migrate old logic |  |  |

## 4. Required Impact Surface

Use `N/A - reason` only when the surface is genuinely not affected.

| Surface | Required answer |
| --- | --- |
| One-Control |  |
| YAML/config_schema/config_resolver/source table |  |
| Data contract |  |
| Manifest/index_contract |  |
| Lineage/fingerprint |  |
| DDP/loss graph |  |
| Eval/rerank |  |
| Logging/metrics/cache/report output |  |
| Guardrail/tests/docs |  |

## 4a. GPU / Preprocess_b_c Constraint

Fill this whenever the request touches GPU visibility, CUDA admission,
preprocess_b, preprocess_c, BGE-large, or embedding/domain probes.

| Question | Required answer |
| --- | --- |
| Admin tmux command | `tmux -L odcr_gpu new-session -A -s odcr` |
| User GPU-node entry | User manually runs `odcr-enter-gpu <JOBID>` inside the same tmux |
| Codex forbidden GPU-management commands | Codex must not execute `odcr-enter-gpu`, `srun`, `sbatch`, or `scancel`; must not create, kill, or switch tmux |
| Controlled bridge exception | `python code/tools/odcr_tmux_gpu_bridge.py` only; user-created, already-entered, uniquely validated GPU pane; whitelist short validation scripts only; not arbitrary send-keys |
| Bridge output/timeout | AI_analysis-only outputs; mode-specific adaptive timeout; `stop_after_first_valid_result` |
| CUDA evidence source | Current tmux session real-time CUDA only; not admin shell or old probe output |
| If CUDA is unavailable | Fail fast and ask the user to manually enter the GPU node in the same tmux, then rerun probe |
| Short validation budget | <= 3 minutes; short probe/benchmark/command smoke/parameter comparison only |
| Commands that require current tmux CUDA context | `nvidia-smi`, `torch.cuda.is_available()`, preprocess_b/c, BGE-large, short-window GPU probes, CUDA admission |
| CPU fallback allowed | no, except explicit test-only debug probes |
| Current tmux GPU visibility checked | yes/no/N/A |
| Formal preprocess_b/c started | no, unless explicitly authorized |

## 5. Logging / Artifact Output Impact

Every new output file must be declared here before implementation. This does
not freeze the current logging structure; it keeps future additions inside a
named role and directory boundary.

| Question | Required answer |
| --- | --- |
| Output role | console / file log / metrics / manifest / lineage / cache / AI_analysis / data artifact / report / other |
| Directory rationale | Why this directory owns the output |
| Duplicate or replacement | Whether it duplicates, replaces, or indexes an existing file |
| run_summary indexing | yes/no/N/A - whether `meta/run_summary.json` should point to it |
| latest.json update | yes/no/N/A - whether a parent `latest.json` should change |
| Default visibility | default console / verbose-only / debug-only / file-only / N/A |
| Guardrail/test needed | yes/no and which rule/test |

Required checks before adding output files:

1. State whether the output is console, file log, metrics, manifest, lineage,
   cache, AI_analysis, data artifact, report, or another explicit role.
2. Explain why the selected directory owns that role.
3. State whether the output duplicates an existing file, replaces one, or is an
   index of another file.
4. State whether `meta/run_summary.json` must index the output.
5. State whether `latest.json` must be updated.
6. State whether the output appears by default or only under verbose/debug/file
   detail.
7. State whether a guardrail or test must be added.

Forbidden output shortcuts:

- Do not write real run logs under `data/` or `merged/`.
- Do not use `AI_analysis/` as a full training log mirror.
- Do not mix cache payloads into ordinary full logs.
- Do not dump full resolved config, full source table, or per-rule guardrail
  PASS detail to default console output.
- Do not add new `code/log.out`, top-level `logs/`, or ad hoc `.log` defaults.

## 6. Post-Edit Validation

Codex must run the post-edit validation suite after edits and before the final
response. Codex must not wait for git commit, and Codex must not leave
validation to the user. For a single-user workflow, prefer repo-local Codex
Hooks: after the project `.codex` layer is trusted, the Codex Hooks Stop hook
in `.codex/hooks.json` runs the absolute wrapper path
`/public/home/zhangliml/lc/ODCR/ODCR-main/.codex/hooks/odcr_post_edit_stop.sh`.
The wrapper verifies the repository root, rejects Python 2, prefers the D4C
Python interpreter, then invokes `.codex/hooks/odcr_post_edit_stop.py`, infers
the scope, and calls the unified gate automatically. Stop hook inference must
use only current-session evidence: `transcript_path` touched files first, then
payload touched files/tool outputs. Dirty workspace is not a post-edit signal;
`git status` is not used for scope inference and may only be recorded as an
optional workspace dirty count in diagnostics. The hook filters ignored
runtime/audit artifacts first:
`audit.log`, `AI_analysis/`, `AI_analysis/01_raw_logs/codex_hooks/`, `runs/`,
`cache/`, `artifacts/`, `data/`, `merged/`, `__pycache__/`, `.pytest_cache/`,
and common temporary/log/bytecode files such as `*.log` and `*.pyc` do not
trigger heavy validation. If only ignored files changed, the hook skips the
post-edit checker. If no current-session touched files are available, if the
transcript is missing/parse-failed/empty, if only historical dirty workspace
state exists, or if current-session files are unknown, the hook selects `skip`
and does not call `odcr_post_edit_check.py`. Docs/governance hook changes use
`governance-fast`; `all` is reserved for current-session multi-business-stage
changes or `ODCR_HOOK_SCOPE=all`. Users can set `ODCR_HOOK_SCOPE=<scope>` to
force a check. Automatic Stop hook checks use `--max-seconds 180` by default;
manual deep checks may use `--max-seconds 900`. Successful Stop stdout is
JSON-only, while human logs and `runtime_last.json` are written under
`AI_analysis/01_raw_logs/codex_hooks`. This does not require git commit; git
hook / CI are optional insurance, not the primary workflow.

If Codex Hooks are unavailable or the `.codex` layer is not trusted, Codex must
run the manual fallback before final delivery:

```bash
python code/tools/odcr_post_edit_check.py --scope <scope>
```

Use `--dry-run` to preview the command list and `--max-seconds` to bound each
command. Choose the narrowest applicable scope by touched files and task
boundary:

| Touched surface | Scope |
| --- | --- |
| docs / governance / AGENTS / Codex template | `governance-fast` or `governance` |
| logging/path/tail/AI_analysis policy | `logging` |
| config / One-Control / resolver / schema / runners | `config` |
| preprocess contract/runtime | `preprocess` |
| Step3 | `step3` |
| Step4 | `step4` |
| Step5 | `step5` |
| eval / rerank | `eval` |
| cross-stage contract / manifest / lineage / cache or checkpoint hard gate / eval-rerank gate | related stage scopes, or `all` when one lightweight scope cannot represent the impact |
| final reclosure / release gate / manual deep validation | `all` |
| ignored-only / dirty-workspace-only / no-session / unknown current-session files | `skip` by hook |

`--scope all` is not permanently banned. It is a high-cost full-chain
lightweight validation scope; it is wrong only when a narrow task is
misclassified into `all` and thereby triggers unrelated Step3/Step4/Step5 or
eval checks. The selected scope must stay lightweight unless the request
explicitly authorizes real preprocess, training, Step4, Step5, eval, or rerank.
The `logging` scope covers run-summary and console/file logging tests without
real stage execution.
`governance-fast` contains only governance-tool py_compile checks plus strict
One-Control guardrail; it does not run compileall, doctor, show, stage dry-run,
full tests, or real training. Step3 show/dry-run commands are required only
when the selected scope actually touches Step3, config changes that affect
Step3, or `all`; they are not defaults for every user-facing change.
Real training and real data-stage execution require explicit user
authorization.

Manual deep validation remains available even when the Stop hook skips:

```bash
python code/tools/odcr_post_edit_check.py --scope governance
python code/tools/odcr_post_edit_check.py --scope all
```

```text
Post-Edit Validation:
- chosen scope:
- Codex Hooks available/trusted:
- commands run:
- compileall:
- guardrail strict:
- doctor:
- show/dry-run:
- tests:
- real training:
- failures fixed:
- final status:
```

- Change scope:
- Selected validation scope:
- Commands run:
- Command results:
- Real training run: no, unless explicitly authorized.
- Needs user authorization before real preprocess/training/Step4/Step5/eval/rerank: yes/no

| Command | PASS/FAIL/not applicable | Evidence |
| --- | --- | --- |
| `python -m compileall -q code` |  |  |
| `python code/tools/check_one_control_guardrails.py --strict` |  |  |
| `./odcr doctor` |  | Scope-owned or manual deep check; not a fixed default for every change. |
| stage show/dry-run |  | Run only for the selected owning stage, config impact, or `all`; Step3 dry-run is not a universal default. |
| tests or task-specific lightweight checks |  |  |
| real training | not run |  |

If a required command fails, Codex must fix and rerun the affected validation
command before final delivery. Final delivery must include the fixed-format
Validation block required by `AGENTS.md`.

## 7. Required Outputs

- Modified files:
- Old logic handling: delete / migrate / retired-fail-fast / docs-history-only / N/A
- Rerun decision:
- AI_analysis file:
- Lightweight verification result:
- Guardrail result:
- Tests result:
- Remaining risk:

## Codex Execution Order

1. Read `AGENTS.md`.
2. Classify the request using `docs/ODCR_EVOLUTION_PROTOCOL.md`.
3. Fill or mirror `docs/ODCR_FEATURE_INTEGRATION_CHECKLIST.md`.
4. Edit only the files allowed by the request.
5. Run post-edit validation immediately after edits; do not wait for commit.
   Use the Codex Hooks Stop hook when available, otherwise run
   `python code/tools/odcr_post_edit_check.py --scope <scope>` manually.
6. Fix validation failures and rerun affected commands before handoff.
7. State whether rerun is required and why.
8. Write the `AI_analysis` ledger/summary/report.
9. Final response must include changed files, verification results, rerun
   decision, and whether training/runtime logic was untouched.
