# ODCR Feature Integration Checklist

Copy this template before adding a feature, parameter, field, artifact, entry,
loss, router, verbalizer, cache, checkpoint, export, eval/rerank behavior, or
large refactor. Fill it before implementation and update it before handoff.
Future changes must not skip checklist coverage; when the checklist is mirrored
elsewhere, the mirror must name the `AI_analysis` ledger path.

## Change Classification

| Row | Required value |
| --- | --- |
| Feature name |  |
| Owning stage | preprocess / step3 / step4 / step5 / eval / rerank / tooling / docs |
| Change type | parameter / field / artifact / entrypoint / model-loss-router-verbalizer / config-control-plane / cache-checkpoint-export / logging-metrics-cache-report / eval-rerank / legacy-cleanup |
| User-visible behavior |  |
| Internal-only behavior |  |
| Explicit non-goals |  |

## Machine-Executable Impact Table

Use `N/A - reason` only when a row truly does not apply. Leave no blank rows at
handoff.

| Required row | Value |
| --- | --- |
| One-Control impact |  |
| YAML path |  |
| schema path |  |
| resolver path |  |
| source table key |  |
| resolved payload key |  |
| `./odcr show` surface |  |
| `./odcr doctor` surface |  |
| CLI `--set` behavior |  |
| producer |  |
| consumer |  |
| data contract path |  |
| contract version |  |
| manifest key |  |
| index_contract key |  |
| fingerprint key |  |
| lineage key |  |
| mismatch policy |  |
| DDP risk |  |
| loss graph risk |  |
| eval/rerank risk |  |
| console_output_changed |  |
| file_log_added |  |
| metrics_file_added |  |
| cache_file_added |  |
| report_file_added |  |
| run_summary_updated |  |
| latest_pointer_updated |  |
| AI_analysis_output_added |  |
| artifact_role |  |
| output_directory |  |
| retention_policy |  |
| verbose_or_default |  |
| post_edit_logging_scope |  |
| legacy cleanup |  |
| guardrail rule |  |
| unit test |  |
| dry-run command |  |
| rerun decision |  |

## Control Plane Detail

| Check | Answer |
| --- | --- |
| Adds configuration | yes/no |
| YAML path |  |
| schema path |  |
| resolver path |  |
| resolved payload key |  |
| source table key |  |
| `./odcr show` output |  |
| `./odcr doctor` validation |  |
| Bare env active source introduced | no |
| Active argparse override introduced | no |
| Hidden hardcode introduced | no |

## Data Contract Detail

| Check | Answer |
| --- | --- |
| Adds data/export field | yes/no |
| Field name |  |
| data_contract/schema path |  |
| contract version |  |
| producer |  |
| transport files |  |
| consumer |  |
| manifest key |  |
| index_contract key |  |
| required/optional policy |  |
| missing-field mismatch policy |  |
| retired field interaction |  |

## Artifact And Lineage Detail

| Check | Answer |
| --- | --- |
| Adds artifact | yes/no |
| Artifact type | cache/checkpoint/export/manifest/index/other |
| Artifact path pattern |  |
| Schema version |  |
| Config hash key |  |
| Data/export contract version key |  |
| Input artifact fingerprint key |  |
| Model path fingerprint key |  |
| Task/domain lineage key |  |
| Consumer validation location |  |
| mismatch policy |  |
| rerun decision on mismatch |  |

## Logging, Metrics, Cache, And Report Detail

Future output files are allowed, but each new output must have an explicit
role and boundary. Use `N/A - reason` only when the row truly does not apply.

| Check | Answer |
| --- | --- |
| console_output_changed | yes/no |
| file_log_added | yes/no |
| metrics_file_added | yes/no |
| cache_file_added | yes/no |
| report_file_added | yes/no |
| run_summary_updated | yes/no/N/A - reason |
| latest_pointer_updated | yes/no/N/A - reason |
| AI_analysis_output_added | yes/no |
| artifact_role | console / file log / metrics / manifest / lineage / cache / AI_analysis / data artifact / report / other |
| output_directory |  |
| producer |  |
| consumer |  |
| retention_policy |  |
| verbose_or_default | default / verbose-only / debug-only / N/A - reason |
| post_edit_logging_scope | logging / governance-fast / config / preprocess / step3 / step4 / step5 / eval / all / N/A - reason |

## Entrypoint Detail

| Check | Answer |
| --- | --- |
| Adds entrypoint | yes/no |
| Active entry route | `./odcr` / `python code/odcr.py` / N/A |
| New shell entrypoint added | no |
| Dry-run command |  |
| Log/meta location |  |
| Data output location |  |

## Training, Router, Verbalizer, And DDP Detail

| Check | Answer |
| --- | --- |
| Adds or changes model/loss/router/verbalizer | yes/no |
| Config block |  |
| Schema fields |  |
| Resolved payload fields |  |
| Total loss insertion point |  |
| Logging fields |  |
| Empty-mask graph-safe zero handling |  |
| Avoids rank-local `mask.any()` graph divergence | yes/no/N/A |
| DDP risk |  |
| Finite-loss/global decision impact |  |

## Eval/Rerank Detail

| Check | Answer |
| --- | --- |
| Affects eval | yes/no |
| Affects rerank | yes/no |
| Required checkpoint/export schema updates |  |
| Required lineage validation updates |  |
| Metrics compatibility |  |
| Old metrics compatibility policy |  |
| eval/rerank risk |  |

## Old Logic Plan

| Check | Answer |
| --- | --- |
| Old logic touched |  |
| legacy cleanup | delete / migrate / retired-fail-fast / docs-history-only / N/A |
| Silent fallback introduced | no |
| Long-term dual active path introduced | no |
| Files to delete |  |
| Files to migrate |  |
| Fail-fast stubs |  |
| History/docs updates |  |

## Verification And Handoff

| Required row | Value |
| --- | --- |
| guardrail rule |  |
| unit test |  |
| dry-run command |  |
| compile/static command | `python -m compileall -q code` |
| guardrail command | `python code/tools/check_one_control_guardrails.py --strict` |
| post_edit_scope |  |
| post_edit_check_command | `python code/tools/odcr_post_edit_check.py --scope <scope>` |
| narrowest scope rule | choose the smallest scope matching current-session touched files; automatic Stop hook degrades `all` to `governance-fast` and records manual follow-up; manual deep validation may use `all` |
| Step3 dry-run applicability | only when the selected scope touches Step3, config changes that affect Step3, or `all`; not a fixed default for every user-facing change |
| post_edit_check_result |  |
| governance-fast applicability | current-session docs/governance hook/tool/test/doc changes only |
| ignored-only hook behavior | `audit.log`, `AI_analysis/`, hook runtime logs, `runs/`, `cache/`, `artifacts/`, `data/`, `merged/`, caches, `*.log`, and `*.pyc` select `skip` when no effective source/config/doc files remain |
| dirty workspace hook behavior | dirty workspace is not a post-edit signal; git status is not used for scope inference and dirty-only state selects `skip` |
| no-session hook behavior | missing/parse-failed/empty transcript, no payload touched files, and unknown current-session touched files select `skip` unless `ODCR_HOOK_SCOPE=<scope>` explicitly overrides |
| automatic hook timeout | wrapper timeout 180 seconds; child `--max-seconds 120` by default and must remain below wrapper timeout |
| manual deep-check timeout | `--max-seconds 900` may be used manually |
| validation_block_in_final_response | yes/no |
| post_edit_validation_scope |  |
| post_edit_validation_commands | Prefer `python code/tools/odcr_post_edit_check.py --scope <scope>`; use `--dry-run` to preview and `--max-seconds` to bound commands. |
| manual deep validation | `governance` or `all`; real training still requires explicit user authorization |
| post_edit_validation_result |  |
| real_training_run |  |
| rerun_required_after_change |  |
| rerun decision |  |
| AI_analysis evidence path |  |
| Known residual risk |  |

## Evolution Guardrail Coverage

| Rule | Required answer |
| --- | --- |
| `R042` active parameter One-Control path |  |
| `R043` data/export field contract path |  |
| `R044` cache/checkpoint/export lineage path |  |
| `R045` entrypoint route |  |
| `R046` env/source ownership |  |
| `R047` loss/router/verbalizer total-loss path |  |
| `R048` mask/gate DDP graph-safety path |  |
| `R049` legacy cleanup handling |  |
| `R050` checklist or AI_analysis ledger path |  |
| `R051` post-edit script exists |  |
| `R052` Codex final-response validation workflow |  |
| `R053` post-edit dry-run/no-real-training safety |  |
| `R060` console default summary/file-detail split |  |
| `R061` active logs avoid `code/log.out` and top-level `logs/` defaults |  |
| `R062` verbose/debug display-only semantics |  |
| `R063` Stop hook ignored path filter |  |
| `R064` ignored-only no-op fast path |  |
| `R065` governance-fast scope |  |
| `R066` unknown/dirty/parse-failed/no-session cases skip, not governance-fast/all |  |
| `R067` automatic hook child timeout < wrapper timeout, auto all degraded, manual deep-check 900 |  |
| `R068` log/report/metrics/cache outputs declare artifact role |  |
| `R069` run-facing outputs update run_summary/latest decisions |  |
| `R070` AI_analysis is not a full training log mirror |  |
| `R071` console default summary avoids full config/source/per-rule dumps |  |
| `R072` new log paths avoid data/merged/top-level logs/code/log.out |  |
| `R073` unknown session touched files skip unless env override |  |
| `R074` runtime diagnostics schema_version is v2.2 |  |
| `R075` runtime diagnostics records workspace_git_status_used_for_scope=false |  |
| `R076` runtime diagnostics omits legacy changed/raw/git fields |  |
| `R077` selected_scope=skip has post_edit_command=null |  |
| `R078` run logs target `runs/<stage>/<unit>/<run_id>/meta` |  |
| `R079` cache artifacts target `cache/<producer>/<cache_key>` |  |
| `R080` AI_analysis is not a full-log mirror |  |
| `R081` data/merged receive data artifacts only |  |
| `R082` daemon/nohup/mirror/fallback log defaults retired |  |
| `R083` metrics/audit filenames canonical |  |
