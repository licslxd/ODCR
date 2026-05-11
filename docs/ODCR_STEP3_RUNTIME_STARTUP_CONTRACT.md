# ODCR Step3 Runtime Startup Contract

Step3 startup has two phases:

1. Pre-DDP tokenizer cache phase: the parent launcher resolves One-Control config, writes `meta/resolved_config.json`, writes an initial `meta/training_runtime_config.json`, validates preprocess upstream lineage, and builds or reuses the tokenizer cache.
2. DDP training phase: only after a completed tokenizer cache manifest exists do torchrun children load the cache, initialize NCCL, wrap the model in DDP, and enter the training loop.

The tokenizer/cache phase must not initialize an NCCL process group and must not call `torch.distributed` collectives. Cache readiness is a filesystem contract: partial directory, completed manifest, completed marker, and final validation. If a child has to wait in a fallback path, it waits by file polling for a completed manifest, never by `dist.barrier` or `all_reduce`.

Formal Step3 tokenizer cache paths are controlled by `configs/odcr.yaml: step3.cache.formal_cache_namespace` and `path_layout.step3_tokenizer_cache_entry_dir`. The active namespace is:

`cache/step3/tokenizer/task{T}/{source}_to_{target}/{compatibility_key}/`

The retired `cache/task{T}/hf` path is not a formal Step3 tokenizer cache source.

Tokenizer cache writes are atomic:

- build in `<cache_key>.partial.<pid>.<uuid>/`
- write `build_started.json`
- run `datasets.map`
- save the HuggingFace dataset to the partial directory
- write `cache_manifest.json` with `completed=false`
- validate dataset files and lineage fields
- write `cache_manifest.json` with `completed=true`
- write `completed.marker`
- atomically rename the partial directory to the final cache directory
- revalidate the final directory before reuse

Failed or partial caches are never reusable. A failed build writes `failed.marker` in the partial directory, and the next run may clean stale partials only as an explicit rebuild event.

Downstream handoff must reject failed latest pointers. `from_step3=latest` and other upstream latest selectors require `latest_status` and `run_summary.status` to be one of `ok`, `completed`, or `success`; failed, running, partial, or interrupted runs are records only. Step3 and Step5 latest handoffs also require the canonical checkpoint, checkpoint sidecar, schema, and checkpoint file hash before downstream resolution succeeds.

`training_runtime_config.json` must exist before tokenization/cache starts. A failed run summary must not point to a missing required runtime config with a null hash; missing startup artifacts are recorded under `optional_artifacts` with a reason. Failed summaries carry a compact root signature with phase, fatal signature, cache key/dir/status, training-loop status, checkpoint status, and NCCL fields when present.

`num_proc`, `OMP_NUM_THREADS`, and `MKL_NUM_THREADS` are One-Control hardware runtime values. Conservative values reduce CPU contention, but they are not the P0 fix. The P0 fix is the startup split: no NCCL collective may wait while CPU tokenization is running.

Dry-run validates config resolution and command assembly only. It does not prove cold-cache wall-clock behavior, cache atomicity under interruption, or formal experiment quality. A later minimal runtime check may verify the P0 startup contract, but it is not a formal train.

Tombstones for retired designs:

- NCCL init before tokenization
- rank0 `datasets.map` plus non-rank NCCL barrier
- cache readiness via `dist.barrier` or scalar all-reduce
- Step3 formal tokenizer cache under `cache/task{T}/hf`
- direct `save_to_disk(final_cache_dir)`
- downstream latest selection accepting failed runs
- run summaries referencing missing `training_runtime_config.json` as a required artifact
