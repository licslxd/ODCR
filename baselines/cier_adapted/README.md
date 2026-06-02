# CIER-adapted External Baseline

This directory adapts `baselines/_archives/CIER.zip` as an external ODCR/D4C
baseline. It does not change `code/`, `configs/odcr.yaml`, `./odcr`, or the
ODCR active path.

## Upstream Audit

Upstream source is unpacked at `baselines/cier_adapted/upstream/CIER-main`.

Confirmed from upstream files:

| Item | Evidence | Result |
| --- | --- | --- |
| Entrypoint | `README.md`, `main.py` | `python main.py --dataset_name <name>` |
| Only-eval mode | `README.md`, `main.py` | `python main.py --dataset_name Yelp --only_eval` |
| Supported datasets | `README.md`, `main.py` | `Yelp`, `Amazon/MoviesAndTV`, `TripAdvisor` |
| Not natively supported | `main.py` hardcoded user/item counts | `AM_CDs` is not native upstream and needs this adapter |
| Input data root | `main.py` | `--data_dir`, default `../data/` |
| Raw data format | `main.py` | reads `<data_dir>/<dataset_name>/reviews.pickle` |
| Required raw columns | `main.py` | `user`, `item`, `rating`, `template` |
| Template usage | `main.py` | `template[0]` becomes keyword, `template[2]` becomes explanation text |
| Split files | `dataloader.py` | `<split_index>/train.index`, `validation.index`, `test.index` |
| Training flow | `main.py` | loops split indices `1..5`, trains, validates by loss, saves best LoRA and prompt encoder |
| Eval flow | `main.py` | loads best adapter and prompt encoder, then calls `test_step` |
| Rating prediction | `main.py`, `model.py` | predicted from verbalizer logits over `1..5` |
| Explanation prediction | `main.py` | greedy argmax token generation for `--word` steps |
| Upstream output | `main.py` | pickle at `<output_dir>/<dataset_name>/<split_index>generate.dataset` with `text` token ids and `rating` |
| Upstream diagnostic metrics | `main.py`, `utils.py` | RMSE, MAE, BLEU, USR, DIV, FCR, FMR, ROUGE in log |
| Dependencies | `README.md` | Python 3.9, PyTorch 2.2.2, transformers 4.37.2, peft 0.3.0, accelerate 0.28.0 |

## ODCR Task Mapping

| Task | Source domain | Target domain |
| --- | --- | --- |
| task2 | `AM_Movies` | `AM_CDs` |
| task5 | `AM_CDs` | `AM_Movies` |
| task8 | `TripAdvisor` | `Yelp` |
| task7 | `Yelp` | `TripAdvisor` |

## ODCR to CIER Field Mapping

Only these ODCR fields are consumed:

| ODCR split CSV field | CIER-adapted field | Notes |
| --- | --- | --- |
| `user` | `user_id`, stable encoded `cier_user` | Encoding is task-run local across active source/target rows. |
| `item` | `item_id`, stable encoded `cier_item` | Encoding is task-run local across active source/target rows. |
| `rating` | `rating`, `rating_index` | `rating_index = rating - 1`, matching upstream CIER. |
| `review` | `review` | Allowed text input and fallback for keyword construction. |
| `explanation` | `explanation`, CIER text target | Used as the generated explanation reference. |
| `explanation` or `review` | `keyword`, `keyword_words` | First words from allowed text only; no ODCR evidence fields. |

The adapter intentionally ignores `confidence_bucket`, `sample_weight_hint`,
`route_scorer`, `route_explainer`, Step4 posterior fields, and preprocess
evidence priors. CIER-adapted is therefore an independent external baseline and
does not use ODCR Step3/Step4 evidence routing.

## Training Protocol

Two modes are supported:

| Mode | Behavior |
| --- | --- |
| `target_only` | Train on target train, select checkpoint on target valid, report target test. |
| `source_to_target` | Source train pretraining, target train fine-tuning, target valid checkpoint selection, target test final report. |

The default mode is `source_to_target`. Full training requires a fresh validated
CUDA pane under the ODCR GPU protocol. Dry-runs and smoke checks are CPU-safe
and do not load the LLM.

## Evaluation Protocol

The final comparable metrics are produced by
`adapter/eval_with_odcr_metrics.py`, which calls ODCR metric utilities from
`code/base_utils.py` and writes:

```text
runs/baselines/cier_adapted/task<T>/<RUN_ID>/eval/<split>/paper_metrics.json
runs/baselines/cier_adapted/task<T>/<RUN_ID>/eval/<split>/eval_metrics.json
```

The decode profile is fixed in baseline-local configs:

```yaml
profile: paper_greedy_25
max_length: 25
do_sample: false
temperature: null
top_p: null
repetition_penalty: 1.0
```

CIER's original evaluator is retained only as a diagnostic reference. Paper
metrics must come from the ODCR official evaluator or this ODCR metric adapter.

## Run Layout

```text
runs/baselines/cier_adapted/task<T>/<RUN_ID>/
  meta/
    run_summary.json
    resolved_config.json
    source_table.json
    stage_status.json
  data/
  model/
  predictions/
    valid_predictions.jsonl
    test_predictions.jsonl
  eval/
    valid/
      paper_metrics.json
      eval_metrics.json
    test/
      paper_metrics.json
      eval_metrics.json
```

## Commands

Task2 required dry-runs:

```bash
python baselines/cier_adapted/adapter/build_cier_dataset.py --task 2 --dry-run
python baselines/cier_adapted/adapter/train_cier_odcr.py --task 2 --mode source_to_target --dry-run
python baselines/cier_adapted/adapter/export_predictions.py --task 2 --run-id smoke --dry-run
python baselines/cier_adapted/adapter/eval_with_odcr_metrics.py --task 2 --run-id smoke --split valid --dry-run
```

Task2 smoke sequence:

```bash
python baselines/cier_adapted/adapter/build_cier_dataset.py --task 2 --mode source_to_target --run-id smoke --smoke
python baselines/cier_adapted/adapter/train_cier_odcr.py --task 2 --mode source_to_target --run-id smoke --smoke --max-steps 1
python baselines/cier_adapted/adapter/export_predictions.py --task 2 --run-id smoke
python baselines/cier_adapted/adapter/eval_with_odcr_metrics.py --task 2 --run-id smoke --split valid
python baselines/cier_adapted/adapter/eval_with_odcr_metrics.py --task 2 --run-id smoke --split test
```

The helper scripts in `scripts/` are external baseline helpers, not ODCR
entrypoints. They default to dry-run unless invoked with `smoke` or `full`.

