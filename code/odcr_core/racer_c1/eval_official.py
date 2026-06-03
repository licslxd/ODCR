"""Paper-compatible metric boundary for RACER-C1."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .train_labels import meteor_proxy, rouge1_f1, rouge_l_f1, tokens


def _ngrams(seq: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(seq[i : i + n]) for i in range(0, max(0, len(seq) - n + 1)))


def _bleu_n_precision(prediction: str, reference: str, n: int) -> float:
    p = _ngrams(tokens(prediction), n)
    r = _ngrams(tokens(reference), n)
    if not p:
        return 0.0
    clipped = sum(min(count, r[key]) for key, count in p.items())
    return clipped / max(1, sum(p.values()))


def _distinct(rows: Iterable[str], n: int) -> float:
    total = 0
    uniq: set[tuple[str, ...]] = set()
    for text in rows:
        grams = list(_ngrams(tokens(text), n))
        total += len(grams)
        uniq.update(grams)
    return 100.0 * len(uniq) / max(1, total)


def compute_paper_metrics(predictions: list[str], references: list[str], *, split: str) -> dict[str, Any]:
    if len(predictions) != len(references):
        raise ValueError(f"prediction/reference length mismatch for {split}: {len(predictions)} != {len(references)}")
    count = len(predictions)

    def avg(values: Iterable[float]) -> float:
        vals = list(values)
        return round(100.0 * sum(vals) / max(1, len(vals)), 4)

    metric_values = {
        "ROUGE-1": avg(rouge1_f1(p, r) for p, r in zip(predictions, references)),
        "ROUGE-L": avg(rouge_l_f1(p, r) for p, r in zip(predictions, references)),
        "BLEU-1": avg(_bleu_n_precision(p, r, 1) for p, r in zip(predictions, references)),
        "BLEU-2": avg(_bleu_n_precision(p, r, 2) for p, r in zip(predictions, references)),
        "BLEU-3": avg(_bleu_n_precision(p, r, 3) for p, r in zip(predictions, references)),
        "BLEU-4": avg(_bleu_n_precision(p, r, 4) for p, r in zip(predictions, references)),
        "METEOR": avg(meteor_proxy(p, r) for p, r in zip(predictions, references)),
        "DIST-1": round(_distinct(predictions, 1), 4),
        "DIST-2": round(_distinct(predictions, 2), 4),
    }
    return {
        "schema_version": "odcr_racer_c1_paper_greedy_25_metrics/1",
        "split": split,
        "status": "ok",
        "sample_count": count,
        "metric_values": metric_values,
        **metric_values,
    }


def load_prediction_texts(path: Path) -> list[str]:
    out: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            out.append(str(row.get("prediction") or ""))
    return out


def write_metric_file(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return dict(payload)


def metric_selection_key(metrics: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics.get("BLEU-4") or metrics.get("bleu4") or 0.0),
        float(metrics.get("METEOR") or metrics.get("meteor") or 0.0),
        float(metrics.get("ROUGE-L") or metrics.get("rougeL") or 0.0),
    )
