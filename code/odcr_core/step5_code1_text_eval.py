"""Code1-compatible explanation-only metrics for completed Step5 predictions.

This module does not launch Step5 eval, training, or rerank.  It only consumes
already-generated Step5 prediction rows and computes the original-code style
text metrics, excluding rating metrics and BERTScore.
"""
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from base_utils import evaluate_text


STEP5_CODE1_TEXT_EVAL_SCHEMA_VERSION = "odcr_step5_code1_text_eval_no_rating_no_bert/1"
STEP5_CODE1_TEXT_EVAL_PROTOCOL = "step5_final_pred_step3_code1_text_eval_no_rating_no_bert"
DEFAULT_CODE1_TEXT_MAX_LENGTH = 25


class Step5Code1TextEvalError(RuntimeError):
    """Raised when Step5 prediction rows cannot support code1-style text eval."""


def _has_value(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _token_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        parsed: Any = None
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(raw)
                break
            except Exception:
                parsed = None
        if parsed is None:
            raw = raw.replace(",", " ")
            return [int(float(x)) for x in raw.split() if x.strip()]
        value = parsed
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                out.extend(_token_ids(item))
            elif hasattr(item, "tolist"):
                out.extend(_token_ids(item.tolist()))
            elif _has_value(item):
                out.append(int(float(item)))
        return out
    if _has_value(value):
        return [int(float(value))]
    return []


def _decode_token_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return str(tokenizer.decode(list(token_ids), skip_special_tokens=True)).strip()
    except TypeError:
        return str(tokenizer.decode(list(token_ids))).strip()


def code1_style_reference_text(row: Mapping[str, Any], tokenizer: Any, *, max_length: int = DEFAULT_CODE1_TEXT_MAX_LENGTH) -> str:
    """Rebuild a reference with the same 25-token encode/decode shape as code1.

    code1 tokenizes the gold explanation with ``padding="max_length"``,
    ``max_length=25`` and ``truncation=True``, then evaluates decoded text.  This
    function mirrors that input protocol for Step5 rows.
    """

    if tokenizer is None:
        raise Step5Code1TextEvalError("code1-style Step5 text eval requires a tokenizer")
    if int(max_length) != DEFAULT_CODE1_TEXT_MAX_LENGTH:
        raise Step5Code1TextEvalError("code1-style Step5 text eval requires max_length=25")
    if _has_value(row.get("raw_ref_text")):
        ref = str(row.get("raw_ref_text"))
    elif "ref_text" in row:
        ref = "" if row.get("ref_text") is None else str(row.get("ref_text"))
    else:
        raise Step5Code1TextEvalError("Step5 prediction row missing raw_ref_text/ref_text")
    encoded = tokenizer(ref, padding="max_length", max_length=int(max_length), truncation=True)
    ids = _token_ids(encoded.get("input_ids") if isinstance(encoded, Mapping) else encoded)
    return _decode_token_ids(tokenizer, ids)


def step5_final_prediction_text(row: Mapping[str, Any], tokenizer: Any | None = None) -> str:
    """Return Step5 final generated text, preferring exact generated token ids."""

    ids = _token_ids(row.get("pred_token_ids"))
    if ids and tokenizer is not None:
        return _decode_token_ids(tokenizer, ids)
    if "pred_text" not in row:
        raise Step5Code1TextEvalError("Step5 prediction row missing pred_text/pred_token_ids")
    return "" if row.get("pred_text") is None else str(row.get("pred_text")).strip()


def build_code1_text_eval_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    tokenizer: Any,
    max_length: int = DEFAULT_CODE1_TEXT_MAX_LENGTH,
) -> list[dict[str, Any]]:
    """Build the exact text pairs consumed by code1-style explanation metrics."""

    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        sample_id = row.get("sample_id", index)
        pred = step5_final_prediction_text(row, tokenizer)
        ref = code1_style_reference_text(row, tokenizer, max_length=max_length)
        out.append(
            {
                "sample_id": sample_id,
                "pred_text": pred,
                "ref_text": ref,
                "prediction_source": "pred_token_ids" if _token_ids(row.get("pred_token_ids")) else "pred_text",
                "reference_source": "raw_ref_text" if _has_value(row.get("raw_ref_text")) else "ref_text",
                "reference_protocol": "tokenizer_padding_max_length_25_truncation_then_decode",
            }
        )
    if not out:
        raise Step5Code1TextEvalError("Step5 code1-style text eval requires at least one prediction row")
    return out


def compute_step5_code1_text_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    tokenizer: Any,
    text_metric_fn: Callable[[Sequence[str], Sequence[str]], Mapping[str, Any]] = evaluate_text,
    max_length: int = DEFAULT_CODE1_TEXT_MAX_LENGTH,
) -> dict[str, Any]:
    """Compute ROUGE/BLEU/DIST/METEOR for Step5 outputs with code1-style inputs."""

    metric_rows = build_code1_text_eval_rows(rows, tokenizer=tokenizer, max_length=max_length)
    predictions = [str(row["pred_text"]) for row in metric_rows]
    references = [str(row["ref_text"]) for row in metric_rows]
    metrics = dict(text_metric_fn(predictions, references))
    return {
        "schema_version": STEP5_CODE1_TEXT_EVAL_SCHEMA_VERSION,
        "protocol": STEP5_CODE1_TEXT_EVAL_PROTOCOL,
        "stage": "step5",
        "mode": "explanation_only",
        "sample_count": int(len(metric_rows)),
        "metric_function": "base_utils.evaluate_text",
        "text_input_policy": {
            "prediction_source": "Step5 final generated pred_text or pred_token_ids",
            "reference_source": "raw_ref_text preferred, otherwise ref_text",
            "reference_builder": "tokenizer(text, padding='max_length', max_length=25, truncation=True) then decode(skip_special_tokens=True)",
            "max_length": int(max_length),
            "matches_code1_reference_protocol": True,
        },
        "excluded_metrics": ["MAE", "RMSE", "BERTScore"],
        "rating_metrics_written": False,
        "bertscore_written": False,
        "explanation": metrics,
    }


def load_step5_prediction_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load Step5 prediction rows from ``predictions.jsonl`` or ``predictions.csv``."""

    p = Path(path).expanduser()
    if not p.is_file():
        raise Step5Code1TextEvalError(f"prediction file missing: {p}")
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                if not isinstance(payload, dict):
                    raise Step5Code1TextEvalError(f"{p}:{line_no} JSONL row must be an object")
                rows.append(payload)
        return rows
    if suffix == ".csv":
        with p.open("r", encoding="utf-8", newline="") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    raise Step5Code1TextEvalError(f"unsupported prediction file suffix: {p.suffix}")
