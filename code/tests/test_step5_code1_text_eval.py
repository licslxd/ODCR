from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from odcr_core.step5_code1_text_eval import (  # noqa: E402
    STEP5_CODE1_TEXT_EVAL_SCHEMA_VERSION,
    build_code1_text_eval_rows,
    compute_step5_code1_text_metrics,
    load_step5_prediction_rows,
)


class TinyCode1Tokenizer:
    def __call__(self, text, padding=None, max_length=None, truncation=False):
        ids = [int(x) for x in str(text).split() if str(x).strip()]
        if truncation and max_length is not None:
            ids = ids[: int(max_length)]
        if padding == "max_length" and max_length is not None:
            ids = ids + [0] * max(0, int(max_length) - len(ids))
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        values = []
        for x in ids:
            ix = int(x)
            if skip_special_tokens and ix == 0:
                continue
            values.append(str(ix))
        return " ".join(values)


def test_build_rows_rebuilds_reference_with_code1_25_token_protocol() -> None:
    tok = TinyCode1Tokenizer()
    rows = [
        {
            "sample_id": 7,
            "pred_token_ids": [101, 102, 0],
            "pred_text": "this should not win",
            "raw_ref_text": " ".join(str(i) for i in range(1, 31)),
        }
    ]

    built = build_code1_text_eval_rows(rows, tokenizer=tok)

    assert built[0]["sample_id"] == 7
    assert built[0]["pred_text"] == "101 102"
    assert built[0]["ref_text"] == " ".join(str(i) for i in range(1, 26))
    assert built[0]["prediction_source"] == "pred_token_ids"
    assert built[0]["reference_source"] == "raw_ref_text"


def test_compute_metrics_uses_text_metric_only_without_rating_or_bert() -> None:
    tok = TinyCode1Tokenizer()
    captured = {}

    def fake_metric(predictions, references):
        captured["predictions"] = list(predictions)
        captured["references"] = list(references)
        return {"rouge": {"l": 12.34}, "bleu": {"4": 5.67}, "dist": {"1": 8.9}, "meteor": 1.23}

    payload = compute_step5_code1_text_metrics(
        [
                {
                    "sample_id": 1,
                    "pred_text": "9 8 7",
                    "ref_text": " ".join(str(i) for i in range(1, 41)),
                }
            ],
            tokenizer=tok,
            text_metric_fn=fake_metric,
    )

    assert payload["schema_version"] == STEP5_CODE1_TEXT_EVAL_SCHEMA_VERSION
    assert payload["mode"] == "explanation_only"
    assert payload["rating_metrics_written"] is False
    assert payload["bertscore_written"] is False
    assert "recommendation" not in payload
    assert "bert" not in payload["explanation"]
    assert captured["predictions"] == ["9 8 7"]
    assert captured["references"] == [" ".join(str(i) for i in range(1, 26))]


def test_load_step5_prediction_rows_supports_jsonl_and_csv(tmp_path: Path) -> None:
    jsonl = tmp_path / "predictions.jsonl"
    jsonl.write_text('{"sample_id": 1, "pred_text": "a", "ref_text": "b"}\n', encoding="utf-8")
    assert load_step5_prediction_rows(jsonl) == [{"sample_id": 1, "pred_text": "a", "ref_text": "b"}]

    csv_path = tmp_path / "predictions.csv"
    csv_path.write_text("sample_id,pred_text,ref_text\n1,a,b\n", encoding="utf-8")
    assert load_step5_prediction_rows(csv_path) == [{"sample_id": "1", "pred_text": "a", "ref_text": "b"}]
