from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from base_utils import build_paper_metric_inputs, official_paper_metrics  # noqa: E402


class TinyTokenizer:
    def __call__(self, text, add_special_tokens=False, truncation=False):
        del add_special_tokens, truncation
        return {"input_ids": [int(x) for x in str(text).split() if str(x).strip()]}

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(int(x)) for x in ids)


def test_build_paper_metric_inputs_truncates_prediction_and_reference_to_25_tokens() -> None:
    tok = TinyTokenizer()
    pred = " ".join(str(i) for i in range(40))
    ref = " ".join(str(i) for i in range(30))
    out = build_paper_metric_inputs(pred, ref, tok, max_len=25)
    assert out["schema_version"] == "odcr_step5_paper_metric_inputs/1"
    assert out["metric_pred"] == " ".join(str(i) for i in range(25))
    assert out["metric_ref"] == " ".join(str(i) for i in range(25))
    assert out["prediction_truncated"] is True
    assert out["reference_truncated"] is True


def test_official_paper_metrics_marks_single_official_schema() -> None:
    metrics = official_paper_metrics(["a b c"], ["a b c"])
    assert metrics["schema_version"] == "odcr_step5_official_paper_metrics/1"
    assert metrics["input_schema_version"] == "odcr_step5_paper_metric_inputs/1"
    assert metrics["token_length_policy"]["prediction_max_length"] == 25
    assert "bleu" in metrics and "rouge" in metrics and "distinct_corpus" in metrics
