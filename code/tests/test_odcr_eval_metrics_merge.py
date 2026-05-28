import pytest

from odcr_eval_metrics import merge_eval_rows_by_sample_id


def test_merge_eval_rows_by_sample_id_sorts_complete_rows() -> None:
    rows = [[{"sample_id": 1, "pred_text": "b"}], [{"sample_id": 0, "pred_text": "a"}]]

    merged = merge_eval_rows_by_sample_id(rows, expected_n=2)

    assert [row["sample_id"] for row in merged] == [0, 1]


def test_merge_eval_rows_by_sample_id_drops_sampler_padding_duplicate() -> None:
    rows = [
        [{"sample_id": 0, "pred_text": "a"}, {"sample_id": 2, "pred_text": "c"}],
        [{"sample_id": 1, "pred_text": "b"}, {"sample_id": 0, "pred_text": "a"}],
    ]

    merged = merge_eval_rows_by_sample_id(rows, expected_n=3)

    assert [row["sample_id"] for row in merged] == [0, 1, 2]


def test_merge_eval_rows_by_sample_id_keeps_first_padding_duplicate() -> None:
    rows = [
        [{"sample_id": 0, "pred_text": "a"}],
        [{"sample_id": 0, "pred_text": "changed"}],
    ]

    merged = merge_eval_rows_by_sample_id(rows, expected_n=1)

    assert merged == [{"sample_id": 0, "pred_text": "a"}]


def test_merge_eval_rows_by_sample_id_rejects_missing_unique_rows() -> None:
    rows = [[{"sample_id": 0, "pred_text": "a"}]]

    with pytest.raises(RuntimeError, match="唯一条数不一致"):
        merge_eval_rows_by_sample_id(rows, expected_n=2)
