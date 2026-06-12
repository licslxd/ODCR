from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from executors import step5_engine


class _TinyTokenizer:
    model_max_length = 128

    def __call__(self, text: str, *, add_special_tokens: bool, truncation: bool) -> dict[str, list[int]]:
        tokens = str(text).split()
        return {"input_ids": list(range(len(tokens) + (1 if add_special_tokens else 0)))}


def test_step5_train_data_audit_token_lengths_are_sampled(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(step5_engine, "STEP5_DATA_AUDIT_TOKEN_LENGTH_SAMPLE_ROWS", 3)
    monkeypatch.setattr(step5_engine, "get_step5_tokenizer", lambda: _TinyTokenizer())

    df = pd.DataFrame(
        {
            "clean_text": [
                "one two",
                "one two three four five",
                "one",
                "one two three",
                "one two three four five six seven",
            ],
            "sample_origin": ["target_gold", "aux_cf", "aux_gold", "target_gold", "aux_cf"],
            "sample_weight_hint": [1.0, 0.5, 0.3, 0.9, 0.4],
            "train_keep": [1, 1, 1, 1, 1],
        }
    )
    log_path = tmp_path / "full.log"

    step5_engine._rank0_step5_train_data_audit(
        df,
        df,
        train_label_max_length=4,
        train_dynamic_padding=True,
        train_padding_strategy="dynamic_batch",
        log_path=str(log_path),
        loader_summary={"cache_hit": True, "cache_dir": "cache/example", "source": {"expected_sha256": "abc"}},
    )

    audit = json.loads((tmp_path / "data_audit.json").read_text(encoding="utf-8"))
    assert audit["token_length_audit"] == {
        "sampled": True,
        "sample_rows": 3,
        "total_rows": 5,
        "sample_cap": 3,
        "sample_policy": "deterministic_even_spacing",
    }
    assert audit["truncation_over_max"]["count"] == 1
