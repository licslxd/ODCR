from __future__ import annotations

import json

from odcr_core.aux.evidence.ai_analysis_writer import get_writer


def test_ai_analysis_writer_adds_metadata(tmp_path) -> None:
    writer = get_writer(tmp_path)
    result = writer.ledger("unit_ledger.md", "# unit", source="unit", stage="governance", validation_result="PASS")
    text = result.path.read_text(encoding="utf-8")
    assert "schema_version" in text
    assert "# unit" in text
    index = tmp_path / "AI_analysis" / "00_index" / "unit_ledger.json"
    payload = json.loads(index.read_text(encoding="utf-8"))
    assert payload["source"] == "unit"
    assert payload["artifact"]["sha256"] == result.sha256


def test_runtime_diagnostic_keeps_status_fields_top_level(tmp_path) -> None:
    writer = get_writer(tmp_path)
    result = writer.runtime_diagnostic(
        "runtime_status.json",
        {"schema_version": "runtime/unit", "success": True},
        source="unit",
        stage="runtime",
        validation_result={"success": True},
    )
    payload = json.loads(result.path.read_text(encoding="utf-8"))
    assert payload["success"] is True
    assert payload["source"] == "unit"
    assert payload["payload_schema_version"] == "runtime/unit"
    assert payload["payload"]["success"] is True
