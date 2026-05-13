from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from odcr_core.csb_contract import CSB_REQUIRED_TENSOR_FIELDS, validate_csb_forward_output_schema, validate_csb_packet
from step3_sidecar_test_utils import build_model, forward_batch


def test_step3_sidecar_retains_all_z_contract_fields() -> None:
    model = build_model(apply_runtime=True)
    out, _batch = forward_batch(model)

    validate_csb_forward_output_schema(out)
    validate_csb_packet(out.csb_packet)
    assert tuple(out.csb_packet["tensor_fields"]) == CSB_REQUIRED_TENSOR_FIELDS
    assert out.csb_contract_hash
    for name in CSB_REQUIRED_TENSOR_FIELDS:
        assert hasattr(out, name)
        assert name in out.structured_loss_inputs
        assert tuple(getattr(out, name).shape) == (8, model.emsize)
    assert out.csb_diagnostics["csb_mode"] == "sidecar"
    assert out.csb_diagnostics["gradient_firewall"] is True

