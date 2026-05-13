"""Stage prompt contracts for future ODCR prompts."""

STAGE_PROMPT_CONTRACTS = {
    "runtime": "Use ./odcr runtime and stage_dispatch allowlist; no arbitrary shell.",
    "preprocess": "preprocess_b/c are GPU stages and fail fast without CUDA.",
    "step3": "No-accum architecture; bounded probes are validation-only.",
    "step4": "RCR validation requires registered bounded preflight evidence.",
    "step5": "Step5A/Step5B semantics stay One-Control-owned.",
    "eval": "Eval/rerank reuse requires strict lineage.",
}

