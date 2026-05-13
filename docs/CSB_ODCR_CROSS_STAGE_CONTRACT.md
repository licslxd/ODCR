# CSB-ODCR Cross-Stage Contract

The active cross-stage contract starts at Step3.

## Step3 Output

`Step3ForwardOutput` must expose:

- `z_content`
- `z_style`
- `z_domain`
- `z_uncertainty`
- `csb_packet`
- `csb_diagnostics`
- `csb_schema_version`
- `csb_contract_hash`

`csb_packet.schema_version` is `csb_odcr_csb_packet/1`.
`csb_contract_hash` is a stable hash of the contract payload with
`contract_hash` excluded from the hash input.

## Step4 Gate

Step4 consumes Step3 through the upstream resolver and must reject formal
upstreams without `meta/readiness_audit.json`,
`stage_status.final_status=step4_ready`, a CSB contract hash, and all four CSB
tensor fields. Step4 RCR remains the posterior routing stage; preprocess route
fields are prior-only.

Step4 reserves CSB-aware route fields:

- `csb_route_confidence`
- `csb_scorer_clean_score`
- `csb_explainer_control_score`

## Step5 Gate

Step5A consumes scorer-clean CSB packets using `z_content` and `z_uncertainty`.
Step5B consumes explainer-rich CSB packets using all four CSB tensor fields and
must keep diversity guard status available to candidate selection.

Missing CSB contract, missing CSB hash, or missing tensor field metadata is a
hard gate failure for CSB-ODCR formal handoff.

Step3 paper-eval handoff and text metrics are not readiness evidence. BLEU,
ROUGE, DIST, and METEOR are only valid after the Step3 -> Step4 -> Step5 ->
eval/rerank chain.
