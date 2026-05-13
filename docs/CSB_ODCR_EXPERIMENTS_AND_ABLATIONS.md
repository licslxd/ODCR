# CSB-ODCR Experiments And Ablations

All CSB-ODCR ablations are One-Control profiles under
`configs/odcr.yaml: step3.experiment_profiles`. Use `--set
experiment_profile=<profile>` with `./odcr show` or dry-run commands to inspect
the resolved surface.

Required profiles:

- `csb_odcr_full`
- `csb_odcr_wo_csb`
- `csb_odcr_wo_controlled_injection`
- `csb_odcr_wo_rcr_uci`
- `csb_odcr_wo_ccv_fca_diversity`
- `csb_odcr_wo_conflict_routing`

`run_summary.json` records `experiment_profile` and `ablation_profile`.
`eval_registry` records the active method as `CSB-ODCR`; evaluation and rerank
stages must not treat BERTScore as a formal paper metric.

The valid paper-facing metrics remain MAE, RMSE, ROUGE, BLEU, METEOR, and
DIST/collapse diagnostics. Low-DIST candidates may remain scorer candidates, but
they must not become explainer downstream best.
