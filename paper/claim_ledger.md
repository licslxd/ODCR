# ODCR Paper Claim Ledger

Scope: paper-writing only. This ledger records manuscript claims that are safe
under the current evidence boundary. It does not add experiments, run
preprocessing, start training/eval/rerank, fill seeds, rebuild references,
adapt baselines, or modify runtime outputs.

Current story: D4C diagnoses explanation degeneration as dynamic textual
feedback \(W^{(t)} \rightarrow E^{(t)} \rightarrow W^{(t+1)}\). ODCR refines
that coarse feedback variable as \(W^{(t)} \Rightarrow (C,S^{(t)},R_{cf})\),
where \(C\) is stable content evidence, \(S^{(t)}\) is hierarchical dynamic
style, and \(R_{cf}\) is counterfactual reliability. EASD, HSS, RCR/UCI, and
CCV/FCA form the full method loop.

Verdict values: `SAFE_MAIN_CLAIM`, `SAFE_WITH_CAVEAT`, `FUTURE_WORK_ONLY`,
`DO_NOT_WRITE`.

| ID | Type | Sections | Claim | Verdict | Evidence status | Safe wording | Unsafe wording | Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M-C01 | Method claim | Abstract, Introduction, Method | D4C's dynamic feedback diagnosis is \(W^{(t)} \rightarrow E^{(t)} \rightarrow W^{(t+1)}\). | SAFE_MAIN_CLAIM | Chat decision; D4C citation | "D4C diagnoses degeneration as dynamic feedback through textual attributes." | "ODCR invented D4C's feedback model." | Keep as foundation credited to D4C. |
| M-C02 | Method claim | Abstract, Introduction, Method, Figure 1 | ODCR refines \(W^{(t)}\) into \(C\), \(S^{(t)}\), and \(R_{cf}\). | SAFE_MAIN_CLAIM | Chat decision; manuscript method | "ODCR refines coarse textual feedback into content, style, and reliability controls." | "ODCR proves all hidden textual causes are recovered." | Keep as central story. |
| M-C03 | Method claim | Method 3.4, Figure 2 | EASD anchors content/style decomposition in evidence. | SAFE_MAIN_CLAIM | Chat decision; method text | "EASD separates stable content evidence from expression style." | "EASD guarantees complete disentanglement." | Keep as method module, not proof claim. |
| M-C04 | Method claim | Method 3.5, Figure 1, Figure 2 | HSS models \(S^{(t)}=(S_{\mathrm{domain}}^{(t)},\Delta S_{\mathrm{local}}^{(t)})\). | SAFE_MAIN_CLAIM | Chat decision; method text | "HSS represents domain-global style and instance-local residual style." | "HSS fully identifies the true style dynamics." | Keep as modeling structure. |
| M-C05 | Method claim | Method 3.6, Figure 3 | RCR/UCI filters and weights counterfactual evidence by reliability and uncertainty. | SAFE_WITH_CAVEAT | Chat decision; current design evidence | "\(R_{cf}\) controls counterfactual influence through eligibility and weighting." | "\(R_{cf}\) alone explains all observed metric changes." | Do not over-attribute without ablation evidence. |
| M-C06 | Method claim | Method 3.7, Figure 2 | CCV generates \(P(E\mid U,I,C,S,R_{cf})\) and FCA aligns explanation content with content-grounded rating evidence. | SAFE_MAIN_CLAIM | Chat decision; method text | "CCV/FCA makes generation content-grounded, style-aware, and reliability-controlled." | "FCA has completed human faithfulness proof." | Keep as method principle. |
| F-C01 | Figure claim | Figure 1 | Figure 1 is a vector/TikZ dynamic causal refinement diagram with D4C feedback, ODCR factorized feedback, and RCR control. | SAFE_MAIN_CLAIM | TikZ source present | "Figure 1 summarizes dynamic causal refinement." | "Figure 1 reports new experiment results." | Keep in Introduction. |
| F-C02 | Figure claim | Figure 2 | Figure 2 is an overall architecture diagram using method module names, not raw workflow stages. | SAFE_MAIN_CLAIM | TikZ source present | "Figure 2 shows the ODCR method loop and separate rating branch." | "Figure 2 is a Step3/Step4/Step5 engineering workflow." | Keep in Method. |
| F-C03 | Figure claim | Figure 3 | Figure 3 shows RCR/UCI eligibility and reliability weighting. | SAFE_MAIN_CLAIM | TikZ source present | "Figure 3 illustrates routing and sampling influence." | "Figure 3 proves ablation gains." | Use as method diagram and ablation hook. |
| R-C01 | Result claim | Results | Main tables are D4C-style overall performance tables organized by dataset block and method. | SAFE_MAIN_CLAIM | Existing table values; no numeric changes | "The tables report current single-run values by dataset block." | "The tables are full multi-seed benchmark dominance evidence." | Keep table numbers unchanged. |
| R-C02 | Result claim | Results, Discussion | Amazon settings are related-domain content/style transfer; TripAdvisor/Yelp settings stress larger style shift and reliability control. | SAFE_WITH_CAVEAT | Dataset/task semantics; current tables | "The settings are interpreted as different transfer regimes." | "The setting proves each module's isolated causal effect." | Keep interpretation descriptive. |
| R-C03 | Result claim | Results, Limitations | Rating and explanation metrics should be read jointly. | SAFE_MAIN_CLAIM | Table structure | "RMSE/MAE and text metrics capture different behavior." | "One metric alone proves method superiority." | Keep joint reading. |
| B-C01 | Boundary claim | All paper | No SOTA, statistical-significance, full reproduction, 5-seed, longest-reference, or modern-baseline dominance claim. | DO_NOT_WRITE | Missing evidence | "The current results are bounded evidence." | "ODCR is statistically significant SOTA over all baselines." | Keep prohibited claims out. |
| B-C02 | Boundary claim | Limitations, Discussion | Ablation evidence can be integrated later; current text should not say ablations are permanently absent or completed. | SAFE_WITH_CAVEAT | Chat decision | "Ablation hooks are exposed for later evidence integration." | "Ablations are finished" or "there are no ablations." | Use careful ongoing-scope wording. |
| I-C01 | Implementation mapping | Method 3.8 | Preprocess/Step3/Step4/Step5_e may appear only as implementation mapping. | SAFE_MAIN_CLAIM | Paper collaboration rules | "Implementation maps modules to workflow components." | "Workflow stages are the main contribution." | Keep out of Abstract, Introduction, Results, captions. |
