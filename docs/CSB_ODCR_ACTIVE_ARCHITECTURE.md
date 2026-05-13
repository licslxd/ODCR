# CSB-ODCR Active Architecture

Paper-facing method name: **CSB-ODCR: Causal Structure Bottleneck for Orthogonal Disentangled Counterfactual Recommendation**.

The repository and command names may still use ODCR as the legacy project
umbrella, but the active method/model family is `CSB-ODCR` / `csb_odcr`.

## Step3 Core

Step3 is split into two paths.

- Primary path: rating-only scorer and rating representation.
- CSB sidecar: detached structural heads and diagnostics.

The Causal Structure Bottleneck emits:

- `z_content`: rating-safe content basis and scorer-clean content signal.
- `z_style`: expression and explanation style basis.
- `z_domain`: domain shift and counterfactual domain variation.
- `z_uncertainty`: reliability, uncertainty, and routing confidence basis.

EASD, HSS, and geometry are sidecar-only diagnostics/training signals. They do
not update the primary encoder, recommender, or rating head.

## Gradient Firewall

`L_rating_shared` is the only primary Step3 formal loss. CSB sidecar losses read
detached primary representations and may update only sidecar heads. Controlled
injection, rating-safe adapters, light explainer CE, and conflict routing do not
participate in Step3 formal backward.

## Routing And Explanation Boundary

Step4 owns posterior routing. Step5 owns explanation generation, controlled text
injection, verbalizers, and diversity metrics. Step3 exports Z variables and
sidecar packets only.

## Handoff

Step3 writes CSB method metadata, contract hash, forward schema,
`readiness_audit.json`, checkpoint lineage, and sidecar diagnostics for Step4
and Step5. Step4 and Step5 formal consumption must refuse missing CSB contract
payloads.
