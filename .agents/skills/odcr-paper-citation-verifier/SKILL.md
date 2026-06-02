---
name: odcr-paper-citation-verifier
description: Verify and insert ODCR paper citations and BibTeX entries using trusted metadata sources only. Never fabricate references, citation keys, or BibTeX entries.
---

# ODCR Paper Citation Verifier

This skill is instruction-only. It must not include or call local scripts.

## Role

Verify citation keys, BibTeX metadata, and source provenance for the ODCR
paper. Citation work supports claims that Chat has approved; it does not create
new claims.

## Hard Rules

- No fake BibTeX.
- No citation from memory.
- If uncertain, keep TODO.
- Every active `\cite` key must exist in `paper/refs.bib`.
- Every BibTeX entry must have a verified source URL, DOI, arXiv, DBLP, ACM,
  IEEE, ACL Anthology, Crossref, OpenReview, or Semantic Scholar record.
- Citation tasks must update
  `AI_analysis/03_evidence_ledgers/paper_citation_source_ledger.md`.
- Do not convert citation metadata into a new ODCR claim.
- Do not use citations to imply unrun baselines, 5-seed results, or
  longest-reference compatibility.

## Workflow

1. Enumerate active cite keys in `paper/`.
2. Check that every active key exists in `paper/refs.bib`.
3. For new entries, verify metadata from trusted sources before insertion.
4. Record key, verified source, and paper usage in the citation source ledger.
5. Leave unresolved items as explicit citation TODOs.

## Trusted Sources

Prefer source metadata from DBLP, ACM, IEEE, ACL Anthology, arXiv, DOI/Crossref,
OpenReview, or Semantic Scholar. Do not rely on memory, blog posts, or
unverified repository text as sole citation evidence.
