# PET Challenge 2025 Audit Reconciliation

Date: 2026-02-24  
Inputs compared:
- `AUDIT_REPORT.md` (Claude Opus 4.6)
- Codex audit (this session)

## Summary

Both audits agree the pipeline is generally strong, but Claude under-reported correctness-risk bugs in pKa-v2 fallback and NaN handling.  
This reconciled report de-duplicates overlaps and resolves conflicts.

Counts after reconciliation:
- Critical: 2
- Important: 7
- Minor: 6

---

## Overlap (Both Audits Agree)

1. `validate_v4.py` is stale for newer scoring branches (v5/v6)
- Claude: `AUDIT_REPORT.md` §3.2
- Codex: Important issue (validation scope mismatch)
- Verdict: Keep as Important (not Minor), because it can falsely reassure versioned runs.

2. pKa strictness/data-integrity checks are insufficient
- Claude: `AUDIT_REPORT.md` §3.8 (invalid v2 file integrity not deeply checked)
- Codex: Important issue (`--require-pka` only strict at file/column level, not row completeness)
- Verdict: Keep as Important.

3. Notebook execution/gating around submission generation is fragile
- Claude: `AUDIT_REPORT.md` §2.1 (Cell 35 unexecuted state risk)
- Codex: Critical notebook logic mismatch (`PREDICT_TEST=False` can still generate v6 if stale v2 file exists)
- Verdict: Keep two separate findings:
  - Important: cell-order reproducibility risk
  - Critical: toggle semantics can select wrong scoring branch

---

## Conflicts Resolved

1. Claude says "Critical issues: 0"
- Resolved: incorrect. At least two correctness blockers remain:
  - pKAI empty-result fallback bug in `compute_pka_features_v2.py`
  - notebook toggle gating bug for v6 generation

2. Claude treats float equality in pKa stats as Important
- Resolved: downgrade to Minor (diagnostic/reporting only, not scoring core behavior).

3. Claude marks ESMC missing `entropy_at_site`/`native_ll_at_site` as High
- Resolved: Important, not Critical. This is feature asymmetry and quality/reliability impact, not direct wrong-result crash.

---

## Final Corrected Issue List

### Critical

1. pKAI empty-result path blocks PROPKA fallback  
File: `scripts/compute_pka_features_v2.py` lines 272-283  
Issue: `[]` from pKAI is treated as success (`residues is not None`), so PROPKA fallback is skipped.

2. Notebook `PREDICT_TEST` toggle does not strictly gate v6  
File: `PET_Challenge_2025_Pipeline_v2.ipynb` (Cell 17 vs Cell 35 logic)  
Issue: stale `pka_features_test_v2.csv` can trigger v6 even when `PREDICT_TEST=False`.

### Important

1. `--require-pka` lacks row-level strictness (partial NaN-heavy files pass and get mean-imputed)  
File: `scripts/generate_submission_v2.py` lines 445-474, 435-443

2. `zscore()` all-NaN path can emit all-NaN outputs  
File: `scripts/generate_submission_v2.py` lines 59-64

3. `wt_idx=-1` fallback may silently map to last WT in CDS feature generation  
File: `scripts/compute_cds_features.py` lines 113, 257

4. ESM embedding trainer/extractor filename mismatch  
Files: `scripts/train_esm_model.py`, `scripts/extract_esm_embeddings.py`

5. Catalytic-His identification heuristic is fragile (`min(pKa)` heuristic)  
Files: `scripts/compute_pka_features.py`, `scripts/compute_pka_features_v2.py`

6. ESMC lacks position-specific feature outputs used by v4+ path  
Files: `scripts/esmc_scoring.py` vs `scripts/generate_submission_v2.py`

7. `mutation_features.csv` required-column validation missing in submission generator  
File: `scripts/generate_submission_v2.py` around feature load block

### Minor

1. `compute_pka_features_v2.py` float equality in diagnostics (`== 0.0`)  
2. `compute_pka_features_v2.py` lacks `--propka-only`/`--pkai-only` exclusivity guard  
3. `compute_cds_features.py` multi-mutant fallback returns all-zero AA delta features  
4. `predict_structures.py --max-seqs` truncation can confuse downstream expectations  
5. `execute_notebook_cells.py` is stale vs current notebook indices  
6. Hardcoded local env paths in `extract_tm_from_pdfs.py` reduce portability

---

## What Was Fixed in This Reconciliation

- Removed duplicate overlap findings between the two audits.
- Corrected severity where one report over/under-estimated impact.
- Restored missing critical findings omitted in `AUDIT_REPORT.md`.
- Produced a single normalized issue list to drive remediation.

