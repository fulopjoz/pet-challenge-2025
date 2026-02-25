# PET Challenge 2025 Pipeline Audit

## Objective

Perform a comprehensive audit of the entire PET Challenge 2025 zero-shot prediction pipeline. Examine every script, notebook, data file, and scoring approach for correctness, consistency, scientific validity, and potential improvements. Produce a detailed written report with specific findings, line references, and severity ratings.

## Scope

Audit ALL of the following:

### 1. Scripts (in `scripts/`)

- `esm2_zero_shot_scoring.py` — ESM2-650M scoring logic, wildtype-marginal method, position-specific features
- `esmc_scoring.py` — ESMC-600M scoring, compare with ESM2 for consistency
- `compute_cds_features.py` — CDS feature extraction (5' AT-richness, rare codons, mutation properties)
- `predict_structures.py` — ESMFold structure prediction, `--mode {wt,test,all}`, resumability
- `compute_pka_features.py` — PROPKA pKa for 313 WTs, WT-to-test mapping
- `compute_pka_features_v2.py` — Per-mutant pKa via pKAI/PROPKA, delta features, fallback chain
- `generate_submission_v2.py` — Main submission generator (v4/v5/v6), weight vectors, z-scoring, rank scaling
- `validate_v4.py` — Validation checks (substitution spread, top-K WT fraction, within-WT correlation)
- All other scripts in the directory — legacy, ML baselines, utilities

### 2. Notebook

- `PET_Challenge_2025_Pipeline_v2.ipynb` — Full Colab pipeline, cell-by-cell review
  - Are cells in correct execution order?
  - Do variable names carry across cells correctly?
  - Are there hardcoded paths that break on Colab vs local?
  - Do toggle variables (USE_PKA, STRICT_PKA, PREDICT_TEST) work in all combinations?

### 3. Data & Results

- `data/petase_challenge_data/` — Input CSVs, verify column names, row counts, sequence validity
- `results/` — Output CSVs, verify schema consistency between scripts
  - Do `esm2_scores.csv` columns match what `generate_submission_v2.py` expects?
  - Do `pka_features_test_v2.csv` columns match what `--pka-v2` loads?
  - Is `wt_idx` mapping consistent across all files?

### 4. Scientific Validity

- Is the wildtype-marginal scoring (Meier et al. 2021) implemented correctly?
- Are Henderson-Hasselbalch protonation fractions computed correctly?
- Do pH-specific charge heuristics match the cited literature (Charlier 2024, Lu 2022, Bell 2022)?
- Is the catalytic His identification logic sound (lowest pKa His = catalytic)?
- Are z-score normalizations applied correctly (especially mutant-only z_delta_ll)?
- Does the sign convention for delta_pka make physical sense?

### 5. Weight Vectors & Scoring

- Verify ALL weight vectors sum to exactly 1.0 in every branch:
  - `compute_activity_1`: v3, v4, v5, v6 (with and without emb_dist)
  - `compute_activity_2`: v3, v4, v5, v6 (with and without emb_dist)
  - `compute_expression`: single branch
- Are the relative weight magnitudes justified by the scientific rationale?
- Does the z_logit/emb_dist conditional branching work correctly in all paths?

### 6. Edge Cases & Robustness

- What happens when pKAI returns empty results? When PROPKA crashes on a malformed PDB?
- What happens with all-NaN columns? Zero-variance columns? Constant z-scores?
- Are there division-by-zero risks in zscore(), protonation_fraction(), rank_scale()?
- What if a test sequence matches no WT (wt_idx=None)?
- What if `--max-seqs` is used — do downstream scripts handle partial results?

### 7. Consistency Between Components

- Does `predict_structures.py --mode test` output naming (`test_0000.pdb`) match what `compute_pka_features_v2.py` expects?
- Does the notebook's `PREDICT_TEST` toggle properly gate all downstream v2/v6 cells?
- Are column names in CSV outputs consistent between producing and consuming scripts?
- Do CLI flag combinations work correctly (`--no-pka --pka-v2` rejected? `--require-pka --pka-v2` works?)

## Tools & Approaches to Use

Use every available tool and agent capability to thoroughly audit:

- **Read every script file** — line by line where logic is complex
- **Grep for patterns** — search for `zscore`, `rank_scale`, `protonation_fraction`, `wt_idx`, `delta_ll`, `NaN`, `fillna`, hardcoded numbers, magic constants
- **Glob for files** — find all `.py`, `.csv`, `.ipynb` files, check for orphaned or unexpected files
- **Run syntax checks** — `python -c "import ast; ast.parse(open(f).read())"` on all scripts
- **Validate CSV schemas** — read headers of all result CSVs, check column counts and types
- **Cross-reference imports** — verify all imported modules are available (propka, pKAI, fair-esm, scipy, etc.)
- **Check the notebook JSON** — parse cell structure, verify cell IDs, check for execution order issues
- **Search for TODO/FIXME/HACK/XXX** comments
- **Search for commented-out code** that might indicate incomplete changes
- **Verify backward compatibility** — does v4 still work identically when v6 code is present but `--pka-v2` is not used?

## Report Format

Structure the report as:

### Executive Summary
- Overall pipeline health (1-2 paragraphs)
- Critical issues count, important issues count, minor issues count

### Critical Issues (blocks correctness or produces wrong results)
For each:
- File, line number(s)
- Description of the bug
- Impact on predictions
- Suggested fix

### Important Issues (may affect result quality or reliability)
Same format as above.

### Minor Issues (style, documentation, maintainability)
Same format as above.

### Scientific Review
- Assessment of each scoring component's validity
- Any questionable assumptions or simplifications
- Literature citations that may be misapplied

### Consistency Matrix
- Table showing which scripts produce which files, and which scripts consume them
- Flag any mismatches in expected vs actual column names or row counts

### Recommendations
- Priority-ordered list of improvements
- Estimate of impact on prediction quality (high/medium/low)

Be specific. Cite line numbers. Show the problematic code. Do not hand-wave.
