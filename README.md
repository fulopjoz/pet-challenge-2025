# PET Challenge 2025 - Zero-Shot Prediction Pipeline

Predicting PETase enzyme **activity** and **expression** for 4988 variant sequences using protein language models (PLMs), with ML baselines validated against literature Tm data.

**Challenge**: ALIGN Bio PET Challenge 2025, Zero-Shot Track
**Evaluation metric**: NDCG (ranking quality)
**Targets**: `activity_1`, `activity_2` (umol TPA/min*mg), `expression` (mg/mL)

## Quick Start (Google Colab)

```bash
# 1. Upload this folder to Colab, then:
!bash setup_colab.sh

# 2. Run ESM2 scoring (~5 min on T4 GPU)
!python scripts/esm2_zero_shot_scoring.py

# 3. Run ESMC scoring (~5 min on T4 GPU)
!python scripts/esmc_scoring.py

# 4. Validate scores (compare models, sanity checks)
!python scripts/validate_scores.py

# 5. Generate submission (ESM2+ESMC ensemble)
!python scripts/generate_submission.py

# Output: results/submission_zero_shot.csv
```

For ESM2-only submission: `python scripts/generate_submission.py --esm2-only`

## Methods

### Approach 1: PLM Zero-Shot Scoring (Primary)

We use **wildtype-marginal scoring** from protein language models. This requires zero training data — the PLM's evolutionary knowledge serves as the prior.

**How it works:**
1. Run one forward pass per wild-type sequence through the PLM
2. Extract log-probabilities at each position: `P(aa | context)`
3. For each single-point mutant, compute:
   - `delta_ll = log P(mut_aa at pos) - log P(wt_aa at pos)` — mutation effect
   - `abs_ll = mean log P(native_aa)` over all positions — absolute fitness
   - `entropy` — positional uncertainty (lower = more conserved)
   - `logit_native` — raw model confidence in native residue
   - `joint_ll` — joint log-likelihood over all standard amino acids

**Models used:**
- **ESM2-650M** (Meta, 2022): Gold standard for zero-shot variant effect prediction. Top performer on ProteinGym benchmarks. Package: `fair-esm`
- **ESMC-600M** (EvolutionaryScale, 2024): Newer model rivaling ESM2-3B performance. Best zero-shot Spearman (0.49) in Kral 2025 thesis on antibody Tm. Package: `esm`

**Score combination for predictions:**
- Activity 1 (pH 5.5): `0.5*z(delta_ll) + 0.3*z(abs_ll) + 0.1*z(-entropy) + 0.1*z(logit_native)`
- Activity 2 (pH 9.0): `0.35*z(delta_ll) + 0.35*z(abs_ll) + 0.2*z(-entropy) + 0.1*z(logit_native)`
- Expression: `0.2*z(delta_ll) + 0.4*z(abs_ll) + 0.2*z(-entropy) + 0.2*z(logit_native)`
- Ensemble: average of ESM2 and ESMC model-level predictions

Rationale: delta_ll is the primary activity signal (mutation tolerance). abs_ll dominates expression (protein foldability/stability correlates with expressibility).

### Approach 2: ML Baselines (Validation)

Traditional ML models validated on **12 verified IsPETase Tm values** from published papers:
- **Ridge Regression** (primary for small n)
- **Random Forest** (comparison)
- **XGBoost** (comparison)
- **Lasso**, **ElasticNet**, **Feature-selected Ridge**

37 sequence-based features: AA composition (20), physicochemical properties (7), active-site distances (3), mutation count features (3), structural proxies (4).

For model fitting and LOOCV, duplicate feature rows (same mutation/sequence with different reported Tm from different assays) are aggregated by mean Tm to avoid leakage-like artifacts in validation.
These models validate the approach but are NOT used for the challenge submission (insufficient training data for activity/expression prediction).

## Validation

### What was validated

1. **Biological prior checks** (automated in `validate_scores.py`):
   - WT sequences score higher than mutants on abs_ll (evolutionary optimum)
   - >50% of mutations have negative delta_ll (most mutations are deleterious)
   - Score distributions are non-degenerate (sufficient variance for ranking)

2. **Cross-model agreement**:
   - Spearman correlation between ESM2 and ESMC predictions
   - High agreement (rho > 0.7) indicates robust signal

3. **Known Tm validation** (12 verified IsPETase variants):
   - Source: Brott et al. 2022 (nanoDSF, identical conditions), Lu 2022, Son 2019, Cui 2021
   - Tm range: 45.1-81.1 C (WT to DuraPETase+SS)
   - ML Ridge LOOCV: validated on these 12 samples

4. **Sanity checks on submission**:
   - WT activity > mutant activity (confirmed: 4.55 vs 2.36)
   - WT expression > mutant expression (confirmed: 2.30 vs 1.45)
   - Score correlations match biological expectations

### What was NOT validated

- No ground truth available for the 4988 challenge sequences (this IS the test set)
- Activity/expression predictions are based on evolutionary conservation, not direct biochemical assays
- The challenge evaluation (NDCG) will be the true validation

## Data

### Challenge Data (`data/petase_challenge_data/`)

| File | Description |
|------|-------------|
| `pet-2025-wildtype-cds.csv` | 313 unique WT PETase sequences with codon-optimized CDS |
| `predictive-pet-zero-shot-test-2025.csv` | 4988 test sequences (314 WT + 4674 single mutants) |
| `pet-2025-experimental-methods.pdf` | Lab protocol (BL21(DE3), pET28a, IMAC, activity at 30C) |
| `pet-2025-expression-vector.gb` | pET28a vector GenBank (T7, 6xHis, KanR) |
| `predictive-phase.pdf` | Challenge website screenshot |

### Validation Data (`data/`)

| File | Description |
|------|-------------|
| `mutations_dataset.csv` | 14 verified PETase Tm values from 4 papers (12 IsPETase + 2 LCC) |
| `features_matrix.csv` | 37 sequence features for the 12 IsPETase variants |
| `petase_ideonella.fasta` | WT IsPETase (A0A0K8P6T7, 290 aa) |
| `lcc_cutinase.fasta` | LCC reference (G9BY57) |

### Literature (`data/validation/`)

PDFs of key papers used for Tm verification:
- Brott et al. 2022, Eng. Life Sci. (7 IsPETase Tm values, nanoDSF)
- Lu et al. 2022, Nature (FAST-PETase)
- Son et al. 2019, ACS Catal (ThermoPETase)
- Cui et al. 2021, ACS Catal (DuraPETase)
- Kral 2025, MSc Thesis (zero-shot scoring methodology reference)

## Scripts

### PLM Scoring Pipeline (Zero-Shot Track)

| Script | What it does | Runtime (T4) |
|--------|-------------|-------------|
| `scripts/esm2_zero_shot_scoring.py` | ESM2-650M WT-marginal scoring for 4988 sequences | ~5 min |
| `scripts/esmc_scoring.py` | ESMC-600M WT-marginal scoring for 4988 sequences | ~5 min |
| `scripts/generate_submission.py` | Combine scores into final submission CSV | <1 sec |
| `scripts/validate_scores.py` | Compare models, sanity checks, Tm validation | <1 sec |

### ML Baseline Pipeline (Tm Prediction)

| Script | What it does |
|--------|-------------|
| `scripts/extract_mutations.py` | Generate `mutations_dataset.csv` from verified literature data |
| `scripts/feature_extraction.py` | Extract 37 sequence features for each variant |
| `scripts/train_rf_baseline.py` | Train Ridge + Random Forest on Tm data (LOOCV) |
| `scripts/alternative_models.py` | Compare Ridge, Lasso, ElasticNet, XGBoost on Tm data |

## Results

Pre-computed results from local run (ESM2-650M on NVIDIA 840M):

| File | Description |
|------|-------------|
| `results/esm2_scores.csv` | ESM2 scores for all 4988 test sequences |
| `results/submission_zero_shot.csv` | Final submission (ESM2-only) |
| `results/model_comparison.csv` | ML model comparison on Tm validation data |
| `results/rf_feature_importance.csv` | Random Forest feature importance |
| `results/*.pkl` | Trained ML model files |

## Challenge Details

- **Organism**: PETase variants from diverse sources (313 unique scaffolds)
- **Test set**: 4988 sequences (314 WT-identical + 4674 single-point mutants)
- **3 main scaffolds**: WT0 (1560 variants, len=259), WT1 (1559, len=257), WT2 (1559, len=259)
- **Expression**: pET28a vector, BL21(DE3), IMAC purification, 6xHis tag
- **Activity assay**: Powdered PET, 30C, pH 5.5 (citrate) and pH 9.0 (glycine-NaOH), mass spec detection of TPA
- **Metric**: NDCG (Normalized Discounted Cumulative Gain) — ranking quality

## References

1. **Brott et al. (2022)** Eng. Life Sci. DOI: 10.1002/elsc.202100105 — Comprehensive Tm comparison of IsPETase variants (nanoDSF)
2. **Lu et al. (2022)** Nature. DOI: 10.1038/s41586-022-04599-z — FAST-PETase, ML-designed thermostable variant
3. **Son et al. (2019)** ACS Catal. DOI: 10.1021/acscatal.9b00568 — ThermoPETase (S121E/D186H/R280A)
4. **Cui et al. (2021)** ACS Catal. DOI: 10.1021/acscatal.0c05126 — DuraPETase (9-point mutant)
5. **Kral (2025)** MSc Thesis, Charles University — Zero-shot PLM scoring methodology for protein thermostability
6. **Lin et al. (2023)** Science. DOI: 10.1126/science.ade2574 — ESM2 (evolutionary scale modeling)
7. **EvolutionaryScale (2024)** — ESMC (ESM Cambrian), efficient protein language models
8. **Meier et al. (2021)** NeurIPS. DOI: 10.1101/2021.07.09.450648 — WT-marginal scoring method for variant effects

## Folder Structure

```
petase_tournament/
├── README.md                    # This file
├── requirements.txt             # Python dependencies (local)
├── requirements_colab.txt       # Python dependencies (Colab)
├── setup_colab.sh               # Colab setup script
├── data/
│   ├── petase_challenge_data/   # Challenge test data (from ALIGN Bio)
│   ├── mutations_dataset.csv    # Verified Tm data (12 IsPETase + 2 LCC)
│   ├── features_matrix.csv      # Sequence features for ML validation
│   ├── validation/              # Literature PDFs + extraction logs
│   └── *.fasta                  # Reference sequences
├── scripts/
│   ├── esm2_zero_shot_scoring.py    # ESM2-650M scoring
│   ├── esmc_scoring.py              # ESMC-600M scoring
│   ├── generate_submission.py       # Combine scores → submission
│   ├── validate_scores.py           # Compare & validate all models
│   ├── extract_mutations.py         # Tm dataset from literature
│   ├── feature_extraction.py        # Sequence feature extraction
│   ├── train_rf_baseline.py         # Ridge/RF training
│   └── alternative_models.py        # Model comparison (Ridge/Lasso/XGB)
├── results/
│   ├── esm2_scores.csv              # ESM2 scores (4988 sequences)
│   ├── esmc_scores.csv              # ESMC scores (after Colab run)
│   ├── submission_zero_shot.csv     # Final submission
│   └── model_comparison.csv         # ML model comparison
└── literature/
    ├── LITERATURE_REVIEW.md
    └── citations.md
```
