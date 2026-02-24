# PET Challenge 2025 - Zero-Shot Prediction Pipeline (v4)

Predicting PETase enzyme **activity** and **expression** for 4988 variant sequences using protein language models (PLMs), CDS features, and amino acid property changes.

**Challenge**: ALIGN Bio PET Challenge 2025, Zero-Shot Track
**Evaluation metric**: NDCG (ranking quality)
**Targets**: `activity_1` (pH 5.5), `activity_2` (pH 9.0) in umol TPA/min·mg, `expression` (mg/mL)

## Quick Start (Google Colab)

```bash
# 1. Clone and setup
!git clone https://github.com/fulopjoz/pet-challenge-2025.git
%cd pet-challenge-2025
!bash setup_colab.sh

# 2. ESM2 scoring (~5 min on T4 GPU)
!python scripts/esm2_zero_shot_scoring.py

# 3. (Optional) ESMC scoring (~5 min on T4)
!python scripts/esmc_scoring.py

# 4. Compute CDS + mutation features (<1 sec)
!python scripts/compute_cds_features.py

# 5. Generate v4 submission
!python scripts/generate_submission_v2.py --esm2-only

# 6. Validate
!python scripts/validate_v4.py

# Output: results/submission_zero_shot_v2.csv
```

The v1 baseline (PLM-only, no CDS/mutation features) can still be generated with `python scripts/generate_submission.py --esm2-only`.

## Methods

### PLM Zero-Shot Scoring

We use **wildtype-marginal scoring** (Meier et al. 2021) - one forward pass per WT scaffold, then score mutations from the WT log-probability distribution. No training data required.

**Scores per sequence:**
- `delta_ll` - log P(mut_aa) - log P(wt_aa) at mutation site (mutation effect)
- `abs_ll` - mean log P(native_aa) over all positions (absolute fitness)
- `entropy` - mean positional entropy (lower = more conserved)
- `logit_native` - mean raw logit for native residue
- `joint_ll` - joint log-likelihood over 20 standard AAs
- `entropy_at_site` - positional entropy at the mutation site (v4)
- `native_ll_at_site` - native log-probability at the mutation site (v4)

**Models:**
- **ESM2-650M** (Meta, Lin et al. 2023): Gold standard for zero-shot variant effect prediction. Package: `fair-esm`
- **ESMC-600M** (EvolutionaryScale, 2024): Newer model rivaling ESM2-3B. Package: `esm`

### Feature Engineering (v2+)

**CDS features** (per WT scaffold, `compute_cds_features.py`):
- 5' AT-richness (codons 2-8): strongest single predictor of E. coli expression (Nieuwkoop et al. 2023, r=0.762)
- Rare codon fraction (E. coli low-abundance tRNAs)
- GC content (total, 5', first 50 nt)

**Mutation features** (per test sequence):
- Delta hydrophobicity, delta charge, delta MW (Kyte-Doolittle scale)
- Charge sign change (relevant for pH-dependent activity)
- Relative mutation position

### Scoring Formulas (v4)

All features are z-score normalized. delta_ll is z-scored only among mutants (WTs get z_delta=0) to prevent WT over-dominance in rankings.

**Activity 1 (pH 5.5)** - suboptimal pH, stability + negative charge matter:
```
0.30·z_delta + 0.25·z_abs + 0.10·z_entropy + 0.05·z_logit
+ 0.05·z_entropy_at_site + 0.05·z_native_ll_at_site
+ 0.10·z(-delta_charge) + 0.10·z(-abs_delta_hydro)
```
Rationale: At pH 5.5 the catalytic His (pKa ~4.9, Charlier 2024 on LCC) is ~80% deprotonated. Negative charge mutations can lower His pKa and maintain activity.

**Activity 2 (pH 9.0)** - near-optimal pH, fitness dominates:
```
0.35·z_delta + 0.20·z_abs + 0.10·z_entropy + 0.05·z_logit
+ 0.05·z_entropy_at_site + 0.05·z_native_ll_at_site
+ 0.10·z(delta_charge) + 0.10·z(-abs_delta_hydro)
```
Rationale: At pH 9.0, His is >99.9% deprotonated. Positive charge aids PET binding at alkaline pH (Lu 2022, Bell 2022).

**Expression (mg/mL)** - CDS structure + PLM foldability:
```
0.30·z_delta + 0.15·z_abs + 0.10·z_entropy + 0.10·z_logit
+ 0.15·z(cds_at_5prime) + 0.10·z(-cds_rare_codon)
+ 0.10·z(-abs_delta_hydro)
```
Rationale: 5' AT-richness reduces mRNA secondary structure, improving ribosome initiation. Rare codons cause ribosome stalling.

### ML Baselines (Validation Only)

Ridge regression validated on **8 aggregated IsPETase Tm values** (LOOCV, from 14 literature data points across 4 papers). These models validate that PLM features capture thermostability signal but are NOT used for the challenge submission.

## Validation

### Automated checks (`validate_scores.py`)
- WT sequences score higher than mutants on abs_ll
- >50% of mutations have negative delta_ll (deleterious)
- Score distributions have sufficient variance for ranking

### v4 validation checks (`validate_v4.py`)
1. **Per-site substitution spread**: Different substitutions at the same position get different delta_ll values
2. **Top-K WT fraction**: Top-ranked entries include both WTs and mutants (not WT-dominated)
3. **Within-WT rank correlation**: act1 vs act2 rankings differ within each WT scaffold (charge direction effect)
4. **entropy_at_site partial correlation**: Site entropy adds signal beyond delta_ll
5. **Weight vector sums**: All weight vectors sum to 1.0
6. **Entropy sign verification**: Mutants at conserved sites get lower predicted activity

### Limitations
- No ground truth for the 4988 challenge sequences
- Validation checks confirm model consistency and biological priors, not predictive accuracy
- The challenge NDCG evaluation is the true test

## Data

### Challenge Data (`data/petase_challenge_data/`)

| File | Description |
|------|-------------|
| `pet-2025-wildtype-cds.csv` | 313 unique WT PETase sequences with codon-optimized CDS |
| `predictive-pet-zero-shot-test-2025.csv` | 4988 test sequences (314 WT + 4674 single mutants) |
| `pet-2025-experimental-methods.pdf` | Lab protocol (BL21(DE3), pET28a, IMAC, activity at 30C) |
| `pet-2025-expression-vector.gb` | pET28a vector GenBank (T7, 6xHis, KanR) |

### Validation Data (`data/`)

| File | Description |
|------|-------------|
| `mutations_dataset.csv` | 31 verified IsPETase Tm values from 4 papers |
| `features_matrix.csv` | 37 sequence features for all 31 IsPETase variants |
| `petase_ideonella.fasta` | WT IsPETase (A0A0K8P6T7, 290 aa) |

## Scripts

### v4 Pipeline (Primary)

| Script | Description | Runtime |
|--------|-------------|---------|
| `esm2_zero_shot_scoring.py` | ESM2-650M scoring for 4988 sequences (includes site features) | ~5 min (T4) |
| `esmc_scoring.py` | ESMC-600M scoring | ~5 min (T4) |
| `compute_cds_features.py` | CDS + mutation feature extraction | <1 sec |
| `generate_submission_v2.py` | v4 submission with PLM + CDS + mutation features | <1 sec |
| `validate_v4.py` | 6 validation checks for v4 submission | <1 sec |
| `validate_scores.py` | Basic PLM sanity checks and cross-model comparison | <1 sec |
| `patch_site_features.py` | Patch entropy_at_site/native_ll_at_site into existing scores (CPU) | ~10 min |

### ML Baseline Pipeline (Validation Only)

| Script | Description |
|--------|-------------|
| `extract_mutations.py` | Generate `mutations_dataset.csv` from literature |
| `feature_extraction.py` | Extract 37 sequence features per variant |
| `train_rf_baseline.py` | Train Ridge + Random Forest on Tm data (LOOCV) |
| `alternative_models.py` | Compare Ridge, Lasso, ElasticNet, XGBoost |

### Utilities

| Script | Description |
|--------|-------------|
| `generate_submission.py` | v1 baseline submission (PLM-only, no CDS features) |
| `execute_notebook_cells.py` | Execute non-GPU notebook cells locally |

## Results

| File | Description |
|------|-------------|
| `esm2_scores.csv` | ESM2 scores (4988 rows, 11 columns incl. site features) |
| `submission_zero_shot_v2.csv` | v4 submission (primary) |
| `submission_zero_shot.csv` | v1 baseline submission |
| `cds_features.csv` | CDS features for 313 WT scaffolds |
| `mutation_features.csv` | Mutation features for 4988 test sequences |
| `model_comparison.csv` | ML model comparison on Tm validation |
| `rf_feature_importance.csv` | Random Forest feature importances |

## Notebook

`PET_Challenge_2025_Pipeline_v2.ipynb` - full pipeline walkthrough designed for Google Colab (T4 GPU). Includes data exploration, PLM scoring, feature engineering, ML validation, submission generation, and diagnostics.

## Challenge Details

- **Organism**: PETase variants from diverse sources (313 unique scaffolds)
- **Test set**: 4988 sequences (314 WT-identical + 4674 single-point mutants)
- **3 main scaffolds**: WT0 (1560 variants, len=259), WT1 (1559, len=257), WT2 (1559, len=259)
- **Expression**: pET28a vector, BL21(DE3), IMAC purification, 6xHis tag
- **Activity assay**: Powdered PET, 30C, pH 5.5 (citrate) and pH 9.0 (glycine-NaOH), mass spec detection of TPA
- **Metric**: NDCG (Normalized Discounted Cumulative Gain)

## References

1. **Meier et al. (2021)** NeurIPS. DOI: 10.1101/2021.07.09.450648 - WT-marginal scoring method
2. **Lin et al. (2023)** Science. DOI: 10.1126/science.ade2574 - ESM2
3. **EvolutionaryScale (2024)** - ESMC (ESM Cambrian)
4. **Nieuwkoop et al. (2023)** NAR, 51(5):2363-2376. PMID 36718935 - Codons 2-8 dominate E. coli expression (r=0.762)
5. **Charlier et al. (2024)** Biophys J - NMR titration of catalytic His in LCC(ICCG), pKa=4.90
6. **Lu et al. (2022)** Nature. DOI: 10.1038/s41586-022-04599-z - FAST-PETase
7. **Bell et al. (2022)** Nature Catalysis - HotPETase, activity at alkaline pH
8. **Brott et al. (2022)** Eng. Life Sci. DOI: 10.1002/elsc.202100105 - IsPETase Tm values (nanoDSF)
9. **Son et al. (2019)** ACS Catal. DOI: 10.1021/acscatal.9b00568 - ThermoPETase
10. **Cui et al. (2021)** ACS Catal. DOI: 10.1021/acscatal.0c05126 - DuraPETase
11. **Kral (2025)** MSc Thesis, Charles University - Zero-shot PLM scoring methodology
12. **Kudla et al. (2009)** Science - mRNA folding and gene expression in E. coli
13. **Han et al. (2017)** Nat Commun. DOI: 10.1038/s41467-017-02255-z - IsPETase crystal structure (1.58A)
14. **Hong et al. (2023)** Nat Commun. DOI: 10.1038/s41467-023-43455-2 - Mesophilic/thermophilic PET hydrolase engineering

## Folder Structure

```
pet-challenge-2025/
├── README.md
├── PET_Challenge_2025_Pipeline_v2.ipynb   # Full pipeline notebook (Colab)
├── requirements.txt                       # Python dependencies (local)
├── requirements_colab.txt                 # Python dependencies (Colab)
├── setup_colab.sh                         # Colab setup script
├── data/
│   ├── petase_challenge_data/             # Challenge data (from ALIGN Bio)
│   ├── mutations_dataset.csv              # Verified Tm data
│   ├── features_matrix.csv                # Sequence features for ML
│   ├── validation/                        # Literature PDFs
│   └── *.fasta                            # Reference sequences
├── scripts/
│   ├── esm2_zero_shot_scoring.py          # ESM2-650M scoring
│   ├── esmc_scoring.py                    # ESMC-600M scoring
│   ├── compute_cds_features.py            # CDS + mutation features
│   ├── generate_submission_v2.py          # v4 submission (primary)
│   ├── generate_submission.py             # v1 baseline submission
│   ├── validate_v4.py                     # v4 validation checks
│   ├── validate_scores.py                 # Basic PLM sanity checks
│   ├── patch_site_features.py             # Patch site features (CPU)
│   ├── execute_notebook_cells.py          # Execute notebook cells locally
│   ├── feature_extraction.py              # Sequence feature extraction
│   ├── train_rf_baseline.py               # ML training (Ridge/RF)
│   ├── alternative_models.py              # Model comparison
│   └── extract_mutations.py               # Tm dataset from literature
├── results/
│   ├── esm2_scores.csv                    # ESM2 scores (4988 sequences)
│   ├── submission_zero_shot_v2.csv        # v4 submission (primary)
│   ├── submission_zero_shot.csv           # v1 baseline
│   ├── cds_features.csv                   # CDS features (313 WTs)
│   ├── mutation_features.csv              # Mutation features (4988 seqs)
│   └── model_comparison.csv               # ML model comparison
└── literature/
    ├── LITERATURE_REVIEW.md
    └── citations.md
```
