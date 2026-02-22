# PETase Literature Review - Executive Summary

**Date:** 2026-02-22
**Literature Source:** PubMed (134+ papers) + Academic search
**Focus:** PETase enzyme engineering papers

---

## Quick Stats

| Metric | Value |
|--------|-------|
| PubMed results found | 134+ papers |
| Semantic Scholar results | 56,787 total |
| Top papers analyzed | 25+ |
| Time frame | 2018-2026 |
| Recent papers (2024-2026) | 8 found |

---

## Top 10 Engineering Papers (Ranked by Impact)

### Gold Standard (Must-Read)

1. **Austin 2018 (PNAS)** - DuraPETase foundation
   - W159H, S238F mutations = game changer
   - First structure-guided engineering

2. **Knott 2020 (PNAS)** - Two-enzyme system
   - MHETase structure revealed
   - Synergy pathway validated

3. **Burgin 2024 (Comm Chem)** - Reaction mechanism
   - Computational mechanism elucidation
   - Rate-limiting step: acylation (not deacylation)

### Recent High-Impact (2023-2026)

4. **Wang 2024 (Eco Env Safety)** - HotPETase + CBM
   - 10x activity at 50°C
   - Domain fusion strategy

5. **Lee YL 2024 (IJBM)** - CLEA immobilization
   - Industrial application focus
   - Reusable enzyme platform

6. **Zurier 2023 (Biotech Bioeng)** - High-throughput screening
   - Cell-free protein synthesis
   - 48-hour mutant testing

7. **Lee SH 2023 (J Haz Mat)** - Z1-PETase
   - Multi-parameter optimization
   - 20x protein yield improvement

8. **Meng 2021 (IJBM)** - Premuse ML tool
   - 1486 homologs analyzed
   - Statistical mutation selection

### Specialized Approaches

9. **Blázquez-Sánchez 2023 (Prot Sci)** - Antarctic PETase
   - Loop exchange engineering
   - Cold-adapted applications

10. **Bhattacharya 2026 (FEBS J)** - Latest work
    - Ongoing engineering
    - Product inhibition focus

---

## Key Mutation Catalog

### Primary Stabilizing Mutations
```
S121E  → +stability (Son 2019 ACS Catal)
D186H  → +stability (Son 2019 ACS Catal)
R280A  → +stability (Son 2019 ACS Catal)
W159H  → +stability (Cui 2021 ACS Catal, part of DuraPETase)
F229Y  → +stability (Meng 2021)
```
NOTE: Austin 2018 reported W159H/S238F structures but did NOT report Tm values.

### Activity-Enhancing Mutations
```
R280A  → +activity (multiple papers)
N233K  → +activity & stability (Lu 2022, part of FAST-PETase)
R224Q  → +stability (Lu 2022, part of FAST-PETase)
```

### Multi-Mutation Variants
- **ThermoPETase:** S121E/D186H/R280A (Son 2019 ACS Catal, Tm ~55-57 C)
- **DuraPETase:** 9 mutations including ThermoPETase core (Cui 2021 ACS Catal, Tm ~75-77 C)
- **FAST-PETase:** N233K/R224Q/S121E/D186H/R280A (Lu 2022 Nature, Tm 67.8 C)
- **Z1-PETase:** Multi-parameter optimized (Lee SH 2023)
- **HotPETase-CBM:** Domain fusion variant (Wang 2024)

---

## Engineering Trends (2018-2026)

| Period | Focus | Techniques |
|--------|-------|------------|
| 2018-2020 | Single-property optimization | Structure-guided, rational design |
| 2020-2022 | Stability + activity | Directed evolution, ML-guided (Premuse) |
| 2022-2024 | Multiparameter engineering | High-throughput screening, cell-free systems |
| 2024-2026 | Industrial application | Immobilization, CBM fusion, PEGylation |
| Future | ? | AI/LLM-guided design ? |

---

## What's Missing

### Classic Papers (Not Found in PubMed Keyword Search)

1. **Yoshida 2016 (Science)** - Original discovery
   - Probably indexed differently
   - Need DOI search: 10.1126/science.aai6419

2. **Son 2019 (ACS Catalysis)** - ThermoPETase
   - S121E/D186H/R280A thermostable variant
   - DOI: 10.1021/acscatal.9b00568

3. **Tournier 2020 (Nature)** - Carbios 10,000x variant
   - Industrial breakthrough
   - DOI: 10.1038/s41586-020-2643-5

4. **Cui 2021 (ACS Catalysis)** - DuraPETase
   - Rational design, 9-mutation variant
   - DOI: 10.1021/acscatal.0c05126

### Action Items
- [ ] Search by DOI for missing papers
- [ ] Extract FASTA sequences for all variants
- [ ] Build mutation-activity database
- [ ] Normalize activity scores across papers

---

## Tournament Relevance

### Papers Most Useful for Competition

1. **Austin 2018** - Baseline mutations
2. **Meng 2021** - ML approach example
3. **Zurier 2023** - Screening methodology
4. **Lee SH 2023** - Multi-objective optimization
5. **Wang 2024** - Non-canonical strategies (domain fusion)

### Data Extraction Priorities

```python
# Data schema to extract
{
    "paper_id": "Austin2018",
    "mutations": ["W159H", "S238F", "R280A"],
    "activity_relative": 1.5,  # vs wild-type
    "tm_delta": 30,  # °C change
    "assay_conditions": {
        "temperature": 50,
        "substrate": "amorphous PET",
        "units": "nmol/min/mg"
    }
}
```

---

## Quick Reference Links

| Paper | PubMed | DOI |
|-------|--------|-----|
| Austin 2018 | 29666242 | 10.1073/pnas.1718804115 |
| Knott 2020 | 32989159 | 10.1073/pnas.2006753117 |
| Burgin 2024 | 38538850 | 10.1038/s42004-024-01154-x |
| Wang 2024 | 38833982 | 10.1016/j.ecoenv.2024.116540 |
| Lee 2024 | 38382786 | 10.1016/j.ijbiomac.2024.130284 |
| Zurier 2023 | 36575047 | 10.1002/bit.28319 |
| Lee SH 2023 | 37595467 | 10.1016/j.jhazmat.2023.132297 |
| Meng 2021 | 33753197 | 10.1016/j.ijbiomac.2021.03.058 |
| Blázquez 2023 | 37574805 | 10.1002/pro.4757 |
| Bhattacharya 2026 | 40847613 | 10.1111/febs.70228 |

---

## Bottom Line

The PETase engineering field is mature with:
- ✅ **Core variants defined** (DuraPETase, FAST-PETase)
- ✅ **Structure understood** (crystal structures available)
- ✅ **Mechanism elucidated** (rate-limiting known)
- ✅ **Screening pipelines** established (cell-free, high-throughput)
- ⚠️ **ML methods emerging** (Premuse, ESM)
- ⏳ **Clinical/industrial scaling** in progress

**Tournament is well-timed** - enough data for ML models, but still room for novel discoveries!

---

**Files Created:**
- `TOP10_ENG_PAPERS.md` - Detailed analysis
- `pubmed_citations.md` - Raw search results
- This summary

**Next:**
1. Fetch missing papers by DOI
2. Extract mutation tables
3. Get variant FASTA sequences
4. Build training dataset
