# PETase Tm Validation Data Extraction Log

**Date:** 2026-02-22
**Goal:** Extract IsPETase variant Tm values from 4 SI PDFs

---

## Target Papers

1. Son 2019 (FAST-PETase) - https://pubs.acs.org/doi/10.1021/acscatal.9b00568
2. Cui 2021 (DuraPETase ML) - https://pubs.acs.org/doi/10.1021/acscatal.0c05126
3. Lu 2022 (Nature) - https://www.nature.com/articles/s41586-022-04599-z
4. Brott 2022 - https://onlinelibrary.wiley.com/doi/10.1002/elsc.202100105

---

## Download Status

| Paper | URL | Status | File Size |
|-------|-----|--------|-----------|
| Son 2019 | ACS Supplementary | ❌ HTTP 403 Forbidden | - |
| Cui 2021 | ACS Supplementary | ❌ HTTP 403 Forbidden | - |
| Lu 2022 | Springer Supplementary | ✅ Downloaded | 2.48 MB (43 pages) |
| Brott 2022 | Wiley Supplementary | ❌ HTTP 403 Forbidden | - |

---

## Lu 2022 (Nature) - FAST-PETase Analysis

### File: lu2022_suppl.pdf

### Content Summary
- **Pages:** 43
- **Structure:**
  - Pages 1-10: Methods, affiliations, discussion
  - Pages 11-16: Prediction tables (probability distributions, rankings)
  - Pages 18-22: PET degradation rate tables (postconsumer products)
  - Pages 23-30: Supplementary figures

### Table Extractions

#### Page 12-16: Probability/Ranking Tables
- **Table:** Predicted mutations ranked by fold change
- **Content:** Probability distributions, rankings, avg_log_ratio
- **Examples of mutations found:**
  - S121E, M262L, N233K, T140D, S58E, S169A, Q119L, N225C, T270V, N114T
  - Q91I, and many more (38 mutations in ranking table)
- **Tm Values:** NO explicit Tm data found
- **Activity Data:** NOT in this table (probability predictions only)

#### Pages 18-22: Postconsumer PET Degradation
- **Table:** Sample characteristics, degradation times, crystallinity
- **Content:** 46 real-world PET samples tested
- **Tm Values:** NO Tm data (activity/degradation rates only)

#### Pages 25-30: Activity Figures
- **Figure 2:** Activity comparison at different temps (30°C, 40°C)
- **Content:** FAST-PETase vs WT, Thermo, Dura, LCC, ICCM
- **Tm Values:** NO thermal stability data (only activity at fixed temperatures)

### Findings Summary

| Data Type | Found? | Details |
|-----------|--------|---------|
| Predicted mutations (MutCompute) | ✅ YES | 38 mutations ranked |
| Tm values | ❌ NO | No explicit thermal stability data |
| Activity measurements | ✅ YES | Relative activity at 30°C, 40°C |
| Postconsumer PET degradation | ✅ YES | Time to complete degradation for 46 samples |

### Mutations Mentioned (from Page 16)
```
S121E, M262L (rank 1,2)
N233K, T140D (rank 3,4)
S58E, S169A, Q119L, N225C, T270V, N114T (rank 5-10)
Q91I, H92D, N114V, L119F, H153Y, W156H (and more)
```

### Tm Data: NOT FOUND

The Lu 2022 supplementary information **does NOT contain Tm values** for FAST-PETase variants.

**Reason:** This paper focuses on:
1. MutCompute predictions (probabilities, not stability)
2. Activity measurements at fixed temperatures (30°C, 40°C)
3. Real-world PET degradation rates

**Possibility:** Tm data might be in:
- Main paper (not supplemental)
- Extended Data figures (mentioned in text)
- Source data files (need to fetch differently)

---

## FireProtDB Search

### Query: UniProt A0A0K8P6T7

**Status:** ❌ FAILED
- **Error:** ENOTFOUND - Domain not accessible
- **Alternative URLs Attempted:**
  - https://fireprotdb.bioinfo.cnio.es - DNS failed
  - No backup URL available

### Status
**No FireProtDB data retrieved**

---

## UniProt A0A0K8P6T7 Query

### Basic Information
- **Protein:** Poly(ethylene terephthalate) hydrolase
- **Length:** 290 aa
- **Organism:** *Piscinibacter sakaiensis* (formerly *Ideonella sakaiensis*)

### Annotations Found
- **Catalytic Activity:** Ester bond hydrolysis
- **Active Site:** Ser160, Asp206, His237 (catalytic triad)

### Natural Variants
- **Count:** 0 polymorphic variants annotated
- **Thermal Annotations:** None

### Tm Data: NOT FOUND

UniProt database does not contain Tm values for PETase variants.

---

## ACS/Wiley Supplementary Download Issues

### Problem: HTTP 403 Forbidden

**Affected Papers:**
1. **Son 2019** (ACS Catalysis) - 403 Forbidden
2. **Cui 2021** (ACS Catalysis) - 403 Forbidden  
3. **Brott 2022** (ELS - Wiley) - 403 Forbidden

### Why 403 Forbidden?

ACS and Wiley require:
1. Session cookies (institutional access needed)
2. Referer headers
3. Potentially user authentication

### Workarounds Attempted

1. **User-Agent Header:** ❌ Still 403
2. **Multiple URLs:** ❌ All blocked
3. **Direct URL encoding:** ❌ No effect

### Alternative Approaches Needed

Since direct downloads blocked, options are:
1. Ask user to download manually
2. Use academic network access
3. Find preprint/arXiv versions
4. Extract data from references in other papers

---

## DATA STATUS SUMMARY

| Source | Download Status | Tm Values Found | Activity Data Found |
|--------|----------------|-----------------|-------------------|
| Son 2019 SI | ❌ Blocked | ? | ? |
| Cui 2021 SI | ❌ Blocked | ? | ? |
| Lu 2022 SI | ✅ Downloaded | ❌ NOT PRESENT | ✅ YES (30°C, 40°C) |
| Brott 2022 SI | ❌ Blocked | ? | ? |
| FireProtDB | ❌ DNS failed | - | - |
| UniProt A0A0K8P6T7 | ✅ Queried | ❌ NOT PRESENT | - |

---

## NEW DATA COLLECTED

### From Lu 2022 (NO Tm values, but useful):

#### Mutations with Predictions (38 total):
```
S121E, M262L, N233K, T140D, S58E, S169A, Q119L, N225C, T270V, N114T,
Q91I, H92D, N114V, L119F, H153Y, W156H, and 22 more (see Page 16)
```

**Note:** These are predictions, NOT experimental Tm values.

---

## VALIDATION DATASET STATUS

**Current State:**
- ❌ ZERO (0) new Tm values extracted
- ✅ 38 mutation positions identified (but no numerical Tm data)
- ✅ FAST-PETase activity data at 30°C, 40°C (relative to WT)

**Target:** 10-18 new Tm entries
**Actual:** 0 entries

---

## OBSTACLES & NEXT STEPS

### Primary Obstacles

1. **Publisher Access Restrictions (403 Forbidden)**
   - Son 2019, Cui 2021, Brott 2022 SIs blocked
   - Need institutional account or manual download

2. **Tm Data Not in Lu 2022 Supplemental**
   - Lu 2022 only has activity data, not stability
   - Tm might be in main paper or Extended Data

3. **FireProtDB Access Failed**
   - DNS resolution failed
   - Alternative: ProTherm, Thermonet

### Recommended Path Forward

#### Option 1: Manual Download (Fastest)
- User downloads 3 blocked SI PDFs from institutional/VPN access
- I extract tables using pdfplumber
- Time: User 10 min + Automated extraction 10 min

#### Option 2: Query Main Papers
- Fetch main papers (web_fetch might work)
- Extract data from main paper tables
- Time: 20 min

#### Option 3: Alternative Databases
- Try ProTherm (if accessible)
- Try Thermonet
- GitHub repos of authors (often include data)
- Time: 30 min

#### Option 4: Literature Mining
- Search papers citing these key works
- Find secondary papers that report Tm values
- Build dataset from multiple sources
- Time: 1-2 hours

---

## FILES CREATED

```
validation/
├── lu2022_suppl.pdf                          ✅ Downloaded
├── validation_mutations.csv                  ❌ Empty (no Tm found)
└── EXTRACTION_LOG.md                         ✅ This file
```

---

## FINDINGS: Tm DATA AVAILABILITY

### Confirmed PRESENT (in other papers, not yet accessed):
- Austin 2018 (DuraPETase) - Has Tm values (in training set)
- Son 2020 (4p variant) - Has Tm values (in training set)
- Lee SH 2023 (Z1-PETase) - Has Tm values (in training set)

### SUSPECTED PRESENT (needs access):
- Son 2019 (FAST-PETase single mutants)
- Cui 2021 (Single mutant Tm table)
- Brott 2022 (DuraPETase+SS combos)

### POSSIBLY MISSING:
- Lu 2022 main paper (not SI)
- Extended Data figures
- Source data files

---

## RECOMMENDATION

**Best Path Forward:**
1. User downloads 3 blocked PDFs manually (10 min)
2. Extract Tm values using automation (10 min)
3. Add to validation_mutations.csv
4. Aim for 10-15 new entries

**Alternative:**
- Query main papers (Son 2019, Cui 2021) via web_fetch
- Extract table data from main text
- May find Tm values there

**Current Blocking Factor:**  
Publisher access restrictions preventing SI download

---

**Next Action:** Request user to manually download or provide alternative access method
