# PETase Sequence Catalog

## Date: 2026-02-22

## Sequences Collected

### 1. PETase from Ideonella sakaiensis (Wild-type)
- **UniProt ID:** A0A0K8P6T7
- **Organism:** Piscinibacter sakaiensis (formerly Ideonella sakaiensis)
- **Length:** 290 aa
- **File:** petase_ideonella.fasta
- **Source:** First discovered PET-degrading enzyme (Yoshida et al., 2016)
- **Tm:** ~45 C (nanoDSF, Brott 2022); ~46 C (DSF, Son 2019 / Cui 2021)
- **Key residues:**
  - Ser160 (catalytic serine)
  - Asp206 (catalytic aspartate)
  - His237 (catalytic histidine)

### 2. LCC (Leaf-branch Compost Cutinase)
- **UniProt ID:** G9BY57
- **Organism:** Unknown prokaryotic organism
- **Length:** 293 aa
- **File:** lcc_cutinase.fasta
- **Source:** Thermophilic cutinase engineered for PET degradation
- **Tm:** ~84.7 C (DSF, Tournier 2020)
- **Notes:** Different enzyme family from IsPETase, more thermostable

## Sequence Statistics

| ID | Length | Source | Tm (C) | Notes |
|----|--------|--------|--------|-------|
| A0A0K8P6T7 | 290 | I. sakaiensis | ~45 | Wild-type, reference |
| G9BY57 | 293 | Thermophilic | ~85 | LCC, different family |

## Known Engineered Variants (with verified Tm)

### IsPETase variants (Brott et al. 2022, DOI: 10.1002/elsc.202100105)
| Variant | Mutations | Tm (C) |
|---------|-----------|--------|
| WT | - | 45.1 |
| ThermoPETase | S121E/D186H/R280A | 56.6 |
| ThermoPETase+KF | +K95N/F201I | 61.6 |
| ThermoPETase+SS | +N233C/S282C | 68.2 |
| ThermoPETase+KF+SS | +K95N/F201I/N233C/S282C | 70.8 |
| DuraPETase | 9 mutations (Cui 2021) | 75.0 |
| DuraPETase+SS | +N233C/S282C | 81.1 |

### Other verified:
- **FAST-PETase** (Lu et al., 2022 Nature): Tm=67.8 C, ML-designed (MutCompute)
- **DuraPETase** (Cui et al., 2021 ACS Catal): Tm=77.0 C, rational design

## Active Site

Catalytic triad: Ser160-Asp206-His237 (numbering for I. sakaiensis PETase)

Binding site residues (crystal structure 5XJH):
- Trp156, Trp185, Trp194 (tryptophan clamp for aromatic ring)
- Tyr87, Met161, Phe209, Ile208, Met215

## Paper Attribution Notes

- **DuraPETase**: Designed by Cui et al. 2021 (ACS Catal, DOI: 10.1021/acscatal.0c05126), NOT Austin 2018
- **FAST-PETase**: Designed by Lu et al. 2022 (Nature, DOI: 10.1038/s41586-022-04599-z), NOT Son 2019
- **ThermoPETase**: Designed by Son et al. 2019 (ACS Catal, DOI: 10.1021/acscatal.9b00568)
- **Austin 2018** (PNAS, DOI: 10.1073/pnas.1718804115): Reported W159H/S238F structure, NO Tm values
