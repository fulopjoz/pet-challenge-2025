#!/usr/bin/env python3
"""
Compute CDS-Based and Mutation-Level Features for PET Challenge 2025

This script produces two feature files:
  1. cds_features.csv — per-WT scaffold features (GC, rare codons, 5' energy proxy)
  2. mutation_features.csv — per-test-sequence features (AA property changes, position)

Rationale (from literature):
  - Expression in E. coli is dominated by 5' mRNA secondary structure, NOT full-gene CAI
    (Kudla et al. 2009; Cambray et al. 2018; Pearson r=0.762 for codons 2-8 alone)
  - Rare codons slow translation and trigger ribosome stalling
  - AA property changes (hydrophobicity, charge, size) affect folding and solubility
  - Charge changes are relevant for pH-dependent activity differences

Usage:
    python scripts/compute_cds_features.py
"""

import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "petase_challenge_data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

TEST_CSV = os.path.join(DATA_DIR, "predictive-pet-zero-shot-test-2025.csv")
CDS_CSV = os.path.join(DATA_DIR, "pet-2025-wildtype-cds.csv")
ESM2_SCORES = os.path.join(RESULTS_DIR, "esm2_scores.csv")

CDS_FEATURES_OUT = os.path.join(RESULTS_DIR, "cds_features.csv")
MUTATION_FEATURES_OUT = os.path.join(RESULTS_DIR, "mutation_features.csv")

# E. coli rare codons (low-abundance tRNAs)
RARE_CODONS_ECOLI = {"AGG", "AGA", "ATA", "CTA", "CGA", "GGA", "CGG", "CCC"}

# Kyte-Doolittle hydrophobicity scale
KD_HYDRO = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Amino acid molecular weights (Da)
AA_MW = {
    "A": 89, "R": 174, "N": 132, "D": 133, "C": 121, "Q": 146, "E": 147,
    "G": 75, "H": 155, "I": 131, "L": 131, "K": 146, "M": 149, "F": 165,
    "P": 115, "S": 105, "T": 119, "W": 204, "Y": 181, "V": 117,
}

# Charge at neutral pH
AA_CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}


def compute_cds_features(cds_seq):
    """Compute expression-relevant features from a coding DNA sequence."""
    n = len(cds_seq)
    codons = [cds_seq[i : i + 3] for i in range(0, n - 2, 3)]
    n_codons = len(codons)

    # Total GC content
    gc = (cds_seq.count("G") + cds_seq.count("C")) / n

    # 5' region GC (codons 2-8 = nucleotides 3-23, ~21 nt)
    # This is the strongest single predictor of E. coli expression
    region_5p = cds_seq[3:24]  # skip start codon, take codons 2-8
    gc_5prime = (
        (region_5p.count("G") + region_5p.count("C")) / len(region_5p)
        if len(region_5p) > 0
        else gc
    )

    # Broader 5' region (first 50 nt) — used in ColiFormer for delta-MFE
    region_50 = cds_seq[:50]
    gc_50nt = (
        (region_50.count("G") + region_50.count("C")) / len(region_50)
        if len(region_50) > 0
        else gc
    )

    # Rare codon fraction
    n_rare = sum(1 for c in codons if c.upper() in RARE_CODONS_ECOLI)
    rare_frac = n_rare / n_codons if n_codons > 0 else 0

    # AT-richness near 5' (inversely correlated with mRNA secondary structure)
    at_5prime = 1.0 - gc_5prime

    return {
        "gc_content": gc,
        "gc_5prime": gc_5prime,
        "gc_50nt": gc_50nt,
        "at_5prime": at_5prime,
        "rare_codon_frac": rare_frac,
        "cds_length": n,
        "n_codons": n_codons,
    }


def map_test_to_wt(test_seqs, wt_seqs):
    """Map each test sequence to its closest parent WT by Hamming distance."""
    wt_by_len = {}
    for i, seq in enumerate(wt_seqs):
        wt_by_len.setdefault(len(seq), []).append((i, seq))

    parent_indices = []
    n_mutations_list = []

    for tseq in test_seqs:
        tlen = len(tseq)
        candidates = wt_by_len.get(tlen, [])
        best_dist, best_wt = 999, -1
        for wt_i, wt_s in candidates:
            dist = sum(1 for a, b in zip(tseq, wt_s) if a != b)
            if dist < best_dist:
                best_dist = dist
                best_wt = wt_i
        parent_indices.append(best_wt)
        n_mutations_list.append(best_dist)

    return np.array(parent_indices), np.array(n_mutations_list)


def compute_mutation_features(test_seq, wt_seq):
    """Compute AA property changes for a single-point mutant."""
    if len(test_seq) != len(wt_seq):
        return _default_mutation_features()

    diffs = [(j, wt_seq[j], test_seq[j]) for j in range(len(test_seq)) if test_seq[j] != wt_seq[j]]

    if len(diffs) == 0:
        return {
            "is_wt": 1,
            "mut_pos": -1,
            "pos_relative": 0.5,
            "delta_hydro": 0.0,
            "delta_mw": 0.0,
            "delta_charge": 0.0,
            "abs_delta_hydro": 0.0,
            "abs_delta_charge": 0.0,
            "wt_hydro_at_site": 0.0,
            "mut_hydro_at_site": 0.0,
            "charge_sign_change": 0,
        }

    if len(diffs) == 1:
        pos, wt_aa, mut_aa = diffs[0]
        dh = KD_HYDRO.get(mut_aa, 0) - KD_HYDRO.get(wt_aa, 0)
        dm = AA_MW.get(mut_aa, 130) - AA_MW.get(wt_aa, 130)
        wt_charge = AA_CHARGE.get(wt_aa, 0)
        mut_charge = AA_CHARGE.get(mut_aa, 0)
        dc = mut_charge - wt_charge

        # Charge sign change: important for pH sensitivity
        sign_change = 0
        if wt_charge * mut_charge < 0:
            sign_change = 1  # opposite charges
        elif (wt_charge == 0 and mut_charge != 0) or (wt_charge != 0 and mut_charge == 0):
            sign_change = 1

        return {
            "is_wt": 0,
            "mut_pos": pos,
            "pos_relative": pos / len(test_seq),
            "delta_hydro": dh,
            "delta_mw": dm,
            "delta_charge": dc,
            "abs_delta_hydro": abs(dh),
            "abs_delta_charge": abs(dc),
            "wt_hydro_at_site": KD_HYDRO.get(wt_aa, 0),
            "mut_hydro_at_site": KD_HYDRO.get(mut_aa, 0),
            "charge_sign_change": sign_change,
        }

    return _default_mutation_features()


def _default_mutation_features():
    return {
        "is_wt": 0,
        "mut_pos": -1,
        "pos_relative": 0.5,
        "delta_hydro": 0.0,
        "delta_mw": 0.0,
        "delta_charge": 0.0,
        "abs_delta_hydro": 0.0,
        "abs_delta_charge": 0.0,
        "wt_hydro_at_site": 0.0,
        "mut_hydro_at_site": 0.0,
        "charge_sign_change": 0,
    }


def main():
    # Load data
    test_df = pd.read_csv(TEST_CSV)
    cds_df = pd.read_csv(CDS_CSV)
    n_test = len(test_df)
    n_wt = len(cds_df)
    print("Test sequences: %d" % n_test)
    print("WT sequences with CDS: %d" % n_wt)

    wt_seqs = list(cds_df["Wt AA Sequence"].values)
    wt_cds = list(cds_df["CDS"].values)
    test_seqs = list(test_df["sequence"].values)

    # Use wt_idx from ESM2 scores if available, otherwise compute mapping
    if os.path.exists(ESM2_SCORES):
        esm2 = pd.read_csv(ESM2_SCORES)
        if "wt_idx" in esm2.columns and len(esm2) == n_test:
            parent_idx = esm2["wt_idx"].astype(int).values
            n_mutations = esm2["n_mutations"].astype(int).values
            print("Using wt_idx from ESM2 scores")
        else:
            parent_idx, n_mutations = map_test_to_wt(test_seqs, wt_seqs)
            print("Computed WT mapping from sequences")
    else:
        parent_idx, n_mutations = map_test_to_wt(test_seqs, wt_seqs)
        print("Computed WT mapping from sequences")

    # --- CDS features per WT ---
    print("\nComputing CDS features for %d WT scaffolds..." % n_wt)
    cds_feat_list = []
    for i in range(n_wt):
        feats = compute_cds_features(wt_cds[i])
        feats["wt_idx"] = i
        feats["wt_aa_len"] = len(wt_seqs[i])
        cds_feat_list.append(feats)

    cds_features = pd.DataFrame(cds_feat_list)

    # Z-score normalize CDS features across WTs
    for col in ["gc_content", "gc_5prime", "gc_50nt", "at_5prime", "rare_codon_frac", "cds_length"]:
        vals = cds_features[col].values
        std = vals.std()
        if std > 1e-10:
            cds_features[col + "_z"] = (vals - vals.mean()) / std
        else:
            cds_features[col + "_z"] = 0.0

    cds_features.to_csv(CDS_FEATURES_OUT, index=False)
    print("CDS features saved to %s" % CDS_FEATURES_OUT)

    # Summary stats
    print("\n=== CDS Feature Statistics ===")
    for col in ["gc_content", "gc_5prime", "rare_codon_frac"]:
        v = cds_features[col]
        print("  %s: mean=%.4f, std=%.4f, range=[%.4f, %.4f]" % (
            col, v.mean(), v.std(), v.min(), v.max()))

    # --- Mutation features per test sequence ---
    print("\nComputing mutation features for %d test sequences..." % n_test)
    mut_feat_list = []
    for i in range(n_test):
        wt_i = parent_idx[i]
        feats = compute_mutation_features(test_seqs[i], wt_seqs[wt_i])
        feats["test_idx"] = i
        feats["wt_idx"] = int(wt_i)
        feats["n_mutations"] = int(n_mutations[i])

        # Propagate CDS features to this test sequence
        cds_row = cds_features[cds_features["wt_idx"] == wt_i].iloc[0]
        feats["cds_gc_5prime_z"] = cds_row["gc_5prime_z"]
        feats["cds_rare_codon_z"] = cds_row["rare_codon_frac_z"]
        feats["cds_gc_50nt_z"] = cds_row["gc_50nt_z"]
        feats["cds_at_5prime_z"] = cds_row["at_5prime_z"]

        mut_feat_list.append(feats)

    mutation_features = pd.DataFrame(mut_feat_list)
    mutation_features.to_csv(MUTATION_FEATURES_OUT, index=False)
    print("Mutation features saved to %s" % MUTATION_FEATURES_OUT)

    # Summary
    print("\n=== Mutation Feature Summary ===")
    wt_count = (mutation_features["is_wt"] == 1).sum()
    mut_count = (mutation_features["is_wt"] == 0).sum()
    print("  WT sequences: %d" % wt_count)
    print("  Mutant sequences: %d" % mut_count)

    muts = mutation_features[mutation_features["is_wt"] == 0]
    for col in ["delta_hydro", "delta_charge", "delta_mw", "pos_relative"]:
        v = muts[col]
        print("  %s: mean=%.3f, std=%.3f" % (col, v.mean(), v.std()))

    # Charge change distribution (important for pH scoring)
    charge_changes = muts["delta_charge"]
    print("\n=== Charge Change Distribution ===")
    print("  Positive (adding charge): %d (%.1f%%)" % (
        (charge_changes > 0).sum(), 100 * (charge_changes > 0).mean()))
    print("  Negative (removing charge): %d (%.1f%%)" % (
        (charge_changes < 0).sum(), 100 * (charge_changes < 0).mean()))
    print("  Neutral: %d (%.1f%%)" % (
        (charge_changes == 0).sum(), 100 * (charge_changes == 0).mean()))

    # Top 3 WT scaffolds
    print("\n=== Top 3 WT Scaffolds ===")
    from collections import Counter
    wt_counts = Counter(parent_idx)
    for wt_i, count in wt_counts.most_common(3):
        cds_row = cds_features[cds_features["wt_idx"] == wt_i].iloc[0]
        print("  WT#%d: %d seqs, GC_5p=%.3f, rare=%.3f" % (
            wt_i, count, cds_row["gc_5prime"], cds_row["rare_codon_frac"]))


if __name__ == "__main__":
    main()
