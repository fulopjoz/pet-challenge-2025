#!/usr/bin/env python3
"""
Compute pKa features from PROPKA analysis of predicted PETase structures.

Uses PROPKA to estimate per-residue pKa values for ionizable residues,
then computes Henderson-Hasselbalch protonation fractions at pH 5.5 and 9.0.

Key scientific basis (Charlier 2024, Biophys J):
  - Catalytic His pKa ~4.9 (NMR titration of LCC His242)
  - Conserved alpha/beta-hydrolase serine protease mechanism (Ser-His-Asp triad)
  - pH 5.5: His ~80% deprotonated (suboptimal)
  - pH 9.0: His >99.9% deprotonated (near-optimal)

Maps per-WT pKa features to all 4988 test sequences via wt_idx.

Usage:
    python scripts/compute_pka_features.py

Requires: pip install propka
Input:  results/structures/*.pdb (from predict_structures.py)
        results/esm2_scores.csv (for wt_idx mapping)
Output: results/pka_features.csv (313 WT rows)
        results/pka_features_test.csv (4988 test rows)
"""

import os
import sys
import glob
import csv
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
STRUCTURES_DIR = os.path.join(BASE_DIR, "results", "structures")
ESM2_SCORES = os.path.join(BASE_DIR, "results", "esm2_scores.csv")
TEST_CSV = os.path.join(BASE_DIR, "data", "petase_challenge_data",
                        "predictive-pet-zero-shot-test-2025.csv")
WT_CSV = os.path.join(BASE_DIR, "data", "petase_challenge_data",
                      "pet-2025-wildtype-cds.csv")
PKA_WT_CSV = os.path.join(BASE_DIR, "results", "pka_features.csv")
PKA_TEST_CSV = os.path.join(BASE_DIR, "results", "pka_features_test.csv")

# Reference pKa values for ionizable amino acids (in isolation)
REFERENCE_PKA = {
    "ASP": 3.80,
    "GLU": 4.50,
    "HIS": 6.50,
    "CYS": 9.00,
    "TYR": 10.50,
    "LYS": 10.50,
    "ARG": 12.50,
}

# Ionizable residue 3-letter codes
IONIZABLE_RESIDUES = set(REFERENCE_PKA.keys())


def protonation_fraction(pH, pKa):
    """Henderson-Hasselbalch protonation fraction for a single ionizable group.

    For acids (Asp, Glu, Cys, Tyr): protonated = neutral, deprotonated = charged (-)
    For bases (His, Lys, Arg): protonated = charged (+), deprotonated = neutral

    Returns fraction of protonated form.
    """
    return 1.0 / (1.0 + 10.0 ** (pH - pKa))


def run_propka_on_pdb(pdb_path):
    """
    Run PROPKA on a single PDB file.
    Returns list of dicts with residue pKa info.
    """
    try:
        import propka.run
        mol = propka.run.single(pdb_path, write_pka=False)
    except Exception as e:
        print("  PROPKA failed on %s: %s" % (pdb_path, e))
        return []

    residues = []
    for group in mol.conformations["AVR"].groups:
        resname = group.residue_type.strip().upper()
        if resname not in IONIZABLE_RESIDUES:
            continue

        pka = group.pka_value
        if pka is None or np.isnan(pka):
            continue

        residues.append({
            "resname": resname,
            "resnum": group.atom.res_num,
            "chain": group.atom.chain_id,
            "pka": float(pka),
            "ref_pka": REFERENCE_PKA.get(resname, 0.0),
            "pka_shift": float(pka) - REFERENCE_PKA.get(resname, 0.0),
        })

    return residues


def compute_wt_pka_features(residues):
    """
    Compute pKa-based features from PROPKA residue list for one WT structure.
    """
    features = {
        "n_ionizable": len(residues),
        "catalytic_his_pka": np.nan,
        "n_his_shifted": 0,
        "mean_ionizable_pka_shift": 0.0,
        "proton_frac_his_pH55": np.nan,
        "proton_frac_his_pH90": np.nan,
        "delta_protonation_his": np.nan,
        "mean_proton_frac_all_pH55": np.nan,
        "mean_proton_frac_all_pH90": np.nan,
    }

    if not residues:
        return features

    # All ionizable residues
    pka_shifts = [abs(r["pka_shift"]) for r in residues]
    features["mean_ionizable_pka_shift"] = float(np.mean(pka_shifts))

    # Histidine-specific features
    his_residues = [r for r in residues if r["resname"] == "HIS"]
    if his_residues:
        # Find the catalytic His: the one with lowest pKa (most shifted from reference 6.5)
        # In PETases, the catalytic His typically has pKa ~4.9 (Charlier 2024)
        cat_his = min(his_residues, key=lambda r: r["pka"])
        features["catalytic_his_pka"] = cat_his["pka"]
        features["proton_frac_his_pH55"] = protonation_fraction(5.5, cat_his["pka"])
        features["proton_frac_his_pH90"] = protonation_fraction(9.0, cat_his["pka"])
        features["delta_protonation_his"] = (
            features["proton_frac_his_pH55"] - features["proton_frac_his_pH90"]
        )

        # Count His with significant pKa shifts (>1 unit from reference 6.50)
        features["n_his_shifted"] = sum(
            1 for r in his_residues if abs(r["pka_shift"]) > 1.0
        )

    # Mean protonation fraction across all ionizable residues at both pHs
    pf_55 = [protonation_fraction(5.5, r["pka"]) for r in residues]
    pf_90 = [protonation_fraction(9.0, r["pka"]) for r in residues]
    features["mean_proton_frac_all_pH55"] = float(np.mean(pf_55))
    features["mean_proton_frac_all_pH90"] = float(np.mean(pf_90))

    return features


def main():
    print("=" * 60)
    print("PROPKA pKa Feature Computation")
    print("=" * 60)

    # Check structures exist
    pdb_files = sorted(glob.glob(os.path.join(STRUCTURES_DIR, "wt_*.pdb")))
    if not pdb_files:
        print("ERROR: No PDB files found in %s" % STRUCTURES_DIR)
        print("Run predict_structures.py first.")
        sys.exit(1)

    print("Found %d PDB structures" % len(pdb_files))

    # Process each WT structure
    wt_features = []
    for i, pdb_path in enumerate(pdb_files):
        basename = os.path.basename(pdb_path)
        # Extract wt_idx from filename (wt_000.pdb -> 0)
        wt_idx = int(basename.replace("wt_", "").replace(".pdb", ""))

        residues = run_propka_on_pdb(pdb_path)
        features = compute_wt_pka_features(residues)
        features["wt_idx"] = wt_idx
        features["pdb_file"] = basename
        wt_features.append(features)

        if (i + 1) % 50 == 0 or i == 0:
            his_pka = features["catalytic_his_pka"]
            his_str = "%.2f" % his_pka if not np.isnan(his_pka) else "N/A"
            print("  [%d/%d] %s: %d ionizable, catalytic His pKa=%s" % (
                i + 1, len(pdb_files), basename, features["n_ionizable"], his_str))

    # Save WT pKa features
    wt_pka_df = pd.DataFrame(wt_features)
    wt_pka_df.to_csv(PKA_WT_CSV, index=False)
    print("\nSaved %d WT pKa features to %s" % (len(wt_pka_df), PKA_WT_CSV))

    # Summary
    valid_his = wt_pka_df["catalytic_his_pka"].dropna()
    if len(valid_his) > 0:
        print("\n=== Catalytic His pKa Summary ===")
        print("  Structures with His: %d / %d" % (len(valid_his), len(wt_pka_df)))
        print("  pKa: mean=%.2f, std=%.2f, min=%.2f, max=%.2f" % (
            valid_his.mean(), valid_his.std(), valid_his.min(), valid_his.max()))
        print("  Expected ~4.9 for IsPETase-like catalytic His (Charlier 2024)")

    # Map to test sequences
    print("\n--- Mapping pKa features to test sequences ---")
    if not os.path.exists(ESM2_SCORES):
        print("WARNING: %s not found, using test CSV for wt_idx mapping" % ESM2_SCORES)
        # Build mapping from test data directly
        test_df = pd.read_csv(TEST_CSV)
        wt_df = pd.read_csv(WT_CSV)
        wt_seqs = list(wt_df["Wt AA Sequence"].values)
        from collections import defaultdict
        wt_by_len = defaultdict(list)
        for wi, seq in enumerate(wt_seqs):
            wt_by_len[len(seq)].append((wi, seq))

        test_wt_idx = []
        for test_seq in test_df["sequence"].values:
            tlen = len(test_seq)
            best_wt, best_diff = None, 999
            for wi, wseq in wt_by_len.get(tlen, []):
                ndiff = sum(1 for a, b in zip(wseq, test_seq) if a != b)
                if ndiff < best_diff:
                    best_diff = ndiff
                    best_wt = wi
                if ndiff == 0:
                    break
            test_wt_idx.append(best_wt)
    else:
        scores_df = pd.read_csv(ESM2_SCORES)
        test_wt_idx = scores_df["wt_idx"].astype(int).tolist()

    # Build wt_idx -> pKa features lookup
    pka_lookup = {}
    for _, row in wt_pka_df.iterrows():
        pka_lookup[int(row["wt_idx"])] = row

    # Map features for each test sequence
    feature_cols = [c for c in wt_pka_df.columns if c not in ("wt_idx", "pdb_file")]
    test_rows = []
    for ti, wi in enumerate(test_wt_idx):
        row = {"test_idx": ti, "wt_idx": wi}
        if wi is not None and wi in pka_lookup:
            wt_row = pka_lookup[wi]
            for col in feature_cols:
                row[col] = wt_row[col]
        else:
            for col in feature_cols:
                row[col] = np.nan
        test_rows.append(row)

    test_pka_df = pd.DataFrame(test_rows)
    test_pka_df.to_csv(PKA_TEST_CSV, index=False)
    print("Saved %d test pKa features to %s" % (len(test_pka_df), PKA_TEST_CSV))

    # Verify
    n_with_his = test_pka_df["catalytic_his_pka"].notna().sum()
    print("  %d / %d test sequences have catalytic His pKa" % (n_with_his, len(test_pka_df)))

    print("\nPROPKA FEATURE COMPUTATION COMPLETE")


if __name__ == "__main__":
    main()
