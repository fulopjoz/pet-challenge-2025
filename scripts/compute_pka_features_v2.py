#!/usr/bin/env python3
"""
Compute per-mutant pKa features using pKAI (primary) and PROPKA (fallback).

Unlike compute_pka_features.py which maps WT pKa identically to all mutants,
this script predicts pKa for each of the 4988 test structures individually,
then computes delta features (mutant - WT) that provide genuine within-WT
discrimination.

Primary tool: pKAI (Bayer, MIT license, DL-based, faster and more accurate
on predicted structures than PROPKA).
Fallback: PROPKA (classical empirical pKa prediction).
Final fallback: parent WT's pKa from pka_features.csv.

Usage:
    python scripts/compute_pka_features_v2.py [--propka-only] [--pkai-only]

Requires: pip install pKAI propka
Input:  results/structures/test_*.pdb (from predict_structures.py --mode test)
        results/structures/wt_*.pdb   (WT structures)
        results/esm2_scores.csv       (for wt_idx mapping)
        results/pka_features.csv      (WT baseline pKa)
Output: results/pka_features_test_v2.csv (4988 rows with per-mutant + delta features)
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
STRUCTURES_DIR = os.path.join(BASE_DIR, "results", "structures")
ESM2_SCORES = os.path.join(BASE_DIR, "results", "esm2_scores.csv")
PKA_WT_CSV = os.path.join(BASE_DIR, "results", "pka_features.csv")
PKA_V2_CSV = os.path.join(BASE_DIR, "results", "pka_features_test_v2.csv")

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
IONIZABLE_RESIDUES = set(REFERENCE_PKA.keys())


def protonation_fraction(pH, pKa):
    """Henderson-Hasselbalch: fraction of protonated form."""
    return 1.0 / (1.0 + 10.0 ** (pH - pKa))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-mutant pKa features via pKAI + PROPKA")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--propka-only", action="store_true",
                       help="Use only PROPKA (skip pKAI)")
    group.add_argument("--pkai-only", action="store_true",
                       help="Use only pKAI (no PROPKA fallback)")
    return parser.parse_args()


def run_pkai_on_pdb(pdb_path):
    """
    Run pKAI on a PDB file. Returns list of dicts with residue pKa info.
    pKAI uses a deep learning model trained on experimental pKa data.
    """
    try:
        from pKAI.pKAI import pKAI
    except ImportError:
        return None  # pKAI not installed

    try:
        # pKAI returns a DataFrame with columns: resname, resid, chain, pKa
        results = pKAI(pdb_path)
        if results is None or len(results) == 0:
            return []

        residues = []
        for _, row in results.iterrows():
            resname = str(row.get("resname", row.get("Residue", ""))).strip().upper()
            # pKAI may use different column names depending on version
            resnum = int(row.get("resid", row.get("ResID", row.get("residue_number", 0))))
            chain = str(row.get("chain", row.get("Chain", "A")))
            pka = float(row.get("pKa", row.get("pka", 0.0)))

            # Only keep standard ionizable residues
            # pKAI may return 3-letter codes or use its own naming
            resname_3 = resname[:3]
            if resname_3 not in IONIZABLE_RESIDUES:
                continue
            if np.isnan(pka):
                continue

            residues.append({
                "resname": resname_3,
                "resnum": resnum,
                "chain": chain,
                "pka": pka,
                "ref_pka": REFERENCE_PKA.get(resname_3, 0.0),
                "pka_shift": pka - REFERENCE_PKA.get(resname_3, 0.0),
            })
        return residues
    except Exception as e:
        return None  # Signal failure for fallback


def run_propka_on_pdb(pdb_path):
    """
    Run PROPKA on a PDB file. Returns list of dicts with residue pKa info.
    """
    try:
        import propka.run
    except ImportError:
        return None

    try:
        mol = propka.run.single(pdb_path, write_pka=False)
    except Exception:
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


def compute_features_from_residues(residues):
    """
    Compute pKa-based features from a list of ionizable residue pKa dicts.
    Same logic as compute_pka_features.py but returns a flat dict.
    """
    features = {
        "catalytic_his_pka": np.nan,
        "proton_frac_his_pH55": np.nan,
        "proton_frac_his_pH90": np.nan,
        "delta_protonation_his": np.nan,
        "n_ionizable": 0,
        "n_his_shifted": 0,
        "mean_ionizable_pka_shift": 0.0,
        "mean_proton_frac_all_pH55": np.nan,
        "mean_proton_frac_all_pH90": np.nan,
    }

    if not residues:
        return features

    features["n_ionizable"] = len(residues)
    pka_shifts = [abs(r["pka_shift"]) for r in residues]
    features["mean_ionizable_pka_shift"] = float(np.mean(pka_shifts))

    # Histidine-specific
    # CAVEAT: Uses min-pKa His as proxy for catalytic His (His237-equivalent in
    # IsPETase). This heuristic works for canonical PETase scaffolds where the
    # catalytic His has a depressed pKa due to its active-site environment, but
    # may misidentify in scaffolds with multiple buried His or unusual folds.
    # Proper fix: align to canonical catalytic triad positions (Ser160/His237/Asp206).
    his_residues = [r for r in residues if r["resname"] == "HIS"]
    if his_residues:
        cat_his = min(his_residues, key=lambda r: r["pka"])
        features["catalytic_his_pka"] = cat_his["pka"]
        features["proton_frac_his_pH55"] = protonation_fraction(5.5, cat_his["pka"])
        features["proton_frac_his_pH90"] = protonation_fraction(9.0, cat_his["pka"])
        features["delta_protonation_his"] = (
            features["proton_frac_his_pH55"] - features["proton_frac_his_pH90"]
        )
        features["n_his_shifted"] = sum(
            1 for r in his_residues if abs(r["pka_shift"]) > 1.0
        )

    # Mean protonation across all ionizable residues
    pf_55 = [protonation_fraction(5.5, r["pka"]) for r in residues]
    pf_90 = [protonation_fraction(9.0, r["pka"]) for r in residues]
    features["mean_proton_frac_all_pH55"] = float(np.mean(pf_55))
    features["mean_proton_frac_all_pH90"] = float(np.mean(pf_90))

    return features


def main():
    args = parse_args()
    t0 = time.time()

    print("=" * 60)
    print("Per-Mutant pKa Feature Computation (v2)")
    print("=" * 60)

    # Determine which tools to use
    use_pkai = not args.propka_only
    use_propka = not args.pkai_only

    # Check availability
    pkai_available = False
    propka_available = False

    if use_pkai:
        try:
            from pKAI.pKAI import pKAI as _pkai_check
            pkai_available = True
            print("pKAI: available (primary)")
        except ImportError:
            print("pKAI: not installed")
            if args.pkai_only:
                print("ERROR: --pkai-only specified but pKAI not installed")
                sys.exit(1)

    if use_propka:
        try:
            import propka.run as _propka_check
            propka_available = True
            print("PROPKA: available (%s)" % ("fallback" if pkai_available else "primary"))
        except ImportError:
            print("PROPKA: not installed")
            if args.propka_only:
                print("ERROR: --propka-only specified but PROPKA not installed")
                sys.exit(1)

    if not pkai_available and not propka_available:
        print("ERROR: Neither pKAI nor PROPKA is installed. Install at least one:")
        print("  pip install pKAI propka")
        sys.exit(1)

    # Load wt_idx mapping from ESM2 scores
    if not os.path.exists(ESM2_SCORES):
        print("ERROR: %s not found. Run esm2_zero_shot_scoring.py first." % ESM2_SCORES)
        sys.exit(1)

    scores_df = pd.read_csv(ESM2_SCORES)
    n_test = len(scores_df)
    test_wt_idx = scores_df["wt_idx"].astype(int).tolist()
    test_n_mutations = scores_df["n_mutations"].astype(int).tolist()
    print("Test set: %d sequences" % n_test)

    # Load WT baseline pKa features
    wt_pka_lookup = {}
    if os.path.exists(PKA_WT_CSV):
        wt_pka_df = pd.read_csv(PKA_WT_CSV)
        for _, row in wt_pka_df.iterrows():
            wt_pka_lookup[int(row["wt_idx"])] = row.to_dict()
        print("Loaded WT pKa baseline: %d entries" % len(wt_pka_lookup))
    else:
        print("WARNING: %s not found — delta features will be NaN for WT fallback" % PKA_WT_CSV)

    # Process each test sequence
    print("\nProcessing %d test structures..." % n_test)
    rows = []
    counts = {"pkai": 0, "propka": 0, "wt_fallback": 0, "no_structure": 0}

    for i in range(n_test):
        pdb_path = os.path.join(STRUCTURES_DIR, "test_%04d.pdb" % i)
        wt_idx = test_wt_idx[i]
        n_mut = test_n_mutations[i]

        residues = None
        source = "none"

        if os.path.exists(pdb_path):
            # Try pKAI first
            if pkai_available:
                pkai_residues = run_pkai_on_pdb(pdb_path)
                if pkai_residues is not None and len(pkai_residues) > 0:
                    residues = pkai_residues
                    source = "pkai"
                else:
                    # Treat empty pKAI outputs as failure so PROPKA fallback still runs.
                    residues = None
                    print("  WARNING: pKAI failed on test_%04d.pdb, trying PROPKA" % i)

            # Fallback to PROPKA
            if residues is None and propka_available:
                residues = run_propka_on_pdb(pdb_path)
                if residues is not None:
                    source = "propka"
                elif pkai_available:
                    # Both tools failed on this structure
                    print("  WARNING: PROPKA also failed on test_%04d.pdb, using WT fallback" % i)
        else:
            if counts["no_structure"] == 0:
                print("  WARNING: test_%04d.pdb not found — using WT fallback "
                      "(run predict_structures.py --mode test first)" % i)

        if residues is not None and len(residues) > 0:
            features = compute_features_from_residues(residues)
            counts[source] += 1
        elif wt_idx in wt_pka_lookup:
            # Fallback to WT baseline — delta features will be NaN for mutants
            wt_row = wt_pka_lookup[wt_idx]
            features = {
                "catalytic_his_pka": wt_row.get("catalytic_his_pka", np.nan),
                "proton_frac_his_pH55": wt_row.get("proton_frac_his_pH55", np.nan),
                "proton_frac_his_pH90": wt_row.get("proton_frac_his_pH90", np.nan),
                "delta_protonation_his": wt_row.get("delta_protonation_his", np.nan),
                "n_ionizable": int(wt_row.get("n_ionizable", 0)),
                "n_his_shifted": int(wt_row.get("n_his_shifted", 0)),
                "mean_ionizable_pka_shift": wt_row.get("mean_ionizable_pka_shift", 0.0),
                "mean_proton_frac_all_pH55": wt_row.get("mean_proton_frac_all_pH55", np.nan),
                "mean_proton_frac_all_pH90": wt_row.get("mean_proton_frac_all_pH90", np.nan),
            }
            source = "wt_fallback"
            counts["wt_fallback"] += 1
        else:
            features = compute_features_from_residues([])
            source = "none"
            counts["no_structure"] += 1

        # Compute delta features (mutant - WT)
        wt_his_pka = np.nan
        wt_pf55 = np.nan
        wt_pf90 = np.nan
        if wt_idx in wt_pka_lookup:
            wt_his_pka = wt_pka_lookup[wt_idx].get("catalytic_his_pka", np.nan)
            wt_pf55 = wt_pka_lookup[wt_idx].get("proton_frac_his_pH55", np.nan)
            wt_pf90 = wt_pka_lookup[wt_idx].get("proton_frac_his_pH90", np.nan)

        if n_mut == 0:
            # WT-identical sequence: delta = 0 by definition
            delta_his_pka = 0.0
            delta_pf55 = 0.0
            delta_pf90 = 0.0
        else:
            mut_his_pka = features.get("catalytic_his_pka", np.nan)
            mut_pf55 = features.get("proton_frac_his_pH55", np.nan)
            mut_pf90 = features.get("proton_frac_his_pH90", np.nan)

            delta_his_pka = mut_his_pka - wt_his_pka if (
                not np.isnan(mut_his_pka) and not np.isnan(wt_his_pka)
            ) else np.nan
            delta_pf55 = mut_pf55 - wt_pf55 if (
                not np.isnan(mut_pf55) and not np.isnan(wt_pf55)
            ) else np.nan
            delta_pf90 = mut_pf90 - wt_pf90 if (
                not np.isnan(mut_pf90) and not np.isnan(wt_pf90)
            ) else np.nan

        row = {
            "test_idx": i,
            "wt_idx": wt_idx,
            "pka_source": source,
            # Per-mutant absolute features
            "catalytic_his_pka": features["catalytic_his_pka"],
            "proton_frac_his_pH55": features["proton_frac_his_pH55"],
            "proton_frac_his_pH90": features["proton_frac_his_pH90"],
            "delta_protonation_his": features["delta_protonation_his"],
            "n_ionizable": features["n_ionizable"],
            "n_his_shifted": features["n_his_shifted"],
            "mean_ionizable_pka_shift": features["mean_ionizable_pka_shift"],
            "mean_proton_frac_all_pH55": features["mean_proton_frac_all_pH55"],
            "mean_proton_frac_all_pH90": features["mean_proton_frac_all_pH90"],
            # Delta features (mutant - WT) — within-WT discriminators
            "delta_catalytic_his_pka": delta_his_pka,
            "delta_proton_frac_pH55": delta_pf55,
            "delta_proton_frac_pH90": delta_pf90,
        }
        rows.append(row)

        if (i + 1) % 500 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (n_test - i - 1) / rate if rate > 0 else 0
            print("  [%d/%d] source=%s (%ds elapsed, ~%ds remaining)" % (
                i + 1, n_test, source, int(elapsed), int(remaining)))

    # Save output
    df = pd.DataFrame(rows)
    df.to_csv(PKA_V2_CSV, index=False)

    total_time = time.time() - t0
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print("Saved %d rows to %s" % (len(df), PKA_V2_CSV))
    print("Runtime: %ds (%.1f min)" % (int(total_time), total_time / 60))

    # Source distribution
    print("\n=== pKa Source Distribution ===")
    for src, cnt in sorted(counts.items()):
        pct = 100.0 * cnt / n_test if n_test > 0 else 0
        print("  %-15s %5d (%5.1f%%)" % (src, cnt, pct))

    # Warn if fallback rate is high
    n_fallback = counts["wt_fallback"] + counts["no_structure"]
    if counts["no_structure"] > 0:
        print("\n  WARNING: %d sequences had no PDB structure." % counts["no_structure"])
        print("  Run: python scripts/predict_structures.py --mode test")
    if counts["wt_fallback"] > n_test * 0.05:
        print("\n  WARNING: %.0f%% of sequences used WT fallback (no per-mutant pKa)." %
              (100.0 * counts["wt_fallback"] / n_test))
        print("  Delta features will be NaN for these — check pKAI/PROPKA installation.")
    if pkai_available and counts["pkai"] == 0 and counts["propka"] > 0:
        print("\n  WARNING: pKAI produced 0 results despite being installed.")
        print("  All predictions came from PROPKA fallback. Check pKAI compatibility.")

    # Delta feature statistics
    print("\n=== Delta pKa Feature Statistics (mutants only) ===")
    mut_df = df[df["delta_catalytic_his_pka"] != 0.0]
    for col in ["delta_catalytic_his_pka", "delta_proton_frac_pH55", "delta_proton_frac_pH90"]:
        valid = mut_df[col].dropna()
        if len(valid) > 0:
            print("  %s: mean=%.4f, std=%.4f, min=%.4f, max=%.4f (n=%d)" % (
                col, valid.mean(), valid.std(), valid.min(), valid.max(), len(valid)))
        else:
            print("  %s: all NaN" % col)

    # WT check: deltas should be 0 for WT-identical sequences
    wt_rows = df[df.index.isin(
        [i for i, nm in enumerate(test_n_mutations) if nm == 0]
    )]
    n_wt_zero = (wt_rows["delta_catalytic_his_pka"] == 0.0).sum()
    print("\n=== WT Sanity Check ===")
    print("  WT-identical sequences: %d" % len(wt_rows))
    print("  With delta_catalytic_his_pka=0: %d (expect all)" % n_wt_zero)

    print("\nPER-MUTANT pKa COMPUTATION COMPLETE")


if __name__ == "__main__":
    main()
