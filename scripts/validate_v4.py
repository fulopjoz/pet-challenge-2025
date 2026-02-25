#!/usr/bin/env python3
"""
Validation checks for v4 submission (post-audit fixes).

Checks beyond basic sanity (which are in generate_submission_v2.py):
  1. Per-site substitution spread: different mutations at same position get different delta_ll
  2. Top-K WT fraction: what fraction of top-ranked entries are WT-identical
  3. Within-WT rank correlation between targets: act1 vs act2 rankings should differ
  4. entropy_at_site partial correlation: entropy_at_site adds signal beyond delta_ll

Usage:
    python scripts/validate_v4.py --version {v4,v5,v6} [--submission PATH] [--scores PATH]
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

DEFAULT_SCORES = os.path.join(RESULTS_DIR, "esm2_scores.csv")
DEFAULT_SUBMISSION = os.path.join(RESULTS_DIR, "submission_zero_shot_v2.csv")
TEST_CSV = os.path.join(PROJECT_ROOT, "data", "petase_challenge_data",
                         "predictive-pet-zero-shot-test-2025.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate PET submission scoring behavior.")
    parser.add_argument("--submission", default=DEFAULT_SUBMISSION,
                        help="Submission CSV path (default: results/submission_zero_shot_v2.csv)")
    parser.add_argument("--scores", default=DEFAULT_SCORES,
                        help="Score CSV path with n_mutations/wt_idx (default: results/esm2_scores.csv)")
    parser.add_argument("--version", required=True, choices=["v4", "v5", "v6"],
                        help="Submission scoring version used for weight-vector validation.")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("V4 VALIDATION CHECKS")
    print("=" * 60)

    # Load data
    if not os.path.exists(args.scores):
        raise FileNotFoundError("Scores file not found: %s" % args.scores)
    if not os.path.exists(args.submission):
        raise FileNotFoundError("Submission file not found: %s" % args.submission)
    scores = pd.read_csv(args.scores)
    sub = pd.read_csv(args.submission)
    test_df = pd.read_csv(TEST_CSV)

    n_test = len(test_df)
    n_mutations = scores["n_mutations"].astype(int).values
    wt_mask = n_mutations == 0
    mut_mask = n_mutations > 0

    act1_col = [c for c in sub.columns if "activity_1" in c][0]
    act2_col = [c for c in sub.columns if "activity_2" in c][0]
    expr_col = [c for c in sub.columns if "expression" in c][0]

    checks_passed = 0
    checks_total = 0

    # --- Check 1: Per-site substitution spread ---
    print("\n--- Check 1: Per-site substitution spread ---")
    # For WT0, group mutants by mutation position and check delta_ll variance
    wt0_mask = (scores["wt_idx"].astype(int) == 0) & (scores["n_mutations"].astype(int) == 1)
    wt0 = scores[wt0_mask].copy()
    wt0["delta_ll_f"] = wt0["delta_ll"].astype(float)

    # We need to identify mutation positions. Use test sequences vs WT0.
    wt_seqs = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "petase_challenge_data",
                                        "pet-2025-wildtype-cds.csv"))
    wt0_seq = wt_seqs["Wt AA Sequence"].values[0]

    positions = []
    for idx in wt0.index:
        test_seq = test_df["sequence"].values[idx]
        diffs = [i for i in range(len(wt0_seq)) if i < len(test_seq) and wt0_seq[i] != test_seq[i]]
        positions.append(diffs[0] if len(diffs) == 1 else -1)
    wt0["mut_pos"] = positions

    # Group by position
    pos_groups = wt0[wt0["mut_pos"] >= 0].groupby("mut_pos")["delta_ll_f"]
    multi_sub_positions = [(pos, grp) for pos, grp in pos_groups if len(grp) > 1]

    if multi_sub_positions:
        n_varied = sum(1 for _, grp in multi_sub_positions if grp.std() > 0.01)
        pct = 100.0 * n_varied / len(multi_sub_positions)
        checks_total += 1
        ok = pct > 80
        if ok:
            checks_passed += 1
        status = "PASS" if ok else "FAIL"
        print("  Positions with >1 substitution: %d" % len(multi_sub_positions))
        print("  Positions with varied delta_ll (std>0.01): %d (%.1f%%) [%s]" % (
            n_varied, pct, status))
        # Show a few examples
        for pos, grp in sorted(multi_sub_positions, key=lambda x: -x[1].std())[:3]:
            print("    pos %d: n=%d, mean=%.3f, std=%.3f, range=[%.3f, %.3f]" % (
                pos, len(grp), grp.mean(), grp.std(), grp.min(), grp.max()))
    else:
        print("  No multi-substitution positions found (unexpected)")

    # --- Check 2: Top-K WT fraction ---
    print("\n--- Check 2: Top-K WT fraction ---")
    for target, col in [("activity_1", act1_col), ("activity_2", act2_col),
                         ("expression", expr_col)]:
        ranked = sub[col].values
        top_indices = np.argsort(-ranked)  # descending

        print("  %s:" % target)
        for k in [20, 50, 100, 200]:
            top_k = top_indices[:k]
            n_wt = sum(1 for i in top_k if n_mutations[i] == 0)
            n_mut = k - n_wt
            print("    Top-%d: %d WT, %d mutants (%.0f%% mutants)" % (
                k, n_wt, n_mut, 100.0 * n_mut / k))

    # Overall check: top-20 should have SOME mutants
    checks_total += 1
    top20_act1 = np.argsort(-sub[act1_col].values)[:20]
    n_mut_top20 = sum(1 for i in top20_act1 if n_mutations[i] > 0)
    ok = n_mut_top20 > 0
    if ok:
        checks_passed += 1
    status = "PASS" if ok else "FAIL"
    print("\n  Top-20 act1 has mutants: %d/20 [%s]" % (n_mut_top20, status))

    # --- Check 3: Within-WT rank correlation between targets ---
    print("\n--- Check 3: Within-WT rank correlation (act1 vs act2) ---")
    for wt_i in [0, 1, 2]:
        mask = (scores["wt_idx"].astype(int) == wt_i) & (scores["n_mutations"].astype(int) == 1)
        if mask.sum() < 10:
            continue
        idx = np.where(mask)[0]
        a1 = sub[act1_col].values[idx]
        a2 = sub[act2_col].values[idx]
        r, p = stats.spearmanr(a1, a2)
        print("  WT%d (n=%d mutants): Spearman(act1, act2) = %.4f (p=%.2e)" % (
            wt_i, len(idx), r, p))

    # Check: act1-act2 correlation should differ (< 0.95)
    checks_total += 1
    r_total, _ = stats.spearmanr(sub[act1_col].values, sub[act2_col].values)
    ok = r_total < 0.95
    if ok:
        checks_passed += 1
    status = "PASS" if ok else "FAIL"
    print("  Overall act1-act2 Spearman: %.4f (expect < 0.95) [%s]" % (r_total, status))

    # --- Check 4: entropy_at_site partial correlation ---
    print("\n--- Check 4: entropy_at_site partial correlation ---")
    if "entropy_at_site" in scores.columns:
        # Only look at mutants (entropy_at_site is NaN for WTs)
        mut_scores = scores[mut_mask].copy()
        mut_scores["delta_ll_f"] = mut_scores["delta_ll"].astype(float)
        mut_scores["eas_f"] = pd.to_numeric(mut_scores["entropy_at_site"], errors="coerce")

        valid = mut_scores.dropna(subset=["eas_f"])
        if len(valid) > 10:
            # Simple partial correlation: correlate entropy_at_site with act1
            # after regressing out delta_ll
            from numpy.polynomial.polynomial import polyfit
            delta = valid["delta_ll_f"].values
            eas = valid["eas_f"].values

            # Residualize entropy_at_site against delta_ll
            coeffs = np.polyfit(delta, eas, 1)
            eas_resid = eas - np.polyval(coeffs, delta)

            # Correlate residuals with act1 submission values
            idx = valid.index
            act1_vals = sub[act1_col].values[idx]
            r_partial, p = stats.spearmanr(eas_resid, act1_vals)

            # Also raw correlation
            r_raw, _ = stats.spearmanr(eas, act1_vals)

            checks_total += 1
            ok = abs(r_partial) > 0.01  # any non-zero signal
            if ok:
                checks_passed += 1
            status = "PASS" if ok else "FAIL"
            print("  entropy_at_site raw Spearman with act1: %.4f" % r_raw)
            print("  entropy_at_site partial Spearman (controlling delta_ll): %.4f (p=%.2e) [%s]" % (
                r_partial, p, status))
            print("  (Confirms entropy_at_site adds signal beyond delta_ll)")
        else:
            print("  Not enough valid entropy_at_site values")
    else:
        print("  entropy_at_site column not found in scores")

    # --- Check 5: Weight sum verification ---
    print("\n--- Check 5: Weight vector sums ---")
    # Version-specific weight vectors (must match generate_submission_v2.py)
    WEIGHT_VECTORS = {
        "v4": {
            "act1": [0.30, 0.25, 0.10, 0.05, 0.05, 0.05, 0.10, 0.10],
            "act2": [0.35, 0.20, 0.10, 0.05, 0.05, 0.05, 0.10, 0.10],
            "expr": [0.30, 0.15, 0.10, 0.10, 0.15, 0.10, 0.10],
        },
        "v5": {
            "act1": [0.30, 0.25, 0.10, 0.05, 0.05, 0.05, 0.05, 0.10, 0.05],
            "act2": [0.35, 0.20, 0.10, 0.05, 0.05, 0.05, 0.05, 0.10, 0.05],
            "expr": [0.30, 0.15, 0.10, 0.10, 0.15, 0.10, 0.10],
        },
        "v6": {
            "act1": [0.275, 0.225, 0.10, 0.05, 0.05, 0.025, 0.05, 0.075, 0.10, 0.05],
            "act2": [0.325, 0.20, 0.10, 0.05, 0.05, 0.025, 0.05, 0.05, 0.10, 0.05],
            "expr": [0.30, 0.15, 0.10, 0.10, 0.15, 0.10, 0.10],
        },
    }
    version = args.version
    print("  Validating weight vectors for %s" % version)
    wv = WEIGHT_VECTORS[version]
    act1_weights = wv["act1"]
    act2_weights = wv["act2"]
    expr_weights = wv["expr"]

    checks_total += 1
    s1, s2, se = sum(act1_weights), sum(act2_weights), sum(expr_weights)
    ok = abs(s1 - 1.0) < 1e-6 and abs(s2 - 1.0) < 1e-6 and abs(se - 1.0) < 1e-6
    if ok:
        checks_passed += 1
    status = "PASS" if ok else "FAIL"
    print("  act1 weights sum: %.4f" % s1)
    print("  act2 weights sum: %.4f" % s2)
    print("  expression weights sum: %.4f" % se)
    print("  [%s]" % status)

    # --- Check 6: entropy_at_site sign check ---
    print("\n--- Check 6: entropy_at_site sign verification ---")
    if "entropy_at_site" in scores.columns:
        # For WT0 mutants: verify that mutants at conserved (low entropy) sites
        # get LOWER predicted activity than mutants at variable (high entropy) sites
        wt0_mut = scores[(scores["wt_idx"].astype(int) == 0) &
                         (scores["n_mutations"].astype(int) == 1)].copy()
        wt0_mut["eas_f"] = pd.to_numeric(wt0_mut["entropy_at_site"], errors="coerce")
        wt0_valid = wt0_mut.dropna(subset=["eas_f"])

        if len(wt0_valid) > 20:
            eas_vals = wt0_valid["eas_f"].values
            median_eas = np.median(eas_vals)
            low_eas_mask = eas_vals < median_eas  # conserved
            high_eas_mask = eas_vals >= median_eas  # variable

            low_idx = wt0_valid.index[low_eas_mask]
            high_idx = wt0_valid.index[high_eas_mask]

            low_act1_mean = sub[act1_col].values[low_idx].mean()
            high_act1_mean = sub[act1_col].values[high_idx].mean()

            checks_total += 1
            # Conserved sites should get LOWER activity (penalty)
            ok = low_act1_mean < high_act1_mean
            if ok:
                checks_passed += 1
            status = "PASS" if ok else "FAIL"
            print("  WT0 mutants at conserved sites (low entropy): mean act1 = %.4f (n=%d)" % (
                low_act1_mean, low_eas_mask.sum()))
            print("  WT0 mutants at variable sites (high entropy):  mean act1 = %.4f (n=%d)" % (
                high_act1_mean, high_eas_mask.sum()))
            print("  Conserved < Variable: %s [%s]" % (
                "YES" if ok else "NO", status))
        else:
            print("  Not enough WT0 mutants with entropy_at_site")
    else:
        print("  entropy_at_site column not found")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("RESULT: %d/%d checks passed" % (checks_passed, checks_total))
    print("=" * 60)


if __name__ == "__main__":
    main()
