#!/usr/bin/env python3
"""
Validate and Compare All Scoring Approaches for PET Challenge 2025

Compares ESM2, ESMC, and ML baseline models using:
1. Internal consistency checks (WT > mutant scores, score distributions)
2. Cross-model agreement (Spearman correlation between models)
3. Validation against known Tm data from literature (Brott 2022, etc.)
4. Ranking quality assessment

This script does NOT require ground truth for the challenge test set.
Instead, it uses biological priors and validated Tm data as proxies.

Usage:
    python scripts/validate_scores.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "petase_challenge_data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
VALIDATION_DIR = os.path.join(PROJECT_ROOT, "data")


def load_scores(name, path):
    """Load score CSV if it exists, return (name, df) or None."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        print("  Loaded %s: %d rows" % (name, len(df)))
        return df
    else:
        print("  %s not found at %s" % (name, path))
        return None


def check_biological_priors(name, scores_df):
    """Check that scores satisfy basic biological expectations."""
    print("\n--- %s: Biological Prior Checks ---" % name)

    n_mutations = scores_df["n_mutations"].astype(int).values
    wt_mask = n_mutations == 0
    mut_mask = n_mutations == 1

    checks_passed = 0
    checks_total = 0

    # Check 1: WT abs_ll > mutant abs_ll (WTs are the natural optimum)
    if wt_mask.sum() > 0 and mut_mask.sum() > 0:
        wt_abs = scores_df.loc[wt_mask, "abs_ll"].astype(float).mean()
        mut_abs = scores_df.loc[mut_mask, "abs_ll"].astype(float).mean()
        passed = wt_abs > mut_abs
        checks_total += 1
        if passed:
            checks_passed += 1
        print("  [%s] WT mean abs_ll (%.4f) > mutant mean abs_ll (%.4f)" % (
            "PASS" if passed else "FAIL", wt_abs, mut_abs))

    # Check 2: Most delta_ll values are negative (most mutations are deleterious)
    delta_ll = scores_df["delta_ll"].astype(float).values
    frac_negative = np.mean(delta_ll[mut_mask] < 0) if mut_mask.sum() > 0 else 0
    passed = frac_negative > 0.5
    checks_total += 1
    if passed:
        checks_passed += 1
    print("  [%s] %.1f%% of mutant delta_ll are negative (expect >50%%)" % (
        "PASS" if passed else "FAIL", frac_negative * 100))

    # Check 3: delta_ll distribution is roughly normal-ish (not degenerate)
    if mut_mask.sum() > 10:
        delta_std = np.std(delta_ll[mut_mask])
        passed = delta_std > 0.1
        checks_total += 1
        if passed:
            checks_passed += 1
        print("  [%s] delta_ll std = %.4f (expect > 0.1, not degenerate)" % (
            "PASS" if passed else "FAIL", delta_std))

    # Check 4: WT entropy < mutant entropy on average
    if "entropy" in scores_df.columns and wt_mask.sum() > 0:
        wt_ent = scores_df.loc[wt_mask, "entropy"].astype(float).mean()
        mut_ent = scores_df.loc[mut_mask, "entropy"].astype(float).mean()
        # This can go either way, just report
        print("  [INFO] WT mean entropy: %.4f, mutant mean entropy: %.4f" % (wt_ent, mut_ent))

    print("  Result: %d/%d checks passed" % (checks_passed, checks_total))
    return checks_passed, checks_total


def cross_model_agreement(models):
    """Compute Spearman correlation between model predictions."""
    if len(models) < 2:
        print("\n--- Cross-Model Agreement: Need >= 2 models ---")
        return

    print("\n=== Cross-Model Agreement (Spearman rho) ===")
    names = list(models.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            df1, df2 = models[n1], models[n2]

            for score_col in ["delta_ll", "abs_ll"]:
                if score_col in df1.columns and score_col in df2.columns:
                    v1 = df1[score_col].astype(float).values
                    v2 = df2[score_col].astype(float).values
                    if len(v1) == len(v2):
                        rho, pval = stats.spearmanr(v1, v2)
                        print("  %s vs %s [%s]: rho=%.4f (p=%.2e)" % (
                            n1, n2, score_col, rho, pval))


def validate_against_tm_data(models):
    """
    Use known Tm data from literature to validate model rankings.

    Logic: Among the 313 WT sequences in the challenge, some may correspond
    to known PETase variants. We use abs_ll as a proxy for Tm ranking
    and check if the model's ranking correlates with known Tm values.

    Even if we can't map challenge WTs to specific known variants,
    we can check: does abs_ll distinguish between different WT scaffolds?
    """
    print("\n=== Validation Against Known Tm Data ===")

    # Load Tm data
    tm_path = os.path.join(VALIDATION_DIR, "mutations_dataset.csv")
    if not os.path.exists(tm_path):
        print("  No Tm validation data found at %s" % tm_path)
        print("  Run: python scripts/extract_mutations.py")
        return

    tm_df = pd.read_csv(tm_path)
    ispetase = tm_df[tm_df["enzyme"] == "IsPETase"]
    print("  Loaded %d IsPETase Tm values (range: %.1f-%.1f C)" % (
        len(ispetase), ispetase["tm"].min(), ispetase["tm"].max()))

    # Load ML model comparison if available
    ml_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    if os.path.exists(ml_path):
        ml_df = pd.read_csv(ml_path)
        print("\n  ML Model Results (on Tm validation data):")
        print("  %-25s %12s %10s %10s" % ("Model", "LOOCV RMSE", "LOOCV R2", "Spearman"))
        print("  " + "-" * 60)
        for _, row in ml_df.iterrows():
            print("  %-25s %10.2f C %10.3f %10.3f" % (
                row["name"], row["loocv_rmse"], row["loocv_r2"],
                row.get("loocv_spearman", 0)))

    # For PLM models: check WT-level score variation
    for name, df in models.items():
        wt_mask = df["n_mutations"].astype(int) == 0
        if wt_mask.sum() > 1:
            wt_abs = df.loc[wt_mask, "abs_ll"].astype(float)
            print("\n  %s WT abs_ll statistics (%d WTs):" % (name, wt_mask.sum()))
            print("    mean=%.4f, std=%.4f, range=[%.4f, %.4f]" % (
                wt_abs.mean(), wt_abs.std(), wt_abs.min(), wt_abs.max()))
            print("    Good discrimination: std > 0.01 => %s" % (
                "YES" if wt_abs.std() > 0.01 else "NO (WTs indistinguishable)"))


def score_distribution_analysis(models):
    """Analyze score distributions for each model."""
    print("\n=== Score Distribution Analysis ===")

    for name, df in models.items():
        print("\n--- %s ---" % name)
        for col in ["delta_ll", "abs_ll", "entropy", "logit_native"]:
            if col in df.columns:
                vals = df[col].astype(float).values
                print("  %-15s mean=%8.4f  std=%8.4f  [%8.4f, %8.4f]" % (
                    col, np.mean(vals), np.std(vals), np.min(vals), np.max(vals)))


def per_wt_analysis(models):
    """Analyze score patterns per wild-type scaffold."""
    print("\n=== Per-WT Scaffold Analysis ===")

    for name, df in models.items():
        print("\n--- %s ---" % name)
        wt_groups = df.groupby("wt_idx")

        wt_stats = []
        for wt_idx, group in wt_groups:
            if len(group) < 5:
                continue
            muts = group[group["n_mutations"].astype(int) == 1]
            if len(muts) == 0:
                continue
            delta = muts["delta_ll"].astype(float)
            wt_stats.append({
                "wt_idx": wt_idx,
                "n_variants": len(muts),
                "mean_delta": delta.mean(),
                "std_delta": delta.std(),
                "frac_positive": (delta > 0).mean(),
            })

        if wt_stats:
            wt_stats_df = pd.DataFrame(wt_stats).sort_values("n_variants", ascending=False)
            print("  Top 10 WTs by variant count:")
            print("  %-8s %10s %12s %12s %12s" % (
                "WT_idx", "N_variants", "Mean_delta", "Std_delta", "Frac_pos"))
            for _, row in wt_stats_df.head(10).iterrows():
                print("  %-8d %10d %12.4f %12.4f %12.3f" % (
                    row["wt_idx"], row["n_variants"], row["mean_delta"],
                    row["std_delta"], row["frac_positive"]))

            # Fraction of beneficial mutations across all WTs
            total_pos = sum(s["frac_positive"] * s["n_variants"] for s in wt_stats)
            total_n = sum(s["n_variants"] for s in wt_stats)
            print("  Overall fraction of beneficial mutations (delta_ll > 0): %.1f%%" % (
                total_pos / total_n * 100 if total_n > 0 else 0))


def main():
    print("=" * 70)
    print("PET CHALLENGE 2025 - SCORING VALIDATION & MODEL COMPARISON")
    print("=" * 70)

    # Load all available score files
    print("\nLoading score files...")
    models = {}

    esm2 = load_scores("ESM2-650M", os.path.join(RESULTS_DIR, "esm2_scores.csv"))
    if esm2 is not None:
        models["ESM2-650M"] = esm2

    esmc = load_scores("ESMC-600M", os.path.join(RESULTS_DIR, "esmc_scores.csv"))
    if esmc is not None:
        models["ESMC-600M"] = esmc

    if len(models) == 0:
        print("\nERROR: No score files found. Run scoring scripts first:")
        print("  python scripts/esm2_zero_shot_scoring.py")
        print("  python scripts/esmc_scoring.py")
        return

    # 1. Biological prior checks
    print("\n" + "=" * 70)
    print("1. BIOLOGICAL PRIOR CHECKS")
    print("=" * 70)
    total_pass, total_checks = 0, 0
    for name, df in models.items():
        p, t = check_biological_priors(name, df)
        total_pass += p
        total_checks += t
    print("\nOverall: %d/%d biological checks passed" % (total_pass, total_checks))

    # 2. Score distributions
    print("\n" + "=" * 70)
    print("2. SCORE DISTRIBUTIONS")
    print("=" * 70)
    score_distribution_analysis(models)

    # 3. Per-WT analysis
    print("\n" + "=" * 70)
    print("3. PER-SCAFFOLD ANALYSIS")
    print("=" * 70)
    per_wt_analysis(models)

    # 4. Cross-model agreement
    print("\n" + "=" * 70)
    print("4. CROSS-MODEL AGREEMENT")
    print("=" * 70)
    cross_model_agreement(models)

    # 5. Validation against Tm data
    print("\n" + "=" * 70)
    print("5. VALIDATION AGAINST KNOWN Tm DATA")
    print("=" * 70)
    validate_against_tm_data(models)

    # 6. Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print("\nModels evaluated: %s" % ", ".join(models.keys()))
    print("Biological checks: %d/%d passed" % (total_pass, total_checks))
    if len(models) >= 2:
        print("Ensemble available: YES (run generate_submission.py for combined prediction)")
    else:
        print("Ensemble available: NO (need both ESM2 + ESMC scores)")

    print("\nRecommendation:")
    if total_pass == total_checks:
        print("  All checks passed. Models are producing biologically sensible scores.")
        print("  Proceed with generate_submission.py to create final submission.")
    else:
        print("  Some checks failed. Review score distributions and model outputs.")
        print("  Consider re-running scoring with different parameters.")


if __name__ == "__main__":
    main()
