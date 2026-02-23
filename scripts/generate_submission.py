#!/usr/bin/env python3
"""
Generate PET Challenge 2025 Zero-Shot Submission

Combines PLM scores into predictions for activity_1, activity_2, and expression.
Supports ESM2-only, ESMC-only, or ESM2+ESMC ensemble.

Strategy:
  - activity_1 & activity_2: delta_ll as primary (mutation fitness)
    Higher delta_ll = mutation more tolerated = likely more functional
    For WT sequences: abs_ll differentiates between scaffolds
  - expression: abs_ll as primary (fitness proxy for foldability/expressibility)
    Per Kral (2025): conservation correlates with expression

Score combination:
  - Z-score normalize each score type per model
  - Weighted combination per output target
  - When ensemble: average model-level predictions (equal weight)

Evaluation: NDCG (ranking metric) — only relative order matters, not absolute values.

Usage:
    python scripts/generate_submission.py [--esm2-only] [--esmc-only]
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats

# Portable paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "petase_challenge_data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

TEST_CSV = os.path.join(DATA_DIR, "predictive-pet-zero-shot-test-2025.csv")
ESM2_SCORES = os.path.join(RESULTS_DIR, "esm2_scores.csv")
ESMC_SCORES = os.path.join(RESULTS_DIR, "esmc_scores.csv")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "submission_zero_shot.csv")


def zscore(x):
    """Z-score normalize, handling constant arrays."""
    s = np.std(x)
    if s < 1e-10:
        return np.zeros_like(x)
    return (x - np.mean(x)) / s


def rank_scale(scores, low, high):
    """Map scores to [low, high] range preserving rank order."""
    ranks = stats.rankdata(scores)
    normalized = (ranks - 1) / max(len(ranks) - 1, 1)
    return low + normalized * (high - low)


def compute_predictions(scores_df):
    """
    Compute activity and expression predictions from a single model's scores.
    Returns (activity_1_score, activity_2_score, expression_score) as numpy arrays.
    """
    delta_ll = scores_df["delta_ll"].astype(float).values
    abs_ll = scores_df["abs_ll"].astype(float).values
    entropy = scores_df["entropy"].astype(float).values
    logit_native = scores_df["logit_native"].astype(float).values

    z_delta = zscore(delta_ll)
    z_abs = zscore(abs_ll)
    z_entropy = zscore(-entropy)  # negate: lower entropy = better
    z_logit = zscore(logit_native)

    # Activity 1 (pH 5.5): mutation tolerance dominates.
    activity_1 = 0.5 * z_delta + 0.3 * z_abs + 0.1 * z_entropy + 0.1 * z_logit

    # Activity 2 (pH 9.0): harsher conditions likely favor stability proxies more.
    activity_2 = 0.35 * z_delta + 0.35 * z_abs + 0.20 * z_entropy + 0.10 * z_logit

    # Expression: abs_ll primary (foldability), entropy as stability signal
    expression = 0.2 * z_delta + 0.4 * z_abs + 0.2 * z_entropy + 0.2 * z_logit

    return activity_1, activity_2, expression


def load_scores_aligned(path, model_name, n_test):
    """
    Load a score file and align rows to test-set order.

    Alignment uses `test_idx` when available; otherwise row order is assumed.
    """
    df = pd.read_csv(path)
    if len(df) != n_test:
        raise ValueError(
            "%s score count mismatch: %d vs %d" % (model_name, len(df), n_test)
        )

    required_cols = ["delta_ll", "abs_ll", "entropy", "logit_native", "n_mutations"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError("%s is missing required columns: %s" % (model_name, missing))

    if "test_idx" in df.columns:
        idx = df["test_idx"].astype(int).values
        expected = np.arange(n_test, dtype=int)
        if np.array_equal(idx, expected):
            return df
        if np.array_equal(np.sort(idx), expected):
            print("Reordering %s by test_idx to match test set order." % model_name)
            return df.sort_values("test_idx").reset_index(drop=True)
        raise ValueError("%s has invalid test_idx values (expected 0..%d)." % (model_name, n_test - 1))

    print("WARNING: %s has no test_idx column; assuming existing row order." % model_name)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esm2-only", action="store_true",
                        help="Use only ESM2 scores (ignore ESMC)")
    parser.add_argument("--esmc-only", action="store_true",
                        help="Use only ESMC scores (ignore ESM2)")
    args = parser.parse_args()

    test_df = pd.read_csv(TEST_CSV)
    n_test = len(test_df)
    print("Test set: %d sequences" % n_test)

    # Collect predictions from available models
    activity1_preds = []
    activity2_preds = []
    expression_preds = []
    models_used = []
    n_mutations = None

    # ESM2 scores
    if not args.esmc_only and os.path.exists(ESM2_SCORES):
        print("Loading ESM2 scores from %s" % ESM2_SCORES)
        esm2 = load_scores_aligned(ESM2_SCORES, "ESM2", n_test)
        act1, act2, expr = compute_predictions(esm2)
        activity1_preds.append(act1)
        activity2_preds.append(act2)
        expression_preds.append(expr)
        models_used.append("ESM2-650M")
        n_mutations = esm2["n_mutations"].astype(int).values
    elif not args.esmc_only:
        print("WARNING: ESM2 scores not found at %s" % ESM2_SCORES)

    # ESMC scores
    if not args.esm2_only and os.path.exists(ESMC_SCORES):
        print("Loading ESMC scores from %s" % ESMC_SCORES)
        esmc = load_scores_aligned(ESMC_SCORES, "ESMC", n_test)
        act1, act2, expr = compute_predictions(esmc)
        activity1_preds.append(act1)
        activity2_preds.append(act2)
        expression_preds.append(expr)
        models_used.append("ESMC-600M")
        esmc_n_mut = esmc["n_mutations"].astype(int).values
        if n_mutations is None:
            n_mutations = esmc_n_mut
        elif not np.array_equal(n_mutations, esmc_n_mut):
            raise ValueError(
                "n_mutations mismatch between loaded score files. "
                "Ensure both files correspond to the same test set/order."
            )
    elif not args.esm2_only:
        print("NOTE: ESMC scores not found at %s (run esmc_scoring.py first)" % ESMC_SCORES)

    if len(activity1_preds) == 0:
        print("ERROR: No score files found. Run esm2_zero_shot_scoring.py "
              "and/or esmc_scoring.py first.")
        return

    print("\nUsing models: %s" % " + ".join(models_used))
    if len(models_used) > 1:
        print("Ensemble mode: averaging %d model predictions" % len(models_used))

    # Ensemble: simple average of z-scored predictions
    activity1_score = np.mean(activity1_preds, axis=0)
    activity2_score = np.mean(activity2_preds, axis=0)
    expression_score = np.mean(expression_preds, axis=0)

    # Scale to physical ranges (rank-based to avoid outliers)
    # Activity: PETase specific activity typically 0-5 umol TPA/min*mg
    # Expression: E. coli typically 0-3 mg/mL
    activity_1 = rank_scale(activity1_score, 0.0, 5.0)
    activity_2 = rank_scale(activity2_score, 0.0, 5.0)
    expression = rank_scale(expression_score, 0.0, 3.0)

    # Build submission with original column names
    submission = test_df.copy()
    col_act1 = [c for c in submission.columns if "activity_1" in c][0]
    col_act2 = [c for c in submission.columns if "activity_2" in c][0]
    col_expr = [c for c in submission.columns if "expression" in c][0]

    submission[col_act1] = activity_1
    submission[col_act2] = activity_2
    submission[col_expr] = expression

    os.makedirs(RESULTS_DIR, exist_ok=True)
    submission.to_csv(OUTPUT_CSV, index=False)
    print("\nSubmission saved to %s" % OUTPUT_CSV)

    # Summary
    print("\n=== Submission Summary ===")
    print("Models: %s" % ", ".join(models_used))
    print("Sequences: %d" % n_test)
    for name, arr in [("activity_1", activity_1), ("activity_2", activity_2),
                      ("expression", expression)]:
        print("%s: mean=%.4f, std=%.4f, min=%.4f, max=%.4f" % (
            name, arr.mean(), arr.std(), arr.min(), arr.max()))

    # Sanity checks
    print("\n=== Sanity Checks ===")
    wt_mask = n_mutations == 0
    mut_mask = n_mutations == 1
    if wt_mask.sum() > 0 and mut_mask.sum() > 0:
        print("Mean activity_1: WT=%.4f, mutants=%.4f (expect WT > mutants)" % (
            activity_1[wt_mask].mean(), activity_1[mut_mask].mean()))
        print("Mean expression: WT=%.4f, mutants=%.4f (expect WT > mutants)" % (
            expression[wt_mask].mean(), expression[mut_mask].mean()))
        if activity_1[wt_mask].mean() <= activity_1[mut_mask].mean():
            print("  WARNING: WT activity not higher than mutants!")
        if expression[wt_mask].mean() <= expression[mut_mask].mean():
            print("  WARNING: WT expression not higher than mutants!")

    # Score correlations between models (if ensemble)
    if len(models_used) > 1:
        print("\n=== Model Agreement (Spearman) ===")
        r_act1, _ = stats.spearmanr(activity1_preds[0], activity1_preds[1])
        r_act2, _ = stats.spearmanr(activity2_preds[0], activity2_preds[1])
        r_exp, _ = stats.spearmanr(expression_preds[0], expression_preds[1])
        print("  Activity 1: r=%.3f" % r_act1)
        print("  Activity 2: r=%.3f" % r_act2)
        print("  Expression: r=%.3f" % r_exp)


if __name__ == "__main__":
    main()
