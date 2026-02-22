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
    Returns (activity_score, expression_score) as numpy arrays.
    """
    delta_ll = scores_df["delta_ll"].astype(float).values
    abs_ll = scores_df["abs_ll"].astype(float).values
    entropy = scores_df["entropy"].astype(float).values
    logit_native = scores_df["logit_native"].astype(float).values

    z_delta = zscore(delta_ll)
    z_abs = zscore(abs_ll)
    z_entropy = zscore(-entropy)  # negate: lower entropy = better
    z_logit = zscore(logit_native)

    # Activity: delta_ll primary (mutation tolerance), abs_ll for cross-scaffold
    activity = 0.5 * z_delta + 0.3 * z_abs + 0.1 * z_entropy + 0.1 * z_logit

    # Expression: abs_ll primary (foldability), entropy as stability signal
    expression = 0.2 * z_delta + 0.4 * z_abs + 0.2 * z_entropy + 0.2 * z_logit

    return activity, expression


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
    activity_preds = []
    expression_preds = []
    models_used = []

    # ESM2 scores
    if not args.esmc_only and os.path.exists(ESM2_SCORES):
        print("Loading ESM2 scores from %s" % ESM2_SCORES)
        esm2 = pd.read_csv(ESM2_SCORES)
        assert len(esm2) == n_test, "ESM2 score count mismatch: %d vs %d" % (len(esm2), n_test)
        act, expr = compute_predictions(esm2)
        activity_preds.append(act)
        expression_preds.append(expr)
        models_used.append("ESM2-650M")
        n_mutations = esm2["n_mutations"].astype(int).values
    elif not args.esmc_only:
        print("WARNING: ESM2 scores not found at %s" % ESM2_SCORES)

    # ESMC scores
    if not args.esm2_only and os.path.exists(ESMC_SCORES):
        print("Loading ESMC scores from %s" % ESMC_SCORES)
        esmc = pd.read_csv(ESMC_SCORES)
        assert len(esmc) == n_test, "ESMC score count mismatch: %d vs %d" % (len(esmc), n_test)
        act, expr = compute_predictions(esmc)
        activity_preds.append(act)
        expression_preds.append(expr)
        models_used.append("ESMC-600M")
        if "n_mutations" not in dir():
            n_mutations = esmc["n_mutations"].astype(int).values
    elif not args.esm2_only:
        print("NOTE: ESMC scores not found at %s (run esmc_scoring.py first)" % ESMC_SCORES)

    if len(activity_preds) == 0:
        print("ERROR: No score files found. Run esm2_zero_shot_scoring.py "
              "and/or esmc_scoring.py first.")
        return

    print("\nUsing models: %s" % " + ".join(models_used))
    if len(models_used) > 1:
        print("Ensemble mode: averaging %d model predictions" % len(models_used))

    # Ensemble: simple average of z-scored predictions
    activity_score = np.mean(activity_preds, axis=0)
    expression_score = np.mean(expression_preds, axis=0)

    # Scale to physical ranges (rank-based to avoid outliers)
    # Activity: PETase specific activity typically 0-5 umol TPA/min*mg
    # Expression: E. coli typically 0-3 mg/mL
    activity_1 = rank_scale(activity_score, 0.0, 5.0)
    activity_2 = rank_scale(activity_score, 0.0, 5.0)
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
        r_act, _ = stats.spearmanr(activity_preds[0], activity_preds[1])
        r_exp, _ = stats.spearmanr(expression_preds[0], expression_preds[1])
        print("  Activity: r=%.3f" % r_act)
        print("  Expression: r=%.3f" % r_exp)


if __name__ == "__main__":
    main()
