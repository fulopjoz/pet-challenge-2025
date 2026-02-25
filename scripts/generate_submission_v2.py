#!/usr/bin/env python3
"""
Generate PET Challenge 2025 Enhanced Zero-Shot Submission (v2/v4)

Improvements over v1:
  1. Expression scoring uses CDS features (GC 5', rare codons) for scaffold-level
     differentiation + AA property changes for mutation-level signal
  2. pH-aware activity scoring: opposite charge directions for act1 vs act2
     - act1 (pH 5.5, suboptimal): negative charge lowers catalytic His pKa → maintains activity
     - act2 (pH 9.0, near-optimal): positive charge helps PET binding + salt bridges
  3. PLM features are properly decomposed: within-WT ranking uses delta_ll,
     between-WT ranking uses abs_ll/entropy/logit_native + CDS features
  4. v4: Position-specific features (entropy_at_site, native_ll_at_site) provide
     within-WT signal beyond delta_ll — conserved positions penalize mutations more

Key insight: Within each WT scaffold, entropy/logit_native/joint_ll are CONSTANT
across all single-point variants. Only delta_ll and abs_ll vary per mutation.
This means expression ranking within a WT was effectively just delta_ll in v1.

Literature basis:
  - Charlier 2024 (Biophys J): NMR titration of catalytic His242 in LCC(ICCG),
    pKa = 4.90 ± 0.05 (conserved alpha/beta-hydrolase mechanism suggests
    similar values for IsPETase and related PETases)
  - pH 5.5: His ~80% deprotonated (suboptimal); negative charge can lower His pKa
  - pH 9.0: His >99.9% deprotonated (near-optimal); fitness dominates
  - Lu 2022 (Nature): FAST-PETase N233K beneficial salt bridge at alkaline pH
  - Bell 2022 (Nature Catalysis): HotPETase maintains activity at pH 9.2
  - Expression: Codons 2-8 dominate E. coli expression (Nieuwkoop et al. 2023 NAR, r=0.762)
  - PLM scoring: WT-marginal delta_ll correlates with evolutionary fitness
    (Meier et al. 2021); rank correlation with masked marginal ≈1

Usage:
    python scripts/generate_submission_v2.py [--esm2-only] [--esmc-only] [--no-pka] [--require-pka] [--pka-v2]

Requires: results/esm2_scores.csv (and optionally esmc_scores.csv)
          results/mutation_features.csv (from compute_cds_features.py)
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "petase_challenge_data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

TEST_CSV = os.path.join(DATA_DIR, "predictive-pet-zero-shot-test-2025.csv")
ESM2_SCORES = os.path.join(RESULTS_DIR, "esm2_scores.csv")
ESMC_SCORES = os.path.join(RESULTS_DIR, "esmc_scores.csv")
MUTATION_FEATURES = os.path.join(RESULTS_DIR, "mutation_features.csv")
PKA_TEST_FEATURES = os.path.join(RESULTS_DIR, "pka_features_test.csv")
PKA_V2_TEST_FEATURES = os.path.join(RESULTS_DIR, "pka_features_test_v2.csv")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "submission_zero_shot_v2.csv")


def zscore(x):
    """Z-score normalize, handling constant arrays and NaN values."""
    s = np.nanstd(x)
    if not np.isfinite(s) or s < 1e-10:
        return np.zeros_like(x, dtype=float)
    return (x - np.nanmean(x)) / s


def rank_scale(scores, low, high):
    """Map scores to [low, high] range preserving rank order."""
    scores_arr = np.asarray(scores, dtype=float)
    bad_idx = np.where(~np.isfinite(scores_arr))[0]
    if bad_idx.size > 0:
        sample = bad_idx[:10].tolist()
        raise ValueError(
            "rank_scale received non-finite scores at indices %s (showing up to 10)." % sample
        )
    ranks = stats.rankdata(scores_arr)
    normalized = (ranks - 1) / max(len(ranks) - 1, 1)
    return low + normalized * (high - low)


def compute_plm_scores(scores_df):
    """
    Extract z-scored PLM features from a single model's scores.
    Returns dict of z-scored feature arrays.
    """
    delta_ll = scores_df["delta_ll"].astype(float).values
    abs_ll = scores_df["abs_ll"].astype(float).values
    entropy = scores_df["entropy"].astype(float).values
    logit_native = scores_df["logit_native"].astype(float).values
    n_mutations = scores_df["n_mutations"].astype(int).values

    # Z-score delta_ll ONLY among mutants to avoid WT inflation.
    # WT delta_ll=0 z-scores to ~+2.1 when pooled with mutants (mean~-9),
    # giving WTs a massive artificial boost. Instead, mutants compete
    # against each other, and WTs get z_delta=0 (neutral).
    mut_mask = n_mutations > 0
    z_delta = np.zeros_like(delta_ll)
    if mut_mask.sum() > 0:
        mut_delta = delta_ll[mut_mask]
        s = np.nanstd(mut_delta)
        if s > 1e-10:
            z_delta[mut_mask] = (mut_delta - np.nanmean(mut_delta)) / s
        # WTs keep z_delta = 0 (neutral, not inflated)

    result = {
        "z_delta": z_delta,
        "z_abs": zscore(abs_ll),
        "z_entropy": zscore(-entropy),  # negate: lower entropy = better
        "z_logit": zscore(logit_native),
    }

    # Position-specific features (v4): NaN for WT rows → fill with mean (neutral z~0)
    if "entropy_at_site" in scores_df.columns:
        eas = pd.to_numeric(scores_df["entropy_at_site"], errors="coerce").values
        nls = pd.to_numeric(scores_df["native_ll_at_site"], errors="coerce").values

        # NaN guard: if ALL values are NaN, nanmean returns NaN → default to 0
        eas_mean = np.nanmean(eas)
        if np.isnan(eas_mean):
            eas_mean = 0.0
        eas_filled = np.where(np.isnan(eas), eas_mean, eas)

        nls_mean = np.nanmean(nls)
        if np.isnan(nls_mean):
            nls_mean = 0.0
        nls_filled = np.where(np.isnan(nls), nls_mean, nls)

        # entropy_at_site: low entropy = conserved site = risky to mutate
        # zscore(eas_filled): low entropy → negative z → negative contribution = penalty
        result["z_entropy_at_site"] = zscore(eas_filled)

        # native_ll_at_site: values are log-probs (all negative; -0.1=confident, -8.0=uncertain)
        # After negation: confident site → small positive → below mean → negative z → penalty
        # So with positive weight: confident (conserved) site → penalty for mutation
        result["z_native_ll_at_site"] = zscore(-nls_filled)

        result["has_site_features"] = True
    else:
        result["has_site_features"] = False

    # ESM2 embedding cosine distance to WT (Step 2.4)
    # Only useful when actual per-mutant embeddings are computed (separate forward passes).
    # If all values are constant (e.g., all 0.0 or all NaN), fall back to z_logit.
    if "emb_cosine_dist_to_wt" in scores_df.columns:
        ecd = pd.to_numeric(scores_df["emb_cosine_dist_to_wt"], errors="coerce").values
        ecd_valid = ecd[~np.isnan(ecd)]
        if len(ecd_valid) > 0 and np.std(ecd_valid) > 1e-10:
            ecd_mean = np.nanmean(ecd)
            ecd_filled = np.where(np.isnan(ecd), ecd_mean, ecd)
            result["z_emb_dist"] = zscore(-ecd_filled)  # closer to WT = higher
            result["has_emb_dist"] = True
        else:
            # No variance in distances — feature is degenerate, fall back to z_logit
            result["has_emb_dist"] = False
    else:
        result["has_emb_dist"] = False

    return result


def compute_activity_1(plm, mut_feats, pka_feats=None, pka_v2_feats=None):
    """
    Activity at pH 5.5 — suboptimal pH, enzyme below alkaline optimum.

    Literature basis (Charlier 2024 — LCC His242, Hong 2023):
    - Catalytic His pKa ~4.9 → ~80% deprotonated at pH 5.5 (~20-30% of max activity)
    - Mutations adding negative charge can electrostatically lower His pKa
      → increase deprotonated fraction → maintain activity at suboptimal pH
    - Stability matters more when enzyme operates below its optimum

    v4: Added position-specific entropy (conserved site → riskier mutation).
    v5: Added PROPKA-based protonation fraction at pH 5.5 (physics-based pKa).
    v6: Per-mutant delta_pka (within-WT discriminator, 0.075 weight).
    Strategy: Moderate fitness weight + site conservation + negative charge + pKa + stability.
    """
    delta_charge = mut_feats["delta_charge"].values

    if plm.get("has_site_features"):
        if pka_v2_feats is not None:
            # v6 weights (sum=1.0): delta_pka provides within-WT pKa signal
            # Sign: negative delta_his_pka means mutation LOWERS pKa → more
            # deprotonated at pH 5.5 → better activity → positive contribution.
            z_pka_abs = zscore(pka_v2_feats["proton_frac_his_pH55"].values)
            z_delta_pka = zscore(-pka_v2_feats["delta_catalytic_his_pka"].values)
            score = (
                0.275 * plm["z_delta"]               # mutation tolerance
                + 0.225 * plm["z_abs"]               # foldability
                + 0.10 * plm["z_entropy"]            # conservation (between-WT)
                + 0.05 * plm["z_entropy_at_site"]    # conserved site → penalty
                + 0.05 * plm["z_native_ll_at_site"]  # confident site → penalty
                + 0.025 * zscore(-delta_charge)       # charge heuristic (reduced)
                + 0.05 * z_pka_abs                    # absolute protonation (between-WT)
                + 0.075 * z_delta_pka                 # delta pKa (NEW within-WT signal!)
                + 0.10 * zscore(-mut_feats["abs_delta_hydro"].values)  # stability
            )
            # z_logit/emb_dist: 0.05 total budget
            if plm.get("has_emb_dist"):
                score += 0.025 * plm["z_logit"] + 0.025 * plm["z_emb_dist"]
            else:
                score += 0.05 * plm["z_logit"]
        elif pka_feats is not None:
            # v5 weights (sum=1.0): pKa takes 0.05 from charge heuristic
            z_pka = zscore(pka_feats["proton_frac_his_pH55"].values)
            score = (
                0.30 * plm["z_delta"]               # mutation tolerance
                + 0.25 * plm["z_abs"]               # foldability
                + 0.10 * plm["z_entropy"]            # conservation (between-WT)
                + 0.05 * plm["z_entropy_at_site"]    # conserved site → penalty
                + 0.05 * plm["z_native_ll_at_site"]  # confident site → penalty
                + 0.05 * zscore(-delta_charge)        # charge heuristic (reduced from 0.10)
                + 0.05 * z_pka                        # physics-based pH 5.5 protonation
                + 0.10 * zscore(-mut_feats["abs_delta_hydro"].values)  # stability
            )
            # z_logit/emb_dist: 0.05 total budget
            if plm.get("has_emb_dist"):
                score += 0.025 * plm["z_logit"] + 0.025 * plm["z_emb_dist"]
            else:
                score += 0.05 * plm["z_logit"]
        else:
            # v4 weights (sum=1.0): site features, no pKa
            score = (
                0.30 * plm["z_delta"]
                + 0.25 * plm["z_abs"]
                + 0.10 * plm["z_entropy"]
                + 0.05 * plm["z_logit"]
                + 0.05 * plm["z_entropy_at_site"]
                + 0.05 * plm["z_native_ll_at_site"]
                + 0.10 * zscore(-delta_charge)
                + 0.10 * zscore(-mut_feats["abs_delta_hydro"].values)
            )
    else:
        # Fallback: v3 weights (no site features available)
        score = (
            0.35 * plm["z_delta"]
            + 0.25 * plm["z_abs"]
            + 0.10 * plm["z_entropy"]
            + 0.10 * plm["z_logit"]
            + 0.10 * zscore(-delta_charge)
            + 0.10 * zscore(-mut_feats["abs_delta_hydro"].values)
        )
    return score


def compute_activity_2(plm, mut_feats, pka_feats=None, pka_v2_feats=None):
    """
    Activity at pH 9.0 — near-optimal pH, fitness dominates.

    Literature basis (Charlier 2024 — LCC His242, Lu 2022, Bell 2022):
    - Catalytic His pKa ~4.9 → >99.9% deprotonated at pH 9.0 (enzyme at optimum)
    - Evolutionary fitness (delta_ll) is the best predictor at optimal pH
    - Positive charge additions help at alkaline pH:
      * N233K in FAST-PETase creates beneficial salt bridge with E204 (Lu 2022)
      * PET surface is more negative at alkaline pH → positive charges aid binding
      * HotPETase maintains activity at pH 9.2 (Bell 2022)

    v4: Added position-specific entropy (conserved site → riskier mutation).
    v5: Added PROPKA-based protonation fraction at pH 9.0 (physics-based pKa).
    v6: Per-mutant delta_pka (within-WT, 0.05 weight — less impact at pH 9).
    Strategy: Fitness-dominated + site conservation + positive charge + pKa + stability.
    """
    delta_charge = mut_feats["delta_charge"].values

    if plm.get("has_site_features"):
        if pka_v2_feats is not None:
            # v6 weights (sum=1.0): delta_pka matters less at pH 9.0 because
            # His is >99.9% deprotonated regardless of small pKa shifts
            z_pka_abs = zscore(pka_v2_feats["proton_frac_his_pH90"].values)
            z_delta_pka = zscore(-pka_v2_feats["delta_catalytic_his_pka"].values)
            score = (
                0.325 * plm["z_delta"]              # mutation tolerance (dominant at pH 9)
                + 0.20 * plm["z_abs"]               # foldability
                + 0.10 * plm["z_entropy"]            # conservation (between-WT)
                + 0.05 * plm["z_entropy_at_site"]    # conserved site → penalty
                + 0.05 * plm["z_native_ll_at_site"]  # confident site → penalty
                + 0.025 * zscore(delta_charge)        # positive charge helps (reduced)
                + 0.05 * z_pka_abs                    # absolute protonation (between-WT)
                + 0.05 * z_delta_pka                  # delta pKa (less impact at pH 9)
                + 0.10 * zscore(-mut_feats["abs_delta_hydro"].values)  # stability
            )
            # z_logit/emb_dist: 0.05 total budget
            if plm.get("has_emb_dist"):
                score += 0.025 * plm["z_logit"] + 0.025 * plm["z_emb_dist"]
            else:
                score += 0.05 * plm["z_logit"]
        elif pka_feats is not None:
            # v5 weights (sum=1.0): pKa takes 0.05 from charge heuristic
            z_pka = zscore(pka_feats["proton_frac_his_pH90"].values)
            score = (
                0.35 * plm["z_delta"]               # mutation tolerance (dominant)
                + 0.20 * plm["z_abs"]               # foldability
                + 0.10 * plm["z_entropy"]            # conservation (between-WT)
                + 0.05 * plm["z_entropy_at_site"]    # conserved site → penalty
                + 0.05 * plm["z_native_ll_at_site"]  # confident site → penalty
                + 0.05 * zscore(delta_charge)         # charge heuristic (reduced from 0.10)
                + 0.05 * z_pka                        # physics-based pH 9.0 protonation
                + 0.10 * zscore(-mut_feats["abs_delta_hydro"].values)  # stability
            )
            # z_logit/emb_dist: 0.05 total budget
            if plm.get("has_emb_dist"):
                score += 0.025 * plm["z_logit"] + 0.025 * plm["z_emb_dist"]
            else:
                score += 0.05 * plm["z_logit"]
        else:
            # v4 weights (sum=1.0): site features, no pKa
            score = (
                0.35 * plm["z_delta"]
                + 0.20 * plm["z_abs"]
                + 0.10 * plm["z_entropy"]
                + 0.05 * plm["z_logit"]
                + 0.05 * plm["z_entropy_at_site"]
                + 0.05 * plm["z_native_ll_at_site"]
                + 0.10 * zscore(delta_charge)
                + 0.10 * zscore(-mut_feats["abs_delta_hydro"].values)
            )
    else:
        # Fallback: v3 weights
        score = (
            0.45 * plm["z_delta"]
            + 0.20 * plm["z_abs"]
            + 0.10 * plm["z_entropy"]
            + 0.10 * plm["z_logit"]
            + 0.10 * zscore(delta_charge)
            + 0.05 * zscore(-mut_feats["abs_delta_hydro"].values)
        )
    return score


def compute_expression(plm, mut_feats):
    """
    Expression level (mg/mL) in E. coli pET28a system.

    Key insight: Within a WT scaffold, entropy/logit_native/joint_ll are CONSTANT.
    Only delta_ll and abs_ll vary per mutation. So we need non-PLM features to
    differentiate expression within a scaffold.

    Literature basis:
    - 5' mRNA structure (codons 2-8) is strongest expression predictor
      (Nieuwkoop et al. 2023, r=0.762 for codons 2-8)
      → cds_at_5prime_z (AT-rich 5' = less secondary structure = better expression)
    - Rare codons slow translation → cds_rare_codon_z (lower = better)
    - PLM abs_ll captures scaffold-level foldability (between-WT)
    - delta_ll captures mutation tolerance (within-WT)
    - Large hydrophobicity changes reduce solubility → abs_delta_hydro

    Weights: CDS scaffold features (0.30) + PLM features (0.55) + AA features (0.15)
    """
    # CDS features (scaffold-level, differs between WTs)
    # Already z-scored in compute_cds_features.py — do NOT re-zscore here
    z_at_5prime = mut_feats["cds_at_5prime_z"].values       # higher AT = better expression
    z_rare_neg = -mut_feats["cds_rare_codon_z"].values      # fewer rare codons = better

    # AA mutation features (mutation-level, differs within WT)
    z_abs_hydro_neg = zscore(-mut_feats["abs_delta_hydro"].values)  # less hydro change = better

    score = (
        0.30 * plm["z_delta"]        # mutation tolerance (within-WT signal)
        + 0.15 * plm["z_abs"]        # foldability (between-WT signal)
        + 0.10 * plm["z_entropy"]    # conservation (between-WT)
        + 0.15 * z_at_5prime         # 5' AT-richness → expression (between-WT)
        + 0.10 * z_rare_neg          # fewer rare codons (between-WT)
        + 0.10 * z_abs_hydro_neg     # mutation disruption (within-WT)
        + 0.10 * plm["z_logit"]      # native residue confidence (between-WT)
    )
    return score


def load_scores_aligned(path, model_name, n_test):
    """Load a score file and align rows to test-set order."""
    df = pd.read_csv(path)
    if len(df) != n_test:
        raise ValueError(
            "%s score count mismatch: %d vs %d" % (model_name, len(df), n_test)
        )

    required_cols = ["delta_ll", "abs_ll", "entropy", "logit_native", "n_mutations"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError("%s is missing required columns: %s" % (model_name, missing))

    # Fail fast on non-finite required PLM values to avoid silent NaN propagation.
    for col in ["delta_ll", "abs_ll", "entropy", "logit_native"]:
        vals = pd.to_numeric(df[col], errors="coerce").values
        bad_idx = np.where(~np.isfinite(vals))[0]
        if bad_idx.size > 0:
            sample = bad_idx[:10].tolist()
            raise ValueError(
                "%s has non-finite values in column '%s' at row indices %s "
                "(showing up to 10)." % (model_name, col, sample)
            )

    n_mut_vals = pd.to_numeric(df["n_mutations"], errors="coerce").values
    bad_n_mut = np.where(~np.isfinite(n_mut_vals))[0]
    if bad_n_mut.size > 0:
        sample = bad_n_mut[:10].tolist()
        raise ValueError(
            "%s has non-finite values in column 'n_mutations' at row indices %s "
            "(showing up to 10)." % (model_name, sample)
        )

    if "test_idx" in df.columns:
        idx = df["test_idx"].astype(int).values
        expected = np.arange(n_test, dtype=int)
        if np.array_equal(idx, expected):
            return df
        if np.array_equal(np.sort(idx), expected):
            print("Reordering %s by test_idx to match test set order." % model_name)
            return df.sort_values("test_idx").reset_index(drop=True)
        raise ValueError(
            "%s has invalid test_idx values (expected 0..%d)." % (model_name, n_test - 1)
        )

    print("WARNING: %s has no test_idx column; assuming existing row order." % model_name)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esm2-only", action="store_true",
                        help="Use only ESM2 scores (ignore ESMC)")
    parser.add_argument("--esmc-only", action="store_true",
                        help="Use only ESMC scores (ignore ESM2)")
    parser.add_argument("--no-pka", action="store_true",
                        help="Force v4 mode: disable pKa features even if file exists")
    parser.add_argument("--require-pka", action="store_true",
                        help="Require valid pKa features; fail fast if missing/invalid")
    parser.add_argument("--pka-v2", action="store_true",
                        help="Use per-mutant pKa v2 features with delta_pka (v6 scoring)")
    args = parser.parse_args()

    if args.esm2_only and args.esmc_only:
        parser.error("--esm2-only and --esmc-only cannot be used together.")
    if args.no_pka and args.require_pka:
        parser.error("--no-pka and --require-pka cannot be used together.")
    if args.no_pka and args.pka_v2:
        parser.error("--no-pka and --pka-v2 cannot be used together.")

    test_df = pd.read_csv(TEST_CSV)
    n_test = len(test_df)
    print("Test set: %d sequences" % n_test)

    # Load mutation features (CDS + AA properties)
    if not os.path.exists(MUTATION_FEATURES):
        print("ERROR: %s not found. Run compute_cds_features.py first." % MUTATION_FEATURES)
        raise SystemExit(2)
    mut_feats = pd.read_csv(MUTATION_FEATURES)
    if len(mut_feats) != n_test:
        raise ValueError("Mutation features count mismatch: %d vs %d" % (len(mut_feats), n_test))
    required_mut_cols = ["delta_charge", "abs_delta_hydro", "cds_at_5prime_z", "cds_rare_codon_z"]
    missing_mut_cols = [c for c in required_mut_cols if c not in mut_feats.columns]
    if missing_mut_cols:
        raise ValueError("Mutation features missing required columns: %s" % missing_mut_cols)
    nan_mut_mask = mut_feats[required_mut_cols].isna().any(axis=1)
    nan_mut_rows = int(nan_mut_mask.sum())
    if nan_mut_rows > 0:
        if "test_idx" in mut_feats.columns:
            sample_rows = mut_feats.loc[nan_mut_mask, "test_idx"].astype(int).head(10).tolist()
            sample_label = "test_idx"
        else:
            sample_rows = mut_feats.index[nan_mut_mask].astype(int).tolist()[:10]
            sample_label = "row"
        raise ValueError(
            "Mutation features contain %d/%d rows with NaN values in required columns %s. "
            "Sample %s values: %s" % (
                nan_mut_rows, len(mut_feats), required_mut_cols, sample_label, sample_rows
            )
        )

    # Ensure alignment by test_idx (same logic as PLM score alignment)
    if "test_idx" in mut_feats.columns:
        expected_idx = np.arange(n_test, dtype=int)
        actual_idx = mut_feats["test_idx"].astype(int).values
        if not np.array_equal(actual_idx, expected_idx):
            print("Reordering mutation features by test_idx")
            mut_feats = mut_feats.sort_values("test_idx").reset_index(drop=True)

    print("Loaded mutation features (CDS + AA properties)")

    # Load pKa features (optional by default; strict if --require-pka)
    pka_feats = None
    pka_v2_feats = None
    required_pka_cols = ["catalytic_his_pka", "proton_frac_his_pH55", "proton_frac_his_pH90"]
    required_pka_v2_cols = required_pka_cols + ["delta_catalytic_his_pka"]

    def _fill_nan_pka(df, cols):
        """Fill NaN pKa values with column means for z-scoring."""
        for col in cols:
            if col in df.columns:
                col_mean = df[col].mean()
                if np.isnan(col_mean):
                    col_mean = 0.0
                df[col] = df[col].fillna(col_mean)
        return df

    def _load_pka_file(path, required_cols, label, strict=False):
        """Load and validate a pKa features file. Returns DataFrame or None."""
        if not os.path.exists(path):
            msg = "pKa features not found at %s" % path
            if strict:
                raise FileNotFoundError("ERROR: %s (required)." % msg)
            print("NOTE: %s" % msg)
            return None
        df = pd.read_csv(path)
        if len(df) != n_test:
            msg = "pKa feature count mismatch (%d vs %d)" % (len(df), n_test)
            if strict:
                raise ValueError("ERROR: %s (required)." % msg)
            print("WARNING: %s, ignoring" % msg)
            return None
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            msg = "pKa features missing required columns: %s" % missing
            if strict:
                raise ValueError("ERROR: %s (required)." % msg)
            print("WARNING: %s, ignoring" % msg)
            return None
        if df["proton_frac_his_pH55"].isna().all() or df["proton_frac_his_pH90"].isna().all():
            msg = "pKa protonation columns are all-NaN"
            if strict:
                raise ValueError("ERROR: %s (required)." % msg)
            print("WARNING: %s, ignoring" % msg)
            return None
        if strict:
            # In strict mode, every row must have core pKa values.
            missing_row_mask = df[required_cols].isna().any(axis=1)
            missing_rows = int(missing_row_mask.sum())
            if missing_rows > 0:
                raise ValueError(
                    "ERROR: pKa features contain %d/%d rows with missing required values "
                    "(strict mode requires complete rows)." % (missing_rows, len(df)))
        print("Loaded %s pKa features from %s" % (label, path))
        return df

    if args.no_pka:
        print("NOTE: pKa disabled via --no-pka (forcing v4 behavior)")
    elif args.pka_v2:
        # v6 mode: load per-mutant pKa v2 features with delta columns
        pka_v2_raw = _load_pka_file(
            PKA_V2_TEST_FEATURES, required_pka_v2_cols, "v2 per-mutant",
            strict=args.require_pka)
        if pka_v2_raw is not None:
            fill_cols = ["proton_frac_his_pH55", "proton_frac_his_pH90",
                         "catalytic_his_pka", "delta_protonation_his",
                         "delta_catalytic_his_pka", "delta_proton_frac_pH55",
                         "delta_proton_frac_pH90"]
            pka_v2_feats = _fill_nan_pka(pka_v2_raw, fill_cols)
            print("  Per-mutant pKa v2 loaded — v6 scoring enabled (delta_pka features)")
        elif args.require_pka:
            raise FileNotFoundError("ERROR: --pka-v2 --require-pka but v2 features unavailable.")
        else:
            print("NOTE: v2 pKa unavailable — falling back to v5/v4")
            # Try loading v1 pKa as fallback
            pka_v1_raw = _load_pka_file(PKA_TEST_FEATURES, required_pka_cols, "v1 WT-mapped")
            if pka_v1_raw is not None:
                fill_cols_v1 = ["proton_frac_his_pH55", "proton_frac_his_pH90",
                                "catalytic_his_pka", "delta_protonation_his"]
                pka_feats = _fill_nan_pka(pka_v1_raw, fill_cols_v1)
    else:
        # Standard v5 mode: load WT-mapped pKa features
        pka_v1_raw = _load_pka_file(
            PKA_TEST_FEATURES, required_pka_cols, "v1 WT-mapped",
            strict=args.require_pka)
        if pka_v1_raw is not None:
            fill_cols_v1 = ["proton_frac_his_pH55", "proton_frac_his_pH90",
                            "catalytic_his_pka", "delta_protonation_his"]
            pka_feats = _fill_nan_pka(pka_v1_raw, fill_cols_v1)
            print("  Physics-based pH scoring enabled (v5)")

    # Collect per-model predictions
    activity1_preds = []
    activity2_preds = []
    expression_preds = []
    models_used = []
    n_mutations = None

    # ESM2 scores
    if not args.esmc_only and os.path.exists(ESM2_SCORES):
        print("Loading ESM2 scores from %s" % ESM2_SCORES)
        esm2 = load_scores_aligned(ESM2_SCORES, "ESM2", n_test)
        plm = compute_plm_scores(esm2)
        activity1_preds.append(compute_activity_1(plm, mut_feats, pka_feats, pka_v2_feats))
        activity2_preds.append(compute_activity_2(plm, mut_feats, pka_feats, pka_v2_feats))
        expression_preds.append(compute_expression(plm, mut_feats))
        models_used.append("ESM2-650M")
        n_mutations = esm2["n_mutations"].astype(int).values
    elif not args.esmc_only:
        print("WARNING: ESM2 scores not found at %s" % ESM2_SCORES)

    # ESMC scores
    if not args.esm2_only and os.path.exists(ESMC_SCORES):
        print("Loading ESMC scores from %s" % ESMC_SCORES)
        esmc = load_scores_aligned(ESMC_SCORES, "ESMC", n_test)
        plm = compute_plm_scores(esmc)
        activity1_preds.append(compute_activity_1(plm, mut_feats, pka_feats, pka_v2_feats))
        activity2_preds.append(compute_activity_2(plm, mut_feats, pka_feats, pka_v2_feats))
        expression_preds.append(compute_expression(plm, mut_feats))
        models_used.append("ESMC-600M")
        esmc_n_mut = esmc["n_mutations"].astype(int).values
        if n_mutations is None:
            n_mutations = esmc_n_mut
        elif not np.array_equal(n_mutations, esmc_n_mut):
            raise ValueError("n_mutations mismatch between models.")
    elif not args.esm2_only:
        print("NOTE: ESMC scores not found at %s" % ESMC_SCORES)

    if len(activity1_preds) == 0:
        print("ERROR: No PLM score files found.")
        raise SystemExit(2)

    print("\nUsing models: %s" % " + ".join(models_used))
    if len(models_used) > 1:
        print("Ensemble mode: averaging %d model predictions" % len(models_used))
    if pka_v2_feats is not None:
        version = "v6"
    elif pka_feats is not None:
        version = "v5"
    else:
        version = "v4"
    print("Scoring mode: %s" % version)

    # Ensemble: average model-level predictions
    activity1_score = np.mean(activity1_preds, axis=0)
    activity2_score = np.mean(activity2_preds, axis=0)
    expression_score = np.mean(expression_preds, axis=0)

    def _ensure_finite_scores(name, arr):
        arr_np = np.asarray(arr, dtype=float)
        bad_idx = np.where(~np.isfinite(arr_np))[0]
        if bad_idx.size > 0:
            sample = bad_idx[:10].tolist()
            raise ValueError(
                "%s contains non-finite values at indices %s (showing up to 10)." %
                (name, sample)
            )

    _ensure_finite_scores("activity1_score", activity1_score)
    _ensure_finite_scores("activity2_score", activity2_score)
    _ensure_finite_scores("expression_score", expression_score)

    # Scale to physical ranges (rank-based)
    activity_1 = rank_scale(activity1_score, 0.0, 5.0)
    activity_2 = rank_scale(activity2_score, 0.0, 5.0)
    expression = rank_scale(expression_score, 0.0, 3.0)

    # Build submission
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

    # Summary (version already set correctly above)
    print("\n=== Submission Summary (%s) ===" % version)
    print("Models: %s" % ", ".join(models_used))
    feat_list = "PLM scores + position-specific (entropy_at_site, native_ll_at_site) + CDS + AA properties"
    if pka_v2_feats is not None:
        feat_list += " + per-mutant pKa v2 (pKAI/PROPKA, delta features)"
    elif pka_feats is not None:
        feat_list += " + PROPKA pKa (protonation fractions)"
    print("Features: %s" % feat_list)
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
        for name, arr in [("activity_1", activity_1), ("activity_2", activity_2),
                          ("expression", expression)]:
            wt_mean = arr[wt_mask].mean()
            mut_mean = arr[mut_mask].mean()
            status = "OK" if wt_mean > mut_mean else "WARNING"
            print("  %s: WT=%.4f, mutants=%.4f [%s]" % (name, wt_mean, mut_mean, status))

    # Compare v1 vs v2 if v1 exists
    v1_path = os.path.join(RESULTS_DIR, "submission_zero_shot.csv")
    if os.path.exists(v1_path):
        v1 = pd.read_csv(v1_path)
        print("\n=== v1 vs v2 Comparison (Spearman) ===")
        for colname, v2_arr in [("activity_1", activity_1), ("activity_2", activity_2),
                                ("expression", expression)]:
            v1_col = [c for c in v1.columns if colname in c]
            if v1_col:
                r, _ = stats.spearmanr(v1[v1_col[0]].values, v2_arr)
                print("  %s: r=%.4f" % (colname, r))

    # Activity 1 vs 2 correlation (should be different now)
    r_12, _ = stats.spearmanr(activity_1, activity_2)
    print("\n  activity_1 vs activity_2: r=%.4f (expect < 1.0)" % r_12)

    # Expression vs activity correlation
    r_e1, _ = stats.spearmanr(expression, activity_1)
    print("  expression vs activity_1: r=%.4f" % r_e1)


if __name__ == "__main__":
    main()
