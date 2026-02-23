#!/usr/bin/env python3
"""
Patch esm2_scores.csv with position-specific features (entropy_at_site, native_ll_at_site).

Only processes the 3 main WT scaffolds (WT0, WT1, WT2) that cover 93.8% of the test set.
All other rows (WT-identical entries with 0 mutations) get NaN for the site features.

This avoids re-running the full scoring pipeline (~70 min on CPU) by only doing 3 forward passes.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
WT_CSV = os.path.join(BASE_DIR, "data", "petase_challenge_data", "pet-2025-wildtype-cds.csv")
TEST_CSV = os.path.join(BASE_DIR, "data", "petase_challenge_data", "predictive-pet-zero-shot-test-2025.csv")
SCORES_CSV = os.path.join(BASE_DIR, "results", "esm2_scores.csv")

def main():
    t0 = time.time()

    # Load data
    print("Loading data...")
    wt_df = pd.read_csv(WT_CSV)
    test_df = pd.read_csv(TEST_CSV)
    scores_df = pd.read_csv(SCORES_CSV)

    wt_seqs = list(wt_df["Wt AA Sequence"].values)
    test_seqs = list(test_df["sequence"].values)

    print("  %d test sequences, %d existing score rows" % (len(test_seqs), len(scores_df)))

    # Identify which WTs need forward passes (those with actual mutants)
    from collections import Counter
    wt_counts = Counter(scores_df["wt_idx"].astype(int).values)
    # WTs with >1 test row have mutant variants
    needed_wts = sorted([wi for wi, count in wt_counts.items() if count > 1 and wi >= 0])
    print("  WTs with mutant variants: %s (counts: %s)" % (
        needed_wts, [wt_counts[wi] for wi in needed_wts]))

    # Load ESM2
    print("\nLoading ESM2-650M...")
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.requires_grad_(False)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    device = "cpu"
    print("Model loaded on CPU")

    # Standard AAs for entropy calculation
    std_aa = "ACDEFGHIKLMNPQRSTVWY"
    std_indices = [alphabet.get_idx(aa) for aa in std_aa]

    # Process each needed WT
    # Store: wt_idx -> {pos_entropy: array, native_ll: array}
    wt_site_data = {}

    for wi in needed_wts:
        seq = wt_seqs[wi]
        print("  Scoring WT%d (len=%d)..." % (wi, len(seq)), end=" ", flush=True)
        st = time.time()

        # Forward pass
        data = [("seq", seq)]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[])

        logits = results["logits"][0][1:-1]  # strip BOS/EOS
        logits_f = logits.float()
        log_probs = torch.nn.functional.log_softmax(logits_f, dim=-1).cpu().numpy()
        probs = torch.nn.functional.softmax(logits_f, dim=-1).cpu().numpy()

        L = len(seq)
        aa_indices = [alphabet.get_idx(aa) for aa in seq]

        # Native log-likelihood per position
        native_ll = np.array([log_probs[i, aa_indices[i]] for i in range(L)])

        # Entropy per position (over 20 standard AAs)
        pos_entropy = np.zeros(L)
        for i in range(L):
            p = probs[i, std_indices]
            p = p / p.sum()
            pos_entropy[i] = -np.sum(p * np.log(p + 1e-10))

        wt_site_data[wi] = {"pos_entropy": pos_entropy, "native_ll": native_ll}
        print("done (%.1fs)" % (time.time() - st))

    # Now patch the scores CSV
    print("\nPatching %d rows..." % len(scores_df))
    entropy_at_site = np.full(len(scores_df), np.nan)
    native_ll_at_site = np.full(len(scores_df), np.nan)

    for idx in range(len(scores_df)):
        wi = int(scores_df.iloc[idx]["wt_idx"])
        n_mut = int(scores_df.iloc[idx]["n_mutations"])

        if n_mut == 0 or wi not in wt_site_data:
            continue  # NaN for WT-identical rows and minor WTs

        test_seq = test_seqs[idx]
        wt_seq = wt_seqs[wi]

        # Find mutation positions
        diffs = [(i, wt_seq[i], test_seq[i]) for i in range(len(wt_seq)) if wt_seq[i] != test_seq[i]]
        if len(diffs) == 0:
            continue

        site_data = wt_site_data[wi]
        entropy_at_site[idx] = float(np.mean([site_data["pos_entropy"][pos] for pos, _, _ in diffs]))
        native_ll_at_site[idx] = float(np.mean([site_data["native_ll"][pos] for pos, _, _ in diffs]))

    # Add columns to dataframe
    scores_df["entropy_at_site"] = entropy_at_site
    scores_df["native_ll_at_site"] = native_ll_at_site

    # Format: empty string for NaN, %.6f otherwise
    scores_df["entropy_at_site"] = scores_df["entropy_at_site"].apply(
        lambda x: "" if pd.isna(x) else "%.6f" % x)
    scores_df["native_ll_at_site"] = scores_df["native_ll_at_site"].apply(
        lambda x: "" if pd.isna(x) else "%.6f" % x)

    # Save
    scores_df.to_csv(SCORES_CSV, index=False)
    print("\nSaved updated scores to %s" % SCORES_CSV)

    # Summary
    n_nan = int(sum(1 for x in entropy_at_site if np.isnan(x)))
    n_filled = len(entropy_at_site) - n_nan
    print("\n=== Patch Summary ===")
    print("  Total rows: %d" % len(scores_df))
    print("  Rows with site features: %d" % n_filled)
    print("  NaN rows (WT/minor): %d" % n_nan)

    # Spot-check: show some values for WT0 mutants
    wt0_mask = (scores_df["wt_idx"].astype(int) == 0) & (scores_df["n_mutations"].astype(int) == 1)
    wt0_eas = scores_df.loc[wt0_mask, "entropy_at_site"].head(5)
    print("\n  Sample entropy_at_site for WT0 mutants:")
    for i, v in wt0_eas.items():
        print("    row %d: %s" % (i, v))

    print("\nDone in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
