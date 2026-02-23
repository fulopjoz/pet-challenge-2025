#!/usr/bin/env python3
"""
ESM2 Zero-Shot Scoring for PET Challenge 2025

Computes wildtype-marginal scores for all 4988 test sequences using ESM2-650M.
For single-point mutants: score = log_prob(mut_aa) - log_prob(wt_aa) at mutation site
For WT sequences: score = mean log_prob(native_aa) over all positions (absolute fitness)

Also computes multiple auxiliary scores per Kral (2025):
  - delta_ll: mutation log-likelihood ratio (primary for single mutants)
  - abs_ll: absolute mean log-likelihood (primary for WT ranking)
  - entropy: mean positional entropy (lower = more confident)
  - logit_native: mean raw logit for native residue

Requires: pip install fair-esm
GPU recommended but works on CPU.

Usage:
    python scripts/esm2_zero_shot_scoring.py [--cpu]
"""

import sys
import os
import time
import csv
import argparse
import numpy as np
import torch
import pandas as pd

# -- Config -------------------------------------------------------------------

MODEL_NAME = "esm2_t33_650M_UR50D"
BATCH_SIZE = 1
USE_HALF = True  # float16 to save VRAM

# Portable paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
WT_CSV = os.path.join(BASE_DIR, "data", "petase_challenge_data", "pet-2025-wildtype-cds.csv")
TEST_CSV = os.path.join(BASE_DIR, "data", "petase_challenge_data", "predictive-pet-zero-shot-test-2025.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "results", "esm2_scores.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="ESM2 zero-shot scoring")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU (skip GPU)")
    parser.add_argument("--no-half", action="store_true",
                        help="Disable float16 and use float32")
    return parser.parse_args()


def load_model(device_preference="auto", use_half=True):
    """Load ESM2-650M with fair-esm."""
    import esm
    if device_preference == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_preference

    print("Loading %s..." % MODEL_NAME)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    # Set to inference mode
    model.requires_grad_(False)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # Try GPU first
    if device == "cuda":
        try:
            if use_half:
                model = model.half()
            model = model.to(device)
            # Test with a short sequence
            _, _, test_tokens = batch_converter([("test", "ACDEFG")])
            with torch.no_grad():
                model(test_tokens.to(device))
            print("Model loaded on %s (half=%s)" % (device, use_half))
            return model, alphabet, batch_converter, device
        except RuntimeError as e:
            print("GPU failed (%s), falling back to CPU" % e)
            model = model.float().cpu()
            return model, alphabet, batch_converter, "cpu"
    else:
        print("Using CPU")
        return model, alphabet, batch_converter, "cpu"


def score_sequence(model, alphabet, batch_converter, sequence, device):
    """
    Run single forward pass on a sequence.
    Returns:
        log_probs: (L, V) log-probabilities for each position
        logits_raw: (L, V) raw logits
        probs: (L, V) probabilities
    where L = sequence length, V = vocabulary size
    """
    data = [("seq", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[])

    # logits shape: (1, L+2, V) -- includes BOS and EOS tokens
    logits = results["logits"][0]  # (L+2, V)

    # Strip BOS (position 0) and EOS (last position)
    logits = logits[1:-1]  # (L, V)

    # Always use float32 for softmax numerical stability
    logits_f = logits.float()
    log_probs = torch.nn.functional.log_softmax(logits_f, dim=-1)
    probs = torch.nn.functional.softmax(logits_f, dim=-1)

    return log_probs.cpu().numpy(), logits_f.cpu().numpy(), probs.cpu().numpy()


def compute_scores(log_probs, logits_raw, probs, sequence, alphabet):
    """
    Compute multiple score types from a single forward pass.
    Returns dict of scores for this sequence.
    """
    L = len(sequence)
    assert log_probs.shape[0] == L, "Length mismatch: %d vs %d" % (log_probs.shape[0], L)

    # Map amino acids to token indices
    aa_indices = [alphabet.get_idx(aa) for aa in sequence]

    # 1. Absolute log-likelihood: mean log_prob of native residues
    native_ll = np.array([log_probs[i, aa_indices[i]] for i in range(L)])
    abs_ll = float(np.mean(native_ll))

    # 2. Native logit score: mean raw logit of native residues
    native_logit = np.array([logits_raw[i, aa_indices[i]] for i in range(L)])
    logit_native = float(np.mean(native_logit))

    # 3. Entropy: mean positional entropy (over 20 standard amino acids)
    std_aa = "ACDEFGHIKLMNPQRSTVWY"
    std_indices = [alphabet.get_idx(aa) for aa in std_aa]
    pos_entropy = np.zeros(L)
    for i in range(L):
        p = probs[i, std_indices]
        p = p / p.sum()  # renormalize to standard AAs
        pos_entropy[i] = -np.sum(p * np.log(p + 1e-10))
    entropy = float(np.mean(pos_entropy))

    # 4. Joint log-likelihood: mean sum of log-likelihoods of all standard AAs
    joint_ll = float(np.mean(np.array([
        np.sum(log_probs[i, std_indices]) for i in range(L)
    ])))

    return {
        "abs_ll": abs_ll,
        "logit_native": logit_native,
        "entropy": entropy,
        "joint_ll": joint_ll,
        "per_position_log_probs": log_probs,
        "pos_entropy": pos_entropy,       # (L,) entropy at each position
        "native_ll": native_ll,           # (L,) native log-prob at each position
    }


def score_mutation(wt_log_probs, wt_seq, mut_seq, alphabet,
                   pos_entropy=None, native_ll=None):
    """
    Score a single-point mutation using WT-marginal method.
    Returns dict with:
      - delta_ll: log_prob(mut_aa, pos) - log_prob(wt_aa, pos)
      - entropy_at_site: positional entropy at mutation site (from WT context)
      - native_ll_at_site: native log-prob at mutation site (from WT context)
    """
    assert len(wt_seq) == len(mut_seq), "Length mismatch: %d vs %d" % (len(wt_seq), len(mut_seq))
    diffs = [(i, wt_seq[i], mut_seq[i]) for i in range(len(wt_seq)) if wt_seq[i] != mut_seq[i]]

    result = {"delta_ll": 0.0, "entropy_at_site": float('nan'), "native_ll_at_site": float('nan')}

    if len(diffs) == 0:
        return result  # identical to WT — site features are NaN

    delta_ll = 0.0
    for pos, wt_aa, mut_aa in diffs:
        wt_idx = alphabet.get_idx(wt_aa)
        mut_idx = alphabet.get_idx(mut_aa)
        delta_ll += wt_log_probs[pos, mut_idx] - wt_log_probs[pos, wt_idx]
    result["delta_ll"] = float(delta_ll)

    if pos_entropy is not None and len(diffs) > 0:
        result["entropy_at_site"] = float(np.mean([pos_entropy[pos] for pos, _, _ in diffs]))
    if native_ll is not None and len(diffs) > 0:
        result["native_ll_at_site"] = float(np.mean([native_ll[pos] for pos, _, _ in diffs]))

    return result


def main():
    args = parse_args()
    t0 = time.time()

    # Load data
    print("Loading data...")
    wt_df = pd.read_csv(WT_CSV)
    test_df = pd.read_csv(TEST_CSV)
    print("  %d wild-type sequences" % len(wt_df))
    print("  %d test sequences" % len(test_df))

    # Build WT lookup by sequence
    wt_seqs = list(wt_df["Wt AA Sequence"].values)

    # Group test sequences by their parent WT
    from collections import defaultdict
    wt_by_len = defaultdict(list)
    for i, seq in enumerate(wt_seqs):
        wt_by_len[len(seq)].append((i, seq))

    print("Mapping test sequences to wild-types...")
    test_wt_idx = []
    test_n_muts = []
    for test_seq in test_df["sequence"].values:
        tlen = len(test_seq)
        best_wt = None
        best_diff = 999
        for wi, wseq in wt_by_len.get(tlen, []):
            ndiff = sum(1 for a, b in zip(wseq, test_seq) if a != b)
            if ndiff < best_diff:
                best_diff = ndiff
                best_wt = wi
            if ndiff == 0:
                break
        test_wt_idx.append(best_wt)
        test_n_muts.append(best_diff)

    n_wt_identical = sum(1 for n in test_n_muts if n == 0)
    n_single_mut = sum(1 for n in test_n_muts if n == 1)
    print("  WT-identical: %d, single-mutant: %d, other: %d" % (
        n_wt_identical, n_single_mut, len(test_df) - n_wt_identical - n_single_mut))

    # Find unique WTs that are actually needed
    needed_wt = set(i for i in test_wt_idx if i is not None)
    print("  Need to score %d unique WTs" % len(needed_wt))

    # Load model
    requested_device = "cpu" if args.cpu else "auto"
    use_half = USE_HALF and (not args.no_half)
    model, alphabet, batch_converter, device = load_model(
        device_preference=requested_device,
        use_half=use_half
    )

    # Score all needed WTs
    print("\nScoring %d wild-type sequences on %s..." % (len(needed_wt), device))
    wt_results = {}
    for count, wi in enumerate(sorted(needed_wt)):
        seq = wt_seqs[wi]
        log_probs, logits_raw, probs = score_sequence(
            model, alphabet, batch_converter, seq, device
        )
        scores = compute_scores(log_probs, logits_raw, probs, seq, alphabet)
        wt_results[wi] = scores

        if (count + 1) % 10 == 0 or count == 0:
            elapsed = time.time() - t0
            rate = (count + 1) / elapsed
            remaining = (len(needed_wt) - count - 1) / rate if rate > 0 else 0
            print("  [%d/%d] WT%d (len=%d) abs_ll=%.4f entropy=%.4f (%ds elapsed, ~%ds remaining)" % (
                count + 1, len(needed_wt), wi, len(seq),
                scores["abs_ll"], scores["entropy"],
                int(elapsed), int(remaining)))

    print("\nWT scoring done in %ds" % int(time.time() - t0))

    # Score all test sequences
    print("\nScoring %d test variants..." % len(test_df))
    results = []
    for idx in range(len(test_df)):
        test_seq = test_df["sequence"].values[idx]
        wi = test_wt_idx[idx]
        n_mut = test_n_muts[idx]

        if wi is not None and wi in wt_results:
            wt_scores = wt_results[wi]
            wt_seq = wt_seqs[wi]

            # Delta LL (mutation score) + position-specific features from WT log_probs
            mut_result = score_mutation(
                wt_scores["per_position_log_probs"],
                wt_seq, test_seq, alphabet,
                pos_entropy=wt_scores["pos_entropy"],
                native_ll=wt_scores["native_ll"],
            )

            # Compute mutant absolute scores from WT log-probs (approximation)
            wt_ll_per_pos = wt_scores["per_position_log_probs"]
            native_lls = [wt_ll_per_pos[i, alphabet.get_idx(test_seq[i])]
                         for i in range(len(test_seq))]
            mut_abs_ll = float(np.mean(native_lls))

            results.append({
                "sequence": test_seq,
                "wt_idx": wi,
                "n_mutations": n_mut,
                "delta_ll": mut_result["delta_ll"],
                "abs_ll": mut_abs_ll,
                "wt_abs_ll": wt_scores["abs_ll"],
                "entropy": wt_scores["entropy"],
                "logit_native": wt_scores["logit_native"],
                "joint_ll": wt_scores["joint_ll"],
                "entropy_at_site": mut_result["entropy_at_site"],
                "native_ll_at_site": mut_result["native_ll_at_site"],
            })
        else:
            # No matching WT found -- score directly
            log_probs, logits_raw, probs = score_sequence(
                model, alphabet, batch_converter, test_seq, device
            )
            scores = compute_scores(log_probs, logits_raw, probs, test_seq, alphabet)
            results.append({
                "sequence": test_seq,
                "wt_idx": -1,
                "n_mutations": n_mut,
                "delta_ll": 0.0,
                "abs_ll": scores["abs_ll"],
                "wt_abs_ll": scores["abs_ll"],
                "entropy": scores["entropy"],
                "logit_native": scores["logit_native"],
                "joint_ll": scores["joint_ll"],
                "entropy_at_site": float('nan'),
                "native_ll_at_site": float('nan'),
            })

        if (idx + 1) % 500 == 0:
            print("  [%d/%d] scored" % (idx + 1, len(test_df)))

    # Save results
    print("\nSaving %d scores to %s" % (len(results), OUTPUT_CSV))
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    out_rows = []
    for i, r in enumerate(results):
        row = {
            "test_idx": i,
            "wt_idx": r["wt_idx"],
            "n_mutations": r["n_mutations"],
            "delta_ll": "%.10f" % r["delta_ll"],   # high precision to avoid ties
            "abs_ll": "%.10f" % r["abs_ll"],        # high precision to avoid ties
            "wt_abs_ll": "%.6f" % r["wt_abs_ll"],
            "entropy": "%.6f" % r["entropy"],
            "logit_native": "%.6f" % r["logit_native"],
            "joint_ll": "%.6f" % r["joint_ll"],
        }
        # New position-specific features (NaN for WT rows)
        eas = r["entropy_at_site"]
        nls = r["native_ll_at_site"]
        row["entropy_at_site"] = "" if np.isnan(eas) else "%.6f" % eas
        row["native_ll_at_site"] = "" if np.isnan(nls) else "%.6f" % nls
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUTPUT_CSV, index=False)

    total_time = time.time() - t0
    print("\nDone! Total time: %ds (%.1f min)" % (int(total_time), total_time / 60))
    print("Output: %s" % OUTPUT_CSV)

    # Summary statistics
    df = pd.DataFrame(results)
    print("\n=== Score Summary ===")
    print("delta_ll: mean=%.4f, std=%.4f, min=%.4f, max=%.4f" % (
        df["delta_ll"].mean(), df["delta_ll"].std(),
        df["delta_ll"].min(), df["delta_ll"].max()))
    print("abs_ll:   mean=%.4f, std=%.4f, min=%.4f, max=%.4f" % (
        df["abs_ll"].mean(), df["abs_ll"].std(),
        df["abs_ll"].min(), df["abs_ll"].max()))
    print("entropy:  mean=%.4f, std=%.4f" % (
        df["entropy"].mean(), df["entropy"].std()))


if __name__ == "__main__":
    main()
