#!/usr/bin/env python3
"""
ESMC-600M Zero-Shot Scoring for PET Challenge 2025

Uses EvolutionaryScale's ESMC-600M (ESM Cambrian) for WT-marginal scoring.
ESMC-600M rivals ESM2-3B performance with 600M parameters.

Requires: pip install esm  (EvolutionaryScale package, NOT fair-esm)
GPU recommended (Colab T4/A100 works well).

Computes same score types as ESM2 script for ensemble:
  - delta_ll: mutation log-likelihood ratio
  - abs_ll: absolute mean log-likelihood
  - entropy: mean positional entropy
  - logit_native: mean raw logit for native residue
  - joint_ll: joint log-likelihood over standard AAs

Usage:
    python scripts/esmc_scoring.py [--model esmc_300m|esmc_600m] [--cpu]
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch

# -- Paths (relative to project root) ----------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
WT_CSV = os.path.join(PROJECT_ROOT, "data", "petase_challenge_data",
                      "pet-2025-wildtype-cds.csv")
TEST_CSV = os.path.join(PROJECT_ROOT, "data", "petase_challenge_data",
                        "predictive-pet-zero-shot-test-2025.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Standard amino acids for entropy calculation
STD_AA = "ACDEFGHIKLMNPQRSTVWY"


def parse_args():
    parser = argparse.ArgumentParser(description="ESMC zero-shot scoring")
    parser.add_argument("--model", default="esmc_600m",
                        choices=["esmc_300m", "esmc_600m"],
                        help="ESMC model size (default: esmc_600m)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU (skip GPU)")
    parser.add_argument("--half", action="store_true", default=True,
                        help="Use float16 (default: True)")
    return parser.parse_args()


def load_model(model_name, use_cpu=False, use_half=True):
    """Load ESMC model from EvolutionaryScale package."""
    from esm.models.esmc import ESMC

    device = "cpu" if use_cpu or not torch.cuda.is_available() else "cuda"
    print("Loading %s..." % model_name)

    model = ESMC.from_pretrained(model_name)

    if device == "cuda":
        try:
            if use_half:
                model = model.half()
            model = model.to(device)
            print("Model loaded on %s (half=%s)" % (device, use_half))
        except RuntimeError as e:
            print("GPU failed (%s), falling back to CPU" % e)
            model = model.float().cpu()
            device = "cpu"
    else:
        print("Using CPU")

    model.requires_grad_(False)
    for param in model.parameters():
        param.requires_grad = False

    return model, device


def build_aa_token_map(model):
    """
    Build mapping from amino acid characters to ESMC token indices.
    Returns dict: {'A': idx, 'C': idx, ...}
    """
    tokenizer = model.tokenizer
    aa_map = {}
    for aa in STD_AA:
        tokens = tokenizer.encode(aa)
        # Encoded as [BOS, aa_token, EOS] — extract the middle token
        # Handle different tokenizer return types
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
        if isinstance(tokens, list) and len(tokens) >= 3:
            aa_map[aa] = tokens[1]  # skip BOS
        elif isinstance(tokens, list) and len(tokens) == 1:
            aa_map[aa] = tokens[0]
        else:
            aa_map[aa] = tokens[1] if len(tokens) > 1 else tokens[0]
    return aa_map


def score_sequence_esmc(model, sequence, device, aa_map):
    """
    Run single forward pass with ESMC on a sequence.
    Returns:
        log_probs: (L, V) log-probabilities
        logits_raw: (L, V) raw logits
        probs: (L, V) probabilities
    """
    from esm.sdk.api import ESMProtein, LogitsConfig

    protein = ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein)

    logits_output = model.logits(
        protein_tensor,
        LogitsConfig(sequence=True, return_embeddings=False)
    )

    # logits_output.logits is a ForwardTrackData wrapper; the actual tensor
    # is in the .sequence attribute: shape (1, L+2, V) including BOS/EOS
    logits = logits_output.logits.sequence
    if logits.dim() == 3:
        logits = logits[0]  # remove batch dim -> (L+?, V)

    # Determine if BOS/EOS tokens are included
    L = len(sequence)
    if logits.shape[0] == L + 2:
        logits = logits[1:-1]  # strip BOS and EOS
    elif logits.shape[0] == L + 1:
        logits = logits[1:]    # strip BOS only
    # else: logits.shape[0] == L, no stripping needed

    assert logits.shape[0] == L, (
        "Logits length %d != sequence length %d" % (logits.shape[0], L))

    logits_f = logits.float()
    log_probs = torch.nn.functional.log_softmax(logits_f, dim=-1)
    probs = torch.nn.functional.softmax(logits_f, dim=-1)

    return log_probs.cpu().numpy(), logits_f.cpu().numpy(), probs.cpu().numpy()


def compute_scores(log_probs, logits_raw, probs, sequence, aa_map):
    """Compute multiple score types from a single forward pass."""
    L = len(sequence)
    aa_indices = [aa_map[aa] for aa in sequence]

    # Indices for standard AAs
    std_indices = [aa_map[aa] for aa in STD_AA]

    # 1. Absolute log-likelihood
    native_ll = np.array([log_probs[i, aa_indices[i]] for i in range(L)])
    abs_ll = float(np.mean(native_ll))

    # 2. Native logit score
    native_logit = np.array([logits_raw[i, aa_indices[i]] for i in range(L)])
    logit_native = float(np.mean(native_logit))

    # 3. Entropy over standard AAs
    pos_entropy = np.zeros(L)
    for i in range(L):
        p = probs[i, std_indices]
        p = p / (p.sum() + 1e-10)
        pos_entropy[i] = -np.sum(p * np.log(p + 1e-10))
    entropy = float(np.mean(pos_entropy))

    # 4. Joint log-likelihood
    joint_ll = float(np.mean(np.array([
        np.sum(log_probs[i, std_indices]) for i in range(L)
    ])))

    return {
        "abs_ll": abs_ll,
        "logit_native": logit_native,
        "entropy": entropy,
        "joint_ll": joint_ll,
        "per_position_log_probs": log_probs,
    }


def score_mutation(wt_log_probs, wt_seq, mut_seq, aa_map):
    """Score mutation(s) using WT-marginal method."""
    assert len(wt_seq) == len(mut_seq)
    diffs = [(i, wt_seq[i], mut_seq[i])
             for i in range(len(wt_seq)) if wt_seq[i] != mut_seq[i]]
    if len(diffs) == 0:
        return 0.0

    delta_ll = 0.0
    for pos, wt_aa, mut_aa in diffs:
        wt_idx = aa_map[wt_aa]
        mut_idx = aa_map[mut_aa]
        delta_ll += wt_log_probs[pos, mut_idx] - wt_log_probs[pos, wt_idx]
    return float(delta_ll)


def main():
    args = parse_args()
    t0 = time.time()

    output_csv = os.path.join(RESULTS_DIR, "esmc_scores.csv")

    # Load data
    print("Loading data...")
    wt_df = pd.read_csv(WT_CSV)
    test_df = pd.read_csv(TEST_CSV)
    print("  %d wild-type sequences" % len(wt_df))
    print("  %d test sequences" % len(test_df))

    wt_seqs = list(wt_df["Wt AA Sequence"].values)

    # Map test sequences to their parent WT
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

    n_wt = sum(1 for n in test_n_muts if n == 0)
    n_single = sum(1 for n in test_n_muts if n == 1)
    print("  WT-identical: %d, single-mutant: %d, other: %d" % (
        n_wt, n_single, len(test_df) - n_wt - n_single))

    needed_wt = set(i for i in test_wt_idx if i is not None)
    print("  Need to score %d unique WTs" % len(needed_wt))

    # Load model
    model, device = load_model(args.model, use_cpu=args.cpu, use_half=args.half)
    aa_map = build_aa_token_map(model)
    print("  AA token map built (%d amino acids)" % len(aa_map))

    # Score all needed WTs
    print("\nScoring %d wild-type sequences on %s..." % (len(needed_wt), device))
    wt_results = {}
    for count, wi in enumerate(sorted(needed_wt)):
        seq = wt_seqs[wi]
        log_probs, logits_raw, probs = score_sequence_esmc(
            model, seq, device, aa_map
        )
        scores = compute_scores(log_probs, logits_raw, probs, seq, aa_map)
        wt_results[wi] = scores

        if (count + 1) % 10 == 0 or count == 0:
            elapsed = time.time() - t0
            rate = (count + 1) / elapsed
            remaining = (len(needed_wt) - count - 1) / rate if rate > 0 else 0
            print("  [%d/%d] WT%d (len=%d) abs_ll=%.4f entropy=%.4f "
                  "(%ds elapsed, ~%ds remaining)" % (
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

            delta_ll = score_mutation(
                wt_scores["per_position_log_probs"],
                wt_seq, test_seq, aa_map
            )

            wt_ll_per_pos = wt_scores["per_position_log_probs"]
            native_lls = [wt_ll_per_pos[i, aa_map[test_seq[i]]]
                          for i in range(len(test_seq))]
            mut_abs_ll = float(np.mean(native_lls))

            results.append({
                "test_idx": idx,
                "wt_idx": wi,
                "n_mutations": n_mut,
                "delta_ll": delta_ll,
                "abs_ll": mut_abs_ll,
                "wt_abs_ll": wt_scores["abs_ll"],
                "entropy": wt_scores["entropy"],
                "logit_native": wt_scores["logit_native"],
                "joint_ll": wt_scores["joint_ll"],
            })
        else:
            log_probs, logits_raw, probs = score_sequence_esmc(
                model, test_seq, device, aa_map
            )
            scores = compute_scores(log_probs, logits_raw, probs, test_seq, aa_map)
            results.append({
                "test_idx": idx,
                "wt_idx": -1,
                "n_mutations": n_mut,
                "delta_ll": 0.0,
                "abs_ll": scores["abs_ll"],
                "wt_abs_ll": scores["abs_ll"],
                "entropy": scores["entropy"],
                "logit_native": scores["logit_native"],
                "joint_ll": scores["joint_ll"],
            })

        if (idx + 1) % 500 == 0:
            print("  [%d/%d] scored" % (idx + 1, len(test_df)))

    # Save
    print("\nSaving %d scores to %s" % (len(results), output_csv))
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    out_rows = []
    for r in results:
        out_rows.append({
            "test_idx": r["test_idx"],
            "wt_idx": r["wt_idx"],
            "n_mutations": r["n_mutations"],
            "delta_ll": "%.6f" % r["delta_ll"],
            "abs_ll": "%.6f" % r["abs_ll"],
            "wt_abs_ll": "%.6f" % r["wt_abs_ll"],
            "entropy": "%.6f" % r["entropy"],
            "logit_native": "%.6f" % r["logit_native"],
            "joint_ll": "%.6f" % r["joint_ll"],
        })

    pd.DataFrame(out_rows).to_csv(output_csv, index=False)

    total_time = time.time() - t0
    print("\nDone! Total time: %ds (%.1f min)" % (int(total_time), total_time / 60))
    print("Output: %s" % output_csv)

    # Summary
    df = pd.DataFrame(results)
    print("\n=== Score Summary (%s) ===" % args.model)
    for col in ["delta_ll", "abs_ll", "entropy", "logit_native"]:
        print("  %s: mean=%.4f, std=%.4f" % (col, df[col].mean(), df[col].std()))


if __name__ == "__main__":
    main()
