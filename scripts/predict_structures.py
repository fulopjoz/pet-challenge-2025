#!/usr/bin/env python3
"""
Predict 3D structures for PETase sequences using ESMFold.

ESMFold predicts structures directly from sequence (no MSA needed).
Uses ~3-4GB VRAM for sequences up to ~400 aa. Fast: ~1-2 sec per sequence on T4.

Uses HuggingFace transformers implementation (no openfold dependency).

Important: Run AFTER ESM2 scoring (unload ESM2 first to free VRAM),
or run in a separate Colab cell.

Modes:
  wt   — Predict 313 WT structures only (default, backward compatible)
  test — Predict all 4988 test sequences (for per-mutant pKa)
  all  — Predict both WT and test sequences

Usage:
    python scripts/predict_structures.py [--mode {wt,test,all}] [--cpu] [--max-seqs N]

Requires: pip install transformers torch
Output: results/structures/wt_*.pdb      (WT mode)
        results/structures/test_*.pdb    (test mode)
        results/structures/plddt_summary.csv      (WT pLDDT)
        results/structures/plddt_summary_test.csv  (test pLDDT)
"""

import os
import sys
import time
import argparse
import csv
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
WT_CSV = os.path.join(BASE_DIR, "data", "petase_challenge_data", "pet-2025-wildtype-cds.csv")
TEST_CSV = os.path.join(BASE_DIR, "data", "petase_challenge_data",
                        "predictive-pet-zero-shot-test-2025.csv")
STRUCTURES_DIR = os.path.join(BASE_DIR, "results", "structures")
PLDDT_CSV = os.path.join(STRUCTURES_DIR, "plddt_summary.csv")
PLDDT_TEST_CSV = os.path.join(STRUCTURES_DIR, "plddt_summary_test.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="ESMFold structure prediction for PETases")
    parser.add_argument("--mode", choices=["wt", "test", "all"], default="wt",
                        help="wt=313 WTs only, test=4988 test seqs, all=both (default: wt)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--max-seqs", type=int, default=0,
                        help="Max sequences to process per mode (0 = all)")
    return parser.parse_args()


def load_esmfold(device="auto"):
    """Load ESMFold model via HuggingFace transformers (no openfold needed)."""
    from transformers import AutoTokenizer, EsmForProteinFolding

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading ESMFold-v1 (HuggingFace transformers)...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1", low_cpu_mem_usage=True
    )
    model.requires_grad_(False)
    model = model.eval()

    if device == "cuda":
        try:
            model = model.cuda()
            # Use FP16 for the language model trunk to save VRAM
            model.esm = model.esm.half()
            print("ESMFold loaded on GPU (ESM trunk in FP16)")
        except RuntimeError as e:
            print("GPU failed (%s), using CPU" % e)
            model = model.cpu()
            device = "cpu"
    else:
        print("ESMFold on CPU (will be slow)")

    return model, tokenizer, device


def convert_outputs_to_pdb(output):
    """Convert ESMFold model outputs to PDB string(s)."""
    from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
    from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

    # Convert atom14 positions to atom37 format
    final_atom_positions = atom14_to_atom37(output["positions"][-1], output)
    final_atom_positions = final_atom_positions.cpu().numpy()

    output_np = {}
    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            output_np[k] = v.cpu().numpy()

    final_atom_mask = output_np["atom37_atom_exists"]

    pdbs = []
    for i in range(final_atom_positions.shape[0]):
        pred = OFProtein(
            aatype=output_np["aatype"][i],
            atom_positions=final_atom_positions[i],
            atom_mask=final_atom_mask[i],
            residue_index=output_np["residue_index"][i] + 1,
            b_factors=output_np["plddt"][i] * 100,  # Scale 0-1 to 0-100
        )
        pdbs.append(to_pdb(pred))
    return pdbs


def predict_structure(model, tokenizer, sequence, device):
    """
    Predict structure for a single sequence.
    Returns PDB string and per-residue pLDDT scores (0-100 scale).
    """
    tokenized = tokenizer(
        [sequence], return_tensors="pt", add_special_tokens=False
    )
    tokenized = {k: v.to(device) for k, v in tokenized.items()}

    with torch.no_grad():
        output = model(**tokenized)

    # Extract pLDDT (0-1 scale from model, convert to 0-100)
    plddt_01 = output["plddt"][0, :len(sequence)].cpu().numpy()
    plddt_100 = plddt_01 * 100.0

    # Convert to PDB string
    pdb_strings = convert_outputs_to_pdb(output)
    pdb_str = pdb_strings[0]

    return pdb_str, plddt_100


def read_plddt_from_pdb(pdb_path):
    """Read per-residue pLDDT scores from B-factor column of CA atoms."""
    plddt_scores = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                plddt_scores.append(float(line[60:66].strip()))
    return np.array(plddt_scores) if plddt_scores else np.array([0.0])


def predict_batch(model, tokenizer, device, sequences, idx_list, prefix,
                  plddt_csv_path, idx_col_name, max_seqs=0):
    """
    Predict structures for a list of sequences.

    Args:
        sequences: list of AA sequences
        idx_list: list of indices (for naming/logging)
        prefix: filename prefix ('wt' or 'test')
        plddt_csv_path: path to save pLDDT summary CSV
        idx_col_name: column name for the index ('wt_idx' or 'test_idx')
        max_seqs: limit number of sequences (0 = all)
    """
    t0 = time.time()

    if max_seqs > 0:
        sequences = sequences[:max_seqs]
        idx_list = idx_list[:max_seqs]
        print("Processing first %d sequences" % len(sequences))

    os.makedirs(STRUCTURES_DIR, exist_ok=True)

    # Use 4-digit zero-padding for test (up to 9999), 3-digit for wt (up to 999)
    pad = 4 if prefix == "test" else 3

    plddt_rows = []
    n_total = len(sequences)
    for pos, (idx, seq) in enumerate(zip(idx_list, sequences)):
        pdb_name = "%s_%0*d.pdb" % (prefix, pad, idx)
        pdb_path = os.path.join(STRUCTURES_DIR, pdb_name)

        # Skip if already predicted
        if os.path.exists(pdb_path):
            if (pos + 1) % 500 == 0 or pos == 0:
                print("  [%d/%d] %s exists, skipping" % (pos + 1, n_total, pdb_name))
            plddt_arr = read_plddt_from_pdb(pdb_path)
            plddt_rows.append({
                idx_col_name: idx,
                "seq_len": len(seq),
                "mean_plddt": float(np.mean(plddt_arr)),
                "min_plddt": float(np.min(plddt_arr)),
                "pdb_file": pdb_name,
            })
            continue

        pdb_str, plddt = predict_structure(model, tokenizer, seq, device)

        with open(pdb_path, "w") as f:
            f.write(pdb_str)

        mean_plddt = float(np.mean(plddt)) if len(plddt) > 0 else 0.0
        min_plddt = float(np.min(plddt)) if len(plddt) > 0 else 0.0

        plddt_rows.append({
            idx_col_name: idx,
            "seq_len": len(seq),
            "mean_plddt": mean_plddt,
            "min_plddt": min_plddt,
            "pdb_file": pdb_name,
        })

        elapsed = time.time() - t0
        rate = (pos + 1) / elapsed if elapsed > 0 else 0
        remaining = (n_total - pos - 1) / rate if rate > 0 else 0

        if (pos + 1) % 50 == 0 or pos == 0:
            print("  [%d/%d] %s len=%d pLDDT=%.1f (%ds elapsed, ~%ds remaining)" % (
                pos + 1, n_total, pdb_name, len(seq), mean_plddt,
                int(elapsed), int(remaining)))

        # Free GPU memory periodically
        if device == "cuda" and (pos + 1) % 50 == 0:
            torch.cuda.empty_cache()

    # Save pLDDT summary
    plddt_df = pd.DataFrame(plddt_rows)
    plddt_df.to_csv(plddt_csv_path, index=False)

    total_time = time.time() - t0
    print("\nDone! %d %s structures in %ds (%.1f min)" % (
        n_total, prefix, int(total_time), total_time / 60))
    print("pLDDT summary: %s" % plddt_csv_path)

    # Summary stats
    print("\n=== pLDDT Summary (%s) ===" % prefix)
    print("Mean pLDDT: %.1f (min=%.1f, max=%.1f)" % (
        plddt_df["mean_plddt"].mean(),
        plddt_df["mean_plddt"].min(),
        plddt_df["mean_plddt"].max()))
    low_conf = (plddt_df["mean_plddt"] < 70).sum()
    if low_conf > 0:
        print("WARNING: %d structures with mean pLDDT < 70" % low_conf)


def main():
    args = parse_args()

    do_wt = args.mode in ("wt", "all")
    do_test = args.mode in ("test", "all")

    # Load model
    device_pref = "cpu" if args.cpu else "auto"
    model, tokenizer, device = load_esmfold(device_pref)

    # WT structures
    if do_wt:
        print("=" * 60)
        print("Predicting WT structures")
        print("=" * 60)
        wt_df = pd.read_csv(WT_CSV)
        wt_seqs = list(wt_df["Wt AA Sequence"].values)
        print("Loaded %d WT sequences" % len(wt_seqs))
        predict_batch(
            model, tokenizer, device,
            sequences=wt_seqs,
            idx_list=list(range(len(wt_seqs))),
            prefix="wt",
            plddt_csv_path=PLDDT_CSV,
            idx_col_name="wt_idx",
            max_seqs=args.max_seqs,
        )

    # Test structures
    if do_test:
        print("=" * 60)
        print("Predicting test-set structures (all 4988 sequences)")
        print("=" * 60)
        test_df = pd.read_csv(TEST_CSV)
        test_seqs = list(test_df["sequence"].values)
        print("Loaded %d test sequences" % len(test_seqs))
        predict_batch(
            model, tokenizer, device,
            sequences=test_seqs,
            idx_list=list(range(len(test_seqs))),
            prefix="test",
            plddt_csv_path=PLDDT_TEST_CSV,
            idx_col_name="test_idx",
            max_seqs=args.max_seqs,
        )


if __name__ == "__main__":
    main()
