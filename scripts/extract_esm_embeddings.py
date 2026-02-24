#!/usr/bin/env python3
"""
Extract ESM2 embeddings from PETase sequences.
Uses ESM2-650M (esm2_t33_650M_UR50D, 1280-dim embeddings).

Updated from ESM-1b to ESM2 for consistency with zero-shot scoring pipeline.

Requires: torch, fair-esm
Run: python scripts/extract_esm_embeddings.py
"""

import sys
import csv
from pathlib import Path
import numpy as np

try:
    import torch
    import importlib
    esm = importlib.import_module("esm")
    if not hasattr(esm, "pretrained"):
        # EvolutionaryScale 'esm' overwrote fair-esm — reinstall
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "fair-esm"],
        )
        for key in list(sys.modules):
            if key == "esm" or key.startswith("esm."):
                del sys.modules[key]
        esm = importlib.import_module("esm")
except ImportError:
    print("ERROR: This script requires torch and fair-esm.")
    print("Install: pip install fair-esm torch")
    sys.exit(1)

# Import WT sequence and mutation parser
sys.path.insert(0, str(Path(__file__).parent))
from feature_extraction import WT_SEQUENCE, apply_mutation, parse_mutation_positions

# Load ESM2 model (cached after first download)
print("Loading ESM2-650M model...")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()

# Set model to inference mode
for param in model.parameters():
    param.requires_grad = False
model.eval()

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = model.to(device)


def get_embedding(sequence, use_mean_pool=True):
    """Extract ESM embedding for a single sequence.

    Args:
        sequence: Protein sequence (1-letter codes)
        use_mean_pool: If True, return mean-pooled per-protein embedding (1280-dim)
                      If False, return per-residue embeddings (seq_len x 1280)

    Returns:
        Embedding vector
    """
    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
        token_representations = results["representations"][33]

    if use_mean_pool:
        # Mean pool over sequence (excluding <cls> and <eos> tokens)
        seq_len = batch_tokens.shape[1]
        token_representations = token_representations[:, 1:seq_len-1, :]
        embedding = token_representations.mean(dim=1)
        return embedding.cpu().numpy()[0]  # Shape: (1280,)
    else:
        seq_len = batch_tokens.shape[1]
        return token_representations[:, 1:seq_len-1, :].cpu().numpy()[0]


def get_delta_embedding(wt_sequence, mutant_sequence):
    """Get difference embedding (mutant - wild-type).

    Returns:
        Difference vector (1280-dim)
    """
    wt_emb = get_embedding(wt_sequence, use_mean_pool=True)
    mut_emb = get_embedding(mutant_sequence, use_mean_pool=True)
    return mut_emb - wt_emb


def extract_all_embeddings(mutations_csv, output_dir):
    """Extract ESM embeddings for all IsPETase variants.

    Args:
        mutations_csv: Path to mutations_dataset.csv
        output_dir: Directory to save embeddings

    Saves:
        esm_embeddings.npy: (n_variants, 1280) absolute embeddings
        esm_delta_embeddings.npy: (n_variants, 1280) difference from WT
        esm_variant_names.txt: variant names in same order
    """
    # Load mutations
    with open(mutations_csv, 'r') as f:
        mutations = list(csv.DictReader(f))

    # Filter to IsPETase only
    is_petase = [m for m in mutations if m.get('enzyme', 'IsPETase') == 'IsPETase']
    print(f"Processing {len(is_petase)} IsPETase variants...")

    # Get WT embedding first
    wt_emb = get_embedding(WT_SEQUENCE)
    print(f"WT embedding shape: {wt_emb.shape}, norm: {np.linalg.norm(wt_emb):.2f}")

    embeddings = []
    delta_embeddings = []
    variant_names = []
    tms = []

    for mut in is_petase:
        name = mut['variant_name']
        mutation_str = mut['mutation']

        # Get variant sequence
        variant_seq = apply_mutation(WT_SEQUENCE, mutation_str)
        if variant_seq is None:
            print(f"  Skipping {name}: mutation application failed")
            continue

        print(f"  Extracting: {name} ({mutation_str})...", end=" ")
        emb = get_embedding(variant_seq)
        delta = emb - wt_emb

        embeddings.append(emb)
        delta_embeddings.append(delta)
        variant_names.append(name)
        tms.append(float(mut['tm']))

        print(f"norm={np.linalg.norm(emb):.2f}, delta_norm={np.linalg.norm(delta):.4f}")

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    emb_path = output_dir / 'esm2_embeddings.npy'
    delta_path = output_dir / 'esm2_delta_embeddings.npy'
    names_path = output_dir / 'esm2_variant_names.txt'
    tms_path = output_dir / 'esm2_tms.npy'

    np.save(emb_path, np.array(embeddings))
    np.save(delta_path, np.array(delta_embeddings))
    np.save(tms_path, np.array(tms))

    with open(names_path, 'w') as f:
        f.write('\n'.join(variant_names))

    print(f"\nSaved {len(embeddings)} embeddings to {output_dir}/")
    print(f"  {emb_path.name}: {np.array(embeddings).shape}")
    print(f"  {delta_path.name}: {np.array(delta_embeddings).shape}")
    print(f"  {names_path.name}: {len(variant_names)} names")
    print(f"  {tms_path.name}: {len(tms)} Tm values")


if __name__ == '__main__':
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / 'data'
    mutations_csv = data_dir / 'mutations_dataset.csv'
    output_dir = data_dir

    extract_all_embeddings(mutations_csv, output_dir)
