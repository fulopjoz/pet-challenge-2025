#!/usr/bin/env python3
"""
UMAP visualization of ESM2 embeddings for 313 WT PETase scaffolds.

Reduces 1280-dim ESM2 mean-pooled embeddings to 2D using UMAP,
creating a scatter plot colored by sequence length to reveal
scaffold clustering and diversity.

Usage:
    python scripts/visualize_embeddings.py

Requires: pip install umap-learn matplotlib
Input:  results/esm2_embeddings.npz (from esm2_zero_shot_scoring.py)
        data/petase_challenge_data/pet-2025-wildtype-cds.csv (for sequence lengths)
Output: results/embedding_umap.png
"""

import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
EMBEDDINGS_NPZ = os.path.join(BASE_DIR, "results", "esm2_embeddings.npz")
WT_CSV = os.path.join(BASE_DIR, "data", "petase_challenge_data", "pet-2025-wildtype-cds.csv")
OUTPUT_PNG = os.path.join(BASE_DIR, "results", "embedding_umap.png")


def main():
    print("=" * 60)
    print("UMAP Visualization of ESM2 WT Embeddings")
    print("=" * 60)

    # Load embeddings
    if not os.path.exists(EMBEDDINGS_NPZ):
        print("ERROR: %s not found. Run esm2_zero_shot_scoring.py first." % EMBEDDINGS_NPZ)
        sys.exit(1)

    data = np.load(EMBEDDINGS_NPZ)
    wt_embs = data["wt_embeddings"]  # (313, 1280)
    print("Loaded WT embeddings: %s" % str(wt_embs.shape))

    # Filter out zero-rows (WTs that weren't scored)
    nonzero_mask = np.any(wt_embs != 0, axis=1)
    valid_indices = np.where(nonzero_mask)[0]
    wt_embs_valid = wt_embs[nonzero_mask]
    print("Valid (non-zero) embeddings: %d / %d" % (len(wt_embs_valid), len(wt_embs)))

    # Load WT metadata for coloring
    wt_df = pd.read_csv(WT_CSV)
    wt_seqs = list(wt_df["Wt AA Sequence"].values)
    seq_lengths = np.array([len(s) for s in wt_seqs])
    valid_lengths = seq_lengths[nonzero_mask]

    # UMAP reduction
    try:
        import umap
    except ImportError:
        print("ERROR: umap-learn not installed. Run: pip install umap-learn")
        sys.exit(1)

    print("\nRunning UMAP (n_neighbors=15, min_dist=0.1)...")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="cosine",
        random_state=42,
    )
    embedding_2d = reducer.fit_transform(wt_embs_valid)
    print("UMAP complete: %s" % str(embedding_2d.shape))

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    scatter = ax.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=valid_lengths,
        cmap="viridis",
        s=20,
        alpha=0.7,
        edgecolors="none",
    )

    cbar = plt.colorbar(scatter, ax=ax, label="Sequence Length (aa)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("ESM2 Embeddings: 313 WT PETase Scaffolds\n(colored by sequence length)")

    # Annotate clusters
    length_bins = [(200, 250, "short"), (250, 300, "medium"), (300, 400, "long")]
    for lo, hi, label in length_bins:
        mask = (valid_lengths >= lo) & (valid_lengths < hi)
        if mask.sum() > 0:
            cx = embedding_2d[mask, 0].mean()
            cy = embedding_2d[mask, 1].mean()
            ax.annotate(
                "%s (%d-%d aa, n=%d)" % (label, lo, hi, mask.sum()),
                (cx, cy),
                fontsize=8,
                fontweight="bold",
                ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print("\nSaved UMAP plot to %s" % OUTPUT_PNG)

    # Print length distribution
    print("\n=== Sequence Length Distribution ===")
    for lo, hi, label in length_bins:
        mask = (valid_lengths >= lo) & (valid_lengths < hi)
        print("  %s (%d-%d aa): %d scaffolds" % (label, lo, hi, mask.sum()))
    remaining = (valid_lengths >= 400).sum()
    if remaining > 0:
        print("  very long (400+ aa): %d scaffolds" % remaining)


if __name__ == "__main__":
    main()
