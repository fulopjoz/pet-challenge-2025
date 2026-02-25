#!/usr/bin/env python3
"""
Train model on ESM embeddings for PETase Tm prediction.

Expects embeddings from extract_esm_embeddings.py:
  data/esm2_embeddings.npy        - absolute embeddings (n, 1280)
  data/esm2_delta_embeddings.npy  - delta from WT (n, 1280)
  data/esm2_tms.npy               - Tm values (n,)
  data/esm2_variant_names.txt     - variant names

Run: python scripts/train_esm_model.py
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
import joblib


def load_esm_data(data_dir):
    """Load ESM embeddings, delta embeddings, Tm values, and variant names."""
    data_dir = Path(data_dir)

    emb = np.load(data_dir / 'esm2_embeddings.npy')
    delta = np.load(data_dir / 'esm2_delta_embeddings.npy')
    tms = np.load(data_dir / 'esm2_tms.npy')

    with open(data_dir / 'esm2_variant_names.txt', 'r') as f:
        names = [line.strip() for line in f if line.strip()]

    print(f"Loaded ESM data: {len(names)} variants, {emb.shape[1]}-dim embeddings")
    print(f"Tm range: {tms.min():.1f} - {tms.max():.1f} C")
    return emb, delta, tms, names


def train_and_evaluate(name, X, y, variant_names, alpha=1.0):
    """Train Ridge on embeddings with LOOCV evaluation."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Samples: {len(y)}, Features: {X.shape[1]}")

    model = Ridge(alpha=alpha)
    loo = LeaveOneOut()

    # LOOCV predictions
    y_pred = cross_val_predict(model, X, y, cv=loo)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    target_range = y.max() - y.min()
    print(f"\nLOOCV Results:")
    print(f"  RMSE: {rmse:.2f} C")
    print(f"  R2:   {r2:.3f}")
    print(f"  RMSE/range: {rmse/target_range*100:.1f}%")

    if len(y) > 2:
        rho, pval = spearmanr(y, y_pred)
        print(f"  Spearman rho: {rho:.3f} (p={pval:.4f})")

    # Per-sample predictions
    print(f"\nPer-sample predictions:")
    for i, vname in enumerate(variant_names):
        error = y_pred[i] - y[i]
        print(f"  {vname:25s} actual={y[i]:5.1f}  pred={y_pred[i]:5.1f}  error={error:+.1f}")

    # Train final model
    model.fit(X, y)
    return model, rmse, r2


def main():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / 'data'
    output_dir = script_dir.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if embeddings exist
    if not (data_dir / 'esm2_embeddings.npy').exists():
        print("ERROR: ESM embeddings not found.")
        print("Run extract_esm_embeddings.py first to generate embeddings.")
        sys.exit(1)

    emb, delta, tms, names = load_esm_data(data_dir)

    n = len(tms)
    if n < 20:
        print(f"\nWARNING: Only n={n} samples. ESM embeddings (1280-dim) may overfit.")
        print("Consider using delta embeddings + high regularization.")

    # 1. Ridge on absolute embeddings
    model_abs, rmse_abs, r2_abs = train_and_evaluate(
        "ESM Absolute Embeddings + Ridge (alpha=10)",
        emb, tms, names, alpha=10.0
    )

    # 2. Ridge on delta embeddings (recommended for small n)
    model_delta, rmse_delta, r2_delta = train_and_evaluate(
        "ESM Delta Embeddings + Ridge (alpha=10)",
        delta, tms, names, alpha=10.0
    )

    # 3. ElasticNet on delta embeddings (feature selection)
    print(f"\n{'='*60}")
    print("ESM Delta + ElasticNet (alpha=0.01, l1_ratio=0.5)")
    print(f"{'='*60}")
    enet = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)
    loo = LeaveOneOut()
    y_pred = cross_val_predict(enet, delta, tms, cv=loo)
    rmse_enet = np.sqrt(mean_squared_error(tms, y_pred))
    r2_enet = r2_score(tms, y_pred)
    enet.fit(delta, tms)
    n_used = np.sum(enet.coef_ != 0)
    print(f"  LOOCV RMSE: {rmse_enet:.2f} C  R2: {r2_enet:.3f}")
    print(f"  ESM dimensions used: {n_used}/{delta.shape[1]}")

    # Summary
    print(f"\n{'='*60}")
    print("ESM MODEL COMPARISON")
    print(f"{'='*60}")
    target_range = tms.max() - tms.min()
    print(f"\n{'Method':<45} {'RMSE':>8} {'R2':>8}")
    print("-" * 65)
    print(f"{'Absolute + Ridge(10)':<45} {rmse_abs:>6.2f} C {r2_abs:>8.3f}")
    print(f"{'Delta + Ridge(10)':<45} {rmse_delta:>6.2f} C {r2_delta:>8.3f}")
    print(f"{'Delta + ElasticNet':<45} {rmse_enet:>6.2f} C {r2_enet:>8.3f}")

    # Save best model
    if rmse_delta <= rmse_abs:
        best_model = model_delta
        best_name = "delta_ridge"
    else:
        best_model = model_abs
        best_name = "abs_ridge"

    model_path = output_dir / f'esm_{best_name}_model.pkl'
    joblib.dump(best_model, model_path)
    print(f"\nSaved best ESM model to {model_path}")


if __name__ == '__main__':
    main()
