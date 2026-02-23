#!/usr/bin/env python3
"""
Train baseline models for PETase Tm prediction.
Primary: Ridge regression (better for small n)
Secondary: Random Forest (comparison, needs n >= 10)
"""

import csv
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import joblib


def load_features(input_path: str):
    """Load features and target from CSV"""
    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        feature_names = header[1:-1]  # Skip variant_name (first) and Tm (last)
        rows = list(reader)

    variant_names = []
    features_list = []
    tm_values = []

    for row in rows:
        variant_names.append(row[0])
        features = [float(val) for val in row[1:-1]]
        tm = float(row[-1])
        features_list.append(features)
        tm_values.append(tm)

    X = np.array(features_list)
    y = np.array(tm_values)

    print(f"Loaded {len(y)} samples, {len(feature_names)} features")
    return X, y, feature_names, variant_names


def aggregate_duplicate_feature_rows(X, y, variant_names):
    """
    Collapse exact duplicate feature rows by averaging targets.

    Duplicate sequence-level features with different Tm values can bias LOOCV.
    """
    groups = {}
    for i, row in enumerate(X):
        key = tuple(float(v) for v in row.tolist())
        groups.setdefault(key, []).append(i)

    if all(len(idxs) == 1 for idxs in groups.values()):
        return X, y, variant_names

    print("\nDetected duplicate feature vectors; aggregating by mean Tm:")
    X_new, y_new, names_new = [], [], []
    for key, idxs in groups.items():
        X_new.append(list(key))
        if len(idxs) == 1:
            y_new.append(float(y[idxs[0]]))
            names_new.append(variant_names[idxs[0]])
        else:
            tm_vals = [float(y[i]) for i in idxs]
            merged_name = " / ".join(variant_names[i] for i in idxs)
            print(f"  - {merged_name} -> mean Tm {np.mean(tm_vals):.2f} C (n={len(idxs)})")
            y_new.append(float(np.mean(tm_vals)))
            names_new.append(merged_name)

    X_arr = np.array(X_new, dtype=float)
    y_arr = np.array(y_new, dtype=float)
    print(f"After aggregation: {len(y_arr)} unique samples (from {len(y)})")
    return X_arr, y_arr, names_new


def train_ridge_primary(X, y, feature_names, variant_names):
    """Train Ridge regression (primary model for small datasets)"""
    print("\n" + "=" * 60)
    print("RIDGE REGRESSION (Primary Model)")
    print("=" * 60)

    target_range = y.max() - y.min()
    print(f"Target range: {y.min():.1f} - {y.max():.1f} C ({target_range:.1f} C span)")

    # LOOCV (gold standard for small n)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    loo = LeaveOneOut()

    # Get LOOCV predictions for each sample
    y_pred_loo = cross_val_predict(ridge, X, y, cv=loo)

    rmse_loo = np.sqrt(mean_squared_error(y, y_pred_loo))
    mae_loo = mean_absolute_error(y, y_pred_loo)
    r2_loo = r2_score(y, y_pred_loo)

    print(f"\nLOOCV Metrics:")
    print(f"  RMSE: {rmse_loo:.2f} C")
    print(f"  MAE:  {mae_loo:.2f} C")
    print(f"  R2:   {r2_loo:.3f}")
    print(f"  RMSE / range: {rmse_loo/target_range*100:.1f}%")

    if len(y) > 2:
        spearman, pval = spearmanr(y, y_pred_loo)
        print(f"  Spearman rho: {spearman:.3f} (p={pval:.4f})")

    # Train final model on all data
    ridge.fit(X, y)
    y_pred_train = ridge.predict(X)
    rmse_train = np.sqrt(mean_squared_error(y, y_pred_train))
    r2_train = r2_score(y, y_pred_train)

    print(f"\nTrain (full data) Metrics:")
    print(f"  RMSE: {rmse_train:.2f} C")
    print(f"  R2:   {r2_train:.3f}")

    # Per-sample LOOCV predictions
    print(f"\nPer-sample LOOCV predictions:")
    for i, name in enumerate(variant_names):
        error = y_pred_loo[i] - y[i]
        print(f"  {name:25s} actual={y[i]:5.1f}  pred={y_pred_loo[i]:5.1f}  "
              f"error={error:+.1f} C")

    # Feature coefficients
    print(f"\nTop Ridge Coefficients (absolute):")
    ridge_model = ridge.named_steps['ridge']
    coef_abs = np.abs(ridge_model.coef_)
    indices = np.argsort(coef_abs)[::-1]
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print(f"  {i+1:2d}. {feature_names[idx]:25s} coef={ridge_model.coef_[idx]:+.4f}")

    return ridge, rmse_loo, r2_loo


def train_rf_comparison(X, y, feature_names, variant_names):
    """Train Random Forest (comparison model, needs n >= 10)"""
    n = len(y)

    if n < 10:
        print(f"\n[RF] Skipping Random Forest: n={n} < 10 (too few for tree ensemble)")
        return None, None, None

    print("\n" + "=" * 60)
    print("RANDOM FOREST (Comparison Model)")
    print("=" * 60)

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,  # Limited depth for small n
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    # LOOCV
    loo = LeaveOneOut()
    y_pred_loo = cross_val_predict(rf, X, y, cv=loo)

    rmse_loo = np.sqrt(mean_squared_error(y, y_pred_loo))
    r2_loo = r2_score(y, y_pred_loo)

    print(f"\nLOOCV Metrics:")
    print(f"  RMSE: {rmse_loo:.2f} C")
    print(f"  R2:   {r2_loo:.3f}")

    # Train final model on all data for feature importance
    rf.fit(X, y)

    # Feature importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print(f"\nTop RF Feature Importances:")
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print(f"  {i+1:2d}. {feature_names[idx]:25s} {importances[idx]:.4f}")

    return rf, rmse_loo, r2_loo


def save_feature_importance(rf, feature_names, output_path):
    """Save RF feature importance to CSV"""
    if rf is None:
        return
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['feature', 'importance', 'rank'])
        for rank, idx in enumerate(indices, 1):
            writer.writerow([feature_names[idx], importances[idx], rank])

    print(f"Saved feature importance to {output_path}")


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("PETASE Tm PREDICTION - BASELINE MODELS")
    print("=" * 60)

    # Paths
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    feature_path = project_dir / 'data' / 'features_matrix.csv'
    output_dir = project_dir / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLOADING DATA")
    X, y, feature_names, variant_names = load_features(feature_path)
    X, y, variant_names = aggregate_duplicate_feature_rows(X, y, variant_names)

    n = len(y)
    if n < 15:
        print(f"\nWARNING: Small dataset (n={n}). Results should be interpreted "
              f"cautiously. Ridge with LOOCV is recommended over RF.")

    # Primary: Ridge regression
    ridge, ridge_rmse, ridge_r2 = train_ridge_primary(X, y, feature_names, variant_names)

    # Save Ridge model
    ridge_path = output_dir / 'ridge_baseline_model.pkl'
    joblib.dump(ridge, ridge_path)
    print(f"\nSaved Ridge model to {ridge_path}")

    # Comparison: Random Forest
    rf, rf_rmse, rf_r2 = train_rf_comparison(X, y, feature_names, variant_names)

    if rf is not None:
        rf_path = output_dir / 'rf_baseline_model.pkl'
        joblib.dump(rf, rf_path)
        print(f"Saved RF model to {rf_path}")

        importance_path = output_dir / 'rf_feature_importance.csv'
        save_feature_importance(rf, feature_names, importance_path)

    # Summary
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    target_range = y.max() - y.min()
    print(f"Dataset: {n} samples, {len(feature_names)} features")
    print(f"Target range: {y.min():.1f} - {y.max():.1f} C ({target_range:.1f} C)")
    print(f"\n{'Model':<20} {'LOOCV RMSE':>12} {'LOOCV R2':>10} {'RMSE/Range':>12}")
    print("-" * 56)
    print(f"{'Ridge (alpha=1)':<20} {ridge_rmse:>10.2f} C {ridge_r2:>10.3f} {ridge_rmse/target_range*100:>10.1f}%")
    if rf is not None:
        print(f"{'Random Forest':<20} {rf_rmse:>10.2f} C {rf_r2:>10.3f} {rf_rmse/target_range*100:>10.1f}%")

    quality = "good" if ridge_rmse / target_range < 0.15 else \
              "acceptable" if ridge_rmse / target_range < 0.30 else "poor"
    print(f"\nModel quality: {quality} (RMSE/range = {ridge_rmse/target_range*100:.1f}%)")
    print("  <15%: good, 15-30%: acceptable, >30%: poor")

    print("\nBASELINE TRAINING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
