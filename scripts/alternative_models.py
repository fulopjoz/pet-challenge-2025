#!/usr/bin/env python3
"""
Alternative models for small-dataset PETase Tm prediction.
Compares: Ridge, Lasso, ElasticNet, Feature-selected Ridge, XGBoost
Saves results to results/model_comparison.csv
"""

import csv
from pathlib import Path
import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: xgboost not available, skipping XGBoost model")


def load_features(input_path):
    """Load features and target from CSV"""
    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        feature_names = header[1:-1]
        rows = list(reader)

    X = np.array([[float(val) for val in row[1:-1]] for row in rows])
    y = np.array([float(row[-1]) for row in rows])
    variant_names = [row[0] for row in rows]

    print(f"Loaded {len(y)} samples, {X.shape[1]} features")
    print(f"Target range: {y.min():.1f} - {y.max():.1f} C")
    return X, y, feature_names, variant_names


def aggregate_duplicate_feature_rows(X, y, variant_names):
    """Collapse exact duplicate feature rows by averaging targets."""
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


def evaluate_model(name, model, X, y):
    """Evaluate model with LOOCV, return metrics dict"""
    loo = LeaveOneOut()
    y_pred = cross_val_predict(model, X, y, cv=loo)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    mae = np.mean(np.abs(y - y_pred))

    spearman_rho = 0.0
    if len(y) > 2:
        spearman_rho, _ = spearmanr(y, y_pred)

    # Train on full data for additional info
    model.fit(X, y)
    y_train_pred = model.predict(X)
    train_rmse = np.sqrt(mean_squared_error(y, y_train_pred))
    train_r2 = r2_score(y, y_train_pred)

    return {
        'name': name,
        'loocv_rmse': rmse,
        'loocv_r2': r2,
        'loocv_mae': mae,
        'loocv_spearman': spearman_rho,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
    }


def main():
    # Paths
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    feature_path = project_dir / 'data' / 'features_matrix.csv'
    output_dir = project_dir / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, feature_names, variant_names = load_features(feature_path)
    X, y, variant_names = aggregate_duplicate_feature_rows(X, y, variant_names)
    n = len(y)
    target_range = y.max() - y.min()

    if n < 15:
        print(f"\nWARNING: Small dataset (n={n}). Regularized linear models recommended.")

    results = []

    print("\n" + "=" * 60)
    print("ALTERNATIVE MODELS FOR PETase Tm PREDICTION")
    print("=" * 60)

    # 1. Ridge Regression
    print("\n1. RIDGE REGRESSION (alpha=1.0)")
    print("-" * 40)
    r = evaluate_model('Ridge (alpha=1)', make_pipeline(StandardScaler(), Ridge(alpha=1.0)), X, y)
    results.append(r)
    print(f"  LOOCV RMSE: {r['loocv_rmse']:.2f} C  R2: {r['loocv_r2']:.3f}")

    # 2. Ridge with higher regularization
    print("\n2. RIDGE REGRESSION (alpha=10)")
    print("-" * 40)
    r = evaluate_model('Ridge (alpha=10)', make_pipeline(StandardScaler(), Ridge(alpha=10.0)), X, y)
    results.append(r)
    print(f"  LOOCV RMSE: {r['loocv_rmse']:.2f} C  R2: {r['loocv_r2']:.3f}")

    # 3. Lasso
    print("\n3. LASSO REGRESSION (alpha=0.1)")
    print("-" * 40)
    lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.1, max_iter=10000))
    r = evaluate_model('Lasso (alpha=0.1)', lasso, X, y)
    results.append(r)
    lasso.fit(X, y)
    n_features_used = np.sum(lasso.named_steps['lasso'].coef_ != 0)
    print(f"  LOOCV RMSE: {r['loocv_rmse']:.2f} C  R2: {r['loocv_r2']:.3f}")
    print(f"  Features used: {n_features_used}/{X.shape[1]}")

    # 4. Elastic Net
    print("\n4. ELASTIC NET (alpha=0.1, l1_ratio=0.5)")
    print("-" * 40)
    r = evaluate_model(
        'ElasticNet',
        make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)),
        X, y
    )
    results.append(r)
    print(f"  LOOCV RMSE: {r['loocv_rmse']:.2f} C  R2: {r['loocv_r2']:.3f}")

    # 5. Feature selection + Ridge
    k = min(10, X.shape[1], n - 1)
    print(f"\n5. FEATURE SELECTION + RIDGE (Top {k})")
    print("-" * 40)
    fs_ridge = Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(score_func=f_regression, k=k)),
        ('ridge', Ridge(alpha=1.0)),
    ])
    r = evaluate_model(f'Ridge+Top{k}', fs_ridge, X, y)
    results.append(r)
    print(f"  LOOCV RMSE: {r['loocv_rmse']:.2f} C  R2: {r['loocv_r2']:.3f}")

    # Show selected features
    selector = SelectKBest(score_func=f_regression, k=k)
    X_scaled = StandardScaler().fit_transform(X)
    selector.fit(X_scaled, y)
    mask = selector.get_support()
    selected_features = [f for f, m in zip(feature_names, mask) if m]
    print(f"  Selected features: {', '.join(selected_features)}")

    # 6. XGBoost (if available and n >= 10)
    if HAS_XGBOOST and n >= 10:
        print("\n6. XGBOOST")
        print("-" * 40)
        xgb_model = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0
        )
        r = evaluate_model('XGBoost', xgb_model, X, y)
        results.append(r)
        print(f"  LOOCV RMSE: {r['loocv_rmse']:.2f} C  R2: {r['loocv_r2']:.3f}")
    elif HAS_XGBOOST:
        print(f"\n6. [SKIPPED] XGBoost: n={n} < 10, too few for tree ensemble")

    # Summary table
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<25} {'LOOCV RMSE':>12} {'LOOCV R2':>10} {'Spearman':>10} {'RMSE/Range':>12}")
    print("-" * 71)

    best = min(results, key=lambda r: r['loocv_rmse'])
    for r in sorted(results, key=lambda r: r['loocv_rmse']):
        marker = " <-- BEST" if r['name'] == best['name'] else ""
        print(f"  {r['name']:<23} {r['loocv_rmse']:>10.2f} C {r['loocv_r2']:>10.3f} "
              f"{r['loocv_spearman']:>10.3f} {r['loocv_rmse']/target_range*100:>10.1f}%{marker}")

    print(f"\nBest model: {best['name']} (LOOCV RMSE = {best['loocv_rmse']:.2f} C)")

    # Save results to CSV
    results_path = output_dir / 'model_comparison.csv'
    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'name', 'loocv_rmse', 'loocv_r2', 'loocv_mae',
            'loocv_spearman', 'train_rmse', 'train_r2'
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved model comparison to {results_path}")


if __name__ == '__main__':
    main()
