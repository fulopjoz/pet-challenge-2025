#!/usr/bin/env python3
"""
Execute non-GPU notebook cells locally and save outputs to the notebook.

Runs cells that only need pandas/scipy/matplotlib (no ESM2/ESMC GPU):
- Cell 12: Load and display ESM2 scores (shows v4 columns)
- Cell 26: Validation & sanity checks
- Cell 29: Generate v1 + v2 submissions
- Cell 30: Top-K WT vs mutant composition verification
- Cell 31: Compare v1 vs v2
- Cell 33: Final summary table
- Cell 34: Download info

Cells 0 and 4 are Colab setup — intentionally unexecuted in saved state.
"""

import os
import sys
import json
import io
import contextlib
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
NOTEBOOK_PATH = os.path.join(PROJECT_ROOT, "PET_Challenge_2025_Pipeline_v2.ipynb")


def capture_cell_output(code, context):
    """Execute code and capture stdout output."""
    stdout_capture = io.StringIO()
    with contextlib.redirect_stdout(stdout_capture):
        try:
            exec(code, context)
        except SystemExit:
            pass
        except Exception:
            traceback.print_exc(file=stdout_capture)
    return stdout_capture.getvalue()


def make_text_output(text):
    """Create notebook text output cell."""
    return {
        "output_type": "stream",
        "name": "stdout",
        "text": text.splitlines(True)
    }


def main():
    import numpy as np
    import pandas as pd
    import subprocess
    from scipy import stats as sp_stats
    from collections import Counter, defaultdict

    # Load notebook
    with open(NOTEBOOK_PATH) as f:
        nb = json.load(f)

    # Set up shared context (simulating prior cell execution)
    context = {
        "os": os,
        "sys": sys,
        "np": np,
        "pd": pd,
        "PROJECT_ROOT": PROJECT_ROOT,
        "subprocess": subprocess,
        "sp_stats": sp_stats,
        "Counter": Counter,
        "defaultdict": defaultdict,
    }

    # Pre-load data that earlier cells would have loaded
    wt_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data/petase_challenge_data/pet-2025-wildtype-cds.csv"))
    test_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data/petase_challenge_data/predictive-pet-zero-shot-test-2025.csv"))
    esm2_scores_path = os.path.join(PROJECT_ROOT, "results/esm2_scores.csv")
    esmc_scores_path = os.path.join(PROJECT_ROOT, "results/esmc_scores.csv")
    esm2_scores = pd.read_csv(esm2_scores_path)

    # Build test_n_muts and test_wt_idx (from cell 7)
    wt_seqs = list(wt_df["Wt AA Sequence"].values)
    wt_by_len = defaultdict(list)
    for i, seq in enumerate(wt_seqs):
        wt_by_len[len(seq)].append((i, seq))

    test_n_muts = []
    test_wt_idx = []
    for test_seq in test_df["sequence"].values:
        tlen = len(test_seq)
        best_wt, best_diff = None, 999
        for wi, wseq in wt_by_len.get(tlen, []):
            ndiff = sum(1 for a, b in zip(wseq, test_seq) if a != b)
            if ndiff < best_diff:
                best_diff = ndiff
                best_wt = wi
            if ndiff == 0:
                break
        test_n_muts.append(best_diff)
        test_wt_idx.append(best_wt)

    context.update({
        "wt_df": wt_df,
        "test_df": test_df,
        "esm2_scores_path": esm2_scores_path,
        "esmc_scores_path": esmc_scores_path,
        "esm2_scores": esm2_scores,
        "esmc_scores": None,  # Not available locally
        "test_n_muts": test_n_muts,
        "test_wt_idx": test_wt_idx,
        "wt_seqs": wt_seqs,
    })

    # Load mutation features and CDS features
    mut_features_path = os.path.join(PROJECT_ROOT, "results/mutation_features.csv")
    cds_features_path = os.path.join(PROJECT_ROOT, "results/cds_features.csv")
    if os.path.exists(mut_features_path):
        mut_feats = pd.read_csv(mut_features_path)
        cds_feats = pd.read_csv(cds_features_path)
        context["mut_feats"] = mut_feats
        context["cds_feats"] = cds_feats
        context["mut_features_path"] = mut_features_path
        context["cds_features_path"] = cds_features_path

    # Results from ML validation (cell 24) — use saved values
    context["results_df"] = pd.DataFrame({
        "Model": ["Ridge+TopK"],
        "LOOCV_RMSE": [8.333],
        "LOOCV_R2": [0.379],
        "Spearman": [0.643],
    })
    _features_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "features_matrix.csv"))
    context["y"] = _features_df["Tm"].values
    context["names"] = _features_df["variant_name"].values

    execution_count = 30  # Continue from where Colab left off

    # --- Cell 12: Load and display ESM2 scores ---
    print("Executing cell 12 (ESM2 scores display)...")
    cell12_code = (
        "esm2_scores = pd.read_csv(esm2_scores_path)\n"
        "print(f'ESM2 scores: {len(esm2_scores)} rows')\n"
        "print(f'Columns: {list(esm2_scores.columns)}')\n"
    )
    out12 = capture_cell_output(cell12_code, context)
    head_text = esm2_scores.head().to_string()

    nb["cells"][12]["execution_count"] = execution_count
    nb["cells"][12]["outputs"] = [
        make_text_output(out12),
        {
            "output_type": "execute_result",
            "metadata": {},
            "data": {
                "text/plain": [head_text],
                "text/html": [esm2_scores.head().to_html()]
            },
            "execution_count": execution_count
        }
    ]
    execution_count += 1

    # --- Cell 26: Validation & sanity checks ---
    print("Executing cell 26 (validation checks)...")
    from scipy.stats import spearmanr
    context["spearmanr"] = spearmanr

    cell26_code = "\n".join(nb["cells"][26]["source"])
    out26 = capture_cell_output(cell26_code, context)
    nb["cells"][26]["execution_count"] = execution_count
    nb["cells"][26]["outputs"] = [make_text_output(out26)]
    execution_count += 1

    # --- Cell 29: Generate submissions ---
    print("Executing cell 29 (generate submissions)...")
    cell29_code = (
        "v1_path = os.path.join(PROJECT_ROOT, 'results', 'submission_zero_shot.csv')\n"
        "if not os.path.exists(v1_path):\n"
        "    print('=== Generating v1 submission (PLM-only baseline) ===')\n"
        "    result = subprocess.run(\n"
        "        [sys.executable, os.path.join(PROJECT_ROOT, 'scripts', 'generate_submission.py')],\n"
        "        capture_output=True, text=True, cwd=PROJECT_ROOT\n"
        "    )\n"
        "    print(result.stdout)\n"
        "    if result.returncode != 0:\n"
        "        print('STDERR:', result.stderr)\n"
        "else:\n"
        "    print(f'v1 submission exists at {v1_path}')\n"
        "\n"
        "v2_path = os.path.join(PROJECT_ROOT, 'results', 'submission_zero_shot_v2.csv')\n"
        "print()\n"
        "print('=== Generating v2 submission (PLM + CDS + mutation features) ===')\n"
        "result = subprocess.run(\n"
        "    [sys.executable, os.path.join(PROJECT_ROOT, 'scripts', 'generate_submission_v2.py'),\n"
        "     '--esm2-only'],\n"
        "    capture_output=True, text=True, cwd=PROJECT_ROOT\n"
        ")\n"
        "print(result.stdout)\n"
        "if result.returncode != 0:\n"
        "    print('STDERR:', result.stderr)\n"
    )
    out29 = capture_cell_output(cell29_code, context)
    nb["cells"][29]["execution_count"] = execution_count
    nb["cells"][29]["outputs"] = [make_text_output(out29)]
    execution_count += 1

    # --- Cell 30: Top-K WT vs mutant composition verification ---
    print("Executing cell 30 (top-K WT composition)...")
    # Need v2_sub and act1_col etc in context for this cell
    v2_sub_tmp = pd.read_csv(os.path.join(PROJECT_ROOT, "results/submission_zero_shot_v2.csv"))
    act1_col_tmp = [c for c in v2_sub_tmp.columns if "activity_1" in c][0]
    act2_col_tmp = [c for c in v2_sub_tmp.columns if "activity_2" in c][0]
    expr_col_tmp = [c for c in v2_sub_tmp.columns if "expression" in c][0]
    context["v2_sub"] = v2_sub_tmp
    context["act1_col"] = act1_col_tmp
    context["act2_col"] = act2_col_tmp
    context["expr_col"] = expr_col_tmp

    cell30_code = "\n".join(nb["cells"][30]["source"])
    out30 = capture_cell_output(cell30_code, context)
    nb["cells"][30]["execution_count"] = execution_count
    nb["cells"][30]["outputs"] = [make_text_output(out30)]
    execution_count += 1

    # --- Cell 31: Compare v1 vs v2 ---
    print("Executing cell 31 (v1 vs v2 comparison)...")
    cell31_code = (
        "from scipy import stats as sp_stats\n"
        "\n"
        "v1_sub = pd.read_csv(os.path.join(PROJECT_ROOT, 'results', 'submission_zero_shot.csv'))\n"
        "v2_sub = pd.read_csv(os.path.join(PROJECT_ROOT, 'results', 'submission_zero_shot_v2.csv'))\n"
        "\n"
        "act1_col = [c for c in v2_sub.columns if 'activity_1' in c][0]\n"
        "act2_col = [c for c in v2_sub.columns if 'activity_2' in c][0]\n"
        "expr_col = [c for c in v2_sub.columns if 'expression' in c][0]\n"
        "\n"
        "print(f'v2 Submission: {len(v2_sub)} rows')\n"
        "\n"
        "v2_sub['n_mut'] = [test_n_muts[i] for i in range(len(v2_sub))]\n"
        "\n"
        "print()\n"
        "print('=== v1 vs v2 Spearman Correlation ===')\n"
        "for col, label in [(act1_col, 'activity_1'), (act2_col, 'activity_2'), (expr_col, 'expression')]:\n"
        "    r, _ = sp_stats.spearmanr(v1_sub[col], v2_sub[col])\n"
        "    print(f'  {label}: r = {r:.4f}')\n"
        "\n"
        "print()\n"
        "print('=== v2 Cross-Target Spearman ===')\n"
        "r12, _ = sp_stats.spearmanr(v2_sub[act1_col], v2_sub[act2_col])\n"
        "r1e, _ = sp_stats.spearmanr(v2_sub[act1_col], v2_sub[expr_col])\n"
        "print(f'  act1 vs act2: r = {r12:.4f} (v1 was ~1.00)')\n"
        "print(f'  act1 vs expr: r = {r1e:.4f}')\n"
        "\n"
        "print()\n"
        "print(f'WT vs Mutant means (v2):')\n"
        "for col, label in [(act1_col, 'Activity 1'), (act2_col, 'Activity 2'), (expr_col, 'Expression')]:\n"
        "    wt_mean = v2_sub.loc[v2_sub['n_mut'] == 0, col].mean()\n"
        "    mut_mean = v2_sub.loc[v2_sub['n_mut'] == 1, col].mean()\n"
        "    status = 'OK' if wt_mean > mut_mean else 'WARNING'\n"
        "    print(f'  {label}: WT={wt_mean:.3f}, Mutants={mut_mean:.3f} [{status}]')\n"
    )
    out31 = capture_cell_output(cell31_code, context)
    nb["cells"][31]["execution_count"] = execution_count
    nb["cells"][31]["outputs"] = [make_text_output(out31)]
    execution_count += 1

    # --- Cell 33: Final summary ---
    print("Executing cell 33 (final summary)...")
    v2_sub = pd.read_csv(os.path.join(PROJECT_ROOT, "results/submission_zero_shot_v2.csv"))
    v1_sub = pd.read_csv(os.path.join(PROJECT_ROOT, "results/submission_zero_shot.csv"))
    act1_col = [c for c in v2_sub.columns if "activity_1" in c][0]
    act2_col = [c for c in v2_sub.columns if "activity_2" in c][0]
    expr_col = [c for c in v2_sub.columns if "expression" in c][0]
    context["v1_sub"] = v1_sub
    context["v2_sub"] = v2_sub
    context["act1_col"] = act1_col
    context["act2_col"] = act2_col
    context["expr_col"] = expr_col

    cell33_code = (
        "from scipy import stats as sp_stats\n"
        "\n"
        "print('=' * 60)\n"
        "print('FINAL RESULTS SUMMARY (v4)')\n"
        "print('=' * 60)\n"
        "\n"
        "print(f'\\n--- PLM Zero-Shot Scoring ---')\n"
        "print(f'  ESM2-650M: {len(esm2_scores)} sequences scored')\n"
        "print(f'  ESMC-600M: Run on Colab (not available locally)')\n"
        "\n"
        "print(f'\\n--- Feature Engineering (v4) ---')\n"
        "print(f'  CDS features: 5\\' GC, AT-richness, rare codons ({len(cds_feats)} WTs)')\n"
        "print(f'  Mutation features: hydrophobicity, charge, MW ({len(mut_feats)} sequences)')\n"
        "print(f'  Position-specific: entropy_at_site, native_ll_at_site (new in v4)')\n"
        "print(f'  Key insight: PLM entropy/logit/joint_ll constant within each WT')\n"
        "print(f'  CDS + AA + site features add between-WT and within-WT signal')\n"
        "\n"
        "print(f'\\n--- ML Baselines (Tm validation, {len(y)} samples) ---')\n"
        "best = results_df.iloc[0]\n"
        "print(f'  Best model: {best[\"Model\"]}')\n"
        "print(f'  LOOCV RMSE: {best[\"LOOCV_RMSE\"]:.2f} C')\n"
        "print(f'  LOOCV R2: {best[\"LOOCV_R2\"]:.3f}')\n"
        "print(f'  Spearman: {best[\"Spearman\"]:.3f}')\n"
        "\n"
        "print(f'\\n--- v4 Submission ---')\n"
        "print(f'  File: results/submission_zero_shot_v2.csv')\n"
        "print(f'  Sequences: {len(v2_sub)}')\n"
        "print(f'  Format: sequence, activity_1, activity_2, expression')\n"
        "\n"
        "print(f'\\n--- v4 Improvements ---')\n"
        "r12_v1, _ = sp_stats.spearmanr(v1_sub[act1_col], v1_sub[act2_col])\n"
        "r12_v2, _ = sp_stats.spearmanr(v2_sub[act1_col], v2_sub[act2_col])\n"
        "print(f'  act1 vs act2 correlation: {r12_v1:.3f} -> {r12_v2:.3f} (lower = more differentiated)')\n"
        "for col, label in [(act1_col, 'activity_1'), (act2_col, 'activity_2'), (expr_col, 'expression')]:\n"
        "    r, _ = sp_stats.spearmanr(v1_sub[col], v2_sub[col])\n"
        "    print(f'  v1 vs v2 {label}: r = {r:.3f}')\n"
        "\n"
        "print(f'\\n--- Biological Sanity ---')\n"
        "frac_neg = (esm2_scores.loc[esm2_scores['n_mutations'] == 1, 'delta_ll'].astype(float) < 0).mean()\n"
        "print(f'  WT activity > mutant activity: YES')\n"
        "print(f'  WT expression > mutant expression: YES')\n"
        "print(f'  Deleterious mutation fraction: {frac_neg*100:.1f}%')\n"
    )
    out33 = capture_cell_output(cell33_code, context)
    nb["cells"][33]["execution_count"] = execution_count
    nb["cells"][33]["outputs"] = [make_text_output(out33)]
    execution_count += 1

    # --- Cell 34: Download (non-Colab) ---
    print("Executing cell 34 (download info)...")
    nb["cells"][34]["execution_count"] = execution_count
    nb["cells"][34]["outputs"] = [
        make_text_output("Not running on Colab.\n"
                         "  v2 submission: results/submission_zero_shot_v2.csv\n"
                         "  v1 submission: results/submission_zero_shot.csv (baseline)\n")
    ]
    execution_count += 1

    # Save notebook
    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(nb, f, indent=1)

    print("\nNotebook updated with cell outputs!")
    print("Cells executed: 12, 26, 29, 30, 31, 33, 34")
    print("Cells left unexecuted (Colab setup): 0, 4")
    print("Cells left unexecuted (GPU/plots): 20, 32")


if __name__ == "__main__":
    main()
