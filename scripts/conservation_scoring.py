#!/usr/bin/env python3
"""
Conservation-Based Scoring for PET Challenge 2025

Standalone, CPU-only alternative to PLM-based scoring (~1 min runtime).
Uses MSA-derived conservation frequencies + Table S6 (Buchholz et al., Proteins 2022)
to penalize mutations at conserved positions.

Biological assumption: "anything atypical is penalized" — mutating a conserved
residue is more likely harmful than beneficial.

Evaluation is NDCG (rank-based), so ranking alone is sufficient.

Usage:
    python scripts/conservation_scoring.py

Requires: mafft (apt-get install -y mafft)
Output:  results/submission_conservation.csv
"""

import os
import sys
import csv
import subprocess
import tempfile
from collections import defaultdict
import numpy as np
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "petase_challenge_data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

WT_CSV = os.path.join(DATA_DIR, "pet-2025-wildtype-cds.csv")
TEST_CSV = os.path.join(DATA_DIR, "predictive-pet-zero-shot-test-2025.csv")
ISPETASE_FASTA = os.path.join(PROJECT_ROOT, "data", "petase_ideonella.fasta")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "submission_conservation.csv")

# Charge at pH 7 (simplified)
AA_CHARGE = {
    "D": -1.0, "E": -1.0,  # acidic
    "K": 1.0, "R": 1.0,    # basic
    "H": 0.1,              # weakly basic
}

# Table S6 from Buchholz et al. (Proteins, 2022)
# Amino acid frequencies at positions conserved >=70% across 2930 PETase core domains.
# Format: {ispetase_full_length_pos: (consensus_aa, frequency_pct)}
# IsPETase = A0A0K8P6T7, full-length numbering (290 aa including signal peptide).
TABLE_S6 = {
    32: ("Y", 74), 34: ("R", 84), 35: ("G", 91), 36: ("P", 92),
    38: ("P", 95), 39: ("T", 87), 42: ("S", 73), 45: ("A", 87),
    48: ("G", 97), 49: ("P", 71), 57: ("V", 91), 62: ("G", 93),
    63: ("F", 93), 64: ("G", 83), 65: ("G", 86), 66: ("G", 93),
    67: ("T", 76), 68: ("I", 79), 69: ("Y", 84), 70: ("Y", 91),
    71: ("P", 98), 72: ("T", 85), 74: ("T", 81), 76: ("G", 90),
    77: ("T", 84), 78: ("F", 74), 79: ("G", 90), 80: ("A", 77),
    85: ("P", 99), 86: ("G", 100), 88: ("T", 76), 92: ("S", 70),
    96: ("W", 93), 98: ("G", 89), 99: ("P", 82), 100: ("R", 81),
    101: ("L", 81), 102: ("A", 97), 103: ("S", 96), 105: ("G", 99),
    106: ("F", 97), 107: ("V", 96), 108: ("V", 94), 111: ("I", 84),
    113: ("T", 96), 118: ("D", 98), 120: ("P", 87), 122: ("S", 71),
    123: ("R", 99), 124: ("G", 73), 126: ("Q", 92), 127: ("L", 83),
    128: ("L", 78), 129: ("A", 88), 130: ("A", 96), 131: ("L", 88),
    132: ("D", 82), 133: ("Y", 77), 134: ("L", 85), 138: ("S", 83),
    145: ("V", 82), 146: ("R", 71), 148: ("R", 81), 150: ("D", 94),
    153: ("R", 94), 154: ("L", 85), 156: ("V", 89), 158: ("G", 100),
    159: ("H", 87), 160: ("S", 100), 161: ("M", 94), 162: ("G", 100),
    163: ("G", 99), 164: ("G", 96), 165: ("G", 97), 167: ("L", 88),
    169: ("A", 89), 170: ("A", 82), 173: ("R", 76), 174: ("P", 76),
    176: ("L", 84), 178: ("A", 95), 179: ("A", 78), 181: ("P", 80),
    182: ("L", 79), 184: ("P", 76), 185: ("W", 77), 197: ("P", 97),
    198: ("T", 93), 202: ("G", 75), 206: ("D", 100), 209: ("A", 87),
    211: ("V", 70), 214: ("H", 77), 217: ("P", 79), 218: ("F", 74),
    219: ("Y", 96), 221: ("S", 70), 228: ("A", 77), 229: ("Y", 83),
    231: ("E", 91), 232: ("L", 76), 235: ("A", 76), 237: ("H", 100),
    240: ("P", 74), 244: ("N", 74), 257: ("W", 90), 258: ("L", 80),
    259: ("K", 94), 260: ("R", 78), 261: ("F", 79), 263: ("D", 94),
    265: ("D", 97), 266: ("T", 76), 267: ("R", 96), 268: ("Y", 92),
    270: ("Q", 77), 271: ("F", 96), 272: ("L", 86), 273: ("C", 95),
    274: ("P", 82),
}
# Note: Position 222.1 (insertion in some sequences) is omitted — not mappable.


def rank_scale(scores, low, high):
    """Map scores to [low, high] range preserving rank order."""
    ranks = stats.rankdata(scores)
    normalized = (ranks - 1) / max(len(ranks) - 1, 1)
    return low + normalized * (high - low)


def load_ispetase():
    """Load full-length IsPETase from FASTA (290 aa including signal peptide)."""
    with open(ISPETASE_FASTA) as f:
        lines = f.readlines()
    seq = "".join(line.strip() for line in lines if not line.startswith(">"))
    return seq


def run_mafft(sequences, names):
    """Run MAFFT --auto on sequences and return aligned (names, sequences)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".fasta", delete=False
    ) as tmp:
        for name, seq in zip(names, sequences):
            tmp.write(">%s\n%s\n" % (name, seq))
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["mafft", "--auto", "--thread", "-1", tmp_path],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print("MAFFT stderr:", result.stderr[:500])
            raise RuntimeError("MAFFT failed (rc=%d)" % result.returncode)

        # Parse aligned FASTA from stdout
        aligned_names = []
        aligned_seqs = []
        cur_name = None
        cur_parts = []

        for line in result.stdout.split("\n"):
            line = line.strip()
            if line.startswith(">"):
                if cur_name is not None:
                    aligned_names.append(cur_name)
                    aligned_seqs.append("".join(cur_parts).upper())
                cur_name = line[1:].split()[0]
                cur_parts = []
            elif line:
                cur_parts.append(line)

        if cur_name is not None:
            aligned_names.append(cur_name)
            aligned_seqs.append("".join(cur_parts).upper())

        return aligned_names, aligned_seqs
    finally:
        os.unlink(tmp_path)


def compute_column_frequencies(aligned_seqs):
    """Compute per-column AA frequencies and gap fractions."""
    n_seqs = len(aligned_seqs)
    n_cols = len(aligned_seqs[0])

    col_counts = [defaultdict(int) for _ in range(n_cols)]
    gap_counts = [0] * n_cols

    for seq in aligned_seqs:
        for col, ch in enumerate(seq):
            if ch == "-" or ch == ".":
                gap_counts[col] += 1
            else:
                col_counts[col][ch] += 1

    col_freq = [{} for _ in range(n_cols)]
    gap_frac = [0.0] * n_cols

    for col in range(n_cols):
        n_nongap = n_seqs - gap_counts[col]
        gap_frac[col] = gap_counts[col] / n_seqs
        if n_nongap > 0:
            for aa, cnt in col_counts[col].items():
                col_freq[col][aa] = cnt / n_nongap

    return col_freq, gap_frac


def build_pos_to_col(aligned_seq):
    """Map ungapped positions (0-indexed) to MSA column indices."""
    mapping = {}
    pos = 0
    for col, ch in enumerate(aligned_seq):
        if ch != "-" and ch != ".":
            mapping[pos] = col
            pos += 1
    return mapping


def blend_table_s6(ispetase_pos_to_col, col_freq, gap_frac):
    """
    Blend Table S6 frequencies (2930 seqs) with our MSA frequencies (314 seqs).

    For columns mapping to a Table S6 position:
      consensus AA: 0.7 * table_s6 + 0.3 * msa
      other AAs:    scaled so all frequencies sum to 1.0
    For columns without Table S6 data: use MSA frequency only.
    """
    n_cols = len(col_freq)
    eff = [dict(d) for d in col_freq]  # deep copy

    mapped = 0
    skipped = 0

    for ispetase_pos, (cons_aa, freq_pct) in TABLE_S6.items():
        seq_idx = ispetase_pos - 1  # 1-indexed → 0-indexed
        if seq_idx not in ispetase_pos_to_col:
            skipped += 1
            continue

        col = ispetase_pos_to_col[seq_idx]
        if col >= n_cols:
            skipped += 1
            continue

        table_freq = freq_pct / 100.0
        msa_cons = col_freq[col].get(cons_aa, 0.0)
        blended_cons = 0.7 * table_freq + 0.3 * msa_cons
        eff[col][cons_aa] = blended_cons

        # Scale other AAs so total sums to ~1.0
        remaining_msa = sum(v for k, v in col_freq[col].items() if k != cons_aa)
        remaining = 1.0 - blended_cons
        if remaining_msa > 1e-10 and remaining > 0:
            scale = remaining / remaining_msa
            for aa in col_freq[col]:
                if aa != cons_aa:
                    eff[col][aa] = col_freq[col][aa] * scale
        mapped += 1

    print("  Table S6: %d/%d positions mapped (%d skipped)" % (
        mapped, len(TABLE_S6), skipped))
    return eff


def main():
    print("=== Conservation-Based Scoring for PET Challenge 2025 ===\n")

    # ── Step 1: Load sequences ──────────────────────────────────────────
    print("Step 1: Loading sequences...")

    wt_seqs = []
    with open(WT_CSV) as f:
        for row in csv.DictReader(f):
            wt_seqs.append(row["Wt AA Sequence"])
    print("  %d wild-type sequences" % len(wt_seqs))

    test_seqs = []
    with open(TEST_CSV) as f:
        for row in csv.DictReader(f):
            test_seqs.append(row["sequence"])
    print("  %d test sequences" % len(test_seqs))

    ispetase = load_ispetase()
    print("  IsPETase: %d aa (full-length, incl. signal peptide)" % len(ispetase))

    # Verify Table S6 positions against IsPETase sequence.
    # Table S6 gives the CONSENSUS AA across 2930 diverse PETases — IsPETase
    # need not match at every position (it's just one specific PETase).
    matched = 0
    diverged = 0
    for pos, (cons_aa, _) in TABLE_S6.items():
        if pos - 1 < len(ispetase):
            if ispetase[pos - 1] == cons_aa:
                matched += 1
            else:
                diverged += 1
    print("  Table S6: %d/%d positions match IsPETase, %d diverge (expected)" % (
        matched, len(TABLE_S6), diverged))

    # ── Step 2: MSA with MAFFT ─────────────────────────────────────────
    print("\nStep 2: Building MSA with MAFFT (%d sequences)..." % (
        len(wt_seqs) + 1))

    names = ["WT_%d" % i for i in range(len(wt_seqs))] + ["IsPETase"]
    sequences = list(wt_seqs) + [ispetase]

    aligned_names, aligned_seqs = run_mafft(sequences, names)
    n_cols = len(aligned_seqs[0])
    print("  Aligned %d sequences × %d columns" % (len(aligned_seqs), n_cols))

    name_to_idx = {n: i for i, n in enumerate(aligned_names)}
    ispetase_aidx = name_to_idx["IsPETase"]
    wt_aligned = [aligned_seqs[name_to_idx["WT_%d" % i]]
                  for i in range(len(wt_seqs))]

    # ── Step 3: Per-column frequencies ─────────────────────────────────
    print("\nStep 3: Computing per-column AA frequencies...")
    col_freq, gap_frac = compute_column_frequencies(aligned_seqs)
    print("  %d columns, gap fraction range [%.3f, %.3f]" % (
        n_cols, min(gap_frac), max(gap_frac)))

    # ── Step 4: Map Table S6 → MSA columns ─────────────────────────────
    print("\nStep 4: Mapping Table S6 to MSA columns...")
    ispetase_pos_to_col = build_pos_to_col(aligned_seqs[ispetase_aidx])
    effective_freq = blend_table_s6(ispetase_pos_to_col, col_freq, gap_frac)

    # Verify catalytic triad (S160, D206, H237 — must be highly conserved)
    for pos, label in [(160, "S160"), (206, "D206"), (237, "H237")]:
        col = ispetase_pos_to_col.get(pos - 1)
        if col is not None:
            aa = ispetase[pos - 1]
            cons = effective_freq[col].get(aa, 0.0) * 100
            gf = gap_frac[col]
            print("  Catalytic %s: col=%d, conservation=%.1f%%, gap=%.3f" % (
                label, col, cons, gf))

    # ── Step 5: Map test seqs → parent WTs + extract mutations ─────────
    print("\nStep 5: Mapping test sequences to parent WTs...")
    wt_by_len = defaultdict(list)
    for i, seq in enumerate(wt_seqs):
        wt_by_len[len(seq)].append((i, seq))

    test_wt_idx = []
    test_mutations = []

    for test_seq in test_seqs:
        best_wt = None
        best_diff = 999
        for wi, wseq in wt_by_len.get(len(test_seq), []):
            ndiff = sum(1 for a, b in zip(wseq, test_seq) if a != b)
            if ndiff < best_diff:
                best_diff = ndiff
                best_wt = wi
            if ndiff == 0:
                break
        test_wt_idx.append(best_wt)

        if best_wt is not None and best_diff > 0:
            muts = [(i, w, t) for i, (w, t)
                    in enumerate(zip(wt_seqs[best_wt], test_seq)) if w != t]
        else:
            muts = []
        test_mutations.append(muts)

    n_wt = sum(1 for m in test_mutations if len(m) == 0)
    n_single = sum(1 for m in test_mutations if len(m) == 1)
    print("  WT-identical: %d, single-mutant: %d, other: %d" % (
        n_wt, n_single, len(test_seqs) - n_wt - n_single))

    # ── Step 6: Position → MSA-column maps per WT ──────────────────────
    print("\nStep 6: Building position-to-column maps...")
    wt_pos_to_col = {}
    needed = set(i for i in test_wt_idx if i is not None)
    for wi in needed:
        wt_pos_to_col[wi] = build_pos_to_col(wt_aligned[wi])
    print("  Maps for %d unique WTs" % len(wt_pos_to_col))

    # ── Step 7: Conservation penalty scores ────────────────────────────
    print("\nStep 7: Computing conservation scores...")
    n_test = len(test_seqs)
    penalties = np.zeros(n_test)
    charge_deltas = np.zeros(n_test)
    n_muts = np.zeros(n_test, dtype=int)

    for idx in range(n_test):
        muts = test_mutations[idx]
        n_muts[idx] = len(muts)
        if not muts or test_wt_idx[idx] is None:
            continue

        wi = test_wt_idx[idx]
        p2c = wt_pos_to_col[wi]
        penalty = 0.0
        dq = 0.0

        for pos, wt_aa, mut_aa in muts:
            if pos not in p2c:
                continue
            col = p2c[pos]

            wt_f = effective_freq[col].get(wt_aa, 0.0)
            mut_f = effective_freq[col].get(mut_aa, 0.0)
            penalty += (wt_f - mut_f) * (1.0 - gap_frac[col])

            dq += AA_CHARGE.get(mut_aa, 0.0) - AA_CHARGE.get(wt_aa, 0.0)

        penalties[idx] = penalty
        charge_deltas[idx] = dq

    cons_score = -penalties  # WTs → 0, deleterious mutations → negative

    wt_mask = n_muts == 0
    mut_mask = n_muts > 0
    print("  Score range: [%.4f, %.4f]" % (cons_score.min(), cons_score.max()))
    if wt_mask.sum() > 0:
        print("  WT mean: %.6f" % cons_score[wt_mask].mean())
    if mut_mask.sum() > 0:
        print("  Mutant mean: %.4f" % cons_score[mut_mask].mean())

    # ── Step 8: Generate submission CSV ────────────────────────────────
    print("\nStep 8: Generating submission...")

    # Minor target-specific differentiation via charge
    # Activity 1 (pH 5.5): negative charge lowers catalytic His pKa → helps
    act1_raw = cons_score + 0.1 * (-charge_deltas)
    # Activity 2 (pH 9.0): positive charge aids PET binding + salt bridges
    act2_raw = cons_score + 0.1 * charge_deltas
    # Expression: conservation only
    expr_raw = cons_score.copy()

    activity_1 = rank_scale(act1_raw, 0.0, 5.0)
    activity_2 = rank_scale(act2_raw, 0.0, 5.0)
    expression = rank_scale(expr_raw, 0.0, 3.0)

    # Write submission
    import pandas as pd
    test_df = pd.read_csv(TEST_CSV)
    col_a1 = [c for c in test_df.columns if "activity_1" in c][0]
    col_a2 = [c for c in test_df.columns if "activity_2" in c][0]
    col_ex = [c for c in test_df.columns if "expression" in c][0]

    test_df[col_a1] = activity_1
    test_df[col_a2] = activity_2
    test_df[col_ex] = expression

    os.makedirs(RESULTS_DIR, exist_ok=True)
    test_df.to_csv(OUTPUT_CSV, index=False)
    print("  Saved to %s" % OUTPUT_CSV)

    # ── Summary ────────────────────────────────────────────────────────
    print("\n=== Submission Summary (Conservation-Based) ===")
    print("Method: MSA conservation + Table S6 (Buchholz 2022, 2930 seqs)")
    print("Sequences: %d" % n_test)
    for name, arr in [("activity_1", activity_1), ("activity_2", activity_2),
                      ("expression", expression)]:
        print("  %s: mean=%.4f  std=%.4f  [%.4f, %.4f]" % (
            name, arr.mean(), arr.std(), arr.min(), arr.max()))

    # ── Sanity checks ──────────────────────────────────────────────────
    print("\n=== Sanity Checks ===")
    single_mask = n_muts == 1
    if wt_mask.sum() > 0 and single_mask.sum() > 0:
        for name, arr in [("activity_1", activity_1), ("activity_2", activity_2),
                          ("expression", expression)]:
            wt_mean = arr[wt_mask].mean()
            mut_mean = arr[single_mask].mean()
            status = "OK" if wt_mean > mut_mean else "WARNING"
            print("  %s: WT=%.4f  mutants=%.4f  [%s]" % (
                name, wt_mean, mut_mean, status))

    # Correlation with PLM-based submission
    plm_path = os.path.join(RESULTS_DIR, "submission_zero_shot_v2.csv")
    if os.path.exists(plm_path):
        plm_sub = pd.read_csv(plm_path)
        print("\n=== Spearman Correlation with PLM Submission ===")
        for colname, arr in [("activity_1", activity_1),
                             ("activity_2", activity_2),
                             ("expression", expression)]:
            plm_col = [c for c in plm_sub.columns if colname in c]
            if plm_col:
                r, p = stats.spearmanr(plm_sub[plm_col[0]].values, arr)
                print("  %s: r=%.4f (p=%.2e)" % (colname, r, p))

    r_12, _ = stats.spearmanr(activity_1, activity_2)
    print("\n  activity_1 vs activity_2: r=%.4f" % r_12)


if __name__ == "__main__":
    main()
