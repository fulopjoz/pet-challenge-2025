#!/usr/bin/env python3
"""
Feature extraction for PETase sequences.
Fast computation of sequence-based features (<1s per sequence).

Features:
  - 20 AA composition frequencies
  - 7 physicochemical properties (MW, GRAVY, Charge, Aromatic, GlyPro, Length, Cys)
  - 3 active-site distances (min distance from mutation sites to Ser160/Asp206/His237)
  - 3 mutation-count features (N_mutations, Mutation_span, Mean_mutation_pos)
  - 4 structural proxies (helix/beta propensity, N/C-term aromatic)
  Total: 37 features
"""

import csv
import re
from pathlib import Path
import numpy as np

# Amino acid properties (fast lookup)
AA_MW = {
    'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10,
    'C': 121.15, 'E': 147.13, 'Q': 146.15, 'G': 75.07,
    'H': 155.16, 'I': 131.17, 'L': 131.17, 'K': 146.19,
    'M': 149.21, 'F': 165.19, 'P': 115.13, 'S': 105.09,
    'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15
}

AA_GRAVY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5,
    'C': 2.5, 'E': -3.5, 'Q': -3.5, 'G': -0.4,
    'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9,
    'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8,
    'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

AA_CHARGE = {
    'K': 1, 'R': 1, 'H': 0.1,  # Positive
    'D': -1, 'E': -1,  # Negative
    'A': 0, 'C': 0, 'N': 0, 'Q': 0, 'G': 0,
    'I': 0, 'L': 0, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
}

# Wild-type PETase sequence (A0A0K8P6T7)
WT_SEQUENCE = """MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPS
GYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSR
SSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPW
DSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQA
LIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCS""".replace('\n', '')

# Active site positions (I. sakaiensis PETase, 1-indexed)
ACTIVE_SITE = {
    'serine': 160,
    'aspartate': 206,
    'histidine': 237
}


def parse_mutation_positions(mutation_str):
    """Parse slash-separated mutation string into list of positions.

    Examples:
        'WT' -> []
        'S121E/D186H/R280A' -> [121, 186, 280]
        'N233K/R224Q/S121E/D186H/R280A' -> [233, 224, 121, 186, 280]

    Returns:
        List of 1-indexed mutation positions, or [] for WT.
    """
    if not mutation_str or mutation_str == 'WT':
        return []

    positions = []
    for part in mutation_str.split('/'):
        part = part.strip()
        match = re.match(r'^([A-Z])(\d+)([A-Z])$', part)
        if match:
            positions.append(int(match.group(2)))
    return positions


class SequenceFeatures:
    """Extract features from protein sequences"""

    def __init__(self):
        self.sequence = None
        self.features = {}

    def from_sequence(self, sequence, mutation_positions=None):
        """Extract all features from a sequence.

        Args:
            sequence: Protein sequence string
            mutation_positions: List of 1-indexed mutation positions ([] for WT)
        """
        self.sequence = sequence
        self.features = {}
        if mutation_positions is None:
            mutation_positions = []

        self.aa_composition()
        self.physicochemical()
        self.position_features(mutation_positions)
        self.mutation_count_features(mutation_positions)
        self.structure_features()

        return self.features

    def aa_composition(self):
        """20 features: amino acid frequencies"""
        total = len(self.sequence)
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            count = self.sequence.count(aa)
            self.features[f'AA_{aa}'] = count / total

    def physicochemical(self):
        """Physicochemical properties"""
        N = len(self.sequence)

        # Molecular weight
        self.features['MW'] = sum(AA_MW.get(aa, 0) for aa in self.sequence)

        # GRAVY score (hydrophobicity)
        self.features['GRAVY'] = sum(AA_GRAVY.get(aa, 0) for aa in self.sequence) / N

        # Net charge (simplified, at pH 7)
        self.features['Charge'] = sum(AA_CHARGE.get(aa, 0) for aa in self.sequence)

        # Aromatic fraction
        self.features['Aromatic'] = sum(1 for aa in self.sequence if aa in 'FWY') / N

        # Gly+Pro fraction (flexibility)
        self.features['GlyPro'] = sum(1 for aa in self.sequence if aa in 'GP') / N

        # Length
        self.features['Length'] = N

        # Cysteine count (relevant for disulfide bonds)
        self.features['Cys_count'] = self.sequence.count('C')

    def position_features(self, mutation_positions):
        """Distance from mutation sites to active site residues.

        For each active site residue, compute the minimum absolute distance
        (in sequence position) from any mutation site to that residue.
        For WT (no mutations), use sequence length as sentinel.
        """
        for name, active_pos in ACTIVE_SITE.items():
            if mutation_positions:
                min_dist = min(abs(pos - active_pos) for pos in mutation_positions)
                self.features[f'Dist_to_{name}'] = min_dist
            else:
                # WT: no mutations, use sentinel value
                self.features[f'Dist_to_{name}'] = len(self.sequence)

    def mutation_count_features(self, mutation_positions):
        """Features based on mutation count and spread."""
        n = len(mutation_positions)
        self.features['N_mutations'] = n

        if n > 0:
            self.features['Mutation_span'] = max(mutation_positions) - min(mutation_positions)
            self.features['Mean_mutation_pos'] = sum(mutation_positions) / n
        else:
            self.features['Mutation_span'] = 0
            self.features['Mean_mutation_pos'] = 0.0

    def structure_features(self):
        """Simple structural proxies from sequence"""
        N = len(self.sequence)

        # Approximate secondary structure propensities
        helix_propensity = sum(1 for aa in self.sequence if aa in 'AELMQ')
        beta_propensity = sum(1 for aa in self.sequence if aa in 'VIFWY')

        self.features['Helix_propensity'] = helix_propensity / N
        self.features['Beta_propensity'] = beta_propensity / N

        # N-term and C-terminal composition
        n_term = self.sequence[:10]
        c_term = self.sequence[-10:]
        self.features['Nterm_aromatic'] = sum(1 for aa in n_term if aa in 'FWY') / 10
        self.features['Cterm_aromatic'] = sum(1 for aa in c_term if aa in 'FWY') / 10


def apply_mutation(wt_sequence, mutation_str):
    """Apply slash-separated mutations to wild-type sequence.

    Args:
        wt_sequence: Wild-type protein sequence
        mutation_str: 'WT' or slash-separated like 'S121E/D186H/R280A'

    Returns:
        Mutated sequence, or None if any mutation fails.
    """
    if mutation_str == 'WT' or not mutation_str:
        return wt_sequence

    seq = list(wt_sequence)
    applied = 0

    for part in mutation_str.split('/'):
        part = part.strip()
        match = re.match(r'^([A-Z])(\d+)([A-Z])$', part)
        if not match:
            print(f"  WARNING: Cannot parse mutation '{part}' in '{mutation_str}'")
            return None

        original = match.group(1)
        position = int(match.group(2)) - 1  # 1-indexed to 0-indexed
        mutant = match.group(3)

        if position < 0 or position >= len(seq):
            print(f"  WARNING: Position {position+1} out of range for '{part}'")
            return None

        if seq[position] != original:
            print(f"  WARNING: Expected {original} at position {position+1}, "
                  f"found {seq[position]} for mutation '{part}'")
            return None

        seq[position] = mutant
        applied += 1

    result = ''.join(seq)

    # Validation: if mutations were specified, result must differ from WT
    if applied > 0 and result == wt_sequence:
        print(f"  WARNING: Mutations '{mutation_str}' produced WT-identical sequence!")
        return None

    return result


def create_feature_dataset(wt_sequence, mutations, output_path):
    """Create feature matrix from mutation dataset.

    Only processes IsPETase entries (skips LCC and other enzymes).

    Args:
        wt_sequence: Wild-type IsPETase sequence
        mutations: List of mutation dictionaries from CSV
        output_path: Path to save features CSV

    Returns:
        X (feature matrix), y (Tm values), feature_names, variant_names
    """
    extractor = SequenceFeatures()
    all_features = []
    all_tms = []
    variant_names = []
    skipped = []

    for mut in mutations:
        # Skip non-IsPETase entries
        if mut.get('enzyme', 'IsPETase') != 'IsPETase':
            skipped.append((mut['variant_name'], 'different enzyme'))
            continue

        mutation_str = mut['mutation']
        mutation_positions = parse_mutation_positions(mutation_str)

        # Apply mutations to get variant sequence
        variant_seq = apply_mutation(wt_sequence, mutation_str)
        if variant_seq is None:
            skipped.append((mut['variant_name'], 'mutation application failed'))
            continue

        # Extract features
        features = extractor.from_sequence(variant_seq, mutation_positions)

        # Validate AA composition sums to ~1.0
        aa_sum = sum(features[f'AA_{aa}'] for aa in 'ACDEFGHIKLMNPQRSTVWY')
        if abs(aa_sum - 1.0) > 0.01:
            print(f"  WARNING: AA composition sums to {aa_sum:.4f} for {mut['variant_name']}")

        all_features.append(features)
        all_tms.append(float(mut['tm']))
        variant_names.append(mut['variant_name'])

    if skipped:
        print(f"\nSkipped {len(skipped)} entries:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")

    # Get feature names from first entry
    feature_names = list(all_features[0].keys())

    # Convert to matrix
    X = np.array([[f[name] for name in feature_names] for f in all_features])
    y = np.array(all_tms)

    print(f"\nCreated feature matrix: {X.shape}")
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"  Tm range: {y.min():.1f} - {y.max():.1f} C")

    # Verify no duplicate feature rows
    unique_rows = len(set(tuple(row) for row in X.tolist()))
    if unique_rows < len(X):
        print(f"  WARNING: {len(X) - unique_rows} duplicate feature rows detected!")
    else:
        print(f"  All {len(X)} feature rows are distinct.")

    # Save to CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['variant_name'] + feature_names + ['Tm'])
        for i in range(len(X)):
            row = [variant_names[i]] + [float(X[i, j]) for j in range(len(feature_names))] + [all_tms[i]]
            writer.writerow(row)

    print(f"Saved features to {output_path}")

    return X, y, feature_names, variant_names


if __name__ == '__main__':
    # Load mutations
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir.parent / 'data' / 'mutations_dataset.csv'
    mutations = list(csv.DictReader(open(data_path)))

    # Create feature matrix
    output_dir = script_dir.parent / 'data'
    feature_path = output_dir / 'features_matrix.csv'

    X, y, feature_names, variant_names = create_feature_dataset(WT_SEQUENCE, mutations, feature_path)

    print(f"\nFEATURE SUMMARY")
    print(f"Total IsPETase samples: {len(y)}")
    print(f"Target range (Tm): {y.min():.1f} - {y.max():.1f} C")
    print(f"Feature count: {len(feature_names)}")
    print(f"\nFeatures: {feature_names}")

    # Show per-variant details
    print(f"\nPer-variant features (selected):")
    for i, name in enumerate(variant_names):
        n_mut_idx = feature_names.index('N_mutations')
        dist_ser_idx = feature_names.index('Dist_to_serine')
        print(f"  {name:25s} N_mut={X[i, n_mut_idx]:.0f}  "
              f"Dist_to_Ser160={X[i, dist_ser_idx]:.0f}  Tm={y[i]:.1f}C")
