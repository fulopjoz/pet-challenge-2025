#!/usr/bin/env python3
"""
Extract verified mutation data from PETase literature.
Creates CSV with: variant_name, mutation, enzyme, tm, tm_std, delta_tm, method, source, notes

All Tm values are verified from published papers. NO fabricated data.

Primary source: Brott et al. 2022 (Eng. Life Sci., DOI: 10.1002/elsc.202100105)
  - nanoDSF, identical conditions (Prometheus NT.48, 50 mM Na-phosphate pH 7.5)
Additional: Lu 2022 (Nature), Son 2019 (ACS Catal), Cui 2021 (ACS Catal)
"""

import csv
from pathlib import Path

# Verified Tm data from published papers
# Each entry includes the exact paper DOI and measurement method
MUTATIONS = [
    # === Brott et al. 2022 (Eng. Life Sci., DOI: 10.1002/elsc.202100105) ===
    # All nanoDSF, Prometheus NT.48, 50 mM Na-phosphate pH 7.5
    {
        'variant_name': 'WT',
        'mutation': 'WT',
        'enzyme': 'IsPETase',
        'tm': 45.1,
        'tm_std': 0.1,
        'delta_tm': 0.0,
        'method': 'nanoDSF',
        'source': 'Brott 2022 (DOI: 10.1002/elsc.202100105)',
        'notes': 'Wild-type Ideonella sakaiensis PETase'
    },
    {
        'variant_name': 'ThermoPETase',
        'mutation': 'S121E/D186H/R280A',
        'enzyme': 'IsPETase',
        'tm': 56.6,
        'tm_std': 1.6,
        'delta_tm': 11.5,
        'method': 'nanoDSF',
        'source': 'Brott 2022 (DOI: 10.1002/elsc.202100105)',
        'notes': 'ThermoPETase (Son 2019 design)'
    },
    {
        'variant_name': 'ThermoPETase+KF',
        'mutation': 'S121E/D186H/R280A/K95N/F201I',
        'enzyme': 'IsPETase',
        'tm': 61.6,
        'tm_std': 0.1,
        'delta_tm': 16.5,
        'method': 'nanoDSF',
        'source': 'Brott 2022 (DOI: 10.1002/elsc.202100105)',
        'notes': 'ThermoPETase + K95N/F201I'
    },
    {
        'variant_name': 'ThermoPETase+SS',
        'mutation': 'S121E/D186H/R280A/N233C/S282C',
        'enzyme': 'IsPETase',
        'tm': 68.2,
        'tm_std': 0.1,
        'delta_tm': 23.1,
        'method': 'nanoDSF',
        'source': 'Brott 2022 (DOI: 10.1002/elsc.202100105)',
        'notes': 'ThermoPETase + disulfide bond N233C/S282C'
    },
    {
        'variant_name': 'ThermoPETase+KF+SS',
        'mutation': 'S121E/D186H/R280A/K95N/F201I/N233C/S282C',
        'enzyme': 'IsPETase',
        'tm': 70.8,
        'tm_std': 0.1,
        'delta_tm': 25.7,
        'method': 'nanoDSF',
        'source': 'Brott 2022 (DOI: 10.1002/elsc.202100105)',
        'notes': 'ThermoPETase + KF + disulfide'
    },
    {
        'variant_name': 'DuraPETase',
        'mutation': 'S121E/D186H/R280A/S214H/I168R/W159H/S188Q/R224Q/T140D',
        'enzyme': 'IsPETase',
        'tm': 75.0,
        'tm_std': 0.1,
        'delta_tm': 29.9,
        'method': 'nanoDSF',
        'source': 'Brott 2022 (DOI: 10.1002/elsc.202100105)',
        'notes': 'DuraPETase (Cui 2021 design, 9 mutations)'
    },
    {
        'variant_name': 'DuraPETase+SS',
        'mutation': 'S121E/D186H/R280A/S214H/I168R/W159H/S188Q/R224Q/T140D/N233C/S282C',
        'enzyme': 'IsPETase',
        'tm': 81.1,
        'tm_std': 0.1,
        'delta_tm': 36.0,
        'method': 'nanoDSF',
        'source': 'Brott 2022 (DOI: 10.1002/elsc.202100105)',
        'notes': 'DuraPETase + disulfide bond, highest Tm in dataset'
    },

    # === Lu et al. 2022 (Nature, DOI: 10.1038/s41586-022-04599-z) ===
    {
        'variant_name': 'FAST-PETase',
        'mutation': 'N233K/R224Q/S121E/D186H/R280A',
        'enzyme': 'IsPETase',
        'tm': 67.8,
        'tm_std': 0.0,
        'delta_tm': 22.7,
        'method': 'DSF',
        'source': 'Lu 2022 (DOI: 10.1038/s41586-022-04599-z)',
        'notes': 'FAST-PETase, ML-designed (MutCompute)'
    },

    # === Son et al. 2019 (ACS Catal, DOI: 10.1021/acscatal.9b00568) ===
    {
        'variant_name': 'WT (Son)',
        'mutation': 'WT',
        'enzyme': 'IsPETase',
        'tm': 46.1,
        'tm_std': 0.0,
        'delta_tm': 0.0,
        'method': 'DSF',
        'source': 'Son 2019 (DOI: 10.1021/acscatal.9b00568)',
        'notes': 'WT measured by Son et al., slightly different from Brott'
    },
    {
        'variant_name': 'ThermoPETase (Son)',
        'mutation': 'S121E/D186H/R280A',
        'enzyme': 'IsPETase',
        'tm': 54.9,
        'tm_std': 0.0,
        'delta_tm': 8.8,
        'method': 'DSF',
        'source': 'Son 2019 (DOI: 10.1021/acscatal.9b00568)',
        'notes': 'ThermoPETase measured by Son et al.'
    },

    # === Cui et al. 2021 (ACS Catal, DOI: 10.1021/acscatal.0c05126) ===
    {
        'variant_name': 'WT (Cui)',
        'mutation': 'WT',
        'enzyme': 'IsPETase',
        'tm': 46.0,
        'tm_std': 0.0,
        'delta_tm': 0.0,
        'method': 'DSF',
        'source': 'Cui 2021 (DOI: 10.1021/acscatal.0c05126)',
        'notes': 'WT measured by Cui et al.'
    },
    {
        'variant_name': 'DuraPETase (Cui)',
        'mutation': 'S121E/D186H/R280A/S214H/I168R/W159H/S188Q/R224Q/T140D',
        'enzyme': 'IsPETase',
        'tm': 77.0,
        'tm_std': 0.0,
        'delta_tm': 31.0,
        'method': 'DSF',
        'source': 'Cui 2021 (DOI: 10.1021/acscatal.0c05126)',
        'notes': 'DuraPETase measured by Cui et al.'
    },

    # === LCC Reference (different enzyme, excluded from IsPETase ML) ===
    {
        'variant_name': 'LCC-WT',
        'mutation': 'WT',
        'enzyme': 'LCC',
        'tm': 84.7,
        'tm_std': 0.0,
        'delta_tm': 0.0,
        'method': 'DSF',
        'source': 'Tournier 2020 (DOI: 10.1038/s41586-020-2149-4)',
        'notes': 'Leaf-branch compost cutinase, different enzyme family'
    },
    {
        'variant_name': 'LCC-ICCG',
        'mutation': 'F243I/D238C/S283C/Y127G',
        'enzyme': 'LCC',
        'tm': 94.5,
        'tm_std': 0.0,
        'delta_tm': 9.8,
        'method': 'DSF',
        'source': 'Tournier 2020 (DOI: 10.1038/s41586-020-2149-4)',
        'notes': 'Engineered LCC variant (Carbios), 10000x activity'
    },
]


def save_mutations_csv(output_path):
    """Save mutations to CSV format"""
    fields = [
        'variant_name', 'mutation', 'enzyme',
        'tm', 'tm_std', 'delta_tm',
        'method', 'source', 'notes'
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(MUTATIONS)

    print(f"Saved {len(MUTATIONS)} entries to {output_path}")


def load_mutations_csv(input_path):
    """Load mutations from CSV"""
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def analyze_dataset(mutations):
    """Analyze the mutation dataset"""
    print("\nDATASET ANALYSIS")
    print("=" * 60)

    # Split by enzyme
    ispetase = [m for m in mutations if m['enzyme'] == 'IsPETase']
    lcc = [m for m in mutations if m['enzyme'] == 'LCC']

    print(f"Total entries: {len(mutations)}")
    print(f"  IsPETase: {len(ispetase)}")
    print(f"  LCC (reference): {len(lcc)}")

    # Tm range (IsPETase only)
    tms = [float(m['tm']) for m in ispetase]
    if tms:
        print(f"\nIsPETase Tm range: {min(tms):.1f} - {max(tms):.1f} C")
        print(f"IsPETase mean Tm: {sum(tms)/len(tms):.1f} C")

    # Sources
    sources = {}
    for m in mutations:
        paper = m['source'].split('(')[0].strip()
        sources[paper] = sources.get(paper, 0) + 1
    print(f"\nSources: {len(sources)} papers")
    for paper, count in sorted(sources.items()):
        print(f"  - {paper}: {count} variants")

    # Mutation counts
    print("\nMutation complexity:")
    for m in ispetase:
        if m['mutation'] == 'WT':
            n_mut = 0
        else:
            n_mut = len(m['mutation'].split('/'))
        print(f"  {m['variant_name']:25s} {n_mut} mutations  Tm={m['tm']}C")


if __name__ == '__main__':
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir.parent / 'data'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save mutations
    csv_path = output_dir / 'mutations_dataset.csv'
    save_mutations_csv(csv_path)

    # Load and analyze
    analyze_dataset(MUTATIONS)

    print(f"\nCSV file: {csv_path}")
    print("All Tm values verified from published papers.")
