#!/usr/bin/env python3
"""
Load verified mutation data from mutations_dataset.csv.

The CSV is produced by the petase_tournament extraction pipeline
(extract_tm_from_pdfs.py + build_verified_dataset.py) and copied here.
This script loads and analyzes it — no hardcoded data.

Primary source: Brott et al. 2022 (Eng. Life Sci., DOI: 10.1002/elsc.202100105)
Additional: Son 2019 (ACS Catal), Bell 2022 (Nat Catal)
"""

import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
MUTATIONS_CSV = DATA_DIR / "mutations_dataset.csv"


def load_mutations_csv(input_path=MUTATIONS_CSV):
    """Load mutations from CSV."""
    if not input_path.exists():
        print(f"ERROR: {input_path} not found.")
        print("Copy from petase_tournament pipeline output:")
        print("  cp petase_tournament/data/mutations_dataset.csv data/")
        return []
    with open(input_path, "r") as f:
        return list(csv.DictReader(f))


def analyze_dataset(mutations):
    """Analyze the mutation dataset."""
    print("\nDATASET ANALYSIS")
    print("=" * 60)

    print(f"Total entries: {len(mutations)} (all IsPETase)")

    tms = [float(m['tm']) for m in mutations if float(m.get('tm', 0)) > 0]
    if tms:
        print(f"\nTm range: {min(tms):.1f} - {max(tms):.1f} C")
        print(f"Mean Tm:  {sum(tms)/len(tms):.1f} C")

    sources = {}
    for m in mutations:
        paper = m['source'].split('(')[0].strip()
        sources[paper] = sources.get(paper, 0) + 1
    print(f"\nSources: {len(sources)} papers")
    for paper, count in sorted(sources.items()):
        print(f"  - {paper}: {count} variants")

    has_activity = sum(1 for m in mutations if m.get("rel_activity_30C_pct", "") != "")
    print(f"\nEntries with activity data: {has_activity}")

    print("\nMutation complexity:")
    for m in mutations:
        mut = m['mutation']
        n_mut = 0 if mut == 'WT' else len(mut.split('/'))
        act = " +activity" if m.get("rel_activity_30C_pct", "") != "" else ""
        print(f"  {m['variant_name']:25s} {n_mut:2d} mutations  Tm={m['tm']}C{act}")


if __name__ == '__main__':
    mutations = load_mutations_csv()
    if mutations:
        analyze_dataset(mutations)
        print(f"\nCSV file: {MUTATIONS_CSV}")
        print("All Tm values from verified extraction pipeline.")
    else:
        print("\nNo data. See instructions above.")
