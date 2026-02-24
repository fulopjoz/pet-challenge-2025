#!/usr/bin/env python3
"""
Extract verified mutation data from PETase literature.
Creates CSV with Tm, activity, and provenance data.

Loads all data from verified_tm_data.csv (produced by the automated extraction
pipeline: extract_tm_from_pdfs.py + build_verified_dataset.py).

Primary source: Brott et al. 2022 (Eng. Life Sci., DOI: 10.1002/elsc.202100105)
Additional: Lu 2022 (Nature), Son 2019 (ACS Catal), Cui 2021 (ACS Catal), Bell 2022 (Nat Catal)
"""

import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
VERIFIED_CSV = DATA_DIR / "validation" / "verified_tm_data.csv"

# DOI-to-citation mapping for source column
DOI_CITATIONS = {
    "10.1002/elsc.202100105": "Brott 2022 (DOI: 10.1002/elsc.202100105)",
    "10.1021/acscatal.9b00568": "Son 2019 (DOI: 10.1021/acscatal.9b00568)",
    "10.1021/acscatal.0c05126": "Cui 2021 (DOI: 10.1021/acscatal.0c05126)",
    "10.1038/s41586-022-04599-z": "Lu 2022 (DOI: 10.1038/s41586-022-04599-z)",
    "10.1038/s41929-022-00821-3": "Bell 2022 (DOI: 10.1038/s41929-022-00821-3)",
}

# Activity columns passed through from verified CSV
ACTIVITY_FIELDS = [
    "rel_activity_30C_pct", "rel_activity_40C_pct",
    "rel_activity_50C_pct", "rel_activity_60C_pct",
    "rel_activity_60C_72h_pct",
]


def load_verified_mutations() -> list[dict]:
    """Load Tm + activity data from the verified extraction pipeline CSV.

    Returns list of mutation dicts with standard fields.
    """
    if not VERIFIED_CSV.exists():
        print(f"WARNING: {VERIFIED_CSV} not found.")
        print("Run the extraction pipeline first:")
        print("  python scripts/extract_tm_from_pdfs.py --all")
        print("  python scripts/build_verified_dataset.py")
        return []

    mutations = []
    with open(VERIFIED_CSV, newline="") as f:
        for row in csv.DictReader(f):
            doi = row.get("source_doi", "")
            source = DOI_CITATIONS.get(doi, f"DOI: {doi}")
            table = row.get("source_table", "")
            if table:
                source += f" [{table}]"

            entry = {
                "variant_name": row.get("variant_name", ""),
                "mutation": row.get("mutation", ""),
                "enzyme": row.get("enzyme", "IsPETase"),
                "tm": float(row.get("tm", 0) or 0),
                "tm_std": float(row.get("tm_std", 0) or 0),
                "delta_tm": float(row.get("delta_tm", 0) or 0),
                "method": row.get("method", ""),
                "source": source,
                "notes": row.get("notes", ""),
                "substrate": row.get("substrate", ""),
                "enzyme_conc_nM": row.get("enzyme_conc_nM", ""),
            }

            # Pass through activity columns
            for field in ACTIVITY_FIELDS:
                val = row.get(field, "")
                entry[field] = val

            mutations.append(entry)

    print(f"Loaded {len(mutations)} entries from {VERIFIED_CSV.name}")

    return mutations


def save_mutations_csv(mutations, output_path):
    """Save mutations to CSV format."""
    fields = [
        'variant_name', 'mutation', 'enzyme',
        'tm', 'tm_std', 'delta_tm',
        'method', 'source', 'notes',
        'substrate', 'enzyme_conc_nM',
    ] + ACTIVITY_FIELDS
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(mutations)
    print(f"Saved {len(mutations)} entries to {output_path}")


def load_mutations_csv(input_path):
    """Load mutations from CSV."""
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


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

    # Activity data summary
    has_activity = sum(1 for m in mutations if m.get("rel_activity_30C_pct", "") != "")
    print(f"\nEntries with activity data: {has_activity}")

    print("\nMutation complexity:")
    for m in mutations:
        mut = m['mutation']
        if mut == 'WT':
            n_mut = 0
        else:
            n_mut = len(mut.split('/'))
        act = " +activity" if m.get("rel_activity_30C_pct", "") != "" else ""
        print(f"  {m['variant_name']:25s} {n_mut:2d} mutations  Tm={m['tm']}C{act}")


if __name__ == '__main__':
    output_dir = DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load from verified pipeline CSV (or empty if not yet generated)
    mutations = load_verified_mutations()

    if mutations:
        csv_path = output_dir / 'mutations_dataset.csv'
        save_mutations_csv(mutations, csv_path)
        analyze_dataset(mutations)
        print(f"\nCSV file: {csv_path}")
        print("All Tm + activity values loaded from verified extraction pipeline.")
    else:
        print("\nNo verified data available. Generate it first:")
        print("  python scripts/extract_tm_from_pdfs.py --all")
        print("  python scripts/build_verified_dataset.py")
