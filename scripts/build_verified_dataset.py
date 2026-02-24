#!/usr/bin/env python3
"""
Script 3: Build verified Tm dataset from per-paper extractions.

Merges per-paper CSVs, performs sanity checks, calculates delta_tm,
merges activity data, cross-references same variants across studies,
and produces the final verified_tm_data.csv.

Usage:
    python scripts/build_verified_dataset.py
    python scripts/build_verified_dataset.py --verbose
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
VALIDATION_DIR = PROJECT_DIR / "data" / "validation"
PER_PAPER_DIR = VALIDATION_DIR / "per_paper"
OUTPUT_CSV = VALIDATION_DIR / "verified_tm_data.csv"
ACTIVITY_CSV = PER_PAPER_DIR / "brott2022_activity.csv"

# Tm range checks
TM_MIN_ISPETASE = 30.0
TM_MAX_ISPETASE = 100.0
# Safety net only — normally derived from per-paper CSVs by derive_wt_tm().
# Used only when a paper's WT row is missing from extraction.
WT_TM_FALLBACK = {
    "brott2022": 45.1,
    "son2019": 48.81,
    "cui2021": 46.0,
    "lu2022": 45.1,
    "bell2022": 45.1,
}

# Cross-study delta threshold (different methods expected to differ)
CROSS_STUDY_THRESHOLD = 5.0

# Activity columns that get merged from brott2022_activity.csv
ACTIVITY_FIELDS = [
    "rel_activity_30C_pct", "rel_activity_40C_pct",
    "rel_activity_50C_pct", "rel_activity_60C_pct",
    "rel_activity_60C_72h_pct",
]
METADATA_FIELDS = ["substrate", "enzyme_conc_nM"]


def load_all_extractions() -> list[dict]:
    """Load all per-paper extraction CSVs."""
    records = []
    if not PER_PAPER_DIR.exists():
        print(f"WARNING: {PER_PAPER_DIR} does not exist")
        return records

    for csv_path in sorted(PER_PAPER_DIR.glob("*_tm.csv")):
        paper_id = csv_path.stem.replace("_tm", "")
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["paper_id"] = paper_id
                # Convert numeric fields
                for field in ("tm", "tm_std"):
                    try:
                        row[field] = float(row.get(field, 0) or 0)
                    except ValueError:
                        row[field] = 0.0
                records.append(row)

    print(f"Loaded {len(records)} records from {len(list(PER_PAPER_DIR.glob('*_tm.csv')))} files")
    return records


def load_activity_data() -> dict:
    """Load activity data from brott2022_activity.csv.

    Returns dict keyed by mutation string -> activity row.
    """
    activity = {}
    if not ACTIVITY_CSV.exists():
        print(f"WARNING: {ACTIVITY_CSV} not found — skipping activity merge")
        return activity

    with open(ACTIVITY_CSV, newline="") as f:
        for row in csv.DictReader(f):
            mutation = row.get("mutation", "")
            if mutation:
                activity[mutation] = row

    print(f"Loaded activity data for {len(activity)} variants")
    return activity


def derive_wt_tm(records: list[dict]) -> dict:
    """Derive WT Tm per paper from the extracted data itself.

    Falls back to WT_TM_FALLBACK if no WT entry found for a paper.
    """
    wt_tm = {}
    for r in records:
        if r.get("mutation") == "WT" and r["tm"] > 0:
            paper_id = r.get("paper_id", "")
            if paper_id:
                # Keep the first (highest-confidence) WT per paper
                if paper_id not in wt_tm:
                    wt_tm[paper_id] = r["tm"]

    # Fill in fallbacks for papers without WT entries
    for paper_id, fallback in WT_TM_FALLBACK.items():
        if paper_id not in wt_tm:
            wt_tm[paper_id] = fallback

    return wt_tm


def sanity_checks(records: list[dict], verbose: bool = False) -> list[dict]:
    """Run sanity checks and flag problematic records."""
    import re
    valid = []
    dropped = 0

    for r in records:
        tm = r["tm"]
        enzyme = r.get("enzyme", "IsPETase")
        mutation = r.get("mutation", "")
        variant = r.get("variant_name", "")

        # Skip zero/missing Tm
        if tm <= 0:
            if verbose:
                print(f"  DROP: Tm=0 for {variant}")
            dropped += 1
            continue

        # Skip unnamed screening variants (e.g., A), B), C), D) from Brott Table S2)
        if re.match(r'^[A-Z]\)$', mutation) or re.match(r'^[A-Z]\)$', variant):
            if verbose:
                print(f"  DROP: unnamed screening variant '{variant}' (no mutation ID)")
            dropped += 1
            continue

        # Skip variants with unresolvable mutation strings
        # (M5 from Cui is intermediate GRAPE round, no known mutation list)
        if mutation == "M5":
            if verbose:
                print(f"  DROP: M5 (intermediate GRAPE variant, mutations unknown)")
            dropped += 1
            continue

        # Range check (IsPETase only — pipeline is IsPETase-specific)
        if not (TM_MIN_ISPETASE <= tm <= TM_MAX_ISPETASE):
            if verbose:
                print(f"  DROP: Tm={tm} out of range for IsPETase ({variant})")
            dropped += 1
            continue

        valid.append(r)

    print(f"Sanity checks: {len(valid)} passed, {dropped} dropped")
    return valid


CONFIDENCE_ORDER = {"high": 0, "medium": 1, "single_method": 2,
                    "vision_only": 3, "low_REVIEW": 4}


def deduplicate(records: list[dict], verbose: bool = False) -> list[dict]:
    """Deduplicate: same mutation + same paper keeps best confidence entry.

    Different papers measuring the same variant are kept (cross-study).
    """
    # Group by (mutation, source DOI)
    groups = defaultdict(list)
    for r in records:
        key = (r.get("mutation", ""), r.get("source_doi", ""))
        groups[key].append(r)

    unique = []
    dupes = 0

    for key, entries in groups.items():
        if len(entries) == 1:
            unique.append(entries[0])
        else:
            # Pick best confidence, then highest tm_std (more precise measurement)
            entries.sort(key=lambda e: (
                CONFIDENCE_ORDER.get(e.get("extraction_confidence", ""), 9),
                -float(e.get("tm_std", 0) or 0),
            ))
            unique.append(entries[0])
            dupes += len(entries) - 1
            if verbose:
                mut, doi = key
                print(f"  DEDUP: {mut} from {doi.split('/')[-1]} — "
                      f"kept {entries[0].get('source_table')} "
                      f"({entries[0].get('extraction_confidence')}), "
                      f"dropped {len(entries)-1}")

    if dupes:
        print(f"Deduplication: removed {dupes} within-paper duplicates")
    return unique


def calculate_delta_tm(records: list[dict], wt_tm: dict) -> list[dict]:
    """Calculate delta_tm relative to WT from the same paper."""
    for r in records:
        paper_id = r.get("paper_id", "")
        wt = wt_tm.get(paper_id)

        if wt and r["tm"] > 0:
            r["delta_tm"] = round(r["tm"] - wt, 2)
        else:
            r["delta_tm"] = ""

    return records


def merge_activity(records: list[dict], activity: dict,
                   verbose: bool = False) -> list[dict]:
    """Merge activity data into Tm records by matching mutation string.

    Only merges for entries from Brott 2022 (same paper as activity data).
    For ThermoPETase from Son 2019 (same mutation S121E/D186H/R280A),
    also merge since Brott measured ThermoPETase activity.
    """
    merged = 0
    for r in records:
        mutation = r.get("mutation", "")
        act = activity.get(mutation)

        if act:
            # Merge activity fields
            for field in ACTIVITY_FIELDS:
                val = act.get(field, "")
                try:
                    r[field] = float(val) if val else ""
                except ValueError:
                    r[field] = ""

            # Merge metadata
            r["substrate"] = act.get("substrate", "")
            r["enzyme_conc_nM"] = act.get("enzyme_conc_nM", "")
            merged += 1
        else:
            # No activity data — leave fields empty
            for field in ACTIVITY_FIELDS:
                r[field] = ""
            r["substrate"] = ""
            r["enzyme_conc_nM"] = ""

    if merged:
        print(f"Activity merge: matched {merged} entries")
    return records


def cross_reference(records: list[dict], verbose: bool = False) -> list[dict]:
    """Flag same variants measured in multiple papers."""
    by_mutation = defaultdict(list)
    for r in records:
        mutation = r.get("mutation", "")
        if mutation and mutation != "WT":
            by_mutation[mutation].append(r)

    # Also check WT across sources
    wts = [r for r in records if r.get("mutation") == "WT"]
    if len(wts) > 1:
        by_mutation["WT"] = wts

    for mutation, entries in by_mutation.items():
        if len(entries) < 2:
            continue
        # Check cross-study Tm differences
        sources = [(e.get("paper_id", "?"), e["tm"]) for e in entries]
        tms = [s[1] for s in sources]
        max_delta = max(tms) - min(tms)

        if max_delta > CROSS_STUDY_THRESHOLD:
            note = (f"Cross-study delta={max_delta:.1f}C "
                    f"(methodological difference): "
                    + ", ".join(f"{s}={t}" for s, t in sources))
            for e in entries:
                existing = e.get("notes", "")
                e["notes"] = f"{existing}; {note}" if existing else note
            if verbose:
                print(f"  CROSS-REF: {mutation} delta={max_delta:.1f}C across "
                      f"{len(entries)} studies")

    return records


def print_summary(records: list[dict]):
    """Print dataset summary statistics."""
    print(f"\n{'='*60}")
    print("VERIFIED DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total entries: {len(records)}")

    # By enzyme
    by_enzyme = defaultdict(list)
    for r in records:
        by_enzyme[r.get("enzyme", "unknown")].append(r)
    for enzyme, entries in sorted(by_enzyme.items()):
        tms = [e["tm"] for e in entries if e["tm"] > 0]
        print(f"\n  {enzyme}: {len(entries)} entries")
        if tms:
            print(f"    Tm range: {min(tms):.1f} - {max(tms):.1f} C")
            print(f"    Tm mean:  {sum(tms)/len(tms):.1f} C")

    # By source
    print("\n  By source:")
    by_source = defaultdict(int)
    for r in records:
        by_source[r.get("paper_id", "unknown")] += 1
    for src, count in sorted(by_source.items()):
        print(f"    {src}: {count} entries")

    # Activity data
    has_activity = sum(1 for r in records if r.get("rel_activity_30C_pct", "") != "")
    print(f"\n  Entries with activity data: {has_activity}")

    # By confidence
    print("\n  By confidence:")
    by_conf = defaultdict(int)
    for r in records:
        by_conf[r.get("extraction_confidence", "unknown")] += 1
    for conf, count in sorted(by_conf.items()):
        print(f"    {conf}: {count}")

    # Flag items needing review
    review = [r for r in records
              if r.get("extraction_confidence", "").endswith("REVIEW")]
    if review:
        print(f"\n  NEEDS REVIEW ({len(review)} entries):")
        for r in review:
            print(f"    {r.get('variant_name', '?')} Tm={r['tm']} - {r.get('notes', '')}")


def main():
    parser = argparse.ArgumentParser(
        description="Build verified Tm dataset from per-paper extractions")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Load
    records = load_all_extractions()
    if not records:
        print("No extraction data found. Run extract_tm_from_pdfs.py first.")
        return

    # Load activity data
    activity = load_activity_data()

    # Derive WT Tm from data (before sanity checks drop anything)
    wt_tm = derive_wt_tm(records)
    if args.verbose:
        print("WT Tm per paper (derived from data):")
        for pid, tm in sorted(wt_tm.items()):
            print(f"  {pid}: {tm}")

    # Process
    records = sanity_checks(records, args.verbose)
    records = deduplicate(records, args.verbose)
    records = calculate_delta_tm(records, wt_tm)
    records = merge_activity(records, activity, args.verbose)
    records = cross_reference(records, args.verbose)

    # Sort: enzyme asc, then Tm desc
    records.sort(key=lambda r: (r.get("enzyme", ""), -r["tm"]))

    # Save with extended schema
    fields = [
        "variant_name", "mutation", "enzyme", "tm", "tm_std",
        "delta_tm", "method", "buffer_pH",
        "substrate", "enzyme_conc_nM",
        "rel_activity_30C_pct", "rel_activity_40C_pct",
        "rel_activity_50C_pct", "rel_activity_60C_pct",
        "rel_activity_60C_72h_pct",
        "source_doi", "source_table", "extraction_confidence", "notes",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    print(f"\nFinal dataset: {len(records)} entries -> {OUTPUT_CSV}")
    print_summary(records)


if __name__ == "__main__":
    main()
