#!/usr/bin/env python3
"""
Extract Tm values from all downloaded PDFs
"""

import pdfplumber
import csv
from pathlib import Path

# All extracted Tm values
mutations = []
BASE_DIR = Path(__file__).resolve().parent

## 1. Brott 2022 Supplementary Material
print("="*60)
print("Extracting from: Engineering and evaluation of thermostable IsPETase...")
print("="*60)

pdf_path = BASE_DIR / "Engineering and evaluation of thermostable IsPETase variants for PET degradation-sup-0001-suppmat.pdf"
with pdfplumber.open(str(pdf_path)) as pdf:
    # Table S2 on page 6
    page = pdf.pages[5]
    tables = page.extract_tables()
    
    for tbl in tables:
        if tbl and len(tbl) > 1:
            # Check if it's the Tm table
            header = tbl[0]
            if header and 'Melting point' in str(header):
                print(f"\nTable S2 - Melting points:")
                print("-" * 50)
                for row in tbl[1:]:
                    if row and len(row) == 3:
                        variant = row[1].strip()
                        tm_str = row[2].strip()
                        
                        # Parse Tm
                        tm, tm_std = 0.0, 0.0
                        if '±' in tm_str:
                            parts = tm_str.split('±')
                            tm = float(parts[0].strip())
                            tm_std = float(parts[1].strip().replace('°C', '').strip())
                        else:
                            try:
                                tm = float(tm_str.replace('°C', '').strip())
                            except:
                                continue
                        
                        mutations.append({
                            'variant_name': f"Brott2022_{variant}",
                            'mutation': variant.upper(),
                            'enzyme': 'IsPETase',
                            'tm': tm,
                            'tm_std': tm_std,
                            'delta_tm': 0.0,  # Will calculate relative to WT
                            'method': 'nanoDSF',
                            'source': 'Brott 2022 (Supp Table S2)',
                            'notes': ''
                        })
                        print(f"  {variant}: {tm} ± {tm_std} °C")
    
    # Table S5 on page 8
    page = pdf.pages[7]
    tables = page.extract_tables()
    
    for tbl in tables:
        if tbl and len(tbl) > 1:
            header = tbl[0]
            if header and 'This study' in str(header):
                print(f"\nTable S5 - Compared melting points:")
                print("-" * 50)
                for row in tbl[1:]:
                    if row and len(row) > 1:
                        variant = str(row[0]).strip()
                        tm_str = str(row[1]).strip()
                        
                        # Skip empty rows or headers
                        if 'variant' in variant.lower() or not tm_str or '-' in tm_str:
                            continue
                        
                        # Parse Tm
                        tm, tm_std = 0.0, 0.0
                        if '±' in tm_str:
                            parts = tm_str.split('±')
                            tm = float(parts[0].strip())
                            tm_std = float(parts[1].strip().replace('°C', '').strip())
                        else:
                            try:
                                tm = float(tm_str.replace('°C', '').strip())
                            except:
                                continue
                        
                        # Clean variant name
                        mutation = None
                        if 'WT' in variant:
                            mutation = 'WT'
                        elif 'TM' in variant:
                            # Extract mutations from IsPETaseTMN233C/S282C format
                            mutation = variant.replace('IsPETaseTM', '').replace('IsPETase', '')
                        if not mutation:
                            continue
                        
                        mutations.append({
                            'variant_name': f"Brott2022_Comparison_{variant}",
                            'mutation': mutation,
                            'enzyme': 'IsPETase',
                            'tm': tm,
                            'tm_std': tm_std,
                            'delta_tm': 0.0,
                            'method': 'nanoDSF (This study)',
                            'source': 'Brott 2022 (Supp Table S5)',
                            'notes': f"Comparison table: {variant}"
                        })
                        print(f"  {variant}: {tm} ± {tm_std} °C")

## 2. Computational redesign (GRAPE strategy)
print("\n" + "="*60)
print("Extracting from: Computational redesign of a PETase...")
print("="*60)

pdf_path = BASE_DIR / "Computational redesign of a PETase for plastic biodegradation_si_001.pdf"
with pdfplumber.open(str(pdf_path)) as pdf:
    for page_num, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text and 'ΔT' in text and 'Table S4' in text:
            print(f"\nFound Table S4 on page {page_num+1}")
            print("-" * 50)
            tables = page.extract_tables()
            
            for tbl in tables:
                if tbl and len(tbl) > 1:
                    header = tbl[0]
                    if header and 'ΔT' in str(header):
                        # This is Table S4
                        for row in tbl[1:]:
                            if row and len(row) > 1:
                                mutation = str(row[0]).strip()
                                delta_tm_str = str(row[1]).strip()
                                
                                # Parse ΔTm
                                delta_tm = 0.0
                                try:
                                    delta_tm = float(delta_tm_str.replace('ΔTm', '').replace('ΔT', '').strip())
                                except:
                                    continue
                                
                                mutations.append({
                                    'variant_name': f"GRAPE2021_{mutation}",
                                    'mutation': mutation,
                                    'enzyme': 'IsPETase',
                                    'tm': 0.0,  # ΔTm only
                                    'tm_std': 0.0,
                                    'delta_tm': delta_tm,
                                    'method': 'thermofluor',
                                    'source': 'Computational redesign (Table S4)',
                                    'notes': 'ΔTm given, absolute Tm not reported'
                                })
                                print(f"  {mutation}: ΔTm = {delta_tm} °C")
                        break
            break

# Save to CSV
print("\n" + "="*60)
print(f"Total mutations extracted: {len(mutations)}")
print("="*60)

if mutations:
    # Sort by tm (non-zero values first)
    mutations_sorted = sorted(mutations, key=lambda x: (x['tm'], x['delta_tm']), reverse=True)
    
    output_csv = BASE_DIR / 'validation_mutations.csv'
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['variant_name', 'mutation', 'enzyme', 'tm', 'tm_std', 'delta_tm', 'method', 'source', 'notes'])
        writer.writeheader()
        writer.writerows(mutations_sorted)

    print(f"\n✅ Saved {len(mutations_sorted)} mutations to {output_csv}")
    
    # Calculate WT tm delta
    wt_tm = 0.0
    for m in mutations:
        if 'WT' in m['mutation'] and m['tm'] > 0:
            wt_tm = m['tm']
            print(f"\nWT Tm reference: {wt_tm} °C")
            break
    
    # Calculate delta tm for entries that have absolute Tm
    for m in mutations:
        if m['tm'] > 0 and wt_tm > 0 and m['delta_tm'] == 0.0:
            m['delta_tm'] = m['tm'] - wt_tm
    
    # Re-save with calculated deltas
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['variant_name', 'mutation', 'enzyme', 'tm', 'tm_std', 'delta_tm', 'method', 'source', 'notes'])
        writer.writeheader()
        writer.writerows(mutations_sorted)
    
    print("\nTop 10 variants by Tm:")
    print("-" * 60)
    for m in mutations_sorted[:10]:
        if m['tm'] > 0:
            print(f"  {m['mutation']:30s} | Tm: {m['tm']:6.2f} | ΔTm: {m['delta_tm']:+6.2f} °C")
else:
    print("\n❌ No mutations extracted")
