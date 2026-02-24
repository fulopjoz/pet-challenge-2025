#!/usr/bin/env python3
"""Validate all Jupyter notebooks for GitHub rendering compatibility.

Checks:
1. nbformat validation (structure, cell IDs, etc.)
2. Widgets metadata has required 'state' key
3. nbformat_minor >= 5 (cell IDs required)
4. No missing cell IDs

Usage:
    python scripts/validate_notebooks.py [--fix]
"""
import json
import sys
import uuid
import glob
import os

def validate_notebook(path, fix=False):
    errors = []
    fixed = []

    with open(path) as f:
        nb = json.load(f)

    # Check nbformat version
    major = nb.get('nbformat', 0)
    minor = nb.get('nbformat_minor', 0)
    if major != 4 or minor < 5:
        if fix:
            nb['nbformat'] = 4
            nb['nbformat_minor'] = 5
            fixed.append(f'Upgraded nbformat to 4.5 (was {major}.{minor})')
        else:
            errors.append(f'nbformat {major}.{minor} < 4.5 — cell IDs not supported')

    # Check widgets metadata
    meta = nb.get('metadata', {})
    if 'widgets' in meta:
        for widget_key, widget_data in meta['widgets'].items():
            if isinstance(widget_data, dict) and 'state' not in widget_data:
                if fix:
                    # Wrap existing entries in a state key
                    meta['widgets'][widget_key] = {
                        'state': widget_data,
                        'version_major': 2,
                        'version_minor': 0
                    }
                    fixed.append(f'Added missing state key to widgets/{widget_key}')
                else:
                    errors.append(f'widgets/{widget_key} missing "state" key — GitHub will fail to render')

    # Check cell IDs
    for i, cell in enumerate(nb.get('cells', [])):
        if 'id' not in cell:
            if fix:
                cell['id'] = str(uuid.uuid4())[:8]
                fixed.append(f'Added cell ID to cell {i}')
            else:
                errors.append(f'Cell {i} missing "id" field')

    # Clean Colab-specific metadata that can cause issues
    if fix:
        colab_keys = ['executionInfo', 'outputId', 'colab']
        for cell in nb.get('cells', []):
            cell_meta = cell.get('metadata', {})
            for key in colab_keys:
                if key in cell_meta:
                    del cell_meta[key]
                    fixed.append(f'Removed Colab metadata "{key}" from cell')

    if fix and fixed:
        with open(path, 'w') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)

    return errors, fixed


def main():
    fix = '--fix' in sys.argv

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    notebooks = sorted(glob.glob(os.path.join(project_root, '*.ipynb')))

    if not notebooks:
        print('No notebooks found in project root.')
        return 0

    all_ok = True
    for nb_path in notebooks:
        name = os.path.basename(nb_path)
        errors, fixed = validate_notebook(nb_path, fix=fix)

        if errors:
            all_ok = False
            print(f'FAIL  {name}')
            for e in errors:
                print(f'  - {e}')
        elif fixed:
            print(f'FIXED {name}')
            for f_msg in fixed[:10]:
                print(f'  + {f_msg}')
            if len(fixed) > 10:
                print(f'  ... and {len(fixed) - 10} more fixes')
        else:
            print(f'OK    {name}')

    if all_ok:
        print('\nAll notebooks are GitHub-compatible.')
        return 0
    else:
        print(f'\nFailed. Run with --fix to auto-repair.')
        return 1


if __name__ == '__main__':
    sys.exit(main())
