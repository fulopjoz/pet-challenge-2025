#!/usr/bin/env bash
# PET Challenge 2025 - Local environment setup
# Usage: bash setup.sh
set -euo pipefail

VENV_DIR=".venv"
KERNEL_NAME="pet-challenge"
KERNEL_DISPLAY="PET Challenge 2025"

echo "=== PET Challenge 2025 - Local Setup ==="
echo

# 1. Check uv
if ! command -v uv &>/dev/null; then
    echo "ERROR: uv is not installed."
    echo "  Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  Docs:    https://docs.astral.sh/uv/"
    exit 1
fi
echo "[1/6] uv found: $(uv --version)"

# 2. Check Python >= 3.10
PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python >= 3.10 not found."
    echo "  Install Python 3.10+ and ensure it is on your PATH."
    exit 1
fi
echo "[2/6] Python found: $PYTHON ($($PYTHON --version))"

# 3. Create venv
if [ -d "$VENV_DIR" ]; then
    echo "[3/6] Venv already exists at $VENV_DIR (reusing)"
else
    uv venv "$VENV_DIR" --python "$PYTHON"
    echo "[3/6] Created venv at $VENV_DIR"
fi

# 4. Install dependencies
uv pip install --python "$VENV_DIR/bin/python" -r requirements_local.txt
echo "[4/6] Installed CPU-only dependencies"

# 5. Register Jupyter kernel
"$VENV_DIR/bin/python" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY"
echo "[5/6] Registered Jupyter kernel: $KERNEL_DISPLAY"

# 6. Check mafft (needed by Conservation notebook)
echo
if command -v mafft &>/dev/null; then
    echo "[6/6] mafft found: $(mafft --version 2>&1 | head -1)"
else
    echo "[6/6] WARNING: mafft not found (required by Conservation_Scoring_Pipeline.ipynb)"
    echo "  Install:"
    if [ "$(uname)" = "Darwin" ]; then
        echo "    brew install mafft"
    else
        echo "    sudo apt install mafft"
    fi
fi

# 7. Verify data files
echo
MISSING=0
for f in data/petase_challenge_data/predictive-pet-zero-shot-test-2025.csv \
         data/petase_challenge_data/pet-2025-wildtype-cds.csv; do
    if [ ! -f "$f" ]; then
        echo "WARNING: Missing data file: $f"
        MISSING=1
    fi
done
if [ "$MISSING" -eq 0 ]; then
    echo "Data files verified."
fi

# Done
echo
echo "=== Setup complete ==="
echo
echo "Next steps:"
echo "  1. Open VS Code in this directory"
echo "  2. Open any .ipynb notebook"
echo "  3. Select kernel: \"$KERNEL_DISPLAY\""
echo "  4. Run All"
echo
echo "Notebooks (CPU-compatible):"
echo "  - Conservation_Scoring_Pipeline.ipynb  (~1 min, requires mafft)"
echo "  - Approach_Comparison.ipynb            (compares approaches)"
echo "  - PET_Challenge_2025_Pipeline_v2.ipynb (GPU cells skipped locally)"
