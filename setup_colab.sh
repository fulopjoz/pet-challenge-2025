#!/bin/bash
# Setup script for Google Colab
#
# Usage (from any directory):
#   !git clone https://github.com/fulopjoz/pet-challenge-2025.git
#   %cd pet-challenge-2025
#   !bash setup_colab.sh
#
# Or if you uploaded the folder:
#   %cd /content/pet-challenge-2025
#   !bash setup_colab.sh

set -e

echo "=== PET Challenge 2025 - Colab Setup ==="

# Auto-detect project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/data/petase_challenge_data" ]; then
    cd "$SCRIPT_DIR"
    echo "Project root: $SCRIPT_DIR"
elif [ -d "data/petase_challenge_data" ]; then
    echo "Project root: $(pwd)"
elif [ -d "pet-challenge-2025/data/petase_challenge_data" ]; then
    cd pet-challenge-2025
    echo "Project root: $(pwd)"
else
    echo "ERROR: Cannot find data directory."
    echo "Make sure you are in the pet-challenge-2025 folder:"
    echo "  %cd /content/pet-challenge-2025"
    exit 1
fi

# Install dependencies
# NOTE: fair-esm and esm (EvolutionaryScale) conflict on the 'esm' namespace.
# Install fair-esm first for ESM2, then esm for ESMC (overwrites).
# The notebook handles this in sequence.
echo ""
echo "Installing Python packages..."
pip install -q numpy pandas scipy scikit-learn xgboost matplotlib seaborn joblib
pip install -q fair-esm

echo ""
echo "NOTE: ESMC requires a separate install (pip install esm)."
echo "      The notebook handles this automatically in the ESMC section."

# Verify GPU
echo ""
python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
else:
    print('WARNING: No GPU detected!')
    print('  Go to Runtime > Change runtime type > T4 GPU')
    print('  CPU-only will be ~10x slower')
"

# Check data files
echo ""
echo "Checking data files..."
all_ok=true
for f in data/petase_challenge_data/pet-2025-wildtype-cds.csv \
         data/petase_challenge_data/predictive-pet-zero-shot-test-2025.csv \
         data/mutations_dataset.csv \
         data/features_matrix.csv \
         scripts/esm2_zero_shot_scoring.py \
         scripts/esmc_scoring.py \
         scripts/generate_submission.py \
         scripts/validate_scores.py; do
    if [ -f "$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f"
        all_ok=false
    fi
done

if [ "$all_ok" = true ]; then
    echo ""
    echo "=== All files present. Setup complete! ==="
else
    echo ""
    echo "=== Some files missing! Check your clone/upload. ==="
fi

echo ""
echo "Next steps (use the notebook or run manually):"
echo "  1. python scripts/esm2_zero_shot_scoring.py   # ESM2 scoring (~5 min on T4)"
echo "  2. python scripts/esmc_scoring.py              # ESMC scoring (~5 min on T4)"
echo "  3. python scripts/validate_scores.py           # Compare models"
echo "  4. python scripts/generate_submission.py       # Generate submission"
echo ""
echo "Or open PET_Challenge_2025_Pipeline.ipynb for the full guided notebook."
