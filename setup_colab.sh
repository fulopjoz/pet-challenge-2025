#!/bin/bash
# Setup script for Google Colab
# Run: !bash setup_colab.sh

set -e

echo "=== PET Challenge 2025 - Colab Setup ==="

# Install dependencies
echo "Installing Python packages..."
pip install -q fair-esm esm numpy pandas scipy scikit-learn xgboost matplotlib seaborn

# Verify GPU
python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
"

# Verify ESM packages
python3 -c "
import esm as fair_esm
print('fair-esm:', fair_esm.__version__ if hasattr(fair_esm, '__version__') else 'installed')
"

# Check data files
echo ""
echo "Checking data files..."
for f in data/petase_challenge_data/pet-2025-wildtype-cds.csv \
         data/petase_challenge_data/predictive-pet-zero-shot-test-2025.csv; do
    if [ -f "$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f"
    fi
done

echo ""
echo "=== Setup complete ==="
echo ""
echo "Run the pipeline:"
echo "  1. python scripts/esm2_zero_shot_scoring.py   # ESM2-650M scoring (~5 min on T4)"
echo "  2. python scripts/esmc_scoring.py              # ESMC-600M scoring (~5 min on T4)"
echo "  3. python scripts/validate_scores.py           # Validate & compare models"
echo "  4. python scripts/generate_submission.py       # Generate submission (ensemble)"
echo ""
echo "For ML baseline (Tm prediction with known data):"
echo "  5. python scripts/extract_mutations.py         # Generate validation dataset"
echo "  6. python scripts/feature_extraction.py        # Extract sequence features"
echo "  7. python scripts/train_rf_baseline.py         # Train Ridge/RF models"
echo "  8. python scripts/alternative_models.py        # Compare all ML models"
