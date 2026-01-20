#!/bin/bash
#
# Run SafeRec SFT Evaluation
#
# This script runs the standard eval_sft_val.py with SafeRec-specific paths
#

set -e

echo "========================================"
echo "SafeRec SFT Evaluation"
echo "========================================"

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration (all paths relative to project root)
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
MODEL_ROOT="$PROJECT_ROOT/results/Qwen/Qwen2.5-0.5B-Instruct"
DATASET_PATH="$PROJECT_ROOT/downloaded_datasets/processed_datasets/saferec_sft_dataset"
CATALOG_PATH="$PROJECT_ROOT/gt_catalog.pkl"
OUTPUT_DIR="$PROJECT_ROOT/figs_saferec"

echo "Model: $MODEL_NAME"
echo "Model root: $MODEL_ROOT"
echo "Dataset: $DATASET_PATH"
echo "Catalog: $CATALOG_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "========================================"
echo ""

# Check if catalog exists
if [ ! -f "$CATALOG_PATH" ]; then
    echo "❌ Error: Catalog file not found at $CATALOG_PATH"
    echo "Please run: python scripts/phase4_sft_eval/generate_gt_catalog.py"
    exit 1
fi

# Navigate to evaluate directory
cd "$PROJECT_ROOT/evaluate"

# Set PYTHONPATH to include libs directory
export PYTHONPATH="$PROJECT_ROOT/libs:$PYTHONPATH"

# Run evaluation
echo "🚀 Starting evaluation..."
python eval_sft_val.py \
    --model_name "$MODEL_NAME" \
    --model_root "$MODEL_ROOT" \
    --dataset_path "$DATASET_PATH" \
    --catalog_path "$CATALOG_PATH" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "✅ Evaluation complete!"
echo "Check results in: $MODEL_ROOT"
echo "========================================"
