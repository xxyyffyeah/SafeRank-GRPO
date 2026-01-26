#!/bin/bash
#
# Process All SFT Datasets Through Phase 2 & 3
#
# For each SFT dataset (train, test, validation):
#   Step 1: Generate SafeRec dataset (constraint injection + safety filtering + CoT)
#   Step 2: Convert to HuggingFace format
#
# Usage:
#   bash scripts/phase2_3_saferec/process_all_sft_datasets.sh
#

set -e

echo "=========================================="
echo "Phase 2 & 3: SafeRec Dataset Generation"
echo "=========================================="
echo ""

# Paths
INPUT_DIR="data/phase0_trait_assignment/expanded"
SAFEREC_DIR="data/phase2_3_saferec"
HF_OUTPUT_DIR="downloaded_datasets/processed_datasets/saferec_sft_dataset"

# Create output directories
mkdir -p "$SAFEREC_DIR"
mkdir -p "$HF_OUTPUT_DIR"

# Parameters
INJECTION_RATE=1.0
RISK_THRESHOLD=0.66

# Function to process a single dataset
process_dataset() {
    local SPLIT=$1
    local INPUT_FILE=$2
    local PREFIX=$3

    echo ""
    echo "=========================================="
    echo "Processing: $SPLIT"
    echo "=========================================="
    echo ""

    # Step 1: Generate SafeRec dataset
    echo "Step 1: Generating SafeRec dataset..."
    python3 scripts/phase2_3_saferec/generate_saferec_dataset.py \
        --input_path "$INPUT_FILE" \
        --output_path "$SAFEREC_DIR/${PREFIX}_saferec.json" \
        --injection_rate $INJECTION_RATE \
        --risk_threshold $RISK_THRESHOLD

    echo ""
    echo "Step 1 complete: $SAFEREC_DIR/${PREFIX}_saferec.json"
    echo ""

    # Step 2: Convert to HuggingFace format
    echo "Step 2: Converting to HuggingFace format..."
    python3 scripts/phase2_3_saferec/convert_to_hf_dataset.py \
        --input_path "$SAFEREC_DIR/${PREFIX}_saferec.json" \
        --output_path "$HF_OUTPUT_DIR/$SPLIT" \
        --split_name "$SPLIT"

    echo ""
    echo "Step 2 complete: $HF_OUTPUT_DIR/$SPLIT"
    echo ""

    echo "$SPLIT dataset processing complete!"
    echo ""
}

# ========================================
# Process Train Dataset
# ========================================
process_dataset \
    "train" \
    "$INPUT_DIR/sft_train_24k_final.json" \
    "sft_train_24k"

# ========================================
# Process Test Dataset
# ========================================
process_dataset \
    "test" \
    "$INPUT_DIR/sft_test_1k_final.json" \
    "sft_test_1k"

# ========================================
# Process Validation Dataset
# ========================================
process_dataset \
    "validation" \
    "$INPUT_DIR/sft_val_1k_final.json" \
    "sft_val_1k"

# ========================================
# Summary
# ========================================
echo ""
echo "=========================================="
echo "All SFT Datasets Processed!"
echo "=========================================="
echo ""
echo "SafeRec JSON files:"
echo "  - $SAFEREC_DIR/sft_train_24k_saferec.json"
echo "  - $SAFEREC_DIR/sft_test_1k_saferec.json"
echo "  - $SAFEREC_DIR/sft_val_1k_saferec.json"
echo ""
echo "HuggingFace datasets:"
echo "  - $HF_OUTPUT_DIR/train"
echo "  - $HF_OUTPUT_DIR/test"
echo "  - $HF_OUTPUT_DIR/validation"
echo ""
echo "Next steps:"
echo "  1. Load datasets with: load_from_disk('$HF_OUTPUT_DIR/{split}')"
echo "  2. Train SFT model: python train_sft_safe.py --dataset_path $HF_OUTPUT_DIR"
echo ""
