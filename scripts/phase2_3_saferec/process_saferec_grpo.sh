#!/bin/bash
#
# Process SafeRec GRPO Dataset Through Phase 2 & 3
#
# Takes Phase 0 GRPO output (grpo_72k_final.json) and:
#   Step 1: Inject constraints into prompts (generate_saferec_dataset.py --mode grpo)
#   Step 2: Convert to HuggingFace format (convert_to_hf_dataset.py)
#
# Note: GT filtering is already done in Phase 0 (filter_violating_groundtruth.py).
#       This script only handles constraint injection + HF conversion.
#
# Usage:
#   bash scripts/phase2_3_saferec/process_saferec_grpo.sh
#

set -e

echo "=========================================="
echo "Phase 2 & 3: SafeRec GRPO Dataset"
echo "=========================================="
echo ""

# Paths
INPUT_DIR="data/phase0_trait_assignment/expanded"
SAFEREC_DIR="data/phase2_3_saferec"
HF_OUTPUT_DIR="downloaded_datasets/processed_datasets/saferec_grpo_dataset"

# Create output directories
mkdir -p "$SAFEREC_DIR"
mkdir -p "$HF_OUTPUT_DIR"

# Parameters
INJECTION_RATE=1.0
RISK_THRESHOLD=0.66

INPUT_FILE="$INPUT_DIR/grpo_72k_final.json"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Phase 0 output not found: $INPUT_FILE"
    echo "Run Phase 0 first: bash scripts/phase0_trait_assignment/process_grpo_dataset.sh"
    exit 1
fi

echo "Input: $INPUT_FILE"
echo ""

# Step 1: Generate SafeRec GRPO dataset (constraint injection only)
echo "=========================================="
echo "Step 1: Injecting constraints into prompts"
echo "=========================================="
echo ""

python3 scripts/phase2_3_saferec/generate_saferec_dataset.py \
    --mode grpo \
    --input_path "$INPUT_FILE" \
    --output_path "$SAFEREC_DIR/grpo_72k_saferec.json" \
    --injection_rate $INJECTION_RATE \
    --risk_threshold $RISK_THRESHOLD

echo ""
echo "Step 1 complete: $SAFEREC_DIR/grpo_72k_saferec.json"
echo ""

# Step 2: Convert to HuggingFace format
echo "=========================================="
echo "Step 2: Converting to HuggingFace format"
echo "=========================================="
echo ""

python3 scripts/phase2_3_saferec/convert_to_hf_dataset.py \
    --input_path "$SAFEREC_DIR/grpo_72k_saferec.json" \
    --output_path "$HF_OUTPUT_DIR/train" \
    --split_name "train"

echo ""
echo "Step 2 complete: $HF_OUTPUT_DIR/train"
echo ""

# Summary
echo "=========================================="
echo "SafeRec GRPO Dataset Complete!"
echo "=========================================="
echo ""
echo "SafeRec JSON:      $SAFEREC_DIR/grpo_72k_saferec.json"
echo "HuggingFace train: $HF_OUTPUT_DIR/train"
echo ""
echo "Usage:"
echo "  from datasets import load_from_disk"
echo "  dataset = load_from_disk('$HF_OUTPUT_DIR/train')"
echo ""
echo "Training:"
echo "  accelerate launch train_rank_grpo_safe.py \\"
echo "      --train_path $HF_OUTPUT_DIR ..."
echo ""
