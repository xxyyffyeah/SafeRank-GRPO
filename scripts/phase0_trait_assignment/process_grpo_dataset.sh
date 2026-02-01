#!/bin/bash
#
# Process GRPO Dataset Pipeline - Complete 4-Step Processing
#
# Task:
# Extract 72000 samples from grpo -> saferec_grpo
#
# The dataset goes through 4 steps:
#   Step 1: filter_sft_samples.py
#   Step 2: assign_traits_via_gpt.py (uses GRPO-specific prompt)
#   Step 3: filter_violating_groundtruth.py
#   Step 4: analyze_trait_distribution.py
#
# Usage:
#   bash scripts/phase0_trait_assignment/process_grpo_dataset.sh [--test]
#

set -e  # Exit on error

# Load .env file if exists
if [ -f .env ]; then
    echo "Loading environment from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# Check if test mode
TEST_MODE=false
MAX_SAMPLES=""
if [[ "$1" == "--test" ]]; then
    TEST_MODE=true
    MAX_SAMPLES="--max_samples 50"
    echo "Running in TEST MODE (limited samples)"
fi

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable not set"
    echo ""
    echo "Please either:"
    echo "  1. Create .env file with: OPENAI_API_KEY=sk-..."
    echo "  2. Or export manually: export OPENAI_API_KEY=\"sk-...\""
    exit 1
fi

echo "=========================================="
echo "GRPO Dataset Pipeline"
echo "=========================================="
echo ""

# Paths
GRPO_PATH="downloaded_datasets/processed_datasets/grpo/grpo_dataset/train"
OUTPUT_DIR="data/phase0_trait_assignment/expanded"
TRAIT_SENSITIVITY_PATH="downloaded_datasets/movie_trait_sensitivity.json"
TITLE_MAPPING_PATH="data/phase1_mapping/title_to_imdb.pkl"
TRAITS_PATH="traits_with_imdb_parentguide_weights.json"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to process a single dataset through all 4 steps
process_dataset() {
    local INPUT_PATH=$1
    local SPLIT_NAME=$2
    local TARGET_SAMPLES=$3
    local MIN_GT=$4
    local OUTPUT_PREFIX=$5

    echo ""
    echo "=========================================="
    echo "Processing: $SPLIT_NAME ($TARGET_SAMPLES samples, min_gt=$MIN_GT)"
    echo "=========================================="
    echo ""

    # File paths for this dataset
    local FILTERED_FILE="$OUTPUT_DIR/${OUTPUT_PREFIX}_filtered.json"
    local TRAITS_FILE="$OUTPUT_DIR/${OUTPUT_PREFIX}_with_traits.json"
    local FINAL_FILE="$OUTPUT_DIR/${OUTPUT_PREFIX}_final.json"
    local STATS_DIR="$OUTPUT_DIR/${OUTPUT_PREFIX}_stats/"

    # Step 1: Filter samples
    echo "Step 1: Filtering $SPLIT_NAME samples..."
    python3 scripts/phase0_trait_assignment/filter_sft_samples.py \
        --input_path "$INPUT_PATH" \
        --output_path "$FILTERED_FILE" \
        --min_groundtruth "$MIN_GT" \
        --target_samples "$TARGET_SAMPLES" \
        --split_name "$SPLIT_NAME"

    echo ""
    echo "Step 1 complete: $FILTERED_FILE"
    echo ""

    # Step 2: Assign traits via GPT
    echo "Step 2: Assigning traits via ChatGPT API..."
    echo "   Model: gpt-5.2"
    echo "   Output: $TRAITS_FILE"
    echo ""

    python3 scripts/phase0_trait_assignment/assign_traits_via_gpt.py \
        --input_path "$FILTERED_FILE" \
        --output_path "$TRAITS_FILE" \
        --traits_path "$TRAITS_PATH" \
        --model gpt-5.2 \
        --temperature 0.3 \
        $MAX_SAMPLES

    echo ""
    echo "Step 2 complete: $TRAITS_FILE"
    echo ""

    # Step 3: Filter violating groundtruth
    echo "Step 3: Filtering violating groundtruth..."
    python3 scripts/phase0_trait_assignment/filter_violating_groundtruth.py \
        --input_path "$TRAITS_FILE" \
        --output_path "$FINAL_FILE" \
        --trait_sensitivity_path "$TRAIT_SENSITIVITY_PATH" \
        --title_mapping_path "$TITLE_MAPPING_PATH" \
        --min_groundtruth_after_filter 1

    echo ""
    echo "Step 3 complete: $FINAL_FILE"
    echo ""

    # Step 4: Analyze trait distribution
    echo "Step 4: Analyzing trait distribution..."
    python3 scripts/phase0_trait_assignment/analyze_trait_distribution.py \
        --input_path "$FINAL_FILE" \
        --output_dir "$STATS_DIR"

    echo ""
    echo "Step 4 complete: $STATS_DIR"
    echo ""

    echo "Dataset $SPLIT_NAME processing complete!"
    echo ""
}

# ========================================
# Process Dataset 4: GRPO (72000, no GT filter)
# ========================================
process_dataset \
    "$GRPO_PATH" \
    "grpo" \
    72000 \
    0 \
    "grpo_72k"

# ========================================
# Summary
# ========================================
echo ""
echo "=========================================="
echo "GRPO Dataset Pipeline Complete!"
echo "=========================================="
echo ""
echo "Output files in $OUTPUT_DIR:"
echo ""
echo "GRPO (72k):"
echo "  - Filtered: ${OUTPUT_DIR}/grpo_72k_filtered.json"
echo "  - With traits: ${OUTPUT_DIR}/grpo_72k_with_traits.json"
echo "  - Final: ${OUTPUT_DIR}/grpo_72k_final.json"
echo "  - Stats: ${OUTPUT_DIR}/grpo_72k_stats/"
echo ""
echo "Next steps:"
echo "  1. Review trait distributions in grpo_72k_stats/ directory"
echo "  2. Run convert_to_hf_dataset.py in phase2_3 to convert to HuggingFace format"
echo ""
