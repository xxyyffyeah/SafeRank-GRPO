#!/bin/bash
#
# Complete Trait Assignment Pipeline
#
# Usage:
#   1. Create .env file with OPENAI_API_KEY
#   2. bash scripts/phase0_trait_assignment/run_trait_assignment_pipeline.sh [--test]
#

set -e  # Exit on error

# Load .env file if exists
if [ -f .env ]; then
    echo "📄 Loading environment from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# Check if test mode
TEST_MODE=false
MAX_SAMPLES=""
if [[ "$1" == "--test" ]]; then
    TEST_MODE=true
    MAX_SAMPLES="--max_samples 100"
    echo "🧪 Running in TEST MODE (100 samples only)"
fi

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY environment variable not set"
    echo ""
    echo "Please either:"
    echo "  1. Create .env file with: OPENAI_API_KEY=sk-..."
    echo "  2. Or export manually: export OPENAI_API_KEY=\"sk-...\""
    echo ""
    echo "See .env.example for template"
    exit 1
fi

echo "================================"
echo "Trait Assignment Pipeline"
echo "================================"
echo ""

# Step 1: Filter samples
echo "📋 Step 1: Filtering SFT samples (GT >= 3)..."
python3 scripts/phase0_trait_assignment/filter_sft_samples.py \
    --input_path downloaded_datasets/processed_datasets/sft_dataset/train \
    --output_path data/phase0_trait_assignment/sft_filtered_8k.json \
    --min_groundtruth 3 \
    --target_samples 8000

echo ""
echo "✅ Step 1 complete"
echo ""

# Step 2: Assign traits via GPT
if [ "$TEST_MODE" = true ]; then
    OUTPUT_PATH="data/phase0_trait_assignment/sft_with_assigned_traits_test.json"
else
    OUTPUT_PATH="data/phase0_trait_assignment/sft_with_assigned_traits.json"
fi

echo "🤖 Step 2: Assigning traits via ChatGPT API..."
echo "   Model: gpt-5.2"
echo "   Output: $OUTPUT_PATH"
echo ""

python3 scripts/phase0_trait_assignment/assign_traits_via_gpt.py \
    --input_path data/phase0_trait_assignment/sft_filtered_8k.json \
    --output_path "$OUTPUT_PATH" \
    --traits_path traits_with_imdb_parentguide_weights.json \
    --model gpt-5.2 \
    --temperature 0.3 \
    $MAX_SAMPLES

echo ""
echo "✅ Step 2 complete"
echo ""

# Step 3: Filter violating groundtruth
if [ "$TEST_MODE" = true ]; then
    FINAL_OUTPUT="data/phase0_trait_assignment/saferec_sft_8k_dataset_test.json"
else
    FINAL_OUTPUT="data/phase0_trait_assignment/saferec_sft_8k_dataset.json"
fi

echo "🔍 Step 3: Filtering violating groundtruth..."
python3 scripts/phase0_trait_assignment/filter_violating_groundtruth.py \
    --input_path "$OUTPUT_PATH" \
    --output_path "$FINAL_OUTPUT" \
    --trait_sensitivity_path downloaded_datasets/movie_trait_sensitivity.json \
    --title_mapping_path data/phase1_mapping/title_to_imdb.pkl \
    --min_groundtruth_after_filter 1

echo ""
echo "✅ Step 3 complete"
echo ""

# Step 4: Analyze distribution
if [ "$TEST_MODE" = true ]; then
    STATS_DIR="data/phase0_trait_assignment/trait_stats_test/"
else
    STATS_DIR="data/phase0_trait_assignment/trait_stats/"
fi

echo "📊 Step 4: Analyzing trait distribution..."
python3 scripts/phase0_trait_assignment/analyze_trait_distribution.py \
    --input_path "$FINAL_OUTPUT" \
    --output_dir "$STATS_DIR"

echo ""
echo "✅ Step 4 complete"
echo ""

# Summary
echo "================================"
echo "🎉 Pipeline Complete!"
echo "================================"
echo ""
echo "Output files:"
echo "  - Filtered samples: data/phase0_trait_assignment/sft_filtered_8k.json"
echo "  - With assigned traits: $OUTPUT_PATH"
echo "  - Final dataset: $FINAL_OUTPUT"
echo "  - Statistics: $STATS_DIR"
echo ""

if [ "$TEST_MODE" = true ]; then
    echo "⚠️  This was a TEST run (100 samples)"
    echo "    To run full pipeline, remove --test flag"
else
    echo "✅ Full pipeline completed successfully"
    echo ""
    echo "Next steps:"
    echo "  - Review trait distribution: cat $STATS_DIR/stats.json | jq .trait_distribution"
    echo "  - Continue to Phase 2: See docs/SAFEREC_IMPLEMENTATION_PLAN.md"
fi
