#!/bin/bash
#
# Monitor trait assignment progress
#
# Usage: bash scripts/phase0_trait_assignment/monitor_progress.sh
#

echo "=== Trait Assignment Progress Monitor ==="
echo ""

# Check if output file exists
OUTPUT_FILE="/tmp/claude/-home-coder-Rank-GRPO/tasks/b57f13c.output"
if [ ! -f "$OUTPUT_FILE" ]; then
    OUTPUT_FILE="logs/trait_assignment_full_run.log"
fi

if [ ! -f "$OUTPUT_FILE" ]; then
    echo "❌ No running job found"
    exit 1
fi

echo "📊 Current Progress:"
echo ""

# Show last 50 lines
tail -50 "$OUTPUT_FILE"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Commands:"
echo "  - View full log: tail -f $OUTPUT_FILE"
echo "  - Check progress: bash scripts/phase0_trait_assignment/monitor_progress.sh"
echo "  - Kill job: kill \$(pgrep -f 'run_trait_assignment_pipeline.sh')"
