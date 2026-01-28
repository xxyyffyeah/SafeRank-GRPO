#!/bin/bash
TRL_PATH=$(python -c "import trl, os; print(os.path.dirname(trl.__file__))")

# Copy trainer file
cp libs/trl/rank_grpo_trainer.py $TRL_PATH/trainer/

# Update top-level __init__.py
cp libs/trl/__module__init__.py $TRL_PATH/__init__.py

# Update trainer/__init__.py to include RankGRPOTrainer
TRAINER_INIT="$TRL_PATH/trainer/__init__.py"
if ! grep -q "rank_grpo_trainer" "$TRAINER_INIT"; then
    # Add to _import_structure (after grpo_trainer line)
    sed -i '/"grpo_trainer": \["GRPOTrainer"\],/a\    "rank_grpo_trainer": ["RankGRPOTrainer"],' "$TRAINER_INIT"
    echo "✅ Updated trainer/__init__.py"
fi

echo "✅ Rank-GRPO trainer installed to $TRL_PATH"