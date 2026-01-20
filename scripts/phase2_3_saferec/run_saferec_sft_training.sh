#!/bin/bash
# SafeRec SFT Training Script
# Run with: bash scripts/phase2_3_saferec/run_saferec_sft_training.sh

set -e  # Exit on error

# Configuration
DATASET_PATH="downloaded_datasets/processed_datasets/saferec_sft_dataset"
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
LOG_DIR="./logs"

# Training hyperparameters (consistent with train_sft.py defaults)
NUM_EPOCHS=10
TRAIN_BATCH_SIZE=12
EVAL_BATCH_SIZE=12
GRAD_ACCUM_STEPS=8
LEARNING_RATE=5e-5
WARMUP_RATIO=0.05
OPTIM="paged_adamw_8bit"
LR_SCHEDULER_TYPE="cosine"
MAX_LENGTH=1024
DATASET_NUM_PROC=64
SAVE_STEPS=50
LOGGING_STEPS=10
EVAL_STEPS=10
SEED=3407

# Create log directory
mkdir -p ${LOG_DIR}

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/saferec_sft_${TIMESTAMP}.log"

echo "======================================"
echo "SafeRec SFT Training"
echo "======================================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_PATH}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Batch size: ${TRAIN_BATCH_SIZE} (grad accum: ${GRAD_ACCUM_STEPS})"
echo "Learning rate: ${LEARNING_RATE}"
echo "Log file: ${LOG_FILE}"
echo "======================================"
echo ""

# Check if dataset exists
if [ ! -d "${DATASET_PATH}" ]; then
    echo "Error: Dataset not found at ${DATASET_PATH}"
    echo "Please run the data generation pipeline first:"
    echo "  python scripts/phase2_3_saferec/generate_saferec_dataset.py"
    echo "  python scripts/phase2_3_saferec/convert_to_hf_dataset.py"
    exit 1
fi

# Run training with nohup
echo "Starting training in background..."
echo "To monitor progress: tail -f ${LOG_FILE}"
echo ""

nohup python train_sft_safe.py \
    --dataset_path ${DATASET_PATH} \
    --model_name ${MODEL_NAME} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --warmup_ratio ${WARMUP_RATIO} \
    --optim ${OPTIM} \
    --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
    --max_length ${MAX_LENGTH} \
    --dataset_num_proc ${DATASET_NUM_PROC} \
    --save_strategy steps \
    --save_steps ${SAVE_STEPS} \
    --logging_steps ${LOGGING_STEPS} \
    --eval_strategy steps \
    --eval_steps ${EVAL_STEPS} \
    --seed ${SEED} \
    --bf16 \
    --gradient_checkpointing \
    > ${LOG_FILE} 2>&1 &

TRAIN_PID=$!

echo "✅ Training started with PID: ${TRAIN_PID}"
echo ""
echo "Useful commands:"
echo "  # Monitor training log:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "  # Check training process:"
echo "  ps aux | grep ${TRAIN_PID}"
echo ""
echo "  # Stop training:"
echo "  kill ${TRAIN_PID}"
echo ""
echo "  # View GPU usage:"
echo "  nvidia-smi"
echo ""

# Wait a moment and check if training started successfully
sleep 3

if ps -p ${TRAIN_PID} > /dev/null; then
    echo "✅ Training is running successfully!"
    echo "Check log: ${LOG_FILE}"
else
    echo "❌ Training failed to start. Check log for errors:"
    echo "  cat ${LOG_FILE}"
    exit 1
fi
