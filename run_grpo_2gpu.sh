#!/bin/bash
set -e

mkdir -p logs

MAX_RETRIES=5
RETRY=0

while [ $RETRY -lt $MAX_RETRIES ]; do
    echo "[$(date)] Attempt $((RETRY+1))/$MAX_RETRIES"

    RESUME_FLAG=""
    if [ $RETRY -gt 0 ]; then
        RESUME_FLAG="--resume"
    fi

    set +e
    accelerate launch --num_processes 2 train_rank_grpo.py \
        --train_path downloaded_datasets/processed_datasets/grpo/grpo_dataset \
        --model_name Qwen/Qwen2.5-0.5B-Instruct \
        --sft_checkpoint 800 \
        --reward_func exp_inf \
        --mu 1 \
        --lr 1e-6 \
        --kl_beta 1e-3 \
        --adam_beta1 0.9 \
        --adam_beta2 0.99 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --num_train_epochs 2 \
        --gradient_accumulation_steps 4 \
        --save_strategy steps \
        --save_steps 200 \
        --logging_steps 10 \
        --use_vllm \
        --vllm_mode colocate \
        --vllm_gpu_memory_utilization 0.35 \
        --vllm_tensor_parallel_size 1 \
        --max_prompt_length 2048 \
        --max_completion_length 1024 \
        --num_generations 8 \
        --seed 3407 \
        --bf16 \
        --gradient_checkpointing \
        --wandb_project rank_grpo \
        --catalog_path gt_catalog_complete.pkl \
        $RESUME_FLAG \
        2>&1 | tee -a logs/rank_grpo_2gpu.txt
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Training completed successfully."
        exit 0
    fi

    RETRY=$((RETRY+1))
    echo "[$(date)] Training crashed (exit $EXIT_CODE). Retrying in 30s..."
    sleep 30
done

echo "[$(date)] Max retries reached. Giving up."
exit 1
