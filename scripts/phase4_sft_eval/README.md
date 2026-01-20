# Phase 4: SafeRec SFT Evaluation

This directory contains scripts for evaluating the trained SafeRec SFT model.

## Overview

After training the SafeRec model (Phase 2-3), this phase evaluates its performance on the validation set, computing standard recommendation metrics (Recall@K, NDCG@K) as well as catalog match ratios.

## Files

- **`generate_gt_catalog.py`**: Extracts all unique movies from SafeRec dataset to create a ground truth catalog
- **`run_eval.sh`**: Convenience script to run evaluation with correct paths

## Step 1: Generate Ground Truth Catalog

The catalog file (`gt_catalog.pkl`) is required to filter out hallucinated or misspelled movies during evaluation.

```bash
# From project root
python scripts/phase4_sft_eval/generate_gt_catalog.py --verbose
```

This will:
- Extract 7,370+ unique movies from the SafeRec dataset
- Create `gt_catalog.pkl` in the project root
- Show statistics about the catalog

**Output:**
```
Total unique movies: 7,370
Year range: 1902 - 2024
Average year: 1998.4
```

## Step 2: Run Evaluation

```bash
# From scripts/phase4_sft_eval directory
bash run_eval.sh

# Or run directly from project root
cd evaluate
python eval_sft_val.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --model_root "../results/Qwen/Qwen2.5-0.5B-Instruct" \
    --dataset_path "../downloaded_datasets/processed_datasets/saferec_sft_dataset" \
    --catalog_path "../gt_catalog.pkl" \
    --output_dir "figs_saferec"
```

## What the Evaluation Does

The evaluation script (`evaluate/eval_sft_val.py`) performs the following:

1. **Load catalog and dataset** - 667 validation samples
2. **Find checkpoints** - Evaluates step 0, 200, 400, ..., 630
3. **Generate recommendations** - Uses vLLM for efficient batch inference
4. **Compute metrics**:
   - **Catalog Match Ratio**: % of recommendations that exist in the catalog
   - **Recall@K**: How many ground truth movies appear in top-K recommendations
   - **NDCG@K**: Quality of ranking (considers position of correct recommendations)

## Understanding the Metrics

### Catalog Match Ratio
- **What it measures**: Percentage of model recommendations that are real movies (not hallucinated)
- **Good score**: > 90%
- **Poor score**: < 70% (model is hallucinating or misspelling movie titles)

### Recall@K
- **What it measures**: Fraction of ground truth movies found in top-K recommendations
- **Example**: If GT has 3 movies and top-10 recommendations contain 2 of them, Recall@10 = 2/3 = 66.7%
- **K values**: 5, 10, 15, 20

### NDCG@K (Normalized Discounted Cumulative Gain)
- **What it measures**: Quality of ranking (penalizes relevant movies that appear lower in the list)
- **Range**: 0.0 to 1.0
- **Higher is better**: Means good movies are ranked higher

## Output Files

After evaluation, check these files in `results/Qwen/Qwen2.5-0.5B-Instruct/`:

- **`analysis.json`**: Metrics summary for all checkpoints
- **`output.pkl`**: Full recommendations for all samples
- **`trainer_state.json`**: Training history
- **`figs_saferec/`**: Loss plots

Example `analysis.json`:
```json
{
  "catalog_ratios": [
    {"step": 0, "mean": 0.85, "std": 0.12},
    {"step": 200, "mean": 0.92, "std": 0.08},
    ...
  ],
  "avg_metrics": {
    "0": {
      "recall": {"5": 0.15, "10": 0.28, "15": 0.35, "20": 0.42},
      "ndcg": {"5": 0.18, "10": 0.24, "15": 0.27, "20": 0.29}
    },
    "630": {
      "recall": {"5": 0.32, "10": 0.52, "15": 0.61, "20": 0.68},
      "ndcg": {"5": 0.35, "10": 0.42, "15": 0.45, "20": 0.47}
    }
  }
}
```

## Requirements

The evaluation requires:
- **vLLM**: For efficient batch inference
  ```bash
  pip install vllm
  ```
- **GPU**: Sufficient memory to load the model (Qwen2.5-0.5B needs ~2GB)
- **Catalog file**: `gt_catalog.pkl` (generated in Step 1)

## Troubleshooting

### Error: "No module named 'vllm'"
```bash
pip install vllm
```

### Error: "Catalog file not found"
```bash
python scripts/phase4_sft_eval/generate_gt_catalog.py
```

### Out of GPU memory
Reduce `gpu_memory_utilization` in the script:
```python
llm = LLM(model=..., gpu_memory_utilization=0.6)  # Default is 0.8
```

## Expected Runtime

- **Catalog generation**: < 1 minute
- **Evaluation** (7 checkpoints × 667 samples):
  - With GPU: ~10-20 minutes
  - Depends on GPU speed and model size

## Integration with W&B (Optional)

To upload results to Weights & Biases:
```bash
python eval_sft_val.py ... --upload_wandb
```

This requires W&B account and login:
```bash
wandb login
```
