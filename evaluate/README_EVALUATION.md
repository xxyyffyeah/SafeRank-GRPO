# SafeRec Model Evaluation Guide

Complete guide for evaluating recommendation models on the **SafeRec benchmark** with both **recommendation quality** and **safety metrics**.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Prerequisites](#prerequisites)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Evaluating SafeRec Models](#evaluating-saferec-models)
6. [Evaluating Baseline Models](#evaluating-baseline-models)
7. [Understanding the Output](#understanding-the-output)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The SafeRec evaluation framework measures two key aspects:

1. **Recommendation Quality**: How well the model recommends relevant movies
   - Recall@K, NDCG@K
   - Catalog match ratio (hallucination rate)

2. **Safety**: How well the model respects user sensitivities
   - Sensitivity DCG@K (lower is better)
   - Sensitive movie count and ratio
   - Measures violations of user trait constraints

**Evaluation Script**: [`eval_sft_val_safe.py`](eval_sft_val_safe.py)

---

## Quick Start

```bash
# 1. Navigate to project root
cd /home/jovyan/Rank-GRPO

# 2. Generate GT catalog (one-time setup)
python3 scripts/phase4_sft_eval/generate_gt_catalog.py \
    --saferec_json data/phase2_3_saferec/sft_train_24k_saferec.json \
    --output gt_catalog_complete.pkl \
    --verbose

# 3. Run evaluation
cd evaluate
python3 eval_sft_val_safe.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --model_root "../results/Qwen/Qwen2.5-0.5B-Instruct" \
    --dataset_path "../downloaded_datasets/processed_datasets/saferec_sft_dataset" \
    --catalog_path "../gt_catalog_complete.pkl" \
    --trait_sensitivity_path "../downloaded_datasets/movie_trait_sensitivity.json" \
    --title_mapping_path "../data/phase1_mapping/title_to_imdb.pkl" \
    --output_dir "figs_saferec"
```

---

## Prerequisites

### 1. Required Files

| File | Path | Description | How to Get |
|------|------|-------------|------------|
| **GT Catalog** | `gt_catalog.pkl` | 7,403 unique movies | Run `generate_gt_catalog.py` |
| **Trait Sensitivity** | `downloaded_datasets/movie_trait_sensitivity.json` | Safety data for 24,408 movies | Already in repo |
| **Title Mapping** | `data/phase1_mapping/title_to_imdb.pkl` | Title → IMDb ID mapping | Already in repo |
| **Dataset** | `downloaded_datasets/processed_datasets/saferec_sft_dataset/` | Test/validation data | Generated in Phase 2&3 |

### 2. Python Dependencies

```bash
pip install vllm datasets tqdm numpy matplotlib
```

### 3. Hardware

- **GPU**: Required for model inference (≥ 8GB VRAM for Qwen2.5-0.5B)
- **RAM**: 16GB+ recommended
- **Disk**: ~2GB for catalog and sensitivity data

---

## Evaluation Metrics

### Recommendation Quality Metrics

#### 1. Catalog Match Ratio
**What**: Percentage of recommendations that are real movies (not hallucinated)

**Formula**: `matched_movies / total_recommendations`

**Interpretation**:
- ✅ **> 90%**: Excellent (model rarely hallucinates)
- ⚠️  **70-90%**: Acceptable (some hallucinations)
- ❌ **< 70%**: Poor (high hallucination rate)

#### 2. Recall@K
**What**: Fraction of ground truth movies found in top-K recommendations

**Formula**: `|GT ∩ Top-K| / |GT|`

**Example**: If GT = {Movie A, Movie B, Movie C} and Top-10 contains {Movie A, Movie C}, then Recall@10 = 2/3 = 66.7%

**K values**: 5, 10, 15, 20

#### 3. NDCG@K (Normalized Discounted Cumulative Gain)
**What**: Quality of ranking (penalizes relevant movies ranked lower)

**Range**: [0.0, 1.0]

**Interpretation**:
- Higher = Better ranking quality
- Considers both relevance AND position

### Safety Metrics (Lower is Better)

#### 4. Sensitivity DCG@K
**What**: Discounted cumulative gain for sensitive movies (measures safety violations)

**Formula**: `DCG(sensitive_hits)`

**Interpretation**:
- **0.0**: Perfect safety (no sensitive movies recommended)
- **Higher**: More safety violations, especially at top positions

#### 5. Sensitive Movie Count@K
**What**: Number of sensitive movies in top-K recommendations

**Range**: [0, K]

**Interpretation**:
- **0**: No safety violations
- **Higher**: More movies that violate user sensitivity traits

#### 6. Sensitive Movie Ratio@K
**What**: Percentage of recommendations that are sensitive

**Formula**: `sensitive_count / K`

**Range**: [0.0, 1.0]

**Interpretation**:
- **0%**: Perfect safety
- **Higher %**: More safety violations

---

## Evaluating SafeRec Models

### Standard Evaluation (All Checkpoints)

```bash
cd evaluate
python3 eval_sft_val_safe.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --model_root "../results/Qwen/Qwen2.5-0.5B-Instruct" \
    --dataset_path "../downloaded_datasets/processed_datasets/saferec_sft_dataset" \
    --catalog_path "../gt_catalog_complete.pkl" \
    --trait_sensitivity_path "../downloaded_datasets/movie_trait_sensitivity.json" \
    --title_mapping_path "../data/phase1_mapping/title_to_imdb.pkl"
```

**What it does**:
- Evaluates checkpoints at steps 0, 200, 400, ...
- Generates recommendations for validation set
- Computes all metrics (quality + safety)
- Saves results to `analysis_state.json`

### Quick Evaluation (Last Checkpoint Only)

```bash
python3 eval_sft_val_safe.py \
    --model_root "../results/Qwen/Qwen2.5-0.5B-Instruct" \
    --dataset_path "../downloaded_datasets/processed_datasets/saferec_sft_dataset" \
    --catalog_path "../gt_catalog_complete.pkl" \
    --trait_sensitivity_path "../downloaded_datasets/movie_trait_sensitivity.json" \
    --title_mapping_path "../data/phase1_mapping/title_to_imdb.pkl" \
    --eval_last_only
```

**Saves time**: Only evaluates final checkpoint (useful for quick validation)

### Upload to Weights & Biases

```bash
python3 eval_sft_val_safe.py ... --upload_wandb
```

**Requirements**: W&B account and login (`wandb login`)

### Evaluating GRPO / Safe-GRPO Models (Full Checkpoints)

```bash
cd evaluate
python3 eval_sft_val_safe.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --model_root "../results/grpo/Qwen/Qwen2.5-0.5B-Instruct_lr1e-06_kl0.001_mu1" \
    --dataset_path "../downloaded_datasets/processed_datasets/saferec_sft_dataset" \
    --catalog_path "../gt_catalog_complete.pkl" \
    --trait_sensitivity_path "../downloaded_datasets/movie_trait_sensitivity.json" \
    --title_mapping_path "../data/phase1_mapping/title_to_imdb.pkl" \
    --output_dir "figs_saferec" \
    --baseline_model "../results/Qwen/Qwen2.5-0.5B-Instruct/checkpoint-800"
```

**`--baseline_model`**: checkpoint-0 使用该模型作为 baseline（而非 base model），通常设为 SFT checkpoint 路径。

### Evaluating LoRA Models (GDPO + LoRA)

LoRA checkpoint 只包含 adapter 权重，需要指定 base model 来加载：

```bash
cd evaluate
python3 eval_sft_val_safe.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --model_root "../results/safe_grpo/Qwen/Qwen2.5-0.5B-Instruct_gdpo_lora_r64_lr1e-06_kl0.001_mu1_lambda1.0_penalty1.0" \
    --dataset_path "../downloaded_datasets/processed_datasets/saferec_sft_dataset" \
    --catalog_path "../gt_catalog_complete.pkl" \
    --trait_sensitivity_path "../downloaded_datasets/movie_trait_sensitivity.json" \
    --title_mapping_path "../data/phase1_mapping/title_to_imdb.pkl" \
    --output_dir "figs_saferec" \
    --baseline_model "../results/Qwen/Qwen2.5-0.5B-Instruct/checkpoint-800" \
    --use_lora \
    --lora_base_model "../results/Qwen/Qwen2.5-0.5B-Instruct/checkpoint-800"
```

**LoRA 专用参数**：

| 参数 | 说明 |
|------|------|
| `--use_lora` | 标记 checkpoint 为 LoRA adapter |
| `--lora_base_model` | LoRA 的 base model 路径（必需），通常与 `--baseline_model` 一致 |

**工作原理**：
- checkpoint-0：加载 `--baseline_model`（SFT checkpoint）
- 其他 step：用 vLLM 的 `enable_lora=True` 加载 base model + LoRA adapter
- 不存在或不完整的 checkpoint 会自动跳过（同时检查 `config.json` 和 `adapter_config.json`）

---

## Evaluating Baseline Models

### Option 1: Using vLLM-Compatible Models

For any model compatible with vLLM (e.g., LLaMA, Mistral, Qwen):

```bash
python3 eval_sft_val_safe.py \
    --model_name "meta-llama/Llama-2-7b-chat-hf" \
    --model_root "../results/baseline_llama2" \
    --dataset_path "../downloaded_datasets/processed_datasets/saferec_sft_dataset" \
    --catalog_path "../gt_catalog_complete.pkl" \
    --trait_sensitivity_path "../downloaded_datasets/movie_trait_sensitivity.json" \
    --title_mapping_path "../data/phase1_mapping/title_to_imdb.pkl" \
    --eval_last_only
```

**Notes**:
- `--model_name`: HuggingFace model ID
- `--model_root`: Directory to save results (will be created)
- No training needed - evaluates base model directly

### Option 2: Evaluating OpenAI/Anthropic Models

For API-based models, create a custom evaluation script:

```python
# eval_baseline_api.py
import json
from datasets import load_from_disk
from openai import OpenAI

# Load dataset
dataset = load_from_disk("../downloaded_datasets/processed_datasets/saferec_sft_dataset/validation")

# Initialize API client
client = OpenAI(api_key="YOUR_API_KEY")

results = []
for sample in dataset:
    prompt = sample['prompt'][0]['content']

    # Call API
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    results.append({
        "sample_id": sample.get("sample_id"),
        "raw_output": response.choices[0].message.content
    })

# Save for post-processing
with open("baseline_gpt4_outputs.json", "w") as f:
    json.dump(results, f)
```

Then evaluate using the same metrics pipeline (see Advanced Usage below).

### Option 3: Evaluating Non-Safety Baselines

For models without safety training:

1. Run standard evaluation
2. Compare safety metrics (expect higher Sensitivity DCG/Ratio)
3. This demonstrates the value of SafeRec training

**Example comparison**:

| Model | Recall@20 | NDCG@20 | Sensitive Ratio@20 |
|-------|-----------|---------|-------------------|
| Base Qwen2.5 | 0.42 | 0.29 | **35.2%** ❌ |
| SafeRec Qwen2.5 | 0.68 | 0.47 | **5.1%** ✅ |

---

## Understanding the Output

### 1. Console Output

During evaluation, you'll see:

```
📂 Loading validation dataset from ../downloaded_datasets/...
Loaded catalog of size: 7403
🔒 Initializing SafetyOracle ...
[SafetyOracle] Loaded trait sensitivity for 24408 movies
Total unique contexts: 1127

🧠 Generating recommendations across checkpoints ...
Processing step 0 ...
Processing step 200 ...
...

🔍 Evaluating catalog match ratios ...
📊 Calculating Recall and NDCG ...
🔒 Calculating Sensitivity metrics (lower is better) ...
  Step 630 - Sensitivity DCG@20: 0.8234, Count: 1.12, Ratio: 5.6%

💾 Saving evaluation results ...
✅ Evaluation complete. Results saved to ../results/...
```

### 2. Output Files

All results are saved to `{model_root}/`:

#### `analysis_state.json`
Complete metrics for all checkpoints:

```json
{
  "log_history": [
    {
      "step": 630,
      "loss": 0.425,
      "eval_recall@5": 0.32,
      "eval_recall@10": 0.52,
      "eval_recall@20": 0.68,
      "eval_ndcg@5": 0.35,
      "eval_ndcg@20": 0.47,
      "eval_sensitivity_dcg@20": 0.8234,
      "eval_sensitive_count@20": 1.12,
      "eval_sensitive_ratio@20": 0.056,
      "catalog_match_ratio": 0.94
    }
  ]
}
```

#### `output.pkl`
Full recommendations for every sample (for detailed analysis):

```python
import pickle

with open("results/.../output.pkl", "rb") as f:
    data = pickle.load(f)

# Each item contains:
# - prompt: User conversation
# - groundtruth: True movies user liked
# - rec_after_step_630: Model recommendations
# - constraints: User sensitivity traits
```

#### `figs_saferec/`
Training loss curves (PNG files)

### 3. Key Metrics to Track

For **production deployment**, monitor:

1. ✅ **Sensitive Ratio@20 < 10%**: Most important safety metric
2. ✅ **Recall@20 > 0.50**: Good recommendation quality
3. ✅ **Catalog Match > 90%**: Low hallucination rate

---

## Advanced Usage

### Complete CLI Reference

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_name` | `Qwen/Qwen2.5-0.5B-Instruct` | 模型名称（HuggingFace ID） |
| `--model_root` | `../results/Qwen/Qwen2.5-0.5B-Instruct` | checkpoint 所在目录 |
| `--dataset_path` | `../processed_datasets/sft_dataset` | 数据集路径 |
| `--catalog_path` | `../gt_catalog.pkl` | GT catalog 文件 |
| `--trait_sensitivity_path` | `../downloaded_datasets/movie_trait_sensitivity.json` | 电影安全特征数据 |
| `--title_mapping_path` | `../data/phase1_mapping/title_to_imdb.pkl` | Title→IMDb 映射 |
| `--output_dir` | `figs` | 图表输出目录 |
| `--split` | `test` | 评估数据集分割：train/validation/test |
| `--eval_last_only` | - | 仅评估最新 checkpoint |
| `--baseline_model` | None | checkpoint-0 baseline 模型路径（默认用 `--model_name`） |
| `--use_lora` | - | checkpoint 为 LoRA adapter |
| `--lora_base_model` | None | LoRA base model 路径（`--use_lora` 时必需） |
| `--upload_wandb` | - | 上传结果到 Weights & Biases |
| `--wandb_project` | `sft_eval_val` | W&B 项目名 |

### Custom Metrics Computation

If you want to add new metrics:

```python
# eval_custom_metrics.py
import pickle
import numpy as np
from libs.safety_oracle import SafetyOracle

# Load evaluation output
with open("results/.../output.pkl", "rb") as f:
    data = pickle.load(f)

# Initialize oracle
oracle = SafetyOracle(...)

# Compute custom metric
custom_scores = []
for item in data:
    recs = item['rec_after_step_630'][:20]
    constraints = item['constraints']

    # Your custom logic
    score = compute_my_metric(recs, constraints, oracle)
    custom_scores.append(score)

print(f"Custom metric: {np.mean(custom_scores):.4f}")
```

### Batch Evaluation of Multiple Models

```bash
#!/bin/bash
# eval_all_models.sh

MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "meta-llama/Llama-2-7b-chat-hf"
)

for model in "${MODELS[@]}"; do
    echo "Evaluating $model..."
    python3 eval_sft_val_safe.py \
        --model_name "$model" \
        --model_root "../results/$(basename $model)" \
        --dataset_path "../downloaded_datasets/processed_datasets/saferec_sft_dataset" \
        --catalog_path "../gt_catalog_complete.pkl" \
        --trait_sensitivity_path "../downloaded_datasets/movie_trait_sensitivity.json" \
        --title_mapping_path "../data/phase1_mapping/title_to_imdb.pkl" \
        --eval_last_only
done
```

### Evaluating on Custom Datasets

If you have a different test set:

```python
# 1. Convert to HuggingFace format
from datasets import Dataset

custom_data = [
    {
        "prompt": [{"role": "user", "content": "..."}],
        "groundtruth_with_release_year": [["Movie", 2020], ...],
        "constraints": {"Anti-gore": True, ...},
        "seen_titles": ["Seen Movie 1", ...]
    },
    ...
]

dataset = Dataset.from_list(custom_data)
dataset.save_to_disk("custom_test_set")

# 2. Run evaluation
python3 eval_sft_val_safe.py \
    --dataset_path "custom_test_set" \
    ...
```

---

## Troubleshooting

### Error: "Catalog file not found"

**Solution**:
```bash
python3 scripts/phase4_sft_eval/generate_gt_catalog.py \
    --saferec_json data/phase2_3_saferec/sft_train_24k_saferec.json \
    --output gt_catalog_complete.pkl
```

### Error: "No module named 'vllm'"

**Solution**:
```bash
pip install vllm
```

### Error: "CUDA out of memory"

**Solution 1**: Reduce GPU memory usage
```python
# In eval_sft_val_safe.py, line 155-158
llm = LLM(
    model=model_to_load,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.6,  # Reduce from 0.8
    max_model_len=4096  # Reduce from 8192
)
```

**Solution 2**: Evaluate fewer checkpoints
```bash
python3 eval_sft_val_safe.py ... --eval_last_only
```

### Error: "Title mapping file not found"

**Solution**: Ensure you have the phase1 mapping file. If missing, you can disable title-based safety checks (will reduce safety metric accuracy):

```python
# Modify eval_sft_val_safe.py to use IMDb ID-based checks only
# Contact the SafeRec team for the mapping file
```

### Slow Evaluation

**Issue**: Evaluation takes > 30 minutes

**Solutions**:
1. Use `--eval_last_only` (evaluates only final checkpoint)
2. Reduce validation set size (subsample dataset)
3. Use faster GPU
4. Increase `tensor_parallel_size` if you have multiple GPUs

---

## Summary: Evaluation Checklist

- [ ] Generate GT catalog (`gt_catalog_complete.pkl`)
- [ ] Verify all required files exist
- [ ] Install dependencies (`vllm`, `datasets`, etc.)
- [ ] Run evaluation script
- [ ] Check `analysis_state.json` for metrics
- [ ] Monitor safety metrics (Sensitive Ratio@20 < 10%)
- [ ] Compare with baseline models
- [ ] Upload to W&B (optional)

---

## Citation

If you use this evaluation framework, please cite:

```bibtex
@dataset{saferec_benchmark_2026,
  title={SafeRec Benchmark: Safety-Aware Movie Recommendations with Dual Reasoning},
  author={SafeRec Team},
  year={2026},
  publisher={HuggingFace},
  url={https://huggingface.co/datasets/Dionysianspirit/saferec-benchmark}
}
```

---

## Support

For questions or issues:
- Open an issue on GitHub
- Check the [SafeRec Benchmark](https://huggingface.co/datasets/Dionysianspirit/saferec-benchmark) documentation
- Contact the SafeRec team

---

**Last Updated**: 2026-01-31
