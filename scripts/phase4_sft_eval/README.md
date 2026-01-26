# Phase 4: SafeRec SFT Evaluation

> **Note**: This directory contains utility scripts for Phase 4. For complete evaluation documentation, see **[`evaluate/README_EVALUATION.md`](../../evaluate/README_EVALUATION.md)**.

---

## Overview

Phase 4 evaluates trained SafeRec models on both **recommendation quality** and **safety metrics**.

**Main Evaluation Script**: [`evaluate/eval_sft_val_safe.py`](../../evaluate/eval_sft_val_safe.py)

---

## Quick Links

- 📖 **[Complete Evaluation Guide](../../evaluate/README_EVALUATION.md)** - Full documentation
- 🔧 **[Evaluation Script](../../evaluate/eval_sft_val_safe.py)** - Main evaluation code
- 📊 **[SafeRec Benchmark](https://huggingface.co/datasets/Dionysianspirit/saferec-benchmark)** - Dataset on HuggingFace

---

## Quick Start

### 1. Generate Ground Truth Catalog

```bash
# From project root
python3 scripts/phase4_sft_eval/generate_gt_catalog.py \
    --saferec_json data/phase2_3_saferec/sft_train_24k_saferec.json \
    --output gt_catalog_complete.pkl \
    --verbose
```

**Output**: `gt_catalog_complete.pkl` (7,403 unique movies)

### 2. Run Evaluation

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

---

## Files in This Directory

### `generate_gt_catalog.py`

**Purpose**: Extract unique movies from SafeRec dataset to create ground truth catalog

**Usage**:
```bash
python3 generate_gt_catalog.py \
    --saferec_json data/phase2_3_saferec/sft_train_24k_saferec.json \
    --output gt_catalog_complete.pkl \
    --verbose
```

**Output Statistics**:
```
Total samples processed: 21,425
Total unique movies: 7,403
Year range: 1902 - 2024
Average year: 1998.4
```

**What it does**:
1. Loads SafeRec JSON datasets (train/test/validation)
2. Extracts all movies from ground truth, recommendations, etc.
3. Creates deduplicated catalog of (movie_title, year) tuples
4. Saves as pickle file for fast loading during evaluation

---

## Evaluation Metrics

The evaluation computes **two categories** of metrics:

### 1. Recommendation Quality

- **Catalog Match Ratio**: % of recommendations that are real movies
- **Recall@K**: Fraction of ground truth found in top-K (K = 5, 10, 15, 20)
- **NDCG@K**: Ranking quality score

### 2. Safety Metrics (Lower is Better)

- **Sensitivity DCG@K**: Cumulative gain for sensitive movies
- **Sensitive Count@K**: Number of unsafe recommendations
- **Sensitive Ratio@K**: Percentage of unsafe recommendations

**See**: [Complete Metrics Documentation](../../evaluate/README_EVALUATION.md#evaluation-metrics)

---

## Evaluating Baseline Models

To compare SafeRec with baseline models, see [Baseline Evaluation Guide](../../evaluate/README_EVALUATION.md#evaluating-baseline-models)

---

## Full Documentation

👉 **[Complete Evaluation Guide](../../evaluate/README_EVALUATION.md)**

---

**Last Updated**: 2026-01-26
