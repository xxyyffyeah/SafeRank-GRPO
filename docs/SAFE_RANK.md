# Safe Training

## 1. Overview

**Rank-GRPO** ([Zhu et al., 2025](https://arxiv.org/abs/2506.05889)) is a reinforcement learning method for training LLM-based conversational recommender systems. Its key innovation is **per-rank independent advantage computation**: each recommendation position has its own advantage value, so the RL signal for rank *k* only affects the tokens that generated the movie at position *k*.

The safe training pipeline has two stages:

1. **Safe SFT** — Supervised fine-tuning on the SafeRec dataset (constraint-filtered ground truth, constraint-injected prompts). This grounds the model in the recommendation catalog while teaching it to respect user sensitivity constraints through data curation alone.
2. **Safe-Rank-GRPO** — RL fine-tuning with per-rank safety penalties, GDPO decoupled normalization, and count reward. This further sharpens safety alignment beyond what SFT achieves.

See [SafeRec Dataset Curation](./SAFEREC_IMPLEMENTATION_PLAN.md) for how the training data is prepared.

---

## 2. Safe SFT

### 2.1 What Safe SFT Does

Safe SFT fine-tunes a pretrained LLM (e.g., Qwen2.5-0.5B-Instruct) on the SafeRec SFT dataset. Compared to the original SFT on Reddit-v2, the SafeRec dataset has two key differences:

1. **Filtered ground truth**: Movies that violate the user's assigned sensitivity traits are removed from the ground truth, so the model never sees constraint-violating recommendations as positive examples.
2. **Constraint-injected prompts**: User prompts include natural-language descriptions of their sensitivity preferences (e.g., "Please avoid movies with gore"), teaching the model to condition on constraints.

No changes to the SFT training code or loss function are needed — the safety alignment comes entirely from the curated data.

### 2.2 Training Command

```bash
accelerate launch --config_file configs/qwen25_0.5b_sft.yaml train_sft_safe.py \
    --dataset_path ./downloaded_datasets/processed_datasets/saferec_sft_dataset \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --gradient_accumulation_steps 8 \
    --save_strategy steps \
    --save_steps 100 \
    --eval_strategy steps \
    --eval_steps 10 \
    --logging_steps 10 \
    --bf16
```

The accelerate config (`configs/qwen25_0.5b_sft.yaml`) uses DeepSpeed ZeRO-3 with 2 GPUs.

### 2.3 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset_path` | (required) | Path to SafeRec SFT dataset (HF `load_from_disk` format) |
| `--model_name` | `meta-llama/Llama-3.2-3B-Instruct` | Base model from Hugging Face Hub |
| `--learning_rate` | 5e-5 | Initial learning rate |
| `--warmup_ratio` | 0.05 | LR warmup fraction |
| `--num_train_epochs` | 10 | Number of training epochs |
| `--max_length` | 1024 | Maximum input sequence length |
| `--optim` | `paged_adamw_8bit` | Optimizer |

### 2.4 Checkpoint Selection

Safe SFT outputs checkpoints to `./results/{model_name}/`. Select the checkpoint with the highest Recall@10 on the validation set. For Qwen2.5-0.5B-Instruct, this is **checkpoint-800** (see Section 7.1 for the full training trajectory).

---

## 3. How Original Rank-GRPO Works

Rank-GRPO treats each position in a top-K recommendation list as an independent RL "item" with its own reward and advantage. This contrasts with standard GRPO, which computes a single scalar reward for the entire output.

### Reward functions

Two reward shaping variants are supported:

| Variant | Per-rank reward | Notes |
|---------|----------------|-------|
| `exp_inf` | `hit[k]` (1 if GT match, else 0) | Per-rank independent |
| `log_decay` | DCG-weighted contribution at rank *k* | Earlier ranks weighted more |

### Advantage computation

For a batch of *G* generations per prompt, each rank *k* has rewards across the group. The advantage is the group-normalized reward:

```
A[k] = (r[k] - mean_group[k]) / (std_group[k] + eps)
```

### Original training setup (from README)

The original paper trains on 4 GPUs with vLLM colocate mode:

- Model: `Qwen2.5-0.5B-Instruct` (SFT checkpoint 1500)
- `per_device_train_batch_size=16`, `gradient_accumulation_steps=6`, `num_generations=8`
- Effective batch size: 4 x 16 x 6 = 384
- Learning rate: 1e-6, KL beta: 1e-3
- Reward function: `exp_inf`
- Full-weight training (no LoRA)

---

## 4. Safe-Rank-GRPO Extensions

### 4.1 Per-rank safety penalties

A `SafetyOracle` checks each recommended movie against the user's sensitivity constraints. If a movie at rank *k* violates a constraint, a penalty is applied only to rank *k*:

```
r_safe[k] = -lambda_safe * penalty_safe * discount[k]   (if violation)
r_safe[k] = 0                                           (if safe)
```

The discount factor matches the relevance reward's weighting scheme (log-decay or exp-inf), so safety penalties and relevance rewards operate on the same scale.

The total per-rank reward in GRPO mode (single combined function) is:

```
r_total[k] = r_rel[k] + r_safe[k]
```

#### Example

User constraint: `{"Anti-gore / squeamish": True}`. Model outputs 20 movies:

| Rank | Movie | GT Match | Violation | r_rel | r_safe | r_total |
|------|-------|----------|-----------|-------|--------|---------|
| 1 | Toy Story | Yes | No | 1.0 | 0.0 | 1.0 |
| 2 | Finding Nemo | No | No | 0.0 | 0.0 | 0.0 |
| 3 | Saw | No | Yes | 0.0 | -1.0 | -1.0 |

Only the tokens generating "Saw" at rank 3 are penalized; ranks 1 and 2 are unaffected.

### 4.2 GDPO decoupled normalization

**Problem**: When relevance and safety rewards are combined into a single scalar before group normalization (standard GRPO mode), the sparse relevance signal can be diluted by the dense safety signal. This causes *reward advantage collapse* — the model receives weak gradients for relevance, stalling quality improvements.

**Solution**: GDPO (`--advantage_mode gdpo`) normalizes each reward component independently before combining:

```python
# For each reward function i:
adv_i[k] = (r_i[k] - group_mean_i[k]) / (group_std_i[k] + eps)

# Weighted sum, then batch normalization:
A[k] = sum(w_i * adv_i[k])
A[k] = (A[k] - batch_mean) / (batch_std + eps)
```

In GDPO mode, relevance and safety are passed as separate reward functions to the trainer, enabling independent normalization.

### 4.3 Count reward

**Problem**: Without explicit encouragement to output the correct number of recommendations, the model learns a shortcut — it outputs *fewer* movies to minimize safety violations. This artificially improves safety metrics but degrades recall.

**Solution**: The count reward (`--lambda_count > 0`) adds a third reward signal:

```
r_count = +lambda_count                                  (if count == target)
r_count = -lambda_count * |count - target| / target      (if count != target)
```

This reward is broadcast uniformly across all ranks (since it reflects the total output length, not any individual position). In GDPO mode it becomes a third independently-normalized reward function.

### 4.4 LoRA training support

Enabling `--use_lora` wraps the model with PEFT LoRA adapters. The reference model is obtained by disabling adapters (`model.disable_adapter()`) rather than maintaining a separate copy, significantly reducing memory usage. Only adapter weights are saved in checkpoints.

Recommended LoRA configuration for multi-signal structured output: `r=256, alpha=512` (effective scale = 2.0).

---

## 5. Data Pipeline

The SafeRec dataset extends the original Reddit-v2 dataset with user sensitivity traits and prompt-level constraint injection.

### Phase 0: Trait assignment

**Script**: `scripts/phase0_trait_assignment/assign_traits_via_gpt.py`

1. GPT assigns sensitivity traits (from a set of 20, defined in `traits_warnings.json`) to each user based on their watch history
2. Ground truth movies that violate the assigned traits are filtered (`filter_violating_groundtruth.py`)
3. SFT samples are filtered to retain only those with valid constrained ground truth (`filter_sft_samples.py`)

### Phase 1: Title mapping

**Script**: `scripts/phase1_mapping/build_title_mapping.py`

Maps Reddit movie titles to canonical catalog entries for consistent safety oracle lookups.

### Phases 2-3: Constraint injection and HF dataset conversion

**Scripts**: `scripts/phase2_3_saferec/generate_saferec_dataset.py`, `convert_to_hf_dataset.py`

1. Injects constraint descriptions into user prompts (e.g., "Please avoid movies with gore")
2. Converts the augmented data into Hugging Face `datasets` format for training

### Dataset statistics

The SafeRec GRPO dataset contains approximately 38,582 training samples, each with:
- `prompt`: conversation history with injected constraints
- `completion`: reference recommendation list
- `groundtruth_with_release_year`: ground truth movies `[(title, year), ...]`
- `seen_titles`: user's watch history
- `constraints`: active sensitivity traits `{"Anti-gore / squeamish": True, ...}`

---

## 6. Training Configuration Comparison

| Parameter | Original Rank-GRPO | Safe-Rank-GRPO |
|-----------|-------------------|----------------|
| GPUs | 4 | 2 |
| Model | Qwen2.5-0.5B-Instruct | Qwen2.5-0.5B-Instruct |
| SFT checkpoint | 1500 | 800 |
| Reward function | `exp_inf` | `exp_inf` |
| Advantage mode | `grpo` | `gdpo` |
| LoRA | No | Yes (r=256, alpha=512) |
| `per_device_train_batch_size` | 16 | 1 |
| `gradient_accumulation_steps` | 6 | 24 |
| `num_generations` | 8 | 4 |
| Effective batch size | 384 | 48 |
| Safety reward | No | Yes (lambda=1.0, penalty=1.0) |
| Count reward | No | Optional (lambda_count > 0) |
| Optimizer | (default) | `paged_adamw_8bit` |
| Gradient checkpointing | No | Yes |
| `max_steps` | epoch-based | 1000-1200 |

---

## 7. Evaluation Results

All models use Qwen2.5-0.5B-Instruct. Evaluation is on the SafeRec validation set with user sensitivity constraints.

### Metrics

| Metric | Description |
|--------|-------------|
| Recall@K | Fraction of ground-truth movies appearing in top-K recommendations |
| NDCG@K | Normalized discounted cumulative gain at K |
| Sensitivity Ratio@K | Fraction of top-K recommendations that violate user constraints |

### 7.1 SFT Baseline Comparison

Two SFT models establish the starting points. The **Official SFT** (checkpoint-1500) was trained on the original Reddit-v2 dataset without any safety awareness. The **Safe SFT** (checkpoint-800) was trained on the SafeRec dataset where constraint-violating ground truth has been filtered out and constraint instructions are injected into prompts.

| Metric | Official SFT (ckpt-1500) | Safe SFT (ckpt-800) |
|--------|--------------------------|----------------------|
| Recall@5 | 0.0401 | 0.0467 |
| Recall@10 | 0.0738 | 0.0732 |
| NDCG@5 | 0.0295 | 0.0333 |
| NDCG@10 | 0.0409 | 0.0422 |
| Sensitivity Ratio@10 | **25.20%** | **1.33%** |

The Safe SFT model matches the Official SFT in recommendation quality (Recall@10: 0.0732 vs 0.0738) while reducing the sensitivity violation rate from 25.2% to 1.3% — a **19x reduction**. This is achieved purely through data curation (filtering violating GT, injecting constraints) without any RL.

#### Safe SFT training trajectory

The Safe SFT model's metrics across training checkpoints:

| Checkpoint | Recall@5 | Recall@10 | NDCG@10 | Sens. Ratio@10 |
|------------|----------|-----------|---------|----------------|
| 200 | 0.0275 | 0.0307 | 0.0225 | 0.72% |
| 400 | 0.0490 | 0.0576 | 0.0400 | 0.93% |
| 600 | 0.0458 | 0.0596 | 0.0391 | 1.19% |
| **800** | **0.0467** | **0.0732** | **0.0422** | **1.33%** |
| 1000 | 0.0416 | 0.0635 | 0.0383 | 1.70% |
| 1200 | 0.0483 | 0.0658 | 0.0404 | 2.14% |

Checkpoint 800 is the peak for Recall@10. Beyond step 800, the model begins overfitting — recall degrades while the sensitivity ratio increases steadily, indicating the model starts generating more constraint-violating recommendations.

### 7.2 Effect of Original GRPO (No Safety Signal)

The official Rank-GRPO (exp_inf, 15,800 steps) improves recommendation quality but **dramatically worsens safety**:

| Metric | Official SFT (ckpt-1500) | Official GRPO (step 15800) | Change |
|--------|--------------------------|----------------------------|--------|
| Recall@5 | 0.0401 | 0.0678 | +69% |
| Recall@10 | 0.0738 | 0.1052 | +43% |
| NDCG@5 | 0.0295 | 0.0487 | +65% |
| NDCG@10 | 0.0409 | 0.0618 | +51% |
| Sensitivity Ratio@10 | 25.20% | **44.47%** | +76% |

GRPO without a safety signal actively optimizes the model to recommend more popular movies (which tend to be more controversial), increasing the violation rate from 25% to 44%. This motivates the need for explicit safety rewards.

### 7.3 Safe-Rank-GRPO Results

Three GRPO runs starting from Safe SFT checkpoint-800 are compared.

#### GDPO without count reward (steps 0 → 1000)

| Metric | Step 0 | Step 600 | Step 1000 | Change |
|--------|--------|----------|-----------|--------|
| Recall@5 | 0.0470 | 0.0478 | 0.0483 | +2.8% |
| Recall@10 | 0.0710 | 0.0650 | 0.0596 | **-16.1%** |
| NDCG@5 | 0.0337 | 0.0346 | 0.0362 | +7.4% |
| NDCG@10 | 0.0419 | 0.0406 | 0.0403 | -3.8% |
| Sensitivity Ratio@10 | 1.44% | 1.07% | 1.14% | -20.8% |

Recall@10 degrades significantly because the model learns to output fewer recommendations as a shortcut to avoid safety violations.

#### GDPO with count reward (steps 0 → 1000)

| Metric | Step 0 | Step 600 | Step 1000 | Change |
|--------|--------|----------|-----------|--------|
| Recall@5 | 0.0470 | 0.0476 | 0.0520 | +10.6% |
| Recall@10 | 0.0710 | 0.0688 | 0.0725 | **+2.1%** |
| NDCG@5 | 0.0337 | 0.0358 | 0.0378 | +12.2% |
| NDCG@10 | 0.0419 | 0.0430 | 0.0447 | +6.7% |
| Sensitivity Ratio@10 | 1.44% | 1.03% | 1.25% | -13.2% |

With the count reward, Recall@10 is maintained near baseline while safety still improves.

### 7.4 End-to-End Summary

| Model | Recall@10 | NDCG@10 | Sens. Ratio@10 |
|-------|-----------|---------|----------------|
| Official SFT (ckpt-1500) | 0.0738 | 0.0409 | 25.20% |
| Official GRPO (step 15800) | 0.1052 | 0.0618 | 44.47% |
| Safe SFT (ckpt-800) | 0.0732 | 0.0422 | 1.33% |
| Safe GDPO w/o count (step 1000) | 0.0596 | 0.0403 | 1.14% |
| Safe GDPO w/ count (step 1000) | 0.0725 | 0.0447 | 1.25% |

The full Safe-Rank-GRPO pipeline (Safe SFT + GDPO with count reward) maintains recommendation quality comparable to the official SFT baseline (Recall@10: 0.0725 vs 0.0738) while reducing sensitivity violations by **95%** (1.25% vs 25.20%).

---

## 8. Key Findings

1. **Data curation alone achieves the majority of safety improvement**. Safe SFT reduces the sensitivity ratio from 25.2% to 1.3% (a 19x reduction) with no recall loss, by filtering constraint-violating ground truth and injecting constraints into prompts.

2. **Standard GRPO makes safety worse**. The official Rank-GRPO (no safety signal) increases the violation rate from 25% to 44% because it optimizes purely for recommendation relevance, pushing the model toward popular but controversial movies.

3. **Safe-Rank-GRPO further improves safety on top of Safe SFT**. The GDPO runs reduce the already-low sensitivity ratio by an additional 13-21%, demonstrating that RL fine-tuning with safety rewards adds value beyond data curation alone.

4. **Without count reward, the model exploits a shortcut**. Rather than learning which specific movies are unsafe, the model reduces its total recommendation count. This reduces safety violations trivially but causes Recall@10 to degrade by 16%.

5. **Count reward closes the shortcut**. By rewarding the model for outputting exactly the target number of recommendations, Recall@10 is maintained (+2.1%) while safety still improves (-13.2% sensitivity ratio). The model is forced to learn *which* movies to avoid rather than *how many* movies to output.

6. **GDPO is necessary for multi-objective training**. Decoupled normalization prevents the dense safety signal from drowning out the sparse relevance signal, which would otherwise cause the model to ignore relevance entirely.

7. **End-to-end result**. The full pipeline (Safe SFT + GDPO with count reward) achieves comparable recommendation quality to the official SFT (Recall@10: 0.0725 vs 0.0738) while reducing sensitivity violations by 95% (1.25% vs 25.20%).

---

## 9. File Reference

| File | Purpose |
|------|---------|
| `train_sft_safe.py` | Safe SFT training entry point |
| `train_rank_grpo_safe.py` | Safe-Rank-GRPO training entry point |
| `libs/safe_reward_funcs.py` | Reward functions (combined, split, count) |
| `libs/safety_oracle.py` | SafetyOracle for violation checking |
| `libs/trl/rank_grpo_trainer.py` | RankGRPOTrainer (GRPO + GDPO modes) |
| `scripts/phase0_trait_assignment/` | Trait assignment and GT filtering |
| `scripts/phase1_mapping/` | Title-to-catalog mapping |
| `scripts/phase2_3_saferec/` | Constraint injection and HF conversion |
| `traits_warnings.json` | Full list of 20 sensitivity traits |

---

## 10. Quick Start

### GDPO + LoRA + count reward (recommended)

```bash
accelerate launch --num_processes 2 train_rank_grpo_safe.py \
    --train_path ./downloaded_datasets/processed_datasets/saferec_grpo_dataset \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --sft_checkpoint 800 \
    --catalog_path gt_catalog_complete.pkl \
    --use_lora \
    --lora_r 256 \
    --lora_alpha 512 \
    --lora_dropout 0.05 \
    --advantage_mode gdpo \
    --reward_func exp_inf \
    --lr 1e-6 \
    --kl_beta 1e-3 \
    --gradient_accumulation_steps 24 \
    --per_device_train_batch_size 1 \
    --num_generations 4 \
    --max_steps 1200 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.35 \
    --vllm_tensor_parallel_size 1 \
    --save_steps 100 \
    --seed 3407 \
    --bf16 \
    --gradient_checkpointing \
    --lambda_safe 1.0 \
    --penalty_safe 1.0 \
    --lambda_count 1.0 \
    --target_count 10
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--advantage_mode` | `grpo` | `grpo` (combined) or `gdpo` (decoupled normalization) |
| `--lambda_safe` | 1.0 | Safety penalty weight |
| `--penalty_safe` | 1.0 | Penalty magnitude per violation |
| `--risk_threshold` | 0.66 | SafetyOracle risk threshold |
| `--lambda_count` | 0.0 | Count reward weight (0 = disabled) |
| `--target_count` | 10 | Target number of recommendations |
| `--use_lora` | off | Enable LoRA training |
| `--lora_r` | 16 | LoRA rank |
| `--lora_alpha` | 32 | LoRA scaling factor |
