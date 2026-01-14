# Quick Start: Trait Assignment

快速上手指南 - 为 SFT 数据自动分配用户敏感特征

---

## 前置条件

✅ 已完成 Phase 1 映射构建:
- [x] `data/title_to_imdb.pkl` 存在
- [x] `downloaded_datasets/movie_trait_sensitivity.json` 存在
- [x] OpenAI API Key 已配置

---

## Step 1: 筛选高质量样本

```bash
python scripts/filter_sft_samples.py \
    --input_path downloaded_datasets/processed_datasets/sft_dataset/train \
    --output_path data/sft_filtered_8k.json \
    --min_groundtruth 3 \
    --target_samples 8000
```

**输出**: `data/sft_filtered_8k.json` (8,000 样本)

**预期耗时**: ~2 分钟

---

## Step 2: ChatGPT API 自动标注

### 设置 API Key

```bash
export OPENAI_API_KEY="sk-..."
```

### 运行标注脚本

```bash
python scripts/assign_traits_via_gpt.py \
    --input_path data/sft_filtered_8k.json \
    --output_path data/sft_with_assigned_traits.json \
    --traits_path traits_warnings.json \
    --model gpt-4o \
    --batch_size 20 \
    --temperature 0.3 \
    --max_workers 5
```

**输出**: `data/sft_with_assigned_traits.json` (带 assigned_trait)

**预期耗时**: ~30-40 分钟

**预期成本**: ~$22 (8,000 samples × $0.0027)

---

## Step 3: 过滤违规 GT 电影

```bash
python scripts/filter_violating_groundtruth.py \
    --input_path data/sft_with_assigned_traits.json \
    --output_path data/saferec_sft_8k_dataset.json \
    --trait_sensitivity_path downloaded_datasets/movie_trait_sensitivity.json \
    --title_mapping_path data/title_to_imdb.pkl \
    --min_groundtruth_after_filter 1
```

**输出**: `data/saferec_sft_8k_dataset.json` (最终训练数据)

**预期耗时**: ~3 分钟

**预期样本保留率**: ~90% (约 7,200 样本)

---

## Step 4: 统计分析

```bash
python scripts/analyze_trait_distribution.py \
    --input_path data/saferec_sft_8k_dataset.json \
    --output_dir data/trait_stats/
```

**输出**:
- `data/trait_stats/stats.json` - 详细统计
- `data/trait_stats/trait_distribution.png` - 分布图
- `data/trait_stats/gt_filtering_stats.png` - 过滤前后对比

**预期耗时**: ~1 分钟

---

## Step 5: 人工验证（可选）

随机抽取 100 个样本进行质量检查：

```bash
python scripts/sample_for_validation.py \
    --input_path data/sft_with_assigned_traits.json \
    --output_path data/validation_sample_100.json \
    --sample_size 100
```

**验证清单**:
- [ ] Assigned trait 不违反 prompt 中的用户偏好
- [ ] Completion 中的电影确实容易违反该 trait
- [ ] GT 过滤逻辑正确

---

## 一键运行（全流程）

```bash
# 设置环境变量
export OPENAI_API_KEY="sk-..."

# 运行完整 pipeline
bash scripts/run_trait_assignment_pipeline.sh
```

**总耗时**: ~40-50 分钟

---

## 常见问题

### Q1: API 调用失败怎么办？

脚本内置重试机制（最多 3 次）。如果仍失败：

```bash
# 从中断点继续
python scripts/assign_traits_via_gpt.py \
    --input_path data/sft_filtered_8k.json \
    --output_path data/sft_with_assigned_traits.json \
    --resume_from data/sft_with_assigned_traits.checkpoint.json
```

### Q2: 如何降低成本？

**方案1**: 减少样本数
```bash
--target_samples 4000  # 成本降至 ~$11
```

**方案2**: 使用 gpt-4o-mini
```bash
--model gpt-4o-mini  # 成本降至 ~$2
```

### Q3: Trait 分布不均怎么办？

运行第二轮补充标注：

```bash
python scripts/balance_trait_distribution.py \
    --input_path data/saferec_sft_8k_dataset.json \
    --target_samples_per_trait 400 \
    --output_path data/saferec_sft_balanced.json
```

---

## 下一步

完成 Trait Assignment 后，继续 SafeRec 训练：

1. **实现 SafetyOracle**: 参见 [SAFEREC_IMPLEMENTATION_PLAN.md](./SAFEREC_IMPLEMENTATION_PLAN.md) Phase 2
2. **生成 CoT 数据**: Phase 4
3. **训练 SafeRec SFT**: Phase 5

---

## 目录结构

完成后应有以下文件：

```
data/
├── sft_filtered_8k.json              # Step 1
├── sft_with_assigned_traits.json     # Step 2
├── saferec_sft_8k_dataset.json       # Step 3（最终数据）
├── trait_stats/                      # Step 4
│   ├── stats.json
│   ├── trait_distribution.png
│   └── gt_filtering_stats.png
└── validation_sample_100.json        # Step 5（可选）
```

---

## 参考文档

- [TRAIT_ASSIGNMENT_PLAN.md](./TRAIT_ASSIGNMENT_PLAN.md) - 详细计划
- [SAFEREC_IMPLEMENTATION_PLAN.md](./SAFEREC_IMPLEMENTATION_PLAN.md) - 完整实施路线图
- [traits_warnings.json](../traits_warnings.json) - 20 个 trait 定义
