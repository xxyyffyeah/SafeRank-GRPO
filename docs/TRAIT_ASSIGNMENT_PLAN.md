# Trait Assignment Plan - 自动为 SFT 数据分配用户敏感特征

## 概述

通过 ChatGPT API 自动为 SFT 数据集中的每个 conversation 分配合适的用户敏感特征（trait），生成用于安全对齐训练的标注数据。

---

## 目标

为 8,000 条高质量 SFT 样本分配 user trait，使得：
1. Trait 不违反 prompt 中的用户特征描述
2. Completion 中的推荐电影**容易违反**该 trait（用于生成负样本）
3. 生成的数据可直接用于 SafeRec 训练

---

## Pipeline 流程

```
SFT Dataset (95,753 samples)
    ↓
[Step 1] 筛选高质量样本 (GT > 3)
    ↓
8,000 filtered samples
    ↓
[Step 2] ChatGPT API 批量调用
    ↓
Assigned Traits (8,000 annotations)
    ↓
[Step 3] 过滤 GT 违规电影
    ↓
SafeRec Training Dataset
    ↓
[Step 4] 统计分析
```

---

## Step 1: 数据筛选

### 筛选标准

| 标准 | 阈值 | 原因 |
|------|------|------|
| **Groundtruth 数量** | ≥ 3 | 有足够的 GT 电影供过滤 |
| **目标样本数** | 8,000 | 平衡数据量和 API 成本 |

### 实现脚本

```bash
python scripts/filter_sft_samples.py \
    --input_path downloaded_datasets/processed_datasets/sft_dataset/train \
    --output_path data/sft_filtered_8k.json \
    --min_groundtruth 3 \
    --target_samples 8000
```

### 预期输出

```json
{
  "samples": [
    {
      "sample_id": "train_12345",
      "prompt": [{"role": "user", "content": "..."}],
      "completion": [{"role": "assistant", "content": "..."}],
      "seen_titles": ["Movie1", "Movie2"],
      "groundtruth_with_release_year": [
        ["Movie A", "2010"],
        ["Movie B", "2015"],
        ["Movie C", "2018"],
        ["Movie D", "2020"]
      ]
    }
  ],
  "stats": {
    "total_samples": 8000,
    "avg_groundtruth": 4.2
  }
}
```

---

## Step 2: ChatGPT API Trait Assignment

### API 配置

| 参数 | 值 |
|------|-----|
| **Model** | `gpt-4o` (或 `gpt-4o-2024-11-20`) |
| **Temperature** | 0.3 (低温度保证一致性) |
| **Max Tokens** | 500 |
| **Batch Size** | 20 samples/request |

### Prompt 设计

```python
SYSTEM_PROMPT = """You are a movie safety expert analyzing user conversations and movie recommendations.

Your task: Identify which user trait sensitivity would be MOST violated by the recommended movies.

Available Traits:
{trait_list}

Guidelines:
1. The assigned trait should NOT conflict with user's explicit preferences in the conversation
2. The recommended movies should be LIKELY to violate this trait (contain sensitive content)
3. Choose the trait with highest violation risk among recommended movies
4. Return ONLY the trait name, nothing else

Example:
User: "I love action movies with intense fight scenes"
Recommendations: "Saw (2004), Hostel (2005), The Human Centipede (2009)"
Output: Anti-gore / squeamish
"""

USER_PROMPT_TEMPLATE = """Conversation:
{prompt}

Recommended Movies:
{completion}

Which trait is MOST likely to be violated by these recommendations?
Return only the trait name."""
```

### 实现脚本

```bash
python scripts/assign_traits_via_gpt.py \
    --input_path data/sft_filtered_8k.json \
    --output_path data/sft_with_assigned_traits.json \
    --traits_path traits_warnings.json \
    --api_key $OPENAI_API_KEY \
    --model gpt-4o \
    --batch_size 20 \
    --temperature 0.3
```

### 输出格式

```json
{
  "sample_id": "train_12345",
  "assigned_trait": "Anti-gore / squeamish",
  "assignment_confidence": "high",
  "gpt_reasoning": "Recommendations include multiple horror films with graphic violence",
  "prompt": [...],
  "completion": [...],
  "groundtruth_with_release_year": [...]
}
```

---

## Step 3: GT 违规过滤

### 过滤逻辑

对于每个样本：
1. 查找 GT 电影的 trait sensitivity 数据
2. 检查 GT 是否违反 assigned trait
3. 移除违规的 GT 电影
4. 如果剩余 GT < 1，则丢弃整个样本

```python
def filter_groundtruth(sample, trait_data, title_mapper):
    """
    过滤违反 assigned trait 的 GT 电影

    Args:
        sample: 包含 assigned_trait 和 groundtruth 的样本
        trait_data: movie_trait_sensitivity.json
        title_mapper: title → imdbId 映射

    Returns:
        filtered_sample or None
    """
    assigned_trait = sample["assigned_trait"]
    filtered_gt = []

    for title, year in sample["groundtruth_with_release_year"]:
        imdb_id = title_mapper.get_imdb_id(title, year)

        if not imdb_id or imdb_id not in trait_data:
            # 未知电影保留
            filtered_gt.append([title, year])
            continue

        traits = trait_data[imdb_id]["traits"]
        trait_info = traits.get(assigned_trait, {})

        # 检查是否违反
        if trait_info.get("unsafe", False):
            # 违反 trait，移除
            continue
        else:
            # 安全，保留
            filtered_gt.append([title, year])

    # 至少保留 1 个 GT
    if len(filtered_gt) < 1:
        return None

    sample["groundtruth_with_release_year"] = filtered_gt
    sample["num_gt_removed"] = len(sample["groundtruth_with_release_year"]) - len(filtered_gt)

    return sample
```

### 实现脚本

```bash
python scripts/filter_violating_groundtruth.py \
    --input_path data/sft_with_assigned_traits.json \
    --output_path data/saferec_sft_8k_dataset.json \
    --trait_sensitivity_path downloaded_datasets/movie_trait_sensitivity.json \
    --title_mapping_path data/title_to_imdb.pkl
```

---

## Step 4: 统计分析与可视化

### 统计指标

```python
stats = {
    "total_samples": 8000,
    "samples_after_filtering": 7245,
    "samples_dropped": 755,
    "avg_gt_before": 4.2,
    "avg_gt_after": 3.1,
    "avg_gt_removed_per_sample": 1.1,

    "trait_distribution": {
        "Anti-gore / squeamish": 1245,
        "Horror avoider": 1089,
        "Sexual violence sensitive": 856,
        ...
    },

    "trait_coverage": {
        "mapped_to_imdb": 6845,
        "found_in_trait_data": 6120,
        "unknown_movies": 1125
    }
}
```

### 可视化

```bash
python scripts/analyze_trait_distribution.py \
    --input_path data/saferec_sft_8k_dataset.json \
    --output_dir data/trait_stats/
```

**生成图表**:
1. `trait_distribution.png` - Trait 分配分布柱状图
2. `gt_filtering_stats.png` - GT 过滤前后对比
3. `trait_safety_heatmap.png` - Trait × 电影 安全性热力图

---

## 成本估算

### OpenAI API 成本

| 模型 | Input Price | Output Price | 每样本 Tokens | 单价 |
|------|-------------|--------------|---------------|------|
| gpt-4o | $2.50/1M | $10.00/1M | ~1000 input + 20 output | $0.0027 |

**总成本**:
- 8,000 samples × $0.0027 ≈ **$21.6**

### 时间估算

- Batch size: 20
- 请求数: 8,000 / 20 = 400
- 平均延迟: 5s/request
- 总时间: 400 × 5s ≈ **33 分钟**

---

## 文件结构

```
Rank-GRPO/
├── scripts/
│   ├── filter_sft_samples.py              # [新增] Step 1 筛选
│   ├── assign_traits_via_gpt.py           # [新增] Step 2 GPT 标注
│   ├── filter_violating_groundtruth.py    # [新增] Step 3 过滤
│   └── analyze_trait_distribution.py      # [新增] Step 4 统计
│
├── data/
│   ├── sft_filtered_8k.json               # Step 1 输出
│   ├── sft_with_assigned_traits.json      # Step 2 输出
│   ├── saferec_sft_8k_dataset.json        # Step 3 输出（最终数据）
│   └── trait_stats/                       # Step 4 输出
│       ├── trait_distribution.png
│       └── stats.json
│
└── docs/
    └── TRAIT_ASSIGNMENT_PLAN.md           # 本文档
```

---

## 质量控制

### 人工抽样验证

从 8,000 样本中随机抽取 100 个进行人工验证：

```bash
python scripts/sample_for_validation.py \
    --input_path data/sft_with_assigned_traits.json \
    --output_path data/validation_sample_100.json \
    --sample_size 100
```

**验证标准**:
1. ✅ Assigned trait 不违反 prompt 中的用户偏好
2. ✅ Completion 中的电影确实容易违反该 trait
3. ✅ GT 过滤逻辑正确

### 错误处理

| 错误类型 | 处理策略 |
|---------|---------|
| **GPT 返回无效 trait** | 重试 3 次，仍失败则标记为 "Unknown" |
| **Title 映射失败** | 保留原始 GT，标记为 "unmapped" |
| **全部 GT 被过滤** | 丢弃样本，记录日志 |

---

## 预期效果

### 数据质量提升

| 指标 | 改进 |
|------|------|
| **Trait 覆盖均衡性** | 自动分配确保各 trait 都有代表样本 |
| **GT 质量** | 移除违规 GT，减少噪声标签 |
| **训练信号强度** | Completion 违反 trait 提供强负样本 |

### 后续用于 SafeRec

```python
# SafeRec SFT 数据格式
{
    "prompt": [{"role": "user", "content": "推荐科幻电影..."}],
    "completion": [{"role": "assistant", "content": """
思考过程：
用户偏好科幻片。注意：部分推荐可能包含血腥暴力内容（Anti-gore trait）。
- 考虑《Alien》(1979)，但包含大量血腥场面，不适合敏感用户。
- 考虑《Event Horizon》(1997)，极端恐怖血腥，排除。

推荐列表：
The Martian (2015)
Interstellar (2014)
Arrival (2016)
..."""}],

    "assigned_trait": "Anti-gore / squeamish",
    "groundtruth_with_release_year": [
        ["The Martian", "2015"],
        ["Interstellar", "2014"]
        # 注意：《Alien》已被过滤（违反 Anti-gore）
    ]
}
```

---

## 实施时间线

| 阶段 | 任务 | 预计时间 |
|------|------|---------|
| **Week 1** | 实现 Step 1-2 脚本 + GPT 调用 | 2 天 |
| **Week 1** | 运行 GPT 标注（8k 样本） | 1 小时 |
| **Week 1** | 人工抽样验证（100 样本） | 2 小时 |
| **Week 2** | 实现 Step 3-4 脚本 | 1 天 |
| **Week 2** | 生成最终数据集 + 统计分析 | 0.5 天 |

**总时间**: 约 **3-4 天**

---

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| **GPT 分配不准确** | 训练数据质量下降 | 人工验证 + 温度调低 |
| **API 限流** | 处理时间延长 | 实现重试机制 + 批处理 |
| **过度过滤 GT** | 样本过少 | 设置最小 GT 保留数 |
| **Trait 分布不均** | 某些 trait 样本过少 | 第二轮补充标注 |

---

## 扩展计划

### 如果效果良好

1. **扩展到全量数据**: 从 8k → 95k 样本
2. **多 Trait 分配**: 每个样本分配 1-3 个 traits
3. **细粒度标注**: 标注 completion 中具体哪些电影违反 trait

### 替代方案

如果 GPT 成本过高，可使用：
- **规则 + 启发式**: 基于电影 trait data 自动推断
- **Smaller LLM**: 使用 Llama-3.1-70B 或 Qwen-72B

---

## 参考

- [SAFEREC_SFT_PLAN.md](./SAFEREC_SFT_PLAN.md) - 原始 SafeRec 计划
- [MAPPING_COVERAGE_SUMMARY.md](./MAPPING_COVERAGE_SUMMARY.md) - 映射覆盖率
- [traits_warnings.json](../traits_warnings.json) - 20 个 trait 定义
