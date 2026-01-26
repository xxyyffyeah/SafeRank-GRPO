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

| 数据集 | GT 阈值 | 目标样本数 | 原因 |
|--------|---------|------------|------|
| **SFT Train** | ≥ 2 | 24,000 | 有足够的 GT 电影供过滤（Step 3 会移除违规 GT） |
| **SFT Test** | ≥ 0 | 1,000 | 测试集不过滤 GT，保持原始分布 |
| **SFT Validation** | ≥ 0 | 1,000 | 验证集不过滤 GT，保持原始分布 |
| **GRPO** | ≥ 0 | 72,000 | GRPO 用于 RL 训练，不需要 GT 过滤 |

### 为什么 Train 使用 GT ≥ 2？

Step 3（GT 违规过滤）会移除违反 assigned trait 的 GT 电影。如果初始 GT < 2：
- 过滤后可能剩余 0 个 GT
- 样本会被丢弃（min_groundtruth_after_filter = 1）
- 导致大量样本损失

因此，Train 集使用 GT ≥ 2 确保过滤后仍有足够 GT。

### SFT 数据筛选

```bash
# Train: 24k samples, GT >= 2
python scripts/phase0_trait_assignment/filter_sft_samples.py \
    --input_path downloaded_datasets/processed_datasets/sft_dataset/train \
    --output_path data/phase0_trait_assignment/expanded/sft_train_24k_filtered.json \
    --min_groundtruth 2 \
    --target_samples 24000 \
    --split_name train

# Test: 1k samples, no GT filter
python scripts/phase0_trait_assignment/filter_sft_samples.py \
    --input_path downloaded_datasets/processed_datasets/sft_dataset/test \
    --output_path data/phase0_trait_assignment/expanded/sft_test_1k_filtered.json \
    --min_groundtruth 0 \
    --target_samples 1000 \
    --split_name test

# Validation: 1k samples, no GT filter
python scripts/phase0_trait_assignment/filter_sft_samples.py \
    --input_path downloaded_datasets/processed_datasets/sft_dataset/validation \
    --output_path data/phase0_trait_assignment/expanded/sft_val_1k_filtered.json \
    --min_groundtruth 0 \
    --target_samples 1000 \
    --split_name validation
```

### GRPO 数据筛选

GRPO 数据集与 SFT 数据集有以下区别：

| 字段 | SFT 数据 | GRPO 数据 |
|------|----------|-----------|
| prompt | ✅ | ✅ |
| completion | ✅ | ❌ 无此字段 |
| seen_titles | ✅ | ✅ |
| groundtruth_with_release_year | ✅ | ✅ |

**GRPO 无 completion 的处理方式**：
- Step 2 (assign_traits_via_gpt.py) **只使用用户对话**推断 trait
- **不使用 groundtruth**（因为 groundtruth 是用户喜欢的电影，无法推断用户避免什么）
- GPT 只从对话中的显式线索（如 "for kids", "avoid gore" 等）推断 trait
- 如果对话中没有明确的敏感性指示，分配 "None"
- 使用专门的 `SYSTEM_PROMPT_TEMPLATE_GRPO` 和 `USER_PROMPT_TEMPLATE_GRPO`

```bash
# GRPO: 72k samples, no GT filter
python scripts/phase0_trait_assignment/filter_sft_samples.py \
    --input_path downloaded_datasets/processed_datasets/grpo/grpo_dataset/train \
    --output_path data/phase0_trait_assignment/expanded/grpo_72k_filtered.json \
    --min_groundtruth 0 \
    --target_samples 72000 \
    --split_name grpo
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
| **Model** | `gpt-5.2` |
| **Temperature** | 0.3 (低温度保证一致性) |
| **Max Tokens** | 150 |
| **Response Format** | JSON Object |

### 双 Prompt 模式

由于 GRPO 数据没有 `completion` 字段，使用两套 Prompt：

#### SFT Prompt（有 completion）

```python
SYSTEM_PROMPT_TEMPLATE_SFT = """You are a movie safety expert analyzing user conversations and movie recommendations.

Your task: Identify which user trait sensitivity would be MOST violated by the recommended movies, but NOT conflict with the user's explicit preferences.

Available Traits:
{trait_list}

Guidelines:
1. The assigned trait should NOT conflict with user's explicit preferences
2. The recommended movies should be LIKELY to violate this trait
3. Choose the trait with highest violation risk
4. Return valid JSON: {"assigned_trait": "exact trait name"}
"""

USER_PROMPT_TEMPLATE_SFT = """User Conversation:
{prompt}

Recommended Movies:
{completion}

Which trait is MOST likely to be violated by these recommendations?
Respond with JSON: {"assigned_trait": "exact trait name from the list"}"""
```

#### GRPO Prompt（无 completion，仅基于对话）

```python
SYSTEM_PROMPT_TEMPLATE_GRPO = """You are a movie safety expert analyzing user conversations about movies.

Your task: Identify which user trait sensitivity is MOST likely based ONLY on the user's conversation tone, language, and explicit mentions. Do NOT infer from movie titles or preferences.

Available Traits:
{trait_list}

Guidelines:
1. ONLY use explicit clues from the conversation (mentions of kids, family, avoiding violence, etc.)
2. If user mentions "kids", "children", "family-friendly" → use "Parent filtering for children"
3. If user mentions avoiding specific content (gore, violence, scary, etc.) → use corresponding trait
4. If NO explicit sensitivity mentioned → use "None"
5. DO NOT infer traits from movie titles or genres mentioned
6. Return valid JSON: {"assigned_trait": "exact trait name", "reason": "explanation"}

Example:
User: "I'm looking for action thrillers, but please nothing with gore or graphic violence - I'm very squeamish about that stuff"
Output: {"assigned_trait": "Anti-gore / squeamish", "reason": "User explicitly states being squeamish about gore and graphic violence, indicating this sensitivity needs to be filtered."}
"""

USER_PROMPT_TEMPLATE_GRPO = """User Conversation:
{prompt}

Based ONLY on the user's conversation (ignore any movie titles), which trait sensitivity is most clearly indicated?
Respond with JSON: {"assigned_trait": "exact trait name from the list", "reason": "brief explanation"}"""
```

**关键差异**：
- **SFT**: 从推荐电影（completion）推断可能违反的 trait
- **GRPO**: 从用户对话中的**显式线索**推断敏感性，不使用 groundtruth

### 自动检测数据类型

```python
def create_prompt(sample, traits):
    # 自动检测是否有 completion
    has_completion = "completion" in sample and sample["completion"]

    if has_completion:
        # SFT 数据：使用 completion（推荐电影）
        system_prompt = SYSTEM_PROMPT_TEMPLATE_SFT.format(...)
        user_prompt = USER_PROMPT_TEMPLATE_SFT.format(
            prompt=prompt_content,
            completion=completion_content
        )
    else:
        # GRPO 数据：只使用对话，忽略 groundtruth
        system_prompt = SYSTEM_PROMPT_TEMPLATE_GRPO.format(...)
        user_prompt = USER_PROMPT_TEMPLATE_GRPO.format(
            prompt=prompt_content
        )

    return system_prompt, user_prompt
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
  "assignment_reason": "User wants action (not conflicting), but recommendations are extreme horror/gore films that feature graphic torture and body horror.",
  "assignment_success": true,
  "assignment_error": null,
  "gpt_usage": {
    "prompt_tokens": 726,
    "completion_tokens": 35,
    "total_tokens": 761
  },
  "prompt": [...],
  "completion": [...],
  "groundtruth_with_release_year": [...]
}
```

**新增字段说明**：
- `assignment_reason`: GPT 给出的 trait 选择理由（1-2句解释）
- `assignment_success`: 是否成功分配 trait
- `assignment_error`: 错误信息（如果有）
- `gpt_usage`: Token 使用统计

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
├── scripts/phase0_trait_assignment/
│   ├── filter_sft_samples.py              # Step 1 筛选（支持 SFT 和 GRPO）
│   ├── assign_traits_via_gpt.py           # Step 2 GPT 标注（双 Prompt 模式）
│   ├── filter_violating_groundtruth.py    # Step 3 过滤
│   ├── analyze_trait_distribution.py      # Step 4 统计
│   ├── run_trait_assignment_pipeline.sh   # 单数据集 Pipeline
│   └── expand_saferec_pipeline.sh         # 扩展 Pipeline（4个数据集）
│
├── data/phase0_trait_assignment/
│   ├── expanded/                          # 扩展数据集输出目录
│   │   ├── sft_train_24k_filtered.json    # Train 24k 筛选结果
│   │   ├── sft_train_24k_with_traits.json # Train 24k 标注结果
│   │   ├── sft_train_24k_final.json       # Train 24k 最终结果
│   │   ├── sft_train_24k_stats/           # Train 24k 统计
│   │   ├── sft_test_1k_*.json             # Test 1k 各阶段输出
│   │   ├── sft_val_1k_*.json              # Validation 1k 各阶段输出
│   │   └── grpo_72k_*.json                # GRPO 72k 各阶段输出
│   └── trait_stats/                       # 原 8k 统计（已完成）
│
├── .env                                   # OpenAI API Key
│
└── docs/
    └── TRAIT_ASSIGNMENT_PLAN.md           # 本文档
```

### 运行扩展 Pipeline

```bash
# 运行完整 4 数据集 Pipeline
bash scripts/phase0_trait_assignment/expand_saferec_pipeline.sh

# 测试模式（每个数据集只处理 50 条）
bash scripts/phase0_trait_assignment/expand_saferec_pipeline.sh --test
```

扩展 Pipeline 会依次处理：
1. **sft_train_24k**: 24,000 条，GT ≥ 2
2. **sft_test_1k**: 1,000 条，无 GT 过滤
3. **sft_val_1k**: 1,000 条，无 GT 过滤
4. **grpo_72k**: 72,000 条，无 GT 过滤，使用 GRPO Prompt

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

## Phase 2 & 3: SafeRec Dataset with Dual Reasoning

### 数据集字段

生成的 HuggingFace 数据集包含以下 7 个字段：

```python
{
    # 输入输出
    "prompt": [{"role": "user", "content": "..."}],
    "completion": [{"role": "assistant", "content": "..."}],

    # 元数据
    "seen_titles": ["Movie 1", "Movie 2", ...],
    "groundtruth_with_release_year": [["Movie A", "2020"], ...],
    "constraints": {"Trait 1": True, "Trait 2": None, ...},  # 20 个 traits

    # 新增：Trait Assignment Reasoning
    "assigned_trait": "Horror avoider (avoids scares & supernatural)",
    "assignment_reason": "User mentioned 'I don't watch horror movies' in conversation."
}
```

### 两层 Reasoning

**1. Trait Assignment Reasoning** (`assigned_trait` + `assignment_reason`)
- **作用**: 解释为什么 GPT 分配了这个 trait
- **来源**: Phase 0 - Trait Assignment
- **示例**:
  ```
  assigned_trait: "Horror avoider (avoids scares & supernatural)"
  assignment_reason: "User mentioned 'I don't watch horror movies' in conversation."
  ```

**2. Safety Filtering Reasoning** (在 `completion` 中的 `<reasoning>` 标签)
- **作用**: 解释哪些电影被过滤以及为什么
- **来源**: Phase 2&3 - SafeRec Dataset Generation
- **示例**:
  ```xml
  <reasoning>
  Safety Analysis:
  User preferences to avoid: Horror avoider (avoids scares & supernatural)

  Movies filtered due to safety concerns:
  - Drop Dead Gorgeous (1999): Horror avoider: risk=1.00
  - Man Bites Dog (1992): Horror avoider: risk=1.00

  Safe recommendations selected: 14 movies
  </reasoning>
  ```

### 数据统计

| Split | Samples | With Trait Reasoning | With Safety CoT |
|-------|---------|---------------------|-----------------|
| Train | 19,086 | 100% | 99.6% |
| Test | 1,212 | 100% | 99.6% |
| Validation | 1,127 | 100% | 99.5% |

### 使用示例

```python
from datasets import load_from_disk

# 加载数据集
dataset = load_from_disk('downloaded_datasets/processed_datasets/saferec_sft_dataset/train')

# 访问双层 reasoning
sample = dataset[0]
print(f"Assigned Trait: {sample['assigned_trait']}")
print(f"Why: {sample['assignment_reason']}")
print(f"Safety CoT: {sample['completion'][0]['content']}")
```

---

## 参考

- [SAFEREC_SFT_PLAN.md](./SAFEREC_SFT_PLAN.md) - 原始 SafeRec 计划
- [MAPPING_COVERAGE_SUMMARY.md](./MAPPING_COVERAGE_SUMMARY.md) - 映射覆盖率
- [traits_warnings.json](../traits_warnings.json) - 20 个 trait 定义
