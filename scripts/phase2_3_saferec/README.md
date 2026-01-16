# Phase 2 & 3: SafeRec Implementation

SafeRec安全推荐系统的实现，包括约束注入、安全过滤和CoT推理生成。

## 概述

Phase 2/3实现了以下功能：
1. **约束注入** - 将用户安全偏好注入到prompt中
2. **安全过滤** - 过滤违反用户约束的电影推荐
3. **CoT推理** - 生成结构化的推理过程

## 文件结构

```
libs/
├── safety_oracle.py          # 电影安全风险评估
├── constraint_injector.py    # 约束文本注入（20个traits的英文模板）
└── safety_filter.py          # 推理时的推荐过滤

scripts/phase2_3_saferec/
├── generate_saferec_dataset.py  # 数据集生成主脚本
├── convert_to_hf_dataset.py     # 转换为HuggingFace格式
└── README.md                     # 本文档
```

## 使用流程

### 1. 生成SafeRec数据集

```bash
python scripts/phase2_3_saferec/generate_saferec_dataset.py \
    --input_path data/phase0_trait_assignment/saferec_sft_8k_dataset.json \
    --output_path data/phase2_3_saferec/saferec_sft_final.json \
    --injection_rate 1.0 \
    --risk_threshold 0.66
```

**参数说明：**
- `--input_path`: Phase 0的输出（带trait标注的数据）
- `--output_path`: SafeRec数据集输出路径
- `--injection_rate`: 约束注入比例（1.0 = 100%）
- `--risk_threshold`: 安全风险阈值（0.66 = 高于此值视为不安全）
- `--no_cot`: 禁用CoT推理生成

### 2. 转换为HuggingFace格式

```bash
python scripts/phase2_3_saferec/convert_to_hf_dataset.py \
    --input_path data/phase2_3_saferec/saferec_sft_final.json \
    --output_path downloaded_datasets/processed_datasets/saferec_sft_dataset \
    --train_ratio 0.9
```

### 3. 训练SFT模型

```bash
python train_sft_safe.py \
    --dataset_path downloaded_datasets/processed_datasets/saferec_sft_dataset \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --num_train_epochs 10
```

## 数据格式

### 输入格式（Phase 0输出）

```json
{
  "sample_id": "train_2",
  "prompt": [{"role": "user", "content": "..."}],
  "completion": [{"role": "assistant", "content": "Movie1 (2020)\nMovie2 (2021)\n..."}],
  "assigned_trait": "Anti-gore / squeamish",
  "groundtruth_with_release_year": [["Movie1", "2020"], ...],
  ...
}
```

### 输出格式（SafeRec）

```json
{
  // 原始字段保留
  "sample_id": "train_2",
  "prompt": [{"role": "user", "content": "I don't like gory content. Recommend..."}],
  "completion": [{
    "role": "assistant",
    "content": "<reasoning>\nSafety Analysis:...\n</reasoning>\n<solution>\nMovie1 (2020)\nMovie2 (2021)\n</solution>"
  }],

  // 新增字段
  "has_constraint_injection": true,
  "injected_traits": ["Anti-gore / squeamish"],
  "constraint_text": "I don't like gory or bloody content. ",

  "original_recommendations": [{"title": "Movie1", "year": 2020}, ...],
  "original_recommendation_count": 20,

  "safe_recommendations": [
    {"title": "SafeMovie1", "year": 2020, "imdb_id": "1234567"},
    ...
  ],
  "safe_recommendation_count": 12,
  "safe_recommendation_imdb_ids": ["1234567", "2345678", ...],

  "filtered_recommendations": [
    {
      "title": "UnsafeMovie",
      "year": 2020,
      "violations": ["Anti-gore / squeamish: risk=0.92"],
      "imdb_id": "3456789"
    },
    ...
  ],
  "filtered_recommendation_count": 8,

  "has_cot": true,
  "cot_reasoning": "Safety Analysis:..."
}
```

### Completion的XML结构

```xml
<reasoning>
Safety Analysis:
User preferences to avoid: Anti-gore / squeamish

Movies filtered due to safety concerns:
- Saw (2004): Anti-gore / squeamish: risk=1.00
- Hostel (2005): Anti-gore / squeamish: risk=0.95

Safe recommendations selected: 12 movies
</reasoning>
<solution>
The Grand Budapest Hotel (2014)
Amélie (2001)
The Princess Bride (1987)
...
</solution>
```

**XML格式的优势：**
1. 结构清晰，易于解析
2. 可以选择性提取`<solution>`部分进行评估
3. 训练时模型学会生成结构化输出
4. 推理时可以控制是否生成`<reasoning>`

## 约束模板

支持20个traits的英文约束模板，例如：

```python
"Anti-gore / squeamish": [
    "I don't like gory or bloody content. ",
    "Please avoid movies with excessive blood and gore. ",
    "I'm squeamish, so nothing too graphic please. ",
    ...
]

"Kid-safety / child harm sensitive": [
    "Something suitable to watch with kids please. ",
    "Family-friendly movies only. ",
    "Please avoid any content harmful to children. ",
    ...
]
```

## 统计信息

以6670个样本的完整数据集为例：

```
Total samples: 6670
Samples with injection: 6670 (100%)
Samples with CoT: 6622 (99.3%)

Recommendations:
  Original: 133400 (avg 20.0/sample)
  Safe: 85517 (avg 12.8/sample)
  Filtered: 47883 (avg 7.2/sample)

Top Traits:
  Avoid torture & extreme violence: 2791
  Sexual violence sensitive: 1243
  Anti-gore / squeamish: 826
  Self-harm & suicide sensitive: 592
  Kid-safety / child harm sensitive: 515
```

## 核心模块API

### SafetyOracle

```python
from libs.safety_oracle import create_oracle

oracle = create_oracle("/path/to/project")

# 检查电影安全性
result = oracle.check_safety(
    title="Saw",
    year=2004,
    constraints={"Anti-gore / squeamish": True},
    threshold=0.66
)
print(result.is_safe)  # False
print(result.violations)  # ["Anti-gore / squeamish: risk=1.00"]
```

### ConstraintInjector

```python
from libs.constraint_injector import ConstraintInjector

injector = ConstraintInjector(seed=42)

result = injector.inject(
    prompt="Recommend some action movies",
    traits=["Anti-gore / squeamish"]
)
print(result.injected_prompt)
# "I don't like gory or bloody content. Recommend some action movies"
```

### SafetyFilter

```python
from libs.safety_filter import SafetyFilter

safety_filter = SafetyFilter(oracle)

result = safety_filter.filter_movies(
    movies=[("Saw", 2004), ("Toy Story", 1995)],
    constraints={"Anti-gore / squeamish": True}
)
print(result.safe_movies)  # [("Toy Story", 1995)]
```

## 注意事项

1. **内存使用**：完整数据集生成需要约500MB内存
2. **处理速度**：约500个样本/分钟
3. **数据依赖**：
   - `downloaded_datasets/movie_trait_sensitivity.json`
   - `data/phase1_mapping/title_to_imdb.pkl`
4. **Risk Threshold**：默认0.66，可调整以平衡安全性和推荐数量

## 未来改进

1. **多语言支持**：添加中文约束模板
2. **动态阈值**：根据trait类型调整阈值
3. **解释性增强**：在CoT中添加更详细的风险分析
4. **批量优化**：使用并行处理加速生成
