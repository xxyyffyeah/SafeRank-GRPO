# SafeRec SFT 实施计划

## 概述

本计划基于 [SAFEREC_SFT_PLAN.md](./SAFEREC_SFT_PLAN.md) 的设计方案，结合当前已有的 `movie_trait_sensitivity.json` 数据，制定具体可执行的实施步骤。

## 当前资产清单

| 文件 | 路径 | 说明 |
|------|------|------|
| Trait Sensitivity | `downloaded_datasets/movie_trait_sensitivity.json` | 24,408 电影 × 20 traits |
| Trait Config | `traits_with_imdb_parentguide_weights.json` | 权重配置 |
| SFT Dataset | `downloaded_datasets/processed_datasets/sft_dataset/` | 95,753 训练样本 |

---

## Phase 0: Trait Assignment (自动标注)

### 0.1 概述

**目标**: 使用 ChatGPT API 为 8,000 个高质量 SFT 样本自动分配用户敏感特征（trait）

**详细计划**: 参见 [TRAIT_ASSIGNMENT_PLAN.md](./TRAIT_ASSIGNMENT_PLAN.md)

### 0.2 流程概览

```
SFT Dataset (95,753)
    ↓ 筛选 GT ≥ 3
8,000 samples
    ↓ ChatGPT API 标注
Assigned Traits
    ↓ 过滤违规 GT
SafeRec Training Data (~7,200 samples)
```

### 0.3 关键产出

| 输出 | 说明 |
|------|------|
| `data/saferec_sft_8k_dataset.json` | 带 assigned_trait 的训练数据 |
| `data/trait_stats/` | Trait 分布统计和可视化 |

### 0.4 为什么需要这一步？

| 挑战 | Trait Assignment 解决方案 |
|------|--------------------------|
| **人工标注成本高** | GPT-4o 自动标注，成本 ~$22 |
| **Trait 分布不均** | 智能分配确保各 trait 有代表样本 |
| **GT 噪声** | 自动过滤违反 trait 的 GT 电影 |

### 0.5 实施时间

- 脚本开发: 2-3 天
- API 调用: ~33 分钟
- 人工验证: 2 小时

---

## Phase 1: Title ↔ imdbId 映射构建

### 1.1 数据源

使用 IMDb 官方数据集：
- **URL**: https://datasets.imdbws.com/title.basics.tsv.gz
- **字段**: tconst (imdbId), titleType, primaryTitle, originalTitle, startYear, ...

### 1.2 映射表结构

```python
# data/title_to_imdb.pkl
{
    ("toy story", 1995): "0114709",
    ("jumanji", 1995): "0113497",
    ...
}
```

### 1.3 实现脚本

```bash
scripts/build_title_mapping.py
```

**输出文件**:
- `data/title_to_imdb.pkl` - 主映射表
- `data/title_mapping_stats.json` - 覆盖率统计

---

## Phase 2: SafetyOracle 模块

### 2.1 文件结构

```
libs/
├── safety_oracle.py      # Oracle 核心类
├── constraint_injector.py # 约束注入
└── safety_filter.py       # Safety Filter
```

### 2.2 SafetyOracle 接口

```python
class SafetyOracle:
    def __init__(self, trait_sensitivity_path, title_mapping_path):
        ...

    def get_movie_traits(self, title: str, year: int) -> dict:
        """返回电影的所有 trait 敏感度"""

    def check_safety(self, title: str, year: int, constraints: dict) -> Tuple[bool, List[str]]:
        """检查电影是否满足约束，返回 (is_safe, violation_reasons)"""
```

---

## Phase 3: 约束注入模块

### 3.1 基于 20 Traits 的约束模板

```python
TRAIT_CONSTRAINT_TEMPLATES = {
    "Anti-gore / squeamish": [
        "我不喜欢血腥暴力的内容，",
        "不要太血腥的电影，",
        "请避免有过多血腥场面的，",
    ],
    "Horror avoider (avoids scares & supernatural)": [
        "我不看恐怖片，",
        "不要有吓人的镜头，",
        "我胆子小，不要恐怖的，",
    ],
    "Kid-safety / child harm sensitive": [
        "适合和孩子一起看的，",
        "要适合家庭观看的，",
        "不要有伤害儿童的内容，",
    ],
    # ... 其他 traits
}
```

### 3.2 注入策略

- **注入率**: 30-40% 的样本注入约束
- **约束数量**: 1-3 个 traits per sample
- **位置**: Prompt 开头或用户偏好描述中

---

## Phase 4: CoT 数据生成

### 4.1 数据结构

```python
{
    "prompt": [{"role": "user", "content": "我不喜欢血腥暴力...推荐电影"}],
    "completion": [{"role": "assistant", "content": """
思考过程：
用户要求避免血腥暴力内容。
- 考虑《Saw》(2004)，但该电影 Anti-gore 风险分 0.92，包含大量血腥场面，排除。
- 考虑《The Shining》(1980)，Horror 风险分 0.85，排除。

推荐列表：
The Grand Budapest Hotel (2014)
...
"""}],
    "constraints": {"Anti-gore / squeamish": True},
    "has_cot": True
}
```

### 4.2 生成脚本

```bash
scripts/generate_saferec_sft_data.py \
    --input_path downloaded_datasets/processed_datasets/sft_dataset \
    --output_path downloaded_datasets/processed_datasets/saferec_sft_dataset \
    --trait_sensitivity_path downloaded_datasets/movie_trait_sensitivity.json \
    --title_mapping_path data/title_to_imdb.pkl \
    --injection_rate 0.35 \
    --cot_threshold 0.66
```

---

## Phase 5: 训练流程修改

### 5.1 修改 train_sft.py

- 支持加载 SafetyOracle
- 支持新的数据格式 (has_cot, constraints 字段)

### 5.2 配置文件

```yaml
# configs/saferec_sft.yaml
safety:
  enabled: true
  trait_sensitivity_path: downloaded_datasets/movie_trait_sensitivity.json
  title_mapping_path: data/title_to_imdb.pkl
  injection_rate: 0.35
  cot_threshold: 0.66
```

---

## 实施时间线

| Phase | 任务 | 依赖 | 状态 |
|-------|------|------|------|
| **0.1** | 筛选 GT ≥ 3 的样本 (8k) | - | 待开始 |
| **0.2** | 开发 GPT 标注脚本 | - | 待开始 |
| **0.3** | 运行 ChatGPT API 标注 | 0.1, 0.2 | 待开始 |
| **0.4** | 过滤违规 GT 电影 | 0.3, 1.3 | 待开始 |
| **0.5** | 统计 Trait 分布 | 0.4 | 待开始 |
| **1.1** | 下载 IMDb title.basics | - | ✅ 已完成 |
| **1.2** | 构建 title→imdbId 映射 | 1.1 | ✅ 已完成 |
| **1.3** | 测试 SFT 数据集覆盖率 | 1.2 | ✅ 已完成 |
| **2.1** | 实现 SafetyOracle | 1.2 | 待开始 |
| **3.1** | 实现约束注入模块 | 2.1, 0.5 | 待开始 |
| **4.1** | 生成 SafeRec SFT 数据 | 3.1, 0.4 | 待开始 |
| **5.1** | 修改训练流程 | 4.1 | 待开始 |

---

## 预期产出

### Phase 0 产出
1. **Trait 标注数据**: `data/saferec_sft_8k_dataset.json` (8k 样本)
2. **Trait 分布统计**: `data/trait_stats/stats.json`
3. **可视化图表**: `data/trait_stats/*.png`

### Phase 1-2 产出
4. **Title 映射表**: `data/title_to_imdb.pkl` ✅
5. **映射覆盖率报告**: `data/mapping_coverage_report.json` ✅
6. **SafetyOracle 模块**: `libs/safety_oracle.py`

### Phase 3-5 产出
7. **完整 SafeRec SFT 数据集**: `downloaded_datasets/processed_datasets/saferec_sft_dataset/`
8. **约束注入模块**: `libs/constraint_injector.py`
9. **配置文件**: `configs/saferec_sft.yaml`

---

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| Title 匹配率低 | 无法为大量电影标注安全属性 | 使用模糊匹配 + 年份容差 |
| 约束样本过少 | 模型学不到安全对齐 | 提高注入率或数据增强 |
| CoT 质量差 | 模型学到错误推理 | 人工抽检 + 质量过滤 |
