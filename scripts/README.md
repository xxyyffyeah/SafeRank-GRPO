# Scripts Directory

脚本按 SafeRec 实施阶段组织。

---

## 目录结构

```
scripts/
├── phase0_trait_assignment/    # Phase 0: Trait Assignment
├── phase1_mapping/              # Phase 1: Title→imdbId Mapping
└── README.md                    # 本文件
```

---

## Phase 0: Trait Assignment (自动标注)

**目录**: `phase0_trait_assignment/`

### 脚本

| 脚本 | 说明 | 用途 |
|------|------|------|
| `filter_sft_samples.py` | 筛选 GT ≥ 3 的样本 | Step 1 |
| `assign_traits_via_gpt.py` | ChatGPT API 自动标注 | Step 2 |
| `filter_violating_groundtruth.py` | 过滤违规 GT 电影 | Step 3 |
| `analyze_trait_distribution.py` | 统计 Trait 分布 | Step 4 |
| `run_trait_assignment_pipeline.sh` | 完整 Pipeline | 一键运行 |

### 使用示例

```bash
# 完整流程（需要 OpenAI API Key）
export OPENAI_API_KEY="sk-..."
bash scripts/phase0_trait_assignment/run_trait_assignment_pipeline.sh

# 测试模式（仅处理 100 样本）
bash scripts/phase0_trait_assignment/run_trait_assignment_pipeline.sh --test

# 单独运行某一步
python3 scripts/phase0_trait_assignment/filter_sft_samples.py
```

### 输入输出

| 输入 | 输出 |
|------|------|
| `downloaded_datasets/processed_datasets/sft_dataset/` | `data/phase0_trait_assignment/sft_filtered_8k.json` |
| `traits_warnings.json` | `data/phase0_trait_assignment/sft_with_assigned_traits.json` |
| `data/phase1_mapping/title_to_imdb.pkl` | `data/phase0_trait_assignment/saferec_sft_8k_dataset.json` |
| `downloaded_datasets/movie_trait_sensitivity.json` | `data/phase0_trait_assignment/trait_stats/` |

---

## Phase 1: Title ↔ imdbId Mapping (映射构建)

**目录**: `phase1_mapping/`

### 脚本

| 脚本 | 说明 | 状态 |
|------|------|------|
| `build_title_mapping.py` | 构建 title→imdbId 映射表 | ✅ 已完成 |
| `test_mapping_coverage.py` | 测试映射覆盖率 | ✅ 已完成 |

### 使用示例

```bash
# 构建映射（已完成，无需重新运行）
python3 scripts/phase1_mapping/build_title_mapping.py

# 测试覆盖率
python3 scripts/phase1_mapping/test_mapping_coverage.py --max_samples 10000
```

### 输入输出

| 输入 | 输出 |
|------|------|
| `data/raw_downloads/title.basics.tsv.gz` | `data/phase1_mapping/title_to_imdb.pkl` |
| `downloaded_datasets/movie_trait_sensitivity.json` | `data/phase1_mapping/title_to_imdb.stats.json` |
| `downloaded_datasets/processed_datasets/sft_dataset/` | `data/phase1_mapping/mapping_coverage_report.json` |

---

## 相关文档

- [TRAIT_ASSIGNMENT_PLAN.md](../docs/TRAIT_ASSIGNMENT_PLAN.md) - Phase 0 详细计划
- [QUICK_START_TRAIT_ASSIGNMENT.md](../docs/QUICK_START_TRAIT_ASSIGNMENT.md) - 快速上手指南
- [MAPPING_COVERAGE_SUMMARY.md](../docs/MAPPING_COVERAGE_SUMMARY.md) - Phase 1 覆盖率报告
- [SAFEREC_IMPLEMENTATION_PLAN.md](../docs/SAFEREC_IMPLEMENTATION_PLAN.md) - 完整实施计划

---

## 常见问题

### Q: 为什么 Phase 0 脚本依赖 Phase 1 的输出？

A: Phase 0 的 Step 3（过滤违规 GT）需要使用 Phase 1 构建的 `title_to_imdb.pkl` 映射表来查找电影的 imdbId，进而获取 trait sensitivity 数据。

### Q: 如何更新映射表？

```bash
# 重新下载 IMDb 数据
cd data/raw_downloads
wget https://datasets.imdbws.com/title.basics.tsv.gz

# 重新构建映射
python3 scripts/phase1_mapping/build_title_mapping.py
```

### Q: Pipeline 脚本失败了怎么办？

每个 step 都是独立的，可以单独重新运行：

```bash
# 例如：只重新运行 Step 2
python3 scripts/phase0_trait_assignment/assign_traits_via_gpt.py \
    --input_path data/phase0_trait_assignment/sft_filtered_8k.json \
    --output_path data/phase0_trait_assignment/sft_with_assigned_traits.json
```
