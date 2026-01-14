# Data Directory

数据文件按 SafeRec 实施阶段组织。

---

## 目录结构

```
data/
├── phase0_trait_assignment/    # Phase 0: Trait Assignment 数据
├── phase1_mapping/             # Phase 1: Title Mapping 数据
├── raw_downloads/              # 原始下载数据
└── README.md                   # 本文件
```

---

## Phase 0: Trait Assignment

**目录**: `phase0_trait_assignment/`

### 文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `sft_filtered_8k.json` | ~21 MB | Step 1: 筛选后的 8k 样本（GT ≥ 3） |
| `sft_with_assigned_traits.json` | - | Step 2: 带 assigned_trait 的样本（待生成） |
| `saferec_sft_8k_dataset.json` | - | Step 3: 过滤后的最终数据集（待生成） |
| `trait_stats/` | - | Step 4: 统计和可视化（待生成） |

### 数据流

```
SFT Dataset (95,753 samples)
    ↓ filter_sft_samples.py
sft_filtered_8k.json (8,000 samples)
    ↓ assign_traits_via_gpt.py
sft_with_assigned_traits.json (8,000 samples with traits)
    ↓ filter_violating_groundtruth.py
saferec_sft_8k_dataset.json (~7,200 samples)
    ↓ analyze_trait_distribution.py
trait_stats/ (statistics & visualizations)
```

### 样本格式

```json
{
  "sample_id": "train_12345",
  "assigned_trait": "Anti-gore / squeamish",
  "assignment_success": true,
  "prompt": [...],
  "completion": [...],
  "groundtruth_with_release_year": [
    ["Movie A", "2015"],
    ["Movie B", "2018"]
  ],
  "num_gt_removed": 2,
  "removed_groundtruth": [...]
}
```

---

## Phase 1: Title ↔ imdbId Mapping

**目录**: `phase1_mapping/`

### 文件

| 文件 | 大小 | 说明 | 状态 |
|------|------|------|------|
| `title_to_imdb.pkl` | 34 MB | 主映射表（922,204 条） | ✅ 已生成 |
| `title_to_imdb.stats.json` | <1 KB | 映射统计 | ✅ 已生成 |
| `mapping_coverage_report.json` | 16 KB | SFT 数据集覆盖率测试 | ✅ 已生成 |

### 映射表结构

```python
# title_to_imdb.pkl
{
    ("toy story", 1995): "0114709",
    ("jumanji", 1995): "0113497",
    ...
}
```

### 覆盖率

- **映射成功率**: 93.2% (SFT 数据集)
- **Trait 覆盖率**: 70.5% (可查到 trait sensitivity)
- **总映射条目**: 922,204

---

## Raw Downloads

**目录**: `raw_downloads/`

### 文件

| 文件 | 大小 | 来源 |
|------|------|------|
| `title.basics.tsv.gz` | 207 MB | IMDb Datasets |

### 重新下载

```bash
cd data/raw_downloads
wget https://datasets.imdbws.com/title.basics.tsv.gz
```

---

## 外部数据依赖

以下数据在项目根目录或 `downloaded_datasets/` 中：

| 文件 | 位置 | 说明 |
|------|------|------|
| `movie_trait_sensitivity.json` | `downloaded_datasets/` | 24,408 电影 × 20 traits |
| `traits_warnings.json` | 项目根目录 | 20 个 trait 定义 |
| `sft_dataset/` | `downloaded_datasets/processed_datasets/` | SFT 训练数据（95,753 样本） |

---

## 磁盘使用

```bash
# 查看各目录大小
du -sh phase0_trait_assignment/
du -sh phase1_mapping/
du -sh raw_downloads/
```

**预计总大小**:
- Phase 0: ~50 MB（完成后）
- Phase 1: ~50 MB
- Raw: ~210 MB

---

## 清理策略

### 保留核心文件

如果需要释放空间，可删除中间文件：

```bash
# 删除原始下载（可重新下载）
rm data/raw_downloads/title.basics.tsv.gz

# 删除中间步骤（可重新生成）
rm data/phase0_trait_assignment/sft_filtered_8k.json
rm data/phase0_trait_assignment/sft_with_assigned_traits.json
```

### 必须保留

- `data/phase1_mapping/title_to_imdb.pkl` - 核心映射表
- `data/phase0_trait_assignment/saferec_sft_8k_dataset.json` - 最终训练数据

---

## 数据版本

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-01-14 | Phase 1 映射完成 |
| v0.2 | TBD | Phase 0 完成（待生成） |

---

## 相关命令

```bash
# 列出所有数据文件
find data -type f -name "*.json" -o -name "*.pkl"

# 检查文件完整性
python3 -c "import pickle; pickle.load(open('data/phase1_mapping/title_to_imdb.pkl', 'rb'))"

# 查看 JSON 文件样本
head -100 data/phase0_trait_assignment/sft_filtered_8k.json | jq .
```
