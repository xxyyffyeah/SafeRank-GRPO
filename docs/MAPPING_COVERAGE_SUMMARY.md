# Title → imdbId 映射覆盖率报告

## 执行总结

✅ **结论**：映射方案可行，覆盖率满足 SafeRec 实施需求

---

## 关键指标

| 指标 | 结果 | 目标 | 状态 |
|------|------|------|------|
| **Title → imdbId 映射率** | 93.2% | ≥ 70% | ✅ 超标完成 |
| **Trait Sensitivity 覆盖率** | 70.5% | ≥ 60% | ✅ 超标完成 |
| **总映射条目** | 922,204 | - | - |
| **Trait 数据电影数** | 24,408 | - | - |

---

## 数据概览

### 测试范围
- **SFT 样本数**: 10,000 / 95,753
- **提取的唯一电影**: 7,389
  - Groundtruth 电影: 3,949
  - Completion 电影: 7,385

### 映射结果
```
总电影: 7,389
  ├─ 映射成功: 6,883 (93.2%)
  │   ├─ 有 Trait 数据: 5,210 (70.5%)
  │   └─ 无 Trait 数据: 1,673 (22.6%)
  └─ 映射失败: 506 (6.8%)
```

---

## 未映射电影分析

### 主要原因：TV 系列被排除

未映射的电影样本（前20）：
```
Ted Lasso (2020)                 - TV Series
Virgin River (2019)              - TV Series
Only Murders in the Building     - TV Series
The Expanse (2015)               - TV Series
Star Trek: Deep Space Nine       - TV Series
Castlevania (2017)               - TV Series
Slow Horses (2022)               - TV Series
...
```

### 建议改进

如果需要覆盖 TV 系列，可在 [scripts/build_title_mapping.py:83](../scripts/build_title_mapping.py#L83) 修改：

```python
# 当前：只包含电影
if title_type not in ("movie", "tvMovie", "tvSpecial"):
    continue

# 改为：包含所有视频内容
if title_type not in ("movie", "tvMovie", "tvSpecial", "tvSeries", "tvMiniSeries"):
    continue
```

**预期提升**：映射率 93.2% → 97%+

---

## 年份分布

推荐系统主要覆盖 2010-2022 年的电影（占比 ~70%）：

| 年份 | 电影数 | 备注 |
|------|--------|------|
| 2022 | 284 | 最新电影 |
| 2019 | 283 | - |
| 2018 | 257 | - |
| 2021 | 237 | - |
| 2017 | 230 | - |
| 2016 | 217 | - |
| 2014 | 216 | - |
| 2013 | 208 | - |
| 2015 | 194 | - |
| 2011 | 188 | - |

---

## SafeRec 可行性评估

### ✅ 可行性确认

| 场景 | 覆盖率 | 评估 |
|------|--------|------|
| **用户约束注入** | 93.2% | 绝大多数电影可映射 |
| **Safety Filter** | 70.5% | 7成电影可进行安全检查 |
| **CoT 生成** | 70.5% | 足够生成高质量训练数据 |

### ⚠️ 边界情况处理

对于未映射或无 trait 数据的电影：

**策略1 - 保守（推荐）**:
```python
if imdb_id is None or imdb_id not in trait_data:
    return True, []  # 未知电影默认通过
```

**策略2 - 激进**:
```python
if imdb_id is None or imdb_id not in trait_data:
    return False, ["Unknown movie - safety uncertain"]
```

**建议**：使用策略1，避免过度惩罚新电影或冷门电影

---

## 下一步行动

### Phase 1: 核心模块实现 ✅ 数据准备完成

- [x] Title → imdbId 映射表 ([data/title_to_imdb.pkl](../data/title_to_imdb.pkl))
- [x] 覆盖率验证 (93.2% 映射率)
- [ ] 实现 `SafetyOracle` 类
- [ ] 实现 `ConstraintInjector` 类

### Phase 2: 数据集生成

- [ ] 基于约束模板生成 SafeRec SFT 数据
- [ ] 质量检查和人工抽样验证

### Phase 3: 训练集成

- [ ] 修改 reward function 支持安全检查
- [ ] 配置文件和训练流程更新

---

## 技术细节

### 映射查找逻辑

```python
def lookup_imdb_id(title: str, year: int) -> str | None:
    norm_title = normalize_title(title)  # 小写 + 去标点

    # 1. 精确匹配
    if (norm_title, year) in mapping:
        return mapping[(norm_title, year)]

    # 2. 年份容差 ±2
    for d in [1, 2]:
        for y in [year + d, year - d]:
            if (norm_title, y) in mapping:
                return mapping[(norm_title, y)]

    return None
```

### 标题标准化

```python
"The Shawshank Redemption" → "the shawshank redemption"
"Star Wars: Episode IV"    → "star wars episode iv"
```

---

## 参考文件

| 文件 | 说明 |
|------|------|
| [scripts/build_title_mapping.py](../scripts/build_title_mapping.py) | 映射构建脚本 |
| [scripts/test_mapping_coverage.py](../scripts/test_mapping_coverage.py) | 覆盖率测试脚本 |
| [data/title_to_imdb.pkl](../data/title_to_imdb.pkl) | 映射数据（922K 条目） |
| [data/mapping_coverage_report.json](../data/mapping_coverage_report.json) | 详细报告 |
| [CURRENT_EVALUATION_MECHANISM.md](./CURRENT_EVALUATION_MECHANISM.md) | 评估机制说明 |
| [SAFEREC_IMPLEMENTATION_PLAN.md](./SAFEREC_IMPLEMENTATION_PLAN.md) | 实施计划 |
