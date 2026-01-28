# SafeRec 文档索引

SafeRec: Safety-Aligned Recommendation System - 完整实施文档

---

## 📋 核心文档

### 1. 实施计划

| 文档 | 说明 |
|------|------|
| [SAFEREC_IMPLEMENTATION_PLAN.md](./SAFEREC_IMPLEMENTATION_PLAN.md) | 📌 **主计划** - 完整实施路线图 |
| [SAFEREC_SFT_PLAN.md](./SAFEREC_SFT_PLAN.md) | 原始 SafeRec SFT 理论设计 |

### 2. Trait Assignment（Phase 0）

| 文档 | 说明 |
|------|------|
| [TRAIT_ASSIGNMENT_PLAN.md](./TRAIT_ASSIGNMENT_PLAN.md) | 📌 **Trait 自动标注详细计划** |
| [QUICK_START_TRAIT_ASSIGNMENT.md](./QUICK_START_TRAIT_ASSIGNMENT.md) | 🚀 **快速上手指南** |

### 3. Safe-Rank-GRPO（Phase 6）

| 文档 | 说明 |
|------|------|
| [SAFE_RANK_GRPO.md](./SAFE_RANK_GRPO.md) | 📌 **Safe-Rank-GRPO 训练实现** |

### 4. 技术参考

| 文档 | 说明 |
|------|------|
| [IMDB_PARENTGUIDE_INTEGRATION.md](./IMDB_PARENTGUIDE_INTEGRATION.md) | IMDb Parent Guide 集成方案 |
| [CURRENT_EVALUATION_MECHANISM.md](./CURRENT_EVALUATION_MECHANISM.md) | 当前评估机制分析 |
| [MAPPING_COVERAGE_SUMMARY.md](./MAPPING_COVERAGE_SUMMARY.md) | Title→imdbId 映射覆盖率报告 |

---

## 🗺️ 实施路线图

```
Phase 0: Trait Assignment (自动标注)
    ├── [0.1] 筛选 GT ≥ 3 的样本 (8k)
    ├── [0.2] ChatGPT API 自动标注
    ├── [0.3] 过滤违规 GT
    └── [0.4] 统计 Trait 分布

Phase 1: Title ↔ imdbId 映射
    ├── [1.1] 下载 IMDb title.basics ✅
    ├── [1.2] 构建映射表 ✅
    └── [1.3] 测试覆盖率 ✅

Phase 2: SafetyOracle 模块
    ├── [2.1] 实现 SafetyOracle 类
    └── [2.2] 实现 TitleToImdbMapper

Phase 3: 约束注入
    ├── [3.1] 设计约束模板
    └── [3.2] 实现 ConstraintInjector

Phase 4: CoT 数据生成
    ├── [4.1] 设计 CoT 格式
    └── [4.2] 生成训练数据

Phase 5: 训练集成
    ├── [5.1] 修改训练流程
    └── [5.2] 配置文件

Phase 6: Safe-Rank-GRPO ✅
    ├── [6.1] safe_reward_funcs.py ✅
    └── [6.2] train_rank_grpo_safe.py ✅
```

---

## 📊 当前进度

### ✅ 已完成

- [x] **Phase 1 完成** (Title 映射)
  - 922,204 条映射
  - 93.2% SFT 数据集覆盖率
  - 70.5% Trait Sensitivity 覆盖率

- [x] **Phase 6 完成** (Safe-Rank-GRPO)
  - libs/safe_reward_funcs.py
  - train_rank_grpo_safe.py
  - 支持 per-rank 安全惩罚

### 🚧 进行中

- [ ] **Phase 0** - Trait Assignment
  - 计划文档已完成
  - 待实现脚本

### 📅 待开始

- [ ] Phase 2-5

---

## 🎯 快速开始

### 新手入门

1. 阅读 [SAFEREC_IMPLEMENTATION_PLAN.md](./SAFEREC_IMPLEMENTATION_PLAN.md) 了解全貌
2. 按照 [QUICK_START_TRAIT_ASSIGNMENT.md](./QUICK_START_TRAIT_ASSIGNMENT.md) 运行 Phase 0
3. 查看 [MAPPING_COVERAGE_SUMMARY.md](./MAPPING_COVERAGE_SUMMARY.md) 了解数据覆盖情况

### 开发者

参考各 Phase 的实施细节：
- Phase 0: [TRAIT_ASSIGNMENT_PLAN.md](./TRAIT_ASSIGNMENT_PLAN.md)
- Phase 1: [CURRENT_EVALUATION_MECHANISM.md](./CURRENT_EVALUATION_MECHANISM.md)
- Phase 2-5: [SAFEREC_IMPLEMENTATION_PLAN.md](./SAFEREC_IMPLEMENTATION_PLAN.md)

---

## 📂 项目文件结构

```
Rank-GRPO/
├── docs/                              # 文档（本目录）
│   ├── README.md                      # 本文件
│   ├── SAFEREC_IMPLEMENTATION_PLAN.md # 主计划
│   ├── TRAIT_ASSIGNMENT_PLAN.md       # Phase 0 详细计划
│   └── ...
│
├── scripts/                           # 实施脚本
│   ├── build_title_mapping.py         # ✅ Phase 1.2
│   ├── test_mapping_coverage.py       # ✅ Phase 1.3
│   ├── filter_sft_samples.py          # Phase 0.1（待开发）
│   ├── assign_traits_via_gpt.py       # Phase 0.2（待开发）
│   └── ...
│
├── data/                              # 数据产出
│   ├── title_to_imdb.pkl              # ✅ 映射表
│   ├── mapping_coverage_report.json   # ✅ 覆盖率报告
│   └── saferec_sft_8k_dataset.json    # Phase 0 产出（待生成）
│
├── downloaded_datasets/
│   ├── movie_trait_sensitivity.json   # ✅ Trait 数据（24,408 电影）
│   └── processed_datasets/sft_dataset # ✅ 原始 SFT 数据
│
└── libs/                              # 模块库
    ├── safety_oracle.py               # ✅ Phase 2
    ├── safe_reward_funcs.py           # ✅ Phase 6 安全奖励函数
    └── constraint_injector.py         # Phase 3（待开发）
```

---

## 💡 关键概念

### Traits (用户敏感特征)

20 个预定义的用户敏感特征，例如：
- Anti-gore / squeamish
- Horror avoider
- Kid-safety / child harm sensitive
- Sexual violence sensitive
- ...

参见：[traits_warnings.json](../traits_warnings.json)

### Trait Sensitivity Data

24,408 部电影的敏感度评分，结合：
- **DoesTheDogDie (DDD)** 细粒度标签
- **IMDb Parent Guide** 强度评级

参见：[IMDB_PARENTGUIDE_INTEGRATION.md](./IMDB_PARENTGUIDE_INTEGRATION.md)

### SafeRec Training

通过以下步骤训练安全对齐的推荐模型：
1. **Remap**: 为用户对话注入安全约束
2. **Reflect**: 过滤违反约束的推荐
3. **Adjust**: 生成 CoT 数据解释过滤原因

参见：[SAFEREC_SFT_PLAN.md](./SAFEREC_SFT_PLAN.md)

---

## 🔗 外部资源

### 数据源

- [IMDb Datasets](https://datasets.imdbws.com/) - title.basics.tsv.gz
- [DoesTheDogDie API](https://www.doesthedogdie.com/) - 细粒度触发标签
- [IMDb Parent Guide](https://www.imdb.com/parentalguide/) - 内容强度评级

### 相关论文

- Rank-GRPO 原始论文（Netflix Research 2025）
- SafeRec 概念来源

---

## 📞 支持

如有问题，请参考：
1. 各文档中的 "常见问题" 部分
2. 项目 issue tracker
3. 相关脚本的 `--help` 输出

---

## 📝 更新日志

| 日期 | 版本 | 更新内容 |
|------|------|---------|
| 2026-01-28 | v0.4 | 完成 Phase 6 Safe-Rank-GRPO 实现 |
| 2026-01-14 | v0.3 | 添加 Trait Assignment 计划（Phase 0） |
| 2026-01-14 | v0.2 | 完成 Phase 1 映射构建 |
| 2026-01-13 | v0.1 | 初始文档结构 |
