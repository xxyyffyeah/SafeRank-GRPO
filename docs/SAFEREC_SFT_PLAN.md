# SafeRec SFT 改造计划 - 可行性分析与实施方案

## 一、当前系统现状总结

| 组件 | 当前状态 | SafeRec 需求 |
|------|----------|--------------|
| **外部知识库** | `gt_catalog.pkl` 仅含 (title, year) | 需要电影属性标签 (Rating, Tags) |
| **Prompt 格式** | 简单推荐指令，无约束 | 需注入安全约束 |
| **数据处理** | 直接加载，无过滤 | 需两阶段过滤 (Safety + Relevance) |
| **Completion** | 纯推荐列表 | 需 CoT 修正数据 |

### 当前 SFT 数据集结构

```python
{
    'prompt': [{'role': 'user', 'content': '...'}],      # 用户提示
    'completion': [{'role': 'assistant', 'content': '...'}],  # 20部电影推荐
    'seen_titles': ['Movie1', 'Movie2'],                 # 已看电影
    'groundtruth_with_release_year': [['Title', 'Year'], ...]  # 真实标签
}
```

---

## 二、可行性分析

### ✅ 高可行性 (直接可实现)

1. **Prompt 约束注入** - 只需修改数据预处理脚本
2. **两阶段 Reflect** - 可在现有 `reward_funcs.py` 基础上扩展
3. **System Prompt 增强** - 修改 prompt template 即可
4. **数据格式兼容** - 现有 `SFTTrainer` 完全支持对话格式

### ⚠️ 需额外构建 (核心瓶颈)

1. **电影安全属性数据库 (Oracle)**
   - 当前 `gt_catalog` 只有 ~数万条电影
   - 需要获取 MPAA Rating、内容标签 (Violence, Gore, Sexual Content 等)
   - **数据源选项**：
     - IMDb Datasets (公开)
     - TMDB API (免费注册)
     - MovieLens + OMDB 组合

2. **约束类型设计**
   - 需要定义有限的约束类别，避免过于复杂

---

## 三、具体实施计划

### 阶段 0：构建 Oracle 数据库 (前置依赖)

```
新建文件: libs/oracle.py
新建文件: data/movie_oracle.pkl
```

**数据结构设计**：
```python
# movie_oracle.pkl 结构
{
    ('Movie Title', year): {
        'rating': 'PG-13',           # MPAA Rating: G, PG, PG-13, R, NC-17
        'tags': ['Violence', 'Gore', 'Language'],  # 内容标签
        'genres': ['Action', 'Thriller'],
        'is_adult': False,
    },
    ...
}
```

**实现方案**：
```python
# libs/oracle.py
import pickle
from typing import Dict, Set, Tuple, Optional

class MovieOracle:
    def __init__(self, oracle_path: str):
        with open(oracle_path, 'rb') as f:
            self.data: Dict[Tuple[str, int], dict] = pickle.load(f)

    def get_movie_info(self, title: str, year: int) -> Optional[dict]:
        return self.data.get((title.lower(), year))

    def check_safety(self, title: str, year: int, constraints: dict) -> Tuple[bool, str]:
        """
        Returns: (is_safe, reason)
        """
        info = self.get_movie_info(title, year)
        if not info:
            return True, "Unknown movie"  # 未知电影默认通过

        # 检查 Rating 约束
        if 'max_rating' in constraints:
            rating_order = ['G', 'PG', 'PG-13', 'R', 'NC-17']
            if rating_order.index(info['rating']) > rating_order.index(constraints['max_rating']):
                return False, f"Rating {info['rating']} exceeds {constraints['max_rating']}"

        # 检查 Tag 约束
        if 'forbidden_tags' in constraints:
            violations = set(info['tags']) & set(constraints['forbidden_tags'])
            if violations:
                return False, f"Contains forbidden content: {violations}"

        return True, "Safe"
```

---

### 阶段 1：Remap - 约束注入模块

```
新建文件: libs/constraint_injector.py
```

**约束类型定义**：
```python
# libs/constraint_injector.py
import random
from typing import Dict, List, Tuple

# 约束模板库
CONSTRAINT_TEMPLATES = {
    'age_restriction': [
        "我是{age}岁的学生，",
        "请推荐适合{age}岁孩子看的电影，",
        "我想和{age}岁的孩子一起看，",
    ],
    'no_violence': [
        "但不要有暴力场面，",
        "不要太血腥的，",
        "我不喜欢打斗场面，",
    ],
    'no_horror': [
        "不要恐怖片，",
        "我胆子小，不要吓人的，",
        "不要有突然吓人的镜头，",
    ],
    'no_explicit': [
        "不要有成人内容，",
        "要适合家庭观看的，",
    ],
}

# 约束到 Oracle 检查规则的映射
CONSTRAINT_TO_RULES = {
    'age_restriction': lambda age: {
        'max_rating': 'G' if age < 8 else ('PG' if age < 13 else 'PG-13')
    },
    'no_violence': {'forbidden_tags': ['Violence', 'Gore', 'Brutal']},
    'no_horror': {'forbidden_tags': ['Horror', 'Jump Scare', 'Disturbing']},
    'no_explicit': {'forbidden_tags': ['Sexual Content', 'Nudity'], 'max_rating': 'PG-13'},
}

class ConstraintInjector:
    def __init__(self, injection_rate: float = 0.3):
        self.injection_rate = injection_rate

    def inject(self, original_prompt: str) -> Tuple[str, Dict]:
        """
        Returns: (modified_prompt, active_constraints)
        """
        if random.random() > self.injection_rate:
            return original_prompt, {}

        # 随机选择约束类型
        constraint_type = random.choice(list(CONSTRAINT_TEMPLATES.keys()))
        template = random.choice(CONSTRAINT_TEMPLATES[constraint_type])

        # 处理特殊参数
        if constraint_type == 'age_restriction':
            age = random.choice([8, 10, 12, 14])
            constraint_text = template.format(age=age)
            rules = CONSTRAINT_TO_RULES[constraint_type](age)
        else:
            constraint_text = template
            rules = CONSTRAINT_TO_RULES[constraint_type]

        # 注入到 prompt
        modified_prompt = self._insert_constraint(original_prompt, constraint_text)

        return modified_prompt, {constraint_type: rules}

    def _insert_constraint(self, prompt: str, constraint: str) -> str:
        # 在用户描述开头或结尾插入约束
        if "USER:" in prompt:
            parts = prompt.split("USER:", 1)
            return parts[0] + "USER: " + constraint + parts[1].strip()
        return constraint + prompt
```

---

### 阶段 2：Reflect - 两阶段判别器

```
修改文件: libs/reward_funcs.py
新建文件: libs/safety_filter.py
```

**Safety Filter 实现**：
```python
# libs/safety_filter.py
from typing import List, Tuple, Dict
from libs.oracle import MovieOracle

class SafetyFilter:
    def __init__(self, oracle: MovieOracle):
        self.oracle = oracle

    def filter_recommendations(
        self,
        recommendations: List[Tuple[str, int]],
        constraints: Dict
    ) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int, str]]]:
        """
        Returns: (safe_recs, unsafe_recs_with_reason)
        """
        safe = []
        unsafe = []

        for title, year in recommendations:
            is_safe, reason = self.oracle.check_safety(title, year, constraints)
            if is_safe:
                safe.append((title, year))
            else:
                unsafe.append((title, year, reason))

        return safe, unsafe
```

**修改 reward_funcs.py**：
```python
# 在现有奖励函数基础上增加安全惩罚
def make_safety_aware_reward_func(rec_num, gt_catalog, oracle, safety_penalty=-1.0):
    def reward_func(completions, prompts, constraints_list, **kwargs):
        rewards = []
        for completion, constraints in zip(completions, constraints_list):
            # 解析推荐
            recs = parse_recommendations(completion)

            # 第一关：Safety Filter
            safe_recs, unsafe_recs = safety_filter.filter_recommendations(recs, constraints)

            # 第二关：Relevance (原有逻辑)
            relevance_reward = compute_relevance_reward(safe_recs, gt_catalog)

            # 安全惩罚
            safety_penalty_total = len(unsafe_recs) * safety_penalty

            total_reward = relevance_reward + safety_penalty_total
            rewards.append(total_reward)

        return rewards
    return reward_func
```

---

### 阶段 3：Adjust - CoT 修正数据生成

```
新建文件: scripts/generate_cot_sft_data.py
```

**CoT 数据生成逻辑**：
```python
# scripts/generate_cot_sft_data.py
from datasets import load_from_disk, Dataset
from libs.oracle import MovieOracle
from libs.constraint_injector import ConstraintInjector
from libs.safety_filter import SafetyFilter

def generate_cot_completion(
    original_recs: List[Tuple[str, int]],
    safe_recs: List[Tuple[str, int]],
    unsafe_recs: List[Tuple[str, int, str]],
    constraints: Dict
) -> str:
    """生成 CoT 格式的 completion"""

    cot_parts = []

    # 思考过程
    if unsafe_recs:
        cot_parts.append("思考过程：")
        cot_parts.append(f"用户要求: {format_constraints(constraints)}")
        for title, year, reason in unsafe_recs[:3]:  # 最多展示3个
            cot_parts.append(f"- 考虑推荐《{title}》({year})，但{reason}，违反用户约束，排除。")
        cot_parts.append("")

    # 最终推荐
    cot_parts.append("推荐列表：")
    for i, (title, year) in enumerate(safe_recs[:20], 1):
        cot_parts.append(f"{title} ({year})")

    return "\n".join(cot_parts)

def process_dataset(input_path: str, output_path: str, oracle_path: str):
    dataset = load_from_disk(input_path)
    oracle = MovieOracle(oracle_path)
    injector = ConstraintInjector(injection_rate=0.4)
    safety_filter = SafetyFilter(oracle)

    new_samples = []
    for sample in dataset:
        # 1. 注入约束
        modified_prompt, constraints = injector.inject(sample['prompt'][0]['content'])

        # 2. 获取原始推荐
        original_recs = parse_completion(sample['completion'][0]['content'])

        # 3. Safety Filter
        safe_recs, unsafe_recs = safety_filter.filter_recommendations(
            original_recs, constraints
        )

        # 4. 生成 CoT completion (如果有违规项)
        if unsafe_recs and len(safe_recs) >= 15:  # 确保有足够安全推荐
            new_completion = generate_cot_completion(
                original_recs, safe_recs, unsafe_recs, constraints
            )
            new_samples.append({
                'prompt': [{'role': 'user', 'content': modified_prompt}],
                'completion': [{'role': 'assistant', 'content': new_completion}],
                'seen_titles': sample['seen_titles'],
                'groundtruth_with_release_year': sample['groundtruth_with_release_year'],
                'has_cot': True,
                'constraints': constraints,
            })
        else:
            # 普通样本（无约束或全部安全）
            new_samples.append({
                **sample,
                'has_cot': False,
                'constraints': constraints,
            })

    new_dataset = Dataset.from_list(new_samples)
    new_dataset.save_to_disk(output_path)
```

---

### 阶段 4：System Prompt 增强

```
修改文件: train_sft.py 或新建 prompt_templates.py
```

**增强的 System Prompt**：
```python
SAFEREC_SYSTEM_PROMPT = """Role: You are a Safety-Aligned Movie Recommendation Assistant.

Guidelines:
1. You must strictly adhere to the user's safety constraints.
2. Before recommending, check each movie against the constraint requirements.
3. If a movie violates any constraint, explicitly state why and exclude it.
4. Prioritize safety over relevance - never recommend unsafe content.

Context Information (Movie Attributes):
{oracle_context}

Task: Based on the user's preferences and constraints, recommend 20 safe and relevant movies.
Format: "Movie Title (Year)" - one per line, with reasoning if constraints are active.
"""

def build_prompt_with_oracle(user_query: str, constraints: dict, oracle: MovieOracle) -> str:
    # 动态构建相关电影的属性信息
    oracle_context = format_oracle_context(oracle, user_query)

    system_prompt = SAFEREC_SYSTEM_PROMPT.format(oracle_context=oracle_context)

    return f"{system_prompt}\n\nUser Query: {user_query}"
```

---

## 四、文件结构变更

```
Rank-GRPO/
├── libs/
│   ├── oracle.py              # [新增] Oracle 数据库接口
│   ├── safety_filter.py       # [新增] Safety Filter
│   ├── constraint_injector.py # [新增] 约束注入器
│   ├── reward_funcs.py        # [修改] 增加安全感知奖励
│   └── data.py                # [修改] 增加 Oracle 加载
├── scripts/
│   ├── build_oracle.py        # [新增] 构建 Oracle 数据库
│   └── generate_cot_sft_data.py # [新增] 生成 CoT SFT 数据
├── data/
│   └── movie_oracle.pkl       # [新增] 电影属性数据库
├── train_sft.py               # [修改] 支持新数据格式
└── configs/
    └── saferec_sft.yaml       # [新增] SafeRec 配置
```

---

## 五、实施优先级

| 优先级 | 任务 | 工作量 | 依赖 |
|--------|------|--------|------|
| **P0** | 构建 Oracle 数据库 | 2-3天 | 外部 API (TMDB/IMDb) |
| **P1** | 约束注入模块 | 1天 | P0 |
| **P1** | Safety Filter | 1天 | P0 |
| **P2** | CoT 数据生成脚本 | 1-2天 | P1 |
| **P2** | 修改训练流程 | 0.5天 | P2 |
| **P3** | System Prompt 增强 | 0.5天 | P2 |

---

## 六、风险与建议

### 风险点

1. **Oracle 数据覆盖率**：`gt_catalog` 中的电影可能在外部 API 中找不到完整属性
   - **建议**：对于缺失属性的电影，设置默认值或标记为 "Unknown"

2. **约束复杂度**：过于复杂的约束组合可能导致可推荐电影过少
   - **建议**：设置最低安全推荐数量阈值 (如 ≥15)，否则放宽约束

3. **CoT 数据质量**：自动生成的 CoT 可能不够自然
   - **建议**：先小规模测试，必要时引入 Teacher LLM 润色

---

## 七、改动对比表

| 环节 | 原版 Rank-GRPO | **SafeRec 改进版** | 核心目的 |
|------|----------------|-------------------|----------|
| **User Prompt** | 仅包含推荐需求 | **注入合成的安全约束 (No Gore, For Kids, etc.)** | 制造多样化的安全场景 |
| **Reflect** | Teacher 打分相关性 | **Rule-based Safety Filter (查表) + Teacher Relevance** | 确保 SFT 数据绝对合规 (Ground Truth) |
| **Reflect 数据利用** | 过滤低分 Item | **利用被过滤的 Item 构造 "CoT 修正" 样本** | 教模型"为什么"这个不能推 |
| **Context** | 仅历史记录 | **历史记录 + Item 敏感属性标签 (Oracle Tags)** | 让模型基于证据 (Evidence) 做判断 |

---

## 八、预期效果

通过这一阶段的严密构造，SFT 模型将能学会：

1. **查阅属性** - 关注 Context 中的电影 Tags
2. **对比约束** - 将 Tags 与 User Constraints 进行逻辑比对
3. **剔除违规** - 显式排除不符合约束的电影
4. **最终排序** - 在安全前提下进行相关性排序

这为第二阶段的 RL (Rank-GRPO) 打下坚实的安全对齐基础。
