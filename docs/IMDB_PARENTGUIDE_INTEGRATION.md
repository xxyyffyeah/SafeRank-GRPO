# IMDb Parent Guide 集成方案：粗粒度强度权重

本文档描述如何将 IMDb Parent Guide 的五大类强度评级作为"粗粒度权重"，与 DoesTheDogDie (DDD) 细标签融合，生成更完整的电影敏感内容标注。

## 概述

**核心思路**：Parent Guide 类别少（仅5类）没关系，把它当作 trait 风险的**粗粒度权重**，与 DDD 的细粒度标签互补。

- **DDD**：提供精确的触发标签（如 gore、rape、drug abuse 等）
- **Parent Guide**：提供整体强度评级，弥补 DDD 覆盖不全的情况

---

## 输入数据准备

每部电影（建议用 `imdbId` 对齐）需要两类信息：

### 1. DDD 细标签
已在 trait 配置的 `avoid` 列表中定义（如 gore、rape、drug 等）

### 2. IMDb Parent Guide 五大类强度

| 类别 | 英文名 |
|------|--------|
| 性与裸露 | Sex & Nudity |
| 暴力与血腥 | Violence & Gore |
| 脏话 | Profanity |
| 烟酒毒品 | Alcohol, Drugs & Smoking |
| 惊悚紧张 | Frightening & Intense Scenes |

每个类别的取值：`None` / `Mild` / `Moderate` / `Severe`

---

## Step 1：强度数值化

将 Parent Guide 等级映射为数值：

| 等级 | 数值 |
|------|------|
| None | 0 |
| Mild | 1 |
| Moderate | 2 |
| Severe | 3 |

对每部电影得到一个5维向量：

$$\mathbf{s}(m) = [s_{\text{sex}}, s_{\text{viol}}, s_{\text{prof}}, s_{\text{sub}}, s_{\text{fright}}]$$

**示例**：某电影的 Parent Guide 为 `[None, Severe, Mild, None, Moderate]`，则 $\mathbf{s}(m) = [0, 3, 1, 0, 2]$

---

## Step 2：为每个 Trait 定义权重向量

给每个 trait 一个5维权重向量 $\mathbf{w}(t)$，不相关的维度设为 0。

### A. Parent Guide 能有效补充的 Trait（建议启用）

| Trait | Sex & Nudity | Violence & Gore | Profanity | Alcohol/Drugs | Frightening | 适用性 |
|-------|--------------|-----------------|-----------|---------------|-------------|--------|
| Anti-gore / squeamish | 0.0 | **1.0** | 0.0 | 0.0 | 0.3 | strong |
| Avoid torture & extreme violence | 0.0 | **1.0** | 0.0 | 0.0 | 0.5 | strong |
| Sexual violence sensitive | **1.0** | 0.3 | 0.0 | 0.0 | 0.2 | strong |
| Kid-safety / child harm sensitive | 0.1 | **0.8** | 0.0 | 0.0 | 0.4 | strong |
| Substance recovery / avoid drugs & alcohol | 0.0 | 0.0 | 0.0 | **1.0** | 0.1 | strong |
| Horror avoider | 0.0 | 0.3 | 0.0 | 0.0 | **1.0** | strong |
| Domestic abuse / stalking sensitive | 0.1 | **0.8** | 0.0 | 0.0 | 0.3 | weak |
| Disaster/accident avoider | 0.0 | **0.7** | 0.0 | 0.0 | 0.4 | weak |
| Photosensitivity & motion sickness | 0.0 | 0.0 | 0.0 | 0.0 | **1.0** | weak |
| Claustrophobia / breathing distress | 0.0 | 0.0 | 0.0 | 0.0 | **0.6** | weak |
| Self-harm & suicide sensitive | 0.0 | 0.4 | 0.0 | 0.0 | 0.4 | weak |
| Medical/health trauma avoider | 0.0 | 0.3 | 0.0 | 0.0 | 0.2 | weak |
| Hate speech / slur-sensitive | 0.0 | 0.0 | **0.6** | 0.0 | 0.0 | very_weak_proxy |

> **注意**：`Hate speech / slur-sensitive` 的 Profanity 强度只能提供"语言粗俗程度"，对"仇恨言论/歧视"并不可靠，仅作为可选的弱回退。

### B. Parent Guide 基本补不了的 Trait（权重全0，只靠DDD）

以下 trait 要么是很细的具体触发物，要么是"结局偏好"，Parent Guide 无法有效覆盖：

| Trait | 原因 |
|-------|------|
| Animal lover (avoid animal harm/death) | 具体动物类型无法从5类强度推断 |
| Arachnophobia / reptile phobia | 蜘蛛、蛇等具体恐惧对象 |
| Needle/medical procedure phobia | 针头等具体医疗器械 |
| Mental health portrayal sensitive | 心理健康表现方式的细节 |
| Gender/LGBTQ respect sensitive | 性别相关内容的处理方式 |
| Pregnancy/infant-loss sensitive | 怀孕/婴儿相关的具体情节 |
| Happy-ending preference | 结局偏好，与强度无关 |

---

## Step 3：计算 PG 风险分

对每个 trait $t$ 和电影 $m$，计算：

$$\text{pg\_risk}(m, t) = \begin{cases} \dfrac{\mathbf{w}(t) \cdot \mathbf{s}(m)}{3 \cdot \sum_i w_i(t)}, & \text{if } \sum_i w_i(t) > 0 \\ 0, & \text{otherwise} \end{cases}$$

**解释**：
- 分母的 `3` 是因为 severity 最大值为 3，将分数归一化到 `[0, 1]`
- 结果为连续分数，便于排序和阈值控制

**计算示例**：

假设电影 $m$ 的 Parent Guide 向量为 $\mathbf{s}(m) = [0, 3, 1, 0, 2]$

对于 "Anti-gore / squeamish" trait，权重为 $\mathbf{w} = [0, 1.0, 0, 0, 0.3]$

$$\text{pg\_risk} = \frac{0 \times 0 + 3 \times 1.0 + 1 \times 0 + 0 \times 0 + 2 \times 0.3}{3 \times (0 + 1.0 + 0 + 0 + 0.3)} = \frac{3.6}{3.9} \approx 0.92$$

---

## Step 4：DDD 硬触发与 PG 软分数融合

对每个电影-trait 组合，输出两个指标：

| 指标 | 说明 |
|------|------|
| `ddd_trigger(m, t)` | 电影是否命中该 trait 的任意 DDD avoid 标签（0 或 1） |
| `pg_risk(m, t)` | Step 3 计算的连续分数（0~1） |

**融合公式**：

$$\text{final}(m, t) = \max\big(\text{ddd\_trigger}(m, t), \text{pg\_risk}(m, t)\big)$$

> 也可以使用 $\alpha \cdot \text{ddd} + (1-\alpha) \cdot \text{pg}$，但 `max` 更符合"DDD 命中必然敏感"的直觉。

---

## Step 5：生成数据集标签

### 阈值设置

建议阈值 $\tau = 0.66$（相当于"加权后接近 Moderate"）

```
unsafe_for_trait = final(m, t) >= τ
```

### 输出格式

```json
{
  "imdbId": "0114709",
  "traits": {
    "Anti-gore / squeamish": {
      "final": 0.78,
      "unsafe": true,
      "source": "ParentGuide",
      "pg_risk": 0.78,
      "ddd_trigger": 0,
      "pg_dims": {
        "Violence & Gore": "Severe",
        "Frightening & Intense Scenes": "Moderate"
      }
    },
    "Sexual violence sensitive": {
      "final": 1.0,
      "unsafe": true,
      "source": "Both",
      "pg_risk": 0.45,
      "ddd_trigger": 1,
      "hit_tags": ["Is someone sexually assaulted"]
    }
  }
}
```

### 字段说明

| 字段 | 说明 |
|------|------|
| `final` | 融合后的最终分数 (0~1) |
| `unsafe` | 是否超过阈值 |
| `source` | 触发来源：`"DDD"` / `"ParentGuide"` / `"Both"` |
| `pg_risk` | Parent Guide 计算的风险分 |
| `ddd_trigger` | DDD 标签是否命中 (0/1) |
| `pg_dims` | Parent Guide 各维度的原始等级 |
| `hit_tags` | 命中的 DDD 标签列表 |

---

## 配置文件

完整的权重配置见 [`traits_with_imdb_parentguide_weights.json`](../traits_with_imdb_parentguide_weights.json)

### 配置结构

```json
{
  "meta": {
    "severity_to_int": { "None": 0, "Mild": 1, "Moderate": 2, "Severe": 3 },
    "risk_formula": { "name": "weighted_severity_normalized" }
  },
  "traits": [
    {
      "trait": "Anti-gore / squeamish",
      "avoid": ["..."],
      "imdb_parent_guide": {
        "pg_applicability": "strong",
        "weights": {
          "Sex & Nudity": 0.0,
          "Violence & Gore": 1.0,
          "Profanity": 0.0,
          "Alcohol, Drugs & Smoking": 0.0,
          "Frightening & Intense Scenes": 0.3
        }
      }
    }
  ]
}
```

### pg_applicability 取值

| 值 | 说明 | 建议用法 |
|----|------|---------|
| `strong` | Parent Guide 能有效补充 | 直接使用 pg_risk |
| `weak` | 弱相关，仅作回退 | 提高决策阈值或仅在 DDD 缺失时使用 |
| `very_weak_proxy` | 极弱代理 | 仅作可选回退 |
| `none` | 无法使用 | 完全依赖 DDD |

---

## 实现流程伪代码

```python
def compute_trait_labels(movie_id, ddd_tags, parent_guide, trait_config, threshold=0.66):
    """
    Args:
        movie_id: IMDb ID
        ddd_tags: set of DDD tags present in this movie
        parent_guide: dict {"Sex & Nudity": "Severe", ...}
        trait_config: loaded from traits_with_imdb_parentguide_weights.json
        threshold: decision threshold (default 0.66)

    Returns:
        dict of trait -> label info
    """
    severity_map = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}

    # Convert parent guide to numeric vector
    s = [severity_map.get(parent_guide.get(cat, "None"), 0)
         for cat in ["Sex & Nudity", "Violence & Gore", "Profanity",
                     "Alcohol, Drugs & Smoking", "Frightening & Intense Scenes"]]

    results = {}
    for trait_info in trait_config["traits"]:
        trait_name = trait_info["trait"]
        avoid_tags = set(trait_info["avoid"])
        weights = trait_info["imdb_parent_guide"]["weights"]

        # DDD trigger
        hit_tags = ddd_tags & avoid_tags
        ddd_trigger = 1 if hit_tags else 0

        # PG risk
        w = [weights["Sex & Nudity"], weights["Violence & Gore"],
             weights["Profanity"], weights["Alcohol, Drugs & Smoking"],
             weights["Frightening & Intense Scenes"]]

        w_sum = sum(w)
        if w_sum > 0:
            pg_risk = sum(wi * si for wi, si in zip(w, s)) / (3 * w_sum)
        else:
            pg_risk = 0.0

        # Fusion
        final = max(ddd_trigger, pg_risk)

        # Determine source
        if ddd_trigger and pg_risk >= threshold:
            source = "Both"
        elif ddd_trigger:
            source = "DDD"
        elif pg_risk >= threshold:
            source = "ParentGuide"
        else:
            source = None

        results[trait_name] = {
            "final": round(final, 3),
            "unsafe": final >= threshold,
            "source": source,
            "pg_risk": round(pg_risk, 3),
            "ddd_trigger": ddd_trigger,
            "hit_tags": list(hit_tags) if hit_tags else None
        }

    return results
```

---

## 适用性总结

| 类型 | Trait 数量 | 说明 |
|------|-----------|------|
| **强适用** | 6 | Anti-gore, Torture, Sexual violence, Kid-safety, Substance, Horror |
| **弱适用** | 6 | Domestic abuse, Disaster, Photosensitivity, Claustrophobia, Self-harm, Medical |
| **极弱代理** | 1 | Hate speech (仅 Profanity 维度) |
| **不适用** | 7 | Animal, Arachnophobia, Needle, Mental health, LGBTQ, Pregnancy, Happy-ending |

---

## 后续扩展

1. **方案3/4**：如果需要更精确的信号，可以使用 Parent Guide 的文本描述做关键词匹配或 LLM 细分类
2. **阈值调优**：可以对不同 `pg_applicability` 级别使用不同阈值
3. **A/B测试**：对比纯 DDD vs DDD+PG 融合的效果
