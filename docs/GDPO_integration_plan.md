# Plan: 在 Rank-GRPO Safe 中集成 GDPO (Decoupled Normalization)

## 背景

`train_rank_grpo_safe.py` 有两个 reward 信号：relevance（稀疏）和 safety penalty。
当前两者在 reward 函数内部直接相加，然后作为单个 reward 进行 group-wise normalization。
由于 relevance reward 非常稀疏，合并后的 normalization 会导致 safety signal 被稀释（GDPO 论文称之为 "reward advantages collapse"）。

**GDPO 方案**：对每个 reward 信号独立做 group-wise normalization，再加权合并，最后做一次 batch normalization。

## 修改概览

### 需要修改的文件

1. **`libs/safe_reward_funcs.py`** — 拆分 reward 函数，使 relevance 和 safety 作为两个独立函数返回
2. **`libs/trl/rank_grpo_trainer.py`** — 添加 GDPO advantage 计算模式（在现有 GRPO normalization 旁边）
3. **`train_rank_grpo_safe.py`** — 添加 `--advantage_mode` 参数（`grpo` / `gdpo`），传两个 reward 函数给 trainer

### 不需要修改的文件
- `libs/reward_funcs.py`（非 safe 版本，不涉及多 reward）
- `train_rank_grpo.py`（非 safe 版本，保持不变）

---

## Step 1: 拆分 safe reward 函数 (`libs/safe_reward_funcs.py`)

目前 `safe_reward_func_log_decay` 和 `safe_reward_func_exp_inf` 内部将 `rewards_rel + safety_penalties` 合并返回。

**新增两对独立的 reward 函数**（保留原有合并版本以兼容 `--advantage_mode grpo`）：

### 1a. 新增 `relevance_only_reward_func_log_decay`
- 从现有 `safe_reward_func_log_decay` 中提取，只返回 `rewards_rel`（不加 safety_penalties）
- 签名与现有 reward 函数一致，接收同样的参数（constraints 参数忽略）

### 1b. 新增 `safety_only_reward_func_log_decay`
- 只返回 `safety_penalties`（不含 relevance）
- 对每个 rank 位置检查 safety violation，返回 penalty 值

### 1c. 同样为 exp_inf 版本新增对应的拆分函数
- `relevance_only_reward_func_exp_inf`
- `safety_only_reward_func_exp_inf`

### 1d. 新增 factory 函数
- `make_safe_relevance_func(rec_num, gt_catalog, ...)` — 只返回 relevance reward
- `make_safe_safety_func(rec_num, safety_oracle, lambda_safe, penalty_safe, ...)` — 只返回 safety penalty
- 同时提供 `_individual` (exp_inf) 版本

**注意**：safety reward 函数也需要接收 `groundtruth_with_release_year` 和 `seen_titles` 等参数（即使不使用），因为 trainer 统一传入所有 kwargs。可在函数内用 `**kwargs` 忽略不需要的参数。

---

## Step 2: 修改 trainer 添加 GDPO 模式 (`libs/trl/rank_grpo_trainer.py`)

在 `_generate_and_score_completions` 方法中（约第 1789-1809 行），当前的 advantage 计算逻辑：

```python
# 现有 GRPO: 先加权求和，再 group normalize
weights = self.reward_weights.to(device).view(1, -1, 1)
rewards_items = (rewards_per_func * weights).nansum(dim=1)  # (N_total, rec_num)
...
group_means = rewards_items.view(Bglob, G, rec_num).mean(dim=1)
group_stds  = rewards_items.view(Bglob, G, rec_num).std(dim=1)
advantages_items = (rewards_items - mean_rep) / (std_rep + 1e-4)
```

### 2a. 添加 `advantage_mode` 属性

在 `__init__` 中（约第 786 行后）新增：
```python
self.advantage_mode = getattr(args, "advantage_mode", "grpo")  # "grpo" or "gdpo"
```

### 2b. 新增 GDPO advantage 计算逻辑

在第 1789 行的 advantage 计算位置，添加分支：

```python
if self.advantage_mode == "gdpo" and rewards_per_func.size(1) > 1:
    # GDPO: 对每个 reward 函数独立 normalize，再加权合并
    G = self.num_generations
    Bglob = rewards_per_func.size(0) // G
    rec_num = self.rec_num if getattr(self, 'rec_num', None) is not None else rewards_per_func.size(2)
    num_funcs = rewards_per_func.size(1)

    all_advantages = []
    for i in range(num_funcs):
        reward_i = rewards_per_func[:, i, :]  # (N_total, rec_num)

        # 每个 reward 独立做 group-wise normalization
        group_mean_i = reward_i.view(Bglob, G, rec_num).mean(dim=1)  # (Bglob, rec_num)
        group_std_i  = reward_i.view(Bglob, G, rec_num).std(dim=1)
        mean_rep_i = group_mean_i.repeat_interleave(G, dim=0)
        std_rep_i  = group_std_i.repeat_interleave(G, dim=0)

        adv_i = reward_i - mean_rep_i
        adv_i = adv_i / (std_rep_i + 1e-4)
        all_advantages.append(adv_i)

    # 加权合并
    stacked = torch.stack(all_advantages, dim=1)  # (N_total, num_funcs, rec_num)
    weights = self.reward_weights.to(device).view(1, -1, 1)
    advantages_items = (stacked * weights).nansum(dim=1)  # (N_total, rec_num)

    # 最终 batch normalization
    bn_mean = advantages_items.mean()
    bn_std  = advantages_items.std()
    advantages_items = (advantages_items - bn_mean) / (bn_std + 1e-4)

else:
    # 原有 GRPO 逻辑（保持不变）
    weights = self.reward_weights.to(device).view(1, -1, 1)
    rewards_items = (rewards_per_func * weights).nansum(dim=1)
    ...（现有代码）
```

### 2c. 同步修改 installed trl package

由于 `from trl import RankGRPOTrainer` 实际导入的是 site-packages 中的版本，需要将修改同步到：
`/home/jovyan/conda_envs/safe/lib/python3.13/site-packages/trl/trainer/rank_grpo_trainer.py`

（或者改为从 `libs.trl` 导入）

---

## Step 3: 修改训练脚本 (`train_rank_grpo_safe.py`)

### 3a. 添加新参数

```python
parser.add_argument(
    "--advantage_mode",
    default="grpo",
    choices=["grpo", "gdpo"],
    help="Advantage computation mode: grpo (combined normalization) or gdpo (per-reward decoupled normalization).",
)
```

### 3b. 根据 advantage_mode 选择 reward 函数构建方式

```python
if args.advantage_mode == "gdpo":
    # GDPO: 两个独立 reward 函数
    if args.reward_func == "exp_inf":
        relevance_func = make_safe_relevance_func_individual(rec_num=20, gt_catalog=gt_catalog)
        safety_func = make_safe_safety_func_individual(
            rec_num=20, safety_oracle=safety_oracle,
            lambda_safe=args.lambda_safe, penalty_safe=args.penalty_safe
        )
    else:  # log_decay
        relevance_func = make_safe_relevance_func(rec_num=20, gt_catalog=gt_catalog)
        safety_func = make_safe_safety_func(
            rec_num=20, safety_oracle=safety_oracle,
            lambda_safe=args.lambda_safe, penalty_safe=args.penalty_safe
        )
    reward_funcs = [relevance_func, safety_func]
else:
    # GRPO: 原有合并 reward 函数（保持不变）
    if args.reward_func == "exp_inf":
        reward_funcs = make_safe_reward_func_individual(...)
    else:
        reward_funcs = make_safe_reward_func(...)
```

### 3c. 传递 advantage_mode 给 GRPOConfig

需要将 `advantage_mode` 传递给 trainer。由于 GRPOConfig 可能不支持该参数，有两种方式：
- 方式 A：通过 `setattr(config, "advantage_mode", args.advantage_mode)` 在 config 创建后添加
- 方式 B：在 trainer 初始化后直接设置 `trainer.advantage_mode = args.advantage_mode`

选择方式 A，在 config 创建后追加属性。

---

## Step 4: 创建新的启动脚本

新建 `run_safe_grpo_gdpo.sh`（可选），示例：

```bash
accelerate launch --num_processes 2 train_rank_grpo_safe.py \
    --advantage_mode gdpo \
    --reward_func exp_inf \
    --lambda_safe 1.0 \
    --penalty_safe 1.0 \
    ...
```

---

## 验证计划

1. **单元测试**：构造 mock rewards_per_func tensor (N_total, 2, rec_num)，验证 GDPO 分支输出 advantages 的 shape 正确且每个 reward 被独立 normalize
2. **运行训练**：分别用 `--advantage_mode grpo` 和 `--advantage_mode gdpo` 启动训练，观察 wandb 日志中 reward 曲线差异
3. **回归验证**：确认 `--advantage_mode grpo` 行为与修改前完全一致
