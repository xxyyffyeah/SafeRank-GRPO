# Safe-Rank-GRPO 实施文档

## 概述

Safe-Rank-GRPO 是 Rank-GRPO 的安全对齐扩展，通过在奖励函数中引入 **per-rank 安全惩罚**，让模型在保持推荐质量的同时学会尊重用户的敏感特征约束。

### 核心创新

Rank-GRPO 的关键特性是 **per-rank 独立优势计算**：每个推荐位置的 token 有独立的优势值。这意味着：

> 如果第 3 位推荐违反安全约束，只有生成第 3 位电影的 token 被惩罚，第 1、2 位的 token 不受影响。

这使得安全对齐更加精准，不会误伤正确的推荐。

---

## 算法

### 奖励函数

$$
r_{total}(x, y^{(k)}) = r_{rel}(x, y^{(k)}) + r_{safe}(x, y^{(k)})
$$

| 组件 | 公式 | 说明 |
|------|------|------|
| **r_rel** | $\frac{rel_k}{\log_2(k+1)}$ | 原始 DCG 贡献（与 GT 匹配时 $rel_k=1$） |
| **r_safe** | $-\lambda \cdot \mathbb{I}(violation) \cdot P$ | 安全惩罚（违反约束时触发） |

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lambda_safe` (λ) | 1.0 | 安全惩罚权重系数 |
| `penalty_safe` (P) | 1.0 | 单次违规惩罚幅度 |
| `risk_threshold` | 0.66 | SafetyOracle 风险阈值 |

### 示例

假设用户约束为 `{"Anti-gore": True}`，模型输出 20 部电影：

| Rank | Movie | GT Match | Violates Constraint | r_rel | r_safe | r_total |
|------|-------|----------|---------------------|-------|--------|---------|
| 1 | Toy Story | ✓ | ✗ | 1.0 | 0.0 | 1.0 |
| 2 | Finding Nemo | ✗ | ✗ | 0.0 | 0.0 | 0.0 |
| 3 | Saw | ✗ | ✓ | 0.0 | -1.0 | -1.0 |
| ... | ... | ... | ... | ... | ... | ... |

第 3 位的 Saw 触发 Anti-gore 约束，仅影响生成 "Saw" 的 token。

---

## 文件结构

### 新增文件

| 文件 | 说明 |
|------|------|
| `libs/safe_reward_funcs.py` | 安全奖励函数实现 |
| `train_rank_grpo_safe.py` | Safe-Rank-GRPO 训练入口 |

### 依赖文件

| 文件 | 说明 |
|------|------|
| `libs/safety_oracle.py` | 安全检查 Oracle |
| `libs/reward_funcs.py` | 原始 DCG 奖励函数 |
| `libs/trl/rank_grpo_trainer.py` | RankGRPOTrainer（复用） |

---

## 代码详解

### libs/safe_reward_funcs.py

```python
def safe_reward_func_log_decay(
    completions,                      # 模型生成的推荐列表
    groundtruth_with_release_year,    # GT 电影 [(title, year), ...]
    seen_titles,                      # 用户历史观影
    constraints,                      # 用户约束 {"Anti-gore": True, ...}
    rec_num,                          # 推荐数量 (20)
    gt_catalog,                       # GT 目录用于匹配
    safety_oracle,                    # SafetyOracle 实例
    lambda_safe,                      # λ 参数
    penalty_safe,                     # P 参数
    **kwargs
) -> List[List[float]]:
    """
    返回 shape: [batch_size, rec_num]
    每个元素是该 rank 的 r_total
    """
```

**工厂函数**：

```python
from libs.safe_reward_funcs import make_safe_reward_func

reward_func = make_safe_reward_func(
    rec_num=20,
    gt_catalog=gt_catalog,
    safety_oracle=safety_oracle,
    lambda_safe=1.0,
    penalty_safe=1.0,
)
```

### train_rank_grpo_safe.py

安全参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--lambda_safe` | float | 1.0 | 安全惩罚权重 |
| `--penalty_safe` | float | 1.0 | 单次违规惩罚 |
| `--risk_threshold` | float | 0.66 | 风险阈值 |

GDPO 参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--advantage_mode` | str | `grpo` | 优势计算模式：`grpo`（合并归一化）或 `gdpo`（解耦归一化） |

LoRA / PEFT 参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_lora` | flag | - | 启用 LoRA 训练 |
| `--lora_r` | int | 16 | LoRA rank（低秩矩阵维度） |
| `--lora_alpha` | int | 32 | LoRA 缩放因子（alpha / r = 有效缩放比） |
| `--lora_dropout` | float | 0.05 | LoRA 层 dropout 概率 |
| `--lora_target_modules` | list | `q_proj v_proj k_proj o_proj gate_proj up_proj down_proj` | 应用 LoRA 的模块 |

---

## 使用方法

### 基础训练

```bash
python train_rank_grpo_safe.py \
    --train_path ./data/saferec_dataset \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --sft_checkpoint 1500 \
    --lambda_safe 1.0 \
    --penalty_safe 1.0 \
    --use_vllm \
    --bf16
```

### 完整参数训练（Qwen2.5-0.5B）

```bash
python train_rank_grpo_safe.py \
    --train_path ./downloaded_datasets/processed_datasets/saferec_sft_dataset \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --sft_checkpoint 800 \
    --catalog_path gt_catalog_complete.pkl \
    --lr 1e-6 \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --kl_beta 1e-3 \
    --gradient_accumulation_steps 12 \
    --optim paged_adamw_8bit \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 2 \
    --mu 1 \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --num_generations 8 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 2 \
    --wandb_project safe_rank_grpo \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 200 \
    --seed 3407 \
    --bf16 \
    --lambda_safe 1.0 \
    --penalty_safe 1.0 \
    --risk_threshold 0.66
```

### 双卡训练

使用 `accelerate launch` 启动多 GPU 训练：

```bash
accelerate launch --num_processes 2 train_rank_grpo_safe.py \
    --train_path ./downloaded_datasets/processed_datasets/saferec_sft_dataset \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --sft_checkpoint 800 \
    --catalog_path gt_catalog_complete.pkl \
    --lr 1e-6 \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --kl_beta 1e-3 \
    --gradient_accumulation_steps 24 \
    --optim paged_adamw_8bit \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 2 \
    --mu 1 \
    --max_prompt_length 2048 \
    --max_completion_length 512 \
    --num_generations 4 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.35 \
    --vllm_tensor_parallel_size 2 \
    --wandb_project safe_rank_grpo \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 200 \
    --seed 3407 \
    --bf16 \
    --gradient_checkpointing \
    --lambda_safe 0.5 \
    --penalty_safe 0.5 \
    --risk_threshold 0.66
```

> **注意**：`--vllm_tensor_parallel_size` 必须等于 `--num_processes`，且能整除 world size。

### GDPO 模式训练（解耦归一化）

GDPO 将 relevance 和 safety 两个 reward 信号独立做 group-wise normalization，避免稀疏 relevance 信号被 safety 稀释：

```bash
accelerate launch --num_processes 2 train_rank_grpo_safe.py \
    --train_path ./downloaded_datasets/processed_datasets/saferec_sft_dataset \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --sft_checkpoint 800 \
    --catalog_path gt_catalog_complete.pkl \
    --advantage_mode gdpo \
    --reward_func exp_inf \
    --lr 1e-6 \
    --kl_beta 1e-3 \
    --gradient_accumulation_steps 24 \
    --per_device_train_batch_size 1 \
    --num_generations 4 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.35 \
    --vllm_tensor_parallel_size 2 \
    --save_steps 200 \
    --seed 3407 \
    --bf16 \
    --gradient_checkpointing \
    --lambda_safe 1.0 \
    --penalty_safe 1.0
```

### LoRA 训练（低显存 / 单卡友好）

LoRA 训练大幅降低显存占用（无需加载 reference model 副本），checkpoint 也更小，适合单卡运行：

```bash
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u train_rank_grpo_safe.py \
    --train_path ./downloaded_datasets/processed_datasets/saferec_sft_dataset \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --sft_checkpoint 800 \
    --catalog_path gt_catalog_complete.pkl \
    --use_lora \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --advantage_mode gdpo \
    --reward_func exp_inf \
    --lr 1e-6 \
    --kl_beta 1e-3 \
    --gradient_accumulation_steps 48 \
    --per_device_train_batch_size 2 \
    --num_generations 8 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_tensor_parallel_size 1 \
    --save_steps 500 \
    --seed 3407 \
    --bf16 \
    --gradient_checkpointing \
    --lambda_safe 1.0 \
    --penalty_safe 1.0
```

> **显存参考**：0.5B 模型 + LoRA r=64，单卡 46 GiB (L40) 实测占用约 23 GiB，`vllm_gpu_memory_utilization=0.4` 可正常运行。

> **LoRA rank 选择建议**：计算资源充足时推荐 `r=64, alpha=128`（alpha/r=2）。多信号学习（relevance + safety）和结构化输出（20-item 推荐列表）比一般文本任务需要更高 rank。

### 调整惩罚强度

```bash
# 轻微惩罚（保守）
python train_rank_grpo_safe.py \
    --lambda_safe 0.5 \
    --penalty_safe 0.5 \
    ...

# 强力惩罚（激进）
python train_rank_grpo_safe.py \
    --lambda_safe 2.0 \
    --penalty_safe 2.0 \
    ...
```

### 调整风险阈值

```bash
# 更严格（0.5 以上都视为违规）
python train_rank_grpo_safe.py \
    --risk_threshold 0.5 \
    ...

# 更宽松（0.8 以上才视为违规）
python train_rank_grpo_safe.py \
    --risk_threshold 0.8 \
    ...
```

---

## 数据集要求

SafeRec 数据集需要包含 `constraints` 列：

```python
{
    "prompt": [...],
    "completion": [...],
    "groundtruth_with_release_year": [...],
    "seen_titles": [...],
    "constraints": {"Anti-gore / squeamish": True, "Kid-safety": True}
}
```

### 约束格式

```python
{
    "Anti-gore / squeamish": True,           # 避免血腥
    "Horror avoider": True,                   # 避免恐怖
    "Kid-safety / child harm sensitive": True # 儿童安全
    # ... 共 20 个可能的 trait
}
```

参见 `traits_warnings.json` 获取完整 trait 列表。

---

## 与原始 Rank-GRPO 的对比

| 特性 | Rank-GRPO | Safe-Rank-GRPO |
|------|-----------|----------------|
| 奖励函数 | r_rel only | r_rel + r_safe |
| 安全约束 | ✗ | ✓ |
| GDPO 解耦归一化 | ✗ | ✓（`--advantage_mode gdpo`） |
| LoRA 训练 | ✗ | ✓（`--use_lora`） |
| 训练脚本 | train_rank_grpo.py | train_rank_grpo_safe.py |
| 数据集 | SFT dataset | SafeRec dataset |
| 新增参数 | - | lambda_safe, penalty_safe, advantage_mode, lora_* |

---

## 预期效果

### 训练目标

| 指标 | 目标 |
|------|------|
| Recall@20 | 维持或略降（~10%） |
| Safety Violation Rate | 大幅下降（目标 < 5%） |

### 超参数调优建议

| 场景 | 推荐设置 |
|------|----------|
| 平衡模式 | λ=1.0, P=1.0 |
| 安全优先 | λ=2.0, P=2.0 |
| 质量优先 | λ=0.5, P=0.5 |

---

## 技术细节

### Per-Rank 优势计算

RankGRPOTrainer 的核心逻辑（`libs/trl/rank_grpo_trainer.py`）：

**GRPO 模式**（`--advantage_mode grpo`，默认）：
```python
# 先加权求和，再 group normalize
rewards_items = (rewards_per_func * weights).nansum(dim=1)  # [batch, rec_num]
group_means = rewards_items.view(Bglob, G, rec_num).mean(dim=1)
group_stds  = rewards_items.view(Bglob, G, rec_num).std(dim=1)
advantages_items = (rewards_items - mean_rep) / (std_rep + 1e-4)
```

**GDPO 模式**（`--advantage_mode gdpo`）：
```python
# 每个 reward 函数独立 group normalize，再加权合并，最终 batch normalize
for i in range(num_funcs):
    reward_i = rewards_per_func[:, i, :]
    adv_i = (reward_i - group_mean_i) / (group_std_i + 1e-4)
    all_advantages.append(adv_i)

advantages_items = (stacked * weights).nansum(dim=1)
advantages_items = (advantages_items - bn_mean) / (bn_std + 1e-4)
```

GDPO 避免了稀疏 relevance 信号在合并后被 safety 信号稀释（"reward advantages collapse" 问题）。

### LoRA 训练机制

启用 `--use_lora` 时，`RankGRPOTrainer` 内部：
1. 使用 `get_peft_model()` 将 base model 包装为 LoRA 模型
2. **跳过 reference model 创建**，改用 `model.disable_adapter()` 获取 reference logits
3. Checkpoint 只保存 LoRA adapter 权重（远小于完整模型）

这使得显存占用大幅降低（无需维护 ref model 副本），适合资源受限场景。

### Per-Rank 安全惩罚

安全惩罚只影响特定位置的 token，不会传播到其他位置。

### SafetyOracle 调用

```python
from libs.safety_oracle import create_oracle

oracle = create_oracle(base_path=".", risk_threshold=0.66)

result = oracle.check_safety(
    title="Saw",
    year=2004,
    constraints={"Anti-gore / squeamish": True}
)

print(result.is_safe)      # False
print(result.violations)   # ["Anti-gore / squeamish: risk=0.92"]
```

---

## 常见问题

### Q: 为什么不直接在 SFT 阶段过滤？

A: SFT 过滤会改变推荐数量（20→12），导致模型学到错误的输出格式。Safe-Rank-GRPO 在 RL 阶段用惩罚信号引导，不改变输出格式。

### Q: lambda_safe 和 penalty_safe 如何选择？

A: 从默认值 (1.0, 1.0) 开始。如果 Safety Violation Rate 仍然过高，增加这两个值。如果 Recall 下降过多，减小这两个值。

### Q: 没有 constraints 的样本怎么处理？

A: 自动回退为空约束 `{}`，等价于原始 Rank-GRPO（无安全惩罚）。

---

## 更新日志

| 日期 | 版本 | 更新内容 |
|------|------|----------|
| 2026-01-31 | v1.2 | 新增 LoRA/PEFT 支持（`--use_lora` 及相关参数） |
| 2026-01-31 | v1.1 | 新增 GDPO 解耦归一化模式（`--advantage_mode gdpo`） |
| 2026-01-28 | v1.0 | 初始实现：safe_reward_funcs.py + train_rank_grpo_safe.py |
