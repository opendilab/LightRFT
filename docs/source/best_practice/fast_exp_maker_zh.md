# FastExperienceMaker 最佳实践指南

## 目录
- [概述](#概述)
- [核心功能](#核心功能)
- [架构组件](#架构组件)
- [使用指南](#使用指南)
- [配置参数](#配置参数)
- [优势估计方法](#优势估计方法)
- [最佳实践](#最佳实践)
- [常见问题与解决方案](#常见问题与解决方案)
- [性能调优](#性能调优)

## 概述

### 什么是 FastExperienceMaker？

FastExperienceMaker 是 LightRFT 中用于 RLHF（从人类反馈中强化学习）训练的优化经验生成引擎。它扩展了基础的 `NaiveExperienceMaker`，支持高性能推理后端（VLLM/SGLang）和高级强化学习特性。

### 核心能力

- **高性能推理**：支持 VLLM 和 SGLang 后端，实现高效文本生成
- **多模态支持**：支持视觉-语言模型（VLM）的图像和视频数据处理
- **高级优势估计**：支持多种方法，包括 GAE、RLOO、REINFORCE 和 Group Normalization
- **灵活的奖励组合**：支持多个奖励模型和自定义聚合函数
- **样本打包**：通过序列打包提高训练效率
- **奖励归一化**：运行时奖励统计，支持归一化和裁剪

## 核心功能

### 1. 经验生成流程

FastExperienceMaker 实现了一个 7 阶段的经验生成流程：

```
阶段 1: 样本生成 (VLLM/SGLang)
    ↓
阶段 2: 分片并行预处理
    ↓
阶段 3: 模型推理 (Actor, Critic, Initial, Reward Models)
    ↓
阶段 4: 分片并行后处理
    ↓
阶段 5: 奖励处理 (归一化、塑形、过滤)
    ↓
阶段 6: 多图像/视频处理
    ↓
阶段 7: 优势计算
```

### 2. 多模态数据处理

`MultimodalDataProcessor` 类处理纯文本和图像-文本混合数据：

- **自动分离**：分离纯文本和多模态样本
- **适当处理**：通过分词器或多模态处理器路由样本
- **顺序保持**：处理后保持原始批次顺序
- **多图像/视频支持**：处理每个样本的多个图像或视频

### 3. 奖励计算引擎

`RewardComputationEngine` 管理奖励模型推理和聚合：

- **远程奖励模型**：支持 HTTP/gRPC 奖励模型
- **本地奖励模型**：基于 PyTorch 的奖励模型
- **自定义奖励函数**：用于自定义奖励逻辑的 Python 函数
- **多模型集成**：使用自定义聚合组合多个奖励模型
- **优化批处理**：高效的批处理，支持可选的样本过滤

## 架构组件

### 类层次结构

```
NaiveExperienceMaker (基类)
    ↓
FastExperienceMaker
    ├── MultimodalDataProcessor
    ├── RewardComputationEngine
    └── AdvantageCalculator (GAE/RLOO/REINFORCE/GroupNorm)
```

### 关键类

#### 1. FastExperienceMaker

**用途**：主要的经验生成类，具有优化推理和高级强化学习特性。

**初始化参数**：
- `packing_samples` (bool)：启用样本打包以提高效率
- `processor`：用于 VLM 模型的多模态处理器
- 其他参数继承自 `NaiveExperienceMaker`

**关键方法**：
- `make_experience_list()`：从提示生成经验
- `generate_samples()`：使用推理引擎生成样本
- `get_advantages_and_returns()`：计算优势和回报

#### 2. MultimodalDataProcessor

**用途**：处理纯文本和多模态混合数据的预处理。

**主要职责**：
- 归一化图像/视频输入（文件路径、PIL 图像、字节）
- 分离纯文本和多模态样本
- 通过适当的流程处理
- 按 `n_samples_per_prompt` 因子扩展样本

#### 3. RewardComputationEngine

**用途**：管理奖励模型推理和分数聚合。

**处理流程**：
1. **收集**：根据奖励配方收集或过滤样本
2. **处理**：通过奖励模型运行前向传播
3. **聚合**：使用 reward_fn 组合分数

## 使用指南

### 基础用法

#### 纯文本生成

```python
from lightrft.trainer.fast_exp_maker import FastExperienceMaker

# 初始化经验生成器
exp_maker = FastExperienceMaker(
    actor=actor_model,
    critic=critic_model,
    reward_model=reward_model,
    initial_model=initial_model,
    tokenizer=tokenizer,
    prompt_max_len=512,
    kl_controller=kl_controller,
    strategy=strategy,
    packing_samples=False,
)

# 生成经验
prompts = ["解释量子计算", "什么是机器学习？"]
experiences = exp_maker.make_experience_list(
    all_prompts=prompts,
    temperature=0.7,
    max_new_tokens=512,
    top_p=0.9,
)
```

#### 视觉-语言生成

```python
from PIL import Image

# 使用处理器初始化以支持 VLM
exp_maker = FastExperienceMaker(
    actor=actor_model,
    critic=critic_model,
    reward_model=reward_model,
    initial_model=initial_model,
    tokenizer=tokenizer,
    processor=multimodal_processor,  # VLM 必需
    prompt_max_len=512,
    kl_controller=kl_controller,
    strategy=strategy,
)

# 准备多模态数据
prompts = ["描述这张图片", "图片里有什么？"]
images = [
    [Image.open("image1.jpg")],  # 单张图片
    [Image.open("img2.jpg"), Image.open("img3.jpg")],  # 多张图片
]
references = ["沙发上的一只猫", "两只狗在玩耍"]

# 生成经验
experiences = exp_maker.make_experience_list(
    all_prompts=prompts,
    all_images=images,
    all_references=references,
    temperature=0.7,
    max_new_tokens=512,
)
```

### 高级用法

#### 多奖励模型与自定义聚合

```python
# 定义自定义奖励聚合函数
def custom_reward_fn(model_reward_list, labels, queries, refs, label_map):
    """
    自定义奖励聚合函数。

    参数：
        model_reward_list: 每个模型的奖励张量列表
        labels: 样本标签
        queries: 生成的文本
        refs: 参考文本
        label_map: 奖励模型名称到索引的映射

    返回：
        aggregated_rewards: 组合的奖励张量
        reward_metrics: 详细指标字典
    """
    # 示例：基于标签的加权平均
    weights = torch.tensor([0.6, 0.4])  # 两个模型的权重
    aggregated = sum(w * r for w, r in zip(weights, model_reward_list))

    metrics = {
        "reward_model_1": model_reward_list[0].mean(),
        "reward_model_2": model_reward_list[1].mean(),
    }

    return aggregated, metrics

# 使用多个奖励模型初始化
exp_maker = FastExperienceMaker(
    actor=actor_model,
    critic=critic_model,
    reward_model=[reward_model_1, reward_model_2],  # 模型列表
    reward_fn=custom_reward_fn,
    reward_fn_label_map={"rm1": 0, "rm2": 1},
    initial_model=initial_model,
    tokenizer=tokenizer,
    strategy=strategy,
)
```

#### 样本打包以提高效率

```python
# 启用样本打包
exp_maker = FastExperienceMaker(
    actor=actor_model,
    critic=critic_model,
    reward_model=reward_model,
    initial_model=initial_model,
    tokenizer=tokenizer,
    strategy=strategy,
    packing_samples=True,  # 启用打包
)

# 打包格式：| prompt1 response1 [EOS] | prompt2 response2 [EOS] | ...
# 优势：
# - 减少填充开销
# - 提高 GPU 利用率
# - 更快的训练吞吐量
```

#### 远程奖励模型

```python
# 通过 HTTP/gRPC 使用远程奖励模型
exp_maker = FastExperienceMaker(
    actor=actor_model,
    critic=critic_model,
    reward_model=None,  # 无本地奖励模型
    remote_rm_url=[
        "http://reward-server-1:8000/score",
        "http://reward-server-2:8000/score",
    ],
    initial_model=initial_model,
    tokenizer=tokenizer,
    strategy=strategy,
)
```

## 配置参数

### 生成参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `temperature` | float | 1.0 | 采样温度（越高越随机） |
| `top_p` | float | 1.0 | 核采样阈值 |
| `top_k` | int | -1 | Top-k 采样（-1 = 禁用） |
| `max_new_tokens` | int | 1024 | 生成的最大令牌数 |
| `min_new_tokens` | int | 1 | 生成的最小令牌数 |
| `skip_special_tokens` | bool | False | 输出中跳过特殊令牌 |

### 奖励处理参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `reward_running_norm` | bool | False | 启用运行时奖励归一化 |
| `reward_running_norm_minus_mean` | bool | False | 归一化时减去均值 |
| `reward_clip` | float | 0 | 奖励裁剪阈值（0 = 禁用） |
| `overlong_buffer` | bool | False | 启用过长序列惩罚 |
| `overlong_buffer_len` | int | 50 | 过长惩罚的缓冲区长度 |
| `overlong_buffer_penalty_factor` | float | 1.0 | 过长序列的惩罚因子 |

### 优势估计参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `advantage_estimator` | str | "gae" | 方法："gae"、"rloo"、"reinforce"、"group_norm" |
| `advantages_norm` | bool | False | 启用优势归一化（白化） |
| `advantage_clip` | float | 0 | 优势裁剪阈值（0 = 禁用） |
| `gamma` | float | 1.0 | 回报的折扣因子 |
| `lambd` | float | 0.95 | GAE lambda 参数 |

## 优势估计方法

### 1. GAE（广义优势估计）

**何时使用**：标准 PPO 训练，需要 critic 模型。

**优势**：
- 通过 lambda 参数平衡偏差-方差权衡
- 平滑的优势估计
- 与价值函数配合良好

**配置**：
```python
strategy.config.advantage_estimator = "gae"
strategy.config.gamma = 1.0
strategy.config.lambd = 0.95
```

### 2. RLOO（REINFORCE Leave-One-Out）

**何时使用**：无 critic 模型训练，每个提示生成多个样本。

**优势**：
- 不需要 critic 模型
- 通过基线减法减少方差
- 对每个提示的多个样本高效

**配置**：
```python
strategy.config.advantage_estimator = "rloo"
strategy.config.n_samples_per_prompt = 4  # 必须 > 1
```

### 3. REINFORCE with Baseline

**何时使用**：简单的策略梯度，使用奖励基线。

**优势**：
- 简单直接
- 适用于每个提示的单个样本
- 不需要 critic 模型

**配置**：
```python
strategy.config.advantage_estimator = "reinforce"
```

### 4. Group Normalization (GRPO)

**何时使用**：基于组的优势归一化，每个提示生成多个样本。

**优势**：
- 在每个提示组内归一化优势
- 减少不同提示之间的方差
- 对多样化的提示分布有效

**配置**：
```python
strategy.config.advantage_estimator = "group_norm"
strategy.config.n_samples_per_prompt = 4  # 必须 > 1
```

### 方法对比表

| 方法 | 需要 Critic | 每个提示的样本数 | 方差 | 偏差 | 复杂度 |
|------|------------|----------------|------|------|--------|
| GAE | 是 | 任意 | 低 | 低 | 中等 |
| RLOO | 否 | > 1 | 中等 | 低 | 低 |
| REINFORCE | 否 | 任意 | 高 | 低 | 低 |
| Group Norm | 否 | > 1 | 中等 | 中等 | 低 |

## 最佳实践

### 1. 选择推理后端

**VLLM**：
- ✅ 最适合：大规模部署、高吞吐量
- ✅ 支持：PagedAttention、连续批处理
- ⚠️ 注意：需要 CUDA 兼容的 GPU

**SGLang**：
- ✅ 最适合：研究、灵活性
- ✅ 支持：自定义采样、结构化生成
- ⚠️ 注意：可能有不同的性能特征

### 2. 多模态数据处理

**图像归一化**：
```python
# 支持的格式：
# 1. PIL Image 对象
images = [[Image.open("img.jpg")]]

# 2. 文件路径（将自动加载）
images = [["path/to/image.jpg"]]

# 3. 混合格式
images = [[Image.open("img1.jpg"), "path/to/img2.jpg"]]
```

**多图像场景**：
```python
# 每个样本多张图片
images = [
    [img1, img2, img3],  # 样本 1：3 张图片
    [img4],              # 样本 2：1 张图片
    None,                # 样本 3：无图片（纯文本）
]
```

### 3. 奖励模型配置

**单个奖励模型**：
```python
exp_maker = FastExperienceMaker(
    reward_model=single_rm,
    # ...
)
```

**多个奖励模型与聚合**：
```python
exp_maker = FastExperienceMaker(
    reward_model=[rm1, rm2, rm3],
    reward_fn=custom_aggregation_fn,
    reward_fn_label_map={"quality": 0, "safety": 1, "helpfulness": 2},
    # ...
)
```

**自定义奖励函数**：
```python
def custom_reward(queries, prompts, labels):
    """自定义奖励计算逻辑"""
    rewards = []
    for query, prompt, label in zip(queries, prompts, labels):
        # 你的自定义逻辑
        score = compute_custom_score(query, prompt, label)
        rewards.append(score)
    return rewards

exp_maker = FastExperienceMaker(
    custom_reward_func=custom_reward,
    # ...
)
```

### 4. 内存优化

**启用样本打包**：
```python
# 减少 30-50% 的填充开销
exp_maker = FastExperienceMaker(
    packing_samples=True,
    # ...
)
```

**调整微批次大小**：
```python
# 平衡内存使用和吞吐量
strategy.config.micro_rollout_batch_size = 8  # 根据 GPU 内存调整
```

**梯度检查点**：
```python
# 为大型模型启用
actor.gradient_checkpointing_enable()
```

### 5. 奖励归一化策略

**运行时归一化**（推荐用于稳定训练）：
```python
strategy.args.reward_running_norm = True
strategy.args.reward_running_norm_minus_mean = True  # 减去均值
```

**奖励裁剪**（防止异常值）：
```python
strategy.config.reward_clip = 10.0  # 裁剪到 [-10, 10]
```

**优势归一化**（稳定策略更新）：
```python
strategy.config.advantages_norm = True
strategy.config.advantage_clip = 5.0  # 可选裁剪
```

### 6. 处理过长序列

```python
# 惩罚过长的序列
strategy.config.overlong_buffer = True
strategy.config.overlong_buffer_len = 50  # 缓冲区长度
strategy.config.overlong_buffer_penalty_factor = 1.0  # 惩罚强度

# 示例：如果 max_new_tokens=512 且 buffer_len=50
# 预期长度 = 512 - 50 = 462
# 长度超过 462 个令牌的序列将受到惩罚
```

## 常见问题与解决方案

### 问题 1：内存不足（OOM）

**症状**：经验生成过程中出现 CUDA 内存不足错误。

**解决方案**：
1. 启用样本打包：`packing_samples=True`
2. 减小微批次大小：`strategy.config.micro_rollout_batch_size = 4`
3. 启用梯度检查点：`actor.gradient_checkpointing_enable()`
4. 减少最大序列长度：`max_new_tokens=256`
5. 使用更小的模型或量化

### 问题 2：生成速度慢

**症状**：经验生成耗时过长。

**解决方案**：
1. 使用 VLLM 后端：`strategy.args.engine_type = "vllm"`
2. 增加批次大小：`strategy.config.micro_rollout_batch_size = 16`
3. 启用样本打包：`packing_samples=True`
4. 检查 GPU 利用率：确保 GPU 充分利用
5. 如果可能，减少 `max_new_tokens`

### 问题 3：训练不稳定

**症状**：训练过程中奖励或损失剧烈波动。

**解决方案**：
1. 启用奖励归一化：
   ```python
   strategy.args.reward_running_norm = True
   strategy.args.reward_running_norm_minus_mean = True
   ```
2. 启用优势归一化：
   ```python
   strategy.config.advantages_norm = True
   ```
3. 添加奖励裁剪：
   ```python
   strategy.config.reward_clip = 10.0
   ```
4. 降低学习率
5. 使用 GAE 并设置适当的 lambda：`strategy.config.lambd = 0.95`

### 问题 4：图像令牌不匹配（VLM）

**症状**：推理过程中出现令牌/补丁不匹配的警告消息。

**原因**：图像令牌数量与像素值补丁不匹配。

**解决方案**：FastExperienceMaker 会自动修复此问题。警告仅供参考。如果频繁出现：
1. 检查图像预处理流程
2. 验证处理器配置
3. 确保样本之间的图像格式一致

### 问题 5：RLOO 需要多个样本

**症状**：使用 RLOO 时 `n_samples_per_prompt = 1` 导致错误。

**原因**：RLOO 需要每个提示生成多个样本来计算基线。

**解决方案**：
```python
# 设置 n_samples_per_prompt > 1
strategy.config.n_samples_per_prompt = 4
strategy.config.advantage_estimator = "rloo"
```

或切换到其他方法：
```python
# 改用 GAE 或 REINFORCE
strategy.config.advantage_estimator = "gae"
```

### 问题 6：远程奖励模型超时

**症状**：使用远程奖励模型时出现超时错误。

**解决方案**：
1. 检查到奖励模型服务器的网络连接
2. 在 remote_rm_fn 配置中增加超时时间
3. 减小批次大小以避免长时间处理
4. 考虑使用本地奖励模型以获得更好的性能
5. 在自定义奖励函数中实现重试逻辑

## 性能调优

### 吞吐量优化

**最大吞吐量的推荐配置**：
```python
strategy.args.engine_type = "vllm"
strategy.config.micro_rollout_batch_size = 16  # 根据 GPU 内存调整
exp_maker = FastExperienceMaker(
    packing_samples=True,
    # ...
)
```

**预期性能**：
- VLLM 后端：比 HuggingFace generate 快 2-5 倍
- 样本打包：减少 30-50% 的填充开销
- 批处理：批次大小线性扩展（直到 GPU 内存限制）

### 内存效率

**内存受限环境的推荐配置**：
```python
strategy.config.micro_rollout_batch_size = 4
strategy.config.max_new_tokens = 256
exp_maker = FastExperienceMaker(
    packing_samples=True,
    # ...
)
actor.gradient_checkpointing_enable()
```

## 参考资料

### 相关文档
- [策略设计哲学](strategy_design_philosophy.md)
- [模型设计文档](model.md)
- [奖励模型最佳实践](reward_model.md)

### 代码引用
- FastExperienceMaker：`lightrft/trainer/fast_exp_maker.py`
- 基础 ExperienceMaker：`lightrft/trainer/experience_maker.py`
- 优势计算器：`lightrft/trainer/advantage_calculator.py`
- VLLM 工具：`lightrft/strategy/vllm_utils/`

### 研究论文
- **GAE**："High-Dimensional Continuous Control Using Generalized Advantage Estimation"（Schulman 等，2016）
- **PPO**："Proximal Policy Optimization Algorithms"（Schulman 等，2017）
- **RLOO**："Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs"（Ahmadian 等，2024）

---

**文档版本**：1.0
**最后更新**：2026-02-03
**维护者**：LightRFT 团队
