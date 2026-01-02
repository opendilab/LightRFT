## 算法速查表

LightRFT 支持丰富的强化学习算法生态系统，用于大语言模型的微调。本综合指南提供算法细节、实现参考。

### 本指南的目的

随着 RFT 领域的快速发展和新算法创新的涌现，本指南帮助你：

1. **快速识别** 哪些算法适合你的需求
2. **理解实现** 通过将算法映射到代码模块
3. **规划集成** 通过识别协同效应或冲突来集成多个算法
4. **保持清晰** 通过记录算法与组件之间的关系

### 算法概览与实现

| 算法 | 类型 | 模块 | 描述 | 实现位置 | 论文 |
|------|------|------|------|---------|------|
| **GRPO** | 策略优化 | 优势估计 | 使用基于组的归一化进行优势估计，无需独立的价值网络 | `FastExperienceMaker._get_return_advs()` | [arXiv:2402.03300](https://arxiv.org/pdf/2402.03300) |
| **GSPO** | 策略优化 | 策略损失 | 序列视角的策略优化方案 | `PolicyLoss.forward()` | [arXiv:2507.18071](https://arxiv.org/abs/2507.18071) |
| **REINFORCE++** | 优势估计 | 优势估计 | 通过改进的基线估计修改回报和优势计算 | `FastExperienceMaker._get_return_advs()` | [arXiv:2501.03262](https://arxiv.org/abs/2501.03262) |
| **CPGD** | 优势估计 | 优势估计 | 添加基于 KL 的漂移约束和裁剪对数比率以实现稳定的回报/优势计算 | `FastExperienceMaker._get_return_advs()` | [arXiv:2505.12504](https://arxiv.org/abs/2505.12504) |
| **FIRE Sampling** | 采样策略 | 经验生成 | 通过过滤和排序策略修改样本生成过程 | `FastExperienceMaker.generate_samples()` | [arXiv:2410.21236](https://arxiv.org/abs/2410.21236) |
| **GMPO** | 策略优化 | 策略损失 | 通过几何平均策略优化修改策略损失 | `PolicyLoss.forward()` | [arXiv:2507.20673](https://arxiv.org/abs/2507.20673) |
| **Dr.GRPO** | 策略优化 | 策略损失 | 引入无偏策略优化以缓解长度偏差并提高 token 效率 | `PolicyLoss.forward()` | [arXiv:2503.20783](https://arxiv.org/abs/2503.20783) |
| **DAPO** | 策略优化 | 策略损失 | 引入解耦裁剪和动态采样方案以稳定大规模 RL 优化 | `PolicyLoss.forward()` | [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) |
| **Token-Level Policy** | 策略优化 | 策略损失 | 在 token 粒度上优化策略以改善稳定性和信用分配 | `PolicyLoss.forward()` | [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) |
| **Reward Norm/Clip** | 奖励处理 | 奖励处理 | 应用奖励归一化和裁剪以稳定优势计算 | `FastExperienceMaker._get_return_advs()` | [GitHub](https://github.com/alibaba/ROLL) |
| **select_high_entropy_tokens** | 策略优化 | 策略损失 | 修改 PolicyLoss 以在训练期间实现高熵 token 选择 |  `PolicyLoss.forward()` | [arXiv:2506.01939](https://arxiv.org/abs/2506.01939) |

### 算法架构

#### 核心训练组件

LightRFT 的算法实现围绕三个主要模块组织：

##### 1. 策略损失计算 (`lightrft/trainer/ppo_loss.py`)
- **用途**：实现具有多种代理目标的 PPO 策略损失
- **核心方法**：`forward(log_probs, old_log_probs, advantages, action_mask)`
- **受影响算法**：GSPO、GMPO、Dr.GRPO、DAPO、Token-Level Policy、select_high_entropy_tokens
- **修改类型**：损失函数设计和 token 选择策略

##### 2. 数据生成 (`lightrft/trainer/fast_exp_maker.py`)
- **用途**：使用 vLLM 和其他推理后端生成经验
- **核心方法**：
  - `generate_samples()`：使用各种策略生成样本
  - `_get_return_advs()`：回报和优势计算
- **受影响算法**：FIRE Sampling
- **修改类型**：采样策略和推理优化

##### 3. 优势与奖励处理 (`lightrft/trainer/fast_exp_maker.py`)
- **用途**：处理奖励并计算策略更新的优势
- **核心方法**：`_get_return_advs()`：使用各种基线的优势估计
- **受影响算法**：GRPO、REINFORCE++、CPGD、Reward Norm/Clip
- **修改类型**：优势估计方法和奖励塑形

#### 修改类型

**算法改变**：
- **损失设计**：核心目标函数修改
- **优势估计**：优势计算方法更新
- **采样策略**：样本生成过程改变
- **Token 选择**：训练中使用哪些 token
- **奖励塑形**：奖励预处理和过滤

**实现改变**：
- **效率优化**：性能改进（例如 FP8）
- **参数调优**：超参数调整
- **流程集成**：新组件或工作流改变

### 策略优化算法

#### GRPO (Group Relative Policy Optimization)

**概述**：GRPO 使用基于组的归一化进行优势估计，无需单独的价值网络即可提供稳定的训练。

**实现位置**：`FastExperienceMaker._get_return_advs()` - 优势估计模块
**修改类型**：优势估计

**核心特性**：
- 不需要 critic 网络
- 组归一化优势
- 大批量稳定训练
- 内存高效

**使用方法**：
```bash
python train.py \
    --advantage_estimator group_norm \
    --n_samples_per_prompt 8 \
    --kl_estimator k3
```

**最适合**：
- 内存有限的大规模训练
- 无价值网络的快速原型设计
- 数学推理和代码任务

---

#### GSPO (Group Sequence Policy Optimization)

**概述**：GSPO 通过灵活的代理函数推广 PPO 目标，允许更好地控制策略更新。

**实现位置**：`PolicyLoss.forward()` - 策略损失模块
**修改类型**：损失设计

**核心特性**：
- 广义裁剪目标
- 自适应信赖域更新
- 更好的样本效率

**使用方法**：
```bash
python train.py \
    --advantage_estimator gspo \
    --gspo_alpha 0.1 \
    --clip_range 0.2
```

**最适合**：
- 需要精确策略控制的任务
- 多任务学习场景

---

#### GMPO (Geometric-Mean Policy Optimization)

**概述**：GMPO 利用镜像下降原理进行策略优化，提供理论保证和改进收敛性。

**实现位置**：`PolicyLoss.forward()` - 策略损失模块
**修改类型**：损失设计

**核心特性**：
- 镜像下降更新
- 理论收敛保证
- 自适应步长

**使用方法**：
```bash
python train.py \
    --advantage_estimator gmpo \
    --mirror_tau 0.01
```

**最适合**：
- 需要理论保证的研究应用
- 复杂的奖励地形

---

#### Dr.GRPO (Group Relative Policy Optimization Done Right)

**概述**：Dr.GRPO 通过显式建模和缓解奖励-长度相关性来解决奖励模型中的长度偏差。

**实现位置**：`PolicyLoss.forward()` - 策略损失模块
**修改类型**：损失设计（长度偏差缓解）

**核心特性**：
- 长度偏差缓解
- 奖励去偏机制
- 改善响应质量

**使用方法**：
```bash
python train.py \
    --advantage_estimator group_norm \
    --use_length_penalty \
    --length_penalty_coef 0.01
```

**最适合**：
- 对响应长度敏感的任务
- 指令遵循
- 开放式生成

---

#### DAPO (Dynamic sAmpling Policy Optimization)

**概述**：DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization) 对优势加权的策略更新使用单独的上下裁剪边界，并结合动态采样策略，提高训练稳定性。

**实现位置**：`PolicyLoss.forward()` - 策略损失模块
**修改类型**：损失设计（解耦裁剪）

**核心特性**：
- 正/负优势的解耦裁剪
- 动态采样策略
- 更好地处理分布偏移
- 改善稳定性

**使用方法**：
```bash
python train.py \
    --use_clip_higher \
    --clip_range_higher 0.3 \
    --clip_range_lower 0.2
```

**最适合**：
- 高噪声奖励信号
- 大分布偏移
- 挑战性领域

---

#### Token-Level Policy

**概述**：在 token 粒度上优化策略，提高稳定性和信用分配。

**实现位置**：`PolicyLoss.forward()` - 策略损失模块
**修改类型**：Token 选择

**核心特性**：
- Token 粒度优化
- 改善信用分配
- 长序列中更好的稳定性

**使用方法**：通常通过实现修改与其他策略优化方法结合使用。

### 优势估计方法

#### REINFORCE++

**概述**：改进的基线估计方法，使用控制变量来降低策略梯度估计的方差。

**实现位置**：`FastExperienceMaker._get_return_advs()` - 优势估计模块
**修改类型**：优势估计

**核心特性**：
- 更低方差梯度
- 更快收敛
- 与所有策略优化方法兼容

**使用方法**：
```bash
python train.py \
    --advantage_estimator reinforce_plus \
    --baseline_type value_network
```

**最适合**：
- 高方差环境
- 稀疏奖励
- 与 PPO 或其他在策略方法结合

---

#### CPGD (Clipped Policy Gradient Optimization with Policy Drift)

**概述**：CPGD 使用 KL 散度约束策略更新，防止灾难性遗忘并保持稳定训练。

**实现位置**：`FastExperienceMaker._get_return_advs()` - 优势估计模块
**修改类型**：优势估计（KL 约束）

**核心特性**：
- KL 约束更新
- 防止灾难性遗忘
- 自适应约束调整

**使用方法**：
```bash
python train.py \
    --advantage_estimator cpgd \
    --kl_target 0.01 \
    --kl_horizon 10000
```

**最适合**：
- 微调预训练模型
- 保留原始能力
- 多阶段训练

### 奖励处理

#### 奖励归一化和裁剪

**概述**：标准奖励预处理技术以稳定训练。

**实现位置**：`FastExperienceMaker._get_return_advs()` - 奖励处理模块
**修改类型**：奖励塑形（归一化/裁剪）

**核心特性**：
- 运行奖励统计
- 优势归一化
- 奖励裁剪

**使用方法**：
```bash
python train.py \
    --reward_running_norm \
    --reward_running_norm_minus_mean \
    --reward_clip 10.0 \
    --advantage_clip 10.0
```

**最适合**：
- 所有训练场景（推荐基线）
- 提示间奖励尺度变化
- 训练稳定性

### 采样策略

#### FIRE Sampling

**概述**：FIRE（过滤和改进奖励估计）结合过滤和排序策略以实现更好的样本选择。

**实现位置**：`FastExperienceMaker.generate_samples()` - 经验生成模块
**修改类型**：采样策略

**核心特性**：
- 多阶段过滤
- 基于奖励的排序
- 样本效率

**使用方法**：
```bash
python train.py \
    --use_fire_sampling \
    --fire_filter_ratio 0.5 \
    --fire_rank_method reward
```

**最适合**：
- 有限计算预算
- 高质量数据生成
- Best-of-N 采样场景



### 实现注意事项

- 所有策略损失算法修改 **PolicyLoss** 模块的 `forward()` 方法
- 优势估计算法修改 **FastExperienceMaker** 的 `_get_return_advs()` 方法
- 采样策略修改 **FastExperienceMaker** 的 `generate_samples()` 方法
- 奖励处理算法主要在 `_get_return_advs()` 方法内工作
- 大多数修改在核心训练循环组件而非外围工具中

### 参考资料

详细的算法描述和实验结果请参阅链接的论文。实现细节可在源代码中找到：
- 策略损失：`lightrft/models/loss.py`
- 经验生成器：`lightrft/trainer/fast_exp_maker.py`
- vLLM 工具：`lightrft/strategy/vllm_utils/`
