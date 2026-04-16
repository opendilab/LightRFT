# LightRFT 在线策略蒸馏 (OPD)

在线策略知识蒸馏使小型学生模型能够在强化学习训练过程中从大型教师模型学习。

## 概述

| 方面 | 描述 |
|------|------|
| **教师模型** | 提供 token 级别对数概率监督的大型模型 |
| **学生模型** | 被训练以匹配教师分布的小型模型 |
| **训练方式** | 在线策略：教师评估学生实际生成的响应 |
| **奖励信号** | 教师的对数概率作为学习信号 |

## 快速开始

### 1. 启动教师模型服务器

```bash
# 启动 SGLang 服务器运行教师模型
CUDA_VISIBLE_DEVICES=7 python3 -m sglang.launch_server \
    --model-path "Qwen/Qwen2.5-7B-Instruct" \
    --host 0.0.0.0 \
    --port 13141 \
    --tp 1 \
    --mem-fraction-static 0.6
```

### 2. 运行训练

```bash
bash examples/on_policy_distillation/run_opd_qwen_2.sh
```

或手动运行：

```bash
torchrun --nproc-per-node 2 examples/gsm8k_geo3k/train_colocate.py \
    --pretrain "Qwen/Qwen2.5-0.5B-Instruct" \
    --advantage_estimator "on_policy_distillation" \
    --teacher_model_url "http://127.0.0.1:13141/generate" \
    --no_task_reward \
    --reward_pretrain "" \
    --n_samples_per_prompt 4 \
    --actor_learning_rate 1e-6 \
    --init_kl_coef 0.01 \
    --num_episodes 30
```

### 分离部署

教师服务器和训练可以在不同终端或不同机器上运行：

```bash
# 终端 1：启动教师服务器
TEACHER_GPU=7 bash examples/on_policy_distillation/start_teacher.sh

# 终端 2：启动训练（教师就绪后）
TEACHER_URL=http://127.0.0.1:13141/generate bash examples/on_policy_distillation/start_training.sh
```

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    OPD 训练流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 生成      提示词 ──► [学生模型] ──► 响应                 │
│                                                             │
│  2. 评估      [提示词 + 响应] ──► [教师服务器]               │
│                            │                                │
│                            ▼                                │
│                    教师对数概率                              │
│                                                             │
│  3. 计算      优势值 = 教师_logp - 学生_logp                 │
│                                                             │
│  4. 更新      学生模型 ◄── 策略梯度损失                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. 优势值计算器

**文件**: `lightrft/trainer/advantage_calculator.py`

```python
class OnPolicyDistillationCalculator(AdvantageCalculator):
    def compute(self, experience, ...):
        teacher_log_probs = experience.info["teacher_log_probs"]
        student_log_probs = experience.action_log_probs

        # 反向 KL：鼓励学生匹配教师
        reverse_kl = student_log_probs - teacher_log_probs
        advantages = -reverse_kl  # = teacher - student

        return advantages, returns, {"opd_reverse_kl": reverse_kl}
```

### 2. 教师对数概率获取器

**文件**: `examples/on_policy_distillation/on_policy_distillation_reward.py`

- 异步 HTTP 请求到教师服务器
- 支持 SGLang 和 vLLM 响应格式
- 自动重试（指数退避）：当教师服务器出现瞬态故障时，重试延迟按指数增长（1s → 2s → 4s → ...），避免在服务器暂时过载期间发送大量重试请求

### 3. Experience Maker 集成

**文件**: `lightrft/trainer/fast_exp_maker.py`

- 当 `--advantage_estimator "on_policy_distillation"` 时，`--teacher_model_url` 指定教师 URL（也可通过 `--remote_rm_url` 传递，但已弃用）
- 教师对数概率存储在 `experience.info["teacher_log_probs"]`
- OPD 指标（`opd_reverse_kl_mean/std/min/max`）记录到 wandb

## 配置

### 必需参数

| 参数 | 值 | 描述 |
|------|---|------|
| `--advantage_estimator` | `"on_policy_distillation"` | 启用 OPD 模式 |
| `--teacher_model_url` | `"http://host:port/generate"` | 教师服务器 URL |
| `--reward_pretrain` | `""` | 空值（不需要奖励模型） |

### 推荐超参数

| 参数 | 值 | 描述 |
|------|---|------|
| `--n_samples_per_prompt` | 4 | 每个提示词的响应数 |
| `--actor_learning_rate` | 1e-6 | 学生学习率 |
| `--init_kl_coef` | 0.01 | KL 正则化系数 |
| `--num_episodes` | 30 | 训练轮数 |

## 教师服务器格式

### SGLang（推荐）

```json
{
    "meta_info": {
        "input_token_logprobs": [[logprob, rank, token], ...]
    }
}
```

### vLLM

```json
{
    "token_logprobs": [logprob1, logprob2, ...]
}
```

## 监控

### 记录的指标

| 指标 | 描述 |
|------|------|
| `opd_reverse_kl_mean` | 平均 KL(学生 \|\| 教师) |
| `opd_reverse_kl_std` | 反向 KL 的标准差 |
| `advantages_mean` | 平均优势值（应接近 0） |
| `policy_loss` | 训练中应下降 |

### 控制台输出

```
📊 详细步骤统计
============================================================
🎁 总奖励:         0.0000 ± 0.0000 (OPD 占位符)
📈 优势值:         0.0012 ± 0.8234 (...)
🎓 OPD 反向 KL:    0.1523 ± 0.0891 (...)
============================================================
```

## 故障排除

### 教师服务器问题

```bash
# 检查端口是否被占用
lsof -i :13141

# 检查 GPU 可用性
nvidia-smi

# 内存不足时减少内存占用
--mem-fraction-static 0.5
```

### 训练内存不足

```bash
--micro_train_batch_size 2
--micro_rollout_batch_size 2
--gradient_checkpointing
--zero_stage 3
```

### 收敛缓慢

```bash
--n_samples_per_prompt 8
--actor_learning_rate 5e-7
--num_episodes 50
```

## 与其他方法对比

| 方法 | 奖励信号 | 模式 | 需要 RM |
|------|---------|------|---------|
| GRPO | 任务特定奖励 | 在线 | 是 |
| DPO | 偏好对 | 离线 | 否 |
| **OPD** | 教师对数概率 | 在线 | 否（使用教师） |

### 优势

- 无需单独训练奖励模型：与 GRPO/PPO 等方法不同，OPD 不需要序列级的结果奖励模型（Outcome Reward Model）。教师模型本身充当 token 粒度的奖励信号——通过在每个 token 位置提供对数概率监督，教师直接指导学生在生成过程中的每一步决策，而非仅在完整序列结束后给出单一的好/坏评分。
- Token 级监督（比序列级更精细）
- 在线策略：适应学生不断变化的分布
- 适用于任何有好教师模型的任务

### 局限性

- 需要运行教师模型（推理开销）
- 学生无法超越教师的能力
- 需要足够的计算资源进行教师推理

## 文件结构

```
examples/on_policy_distillation/
├── README.md                           # 英文文档
├── README_zh.md                        # 本文件
├── run_opd_qwen.sh                   # 一体化训练脚本
├── start_teacher.sh                  # 仅启动教师服务器
├── start_training.sh                 # 仅启动训练（需要 TEACHER_URL）
├── test_opd.py                       # 单元测试
└── on_policy_distillation_reward.py   # 教师对数概率获取器
```

## 参考资料

- [LightRFT 文档](../../README.md)
- [优势值计算器源码](../../lightrft/trainer/advantage_calculator.py)
- [Fast Experience Maker 源码](../../lightrft/trainer/fast_exp_maker.py)
- [On-Policy Distillation Blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
