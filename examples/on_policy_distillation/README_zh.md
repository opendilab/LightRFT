# 在线策略蒸馏 (On-Policy Distillation, OPD)

在线策略蒸馏（[Blog](https://thinkingmachines.ai/blog/on-policy-distillation/)）通过 token 级对数概率监督，让小型学生模型在 RL 训练过程中从大型教师模型学习。与 GRPO/PPO 不同，OPD 不需要单独的奖励模型——教师本身在每个 token 位置提供监督信号。

```
训练流程:  Prompt ──► [学生] ──► Response ──► [教师服务器] ──► Teacher LogProbs
          Advantage = Teacher_logp - Student_logp ──► 策略梯度更新学生
```

## 快速开始

### 方式一：一体化执行（推荐）

`run_opd_qwen.sh` 内部已包含教师服务器的启动、健康检查和自动清理，无需手动管理教师进程：

```bash
# 直接运行（需先编辑脚本中的模型和数据集路径）
bash examples/on_policy_distillation/run_opd_qwen.sh

# 或通过环境变量覆盖
TEACHER_MODEL_PATH=/path/to/teacher \
STUDENT_MODEL_PATH=/path/to/student \
DATASET_PATH=/path/to/data.jsonl \
bash examples/on_policy_distillation/run_opd_qwen.sh

# 开启任务奖励混合模式（GRPO 任务奖励 + OPD KL 信号）
USE_TASK_REWARD=true bash examples/on_policy_distillation/run_opd_qwen.sh
```

### 方式二：分离部署

适用于教师和训练在不同终端或不同机器上运行的场景：

```bash
# 终端 1：启动教师服务器
bash examples/on_policy_distillation/start_teacher.sh

# 终端 2：启动训练（教师就绪后）
TEACHER_URL=http://127.0.0.1:13141/generate \
bash examples/on_policy_distillation/start_training.sh
```

## 环境变量参考

### 模型与数据

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TEACHER_MODEL_PATH` | `Qwen/Qwen2.5-7B-Instruct` | 教师模型路径 |
| `STUDENT_MODEL_PATH` | `Qwen/Qwen2.5-0.5B-Instruct` | 学生模型路径 |
| `DATASET_PATH` | — | 训练数据集路径（jsonl 格式） |

### 训练超参数

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `USE_TASK_REWARD` | `false` | `true` 开启 GRPO 任务奖励 + OPD KL 混合模式 |
| `OPD_KL_COEF` | `1.0` | OPD KL 损失系数 |
| `N_SAMPLES` | `8` | 每个提示词的采样数 |
| `LR` | `5e-7` | 学生学习率 |
| `EPISODE` | `30` | 训练轮数 |

### 教师服务器（分离部署时）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TEACHER_GPU` | `0` | 教师模型使用的 GPU |
| `TEACHER_PORT` | `13141` | 教师服务器端口 |
| `TEACHER_URL` | — | 教师服务器地址（`start_training.sh` 必需） |

## 监控指标

训练过程中以下指标会记录到 W&B：

| 指标 | 说明 |
|------|------|
| `opd_reverse_kl_mean` | 平均 KL(学生 \|\| 教师)，应随训练下降 |
| `opd_reverse_kl_std` | 反向 KL 标准差 |
| `advantages_mean` | 平均优势值（应接近 0） |
| `policy_loss` | 策略损失，应随训练下降 |

## 故障排除

| 问题 | 解决方案 |
|------|----------|
| 教师服务器启动失败 | `lsof -i :13141` 检查端口占用；`nvidia-smi` 检查 GPU 可用性 |
| 教师 OOM | 降低 `MEM_FRACTION`（默认 0.7） |
| 训练 OOM | 减小 `MICRO_TRAIN_BS`/`MICRO_ROLLOUT_BS`，或设置 `--zero_stage 3` |
| 收敛缓慢 | 增大 `N_SAMPLES`（如 16），降低 `LR`（如 1e-7），增大 `EPISODE` |

## 与其他方法对比

| 方法 | 奖励信号 | 监督粒度 | 需要 RM |
|------|---------|----------|---------|
| GRPO | 任务特定奖励 | 序列级 | 是 |
| DPO | 偏好对 | 序列级 | 否 |
| **OPD** | 教师对数概率 | Token 级 | 否（使用教师） |

## 文件结构

```
examples/on_policy_distillation/
├── run_opd_qwen.sh       # 一体化脚本：教师启动 + 训练（推荐）
├── start_teacher.sh      # 仅启动教师服务器
├── start_training.sh     # 仅启动训练（需要 TEACHER_URL）
├── test_opd.py           # 单元测试
├── README.md             # English
└── README_zh.md          # 本文件
```
