# LightRFT

<div align="center">

<img src="assets/logo.png" alt="LightRFT Logo" width="600"/>

**轻量化、全模态和奖励模型驱动的强化学习微调框架**

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/opendilab/lightrft)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

[English](README.md) | 简体中文

</div>

---

## 📖 简介

**LightRFT** (Light Reinforcement Fine-Tuning) 是一个先进的多模态强化学习微调框架，专为大语言模型（LLM）和视觉语言模型（VLM）设计。该框架提供了高效、可扩展的 RLVR（Reinforcement Learning with Verifiable Rewards） 和 RLHF（Reinforcement Learning from Human Feedback）训练能力，支持多种前沿算法和分布式训练策略。

### ✨ 核心特性

- 🚀 **高性能推理引擎**
  - 集成 vLLM 和 SGLang 用于高效采样和推理
  - 支持 FP8 推理优化，显著降低延迟和显存占用
  - 灵活的引擎睡眠/唤醒机制优化资源利用

- 🧠 **丰富的算法生态**
  - **Policy Optimization**: GRPO, GSPO, GMPO, Dr.GRPO
  - **Advantage Estimation**: REINFORCE++, CPGD
  - **Reward Processing**: Reward Norm/Clip
  - **Sampling Strategy**: FIRE Sampling, Token-Level Policy
  - **Stability Enhancement**: DAPO, select_high_entropy_tokens

- 🔧 **灵活的训练策略**
  - 支持 FSDP (Fully Sharded Data Parallel) v2
  - 支持 DeepSpeed ZeRO (Stage 1/2/3)
  - 梯度检查点和混合精度训练（BF16/FP16）
  - Adam Offload 和内存优化技术

- 🎯 **创新的资源协同机制**
  - **Colocate Anything**: 奖励模型与训练模型协同定位，最大化 GPU 利用率
    - 支持多个奖励模型在同一设备上并行推理
    - 动态显存管理，训练/推理阶段自动切换
    - 减少跨设备通信开销，提升端到端训练效率
  - **Balance Anything** 🚧 (开发中): 智能负载均衡系统
    - 自适应任务调度和资源分配
    - 多节点训练负载自动均衡
    - 异构硬件环境性能优化

- 🌐 **全面的多模态支持**
  - **原生 Vision-Language Model (VLM) 训练**
    - 支持 Qwen-VL 等主流视觉语言模型
    - 图像-文本多模态数据并行处理
    - 高效的多模态 tokenization 和批处理
  - **多模态奖励建模**
    - 支持多个视觉奖励模型协同工作
    - 图像理解与文本生成的联合优化
  - **完整的视觉-语言对齐训练流程**
    - 专为多模态 RLVR/RLHF 优化
    - 内置视觉-语言模型微调支持

- 📊 **完整的实验工具链**
  - Weights & Biases (W&B) 集成
  - 数学能力基准测试（GSM8K, Geo3K 等）
  - 轨迹保存和分析工具
  - 自动检查点管理

---

## 🎯 支持的算法

详细算法说明、实现细节和使用指南请参考 [算法文档](docs/source/quick_start/algorithms_cn.md)。

| 算法 | 类型 | 主要改进 | 论文链接 |
|------|------|----------|---------|
| **GRPO** | Policy Optimization | 组归一化优势估计 |  [arXiv:2402.03300](https://arxiv.org/pdf/2402.03300)  |
| **GSPO** | Policy Optimization | 广义替代目标 | [arXiv:2507.18071](https://arxiv.org/abs/2507.18071) |
| **GMPO (WIP)** | Policy Optimization | 广义镜像策略优化 | [arXiv:2507.20673](https://arxiv.org/abs/2507.20673) |
| **Dr.GRPO** | Policy Optimization | 缓解长度偏差 | [arXiv:2503.20783](https://arxiv.org/abs/2503.20783) |
| **REINFORCE++** | Advantage Estimation | 改进基线估计 | [arXiv:2501.03262](https://arxiv.org/abs/2501.03262) |
| **DAPO** | Policy Optimization | 解耦剪裁和动态采样策略优化 | [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) |
| **CPGD** | Advantage Estimation | KL漂移约束 | [arXiv:2505.12504](https://arxiv.org/abs/2505.12504) |
| **FIRE Sampling** | Sampling Strategy | 过滤与排序策略 | [arXiv:2410.21236](https://arxiv.org/abs/2410.21236) |

---

## 🚀 快速开始

### 环境要求

- Python >= 3.10
- CUDA >= 12.8
- PyTorch >= 2.5.1

### Docker 镜像

TO BE DONE

### 安装步骤

克隆并安装 LightRFT:

```bash
# 克隆仓库
git clone https://github.com/opendilab/LightRFT.git
cd LightRFT

# 安装依赖
pip install -r requirements.txt

# 安装 LightRFT
pip install -e .
```


---

## 📚 使用指南

### 基础示例：GRPO 训练

```bash
# 单节点 8 GPU 训练示例
cd LightRFT

# 运行 GRPO 训练 (GSM8K 数学推理任务)
bash examples/gsm8k_geo3k/run_grpo_gsm8k_qwen2.5_0.5b.sh

# 或者运行 Geo3K 几何问题训练 (VLM 多模态)
bash examples/gsm8k_geo3k/run_grpo_geo3k_qwen2.5_vl_7b.sh
```

---

## 🏗️ 项目结构

```
LightRFT/
├── lightrft/                      # 核心库
│   ├── strategy/                  # 训练&推理策略
│   │   ├── fsdp/                  # FSDP 实现
│   │   ├── deepspeed/             # DeepSpeed 实现
│   │   ├── vllm_utils/            # vLLM 工具
│   │   └── sglang_utils/          # SGLang 工具
│   ├── models/                    # 模型定义
│   │   ├── actor_language.py      # 语言模型 Actor
│   │   ├── actor_vl.py            # 视觉语言模型 Actor
│   │   └── monkey_patch/          # 模型适配补丁
│   ├── trainer/                   # 训练器实现
│   │   ├── ppo_trainer.py         # PPO 训练器
│   │   ├── ppo_trainer_vl.py      # VLM PPO 训练器
│   │   ├── fast_exp_maker.py      # 经验生成器
│   │   ├── experience_maker.py    # 基础经验生成器
│   │   ├── experience_maker_vl.py # VLM 经验生成器
│   │   └── spmd_ppo_trainer.py    # SPMD PPO 训练器
│   ├── datasets/                  # 数据集处理
│   └── utils/                     # 工具函数
│       └── ckpt_scripts/          # 检查点处理脚本
│
├── examples/                      # 使用示例
│   ├── gsm8k_geo3k/               # GSM8K/Geo3K 数学推理训练示例
│   ├── grm_training/              # 生成式奖励模型训练示例
│   ├── srm_training/              # 标量奖励模型训练示例
│   ├── chat/                      # 模型对话示例
│
├── docs/                          # 📚 Sphinx 文档
│   └── source/
│       ├── installation/          # 安装指南
│       ├── quick_start/           # 快速开始 & 用户指南
│       │   ├── algorithms.md      # 算法文档（英文）
│       │   ├── algorithms_cn.md   # 算法文档（中文）
│       │   └── configuration.md   # 配置参数参考
│       └── best_practice/         # 最佳实践 & 资源
│           ├── strategy_usage.rst   # 训练策略使用指南（英文）
│           ├── strategy_usage_zh.md # 训练策略使用指南（中文）
│           ├── faq.md              # 常见问题
│           ├── troubleshooting.md  # 问题排查指南
│           └── contributing.md     # 贡献指南
│
├── assets/                        # 资源文件
│   └── logo.png                   # 项目Logo
│
├── results/                       # 训练结果
├── rft_logs/                      # 训练日志
└── README.md                      # 项目文档
```

### 🔑 关键目录说明

- **`lightrft/`**: LightRFT 核心库，提供训练策略、模型定义和训练器实现
- **`examples/`**: 完整的训练示例和脚本
  - `gsm8k_geo3k/`: GSM8K和Geo3K数学推理训练示例
  - `grm_training/`: 生成式奖励模型训练示例
  - `srm_training/`: 标量奖励模型训练示例
  - `chat/`: 模型对话示例
- **`docs/`**: Sphinx文档，包含完整的使用指南和API文档

---

## ⚙️ 关键配置参数

### 批次大小配置

```bash
TBS=128                           # 训练批次大小
RBS=128                           # Rollout 批次大小
micro_train_batch_size=1          # 每张卡的微批次大小
micro_rollout_batch_size=2        # Rollout 微批次大小
```

### 算法参数

```bash
--advantage_estimator group_norm  # 优势估计器：group_norm, reinforce, cpgd
--n_samples_per_prompt 8          # 每个提示采样数量
--max_epochs 1                    # 每个episode的训练轮数
--num_episodes 3                  # 总训练轮数
--kl_estimator k3                 # KL 估计器类型
--init_kl_coef 0.001              # KL 惩罚系数
```

### 分布式训练

```bash
--fsdp                            # 启用 FSDP
--zero_stage 3                    # DeepSpeed ZeRO Stage
--gradient_checkpointing          # 梯度检查点
--adam_offload                    # Adam 优化器卸载
--bf16                            # BF16 混合精度
```

### 推理引擎

```bash
--rm_use_engine                   # 使用推理引擎（vLLM/SGLang）
--engine_mem_util 0.4             # 引擎显存利用率
--engine_tp_size 1                # 引擎张量并行度
--enable_engine_sleep             # 启用引擎睡眠机制
```

---

## 🔧 常见问题排查


详细说明见训练脚本中的参数验证逻辑。

### 1. OOM (显存不足)

**解决方案**：
- 减小 `micro_train_batch_size` 和 `micro_rollout_batch_size`
- 启用 `--gradient_checkpointing`
- 降低 `--engine_mem_util`
- 使用 ZeRO Stage 3

### 2. 训练不稳定

**解决方案**：
- 启用 Reward Normalization: `--normalize_reward`
- 降低学习率
- 使用 `--advantage_estimator group_norm`
- 尝试 DAPO 算法


## 📖 文档

### 📚 完整文档指南

**快速开始：**
- [安装指南](docs/source/installation/index_cn.rst) - Docker 镜像、安装方法和问题排查
- [支持的算法](docs/source/quick_start/algorithms_cn.md) - 详细算法指南及实现细节
- [配置参数参考](docs/source/quick_start/configuration.md) - 完整参数文档

**最佳实践：**
- [训练策略使用](docs/source/best_practice/strategy_usage_zh.md) - FSDP、DeepSpeed 和推理引擎配置
- [常见问题](docs/source/best_practice/faq.md) - 常见问题与解决方案
- [问题排查指南](docs/source/best_practice/troubleshooting.md) - 常见问题和调试方法
- [贡献指南](docs/source/best_practice/contributing.md) - 如何为 LightRFT 做贡献

### 本地构建文档

安装文档依赖：
```bash
pip install -r requirements-doc.txt
```

生成 HTML 文档：
```bash
make docs
# 打开 docs/build/index.html 查看文档
```

实时预览文档：
```bash
make docs-live
# 访问 http://localhost:8000
```


## 🤝 贡献指南

我们欢迎社区贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码规范

```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 代码格式化（YAPF）
yapf -i -r lightrft/

# 代码检查（Pylint）
pylint lightrft/
```

---

## 📄 许可证

本项目采用 Apache 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

**LightRFT 是基于 [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) 开发的。** 我们向 OpenRLHF 团队的杰出工作表示衷心的感谢。本项目中的部分文件和实现是从 OpenRLHF 改编和复用的。

### 合作单位

本项目是与**上海人工智能实验室系统平台中心**和**安全可信AI中心**的同事合作开发，我们向其表示衷心的感谢。

### 开源依赖

本项目依托于以下优秀的开源项目（包括但不限于）:

- **[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)**、**[verl](https://github.com/volcengine/verl)** - 核心 RL 框架基础（部分关键组件改造和复用）
- [vLLM](https://github.com/vllm-project/vllm) - 高性能推理引擎
- [SGLang](https://github.com/sgl-project/sglang) - 结构化生成语言运行时
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - 分布式训练优化
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html) - 全分片数据并行

感谢所有贡献者和支持者！

---

## 🗓️ RoadMap

我们正在进行以下改进和功能开发：

### 核心功能增强

- [ ] **Trajectory 功能扩展**
  - 新增更多分析指标
  - 增强轨迹保存和分析能力

- [ ] **Reward 机制重构**
  - 重构 rule-based 和 model-based reward 计算
  - 优化 reward dataset 处理流程

### 算法优化与集成

- [ ] **更多算法支持**
  - Entropy-based token selection
  - GMPO (Generalized Mirror Policy Optimization)
  - GSPO (Generalized Surrogate Policy Optimization)

- [ ] **Advantage 计算重构**
  - 优化 advantage estimation 模块架构
  - 统一不同算法的 advantage 计算接口

- [ ] **Loss-Filter 机制优化**
  - 重构 loss filtering 实现
  - 完成 GSM8K/Geo3K 基准测试
  - 实验结果记录和分析


欢迎社区贡献和反馈！

---

## 📮 联系方式

如有问题或建议，请通过以下方式联系：

- **Issues**: [GitHub Issues](https://github.com/yourusername/lightrft/issues)
- **邮件**: opendilab@pjlab.org.cn

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给我们一个星标！**

Made with ❤️ by LightRFT Team

</div>
