# LightRFT

<div align="center">

<img src="assets/logo.png" alt="LightRFT Logo" width="600"/>

**轻量化、全模态和奖励模型驱动的强化学习微调框架**

[![Version](https://img.shields.io/badge/version-0.1.1-blue.svg)](https://github.com/opendilab/lightrft)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1+-ee4c2c.svg)](https://pytorch.org/)
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

详细算法说明、实现细节和使用指南请参考 [算法文档](docs/source/quick_start/algorithms_zh.md)。

| 算法 | 类型 | 主要改进 | 论文链接 |
|------|------|----------|---------|
| **GRPO** | Policy Optimization | 组归一化优势估计 |  [arXiv:2402.03300](https://arxiv.org/pdf/2402.03300)  |
| **GSPO** | Policy Optimization | 组序列策略优化 | [arXiv:2507.18071](https://arxiv.org/abs/2507.18071) |
| **GMPO (WIP)** | Policy Optimization | 几何平均策略优化 | [arXiv:2507.20673](https://arxiv.org/abs/2507.20673) |
| **Dr.GRPO** | Policy Optimization | 缓解长度偏差 | [arXiv:2503.20783](https://arxiv.org/abs/2503.20783) |
| **REINFORCE++** | Advantage Estimation | 改进基线估计 | [arXiv:2501.03262](https://arxiv.org/abs/2501.03262) |
| **DAPO** | Policy Optimization | 解耦剪裁和动态采样策略优化 | [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) |
| **CPGD** | Advantage Estimation | KL漂移约束 | [arXiv:2505.12504](https://arxiv.org/abs/2505.12504) |
| **FIRE Sampling** | Sampling Strategy | 高温度首token采样提升多样性 | [arXiv:2410.21236](https://arxiv.org/abs/2410.21236) |

---

## 🚀 快速开始

### 环境要求

- Python >= 3.12
- CUDA >= 12.8
- PyTorch >= 2.9.1

### Docker 镜像

我们提供预构建的 Docker 镜像，以便于快速部署并确保环境的一致性。您也可以使用项目中提供的 `Dockerfile` 和 `Makefile` 自行构建镜像。

#### 使用预构建镜像

官方 Docker 镜像托管在 [Docker Hub](https://hub.docker.com/r/opendilab/lightrft)。您可以使用以下命令获取最新版本：

```shell
docker pull opendilab/lightrft:v0.1.0
```

使用 GPU 支持运行容器：

```shell
docker run --gpus all -it --rm \
    -v /path/to/your/data:/app/data \
    -v /path/to/your/checkpoints:/app/checkpoints \
    opendilab/lightrft:v0.1.0 /bin/bash
```

#### 自行构建镜像

如果您需要自定义环境或基于特定分支进行构建，可以使用提供的 `Makefile` 在本地构建镜像。

1. **前提条件**：确保您的系统已安装 Docker 和 NVIDIA Container Toolkit。
2. **构建镜像**：
   ```shell
   # 使用默认名称构建镜像 (opendilab/lightrft:v${VERSION})
   make dbuild
   ```
   `IMAGE_NAME` 将根据项目的当前版本自动确定。您也可以手动指定标签：
   ```shell
   make dbuild IMAGE_NAME=your-custom-tag:latest
   ```

3. **技术细节**：
   - **基础镜像**：采用 `nvcr.io/nvidia/pytorch:25.01-py3`（包含 PyTorch 2.5+ 和 CUDA 12.8）。
   - **依赖安装**：构建过程会按照严格的顺序安装 `vLLM`、`DeepSpeed`、`Flash-Attention` 和 `SGLang` 等核心组件，以确保环境稳定性。
   - **优化策略**：`Dockerfile` 采用了多层构建优化，并配置了非交互式安装的环境变量。

### 安装步骤

#### 标准安装

LightRFT 默认使用 **SGLang** 作为推理后端，并包含 **Flash-Attention** 以优化性能。

```bash
# 克隆仓库
git clone https://github.com/opendilab/LightRFT.git
cd LightRFT

# 安装 LightRFT 及所有核心依赖
pip install -e .
```

**安装内容**: PyTorch、SGLang、Flash-Attention、Transformers、DeepSpeed 和其他核心依赖。

#### 可选：安装 vLLM 后端

如果您想使用 vLLM 替代（或配合）SGLang：

```bash
# 安装 vLLM 后端
pip install ".[vllm]"

# 或直接安装 vLLM
pip install vllm>=0.13.3
```

#### Flash-Attention 安装问题排查

Flash-Attention 默认包含在安装中，但在某些系统上可能因 CUDA 兼容性而安装失败。如果遇到问题，请尝试：

**方式 1: 使用预编译的 wheel 文件（推荐）**
```bash
# 从 https://github.com/Dao-AILab/flash-attention/releases 下载适合的 wheel 文件
# 例如 CUDA 12.x 和 PyTorch 2.9:
pip install flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

**方式 2: 使用 Docker（最简单）**
```bash
# 官方 Docker 镜像包含所有依赖
docker pull opendilab/lightrft:v0.1.0
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
├── lightrft/                         # 核心库
│   ├── strategy/                     # 训练&推理策略
│   │   ├── fsdp/                     # FSDP 实现
│   │   ├── deepspeed/                # DeepSpeed 实现
│   │   ├── vllm_utils/               # vLLM 工具
│   │   ├── sglang_utils/             # SGLang 工具
│   │   └── utils/                    # 策略工具函数
│   ├── models/                       # 模型定义
│   │   ├── actor_al.py               # 音频-语言模型 Actor
│   │   ├── actor_language.py         # 语言模型 Actor
│   │   ├── actor_vl.py               # 视觉-语言模型 Actor
│   │   ├── grm_vl.py                 # 生成式奖励模型（视觉-语言）
│   │   ├── srm_al.py                 # 标量奖励模型（音频-语言）
│   │   ├── srm_vl.py                 # 标量奖励模型（视觉-语言）
│   │   ├── loss.py                   # 损失函数
│   │   ├── monkey_patch/             # 分布式训练模型适配补丁
│   │   ├── tests/                    # 模型测试
│   │   └── utils.py                  # 模型工具函数
│   ├── trainer/                      # 训练器实现
│   │   ├── ppo_trainer.py            # LLM PPO 训练器
│   │   ├── ppo_trainer_vl.py         # VLM PPO 训练器
│   │   ├── spmd_ppo_trainer.py       # SPMD PPO 训练器扩展（**核心**）
│   │   ├── grm_trainer_vl.py         # 生成式奖励模型训练器（视觉-语言）
│   │   ├── srm_trainer_al.py         # 标量奖励模型训练器（音频-语言）
│   │   ├── srm_trainer_vl.py         # 标量奖励模型训练器（视觉-语言）
│   │   ├── fast_exp_maker.py         # 快速经验生成器（**核心**）
│   │   ├── experience_maker.py       # 基础经验生成器
│   │   ├── experience_maker_vl.py    # VLM 基础经验生成器
│   │   ├── replay_buffer.py          # 经验回放缓冲区
│   │   ├── replay_buffer_vl.py       # VLM 经验回放缓冲区
│   │   ├── replay_buffer_utils.py    # 经验回放缓冲区工具函数
│   │   ├── kl_controller.py          # KL 散度控制器
│   │   ├── image_utils.py            # 图像工具函数
│   │   ├── video_utils.py            # 视频工具函数
│   │   └── utils.py                  # 训练器工具函数
│   ├── datasets/                     # 数据集处理
│   │   ├── audio_alpaca.py           # Audio Alpaca 数据集的 Data Handler
│   │   ├── genai_bench.py            # GenAI Bench 数据集的 Data Handler
│   │   ├── grm_dataset.py            # 生成式奖励模型数据集
│   │   ├── hpdv3.py                  # HPDv3 奖励模型数据集的 Data Handler
│   │   ├── image_reward_db.py        # ImageRewardDB 数据集的 Data Handler
│   │   ├── imagegen_cot_reward.py    # ImageGen-CoT-Reward 数据集的 Data Handler
│   │   ├── omnirewardbench.py        # OmniRewardBench 数据集的 Data Handler
│   │   ├── process_reward_dataset.py # 奖励数据集处理
│   │   ├── prompts_dataset.py        # LLM 提示词数据集
│   │   ├── prompts_dataset_vl.py     # 视觉-语言提示词数据集
│   │   ├── rapidata.py               # Rapidata T2I/T2V 数据集的 Data Handler
│   │   ├── rft_dataset.py            # 强化微调 (RFT) 数据集
│   │   ├── sft_dataset.py            # SFT 数据集
│   │   ├── sft_dataset_vl.py         # VLM SFT 数据集
│   │   ├── srm_dataset.py            # 标量奖励模型基础数据集
│   │   ├── videodpo.py               # VideoDPO 数据集的 Data Handler
│   │   ├── videogen_rewardbench.py   # VideoGen-RewardBench 数据集的 Data Handler
│   │   └── utils.py                  # 数据集工具函数
│   └── utils/                        # 工具函数
│       ├── ckpt_scripts/             # 检查点处理脚本
│       ├── cli_args.py               # 命令行参数解析
│       ├── distributed_sampler.py    # 分布式采样器
│       ├── logging_utils.py          # 日志工具函数
│       ├── processor.py              # HuggingFace 模型数据处理器
│       ├── remote_rm_utils.py        # 远程奖励模型工具函数
│       ├── timer.py                  # 计时器工具函数
│       ├── trajectory_saver.py       # 轨迹保存器
│       └── utils.py                  # 通用工具函数
│
├── examples/                         # 使用示例
│   ├── gsm8k_geo3k/                  # GSM8K/Geo3K 数学推理训练示例
│   ├── grm_training/                 # 生成式奖励模型训练示例
│   ├── grm_vl_rl/                    # 强化微调生成式奖励模型训练示例
│   ├── srm_training/                 # 标量奖励模型训练示例
│   ├── chat/                         # 模型对话示例
│
├── docs/                             # 📚 Sphinx 文档
│   ├── Makefile                      # 文档构建 Makefile
│   ├── make.bat                      # 文档构建批处理文件
│   └── source/                       # 文档源码
│       ├── _static/                  # 静态文件（CSS 等）
│       ├── api_doc/                  # API 文档
│       ├── best_practice/            # 最佳实践 & 资源
│       ├── installation/             # 安装指南
│       └── quick_start/              # 快速开始 & 用户指南
│
├── assets/                           # 资源文件
│   └── logo.png                      # 项目 Logo
│
├── CHANGELOG.md                      # 更新日志
├── LICENSE                           # 许可证文件
├── Makefile                          # 项目 Makefile
├── README.md                         # 项目文档（英文）
├── README_zh.md                      # 项目文档（中文）
├── requirements.txt                  # Python 依赖
├── requirements-dev.txt              # 开发依赖
├── requirements-doc.txt              # 文档依赖
└── setup.py                          # 包安装脚本
```

### 🔑 关键目录说明

- **`lightrft/`**: LightRFT 核心库，提供训练策略、模型定义和训练器实现
- **`examples/`**: 完整的训练示例和脚本
  - `gsm8k_geo3k/`: GSM8K和Geo3K数学推理训练示例
  - `grm_training/`: 生成式奖励模型训练示例
  - `grm_vl_rl/`: 强化微调生成式奖励模型训练示例
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
- [安装指南](docs/source/installation/index_zh.rst) - Docker 镜像、安装方法和问题排查
- [支持的算法](docs/source/quick_start/algorithms_zh.md) - 详细算法指南及实现细节
- [配置参数参考](docs/source/quick_start/configuration_zh.md) - 完整参数文档

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


## 开发计划

- [v0.1.2](https://github.com/opendilab/LightRFT/issues/28)
- [v0.1.1](https://github.com/opendilab/LightRFT/issues/19)


## 🤝 贡献指南

非常欢迎并感谢您的贡献！为了确保协作顺畅，请遵循以下开发流程：

1.  **Fork 本仓库**：点击右上角的 "Fork" 按钮，将项目复刻到您的 GitHub 账户下。
2.  **创建特性分支**：建议基于 `main` 分支创建新分支。确保属于文档的分支以 *doc* 模式命名，以便自动部署文档站点。
    ```bash
    git checkout -b feature/your-feature-name
    ```
3.  **提交更改**：请遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范撰写提交信息。
    *   格式示例：`feature(user): 简短描述您的更改`
    *   常用类型：`feature` (新功能), `fix` (修复), `polish` (润色优化), `docs` (文档), `style` (格式), `refactor` (重构)。
    ```bash
    git commit -m 'feature(user): add an amazing feature'
    ```
4.  **推送到分支**：将更改推送到您的远程仓库。
    ```bash
    git push origin feature/your-feature-name
    ```
5.  **开启 Pull Request**：前往原仓库，创建一个指向 `main` (或指定开发分支) 的 Pull Request，并详细描述您的更改内容。

### 代码规范

```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 代码格式化（YAPF）
make format

# 代码检查（Flake8）
make fcheck
```

---

## 📚 引用

如果您在研究和应用中使用了本代码库，请按下列说明引用：

```bibtex
@misc{lightrft,
  title={LightRFT},
  author={Niu, Yazhe and Pu, Yuan and Shi, Dongxing and Lu, Yudong and Xiong, Yingtong and Ge, Ruijun and Sun, Jiaxuan and Wan, Zunian and Zhang, Shaoang and others},
  publisher={GitHub},
  howpublished={\url{https://github.com/opendilab/LightRFT}},
  year={2025},
}
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

## 📮 联系方式

如有问题或建议，请通过以下方式联系：

- **Issues**: [GitHub Issues](https://github.com/opendilab/LightRFT/issues)
- **邮件**: opendilab@pjlab.org.cn

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给我们一个星标！**

Made with ❤️ by LightRFT Team

</div>
