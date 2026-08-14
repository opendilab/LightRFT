# LightRFT

<div align="center">

<img src="assets/logo.png" alt="LightRFT Logo" width="600"/>

**轻量、高效、全模态、奖励模型驱动的强化学习微调框架**

[![Version](https://img.shields.io/badge/version-0.1.1-blue.svg)](https://github.com/opendilab/LightRFT)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

[English](README.md) | 简体中文

</div>

## 项目概述

LightRFT（Light Reinforcement Fine-Tuning）是面向大语言模型（LLM）和多模态模型（MLLM）的强化学习微调框架，旨在为可验证奖励强化学习（RLVR）、基于人类反馈的强化学习（RLHF）和基于模型奖励的策略优化提供高效、可扩展的训练流程。

项目以 `torchrun` 等 PyTorch 分布式通信为运行基础，通过统一的 Strategy 接口衔接 FSDP v2、DeepSpeed ZeRO 训练后端与 SGLang、vLLM 采样后端。当前代码路径和示例覆盖文本、图像、视频与音频任务。

## 目录

- [设计特点](#设计特点)
- [训练方法与优化机制](#训练方法与优化机制)
- [系统架构](#系统架构)
- [安装](#安装)
- [快速上手](#快速上手)
- [配置说明](#配置说明)
- [示例与应用](#示例与应用)
- [监控、轨迹与检查点](#监控轨迹与检查点)
- [项目结构](#项目结构)
- [文档与问题排查](#文档与问题排查)
- [开发计划](#开发计划)
- [贡献](#贡献)
- [引用、许可证与致谢](#引用许可证与致谢)

## 设计特点

### 统一的 Strategy 抽象与训推闭环

- 通过统一接口衔接 DeepSpeed/FSDP v2 训练后端与 SGLang/vLLM 采样后端，降低上层训练流程对具体后端的耦合。
- 基于 `torchrun` 和单程序多数据（SPMD）方式组织分布式任务，无需额外的 Ray 调度层，并可沿用原生 PyTorch 分布式工具进行调试。
- 训练与采样按阶段复用同一组 GPU；推理引擎支持休眠/唤醒，Actor 更新后再将新权重同步至采样侧。

项目将这种逻辑共置与分阶段资源复用方式称为 **Colocate Anything**。Strategy 的职责边界、验证流程、模型放置和权重同步机制见[运行时架构与资源复用原理](docs/source/best_practice/runtime_architecture_zh.md)。

### 分布式训练与参数高效微调

- FSDP v2 与 DeepSpeed ZeRO Stage 1/2/3；未指定 `--fsdp` 时使用 DeepSpeed 策略。
- BF16、梯度检查点、Adam 优化器卸载和 FSDP CPU offload。
- LoRA、视觉编码器前缀冻结及序列打包。
- 可选 FlashAttention 2，以及用于 log-probability 计算的融合实现。

### 奖励模型驱动的训练

- 支持规则奖励、自定义奖励函数、本地奖励模型和远程奖励服务。
- 支持组合多个奖励来源，并通过 reward recipe 对各项奖励进行聚合。
- 提供视觉序列奖励模型（SRM）、视觉生成奖励模型（GRM）和音频序列奖励模型训练入口。
- 支持在线策略蒸馏（On-Policy Distillation，OPD），可使用纯蒸馏目标，也可将教师信号与任务奖励结合。

### 多模态任务支持

- 文本策略模型：`ActorLanguage`。
- 视觉语言策略模型：`ActorVL`，支持图像输入；经验生成路径同时处理视频字段。
- 音频语言策略模型：`ActorAL`，仓库提供 R1-AQA 训练示例。
- 示例任务包括 GSM8K 文本推理、Geo3K 几何图像推理、视频奖励模型强化学习和音频问答。

### 实验记录与分析

- 支持 Weights & Biases 和 TensorBoard。
- 可保存训练轨迹、统计重复度、反思模式与策略熵等指标。
- 可标记并可视化高熵 token。
- 提供训练检查点、Hugging Face 格式保存及检查点转换工具。

## 训练方法与优化机制

主训练入口为 `examples/gsm8k_geo3k/train_colocate.py`。下表仅列出能够从该入口进入优势计算与训练流程的方法。

| 方法 | `--advantage_estimator` | Critic | 采样要求 | 当前实现说明 |
| --- | --- | --- | --- | --- |
| PPO / GAE | `gae` | 需要 | 无组采样要求 | 计算 GAE 和 value loss；未指定 `--critic_pretrain` 时使用策略模型路径初始化 Critic |
| REINFORCE | `reinforce` | 不需要 | 无组采样要求 | 使用序列奖励构造 token 级回报 |
| RLOO | `rloo` | 不需要 | `--n_samples_per_prompt > 1` | 使用 leave-one-out 组内基线 |
| REINFORCE with baseline | `reinforce_baseline` | 不需要 | `--n_samples_per_prompt > 1` | 使用组均值基线，不进行组标准差缩放 |
| GRPO | `group_norm` | 不需要 | `--n_samples_per_prompt > 1` | 对同一 prompt 的奖励进行组内中心化与标准化；命令行应使用 `group_norm` |
| CPGD | `cpgd` | 不需要 | 由任务配置决定 | 提供 CPGD 优势计算；配合 `--use_cpg_loss` 启用非对称裁剪策略损失 |
| 在线策略蒸馏 | `on_policy_distillation` | 不需要 | `--n_samples_per_prompt > 1` | 从 `--teacher_model_url` 获取教师 log-probability，并由 `--opd_kl_coef` 控制蒸馏项 |

可与上述方法组合使用的机制包括：

- `--dynamic_sampling`：在 GRPO 的组优势计算中屏蔽奖励全相同的组。
- `--overlong_buffer`：对超过指定长度阈值的响应施加长度相关惩罚。
- `--use_fire` 与 `--first_token_temperature`：仅对第一个生成 token 使用独立采样温度。
- `--high_entropy_token_ratio`：只使用每个样本中指定比例的高熵 token 计算策略梯度；`0.0` 表示不筛选。
- `--reward_running_norm`、`--reward_clip`、`--advantages_norm` 和 `--advantage_clip`：分别控制奖励运行时归一化、奖励裁剪、优势白化与优势裁剪。

实现边界说明：动态采样和超长响应缓冲是 DAPO 风格的独立机制，不等同于完整实现 DAPO 的全部算法组件。主入口虽然保留 `--use_gspo` 等实验性参数，并在参数选项中保留 `reinforce++`，但当前主训练损失或优势计算工厂未形成相应的完整可运行路径，因此本 README 不将其列为已支持方法。算法背景可参阅[算法说明](docs/source/quick_start/algorithms_zh.md)，运行能力仍以源码和本节表格为准。

相关方法资料包括 [GRPO](https://arxiv.org/abs/2402.03300)、[DAPO](https://arxiv.org/abs/2503.14476)、[CPGD](https://arxiv.org/abs/2505.12504)、[FIRE](https://arxiv.org/abs/2410.21236)、[高熵 token 筛选](https://arxiv.org/abs/2506.01939)和[在线策略蒸馏](https://thinkingmachines.ai/blog/on-policy-distillation/)。这些链接用于说明原始方法，LightRFT 的实现范围仍以本节的源码核对结果为准。

## 系统架构

LightRFT 的典型训练闭环如下：

```text
数据预处理 → rollout 采样 → 奖励与 experience 构造
           → 优势计算与策略更新 → 权重同步 → 下一轮采样
```

训练器负责组织迭代过程，Strategy 提供分布式训练与采样后端的运行接口，奖励组件完成规则或模型奖励计算。Actor、Reference Model、Critic 和 rollout policy 的关系，以及 engine sleep/wake、模型 reload/offload 和权重同步的具体时序，统一放在[运行时架构文档](docs/source/best_practice/runtime_architecture_zh.md)中说明。

## 安装

### 环境要求

| 组件 | 源码安装要求或说明 |
| --- | --- |
| Python | `>= 3.12` |
| PyTorch | `>= 2.9.1`（见 `pyproject.toml`） |
| GPU | 分布式训练需要 CUDA 可用的 NVIDIA GPU 环境 |
| 默认推理后端 | SGLang `>= 0.5.6.post2` |
| 可选推理后端 | vLLM `>= 0.18.1` |
| 训练后端 | DeepSpeed `>= 0.18.3`，或 PyTorch FSDP v2 |

CUDA、PyTorch、FlashAttention、SGLang 与 vLLM 之间存在二进制兼容约束。源码安装时应根据实际驱动和 CUDA 环境选择相互兼容的构建；仓库 `Dockerfile` 提供的是一个固定版本的参考环境。

### 源码安装

默认依赖包含 SGLang：

```bash
git clone https://github.com/opendilab/LightRFT.git
cd LightRFT
pip install -e .
```

如需使用 vLLM：

```bash
pip install -e ".[vllm]"
```

也可以在默认安装后单独安装兼容版本：

```bash
pip install "vllm>=0.18.1"
```

### Docker

运行 GPU 容器需要 Docker 和 NVIDIA Container Toolkit。[Docker Hub](https://hub.docker.com/r/opendilab/lightrft) 提供了 `v0.1.0` 示例镜像：

```bash
docker pull opendilab/lightrft:v0.1.0
docker run --gpus all -it --rm \
  --ipc=host \
  -v /path/to/data:/app/data \
  -v /path/to/checkpoints:/app/checkpoints \
  opendilab/lightrft:v0.1.0 /bin/bash
```

也可以基于仓库中的 `Dockerfile` 构建：

```bash
make dbuild

# 指定自定义镜像名称
make dbuild IMAGE_NAME=your-custom-tag:latest
```

当前 `Dockerfile` 以 `nvcr.io/nvidia/pytorch:25.01-py3` 为基础镜像，并显式安装 PyTorch 2.9.0（CUDA 12.8 wheel）、DeepSpeed 0.18.3、vLLM 0.18.1、FlashAttention 2.8.3 和 SGLang 0.5.6.post2。其中 PyTorch 版本低于源码包声明的 `>=2.9.1`。在将 Dockerfile 作为发布环境依据前，应先确认预期版本组合。

### FlashAttention 安装问题

如果 FlashAttention 源码构建失败，可以从 [FlashAttention Releases](https://github.com/Dao-AILab/flash-attention/releases) 选择与 Python、PyTorch、CUDA 和 C++ ABI 完全匹配的 wheel。例如，仓库 Docker 环境使用：

```bash
pip install flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

若没有匹配的预编译 wheel，需要在具备编译工具链的环境中从源码安装。详细说明见[安装文档](docs/source/installation/index_zh.rst)和[问题排查指南](docs/source/best_practice/troubleshooting_zh.md)。

## 快速上手

以下示例使用 GSM8K、Qwen2.5-0.5B-Instruct、GRPO 与规则奖励。示例脚本是训练模板：运行前必须检查数据路径、模型路径、GPU 数量、推理并行规模和日志配置。

### 1. 预处理 GSM8K

```bash
python examples/gsm8k_geo3k/data_preprocess/gsm8k.py \
  --local_save_dir /path/to/data/gsm8k
```

脚本从 `openai/gsm8k` 读取数据并生成训练、测试 Parquet 文件。数据条目包含 prompt、参考答案及 `gsm8k_rule` 奖励标签；训练使用答案正确性和输出格式组成的规则奖励，不需要神经奖励模型。

### 2. 修改训练脚本

编辑 `examples/gsm8k_geo3k/run_grpo_gsm8k_qwen2.5_0.5b.sh` 中的用户配置区，至少确认：

```bash
PATH_TO_YOUR_BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PATH_TO_YOUR_GSM8K_DATASET="/path/to/data/gsm8k"

export NNODES=1
export GPUS_PER_NODE=8
ENGINE_TP=2
```

如不使用 W&B，将 `WANDB_API_KEY` 留空，并按需移除或调整脚本中的 W&B 参数。`ENGINE_TP` 必须能够整除总进程数；多节点运行还需正确设置 `NODE_RANK`、`MASTER_ADDR` 和 `MASTER_PORT`。

### 3. 启动训练

使用默认 SGLang 后端：

```bash
ENGINE_TYPE=sglang \
  bash examples/gsm8k_geo3k/run_grpo_gsm8k_qwen2.5_0.5b.sh
```

使用已额外安装的 vLLM 后端：

```bash
ENGINE_TYPE=vllm \
  bash examples/gsm8k_geo3k/run_grpo_gsm8k_qwen2.5_0.5b.sh
```

脚本通过 `torchrun` 启动 `train_colocate.py`，默认示例使用 FSDP、BF16、FlashAttention、引擎 sleep/wake、规则奖励和组归一化优势。资源不足时，应同步缩小 `GPUS_PER_NODE`、`ENGINE_TP`、批量大小、序列长度或模型规模；不应仅修改进程数而保留不相容的推理张量并行配置。

Geo3K 图像任务可使用：

```bash
python examples/gsm8k_geo3k/data_preprocess/geo3k.py \
  --local_save_dir /path/to/data/geo3k

# 修改脚本中的模型和数据路径后运行
ENGINE_TYPE=sglang \
  bash examples/gsm8k_geo3k/run_grpo_geo3k_qwen2.5_vl_7b.sh
```

完整分步说明见 [GSM8K/Geo3K 教程](docs/source/quick_start/grpo_gsm8k_geo3k_tutorial_zh.md)。

## 配置说明

模型、数据、算法、分布式后端、推理引擎、日志和检查点参数统一整理在[配置参数详解](docs/source/quick_start/configuration_zh.md)中。不同训练入口暴露的参数并不完全相同，实际使用时还应查看对应脚本及命令行帮助：

```bash
python examples/gsm8k_geo3k/train_colocate.py --help
```

复现实验时应使用同一仓库版本的启动脚本、配置文档和参数解析代码，并以当前入口的 `--help` 与源码实现为最终依据。

## 示例与应用

| 目录 | 模态或任务 | 主要用途 |
| --- | --- | --- |
| `examples/gsm8k_geo3k/` | 文本、图像 | GSM8K/Geo3K 上的 GRPO、PPO、LoRA 和规则奖励训练 |
| `examples/orm_rl_demo/` | 图像 | 组合格式奖励、通用视觉奖励模型和准确率奖励的完整示例 |
| `examples/grm_training/` | 图像/视频奖励 | 视觉 GRM 训练 |
| `examples/grm_vl_rl/` | 视频 | 使用视觉奖励模型进行策略优化 |
| `examples/srm_training/` | 图像、音频 | 视觉或音频 SRM 训练 |
| `examples/r1_aqa/` | 音频 | R1-AQA 音频问答 GRPO 示例 |
| `examples/on_policy_distillation/` | 文本 | 在线策略蒸馏服务与训练示例 |
| `examples/math_benchmarks/` | 文本评测 | Math500、AIME 和 GPQA 等数学/推理评测入口 |
| `examples/entropy_viz/` | 分析工具 | 高熵 token 轨迹的本地 HTML 可视化 |
| `examples/chat/` | 交互工具 | 模型对话与生成检查示例 |

奖励模型的训练数据、模型结构和实践说明见[奖励模型文档](docs/source/best_practice/reward_model_zh.md)。各示例中的 shell 文件包含面向特定集群的路径、端口和 GPU 配置，使用前应将其视为模板进行审查。

## 监控、轨迹与检查点

LightRFT 支持 Weights & Biases、TensorBoard、训练轨迹保存与分析、高熵 token 可视化、分布式训练状态恢复及 Hugging Face 格式检查点。相关参数见[配置参数详解](docs/source/quick_start/configuration_zh.md)，检查点转换方法见[`lightrft/utils/ckpt_scripts/README_zh.md`](lightrft/utils/ckpt_scripts/README_zh.md)，轨迹可视化入口位于 `examples/entropy_viz/render_trajectories.html`。

## 项目结构

```text
LightRFT/
├── lightrft/
│   ├── datasets/                 # 文本与多模态数据集
│   ├── evaluation/               # 评测与奖励函数
│   ├── models/                   # 文本、视觉、音频 Actor 与奖励模型
│   ├── strategy/
│   │   ├── deepspeed/            # DeepSpeed 策略
│   │   ├── fsdp/                 # FSDP v2 策略
│   │   ├── sglang_utils/         # SGLang 引擎与权重同步
│   │   └── vllm_utils/           # vLLM 引擎与权重同步
│   ├── trainer/                  # 优势计算、经验生成与训练器
│   └── utils/                    # 日志、轨迹和检查点工具
├── examples/                     # 训练、蒸馏、评测与分析示例
├── docs/                         # Sphinx 文档
├── tools/                        # 版本与 Docker 辅助工具
├── README.md
└── README_zh.md
```

## 文档与问题排查

### 文档索引

- [安装指南](docs/source/installation/index_zh.rst)
- [GSM8K/Geo3K 快速教程](docs/source/quick_start/grpo_gsm8k_geo3k_tutorial_zh.md)
- [算法说明](docs/source/quick_start/algorithms_zh.md)
- [配置说明](docs/source/quick_start/configuration_zh.md)
- [训练策略](docs/source/best_practice/strategy_zh.md)
- [策略设计说明](docs/source/best_practice/strategy_design_philosophy_zh.md)
- [运行时架构与资源复用原理](docs/source/best_practice/runtime_architecture_zh.md)
- [奖励模型](docs/source/best_practice/reward_model_zh.md)
- [常见问题](docs/source/best_practice/faq_zh.md)
- [问题排查](docs/source/best_practice/troubleshooting_zh.md)
- [贡献指南](docs/source/best_practice/contributing_zh.md)

推理后端、显存、分布式初始化、多模态数据及训练稳定性问题请优先查阅[常见问题](docs/source/best_practice/faq_zh.md)和[问题排查指南](docs/source/best_practice/troubleshooting_zh.md)。

### 本地构建文档

```bash
pip install -r requirements-doc.txt
make docs
```

生成结果位于 `docs/build/html/index.html`。实时预览：

```bash
make docs-live
# 浏览器访问 http://localhost:8000
```

## 开发计划

- [v0.1.2 开发计划](https://github.com/opendilab/LightRFT/issues/28)
- [v0.1.1 开发计划](https://github.com/opendilab/LightRFT/issues/19)

开发计划用于记录拟议工作，不代表相关功能已经进入当前发布版本。

## 贡献

欢迎通过 Issue 和 Pull Request 参与开发。建议流程如下：

1. Fork 仓库，并基于 `main` 创建特性或文档分支。
2. 完成修改及必要测试，避免在同一提交中混入无关变更。
3. 按 [Conventional Commits](https://www.conventionalcommits.org/) 规范撰写提交信息。
4. 推送分支并创建 Pull Request，说明问题背景、修改内容与验证方法。

```bash
git checkout -b feature/your-feature-name
git commit -m "feature(user): add an example"
git push origin feature/your-feature-name
```

仓库当前使用的常见提交类型包括 `feature`、`fix`、`polish`、`docs`、`style` 和 `refactor`。文档分支建议在名称中包含 `doc`，以配合文档部署流程。完整规范见[贡献指南](docs/source/best_practice/contributing_zh.md)。

开发检查命令：

```bash
pip install -r requirements-dev.txt
make format   # YAPF
make fcheck   # Flake8
```

## 引用、许可证与致谢

### 引用

如果 LightRFT 对您的研究或应用有所帮助，请引用：

```bibtex
@misc{lightrft,
  title={LightRFT: Light, Efficient, Omni-modal & Reward-model Driven Reinforcement Fine-Tuning Framework},
  author={Niu, Yazhe and Pu, Yuan and Shi, Dongxing and Lu, Yudong and Xiong, Yingtong and Ge, Ruijun and Sun, Jiaxuan and Wan, Zunian and Zhang, Shaoang},
  publisher={GitHub},
  howpublished={\url{https://github.com/opendilab/LightRFT}},
  year={2025},
}
```

### 许可证

本项目采用 [Apache License 2.0](LICENSE)。

### 致谢

LightRFT 基于 [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) 开发，部分文件与实现由 OpenRLHF 改编或复用。项目亦借鉴或使用了 [verl](https://github.com/volcengine/verl)、[SGLang](https://github.com/sgl-project/sglang)、[vLLM](https://github.com/vllm-project/vllm)、[DeepSpeed](https://github.com/microsoft/DeepSpeed) 与 [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html) 等开源项目。

本项目由上海人工智能实验室系统平台中心与安全可信 AI 中心的同事合作开发。感谢所有贡献者与相关开源社区。

### 联系方式

- GitHub Issues：[opendilab/LightRFT Issues](https://github.com/opendilab/LightRFT/issues)
- 邮箱：opendilab@pjlab.org.cn
