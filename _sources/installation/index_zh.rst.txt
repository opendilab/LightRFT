.. _installation_zh:

==================================
安装指南
==================================

本指南提供了 LightRFT 的安装和设置说明。LightRFT 是一个轻量化、高性能的强化学习微调框架，专为大语言模型（LLM）和视觉语言模型（VLM）设计。

环境要求
========

在安装 LightRFT 之前，请确保您的环境满足以下要求：

* Python >= 3.12
* CUDA >= 12.8
* PyTorch >= 2.9.1
* 支持 CUDA 的 GPU

Docker 镜像
===========

TO BE DONE

安装步骤
========

标准安装
--------

LightRFT 默认使用 **SGLang** 作为推理后端，并包含 **Flash-Attention** 以优化性能。

.. code-block:: bash

   # 克隆仓库
   git clone https://github.com/opendilab/LightRFT.git
   cd LightRFT

   # 安装 LightRFT 及所有核心依赖
   pip install -e .

**安装内容**: PyTorch、SGLang、Flash-Attention、Transformers、DeepSpeed 和其他核心依赖。

可选：安装 vLLM 后端
--------------------

如果您想使用 vLLM 替代（或配合）SGLang：

.. code-block:: bash

   # 安装 vLLM 后端
   pip install ".[vllm]"

   # 或直接安装 vLLM
   pip install vllm>=0.13.3

Flash-Attention 安装问题排查
-----------------------------

Flash-Attention 默认包含在安装中，但在某些系统上可能因 CUDA 兼容性而安装失败。如果遇到问题，请尝试：

**方式 1: 使用预编译的 wheel 文件（推荐）**

.. code-block:: bash

   # 从 https://github.com/Dao-AILab/flash-attention/releases 下载适合的 wheel 文件
   # 例如 CUDA 12.x 和 PyTorch 2.9:
   pip install flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

**方式 2: 使用 Docker（最简单）**

.. code-block:: bash

   # 官方 Docker 镜像包含所有依赖
   docker pull opendilab/lightrft:v0.1.0

生成文档（可选）
================

安装文档生成所需的依赖：

.. code-block:: bash

   pip install -r requirements-doc.txt

生成 HTML 文档：

.. code-block:: bash

   make docs

文档将生成在 ``docs/build`` 目录中，打开 ``index.html`` 即可查看。

启动带自动重载的实时文档服务器：

.. code-block:: bash

   make docs-live

项目结构
========

LightRFT 组织成几个关键模块：

.. code-block:: text

   LightRFT/
   ├── lightrft/                      # 核心库
   │   ├── datasets/                  # 数据集实现
   │   │   ├── audio_alpaca.py        # 音频数据集
   │   │   ├── grm_dataset.py         # 生成式奖励模型数据集
   │   │   ├── prompts_dataset.py     # 提示数据集
   │   │   ├── prompts_dataset_vl.py  # 视觉语言提示数据集
   │   │   ├── sft_dataset.py         # SFT 数据集
   │   │   ├── sft_dataset_vl.py      # 视觉语言 SFT 数据集
   │   │   ├── srm_dataset.py         # 标量奖励模型数据集
   │   │   └── utils.py               # 数据集工具
   │   ├── models/                    # 模型定义
   │   │   ├── actor_al.py            # 音频-语言 Actor 模型
   │   │   ├── actor_language.py      # 语言 Actor 模型
   │   │   ├── actor_vl.py            # 视觉-语言 Actor 模型
   │   │   ├── grm_vl.py              # 生成式奖励模型 (视觉-语言)
   │   │   ├── srm_al.py              # 标量奖励模型 (音频-语言)
   │   │   ├── srm_vl.py              # 标量奖励模型 (视觉-语言)
   │   │   ├── loss.py                # 损失函数
   │   │   ├── utils.py               # 模型工具
   │   │   └── monkey_patch/          # 模型适配补丁
   │   │       ├── apply.py           # 补丁应用
   │   │       ├── hf_generate_patch.py  # HuggingFace generate 补丁
   │   │       ├── llama.py           # LLaMA 补丁
   │   │       └── qwen.py            # Qwen 补丁
   │   ├── strategy/                  # 训练&推理策略
   │   │   ├── config.py              # 策略配置
   │   │   ├── fake_strategy.py       # 测试用假策略
   │   │   ├── strategy.py            # 主策略实现
   │   │   ├── strategy_base.py       # 策略基类
   │   │   ├── deepspeed/             # DeepSpeed 实现
   │   │   │   ├── deepspeed.py       # DeepSpeed 策略
   │   │   │   └── deepspeed_utils.py # DeepSpeed 工具
   │   │   ├── fsdp/                  # FSDP 实现
   │   │   │   ├── fsdp_optimizer.py  # FSDP 优化器
   │   │   │   ├── fsdp_utils.py      # FSDP 工具
   │   │   │   └── fsdpv2.py          # FSDP v2 实现
   │   │   ├── sglang_utils/          # SGLang 工具
   │   │   │   ├── sglang_engine.py   # SGLang 引擎
   │   │   │   └── sgl_model_saver.py # SGLang 模型保存器
   │   │   ├── vllm_utils/            # vLLM 工具
   │   │   │   └── vllm_worker_wrap_no_ray.py  # vLLM worker 包装器
   │   │   └── utils/                 # 策略工具
   │   │       ├── broadcast_utils.py # 广播工具
   │   │       ├── ckpt_utils.py      # 检查点工具
   │   │       ├── data_utils.py      # 数据工具
   │   │       ├── distributed_util.py  # 分布式工具
   │   │       ├── optimizer_utils.py # 优化器工具
   │   │       ├── parallel_utils.py  # 并行工具
   │   │       └── statistic.py       # 统计工具
   │   ├── trainer/                   # Trainer 实现
   │   │   ├── experience_maker.py    # 经验生成器
   │   │   ├── experience_maker_vl.py # VLM 经验生成器
   │   │   ├── fast_exp_maker.py      # 快速经验生成器
   │   │   ├── grm_trainer_vl.py      # 生成式奖励模型训练器 (视觉-语言)
   │   │   ├── kl_controller.py       # KL 散度控制器
   │   │   ├── ppo_trainer.py         # PPO 训练器
   │   │   ├── ppo_trainer_vl.py      # 视觉-语言 PPO 训练器
   │   │   ├── replay_buffer.py       # 回放缓冲区
   │   │   ├── replay_buffer_utils.py # 回放缓冲区工具
   │   │   ├── replay_buffer_vl.py    # 视觉-语言回放缓冲区
   │   │   ├── spmd_ppo_trainer.py    # SPMD PPO 训练器
   │   │   ├── srm_trainer_al.py      # 标量奖励模型训练器 (音频-语言)
   │   │   ├── srm_trainer_vl.py      # 标量奖励模型训练器 (视觉-语言)
   │   │   └── utils.py               # 训练器工具
   │   └── utils/                     # 工具函数
   │       ├── cli_args.py            # CLI 参数解析
   │       ├── distributed_sampler.py # 分布式采样器
   │       ├── logging_utils.py       # 日志工具
   │       ├── processor.py           # 数据处理器
   │       ├── remote_rm_utils.py     # 远程奖励模型工具
   │       ├── timer.py               # 计时器工具
   │       ├── trajectory_saver.py    # 轨迹保存工具
   │       └── utils.py               # 通用工具
   │
   ├── examples/                      # 使用示例
   │   ├── chat/                      # 对话模型训练示例
   │   ├── grm_training/              # 通用奖励模型训练示例
   │   ├── gsm8k_geo3k/               # GSM8K/Geo3K 数学推理示例
   │   │   ├── data_preprocess/       # 数据预处理脚本
   │   │   ├── train_colocate.py      # 协同定位训练脚本
   │   │   ├── reward_models_utils.py # 奖励模型工具
   │   │   ├── run_grpo_gsm8k_qwen2.5_0.5b.sh    # GSM8K 训练脚本
   │   │   └── run_grpo_geo3k_qwen2.5_vl_7b.sh   # Geo3K VLM 训练脚本
   │   └── srm_training/              # 标量奖励模型训练示例
   │
   ├── docs/                          # 📚 Sphinx 文档
   │   └── source/
   │       ├── installation/          # 安装指南
   │       ├── quick_start/           # 快速开始 & 用户指南
   │       ├── best_practice/         # 最佳实践 & 资源
   │       └── api_doc/               # API 文档
   │           ├── datasets/          # 数据集 API
   │           ├── models/            # 模型 API
   │           ├── strategy/          # 策略 API
   │           ├── trainer/           # 训练器 API
   │           └── utils/             # 工具 API
   │
   ├── assets/                        # 资源文件
   │   └── logo.png                   # 项目Logo
   │
   ├── results/                       # 训练结果
   ├── rft_logs/                      # 训练日志
   ├── requirements.txt               # Python 依赖
   ├── requirements-dev.txt           # 开发依赖
   ├── requirements-doc.txt           # 文档依赖
   ├── setup.py                       # 包设置
   └── README.md                      # 项目文档

关键目录说明
------------

* **lightrft/**：LightRFT 核心库，包含五个主要模块：

  * ``datasets/``：数据集实现，支持提示、SFT、奖励建模（文本、视觉-语言、音频-语言）
  * ``models/``：Actor 模型（语言、视觉-语言、音频-语言）、奖励模型和损失函数
  * ``strategy/``：训练策略，包括 FSDP、DeepSpeed、vLLM/SGLang 集成
  * ``trainer/``：训练器实现，包括 PPO、经验生成和回放缓冲区
  * ``utils/``：工具函数，用于 CLI、日志、分布式训练和轨迹保存

* **examples/**：完整的训练示例和脚本

  * ``gsm8k_geo3k/``：GSM8K 和 Geo3K 数学推理训练示例
  * ``grm_training/``：生成式奖励模型训练示例
  * ``srm_training/``：标量奖励模型训练示例
  * ``chat/``：对话模型训练示例

* **docs/**：Sphinx 文档，包含完整的用户指南和 API 文档

验证安装
========

验证安装是否成功，运行简单测试：

.. code-block:: bash

   python -c "import lightrft; print(lightrft)"

如果没有导入错误，您应该会看到模块路径。

快速开始示例
============

安装完成后，尝试一个基础的 GRPO 训练示例：

.. code-block:: bash

   # 单节点 8 GPU 训练示例
   cd /path/to/LightRFT

   # 运行 GRPO 训练（GSM8K 数学推理任务）
   bash examples/gsm8k_geo3k/run_grpo_gsm8k_qwen2.5_0.5b.sh

   # 或者运行 Geo3K 几何问题训练（VLM 多模态）
   bash examples/gsm8k_geo3k/run_grpo_geo3k_qwen2.5_vl_7b.sh

故障排除
========

常见问题
--------

**问题**：CUDA 错误或版本不匹配

* **解决方案**：确保 CUDA 驱动和工具包版本与 PyTorch 安装匹配。使用 ``nvcc --version`` 和 ``python -c "import torch; print(torch.version.cuda)"`` 检查

**问题**：训练时内存不足错误

* **解决方案**：

  * 减小 ``micro_train_batch_size`` 或 ``micro_rollout_batch_size``
  * 启用梯度检查点：``--gradient_checkpointing``
  * 使用 FSDP + CPU 卸载：``--fsdp --fsdp_cpu_offload``
  * 调整引擎内存利用率：``--engine_mem_util 0.4``

**问题**：评测依赖安装缓慢

* **解决方案**：使用镜像或代理进行 pip 安装：

  .. code-block:: bash

     pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package>

获取更多支持
------------

如果遇到此处未涵盖的问题：

* 查看项目的 `GitHub Issues <https://github.com/opendilab/LightRFT/issues>`_
* 查阅 :doc:`../best_practice/strategy_zh` 了解训练配置
* 参考 ``examples/`` 目录中的示例脚本

后续步骤
========

安装成功后：

1. 查阅 :doc:`../quick_start` 指南了解基本使用方法
2. 探索 :doc:`../best_practice/strategy_zh` 了解分布式训练策略
3. 查看 ``examples/`` 目录中的完整训练示例
4. 阅读算法文档了解具体实现细节

相关文档
========

* :doc:`../quick_start` - 快速开始指南
* :doc:`../best_practice/strategy_usage` - Strategy 使用指南
* :doc:`../api/index` - API 参考
