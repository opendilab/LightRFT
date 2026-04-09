<div align="center">

# ORM RL Demo 训练示例

LightRFT 中用于 ORM 强化学习训练的最小化示例材料。

</div>

## 概述

这个目录正在统一为更中性的 `orm_rl_demo` 命名。

当前保留的材料主要包括：
- 基于 Qwen2.5-VL 风格模型的多模态 actor 训练
- 协同部署的 outcome reward model 打分
- 基于 FSDP 的训练方式，以及 SGLang / vLLM 生成后端
- 用于调试的轻量 trajectory 保存能力

## 项目结构

```text
orm_rl_demo/
├── train_colocate.py
├── reward_models.py
├── reward_models_utils.py
├── test_reward_models.py
├── run_general_fsdp_qwenvl.sh
├── run_kg_fsdp_qwenvl.sh
└── run_fsdp_deepseek.sh
```

## 快速开始

当前目录中的主要通用入口脚本是：

```bash
bash examples/orm_rl_demo/run_general_fsdp_qwenvl.sh
```

这个脚本保持现有训练流程不变，只把目录名和示例名统一到了 `orm_rl_demo`。

## 环境要求

- Python >= 3.8
- CUDA >= 11.8（用于 GPU 训练）
- 更大的奖励模型配置建议使用 8x A100 (80GB) 或类似规格硬件

## 说明

- 当前目录中的训练逻辑仍然保持原样。
- 这次提交只做命名和路径引用的统一，不调整训练流程本身。
- 后续如果需要进一步缩减示例范围，可以在当前命名统一的基础上继续进行。

## 许可证

本项目采用 Apache 2.0 许可证。详见 [LICENSE](../../LICENSE)。
