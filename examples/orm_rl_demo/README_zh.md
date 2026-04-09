<div align="center">

# ORM RL Demo 训练示例

LightRFT 中面向 Geo3K 的最小 ORM 强化学习示例。

</div>

## 概述

这个示例目录现在收敛为一个单独可运行的 ORM RL demo，用来理清已有训练流程：
- 数据集：Geo3K
- actor：Qwen2.5-VL 7B actor checkpoint
- reward 侧：单一路 general outcome reward model
- 后端：FSDP 训练 + engine 侧 reward 推理

## 项目结构

```text
orm_rl_demo/
├── train_colocate.py
├── reward_models.py
├── reward_models_utils.py
├── test_reward_models.py
└── run_general_fsdp_qwenvl.sh
```

## 快速开始

这个 demo 只保留一个入口脚本：

```bash
bash examples/orm_rl_demo/run_general_fsdp_qwenvl.sh
```

这个脚本尽量复用了仓库里已经出现过的集群路径风格，继续使用当前示例中已有的 actor / reward model 路径。

## Demo 流程

这个 demo 的目标是更直观地看到 ORM RL 的主流程：
- actor 在 Geo3K 上生成 trajectory
- general ORM 对 trajectory 打分
- 保留 trajectory 保存，便于调试和理解训练过程

为了不去改写现有 Geo3K 数据文件，这个 demo 在运行时把数据标签覆盖成 `general`，这样可以沿用原始数据路径，同时走 general ORM 的 reward recipe。

## 环境要求

- Python >= 3.8
- CUDA >= 11.8（用于 GPU 训练）
- 72B 奖励模型配置建议使用 8x A100 (80GB) 或类似规格硬件

## 说明

- 这个 demo 有意只保留一个 shell 入口。
- Geo3K 的 reward 路由通过运行时标签覆盖完成，不直接改写数据集本身。
- 当前 reward model 路径继续保留为这个示例目录原本使用的集群路径风格。

## 许可证

本项目采用 Apache 2.0 许可证。详见 [LICENSE](../../LICENSE)。
