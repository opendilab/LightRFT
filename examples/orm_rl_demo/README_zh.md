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
export DATA_PATH=/path/to/geo3k
export PRETRAIN_PATH=/path/to/Qwen2.5-VL-7B-Instruct
export REWARD_PRETRAIN_PATHS='{"general":"/path/to/general-reward-model"}'
bash examples/orm_rl_demo/run_general_fsdp_qwenvl.sh
```

脚本本身是模板，不内置任何集群或个人路径；运行前请先按上面方式显式设置数据和模型路径。

## Demo 流程

这个 demo 的目标是更直观地看到 ORM RL 的主流程：
- actor 在 Geo3K 上生成 trajectory
- general ORM 对 trajectory 打分
- 保留 trajectory 保存，便于调试和理解训练过程

为了不去改写现有 Geo3K 数据文件，这个 demo 在运行时把数据标签覆盖成 `geo3k_general`，这样可以沿用原始数据路径，同时走本 demo 的 general ORM reward 融合逻辑。

## 环境要求

环境要求与仓库根目录 [README_zh.md](../../README_zh.md#环境要求) 保持一致，请直接参考主文档。

## 说明

- 这个 demo 有意只保留一个 shell 入口。
- Geo3K 的 reward 路由通过运行时标签覆盖完成，不直接改写数据集本身。
- 运行所需路径通过环境变量传入，避免把集群或个人信息写进示例脚本。

## 真实全量实验记录

这个 demo 已经基于一次真实的 2 卡全量训练完成了验通，配置是 `sglang` rollout + `rm_use_engine=True` 的 general ORM 路径，而不是只做本地 smoke。

- 汇报形式参考：upstream PR54 <https://github.com/opendilab/LightRFT/pull/54>
- 本次 upstream PR 实验汇报 comment：<https://github.com/opendilab/LightRFT/pull/56#issuecomment-4272514537>
- W&B run：<https://wandb.ai/hansbug/ORM-RL-Demo-QwenVL-7B-Geo3K/runs/zrekazyw>
- run name：`ORM-RL-Demo-Geo3K-General-SGLang-20260417_150451`
- worker 启动脚本：`/mnt/shared-storage-user/zhangshaoang/.orm_rl_demo_full_sglang_20260417.sh`
- 原始训练日志：`/mnt/shared-storage-user/zhangshaoang/.orm_rl_demo_full_sglang_20260417_150345.log`
- 结果目录：`/mnt/shared-storage-user/zhangshaoang/LightRFT/results/orm-rl-demo-general-geo3k-sglang/LightRFT-geo3k-general-orm-sglang-len_1024_2048-tbs_128-rbs_128-sample_8-kl_0.001-warmup_0.03-ep_20-lr_1e-6-20260417_150451`
- trajectory 目录：`/mnt/shared-storage-user/zhangshaoang/LightRFT/results/orm-rl-demo-general-geo3k-sglang/LightRFT-geo3k-general-orm-sglang-len_1024_2048-tbs_128-rbs_128-sample_8-kl_0.001-warmup_0.03-ep_20-lr_1e-6-20260417_150451/trajectories`

### 实际生效配置

| 项目 | 值 |
| --- | --- |
| 集群资源 | `2 GPU / 40 CPU / 500000 memory` |
| 镜像 | `registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/easyr1:lightrft-20260119` |
| Conda 环境 | `/root/miniconda3/envs/lightrft` |
| Actor | `/mnt/shared-storage-user/puyuan/model/Qwen2.5-VL-7B-Instruct` |
| General RM | `/mnt/shared-storage-user/puyuan/model/Qwen2.5-VL-7B-Instruct` |
| 数据 | `/mnt/shared-storage-user/puyuan/data/geo3k` |
| Rollout engine | `sglang` |
| RM 推理 | `rm_use_engine=True`，backend=`sglang` |
| Reward 融合 | `format 0.1 + general_model 0.2 + accuracy 0.7` |
| Batch 大小 | `train_batch_size=128`, `rollout_batch_size=128` |
| Micro batch 大小 | `micro_train_batch_size=4`, `micro_rollout_batch_size=4` |
| 采样配置 | `n_samples_per_prompt=8`, `num_episodes=20` |
| 长度配置 | `prompt_max_len=1024`, `generate_max_len=2048` |
| 优化 / KL | `actor_learning_rate=1e-6`, `init_kl_coef=0.001`, `lr_warmup_ratio=0.03` |
| 保存配置 | `max_ckpt_num=1`, `save_trajectories=True`, `num_trajectories_to_save=16` |

这次 worker 在启动训练前，还显式补齐了 `sglang` 所需的 runtime 环境：

- `conda activate /root/miniconda3/envs/lightrft`
- `PYTHONPATH=/mnt/shared-storage-user/zhangshaoang/LightRFT:$PYTHONPATH`
- `LD_LIBRARY_PATH` 额外加入：
- `/usr/local/nvidia/lib`
- `/usr/local/nvidia/lib64`
- `/root/miniconda3/envs/lightrft/lib/python3.12/site-packages/nvidia/cuda_runtime/lib`
- `/root/miniconda3/envs/lightrft/lib/python3.12/site-packages/nvidia/cudnn/lib`
- `/root/miniconda3/envs/lightrft/lib/python3.12/site-packages/nvidia/cublas/lib`
- `/root/miniconda3/envs/lightrft/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib`
- `/root/miniconda3/envs/lightrft/lib`

### 核心结果

- 训练完整跑完，最终 `train/global_step=320`
- 整个过程中一共触发了 `16` 次 eval
- `eval/reward_mean` 从 `0.4636` 提升到 `0.5679`
- best `eval/reward_mean=0.5686`，出现在 `train_step=260`
- final `eval/accuracy_reward_mean=0.5166`
- final `eval/format_reward_mean=0.9956`
- final `eval/general_model_reward_mean=0.1067`
- final `train/general_model_reward_mean=0.1309`
- final `train/step_reward_mean=0.6883`
- final `train/kl=0.5952`

从结果上看，可以比较明确地得到下面这些结论：

- 这条 ORM RL demo 链路不只是“能启动”，而是已经在真实 `rlaunch` 环境里完整跑完一版
- `accuracy_reward` 是中后期主要的增益来源
- `general_model_reward` 始终是正向项，能提供额外加分
- `format_reward` 很早就接近饱和，后续基本稳定在接近 `1.0`

### 实验图表

#### Summary Card

![](assets/verified_full_run_20260417/summary_card.png)

#### Reward Dashboard

![](assets/verified_full_run_20260417/reward_dashboard.png)

#### Optimization Dashboard

![](assets/verified_full_run_20260417/optimization_dashboard.png)

### 同题从 Step 80 到 Step 320 的对照样例

这次 run 在 `step80` 和 `step320` 之间，实际只有 2 道共同题目，所以最直接、也最不容易歧义的展示方式，就是对这 2 道题分别做“同题早期 vs 同题末期”的真实对照。

因此这里改成 4 张真实卡片：

- Question A at step 80
- Question A at step 320
- Question B at step 80
- Question B at step 320

#### Question A：平行四边形面积题

这道题体现的是“答案已经接近正确，但还没命中规则答案；后期修正成规则答案后 reward 跳升”的过程。

![](assets/verified_full_run_20260417/question_a_step80.png)

![](assets/verified_full_run_20260417/question_a_step320.png)

- 共同题面：`Find the area of the parallelogram. Round to the nearest tenth if necessary.`
- Step 80 来源：`trajectories_step_80.json`, `idx=0`, image `images/step80_exp0_sample0_img0.png`
- Step 320 来源：`trajectories_step_320.json`, `idx=0`, image `images/step320_exp0_sample0_img0.png`
- Step 80 输出摘录：`... The area of the parallelogram is approximately 38.97 square feet. \boxed{38.97}`
- Step 320 输出摘录：`... The area of the parallelogram is approximately \boxed{39.0}.`
- Step 80 reward 拆解：`total=0.3`, `format=1.0`, `accuracy=0.0`, `general_model=0.2`, `rule=0.1`
- Step 320 reward 拆解：`total=1.0`, `format=1.0`, `accuracy=1.0`, `general_model=0.2`, `rule=0.8`
- 含义：step 80 时 actor 已经给出了非常接近的答案，所以 `general_model_reward` 已经是正的；到 step 320 时，输出从 `38.97` 修正成规则答案 `39.0`，于是 `accuracy_reward` 从 `0.0` 跳到了 `1.0`。

#### Question B：切线几何 `y`

这道题体现的是更剧烈的变化，也就是从明显错误的解答，演化到完整正确的解答。

![](assets/verified_full_run_20260417/question_b_step80.png)

![](assets/verified_full_run_20260417/question_b_step320.png)

- 共同题面：`Find y. Assume that segments that appear to be tangent are tangent. Round to the nearest tenth if necessary.`
- Step 80 来源：`trajectories_step_80.json`, `idx=8`, image `images/step80_exp8_sample0_img0.png`
- Step 320 来源：`trajectories_step_320.json`, `idx=8`, image `images/step320_exp8_sample0_img0.png`
- Step 80 输出摘录：`... However, the correct value is: \[ y = 10 \] </think> The radius \( y \) is \boxed{10}.`
- Step 320 输出摘录：`... \[ y = \sqrt{160} = 4\sqrt{10} \approx 12.6 \] </think> The value of \(y\) is approximately \boxed{12.6}.`
- Step 80 reward 拆解：`total=0.1`, `format=1.0`, `accuracy=0.0`, `general_model=0.0`, `rule=0.1`
- Step 320 reward 拆解：`total=1.0`, `format=1.0`, `accuracy=1.0`, `general_model=0.2`, `rule=0.8`
- 含义：step 80 时基本只保住了 format，accuracy 和 general RM 都没有给分；到 step 320 时，这两项都变成了正向贡献。

## 许可证

本项目采用 Apache 2.0 许可证。详见 [LICENSE](../../LICENSE)。
