# Tiny Python Expression RL Demo

[English](README.md)

这是仓库里最小的纯文本 RL fine-tuning 示例。

它保留了完整的 LightRFT 训练链路，但把任务收敛成：

- 模型：本地 Qwen 文本 checkpoint
- 任务：求解非常小的算术表达式
- reward：`format + correctness`
- 数据：由本地 Python 脚本现场生成

核心 `lightrft/` 包完全不改，任务相关逻辑全部收在 `examples/tiny_python_expr/` 下。

## 文件说明

- `build_dataset.py`：生成一个很小的算术数据集，并保存 `train` / `test`
- `reward_models_utils.py`：纯规则 reward，不加载神经 reward model
- `train_colocate.py`：自包含的最小 LightRFT 训练入口
- `run_qwen25_3b.sh`：本地和集群 worker 都可直接调用的最小启动脚本
- `.gitignore`：忽略运行时生成的 `data/` 和 `artifacts/`

## 这个 Demo 想说明什么

这个例子主要是为了把 LightRFT 里“任务定制面”压到最小，只保留三件事：

1. 定义数据格式。
2. 定义 reward 函数。
3. 写一个只保留必要参数的极简训练入口。

## 本地快速开始

最小直接运行方式：

```bash
bash examples/tiny_python_expr/run_qwen25_3b.sh
```

脚本默认会：

- 在 `examples/tiny_python_expr/data/generated` 下生成数据
- 把输出写到 `examples/tiny_python_expr/artifacts/`
- 使用 `/mnt/shared-storage-user/puyuan/model/Qwen2.5-3B-Instruct`
- 运行纯文本 GRPO，reward 只有规则项
- 默认 `WANDB_MODE=offline`
- 训练结束只写一个轻量 `training_complete.txt` 标记，不额外导出完整 final checkpoint

一个最小 2 卡 smoke：

```bash
NAME=tiny-python-expr-smoke \
TRAIN_SIZE=16 TEST_SIZE=8 \
N_SAMPLES=2 EPISODE=1 \
RBS=8 TBS=8 \
PROMPT_MAX_LEN=128 GENERATE_MAX_LEN=64 \
ENGINE_MEM_UTIL=0.35 \
bash examples/tiny_python_expr/run_qwen25_3b.sh
```

一个更长一些、适合看曲线的运行：

```bash
NAME=tiny-python-expr-20ep \
TRAIN_SIZE=32 TEST_SIZE=16 \
N_SAMPLES=4 EPISODE=20 \
RBS=8 TBS=8 \
PROMPT_MAX_LEN=128 GENERATE_MAX_LEN=64 \
ENGINE_MEM_UTIL=0.35 \
bash examples/tiny_python_expr/run_qwen25_3b.sh
```

## 单独构建数据集

`build_dataset.py` 导出的是 Hugging Face `DatasetDict` 格式，里面会有 `train/` 和 `test/` 两个 split。这个输出目录可以直接通过 `DATA_DIR` 或 `--prompt_data` 接到训练里。

最小可复制示例：

```bash
export DATA_DIR=/tmp/tiny_python_expr_dataset

python3 examples/tiny_python_expr/build_dataset.py \
  --output_dir "${DATA_DIR}" \
  --train_size 32 \
  --test_size 16 \
  --seed 42
```

然后直接复用这份已经导出的数据做训练：

```bash
DATA_DIR=/tmp/tiny_python_expr_dataset \
SKIP_DATASET_BUILD=1 \
NAME=tiny-python-expr-from-exported-data \
N_SAMPLES=4 EPISODE=4 \
RBS=8 TBS=8 \
PROMPT_MAX_LEN=128 GENERATE_MAX_LEN=64 \
ENGINE_MEM_UTIL=0.35 \
bash examples/tiny_python_expr/run_qwen25_3b.sh
```

如果你想看得更直白一点，训练入口最终读取的就是同一个目录，只不过参数名叫 `--prompt_data`：

```bash
torchrun \
  --nproc-per-node 2 \
  examples/tiny_python_expr/train_colocate.py \
  --pretrain /mnt/shared-storage-user/puyuan/model/Qwen2.5-3B-Instruct \
  --prompt_data /tmp/tiny_python_expr_dataset \
  --save_path examples/tiny_python_expr/artifacts/results/manual-run \
  --ckpt_path examples/tiny_python_expr/artifacts/results/manual-run \
  --micro_train_batch_size 1 \
  --train_batch_size 8 \
  --micro_rollout_batch_size 1 \
  --rollout_batch_size 8 \
  --num_episodes 1 \
  --n_samples_per_prompt 2 \
  --prompt_max_len 128 \
  --generate_max_len 64 \
  --actor_learning_rate 1e-6 \
  --init_kl_coef 0.001 \
  --engine_type sglang \
  --engine_mem_util 0.35 \
  --engine_tp_size 1
```

## `rlaunch` 集群启动流程

这个 example 不再单独保留 `run_rlaunch.sh`，完整集群启动流程直接写在这里。

运行前请先替换这些占位符：

- `<your-user>`：你的共享存储用户名
- `<model-owner>`：保存模型 checkpoint 的共享存储用户名
- `<your-wandb-entity>`：如果你要在线同步 W&B，这里换成你自己的 entity

推荐先在宿主机侧准备：

```bash
source .env

# 可选。只有宿主机需要在线访问 W&B 时才需要。
source /nfs/enable_proxy

export REPO_ROOT=/mnt/shared-storage-user/<your-user>/LightRFT
export MODEL_PATH=/mnt/shared-storage-user/<model-owner>/model/Qwen2.5-3B-Instruct
export WANDB_MODE=offline
export WANDB_PROJECT=tiny-python-expr
export WANDB_ORG=<your-wandb-entity>
export LIGHTRFT_WANDB_API_KEY="${LIGHTRFT_WANDB_API_KEY:-${WANDB_API_KEY:-}}"
```

然后提交一个最小 2 卡任务：

```bash
rlaunch \
  --memory=500000 \
  --cpu=40 \
  --gpu=2 \
  --charged-group=rlinfra_gpu \
  --private-machine=yes \
  --custom-resources brainpp.cn/fuse=1 \
  --image=registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/easyr1:lightrft-20260119 \
  --mount=gpfs://gpfs1/<model-owner>:/mnt/shared-storage-user/<model-owner> \
  --mount=gpfs://gpfs1/<your-user>:/mnt/shared-storage-user/<your-user> \
  -e NCCL_IB_DISABLE=1 \
  -e WANDB_MODE="${WANDB_MODE}" \
  -e WANDB_PROJECT="${WANDB_PROJECT}" \
  -e WANDB_ORG="${WANDB_ORG}" \
  -e LIGHTRFT_WANDB_API_KEY="${LIGHTRFT_WANDB_API_KEY}" \
  -e NAME=tiny-python-expr-rlaunch \
  -e MODEL_PATH="${MODEL_PATH}" \
  -e TRAIN_SIZE=16 \
  -e TEST_SIZE=8 \
  -e N_SAMPLES=2 \
  -e EPISODE=1 \
  -e RBS=8 \
  -e TBS=8 \
  -e PROMPT_MAX_LEN=128 \
  -e GENERATE_MAX_LEN=64 \
  -e ENGINE_MEM_UTIL=0.35 \
  -d -- bash -lc '
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/miniconda3/envs/lightrft

REPO_ROOT=/mnt/shared-storage-user/<your-user>/LightRFT
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export LD_LIBRARY_PATH=/root/miniconda3/envs/lightrft/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/root/miniconda3/envs/lightrft/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/root/miniconda3/envs/lightrft/lib/python3.12/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/root/miniconda3/envs/lightrft/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/root/miniconda3/envs/lightrft/lib:${LD_LIBRARY_PATH}

export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_DEBUG=WARN
export IGNORE_EOS=0

PYTHONUNBUFFERED=1 bash examples/tiny_python_expr/run_qwen25_3b.sh \
  2>&1 | tee -a examples/tiny_python_expr/artifacts/rlaunch_smoke.log
'
```

## W&B 说明

- 这个 example 默认离线运行，不依赖 W&B 凭据。
- 如果你想在线记录，把 `WANDB_MODE=online`，同时提供 `LIGHTRFT_WANDB_API_KEY` 或 `WANDB_API_KEY`，并把 `WANDB_ORG` 改成你真实可用的 entity。
- W&B 运行目录在 `examples/tiny_python_expr/artifacts/wandb/` 下，这部分已经被 example 自己的 `.gitignore` 忽略。

## 生成文件说明

这个 example 故意不把运行产物放进 git：

- `examples/tiny_python_expr/data/`
- `examples/tiny_python_expr/artifacts/`
- `examples/tiny_python_expr/__pycache__/`
