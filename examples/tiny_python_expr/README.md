# Tiny Python Expression RL Demo

[简体中文](README_zh.md)

This example is the smallest text-only RL fine-tuning demo in this repository.

It keeps the full LightRFT training stack, but simplifies the task to:

- model: a local Qwen text checkpoint
- task: solve tiny arithmetic expressions
- reward: `format + correctness`
- data: generated on the fly by a local Python script

The core `lightrft/` package is intentionally untouched. Everything task-specific lives under `examples/tiny_python_expr/`.

## Files

- `build_dataset.py`: generates a tiny arithmetic dataset and saves `train` / `test`
- `reward_models_utils.py`: pure rule-based reward, no neural reward model
- `train_colocate.py`: self-contained minimal LightRFT training entry
- `run_qwen25_3b.sh`: minimal runnable launcher for local or cluster workers
- `.gitignore`: ignores generated `data/` and `artifacts/`

## What The Demo Shows

This example is meant to show the minimum task-specific surface area in LightRFT:

1. Define a dataset format.
2. Define a reward function.
3. Write a tiny training entry that only keeps the arguments this demo really needs.

## Local Quick Start

The smallest direct run is:

```bash
bash examples/tiny_python_expr/run_qwen25_3b.sh
```

By default the script:

- generates a dataset under `examples/tiny_python_expr/data/generated`
- stores outputs under `examples/tiny_python_expr/artifacts/`
- uses `/mnt/shared-storage-user/puyuan/model/Qwen2.5-3B-Instruct`
- runs text-only GRPO with rule-based reward only
- keeps `WANDB_MODE=offline` unless you override it
- writes a lightweight `training_complete.txt` marker instead of exporting a full final checkpoint

A tiny 2-GPU smoke run:

```bash
NAME=tiny-python-expr-smoke \
TRAIN_SIZE=16 TEST_SIZE=8 \
N_SAMPLES=2 EPISODE=1 \
RBS=8 TBS=8 \
PROMPT_MAX_LEN=128 GENERATE_MAX_LEN=64 \
ENGINE_MEM_UTIL=0.35 \
bash examples/tiny_python_expr/run_qwen25_3b.sh
```

A longer run for checking curves:

```bash
NAME=tiny-python-expr-20ep \
TRAIN_SIZE=32 TEST_SIZE=16 \
N_SAMPLES=4 EPISODE=20 \
RBS=8 TBS=8 \
PROMPT_MAX_LEN=128 GENERATE_MAX_LEN=64 \
ENGINE_MEM_UTIL=0.35 \
bash examples/tiny_python_expr/run_qwen25_3b.sh
```

## Build Dataset Separately

`build_dataset.py` exports a Hugging Face `DatasetDict` with `train/` and `test/` splits, and that output can be passed directly to training through `DATA_DIR` or `--prompt_data`.

Minimal copy-paste example:

```bash
export DATA_DIR=/tmp/tiny_python_expr_dataset

python3 examples/tiny_python_expr/build_dataset.py \
  --output_dir "${DATA_DIR}" \
  --train_size 32 \
  --test_size 16 \
  --seed 42
```

Then reuse exactly that exported dataset for training:

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

If you want the most explicit connection, the training entry ultimately reads the same directory via `--prompt_data`:

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

## `rlaunch` Cluster Flow

This example does not keep a separate `run_rlaunch.sh`. The full cluster launch flow is documented here instead.

Before you submit the job, replace these placeholders:

- `<your-user>`: your shared-storage user name
- `<model-owner>`: the shared-storage owner that holds the model checkpoint
- `<your-wandb-entity>`: your W&B entity when you want online sync

Recommended host-side setup:

```bash
source .env

# Optional. Only needed when you want online W&B access from this machine.
source /nfs/enable_proxy

export REPO_ROOT=/mnt/shared-storage-user/<your-user>/LightRFT
export MODEL_PATH=/mnt/shared-storage-user/<model-owner>/model/Qwen2.5-3B-Instruct
export WANDB_MODE=offline
export WANDB_PROJECT=tiny-python-expr
export WANDB_ORG=<your-wandb-entity>
export LIGHTRFT_WANDB_API_KEY="${LIGHTRFT_WANDB_API_KEY:-${WANDB_API_KEY:-}}"
```

Then submit a minimal 2-GPU run:

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

## W&B Notes

- The example defaults to offline mode, so it can run without W&B credentials.
- If you want online logging, set `WANDB_MODE=online`, provide `LIGHTRFT_WANDB_API_KEY` or `WANDB_API_KEY`, and override `WANDB_ORG` with your real entity.
- Generated W&B files stay under `examples/tiny_python_expr/artifacts/wandb/`, which is ignored by this example's `.gitignore`.

## Generated Files

This example intentionally keeps generated files out of git:

- `examples/tiny_python_expr/data/`
- `examples/tiny_python_expr/artifacts/`
- `examples/tiny_python_expr/__pycache__/`
