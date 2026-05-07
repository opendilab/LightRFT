#!/bin/bash
# Initial-eval-only smoke: load base URSA-8B onto the 8-GPU FSDP training pipeline,
# run a single 500-sample eval at step 0 (NO PPO update), exit.
#
# Goal: measure the TRUE baseline outcome in the training pipeline (8-rank FSDP +
# bs=4 batched generate + no patch + same DistributedSampler). This isolates how
# much of the "0.5833 step 1 vs 0.694 standalone bs=1" gap is bs-or-FSDP induced
# vs. PPO step drift.

set -euo pipefail

PATH_TO_YOUR_BASE_MODEL="/home/ubuntu/URSA-MATH/checkpoints/URSA-8B"
PATH_TO_URSA_RM="/home/ubuntu/URSA-MATH/checkpoints/URSA-RM-8B"
PATH_TO_YOUR_MATH_DATASET="/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_psgrpo.jsonl"

EXPERIMENT_NAME="lightrft-ursa8b-base-eval-only"
export WANDB_MODE="offline"
export WANDB_PROJECT="LightRFT-URSA8B-MathPRM-Smoke"

EVAL_HOLDOUT_SIZE=500
MAX_EVAL_SAMPLES=500

export NNODES="${NNODES:-1}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-20195}"

current_time=$(date +"%Y%m%d_%H%M%S")
SAVE_MODEL_NAME="${EXPERIMENT_NAME}-${current_time}"
WANDB_RUN_NAME="${EXPERIMENT_NAME}-${current_time}"

mkdir -p "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}"
mkdir -p "rft_logs/${EXPERIMENT_NAME}"

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_DEBUG="WARN"

REWARD_PRETRAIN_PATHS="{\"math_prm\":\"${PATH_TO_URSA_RM}\"}"

SYSTEM_PROMPT='A conversation between the User and Assistant. The User asks a question that may require mathematical or visual reasoning, and the Assistant solves it step by step. Each step MUST begin with "Step N:" (e.g. "Step 1:", "Step 2:") on its own line. After all steps, output exactly one final answer line prefixed with "†Answer:" (e.g. "†Answer: 42"). Stop immediately after the "†Answer:" line and do not output any extra text, repeated answer markers, or additional steps.'

set -x

/home/ubuntu/miniconda3/envs/lightrft/bin/torchrun \
    --nnodes $NNODES \
    --nproc-per-node $GPUS_PER_NODE \
    --node_rank $NODE_RANK \
    --master-port $MASTER_PORT \
    --master-addr $MASTER_ADDR \
    examples/math_prm/train_colocate.py \
    --pretrain "${PATH_TO_YOUR_BASE_MODEL}" \
    --reward_pretrain "${REWARD_PRETRAIN_PATHS}" \
    --prompt_data "${PATH_TO_YOUR_MATH_DATASET}" \
    --max_samples 32 \
    --input_key "prompt" \
    --images_key "images" \
    --label_key "label" \
    --apply_chat_template \
    --system_prompt "${SYSTEM_PROMPT}" \
    --save_path "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    --ckpt_path "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    --save_steps 999999 \
    --max_ckpt_num 2 \
    --print_replay_buffer_stats \
    --fsdp \
    --bf16 \
    --flash_attn \
    --gradient_checkpointing \
    --zero_stage 3 \
    --adam_offload \
    --freeze_prefix \
    --l2 1.0e-2 \
    --mixed_mm_data \
    --limit_mm_image_per_prompt 10 \
    --loss_agg_mode "seq-mean-token-mean" \
    --advantage_estimator "group_norm" \
    --max_epochs 1 \
    --num_episodes 1 \
    --lr_warmup_ratio 0.0 \
    --n_samples_per_prompt 2 \
    --train_batch_size 32 \
    --rollout_batch_size 32 \
    --prompt_max_len 1024 \
    --generate_max_len 512 \
    --actor_learning_rate 1e-6 \
    --use_kl_loss \
    --init_kl_coef 0.001 \
    --kl_estimator k3 \
    --engine_type "hf" \
    --engine_mem_util 0.6 \
    --local_hf_generate_max_batch_size 4 \
    --local_hf_max_new_tokens 512 \
    --hf_separate_rollout_actor \
    --hf_separate_rollout_keep_on_gpu \
    --enable_engine_sleep \
    --eval_steps 999999 \
    --eval_holdout_size ${EVAL_HOLDOUT_SIZE} \
    --max_eval_samples ${MAX_EVAL_SAMPLES} \
    --initial_eval \
    --initial_eval_only \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    2>&1 | tee "rft_logs/${EXPERIMENT_NAME}/node${NODE_RANK}_${current_time}.log"
