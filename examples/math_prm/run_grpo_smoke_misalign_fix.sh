#!/bin/bash
# Short smoke test of the silent-gather misalignment fix.
#
# Goal: confirm that with the patched UrsaActor.forward + log_probs_from_logits
# shape assert, the wandb metric "train/kl" comes back at ~0.04 (the real
# policy KL) instead of ~30 (the silent-misalignment artifact).
#
# This script reuses the same checkpoints + dataset as the dev-train logs
# at rft_logs/lightrft-ursa8b-mathprm-dev-train/node0_20260427_222814.log.
# It overrides batch sizes / eval / save to keep the run short (~5 PPO steps).

set -euo pipefail

PATH_TO_YOUR_BASE_MODEL="/home/ubuntu/URSA-MATH/checkpoints/URSA-8B"
PATH_TO_URSA_RM="/home/ubuntu/URSA-MATH/checkpoints/URSA-RM-8B"
PATH_TO_YOUR_MATH_DATASET="/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_psgrpo.jsonl"

EXPERIMENT_NAME="lightrft-ursa8b-mathprm-misalign-smoke"

# wandb offline (we read metrics from the log + local wandb dir)
export WANDB_MODE="offline"
export WANDB_PROJECT="LightRFT-URSA8B-MathPRM-Smoke"

# Tiny rollout to keep the smoke fast.
N_SAMPLES=2              # 2 samples per prompt (smoke only)
EPISODE=1                # one pass
WARMUP=0.0               # no warmup so LR is full from step 1
RBS=32                   # 32 prompts per rollout (must be divisible by world_size=8)
TBS=32                   # train batch (must be divisible by micro_train_batch_size * world_size = 4*8 = 32)
KL_ESTIMATOR=k3          # use the SAME estimator as the broken historical run,
                         # so the fix's effect is unambiguous
KL=0.001                 # SAME kl_coef as the broken historical run
KL_TARGET=""
LR=1e-6
PROMPT_MAX_LEN=1024
GENERATE_MAX_LEN=768     # short to keep smoke fast
MAX_SAMPLES=128          # 4 rollouts of RBS=32 prompts
limit_mm_image_per_prompt=10

# No eval, no save (smoke test only).
EVAL_STEPS=999999
EVAL_HOLDOUT_SIZE=8
MAX_EVAL_SAMPLES=8

export NNODES="${NNODES:-1}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-20193}"

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
    --max_samples ${MAX_SAMPLES} \
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
    --limit_mm_image_per_prompt $limit_mm_image_per_prompt \
    --loss_agg_mode "seq-mean-token-mean" \
    --advantage_estimator "group_norm" \
    --max_epochs 1 \
    --num_episodes ${EPISODE} \
    --lr_warmup_ratio ${WARMUP} \
    --n_samples_per_prompt $N_SAMPLES \
    --train_batch_size ${TBS} \
    --rollout_batch_size ${RBS} \
    --prompt_max_len $PROMPT_MAX_LEN \
    --generate_max_len $GENERATE_MAX_LEN \
    --actor_learning_rate $LR \
    --use_kl_loss \
    --init_kl_coef $KL \
    --kl_estimator "${KL_ESTIMATOR}" \
    --engine_type "hf" \
    --engine_mem_util 0.6 \
    --local_hf_generate_max_batch_size 4 \
    --local_hf_max_new_tokens 384 \
    --hf_separate_rollout_actor \
    --hf_separate_rollout_keep_on_gpu \
    --enable_engine_sleep \
    --eval_steps ${EVAL_STEPS} \
    --eval_holdout_size ${EVAL_HOLDOUT_SIZE} \
    --max_eval_samples ${MAX_EVAL_SAMPLES} \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    2>&1 | tee "rft_logs/${EXPERIMENT_NAME}/node${NODE_RANK}_${current_time}.log"
