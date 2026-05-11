#!/bin/bash
# Smoke test: verify the eval-pipeline fix in math_prm_trainer._runtime_eval_context.
#
# What this script verifies:
#   - With the fix (rollout_eos_patch detached during eval), the wandb-logged
#     eval/outcome_correct from the training pipeline should jump from
#     ~0.50 (broken pipeline at any RL ckpt) to ~0.69 (base URSA-8B real ability).
#   - We resume from base URSA-8B (no ckpt load), train for 1 step, then run
#     a full 500-sample eval. The first eval (after step 1) reports the model
#     ability under the FIXED pipeline.
#
# Expected wandb signature when fix is correct:
#   eval/outcome_correct ≈ 0.62-0.70   (not 0.50)
#   eval/answer_extraction_failed ≈ 0.01   (not 0.06)
#   The training log shows two new lines:
#     [eval] rollout_eos_patch detached for the eval pass
#     [eval] rollout_eos_patch reattached after eval
#
# Compare with the historical bug pipeline (PR #53 issuecomment-4394071500):
#   step20 wandb eval/outcome_correct = 0.379 (pre-fix, base + 20 RL steps)
#   step540 wandb eval/outcome_correct = 0.474 (pre-fix)

set -euo pipefail

PATH_TO_YOUR_BASE_MODEL="/home/ubuntu/URSA-MATH/checkpoints/URSA-8B"
PATH_TO_URSA_RM="/home/ubuntu/URSA-MATH/checkpoints/URSA-RM-8B"
PATH_TO_YOUR_MATH_DATASET="/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_psgrpo.jsonl"

EXPERIMENT_NAME="lightrft-ursa8b-mathprm-padding-fix-verify"

export WANDB_MODE="offline"
export WANDB_PROJECT="LightRFT-URSA8B-MathPRM-Smoke"

# Tiny rollout to keep the smoke fast — just enough to enter eval.
N_SAMPLES=2
EPISODE=1
WARMUP=0.0
RBS=128
TBS=128
KL_ESTIMATOR=k3
KL=0.001
LR=1e-6
PROMPT_MAX_LEN=1024
GENERATE_MAX_LEN=512
MAX_SAMPLES=128  # 1 rollout-step: 128 prompts / (RBS=128) = 1 step
limit_mm_image_per_prompt=10

# Run eval after every train step. Holdout = full 500 sample → outcome stats are
# directly comparable to the historical wandb numbers.
EVAL_STEPS=1
EVAL_HOLDOUT_SIZE=500
MAX_EVAL_SAMPLES=500

export NNODES="${NNODES:-1}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-20196}"  # different port from misalign-fix smoke

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
    --local_hf_max_new_tokens 512 \
    --hf_separate_rollout_actor \
    --hf_separate_rollout_keep_on_gpu \
    --enable_engine_sleep \
    --eval_steps ${EVAL_STEPS} \
    --eval_holdout_size ${EVAL_HOLDOUT_SIZE} \
    --max_eval_samples ${MAX_EVAL_SAMPLES} \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    2>&1 | tee "rft_logs/${EXPERIMENT_NAME}/node${NODE_RANK}_${current_time}.log"
