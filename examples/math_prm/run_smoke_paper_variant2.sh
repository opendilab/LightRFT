#!/bin/bash
# Strict paper Eq.9 (URSA variant 2) smoke test.
#
# Differences vs run_smoke_per_step_prm_groupnorm.sh:
#   - Uses the new --advantage_estimator ursa_variant2 (paper Eq.9 strict
#     advantage formula, no cumsum, outcome reward retained as second
#     additive term).
#   - Forces --per_step_reward_mode raw because the variant-2 calculator
#     does its OWN group normalization on r̄_s; pre-normalizing step
#     rewards in fast_exp_maker would double-norm and break Eq.9 semantics.
#
# Expected outcome (post-run check_paper_variant2_smoke.py validates):
#   AC5  rollout/ursa_v2_adv_pos_frac and *_neg_frac both > 5%
#   AC6  rollout/alignment_failed < 5%
#   AC7  ≥5 train_step + ≥1 eval pass without NaN / crash
#   AC8  rollout/ursa_v2_msp_normed_std ≈ 1, ursa_v2_oc_normed_std ≈ 1
#
# Local resources (this box):
#   8x A100, /mnt/shared-storage-user/puyuan/... has URSA-8B + URSA-RM-8B.

set -euo pipefail

# Paths — overridable via env so this script also runs on the original
# /home/ubuntu/URSA-MATH layout if needed.
PATH_TO_YOUR_BASE_MODEL="${PATH_TO_YOUR_BASE_MODEL:-/mnt/shared-storage-user/puyuan/zhangshaoang/LightRFT/models/URSA-MATH/URSA-8B}"
PATH_TO_URSA_RM="${PATH_TO_URSA_RM:-/mnt/shared-storage-user/puyuan/zhangshaoang/LightRFT/models/URSA-MATH/URSA-RM-8B}"
PATH_TO_YOUR_MATH_DATASET="${PATH_TO_YOUR_MATH_DATASET:-/mnt/shared-storage-user/puyuan/zhangshaoang/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_psgrpo.jsonl}"
LIGHTRFT_OUTPUT_ROOT="${LIGHTRFT_OUTPUT_ROOT:-/mnt/shared-storage-user/puyuan/zhangshaoang/LightRFT/outputs}"

EXPERIMENT_NAME="lightrft-ursa8b-mathprm-paper-variant2-smoke"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-LightRFT-URSA8B-MathPRM-Smoke}"

# Small-scale: 5 train steps × 1 episode is enough to verify advantage shape
# + alignment success rate. We pick batch sizes so that 5 train steps cover
# multiple gather→train cycles (rollout_batch_size=16 → 4 micro batches per
# train step at micro_train_batch_size=4 default).
N_SAMPLES=4
EPISODE=1
WARMUP=0.0
# 8 GPU × default micro_train_batch_size=4 → TBS must be a multiple of 32.
# Pick 32 (smallest that satisfies the constraint).
RBS=32
TBS=32
KL_ESTIMATOR=k3
KL=0.001
LR=1e-6
PROMPT_MAX_LEN=1024
GENERATE_MAX_LEN=512
# 80 prompts × 4 samples = 320 trajectories total; with TBS=16 that's 20
# train steps if we ran the whole episode through. We cut short via
# --max_samples to ~5 train batches' worth of prompts.
MAX_SAMPLES=80

EVAL_STEPS=5
EVAL_HOLDOUT_SIZE=64
MAX_EVAL_SAMPLES=64

# Build a one-shot "math_per_step_prm" copy of the dataset (just relabels
# the rows; PRM extracts step boundaries from the response itself).
OVERRIDE_LABEL_DATASET="${PATH_TO_YOUR_MATH_DATASET%.jsonl}.per_step_prm.jsonl"
if [ ! -f "$OVERRIDE_LABEL_DATASET" ]; then
    echo "Building per_step_prm-labeled dataset from psgrpo source..."
    sed 's/"label":[ ]*"math_psgrpo"/"label": "math_per_step_prm"/g' \
        "$PATH_TO_YOUR_MATH_DATASET" > "$OVERRIDE_LABEL_DATASET"
    echo "  done: $(wc -l < $OVERRIDE_LABEL_DATASET) rows"
fi

export NNODES="${NNODES:-1}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-20198}"

current_time=$(date +"%Y%m%d_%H%M%S")
SAVE_MODEL_NAME="${EXPERIMENT_NAME}-${current_time}"
WANDB_RUN_NAME="${EXPERIMENT_NAME}-${current_time}"

mkdir -p "${LIGHTRFT_OUTPUT_ROOT}/results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}"
mkdir -p "${LIGHTRFT_OUTPUT_ROOT}/rft_logs/${EXPERIMENT_NAME}"

# repo root has /wandb owned by root on this box; redirect wandb to a writable
# location alongside the training output. This must come BEFORE wandb.init.
export WANDB_DIR="${LIGHTRFT_OUTPUT_ROOT}/wandb"
mkdir -p "${WANDB_DIR}"

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_DEBUG="WARN"

# CRITICAL: pip has lightrft editable-installed pointing at the puyuan code
# refactor copy, which (a) lacks the paper-Eq.9 estimator wiring this script
# depends on, and (b) eagerly imports sglang.srt at strategy_base import time
# which is broken on this box's sgl_kernel install. Force PYTHONPATH to our
# in-repo lightrft so torchrun-spawned workers pick it up.
export PYTHONPATH="$(cd "$(dirname "$0")/../.." && pwd):${PYTHONPATH:-}"

# Source .env (WANDB_API_KEY etc.) if available
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi
if [ -n "${LIGHTRFT_WANDB_API_KEY:-}" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY="$LIGHTRFT_WANDB_API_KEY"
fi

REWARD_PRETRAIN_PATHS="{\"math_prm\":\"${PATH_TO_URSA_RM}\"}"

SYSTEM_PROMPT='A conversation between the User and Assistant. The User asks a question that may require mathematical or visual reasoning, and the Assistant solves it step by step. Each step MUST begin with "Step N:" (e.g. "Step 1:", "Step 2:") on its own line. After all steps, output exactly one final answer line prefixed with "†Answer:" (e.g. "†Answer: 42"). Stop immediately after the "†Answer:" line and do not output any extra text, repeated answer markers, or additional steps.'

set -x

torchrun \
    --nnodes $NNODES \
    --nproc-per-node $GPUS_PER_NODE \
    --node_rank $NODE_RANK \
    --master-port $MASTER_PORT \
    --master-addr $MASTER_ADDR \
    examples/math_prm/train_colocate.py \
    --pretrain "${PATH_TO_YOUR_BASE_MODEL}" \
    --reward_pretrain "${REWARD_PRETRAIN_PATHS}" \
    --prompt_data "${OVERRIDE_LABEL_DATASET}" \
    --max_samples ${MAX_SAMPLES} \
    --input_key "prompt" \
    --images_key "images" \
    --label_key "label" \
    --apply_chat_template \
    --system_prompt "${SYSTEM_PROMPT}" \
    --save_path "${LIGHTRFT_OUTPUT_ROOT}/results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    --ckpt_path "${LIGHTRFT_OUTPUT_ROOT}/results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    --save_steps 999999 \
    --per_step_reward_mode raw \
    --max_ckpt_num 2 \
    --save_trajectories \
    --num_trajectories_to_save 4 \
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
    --advantage_estimator "ursa_variant2" \
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
    --kl_estimator ${KL_ESTIMATOR} \
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
    --use_wandb true \
    --wandb_org "${WANDB_ORG:-hansbug}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    2>&1 | tee "${LIGHTRFT_OUTPUT_ROOT}/rft_logs/${EXPERIMENT_NAME}/node${NODE_RANK}_${current_time}.log"
