#!/bin/bash
# Stage 3 smoke: variant 2 (per-step PRM reward) end-to-end verification.
#
# What this verifies:
# 1. MathPRMReward.forward returns step_rewards / step_token_indices when
#    label == "math_per_step_prm" (vs trajectory scalar for math_psgrpo).
# 2. fast_exp_maker plumbs them through _RewardBatchResult → outputs[i] →
#    experience.info → _compute_advantages_and_returns.
# 3. compute_reward scatters per-step rewards to step boundary tokens
#    (NOT only EOS) when step_rewards is provided.
# 4. cumulative_returns + GroupNorm produce per-token advantages with
#    higher within-trajectory variance than trajectory-scalar mode.
# 5. Training step succeeds (no NaN, no shape mismatch, no crash).
#
# How to read the wandb output:
#   - rollout_alignment_failed_rate < 5%  (step boundaries align with PRM)
#   - rollout_n_aligned_steps > 0         (most trajectories produce step rewards)
#   - train/advantages_std significantly > 0 within a trajectory
#     (per-step credit gives non-trivial advantage variance, vs trajectory-
#      scalar mode where every token in a traj has the same advantage)
#   - eval/outcome_correct comparable to smoke v2 (~0.58 ± noise) — 1 PPO step
#     shouldn't change outcome dramatically; this confirms the new path
#     doesn't catastrophically break.
#
# Compared to run_smoke_eval_fix_verify.sh:
#   - Same base URSA-8B + URSA-RM-8B + 8-rank FSDP + 1 PPO step + 500 eval
#   - Different label: "math_per_step_prm" instead of "math_psgrpo"
#   - PRM forward emits per-step credit; everything else identical

set -euo pipefail

PATH_TO_YOUR_BASE_MODEL="/home/ubuntu/URSA-MATH/checkpoints/URSA-8B"
PATH_TO_URSA_RM="/home/ubuntu/URSA-MATH/checkpoints/URSA-RM-8B"
PATH_TO_YOUR_MATH_DATASET="/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_psgrpo.jsonl"

EXPERIMENT_NAME="lightrft-ursa8b-mathprm-per-step-smoke"
export WANDB_MODE="offline"
export WANDB_PROJECT="LightRFT-URSA8B-MathPRM-Smoke"

N_SAMPLES=2
EPISODE=1
WARMUP=0.0
RBS=32
TBS=32
KL_ESTIMATOR=k3
KL=0.001
LR=1e-6
PROMPT_MAX_LEN=1024
GENERATE_MAX_LEN=512
MAX_SAMPLES=32

# Eval cycle
EVAL_STEPS=1
EVAL_HOLDOUT_SIZE=500
MAX_EVAL_SAMPLES=500

# IMPORTANT: override the dataset's label_key. The mmathcot_stage3
# dataset has every row labeled "math_psgrpo"; for this smoke we treat
# them as "math_per_step_prm" to exercise the new code path. We use
# argparse default override via env: see train_colocate.py logic for
# `args.label_key` mapping the dataset's label_key column. The simplest
# end-to-end path here is to monkey-patch the dataset by post-filtering
# in train_colocate.py — but instead we use the cleaner approach of a
# new flag --override_label that wraps the prompts dataset.
#
# For this smoke we add the override by re-using --label_key but
# pointing at a custom column. The mmathcot manifest already has
# 'label' field == 'math_psgrpo'. We add a sed-injected sibling with
# a "math_per_step_prm" label only when smoking — see SETUP below.

OVERRIDE_LABEL_DATASET="/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_per_step_prm.jsonl"
if [ ! -f "$OVERRIDE_LABEL_DATASET" ]; then
    echo "Building per_step_prm-labeled dataset from psgrpo source ..."
    sed 's/"label":[ ]*"math_psgrpo"/"label": "math_per_step_prm"/g' \
        "$PATH_TO_YOUR_MATH_DATASET" > "$OVERRIDE_LABEL_DATASET"
    echo "  done: $(wc -l < $OVERRIDE_LABEL_DATASET) rows"
fi

export NNODES="${NNODES:-1}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-20196}"

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
    --prompt_data "${OVERRIDE_LABEL_DATASET}" \
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
    --limit_mm_image_per_prompt 10 \
    --loss_agg_mode "seq-mean-token-mean" \
    --advantage_estimator "group_norm" \
    --per_step_reward_mode raw \
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
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    2>&1 | tee "rft_logs/${EXPERIMENT_NAME}/node${NODE_RANK}_${current_time}.log"
