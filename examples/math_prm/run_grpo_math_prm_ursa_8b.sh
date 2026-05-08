#!/bin/bash
#
# LightRFT GRPO Training Script - URSA-8B with URSA-8B-RM (Math PRM).
#
# This script trains URSA-8B (a multimodal math VLM built on Qwen2.5-Math) with
# URSA-8B-RM as a Process Reward Model. The reward signal is PS-GRPO over the
# PRM step scores: r in {0, 0.5, 1} based on outcome correctness and whether
# any step-score drop event was observed in the response.
#
# - Actor:    URSA-8B    (hybrid SAM-B + SigLIP-L vision tower + Qwen2.5-Math)
# - Reward:   URSA-8B-RM (process reward model for step-level scoring)
# - Engine:   local HF rollout (vLLM/SGLang URSA support is future work)
# - Algorithm: GRPO with PS-GRPO reward via the math_psgrpo label
#

# Auto-load credentials/paths from .env if present (no-op when missing).
# Useful keys: WANDB_API_KEY, WANDB_PROJECT, HF_TOKEN, PATH_TO_YOUR_BASE_MODEL,
# PATH_TO_URSA_RM, PATH_TO_YOUR_MATH_DATASET (any var here is overridable via
# the outer environment).
if [ -f "$(dirname "$0")/../../.env" ]; then
    set -a; . "$(dirname "$0")/../../.env"; set +a
fi

################################################################################
#                           Part 1: User Configuration                         #
# Please update the following paths and settings to match your environment.    #
################################################################################

# --- Model and Dataset Paths ---
# Each value can be overridden by exporting the env var with the same name
# before invoking this script (e.g. for CI or per-machine paths). The strings
# below are placeholders to make the script self-documenting; a real run must
# either edit them or override via env.
PATH_TO_YOUR_BASE_MODEL="${PATH_TO_YOUR_BASE_MODEL:-/path/to/your/URSA-8B}"
PATH_TO_URSA_RM="${PATH_TO_URSA_RM:-/path/to/your/URSA-RM-8B}"
PATH_TO_YOUR_MATH_DATASET="${PATH_TO_YOUR_MATH_DATASET:-/path/to/your/preprocessed/math_psgrpo.jsonl}"

# --- Experiment and Logging ---
EXPERIMENT_NAME="${EXPERIMENT_NAME:-lightrft-ursa8b-math-prm}"

# W&B configuration. Leave WANDB_API_KEY empty to disable W&B.
export WANDB_API_KEY="${WANDB_API_KEY:-YOUR_WANDB_API_KEY}"
export WANDB_PROJECT="${WANDB_PROJECT:-LightRFT-URSA8B-MathPRM}"


################################################################################
#                       Part 2: Training Hyperparameters                       #
# These settings control the training process. Adjust them as needed.          #
################################################################################

# --- GRPO settings ---
N_SAMPLES=8              # Number of samples per prompt for GRPO (must be > 1).
EPISODE=10               # Total number of training episodes.
WARMUP=0.03              # Learning rate warmup ratio.
RBS=128                  # Rollout Batch Size.
TBS=128                  # Training Batch Size.

# --- Learning and model settings ---
# K3 estimator (Schulman) at the historical default 0.001. The earlier proposal
# to switch to K2 + 0.005 was justified by KL ~ 11 nats observed on the broken
# run; once the silent log-prob misalignment was fixed (see PR #53), the real
# K3 sits at ~0.04 and the K2/K3/K1 ratios collapse to numerically equivalent
# small values, so the estimator + coefficient change has no remaining
# justification. Keep historical values to minimize the PR's behavior diff.
KL_ESTIMATOR=k3          # Schulman K3 = exp(-r) - 1 + r. Historical default.
KL=0.001                 # Historical default. K3 * 0.001 ~= 4e-5 budget on real KL.
KL_TARGET=""             # If set (e.g. "0.5"), enables AdaptiveKLController.
# Variant 2 per-step PRM reward mode. Only meaningful when prompts have label
# "math_per_step_prm" (see fast_exp_maker._apply_step_reward_group_norm). Values:
#   raw         : scatter raw sigmoid step_score (paper Figure ablation; default)
#   group_norm  : per-step group-relative baseline (GRPO convention)
PER_STEP_REWARD_MODE="${PER_STEP_REWARD_MODE:-raw}"

LR=1e-6                  # Actor learning rate.
PROMPT_MAX_LEN=1024      # Max length of the input prompt.
GENERATE_MAX_LEN=3072    # Max length of the generated response.
MAX_SAMPLES=15360        # Cap on the training subset size.

# --- Multi-modal settings ---
limit_mm_image_per_prompt=10

# --- Evaluation settings ---
# Eval pulls a fixed deterministic held-out subset out of the training manifest
# (URSA Stage 3 protocol).
EVAL_STEPS=20
EVAL_HOLDOUT_SIZE=500
MAX_EVAL_SAMPLES=500


################################################################################
#                    Part 3: Distributed Training Setup                        #
# Configure settings for multi-GPU and multi-node training.                    #
################################################################################

export NNODES="${NNODES:-1}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-20092}"


################################################################################
#                      Part 4: Execution and Logging                           #
# This section prepares and launches the training command.                     #
################################################################################

# --- Generate dynamic names and paths ---
# SAVE_MODEL_NAME / WANDB_RUN_NAME are env-overridable so a resumed run can target
# the existing ckpt directory instead of creating a fresh timestamped one.
current_time=$(date +"%Y%m%d_%H%M%S")
SAVE_MODEL_NAME="${SAVE_MODEL_NAME:-${EXPERIMENT_NAME}-ep${EPISODE}-kl${KL}-lr${LR}-${current_time}}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXPERIMENT_NAME}-${current_time}}"

mkdir -p "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}"
mkdir -p "rft_logs/${EXPERIMENT_NAME}"

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_DEBUG="WARN"
if [[ -n "${WANDB_API_KEY}" && "${WANDB_API_KEY}" != "YOUR_WANDB_API_KEY" ]]; then
    export WANDB_MODE="${WANDB_MODE:-online}"
else
    export WANDB_MODE="${WANDB_MODE:-offline}"
fi

# Optional adaptive-KL flag block (only added when KL_TARGET is non-empty).
KL_TARGET_ARGS=()
if [[ -n "${KL_TARGET}" ]]; then
    KL_TARGET_ARGS=(--kl_target "${KL_TARGET}")
fi

# Optional resume-from-checkpoint flag. Set LOAD_CHECKPOINT=1 in the environment
# to continue training from ${ckpt_path}/_actor (and _critic if applicable).
RESUME_ARGS=()
if [[ "${LOAD_CHECKPOINT:-0}" == "1" ]]; then
    RESUME_ARGS=(--load_checkpoint)
fi

# Math PRM uses a single URSA-RM checkpoint registered under the math_prm label.
REWARD_PRETRAIN_PATHS="{\"math_prm\":\"${PATH_TO_URSA_RM}\"}"

# URSA enforces a fixed structured response format for the PRM scorer.
SYSTEM_PROMPT='A conversation between the User and Assistant. The User asks a question that may require mathematical or visual reasoning, and the Assistant solves it step by step. Each step MUST begin with "Step N:" (e.g. "Step 1:", "Step 2:") on its own line. After all steps, output exactly one final answer line prefixed with "†Answer:" (e.g. "†Answer: 42"). Stop immediately after the "†Answer:" line and do not output any extra text, repeated answer markers, or additional steps.'

set -x


################################################################################
#                         Part 5: Main Training Command                        #
################################################################################

# Use the conda env's torchrun explicitly: under bash -c, `conda activate` does
# not propagate to subprocesses, so a plain `torchrun` may resolve to a system
# python that lacks transformers/flash_attn etc. Override with TORCHRUN= if you
# launch from a different env.
TORCHRUN="${TORCHRUN:-torchrun}"
"${TORCHRUN}" \
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
    --save_steps 20 \
    --max_ckpt_num 2 \
    --save_trajectories \
    --num_trajectories_to_save 16 \
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
    --per_step_reward_mode "${PER_STEP_REWARD_MODE}" \
    "${KL_TARGET_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
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
    --use_wandb "${WANDB_API_KEY}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    2>&1 | tee "rft_logs/${EXPERIMENT_NAME}/node${NODE_RANK}_${current_time}.log"


################################################################################
#                           Usage Instructions                                 #
#                                                                              #
# Step 1: Prepare the URSA-8B actor and URSA-8B-RM reward model checkpoints.   #
#   Both are public on Hugging Face under the URSA-MATH project. Set           #
#   PATH_TO_YOUR_BASE_MODEL and PATH_TO_URSA_RM to the local directories.      #
#                                                                              #
# Step 2: Preprocess the math PRM dataset.                                     #
#   `python examples/math_prm/tools/prepare_ursa_stage3_manifest.py`           #
#   produces a JSONL manifest with fields {prompt, images, reference, label}   #
#   where label="math_psgrpo" enables the PS-GRPO reward path.                 #
#                                                                              #
# Step 3: Configure the script.                                                #
#   Edit "Part 1: User Configuration" at the top of this file. Set the paths   #
#   to your URSA-8B actor, URSA-8B-RM reward model, and preprocessed manifest. #
#                                                                              #
# Step 4: Run the training script.                                             #
#   `bash examples/math_prm/run_grpo_math_prm_ursa_8b.sh`                      #
#                                                                              #
################################################################################
