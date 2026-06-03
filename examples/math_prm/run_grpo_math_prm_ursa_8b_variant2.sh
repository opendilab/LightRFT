#!/bin/bash
# Fail fast: a crashed torchrun must propagate its exit code through the
# `2>&1 | tee` pipeline below so multi-node orchestrators / CI see the error.
set -eo pipefail
#
# LightRFT GRPO Training Script — URSA-8B with URSA-8B-RM, strict URSA-paper
# Eq.9 (variant 2) advantage.
#
# Differs from run_grpo_math_prm_ursa_8b.sh ONLY in --advantage_estimator:
#   ursa_variant2  — strict paper Eq.9 form, computed in
#                    examples/math_prm/ursa_variant2.py:
#                    A_t^i = r_{s,t}^i · GroupNorm_G(r̄_s^i)
#                          +              GroupNorm_G(r_o^i)
#                    A_t broadcast to every token in step t's span.
#                    No cumulative return. Outcome term retained.
#
# Auto-swaps PATH_TO_YOUR_MATH_DATASET to the .per_step_prm.jsonl sibling
# (label="math_per_step_prm") because variant 2 needs per-step labels.
#
# - Actor:    URSA-8B    (hybrid SAM-B + SigLIP-L vision tower + Qwen2.5-Math)
# - Reward:   URSA-8B-RM (process reward model for step-level scoring)
# - Engine:   local HF rollout (vLLM/SGLang URSA support is future work)
# - Algorithm: GRPO with PS-GRPO reward via the math_psgrpo label
#

# Auto-load credentials/paths from .env if present (no-op when missing).
# Useful keys: WANDB_API_KEY, WANDB_PROJECT, HF_TOKEN, PATH_TO_YOUR_BASE_MODEL,
# PATH_TO_URSA_RM, PATH_TO_YOUR_MATH_DATASET, LIGHTRFT_OUTPUT_ROOT.
if [ -f "$(dirname "$0")/../../.env" ]; then
    set -a; . "$(dirname "$0")/../../.env"; set +a
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# Alias project-specific WANDB key names to the canonical WANDB_API_KEY so
# the rest of the script (and wandb itself) can use the canonical name.
: "${WANDB_API_KEY:=${LIGHTRFT_WANDB_API_KEY:-${WANDB_TOKEN:-${WANDB_KEY:-}}}}"
export WANDB_API_KEY

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
# variant 2 NEEDS rows labeled "math_per_step_prm". The PS-GRPO dataset has
# label="math_psgrpo" everywhere — running variant 2 on it would silently
# emit zero step_rewards. .env on this box still points
# PATH_TO_YOUR_MATH_DATASET at the psgrpo .jsonl (legacy default), so we
# auto-swap to its sed-relabeled sibling (built once by the smoke script).
# If the caller wants a custom path, set PATH_TO_YOUR_MATH_DATASET_VARIANT2.
if [ -n "${PATH_TO_YOUR_MATH_DATASET_VARIANT2:-}" ]; then
    PATH_TO_YOUR_MATH_DATASET="${PATH_TO_YOUR_MATH_DATASET_VARIANT2}"
elif [ -n "${PATH_TO_YOUR_MATH_DATASET:-}" ] && [[ "${PATH_TO_YOUR_MATH_DATASET}" != *per_step_prm* ]]; then
    PATH_TO_YOUR_MATH_DATASET="${PATH_TO_YOUR_MATH_DATASET%.jsonl}.per_step_prm.jsonl"
fi
PATH_TO_YOUR_MATH_DATASET="${PATH_TO_YOUR_MATH_DATASET:-/path/to/your/preprocessed/math_per_step_prm.jsonl}"
if [ ! -f "${PATH_TO_YOUR_MATH_DATASET}" ]; then
    echo "[variant2 launch] FATAL: dataset not found: ${PATH_TO_YOUR_MATH_DATASET}" >&2
    exit 1
fi
# Sanity: first row must already be relabeled, otherwise variant 2 silently fails.
FIRST_LABEL=$(head -1 "${PATH_TO_YOUR_MATH_DATASET}" | python3 -c 'import sys,json; print(json.loads(sys.stdin.read()).get("label",""))' 2>/dev/null || echo "")
if [ "${FIRST_LABEL}" != "math_per_step_prm" ]; then
    echo "[variant2 launch] FATAL: dataset first row label='${FIRST_LABEL}', expected 'math_per_step_prm'." >&2
    echo "Pre-process with:" >&2
    echo "  sed 's/\"label\":[ ]*\"math_psgrpo\"/\"label\": \"math_per_step_prm\"/g' SRC > DST" >&2
    exit 1
fi
echo "[variant2 launch] using dataset: ${PATH_TO_YOUR_MATH_DATASET}"

# --- Experiment and Logging ---
EXPERIMENT_NAME="${EXPERIMENT_NAME:-lightrft-ursa8b-math-prm-variant2}"
LIGHTRFT_OUTPUT_ROOT="${LIGHTRFT_OUTPUT_ROOT:-.}"

# W&B configuration. Leave WANDB_API_KEY empty to disable W&B.
export WANDB_API_KEY="${WANDB_API_KEY:-YOUR_WANDB_API_KEY}"
WANDB_ORG="${WANDB_ORG:-${WANDB_ENTITY:-}}"
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
SAVE_DIR="${LIGHTRFT_OUTPUT_ROOT}/results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}"
LOG_DIR="${LIGHTRFT_OUTPUT_ROOT}/rft_logs/${EXPERIMENT_NAME}"
export WANDB_DIR="${WANDB_DIR:-${LIGHTRFT_OUTPUT_ROOT}/wandb}"

mkdir -p "${SAVE_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${WANDB_DIR}"
TRAIN_LOG="${LOG_DIR}/node${NODE_RANK}_${current_time}.log"

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

WANDB_ORG_ARGS=()
if [[ -n "${WANDB_ORG}" ]]; then
    WANDB_ORG_ARGS=(--wandb_org "${WANDB_ORG}")
fi

# Math PRM uses a single URSA-RM checkpoint registered under the math_prm label.
REWARD_PRETRAIN_PATHS="{\"math_prm\":\"${PATH_TO_URSA_RM}\"}"

# URSA enforces a fixed structured response format for the PRM scorer.
SYSTEM_PROMPT='A conversation between the User and Assistant. The User asks a question that may require mathematical or visual reasoning, and the Assistant solves it step by step. Each step MUST begin with "Step N:" (e.g. "Step 1:", "Step 2:") on its own line. After all steps, output exactly one final answer line prefixed with "†Answer:" (e.g. "†Answer: 42"). Stop immediately after the "†Answer:" line and do not output any extra text, repeated answer markers, or additional steps.'


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
    --save_path "${SAVE_DIR}" \
    --ckpt_path "${SAVE_DIR}" \
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
    --use_wandb "true" \
    "${WANDB_ORG_ARGS[@]}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    2>&1 | tee "${TRAIN_LOG}"


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
