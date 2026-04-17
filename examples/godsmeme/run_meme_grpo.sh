#!/usr/bin/env bash
#
# LightRFT GRPO training script for the GodsMeme example.
#
# Compared with examples/gsm8k_geo3k/run_grpo_geo3k_qwen2.5_vl_7b.sh,
# GodsMeme has a few task-specific constraints:
# 1. The dataset is loaded from --annotation_path + --root_dir, not --prompt_data.
# 2. The reward is not pure rule-based; it uses a local pairwise meme judge model.
# 3. The current meme reward model does not support --rm_use_engine.
# 4. Each GRPO group must stay inside one micro-rollout batch:
#      micro_rollout_batch_size % n_samples_per_prompt == 0
# 5. Pairwise judge cost grows quadratically with n_samples_per_prompt unless you cap
#    max_pairs_per_group.
#

set -euo pipefail

################################################################################
#                           Part 1: User Configuration                         #
################################################################################

# --- Model and Dataset Paths ---
# The policy model can be either a local path or a Hugging Face model id.
POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-/path/to/your/policy-model}"

# The meme judge model is loaded locally by examples/godsmeme/reward_model.py.
# In the simplest setup, this can be the same model family as the policy model.
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-/path/to/your/reward-model}"

# GodsMeme expects pre-built RL rows in JSON or JSONL format.
ANNOTATION_PATH="${ANNOTATION_PATH:-/path/to/your/train_data.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-/path/to/your/image_root}"

# --- Experiment and Logging ---
EXPERIMENT_NAME="${EXPERIMENT_NAME:-lightrft-godsmeme-grpo-training}"
RESULT_ROOT="${RESULT_ROOT:-results}"
LOG_ROOT="${LOG_ROOT:-rft_logs}"

# Set WANDB_API_KEY="" to disable W&B cleanly.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-LightRFT-GodsMeme-Experiments}"
export WANDB_MODE="${WANDB_MODE:-offline}"


################################################################################
#                    Part 2: GodsMeme Reward Configuration                     #
################################################################################

# Reward prompt template used by the pairwise judge.
REWARD_PROMPT_PATH="${REWARD_PROMPT_PATH:-examples/godsmeme/prompts/reward_compare.txt}"

# Reward cost control. 0 means use all pairs inside each rollout group.
MAX_PAIRS_PER_GROUP="${MAX_PAIRS_PER_GROUP:-0}"
PAIR_BATCH_SIZE="${PAIR_BATCH_SIZE:-4}"
REWARD_MAX_NEW_TOKENS="${REWARD_MAX_NEW_TOKENS:-96}"

# Reward composition:
# final_reward = model_reward_weight * pairwise_reward
#              + format_reward_weight * format_reward
MODEL_REWARD_WEIGHT="${MODEL_REWARD_WEIGHT:-1.0}"
FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-0.1}"


################################################################################
#                       Part 3: Training Hyperparameters                       #
################################################################################

# --- GRPO Settings ---
N_SAMPLES="${N_SAMPLES:-8}"
EPISODE="${EPISODE:-20}"
WARMUP="${WARMUP:-0.03}"

# --- Batch Size Settings ---
RBS="${RBS:-128}"
TBS="${TBS:-128}"
MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-8}"
MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-4}"

# --- Learning and Generation Settings ---
KL="${KL:-0.01}"
LR="${LR:-1e-6}"
PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-2048}"
GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-1024}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"

# The actor only sees one source image per prompt in GodsMeme.
LIMIT_MM_IMAGE_PER_PROMPT="${LIMIT_MM_IMAGE_PER_PROMPT:-1}"


################################################################################
#                    Part 4: Distributed Training Setup                        #
################################################################################

export NNODES="${NNODES:-1}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-20091}"

# The rollout engine is for the actor only. The meme reward model is loaded
# directly and kept outside rm_use_engine.
ENGINE_TYPE="${ENGINE_TYPE:-sglang}"
ENGINE_TP="${ENGINE_TP:-2}"
ENGINE_MEM_UTIL="${ENGINE_MEM_UTIL:-0.4}"


################################################################################
#                        Part 5: Validation and Launch                         #
################################################################################

if (( N_SAMPLES <= 1 )); then
    echo "[GodsMeme] N_SAMPLES must be > 1 when using group_norm / GRPO." >&2
    exit 1
fi

if (( MICRO_ROLLOUT_BATCH_SIZE % N_SAMPLES != 0 )); then
    echo "[GodsMeme] MICRO_ROLLOUT_BATCH_SIZE must be divisible by N_SAMPLES." >&2
    echo "[GodsMeme] This keeps every rollout group inside one micro-batch for pairwise reward computation." >&2
    exit 1
fi

if (( ENGINE_TP <= 0 )); then
    echo "[GodsMeme] ENGINE_TP must be a positive integer." >&2
    exit 1
fi

if (( GPUS_PER_NODE % ENGINE_TP != 0 )); then
    echo "[GodsMeme] GPUS_PER_NODE must be divisible by ENGINE_TP." >&2
    exit 1
fi

PAIR_TAG="allpairs"
if (( MAX_PAIRS_PER_GROUP > 0 )); then
    PAIR_TAG="pairs${MAX_PAIRS_PER_GROUP}"
fi

DEFAULT_REWARD_PRETRAIN="$({
    REWARD_MODEL_PATH="$REWARD_MODEL_PATH" \
    REWARD_PROMPT_PATH="$REWARD_PROMPT_PATH" \
    PAIR_BATCH_SIZE="$PAIR_BATCH_SIZE" \
    MAX_PAIRS_PER_GROUP="$MAX_PAIRS_PER_GROUP" \
    MODEL_REWARD_WEIGHT="$MODEL_REWARD_WEIGHT" \
    FORMAT_REWARD_WEIGHT="$FORMAT_REWARD_WEIGHT" \
    REWARD_MAX_NEW_TOKENS="$REWARD_MAX_NEW_TOKENS" \
    N_SAMPLES="$N_SAMPLES" \
    python - <<'PY'
import json
import os

cfg = {
    "pairwise": {
        "path": os.environ["REWARD_MODEL_PATH"],
        "reward_prompt_path": os.environ["REWARD_PROMPT_PATH"],
        "pair_batch_size": int(os.environ["PAIR_BATCH_SIZE"]),
        "max_pairs_per_group": int(os.environ["MAX_PAIRS_PER_GROUP"]),
        "model_reward_weight": float(os.environ["MODEL_REWARD_WEIGHT"]),
        "format_reward_weight": float(os.environ["FORMAT_REWARD_WEIGHT"]),
        "max_new_tokens": int(os.environ["REWARD_MAX_NEW_TOKENS"]),
        "n_samples_per_prompt": int(os.environ["N_SAMPLES"]),
    }
}
print(json.dumps(cfg, ensure_ascii=True))
PY
})"
REWARD_PRETRAIN="${REWARD_PRETRAIN:-$DEFAULT_REWARD_PRETRAIN}"

current_time=$(date +"%Y%m%d_%H%M%S")
SAVE_MODEL_NAME="${EXPERIMENT_NAME}-ep${EPISODE}-kl${KL}-lr${LR}-${PAIR_TAG}-${current_time}"
WANDB_RUN_NAME="${EXPERIMENT_NAME}-${PAIR_TAG}-${current_time}"

mkdir -p "${RESULT_ROOT}/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}"
mkdir -p "${LOG_ROOT}/${EXPERIMENT_NAME}"

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export IGNORE_EOS="${IGNORE_EOS:-0}"

set -x

torchrun \
    --nnodes "$NNODES" \
    --nproc-per-node "$GPUS_PER_NODE" \
    --node_rank "$NODE_RANK" \
    --master-port "$MASTER_PORT" \
    --master-addr "$MASTER_ADDR" \
    examples/godsmeme/train_colocate.py \
    --pretrain "${POLICY_MODEL_PATH}" \
    --reward_pretrain "${REWARD_PRETRAIN}" \
    --annotation_path "${ANNOTATION_PATH}" \
    --root_dir "${IMAGE_ROOT}" \
    --save_path "${RESULT_ROOT}/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    --ckpt_path "${RESULT_ROOT}/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    --advantage_estimator "group_norm" \
    --use_kl_loss \
    --kl_estimator "k3" \
    --fsdp \
    --bf16 \
    --flash_attn \
    --gradient_checkpointing \
    --save_hf_ckpt \
    --micro_train_batch_size "${MICRO_TRAIN_BATCH_SIZE}" \
    --train_batch_size "${TBS}" \
    --micro_rollout_batch_size "${MICRO_ROLLOUT_BATCH_SIZE}" \
    --rollout_batch_size "${RBS}" \
    --max_epochs 1 \
    --num_episodes "${EPISODE}" \
    --lr_warmup_ratio "${WARMUP}" \
    --n_samples_per_prompt "${N_SAMPLES}" \
    --prompt_max_len "${PROMPT_MAX_LEN}" \
    --generate_max_len "${GENERATE_MAX_LEN}" \
    --actor_learning_rate "${LR}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --init_kl_coef "${KL}" \
    --l2 1.0e-2 \
    --freeze_prefix \
    --adam_offload \
    --engine_type "${ENGINE_TYPE}" \
    --engine_mem_util "${ENGINE_MEM_UTIL}" \
    --engine_tp_size "${ENGINE_TP}" \
    --enable_engine_sleep \
    --limit_mm_image_per_prompt "${LIMIT_MM_IMAGE_PER_PROMPT}" \
    --use_wandb "${WANDB_API_KEY}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    2>&1 | tee "${LOG_ROOT}/${EXPERIMENT_NAME}/node${NODE_RANK}_${current_time}.log"


################################################################################
#                                Usage Notes                                   #
#                                                                              #
# 1. GodsMeme data must already be prepared as conversation-style RL rows.     #
# 2. Do not add --rm_use_engine here: the current meme reward model raises     #
#    NotImplementedError when rm_use_engine is enabled.                        #
# 3. LIMIT_MM_IMAGE_PER_PROMPT defaults to 1 because each policy prompt only   #
#    contains one source image. The pairwise judge images are handled inside   #
#    examples/godsmeme/reward_model.py.                                        #
# 4. If reward evaluation is too slow, first try lowering N_SAMPLES or set     #
#    MAX_PAIRS_PER_GROUP to a small positive integer.                          #
#                                                                              #
################################################################################
