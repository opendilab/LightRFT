#!/bin/bash
#
# Start OPD training. Requires TEACHER_URL env var.
#
# Usage:
#   TEACHER_URL=http://127.0.0.1:13141/generate bash examples/on_policy_distillation/start_training.sh
#

set -euo pipefail

if [ -z "${TEACHER_URL:-}" ]; then
    echo "ERROR: TEACHER_URL not set."
    echo "Start the teacher first:  bash examples/on_policy_distillation/start_teacher.sh"
    echo "Then export TEACHER_URL=http://host:port/generate"
    exit 1
fi

# --- Configuration ---
STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
DATASET_PATH="${DATASET_PATH:-/path/to/your/dataset.jsonl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-qwen}"

export WANDB_API_KEY="${WANDB_API_KEY:-YOUR_WANDB_API_KEY}"
export WANDB_PROJECT="${WANDB_PROJECT:-LightRFT-OnPolicyDistillation}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# --- GPU setup ---
GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
ENGINE_TP=$( [ "$GPUS_PER_NODE" -ge 2 ] && echo 2 || echo 1 )

# --- Hyperparameters ---
N_SAMPLES=${N_SAMPLES:-8}
EPISODE=${EPISODE:-30}
OPD_KL_COEF=${OPD_KL_COEF:-1.0}
MICRO_TRAIN_BS=${MICRO_TRAIN_BS:-4}
MICRO_ROLLOUT_BS=${MICRO_ROLLOUT_BS:-4}
LR=${LR:-5e-7}

WORLD_SIZE=$((1 * GPUS_PER_NODE))
ALIGN=$((MICRO_TRAIN_BS * WORLD_SIZE))
RBS=$(( (${RBS:-128} / ALIGN) * ALIGN ))
TBS=$(( (${TBS:-128} / ALIGN) * ALIGN ))
[ "$RBS" -lt "$ALIGN" ] && RBS=$ALIGN
[ "$TBS" -lt "$ALIGN" ] && TBS=$ALIGN

ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-on_policy_distillation}"
USE_TASK_REWARD="${USE_TASK_REWARD:-false}"

if [ "$USE_TASK_REWARD" = "true" ]; then
    TASK_REWARD_FLAG="--use_task_reward"
else
    TASK_REWARD_FLAG="--no_task_reward"
fi

if [ "$USE_TASK_REWARD" = "true" ]; then
    KL=${KL:-0.01}
else
    KL=${KL:-0.00}
fi

current_time=$(date +"%Y%m%d_%H%M%S")
SAVE_MODEL_NAME="${EXPERIMENT_NAME}-${ADVANTAGE_ESTIMATOR}-ep${EPISODE}-lr${LR}-${current_time}"
LOG_DIR="rft_logs/${EXPERIMENT_NAME}"
mkdir -p "$LOG_DIR" "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}"

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_DEBUG="WARN"
export NCCL_TIMEOUT=3600

echo "========================================="
echo "On-Policy Distillation Training"
echo "  Estimator: $ADVANTAGE_ESTIMATOR"
echo "  Student: $STUDENT_MODEL_PATH"
echo "  Teacher: $TEACHER_URL"
echo "  GPUs:    $GPUS_PER_NODE"
echo "========================================="

set -x

torchrun \
    --nnodes 1 \
    --nproc-per-node $GPUS_PER_NODE \
    --node_rank 0 \
    --master-port ${MASTER_PORT:-20090} \
    --master-addr localhost \
    examples/gsm8k_geo3k/train_colocate.py \
    --pretrain "$STUDENT_MODEL_PATH" \
    --save_trajectories \
    --advantage_estimator "${ADVANTAGE_ESTIMATOR}" \
    --opd_kl_coef ${OPD_KL_COEF} \
    --fsdp \
    --use_kl_loss \
    --flash_attn \
    --engine_type sglang \
    --enable_engine_sleep \
    --rm_use_engine \
    --reward_pretrain "" \
    --teacher_model_url "$TEACHER_URL" \
    ${TASK_REWARD_FLAG} \
    --save_path "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    --ckpt_path "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    --micro_train_batch_size ${MICRO_TRAIN_BS} \
    --train_batch_size ${TBS} \
    --micro_rollout_batch_size ${MICRO_ROLLOUT_BS} \
    --rollout_batch_size ${RBS} \
    --max_epochs 1 \
    --num_episodes ${EPISODE} \
    --lr_warmup_ratio 0.03 \
    --n_samples_per_prompt $N_SAMPLES \
    --prompt_max_len ${PROMPT_MAX_LEN:-1024} \
    --generate_max_len ${GENERATE_MAX_LEN:-2048} \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate $LR \
    --init_kl_coef $KL \
    --kl_estimator "k3" \
    --prompt_data "$DATASET_PATH" \
    --input_key "prompt" \
    --label_key "label" \
    --eval_steps 20 \
    --eval_split "test" \
    --apply_chat_template \
    --gradient_checkpointing \
    --save_steps 20 \
    --max_ckpt_num 3 \
    --engine_mem_util 0.6 \
    --engine_tp_size $ENGINE_TP \
    --l2 1.0e-2 \
    --freeze_prefix \
    --adam_offload \
    --text_only \
    --use_wandb "${WANDB_API_KEY}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${EXPERIMENT_NAME}-${ADVANTAGE_ESTIMATOR}-${current_time}" \
    2>&1 | tee "${LOG_DIR}/train_${current_time}.log"

exit ${PIPESTATUS[0]}
