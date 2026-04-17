#!/bin/bash
#
# LightRFT On-Policy Distillation Training Script (Template)
# Knowledge distillation from a teacher model to a student model.
#
# Features:
#   - Auto GPU detection and allocation (teacher + training)
#   - Robust teacher server with health monitoring
#
# Usage:
#   # Edit paths below, then:
#   bash examples/on_policy_distillation/run_opd_qwen.sh
#   USE_TASK_REWARD=true bash examples/on_policy_distillation/run_opd_qwen.sh
#

set -euo pipefail

################################################################################
#                           Part 1: User Configuration                         #
################################################################################

# --- Model Paths (EDIT THESE) ---
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
DATASET_PATH="${DATASET_PATH:-/path/to/your/dataset.jsonl}"

# --- Experiment ---
EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-qwen-7b-to-0.5b}"
export WANDB_API_KEY="${WANDB_API_KEY:-YOUR_WANDB_API_KEY}"
export WANDB_PROJECT="${WANDB_PROJECT:-LightRFT-OnPolicyDistillation}"

# --- Teacher Server ---
TEACHER_IP="127.0.0.1"
TEACHER_PORT=${TEACHER_PORT:-13141}

################################################################################
#                       Part 2: Auto GPU Detection & Allocation                #
################################################################################

TOTAL_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$TOTAL_GPUS" -lt 2 ]; then
    echo "ERROR: Need at least 2 GPUs (1 teacher + 1 training). Found: $TOTAL_GPUS"
    exit 1
fi

# Last GPU for teacher, rest for training
TEACHER_GPU=$((TOTAL_GPUS - 1))
TRAIN_GPUS=$((TOTAL_GPUS - 1))

if [ "$TRAIN_GPUS" -ge 2 ]; then
    ENGINE_TP=2
else
    ENGINE_TP=1
fi

export NNODES=1
export GPUS_PER_NODE=$TRAIN_GPUS
export NODE_RANK=0
export MASTER_ADDR="localhost"
export MASTER_PORT=${MASTER_PORT:-20090}

echo "GPU Allocation: ${TOTAL_GPUS} total → Teacher: GPU ${TEACHER_GPU}, Training: GPU 0-$((TRAIN_GPUS-1)) (TP=${ENGINE_TP})"

################################################################################
#                       Part 3: Training Hyperparameters                       #
################################################################################

# --- Mode control (override via env) ---
#   USE_TASK_REWARD=false  - Pure distillation: rewards=0, only OPD KL signal
#   USE_TASK_REWARD=true   - GRPO task rewards + OPD KL penalty
ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-on_policy_distillation}"
USE_TASK_REWARD="${USE_TASK_REWARD:-false}"

N_SAMPLES=${N_SAMPLES:-8}
EPISODE=${EPISODE:-30}
WARMUP=${WARMUP:-0.03}
OPD_KL_COEF=${OPD_KL_COEF:-1.0}
MICRO_TRAIN_BS=${MICRO_TRAIN_BS:-4}
MICRO_ROLLOUT_BS=${MICRO_ROLLOUT_BS:-4}

# Auto-align batch sizes to (micro_batch * world_size)
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
ALIGN=$((MICRO_TRAIN_BS * WORLD_SIZE))
RBS=$(( (${RBS:-128} / ALIGN) * ALIGN ))
TBS=$(( (${TBS:-128} / ALIGN) * ALIGN ))
[ "$RBS" -lt "$ALIGN" ] && RBS=$ALIGN
[ "$TBS" -lt "$ALIGN" ] && TBS=$ALIGN

echo "Batch sizes: TBS=${TBS}, RBS=${RBS} (aligned to micro=${MICRO_TRAIN_BS} * world=${WORLD_SIZE} = ${ALIGN})"

if [ "$USE_TASK_REWARD" = "true" ]; then
    TASK_REWARD_FLAG="--use_task_reward"
else
    TASK_REWARD_FLAG="--no_task_reward"
fi

if [ "$USE_TASK_REWARD" = "true" ]; then
    KL=${KL:-0.01}
    LR=${LR:-5e-7}
else
    KL=${KL:-0.00}
    LR=${LR:-5e-7}
fi

PROMPT_MAX_LEN=${PROMPT_MAX_LEN:-1024}
GENERATE_MAX_LEN=${GENERATE_MAX_LEN:-2048}

################################################################################
#                      Part 4: Teacher Model Server                            #
################################################################################

TEACHER_PID=""
LOG_DIR="rft_logs/${EXPERIMENT_NAME}"
mkdir -p "$LOG_DIR"
TEACHER_LOG="${LOG_DIR}/teacher_model_$(date +%Y%m%d_%H%M%S).log"

cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    if [ -n "$TEACHER_PID" ] && kill -0 "$TEACHER_PID" 2>/dev/null; then
        echo "Stopping teacher server (PID: $TEACHER_PID)..."
        kill "$TEACHER_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$TEACHER_PID" 2>/dev/null || true
    fi
    pkill -f "sglang.launch_server.*${TEACHER_PORT}" 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

start_teacher_server() {
    if lsof -Pi :"$TEACHER_PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "Port $TEACHER_PORT in use, killing existing process..."
        lsof -ti:"$TEACHER_PORT" | xargs kill -9 2>/dev/null || true
        sleep 3
    fi

    echo "Starting teacher server on GPU $TEACHER_GPU..."
    CUDA_VISIBLE_DEVICES=$TEACHER_GPU python3 -m sglang.launch_server \
        --model-path "$TEACHER_MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$TEACHER_PORT" \
        --tp 1 \
        --chunked-prefill-size 4096 \
        --mem-fraction-static 0.7 \
        --disable-radix-cache \
        --request-timeout 300 \
        --max-running-requests 64 \
        >> "$TEACHER_LOG" 2>&1 &

    TEACHER_PID=$!
    echo "Teacher PID: $TEACHER_PID, Log: $TEACHER_LOG"

    local max_wait=600 waited=0
    while ! curl -sf "http://$TEACHER_IP:$TEACHER_PORT/health" >/dev/null 2>&1; do
        if [ $waited -ge $max_wait ]; then
            echo "ERROR: Teacher server failed to start in ${max_wait}s"
            tail -30 "$TEACHER_LOG"
            exit 1
        fi
        if ! kill -0 "$TEACHER_PID" 2>/dev/null; then
            echo "ERROR: Teacher server process died"
            tail -30 "$TEACHER_LOG"
            exit 1
        fi
        printf "."
        sleep 5
        waited=$((waited + 5))
    done
    echo ""
    echo "Teacher server ready at $TEACHER_IP:$TEACHER_PORT"
}

# Validate model paths
for p in "$TEACHER_MODEL_PATH" "$STUDENT_MODEL_PATH"; do
    [ -e "$p" ] || { echo "ERROR: Path not found: $p"; exit 1; }
done

start_teacher_server
sleep 3

################################################################################
#                         Part 5: Launch Training                              #
################################################################################

current_time=$(date +"%Y%m%d_%H%M%S")
SAVE_MODEL_NAME="${EXPERIMENT_NAME}-${ADVANTAGE_ESTIMATOR}-ep${EPISODE}-kl${KL}-lr${LR}-${current_time}"
WANDB_RUN_NAME="${EXPERIMENT_NAME}-${ADVANTAGE_ESTIMATOR}-${current_time}"
TEACHER_URL="http://$TEACHER_IP:$TEACHER_PORT/generate"

mkdir -p "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}"

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_DEBUG="WARN"
export NCCL_TIMEOUT=3600
export IGNORE_EOS=0
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "========================================="
echo "On-Policy Distillation Training"
echo "========================================="
echo "Estimator: $ADVANTAGE_ESTIMATOR"
echo "Student: $STUDENT_MODEL_PATH"
echo "Teacher: $TEACHER_URL"
echo "GPUs:    Training=0-$((TRAIN_GPUS-1)), Teacher=$TEACHER_GPU"
echo "========================================="

set -x

torchrun \
    --nnodes $NNODES \
    --nproc-per-node $GPUS_PER_NODE \
    --node_rank $NODE_RANK \
    --master-port $MASTER_PORT \
    --master-addr $MASTER_ADDR \
    examples/gsm8k_geo3k/train_colocate.py \
    --pretrain "$STUDENT_MODEL_PATH" \
    --save_trajectories \
    --advantage_estimator "${ADVANTAGE_ESTIMATOR}" \
    --opd_kl_coef ${OPD_KL_COEF} \
    --fsdp \
    --use_kl_loss \
    --flash_attn \
    --engine_type sglang \
    --reward_pretrain "" \
    --teacher_model_url "$TEACHER_URL" \
    ${TASK_REWARD_FLAG} \
    --save_path "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    --ckpt_path "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    --micro_train_batch_size ${MICRO_TRAIN_BS} \
    --train_batch_size ${TBS} \
    --micro_rollout_batch_size ${MICRO_ROLLOUT_BS} \
    --rollout_batch_size ${RBS} \
    --num_episodes ${EPISODE} \
    --lr_warmup_ratio ${WARMUP} \
    --n_samples_per_prompt $N_SAMPLES \
    --prompt_max_len $PROMPT_MAX_LEN \
    --generate_max_len $GENERATE_MAX_LEN \
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
    --wandb_run_name "${WANDB_RUN_NAME}" \
    2>&1 | tee "${LOG_DIR}/node${NODE_RANK}_${current_time}.log"

TRAINING_EXIT_CODE=${PIPESTATUS[0]}
set +x

echo ""
echo "========================================="
echo "Training Complete (exit: $TRAINING_EXIT_CODE)"
echo "Model: results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}"
echo "========================================="

exit $TRAINING_EXIT_CODE
