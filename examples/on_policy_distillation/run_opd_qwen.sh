#!/bin/bash
#
# LightRFT On-Policy Distillation Training Script
# This script demonstrates knowledge distillation from Qwen2.5-7B (teacher) to Qwen2.5-0.5B (student)
# using on-policy distillation during reinforcement learning.
#
# Key Features:
# - No separate reward model needed - teacher model provides the learning signal
# - Token-level supervision from teacher log probabilities
# - On-policy: teacher evaluates student's actual generated responses
#

set -e

################################################################################
#                           Part 1: User Configuration                         #
################################################################################

# --- Model Paths ---
# Teacher model (larger, provides learning signal)
TEACHER_MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"

# Student model (smaller, being trained)
STUDENT_MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"

# --- Dataset Path ---
# Path to your training dataset (JSONL format)
# Each line should be a JSON object with a "prompt" field
DATASET_PATH="/path/to/your/dataset.jsonl"

# --- Teacher Model Server Configuration ---
TEACHER_IP="127.0.0.1"
TEACHER_PORT=13141
TEACHER_GPU=7  # GPU to run teacher model on

# --- Experiment Configuration ---
EXPERIMENT_NAME="opd-qwen-7b-to-0.5b"
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
export WANDB_PROJECT="LightRFT-OnPolicyDistillation"

################################################################################
#                       Part 2: Training Hyperparameters                       #
################################################################################

# --- Distillation Settings ---
N_SAMPLES=4              # Number of samples per prompt
EPISODE=30               # Total number of training episodes
WARMUP=0.03              # Learning rate warmup ratio

# --- Batch Size Configuration ---
RBS=128       # Rollout Batch Size
TBS=128       # Train Batch Size

# --- Learning Settings ---
KL=0.01                  # KL divergence coefficient (for regularization)
LR=1e-6                  # Student learning rate
MAX_LENGTH=3072          # Max sequence length
PROMPT_MAX_LEN=1024      # Max prompt length
GENERATE_MAX_LEN=2048    # Max generation length

################################################################################
#                    Part 3: Distributed Training Setup                        #
################################################################################

# --- Single-Node Setup ---
export MLP_WORKER_NUM=1
export MLP_WORKER_GPU=8
export MLP_ROLE_INDEX=0
export MLP_WORKER_0_HOST="localhost"
export MLP_WORKER_0_PORT=20090

# --- PyTorch Distributed Variables ---
export MASTER_ADDR=$MLP_WORKER_0_HOST
export MASTER_PORT=$MLP_WORKER_0_PORT
export NNODES=$MLP_WORKER_NUM
export NODE_RANK=$MLP_ROLE_INDEX
export GPUS_PER_NODE=$MLP_WORKER_GPU

# --- vLLM Engine Settings ---
ENGINE_TP=2  # Tensor parallelism for inference engine

################################################################################
#                      Part 4: Start Teacher Model Server                      #
################################################################################

echo "========================================="
echo "Starting Teacher Model Server"
echo "========================================="

# Generate unique log file for teacher server
LOG_FILE="/tmp/teacher_model_$(date +%Y%m%d_%H%M%S).log"

# Launch teacher model server in background
CUDA_VISIBLE_DEVICES=$TEACHER_GPU python3 -m sglang.launch_server \
    --model-path "$TEACHER_MODEL_PATH" \
    --host 0.0.0.0 \
    --port $TEACHER_PORT \
    --tp 1 \
    --chunked-prefill-size 4096 \
    --mem-fraction-static 0.6 \
    > "$LOG_FILE" 2>&1 &

TEACHER_PID=$!
echo "Teacher model server starting (PID: $TEACHER_PID)..."
echo "Logs: $LOG_FILE"

# Wait for teacher model server to be ready
MAX_WAIT=300  # Maximum wait time in seconds
WAITED=0
until curl -sf http://$TEACHER_IP:$TEACHER_PORT/health > /dev/null 2>&1; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: Teacher model server failed to start within $MAX_WAIT seconds"
        echo "Last 20 lines of log:"
        tail -n 20 "$LOG_FILE"
        kill $TEACHER_PID 2>/dev/null || true
        exit 1
    fi
    echo "Waiting for teacher model server to start... ($WAITED/$MAX_WAIT seconds)"
    tail -n 5 "$LOG_FILE"
    sleep 5
    WAITED=$((WAITED + 5))
done

echo "✓ Teacher model server is up and running at $TEACHER_IP:$TEACHER_PORT"
sleep 5

################################################################################
#                         Part 5: Training Setup                               #
################################################################################

# --- Generate dynamic names ---
current_time=$(date +"%Y%m%d_%H%M%S")
SAVE_MODEL_NAME="${EXPERIMENT_NAME}-ep${EPISODE}-kl${KL}-lr${LR}-${current_time}"
WANDB_RUN_NAME="${EXPERIMENT_NAME}-${current_time}"

# --- Create directories ---
mkdir -p "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}"
mkdir -p "rft_logs/${EXPERIMENT_NAME}"

# --- Environment optimizations ---
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_DEBUG="WARN"
export IGNORE_EOS=0
export WANDB_MODE="offline"  # Set to "online" for real-time logging

# --- Teacher model URL for distillation ---
TEACHER_URL="http://$TEACHER_IP:$TEACHER_PORT/generate"

set -x

################################################################################
#                         Part 6: Launch Training                              #
################################################################################

echo "========================================="
echo "Starting On-Policy Distillation Training"
echo "========================================="

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up..."
    kill $TEACHER_PID 2>/dev/null || true
    pkill -f "sglang.launch_server" 2>/dev/null || true
    echo "Cleanup complete"
}
trap cleanup EXIT

torchrun \
    --nnodes $NNODES \
    --nproc-per-node $GPUS_PER_NODE \
    --node_rank $NODE_RANK \
    --master-port $MASTER_PORT \
    --master-addr $MASTER_ADDR \
    examples/gsm8k_geo3k/train_colocate.py \
    --pretrain "$STUDENT_MODEL_PATH" \
    --save_trajectories \
    --advantage_estimator "on_policy_distillation" \
    --fsdp \
    --use_kl_loss \
    --flash_attn \
    --rm_use_engine \
    --reward_pretrain "{}" \
    --remote_rm_url "$TEACHER_URL" \
    --save_path "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    --ckpt_path "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    --micro_train_batch_size 4 \
    --train_batch_size ${TBS} \
    --micro_rollout_batch_size 4 \
    --rollout_batch_size ${RBS} \
    --max_epochs 1 \
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
    --max_ckpt_num 3 \
    --max_ckpt_mem 160 \
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --logging_steps 1 \
    --eval_steps -1 \
    --rm_engine_tp $ENGINE_TP

echo "Training complete!"

################################################################################
#                         Part 7: Usage Instructions                           #
################################################################################

: <<'USAGE'
===============================================================================
Usage Instructions
===============================================================================

1. Prerequisites:
   - Install LightRFT and dependencies
   - Install SGLang: pip install sglang
   - Prepare your training dataset in JSONL format

2. Configure the script:
   - Set TEACHER_MODEL_PATH to your teacher model
   - Set STUDENT_MODEL_PATH to your student model
   - Set DATASET_PATH to your training data
   - Adjust hyperparameters as needed

3. Run the script:
   bash examples/on_policy_distillation/run_opd_qwen.sh

4. Monitor training:
   - Check W&B dashboard for training metrics
   - Logs are saved in rft_logs/${EXPERIMENT_NAME}/
   - Checkpoints are saved in results/${EXPERIMENT_NAME}/

5. Key Parameters:
   - N_SAMPLES: Number of responses per prompt (higher = more stable but slower)
   - LR: Learning rate (typically 1e-6 for distillation)
   - KL: KL coefficient for regularization (keeps student close to initialization)
   - EPISODE: Number of training episodes

6. Expected Behavior:
   - Student model should gradually match teacher's probability distribution
   - Training loss should decrease over episodes
   - Student responses should become more similar to teacher's style

7. Troubleshooting:
   - If teacher server fails: Check GPU memory and CUDA availability
   - If training OOMs: Reduce batch sizes or enable gradient checkpointing
   - If convergence is slow: Adjust learning rate or increase N_SAMPLES

===============================================================================
USAGE
