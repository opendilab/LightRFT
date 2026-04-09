#!/bin/bash
#
# LightRFT Training Script for R1-AQA (Audio Question Answering)
#
# This script fine-tunes Qwen2-Audio-7B-Instruct using the GRPO algorithm
# with rule-based rewards, faithfully migrating the R1-AQA training pipeline.
#
#
# Migration from R1-AQA:
#   R1-AQA num_generations=8      → N_SAMPLES=8
#   R1-AQA temperature=1.0        → TEMPERATURE=1.0
#   R1-AQA max_prompt_length=512  → PROMPT_MAX_LEN=512
#   R1-AQA per_device_batch=1     → MICRO_TRAIN=1
#   R1-AQA grad_accum=2           → TBS adjusted
#   R1-AQA DeepSpeed ZeRO3        → --zero_stage 3
#

################################################################################
#                           Part 1: User Configuration                         #
################################################################################

# --- Model and Dataset Paths ---
# Qwen2-Audio-7B-Instruct base model
PATH_TO_YOUR_BASE_MODEL=""

# Path to the preprocessed AVQA dataset (output of data_preprocess/avqa.py)
PATH_TO_YOUR_AVQA_DATASET=""

# --- Experiment and Logging ---
EXPERIMENT_NAME="lightrft-r1-aqa-grpo-training"

export WANDB_API_KEY=""
export WANDB_PROJECT="LightRFT-R1-AQA"


################################################################################
#                       Part 2: Training Hyperparameters                       #
#                                                                              #
# These match R1-AQA's defaults as closely as possible.                        #
# See README.md for the full mapping from R1-AQA to LightRFT parameters.      #
################################################################################

# --- GRPO Settings (from R1-AQA) ---
GROUP_METHOD="normal"
N_SAMPLES=4              # num_generations reduced from 8→4 to save memory
EPISODE=10               # Number of training episodes
WARMUP=0.03              # Learning rate warmup ratio
TEMPERATURE=1.0          # Sampling temperature (R1-AQA default: 1.0)

# --- Batch Size Configuration ---
# Constraint: train_batch_size >= rollout_batch_size * n_samples_per_prompt
# Reduced for single-GPU memory constraints (140 GiB GPU).
RBS=4                    # Rollout Batch Size (reduced from 128 to fit in memory)
TBS=16                   # Train Batch Size (RBS * N_SAMPLES = 4 * 4 = 16)
MICRO_ROLLOUT=1          # Micro rollout batch size per GPU (reduced from 2)
MICRO_TRAIN=1            # Micro train batch size per GPU (R1-AQA: per_device=1)

# --- Learning and Model Settings ---
KL=0.01                  # KL divergence coefficient
LR=1e-6                  # Actor learning rate
PROMPT_MAX_LEN=512        # Max prompt length (reduced from 1024 to save memory)
GENERATE_MAX_LEN=1024     # Max generation length (reduced from 2048 to save memory)
MAX_LENGTH=1536          # Total max sequence length (512 + 1024)

# --- Evaluation Settings ---
EVAL_STEPS=50            # Evaluate every N steps
SAVE_STEPS=50            # Save checkpoint every N steps


################################################################################
#                    Part 3: Distributed Training Setup                        #
################################################################################

# --- Single-Node Setup ---
export MLP_WORKER_NUM=1
export MLP_WORKER_GPU=1                 # Number of GPUs per node
export MLP_ROLE_INDEX=0
export MLP_WORKER_0_HOST="localhost"
export MLP_WORKER_0_PORT=20092

# --- PyTorch Distributed ---
export MASTER_ADDR=$MLP_WORKER_0_HOST
export MASTER_PORT=$MLP_WORKER_0_PORT
export NNODES=$MLP_WORKER_NUM
export NODE_RANK=$MLP_ROLE_INDEX
export GPUS_PER_NODE=$MLP_WORKER_GPU

# --- Inference Engine ---
ENGINE_TP=1              # Tensor parallelism for inference engine
ENGINE_MEM_UTIL=0.3      # Memory utilization for inference engine (reduced from 0.6)

# --- Checkpoint local path (use if NFS causes OSError) ---
CKPT_PATH_LOCAL=""


################################################################################
#                      Part 4: Execution and Logging                           #
################################################################################

current_time=$(date +"%Y%m%d_%H%M%S")
SAVE_MODEL_NAME="${EXPERIMENT_NAME}-ep${EPISODE}-kl${KL}-lr${LR}-${current_time}"
WANDB_RUN_NAME="${EXPERIMENT_NAME}-${current_time}"

mkdir -p "./results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}"
mkdir -p "./rft_logs/${EXPERIMENT_NAME}"

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_DEBUG="WARN"
export IGNORE_EOS=0
export WANDB_MODE="offline"
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -x


################################################################################
#                         Part 5: Main Training Command                        #
################################################################################

torchrun \
    --nnodes $NNODES \
    --nproc-per-node $GPUS_PER_NODE \
    --node_rank $NODE_RANK \
    --master-port $MASTER_PORT \
    --master-addr $MASTER_ADDR \
    examples/r1_aqa/train_colocate.py \
    --pretrain "${PATH_TO_YOUR_BASE_MODEL}" \
    --save_trajectories \
    --advantage_estimator "group_norm" \
    --fsdp \
    --use_kl_loss \
    --flash_attn \
    --rm_use_engine \
    --reward_pretrain "{}" \
    --save_path "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    --ckpt_path "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    $( [ -n "${CKPT_PATH_LOCAL}" ] && echo "--ckpt_path_local ${CKPT_PATH_LOCAL}" ) \
    --micro_train_batch_size ${MICRO_TRAIN} \
    --train_batch_size ${TBS} \
    --micro_rollout_batch_size ${MICRO_ROLLOUT} \
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
    --temperature $TEMPERATURE \
    --prompt_data "${PATH_TO_YOUR_AVQA_DATASET}" \
    --input_key "prompt" \
    --label_key "label" \
    --reference_key "reference" \
    --eval_steps ${EVAL_STEPS} \
    --gradient_checkpointing \
    --save_steps ${SAVE_STEPS} \
    --max_ckpt_num 3 \
    --engine_type vllm \
    --engine_mem_util ${ENGINE_MEM_UTIL} \
    --engine_tp_size $ENGINE_TP \
    --enable_engine_sleep \
    --l2 1.0e-2 \
    --adam_offload \
    --use_wandb "${WANDB_API_KEY}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    2>&1 | tee "rft_logs/${EXPERIMENT_NAME}/node${NODE_RANK}_${current_time}.log"


################################################################################
#                           Usage Instructions                                 #
#                                                                              #
# Step 1: Preprocess the AVQA Dataset                                          #
#   Convert R1-AQA's JSONL to LightRFT parquet:                               #
#                                                                              #
#   python examples/r1_aqa/data_preprocess/avqa.py \                           #
#       --input_jsonl data/AVQA/train_qa.data \                                #
#       --audio_dir data/AVQA/audios \                                         #
#       --local_save_dir /path/to/preprocessed/avqa_lightrft                   #
#                                                                              #
# Step 2: Configure the Script                                                 #
#   Edit "Part 1: User Configuration" above:                                   #
#   - Set PATH_TO_YOUR_BASE_MODEL (Qwen2-Audio-7B-Instruct)                   #
#   - Set PATH_TO_YOUR_AVQA_DATASET                                            #
#   - Set GPU count in MLP_WORKER_GPU                                          #
#                                                                              #
# Step 3: Run Training                                                         #
#   bash examples/r1_aqa/run_grpo_r1_aqa_qwen2_audio_7b.sh                    #
#                                                                              #
# Step 4: Evaluate on MMAU Test-Mini                                           #
#   python examples/r1_aqa/eval_mmau.py \                                      #
#       --model_path results/lightrft-r1-aqa-grpo-training/... \               #
#       --data_file /path/to/mmau-test-mini.json \                             #
#       --audio_dir /path/to/mmau/audio \                                      #
#       --out_file results/res_mmau_mini.json                                  #
#                                                                              #
#   python /path/to/mmau/evaluation.py --input results/res_mmau_mini.json      #
#                                                                              #
# Notes:                                                                       #
# - This uses the AUDIO pipeline (not VL). Audio is processed via Qwen2-Audio. #
# - TBS must >= RBS * N_SAMPLES for GRPO constraint.                           #
# - For dry-run: set EPISODE=1, RBS=4, TBS=32, N_SAMPLES=4.                   #
# - For 1-GPU: set MLP_WORKER_GPU=1, ENGINE_TP=1, ENGINE_MEM_UTIL=0.5.        #
################################################################################
