#!/bin/bash
#
# LightRFT Multi-Modal Training Script for the Geo3K Dataset.
# This script is designed for fine-tuning a large multi-modal model using the GRPO algorithm.
#
# Key Feature:
# This training process utilizes a PURE RULE-BASED REWARD mechanism, eliminating the need for a separate reward model.
# The reward is calculated based on two criteria:
# - Format Correctness (10%): Adherence to the required <think>...</think> and \boxed{} format.
# - Answer Accuracy (90%): Correctness of the final answer.
#

################################################################################
#                           Part 1: User Configuration                         #
# Please update the following paths and settings to match your environment.    #
################################################################################

# --- Model and Dataset Paths ---
# Path to the base model. Can be a Hugging Face model name (e.g., "Qwen/Qwen2.5-VL-7B-Instruct")
# or a local directory containing the model files.
PATH_TO_YOUR_BASE_MODEL="/path/to/your/base/model"

# Path to the preprocessed geo3k dataset.
# See "Usage Instructions" at the end of the script for preprocessing steps.
PATH_TO_YOUR_GEO3K_DATASET="/path/to/your/preprocessed/geo3k_dataset"

# --- Experiment and Logging ---
# A descriptive name for your experiment. Used for organizing logs and checkpoints.
EXPERIMENT_NAME="lightrft-geo3k-grpo-training"

# Your Weights & Biases API key.
# Set to an empty string "" if you are not using W&B.
# It is strongly recommended to set this as an environment variable instead of hardcoding.
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
export WANDB_PROJECT="LightRFT-Geo3K-Experiments"


################################################################################
#                       Part 2: Training Hyperparameters                       #
# These settings control the training process. Adjust them as needed.          #
################################################################################

# --- GRPO Settings ---
GROUP_METHOD="normal"
N_SAMPLES=8              # Number of samples per prompt for GRPO (must be > 1).
EPISODE=20               # Total number of training episodes.
WARMUP=0.03              # Learning rate warmup ratio.
RBS=128                  # Rollout Batch Size.
TBS=128                  # Training Batch Size.

# --- Learning and Model Settings ---
KL=0.01                  # KL divergence coefficient.
LR=1e-6                  # Actor learning rate.
MAX_LENGTH=3072          # Max sequence length (prompt + generation).
PROMPT_MAX_LEN=1024      # Max length of the input prompt.
GENERATE_MAX_LEN=2048    # Max length of the generated response.

# --- Multi-modal Settings ---
limit_mm_image_per_prompt=10  # Max number of images per prompt.

# --- Evaluation Settings ---
EVAL_SPLIT="test"             # Dataset split to use for evaluation ("test", "validation").
MAX_EVAL_SAMPLES=700          # Max samples for evaluation to keep it fast.
# Note: hiyouga/geometry3k dataset splits: train (2.1k), validation (300), test (601).


################################################################################
#                    Part 3: Distributed Training Setup                        #
# Configure settings for multi-GPU and multi-node training.                    #
################################################################################

# --- Single-Node Distributed Setup ---
# Update these if you are running in a multi-node environment.
export MLP_WORKER_NUM=1                 # Number of nodes.
export MLP_WORKER_GPU=8                 # Number of GPUs per node.
export MLP_ROLE_INDEX=0                 # Rank of the current node.
export MLP_WORKER_0_HOST="localhost"    # IP address of the master node (node 0).
export MLP_WORKER_0_PORT=20091          # Port for the master node.

# --- PyTorch Distributed Environment Variables ---
export MASTER_ADDR=$MLP_WORKER_0_HOST
export MASTER_PORT=$MLP_WORKER_0_PORT
export NNODES=$MLP_WORKER_NUM
export NODE_RANK=$MLP_ROLE_INDEX
export GPUS_PER_NODE=$MLP_WORKER_GPU

# --- vLLM/SGLang Engine Settings ---
ENGINE_TP=2  # Tensor parallelism size for the inference engine. Adjust based on your model and GPU setup.


################################################################################
#                      Part 4: Execution and Logging                           #
# This section prepares and launches the training command.                     #
################################################################################

# --- Generate dynamic names and paths ---
current_time=$(date +"%Y%m%d_%H%M%S")
SAVE_MODEL_NAME="${EXPERIMENT_NAME}-ep${EPISODE}-kl${KL}-lr${LR}-${current_time}"
WANDB_RUN_NAME="${EXPERIMENT_NAME}-${current_time}"

# --- Create directories for logs and checkpoints ---
mkdir -p "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}"
mkdir -p "rft_logs/${EXPERIMENT_NAME}"

# --- System and Environment Optimizations ---
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_DEBUG="WARN"
export IGNORE_EOS=0
export WANDB_MODE="offline" # Set to "online" for real-time W&B logging.

# --- Set execution verbosity ---
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
    examples/safework_t1/train_colocate.py \
    --pretrain "${PATH_TO_YOUR_BASE_MODEL}" \
    --save_trajectories \
    --print_replay_buffer_stats \
    --loss_agg_mode "seq-mean-token-mean" \
    --fsdp \
    --rm_use_engine \
    --mixed_mm_data \
    --reward_pretrain "{}" \
    --save_path "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    --ckpt_path "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    --micro_train_batch_size 4 \
    --train_batch_size ${TBS} \
    --micro_rollout_batch_size 8 \
    --rollout_batch_size ${RBS} \
    --advantage_estimator "group_norm" \
    --max_epochs 1 \
    --num_episodes ${EPISODE} \
    --lr_warmup_ratio ${WARMUP} \
    --n_samples_per_prompt $N_SAMPLES \
    --prompt_max_len $PROMPT_MAX_LEN \
    --generate_max_len $GENERATE_MAX_LEN \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate $LR \
    --use_kl_loss \
    --init_kl_coef $KL \
    --kl_estimator "k3" \
    --prompt_data "${PATH_TO_YOUR_GEO3K_DATASET}" \
    --input_key "prompt" \
    --images_key "images" \
    --label_key "label" \
    --eval_split "${EVAL_SPLIT}" \
    --apply_chat_template \
    --flash_attn \
    --gradient_checkpointing \
    --save_steps 20 \
    --max_ckpt_num 2 \
    --engine_mem_util 0.6 \
    --engine_tp_size $ENGINE_TP \
    --enable_engine_sleep \
    --system_prompt 'A conversation between the User and Assistant. The User asks a question, and the Assistant provides a solution. The Assistant first thinks through the reasoning process internally with self-reflection and consistency check and then gives the final analysis and answer. The reasoning process should be enclosed within <think></think>, followed directly by the final thought and answer, the final answer MUST BE put in \\boxed{}, like this: <think> reasoning process here </think> final thought and \\boxed{answer} here.' \
    --l2 1.0e-2 \
    --freeze_prefix \
    --adam_offload \
    --limit_mm_image_per_prompt $limit_mm_image_per_prompt \
    --use_wandb "${WANDB_API_KEY}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    2>&1 | tee "rft_logs/${EXPERIMENT_NAME}/node${NODE_RANK}_${current_time}.log"


################################################################################
#                           Usage Instructions                                 #
#                                                                              #
# Step 1: Preprocess the Geo3K Dataset                                         #
#   Run the provided preprocessing script to prepare the dataset.              #
#   Make sure the output directory matches `PATH_TO_YOUR_GEO3K_DATASET`.       #
#                                                                              #
#   `python ./examples/data_preprocess/geo3k.py --local_save_dir /path/to/your/preprocessed/geo3k_dataset`
#                                                                              #
# Step 2: Configure the Script                                                 #
#   Edit "Part 1: User Configuration" at the top of this file. Set the paths   #
#   to your base model and the preprocessed dataset.                           #
#                                                                              #
# Step 3: Run the Training Script                                              #
#   Execute this script from your shell.                                       #
#                                                                              #
#   `bash /path/to/this/script.sh`                                             #
#                                                                              #
################################################################################