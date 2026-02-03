#!/bin/bash
#
# LightRFT Multi-Modal PPO Training Script for the Geo3K Dataset.
# This script is designed for fine-tuning a large multi-modal model using the PPO algorithm.
#
# Key Feature:
# This training process utilizes a PURE RULE-BASED REWARD mechanism.
# - Format Correctness (10%): Adherence to <think>...</think> and \boxed{} format.
# - Answer Accuracy (90%): Correctness of the final answer.
#
# PPO vs GRPO Note:
# Unlike GRPO, this PPO setup uses GAE (Generalized Advantage Estimation), requires a 
# Critic model, and treats KL divergence as a reward penalty rather than a loss term.
#

################################################################################
#                           Part 1: User Configuration                         #
# Please update the following paths and settings to match your environment.    #
################################################################################

# --- Model and Dataset Paths ---
# Path to the base model (Actor). Can be a Hugging Face model name or local path.
# e.g., "Qwen/Qwen2.5-VL-7B-Instruct"
PATH_TO_YOUR_BASE_MODEL="/path/to/your/base/model"

# Path to the preprocessed geo3k dataset.
PATH_TO_YOUR_GEO3K_DATASET="/path/to/your/preprocessed/geo3k_dataset"

# --- Critic Model Configuration ---
# PPO requires a Critic model for value estimation.
# Usually initialized from the base model (Actor) for the first run.
PATH_TO_YOUR_CRITIC_MODEL="${PATH_TO_YOUR_BASE_MODEL}"

# --- Experiment and Logging ---
EXPERIMENT_NAME="lightrft-geo3k-ppo-training"

# Your Weights & Biases API key.
# Set to an empty string "" if you are not using W&B.
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
export WANDB_PROJECT="LightRFT-Geo3K-PPO"


################################################################################
#                       Part 2: Training Hyperparameters                       #
# These settings control the training process. Adjust them as needed.          #
################################################################################

# --- PPO Sampling Settings ---
N_SAMPLES=2              # Number of samples per prompt (PPO typically uses 1-2).
EPISODE=30               # Total number of training episodes.
WARMUP=0.03              # Learning rate warmup ratio.
RBS=64                   # Rollout Batch Size (prompts per rollout).
TBS=$((RBS * N_SAMPLES)) # Training Batch Size (calculated automatically).

# --- Learning Rates ---
ACTOR_LR=1e-6            # Actor learning rate.
CRITIC_LR=9e-6           # Critic learning rate (typically higher than actor).

# --- PPO Specific Hyperparameters ---
EPS_CLIP=0.2             # PPO policy clip range.
VALUE_CLIP=0.2           # PPO value clip range.
GAMMA=1.0                # GAE gamma (discount factor).
LAMBD=0.95               # GAE lambda (TD lambda).
KL_COEF=0.01             # KL coefficient (used as reward penalty).

# --- Model Constraints ---
MAX_LENGTH=3072          # Max sequence length (prompt + generation).
PROMPT_MAX_LEN=1024      # Max length of the input prompt.
GENERATE_MAX_LEN=2048    # Max length of the generated response.

# --- Multi-modal Settings ---
limit_mm_image_per_prompt=10  # Max number of images per prompt.

# --- Evaluation Settings ---
EVAL_SPLIT="test"             # Dataset split to use for evaluation.
MAX_EVAL_SAMPLES=700          # Max samples for evaluation.


################################################################################
#                    Part 3: Distributed Training Setup                        #
# Configure settings for multi-GPU and multi-node training.                    #
################################################################################

# --- Single-Node Distributed Setup ---
export MLP_WORKER_NUM=1                 # Number of nodes.
export MLP_WORKER_GPU=8                 # Number of GPUs per node.
export MLP_ROLE_INDEX=0                 # Rank of the current node.
export MLP_WORKER_0_HOST="localhost"    # IP address of the master node.
export MLP_WORKER_0_PORT=20091          # Port for the master node.

# --- PyTorch Distributed Environment Variables ---
export MASTER_ADDR=$MLP_WORKER_0_HOST
export MASTER_PORT=$MLP_WORKER_0_PORT
export NNODES=$MLP_WORKER_NUM
export NODE_RANK=$MLP_ROLE_INDEX
export GPUS_PER_NODE=$MLP_WORKER_GPU

# --- vLLM/SGLang Engine Settings ---
ENGINE_TP=2  # Tensor parallelism size for the inference engine.


################################################################################
#                      Part 4: Execution and Logging                           #
# This section prepares and launches the training command.                     #
################################################################################

# --- Generate dynamic names and paths ---
current_time=$(date +"%Y%m%d_%H%M%S")
SAVE_MODEL_NAME="${EXPERIMENT_NAME}-ep${EPISODE}-kl${KL_COEF}-lr${ACTOR_LR}-${current_time}"
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

# Note: We use '{}' for reward_pretrain because we use a rule-based reward.
# We explicitly set --advantage_estimator to "gae" for PPO.

torchrun \
    --nnodes $NNODES \
    --nproc-per-node $GPUS_PER_NODE \
    --node_rank $NODE_RANK \
    --master-port $MASTER_PORT \
    --master-addr $MASTER_ADDR \
    examples/openrlhf_v/train_grpo_rm_colocate.py \
    --pretrain "${PATH_TO_YOUR_BASE_MODEL}" \
    --critic_pretrain "${PATH_TO_YOUR_CRITIC_MODEL}" \
    --reward_pretrain "{}" \
    --save_trajectories \
    --fsdp \
    --mixed_mm_data \
    --rm_use_engine \
    --save_path "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    --ckpt_path "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    --micro_train_batch_size 4 \
    --train_batch_size ${TBS} \
    --micro_rollout_batch_size 4 \
    --rollout_batch_size ${RBS} \
    --advantage_estimator "gae" \
    --max_epochs 1 \
    --num_episodes ${EPISODE} \
    --lr_warmup_ratio ${WARMUP} \
    --n_samples_per_prompt $N_SAMPLES \
    --prompt_max_len $PROMPT_MAX_LEN \
    --generate_max_len $GENERATE_MAX_LEN \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate $ACTOR_LR \
    --critic_learning_rate $CRITIC_LR \
    --init_kl_coef $KL_COEF \
    --kl_estimator "k1" \
    --eps_clip $EPS_CLIP \
    --value_clip $VALUE_CLIP \
    --gamma $GAMMA \
    --lambd $LAMBD \
    --prompt_data "${PATH_TO_YOUR_GEO3K_DATASET}" \
    --input_key "prompt" \
    --images_key "images" \
    --label_key "label" \
    --eval_split "${EVAL_SPLIT}" \
    --max_eval_samples $MAX_EVAL_SAMPLES \
    --apply_chat_template \
    --flash_attn \
    --gradient_checkpointing \
    --save_steps 20 \
    --max_ckpt_num 3 \
    --engine_mem_util 0.6 \
    --engine_tp_size $ENGINE_TP \
    --enable_engine_sleep \
    --system_prompt 'A conversation between the User and Assistant. The User asks a question, and the Assistant provides a solution. The Assistant first thinks through the reasoning process internally with self-reflection and consistency check and then gives the final analysis and answer. The reasoning process should be enclosed within <think></think>, followed directly by the final thought and answer, the final answer MUST BE put in \\boxed{final answer}, like this: <think> reasoning process here </think> final thought and \\boxed{final answer} here.' \
    --advantages_norm \
    --l2 1.0e-2 \
    --freeze_prefix \
    --fsdp_cpu_offload \
    --limit_mm_image_per_prompt $limit_mm_image_per_prompt \
    --use_wandb "${WANDB_API_KEY}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    2>&1 | tee "rft_logs/${EXPERIMENT_NAME}/node${NODE_RANK}_${current_time}.log"


################################################################################
#                           Usage Instructions                                 #
#                                                                              #
# Step 1: Preprocess the Geo3K Dataset                                         #
#   Ensure your dataset is preprocessed and points to `PATH_TO_YOUR_GEO3K_DATASET`.
#   `python examples/data_preprocess/geo3k.py --local_save_dir /path/to/dataset`
#                                                                              #
# Step 2: Configure the Script                                                 #
#   Edit "Part 1: User Configuration" to set your Model and Dataset paths.     #
#   Ensure `PATH_TO_YOUR_CRITIC_MODEL` is set (usually same as base model).    #
#                                                                              #
# Step 3: Run the Training Script                                              #
#   `bash /path/to/this/script.sh`                                             #
#                                                                              #
################################################################################