#!/bin/bash

# --- 用户配置区域 ---
# 1. 设置一个您有权限写入的目录作为所有输出的根目录
#    请确保这个目录存在且可写，例如 /root/my_LightRFT_outputs
# WRITABLE_BASE_DIR="/root/my_LightRFT_outputs"
# WRITABLE_BASE_DIR="/mnt/shared-storage-user/rft_outputs"
WRITABLE_BASE_DIR="/mnt/shared-storage-user/puyuan/code/code_refactor/LightRFT/"

# ==============================================================================
# III. EXPERIMENT HYPERPARAMETERS
# ==============================================================================
# Core settings that control the training process and model architecture.

# --- RL Training Parameters ---
N_SAMPLES=8          # Number of responses to generate for each prompt.
EPISODE=3            # Total number of training episodes.
WARMUP=0.03          # Learning rate warmup ratio.
KL=0             # Initial coefficient for the KL-divergence penalty term.
# KL=0.001             # Initial coefficient for the KL-divergence penalty term.

LR=1e-6              # Learning rate for the actor model.
MAX_LENGTH=8192      # Maximum sequence length for prompts and generations.

# --- Batch Sizes ---
# TBS=64               # Total training batch size across all GPUs.
# RBS=128              # Total rollout batch size for generating experiences.

# TODO ==========
# TBS=32               # Total training batch size across all GPUs.
# RBS=64              # Total rollout batch size for generating experiences.

# TODO ==========
TBS=24               # Total training batch size across all GPUs.
RBS=48              # Total rollout batch size for generating experiences.


# ==============================================================================
# IV. FILE PATHS & MODEL LOCATIONS
# ==============================================================================
# Configure paths to datasets, pretrained models, and reward models.

# --- Dataset ---
DATA_PATH="/mnt/shared-storage-user/puyuan/rft_20250828/dataset/svki_text_20250722"

# --- Base Model ---
PRETRAIN_PATH="/mnt/shared-storage-user/puyuan/rft_20250828/base_model_deepseek_after_sft_20250828" # Deepseek-r1-distill-llama70B
# --- Inference Engine Parallelism ---
# Tensor Parallelism (TP) size for the vLLM/SGLang inference engine.
# Adjust based on the model size.
ENGINE_TP=8          # For a 72B model
limit_mm_image_per_prompt=0  # multi-modal model
NAME="ds-uni-1221"

# ====================================The following is only for debug====================================
# PRETRAIN_PATH="/mnt/shared-storage-user/puyuan/rft_20250828/base_model_after_kg_20250905" # 在kg上训练后的qwen-vl 7b
# ENGINE_TP=1        # Example for a 7B model
# limit_mm_image_per_prompt=0  # multi-modal model
# NAME="ds-qwen-uni-1221"


# Path to the initial weights of the actor model to be trained.

# --- Reward Models ---
# A JSON-formatted string specifying paths to different pretrained reward models.
# The training script uses multiple reward models for different aspects (e.g., safety, value).
# svkng
REWARD_PRETRAIN_PATHS='{"safety":"/mnt/shared-storage-user/puyuan/rft_20250828/reward_model_20250828/safe_orm/","value":"/mnt/shared-storage-user/puyuan/rft_20250828/reward_model_20250828/value_orm/","knowledge":"/mnt/shared-storage-user/puyuan/rft_20250828/reward_model_20250828/knowledge_orm/","normal":"/mnt/shared-storage-user/puyuan/rft_20250828/reward_model_20250828/knowledge_orm/","general":"/mnt/shared-storage-user/puyuan/rft_20250828/reward_model_20250828/knowledge_orm/"}'

# only for debug
# vkng
# REWARD_PRETRAIN_PATHS='{"value":"/mnt/shared-storage-user/puyuan/rft_20250828/reward_model_20250828/value_orm/","knowledge":"/mnt/shared-storage-user/puyuan/rft_20250828/reward_model_20250828/knowledge_orm/","normal":"/mnt/shared-storage-user/puyuan/rft_20250828/reward_model_20250828/knowledge_orm/","general":"/mnt/shared-storage-user/puyuan/rft_20250828/reward_model_20250828/knowledge_orm/"}'
# v
# REWARD_PRETRAIN_PATHS='{"value":"/mnt/shared-storage-user/puyuan/rft_20250828/reward_model_20250828/value_orm/"}'
# REWARD_PRETRAIN_PATHS='{}'


# ====================================The following is only for debug====================================
# ENGINE_TP=1  # vLLM/SGLang, for 7b base model
# PRETRAIN_PATH="/mnt/shared-storage-user/puyuan/rft_20250828/base_model_after_kg_20250828" # 在kg上训练后的qwen-vl 7b
# PRETRAIN_PATH="/fs-computility/shaowenqi/shared/dingyizhuo/ckpt/sft_7b_0522" # 在sft后的qwen-vl 7b

# ==============================================================================
# V. LOGGING & OUTPUT CONFIGURATION
# ==============================================================================
# Settings for saving checkpoints, logs, and experiment tracking.

# --- Experiment Naming and Directories ---
current_time=$(date +"%m%d%H%M")
SAVE_MODEL_NAME="LightRFT-len_${MAX_LENGTH}-tbs_${TBS}-rbs_${RBS}-sample_${N_SAMPLES}-kl_${KL}-warmup_${WARMUP}-ep_${EPISODE}-plr_${LR}-rm-colocate-kg-${current_time}"

# --- 新增：为 wandb 配置一个可写的离线日志目录 ---
# 我们在可写目录下创建一个专门用于存放 wandb 离线数据的子目录
WANDB_OFFLINE_DIR="${WRITABLE_BASE_DIR}/wandb_offline_logs"
mkdir -p "${WANDB_OFFLINE_DIR}"
# 使用 WANDB_DIR 环境变量告诉 wandb 将所有数据写入此目录
export WANDB_DIR="${WANDB_OFFLINE_DIR}"
# --- 修改结束 ---

SAVE_PATH="${WRITABLE_BASE_DIR}/results/$NAME/${SAVE_MODEL_NAME}"
LOG_DIR="${WRITABLE_BASE_DIR}/rft_logs/$NAME"


mkdir -p "${SAVE_PATH}"
mkdir -p "${LOG_DIR}"


export WANDB_MODE="offline" # TODO
# --- Weights & Biases (W&B) Logging ---
# It's recommended to set this as an environment variable rather than hardcoding.
# export WANDB_API_KEY="YOUR_WANDB_API_KEY" # Replace with your key
export WANDB_API_KEY="968275bc822c87ac741ecce2f06cdfb54dbc1608" # Replace with your key

WANDB_PROJECT="Deepseek-r1-distill-llama70B-MultiORM-RL"
WANDB_RUN_NAME="Deepseek-r1-distill-llama70B-svki-grpo-${current_time}"


# ==============================================================================
# VI. DISTRIBUTED ENVIRONMENT SETUP (Volcengine)
# ==============================================================================
# These environment variables are specific to the Volcengine MLP platform and are
# used by torchrun to initialize the distributed process group.

# This may help reduce memory usage in some distributed setups.
export TORCH_NCCL_AVOID_RECORD_STREAMS=1


# ==============================================================================
# OOM FIX: PyTorch CUDA Memory Management Optimizations
# ==============================================================================
# Reduce memory fragmentation by using expandable segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MLP_WORKER_NUM=1
export MLP_WORKER_GPU=8
export MLP_ROLE_INDEX=0
export MLP_WORKER_0_PORT=20091

# export MLP_WORKER_0_HOST=10.102.97.181 # TODO:需要根据node的实际IP进行修改
export MLP_WORKER_0_HOST=localhost

# --- Process Group Initialization ---
export MASTER_ADDR=$MLP_WORKER_0_HOST      # IP address of the master node
export NNODES=$MLP_WORKER_NUM              # Total number of nodes
export NODE_RANK=$MLP_ROLE_INDEX           # Rank of the current node (0 to NNODES-1)
export GPUS_PER_NODE=$MLP_WORKER_GPU       # Number of GPUs per node
export MASTER_PORT=$MLP_WORKER_0_PORT      # Port on the master node for communication
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE)) # Total number of GPU processes


# ==============================================================================
# VII. EXECUTION
# ==============================================================================
# The main command to start the distributed training job.

# Enable command echoing for easier debugging.
set -x


#    --micro_rollout_batch_size 2 \

# 如果是deepseek需要加上下面的，如果是qwen-vl测试则需要去掉这一个参数
#    --text_only \
#    --fsdp_cpu_offload \

torchrun --nnodes $NNODES --nproc-per-node $GPUS_PER_NODE --node_rank $NODE_RANK --master-port $MASTER_PORT --master-addr $MASTER_ADDR examples/safework_t1/train_colocate.py \
   --pretrain ${PRETRAIN_PATH} \
   --text_only \
   --loss_agg_mode seq-mean-token-mean \
   --save_trajectories \
   --num_trajectories_to_save 16 \
   --print_replay_buffer_stats \
   --advantage_estimator group_norm \
   --fsdp \
   --adam_offload \
   --flash_attn \
   --rm_use_engine \
   --mixed_mm_data \
   --reward_pretrain ${REWARD_PRETRAIN_PATHS} \
   --save_path ${SAVE_PATH} \
   --ckpt_path ${SAVE_PATH} \
   --micro_train_batch_size 1 \
   --train_batch_size ${TBS} \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size ${RBS} \
   --max_epochs 1 \
   --num_episodes ${EPISODE} \
   --lr_warmup_ratio ${WARMUP} \
   --n_samples_per_prompt $N_SAMPLES \
   --prompt_max_len $MAX_LENGTH \
   --generate_max_len $MAX_LENGTH \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate $LR \
   --init_kl_coef $KL \
   --kl_estimator k3 \
   --prompt_data $DATA_PATH \
   --input_key prompt \
   --images_key images \
   --reference_key chosen \
   --apply_chat_template \
   --gradient_checkpointing \
   --save_steps 20 \
   --max_ckpt_num 1 \
   --engine_mem_util 0.2 \
   --engine_tp_size $ENGINE_TP \
   --limit_mm_image_per_prompt $limit_mm_image_per_prompt \
   --enable_engine_sleep \
   --system_prompt 'A conversation between the User and Assistant. The User asks a question, and the Assistant provides a solution. The Assistant first thinks through the reasoning process internally with self-reflection and consistency check and then gives the final analysis and answer. The reasoning process should be enclosed within <think></think>, followed directly by the final thought and answer, like this: <think> reasoning process here </think> final thought and answer here.' \
   --l2 1.0e-2 \
   --freeze_prefix \
   --use_wandb "${WANDB_API_KEY}" \
   --wandb_project "${WANDB_PROJECT}" \
   --wandb_run_name "${WANDB_RUN_NAME}" \
   2>&1 | tee "${WRITABLE_BASE_DIR}/rft_logs/$NAME/deepseek72b-after-kg_svkng-orm_no-kl_1node_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
   


# cd /mnt/shared-storage-user/puyuan/code/code_refactor/LightRFT
# bash /mnt/shared-storage-user/puyuan/code/code_refactor/LightRFT/examples/safework_t1/run_grpo_svki_fsdp_deepseek.sh 2>&1 | tee "/mnt/shared-storage-user/puyuan/code/code_refactor/LightRFT/rft_logs/${NAME}/deepseek_${NAME}_1node_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"