#!/bin/bash

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

#############################  kwargs ##########################
WARMUP=0.0
TBS=8
LR=1e-5
MAX_LENGTH=8196
FPS=2.0

# Training data paths
data_files=(
    "imagegen-cot-reward-5k:/path/to/imagegen-cot-reward-5k/train.json"
    "hpdv3:/path/to/hpdv3/train.json"
    "your-source:/path/to/your/other/dataset"
)
DATA_PATH=$(printf '%s,' "${data_files[@]}" | sed 's/,$//')

# Path to evaluation data (Optional)
EVAL_DATA_PATH="Path/to/eval/data"

# Example Task Instruction
TASK_INSTRUCTION="""
Provide the task instruction here.
The task instruction should clearly explain the evaluation criteria.
It should include a {prompt} placeholder for the text prompt.
"""

# Pretrained model path
PRETRAIN_PATH="/path/to/the/pretrained/model"

# Save and log paths
current_time=$(date +"%m%d%H%M")
EXPERIMENT_NAME=lightrft-grm-vl-training
SAVE_MODEL_NAME=${EXPERIMENT_NAME}-lr$LR-dataset-pretrained_model-$current_time
mkdir -p results/$EXPERIMENT_NAME/$SAVE_MODEL_NAME

LOG_BASE=log

mkdir -p $LOG_BASE

# Wandb settings
WANDB_API_KEY="your_wandb_api_key"
WANDB_PROJECT=${EXPERIMENT_NAME}
WANDB_RUN_NAME="${SAVE_MODEL_NAME}"
export WANDB_MODE="offline"
# Set WandB cache dir to current log dir
export WANDB_CACHE_DIR="$(pwd)/wandb/wandb_cache"
export WANDB_DATA_DIR="$(pwd)/wandb/wandb_data"

############################### env settings #####################
# Single-Node Distributed Setup
export MLP_WORKER_NUM=1 
export MLP_WORKER_GPU=8      # Number of GPUs per node
export MLP_ROLE_INDEX=0
export MLP_WORKER_0_PORT=20091
export MLP_WORKER_0_HOST=localhost # or 127.0.0.1

# PyTorch Distributed Environment Variables
export MASTER_ADDR=$MLP_WORKER_0_HOST
export NNODES=$MLP_WORKER_NUM
export NODE_RANK=$MLP_ROLE_INDEX
export GPUS_PER_NODE=$MLP_WORKER_GPU
export MASTER_PORT=$MLP_WORKER_0_PORT

# Compute total world size (number of processes)
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

# export NCCL_DEBUG=INFO

############################### torchrun ##########################
set -x

torchrun --nnodes $NNODES --nproc-per-node $GPUS_PER_NODE --node_rank $NODE_RANK --master-port $MASTER_PORT --master-addr $MASTER_ADDR examples/grm_training/train_grm_vl.py \
    --pretrain ${PRETRAIN_PATH} \
    --save_path results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME} \
    --ckpt_path results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME} \
    --train_batch_size ${TBS} \
    --micro_train_batch_size 4 \
    --max_epochs 5 \
    --lr_warmup_ratio ${WARMUP} \
    --prompt_max_len $MAX_LENGTH \
    --generate_max_len 512 \
    --fps $FPS \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate $LR \
    --train_data $DATA_PATH \
    --eval_data ${EVAL_DATA_PATH} \
    --eval_steps 500 \
    --gradient_checkpointing \
    --save_steps 500 \
    --max_ckpt_num 10 \
    --use_tensorboard "tensorboard/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    --use_wandb "${WANDB_API_KEY}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    --l2 1.0e-4 \
    --flash_attn \
    --task_instruction "$TASK_INSTRUCTION" \
    2>&1 | tee log/lightrft_grm_${NODE_RANK}.log

#    --fsdp \
#    --lora_rank 16 \
#    --lora_alpha 32 \
#    --lora_dropout 0.1 \
#    --target_modules q_proj k_proj v_proj o_proj \

