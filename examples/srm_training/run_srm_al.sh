#!/bin/bash

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

#############################  kwargs ##########################
WARMUP=0.0
TBS=32
LR=1e-5
MAX_LENGTH=8196

# Path to training data
data_files=(
    "audio-alpaca:/path/to/your/audio-alpaca/train.json"
)
DATA_PATH=$(printf '%s,' "${data_files[@]}" | sed 's/,$//')

# Path to evaluation data (Optional)
EVAL_DATA_PATH="Path/to/eval/data"


# Example Task Instruction
TASK_INSTRUCTION="You will act as an expert audio evaluator for text-to-audio generation.
Given a text prompt and a generated audio clip, your task is to assess the overall quality of the audio in relation to the prompt.
Your evaluation should focus on the following key aspects:
• Preference: Which audio would a human listener find more satisfying or acoustically pleasing overall (considering audio fidelity, clarity, and musicality/naturalness).
• Alignment: How well the audio content matches the given text prompt in semantics, sound events, mood, and acoustic attributes (e.g., genre, tempo, instruments).
Your task is provided in the following, please give your judgement based on above criteria.
The prompt used for generation is as follows: {prompt}.
"

# Path to the pretrained model
PRETRAIN_PATH="path/to/your/pretrained/audio-language/model"

# Save and log paths
current_time=$(date +"%m%d%H%M")
EXPERIMENT_NAME=lightrft-srm-al-training
SAVE_MODEL_NAME=${EXPERIMENT_NAME}-lr$LR-loss_type-dataset-pretrained_model-$current_time
mkdir -p results/$EXPERIMENT_NAME/$SAVE_MODEL_NAME

LOG_BASE=log
mkdir -p $LOG_BASE

# Wandb settings
WANDB_API_KEY="your_wandb_api_key"
WANDB_PROJECT=${EXPERIMENT_NAME}
WANDB_RUN_NAME="${SAVE_MODEL_NAME}"
export WANDB_MODE="offline"

############################### env settings #####################
# Single-Node Distributed Setup
export MLP_WORKER_NUM=1 
export MLP_WORKER_GPU=8      # Number of GPUs per node
export MLP_ROLE_INDEX=0
export MLP_WORKER_0_PORT=20092
export MLP_WORKER_0_HOST=localhost # or 127.0.0.1

# PyTorch Distributed Environment Variables
export MASTER_ADDR=$MLP_WORKER_0_HOST
export NNODES=$MLP_WORKER_NUM
export NODE_RANK=$MLP_ROLE_INDEX
export GPUS_PER_NODE=$MLP_WORKER_GPU
export MASTER_PORT=$MLP_WORKER_0_PORT

# export NCCL_DEBUG=INFO

############################### torchrun #####################
set -x

torchrun --nnodes $NNODES --nproc-per-node $GPUS_PER_NODE --node_rank $NODE_RANK --master-port $MASTER_PORT --master-addr $MASTER_ADDR examples/srm_training/train_srm_al.py \
    --pretrain ${PRETRAIN_PATH} \
    --save_path results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME} \
    --ckpt_path results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME} \
    --train_batch_size ${TBS} \
    --micro_train_batch_size 4 \
    --max_epochs 5 \
    --lr_warmup_ratio ${WARMUP} \
    --prompt_max_len $MAX_LENGTH \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate $LR \
    --train_data $DATA_PATH \
    --eval_data ${EVAL_DATA_PATH} \
    --eval_steps 500 \
    --gradient_checkpointing \
    --save_steps 100 \
    --max_ckpt_num 5 \
    --use_tensorboard "tensorboard/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    --use_wandb "${WANDB_API_KEY}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    --l2 1.0e-4 \
    --flash_attn \
    --loss_type hps \
    --margin 0.1 \
    --scale_for_train \
    --pooling_method attn \
    --heads_types preference \
    --task_instruction "$TASK_INSTRUCTION" \
    2>&1 | tee log/lightrft_srm_al_${NODE_RANK}.log

#    --fsdp \
#    --adam_offload \
#    --probing_layer 17 \  # Default is -1, the last layer


