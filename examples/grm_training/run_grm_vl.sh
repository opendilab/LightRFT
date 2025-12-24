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
SAVE_MODEL_NAME=lightrft-grm-lr$LR-dataset-pretrained_model-$current_time
mkdir -p results/$SAVE_MODEL_NAME

LOG_BASE=log

mkdir -p $LOG_BASE

############################### env settings #####################
export GPUS_PER_NODE=$GPUS_PER_NODE
export NNODES=$NNODES
export NODE_RANK=$RANK
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

# Compute total world size (number of processes)
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

# export NCCL_DEBUG=INFO

############################### torchrun ##########################
set -x

torchrun --nnodes $NNODES --nproc-per-node $GPUS_PER_NODE --node_rank $NODE_RANK --master-port $MASTER_PORT --master-addr $MASTER_ADDR examples/grm_training/train_grm_vl.py \
    --pretrain ${PRETRAIN_PATH} \
    --save_path results/$SAVE_MODEL_NAME \
    --ckpt_path results/$SAVE_MODEL_NAME \
    --train_batch_size ${TBS} \
    --micro_train_batch_size 4 \
    --max_epochs 5 \
    --lr_warmup_ratio ${WARMUP} \
    --prompt_max_len $MAX_LENGTH \
    --fps $FPS \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate $LR \
    --train_data $DATA_PATH \
    --gradient_checkpointing \
    --save_steps 500 \
    --max_ckpt_num 10 \
    --use_tensorboard "tensorboard/grm/$SAVE_MODEL_NAME" \
    --l2 1.0e-4 \
    --flash_attn \
    --task_instruction "$TASK_INSTRUCTION" \
    2>&1 | tee log/lightrft_grm_${NODE_RANK}.log

#    --eval_data ${EVAL_DATA_PATH} \
#    --eval_steps 500 \
#    --task_instruction "$TASK_INSTRUCTION" \
#    --lora_rank 16 \
#    --lora_alpha 32 \
#    --lora_dropout 0.1 \
#    --target_modules q_proj k_proj v_proj o_proj \

