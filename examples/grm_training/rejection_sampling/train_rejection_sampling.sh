#!/bin/bash

# Training script for rejection sampling data
# This script trains the model on the filtered rejection sampling data

set -e

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

#############################  Configuration ##########################
# Model path (pretrained model to continue training from)
MODEL_PATH="/mnt/shared-storage-user/puyuan/wanzunian/models/lightrlhf-grm-lr1e5-imagegen_cot_reward-qwen2.5vl3B-gs3000"

# Training data path (already converted rejection sampling data)
TRAINING_DATA_PATH="/mnt/shared-storage-user/sunjiaxuan/dec/LightRFT/results/rejection_sampling_20260102_022303/rejection_sampling_train.json"

# Output directory for checkpoints
OUTPUT_DIR="/mnt/shared-storage-user/sunjiaxuan/dec/LightRFT/results/rejection_sampling_20260102_022303/checkpoint"
LOG_DIR="/mnt/shared-storage-user/sunjiaxuan/dec/LightRFT/results/rejection_sampling_20260102_022303/logs"

# Training hyperparameters
TBS=4  # Reduced from 8 to save memory
LR=2.5e-6
MAX_LENGTH=13000
MAX_EPOCHS=3
MICRO_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=16  # Increase to maintain effective batch size (4 * 16 = 64)

# Task instruction for CoT reasoning (must match the one used during inference)
TASK_INSTRUCTION="""Given a caption and two images generated based on this caption, please analyze in detail the two provided images. 
Evaluate them on various dimensions such as semantic consistency (how closely the image content aligns with the caption), 
aesthetics (composition, color usage, artistic expression), authenticity (realism and attention to detail), 
and any other factors you deem relevant. For each evaluation dimension, 
provide a score between 1-10 for both images (e.g., Image 1: 8/10, Image 2: 6/10) and provide a concise rationale for the score. 
Calculate the total score for each image by summing all dimension scores. 
Use a chain-of-thought process to detail your reasoning steps, and enclose all your detailed reasoning within tags. 
Then, in the <answer> tag, output exactly one of the following strings: 'Image 1 is better' or 'Image 2 is better' or 'Both are equal' based on the total scores. 
No additional text is allowed in the <answer> section.
Example output format:
<think>
Semantic consistency: Image 1 (9/10) - ...; Image 2 (7/10) - ...
Aesthetics: Image 2 (8/10) - ...; Image 1 (8/10) - ...
Authenticity: Image 1 (8/10) - ...; Image 2 (5/10) - ...
[Additional dimensions if any]: Image 2 (8/10) - ...; Image 1 (6/10) - ...
Total score:
Image 1: 9+8+8+6=31
Image 2: 7+8+5+8=28
</think>
<answer>Image 1 is better</answer>
Note: In the example above, scores and the final answer are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of two given images.
Your task is provided as follows:
Text Caption: **{prompt}**
"""

############################### Environment #####################
export GPUS_PER_NODE=${GPUS_PER_NODE:-2}  # Use 2 GPUs by default
export NNODES=${NNODES:-1}
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-29500}

export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

# Memory optimization: reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Validate required configuration
if [ ! -f "${TRAINING_DATA_PATH}" ]; then
    echo "Error: Training data file not found: ${TRAINING_DATA_PATH}"
    exit 1
fi

if [ -z "${MODEL_PATH}" ]; then
    echo "Error: MODEL_PATH is not set. Please configure it in the script."
    exit 1
fi

# Create output directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "Rejection Sampling Training"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Training Data: ${TRAINING_DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${GPUS_PER_NODE}"
echo "=========================================="

# Use imagegen-cot-reward handler for the converted data
TRAINING_DATA_SOURCE="imagegen-cot-reward-5k:${TRAINING_DATA_PATH}"

############################### Training ##########################
echo ""
echo "Starting training on rejection sampling data..."
echo "=========================================="

torchrun --nnodes $NNODES --nproc-per-node $GPUS_PER_NODE \
    --node_rank $NODE_RANK --master-port $MASTER_PORT --master-addr $MASTER_ADDR \
    examples/grm_training/train_grm_vl.py \
    --pretrain ${MODEL_PATH} \
    --save_path ${OUTPUT_DIR} \
    --ckpt_path ${OUTPUT_DIR} \
    --train_batch_size ${TBS} \
    --micro_train_batch_size ${MICRO_BATCH_SIZE} \
    --max_epochs ${MAX_EPOCHS} \
    --lr_warmup_ratio 0.03 \
    --prompt_max_len ${MAX_LENGTH} \
    --fps 2.0 \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate ${LR} \
    --train_data ${TRAINING_DATA_SOURCE} \
    --gradient_checkpointing \
    --save_steps 1000 \
    --max_ckpt_num 2 \
    --use_tensorboard "${OUTPUT_DIR}/../tensorboard" \
    --l2 0.0 \
    --flash_attn \
    --task_instruction "${TASK_INSTRUCTION}" \
    2>&1 | tee ${LOG_DIR}/training.log

echo ""
echo "=========================================="
echo "Training Completed!"
echo "=========================================="
echo "Final checkpoint: ${OUTPUT_DIR}/final_checkpoint"
echo "Training logs: ${LOG_DIR}/training.log"
echo "=========================================="

