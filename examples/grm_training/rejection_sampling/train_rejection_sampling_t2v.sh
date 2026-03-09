#!/bin/bash

# Training script for rejection sampling T2V (text-to-video) data
# This script trains the model on the filtered rejection sampling video data

set -e

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

#############################  Configuration ##########################
# Model path (pretrained model to continue training from)
MODEL_PATH="path/to/your/model"

# Training data path (already converted rejection sampling data)
# This should be the output from convert_to_rejection_sampling_data_t2v.py
TRAINING_DATA_PATH="path/to/rejection_sampling_train.json"

# Output directory for checkpoints
OUTPUT_DIR="./results/rejection_sampling_t2v_training_$(date +%Y%m%d_%H%M%S)/checkpoint"
LOG_DIR="./results/rejection_sampling_t2v_training_$(date +%Y%m%d_%H%M%S)/logs"

# Training hyperparameters
TBS=8
LR=2.5e-6
MAX_LENGTH=13000
MAX_EPOCHS=3
MICRO_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=32  # Increase to maintain effective batch size

# Video FPS configuration
VIDEO_FPS=2.0

# Task instruction for CoT reasoning (T2V) - must match the one used during inference
TASK_INSTRUCTION="""Given a caption and two videos generated based on this caption, please analyze in detail the two provided videos. 
Evaluate them on various dimensions such as semantic consistency (how closely the video content aligns with the caption), temporal coherence (smoothness and logical flow of motion across frames), authenticity (realism and attention to detail), and any other factors you deem relevant. 
For each evaluation dimension, provide a score between 1-10 for both videos (e.g., Video 1: 8/10, Video 2: 6/10) and provide a concise rationale for the score. 
Calculate the total score for each video by summing all dimension scores. 
Use a chain-of-thought process to detail your reasoning steps, and enclose all your detailed reasoning within <think> and </think> tags. Then, in the <answer> tag, output exactly one of the following strings:
'Video 1 is better' or 'Video 2 is better' based on the total scores. No additional text is allowed in the <answer> section.
Example output format:
<think>
1. Semantic consistency: Video 1 (9/10) - ...; Video 2 (7/10) - ...
2. Temporal coherence: Video 1 (8/10) - ...; Video 2 (6/10) - ...
3. Authenticity: Video 1 (7/10) - ...; Video 2 (5/10) - ...
...
[Additional dimensions if any]: Video 2 (8/10) - ...; Video 1 (6/10) - ...
Total score:
Video 1: 9+8+7+6=30
Video 2: 7+6+5+8=26
</think>
<answer>Video 1 is better</answer>

Note: In the example above, scores and the final answer are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of two given videos.
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
    echo "Please run convert_to_rejection_sampling_data_t2v.py first to convert filtered samples."
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
echo "Rejection Sampling Training (T2V)"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Training Data: ${TRAINING_DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${GPUS_PER_NODE}"
echo "Video FPS: ${VIDEO_FPS}"
echo "=========================================="

# Use rejection-sampling-t2v handler for the converted data
TRAINING_DATA_SOURCE="rejection-sampling-t2v:${TRAINING_DATA_PATH}"

############################### Training ##########################
echo ""
echo "Starting training on rejection sampling T2V data..."
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
    --fps ${VIDEO_FPS} \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate ${LR} \
    --train_data ${TRAINING_DATA_SOURCE} \
    --gradient_checkpointing \
    --save_steps 1000 \
    --max_ckpt_num 8 \
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
