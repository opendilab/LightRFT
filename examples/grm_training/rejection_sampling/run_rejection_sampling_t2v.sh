#!/bin/bash

# Rejection Sampling Training Script for Text-to-Video (T2V)
# This script performs the complete rejection sampling pipeline:
# 1. Inference on dataset and filter correct samples
# 2. Convert filtered samples to training format
# 3. Train the model on filtered samples

set -e

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

#############################  Configuration ##########################
# Model path (cold-start model)
# Please set your model path here
MODEL_PATH="/mnt/shared-storage-user/puyuan/wanzunian/models/lightrlhf-grm-lr1e5-imagegen_cot_reward-qwen2.5vl3B-gs3000"

# Dataset configuration
# Multiple rapidata-t2v datasets
DATA_PATH=(
    "rapidata-t2v:/mnt/shared-storage-user/puyuan/wanzunian/datasets/rapidata/text-2-video-human-preferences-veo3/data/train-00000-of-00001.parquet"
    "rapidata-t2v:/mnt/shared-storage-user/puyuan/wanzunian/datasets/rapidata/text-2-video-human-preferences-pika2.2/data/train-00000-of-00001.parquet"
    "rapidata-t2v:/mnt/shared-storage-user/puyuan/wanzunian/datasets/rapidata/text-2-video-human-preferences-wan2.1/data/train-00000-of-00001.parquet"
    "rapidata-t2v:/mnt/shared-storage-user/puyuan/wanzunian/datasets/rapidata/text-2-video-human-preferences/data/train-00000-of-00001.parquet"
)

# Output paths
OUTPUT_DIR="./results/rejection_sampling_t2v_$(date +%Y%m%d_%H%M%S)"
FILTERED_SAMPLES_PATH="${OUTPUT_DIR}/filtered_samples.json"
TRAINING_DATA_PATH="${OUTPUT_DIR}/rejection_sampling_train.json"
FINAL_CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoint"

# Training hyperparameters
TBS=8
LR=2.5e-6
MAX_LENGTH=13000
MAX_EPOCHS=3
MICRO_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=32

# Inference parameters (reduced to avoid OOM)
INFERENCE_BATCH_SIZE=8
MAX_NEW_TOKENS=2048

# Video FPS configuration
VIDEO_FPS=2.0

# Task instruction for CoT reasoning (T2V)
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
export GPUS_PER_NODE=${GPUS_PER_NODE:-2}  # Use 2 GPUs
export NNODES=${NNODES:-1}
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-29500}

export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

# Validate required configuration
if [ -z "${MODEL_PATH}" ]; then
    echo "Error: MODEL_PATH is not set. Please configure it in the script."
    exit 1
fi

if [ ${#DATA_PATH[@]} -eq 0 ]; then
    echo "Error: DATA_PATH is not set. Please configure it in the script."
    exit 1
fi

# Create output directory
mkdir -p ${OUTPUT_DIR}
LOG_BASE="${OUTPUT_DIR}/logs"
mkdir -p ${LOG_BASE}

echo "=========================================="
echo "Rejection Sampling Training Pipeline (T2V)"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Data: ${DATA_PATH[@]}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

############################### Step 1: Inference and Filter ##########################
echo ""
echo "Step 1: Running inference and filtering correct samples..."
echo "=========================================="

# Convert array to comma-separated string for Python script
DATA_PATH_STR=$(IFS=','; echo "${DATA_PATH[*]}")

# Use vLLM for inference (vLLM handles multi-GPU internally via tensor_parallel_size)
python examples/grm_training/rejection_sampling/rejection_sampling_inference_t2v.py \
    --model_path ${MODEL_PATH} \
    --data_path ${DATA_PATH_STR} \
    --output_path ${FILTERED_SAMPLES_PATH} \
    --batch_size ${INFERENCE_BATCH_SIZE} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --use_cot \
    --task_instruction "${TASK_INSTRUCTION}" \
    --tensor_parallel_size ${GPUS_PER_NODE} \
    --gpu_memory_utilization 0.9 \
    --video_fps ${VIDEO_FPS} \
    2>&1 | tee ${LOG_BASE}/inference.log

if [ ! -f "${FILTERED_SAMPLES_PATH}" ]; then
    echo "Error: Filtered samples file not created!"
    exit 1
fi

echo "Step 1 completed. Filtered samples saved to: ${FILTERED_SAMPLES_PATH}"

############################### Step 2: Convert to Training Format ##########################
echo ""
echo "Step 2: Converting filtered samples to training format..."
echo "=========================================="

python examples/grm_training/rejection_sampling/convert_to_rejection_sampling_data_t2v.py \
    --filtered_samples_path ${FILTERED_SAMPLES_PATH} \
    --output_path ${TRAINING_DATA_PATH} \
    --task_instruction "${TASK_INSTRUCTION}" \
    --video_fps ${VIDEO_FPS} \
    2>&1 | tee ${LOG_BASE}/convert.log

if [ ! -f "${TRAINING_DATA_PATH}" ]; then
    echo "Error: Training data file not created!"
    exit 1
fi

echo "Step 2 completed. Training data saved to: ${TRAINING_DATA_PATH}"

############################### Step 3: Training ##########################
echo ""
echo "Step 3: Training on filtered samples..."
echo "=========================================="

# Use imagegen-cot-reward handler for the converted data (it supports video too)
TRAINING_DATA_SOURCE="imagegen-cot-reward-5k:${TRAINING_DATA_PATH}"

torchrun --nnodes $NNODES --nproc-per-node $GPUS_PER_NODE \
    --node_rank $NODE_RANK --master-port $MASTER_PORT --master-addr $MASTER_ADDR \
    examples/grm_training/train_grm_vl.py \
    --pretrain ${MODEL_PATH} \
    --save_path ${FINAL_CHECKPOINT_PATH} \
    --ckpt_path ${FINAL_CHECKPOINT_PATH} \
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
    --use_tensorboard "${OUTPUT_DIR}/tensorboard" \
    --l2 0.0 \
    --flash_attn \
    --task_instruction "${TASK_INSTRUCTION}" \
    2>&1 | tee ${LOG_BASE}/training.log

echo ""
echo "=========================================="
echo "Rejection Sampling Training Completed (T2V)!"
echo "=========================================="
echo "Final checkpoint: ${FINAL_CHECKPOINT_PATH}/final_checkpoint"
echo "All outputs saved to: ${OUTPUT_DIR}"
echo "=========================================="

