#!/bin/bash

# Rejection Sampling Data Preparation Script
# This script performs rejection sampling data preparation:
# 1. Inference on dataset and filter correct samples
# 2. Convert filtered samples to training format

set -e

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

#############################  Configuration ##########################
# Model path (cold-start model)
# Please set your model path here
MODEL_PATH="path/to/your/model"

# Dataset configuration
# Please set your dataset path here (format: "source:path")
DATA_PATH="hpdv3:path/to/dataset.json"
# Please set your dataset root directory here
DATA_ROOT="path/to/dataset/root"

# Output paths
OUTPUT_DIR="./results/rejection_sampling_$(date +%Y%m%d_%H%M%S)"
FILTERED_SAMPLES_PATH="${OUTPUT_DIR}/filtered_samples.json"
TRAINING_DATA_PATH="${OUTPUT_DIR}/rejection_sampling_train.json"

# Inference parameters
INFERENCE_BATCH_SIZE=8
MAX_NEW_TOKENS=2048

# Task instruction for CoT reasoning
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

if [ -z "${DATA_PATH}" ]; then
    echo "Error: DATA_PATH is not set. Please configure it in the script."
    exit 1
fi

if [ -z "${DATA_ROOT}" ]; then
    echo "Error: DATA_ROOT is not set. Please configure it in the script."
    exit 1
fi

# Create output directory
mkdir -p ${OUTPUT_DIR}
LOG_BASE="${OUTPUT_DIR}/logs"
mkdir -p ${LOG_BASE}

echo "=========================================="
echo "Rejection Sampling Data Preparation"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Data: ${DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

############################### Step 1: Inference and Filter ##########################
echo ""
echo "Step 1: Running inference and filtering correct samples..."
echo "=========================================="

# Use vLLM for inference (vLLM handles multi-GPU internally via tensor_parallel_size)
python examples/grm_training/rejection_sampling/rejection_sampling_inference.py \
    --model_path ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --output_path ${FILTERED_SAMPLES_PATH} \
    --batch_size ${INFERENCE_BATCH_SIZE} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --use_cot \
    --task_instruction "${TASK_INSTRUCTION}" \
    --tensor_parallel_size ${GPUS_PER_NODE} \
    --gpu_memory_utilization 0.9 \
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

python examples/grm_training/rejection_sampling/convert_to_rejection_sampling_data.py \
    --filtered_samples_path ${FILTERED_SAMPLES_PATH} \
    --output_path ${TRAINING_DATA_PATH} \
    --data_root ${DATA_ROOT} \
    --task_instruction "${TASK_INSTRUCTION}" \
    2>&1 | tee ${LOG_BASE}/convert.log

if [ ! -f "${TRAINING_DATA_PATH}" ]; then
    echo "Error: Training data file not created!"
    exit 1
fi

echo "Step 2 completed. Training data saved to: ${TRAINING_DATA_PATH}"

echo ""
echo "=========================================="
echo "Rejection Sampling Data Preparation Completed!"
echo "=========================================="
echo "Filtered samples: ${FILTERED_SAMPLES_PATH}"
echo "Training data: ${TRAINING_DATA_PATH}"
echo "All outputs saved to: ${OUTPUT_DIR}"
echo "=========================================="

