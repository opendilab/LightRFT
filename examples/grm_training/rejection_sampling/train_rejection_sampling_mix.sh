#!/bin/bash

# Mixed training script for rejection sampling Image (T2I) + Video (T2V) data
# 使用同一个 GRM 模型，在图像和视频拒绝采样数据上进行联合训练。

set -e

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

#############################  Configuration ##########################
# 预训练模型路径（从该模型继续训练）
MODEL_PATH="/mnt/shared-storage-user/puyuan/wanzunian/models/lightrlhf-grm-lr1e5-imagegen_cot_reward-qwen2.5vl3B-gs3000"

# 图像拒绝采样数据（convert_to_rejection_sampling_data.py 的输出）
T2I_TRAINING_DATA_PATH="/mnt/shared-storage-user/sunjiaxuan/dec/LightRFT/results/rejection_sampling_20260115_170931/rejection_sampling_train.json"

# 视频拒绝采样数据（convert_to_rejection_sampling_data_t2v.py 的输出）
T2V_TRAINING_DATA_PATH="/mnt/shared-storage-user/sunjiaxuan/dec/LightRFT/results/rejection_sampling_t2v_20260104_193830/rejection_sampling_train.json"

# 输出目录
OUTPUT_DIR="/mnt/shared-storage-user/sunjiaxuan/dec/LightRFT/results/rejection_sampling_mix_$(date +%Y%m%d_%H%M%S)/checkpoint"
LOG_DIR="$(dirname "${OUTPUT_DIR}")/logs"

# 训练超参（可以按需修改）
TBS=8                       # global train batch size
LR=2.5e-6
MAX_LENGTH=13000
MAX_EPOCHS=3
MICRO_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=32

# 视频 FPS 配置
VIDEO_FPS=2.0

# 注意：
#  - Image 数据的 system prompt（task_instruction）从 T2I_TRAINING_DATA_PATH 对应的 json 里读取；
#  - Video 数据的 system prompt 从 T2V_TRAINING_DATA_PATH 对应的 json 里读取；
#  每条样本在 json 的 conversations[0]['value'] 里已经带了各自的 CoT 说明，因此这里不再额外传统一的 TASK_INSTRUCTION。

############################### Environment #####################
export GPUS_PER_NODE=${GPUS_PER_NODE:-2}
export NNODES=${NNODES:-1}
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-29500}

export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

# 减少显存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 检查配置
if [ ! -f "${T2I_TRAINING_DATA_PATH}" ]; then
    echo "Error: T2I training data file not found: ${T2I_TRAINING_DATA_PATH}"
    exit 1
fi

if [ ! -f "${T2V_TRAINING_DATA_PATH}" ]; then
    echo "Error: T2V training data file not found: ${T2V_TRAINING_DATA_PATH}"
    exit 1
fi

if [ -z "${MODEL_PATH}" ]; then
    echo "Error: MODEL_PATH is not set. Please configure it in the script."
    exit 1
fi

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Rejection Sampling Mixed Training (T2I + T2V)"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "T2I Training Data: ${T2I_TRAINING_DATA_PATH}"
echo "T2V Training Data: ${T2V_TRAINING_DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${GPUS_PER_NODE}"
echo "Video FPS: ${VIDEO_FPS}"
echo "=========================================="

# 这里利用 GRMDataset 中的多 handler 能力：
# args.train_data 会在 train_grm_vl.py 中被按逗号切分成 list，
# 每个元素是 "source:path" 的形式。
T2I_SOURCE="imagegen-cot-reward-5k:${T2I_TRAINING_DATA_PATH}"
T2V_SOURCE="rejection-sampling-t2v:${T2V_TRAINING_DATA_PATH}"

TRAINING_DATA_SOURCES="${T2I_SOURCE},${T2V_SOURCE}"

############################### Training ##########################
echo ""
echo "Starting mixed training on T2I + T2V rejection sampling data..."
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
    --train_data ${TRAINING_DATA_SOURCES} \
    --gradient_checkpointing \
    --save_steps 1000 \
    --max_ckpt_num 8 \
    --use_tensorboard "$(dirname "${OUTPUT_DIR}")/tensorboard" \
    --l2 0.0 \
    --flash_attn \
    2>&1 | tee ${LOG_DIR}/training.log

echo ""
echo "=========================================="
echo "Mixed Training Completed!"
echo "=========================================="
echo "Final checkpoint: ${OUTPUT_DIR}/final_checkpoint"
echo "Training logs: ${LOG_DIR}/training.log"
echo "=========================================="

