#!/bin/bash
#
# Evaluation script for mathematical reasoning benchmarks
#
# Usage:
#   bash run_eval.sh <model_path> [benchmarks...]
#
# Examples:
#   # Evaluate on all benchmarks
#   bash run_eval.sh /path/to/model
#
#   # Evaluate on specific benchmarks
#   bash run_eval.sh /path/to/model math500 aime2024
#
#   # With vLLM for faster inference
#   USE_VLLM=1 bash run_eval.sh /path/to/model
#

set -e

# Default configuration
MODEL_PATH=${1:-""}
shift || true
BENCHMARKS=${@:-"math500 aime2024 gpqa_diamond"}

# Data root (from Open-Reasoner-Zero)
DATA_ROOT=${DATA_ROOT:-"/mnt/shared-storage-user/sunjiaxuan/oct/Open-Reasoner-Zero/data/eval_data"}

# Generation config
MAX_TOKENS=${MAX_TOKENS:-8192}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-0.95}
BATCH_SIZE=${BATCH_SIZE:-1}

# vLLM config
USE_VLLM=${USE_VLLM:-0}
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-1}

# Output directory
OUTPUT_DIR=${OUTPUT_DIR:-"./eval_results"}

# Check if model path is provided
if [ -z "$MODEL_PATH" ]; then
    echo "Error: Model path is required"
    echo "Usage: bash run_eval.sh <model_path> [benchmarks...]"
    exit 1
fi

echo "=========================================="
echo "Math Benchmarks Evaluation"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Benchmarks: $BENCHMARKS"
echo "Data Root: $DATA_ROOT"
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Build command
CMD="python eval_math_benchmarks.py \
    --model_path $MODEL_PATH \
    --benchmarks $BENCHMARKS \
    --data_root $DATA_ROOT \
    --max_tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --batch_size $BATCH_SIZE \
    --output_dir $OUTPUT_DIR"

# Add vLLM if enabled
if [ "$USE_VLLM" = "1" ]; then
    CMD="$CMD --use_vllm --vllm_tensor_parallel_size $VLLM_TENSOR_PARALLEL_SIZE"
    echo "Using vLLM with tensor_parallel_size=$VLLM_TENSOR_PARALLEL_SIZE"
fi

# Run evaluation
echo "Running evaluation..."
echo ""
eval $CMD

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

