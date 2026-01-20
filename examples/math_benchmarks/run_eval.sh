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
#   # Override configuration via environment variables
#   DATA_ROOT=/custom/path MAX_TOKENS=4096 bash run_eval.sh /path/to/model
#

set -e

# ============================================================================
# Model Configuration
# ============================================================================

# Path to your trained model checkpoint
# Can be provided as first argument or via MODEL_PATH environment variable
MODEL_PATH=${1:-"${MODEL_PATH:-}"}
shift 2>/dev/null || true

# ============================================================================
# Benchmark Selection
# ============================================================================

# Available benchmarks:
#   - math500: 500 challenging math problems
#   - aime2024: AIME 2024 competition problems
#   - aime2025: AIME 2025 competition problems (if available)
#   - gpqa_diamond: Graduate-level STEM questions

# Benchmarks to evaluate (can be provided as arguments or via BENCHMARKS env var)
# Default: evaluate on all benchmarks
BENCHMARKS=${@:-"${BENCHMARKS:-math500 aime2024 gpqa_diamond}"}

# ============================================================================
# Data Configuration
# ============================================================================

# Root directory containing the benchmark data files
DATA_ROOT=${DATA_ROOT:-"/path/to/eval_data"}

# ============================================================================
# Generation Parameters
# ============================================================================

# Maximum tokens to generate per example
MAX_TOKENS=${MAX_TOKENS:-8192}

# Sampling temperature (0.0 for greedy decoding, recommended for math)
TEMPERATURE=${TEMPERATURE:-0.0}

# Top-p sampling parameter
TOP_P=${TOP_P:-0.95}

# Batch size for inference
# Increase for faster evaluation if you have enough GPU memory
BATCH_SIZE=${BATCH_SIZE:-1}

# ============================================================================
# vLLM Configuration (Optional)
# ============================================================================

# Enable vLLM for faster inference (requires vLLM installation)
# Set to 1 to enable, 0 to disable
USE_VLLM=${USE_VLLM:-0}

# Number of GPUs for tensor parallelism (only when USE_VLLM=1)
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-1}

# ============================================================================
# Output Configuration
# ============================================================================

# Directory to save evaluation results
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

