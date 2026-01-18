#!/bin/bash
#
# Example configuration for math benchmarks evaluation
# 
# Copy this file and modify the parameters according to your needs.
#

# ============================================================================
# Model Configuration
# ============================================================================

# Path to your trained model checkpoint
export MODEL_PATH="/path/to/your/model/checkpoint"

# ============================================================================
# Benchmark Selection
# ============================================================================

# Available benchmarks:
#   - math500: 500 challenging math problems
#   - aime2024: AIME 2024 competition problems
#   - aime2025: AIME 2025 competition problems (if available)
#   - gpqa_diamond: Graduate-level STEM questions

# Evaluate on all benchmarks
export BENCHMARKS="math500 aime2024 gpqa_diamond"

# Or evaluate on specific benchmarks
# export BENCHMARKS="math500"
# export BENCHMARKS="math500 aime2024"

# ============================================================================
# Data Configuration
# ============================================================================

# Root directory containing the benchmark data files
# Default: Open-Reasoner-Zero eval_data directory
export DATA_ROOT="/mnt/shared-storage-user/sunjiaxuan/oct/Open-Reasoner-Zero/data/eval_data"

# ============================================================================
# Generation Parameters
# ============================================================================

# Maximum tokens to generate per example
export MAX_TOKENS=8192

# Sampling temperature (0.0 for greedy decoding, recommended for math)
export TEMPERATURE=0.0

# Top-p sampling parameter
export TOP_P=0.95

# Batch size for inference
# Increase for faster evaluation if you have enough GPU memory
export BATCH_SIZE=1

# ============================================================================
# vLLM Configuration (Optional)
# ============================================================================

# Enable vLLM for faster inference (requires vLLM installation)
# Set to 1 to enable, 0 to disable
export USE_VLLM=0

# Number of GPUs for tensor parallelism (only when USE_VLLM=1)
export VLLM_TENSOR_PARALLEL_SIZE=1

# ============================================================================
# Output Configuration
# ============================================================================

# Directory to save evaluation results
export OUTPUT_DIR="./eval_results"

# ============================================================================
# Run Evaluation
# ============================================================================

echo "Starting evaluation with configuration:"
echo "  Model: $MODEL_PATH"
echo "  Benchmarks: $BENCHMARKS"
echo "  Output: $OUTPUT_DIR"
echo ""

bash run_eval.sh "$MODEL_PATH" $BENCHMARKS

