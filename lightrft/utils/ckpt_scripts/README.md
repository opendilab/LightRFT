# Checkpoint Conversion Scripts

This directory contains scripts for converting checkpoints between different formats (DeepSpeed, FSDP, HuggingFace) and testing converted models.

## Available Scripts

### 1. DeepSpeed to HuggingFace (`ds2hf.py`)

Converts DeepSpeed checkpoints (Zero-1, Zero-2, Zero-3) to HuggingFace format.

**Required Parameters:**
- `--checkpoint_dir`: Path to the DeepSpeed checkpoint directory
- `--output_dir`: Path to save the converted HuggingFace model
- `--hf_base`: Path to the base HuggingFace model (training start point)

**Optional Parameters:**
- `--tag`, `-t`: Checkpoint tag used as unique identifier (e.g., global_step1)
- `--exclude_frozen_parameters`: Exclude frozen parameters from conversion
- `--debug`, `-d`: Enable debug mode
- `--model_type`: Type of model to initialize (choices: vision2seq, grm, srm_vl, srm_al; default: vision2seq)

**Example:**
```bash
python lightrft/utils/ckpt_scripts/ds2hf.py \
    --checkpoint_dir results/lightrft-7B-ds/_actor/global_step2/ \
    --output_dir results/lightrft-7B-ds2hf \
    --hf_base checkpoints/qwen25-vl-7b-s10-0321-150
```

### 2. FSDP to HuggingFace (`fsdp2hf.py`)

Converts PyTorch Distributed Checkpoint (DCP/FSDP) to HuggingFace format. Automatically detects and handles both Causal Language Models (LLMs) and Vision Language Models (VLMs).

**Required Parameters:**
- `--hf_base`: Path to the base HuggingFace model config directory
- `--checkpoint_dir`: Path to the FSDP checkpoint directory
- `--output_dir`: Path to save the converted HuggingFace checkpoint

**Example:**
```bash
python lightrft/utils/ckpt_scripts/fsdp2hf.py \
    --hf_base checkpoints/qwen25-vl-7b-s10-0321-150 \
    --checkpoint_dir results/lightrft-7B-fsdp/_actor/global_step6/ \
    --output_dir results/lightrft-7B-fsdp2hf
```

### 3. HuggingFace to FSDP (`hf2fsdp.py`)

Converts HuggingFace checkpoint (safetensors format) to PyTorch Distributed Checkpoint (DCP/FSDP) format.

**Required Parameters:**
- `--hf_checkpoint`: Path to the HuggingFace checkpoint directory (containing .safetensors files)
- `--output`: Path to save the converted FSDP checkpoint

**Example:**
```bash
python lightrft/utils/ckpt_scripts/hf2fsdp.py \
    --hf_checkpoint checkpoints/qwen25-vl-7b-s10-0321-150 \
    --output results/qwen25-vl-7b-fsdp
```

### 4. Test Model (`test_model.py`)

Tests a converted HuggingFace model with sample prompts and optional benchmarking.

**Parameters:**
- `--model_path`: Path to the converted HuggingFace model (required)
- `--device`: Device to run on (default: cuda)
- `--test_prompts`: Test prompts (multiple prompts can be provided)
- `--max_new_tokens`: Maximum new tokens to generate (default: 256)
- `--benchmark`: Run benchmark test
- `--benchmark_samples`: Number of benchmark samples (default: 5)
- `--output_file`: Output file for results (default: test_results.json)

**Example:**
```bash
python lightrft/utils/ckpt_scripts/test_model.py \
    --model_path results/lightrft-7B-ds2hf \
    --test_prompts "Write a Python function to calculate Fibonacci numbers." \
    --max_new_tokens 512 \
    --benchmark \
    --output_file test_results.json
```

## Workflow Examples

### Complete DeepSpeed Training to Inference Workflow

```bash
# 1. Train with DeepSpeed (example command, not part of these scripts)
# Your training produces: results/lightrft-7B-ds/_actor/global_step2/

# 2. Convert DeepSpeed checkpoint to HuggingFace
python lightrft/utils/ckpt_scripts/ds2hf.py \
    --checkpoint_dir results/lightrft-7B-ds/_actor/global_step2/ \
    --output_dir results/lightrft-7B-ds2hf \
    --hf_base checkpoints/qwen25-vl-7b-s10-0321-150

# 3. Test the converted model
python lightrft/utils/ckpt_scripts/test_model.py \
    --model_path results/lightrft-7B-ds2hf \
    --benchmark
```

### Complete FSDP Training to Inference Workflow

```bash
# 1. Train with FSDP (example command, not part of these scripts)
# Your training produces: results/lightrft-7B-fsdp/_actor/global_step6/

# 2. Convert FSDP checkpoint to HuggingFace
python lightrft/utils/ckpt_scripts/fsdp2hf.py \
    --hf_base checkpoints/qwen25-vl-7b-s10-0321-150 \
    --checkpoint_dir results/lightrft-7B-fsdp/_actor/global_step6/ \
    --output_dir results/lightrft-7B-fsdp2hf

# 3. Test the converted model
python lightrft/utils/ckpt_scripts/test_model.py \
    --model_path results/lightrft-7B-fsdp2hf \
    --benchmark
```

## Notes

- All conversion scripts support both Language Models (LLMs) and Vision-Language Models (VLMs)
- The `hf_base` parameter should point to the original HuggingFace model that was used as the starting point for training
- Converted checkpoints are fully compatible with HuggingFace's inference and deployment tools
- For large models, ensure sufficient CPU memory is available during conversion
