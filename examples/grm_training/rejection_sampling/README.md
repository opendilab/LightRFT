# Rejection Sampling for GRM Training

This directory contains scripts and tools for preparing rejection sampling training data and training GRM (Generative Reward Model) models on both text-to-image (T2I) and text-to-video (T2V) tasks.

## Overview

Rejection sampling is a technique to filter high-quality training samples by:
1. Running inference on a dataset using a trained GRM model
2. Filtering correctly predicted samples (where model prediction matches ground truth)
3. Converting filtered samples into training format with Chain-of-Thought (CoT) reasoning
4. Training the model on these high-quality filtered samples

## Directory Structure

```
rejection_sampling/
├── README.md                                    # This file
├── run_rejection_sampling_t2i.sh                # T2I rejection sampling data preparation
├── run_rejection_sampling_t2v.sh                # T2V rejection sampling data preparation
├── rejection_sampling_inference_t2i.py          # T2I inference and filtering script
├── rejection_sampling_inference_t2v.py          # T2V inference and filtering script
├── convert_to_rejection_sampling_data_t2i.py    # Convert T2I filtered samples to training format
├── convert_to_rejection_sampling_data_t2v.py    # Convert T2V filtered samples to training format
├── train_rejection_sampling_t2i.sh              # Train GRM on T2I rejection sampling data
├── train_rejection_sampling_t2v.sh              # Train GRM on T2V rejection sampling data
└── train_rejection_sampling_mix.sh              # Train GRM on mixed T2I + T2V data
```

## Workflow

### Text-to-Image (T2I) Rejection Sampling

#### Step 1: Data Preparation

Run the rejection sampling data preparation pipeline:

```bash
bash run_rejection_sampling_t2i.sh
```

**Configuration** (edit the script before running):
- `MODEL_PATH`: Path to your pre-trained GRM model
- `DATA_PATH`: Dataset path in format `"source:path"` (e.g., `"hpdv3:path/to/dataset.json"`)
- `DATA_ROOT`: Root directory of the dataset (for resolving image paths)
- `OUTPUT_DIR`: Directory to save filtered samples and training data
- `INFERENCE_BATCH_SIZE`: Batch size for inference (default: 8)
- `MAX_NEW_TOKENS`: Maximum tokens to generate (default: 2048)
- `TASK_INSTRUCTION`: CoT instruction template for the model

This script will:
1. Run inference on the dataset using vLLM
2. Filter correctly predicted samples
3. Convert filtered samples to training format
4. Save the results to `${OUTPUT_DIR}/rejection_sampling_train.json`

#### Step 2: Training

Train the model on the filtered rejection sampling data:

```bash
bash train_rejection_sampling_t2i.sh
```

**Configuration** (edit the script before running):
- `MODEL_PATH`: Path to your pre-trained model
- `TRAINING_DATA_PATH`: Path to the training data generated in Step 1
- `OUTPUT_DIR`: Directory to save model checkpoints
- `LOG_DIR`: Directory to save training logs
- Training hyperparameters: `TBS`, `LR`, `MAX_LENGTH`, `MAX_EPOCHS`, etc.

### Text-to-Video (T2V) Rejection Sampling

#### Step 1: Data Preparation

Run the T2V rejection sampling data preparation pipeline:

```bash
bash run_rejection_sampling_t2v.sh
```

**Configuration** (edit the script before running):
- `MODEL_PATH`: Path to your pre-trained GRM model
- `DATA_PATH`: Array of dataset paths in format `"rapidata-t2v:path/to/dataset.parquet"`
- `OUTPUT_DIR`: Directory to save filtered samples and training data
- `INFERENCE_BATCH_SIZE`: Batch size for inference (default: 8)
- `MAX_NEW_TOKENS`: Maximum tokens to generate (default: 2048)
- `VIDEO_FPS`: Video frames per second for processing (default: 2.0)
- `TASK_INSTRUCTION`: CoT instruction template for video evaluation

This script will:
1. Run inference on video datasets using vLLM
2. Filter correctly predicted samples based on video quality scores
3. Convert filtered samples to training format
4. Save the results to `${OUTPUT_DIR}/rejection_sampling_train.json`

#### Step 2: Training

Train the model on the filtered T2V rejection sampling data:

```bash
bash train_rejection_sampling_t2v.sh
```

**Configuration** (edit the script before running):
- `MODEL_PATH`: Path to your pre-trained model
- `TRAINING_DATA_PATH`: Path to the training data generated in Step 1
- `OUTPUT_DIR`: Directory to save model checkpoints
- `LOG_DIR`: Directory to save training logs
- `VIDEO_FPS`: Video FPS (must match the value used in data preparation)
- Training hyperparameters: `TBS`, `LR`, `MAX_LENGTH`, `MAX_EPOCHS`, etc.

### Mixed Training (T2I + T2V)

Train a single GRM model on both image and video rejection sampling data:

```bash
bash train_rejection_sampling_mix.sh
```

**Configuration** (edit the script before running):
- `MODEL_PATH`: Path to your pre-trained model
- `T2I_TRAINING_DATA_PATH`: Path to T2I rejection sampling training data
- `T2V_TRAINING_DATA_PATH`: Path to T2V rejection sampling training data
- `OUTPUT_DIR`: Directory to save model checkpoints
- `LOG_DIR`: Directory to save training logs
- Training hyperparameters: `TBS`, `LR`, `MAX_LENGTH`, `MAX_EPOCHS`, etc.

## Python Scripts

### Inference Scripts

#### `rejection_sampling_inference_t2i.py`

Performs inference on T2I datasets and filters correctly predicted samples.

**Usage:**

```bash
python rejection_sampling_inference_t2i.py \
    --model_path path/to/model \
    --data_path "hpdv3:path/to/dataset.json" \
    --output_path path/to/filtered_samples.json \
    --batch_size 8 \
    --max_new_tokens 2048 \
    --use_cot \
    --task_instruction "Your task instruction here" \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.9
```

**Key Features:**
- Uses vLLM for efficient inference
- Supports multi-GPU inference via tensor parallelism
- Extracts CoT reasoning from generated text
- Filters samples where model prediction matches ground truth
- Saves filtered samples and statistics

#### `rejection_sampling_inference_t2v.py`

Performs inference on T2V datasets and filters correctly predicted samples.

**Usage:**

```bash
python rejection_sampling_inference_t2v.py \
    --model_path path/to/model \
    --data_path "rapidata-t2v:path/to/dataset1.parquet,rapidata-t2v:path/to/dataset2.parquet" \
    --output_path path/to/filtered_samples.json \
    --batch_size 8 \
    --max_new_tokens 2048 \
    --use_cot \
    --task_instruction "Your task instruction here" \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.9 \
    --video_fps 2.0
```

**Key Features:**
- Supports multiple T2V datasets (comma-separated)
- Computes ground truth preference based on video quality scores (Alignment + Coherence + Preference)
- Processes videos at specified FPS
- Extracts CoT reasoning from generated text

### Conversion Scripts

#### `convert_to_rejection_sampling_data_t2i.py`

Converts filtered T2I samples to training format.

**Usage:**

```bash
python convert_to_rejection_sampling_data_t2i.py \
    --filtered_samples_path path/to/filtered_samples.json \
    --output_path path/to/training_data.json \
    --data_root path/to/dataset/root \
    --task_instruction "Your task instruction here"
```

**Output Format:**

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Task instruction with prompt"
    },
    {
      "from": "gpt",
      "value": "<think>\nReasoning process...\n</think>\n<answer>Image 1 is better</answer>"
    }
  ],
  "images": [
    "path/to/image1.jpg",
    "path/to/image2.jpg"
  ]
}
```

#### `convert_to_rejection_sampling_data_t2v.py`

Converts filtered T2V samples to training format.

**Usage:**

```bash
python convert_to_rejection_sampling_data_t2v.py \
    --filtered_samples_path path/to/filtered_samples.json \
    --output_path path/to/training_data.json \
    --task_instruction "Your task instruction here" \
    --video_fps 2.0
```

**Output Format:**

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Task instruction with prompt"
    },
    {
      "from": "gpt",
      "value": "<think>\nReasoning process...\n</think>\n<answer>Video 1 is better</answer>"
    }
  ],
  "images": [
    "path/to/video1.mp4",
    "path/to/video2.mp4"
  ],
  "video_fps": 2.0
}
```

## Environment Variables

The scripts support the following environment variables:

- `GPUS_PER_NODE`: Number of GPUs to use (default: 2)
- `NNODES`: Number of nodes (default: 1)
- `NODE_RANK`: Current node rank (default: 0)
- `MASTER_ADDR`: Master node address (default: "localhost")
- `MASTER_PORT`: Master port (default: 29500)

## Requirements

- PyTorch
- vLLM
- Transformers
- LightRFT
- loguru
- tqdm

## Task Instructions

### T2I Task Instruction

The default T2I task instruction asks the model to:
1. Evaluate two images on multiple dimensions (semantic consistency, aesthetics, authenticity)
2. Provide scores (1-10) for each dimension with rationale
3. Calculate total scores by summing dimension scores
4. Use Chain-of-Thought reasoning within `<think>` tags
5. Output the final answer in `<answer>` tags

### T2V Task Instruction

The default T2V task instruction asks the model to:
1. Evaluate two videos on multiple dimensions (semantic consistency, temporal coherence, authenticity)
2. Provide scores (1-10) for each dimension with rationale
3. Calculate total scores by summing dimension scores
4. Use Chain-of-Thought reasoning within `<think>` tags
5. Output the final answer in `<answer>` tags

## Notes

- The inference scripts use vLLM for efficient batched inference
- Tensor parallelism is used for multi-GPU inference
- Ground truth preferences are determined from dataset annotations
- Only correctly predicted samples are used for training
- The training format is compatible with the GRM training pipeline

## Troubleshooting

### Out of Memory (OOM) Issues

1. Reduce `INFERENCE_BATCH_SIZE` or `MICRO_BATCH_SIZE`
2. Reduce `gpu_memory_utilization`
3. Enable gradient checkpointing (already enabled in training scripts)
4. Reduce `MAX_LENGTH` or `prompt_max_len`

### Low Accuracy

1. Check if the task instruction matches the training instruction
2. Verify that the model is properly trained on similar tasks
3. Check if the dataset format is correct
4. Review sample predictions in the generated text

### Dataset Loading Issues

1. Verify dataset paths are correct
2. Check dataset format (JSON for T2I, Parquet for T2V)
3. Ensure `DATA_ROOT` is set correctly for T2I
4. Check image/video file paths in the dataset
