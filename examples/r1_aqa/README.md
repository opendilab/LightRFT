# R1-AQA on LightRFT: Audio Question Answering with GRPO

This example migrates [R1-AQA](https://github.com/xiaomi-research/r1-aqa) (Audio Question Answering via GRPO on Qwen2-Audio) into the [LightRFT](https://github.com/opendilab/LightRFT) training framework.

## Overview

R1-AQA applies Group Relative Policy Optimization (GRPO) to Qwen2-Audio-7B-Instruct for audio question-answering tasks. The training uses rule-based rewards (accuracy + format) on the AVQA dataset. This LightRFT example faithfully reproduces the core training pipeline while leveraging LightRFT's distributed training infrastructure, GRPO implementation, and reward processing system.

## File Structure

```
examples/r1_aqa/
├── data_preprocess/
│   └── avqa.py                          # Convert R1-AQA JSONL → LightRFT parquet
├── audio_dataset.py                    # Audio multimodal pipeline extensions
│   ├── AudioPromptDataset               # Dataset class for audio prompts
│   ├── AudioMultimodalProcessor         # Audio feature extraction for pipeline
│   └── patch_* functions                # Pipeline monkey-patches for audio
├── reward_models_utils.py               # Rule-based reward (accuracy + format)
├── train_colocate.py                    # GRPO training entry point
├── eval_mmau.py                         # MMAU test-mini evaluation
├── run_grpo_r1_aqa_qwen2_audio_7b.sh   # Training launch script
└── README.md                            # This file
```

## Quick Start

### Prerequisites

```bash
# Core dependencies (should already be installed with LightRFT)
pip install transformers torch deepspeed

# Audio dependencies
pip install librosa soundfile

# Optional: math_verify for symbolic answer verification
pip install math_verify
```

### Step 1: Prepare the AVQA Dataset

First, obtain the R1-AQA AVQA training data (JSONL format). See [R1-AQA README](https://github.com/xiaomi-research/r1-aqa) for the data conversion from the original AVQA dataset.

The JSONL file should have one JSON object per line with fields:
```json
{
  "id": 183,
  "question_text": "What happened in the video?",
  "multi_choice": ["motorboat", "Yacht consignment", "Sailboat set sail", "Consignment car"],
  "answer": 1,
  "dataset_name": "AVQA",
  "audio_path": "path/to/-HG3Omg_89c_30.wav"
}
```

Convert to LightRFT format:
```bash
python examples/r1_aqa/data_preprocess/avqa.py \
    --input_jsonl data/AVQA/train_qa.data \
    --audio_dir data/AVQA/audios \
    --local_save_dir ~/data/avqa_lightrft
```

### Step 2: Configure and Run Training

Edit the shell script to set your paths:
```bash
# In run_grpo_r1_aqa_qwen2_audio_7b.sh:
PATH_TO_YOUR_BASE_MODEL="Qwen/Qwen2-Audio-7B-Instruct"
PATH_TO_YOUR_AVQA_DATASET="/path/to/your/avqa_lightrft"
```

Run training:
```bash
bash examples/r1_aqa/run_grpo_r1_aqa_qwen2_audio_7b.sh
```

### Step 3: Dry-Run (Minimal Test)

For a quick test with minimal resources:
```bash
# In the shell script, change:
EPISODE=1
RBS=4
TBS=32
N_SAMPLES=4
MLP_WORKER_GPU=1
ENGINE_TP=1
ENGINE_MEM_UTIL=0.5
```

Expected output for dry-run:
- Training log with reward metrics (accuracy_reward, format_reward)
- Checkpoint saved to `results/lightrft-r1-aqa-grpo-training/...`
- wandb logs (if configured)

### Step 4: Evaluate on MMAU Test-Mini

```bash
# Using HuggingFace inference
python examples/r1_aqa/eval_mmau.py \
    --model_path results/lightrft-r1-aqa-grpo-training/<your_run>/  \
    --data_file /path/to/mmau-test-mini.json \
    --audio_dir /path/to/mmau/audio \
    --out_file results/res_mmau_mini.json \
    --engine hf

# Using vLLM for faster inference
python examples/r1_aqa/eval_mmau.py \
    --model_path results/lightrft-r1-aqa-grpo-training/<your_run>/  \
    --data_file /path/to/mmau-test-mini.json \
    --audio_dir /path/to/mmau/audio \
    --out_file results/res_mmau_mini.json \
    --engine vllm

# Run MMAU official evaluation
python /path/to/mmau/evaluation.py --input results/res_mmau_mini.json
```

## Batch Size Constraints

LightRFT enforces specific batch size relationships for GRPO:

```
train_batch_size >= rollout_batch_size × n_samples_per_prompt
```

For R1-AQA defaults (n_samples=8):
| Config | rollout_batch_size | n_samples | train_batch_size | Valid? |
|---|---|---|---|---|
| Default | 16 | 8 | 128 | 128 >= 16×8=128 ✓ |
| Dry-run | 4 | 4 | 32 | 32 >= 4×4=16 ✓ |
| 1-GPU | 4 | 4 | 16 | 16 >= 4×4=16 ✓ |

## Common Issues

### 1. Audio Path Not Found
Ensure `audio_dir` in the preprocessing script points to the directory containing `.wav` files. Audio paths in the JSONL can be relative or absolute.

### 2. VRAM / OOM
- Reduce `MICRO_TRAIN` and `MICRO_ROLLOUT` (e.g., 1)
- Reduce `N_SAMPLES` (e.g., 4 instead of 8)
- Enable `--gradient_checkpointing` and `--adam_offload`
- Reduce `ENGINE_MEM_UTIL` (e.g., 0.4)

### 3. Inference Engine Issues
- Qwen2-Audio requires vLLM or SGLang with audio model support
- Check that your vLLM version supports `Qwen2AudioForConditionalGeneration`
- For SGLang, ensure audio multimodal support is available

### 4. Output Key Mismatch with MMAU
The evaluation script outputs `model_prediction` which matches MMAU's expected field. If using a different evaluation script, check the expected field name.

### 5. Think Mode
R1-AQA supports an optional `<think></think>` mode. To enable:
```bash
# In data preprocessing:
python examples/r1_aqa/data_preprocess/avqa.py --enable_think ...
```
The reward function automatically handles both modes. When `enable_think=True`, the format reward also checks for `<think>...</think>` tags.

## Design Decisions

### 1. Reward Summation (not Weighting)
R1-AQA sums accuracy and format rewards (max=2.0) while GSM8K/Geo3K in LightRFT uses weighted combination (0.9×accuracy + 0.1×format, max=1.0). We keep R1-AQA's summation to ensure identical reward signal. The GRPO normalization handles the scale difference.

### 2. Audio Pipeline via Image Slot
LightRFT's VL pipeline is built for images/videos. We repurpose the image data slots to carry audio data:
- `pixel_values` → `input_features` (audio features)
- `image_grid_thw` → `feature_attention_mask`
- `raw_images` → raw audio tuples `(np.array, sr)`
- `multi_modal_data["image"]` → `multi_modal_data["audio"]`

This is done via targeted monkey patches in `audio_dataset.py` rather than modifying core LightRFT code.

### 3. ActorAL (Audio Language Actor)
Qwen2-Audio uses `Qwen2AudioForConditionalGeneration` (not `AutoModelForVision2Seq`), and its forward pass expects `audio_values` instead of `pixel_values` + `image_grid_thw`. We use `ActorAL` from `lightrft.models.actor_al`, which natively supports Qwen2-Audio's parameter interface.

### 4. Chat Template
R1-AQA embeds audio URLs in the chat message content as `{"type": "audio", "audio_url": path}`. We preserve this format and use the Qwen2-Audio processor's `apply_chat_template` to convert it to the correct token format with audio placeholders.

### 5. No `<think>` by Default
R1-AQA's README notes that explicit reasoning did not show significant benefits for AQA tasks. The `<think>` mode is supported but disabled by default, matching R1-AQA's default configuration.

## Verification Checklist

- [ ] Data preprocessing reads JSONL and outputs valid parquet
- [ ] Reward function produces correct scores for sample inputs
- [ ] Training script launches without errors (dry-run)
- [ ] GRPO advantage calculation works (check wandb/logs for reward metrics)
- [ ] Evaluation script produces MMAU-compatible output format
- [ ] Model prediction field matches MMAU evaluation.py expectations
