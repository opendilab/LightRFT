<div align="center">

# SafeWork-R1 Training Code

Official training code implementation for SafeWork-R1 models using the SafeLadder framework.

[🤗Huggingface Models](https://huggingface.co/collections/AI45Research/safework-r1) • [📜Technical Report](https://arxiv.org/abs/2507.18576) • [💬Online Chat](https://safework-r1.ai45.shlab.org.cn/)

</div>

## Overview

This repository contains the official training code for **SafeWork-R1**, a cutting-edge multimodal reasoning model that demonstrates the coevolution of safety and general intelligence under the AI-45° Law.

The training implementation is built upon the **SafeLadder framework**, featuring:
- **Multi-stage reinforcement learning** pipeline with progressive safety alignment
- **Multi-principled verifiers** (Safety, Value, Knowledge) for robust reward signals
- **Group Relative Policy Optimization (GRPO)** for efficient training
- **Co-located reward models** for multi-dimensional evaluation

## Key Features

### Training Capabilities

- ✅ **Multi-Modal Support**: Both text-only and vision-language models (Qwen2.5-VL, InternVL3, DeepSeek-R1)
- ✅ **Multiple Reward Models**: Value, Safety, Knowledge, Normal, and General verifiers
- ✅ **Flexible Distributed Training**: DeepSpeed ZeRO (Stage 1/2/3) and PyTorch FSDP support
- ✅ **Memory Optimization**: Meta device initialization, gradient checkpointing, CPU offloading
- ✅ **Inference Engines**: vLLM and SGLang integration for efficient generation
- ✅ **EMA Support**: Exponential Moving Average for model stability
- ✅ **Advanced Techniques**: DAPO (Dynamic sampling and overlong buffer penalties)

### SafeLadder Framework

The training follows the SafeLadder multi-stage pipeline:

1. **CoT-SFT**: Chain-of-Thought supervised fine-tuning
2. **M³-RL**: Multi-principled Multi-model Multi-turn reinforcement learning
3. **Safe-and-Efficient RL**: Safety-focused optimization with efficiency constraints
4. **Deliberative Search RL**: Step-level verification with search mechanisms

## Project Structure

```
safework_t1/
├── train_colocate.py              # Main training script for GRPO with co-located RMs
├── reward_models.py               # Reward model implementations (Value, Safety, Knowledge)
├── reward_models_utils.py         # Utilities for loading and managing reward models
├── test_reward_models.py          # Testing script for reward models
├── run_grpo_kg_qwenvl.sh         # Training script for Knowledge + General RMs (Qwen2.5-VL)
├── run_grpo_svki_fsdp_deepseek.sh # Training script for Safety + Value + Knowledge (DeepSeek-70B)
└── run_grpo_svkng_fsdp_qwenvl.sh # Training script for all RMs (Qwen2.5-VL)
```

## Installation

### Prerequisites

- Python >= 3.8
- CUDA >= 11.8 (for GPU training)
- 8x A100 (80GB) or equivalent GPUs recommended

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/AI45Research/SafeWork-R1.git
cd SafeWork-R1/training_code
```

2. **Install dependencies**:
```bash
# Install core training framework
pip install -e .

# Install instruction-following reward library
pip install git+https://github.com/puyuan1996/if_reward.git

# Install additional dependencies
pip install zhconv nltk
python -m nltk.downloader punkt punkt_tab
```

3. **Configure environment** (if behind proxy):
```bash
export http_proxy="http://your-proxy:port"
export https_proxy="http://your-proxy:port"
```

## Quick Start

### 1. Prepare Your Data

Organize your training data in JSONL format:
```json
{"conversations": [{"from": "human", "value": "prompt with optional <image>"}, {"from": "gpt", "value": "response"}]}
```

### 2. Prepare Reward Models

Download the SafeWork-R1 reward models:
- [SafeWork-RM-Safety-7B](https://huggingface.co/AI45Research/SafeWork-RM-Safety-7B)
- [SafeWork-RM-Value-72B](https://huggingface.co/AI45Research/SafeWork-RM-Value-72B)
- [SafeWork-RM-Knowledge-72B](https://huggingface.co/AI45Research/SafeWork-RM-Knowledge-72B)

### 3. Run Training

#### Option A: Quick Start with Qwen2.5-VL-7B

```bash
bash run_grpo_kg_qwenvl.sh
```

This script trains a Qwen2.5-VL-7B model with Knowledge and General reward models.

#### Option B: Full Training with All Verifiers (Qwen2.5-VL)

```bash
bash run_grpo_svkng_fsdp_qwenvl.sh
```

This script uses all reward models (Safety, Value, Knowledge, Normal, General) for comprehensive alignment.

#### Option C: DeepSeek-R1-70B Training

```bash
bash run_grpo_svki_fsdp_deepseek.sh
```

This script trains the DeepSeek-R1-Distill-Llama-70B model with Safety, Value, and Knowledge verifiers.

### 4. Monitor Training

Training logs and checkpoints will be saved to the output directory specified in the script. You can monitor training progress via:
- **Weights & Biases**: Automatically logged if wandb is configured
- **Console logs**: Training loss, reward scores, KL divergence
- **Checkpoint files**: Model states saved at regular intervals

## Configuration

### Key Training Parameters

Edit the training scripts to customize these parameters:

```bash
# RL Training Parameters
N_SAMPLES=8          # Number of responses per prompt
EPISODE=3            # Total training episodes
LR=1e-6              # Learning rate
MAX_LENGTH=8192      # Maximum sequence length

# Batch Sizes
TBS=32               # Total training batch size
RBS=64               # Total rollout batch size

# Reward Model Weights
RM_VALUE_WEIGHT=1.0      # Weight for value verifier
RM_SAFETY_WEIGHT=1.0     # Weight for safety verifier
RM_KNOWLEDGE_WEIGHT=1.0  # Weight for knowledge verifier
```

### Distributed Training Strategy

**DeepSpeed ZeRO**:
```bash
--zero_stage 2 \           # ZeRO optimization stage (1/2/3)
--bf16 \                   # Use BF16 mixed precision
--gradient_checkpointing   # Enable gradient checkpointing
```

**PyTorch FSDP**:
```bash
--fsdp \                   # Enable FSDP mode
--bf16 \                   # Use BF16 mixed precision
--gradient_checkpointing   # Enable gradient checkpointing
```

### Reward Model Configuration

Specify reward models in `reward_models_utils.py` or via command-line:

```python
RECIPE = {
    "value": {
        "path": "AI45Research/SafeWork-RM-Value-72B",
        "weight": 1.0,
        "use_engine": False  # Use HF inference (True for SGLang)
    },
    "safety": {
        "path": "AI45Research/SafeWork-RM-Safety-7B",
        "weight": 1.0,
        "use_engine": True   # Use SGLang for faster inference
    },
    # ... more reward models
}
```

## Advanced Usage

### Custom Reward Models

To add your own reward model:

1. **Implement the reward model class** in `reward_models.py`:
```python
class MyCustomRM(nn.Module):
    def forward(self, input_ids, attention_mask, **kwargs):
        # Your reward computation logic
        return scores
```

2. **Register in reward_models_utils.py**:
```python
RECIPE["custom"] = {
    "path": "path/to/your/model",
    "weight": 1.0,
    "class": "MyCustomRM"
}
```

3. **Update training script** to include your reward model.

### Multi-Turn Training

Enable multi-turn RL training with conversation history:

```bash
--multi_turn \
--max_turns 3 \
--turn_separator "<|end_of_turn|>"
```

### EMA Model

Enable Exponential Moving Average for training stability:

```bash
--enable_ema \
--ema_decay 0.999 \
--ema_update_interval 10
```

## Trained Models

Using this training code, we have successfully trained the following SafeWork-R1 models:

| Model | Base Model | Parameters | Link |
|-------|------------|------------|------|
| SafeWork-R1 | Qwen2.5-VL-72B | 72B | [🤗 HF](https://huggingface.co/AI45Research/SafeWork-R1) |
| SafeWork-R1-InternVL3-78B | InternVL3-78B | 78B | [🤗 HF](https://huggingface.co/AI45Research/SafeWork-R1-InternVL3-78B) |
| SafeWork-R1-DeepSeek-70B | DeepSeek-R1-Distill-Llama-70B | 70B | [🤗 HF](https://huggingface.co/AI45Research/SafeWork-R1-DeepSeek-70B) |
| SafeWork-R1-Qwen2.5VL-7B | Qwen2.5-VL-7B | 7B | [🤗 HF](https://huggingface.co/AI45Research/SafeWork-R1-Qwen2.5VL-7B) |

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size (`TBS`, `RBS`)
   - Enable gradient checkpointing
   - Use DeepSpeed ZeRO-3 or FSDP CPU offloading
   - Reduce `MAX_LENGTH`

2. **Reward Model Loading Errors**
   - Verify reward model paths are correct
   - Ensure sufficient GPU memory for all reward models
   - Use `--rm_use_engine` to offload RMs to SGLang

3. **Slow Training**
   - Enable SGLang engine for reward models (`use_engine: True`)
   - Use vLLM for faster generation
   - Increase batch size if memory allows
   - Check network bandwidth for data loading

4. **Wandb Upload Failures**
   - Configure proxy settings if behind firewall
   - Use `--wandb_mode offline` for offline logging
   - Check wandb API key: `wandb login`

## Performance Tips

- **Use mixed precision (BF16)** for faster training on A100/H100 GPUs
- **Enable flash attention** if your model supports it
- **Use SGLang engine** for reward models to reduce inference overhead
- **Tune gradient accumulation** to maximize GPU utilization
- **Profile your training** to identify bottlenecks

## Citation

If you use this training code, please cite:

```bibtex
@misc{lab2025safework,
  title={SafeWork-R1: Coevolving Safety and Intelligence under the AI-45 Law},
  author={Lab, Shanghai AI and Bao, Yicheng and Chen, Guanxu and Chen, Mingkang and Chen, Yunhao and Chen, Chiyu and Chen, Lingjie and Chen, Sirui and Chen, Xinquan and Cheng, Jie and others},
  journal={arXiv preprint arXiv:2507.18576},
  year={2025}
}
```

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](../../LICENSE) for details.

## Acknowledgments

- The SafeLadder framework builds upon research in safe RLHF and multi-principled alignment
- We thank the open-source community for DeepSpeed, FSDP, vLLM, and SGLang
- Special thanks to the Qwen, InternVL, and DeepSeek teams for their excellent base models

## Contact

For questions or issues:
- Open an issue on [GitHub](https://github.com/AI45Research/SafeWork-R1/issues)
- Visit our [project page](https://safework-r1.ai45.shlab.org.cn/)
- Check the [technical report](https://arxiv.org/abs/2507.18576)
