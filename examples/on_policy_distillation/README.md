# On-Policy Distillation for LightRFT

This directory contains a complete implementation of on-policy knowledge distillation for LightRFT, enabling smaller student models to learn from larger teacher models during reinforcement learning.

## Overview

On-policy distillation is a technique where:
- A **teacher model** (large, powerful) provides token-level supervision
- A **student model** (small, efficient) learns to match the teacher's probability distribution
- Training happens **on-policy**: teacher evaluates student's actual generated responses
- No separate reward model is needed - teacher's log probabilities serve as the learning signal

## Quick Start

### 1. Installation

Ensure you have LightRFT installed with SGLang support:

```bash
pip install lightrft
pip install sglang  # For teacher model inference server
```

### 2. Prepare Your Dataset

Your dataset should be in JSONL format with prompts:

```json
{"prompt": "Solve: What is 2 + 2?"}
{"prompt": "Explain the theory of relativity."}
```

### 3. Run Training

```bash
# Edit the configuration in run_opd_qwen.sh
bash examples/on_policy_distillation/run_opd_qwen.sh
```

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Training Pipeline                     │
└─────────────────────────────────────────────────────────┘

1. Student generates responses:
   Prompt → [Student Model] → Response

2. Teacher evaluates responses:
   [Prompt + Response] → [Teacher Model] → Teacher Log Probs

3. Advantage calculation:
   Advantage = Teacher Log Probs - Student Log Probs

4. Student optimization:
   Update student to increase probability where teacher has high probability
```

### Key Components

#### 1. OnPolicyDistillationCalculator (`lightrft/trainer/advantage_calculator.py`)

Computes advantages using teacher log probabilities:

```python
advantage = teacher_log_probs - student_log_probs
```

This encourages the student to match the teacher's token-level distribution.

#### 2. Teacher Logprob Function (`on_policy_distillation_reward.py`)

Queries the teacher model inference server to get log probabilities:

```python
teacher_log_probs = get_teacher_logprobs_sync(
    teacher_url=teacher_url,
    sequences=sequences,
    response_lengths=response_lengths
)
```

#### 3. Experience Maker Integration

Modified `experience_maker.py` to:
- Query teacher model during experience collection
- Store teacher log probs in `experience.info["teacher_log_probs"]`
- Use OnPolicyDistillationCalculator for advantage computation

## Configuration

### Required Arguments

```bash
--advantage_estimator "on_policy_distillation"  # Enable on-policy distillation
--remote_rm_url "http://localhost:13141/generate"  # Teacher model URL
--pretrain "Qwen/Qwen2.5-0.5B-Instruct"  # Student model
```

### Recommended Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_samples_per_prompt` | 4 | Number of responses per prompt |
| `actor_learning_rate` | 1e-6 | Learning rate for student |
| `init_kl_coef` | 0.01 | KL coefficient for regularization |
| `num_episodes` | 30 | Number of training episodes |

## Example Use Cases

### 1. Math Reasoning (GSM8K)

Train a small model to solve math problems like a larger model:

```bash
TEACHER_MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
STUDENT_MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
DATASET_PATH="path/to/gsm8k.jsonl"
```

### 2. General Instruction Following

Distill instruction-following capabilities:

```bash
TEACHER_MODEL_PATH="Qwen/Qwen2.5-14B-Instruct"
STUDENT_MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
DATASET_PATH="path/to/instruction_data.jsonl"
```

### 3. Domain-Specific Tasks

Transfer domain expertise from a fine-tuned teacher to a smaller student:

```bash
TEACHER_MODEL_PATH="path/to/finetuned_teacher"
STUDENT_MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
DATASET_PATH="path/to/domain_data.jsonl"
```

## Technical Details

### Advantage Computation

The advantage estimator computes:

```python
# Get teacher and student log probs for each token
teacher_log_probs = experience.info["teacher_log_probs"]
student_log_probs = experience.action_log_probs

# Compute advantage (encourages matching teacher)
advantages = teacher_log_probs - student_log_probs

# Apply action mask (only consider generated tokens)
advantages = advantages * experience.action_mask

# Optional: normalize advantages
if config.advantages_norm:
    advantages = (advantages - mean) / (std + 1e-8)
```

### Teacher Server Format

The implementation supports both SGLang and vLLM formats:

**SGLang format:**
```json
{
    "meta_info": {
        "input_token_logprobs": [[logprob, rank, token], ...]
    }
}
```

**vLLM format:**
```json
{
    "token_logprobs": [logprob1, logprob2, ...]
}
```

## Performance Tips

### 1. GPU Memory Optimization

- Run teacher on separate GPU(s) from training
- Use tensor parallelism for large teachers: `--tp 2`
- Adjust memory fraction: `--mem-fraction-static 0.6`

### 2. Training Speed

- Increase `n_samples_per_prompt` for more stable gradients (but slower)
- Use larger batch sizes if memory permits
- Enable gradient checkpointing for memory-intensive models

### 3. Convergence

- Start with lower learning rate (1e-6) for stable distillation
- Use KL coefficient to prevent student from diverging too far
- Monitor teacher-student log prob difference in W&B

## Troubleshooting

### Teacher server won't start

```bash
# Check GPU availability
nvidia-smi

# Check if port is already in use
lsof -i :13141

# Try different memory fraction
--mem-fraction-static 0.5
```

### Training OOM (Out of Memory)

```bash
# Reduce batch sizes
--micro_train_batch_size 2
--micro_rollout_batch_size 2

# Enable gradient checkpointing
--gradient_checkpointing

# Use ZeRO-3 optimization
--zero_stage 3
```

### Slow convergence

```bash
# Increase samples per prompt
--n_samples_per_prompt 8

# Adjust learning rate
--actor_learning_rate 5e-7

# Increase training episodes
--num_episodes 50
```

## Comparison with Other Methods

| Method | Reward Signal | Offline/Online | Requires RM |
|--------|--------------|----------------|-------------|
| GRPO | Task-specific reward | Online | Yes |
| DPO | Preference pairs | Offline | No |
| **On-Policy Distillation** | Teacher log probs | Online | No (uses teacher) |

**Advantages:**
- ✅ No need to train a separate reward model
- ✅ Token-level supervision (finer-grained than sequence-level rewards)
- ✅ On-policy: adapts to student's changing distribution
- ✅ Works for any task where you have a good teacher model

**Limitations:**
- ⚠️ Requires a teacher model (inference overhead)
- ⚠️ Student cannot exceed teacher's capabilities
- ⚠️ Needs sufficient compute for teacher inference

## References

- [Original slime implementation](https://github.com/OpenRLHF/slime)
- [LightRFT Documentation](../../README.md)
- [On-Policy Distillation Paper](https://arxiv.org/abs/XXXX.XXXXX)

## Citation

If you use this implementation, please cite:

```bibtex
@software{lightrft_opd,
  title={On-Policy Distillation for LightRFT},
  author={LightRFT Team},
  year={2024},
  url={https://github.com/yourusername/LightRFT}
}
```

## Support

For questions or issues:
- Open an issue on GitHub
- Check the [FAQ](../../docs/source/best_practice/faq.md)
- Review [troubleshooting guide](../../docs/source/best_practice/troubleshooting.md)
