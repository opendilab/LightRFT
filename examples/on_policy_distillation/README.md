# On-Policy Distillation (OPD) for LightRFT

On-policy knowledge distillation enables smaller student models to learn from larger teacher models during reinforcement learning training.

## Overview

| Aspect | Description |
|--------|-------------|
| **Teacher** | Large model providing token-level log probability supervision |
| **Student** | Smaller model being trained to match teacher's distribution |
| **Training** | On-policy: teacher evaluates student's actual generated responses |
| **Reward** | Teacher's log probabilities serve as the learning signal |

## Quick Start

### 1. Start Teacher Model Server

```bash
# Launch SGLang server for teacher model
CUDA_VISIBLE_DEVICES=7 python3 -m sglang.launch_server \
    --model-path "Qwen/Qwen2.5-7B-Instruct" \
    --host 0.0.0.0 \
    --port 13141 \
    --tp 1 \
    --mem-fraction-static 0.6
```

### 2. Run Training

```bash
bash examples/on_policy_distillation/run_opd_qwen_2.sh
```

Or manually:

```bash
torchrun --nproc-per-node 2 examples/gsm8k_geo3k/train_colocate.py \
    --pretrain "Qwen/Qwen2.5-0.5B-Instruct" \
    --advantage_estimator "on_policy_distillation" \
    --remote_rm_url "http://127.0.0.1:13141/generate" \
    --reward_pretrain "" \
    --n_samples_per_prompt 4 \
    --actor_learning_rate 1e-6 \
    --init_kl_coef 0.01 \
    --num_episodes 30
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OPD Training Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Generate    Prompt ──► [Student] ──► Response           │
│                                                             │
│  2. Evaluate    [Prompt + Response] ──► [Teacher Server]    │
│                            │                                │
│                            ▼                                │
│                    Teacher Log Probs                        │
│                                                             │
│  3. Compute     Advantage = Teacher_logp - Student_logp     │
│                                                             │
│  4. Update      Student ◄── Policy Gradient Loss            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Advantage Calculator

**File**: `lightrft/trainer/advantage_calculator.py`

```python
class OnPolicyDistillationCalculator(AdvantageCalculator):
    def compute(self, experience, ...):
        teacher_log_probs = experience.info["teacher_log_probs"]
        student_log_probs = experience.action_log_probs

        # Reverse KL: encourages student to match teacher
        reverse_kl = student_log_probs - teacher_log_probs
        advantages = -reverse_kl  # = teacher - student

        return advantages, returns, {"opd_reverse_kl": reverse_kl}
```

### 2. Teacher Log Prob Fetcher

**File**: `examples/on_policy_distillation/on_policy_distillation_reward.py`

- Async HTTP requests to teacher server
- Supports SGLang and vLLM response formats
- Automatic retry with exponential backoff

### 3. Experience Maker Integration

**File**: `lightrft/trainer/fast_exp_maker.py`

- `--remote_rm_url` is used as teacher URL (not reward model) when `--advantage_estimator "on_policy_distillation"`
- Teacher log probs stored in `experience.info["teacher_log_probs"]`
- OPD metrics (`opd_reverse_kl_mean/std/min/max`) logged to wandb

## Configuration

### Required Arguments

| Argument | Value | Description |
|----------|-------|-------------|
| `--advantage_estimator` | `"on_policy_distillation"` | Enable OPD mode |
| `--remote_rm_url` | `"http://host:port/generate"` | Teacher server URL |
| `--reward_pretrain` | `""` | Empty (no reward model needed) |

### Recommended Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--n_samples_per_prompt` | 4 | Responses per prompt |
| `--actor_learning_rate` | 1e-6 | Student learning rate |
| `--init_kl_coef` | 0.01 | KL regularization coefficient |
| `--num_episodes` | 30 | Training episodes |

## Teacher Server Formats

### SGLang (Recommended)

```json
{
    "meta_info": {
        "input_token_logprobs": [[logprob, rank, token], ...]
    }
}
```

### vLLM

```json
{
    "token_logprobs": [logprob1, logprob2, ...]
}
```

## Monitoring

### Logged Metrics

| Metric | Description |
|--------|-------------|
| `opd_reverse_kl_mean` | Average KL(student \|\| teacher) |
| `opd_reverse_kl_std` | Standard deviation of reverse KL |
| `advantages_mean` | Average advantage (should center ~0) |
| `policy_loss` | Should decrease during training |

### Console Output

```
📊 Detailed Step Statistics
============================================================
🎁 Total Reward:     0.0000 ± 0.0000 (placeholder for OPD)
📈 Advantages:       0.0012 ± 0.8234 (...)
🎓 OPD Reverse KL:   0.1523 ± 0.0891 (...)
============================================================
```

## Troubleshooting

### Teacher Server Issues

```bash
# Check if port is in use
lsof -i :13141

# Check GPU availability
nvidia-smi

# Reduce memory if OOM
--mem-fraction-static 0.5
```

### Training OOM

```bash
--micro_train_batch_size 2
--micro_rollout_batch_size 2
--gradient_checkpointing
--zero_stage 3
```

### Slow Convergence

```bash
--n_samples_per_prompt 8
--actor_learning_rate 5e-7
--num_episodes 50
```

## Comparison with Other Methods

| Method | Reward Signal | Mode | Requires RM |
|--------|--------------|------|-------------|
| GRPO | Task-specific reward | Online | Yes |
| DPO | Preference pairs | Offline | No |
| **OPD** | Teacher log probs | Online | No (uses teacher) |

### Advantages

- No separate reward model training required
- Token-level supervision (finer than sequence-level)
- On-policy: adapts to student's changing distribution
- Works for any task with a good teacher model

### Limitations

- Requires running teacher model (inference overhead)
- Student cannot exceed teacher's capabilities
- Needs sufficient compute for teacher inference

## File Structure

```
examples/on_policy_distillation/
├── README.md                           # This file
├── README_zh.md                        # Chinese version
├── run_opd_qwen_2.sh                   # Training script
└── on_policy_distillation_reward.py   # Teacher logprob fetcher
```

## References

- [LightRFT Documentation](../../README.md)
- [Advantage Calculator Source](../../lightrft/trainer/advantage_calculator.py)
- [Fast Experience Maker Source](../../lightrft/trainer/fast_exp_maker.py)
