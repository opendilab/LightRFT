# On-Policy Distillation (OPD)

On-Policy Distillation ([Blog](https://thinkingmachines.ai/blog/on-policy-distillation/)) lets a small student model learn from a large teacher model during RL training via token-level log probability supervision. Unlike GRPO/PPO, OPD requires no separate reward model — the teacher itself provides the supervision signal at every token position.

```
Pipeline:  Prompt ──► [Student] ──► Response ──► [Teacher Server] ──► Teacher LogProbs
           Advantage = Teacher_logp - Student_logp ──► Policy gradient update
```

## Quick Start

### Option 1: All-in-One (Recommended)

`run_opd_qwen.sh` handles teacher server startup, health checking, and cleanup internally:

```bash
# Run directly (edit model/dataset paths in the script first)
bash examples/on_policy_distillation/run_opd_qwen.sh

# Or override via environment variables
TEACHER_MODEL_PATH=/path/to/teacher \
STUDENT_MODEL_PATH=/path/to/student \
DATASET_PATH=/path/to/data.jsonl \
bash examples/on_policy_distillation/run_opd_qwen.sh

# Enable hybrid mode (GRPO task reward + OPD KL signal)
USE_TASK_REWARD=true bash examples/on_policy_distillation/run_opd_qwen.sh
```

### Option 2: Separate Deployment

For running teacher and training in different terminals or on different machines:

```bash
# Terminal 1: Start teacher server
bash examples/on_policy_distillation/start_teacher.sh

# Terminal 2: Start training (after teacher is ready)
TEACHER_URL=http://127.0.0.1:13141/generate \
bash examples/on_policy_distillation/start_training.sh
```

## Environment Variables

### Model & Data

| Variable | Default | Description |
|----------|---------|-------------|
| `TEACHER_MODEL_PATH` | `Qwen/Qwen2.5-7B-Instruct` | Teacher model path |
| `STUDENT_MODEL_PATH` | `Qwen/Qwen2.5-0.5B-Instruct` | Student model path |
| `DATASET_PATH` | — | Training dataset path (jsonl format) |

### Training Hyperparameters

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_TASK_REWARD` | `false` | `true` enables GRPO task reward + OPD KL hybrid mode |
| `OPD_KL_COEF` | `1.0` | OPD KL loss coefficient |
| `N_SAMPLES` | `8` | Samples per prompt |
| `LR` | `5e-7` | Student learning rate |
| `EPISODE` | `30` | Training episodes |

### Teacher Server (Separate Deployment)

| Variable | Default | Description |
|----------|---------|-------------|
| `TEACHER_GPU` | `0` | GPU for teacher model |
| `TEACHER_PORT` | `13141` | Teacher server port |
| `TEACHER_URL` | — | Teacher server address (required by `start_training.sh`) |

## Monitoring

The following metrics are logged to W&B during training:

| Metric | Description |
|--------|-------------|
| `opd_reverse_kl_mean` | Average KL(student \|\| teacher), should decrease |
| `opd_reverse_kl_std` | Standard deviation of reverse KL |
| `advantages_mean` | Average advantage (should center ~0) |
| `policy_loss` | Policy loss, should decrease |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Teacher server won't start | `lsof -i :13141` to check port; `nvidia-smi` to check GPU |
| Teacher OOM | Lower `MEM_FRACTION` (default 0.7) |
| Training OOM | Reduce `MICRO_TRAIN_BS`/`MICRO_ROLLOUT_BS`, or set `--zero_stage 3` |
| Slow convergence | Increase `N_SAMPLES` (e.g. 16), lower `LR` (e.g. 1e-7), increase `EPISODE` |

## Comparison with Other Methods

| Method | Reward Signal | Supervision Granularity | Requires RM |
|--------|--------------|------------------------|-------------|
| GRPO | Task-specific reward | Sequence-level | Yes |
| DPO | Preference pairs | Sequence-level | No |
| **OPD** | Teacher log probs | Token-level | No (uses teacher) |

## File Structure

```
examples/on_policy_distillation/
├── run_opd_qwen.sh       # All-in-one: teacher startup + training (recommended)
├── start_teacher.sh      # Teacher server only
├── start_training.sh     # Training only (requires TEACHER_URL)
├── test_opd.py           # Unit tests
├── README.md             # This file
└── README_zh.md          # Chinese version
```
