# Configuration Parameters

This comprehensive guide covers all configuration parameters available in LightRFT. Parameters are organized by category for easy reference.

## Table of Contents

1. [Model Parameters](#model-parameters)
2. [Training Parameters](#training-parameters)
3. [Batch Size Configuration](#batch-size-configuration)
4. [Algorithm Parameters](#algorithm-parameters)
5. [Distributed Training](#distributed-training)
6. [Inference Engine](#inference-engine)
7. [Memory Optimization](#memory-optimization)
8. [Logging and Monitoring](#logging-and-monitoring)
9. [Checkpoint Management](#checkpoint-management)

## Model Parameters

### `--pretrain`
- **Type**: `str`
- **Required**: Yes
- **Description**: Path to pre-trained model and tokenizer
- **Example**: `/path/to/Qwen2.5-7B-Instruct`

### `--reward_pretrain`
- **Type**: `str`
- **Default**: Same as `--pretrain`
- **Description**: Path to reward model
- **Example**: `/path/to/reward-model`

### `--remote_rm_url`
- **Type**: `str`
- **Default**: `None`
- **Description**: URL for remote reward model server
- **Example**: `http://localhost:5000`

### `--max_len`
- **Type**: `int`
- **Default**: `4096`
- **Description**: Maximum sequence length (prompt + response)

### `--prompt_max_len`
- **Type**: `int`
- **Default**: `2048`
- **Description**: Maximum prompt length

## Training Parameters

### `--num_episodes`
- **Type**: `int`
- **Default**: `1`
- **Description**: Total number of training episodes
- **Recommended**: 10-100 for most tasks

### `--max_epochs`
- **Type**: `int`
- **Default**: `1`
- **Description**: Number of training epochs per episode
- **Recommended**: 1-3

### `--actor_learning_rate`
- **Type**: `float`
- **Default**: `5e-7`
- **Description**: Learning rate for actor (policy) model
- **Recommended Range**: `1e-7` to `5e-6`

### `--critic_learning_rate`
- **Type**: `float`
- **Default**: `9e-6`
- **Description**: Learning rate for critic (value) model
- **Recommended Range**: `1e-6` to `1e-5`

### `--lr_warmup_ratio`
- **Type**: `float`
- **Default**: `0.03`
- **Description**: Ratio of warmup steps to total steps
- **Range**: `0.0` to `0.1`

### `--max_norm`
- **Type**: `float`
- **Default**: `1.0`
- **Description**: Maximum gradient norm for clipping

### `--l2`
- **Type**: `float`
- **Default**: `0.0`
- **Description**: L2 regularization coefficient

### `--adam_betas`
- **Type**: `tuple[float, float]`
- **Default**: `(0.9, 0.95)`
- **Description**: Adam optimizer beta parameters

## Batch Size Configuration

### Important Constraint
**Rule**: `train_batch_size >= rollout_batch_size × n_samples_per_prompt`

### `--train_batch_size` (TBS)
- **Type**: `int`
- **Required**: Yes
- **Description**: Global training batch size across all GPUs
- **Example**: `256`
- **Calculation**: `micro_train_batch_size × num_gpus × gradient_accumulation_steps`

### `--micro_train_batch_size`
- **Type**: `int`
- **Default**: `1`
- **Description**: Per-GPU batch size for training
- **Typical Values**: `1`, `2`, `4`

### `--rollout_batch_size` (RBS)
- **Type**: `int`
- **Required**: Yes
- **Description**: Global batch size for experience generation
- **Example**: `64`
- **Note**: Must be divisible by number of GPUs

### `--micro_rollout_batch_size`
- **Type**: `int`
- **Default**: `2`
- **Description**: Per-GPU batch size for rollout
- **Typical Values**: `2`, `4`, `8`

### Example Configurations

**Configuration 1: 8 GPUs, Memory-Constrained**
```bash
--train_batch_size 128 \
--micro_train_batch_size 1 \
--rollout_batch_size 64 \
--micro_rollout_batch_size 2
```

**Configuration 2: 8 GPUs, High-Throughput**
```bash
--train_batch_size 512 \
--micro_train_batch_size 2 \
--rollout_batch_size 256 \
--micro_rollout_batch_size 8
```

## Algorithm Parameters

### `--advantage_estimator`
- **Type**: `str`
- **Choices**: `group_norm`, `reinforce`, `cpgd`, `gspo`, `gmpo`
- **Default**: `group_norm`
- **Description**: Method for advantage estimation
- **Recommendation**:
  - `group_norm`: General purpose (GRPO)
  - `reinforce`: Low variance needed
  - `cpgd`: Preserve base capabilities

### `--n_samples_per_prompt`
- **Type**: `int`
- **Default**: `4`
- **Description**: Number of responses to sample per prompt
- **Typical Values**: `4`, `8`, `16`
- **Note**: Higher values = better but slower

### `--kl_estimator`
- **Type**: `str`
- **Choices**: `k1`, `k2`, `k3`
- **Default**: `k3`
- **Description**: KL divergence estimator type
- **Recommendation**: `k3` for most cases

### `--init_kl_coef`
- **Type**: `float`
- **Default**: `0.001`
- **Description**: Initial KL penalty coefficient
- **Range**: `0.0001` to `0.01`

### `--kl_target`
- **Type**: `float`
- **Default**: `0.01`
- **Description**: Target KL divergence (for CPGD)

### `--clip_range`
- **Type**: `float`
- **Default**: `0.2`
- **Description**: PPO clipping range
- **Range**: `0.1` to `0.3`

### `--clip_range_higher`
- **Type**: `float`
- **Default**: `0.3`
- **Description**: Upper clipping range (Clip Higher algorithm)

### `--temperature`
- **Type**: `float`
- **Default**: `1.0`
- **Description**: Sampling temperature
- **Range**: `0.6` to `1.2`
- **Note**: Lower = more deterministic

### `--top_p`
- **Type**: `float`
- **Default**: `0.9`
- **Description**: Nucleus sampling probability
- **Range**: `0.8` to `1.0`

## Distributed Training

### `--zero_stage`
- **Type**: `int`
- **Choices**: `1`, `2`, `3`
- **Default**: `2`
- **Description**: DeepSpeed ZeRO optimization stage
- **Recommendation**:
  - Stage 1: Optimizer state partitioning
  - Stage 2: + Gradient partitioning (recommended)
  - Stage 3: + Parameter partitioning (max memory saving)

### `--fsdp`
- **Action**: `store_true`
- **Default**: `False`
- **Description**: Use FSDP instead of DeepSpeed
- **When to Use**: PyTorch-native workflows, maximum memory efficiency

### `--fsdp_cpu_offload`
- **Action**: `store_true`
- **Default**: `False`
- **Description**: Offload FSDP optimizer states to CPU
- **Note**: Reduces GPU memory at cost of speed

### `--bf16`
- **Action**: `store_true`
- **Default**: Typically enabled
- **Description**: Use bfloat16 mixed precision

### `--gradient_checkpointing`
- **Action**: `store_true`
- **Default**: `False`
- **Description**: Enable gradient checkpointing
- **Note**: Trades computation for memory

### `--sp_size`
- **Type**: `int`
- **Default**: `1`
- **Description**: Sequence parallelism size
- **Recommendation**: `1`, `2`, `4` for very long sequences

## Inference Engine

### `--engine_type`
- **Type**: `str`
- **Choices**: `vllm`, `sglang`
- **Default**: `vllm`
- **Description**: Inference engine type

### `--engine_tp_size`
- **Type**: `int`
- **Default**: `1`
- **Description**: Tensor parallelism size for inference engine
- **Recommendation**:
  - 7B models: `1` or `2`
  - 13B-34B models: `2` or `4`
  - 70B+ models: `4` or `8`
- **Constraint**: `world_size % engine_tp_size == 0`

### `--engine_mem_util`
- **Type**: `float`
- **Default**: `0.5`
- **Range**: `0.3` to `0.9`
- **Description**: GPU memory utilization for KV cache
- **Recommendation**:
  - High memory: `0.8` - `0.9`
  - Medium memory: `0.5` - `0.7`
  - Low memory: `0.3` - `0.5`

### `--enable_engine_sleep`
- **Action**: `store_true`
- **Default**: `True`
- **Description**: Enable inference engine sleep mode
- **Note**: Saves memory when engine not in use

### `--disable_engine_sleep`
- **Action**: `store_false`
- **Dest**: `enable_engine_sleep`
- **Description**: Disable engine sleep mode

### `--rm_use_engine`
- **Action**: `store_true`
- **Default**: `False`
- **Description**: Use inference engine for reward model
- **When to Use**: High-throughput reward computation

## Memory Optimization

### `--adam_offload`
- **Action**: `store_true`
- **Default**: `False`
- **Description**: Offload Adam optimizer states to CPU

### `--use_mp_opt`
- **Action**: `store_true`
- **Default**: `False`
- **Description**: Use mixed precision optimizer (FSDP)

### `--packing_samples`
- **Action**: `store_true`
- **Default**: `False`
- **Description**: Pack multiple samples into sequences
- **When to Use**: Varied sequence lengths, improve GPU utilization

### `--fused_linear_logprob`
- **Action**: `store_true`
- **Default**: `False`
- **Description**: Fused linear layer and logprob computation
- **Note**: Saves memory for large vocabulary models

### `--chunk_size`
- **Type**: `int`
- **Default**: `4096`
- **Description**: Chunk size for fused operations

## Logging and Monitoring

### `--log_dir`
- **Type**: `str`
- **Default**: `None`
- **Description**: Directory for logs and visualizations

### `--plot_every`
- **Type**: `int`
- **Default**: `10`
- **Description**: Plot generation length distribution every N steps

### `--use_tensorboard`
- **Type**: `str`
- **Default**: `None`
- **Description**: TensorBoard log directory

### `--use_wandb`
- **Type**: `str`
- **Default**: `None`
- **Description**: Weights & Biases project name

### `--wandb_org`
- **Type**: `str`
- **Default**: `None`
- **Description**: W&B organization name

### `--wandb_run_name`
- **Type**: `str`
- **Default**: Auto-generated
- **Description**: W&B run name

## Checkpoint Management

### `--save_path`
- **Type**: `str`
- **Required**: Yes
- **Description**: Directory to save checkpoints

### `--ckpt_path`
- **Type**: `str`
- **Default**: `None`
- **Description**: Path to load checkpoint from

### `--load_checkpoint`
- **Action**: `store_true`
- **Default**: `False`
- **Description**: Enable checkpoint loading

### `--save_interval`
- **Type**: `int`
- **Default**: `1`
- **Description**: Save checkpoint every N episodes

### `--max_ckpt_num`
- **Type**: `int`
- **Default**: `3`
- **Description**: Maximum number of checkpoints to keep

### `--max_ckpt_mem`
- **Type**: `int`
- **Default**: `1000`
- **Description**: Maximum checkpoint memory in GB

## Reward Processing

### `--reward_running_norm`
- **Action**: `store_true`
- **Default**: `False`
- **Description**: Apply running normalization to rewards

### `--reward_running_norm_minus_mean`
- **Action**: `store_true`
- **Default**: `False`
- **Description**: Subtract mean during reward normalization

### `--advantages_norm`
- **Action**: `store_true`
- **Default**: `False`
- **Description**: Normalize advantages

### `--advantage_clip`
- **Type**: `float`
- **Default**: `0.0`
- **Description**: Clip advantages (0 = no clipping)

### `--reward_clip`
- **Type**: `float`
- **Default**: `0.0`
- **Description**: Clip rewards (0 = no clipping)

## Multimodal (VLM) Parameters

### `--mixed_mm_data`
- **Action**: `store_true`
- **Default**: `False`
- **Description**: Handle mixed multimodal and text-only data

### `--processor`
- **Type**: `str`
- **Default**: Auto-detected
- **Description**: Multimodal processor type

## Complete Example Configuration

### Math Reasoning (GSM8K, MATH)
```bash
python train.py \
    # Model
    --pretrain /path/to/Qwen2.5-7B-Instruct \
    --reward_pretrain /path/to/reward-model \
    --max_len 4096 \
    --prompt_max_len 2048 \
    \
    # Training
    --num_episodes 20 \
    --max_epochs 1 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    \
    # Batch Size
    --train_batch_size 128 \
    --micro_train_batch_size 1 \
    --rollout_batch_size 64 \
    --micro_rollout_batch_size 2 \
    \
    # Algorithm
    --advantage_estimator group_norm \
    --n_samples_per_prompt 8 \
    --kl_estimator k3 \
    --init_kl_coef 0.001 \
    --temperature 0.6 \
    \
    # Distributed
    --zero_stage 2 \
    --bf16 \
    --gradient_checkpointing \
    \
    # Engine
    --engine_type vllm \
    --engine_tp_size 1 \
    --engine_mem_util 0.85 \
    --enable_engine_sleep \
    \
    # Memory
    --adam_offload \
    --fused_linear_logprob \
    \
    # Logging
    --use_tensorboard ./tb_logs \
    --plot_every 10 \
    \
    # Checkpoint
    --save_path ./checkpoints \
    --save_interval 1 \
    --max_ckpt_num 3 \
    \
    # Reward
    --reward_running_norm \
    --advantages_norm
```

### Multimodal VLM Training
```bash
python train_vl.py \
    # Model
    --pretrain /path/to/Qwen2-VL-7B-Instruct \
    --max_len 4096 \
    \
    # Training
    --num_episodes 10 \
    --actor_learning_rate 1e-6 \
    \
    # Batch Size
    --train_batch_size 64 \
    --micro_train_batch_size 1 \
    --rollout_batch_size 32 \
    --micro_rollout_batch_size 1 \
    \
    # Algorithm
    --advantage_estimator group_norm \
    --n_samples_per_prompt 4 \
    \
    # Distributed
    --fsdp \
    --fsdp_cpu_offload \
    --gradient_checkpointing \
    \
    # Engine
    --engine_tp_size 4 \
    --engine_mem_util 0.6 \
    \
    # VLM Specific
    --mixed_mm_data \
    --packing_samples
```

## Parameter Validation

LightRFT performs automatic validation of parameters. Common validation rules:

1. **Batch Size**: `train_batch_size >= rollout_batch_size × n_samples_per_prompt`
2. **Divisibility**: Batch sizes must be divisible by number of GPUs
3. **Memory**: Engine TP size must divide world size evenly
4. **Learning Rate**: Actor LR typically < Critic LR
5. **KL Target**: Should be small (0.001-0.01) for stable training

## Environment Variables

Useful environment variables for optimization:

```bash
# NCCL optimization
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# CUDA optimization
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Debugging
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
```

## See Also

- [Algorithm Guide](algorithms.md) - Detailed algorithm descriptions
- [Strategy Usage](../best_practice/strategy_usage.md) - Distributed training strategies
- [Installation](../installation/index.rst) - Setup instructions
