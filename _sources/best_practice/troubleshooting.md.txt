# Troubleshooting Guide

This guide helps you diagnose and resolve common issues when using LightRFT.

## Quick Diagnosis

Use this flowchart to quickly identify your issue:

```
Issue Type?
├─ Installation/Setup → See [Installation Issues](#installation-issues)
├─ Out of Memory → See [Memory Issues](#memory-issues)
├─ Training Issues → See [Training Problems](#training-problems)
├─ Performance → See [Performance Issues](#performance-issues)
└─ Distributed Training → See [Distributed Issues](#distributed-training-issues)
```

## Installation Issues

### Problem: Package import errors

**Symptoms**:
```
ModuleNotFoundError: No module named 'lightrft'
```

**Solution**:
```bash
# Ensure you're in the correct directory
cd /path/to/LightRFT
pip install -r requirements.txt
pip install -e .
```

### Problem: CUDA version mismatch

**Symptoms**:
```
RuntimeError: CUDA error: no kernel image is available for execution
```

**Solution**:
```bash
# Check CUDA version
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch with correct CUDA version
pip install torch==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### Problem: vLLM installation fails

**Symptoms**:
```
ERROR: Failed building wheel for vllm
```

**Solution**:
```bash
# Install build dependencies
pip install ninja packaging wheel

# Install vLLM from source
pip install vllm --no-build-isolation

# Or use pre-built wheel
pip install vllm==0.5.3.post1
```

## Memory Issues

### Problem: Out of Memory (OOM) Errors

**Symptoms**:
```
RuntimeError: CUDA out of memory
torch.cuda.OutOfMemoryError
```

**Solution Strategy** (try in order):

**1. Reduce Batch Sizes**
```bash
# Before
--micro_train_batch_size 2
--micro_rollout_batch_size 4

# After
--micro_train_batch_size 1
--micro_rollout_batch_size 2
```

**2. Enable Gradient Checkpointing**
```bash
--gradient_checkpointing
```
Trades ~20% speed for ~50% memory savings.

**3. Lower Engine Memory**
```bash
# Before
--engine_mem_util 0.9

# After
--engine_mem_util 0.5  # Or 0.4 for very low memory
```

**4. Use FSDP with CPU Offload**
```bash
--fsdp \
--fsdp_cpu_offload \
--use_mp_opt
```

**5. Enable Adam Offload**
```bash
--adam_offload
```

**6. Use ZeRO-3**
```bash
--zero_stage 3
```

**7. Reduce Model/Sequence Length**
```bash
--max_len 2048  # Instead of 4096
--prompt_max_len 1024
```

**Complete Low-Memory Configuration**:
```bash
python train.py \
    --micro_train_batch_size 1 \
    --micro_rollout_batch_size 1 \
    --gradient_checkpointing \
    --engine_mem_util 0.4 \
    --fsdp \
    --fsdp_cpu_offload \
    --adam_offload \
    --max_len 2048 \
    --use_mp_opt
```

### Problem: vLLM Engine OOM

**Symptoms**:
```
Failed to allocate memory for KV cache
```

**Solution**:
```bash
# Reduce KV cache memory
--engine_mem_util 0.3

# Increase tensor parallelism
--engine_tp_size 2  # or 4

# Enable engine sleep
--enable_engine_sleep

# Use smaller max length
--max_len 2048
```

### Problem: Memory Leak During Training

**Symptoms**:
- Memory gradually increases
- Eventually OOMs after several episodes

**Solution**:
```bash
# Enable NCCL optimization
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# Clear cache periodically
# Add to training code:
torch.cuda.empty_cache()

# Use engine sleep
--enable_engine_sleep
```

## Training Problems

### Problem: `num_rollouts_per_episodes = 0`

**Symptoms**:
```
AssertionError: num_rollouts_per_episodes should be > 0
```

**Root Cause**:
`train_batch_size` < `rollout_batch_size × n_samples_per_prompt`

**Solution**:
```bash
# Ensure TBS >= RBS × n_samples
# Example: RBS=64, n_samples=8
--train_batch_size 512  # Must be >= 64×8=512
--rollout_batch_size 64
--n_samples_per_prompt 8
```

### Problem: Training Not Converging

**Symptoms**:
- Reward not increasing
- Loss oscillating
- Model not improving

**Diagnosis & Solutions**:

**1. Check Learning Rate**
```bash
# If too high (loss spikes):
--actor_learning_rate 1e-7  # Lower

# If too low (no progress):
--actor_learning_rate 1e-6  # Higher
```

**2. Enable Reward Normalization**
```bash
--reward_running_norm \
--reward_running_norm_minus_mean \
--advantages_norm
```

**3. Check KL Penalty**
```bash
# If KL too large (policy not updating):
--init_kl_coef 0.0001  # Lower

# If KL too small (instability):
--init_kl_coef 0.01  # Higher
```

**4. Try Different Algorithm**
```bash
# Switch from GRPO to CPGD
--advantage_estimator cpgd \
--kl_target 0.01
```

**5. Check Reward Model Quality**
```python
# Test reward model separately
python test_reward_model.py --model /path/to/rm
```

### Problem: NaN Loss or Gradients

**Symptoms**:
```
Loss: nan
Gradient: nan
```

**Solution**:
```bash
# 1. Enable gradient clipping
--max_norm 1.0

# 2. Lower learning rate
--actor_learning_rate 1e-7

# 3. Use BF16 instead of FP16
--bf16

# 4. Enable reward clipping
--reward_clip 10.0

# 5. Check for division by zero
--advantages_norm  # Normalizes before use
```

### Problem: Training Extremely Slow

**Symptoms**:
- < 100 samples/min on 8×A100
- Each episode takes hours

**Solutions**:

**1. Profile Bottleneck**
```python
# Add profiling
with torch.profiler.profile() as prof:
    trainer.fit()
print(prof.key_averages())
```

**2. Check Data Loading**
```bash
# Increase workers
--num_workers 8

# Use faster dataloader
--dataloader_pin_memory
```

**3. Optimize Generation**
```bash
# Use FP8 inference
--engine_type vllm  # vLLM supports FP8

# Increase TP for generation
--engine_tp_size 2

# Reduce max length if possible
--max_len 2048
```

**4. Reduce Logging**
```bash
# Don't log every step
--log_interval 100
```

## Distributed Training Issues

### Problem: NCCL Timeout

**Symptoms**:
```
RuntimeError: NCCL timeout
[E ProcessGroupNCCL.cpp] Caught collective operation timeout
```

**Solution**:
```bash
# Increase timeout
export NCCL_TIMEOUT=1800

# Debug NCCL
export NCCL_DEBUG=INFO

# Try different network interface
export NCCL_SOCKET_IFNAME=eth0

# Disable InfiniBand if issues
export NCCL_IB_DISABLE=1

# Use GLOO for debugging
export NCCL_BACKEND=gloo
```

### Problem: Distributed Initialization Hanging

**Symptoms**:
- Script hangs at "Initializing process group"
- No error message

**Solution**:
```bash
# 1. Check network connectivity
ping $MASTER_ADDR

# 2. Check port availability
nc -zv $MASTER_ADDR $MASTER_PORT

# 3. Set correct environment variables
export MASTER_ADDR=192.168.1.1
export MASTER_PORT=29500
export WORLD_SIZE=8
export RANK=0  # 0 to 7 for each GPU

# 4. Use explicit init_method
torchrun --nproc_per_node=8 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py

# 5. Enable debug logging
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

### Problem: Uneven GPU Utilization

**Symptoms**:
- Some GPUs at 100%, others idle
- Slow training despite multiple GPUs

**Solution**:
```bash
# 1. Check batch size divisibility
# Ensure batch_size % world_size == 0

# 2. Use tensor parallelism
--engine_tp_size 2  # Splits model across GPUs

# 3. Check for pipeline bubbles
# Ensure train_batch_size is large enough

# 4. Monitor GPU utilization
nvidia-smi dmon -i 0,1,2,3,4,5,6,7 -s u

# 5. Use sequence parallelism for long sequences
--sp_size 2
```

### Problem: Multi-Node Training Fails

**Symptoms**:
- Works on single node
- Fails on multiple nodes

**Solution**:
```bash
# 1. Use SLURM
srun -N2 --gres=gpu:8 --ntasks-per-node=8 bash train.sh

# 2. Or explicit torchrun
# On each node:
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    train.py

# 3. Check firewall rules
# Ensure ports are open between nodes

# 4. Use shared filesystem
# Ensure all nodes can access model/data paths
```

## Performance Issues

### Problem: Low GPU Utilization

**Symptoms**:
- GPU utilization < 80%
- Training slower than expected

**Solutions**:

**1. Increase Batch Size**
```bash
--micro_train_batch_size 2  # Double it
--micro_rollout_batch_size 4
```

**2. Reduce CPU Bottleneck**
```bash
--num_workers 8
--prefetch_factor 2
```

**3. Enable Flash Attention**
```bash
--flash_attn
```

**4. Use Fused Kernels**
```bash
--fused_linear_logprob
```

### Problem: Generation Too Slow

**Symptoms**:
- Rollout phase takes majority of time
- < 100 tokens/sec generation

**Solutions**:

**1. Use vLLM**
```bash
--engine_type vllm  # Instead of HF
--engine_tp_size 2
```

**2. Optimize KV Cache**
```bash
--engine_mem_util 0.9  # If memory allows
```

**3. Use FP8 (if supported)**
```bash
# vLLM automatically uses FP8 on H100
--engine_type vllm
```

**4. Reduce Samples**
```bash
--n_samples_per_prompt 4  # Instead of 8
```

## Inference Engine Issues

### Problem: vLLM Engine Fails to Initialize

**Symptoms**:
```
Failed to initialize vLLM engine
RuntimeError: Cannot allocate memory
```

**Solution**:
```bash
# 1. Check GPU memory
nvidia-smi

# 2. Reduce memory allocation
--engine_mem_util 0.5

# 3. Use smaller TP size
--engine_tp_size 1

# 4. Check model compatibility
# Some models need specific vLLM versions

# 5. Update vLLM
pip install -U vllm
```

### Problem: Engine Not Updating Weights

**Symptoms**:
- Policy model updates but generations don't change
- Rewards stay constant

**Solution**:
```python
# Ensure update_engine_weights is called
self.strategy.update_engine_weights(self.actor)

# Check in training loop:
def ppo_train(self):
    ...
    # After training
    self.strategy.update_engine_weights(self.actor)
```

### Problem: Engine Sleep/Wake Issues

**Symptoms**:
- Training hangs after generation
- "Engine already sleeping" errors

**Solution**:
```bash
# 1. Disable engine sleep for debugging
--disable_engine_sleep

# 2. Or use automatic management
# gather_and_generate handles sleep/wake automatically
all_outputs = self.strategy.gather_and_generate(
    ...,
    sleep_engine=True  # Automatic management
)
```

## Checkpoint Issues

### Problem: Cannot Load Checkpoint

**Symptoms**:
```
FileNotFoundError: Checkpoint not found
RuntimeError: Error loading state dict
```

**Solution**:
```bash
# 1. Check checkpoint path
ls -la /path/to/checkpoint

# 2. Load with relaxed matching
--load_checkpoint \
--ckpt_path /path/to/checkpoint

# 3. Skip optimizer states if incompatible
# Edit code to load model only:
model.load_state_dict(torch.load(ckpt_path))
```

### Problem: Checkpoint Saving Fails

**Symptoms**:
```
OSError: Disk quota exceeded
RuntimeError: Cannot save checkpoint
```

**Solution**:
```bash
# 1. Check disk space
df -h

# 2. Limit checkpoint number
--max_ckpt_num 3

# 3. Set max checkpoint size
--max_ckpt_mem 1000  # GB

# 4. Use different save path
--save_path /path/with/space
```

## Debugging Tips

### Enable Debug Logging

```bash
# PyTorch distributed
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# NCCL
export NCCL_DEBUG=INFO

# CUDA
export CUDA_LAUNCH_BLOCKING=1
```

### Memory Profiling

```python
import torch

# Track memory allocation
torch.cuda.memory._record_memory_history()

# Training loop
...

# Dump memory snapshot
torch.cuda.memory._dump_snapshot("memory.pickle")
```

### Performance Profiling

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=True
) as prof:
    trainer.fit()

# View results
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Debugging Checklist

When reporting bugs, include:

- [ ] Hardware: GPU model, count, memory
- [ ] Software: CUDA, PyTorch, vLLM versions
- [ ] Full command with all arguments
- [ ] Full error traceback
- [ ] Environment variables set
- [ ] Minimal reproduction script
- [ ] What you've tried already

## Getting Help

If you can't resolve the issue:

1. **Check FAQ**: [FAQ](faq.md)
2. **Search Issues**: [GitHub Issues](https://github.com/opendilab/LightRFT/issues)
3. **Ask Community**: GitHub Discussions
4. **Report Bug**: Open new issue with debugging info

## Common Error Messages Reference

| Error Message | Section | Quick Fix |
|---------------|---------|-----------|
| `CUDA out of memory` | [Memory Issues](#memory-issues) | Reduce batch size, enable checkpointing |
| `num_rollouts_per_episodes = 0` | [Training Problems](#training-problems) | Increase `train_batch_size` |
| `NCCL timeout` | [Distributed Issues](#distributed-training-issues) | `export NCCL_TIMEOUT=1800` |
| `Failed to initialize vLLM` | [Inference Engine Issues](#inference-engine-issues) | Reduce `engine_mem_util` |
| `NaN loss` | [Training Problems](#training-problems) | Lower learning rate, clip gradients |

## See Also

- [FAQ](faq.md) - Frequently asked questions
- [Configuration](../user_guide/configuration.md) - All parameters
- [Best Practices](../best_practice/strategy_usage.md) - Optimization tips
