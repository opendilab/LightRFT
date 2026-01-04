# Frequently Asked Questions (FAQ)

Common questions and answers about LightRFT.

## General Questions

### Q: What is LightRFT?

**A**: LightRFT (Light Reinforcement Fine-Tuning) is an advanced reinforcement learning framework for fine-tuning Large Language Models (LLMs) and Vision-Language Models (VLMs). It provides efficient and scalable RLHF training with support for multiple algorithms and distributed training strategies.

### Q: What are the main differences between LightRFT and OpenRLHF?

**A**: LightRFT extends OpenRLHF with:
- Enhanced multimodal (VLM) support
- More RL algorithms (GRPO, GSPO, GMPO, REINFORCE++, CPGD, etc.)
- Better memory optimization (engine sleep, optimizer offload)
- Improved inference engines (vLLM, SGLang with FP8)
- Reward model co-location for efficiency
- More flexible distributed training strategies

### Q: Which models are supported?

**A**: LightRFT supports:
- **LLMs**: Qwen, Qwen2.5, LLaMA, Mistral, and most HuggingFace models
- **VLMs**: Qwen-VL, Qwen2-VL, LLaVA
- **Custom**: Easy to add new models via monkey patching

### Q: What hardware is required?

**A**: Minimum requirements:
- **GPU**: NVIDIA GPUs with CUDA 11.8+
- **Memory**: 40GB+ VRAM per GPU recommended (24GB possible with optimizations)
- **PyTorch**: 2.5.1+
- **Python**: 3.8+

For production: 8× A100/H100 80GB recommended

## Installation Questions

### Q: How do I install LightRFT?

**A**: Simple installation:
```bash
git clone https://github.com/opendilab/LightRFT.git
cd LightRFT
pip install -r requirements.txt && pip install -e .
```

### Q: Do I need to install vLLM separately?

**A**: No, vLLM is included in the requirements. However, for the latest features, you can install from source.

## Training Questions

### Q: What's the difference between FSDP and DeepSpeed?

**A**:
- **FSDP**: PyTorch-native, better integration, supports CPU offload
- **DeepSpeed**: More mature, ZeRO-3 optimization, generally faster

Use FSDP for maximum memory efficiency, DeepSpeed for speed.

### Q: How do I choose batch sizes?

**A**: Follow this constraint:
```
train_batch_size >= rollout_batch_size × n_samples_per_prompt
```

Example for 8 GPUs:
- `train_batch_size=256`
- `rollout_batch_size=64`
- `n_samples_per_prompt=8`
- `micro_train_batch_size=1`
- `micro_rollout_batch_size=2`

### Q: Which algorithm should I use?

**A**: By task:
- **Math/Coding**: GRPO, Dr.GRPO
- **Instruction Following**: CPGD, GSPO
- **Open-ended**: FIRE Sampling
- **Low Memory**: GRPO (no critic)
- **Research**: GMPO, REINFORCE++

### Q: How many samples per prompt should I use?

**A**: Typical values:
- **4-8**: Standard, good balance
- **16+**: Better quality, slower training
- **32+**: Best-of-N scenarios

More samples = better advantage estimation but slower.

### Q: Can I use multiple reward models?

**A**: Yes! LightRFT supports:
- Multiple reward models in parallel
- Reward model co-location (same GPU as training)
- Remote reward model servers
- Weighted reward combination

### Q: How do I enable multimodal (VLM) training?

**A**: Use the VLM training script:
```bash
python train_vl.py \
    --pretrain /path/to/Qwen2-VL \
    --mixed_mm_data \
    --packing_samples
```

## Performance Questions

### Q: How do I reduce memory usage?

**A**: Use these techniques:
1. Enable gradient checkpointing: `--gradient_checkpointing`
2. Use FSDP with CPU offload: `--fsdp --fsdp_cpu_offload`
3. Lower engine memory: `--engine_mem_util 0.4`
4. Use ZeRO-3: `--zero_stage 3`
5. Reduce batch sizes
6. Enable engine sleep: `--enable_engine_sleep`

### Q: How do I speed up training?

**A**:
1. Increase batch sizes (if memory allows)
2. Use FP8 inference (vLLM)
3. Enable Flash Attention: `--flash_attn`
4. Reduce `n_samples_per_prompt` if possible
5. Use tensor parallelism for inference: `--engine_tp_size 2`
6. Optimize NCCL: `export TORCH_NCCL_AVOID_RECORD_STREAMS=1`

### Q: What's the typical training speed?

**A**: On 8× A100 80GB:
- **7B model**: ~1000 samples/min
- **13B model**: ~500 samples/min
- **34B model**: ~200 samples/min
- **70B model**: ~50 samples/min

With FSDP and optimizations.

### Q: How do I use multiple nodes?

**A**: Use SLURM or Ray:
```bash
# SLURM example
srun -N2 --gres=gpu:8 --ntasks-per-node=8 bash train.sh

# Or use torchrun
torchrun --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py
```

## Algorithm Questions

### Q: What's the difference between GRPO and PPO?

**A**:
- **GRPO**: Group-normalized advantages, no critic network
- **PPO**: Uses separate value network (critic)

GRPO is simpler and more memory-efficient.

### Q: When should I use CPGD?

**A**: Use CPGD when:
- Fine-tuning pre-trained models
- Want to preserve base capabilities
- Need controlled policy updates
- Preventing catastrophic forgetting

### Q: What is Clip Higher?

**A**: An improved clipping scheme with separate upper/lower bounds for positive/negative advantages. Better for:
- Noisy rewards
- Large distribution shifts
- Unstable training

## Debugging Questions

### Q: Training crashes with OOM error

**A**: See the [Troubleshooting Guide](troubleshooting.md#out-of-memory-oom-errors)

### Q: `num_rollouts_per_episodes = 0` error

**A**: Your `train_batch_size` is too small. Ensure:
```
train_batch_size >= rollout_batch_size × n_samples_per_prompt
```

### Q: Model not improving / Reward not increasing

**A**: Check:
1. Learning rate too high/low
2. KL penalty too large
3. Reward model quality
4. Enable reward normalization: `--reward_running_norm`
5. Try different advantage estimator

### Q: NCCL timeout or hanging

**A**:
```bash
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# Increase timeout
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
```

### Q: vLLM engine initialization fails

**A**:
1. Check GPU memory: `--engine_mem_util 0.5`
2. Reduce TP size: `--engine_tp_size 1`
3. Check CUDA compatibility
4. Update vLLM: `pip install -U vllm`

## Evaluation Questions

### Q: How do I evaluate on benchmarks?

**A**: For math benchmarks, use the evaluation scripts in the examples directory:
```bash
# Refer to the examples/gsm8k_geo3k directory for evaluation scripts
# See the example training scripts for evaluation configurations
```

### Q: Can I save generation trajectories?

**A**: Yes, use the trajectory saver:
```python
from lightrft.utils import TrajectorySaver

saver = TrajectorySaver(output_dir="./trajectories")
# Automatically saves prompts, responses, rewards
```

### Q: How do I integrate with W&B?

**A**:
```bash
python train.py \
    --use_wandb your-project \
    --wandb_org your-org \
    --wandb_run_name experiment-1
```

## Advanced Questions

### Q: Can I implement custom algorithms?

**A**: Yes! Extend the trainer class:
```python
from lightrft.trainer import SPMDPPOTrainer

class CustomTrainer(SPMDPPOTrainer):
    def compute_advantages(self, ...):
        # Your custom advantage computation
        pass
```

### Q: How do I add a new model architecture?

**A**: Create a monkey patch in `lightrft/models/monkey_patch/`:
```python
# your_model.py
def patch_your_model(model):
    # Add custom forward methods
    pass

# In apply.py
from .your_model import patch_your_model
```

### Q: Can I use custom reward functions?

**A**: Yes, pass a callable:
```python
def custom_reward_fn(responses, labels):
    # Your reward computation
    return rewards

trainer = SPMDPPOTrainer(
    ...,
    reward_fn=custom_reward_fn
)
```

### Q: How do I checkpoint during training?

**A**: Checkpoints are automatic:
```bash
--save_path ./checkpoints \
--save_interval 1 \
--max_ckpt_num 3
```

Resume with:
```bash
--load_checkpoint \
--ckpt_path ./checkpoints/episode_5
```

## Contributing Questions

### Q: How can I contribute to LightRFT?

**A**:
1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

See [Contributing Guide](contributing.md) for details.

### Q: How do I report bugs?

**A**: Open an issue on [GitHub Issues](https://github.com/opendilab/LightRFT/issues) with:
- Environment details (GPU, CUDA, PyTorch versions)
- Full error traceback
- Minimal reproduction script
- Expected vs actual behavior

### Q: Where can I get help?

**A**:
- GitHub Issues for bugs
- Discussions for questions
- Documentation for guides
- Examples directory for code samples

## Additional Resources

- [Installation Guide](../installation/index.rst)
- [Quick Start](../quick_start/index.rst)
- [Algorithm Guide](algorithms.md)
- [Configuration Reference](configuration.md)
- [Troubleshooting Guide](troubleshooting.md)
- [Best Practices](../best_practice/index.rst)
