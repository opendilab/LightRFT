# FastExperienceMaker Best Practice Guide

## Table of Contents
- [Overview](#overview)
- [Core Features](#core-features)
- [Architecture Components](#architecture-components)
- [Usage Guide](#usage-guide)
- [Configuration Parameters](#configuration-parameters)
- [Advantage Estimation Methods](#advantage-estimation-methods)
- [Best Practices](#best-practices)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Performance Tuning](#performance-tuning)

## Overview

### What is FastExperienceMaker?

FastExperienceMaker is an optimized experience generation engine for RLHF (Reinforcement Learning from Human Feedback) training in LightRFT. It extends the base `NaiveExperienceMaker` with high-performance inference backends (VLLM/SGLang) and advanced RL features.

### Key Capabilities

- **High-Performance Inference**: VLLM and SGLang backend support for efficient text generation
- **Multimodal Support**: Vision-language model (VLM) data processing with images and videos
- **Advanced Advantage Estimation**: Multiple methods including GAE, RLOO, REINFORCE, and Group Normalization
- **Flexible Reward Composition**: Support for multiple reward models with custom aggregation functions
- **Sample Packing**: Improved training efficiency through sequence packing
- **Reward Normalization**: Running reward statistics with normalization and clipping

## Core Features

### 1. Experience Generation Pipeline

The FastExperienceMaker implements a 7-stage pipeline for experience generation:

```
Stage 1: Sample Generation (VLLM/SGLang)
    ↓
Stage 2: Shard-Parallel Preprocessing
    ↓
Stage 3: Model Inference (Actor, Critic, Initial, Reward Models)
    ↓
Stage 4: Shard-Parallel Postprocessing
    ↓
Stage 5: Reward Processing (Normalization, Shaping, Filtering)
    ↓
Stage 6: Multi-Image/Video Handling
    ↓
Stage 7: Advantage Computation
```

### 2. Multimodal Data Processing

The `MultimodalDataProcessor` class handles mixed text-only and image-text data:

- **Automatic Separation**: Separates text-only and multimodal samples
- **Appropriate Processing**: Routes samples through tokenizer or multimodal processor
- **Order Preservation**: Maintains original batch ordering after processing
- **Multi-Image/Video Support**: Handles multiple images or videos per sample

### 3. Reward Computation Engine

The `RewardComputationEngine` manages reward model inference and aggregation:

- **Remote Reward Models**: HTTP/gRPC reward model support
- **Local Reward Models**: PyTorch-based reward models
- **Custom Reward Functions**: Python functions for custom reward logic
- **Multi-Model Ensemble**: Combine multiple reward models with custom aggregation
- **Optimized Batching**: Efficient batch processing with optional sample filtering

## Architecture Components

### Class Hierarchy

```
NaiveExperienceMaker (Base Class)
    ↓
FastExperienceMaker
    ├── MultimodalDataProcessor
    ├── RewardComputationEngine
    └── AdvantageCalculator (GAE/RLOO/REINFORCE/GroupNorm)
```

### Key Classes

#### 1. FastExperienceMaker

**Purpose**: Main experience generation class with optimized inference and advanced RL features.

**Initialization Parameters**:
- `packing_samples` (bool): Enable sample packing for efficiency
- `processor`: Multimodal processor for VLM models
- Other parameters inherited from `NaiveExperienceMaker`

**Key Methods**:
- `make_experience_list()`: Generate experiences from prompts
- `generate_samples()`: Generate samples using inference engine
- `get_advantages_and_returns()`: Compute advantages and returns

#### 2. MultimodalDataProcessor

**Purpose**: Handles preprocessing of mixed text-only and multimodal data.

**Key Responsibilities**:
- Normalize image/video inputs (file paths, PIL images, bytes)
- Separate text-only and multimodal samples
- Process through appropriate pipelines
- Expand samples by `n_samples_per_prompt` factor

#### 3. RewardComputationEngine

**Purpose**: Manages reward model inference and score aggregation.

**Processing Pipeline**:
1. **Gather**: Collect or filter samples based on reward recipe
2. **Process**: Run forward pass through reward model(s)
3. **Aggregate**: Combine scores using reward_fn

## Usage Guide

### Basic Usage

#### Text-Only Generation

```python
from lightrft.trainer.fast_exp_maker import FastExperienceMaker

# Initialize experience maker
exp_maker = FastExperienceMaker(
    actor=actor_model,
    critic=critic_model,
    reward_model=reward_model,
    initial_model=initial_model,
    tokenizer=tokenizer,
    prompt_max_len=512,
    kl_controller=kl_controller,
    strategy=strategy,
    packing_samples=False,
)

# Generate experiences
prompts = ["Explain quantum computing", "What is machine learning?"]
experiences = exp_maker.make_experience_list(
    all_prompts=prompts,
    temperature=0.7,
    max_new_tokens=512,
    top_p=0.9,
)
```

#### Vision-Language Generation

```python
from PIL import Image

# Initialize with processor for VLM support
exp_maker = FastExperienceMaker(
    actor=actor_model,
    critic=critic_model,
    reward_model=reward_model,
    initial_model=initial_model,
    tokenizer=tokenizer,
    processor=multimodal_processor,  # Required for VLM
    prompt_max_len=512,
    kl_controller=kl_controller,
    strategy=strategy,
)

# Prepare multimodal data
prompts = ["Describe this image", "What's in this picture?"]
images = [
    [Image.open("image1.jpg")],  # Single image
    [Image.open("img2.jpg"), Image.open("img3.jpg")],  # Multiple images
]
references = ["A cat on a sofa", "Two dogs playing"]

# Generate experiences
experiences = exp_maker.make_experience_list(
    all_prompts=prompts,
    all_images=images,
    all_references=references,
    temperature=0.7,
    max_new_tokens=512,
)
```

### Advanced Usage

#### Multiple Reward Models with Custom Aggregation

```python
# Define custom reward aggregation function
def custom_reward_fn(model_reward_list, labels, queries, refs, label_map):
    """
    Custom reward aggregation function.

    Args:
        model_reward_list: List of reward tensors from each model
        labels: Sample labels
        queries: Generated text
        refs: Reference texts
        label_map: Mapping from reward model names to indices

    Returns:
        aggregated_rewards: Combined reward tensor
        reward_metrics: Dictionary of detailed metrics
    """
    # Example: Weighted average based on label
    weights = torch.tensor([0.6, 0.4])  # Weights for two models
    aggregated = sum(w * r for w, r in zip(weights, model_reward_list))

    metrics = {
        "reward_model_1": model_reward_list[0].mean(),
        "reward_model_2": model_reward_list[1].mean(),
    }

    return aggregated, metrics

# Initialize with multiple reward models
exp_maker = FastExperienceMaker(
    actor=actor_model,
    critic=critic_model,
    reward_model=[reward_model_1, reward_model_2],  # List of models
    reward_fn=custom_reward_fn,
    reward_fn_label_map={"rm1": 0, "rm2": 1},
    initial_model=initial_model,
    tokenizer=tokenizer,
    strategy=strategy,
)
```

#### Sample Packing for Efficiency

```python
# Enable sample packing
exp_maker = FastExperienceMaker(
    actor=actor_model,
    critic=critic_model,
    reward_model=reward_model,
    initial_model=initial_model,
    tokenizer=tokenizer,
    strategy=strategy,
    packing_samples=True,  # Enable packing
)

# Packed format: | prompt1 response1 [EOS] | prompt2 response2 [EOS] | ...
# Benefits:
# - Reduced padding overhead
# - Improved GPU utilization
# - Faster training throughput
```

#### Remote Reward Models

```python
# Use remote reward models via HTTP/gRPC
exp_maker = FastExperienceMaker(
    actor=actor_model,
    critic=critic_model,
    reward_model=None,  # No local reward model
    remote_rm_url=[
        "http://reward-server-1:8000/score",
        "http://reward-server-2:8000/score",
    ],
    initial_model=initial_model,
    tokenizer=tokenizer,
    strategy=strategy,
)
```

## Configuration Parameters

### Generation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 1.0 | Sampling temperature (higher = more random) |
| `top_p` | float | 1.0 | Nucleus sampling threshold |
| `top_k` | int | -1 | Top-k sampling (-1 = disabled) |
| `max_new_tokens` | int | 1024 | Maximum number of tokens to generate |
| `min_new_tokens` | int | 1 | Minimum number of tokens to generate |
| `skip_special_tokens` | bool | False | Skip special tokens in output |

### Reward Processing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reward_running_norm` | bool | False | Enable running reward normalization |
| `reward_running_norm_minus_mean` | bool | False | Subtract mean in normalization |
| `reward_clip` | float | 0 | Reward clipping threshold (0 = disabled) |
| `overlong_buffer` | bool | False | Enable overlong sequence penalty |
| `overlong_buffer_len` | int | 50 | Buffer length for overlong penalty |
| `overlong_buffer_penalty_factor` | float | 1.0 | Penalty factor for overlong sequences |

### Advantage Estimation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `advantage_estimator` | str | "gae" | Method: "gae", "rloo", "reinforce", "group_norm" |
| `advantages_norm` | bool | False | Enable advantage normalization (whitening) |
| `advantage_clip` | float | 0 | Advantage clipping threshold (0 = disabled) |
| `gamma` | float | 1.0 | Discount factor for returns |
| `lambd` | float | 0.95 | GAE lambda parameter |

## Advantage Estimation Methods

### 1. GAE (Generalized Advantage Estimation)

**When to Use**: Standard PPO training with critic model.

**Advantages**:
- Balances bias-variance tradeoff via lambda parameter
- Smooth advantage estimates
- Works well with value function

**Configuration**:
```python
strategy.config.advantage_estimator = "gae"
strategy.config.gamma = 1.0
strategy.config.lambd = 0.95
```

### 2. RLOO (REINFORCE Leave-One-Out)

**When to Use**: Training without critic model, multiple samples per prompt.

**Advantages**:
- No critic model required
- Reduces variance through baseline subtraction
- Efficient for multiple samples per prompt

**Configuration**:
```python
strategy.config.advantage_estimator = "rloo"
strategy.config.n_samples_per_prompt = 4  # Required > 1
```

### 3. REINFORCE with Baseline

**When to Use**: Simple policy gradient with reward baseline.

**Advantages**:
- Simple and straightforward
- Works with single sample per prompt
- No critic model required

**Configuration**:
```python
strategy.config.advantage_estimator = "reinforce"
```

### 4. Group Normalization (GRPO)

**When to Use**: Group-based advantage normalization, multiple samples per prompt.

**Advantages**:
- Normalizes advantages within each prompt group
- Reduces variance across different prompts
- Effective for diverse prompt distributions

**Configuration**:
```python
strategy.config.advantage_estimator = "group_norm"
strategy.config.n_samples_per_prompt = 4  # Required > 1
```

### Comparison Table

| Method | Critic Required | Samples per Prompt | Variance | Bias | Complexity |
|--------|----------------|-------------------|----------|------|------------|
| GAE | Yes | Any | Low | Low | Medium |
| RLOO | No | > 1 | Medium | Low | Low |
| REINFORCE | No | Any | High | Low | Low |
| Group Norm | No | > 1 | Medium | Medium | Low |

## Best Practices

### 1. Choosing Inference Backend

**VLLM**:
- ✅ Best for: Large-scale deployment, high throughput
- ✅ Supports: PagedAttention, continuous batching
- ⚠️ Note: Requires CUDA-compatible GPU

**SGLang**:
- ✅ Best for: Research, flexibility
- ✅ Supports: Custom sampling, structured generation
- ⚠️ Note: May have different performance characteristics

### 2. Multimodal Data Handling

**Image Normalization**:
```python
# Supported formats:
# 1. PIL Image objects
images = [[Image.open("img.jpg")]]

# 2. File paths (will be loaded automatically)
images = [["path/to/image.jpg"]]

# 3. Mixed formats
images = [[Image.open("img1.jpg"), "path/to/img2.jpg"]]
```

**Multi-Image Scenarios**:
```python
# Multiple images per sample
images = [
    [img1, img2, img3],  # Sample 1: 3 images
    [img4],              # Sample 2: 1 image
    None,                # Sample 3: No image (text-only)
]
```

### 3. Reward Model Configuration

**Single Reward Model**:
```python
exp_maker = FastExperienceMaker(
    reward_model=single_rm,
    # ...
)
```

**Multiple Reward Models with Aggregation**:
```python
exp_maker = FastExperienceMaker(
    reward_model=[rm1, rm2, rm3],
    reward_fn=custom_aggregation_fn,
    reward_fn_label_map={"quality": 0, "safety": 1, "helpfulness": 2},
    # ...
)
```

**Custom Reward Function**:
```python
def custom_reward(queries, prompts, labels):
    """Custom reward computation logic"""
    rewards = []
    for query, prompt, label in zip(queries, prompts, labels):
        # Your custom logic here
        score = compute_custom_score(query, prompt, label)
        rewards.append(score)
    return rewards

exp_maker = FastExperienceMaker(
    custom_reward_func=custom_reward,
    # ...
)
```

### 4. Memory Optimization

**Enable Sample Packing**:
```python
# Reduces padding overhead by 30-50%
exp_maker = FastExperienceMaker(
    packing_samples=True,
    # ...
)
```

**Adjust Micro Batch Size**:
```python
# Balance memory usage and throughput
strategy.config.micro_rollout_batch_size = 8  # Adjust based on GPU memory
```

**Gradient Checkpointing**:
```python
# Enable for large models
actor.gradient_checkpointing_enable()
```

### 5. Reward Normalization Strategy

**Running Normalization** (Recommended for stable training):
```python
strategy.args.reward_running_norm = True
strategy.args.reward_running_norm_minus_mean = True  # Subtract mean
```

**Reward Clipping** (Prevent outliers):
```python
strategy.config.reward_clip = 10.0  # Clip to [-10, 10]
```

**Advantage Normalization** (Stabilize policy updates):
```python
strategy.config.advantages_norm = True
strategy.config.advantage_clip = 5.0  # Optional clipping
```

### 6. Handling Overlong Sequences

```python
# Penalize sequences that are too long
strategy.config.overlong_buffer = True
strategy.config.overlong_buffer_len = 50  # Buffer length
strategy.config.overlong_buffer_penalty_factor = 1.0  # Penalty strength

# Example: If max_new_tokens=512 and buffer_len=50
# Expected length = 512 - 50 = 462
# Sequences longer than 462 tokens receive penalty
```

## Common Issues and Solutions

### Issue 1: Out of Memory (OOM)

**Symptoms**: CUDA out of memory error during experience generation.

**Solutions**:
1. Enable sample packing: `packing_samples=True`
2. Reduce micro batch size: `strategy.config.micro_rollout_batch_size = 4`
3. Enable gradient checkpointing: `actor.gradient_checkpointing_enable()`
4. Reduce max sequence length: `max_new_tokens=256`
5. Use smaller model or quantization

### Issue 2: Slow Generation Speed

**Symptoms**: Experience generation takes too long.

**Solutions**:
1. Use VLLM backend: `strategy.args.engine_type = "vllm"`
2. Increase batch size: `strategy.config.micro_rollout_batch_size = 16`
3. Enable sample packing: `packing_samples=True`
4. Check GPU utilization: Ensure GPU is fully utilized
5. Reduce `max_new_tokens` if possible

### Issue 3: Unstable Training

**Symptoms**: Reward or loss fluctuates wildly during training.

**Solutions**:
1. Enable reward normalization:
   ```python
   strategy.args.reward_running_norm = True
   strategy.args.reward_running_norm_minus_mean = True
   ```
2. Enable advantage normalization:
   ```python
   strategy.config.advantages_norm = True
   ```
3. Add reward clipping:
   ```python
   strategy.config.reward_clip = 10.0
   ```
4. Reduce learning rate
5. Use GAE with appropriate lambda: `strategy.config.lambd = 0.95`

### Issue 4: Image Token Mismatch (VLM)

**Symptoms**: Warning message about token/patch mismatch during rollout.

**Cause**: Number of image tokens doesn't match pixel value patches.

**Solution**: This is automatically fixed by FastExperienceMaker. The warning is informational only. If it occurs frequently:
1. Check image preprocessing pipeline
2. Verify processor configuration
3. Ensure consistent image format across samples

### Issue 5: RLOO Requires Multiple Samples

**Symptoms**: Error when using RLOO with `n_samples_per_prompt = 1`.

**Cause**: RLOO requires multiple samples per prompt for baseline computation.

**Solution**:
```python
# Set n_samples_per_prompt > 1
strategy.config.n_samples_per_prompt = 4
strategy.config.advantage_estimator = "rloo"
```

Or switch to another method:
```python
# Use GAE or REINFORCE instead
strategy.config.advantage_estimator = "gae"
```

### Issue 6: Remote Reward Model Timeout

**Symptoms**: Timeout errors when using remote reward models.

**Solutions**:
1. Check network connectivity to reward model server
2. Increase timeout in remote_rm_fn configuration
3. Reduce batch size to avoid long processing times
4. Consider using local reward models for better performance
5. Implement retry logic in custom reward function

## Performance Tuning

### Throughput Optimization

**Recommended Configuration for Maximum Throughput**:
```python
strategy.args.engine_type = "vllm"
strategy.config.micro_rollout_batch_size = 16  # Adjust based on GPU memory
exp_maker = FastExperienceMaker(
    packing_samples=True,
    # ...
)
```

**Expected Performance**:
- VLLM backend: 2-5x faster than HuggingFace generate
- Sample packing: 30-50% reduction in padding overhead
- Batch processing: Linear scaling with batch size (up to GPU memory limit)

### Memory Efficiency

**Recommended Configuration for Memory-Constrained Environments**:
```python
strategy.config.micro_rollout_batch_size = 4
strategy.config.max_new_tokens = 256
exp_maker = FastExperienceMaker(
    packing_samples=True,
    # ...
)
actor.gradient_checkpointing_enable()
```

## References

### Related Documentation
- [Strategy Design Philosophy](strategy_design_philosophy.md)
- [Model Design Document](model.md)
- [Reward Model Best Practices](reward_model.md)

### Code References
- FastExperienceMaker: `lightrft/trainer/fast_exp_maker.py`
- Base ExperienceMaker: `lightrft/trainer/experience_maker.py`
- Advantage Calculators: `lightrft/trainer/advantage_calculator.py`
- VLLM Utils: `lightrft/strategy/vllm_utils/`

### Research Papers
- **GAE**: "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (Schulman et al., 2016)
- **PPO**: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- **RLOO**: "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs" (Ahmadian et al., 2024)

---

**Document Version**: 1.0
**Last Updated**: 2026-02-03
**Maintainer**: LightRFT Team
