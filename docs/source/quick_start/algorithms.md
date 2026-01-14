# Supported Algorithms

LightRFT supports a rich ecosystem of reinforcement learning algorithms for fine-tuning large language models. This comprehensive guide provides algorithm details and implementation references.

### Purpose of This Guide

With the rapid development in the RFT field and emerging algorithmic innovations, this guide helps you:

1. **Quickly identify** which algorithms suit your needs
2. **Understand implementation** by mapping algorithms to code modules
3. **Plan integration** of multiple algorithms by identifying synergies or conflicts
4. **Maintain clarity** through documented relationships between algorithms and components

### Algorithm Overview with Implementation

| Algorithm | Type | Module | Description | Implementation | Paper |
|-----------|------|--------|-------------|----------------|-------|
| **GRPO** | Policy Optimization | Advantage Estimation | Uses group-based normalization for advantage estimation without requiring a separate value network | `FastExperienceMaker._get_return_advs()` | [arXiv:2402.03300](https://arxiv.org/pdf/2402.03300) |
| **GSPO** | Policy Optimization | Policy Loss | Group sequence policy optimization | `PolicyLoss.forward()` | [arXiv:2507.18071](https://arxiv.org/abs/2507.18071) |
| **REINFORCE++** | Advantage Estimation | Advantage Estimation | Modifies return and advantage calculation with improved baseline estimation | `FastExperienceMaker._get_return_advs()` | [arXiv:2501.03262](https://arxiv.org/abs/2501.03262) |
| **CPGD** | Advantage Estimation | Advantage Estimation | Adds KL-based drift constraint and clipped log-ratio for stable return/advantage computation | `FastExperienceMaker._get_return_advs()` | [arXiv:2505.12504](https://arxiv.org/abs/2505.12504) |
| **FIRE Sampling** | Sampling Strategy | Experience Generation | Modifies sample generation process with filtering and ranking strategies | `FastExperienceMaker.generate_samples()` | [arXiv:2410.21236](https://arxiv.org/abs/2410.21236) |
| **GMPO** | Policy Optimization | Policy Loss | Geometric-Mean Policy Optimization | `PolicyLoss.forward()` | [arXiv:2507.20673](https://arxiv.org/abs/2507.20673) |
| **Dr.GRPO** | Policy Optimization | Policy Loss | Introduces an unbiased policy optimization to mitigate length bias and improve token efficiency | `PolicyLoss.forward()` | [arXiv:2503.20783](https://arxiv.org/abs/2503.20783) |
| **DAPO** | Policy Optimization | Policy Loss | Introduces decoupled clipping and dynamic sampling scheme to stabilize large-scale RL optimization | `PolicyLoss.forward()` | [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) |
| **Token-Level Policy** | Policy Optimization | Policy Loss | Optimizes policy at token granularity to improve stability and credit assignment | `PolicyLoss.forward()` | [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) |
| **Reward Norm/Clip** | Reward Processing | Reward Processing | Applies reward normalization and clipping to stabilize advantage computation | `FastExperienceMaker._get_return_advs()` | [GitHub](https://github.com/alibaba/ROLL) |
| **select_high_entropy_tokens** | Policy Optimization | Policy Loss | Modifies PolicyLoss to implement high entropy token selection during training |  `PolicyLoss.forward()` | [arXiv:2506.01939](https://arxiv.org/abs/2506.01939) |

### Algorithm Architecture

#### Core Training Components

LightRFT's algorithm implementations are organized around three main modules:

##### 1. Policy Loss Computation (`lightrft/trainer/ppo_loss.py`)
- **Purpose**: Implements PPO policy loss with multiple surrogate objectives
- **Key Method**: `forward(log_probs, old_log_probs, advantages, action_mask)`
- **Affected by**: GSPO, GMPO, Dr.GRPO, DAPO, Token-Level Policy, select_high_entropy_tokens
- **Modification Type**: Loss function design and token selection strategies

##### 2. Experience Generation (`lightrft/trainer/fast_exp_maker.py`)
- **Purpose**: Generates experiences using vLLM and other inference backends
- **Key Methods**:
  - `generate_samples()`: Sample generation with various strategies
  - `_get_return_advs()`: Returns and advantages calculation
- **Affected by**: FIRE Sampling
- **Modification Type**: Sampling strategies and inference optimization

##### 3. Advantage & Reward Processing (`lightrft/trainer/fast_exp_maker.py`)
- **Purpose**: Processes rewards and computes advantages for policy updates
- **Key Method**: `_get_return_advs()`: Advantage estimation with various baselines
- **Affected by**: GRPO, REINFORCE++, CPGD, Reward Norm/Clip
- **Modification Type**: Advantage estimation methods and reward shaping

#### Modification Types

**Algorithmic Changes**:
- **Loss Design**: Core objective function modifications
- **Advantage Estimation**: Updates to advantage calculation methods
- **Sampling Strategy**: Changes to sample generation processes
- **Token Selection**: Which tokens are used in training
- **Reward Shaping**: Reward preprocessing and filtering

**Implementation Changes**:
- **Efficiency Optimization**: Performance improvements (e.g., FP8)
- **Parameter Tuning**: Hyperparameter adjustments
- **Pipeline Integration**: New components or workflow changes

### Policy Optimization Algorithms

#### GRPO (Group Relative Policy Optimization)

**Overview**: GRPO uses group-based normalization for advantage estimation, providing stable training without requiring a separate value network.

**Implementation**: `FastExperienceMaker._get_return_advs()` - Advantage Estimation module
**Modification Type**: Advantage Estimation

**Key Features**:
- No critic network required
- Group-normalized advantages
- Stable training with large batch sizes
- Memory efficient

**Usage**:
```bash
python train.py \
    --advantage_estimator group_norm \
    --n_samples_per_prompt 8 \
    --kl_estimator k3
```

**Best For**:
- Large-scale training with limited memory
- Quick prototyping without value network
- Math reasoning and coding tasks

---

#### GSPO (Group Sequence Policy Optimization)

**Overview**: GSPO generalizes the PPO objective with flexible surrogate functions, allowing for better control over policy updates.

**Implementation**: `PolicyLoss.forward()` - Policy Loss module
**Modification Type**: Loss Design

**Key Features**:
- Generalized clipping objectives
- Adaptive trust region updates
- Better sample efficiency

**Usage**:
```bash
python train.py \
    --advantage_estimator gspo \
    --gspo_alpha 0.1 \
    --clip_range 0.2
```

**Best For**:
- Tasks requiring precise policy control
- Multi-task learning scenarios

---

#### GMPO (Geometric-Mean Policy Optimization)

**Overview**: GMPO leverages mirror descent principles for policy optimization, providing theoretical guarantees and improved convergence.

**Implementation**: `PolicyLoss.forward()` - Policy Loss module
**Modification Type**: Loss Design

**Key Features**:
- Mirror descent updates
- Theoretical convergence guarantees
- Adaptive step sizes

**Usage**:
```bash
python train.py \
    --advantage_estimator gmpo \
    --mirror_tau 0.01
```

**Best For**:
- Research applications requiring theoretical guarantees
- Complex reward landscapes

---

#### Dr.GRPO (Group Relative Policy Optimization Done Right)

**Overview**: Dr.GRPO addresses length bias in reward models by explicitly modeling and mitigating the reward-length correlation.

**Implementation**: `PolicyLoss.forward()` - Policy Loss module
**Modification Type**: Loss Design (length bias mitigation)

**Key Features**:
- Length bias mitigation
- Reward debiasing mechanisms
- Improved response quality

**Usage**:
```bash
python train.py \
    --advantage_estimator group_norm \
    --use_length_penalty \
    --length_penalty_coef 0.01
```

**Best For**:
- Tasks sensitive to response length
- Instruction following
- Open-ended generation

---

#### DAPO (Dynamic sAmpling Policy Optimization)

**Overview**: DAPO uses separate upper and lower clipping bounds for advantage-weighted policy updates combined with dynamic sampling strategies, improving training stability.

**Implementation**: `PolicyLoss.forward()` - Policy Loss module
**Modification Type**: Loss Design (decoupled clipping)

**Key Features**:
- Decoupled clipping for positive/negative advantages
- Dynamic sampling strategy
- Better handling of distribution shifts
- Improved stability

**Usage**:
```bash
python train.py \
    --use_clip_higher \
    --clip_range_higher 0.3 \
    --clip_range_lower 0.2
```

**Best For**:
- Highly noisy reward signals
- Large distribution shifts
- Challenging domains

---

#### Token-Level Policy

**Overview**: Optimizes policy at token granularity to improve stability and credit assignment.

**Implementation**: `PolicyLoss.forward()` - Policy Loss module
**Modification Type**: Token Selection

**Key Features**:
- Token-granular optimization
- Improved credit assignment
- Better stability in long sequences

**Usage**: Typically combined with other policy optimization methods through implementation modifications.

### Advantage Estimation Methods

#### REINFORCE++

**Overview**: An improved baseline estimation method that uses control variates to reduce variance in policy gradient estimates.

**Implementation**: `FastExperienceMaker._get_return_advs()` - Advantage Estimation module
**Modification Type**: Advantage Estimation

**Key Features**:
- Lower variance gradients
- Faster convergence
- Compatible with all policy optimization methods

**Usage**:
```bash
python train.py \
    --advantage_estimator reinforce_plus \
    --baseline_type value_network
```

**Best For**:
- High-variance environments
- Sparse rewards
- Combining with PPO or other on-policy methods

---

#### CPGD (Clipped Policy Gradient Optimization with Policy Drift)

**Overview**: CPGD constrains policy updates using KL-divergence to prevent catastrophic forgetting and maintain stable training.

**Implementation**: `FastExperienceMaker._get_return_advs()` - Advantage Estimation module
**Modification Type**: Advantage Estimation (KL-constrained)

**Key Features**:
- KL-constrained updates
- Prevents catastrophic forgetting
- Adaptive constraint adjustment

**Usage**:
```bash
python train.py \
    --advantage_estimator cpgd \
    --kl_target 0.01 \
    --kl_horizon 10000
```

**Best For**:
- Fine-tuning pre-trained models
- Preserving original capabilities
- Multi-stage training

### Reward Processing

#### Reward Normalization and Clipping

**Overview**: Standard reward preprocessing techniques to stabilize training.

**Implementation**: `FastExperienceMaker._get_return_advs()` - Reward Processing module
**Modification Type**: Reward Shaping (normalization/clipping)

**Key Features**:
- Running reward statistics
- Advantage normalization
- Reward clipping

**Usage**:
```bash
python train.py \
    --reward_running_norm \
    --reward_running_norm_minus_mean \
    --reward_clip 10.0 \
    --advantage_clip 10.0
```

**Best For**:
- All training scenarios (recommended baseline)
- Reward scale varies across prompts
- Training stability

### Sampling Strategies

#### FIRE Sampling

**Overview**: FIRE (Filtered and Improved Reward Estimation) combines filtering and ranking strategies for better sample selection.

**Implementation**: `FastExperienceMaker.generate_samples()` - Experience Generation module
**Modification Type**: Sampling Strategy

**Key Features**:
- Multi-stage filtering
- Reward-based ranking
- Sample efficiency

**Usage**:
```bash
python train.py \
    --use_fire_sampling \
    --fire_filter_ratio 0.5 \
    --fire_rank_method reward
```

**Best For**:
- Limited computational budgets
- High-quality data generation
- Best-of-N sampling scenarios



### Implementation Notes

- All policy loss algorithms modify the **PolicyLoss** module's `forward()` method
- Advantage estimation algorithms modify **FastExperienceMaker**'s `_get_return_advs()` method
- Sampling strategies modify **FastExperienceMaker**'s `generate_samples()` method
- Reward processing algorithms primarily work within `_get_return_advs()` method
- Most modifications are in core training loop components rather than peripheral utilities

### References

For detailed algorithm descriptions and experimental results, refer to the linked papers. Implementation details can be found in the source code:
- Policy Loss: `lightrft/models/loss.py`
- Experience Maker: `lightrft/trainer/fast_exp_maker.py`
- vLLM Utils: `lightrft/strategy/vllm_utils/`
