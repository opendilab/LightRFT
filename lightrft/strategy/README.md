# LightRFT Strategy Module

The LightRFT strategy module provides a unified interface for distributed training strategies, enabling seamless switching between different distributed training backends while maintaining a consistent API.

## Overview

This module abstracts the complexities of different distributed training frameworks (DeepSpeed, FSDP) behind a common interface, allowing you to:

- Switch between training backends with minimal code changes
- Maintain type safety through configuration-driven design
- Test distributed training workflows without distributed environments
- Extend the framework with new strategies easily

## Core Design Principles

### 1. Abstraction and Unification

All strategies inherit from `StrategyBase` and implement a common interface. This allows you to write training code once and run it with any supported backend.

**Key Features:**
- Unified methods: `backward()`, `optimizer_step()`, `save_ckpt()`, `load_ckpt()`
- Consistent signatures across all strategies
- Strategy-specific implementations are encapsulated

### 2. Configuration-Driven Design

Use typed configuration objects (`StrategyConfig`) instead of dynamic attribute access for better type safety and code clarity.

**Benefits:**
- Type checking and IDE autocompletion
- Clear parameter definitions
- Reduced runtime errors

### 3. Backward Compatibility

The module maintains compatibility with existing code while introducing improvements:
- `StrategyConfig.from_args()` extracts parameters from legacy argument objects
- Original `args` object is preserved
- `get_extra_arg()` provides access to non-standard parameters

### 4. Testability

`FakeStrategy` enables comprehensive testing without requiring distributed environments, providing drop-in mock implementations for all strategy methods.

## Architecture

### Strategy Hierarchy

```
StrategyBase (Abstract Base Class)
├── DeepspeedStrategy      # DeepSpeed (Zero-1, Zero-2, Zero-3) support
├── FSDPV2Strategy         # PyTorch FSDP (Fully Sharded Data Parallel) support
└── FakeStrategy           # Testing and development mock
```

### Directory Structure

```
strategy/
├── __init__.py                 # Module exports
├── strategy.py                 # Strategy factory (get_strategy)
├── strategy_base.py            # Base class defining common interface
├── config.py                   # StrategyConfig for typed configuration
├── fake_strategy.py            # FakeStrategy for testing
├── test_fake_strategy.py       # Unit tests
├── deepspeed/                  # DeepSpeed implementation
│   ├── __init__.py
│   ├── deepspeed.py           # DeepspeedStrategy class
│   └── deepspeed_utils.py     # DeepSpeed utilities
├── fsdp/                       # FSDP implementation
│   ├── __init__.py
│   ├── fsdpv2.py              # FSDPV2Strategy class
│   ├── fsdp_optimizer.py      # FSDP optimizer wrapper
│   └── fsdp_utils.py          # FSDP utilities
├── vllm_utils/                 # vLLM inference engine utilities
├── sglang_utils/               # SGLang inference engine utilities
└── utils/                      # Common utilities
```

## Key Components

### 1. Strategy Factory

The `get_strategy()` function automatically selects the appropriate strategy based on configuration:

```python
from lightrft.strategy import get_strategy

# Automatically selects DeepSpeed or FSDP based on args.fsdp
strategy = get_strategy(args)
```

### 2. Configuration Management

`StrategyConfig` centralizes all configuration parameters with type safety:

```python
from lightrft.strategy import StrategyConfig

# Extract from args object
config = StrategyConfig.from_args(args)

# Access with type safety
learning_rate = config.actor_learning_rate  # Type: float
use_bf16 = config.bf16                      # Type: bool
```

### 3. Strategy Base Interface

All strategies implement this core interface:

```python
class StrategyBase(ABC):
    def setup_distributed(self, timeout=None) -> None:
        """Initialize distributed environment"""

    def create_optimizer(self, model, **kwargs) -> Optimizer:
        """Create optimizer for the model"""

    def prepare(self, *models, is_rlhf=False) -> Any:
        """Prepare models for distributed training"""

    def backward(self, loss, model, optimizer, **kwargs) -> None:
        """Perform backward pass"""

    def optimizer_step(self, optimizer, model, scheduler, **kwargs) -> None:
        """Perform optimizer step"""

    def save_ckpt(self, model, save_dir, **kwargs) -> None:
        """Save checkpoint"""

    def load_ckpt(self, model, load_dir, **kwargs) -> Any:
        """Load checkpoint"""
```

## Usage Examples

### Basic Usage

```python
from lightrft.strategy import get_strategy

# Initialize strategy
strategy = get_strategy(args)

# Setup distributed environment
strategy.setup_distributed()

# Prepare models
actor = strategy.prepare(actor, is_rlhf=True)

# Create optimizer
optimizer = strategy.create_optimizer(actor)

# Training loop
for batch in dataloader:
    loss = compute_loss(batch)
    strategy.backward(loss, actor, optimizer)
    strategy.optimizer_step(optimizer, actor, scheduler)

# Save checkpoint
strategy.save_ckpt(actor, "checkpoints/step_1000")
```

### Configuration-Driven Usage

```python
from lightrft.strategy.config import StrategyConfig
from lightrft.strategy import get_strategy

# Create typed configuration
config = StrategyConfig(
    seed=42,
    max_norm=1.0,
    micro_train_batch_size=4,
    train_batch_size=32,
    bf16=True,
    fsdp=False,  # Use DeepSpeed
    zero_stage=2
)

# Create strategy from configuration
strategy = get_strategy(config)
```

### Testing with FakeStrategy

```python
from lightrft.strategy.fake_strategy import FakeStrategy

# Use fake strategy for unit testing
strategy = FakeStrategy()

# All operations work without distributed environment
strategy.setup_distributed()
strategy.backward(loss, model, optimizer)
strategy.save_ckpt(model, "test_checkpoints")
```

### Switching Between Backends

```python
# Use DeepSpeed
args.fsdp = False
args.zero_stage = 2
strategy = get_strategy(args)

# Or use FSDP (just change one flag)
args.fsdp = True
strategy = get_strategy(args)

# Rest of your code remains unchanged!
```

## Supported Features

### DeepSpeed Strategy
- Zero-1, Zero-2, Zero-3 optimization stages
- Mixed precision training (BF16/FP16)
- Gradient accumulation
- Gradient clipping
- Checkpoint saving/loading
- vLLM/SGLang inference engine integration

### FSDP Strategy
- Fully Sharded Data Parallel training
- Mixed precision training (BF16/FP16)
- Gradient accumulation
- Gradient clipping
- Checkpoint saving/loading (DCP format)
- CPU offloading support

### Common Features
- Unified training interface
- Automatic gradient accumulation
- Memory-efficient checkpointing
- Inference engine sleep/wake management
- Distributed logging and printing

## Best Practices

### 1. Use StrategyConfig for Parameter Access

**Good:**
```python
config = StrategyConfig.from_args(args)
seed = config.seed  # Type: int, IDE autocomplete available
```

**Avoid:**
```python
seed = getattr(args, "seed", 42)  # Type: Any, no autocomplete
```

### 2. Let the Factory Choose the Strategy

**Good:**
```python
strategy = get_strategy(args)  # Factory selects appropriate backend
```

**Avoid:**
```python
if args.fsdp:
    strategy = FSDPV2Strategy(...)
else:
    strategy = DeepspeedStrategy(...)
```

### 3. Use FakeStrategy for Testing

```python
# In test files
def test_training_loop():
    strategy = FakeStrategy()
    # Test your training code without distributed setup
    assert strategy.is_rank_0()
```

### 4. Consistent Logging

Use the strategy's `print()` method for rank-0 only logging:

```python
strategy.print(f"Training step {step}, loss: {loss.item()}")
# Only prints on rank 0 in distributed training
```

## Advanced Features

### Inference Engine Integration

The strategy module supports integration with vLLM and SGLang inference engines for efficient generation during RLHF training:

```python
# Engines can be put to sleep to save memory
strategy.engines_sleep([vllm_engine, sglang_engine])

# Wake them up when needed
strategy.engines_wake_up([vllm_engine, sglang_engine])
```

### Multi-Model Preparation

Prepare multiple models at once for RLHF training:

```python
actor, critic, reward_model, ref_model = strategy.prepare(
    actor, critic, reward_model, ref_model,
    is_rlhf=True
)
```

### Custom Optimizer Creation

Create optimizers with strategy-specific optimizations:

```python
optimizer = strategy.create_optimizer(
    model,
    lr=1e-5,
    betas=(0.9, 0.95),
    weight_decay=0.1
)
```

## Extending the Framework

To add a new strategy:

1. Create a new class inheriting from `StrategyBase`
2. Implement all abstract methods
3. Add your strategy to the factory in `strategy.py`

```python
from lightrft.strategy.strategy_base import StrategyBase

class MyCustomStrategy(StrategyBase):
    def setup_distributed(self, timeout=None):
        # Your implementation
        pass

    def backward(self, loss, model, optimizer, **kwargs):
        # Your implementation
        pass

    # Implement other required methods...
```

## Performance Considerations

- **Memory**: FSDP typically uses less memory than DeepSpeed Zero-3 for the same model
- **Speed**: DeepSpeed may be faster for smaller models; FSDP scales better for very large models
- **Checkpointing**: FSDP uses PyTorch DCP format; DeepSpeed uses its own format
- **Compatibility**: Consider checkpoint format when switching between strategies

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure DeepSpeed or FSDP dependencies are installed
2. **Checkpoint incompatibility**: Use conversion scripts in `utils/ckpt_scripts/`
3. **Out of memory**: Try increasing `zero_stage` (DeepSpeed) or enabling CPU offload (FSDP)
4. **Gradient explosion**: Check `max_norm` setting for gradient clipping

## References

- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [LightRFT Strategy Design Philosophy](../../docs/source/best_practice/strategy_design_philosophy.md)

## Summary

The LightRFT strategy module provides a production-ready abstraction layer for distributed training, combining:
- **Flexibility**: Easy switching between training backends
- **Type Safety**: Configuration-driven design with type checking
- **Testability**: Comprehensive testing without distributed setup
- **Extensibility**: Simple addition of new strategies

This design enables you to focus on your training logic while the strategy module handles the complexities of distributed training.
