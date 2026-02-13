# LightRFT 策略设计理念

## 概述

LightRFT 的策略（Strategy）模块为分布式训练策略提供了一个统一的接口，使得在保持 API 一致性的同时，能够无缝切换不同的分布式训练后端。本文档概述了策略模块的设计原则、架构和使用模式。

## 核心设计原则

### 1. 抽象与统一

**原则**：提供统一的接口，抽象掉不同分布式训练框架的复杂性。

**实现**：
- 所有策略均继承自 [`StrategyBase`](../../lightrft/strategy/strategy_base.py)
- 通用方法如 [`backward()`](../../lightrft/strategy/strategy_base.py)、[`optimizer_step()`](../../lightrft/strategy/strategy_base.py) 和 [`save_ckpt()`](../../lightrft/strategy/strategy_base.py) 具有一致的签名。
- 特定策略的实现封装在具体的策略类中。

### 2. 配置驱动设计

**原则**：使用类型化的配置对象（Typed Configuration Objects）代替动态属性访问，以实现更好的类型安全性和代码清晰度。

**实现**：
- [`StrategyConfig`](../../lightrft/strategy/config.py) 数据类提供了对所有配置参数的类型化访问。
- 消除了对 `getattr(args, "parameter", default)` 模式的需求。
- 支持 IDE 自动补全和静态类型检查。

### 3. 向后兼容性

**原则**：在引入改进的同时，保持与现有代码的兼容性。

**实现**：
- [`StrategyConfig.from_args()`](../../lightrft/strategy/config.py) 方法可从旧版本的参数对象中提取参数。
- 保留原始 `args` 对象以保持兼容性。
- [`get_extra_arg()`](../../lightrft/strategy/config.py) 方法提供了对非标准参数的访问。

### 4. 可测试性

**原则**：无需分布式环境即可进行全面测试。

**实现**：
- [`FakeStrategy`](../../lightrft/strategy/fake_strategy.py) 提供了一个可用于测试的直接替代方案。
- 所有策略方法都有用于单进程测试的 Mock 实现。
- 单元测试验证功能和 API 的一致性。

## 架构

### 策略层次结构

```
StrategyBase (抽象基类)
├── DeepspeedStrategy (DeepSpeed 策略)
├── FSDPV2Strategy    (FSDP 策略)
└── FakeStrategy      (用于测试的伪策略)
```

### 关键组件

#### 1. 策略工厂

[`get_strategy()`](../../lightrft/strategy/strategy.py) 函数作为入口点，根据配置自动选择合适的策略：

```python
from lightrft.strategy import get_strategy

# 根据 args.fsdp 自动选择 DeepSpeed 或 FSDP
strategy = get_strategy(args)
```

#### 2. 配置管理

[`StrategyConfig`](../../lightrft/strategy/config.py) 类集中管理所有配置参数：

```python
from lightrft.strategy.config import StrategyConfig

config = StrategyConfig.from_args(args)
# 类型安全地访问参数
learning_rate = config.actor_learning_rate
use_bf16 = config.bf16
```

#### 3. 通用接口

所有策略都实现相同的核心接口：

```python
class StrategyBase(ABC):
    def setup_distributed(self, timeout=None) -> None: ...
    def create_optimizer(self, model, **kwargs) -> Optimizer: ...
    def prepare(self, *models, is_rlhf=False) -> Any: ...
    def backward(self, loss, model, optimizer, **kwargs) -> None: ...
    def optimizer_step(self, optimizer, model, scheduler, **kwargs) -> None: ...
    def save_ckpt(self, model, save_dir, **kwargs) -> None: ...
    def load_ckpt(self, model, load_dir, **kwargs) -> Any: ...
```

## 使用模式

### 1. 基本用法

```python
from lightrft.strategy import get_strategy

# 初始化策略
strategy = get_strategy(args)

# 准备模型和优化器
actor, critic, reward_models, initial_model = strategy.prepare_models_and_optimizers(
    actor, critic, reward_models, initial_model, args, max_steps
)

# 训练循环
for batch in dataloader:
    loss = compute_loss(batch)
    strategy.backward(loss, actor, actor_optimizer)
    strategy.optimizer_step(actor_optimizer, actor, actor_scheduler)
```

### 2. 配置驱动用法

```python
from lightrft.strategy.config import StrategyConfig

# 创建配置
config = StrategyConfig(
    seed=42,
    max_norm=1.0,
    micro_train_batch_size=4,
    train_batch_size=32,
    bf16=True,
    zero_stage=2
)

# 使用配置创建策略
strategy = get_strategy(config)
```

### 3. 使用 FakeStrategy 进行测试

```python
from lightrft.strategy import get_fake_strategy

# 使用伪策略进行测试
strategy = get_fake_strategy()

# 所有操作无需分布式环境即可运行
strategy.setup_distributed()
strategy.backward(loss, model, optimizer)
strategy.save_ckpt(model, "checkpoints")
```

## 设计优势

### 1. 提高类型安全性

**改进前** (使用 getattr):
```python
seed = getattr(args, "seed", 42)  # 类型: Any
max_norm = getattr(args, "max_norm", 1.0)  # 类型: Any
```

**改进后** (使用 StrategyConfig):
```python
config = StrategyConfig.from_args(args)
seed = config.seed  # 类型: int
max_norm = config.max_norm  # 类型: float
```

### 2. 更好的代码组织

- 配置参数在 [`StrategyConfig`](../../lightrft/strategy/config.py) 中明确定义。
- 策略特定的逻辑封装在具体的策略类中。
- 通用功能在 [`StrategyBase`](../../lightrft/strategy/strategy_base.py) 中实现。

### 3. 增强可测试性

- [`FakeStrategy`](../../lightrft/strategy/fake_strategy.py) 使得无需分布式设置即可进行测试。
- 单元测试可以验证所有策略功能。
- Mock 实现确保了行为的一致性。

### 4. 未来的扩展性

- 通过实现 [`StrategyBase`](../../lightrft/strategy/strategy_base.py) 接口，可以轻松添加新策略。
- 配置可以扩展而不破坏现有代码。
- 工厂模式使得添加新策略类型变得非常简单。

## 最佳实践

### 1. 配置管理

- 使用 [`StrategyConfig`](../../lightrft/strategy/config.py) 进行所有参数访问。
- 避免直接对参数对象调用 `getattr`。
- 对于非标准参数，使用 [`get_extra_arg()`](../../lightrft/strategy/config.py)。

### 2. 策略选择

- 使用 [`get_strategy()`](../../lightrft/strategy/strategy.py) 工厂函数创建策略。
- 让工厂根据配置确定适当的策略。
- 在开发和测试中使用 [`FakeStrategy`](../../lightrft/strategy/fake_strategy.py)。

### 3. 错误处理

- 策略应对不支持的操作提供清晰的错误消息。
- 使用策略的 [`print()`](../../lightrft/strategy/strategy_base.py) 方法进行日志记录。
- 在上下文管理器中实现适当的清理。

### 4. 测试

- 在单元测试中使用 [`FakeStrategy`](../../lightrft/strategy/fake_strategy.py)。
- 测试策略特定的功能和通用功能。
- 验证所有策略是否都实现了所需的接口。

## 结论

LightRFT 策略模块代表了分布式训练抽象设计的模式转变，为 RLHF 训练系统中的灵活性、类型安全和开发者体验树立了新标准。通过采用以抽象统一和配置驱动开发为核心的整体设计理念，该模块在各种分布式训练框架之间实现了前所未有的互操作性。

### 基础设施设计创新

**统一抽象架构**：该模块引入了一个复杂的抽象层，将 DeepSpeed、FSDP 和其他分布式训练后端无缝统一在单一、一致的 API 之下。这一架构突破消除了不同分布式训练方法之间的传统碎片化，使开发者能够零代码更改地在策略之间切换。

**类型安全配置生态系统**：通过 [`StrategyConfig`](../../lightrft/strategy/config.py) 将动态属性访问模式替换为强类型配置对象，该模块建立了配置管理的新模式，消除了运行时错误，支持 IDE 自动补全，并提供编译时安全性保证。

**智能策略选择**：[`get_strategy()`](../../lightrft/strategy/strategy.py) 中的工厂方法实现了基于配置参数的自动策略选择，在抽象后端选择复杂性的同时，通过显式配置保持了完全的控制。

### 先进能力

**多模态推理引擎集成**：策略模块率先通过复杂的引擎管理，实现了对纯文本和多模态生成的统一支持，通过一致的接口支持 vLLM 和 SGLang 后端，以应对多样化的 AI 工作负载。

**全面的测试基础设施**：通过 [`FakeStrategy`](../../lightrft/strategy/fake_strategy.py)，该模块支持在无需复杂分布式环境的情况下对分布式训练工作流进行全功能测试，极大地提高了开发速度和测试覆盖率。

**内存高效的训练编排**：推理引擎休眠/唤醒循环、梯度累积优化和感知内存的检查点保存等先进功能，体现了该模块在大规模资源效率方面的追求。

### 行业影响

这一设计理念为分布式训练抽象建立了新的最佳实践，证明了类型安全、开发者体验和性能优化并不是互斥的目标。该模块的可扩展架构确保了长期生命力，为集成未来分布式训练技术提供了坚实基础，同时保持了与现有代码库的向后兼容性。

LightRFT 的策略模块可以看作是下一代分布式训练系统的一个参考实现。它通过良好的抽象设计，在保证上手简单的同时，也为有经验的用户提供了足够的灵活性，能够覆盖从 RLHF 到 RLVR 的各种训练需求。
