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

LightRFT 策略模块优化了分布式训练的抽象设计，旨在提升 RLHF 系统的灵活性、类型安全与开发效率。通过统一抽象层与配置驱动开发，该模块实现了不同训练框架间的互操作性。

### 核心设计

*   **统一接口架构**：封装 DeepSpeed、FSDP 等分布式后端，提供一致的 API。开发者无需修改业务代码即可切换底层策略。
*   **类型安全配置**：通过 [`StrategyConfig`](../../lightrft/strategy/config.py) 将动态配置转为强类型对象，减少运行时错误，并支持 IDE 自动补全。
*   **工厂模式选择**：[`get_strategy()`](../../lightrft/strategy/strategy.py) 根据配置参数自动实例化策略，在简化调用的同时保留了对后端的控制权。

### 功能特性

*   **推理引擎集成**：通过统一接口支持纯文本及多模态生成，兼容 vLLM 和 SGLang 后端。
*   **便捷测试支持**：提供 [`FakeStrategy`](../../lightrft/strategy/fake_strategy.py)，允许在无分布式环境下测试训练工作流，降低调试成本。
*   **资源效率优化**：支持推理引擎休眠/唤醒、梯度累积及内存感知检查点，优化大规模训练时的资源使用。

### 总结

该模块在易用性与灵活性之间取得了平衡。其架构设计兼顾了开发体验与性能需求，既降低了分布式训练的上手门槛，也为从 RLHF 到 RLVR 的多样化场景提供了可扩展的技术基础。