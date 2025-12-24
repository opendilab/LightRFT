# LightRFT 策略模块

LightRFT 策略模块提供了统一的分布式训练策略接口，支持在不同的分布式训练后端之间无缝切换，同时保持一致的 API。

## 概述

本模块将不同分布式训练框架（DeepSpeed、FSDP）的复杂性抽象到统一的接口之后，使您能够：

- 以最小的代码改动切换训练后端
- 通过配置驱动的设计保持类型安全
- 在无分布式环境下测试分布式训练工作流
- 轻松扩展框架并添加新策略

## 核心设计原则

### 1. 抽象与统一

所有策略都继承自 `StrategyBase` 并实现统一的接口。这使您能够编写一次训练代码，然后在任何支持的后端上运行。

**关键特性：**
- 统一的方法：`backward()`、`optimizer_step()`、`save_ckpt()`、`load_ckpt()`
- 所有策略的签名保持一致
- 策略特定的实现被封装在具体策略类中

### 2. 配置驱动设计

使用类型化配置对象（`StrategyConfig`）而不是动态属性访问，以获得更好的类型安全性和代码清晰度。

**优势：**
- 类型检查和 IDE 自动补全
- 清晰的参数定义
- 减少运行时错误

### 3. 向后兼容性

模块在引入改进的同时保持与现有代码的兼容性：
- `StrategyConfig.from_args()` 从传统参数对象中提取参数
- 保留原始 `args` 对象
- `get_extra_arg()` 提供对非标准参数的访问

### 4. 可测试性

`FakeStrategy` 支持在无需分布式环境的情况下进行全面测试，为所有策略方法提供即插即用的模拟实现。

## 架构

### 策略层次结构

```
StrategyBase (抽象基类)
├── DeepspeedStrategy      # DeepSpeed (Zero-1、Zero-2、Zero-3) 支持
├── FSDPV2Strategy         # PyTorch FSDP (全分片数据并行) 支持
└── FakeStrategy           # 测试和开发模拟
```

### 目录结构

```
strategy/
├── __init__.py                 # 模块导出
├── strategy.py                 # 策略工厂 (get_strategy)
├── strategy_base.py            # 定义通用接口的基类
├── config.py                   # 类型化配置的 StrategyConfig
├── fake_strategy.py            # 用于测试的 FakeStrategy
├── test_fake_strategy.py       # 单元测试
├── deepspeed/                  # DeepSpeed 实现
│   ├── __init__.py
│   ├── deepspeed.py           # DeepspeedStrategy 类
│   └── deepspeed_utils.py     # DeepSpeed 工具函数
├── fsdp/                       # FSDP 实现
│   ├── __init__.py
│   ├── fsdpv2.py              # FSDPV2Strategy 类
│   ├── fsdp_optimizer.py      # FSDP 优化器包装器
│   └── fsdp_utils.py          # FSDP 工具函数
├── vllm_utils/                 # vLLM 推理引擎工具
├── sglang_utils/               # SGLang 推理引擎工具
└── utils/                      # 通用工具函数
```

## 关键组件

### 1. 策略工厂

`get_strategy()` 函数根据配置自动选择合适的策略：

```python
from lightrft.strategy import get_strategy

# 根据 args.fsdp 自动选择 DeepSpeed 或 FSDP
strategy = get_strategy(args)
```

### 2. 配置管理

`StrategyConfig` 以类型安全的方式集中管理所有配置参数：

```python
from lightrft.strategy import StrategyConfig

# 从 args 对象提取
config = StrategyConfig.from_args(args)

# 类型安全的访问
learning_rate = config.actor_learning_rate  # 类型: float
use_bf16 = config.bf16                      # 类型: bool
```

### 3. 策略基类接口

所有策略都实现这个核心接口：

```python
class StrategyBase(ABC):
    def setup_distributed(self, timeout=None) -> None:
        """初始化分布式环境"""

    def create_optimizer(self, model, **kwargs) -> Optimizer:
        """为模型创建优化器"""

    def prepare(self, *models, is_rlhf=False) -> Any:
        """为分布式训练准备模型"""

    def backward(self, loss, model, optimizer, **kwargs) -> None:
        """执行反向传播"""

    def optimizer_step(self, optimizer, model, scheduler, **kwargs) -> None:
        """执行优化器步骤"""

    def save_ckpt(self, model, save_dir, **kwargs) -> None:
        """保存检查点"""

    def load_ckpt(self, model, load_dir, **kwargs) -> Any:
        """加载检查点"""
```

## 使用示例

### 基础用法

```python
from lightrft.strategy import get_strategy

# 初始化策略
strategy = get_strategy(args)

# 设置分布式环境
strategy.setup_distributed()

# 准备模型
actor = strategy.prepare(actor, is_rlhf=True)

# 创建优化器
optimizer = strategy.create_optimizer(actor)

# 训练循环
for batch in dataloader:
    loss = compute_loss(batch)
    strategy.backward(loss, actor, optimizer)
    strategy.optimizer_step(optimizer, actor, scheduler)

# 保存检查点
strategy.save_ckpt(actor, "checkpoints/step_1000")
```

### 配置驱动用法

```python
from lightrft.strategy.config import StrategyConfig
from lightrft.strategy import get_strategy

# 创建类型化配置
config = StrategyConfig(
    seed=42,
    max_norm=1.0,
    micro_train_batch_size=4,
    train_batch_size=32,
    bf16=True,
    fsdp=False,  # 使用 DeepSpeed
    zero_stage=2
)

# 从配置创建策略
strategy = get_strategy(config)
```

### 使用 FakeStrategy 进行测试

```python
from lightrft.strategy.fake_strategy import FakeStrategy

# 使用假策略进行单元测试
strategy = FakeStrategy()

# 所有操作都可以在无分布式环境下工作
strategy.setup_distributed()
strategy.backward(loss, model, optimizer)
strategy.save_ckpt(model, "test_checkpoints")
```

### 在后端之间切换

```python
# 使用 DeepSpeed
args.fsdp = False
args.zero_stage = 2
strategy = get_strategy(args)

# 或使用 FSDP（只需更改一个标志）
args.fsdp = True
strategy = get_strategy(args)

# 其余代码保持不变！
```

## 支持的功能

### DeepSpeed 策略
- Zero-1、Zero-2、Zero-3 优化阶段
- 混合精度训练（BF16/FP16）
- 梯度累积
- 梯度裁剪
- 检查点保存/加载
- vLLM/SGLang 推理引擎集成

### FSDP 策略
- 全分片数据并行训练
- 混合精度训练（BF16/FP16）
- 梯度累积
- 梯度裁剪
- 检查点保存/加载（DCP 格式）
- CPU 卸载支持

### 通用功能
- 统一的训练接口
- 自动梯度累积
- 内存高效的检查点保存
- 推理引擎睡眠/唤醒管理
- 分布式日志记录和打印

## 最佳实践

### 1. 使用 StrategyConfig 访问参数

**推荐：**
```python
config = StrategyConfig.from_args(args)
seed = config.seed  # 类型: int，支持 IDE 自动补全
```

**避免：**
```python
seed = getattr(args, "seed", 42)  # 类型: Any，无自动补全
```

### 2. 让工厂选择策略

**推荐：**
```python
strategy = get_strategy(args)  # 工厂选择合适的后端
```

**避免：**
```python
if args.fsdp:
    strategy = FSDPV2Strategy(...)
else:
    strategy = DeepspeedStrategy(...)
```

### 3. 使用 FakeStrategy 进行测试

```python
# 在测试文件中
def test_training_loop():
    strategy = FakeStrategy()
    # 在无分布式设置下测试训练代码
    assert strategy.is_rank_0()
```

### 4. 一致的日志记录

使用策略的 `print()` 方法进行仅 rank-0 的日志记录：

```python
strategy.print(f"Training step {step}, loss: {loss.item()}")
# 在分布式训练中仅在 rank 0 上打印
```

## 高级功能

### 推理引擎集成

策略模块支持与 vLLM 和 SGLang 推理引擎集成，以在 RLHF 训练期间实现高效生成：

```python
# 可以让引擎休眠以节省内存
strategy.engines_sleep([vllm_engine, sglang_engine])

# 需要时唤醒它们
strategy.engines_wake_up([vllm_engine, sglang_engine])
```

### 多模型准备

一次性为 RLHF 训练准备多个模型：

```python
actor, critic, reward_model, ref_model = strategy.prepare(
    actor, critic, reward_model, ref_model,
    is_rlhf=True
)
```

### 自定义优化器创建

使用策略特定的优化创建优化器：

```python
optimizer = strategy.create_optimizer(
    model,
    lr=1e-5,
    betas=(0.9, 0.95),
    weight_decay=0.1
)
```

## 扩展框架

添加新策略的步骤：

1. 创建继承自 `StrategyBase` 的新类
2. 实现所有抽象方法
3. 在 `strategy.py` 的工厂中添加您的策略

```python
from lightrft.strategy.strategy_base import StrategyBase

class MyCustomStrategy(StrategyBase):
    def setup_distributed(self, timeout=None):
        # 您的实现
        pass

    def backward(self, loss, model, optimizer, **kwargs):
        # 您的实现
        pass

    # 实现其他必需的方法...
```

## 性能考虑

- **内存**：对于相同的模型，FSDP 通常比 DeepSpeed Zero-3 使用更少的内存
- **速度**：对于较小的模型，DeepSpeed 可能更快；对于非常大的模型，FSDP 扩展性更好
- **检查点**：FSDP 使用 PyTorch DCP 格式；DeepSpeed 使用自己的格式
- **兼容性**：在策略之间切换时需要考虑检查点格式

## 故障排除

### 常见问题

1. **导入错误**：确保已安装 DeepSpeed 或 FSDP 依赖项
2. **检查点不兼容**：使用 `utils/ckpt_scripts/` 中的转换脚本
3. **内存不足**：尝试增加 `zero_stage`（DeepSpeed）或启用 CPU 卸载（FSDP）
4. **梯度爆炸**：检查梯度裁剪的 `max_norm` 设置

## 参考资料

- [DeepSpeed 文档](https://www.deepspeed.ai/)
- [PyTorch FSDP 文档](https://pytorch.org/docs/stable/fsdp.html)
- [LightRFT 策略设计哲学](../../docs/source/best_practice/strategy_design_philosophy.md)

## 总结

LightRFT 策略模块提供了一个生产就绪的分布式训练抽象层，结合了：
- **灵活性**：在训练后端之间轻松切换
- **类型安全**：配置驱动的设计与类型检查
- **可测试性**：无需分布式设置的全面测试
- **可扩展性**：简单地添加新策略

这种设计使您能够专注于训练逻辑，而策略模块会处理分布式训练的复杂性。
