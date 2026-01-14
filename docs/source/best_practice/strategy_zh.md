# Strategy 使用指南

## 概述

LightRFT 的 strategy 模块是分布式训练能力的核心，为高效的强化学习微调提供了额外优化。Strategy 提供统一的接口来管理：

- **分布式训练后端**：DeepSpeed ZeRO 和 FSDP（完全分片数据并行）
- **推理引擎集成**：vLLM 和 SGLang 用于高吞吐量生成
- **内存优化**：优化器卸载、梯度累积和引擎睡眠模式
- **序列并行**：跨多个 GPU 高效处理长序列

## 核心 API 扩展

LightRFT 为 strategy 接口添加了以下关键方法：

| 方法 | 功能 |
|------|------|
| `setup_inference_engine()` | 初始化 vLLM 或 SGLang 推理引擎 |
| `update_engine_weights()` | 将 actor 模型权重同步到推理引擎 |
| `gather_and_generate()` | 分布式生成，自动收集 prompts |
| `maybe_load_optimizer()` | 从 CPU 加载优化器状态（仅 FSDP）|
| `maybe_offload_optimizer()` | 将优化器状态卸载到 CPU（仅 FSDP）|
| `wakeup_inference_engine()` | 从睡眠模式唤醒推理引擎 |
| `maybe_sleep_inference_engine()` | 将推理引擎置于睡眠状态以节省内存 |

## 创建 Strategy

### 基础设置

使用工厂函数 `get_strategy()` 创建 strategy 实例：

```python
from lightrft.strategy import get_strategy
from lightrft.utils import add_arguments

def train(args):
    # 创建 strategy（根据 args 自动选择 DeepSpeed 或 FSDP）
    strategy = get_strategy(args)

    # 设置用于生成的推理引擎
    strategy.setup_inference_engine(args, engine_type='vllm')

    # 如需要可以访问引擎
    vllm_engine = strategy.inference_engine

    # 创建训练器
    trainer = SPMDPPOTrainer(
        strategy=strategy,
        actor=actor,
        critic=critic,
        reward_model=reward_model,
        initial_model=initial_model,
        ema_model=ema_model,
        actor_optim=actor_optim,
        critic_optim=critic_optim,
        actor_scheduler=actor_scheduler,
        critic_scheduler=critic_scheduler,
        ...
    )
```

### Strategy 选择

Strategy 类型由配置参数自动决定：

- **FSDP**：设置 `--fsdp` 标志
- **DeepSpeed**：不设置 `--fsdp` 时的默认选项（可通过 `--zero_stage` 配置）

## 在 Trainer 中使用 Strategy

### 标准训练操作

Strategy 提供标准的分布式训练操作：

```python
# 反向传播
strategy.backward(loss, model, optimizer)

# 优化器步进（带梯度裁剪）
strategy.optimizer_step(optimizer, model, scheduler, name="actor")

# 分布式通信
averaged_value = strategy.all_reduce(local_value, op="mean")
gathered_values = strategy.all_gather(local_value)
```

### 内存优化训练

对于基于 FSDP 的训练，使用优化器卸载来减少 GPU 内存占用：

```python
def ppo_train(self, global_steps=0):
    torch.cuda.synchronize()
    train_begin = time.time()

    # 从 CPU 加载优化器状态到 GPU（仅 FSDP）
    self.strategy.maybe_load_optimizer(self.actor_optim)

    # 执行训练
    train_ret = super().ppo_train(global_steps)

    # 从 GPU 卸载优化器状态到 CPU（仅 FSDP）
    self.strategy.maybe_offload_optimizer(self.actor_optim)

    torch.cuda.synchronize()
    self.strategy.print(f"PPO Train TIMECOST {time.time() - train_begin}")

    # 将 actor 权重同步到推理引擎
    self.strategy.update_engine_weights(self.actor)

    return train_ret
```

### 引擎权重同步

训练更新后，将模型权重同步到推理引擎：

```python
# 用最新的 actor 权重更新推理引擎
strategy.update_engine_weights(actor)
```

这确保推理引擎使用最新的模型参数进行生成。

## 在 Experience Maker 中使用 Strategy

### 文本生成（LLM）

使用 `gather_and_generate()` 进行分布式文本生成：

```python
# 对 prompts 进行分词（为提高效率不使用 padding）
all_prompt_token_ids = self.tokenize_fn(
    all_prompts,
    self.prompt_max_len,
    padding=False
)["input_ids"]

# 自动分布式生成响应
all_outputs = self.strategy.gather_and_generate(
    sampling_params=sampling_params,
    all_prompt_token_ids=all_prompt_token_ids,
    sleep_engine=True  # 生成后自动使引擎进入睡眠状态
)

if dist.get_rank(self.vllm_mp_group) == 0:
    self.strategy.print(f"生成了 {len(all_outputs)} 个输出")
```

### 多模态生成（VLM）

对于带有图像的视觉-语言模型：

```python
# 使用多模态输入生成
all_outputs = self.strategy.gather_and_generate(
    sampling_params=sampling_params,
    all_prompts=all_prompts,        # 文本 prompts
    all_images=all_images,          # 图像数据
    images_num=images_num,          # 每个 prompt 的图像数量
    sleep_engine=True
)
```

### `gather_and_generate()` 工作原理

该方法执行以下操作：

1. **收集（Gather）**：将张量并行组内所有 rank 的 prompts 收集到 rank 0
   - 示例：当 `world_size=8` 且 `engine_tp_size=4` 时，ranks [0,1,2,3] 收集到 rank 0，ranks [4,5,6,7] 收集到 rank 4

2. **生成（Generate）**：使用 vLLM/SGLang 引擎对收集的 prompts 执行推理

3. **分发（Distribute）**：将生成的输出按相同顺序分散回原始 ranks

4. **睡眠管理（Sleep Management）**：根据 `sleep_engine` 参数自动处理引擎的睡眠/唤醒周期

**注意**：使用此接口时，用户无需手动管理引擎睡眠状态。

## 必需参数

将 LightRFT 特定的参数添加到你的参数解析器：

```python
from lightrft.utils import add_arguments
import argparse

# 创建解析器
parser = argparse.ArgumentParser()

# 添加 LightRFT 参数
add_arguments(parser)

# 解析参数
args = parser.parse_args()
```

### 关键参数

**推理引擎配置：**
```bash
--engine_tp_size 4              # 推理引擎的张量并行大小
--engine_mem_util 0.85          # KV 缓存的 GPU 内存利用率（0.0-1.0）
--engine_type vllm              # 引擎类型：'vllm' 或 'sglang'
--enable_engine_sleep           # 启用引擎睡眠模式（默认：True）
--disable_engine_sleep          # 禁用引擎睡眠模式
```

**分布式训练：**
```bash
--fsdp                          # 使用 FSDP 而非 DeepSpeed
--zero_stage 2                  # DeepSpeed ZeRO 阶段（1、2 或 3）
--fsdp_cpu_offload              # 将 FSDP 优化器状态卸载到 CPU
--adam_offload                  # 卸载 Adam 优化器状态
--sp_size 2                     # 序列并行大小
```

**训练优化：**
```bash
--packing_samples               # 将多个样本打包到序列中
--use_mp_opt                    # 使用混合精度优化器（FSDP）
--fused_linear_logprob          # 融合线性层和 logprob 计算
--chunk_size 4096               # 融合操作的块大小
```

**监控：**
```bash
--log_dir ./logs                # 日志和可视化的目录
--plot_every 10                 # 每 N 步绘制生成长度分布
```

## Strategy 实现细节

### 可用的 Strategy

LightRFT 提供两种主要的 strategy 实现：

1. **DeepspeedStrategy**（默认）
   - 使用 DeepSpeed ZeRO 进行内存高效训练
   - 可配置 ZeRO 阶段（1、2 或 3）
   - 支持梯度累积和混合精度
   - 最适合：通用 RLHF 训练、成熟的工作流程

2. **FSDPV2Strategy**（设置 `--fsdp` 时）
   - 使用 PyTorch 的完全分片数据并行
   - 支持优化器状态的 CPU 卸载
   - 原生 PyTorch 实现，集成更好
   - 最适合：最大内存效率、PyTorch 原生工作流程

### Strategy 选择逻辑

```python
# 在 get_strategy() 函数中
if args.fsdp:
    strategy = FSDPV2Strategy(...)
else:
    strategy = DeepspeedStrategy(...)
```

## 引擎睡眠/唤醒机制

Strategy 通过引擎睡眠模式提供自动内存管理：

```python
# 引擎生命周期管理
strategy.setup_inference_engine(args, engine_type='vllm')  # 创建并唤醒引擎
strategy.maybe_sleep_inference_engine()                     # 睡眠以节省内存
strategy.wakeup_inference_engine()                          # 唤醒以进行生成
```

**自动管理**：当使用 `gather_and_generate()` 并设置 `sleep_engine=True` 时，睡眠/唤醒周期会自动处理。

## 配置示例

### 高吞吐量设置（8 GPU，DeepSpeed）

```bash
# 使用 DeepSpeed ZeRO-2 和大张量并行
python train.py \
    --zero_stage 2 \
    --engine_tp_size 4 \
    --engine_mem_util 0.9 \
    --enable_engine_sleep \
    --micro_train_batch_size 1 \
    --train_batch_size 128
```

### 内存高效设置（8 GPU，FSDP + CPU 卸载）

```bash
# 使用 FSDP + CPU 卸载以实现最大内存效率
python train.py \
    --fsdp \
    --fsdp_cpu_offload \
    --use_mp_opt \
    --engine_tp_size 2 \
    --engine_mem_util 0.85 \
    --enable_engine_sleep \
    --micro_train_batch_size 1 \
    --train_batch_size 64
```

### 视觉-语言模型设置

```bash
# 使用多模态数据训练 VLM
python train_vl.py \
    --fsdp \
    --engine_tp_size 4 \
    --mixed_mm_data \
    --packing_samples \
    --enable_engine_sleep \
    --plot_every 20
```

## 最佳实践

### 1. 张量并行配置

- 根据模型大小和 GPU 数量设置 `engine_tp_size`
- 7B 模型：`engine_tp_size=1` 或 `2`
- 13B-70B 模型：`engine_tp_size=4` 或 `8`
- 确保 `world_size % engine_tp_size == 0`

### 2. 内存管理

- 对于内存受限的设置启用引擎睡眠模式：`--enable_engine_sleep`
- 根据可用内存调整 `engine_mem_util`（0.5-0.9）
- 使用 FSDP + CPU 卸载以最大化节省内存：`--fsdp --fsdp_cpu_offload`

### 3. 性能优化

- 对于不同长度的序列使用 `--packing_samples`
- 对于大词汇表模型启用 `--fused_linear_logprob`
- 设置适当的 `micro_train_batch_size` 以充分利用 GPU

### 4. 调试和监控

- 使用 `--plot_every` 和 `--log_dir` 跟踪生成长度分布
- 使用 `strategy.report_memory(prefix="checkpoint_name")` 监控内存
- 使用 `strategy.inference_engine_status` 检查引擎状态

## 高级功能

### 序列并行

为超长序列启用序列并行：

```bash
# 在参数中
--sp_size 4  # 将序列分割到 4 个 GPU
```

Strategy 会自动创建序列并行组并处理通信。

### 自定义奖励模型

支持多个奖励模型或远程奖励 API：

```python
# 多个奖励模型
reward_models = [reward_model_1, reward_model_2, reward_model_3]
strategy = get_strategy(args)

# 模型自动分片到各个 GPU
prepared_rms = [strategy.prepare_model(rm, shard_size=8) for rm in reward_models]
```

### 混合精度训练

控制混合精度行为：

```bash
# 启用 BF16 训练
--bf16

# 使用混合精度优化器（FSDP）
--use_mp_opt
```

## 故障排除

### 常见问题

**问题**：生成时内存不足
- **解决方案**：降低 `engine_mem_util` 或增加 `engine_tp_size`

**问题**：引擎未使用新权重更新
- **解决方案**：确保训练后调用 `update_engine_weights()`

**问题**：生成速度慢
- **解决方案**：增加 `micro_rollout_batch_size` 或减少 `engine_tp_size`

**问题**：FSDP 优化器卸载错误
- **解决方案**：验证是否使用 FSDP strategy（`--fsdp`）并成对调用 offload/load

## API 参考

详细的 API 文档请参阅：
- `lightrft.strategy.strategy_base.StrategyBase` - 基础 strategy 类
- `lightrft.strategy.get_strategy()` - Strategy 工厂函数
- `lightrft.utils.add_arguments()` - 参数配置
