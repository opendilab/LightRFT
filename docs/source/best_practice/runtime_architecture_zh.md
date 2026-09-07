# LightRFT 运行时架构与资源复用原理

本文从实现角度解释 LightRFT 的运行时设计，重点回答以下问题：

- “基于原生 `torchrun`/SPMD 运行时”具体指什么；
- `Strategy` 抽象负责什么，是否控制验证任务；
- Actor、rollout policy、old policy 与 Reference Model 之间是什么关系；
- Colocate、引擎 `sleep`/`wake_up`、模型 `reload`/`offload` 如何配合；
- 训练模型的参数如何同步到 vLLM 或 SGLang。

本文描述以当前仓库源码为准。推理引擎的底层内存行为还可能随 vLLM、SGLang 版本变化，部署时应同时核对实际安装版本的文档。

## 1. 先建立完整的职责边界

LightRFT 不是由一个 `Strategy` 对象包办训练、采样和奖励计算，而是由多个层次协同完成一个 RL 迭代：

```text
训练脚本（组装模型、数据与配置）
        │
        ▼
PPO Trainer（训练循环、评估、日志与检查点）
        │
        ├── FastExperienceMaker（采样并构造 Experience）
        │       ├── rollout engine：生成响应
        │       ├── Actor：计算行为策略 log-probability
        │       ├── Reference Model：计算参考 log-probability
        │       ├── Critic：估计 value（需要时）
        │       └── RewardComputationEngine：计算并聚合奖励
        │
        └── Strategy（分布式执行、模型封装、通信、显存生命周期、权重同步）
                ├── DeepSpeed 或 FSDP v2
                └── vLLM 或 SGLang 适配
```

| 组件 | 主要职责 | 不负责的内容 |
| --- | --- | --- |
| `Strategy` | 初始化分布式环境；封装 DeepSpeed/FSDP v2；创建进程组；提供反向传播、优化、保存、推理引擎管理及权重广播等统一接口 | 不定义任务答案是否正确，也不决定具体奖励规则 |
| `PPOTrainer` / `SPMDPPOTrainerVL` | 组织 rollout、经验回放、策略更新、评估、日志和检查点 | 不实现底层 ZeRO/FSDP 通信 |
| `FastExperienceMaker` | 将生成结果依次送入 Actor、Reference、Critic 和奖励组件，构造可训练的 `Experience` | 不直接实现分布式模型分片 |
| `RewardComputationEngine` | 调用本地奖励模型、远程奖励服务或自定义奖励函数，并聚合奖励 | 不负责梯度更新和推理引擎参数同步 |
| vLLM/SGLang engine | 使用 Actor 参数的推理表示执行高吞吐生成 | 不是可直接反向传播的训练 Actor |

因此，更准确的项目表述是：

> LightRFT 以统一的 Strategy 接口衔接 DeepSpeed/FSDP v2 训练后端与 vLLM/SGLang rollout 后端，并由 Trainer、ExperienceMaker 和 RewardComputationEngine 分别组织训练循环、经验构造与奖励计算。

不宜表述为“Strategy 同时控制训练、推理和验证策略”。Strategy 为这些阶段提供共同的分布式运行时和底层操作，但验证语义属于 Trainer 与奖励计算层。

## 2. 原生 `torchrun`/SPMD 运行时

### 2.1 `torchrun` 做了什么

典型任务由如下形式启动：

```bash
torchrun --nproc_per_node 8 examples/gsm8k_geo3k/train_colocate.py ...
```

`torchrun` 在每个节点创建多个 Python 进程，并为进程设置 `RANK`、`LOCAL_RANK`、`WORLD_SIZE`、`MASTER_ADDR` 和 `MASTER_PORT` 等环境变量。通常一个进程绑定一张 GPU。每个进程执行同一份训练脚本，但依据自己的 rank 处理不同的数据分片；在 FSDP 或 ZeRO-3 等配置下，还会持有不同的参数分片，并参与相应的集合通信。

LightRFT 的 FSDP v2 路径使用 `torch.distributed.init_process_group` 初始化通信；DeepSpeed 路径使用 `deepspeed.init_distributed`，底层仍建立 PyTorch 分布式进程组。vLLM 采用 `external_launcher`，SGLang 也复用外部 `torchrun` 环境，因此不需要再引入一套独立的 Ray 调度运行时。

这带来两个直接结果：

1. 训练、rollout 和参数同步位于同一组 rank 拓扑中，进程身份与 GPU 映射清晰；
2. 可直接使用 PyTorch 分布式工具、日志和调试方法定位某个 rank 的阻塞或显存问题。

“原生”并不表示系统没有 DeepSpeed、FSDP、vLLM 或 SGLang，而是指进程的创建与全局协调以 `torchrun` 和 `torch.distributed` 为基础。

### 2.2 SPMD 的含义

SPMD（Single Program, Multiple Data）表示所有 rank 执行同一个程序和大致相同的控制流，但各自处理不同的数据或参数分片。以 8 个 rank、`engine_tp_size=2` 为例，LightRFT 会构造：

```text
rollout TP 组： [0, 1] [2, 3] [4, 5] [6, 7]
正交 DP 组：   [0, 2, 4, 6] [1, 3, 5, 7]
```

在每个 rollout TP 组内，`gather_and_generate` 的基本过程是：

1. 各 rank 准备自己的 prompt；
2. 通过 `all_gather_object` 收集组内输入；
3. TP 组协同执行相同一批输入的生成；
4. 将输出按 rank 切片，每个 rank 仅取回自己的部分；
5. 如启用休眠，则在生成结束后使推理引擎进入休眠状态。

SPMD 对控制流一致性要求很高。若部分 rank 跳过一次 `barrier`、权重广播或生成调用，而其他 rank 进入该集合通信，任务便可能等待甚至死锁。因此调试时应先确认所有 rank 是否以相同次序进入关键阶段，而不能只观察 rank 0。

## 3. Strategy 抽象如何起作用

`get_strategy(args)` 根据 `--fsdp` 选择 `FSDPV2Strategy`；未启用时选择 `DeepspeedStrategy`。上层 Trainer 面向共同接口编程，主要包括：

- 分布式初始化、数据采样器和进程组管理；
- 模型与优化器的 `prepare`；
- `backward`、`optimizer_step`、梯度裁剪和学习率调度；
- 模型与检查点保存/加载；
- rollout 引擎创建、生成、休眠和唤醒；
- Actor 到推理引擎的参数同步；
- FSDP 路径中的模型与优化器 CPU/GPU 迁移。

Strategy 的价值是隔离训练后端差异。例如，上层调用同一个 `strategy.backward(...)`，具体实现可以交由 DeepSpeed engine，也可以由 FSDP v2 包装后的模型执行；调用 `update_engine_weights(actor)` 时，内部再根据训练后端选择 ZeRO 参数聚合或 FSDP `DTensor` 还原。

### 3.1 Strategy 是否控制“验证”

需要区分两个常被称为“验证”的概念：

- **训练过程中的 evaluation**：由 `PPOTrainer.evaluate` 等方法触发。它复用 ExperienceMaker 完成生成和奖励计算，但不对这些经验执行训练更新。
- **答案验证或奖励判定**：由自定义奖励函数、本地/远程 Reward Model，以及 `RewardComputationEngine` 的聚合逻辑完成。

Strategy 会为 evaluation 提供生成、通信、模型放置等运行能力，但不定义评价指标，也不决定规则奖励或模型奖励的语义。因此它“支撑验证任务的执行”，而不是“控制验证策略”。

## 4. Actor、rollout policy 与 Reference Model

### 4.1 Actor 的两种运行表示

同一策略在运行时存在两种用途不同的表示：

- **训练 Actor**：Hugging Face 模型经 DeepSpeed 或 FSDP v2 封装后执行前向、反向传播与优化；
- **rollout policy**：vLLM 或 SGLang 中的推理表示，仅用于高吞吐生成。

二者在逻辑上表示同一策略，但不是同一个 Python 模型对象，也不会自动共享参数。每轮策略更新后，LightRFT 必须显式把训练 Actor 的新参数同步到 rollout engine。

### 4.2 Reference Model 是什么

Reference Model 在代码中通常命名为 `initial_model`。启用 KL 约束时，它通常从与 Actor 相同的预训练或 SFT 检查点初始化，随后保持冻结：

```text
相同初始检查点
    ├── Actor：参与优化，参数随训练变化
    └── Reference Model：冻结，作为相对稳定的 KL 锚点
```

对于响应 token，Reference Model 提供参考对数概率 `log π_ref(a|s)`，Actor 提供当前对数概率 `log π_θ(a|s)`。二者用于估计 KL 偏离，并通过奖励惩罚或显式 KL loss 限制策略过快偏离初始模型。当前示例在 `init_kl_coef == 0` 时不创建 Reference Model。

### 4.3 Reference Model 不等于 old policy

PPO 中还存在用于概率比率的 old policy。LightRFT 不必长期保存第三个完整模型：生成 Experience 时记录的 `action_log_probs` 就代表更新前行为策略的对数概率；训练若干 epoch 时再由当前 Actor 计算新概率，形成近似比率

```text
ratio = exp(log π_θ(a|s) - log π_old(a|s))
```

两类基准的作用不同：

| 基准 | 来源 | 生命周期 | 用途 |
| --- | --- | --- | --- |
| old policy | Experience 中保存的 rollout/更新前 Actor 概率 | 一批经验或一个 PPO 更新周期 | PPO clipping 的概率比率 |
| Reference Model | 冻结的初始模型 | 通常贯穿整个训练任务 | KL 正则与策略漂移约束 |

## 5. Colocate 的准确含义

LightRFT 的 Colocate 指训练 Actor、本地 Reference/Critic/Reward Model 与 rollout engine 共享同一组 GPU/rank，并按阶段复用资源。它是一种**逻辑共置与时间复用**，不意味着所有模型始终同时计算，也不保证所有权重在任意时刻均离开 GPU。

相较于为训练和推理分别保留固定 GPU 池，这种方式可以提高单组资源的使用弹性；代价是阶段切换会引入同步、CPU/GPU 迁移、缓存重建和参数广播开销，并可能增加主机内存需求。

### 5.1 三类容易混淆的内存操作

| 操作 | 作用对象 | LightRFT 中的含义 |
| --- | --- | --- |
| engine `sleep` / `wake_up` | vLLM/SGLang rollout engine | 释放或恢复引擎管理的缓存，是否包含权重取决于引擎实现和调用参数 |
| model `offload_model` / `reload_model` | FSDP v2 管理的 PyTorch 模型 | 通过 `.to("cpu")` 与 `.to(current_cuda_device)` 在 CPU/GPU 间迁移参数和 buffer |
| optimizer offload/load | FSDP v2 优化器状态 | 启用 `--adam_offload` 时，在训练阶段边界迁移优化器状态 |

这些操作不是从磁盘重新读取 checkpoint。“reload”在此表示从 CPU 内存迁回 GPU；只有显式的 checkpoint 加载才涉及持久化存储。

### 5.2 vLLM 与 SGLang 的休眠并不等价

LightRFT 对两种引擎统一调用无参数的 `sleep()` 与 `wake_up()`，但底层语义不同：

- **vLLM**：LightRFT 未显式传入 sleep level，因此具体释放哪些内存采用所安装 vLLM 版本的默认行为。不能脱离版本直接断言“必然卸载全部权重”；应依据对应版本的 `LLM.sleep` 文档和实际显存观测确认。
- **SGLang**：LightRFT 的包装器默认 `release_weights=False`，只释放 KV cache 与 CUDA Graph 占用，保留模型权重；`wake_up()` 再恢复这些缓存资源。代码明确提示释放权重的路径需要谨慎使用。

因此，“通过引擎 sleep/wake 复用 GPU”是正确的概括，但“vLLM 与 SGLang 都会在 sleep 时卸载模型权重”并不严谨。

### 5.3 Reference、Critic 与 Reward Model 的放置

当前快速经验构造路径按角色批处理：

1. Actor 为全部样本计算行为策略 log-probability；
2. 若存在 Reference Model，FSDP v2 路径将其迁回 GPU，计算参考 log-probability，再迁回 CPU；
3. 若存在 Critic，则计算 value；当其采用 FSDP `CPUOffloadPolicy` 时，参数迁移由 FSDP 自动管理，不再手工调用 reload/offload；
4. 奖励计算引擎处理本地 PyTorch Reward Model、自定义奖励或远程奖励服务。

对多个本地 PyTorch Reward Model，当前实现会先将这些模型全部 reload 到 GPU，再逐个计算，并在每个模型完成后立即 offload。因此它减少了计算完成后的持续占用，但峰值显存仍可能包含多个已 reload 的 Reward Model；不应描述为“每次仅加载一个 RM”。

手工 `reload_model`/`offload_model` 是当前 `FSDPV2Strategy` 的明确实现。DeepSpeed 路径主要依赖 ZeRO 和相应 offload 配置；在采用本地 PyTorch Reward Model 等组合前，应根据实际脚本验证其模型放置路径，不能将 FSDP 的手工迁移机制直接推广到所有后端。

### 5.4 Replay Buffer 与优化器状态

PPO Trainer 的 replay buffer 默认支持将经验放到 CPU，训练某个 micro-batch 时再迁回计算设备。这可避免序列、log-probability、advantage 等经验张量长期占用显存。

FSDP v2 下启用 `--adam_offload` 后，策略更新前加载优化器状态，更新完成后再卸载到 CPU。DeepSpeed 的优化器 offload 则由生成的 DeepSpeed 配置与 ZeRO 运行时管理，其生命周期不等同于 FSDP 的显式迁移调用。

## 6. 训练参数如何同步到推理引擎

推理引擎中的 rollout policy 不会因为训练 Actor 完成 `optimizer_step` 而自动更新。`Strategy.update_engine_weights(actor)` 执行以下步骤：

1. 若引擎已休眠，先同步各 rank、清理缓存并唤醒引擎；
2. 创建或复用 `BroadcastManager`；
3. 从训练后端还原当前 Actor 参数；
4. 按推理引擎要求映射参数名并发送权重；
5. 同步 CUDA 与分布式 rank，并清理缓存。

训练后端不同，参数还原方式也不同：

- **DeepSpeed**：在 ZeRO-3 下用 `deepspeed.zero.GatheredParameters` 逐参数聚合，再发送给引擎；
- **FSDP v2**：对 `DTensor` 调用 `full_tensor()` 取得完整参数，再发送给引擎；
- **LoRA**：根据后端与引擎组合，在广播时合并 adapter 或现场构造合并后的完整权重；源码对部分 vLLM + LoRA 组合仍有显式 `NotImplementedError`，使用前应核对限制。

权重同步和引擎唤醒是两件事：`wake_up()` 只使引擎恢复可用，不会自动取得 Actor 的最新参数；真正更新参数的是随后的 `BroadcastManager.broadcast_to_engine()`。

## 7. 一个训练迭代的时序

忽略可选分支后，一个典型迭代可概括为：

```text
初始化
  ├─ torchrun 创建 rank，Strategy 初始化进程组
  ├─ DeepSpeed/FSDP 包装训练模型
  ├─ 创建 rollout engine
  ├─ Actor 参数首次同步到 engine
  └─ 如启用，engine 进入 sleep

一次迭代
  1. wake_up engine
  2. vLLM/SGLang 生成响应
  3. sleep engine（若启用）
  4. Actor 计算 old action log-probability
  5. Reference 计算参考 log-probability（若启用）
  6. Critic 计算 value（若算法需要）
  7. 规则、Reward Model 或远程服务计算奖励
  8. 构造 Experience，必要时存入 CPU replay buffer
  9. Actor/Critic 执行一个或多个训练 epoch
 10. 将更新后的 Actor 参数广播到 engine
 11. 进入下一轮 rollout
```

evaluation 使用相同的生成与奖励链路，并仍会构造 Experience 以提取指标，但不会把这些 Experience 用于 replay-buffer 训练，也不会执行参数更新。具体是否创建 Critic、Reference Model，以及 KL 进入奖励还是 loss，由算法和配置共同决定。

## 8. 性能与稳定性的主要权衡

| 配置或机制 | 主要收益 | 主要代价或风险 |
| --- | --- | --- |
| 增大 `engine_tp_size` | 可容纳更大 rollout 模型，并利用 TP 推理 | TP 通信增加，且 `WORLD_SIZE` 必须能被其整除 |
| 启用 engine sleep | 在训练阶段释放部分引擎内存 | 唤醒、缓存重建或权重恢复会增加阶段切换延迟 |
| FSDP 模型 CPU offload | 降低模型的 GPU 常驻内存 | 增加 CPU 内存占用和 PCIe/NVLink 数据迁移 |
| Adam offload | 降低优化器状态的 GPU 占用 | 优化阶段边界增加迁移开销 |
| replay buffer CPU offload | 降低经验张量显存 | 训练 micro-batch 时需要 Host-to-Device 传输 |
| 频繁同步 rollout 权重 | 使采样策略更接近最新 Actor | 参数聚合和广播成本上升 |

Colocate 是否更快取决于模型规模、显存容量、rollout/训练耗时比例、主机内存和互连带宽。代码提供的是资源复用机制，而不是对所有硬件和工作负载都成立的吞吐保证。

## 9. 调试检查表

### 集合通信阻塞

- 检查所有 rank 是否进入相同的生成、广播和 `barrier` 调用；
- 检查 `WORLD_SIZE % engine_tp_size == 0`；
- 分别打印全局 rank、TP 组 rank 和本地 GPU，避免混淆进程组。

### rollout 结果未反映最新训练参数

- 确认优化后实际调用了 `update_engine_weights(actor)`；
- 不要把 `wake_up()` 当成参数同步；
- 检查 LoRA 与所选训练/推理后端组合是否受支持。

### sleep 后显存释放少于预期

- 先区分权重、KV cache、CUDA Graph、优化器状态和 replay buffer；
- SGLang 默认保留 rollout 权重；
- vLLM 的实际释放范围应按安装版本确认；
- `torch.cuda.empty_cache()` 只能释放缓存分配器中未被张量引用的内存，不会释放仍存活的参数。

### Reference Model 导致 OOM

- 确认是否确实需要非零 KL 系数；
- FSDP v2 下检查 `initial_model_shard_size` 与 CPU 内存；
- 确认 Reference 已在计算后 offload，且没有在其他对象中保留额外 GPU 副本。

## 10. 源码导航

理解上述机制时，建议按以下顺序阅读：

- `lightrft/strategy/strategy.py`：Strategy 工厂；
- `lightrft/strategy/strategy_base.py`：分布式初始化、rollout 和引擎生命周期；
- `lightrft/strategy/fsdp/fsdpv2.py`：FSDP v2 模型/优化器迁移；
- `lightrft/strategy/deepspeed/deepspeed.py`：DeepSpeed 后端；
- `lightrft/strategy/utils/distributed_util.py`：TP/DP 子进程组与输入聚合；
- `lightrft/strategy/utils/broadcast_utils.py`：Actor 到推理引擎的权重广播；
- `lightrft/strategy/sglang_utils/sglang_engine.py`：SGLang 内存释放与恢复；
- `lightrft/strategy/vllm_utils/`：vLLM 外部启动与 worker 适配；
- `lightrft/trainer/fast_exp_maker.py`：经验构造、Reference/Critic/Reward 的执行顺序；
- `lightrft/trainer/ppo_trainer.py` 与 `ppo_trainer_vl.py`：训练和 evaluation 循环；
- `examples/gsm8k_geo3k/train_colocate.py`：文本/视觉语言 Colocate 任务的完整组装示例。

相关概念与参数说明还可参阅 [Strategy 使用指南](strategy_zh.md) 和 [Strategy 设计理念](strategy_design_philosophy_zh.md)。
