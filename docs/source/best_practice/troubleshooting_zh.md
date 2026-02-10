# 故障排除指南

本指南旨在帮助您诊断并解决在使用 LightRFT 过程中遇到的常见问题。

## 快速诊断

使用以下流程图快速定位您的问题：

```
问题类型？
├─ 安装与设置 → 参见 [安装问题](#安装问题)
├─ 显存溢出 (OOM) → 参见 [显存问题](#显存问题)
├─ 训练问题 → 参见 [训练问题](#训练问题)
├─ 性能问题 → 参见 [性能问题](#性能问题)
└─ 分布式训练 → 参见 [分布式训练问题](#分布式训练问题)
```

## 安装问题

### 问题：包导入错误

**现象**：
```
ModuleNotFoundError: No module named 'lightrft'
```

**解决方案**：
```bash
# 确保您处于正确的目录
cd /path/to/LightRFT
pip install -r requirements.txt
pip install -e .
```

### 问题：CUDA 版本不匹配

**现象**：
```
RuntimeError: CUDA error: no kernel image is available for execution
```

**解决方案**：
```bash
# 检查 CUDA 版本
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# 使用正确的 CUDA 版本重新安装 PyTorch
pip install torch==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### 问题：vLLM 安装失败

**现象**：
```
ERROR: Failed building wheel for vllm
```

**解决方案**：
```bash
# 安装构建依赖
pip install ninja packaging wheel

# 从源码安装 vLLM
pip install vllm --no-build-isolation

# 或者使用预编译的 wheel 包
pip install vllm==0.5.3.post1
```

## 显存问题

### 问题：显存溢出 (Out of Memory, OOM) 错误

**现象**：
```
RuntimeError: CUDA out of memory
torch.cuda.OutOfMemoryError
```

**解决策略**（建议按顺序尝试）：

**1. 减小 Batch Size**
```bash
# 修改前
--micro_train_batch_size 2
--micro_rollout_batch_size 4

# 修改后
--micro_train_batch_size 1
--micro_rollout_batch_size 2
```

**2. 启用梯度检查点 (Gradient Checkpointing)**
```bash
--gradient_checkpointing
```
以约 20% 的速度损失换取约 50% 的显存节省。

**3. 降低推理引擎显存占用**
```bash
# 修改前
--engine_mem_util 0.9

# 修改后
--engine_mem_util 0.5  # 对于极低显存设备可尝试 0.4
```

**4. 使用 FSDP 并开启 CPU 卸载 (Offload)**
```bash
--fsdp \
--fsdp_cpu_offload \
--use_mp_opt
```

**5. 启用 Adam 状态卸载 (Offload)**
```bash
--adam_offload
```

**6. 使用 ZeRO-3**
```bash
--zero_stage 3
```

**7. 减小模型/序列长度**
```bash
--max_len 2048  # 代替 4096
--prompt_max_len 1024
```

**完整的低显存配置示例**：
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

### 问题：vLLM 引擎引发的 OOM

**现象**：
```
Failed to allocate memory for KV cache
```

**解决方案**：
```bash
# 减少用于 KV 缓存的显存比例
--engine_mem_util 0.3

# 增加张量并行 (Tensor Parallelism) 大小
--engine_tp_size 2  # 或 4

# 启用引擎休眠模式
--enable_engine_sleep

# 使用较小的最大长度
--max_len 2048
```

### 问题：训练过程中的显存泄漏

**现象**：
- 显存占用随时间逐渐增加
- 在训练数个 episode 后最终导致 OOM

**解决方案**：
```bash
# 启用 NCCL 优化
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# 定期清理缓存
# 在训练代码中添加：
torch.cuda.empty_cache()

# 使用引擎休眠模式
--enable_engine_sleep
```

## 训练问题

### 问题：`num_rollouts_per_episodes = 0`

**现象**：
```
AssertionError: num_rollouts_per_episodes should be > 0
```

**根本原因**：
`train_batch_size` < `rollout_batch_size × n_samples_per_prompt`

**解决方案**：
```bash
# 确保 TBS >= RBS × n_samples
# 示例：RBS=64, n_samples=8
--train_batch_size 512  # 必须 >= 64 × 8 = 512
--rollout_batch_size 64
--n_samples_per_prompt 8
```

### 问题：训练不收敛

**现象**：
- 奖励值不增加
- Loss 剧烈震荡
- 模型表现无提升

**诊断与解决方案**：

**1. 检查学习率**
```bash
# 如果过高（Loss 飙升）：
--actor_learning_rate 1e-7  # 调低

# 如果过低（无进展）：
--actor_learning_rate 1e-6  # 调高
```

**2. 启用奖励归一化**
```bash
--reward_running_norm \
--reward_running_norm_minus_mean \
--advantages_norm
```

**3. 检查 KL 惩罚系数**
```bash
# 如果 KL 过大（策略更新过慢）：
--init_kl_coef 0.0001  # 调低

# 如果 KL 过小（训练不稳定）：
--init_kl_coef 0.01  # 调高
```

**4. 尝试不同的算法架构**
```bash
# 从 GRPO 切换到 CPGD
--advantage_estimator cpgd \
--kl_target 0.01
```

**5. 检查奖励模型 (RM) 质量**
```python
# 单独测试奖励模型
python test_reward_model.py --model /path/to/rm
```

### 问题：Loss 或梯度值为 NaN

**现象**：
```
Loss: nan
Gradient: nan
```

**解决方案**：
```bash
# 1. 启用梯度裁剪
--max_norm 1.0

# 2. 调低学习率
--actor_learning_rate 1e-7

# 3. 使用 BF16 代替 FP16
--bf16

# 4. 启用奖励裁剪
--reward_clip 10.0

# 5. 检查是否存在除以零的情况
--advantages_norm  # 使用前自动进行归一化
```

### 问题：训练极其缓慢

**现象**：
- 在 8 张 A100 上每分钟处理样本数 < 100
- 每个 episode 需要耗费数小时

**解决方案**：

**1. 定位性能瓶颈**
```python
# 使用 profiler 进行性能分析
with torch.profiler.profile() as prof:
    trainer.fit()
print(prof.key_averages())
```

**2. 检查数据加载**
```bash
# 增加 worker 数量
--num_workers 8

# 使用更快的 dataloader 配置
--dataloader_pin_memory
```

**3. 优化生成阶段**
```bash
# 使用 FP8 推理
--engine_type vllm  # vLLM 支持 FP8 推理

# 增加生成阶段的张量并行 (TP)
--engine_tp_size 2

# 如果可能，减小最大长度
--max_len 2048
```

**4. 减少日志记录频率**
```bash
# 不要每步都记录日志
--log_interval 100
```

## 分布式训练问题

### 问题：NCCL 超时

**现象**：
```
RuntimeError: NCCL timeout
[E ProcessGroupNCCL.cpp] Caught collective operation timeout
```

**解决方案**：
```bash
# 增加超时时间
export NCCL_TIMEOUT=1800

# 开启 NCCL 调试日志
export NCCL_DEBUG=INFO

# 尝试使用不同的网络接口
export NCCL_SOCKET_IFNAME=eth0

# 如果存在 IB 问题，尝试禁用 InfiniBand
export NCCL_IB_DISABLE=1

# 使用 GLOO 模式进行调试
export NCCL_BACKEND=gloo
```

### 问题：分布式初始化进程卡住

**现象**：
- 脚本停在 "Initializing process group"
- 无任何错误提示

**解决方案**：
```bash
# 1. 检查网络连通性
ping $MASTER_ADDR

# 2. 检查端口可用性
nc -zv $MASTER_ADDR $MASTER_PORT

# 3. 设置正确的环境变量
export MASTER_ADDR=192.168.1.1
export MASTER_PORT=29500
export WORLD_SIZE=8
export RANK=0  # 每张 GPU 分别设置为 0 到 7

# 4. 使用显式的初始化方法
torchrun --nproc_per_node=8 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py

# 5. 开启详细的调试日志
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

### 问题：GPU 利用率不均衡

**现象**：
- 部分 GPU 负载 100%，其余处于空闲
- 即使有多个 GPU，训练依然缓慢

**解决方案**：
```bash
# 1. 检查 batch size 的整除性
# 确保 batch_size % world_size == 0

# 2. 使用张量并行 (Tensor Parallelism)
--engine_tp_size 2  # 将模型拆分到多张 GPU

# 3. 检查流水线气泡
# 确保 train_batch_size 足够大

# 4. 监控 GPU 实时负载
nvidia-smi dmon -i 0,1,2,3,4,5,6,7 -s u

# 5. 针对长序列开启序列并行 (Sequence Parallelism)
--sp_size 2
```

### 问题：多节点训练失败

**现象**：
- 单节点运行正常
- 跨节点运行时失败

**解决方案**：
```bash
# 1. 使用 SLURM 调度器
srun -N2 --gres=gpu:8 --ntasks-per-node=8 bash train.sh

# 2. 使用显式的 torchrun 命令
# 在每个节点上执行：
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    train.py

# 3. 检查防火墙规则
# 确保节点间的通信端口已开放

# 4. 使用共享文件系统
# 确保所有节点都能访问相同的模型/数据路径
```

## 性能问题

### 问题：GPU 利用率过低

**现象**：
- GPU 利用率 < 80%
- 训练速度慢于预期

**解决方案**：

**1. 增大 Batch Size**
```bash
--micro_train_batch_size 2  # 尝试翻倍
--micro_rollout_batch_size 4
```

**2. 减少 CPU 瓶颈**
```bash
--num_workers 8
--prefetch_factor 2
```

**3. 启用 Flash Attention**
```bash
--flash_attn
```

**4. 使用融合算子 (Fused Kernels)**
```bash
--fused_linear_logprob
```

### 问题：生成速度过慢

**现象**：
- 生成 (rollout) 阶段占据了绝大部分训练时间
- 生成速度 < 100 tokens/sec

**解决方案**：

**1. 使用 vLLM 推理引擎**
```bash
--engine_type vllm  # 代替传统的 Transformers 推理
--engine_tp_size 2
```

**2. 优化 KV 缓存**
```bash
--engine_mem_util 0.9  # 在显存允许的情况下
```

**3. 使用 FP8 推理 (如硬件支持)**
```bash
# vLLM 在 H100 等卡上会自动尝试使用 FP8
--engine_type vllm
```

**4. 减少采样样本数**
```bash
--n_samples_per_prompt 4  # 代替默认的 8
```

## 推理引擎问题

### 问题：vLLM 引擎初始化失败

**现象**：
```
Failed to initialize vLLM engine
RuntimeError: Cannot allocate memory
```

**解决方案**：
```bash
# 1. 检查 GPU 剩余显存
nvidia-smi

# 2. 降低显存分配比例
--engine_mem_util 0.5

# 3. 减小 TP 大小
--engine_tp_size 1

# 4. 检查模型兼容性
# 部分模型可能需要特定版本的 vLLM

# 5. 更新 vLLM
pip install -U vllm
```

### 问题：引擎权重未能同步更新

**现象**：
- 策略模型 (Policy Model) 已更新，但生成的文本内容无变化
- 奖励值保持恒定

**解决方案**：
```python
# 确保调用了 update_engine_weights
self.strategy.update_engine_weights(self.actor)

# 在训练循环中检查：
def ppo_train(self):
    ...
    # 训练结束后
    self.strategy.update_engine_weights(self.actor)
```

### 问题：引擎休眠/唤醒失败

**现象**：
- 训练在生成结束后卡住
- 报错 "Engine already sleeping"

**解决方案**：
```bash
# 1. 调试时可尝试禁用引擎休眠
--disable_engine_sleep

# 2. 或使用自动化管理
# gather_and_generate 函数会自动处理休眠与唤醒
all_outputs = self.strategy.gather_and_generate(
    ...,
    sleep_engine=True  # 启用自动管理
)
```

## 检查点 (Checkpoint) 问题

### 问题：无法加载检查点

**现象**：
```
FileNotFoundError: Checkpoint not found
RuntimeError: Error loading state dict
```

**解决方案**：
```bash
# 1. 检查路径是否存在
ls -la /path/to/checkpoint

# 2. 尝试宽松加载
--load_checkpoint \
--ckpt_path /path/to/checkpoint

# 3. 如果不兼容，可尝试跳过优化器状态
# 修改代码仅加载模型权重：
model.load_state_dict(torch.load(ckpt_path))
```

### 问题：保存检查点失败

**现象**：
```
OSError: Disk quota exceeded
RuntimeError: Cannot save checkpoint
```

**解决方案**：
```bash
# 1. 检查磁盘剩余空间
df -h

# 2. 限制保存的检查点数量
--max_ckpt_num 3

# 3. 设置最大存储占用的限制
--max_ckpt_mem 1000  # 单位：GB

# 4. 更换保存路径
--save_path /path/with/space
```

## 调试技巧

### 开启详细调试日志

```bash
# PyTorch 分布式调试
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# NCCL 调试
export NCCL_DEBUG=INFO

# 强制同步执行（精确定位 CUDA 错误）
export CUDA_LAUNCH_BLOCKING=1
```

### 显存画像分析 (Memory Profiling)

```python
import torch

# 记录显存分配历史
torch.cuda.memory._record_memory_history()

# 训练循环
...

# 导出快照文件
torch.cuda.memory._dump_snapshot("memory.pickle")
```

### 性能画像分析 (Performance Profiling)

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

# 查看耗时分布
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### 问题反馈清单

在报告 Bug 时，请提供以下信息：

- [ ] 硬件设备：GPU 型号、数量、显存容量
- [ ] 软件版本：CUDA、PyTorch、vLLM 版本号
- [ ] 运行命令：包含所有启动参数的完整命令
- [ ] 运行日志：完整的错误堆栈追踪 (Traceback)
- [ ] 环境变量：已生效的相关环境变量设置
- [ ] 最小复现脚本
- [ ] 您已尝试过的解决方法

## 获取更多帮助

如果上述方法无法解决您的问题：

1. **查阅 FAQ**：[常见问题](faq_zh.md)
2. **搜索 Issues**：[GitHub Issues](https://github.com/opendilab/LightRFT/issues)
3. **社区交流**：GitHub Discussions
4. **提交 Bug**：提交带有调试信息的新 Issue

## 常见错误信息快速索引

| 错误信息 | 对应章节 | 快速修复 |
|---------------|---------|-----------|
| `CUDA out of memory` | [显存问题](#显存问题) | 减小 batch size，开启梯度检查点 |
| `num_rollouts_per_episodes = 0` | [训练问题](#训练问题) | 增大 `train_batch_size` |
| `NCCL timeout` | [分布式训练问题](#分布式训练问题) | `export NCCL_TIMEOUT=1800` |
| `Failed to initialize vLLM` | [推理引擎问题](#推理引擎问题) | 调低 `engine_mem_util` |
| `NaN loss` | [训练问题](#训练问题) | 调低学习率，开启梯度裁剪 |

## 另请参阅

- [常见问题 (FAQ)](faq_zh.md) - 汇总的常见问题解答
- [配置指南](../user_guide/configuration.md) - 全量参数说明
- [最佳实践](../best_practice/strategy_usage.md) - 框架优化建议
