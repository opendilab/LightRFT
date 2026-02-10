# 常见问题解答 (FAQ)

关于 LightRFT 的常见问题及其回答。

## 常规问题

### Q: 什么是 LightRFT？

**A**: LightRFT (Light Reinforcement Fine-Tuning) 是一个先进的强化学习框架，专为大语言模型 (LLM) 和视觉语言模型 (VLM) 的微调而设计。它通过支持多种算法和分布式训练策略，提供高效且可扩展的 RLHF 训练能力。

### Q: LightRFT 与 OpenRLHF 的主要区别是什么？

**A**: LightRFT 在 OpenRLHF 的基础上进行了以下扩展：
- 增强了多模态 (VLM) 支持
- 更多的强化学习算法（GRPO, GSPO, GMPO, REINFORCE++, CPGD 等）
- 更好的显存优化（推理引擎休眠、优化器卸载）
- 改进的推理引擎（支持 FP8 的 vLLM, SGLang）
- 为了提高效率，支持奖励模型 (RM) 的同机部署 (Co-location)
- 更灵活的分布式训练策略

### Q: 支持哪些模型？

**A**: LightRFT 支持：
- **LLM**: Qwen, Qwen2.5, LLaMA, Mistral 以及大多数 HuggingFace 模型
- **VLM**: Qwen-VL, Qwen2-VL, LLaVA
- **自定义模型**: 通过 monkey patching 可以轻松添加新模型

### Q: 需要什么样的硬件？

**A**: 最低要求：
- **GPU**: 支持 CUDA 11.8+ 的 NVIDIA GPU
- **显存**: 建议每张 GPU 40GB+ VRAM（通过优化可以实现 24GB 运行）
- **PyTorch**: 2.5.1+
- **Python**: 3.8+

生产环境建议：8× A100/H100 80GB。

## 安装问题

### Q: 如何安装 LightRFT？

**A**: 简单安装步骤：
```bash
git clone https://github.com/opendilab/LightRFT.git
cd LightRFT
pip install -r requirements.txt && pip install -e .
```

### Q: 我需要单独安装 vLLM 吗？

**A**: 不需要，vLLM 已包含在依赖列表中。不过，为了使用最新特性，您可以从源码进行安装。

## 训练问题

### Q: FSDP 和 DeepSpeed 有什么区别？

**A**:
- **FSDP**: PyTorch 原生支持，集成度更高，支持 CPU 卸载 (Offload)。
- **DeepSpeed**: 更成熟，ZeRO-3 优化效果好，通常速度更快。

追求极致显存效率时使用 FSDP，追求速度时使用 DeepSpeed。

### Q: 如何选择 Batch Size？

**A**: 请遵循以下约束：
```
train_batch_size >= rollout_batch_size × n_samples_per_prompt
```

以 8 张 GPU 为例：
- `train_batch_size=256`
- `rollout_batch_size=64`
- `n_samples_per_prompt=8`
- `micro_train_batch_size=1`
- `micro_rollout_batch_size=2`

### Q: 我该使用哪种算法？

**A**: 根据任务类型选择：
- **数学/编程**: GRPO, Dr.GRPO
- **指令遵循**: CPGD, GSPO
- **开放式生成**: FIRE Sampling
- **低显存环境**: GRPO（无需 Critic 网络）
- **研究用途**: GMPO, REINFORCE++

### Q: 每个 Prompt 应该设置多少个采样样本 (Samples per prompt)？

**A**: 常见取值：
- **4-8**: 标准设置，平衡性好。
- **16+**: 质量更高，训练较慢。
- **32+**: 适用于 Best-of-N 场景。

样本数越多，优势估计越准，但训练速度越慢。

### Q: 我可以使用多个奖励模型吗？

**A**: 可以！LightRFT 支持：
- 多个奖励模型并行运行
- 奖励模型同机部署（与训练共用 GPU）
- 远程奖励模型服务器
- 加权奖励组合

### Q: 如何开启多模态 (VLM) 训练？

**A**: 使用 VLM 训练脚本：
```bash
python train_vl.py \
    --pretrain /path/to/Qwen2-VL \
    --mixed_mm_data \
    --packing_samples
```

## 性能问题

### Q: 如何减少显存占用？

**A**: 使用以下技术：
1. 开启梯度检查点：`--gradient_checkpointing`
2. 使用 FSDP 并开启 CPU 卸载：`--fsdp --fsdp_cpu_offload`
3. 降低推理引擎显存占比：`--engine_mem_util 0.4`
4. 使用 ZeRO-3：`--zero_stage 3`
5. 减小 Batch Size
6. 开启引擎休眠：`--enable_engine_sleep`

### Q: 如何加速训练？

**A**:
1. 增大 Batch Size（如果显存允许）
2. 使用 FP8 推理 (vLLM)
3. 开启 Flash Attention：`--flash_attn`
4. 如果可能，减小 `n_samples_per_prompt`
5. 生成阶段使用张量并行：`--engine_tp_size 2`
6. 优化 NCCL：`export TORCH_NCCL_AVOID_RECORD_STREAMS=1`

### Q: 典型的训练速度是多少？

**A**: 在 8× A100 80GB 上（使用 FSDP 及相关优化）：
- **7B 模型**: 约 1000 样本/分钟
- **13B 模型**: 约 500 样本/分钟
- **34B 模型**: 约 200 样本/分钟
- **70B 模型**: 约 50 样本/分钟

### Q: 如何使用多节点训练？

**A**: 使用 SLURM 或 Ray：
```bash
# SLURM 示例
srun -N2 --gres=gpu:8 --ntasks-per-node=8 bash train.sh

# 或使用 torchrun
torchrun --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py
```

## 算法问题

### Q: GRPO 和 PPO 有什么区别？

**A**:
- **GRPO**: 基于组归一化的优势估计，不需要单独的 Critic 网络。
- **PPO**: 使用独立的价值网络 (Critic)。

GRPO 系统更简洁，显存效率更高。

### Q: 什么时候应该使用 CPGD？

**A**: 在以下情况使用 CPGD：
- 对预训练模型进行微调
- 想要保留基础模型的能力
- 需要受控的策略更新
- 防止灾难性遗忘

### Q: 什么是 Clip Higher？

**A**: 一种改进的裁剪方案，针对正负优势分别设置不同的上下界。适用于：
- 有噪声的奖励信号
- 较大的分布偏移
- 训练不稳定的情况

## 调试问题

### Q: 训练因显存溢出 (OOM) 崩溃

**A**: 请参阅 [故障排除指南](troubleshooting_zh.md#显存问题)

### Q: 报错 `num_rollouts_per_episodes = 0`

**A**: 您的 `train_batch_size` 设置过小。请确保：
```
train_batch_size >= rollout_batch_size × n_samples_per_prompt
```

### Q: 模型没有提升 / 奖励没有增加

**A**: 请检查：
1. 学习率是否过高或过低
2. KL 惩罚是否过大
3. 奖励模型的质量
4. 开启奖励归一化：`--reward_running_norm`
5. 尝试不同的优势估计器

### Q: NCCL 超时或卡住

**A**:
```bash
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# 增加超时时间
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
```

### Q: vLLM 引擎初始化失败

**A**:
1. 检查 GPU 显存：尝试 `--engine_mem_util 0.5`
2. 减小 TP 大小：`--engine_tp_size 1`
3. 检查 CUDA 兼容性
4. 更新 vLLM：`pip install -U vllm`

## 评估问题

### Q: 如何在 Benchmark 上进行评估？

**A**: 对于数学评测，请使用 examples 目录下的评测脚本：
```bash
# 请参考 examples/gsm8k_geo3k 目录下的评测脚本
# 也可以参考训练示例脚本中的评测配置部分
```

### Q: 我可以保存生成轨迹 (Trajectories) 吗？

**A**: 可以，使用轨迹保存器：
```python
from lightrft.utils import TrajectorySaver

saver = TrajectorySaver(output_dir="./trajectories")
# 自动保存 Prompts, Responses, Rewards
```

### Q: 如何集成 W&B？

**A**:
```bash
python train.py \
    --use_wandb your-project \
    --wandb_org your-org \
    --wandb_run_name experiment-1
```

## 进阶问题

### Q: 我能实现自定义算法吗？

**A**: 可以！扩展 Trainer 类即可：
```python
from lightrft.trainer import SPMDPPOTrainer

class CustomTrainer(SPMDPPOTrainer):
    def compute_advantages(self, ...):
        # 您的自定义优势计算逻辑
        pass
```

### Q: 如何添加新的模型架构？

**A**: 在 `lightrft/models/monkey_patch/` 中创建 monkey patch：
```python
# your_model.py
def patch_your_model(model):
    # 添加自定义 forward 方法等
    pass

# 在 apply.py 中注册
from .your_model import patch_your_model
```

### Q: 我可以使用自定义奖励函数吗？

**A**: 可以，传入一个可调用对象：
```python
def custom_reward_fn(responses, labels):
    # 您的奖励计算逻辑
    return rewards

trainer = SPMDPPOTrainer(
    ...,
    reward_fn=custom_reward_fn
)
```

### Q: 训练过程中如何保存端点 (Checkpoint)？

**A**: 检查点保存是自动的：
```bash
--save_path ./checkpoints \
--save_interval 1 \
--max_ckpt_num 3
```

恢复训练：
```bash
--load_checkpoint \
--ckpt_path ./checkpoints/episode_5
```

## 贡献问题

### Q: 我如何为 LightRFT 做贡献？

**A**:
1. Fork 仓库
2. 创建特性分支
3. 实现您的修改
4. 添加测试
5. 提交 Pull Request

详情请参阅 [贡献指南](contributing_zh.md)。

### Q: 如何报告 Bug？

**A**: 在 [GitHub Issues](https://github.com/opendilab/LightRFT/issues) 上提交 Issue，包含：
- 环境详情（GPU, CUDA, PyTorch 版本）
- 完整的错误堆栈追踪
- 最小复现脚本
- 预期行为与实际行为的对比

### Q: 哪里可以获得帮助？

**A**:
- Bug 反馈：GitHub Issues
- 提问交流：GitHub Discussions
- 指南文档：Documentation
- 代码参考：Examples 目录

## 附加资源

- [安装指南](../installation/index_zh.rst)
- [快速入门](../quick_start/index_zh.rst)
- [算法指南](../quick_start/algorithms_zh.md)
- [配置参考](../quick_start/configuration_zh.md)
- [故障排除指南](troubleshooting_zh.md)
- [最佳实践](index_zh.rst)
