# 配置参数详解

本指南详细介绍了 LightRFT 中提供的所有配置参数。为了方便查阅，参数按类别进行了归类。

## 目录

1. [模型参数](#模型参数)
2. [训练参数](#训练参数)
3. [Batch Size 配置](#batch-size-配置)
4. [算法参数](#算法参数)
5. [分布式训练](#分布式训练)
6. [推理引擎](#推理引擎)
7. [内存优化](#内存优化)
8. [日志与监控](#日志与监控)
9. [检查点管理](#检查点管理)

## 模型参数

### `--pretrain`
- **类型**: `str`
- **是否必填**: 是
- **描述**: 预训练模型和分词器 (tokenizer) 的路径
- **示例**: `/path/to/Qwen2.5-7B-Instruct`

### `--reward_pretrain`
- **类型**: `str`
- **默认值**: 与 `--pretrain` 相同
- **描述**: 奖励模型 (Reward Model) 的路径
- **示例**: `/path/to/reward-model`

### `--remote_rm_url`
- **类型**: `str`
- **默认值**: `None`
- **描述**: 远程奖励模型服务器的 URL
- **示例**: `http://localhost:5000`

### `--max_len`
- **类型**: `int`
- **默认值**: `4096`
- **描述**: 最大序列长度（Prompt + Response）

### `--prompt_max_len`
- **类型**: `int`
- **默认值**: `2048`
- **描述**: 最大 Prompt 长度

## 训练参数

### `--num_episodes`
- **类型**: `int`
- **默认值**: `1`
- **描述**: 训练的总 Episode 数量
- **建议**: 大多数任务建议设置为 10-100

### `--max_epochs`
- **类型**: `int`
- **默认值**: `1`
- **描述**: 每个 Episode 中训练的 Epoch 数量
- **建议**: 1-3

### `--actor_learning_rate`
- **类型**: `float`
- **默认值**: `5e-7`
- **描述**: Actor（策略）模型的学习率
- **建议范围**: `1e-7` 到 `5e-6`

### `--critic_learning_rate`
- **类型**: `float`
- **默认值**: `9e-6`
- **描述**: Critic（价值）模型的学习率
- **建议范围**: `1e-6` 到 `1e-5`

### `--lr_warmup_ratio`
- **类型**: `float`
- **默认值**: `0.03`
- **描述**: Warmup 步数占总步数的比例
- **范围**: `0.0` 到 `0.1`

### `--max_norm`
- **类型**: `float`
- **默认值**: `1.0`
- **描述**: 梯度裁剪的最大范数 (Gradient clipping norm)

### `--l2`
- **类型**: `float`
- **默认值**: `0.0`
- **描述**: L2 正则化系数

### `--adam_betas`
- **类型**: `tuple[float, float]`
- **默认值**: `(0.9, 0.95)`
- **描述**: Adam 优化器的 beta 参数

## Batch Size 配置

### 重要约束
**规则**: `train_batch_size >= rollout_batch_size × n_samples_per_prompt`

### `--train_batch_size` (TBS)
- **类型**: `int`
- **是否必填**: 是
- **描述**: 所有 GPU 上的全局训练 Batch Size
- **示例**: `256`
- **计算公式**: `micro_train_batch_size × num_gpus × gradient_accumulation_steps`

### `--micro_train_batch_size`
- **类型**: `int`
- **默认值**: `1`
- **描述**: 单个 GPU 上的训练 Batch Size
- **常见值**: `1`, `2`, `4`

### `--rollout_batch_size` (RBS)
- **类型**: `int`
- **是否必填**: 是
- **描述**: 所有 GPU 上的全局生成 (Rollout) Batch Size
- **示例**: `64`
- **注意**: 必须能被 GPU 总数整除

### `--micro_rollout_batch_size`
- **类型**: `int`
- **默认值**: `2`
- **描述**: 单个 GPU 上的生成 (Rollout) Batch Size
- **常见值**: `2`, `4`, `8`

### 配置示例

**配置 1: 8 张 GPU，显存受限**
```bash
--train_batch_size 128 \
--micro_train_batch_size 1 \
--rollout_batch_size 64 \
--micro_rollout_batch_size 2
```

**配置 2: 8 张 GPU，追求高吞吐量**
```bash
--train_batch_size 512 \
--micro_train_batch_size 2 \
--rollout_batch_size 256 \
--micro_rollout_batch_size 8
```

## 算法参数

### `--advantage_estimator`
- **类型**: `str`
- **选项**: `group_norm`, `reinforce`, `cpgd`, `gspo`, `gmpo`
- **默认值**: `group_norm`
- **描述**: 优势估计 (Advantage estimation) 的方法
- **建议**:
  - `group_norm`: 通用选择 (GRPO)
  - `reinforce`: 需要低方差时使用
  - `cpgd`: 旨在保持基础模型能力时使用

### `--n_samples_per_prompt`
- **类型**: `int`
- **默认值**: `4`
- **描述**: 每个 Prompt 采样生成的 Response 数量
- **常见值**: `4`, `8`, `16`
- **注意**: 值越高效果通常越好，但速度越慢

### `--kl_estimator`
- **类型**: `str`
- **选项**: `k1`, `k2`, `k3`
- **默认值**: `k3`
- **描述**: KL 散度估计器类型
- **建议**: 大多数情况下使用 `k3`

### `--init_kl_coef`
- **类型**: `float`
- **默认值**: `0.001`
- **描述**: 初始 KL 惩罚系数
- **范围**: `0.0001` 到 `0.01`

### `--kl_target`
- **类型**: `float`
- **默认值**: `0.01`
- **描述**: 目标 KL 散度（用于 CPGD）

### `--clip_range`
- **类型**: `float`
- **默认值**: `0.2`
- **描述**: PPO 的裁剪范围 (Clipping range)
- **范围**: `0.1` 到 `0.3`

### `--clip_range_higher`
- **类型**: `float`
- **默认值**: `0.3`
- **描述**: 上限裁剪范围（用于 Clip Higher 算法）

### `--temperature`
- **类型**: `float`
- **默认值**: `1.0`
- **描述**: 采样温度 (Sampling temperature)
- **范围**: `0.6` 到 `1.2`
- **注意**: 值越低生成结果越确定

### `--top_p`
- **类型**: `float`
- **默认值**: `0.9`
- **描述**: 核采样 (Nucleus sampling) 概率
- **范围**: `0.8` 到 `1.0`

## 分布式训练

### `--zero_stage`
- **类型**: `int`
- **选项**: `1`, `2`, `3`
- **默认值**: `2`
- **描述**: DeepSpeed ZeRO 优化阶段
- **建议**:
  - Stage 1: 优化器状态分片
  - Stage 2: 优化器状态 + 梯度分片（推荐）
  - Stage 3: 优化器状态 + 梯度 + 参数分片（最大程度节省显存）

### `--fsdp`
- **操作**: `store_true`
- **默认值**: `False`
- **描述**: 使用 FSDP 代替 DeepSpeed
- **适用场景**: 原生 PyTorch 工作流，追求极致的显存效率

### `--fsdp_cpu_offload`
- **操作**: `store_true`
- **默认值**: `False`
- **描述**: 将 FSDP 优化器状态卸载 (Offload) 到 CPU
- **注意**: 减少 GPU 显存占用，但会降低速度

### `--bf16`
- **操作**: `store_true`
- **默认值**: 通常默认启用
- **描述**: 使用 bfloat16 混合精度

### `--gradient_checkpointing`
- **操作**: `store_true`
- **默认值**: `False`
- **描述**: 启用梯度检查点 (Gradient checkpointing)
- **注意**: 以增加计算量为代价换取显存节省

### `--sp_size`
- **类型**: `int`
- **默认值**: `1`
- **描述**: 序列并行 (Sequence parallelism) 大小
- **建议**: 处理极长序列时建议设置为 `1`, `2`, `4`

## 推理引擎

### `--engine_type`
- **类型**: `str`
- **选项**: `vllm`, `sglang`
- **默认值**: `vllm`
- **描述**: 推理引擎类型

### `--engine_tp_size`
- **类型**: `int`
- **默认值**: `1`
- **描述**: 推理引擎的张量并行 (Tensor Parallelism) 大小
- **建议**:
  - 7B 模型: `1` 或 `2`
  - 13B-34B 模型: `2` 或 `4`
  - 70B+ 模型: `4` 或 `8`
- **约束**: `GPU 总数 (world_size) % engine_tp_size == 0`

### `--engine_mem_util`
- **类型**: `float`
- **默认值**: `0.5`
- **范围**: `0.3` 到 `0.9`
- **描述**: 用于 KV 缓存的 GPU 显存占比
- **建议**:
  - 显存充裕: `0.8` - `0.9`
  - 显存适中: `0.5` - `0.7`
  - 显存紧张: `0.3` - `0.5`

### `--enable_engine_sleep`
- **操作**: `store_true`
- **默认值**: `True`
- **描述**: 启用推理引擎休眠模式
- **注意**: 在不使用引擎阶段释放显存

### `--disable_engine_sleep`
- **操作**: `store_false`
- **目标名为**: `enable_engine_sleep`
- **描述**: 禁用推理引擎休眠模式

### `--rm_use_engine`
- **操作**: `store_true`
- **默认值**: `False`
- **描述**: 奖励模型 (Reward Model) 使用推理引擎进行计算
- **适用场景**: 需要高吞吐量的奖励计算

## 内存优化

### `--adam_offload`
- **操作**: `store_true`
- **默认值**: `False`
- **描述**: 将 Adam 优化器状态卸载 (Offload) 到 CPU

### `--use_mp_opt`
- **操作**: `store_true`
- **默认值**: `False`
- **描述**: 使用混合精度优化器 (FSDP)

### `--packing_samples`
- **操作**: `store_true`
- **默认值**: `False`
- **描述**: 将多个样本打包 (Pack) 到同一序列中
- **适用场景**: 序列长度差异较大时，可提高 GPU 利用率

### `--fused_linear_logprob`
- **操作**: `store_true`
- **默认值**: `False`
- **描述**: 融合线性层和 logprob 计算
- **注意**: 对于大词表模型可节省显存

### `--chunk_size`
- **类型**: `int`
- **默认值**: `4096`
- **描述**: 融合操作的分块大小 (Chunk size)

## 日志与监控

### `--log_dir`
- **类型**: `str`
- **默认值**: `None`
- **描述**: 日志及可视化结果的保存目录

### `--plot_every`
- **类型**: `int`
- **默认值**: `10`
- **描述**: 每隔 N 步绘制一次生成长度分布图

### `--use_tensorboard`
- **类型**: `str`
- **默认值**: `None`
- **描述**: TensorBoard 日志目录

### `--use_wandb`
- **类型**: `str`
- **默认值**: `None`
- **描述**: Weights & Biases 项目名称

### `--wandb_org`
- **类型**: `str`
- **默认值**: `None`
- **描述**: W&B 组织名称

### `--wandb_run_name`
- **类型**: `str`
- **默认值**: 自动生成
- **描述**: W&B Run 名称

## 检查点管理

### `--save_path`
- **类型**: `str`
- **是否必填**: 是
- **描述**: 检查点 (Checkpoint) 保存目录

### `--ckpt_path`
- **类型**: `str`
- **默认值**: `None`
- **描述**: 加载检查点的路径

### `--load_checkpoint`
- **操作**: `store_true`
- **默认值**: `False`
- **描述**: 启用检查点加载

### `--save_interval`
- **类型**: `int`
- **默认值**: `1`
- **描述**: 每隔 N 个 Episode 保存一次检查点

### `--max_ckpt_num`
- **类型**: `int`
- **默认值**: `3`
- **描述**: 保留检查点的最大数量

### `--max_ckpt_mem`
- **类型**: `int`
- **默认值**: `1000`
- **描述**: 最大检查点存储空间（GB）

## 奖励处理

### `--reward_running_norm`
- **操作**: `store_true`
- **默认值**: `False`
- **描述**: 对奖励应用实时归一化 (Running normalization)

### `--reward_running_norm_minus_mean`
- **操作**: `store_true`
- **默认值**: `False`
- **描述**: 奖励归一化时减去均值

### `--advantages_norm`
- **操作**: `store_true`
- **默认值**: `False`
- **描述**: 对优势 (Advantages) 进行归一化

### `--advantage_clip`
- **类型**: `float`
- **默认值**: `0.0`
- **描述**: 对优势进行裁剪 (0 表示不裁剪)

### `--reward_clip`
- **类型**: `float`
- **默认值**: `0.0`
- **描述**: 对奖励进行裁剪 (0 表示不裁剪)

## 多模态 (VLM) 参数

### `--mixed_mm_data`
- **操作**: `store_true`
- **默认值**: `False`
- **描述**: 处理混合的多模态和纯文本数据

### `--processor`
- **类型**: `str`
- **默认值**: 自动检测
- **描述**: 多模态处理器 (Processor) 类型

## 完整配置示例

### 数学推理 (GSM8K, MATH)
```bash
python train.py \
    # 模型
    --pretrain /path/to/Qwen2.5-7B-Instruct \
    --reward_pretrain /path/to/reward-model \
    --max_len 4096 \
    --prompt_max_len 2048 \
    \
    # 训练
    --num_episodes 20 \
    --max_epochs 1 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    \
    # Batch Size
    --train_batch_size 128 \
    --micro_train_batch_size 1 \
    --rollout_batch_size 64 \
    --micro_rollout_batch_size 2 \
    \
    # 算法
    --advantage_estimator group_norm \
    --n_samples_per_prompt 8 \
    --kl_estimator k3 \
    --init_kl_coef 0.001 \
    --temperature 0.6 \
    \
    # 分布式
    --zero_stage 2 \
    --bf16 \
    --gradient_checkpointing \
    \
    # 引擎
    --engine_type vllm \
    --engine_tp_size 1 \
    --engine_mem_util 0.85 \
    --enable_engine_sleep \
    \
    # 内存
    --adam_offload \
    --fused_linear_logprob \
    \
    # 日志
    --use_tensorboard ./tb_logs \
    --plot_every 10 \
    \
    # 检查点
    --save_path ./checkpoints \
    --save_interval 1 \
    --max_ckpt_num 3 \
    \
    # 奖励
    --reward_running_norm \
    --advantages_norm
```

### 多模态 VLM 训练
```bash
python train_vl.py \
    # 模型
    --pretrain /path/to/Qwen2-VL-7B-Instruct \
    --max_len 4096 \
    \
    # 训练
    --num_episodes 10 \
    --actor_learning_rate 1e-6 \
    \
    # Batch Size
    --train_batch_size 64 \
    --micro_train_batch_size 1 \
    --rollout_batch_size 32 \
    --micro_rollout_batch_size 1 \
    \
    # 算法
    --advantage_estimator group_norm \
    --n_samples_per_prompt 4 \
    \
    # 分布式
    --fsdp \
    --fsdp_cpu_offload \
    --gradient_checkpointing \
    \
    # 引擎
    --engine_tp_size 4 \
    --engine_mem_util 0.6 \
    \
    # VLM 特定
    --mixed_mm_data \
    --packing_samples
```

## 参数依赖校验

LightRFT 会自动验证参数的合法性。常见的校验规则包括：

1. **Batch Size**: `train_batch_size >= rollout_batch_size × n_samples_per_prompt`
2. **整除性**: Batch Size 必须能被 GPU 数量整除
3. **显存**: 引擎的 TP size 必须能整除 world size
4. **学习率**: Actor LR 通常小于 Critic LR
5. **KL Target**: 为了训练稳定，建议设置较小的值 (0.001-0.01)

## 环境变量

用于优化性能的有用环境变量：

```bash
# NCCL 优化
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# CUDA 显卡设置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 调试
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
```

## 另请参阅

- [算法指南](algorithms_zh.md) - 详细的算法描述
- [策略使用指南](../best_practice/strategy_zh.md) - 分布式训练策略
- [安装指南](../installation/index_zh.rst) - 设置说明
