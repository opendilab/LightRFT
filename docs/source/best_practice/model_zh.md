# LightRFT 模型设计文档

## 概述

`lightrft/models` 模块提供了一个全面的框架，用于在强化学习场景中实现 Actor 模型，特别为语言模型微调和人类反馈集成而设计。本文档概述了 models 包的设计理念、架构和实现细节。

## 设计理念

### 1. 模块化架构
models 包采用了模块化设计方法，实现了关注点分离并提升了代码的复用性：

- **Actor 基类**：为不同类型的 Actor 提供基础功能。
- **奖励模型基类**：为不同类型的奖励模型提供基础功能。
- **工具函数**：在不同模型之间共享的常用操作和辅助函数。
- **模型补丁 (Model Patches)**：针对特定模型架构的专门适配。

### 2. 灵活性与可扩展性
设计优先考虑灵活性，以支持各种模型类型和使用场景：

- 支持纯文本模型和视觉语言 (Vision-Language) 模型。
- 可配置的优化策略（LoRA、量化、Flash Attention）。
- 能够适配不同的模型架构和规模。

### 3. 性能优化
内置优化以实现高效的训练和推理：

- 通过梯度检查点 (Gradient Checkpointing) 实现显存高效的实现。
- 支持使用 DeepSpeed 和 FSDP 进行分布式训练。
- 样本打包 (Sample Packing) 以提高批处理效率。

## 架构组件

### 核心类

#### 1. ActorModality (模态定义)
**用途**：位于 `models/actor_modality.py`，定义 Actor 模型的模态类型并管理不同模态支持的参数。

**核心特性**：
- **分类管理**：通过 `ActorModality` 枚举（如 `LANGUAGE_ONLY`, `VISION_LANGUAGE`, `AUDIO_LANGUAGE`, `OMNI`）定义模型类型。
- **参数映射**：`MODALITY_PARAMETERS` 字典定义了各模态支持的特殊参数（如 `pixel_values` 对应视觉模型，`audio_values` 对应音频模型）。
- **解耦设计**：Trainer 通过 `get_supported_parameters` 接口动态获取模型所需的参数，实现了训练逻辑与具体模型输入格式的解耦。

#### 2. ActorLanguage
**用途**：用于纯文本语言模型的通用 Actor。

**核心特性**：
- **纯文本支持**：模态被显式声明为 `ActorModality.LANGUAGE_ONLY`。
- **广泛兼容**：支持 HuggingFace 上的大多数因果语言模型 (Causal Language Model) 架构。
- **性能优化**：自动检测线性模块进行 LoRA 注入，集成 Flash Attention 2.0。

#### 3. ActorVL (Vision-Language)
**用途**：用于视觉语言模型的专门 Actor，处理图像、视频等多模态输入。

**核心特性**：
- **多模态能力**：模态声明为 `ActorModality.VISION_LANGUAGE`。
- **架构适配**：支持 Qwen2-VL 和 Qwen2.5-VL
- **输入处理**：内部处理图像网格 (Image Grid) 和变长视觉序列。

#### 4. ActorAL (Audio-Language)
**用途**：用于音频语言模型（如 Qwen2-Audio）的专用 Actor，模态为 `ActorModality.AUDIO_LANGUAGE`，支持音频采集和处理。

#### 5. 奖励模型 (Reward Models)
**用途**：评估 Response 质量的标量 (SRM) 或生成式 (GRM) 奖励模型。

**核心类**：
- **ScalarRewardModelVL/AL**：标量奖励模型 (SRM)，将多模态输入映射为标量分数。支持 Bradley-Terry 偏好损失。
- **GenerativeRewardModelVL**：生成式奖励模型 (GRM)，输出带推理过程 (CoT) 的文本评估。

### 工具函数

#### 1. LoRA 配置 (`apply_lora_configuration`)
**用途**：集中的 LoRA 设置和配置。

**设计理由**：
- 消除不同 Actor 类型之间的代码重复。
- 在整个框架中提供一致的 LoRA 配置。

#### 2. 对数概率计算 (`log_probs_from_logits`)
**用途**：从模型 Logits 中高效计算对数概率。

**设计特性**：
- 显存优化的实现，采用逐行处理。
- 支持不同的数据类型（float32, float16, bfloat16）。
- 集成 Flash Attention 以提升性能。
- 针对不支持的配置提供自动回退机制。

#### 3. 位置编码管理 (`reset_position_ids`)
**用途**：处理打包序列 (Packed Sequences) 的位置编码 ID。

**设计理由**：
- 对样本打包优化至关重要。
- 在拼接的序列中保持正确的位置编码。
- 支持打包格式下的变长序列。

**设计特性**：
- 模型架构感知的检测。
- 支持配置排除特定模块（如 Vision Towers 等）。
- 支持多种模型类型和架构。

### 模型补丁 (Model Patches)

#### 用途
`monkey_patch` 目录包含模型特定的适配和优化：

- **架构特定优化**：为特定模型架构量身定制的改进。
- **生成方法补丁**：增强的生成能力。
- **性能优化**：模型特定的性能提升。

## 实现细节

### 1. 模型初始化策略

模型支持两种初始化模式：

#### 模式 A：从预训练路径加载
```python
actor = ActorText(
    pretrain_or_model="model_path",
    lora_rank=16,
    use_flash_attention_2=True
)
```

#### 模式 B：从现有模型加载
```python
actor = ActorText(
    pretrain_or_model=existing_model,
    packing_samples=True
)
```

**设计理由**：
- 同时支持从零开始训练和对现有模型进行微调。
- 支持灵活的模型部署场景。
- 保持与现有工作流的向后兼容性。

### 2. 生成与前向传播 (Forward Pass) 设计

#### 生成方法
- **输入处理**：处理各种输入格式和参数。
- **模型生成**：委托给底层模型及其配置参数。
- **后处理**：为 RL 训练创建 Attention Mask 和 Action Mask。

#### 前向方法 (Forward Method)
- **位置 ID 处理**：管理不同序列格式的位置编码。
- **对数概率计算**：高效计算动作概率。
- **打包序列支持**：在单个 Batch 中处理多个序列。

### 3. 显存与性能优化

#### 梯度检查点 (Gradient Checkpointing)
- 可选的节省显存技术。
- 通过 `gradient_checkpointing_enable/disable` 进行配置。
- 在显存使用与计算开销之间取得平衡。

#### 样本打包 (Sample Packing)
- 将多个序列拼接以实现高效的批处理。
- 通过位置 ID 管理维持正确的注意力模式。
- 显著提高变长序列的训练吞吐量。

## 配置与自定义

### 1. LoRA 配置
- **Rank 和 Alpha**：可配置的 LoRA 维度和缩放。
- **目标模块 (Target Modules)**：支持自动检测以及手动覆盖。
- **Dropout**：可配置的正则化强度。

### 2. 注意力机制
- **Flash Attention 2.0**：可选的高性能注意力实现。
- **回退支持**：需要时自动回退到标准注意力。
- **架构兼容性**：跨不同模型架构工作。

### 3. 设备与分布式训练
- **设备映射**：多 GPU 设置下的灵活设备分配。
- **DeepSpeed 集成**：原生支持 DeepSpeed ZeRO 优化。
- **FSDP 兼容性**：支持完全分片数据并行 (Fully Sharded Data Parallel) 训练。

## 错误处理与健壮性

### 1. 优雅降级
- 自动回退不支持的特性。
- 针对配置问题提供清晰的错误提示。
- 检查模型需求的兼容性。

### 2. 验证与断言
- 关键参数的输入验证。
- 针对不兼容配置的断言检查。
- 模型兼容性的运行时验证。

## 结论

LightRFT 的 models 包为基于语言模型的强化学习提供了一个强大、灵活且高效的基础。模块化设计确保了可维护性和可扩展性，而全面的优化支持使得能够跨各种硬件配置和模型架构进行高效的训练和部署。

设计的首要目标包括：
- **简洁性**：易于使用和理解。
- **灵活性**：适配各种使用场景。
- **性能**：针对效率进行了优化。
- **可靠性**：健壮的错误处理和验证。
- **可扩展性**：易于添加新功能和模型类型。

该架构为当前需求奠定了坚实基础，同时也为未来的功能增强和适配提供了清晰的路径。
