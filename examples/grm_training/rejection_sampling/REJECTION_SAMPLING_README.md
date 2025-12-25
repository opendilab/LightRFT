# Rejection Sampling 实现说明

本文档说明如何在 LightRFT 框架下实现 rejection_sampling 训练流程。

## 概述

Rejection Sampling 是 UnifiedReward-Think 训练流程的第二阶段，主要步骤包括：

1. **推理阶段**：使用 cold-start 阶段训练好的模型对大规模数据进行推理
2. **筛选阶段**：筛选出模型预测正确的样本
3. **数据转换**：将筛选出的样本转换为包含 CoT reasoning 的训练数据格式
4. **训练阶段**：使用筛选出的正确样本进行监督学习训练

## 文件说明

- `rejection_sampling_inference.py`: 推理脚本，对数据集进行推理并筛选正确样本
- `convert_to_rejection_sampling_data.py`: 数据转换脚本，将筛选出的样本转换为训练格式
- `run_rejection_sampling.sh`: 完整的运行脚本，整合整个流程

## 使用方法

### 方法一：使用完整脚本（推荐）

直接运行完整的 rejection sampling 流程：

```bash
cd /mnt/shared-storage-user/sunjiaxuan/dec/LightRFT

# 修改脚本中的配置（如需要）
# - MODEL_PATH: 你的 cold-start 模型路径
# - DATA_PATH: 数据集路径
# - DATA_ROOT: 数据集根目录

bash examples/grm_training/rejection_sampling/run_rejection_sampling.sh
```

### 方法二：分步执行

#### 步骤 1: 推理和筛选

```bash
python examples/grm_training/rejection_sampling/rejection_sampling_inference.py \
    --model_path /mnt/shared-storage-user/puyuan/wanzunian/models/lightrlhf-grm-lr1e5-imagegen_cot_reward-qwen2.5vl3B-gs3000 \
    --data_path "hpdv3:/mnt/shared-storage-user/puyuan/wanzunian/datasets/HPDv3/train_subset_5percent.json" \
    --output_path ./results/filtered_samples.json \
    --batch_size 32 \
    --max_new_tokens 2048 \
    --use_cot
```

#### 步骤 2: 数据转换

```bash
python examples/grm_training/rejection_sampling/convert_to_rejection_sampling_data.py \
    --filtered_samples_path ./results/filtered_samples.json \
    --output_path ./results/rejection_sampling_train.json \
    --data_root /mnt/shared-storage-user/puyuan/wanzunian/datasets/HPDv3
```

#### 步骤 3: 训练

```bash
torchrun --nnodes 1 --nproc-per-node 8 \
    examples/grm_training/train_grm_vl.py \
    --pretrain /mnt/shared-storage-user/puyuan/wanzunian/models/lightrlhf-grm-lr1e5-imagegen_cot_reward-qwen2.5vl3B-gs3000 \
    --save_path ./results/rejection_sampling_checkpoint \
    --train_data "imagegen-cot-reward-5k:./results/rejection_sampling_train.json" \
    --train_batch_size 8 \
    --micro_train_batch_size 1 \
    --max_epochs 3 \
    --prompt_max_len 13000 \
    --actor_learning_rate 2.5e-6 \
    --zero_stage 3 \
    --bf16 \
    --gradient_checkpointing \
    --flash_attn
```

## 配置说明

### 推理阶段参数

- `--model_path`: Cold-start 阶段训练好的模型路径
- `--data_path`: 数据集路径，格式为 `"source:path"`，例如 `"hpdv3:/path/to/data.json"`
- `--output_path`: 筛选出的样本保存路径
- `--batch_size`: 推理批次大小（默认 32）
- `--max_new_tokens`: 最大生成 token 数（默认 2048）
- `--use_cot`: 是否使用 CoT 指令生成推理过程

### 训练阶段参数

- `--pretrain`: 预训练模型路径（通常是 cold-start 模型）
- `--train_data`: 训练数据路径，格式为 `"source:path"`，使用 `imagegen-cot-reward-5k` 作为 source
- `--train_batch_size`: 全局训练批次大小
- `--micro_train_batch_size`: 每张 GPU 的微批次大小
- `--max_epochs`: 训练轮数（默认 3）
- `--prompt_max_len`: 最大序列长度（默认 13000，支持长 CoT）
- `--actor_learning_rate`: 学习率（默认 2.5e-6）

## 数据格式

### 输入数据格式（HPDv3）

```json
{
  "path1": "images/image1.jpg",
  "path2": "images/image2.jpg",
  "prompt": "A beautiful landscape",
  "confidence": null,
  "choice_dist": null,
  "model1": "model_name",
  "model2": "model_name"
}
```

### 输出训练数据格式

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Task instruction with {prompt} placeholder..."
    },
    {
      "from": "gpt",
      "value": "<think>\nCoT reasoning here...\n</think>\n<answer>Image 1 is better</answer>"
    }
  ],
  "images": [
    "/path/to/image1.jpg",
    "/path/to/image2.jpg"
  ]
}
```

## 注意事项

1. **模型路径**：确保 cold-start 模型路径正确
2. **数据路径**：确保数据集路径和根目录配置正确
3. **显存要求**：训练时可能需要较大的显存，建议使用梯度检查点和 ZeRO Stage 3
4. **CoT 格式**：生成的 CoT reasoning 应该包含在 `<think>...</think>` 标签中
5. **答案格式**：最终答案应该在 `<answer>...</answer>` 标签中，格式为 "Image 1 is better" 或 "Image 2 is better"

## 输出文件

运行完成后，会在输出目录生成以下文件：

- `filtered_samples.json`: 筛选出的正确样本（原始格式）
- `filtered_samples_stats.txt`: 推理统计信息
- `rejection_sampling_train.json`: 转换后的训练数据
- `checkpoint/`: 训练好的模型检查点
- `logs/`: 各阶段的日志文件

## 故障排查

1. **推理阶段失败**：检查模型路径和数据路径是否正确
2. **数据转换失败**：检查图像路径是否存在，确保 `data_root` 配置正确
3. **训练阶段 OOM**：减小 `micro_train_batch_size` 或启用 `--gradient_checkpointing`
4. **准确率低**：检查模型是否在 cold-start 阶段训练充分

## 参考

- UnifiedReward-Think 论文: https://arxiv.org/pdf/2505.03318
- LightRFT 文档: 查看项目 README 和文档目录


