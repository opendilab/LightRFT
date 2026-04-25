# LightRFT 上的 R1-AQA：使用 GRPO 的音频问答

[English](README.md) | **中文**

本示例将 [R1-AQA](https://github.com/xiaomi-research/r1-aqa)（基于 Qwen2-Audio 的音频问答 GRPO 训练）迁移到 [LightRFT](https://github.com/opendilab/LightRFT) 训练框架中。

## 概述

R1-AQA 将 Group Relative Policy Optimization（GRPO）应用到 Qwen2-Audio-7B-Instruct 上，用于音频问答任务。训练使用 AVQA 数据集上的规则奖励（准确率 + 格式奖励）。这个 LightRFT 示例在保留核心训练流程的同时，复用了 LightRFT 的分布式训练基础设施、GRPO 实现和奖励处理系统。

## 文件结构

```
examples/r1_aqa/
├── data_preprocess/
│   ├── avqa.py                        # 将 R1-AQA JSONL 转成 LightRFT parquet
│   └── clean_audio_dataset.py         # 删除音频缺失或无法读取的样本
├── audio_dataset.py                  # 音频数据集与多模态输入封装
├── reward_models_utils.py            # 规则奖励（准确率 + 格式）
├── train_colocate.py                 # GRPO 训练入口
├── eval.py                           # 评测脚本（例如 MMAU 风格测试）
├── run_grpo_r1_aqa_qwen2_audio_7b.sh # 训练启动脚本
├── README.md                         # 英文版
└── README_zh.md                      # 本文件
```

## 快速开始

### 前置依赖

```bash
# 核心依赖（通常已随 LightRFT 安装）
pip install transformers torch deepspeed

# 音频依赖
pip install librosa soundfile

# 可选：用于符号化答案校验的 math_verify
pip install math_verify
```

### 第 1 步：准备 AVQA 数据集

首先获取 R1-AQA 使用的 AVQA 训练数据（JSONL 格式）。原始 AVQA 数据如何转换，可参考 [R1-AQA README](https://github.com/xiaomi-research/r1-aqa)。

JSONL 文件中每一行应是一个 JSON 对象，字段类似：

```json
{
  "id": 183,
  "question_text": "What happened in the video?",
  "multi_choice": ["motorboat", "Yacht consignment", "Sailboat set sail", "Consignment car"],
  "answer": 1,
  "dataset_name": "AVQA",
  "audio_path": "path/to/-HG3Omg_89c_30.wav"
}
```

你也可以直接从 https://huggingface.co/datasets/Joysw909/AVQA 下载数据：

```bash
huggingface-cli download --repo-type dataset --resume-download Joysw909/AVQA --local-dir path/to/AVQA
cd path/to/AVQA
mkdir -p all_audios
# 将各个 VGG 目录下的音频复制到 all_audios
cp VGG10000/* VGG20000/* VGG30000/* VGG40000/* all_audios/ 2>/dev/null || true
```

转换为 LightRFT 使用的格式：

```bash
python examples/r1_aqa/data_preprocess/avqa.py \\
    --input_jsonl path/to/AVQA/train_r1aqa_line.json \\
    --audio_dir path/to/AVQA/all_audios \\
    --local_save_dir ./avqa_lightrft
```

### 第 2 步：清理缺失或损坏的音频样本

在训练前，强烈建议先对 parquet 数据做一次清理。在分布式 GRPO 训练中，如果某些样本的 prompt 仍然包含音频占位符，但其 `audio_path` 指向的文件已缺失，那么某些 rank 可能会走文本分支，而其他 rank 仍然走音频分支，后续常见表现就是 actor forward 阶段卡住。

运行：

```bash
python examples/r1_aqa/data_preprocess/clean_audio_dataset.py \\
    --input_dataset ./avqa_lightrft \\
    --output_dir ./avqa_lightrft_clean
```

如果想做更严格的校验：

```bash
python examples/r1_aqa/data_preprocess/clean_audio_dataset.py \\
    --input_dataset ./avqa_lightrft \\
    --output_dir ./avqa_lightrft_clean \\
    --verify_decode
```

该脚本会输出：

- `train.parquet`：仅保留有效音频样本的清洗后数据
- `train.dropped.jsonl`：被丢弃样本的记录，包含原始数据索引、`audio_path` 和原因

推荐工作流：

1. 先运行一次 `avqa.py` 生成 parquet 数据集。
2. 再对该 parquet 目录运行一次 `clean_audio_dataset.py`。
3. 训练时使用清理后的输出目录，而不是原始 parquet 目录。

### 第 3 步：配置并启动训练

先编辑脚本，填入你的路径：

```bash
# 在 run_grpo_r1_aqa_qwen2_audio_7b.sh 中：
PATH_TO_YOUR_BASE_MODEL="Qwen/Qwen2-Audio-7B-Instruct"
PATH_TO_YOUR_AVQA_DATASET="/path/to/your/avqa_lightrft_clean"
```

启动训练：

```bash
bash examples/r1_aqa/run_grpo_r1_aqa_qwen2_audio_7b.sh
```

### 第 4 步：在 MMAU / MMAR 上评测

```bash
# MMAU (test-mini)
python examples/r1_aqa/eval.py \
    --benchmark mmau \
    --model_path results/lightrft-r1-aqa-grpo-training/<your_run>/ \
    --data_file /path/to/mmau-test-mini.json \
    --audio_dir /path/to/mmau/audio \
    --out_file results/res_mmau_mini.json

# 运行 MMAU 官方评测脚本
python /path/to/mmau/evaluation.py --input results/res_mmau_mini.json


# MMAR
python examples/r1_aqa/eval.py \
    --benchmark mmar \
    --model_path results/lightrft-r1-aqa-grpo-training/<your_run>/ \
    --data_file /path/to/MMAR-meta.jsonl \
    --audio_dir /path/to/mmar/audio \
    --out_file results/res_mmar.jsonl

# 运行 MMAR 官方评测脚本
python /path/to/mmar/code/evaluation.py --input results/res_mmar.jsonl
```

## Batch Size 约束

LightRFT 对 GRPO 有如下 batch size 关系约束：

```
train_batch_size >= rollout_batch_size × n_samples_per_prompt
```

R1-AQA 默认配置（`n_samples=8`）下示例：

| 配置 | rollout_batch_size | n_samples | train_batch_size | 合法？ |
|---|---|---|---|---|
| 默认 | 16 | 8 | 128 | 128 >= 16×8=128 ✓ |
| 最小 | 4 | 4 | 32 | 32 >= 4×4=16 ✓ |
| 单卡 | 4 | 4 | 16 | 16 >= 4×4=16 ✓ |

## 常见问题

### 1. 找不到音频路径

确保预处理脚本中的 `audio_dir` 指向实际存放 `.wav` 文件的目录。JSONL 中的音频路径既可以是相对路径，也可以是绝对路径。

如果训练日志显示各个 rank 的音频样本数不一致，例如某个 rank 打印出的 `<|AUDIO|>` prompt 数量更少，或者成功加载的音频数比其他 rank 少，先清理 parquet 数据，再使用清理后的目录进行训练：

```bash
python examples/r1_aqa/data_preprocess/clean_audio_dataset.py \\
    --input_dataset /path/to/avqa_lightrft \\
    --output_dir /path/to/avqa_lightrft_clean
```

然后把 `PATH_TO_YOUR_AVQA_DATASET` 更新为清理后的输出目录。

### 2. 显存 / OOM

- 减小 `MICRO_TRAIN` 和 `MICRO_ROLLOUT`（例如设为 1）
- 减小 `N_SAMPLES`（例如从 8 改成 4）
- 开启 `--gradient_checkpointing` 和 `--adam_offload`
- 调低 `ENGINE_MEM_UTIL`（例如设为 0.4）

### 3. 推理引擎问题

- Qwen2-Audio 需要支持音频模型的 vLLM 或 SGLang
- 检查你的 vLLM 版本是否支持 `Qwen2AudioForConditionalGeneration`
- 如果使用 SGLang，确认已经具备音频多模态支持

### 4. MMAU 输出字段不匹配

评测脚本输出的是 `model_prediction`，这与 MMAU 期望的字段名一致。如果你使用自定义评测脚本，请确认输出字段名是否匹配。

### 5. Think Mode

R1-AQA 支持可选的 `<think></think>` 模式。启用方式如下：

```bash
# 在数据预处理阶段：
python examples/r1_aqa/data_preprocess/avqa.py --enable_think ...
```

奖励函数会自动兼容两种模式。当 `enable_think=True` 时，格式奖励还会额外检查 `<think>...</think>` 标签。

## 设计说明

### 1. 奖励求和，而不是加权

R1-AQA 直接将准确率奖励和格式奖励相加（最大值为 2.0）；而 LightRFT 中 GSM8K/Geo3K 的实现使用加权组合（`0.9×accuracy + 0.1×format`，最大值为 1.0）。这里保留 R1-AQA 的求和方式，以确保奖励信号与原实现一致。GRPO 的归一化过程会处理这部分量纲差异。

### 2. 原生音频 rollout 路径

音频 RL 现在在 LightRFT 核心代码里走专门的 rollout 路径：

- 原始音频负载保留在生成侧，并以 `audio_data` 的形式传给 SGLang
- 处理后的 mel 特征会显式保存在 `audio_values` 中
- Qwen2-Audio 的特征掩码会显式保存在 `feature_attention_mask` 中

### 3. ActorAL（音频语言 Actor）

Qwen2-Audio 使用的是 `Qwen2AudioForConditionalGeneration`（而不是 `AutoModelForVision2Seq`），其 forward 也需要 `audio_values`，而不是 `pixel_values` + `image_grid_thw`。因此这里使用 `lightrft.models.actor_al` 中的 `ActorAL`，它原生支持 Qwen2-Audio 所需的参数接口。

### 4. Chat Template

R1-AQA 会把音频 URL 以 `{"type": "audio", "audio_url": path}` 的形式嵌入到 chat message 的 content 中。这里保留这一格式，并使用 Qwen2-Audio processor 的 `apply_chat_template` 将其转换成带有音频占位符的正确 token 格式。
