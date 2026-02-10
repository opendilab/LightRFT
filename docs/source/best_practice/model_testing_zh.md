# 模型测试指南

本指南介绍了如何测试和评估训练好的模型，特别是视觉语言模型 (VLMs)。

## 概述

LightRFT 提供了用于交互式模型测试的工具，支持：

- **交互式对话**：实时对话测试。
- **多模态支持**：支持文本和图像输入。
- **批量测试**：使用 JSON 文件进行自动化测试。
- **性能优化**：支持 Flash Attention 2 和 bfloat16。
- **命令行界面**：便捷的测试命令。

## 快速开始

### 基本文本对话

```bash
python test_chat.py --model_path <checkpoint-path>
```

### 基于图像的测试

```bash
# 启动交互模式
python test_chat.py --model_path <checkpoint-path>
```

在交互模式下：
```
[You] /image <image-path>
✓ Image loaded: <image-path>
[You] What do you see in this image?
[Assistant] ...
```

### 批量测试

```bash
python test_chat.py \
  --model_path <checkpoint-path> \
  --batch <test-file.json> \
  --output <results.json>
```

### 自定义生成参数

```bash
python test_chat.py \
  --model_path <checkpoint-path> \
  --max_tokens 4096 \
  --temperature 0.5 \
  --top_p 0.9
```

## 交互命令

交互模式下的可用命令：

| 命令 | 描述 |
|---------|-------------|
| `/image <path>` | 为下一次查询加载图像 |
| `/clear` | 清除对话历史 |
| `/reset` | 重置已加载的图像 |
| `/help` | 显示帮助信息 |
| `/quit` 或 `/exit` | 退出程序 |

## 批量测试文件格式

### 纯文本测试

```json
[
  {
    "query": "What is 2 + 2?",
    "expected": "4"
  },
  {
    "query": "Explain the Pythagorean theorem."
  }
]
```

### 基于图像的测试

```json
[
  {
    "query": "Describe what you see in this image.",
    "images": ["<image-path-1>"],
    "expected": "Description of the image"
  },
  {
    "query": "Compare these two images.",
    "images": ["<image-path-1>", "<image-path-2>"]
  }
]
```

## 配置参数

| 参数 | 默认值 | 描述 |
|-----------|---------|-------------|
| `--model_path` | - | 模型检查点路径（必填） |
| `--device` | `cuda` | 推理设备 (cuda/cpu) |
| `--max_tokens` | `8192` | 最大生成 Token 数 |
| `--temperature` | `0.7` | 采样温度（0 表示贪婪解码） |
| `--top_p` | `0.95` | Top-p 采样参数 |
| `--system_prompt` | (默认) | 自定义系统提示词 |
| `--batch` | `None` | 批量测试 JSON 文件路径 |
| `--output` | `None` | 批量测试结果输出文件路径 |

## 使用示例

### 示例 1：数学问题求解

```bash
python test_chat.py --model_path <checkpoint-path>
```

交互示例：
```
[You] If a triangle has sides 3, 4, and 5, what is its area?

[Assistant] <think>
This is a right triangle since 3² + 4² = 9 + 16 = 25 = 5².
For a right triangle, the area is (1/2) × base × height.
Using the two perpendicular sides: Area = (1/2) × 3 × 4 = 6
</think>

The area of the triangle is 6 square units.
```

### 示例 2：几何图形识别

```bash
python test_chat.py --model_path <checkpoint-path>
```

交互示例：
```
[You] /image <geometry-image-path>
✓ Image loaded: <geometry-image-path>

[You] Solve the geometry problem shown in this image.

[Assistant] <think>
Looking at the diagram, I can see a triangle ABC with...
[详细的推理过程]
</think>

The answer is [solution].
```

### 示例 3：批量性能测试

创建测试文件 `test_questions.json`:
```json
[
  {
    "query": "Find the area of triangle with base 6 and height 8.",
    "expected": "24"
  },
  {
    "query": "What is the perimeter of a square with side length 5?",
    "expected": "20"
  }
]
```

运行批量测试：
```bash
python test_chat.py \
  --model_path <checkpoint-path> \
  --batch test_questions.json \
  --output test_results.json \
  --temperature 0.0
```

## 性能优化

测试脚本包含：

1. **Flash Attention 2**：加速注意力计算。
2. **BFloat16**：减少显存占用并加快推理。
3. **批量处理**：提高批量测试的吞吐量。
4. **内存管理**：自动 GPU 显存清理。

## 常见问题排查

### 显存不足 (OOM)

如果遇到显存问题：

**1. 减少 max_tokens：**
```bash
python test_chat.py --model_path <checkpoint-path> --max_tokens 4096
```

**2. 使用 CPU 推理（较慢）：**
```bash
python test_chat.py --model_path <checkpoint-path> --device cpu
```

### 图像加载失败

确保图像路径正确且格式受支持（JPG, PNG 等）：
```bash
ls -lh <image-path>
```

### 生成质量问题

调整采样参数：
- **更确定性**：`--temperature 0.0`（贪婪解码）
- **更多样化**：`--temperature 1.0 --top_p 0.9`
- **平衡**：`--temperature 0.7 --top_p 0.95`（默认）

## 依赖要求

所需 Python 包：
```bash
pip install torch transformers pillow flash-attn
```

## 最佳实践

1. **模型加载**：初次运行需要一定的模型加载时间。
2. **图像重置**：每次对话后图像会自动重置。
3. **历时管理**：使用 `/clear` 重置对话历史。
4. **批量独立性**：每个批量测试任务独立运行。

## 更多资源

- [快速开始指南](../quick_start/index_zh.rst)
- [配置参考](../quick_start/configuration_zh.md)
- [常见问题解答 (FAQ)](faq_zh.md)
- [问题排查指南](troubleshooting_zh.md)
