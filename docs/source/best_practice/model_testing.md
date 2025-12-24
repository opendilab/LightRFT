# Model Testing Guide

Guide for testing and evaluating trained models, particularly Vision-Language Models (VLMs).

## Overview

LightRFT provides tools for interactive model testing with support for:

- **Interactive Chat**: Real-time conversation testing
- **Multimodal Support**: Text and image inputs
- **Batch Testing**: Automated testing with JSON files
- **Performance Optimization**: Flash Attention 2 and bfloat16
- **Command-line Interface**: Convenient testing commands

## Quick Start

### Basic Text Conversation

```bash
python test_chat.py --model_path <checkpoint-path>
```

### Image-based Testing

```bash
# Start interactive mode
python test_chat.py --model_path <checkpoint-path>
```

In interactive mode:
```
[You] /image <image-path>
✓ Image loaded: <image-path>
[You] What do you see in this image?
[Assistant] ...
```

### Batch Testing

```bash
python test_chat.py \
  --model_path <checkpoint-path> \
  --batch <test-file.json> \
  --output <results.json>
```

### Custom Generation Parameters

```bash
python test_chat.py \
  --model_path <checkpoint-path> \
  --max_tokens 4096 \
  --temperature 0.5 \
  --top_p 0.9
```

## Interactive Commands

Available commands in interactive mode:

| Command | Description |
|---------|-------------|
| `/image <path>` | Load image for next query |
| `/clear` | Clear conversation history |
| `/reset` | Reset loaded images |
| `/help` | Show help information |
| `/quit` or `/exit` | Exit program |

## Batch Test File Format

### Text-only Tests

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

### Image-based Tests

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

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | - | Model checkpoint path (required) |
| `--device` | `cuda` | Inference device (cuda/cpu) |
| `--max_tokens` | `8192` | Maximum generation tokens |
| `--temperature` | `0.7` | Sampling temperature (0 for greedy) |
| `--top_p` | `0.95` | Top-p sampling parameter |
| `--system_prompt` | (default) | Custom system prompt |
| `--batch` | `None` | Batch test JSON file path |
| `--output` | `None` | Batch test results output file |

## Usage Examples

### Example 1: Math Problem Solving

```bash
python test_chat.py --model_path <checkpoint-path>
```

Example interaction:
```
[You] If a triangle has sides 3, 4, and 5, what is its area?

[Assistant] <think>
This is a right triangle since 3² + 4² = 9 + 16 = 25 = 5².
For a right triangle, the area is (1/2) × base × height.
Using the two perpendicular sides: Area = (1/2) × 3 × 4 = 6
</think>

The area of the triangle is 6 square units.
```

### Example 2: Geometry Recognition

```bash
python test_chat.py --model_path <checkpoint-path>
```

Example interaction:
```
[You] /image <geometry-image-path>
✓ Image loaded: <geometry-image-path>

[You] Solve the geometry problem shown in this image.

[Assistant] <think>
Looking at the diagram, I can see a triangle ABC with...
[detailed reasoning process]
</think>

The answer is [solution].
```

### Example 3: Batch Performance Testing

Create test file `test_questions.json`:
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

Run batch test:
```bash
python test_chat.py \
  --model_path <checkpoint-path> \
  --batch test_questions.json \
  --output test_results.json \
  --temperature 0.0
```

## Performance Optimizations

The testing script includes:

1. **Flash Attention 2**: Accelerated attention computation
2. **BFloat16**: Reduced memory usage and faster inference
3. **Batch Processing**: Improved throughput for batch tests
4. **Memory Management**: Automatic GPU memory cleanup

## Troubleshooting

### Out of Memory (OOM)

If encountering memory issues:

**1. Reduce max_tokens:**
```bash
python test_chat.py --model_path <checkpoint-path> --max_tokens 4096
```

**2. Use CPU inference (slower):**
```bash
python test_chat.py --model_path <checkpoint-path> --device cpu
```

### Image Loading Failure

Ensure image path is correct and format is supported (JPG, PNG, etc.):
```bash
ls -lh <image-path>
```

### Generation Quality Issues

Adjust sampling parameters:
- **More deterministic**: `--temperature 0.0` (greedy decoding)
- **More diverse**: `--temperature 1.0 --top_p 0.9`
- **Balanced**: `--temperature 0.7 --top_p 0.95` (default)

## Dependencies

Required packages:
```bash
pip install torch transformers pillow flash-attn
```

## Best Practices

1. **Model Loading**: First run requires model loading time
2. **Image Reset**: Images auto-reset after each conversation
3. **History Management**: Use `/clear` to reset conversation history
4. **Batch Independence**: Each batch test runs independently

## Additional Resources

- [Quick Start Guide](../quick_start/index.rst)
- [Configuration Reference](../quick_start/configuration.md)
- [FAQ](faq.md)
- [Troubleshooting Guide](troubleshooting.md)
