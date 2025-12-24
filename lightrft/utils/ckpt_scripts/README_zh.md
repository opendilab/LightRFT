# 检查点转换脚本

本目录包含用于在不同格式（DeepSpeed、FSDP、HuggingFace）之间转换检查点以及测试转换后模型的脚本。

## 可用脚本

### 1. DeepSpeed 转 HuggingFace (`ds2hf.py`)

将 DeepSpeed 检查点（Zero-1、Zero-2、Zero-3）转换为 HuggingFace 格式。

**必需参数：**
- `--checkpoint_dir`：DeepSpeed 检查点目录路径
- `--output_dir`：保存转换后的 HuggingFace 模型的路径
- `--hf_base`：基础 HuggingFace 模型路径（训练起点模型）

**可选参数：**
- `--tag`, `-t`：检查点标签，用作唯一标识符（例如 global_step1）
- `--exclude_frozen_parameters`：从转换中排除冻结参数
- `--debug`, `-d`：启用调试模式
- `--model_type`：要初始化的模型类型（可选项：vision2seq、grm、srm_vl、srm_al；默认：vision2seq）

**示例：**
```bash
python lightrft/utils/ckpt_scripts/ds2hf.py \
    --checkpoint_dir results/lightrft-7B-ds/_actor/global_step2/ \
    --output_dir results/lightrft-7B-ds2hf \
    --hf_base checkpoints/qwen25-vl-7b-s10-0321-150
```

### 2. FSDP 转 HuggingFace (`fsdp2hf.py`)

将 PyTorch 分布式检查点（DCP/FSDP）转换为 HuggingFace 格式。自动检测并处理因果语言模型（LLM）和视觉语言模型（VLM）。

**必需参数：**
- `--hf_base`：基础 HuggingFace 模型配置目录路径
- `--checkpoint_dir`：FSDP 检查点目录路径
- `--output_dir`：保存转换后的 HuggingFace 检查点的路径

**示例：**
```bash
python lightrft/utils/ckpt_scripts/fsdp2hf.py \
    --hf_base checkpoints/qwen25-vl-7b-s10-0321-150 \
    --checkpoint_dir results/lightrft-7B-fsdp/_actor/global_step6/ \
    --output_dir results/lightrft-7B-fsdp2hf
```

### 3. HuggingFace 转 FSDP (`hf2fsdp.py`)

将 HuggingFace 检查点（safetensors 格式）转换为 PyTorch 分布式检查点（DCP/FSDP）格式。

**必需参数：**
- `--hf_checkpoint`：HuggingFace 检查点目录路径（包含 .safetensors 文件）
- `--output`：保存转换后的 FSDP 检查点的路径

**示例：**
```bash
python lightrft/utils/ckpt_scripts/hf2fsdp.py \
    --hf_checkpoint checkpoints/qwen25-vl-7b-s10-0321-150 \
    --output results/qwen25-vl-7b-fsdp
```

### 4. 测试模型 (`test_model.py`)

使用示例提示词测试转换后的 HuggingFace 模型，并可选择性地进行基准测试。

**参数：**
- `--model_path`：转换后的 HuggingFace 模型路径（必需）
- `--device`：运行设备（默认：cuda）
- `--test_prompts`：测试提示词（可提供多个提示词）
- `--max_new_tokens`：生成的最大新token数（默认：256）
- `--benchmark`：运行基准测试
- `--benchmark_samples`：基准测试样本数（默认：5）
- `--output_file`：结果输出文件（默认：test_results.json）

**示例：**
```bash
python lightrft/utils/ckpt_scripts/test_model.py \
    --model_path results/lightrft-7B-ds2hf \
    --test_prompts "写一个Python函数来计算斐波那契数列。" \
    --max_new_tokens 512 \
    --benchmark \
    --output_file test_results.json
```

## 工作流程示例

### 完整的 DeepSpeed 训练到推理工作流程

```bash
# 1. 使用 DeepSpeed 训练（示例命令，不属于这些脚本）
# 训练产生：results/lightrft-7B-ds/_actor/global_step2/

# 2. 将 DeepSpeed 检查点转换为 HuggingFace 格式
python lightrft/utils/ckpt_scripts/ds2hf.py \
    --checkpoint_dir results/lightrft-7B-ds/_actor/global_step2/ \
    --output_dir results/lightrft-7B-ds2hf \
    --hf_base checkpoints/qwen25-vl-7b-s10-0321-150

# 3. 测试转换后的模型
python lightrft/utils/ckpt_scripts/test_model.py \
    --model_path results/lightrft-7B-ds2hf \
    --benchmark
```

### 完整的 FSDP 训练到推理工作流程

```bash
# 1. 使用 FSDP 训练（示例命令，不属于这些脚本）
# 训练产生：results/lightrft-7B-fsdp/_actor/global_step6/

# 2. 将 FSDP 检查点转换为 HuggingFace 格式
python lightrft/utils/ckpt_scripts/fsdp2hf.py \
    --hf_base checkpoints/qwen25-vl-7b-s10-0321-150 \
    --checkpoint_dir results/lightrft-7B-fsdp/_actor/global_step6/ \
    --output_dir results/lightrft-7B-fsdp2hf

# 3. 测试转换后的模型
python lightrft/utils/ckpt_scripts/test_model.py \
    --model_path results/lightrft-7B-fsdp2hf \
    --benchmark
```

## 注意事项

- 所有转换脚本都支持语言模型（LLM）和视觉语言模型（VLM）
- `hf_base` 参数应指向用作训练起点的原始 HuggingFace 模型
- 转换后的检查点完全兼容 HuggingFace 的推理和部署工具
- 对于大型模型，请确保在转换过程中有足够的 CPU 内存
