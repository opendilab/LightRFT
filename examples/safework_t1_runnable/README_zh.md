<div align="center">

# SafeWork-R1 训练代码

SafeWork-R1 模型的官方训练代码实现，采用 SafeLadder 框架。

[🤗Huggingface 模型](https://huggingface.co/collections/AI45Research/safework-r1) • [📜技术报告](https://arxiv.org/abs/2507.18576) • [💬在线体验](https://safework-r1.ai45.shlab.org.cn/)

</div>

## 概述

本仓库包含 **SafeWork-R1** 的官方训练代码，SafeWork-R1 是一个前沿的多模态推理模型，展示了在 AI-45° 法则指导下安全性与通用智能的协同演化。

训练实现基于 **SafeLadder 框架**，具有以下特点：
- **多阶段强化学习**流程，具有渐进式安全对齐
- **多原则验证器**（Safety、Value、Knowledge）提供稳健的奖励信号
- **组相对策略优化（GRPO）**实现高效训练
- **协同部署的奖励模型**进行多维度评估

## 核心特性

### 训练能力

- ✅ **多模态支持**：同时支持纯文本和视觉语言模型（Qwen2.5-VL、InternVL3、DeepSeek-R1）
- ✅ **多个奖励模型**：Value、Safety、Knowledge、Normal 和 General 验证器
- ✅ **灵活的分布式训练**：支持 DeepSpeed ZeRO（Stage 1/2/3）和 PyTorch FSDP
- ✅ **推理引擎**：集成 vLLM 和 SGLang 实现高效生成
- ✅ **内存优化**：梯度检查点、CPU 卸载
- ✅ **高级技术**：例如 DAPO（动态采样和超长缓冲区惩罚）

### SafeLadder 框架

训练遵循 SafeLadder 多阶段流程：

1. **CoT-SFT**：思维链监督微调
2. **M³-RL**：多原则多模型多轮强化学习
3. **Safe-and-Efficient RL**：安全性聚焦优化与效率约束
4. **Deliberative Search RL**：带有搜索机制的步骤级验证

本仓库主要完成第2部分 **M³-RL**：多原则多模型多轮强化学习 的内容。

## 项目结构

```
safework_t1/
├── train_colocate.py              # GRPO 与协同部署奖励模型的主训练脚本
├── reward_models.py               # 奖励模型实现（Value、Safety、Knowledge）
├── reward_models_utils.py         # 加载和管理奖励模型的工具函数
├── test_reward_models.py          # 奖励模型测试脚本
├── run_grpo_kg_qwenvl.sh         # Knowledge + General 奖励模型训练脚本（Qwen2.5-VL）
├── run_grpo_svki_fsdp_deepseek.sh # Safety + Value + Knowledge 训练脚本（DeepSeek-70B）
└── run_grpo_svkng_fsdp_qwenvl.sh # 全部奖励模型训练脚本（Qwen2.5-VL）
```

## 安装

### 前置要求

- Python >= 3.8
- CUDA >= 11.8（用于 GPU 训练）
- 推荐 8x A100 (80GB) 或同等规格 GPU

### 环境配置

1. **克隆仓库**：
```bash
git clone https://github.com/AI45Research/SafeWork-R1.git
cd SafeWork-R1/training_code

```

2. **安装依赖**：
```bash
# 安装核心训练框架
pip install lightrft

```


## 快速开始

### 1. 准备训练数据

请在训练脚本中修改 DATA_PATH 指向您的数据集目录。

### 2. 准备奖励模型 和 SFT模型

下载 SafeWork-R1 奖励模型：
- [SafeWork-RM-Safety-7B](https://huggingface.co/AI45Research/SafeWork-RM-Safety-7B)
- [SafeWork-RM-Value-72B](https://huggingface.co/AI45Research/SafeWork-RM-Value-72B)
- [SafeWork-RM-Knowledge-72B](https://huggingface.co/AI45Research/SafeWork-RM-Knowledge-72B)


### 3. 运行训练

#### 选项 A：使用 Qwen2.5-VL-7B 快速开始

```bash
bash run_grpo_kg_qwenvl.sh
```

该脚本使用 Knowledge 和 General 奖励模型训练 Qwen2.5-VL-7B 模型。

#### 选项 B：使用全部验证器训练（Qwen2.5-VL）

```bash
bash run_grpo_svkng_fsdp_qwenvl.sh
```

该脚本使用所有奖励模型（Safety、Value、Knowledge、Normal、General）进行全面对齐。

#### 选项 C：DeepSeek-R1-70B 训练

```bash
bash run_grpo_svki_fsdp_deepseek.sh
```

该脚本使用 Safety、Value 和 Knowledge 验证器训练 DeepSeek-R1-Distill-Llama-70B 模型。

### 4. 监控训练

训练日志和检查点将保存到脚本中指定的输出目录。您可以通过以下方式监控训练进度：
- **Weights & Biases**：如果配置了 wandb 将自动记录
- **控制台日志**：训练损失、奖励分数、KL 散度
- **检查点文件**：定期保存的模型状态

## 配置

### 关键训练参数

编辑训练脚本以自定义这些参数：

```bash
# 强化学习训练参数
N_SAMPLES=8          # 每个提示词生成的响应数量
EPISODE=3            # 总训练轮数
LR=1e-6              # 学习率
MAX_LENGTH=8192      # 最大序列长度

# 批次大小
TBS=32               # 总训练批次大小
RBS=64               # 总rollout批次大小

# 奖励模型权重
RM_VALUE_WEIGHT=1.0      # Value 验证器权重
RM_SAFETY_WEIGHT=1.0     # Safety 验证器权重
RM_KNOWLEDGE_WEIGHT=1.0  # Knowledge 验证器权重
```

### 分布式训练策略

**DeepSpeed ZeRO**：
```bash
--zero_stage 2 \           # ZeRO 优化阶段（1/2/3）
--bf16 \                   # 使用 BF16 混合精度
--gradient_checkpointing   # 启用梯度检查点
```

**PyTorch FSDP**：
```bash
--fsdp \                   # 启用 FSDP 模式
--bf16 \                   # 使用 BF16 混合精度
--gradient_checkpointing   # 启用梯度检查点
```

### 奖励模型配置

在 `reward_models_utils.py` 中指定奖励模型或通过命令行配置：

```python
RECIPE = {
    "value": {
        "path": "AI45Research/SafeWork-RM-Value-72B",
        "weight": 1.0,
        "use_engine": False  # 使用 HF 推理（True 表示 SGLang）
    },
    "safety": {
        "path": "AI45Research/SafeWork-RM-Safety-7B",
        "weight": 1.0,
        "use_engine": True   # 使用 SGLang 加速推理
    },
    # ... 更多奖励模型
}
```

## 高级用法

### 自定义奖励模型

添加您自己的奖励模型：

1. **在 `reward_models.py` 中实现奖励模型类**：
```python
class MyCustomRM(nn.Module):
    def forward(self, input_ids, attention_mask, **kwargs):
        # 您的奖励计算逻辑
        return scores
```

2. **在 reward_models_utils.py 中注册**：
```python
RECIPE["custom"] = {
    "path": "path/to/your/model",
    "weight": 1.0,
    "class": "MyCustomRM"
}
```

3. **更新训练脚本**以包含您的奖励模型。


## 已训练模型

使用本训练代码，我们成功训练了以下 SafeWork-R1 模型：

| 模型 | 基础模型 | 参数量 | 链接 |
|------|----------|--------|------|
| SafeWork-R1 | Qwen2.5-VL-72B | 72B | [🤗 HF](https://huggingface.co/AI45Research/SafeWork-R1) |
| SafeWork-R1-InternVL3-78B | InternVL3-78B | 78B | [🤗 HF](https://huggingface.co/AI45Research/SafeWork-R1-InternVL3-78B) |
| SafeWork-R1-DeepSeek-70B | DeepSeek-R1-Distill-Llama-70B | 70B | [🤗 HF](https://huggingface.co/AI45Research/SafeWork-R1-DeepSeek-70B) |
| SafeWork-R1-Qwen2.5VL-7B | Qwen2.5-VL-7B | 7B | [🤗 HF](https://huggingface.co/AI45Research/SafeWork-R1-Qwen2.5VL-7B) |

## 故障排除

### 常见问题

1. **CUDA 内存不足**
   - 减少批次大小（`TBS`、`RBS`）
   - 启用梯度检查点
   - 使用 DeepSpeed ZeRO-3 或 FSDP CPU 卸载
   - 减少 `MAX_LENGTH`

2. **奖励模型加载错误**
   - 验证奖励模型路径是否正确
   - 确保有足够的 GPU 内存容纳所有奖励模型
   - 使用 `--rm_use_engine` 将奖励模型卸载到 SGLang

3. **训练速度慢**
   - 为奖励模型启用 SGLang 引擎（`use_engine: True`）
   - 使用 vLLM 加速生成
   - 如果内存允许，增加批次大小
   - 检查数据加载的网络带宽

4. **Wandb 上传失败**
   - 如果在防火墙后面，配置代理设置
   - 使用 `--wandb_mode offline` 进行离线日志记录
   - 检查 wandb API key：`wandb login`

## 性能优化建议

- **使用混合精度（BF16）**在 A100/H100 GPU 上加速训练
- **启用 flash attention**（如果您的模型支持）
- **使用 SGLang 引擎**处理奖励模型以减少推理开销
- **调整梯度累积**以最大化 GPU 利用率
- **分析您的训练**以识别瓶颈

## 引用

如果您使用本训练代码，请引用：

```bibtex
@misc{lab2025safework,
  title={SafeWork-R1: Coevolving Safety and Intelligence under the AI-45 Law},
  author={Lab, Shanghai AI and Bao, Yicheng and Chen, Guanxu and Chen, Mingkang and Chen, Yunhao and Chen, Chiyu and Chen, Lingjie and Chen, Sirui and Chen, Xinquan and Cheng, Jie and others},
  journal={arXiv preprint arXiv:2507.18576},
  year={2025}
}
```

## 许可证

本项目采用 Apache 2.0 许可证。详见 [LICENSE](../../LICENSE)。

## 致谢

- 基于 [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) 开发的。我们向 OpenRLHF 团队的杰出工作表示衷心的感谢。本项目中的部分文件和实现是从 OpenRLHF 改编和复用的。
- SafeLadder 框架建立在安全 RLHF 和多原则对齐研究的基础上
- 我们感谢开源社区提供的 DeepSpeed、FSDP、vLLM 和 SGLang
- 特别感谢 Qwen、InternVL 和 DeepSeek 团队提供的优秀基础模型

## 联系方式

如有问题或反馈：
- 在 [GitHub](https://github.com/AI45Research/SafeWork-R1/issues) 上提交 issue
- 访问我们的[项目页面](https://safework-r1.ai45.shlab.org.cn/)
- 查看[技术报告](https://arxiv.org/abs/2507.18576)
