<div align="center">

# LightRFT 中的 Math PRM 训练

面向 URSA-MATH Stage 3 复现的 LightRFT 工作目录。

</div>

## 范围说明

这个目录已经不再是通用的多模态 reward 示例，而是只保留和 URSA-MATH Stage 3 迁移与复现仍然相关的内容。

当前目标：

- actor: `URSA-8B`
- reward model: `URSA-RM-8B`
- reward label: `math_prm`、`math_psgrpo`、`math_prm_combined`、`math_rule`
- 训练主线：LightRFT PPO/GRPO + 本地 `hf` rollout
- 原始数据：`MMathCoT-1M`

## 运行时基线

运行时基线由 `/data/LightRFT/Dockerfile` 冻结。

- 不要把升级/降级依赖包当作日常调试手段。
- 优先修代码、数据转换、prompt 格式、rollout 配置和 reward wiring。
- `vllm` / `sglang` 对 URSA 的适配不属于这次最小化 upstream 示例面；当前 Stage 3 主线是本地 `hf` rollout。

## 目录结构

```text
examples/math_prm/
├── README.md                    # 当前 URSA-MATH Stage 3 布局说明（英文）
├── README_zh.md                 # 当前目录说明（中文）
├── train_colocate.py            # 主训练入口
├── run_grpo_math_prm_ursa_8b.sh # 主 Stage 3 启动脚本
├── ursa_actor.py                # URSA 专用 actor wrapper
├── reward_models.py             # 仅保留 math-only 的 URSA-RM reward 实现
├── reward_models_utils.py       # 仅保留 math-only 的 reward loader / recipe / reward_fn
├── sitecustomize.py             # 当前示例栈的本地运行时兼容钩子
├── tools/                       # 这次示例保留的最小数据准备与引擎准备工具
│   ├── __init__.py
│   ├── prepare_ursa_stage3_manifest.py
│   ├── prepare_ursa_engine_checkpoint.py
└── ursa_model/                  # 自包含的 URSA 模型代码
```

## 顶层文件职责

### 核心训练主线

- `run_grpo_math_prm_ursa_8b.sh`
  - 当前 Stage 3 复现的主启动脚本。
  - 负责串 actor 路径、reward 路径、数据集路径、FSDP、rollout 参数和可选 W&B。
- `train_colocate.py`
  - 真实的 `torchrun` 入口。
  - 构建 actor、reference model、reward model、dataset、trainer 和 rollout engine。
- `ursa_actor.py`
  - URSA 专用 actor wrapper。
  - 让 LightRFT 按 `UrsaForConditionalGeneration` 加载 actor。

### Reward 路径

- `reward_models.py`
  - 现在只保留 `MathPRMReward` 这一条活跃主线。
  - 旧的 Qwen/SafeWork reward class 已经从这里清掉。
- `reward_models_utils.py`
  - 现在只保留 math-only 的 reward loader / recipe / reward 聚合逻辑。
  - 负责 `math_prm`、`math_psgrpo`、`math_prm_combined`、`math_rule`。
- `sitecustomize.py`
  - 在冻结环境下维持这个示例栈可运行的本地兼容层。

### 自包含 URSA runtime

- `ursa_model/`
  - 本地拷贝的 URSA config、processor、image processor、projector、vision tower 和模型定义。
  - 这使得当前 Stage 3 主线不再需要直接从外部 URSA-MATH repo 动态导入运行时代码。

## `tools/` 里放的是什么

当前 upstream 示例只保留了文档主线真正需要的最小辅助脚本。

- `tools/prepare_ursa_stage3_manifest.py`
  - 把原始 `MMathCoT-1M` Stage 3 jsonl 转成 LightRFT manifest。
- `tools/prepare_ursa_engine_checkpoint.py`
  - 给 `vllm` / `sglang` 兼容性实验生成 wrapper checkpoint。

更多校验、profiling 和迁移辅助脚本会在最小 upstream PR 面之外单独维护。

## 当前主入口

如果你只关心当前 Stage 3 复现主线，通常只需要看这些文件：

- `run_grpo_math_prm_ursa_8b.sh`
- `train_colocate.py`
- `reward_models.py`
- `reward_models_utils.py`
- `tools/prepare_ursa_stage3_manifest.py`
- `tools/prepare_ursa_engine_checkpoint.py`

## 本机资源路径

当前机器上的资源布局：

```bash
URSA actor:      /home/ubuntu/URSA-MATH/checkpoints/URSA-8B
URSA reward:     /home/ubuntu/URSA-MATH/checkpoints/URSA-RM-8B
MMathCoT-1M raw: /home/ubuntu/URSA-MATH/datasets/URSA-MATH/MMathCoT-1M/train.jsonl
Image root:      /home/ubuntu/URSA-MATH/datasets/URSA-MATH/images
```

当前转换后的 manifest：

```bash
/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_psgrpo.jsonl
```

当前 manifest summary：

```bash
/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_psgrpo.summary.json
```

## 数据准备

原始 Stage 3 数据不能直接喂给 `PromptDatasetVL`。

原始 schema：

```json
{
  "image_url": "...",
  "instruction": "...",
  "output": "..."
}
```

转换后的 LightRFT schema：

```json
{
  "prompt": "...",
  "images": ["/abs/path/to/image.png"],
  "reference": "...",
  "label": "math_psgrpo"
}
```

小规模 smoke 转换：

```bash
python examples/math_prm/tools/prepare_ursa_stage3_manifest.py \
  --max-samples 32 \
  --output-path /data/LightRFT/tmp/ursa_stage3/smoke_manifest.jsonl \
  --summary-path /data/LightRFT/tmp/ursa_stage3/smoke_manifest.summary.json
```

默认全量转换：

```bash
python examples/math_prm/tools/prepare_ursa_stage3_manifest.py
```

## 训练

当前机器上，`examples/math_prm/run_grpo_math_prm_ursa_8b.sh` 里的关键默认值应当是：

```bash
PATH_TO_YOUR_BASE_MODEL="/home/ubuntu/URSA-MATH/checkpoints/URSA-8B"
PATH_TO_URSA_RM="/home/ubuntu/URSA-MATH/checkpoints/URSA-RM-8B"
PATH_TO_YOUR_MATH_DATASET="/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_psgrpo.jsonl"
EXPECTED_REWARD_LABEL="math_psgrpo"
```

启动训练：

```bash
bash examples/math_prm/run_grpo_math_prm_ursa_8b.sh
```

## Reward Label 语义

- `math_prm`
  - 纯 PRM reward，直接使用 `min(step_scores)`。
- `math_psgrpo`
  - 在 `MathPRMReward` 内部计算的 PS-GRPO reward。
- `math_prm_combined`
  - PRM + 显式 rule baseline。
- `math_rule`
  - 纯 rule-only ablation。

## 常用排查命令

- 重建 manifest：

```bash
python examples/math_prm/tools/prepare_ursa_stage3_manifest.py
```

- 测试引擎加载时重建 wrapper checkpoint：

```bash
python examples/math_prm/tools/prepare_ursa_engine_checkpoint.py
```
