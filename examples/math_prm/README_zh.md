# Math PRM:基于 Process Reward Model 的 GRPO 训练

本示例使用 GRPO 算法训练 [URSA-8B](https://huggingface.co/URSA-MATH/URSA-8B)(一个多模态数学 VLM),将 [URSA-8B-RM](https://huggingface.co/URSA-MATH/URSA-RM-8B)作为过程奖励模型(PRM),奖励信号采用 [URSA 论文(NeurIPS 2025)](https://arxiv.org/abs/2501.04686) 中提出的 **PS-GRPO** 形式。

与 `examples/gsm8k_geo3k/` 下的纯规则奖励示例不同,本目录的奖励来自一个对**每个推理步骤**打分的神经奖励模型,trajectory 级别的最终奖励由 step score 在整段回答里的演化方式决定,而不仅仅取决于最终答案是否正确。

## 概览

| 项目 | Math PRM |
|------|----------|
| 任务 | 多模态数学推理(图文混合题) |
| 模态 | Multi-modal(文本 + 图像) |
| 策略模型 | URSA-8B(SAM-B + SigLIP-L 混合视觉塔 + Qwen2.5-Math-Instruct) |
| 奖励模型 | URSA-8B-RM(过程奖励模型,逐步打分) |
| 奖励公式 | PS-GRPO:`r ∈ {0, 0.5, 1}`(正确性 × 步骤稳定性) |
| 算法 | GRPO(`group_norm` advantage estimator) |
| Rollout 引擎 | 本地 HuggingFace(URSA 的 vLLM/SGLang 适配是后续工作) |

PS-GRPO 奖励在 `MathPRMReward`([reward_models.py](reward_models.py))中计算,公式与 URSA 论文一致:

```text
r =  0                          if outcome_correct == 0
r =  1                          if outcome_correct == 1 且 没有 step-score drop
r =  0.5  ( = 1 - DROP_GAMMA)   if outcome_correct == 1 但出现了 step-score drop
```

**Step-score drop** 的判定:相邻两个 step 的 score 出现相对下降 ≥ `_DROP_THRESHOLD = 0.3` 时触发。

---

## 1. 数据预处理

训练数据为 `MMathCoT-1M`(Stage 3 切片),需要先转换成 LightRFT 的 manifest 格式。

```bash
python examples/math_prm/tools/prepare_ursa_stage3_manifest.py \
    --output-path /path/to/output/math_psgrpo.jsonl
```

转换后每行的 schema:

```json
{
  "prompt": "数学题文本",
  "images": ["/abs/path/to/image.png"],
  "reference": "标准答案",
  "label": "math_psgrpo"
}
```

`label` 决定走哪条奖励路径:

| Label | 奖励信号 |
|---|---|
| `math_psgrpo` | PS-GRPO:`{0, 0.5, 1}`(本示例默认) |
| `math_prm` | 纯 PRM 聚合 step score(连续值,`[0, 1]`) |
| `math_prm_combined` | PRM 聚合分 + 0.5 × 规则正确性 |
| `math_rule` | 纯规则基线 `{0, 1}`,只看答案是否对 |

要做小规模 smoke 转换(32 条),传 `--max-samples 32`。

---

## 2. 模型 checkpoint

需要同时准备 URSA-8B 策略模型和 URSA-8B-RM 奖励模型:

```bash
# Hugging Face 模型 ID
URSA-MATH/URSA-8B       # 策略模型
URSA-MATH/URSA-RM-8B    # 奖励模型
```

下载到本地目录后,在 `run_grpo_math_prm_ursa_8b.sh` 里设置路径。

---

## 3. 配置并启动训练

编辑 [run_grpo_math_prm_ursa_8b.sh](run_grpo_math_prm_ursa_8b.sh) 顶部的 `Part 1: User Configuration`:

```bash
PATH_TO_YOUR_BASE_MODEL="/path/to/URSA-8B"
PATH_TO_URSA_RM="/path/to/URSA-RM-8B"
PATH_TO_YOUR_MATH_DATASET="/path/to/math_psgrpo.jsonl"
EXPERIMENT_NAME="lightrft-ursa8b-math-prm"
export WANDB_API_KEY="YOUR_WANDB_API_KEY"   # 留空表示禁用 W&B
```

然后运行:

```bash
bash examples/math_prm/run_grpo_math_prm_ursa_8b.sh
```

默认机器配置是 `1 node × 8 A100 GPU`。多机或不同 GPU 数,通过标准环境变量覆盖:

```bash
NNODES=2 GPUS_PER_NODE=8 NODE_RANK=0 \
MASTER_ADDR=10.0.0.1 MASTER_PORT=20092 \
bash examples/math_prm/run_grpo_math_prm_ursa_8b.sh
```

---

## 4. 关键超参

启动器默认值与 URSA-MATH 论文 Stage 3 一致:

| 参数 | 默认值 | 说明 |
|---|---|---|
| `N_SAMPLES` | 8 | 每个 prompt 的 GRPO 采样数 |
| `EPISODE` | 10 | 训练总轮数 |
| `RBS` / `TBS` | 128 / 128 | rollout / 训练 batch size |
| `KL` | 0.001 | 初始 KL 系数 |
| `KL_TARGET` | (默认关) | 设了之后切换到 AdaptiveKLController |
| `LR` | 1e-6 | actor 学习率 |
| `PROMPT_MAX_LEN` | 1024 | |
| `GENERATE_MAX_LEN` | 3072 | |
| `MAX_SAMPLES` | 15360 | 训练子集上限(论文规模代理) |
| `EVAL_HOLDOUT_SIZE` | 500 | 从 `prompt_data` 中确定性切出来的 in-domain 验证集大小 |

如果观察到 KL 漂移,推荐打开自适应 KL 控制器:`KL_TARGET=0.5`(或更小)。

---

## 5. WandB 指标

WandB 面板分三个 namespace:

- `rollout/*` — 每步 rollout 统计:`reward`、`outcome_correct`、`model_reward`、`has_drop_moment`、`response_length`。
- `train/*` — 每步训练统计:`policy_loss`、`kl`、`actor_lr`、`advantages`、`return`。
- `eval/*` — 验证集评测:`reward`、`outcome_correct`、`response_length`、`answer_extraction_failed`。

`MathPRMReward` 输出的全套 per-sample 奖励 metric,见 [reward_models.py](reward_models.py) 中 `forward()` 顶部的注释。

---

## 6. 目录文件说明

```text
examples/math_prm/
├── README.md / README_zh.md      - 本指南
├── train_colocate.py             - torchrun 入口
├── run_grpo_math_prm_ursa_8b.sh  - 启动脚本
├── reward_models.py              - MathPRMReward 实现(PS-GRPO)
├── reward_models_utils.py        - 按 label 选择奖励配方的逻辑
├── ursa_actor.py                 - URSA 专用 actor wrapper
├── math_prm_trainer.py           - MathPRMSPMDPPOTrainerVL(精简的 wandb 指标映射)
├── math_prm_output.py            - "†Answer:" marker / 结构化停止辅助函数
├── rollout_eos_patch.py          - 在 FSDP 下注入 StoppingCriteria 保证可靠 EOS
├── ursa_model/                   - URSA 模型代码(config / processor / model)
└── tools/
    ├── prepare_ursa_stage3_manifest.py     - 数据集转换脚本
    └── prepare_ursa_engine_checkpoint.py   - engine 模式 checkpoint 包装工具
```

---

## 7. 引用

如果使用了本示例,请引用 URSA 论文:

```bibtex
@article{luo2025ursa,
  title={URSA: Understanding and Verifying Chain-of-Thought Reasoning in Multimodal Mathematics},
  author={Luo, Ruilin and Zheng, Zhuofan and Wang, Yifan and Yu, Yiyao and Ni, Xinzhe and Lin, Zicheng and Zeng, Jin and Yang, Yujiu},
  journal={NeurIPS},
  year={2025}
}
```
