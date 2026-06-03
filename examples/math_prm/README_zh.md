# Math PRM：基于过程奖励模型 (PRM) 的 GRPO 训练

本示例使用 [URSA-8B](https://huggingface.co/URSA-MATH/URSA-8B)（多模态数学 VLM）作为 actor，配合 [URSA-8B-RM](https://huggingface.co/URSA-MATH/URSA-RM-8B) 作为过程奖励模型 (PRM)，按 [URSA 论文（NeurIPS 2025）](https://arxiv.org/abs/2501.04686)所提的 **PS-GRPO** 奖励路径用 GRPO 算法训练。

不同于 `examples/gsm8k_geo3k/` 那类规则型 reward 示例，这里的 reward 来自一个对**每一步推理**打分的神经网络奖励模型，最终的 trajectory-level reward 取决于 step scores 沿 response 的**演化形态**，而不仅仅是最终答案是否正确。

本 example 同时附带**两条算法路径**用于对比：

1. **PS-GRPO**（`run_grpo_math_prm_ursa_8b.sh`）—— 论文最终采纳的 `r ∈ {0, 0.5, 1}` 单标量奖励，由标准 GRPO 处理。**生产推荐配方**。
2. **Paper Eq.9 严格 variant 2**（`run_grpo_math_prm_ursa_8b_variant2.sh`）—— 论文附录 B.1 的逐 step PRM advantage：`A_t^i = r_{s,t}^i · GroupNorm_G(r̄_s^i) + GroupNorm_G(r_o^i)`。论文自身否决了它，本 example 保留只为做 ablation 对照。完整实现位于 [`ursa_variant2.py`](ursa_variant2.py)（不修改 `lightrft/`）。

## 总览

| 项 | Math PRM |
|------|----------|
| 任务 | 多模态数学推理（文本+图像题） |
| 模态 | 多模态（文本 + 图像） |
| Actor | URSA-8B（SAM-B + SigLIP-L 视觉塔 + Qwen2.5-Math-Instruct） |
| Reward Model | URSA-8B-RM（过程奖励模型，step-level scoring） |
| Reward 公式（PS-GRPO） | `r ∈ {0, 0.5, 1}`（正确性 × step 稳定性） |
| 算法 | GRPO（group_norm 优势估计器）或 paper Eq.9 的 `ursa_variant2` |
| Rollout 引擎 | 本地 Hugging Face（vLLM/SGLang 对 URSA 的支持待后续） |

PS-GRPO 奖励在 `MathPRMReward`（[reward_models.py](reward_models.py)）中按 URSA 论文公式计算：

```text
r =  0                          若 outcome_correct == 0
r =  1                          若 outcome_correct == 1 且无 step-score drop
r =  0.5  ( = 1 - DROP_GAMMA)   若 outcome_correct == 1 但存在 step-score drop
```

**Step-score drop** 的判定：任意相邻 step score 出现相对降幅 ≥ `_DROP_THRESHOLD = 0.3`。

---

## 1. 数据预处理

训练数据为 `MMathCoT-1M`（Stage 3 子集），需要转换成 LightRFT 的 manifest 格式。`--input-path` 与 `--image-root` 均**必填**（无默认值——路径与环境相关）：

```bash
python examples/math_prm/tools/prepare_ursa_stage3_manifest.py \
    --input-path  /your/data/URSA-MATH/MMathCoT-1M/train.jsonl \
    --image-root  /your/data/URSA-MATH/images \
    --output-path /your/output/math_psgrpo.jsonl
```

转换后每行 manifest 形如：

```json
{
  "prompt": "数学题目文本",
  "images": ["/abs/path/to/image.png"],
  "reference": "标准答案",
  "label": "math_psgrpo"
}
```

`label` 字段决定选择哪一条 reward 路径。可选值：

| Label | Reward 信号 |
|---|---|
| `math_psgrpo` | PS-GRPO：`{0, 0.5, 1}`（本 example 默认） |
| `math_prm` | 纯 PRM 聚合 step score（连续值 `[0, 1]`） |
| `math_prm_combined` | PRM 聚合分数 + 0.5 × 规则正确性 |
| `math_rule` | 规则基线：`{0, 1}` 按答案匹配 |
| `math_per_step_prm` | 逐 step PRM 分数，供 `--advantage_estimator ursa_variant2`（paper Eq.9，详见 §6）使用 |

需要 32 行小规模转换做 smoke 时用 `--max-samples 32`。

---

## 2. 模型 checkpoint

需要 URSA-8B（actor）与 URSA-8B-RM（reward model）两个权重：

```bash
# Hugging Face IDs
URSA-MATH/URSA-8B       # actor
URSA-MATH/URSA-RM-8B    # reward model
```

下载到本地后在 `run_grpo_math_prm_ursa_8b.sh` 里配置路径。

---

## 3. 配置并启动训练（PS-GRPO 配方）

编辑 [run_grpo_math_prm_ursa_8b.sh](run_grpo_math_prm_ursa_8b.sh) 顶部的 `Part 1: User Configuration`：

```bash
PATH_TO_YOUR_BASE_MODEL="/path/to/URSA-8B"
PATH_TO_URSA_RM="/path/to/URSA-RM-8B"
PATH_TO_YOUR_MATH_DATASET="/path/to/math_psgrpo.jsonl"
EXPERIMENT_NAME="lightrft-ursa8b-math-prm"
export WANDB_API_KEY="YOUR_WANDB_API_KEY"   # 留空则禁用 W&B
```

然后运行：

```bash
bash examples/math_prm/run_grpo_math_prm_ursa_8b.sh
```

默认目标硬件 `1 节点 × 8 A100/H100`。改 topology 用环境变量 override：

```bash
NNODES=2 GPUS_PER_NODE=8 NODE_RANK=0 \
MASTER_ADDR=10.0.0.1 MASTER_PORT=20092 \
bash examples/math_prm/run_grpo_math_prm_ursa_8b.sh
```

---

## 4. 关键超参

启动脚本使用 URSA-MATH 论文 Stage 3 默认值：

| 参数 | 值 | 备注 |
|---|---|---|
| `N_SAMPLES` | 8 | 每个 prompt GRPO 采样数 |
| `EPISODE` | 10 | 总训练 episodes |
| `RBS` / `TBS` | 128 / 128 | rollout / training batch size |
| `KL` | 0.001 | 初始 KL 系数 |
| `KL_TARGET` | (off) | 设值后切到 AdaptiveKLController |
| `LR` | 1e-6 | Actor 学习率 |
| `PROMPT_MAX_LEN` | 1024 | |
| `GENERATE_MAX_LEN` | 3072 | |
| `MAX_SAMPLES` | 15360 | 训练子集上限（论文 proxy） |
| `EVAL_HOLDOUT_SIZE` | 500 | 从 `prompt_data` 中保留的确定性 held-out 子集 |

观察到 KL 飘移时建议开 adaptive KL 控制器：`KL_TARGET=0.5`。

---

## 5. 日志字段

W&B 面板按三个 namespace 划分：

- `rollout/*` — 每步 rollout 统计：`reward`、`outcome_correct`、`model_reward`、`has_drop_moment`、`response_length`、`step_score_min/mean/last`、`step_count`、`final_reward`、`max_relative_drop`、`answer_tag_present`、`answer_extraction_failed`、`used_answer_fallback`、`used_mathruler`、`reference_supported`，以及 variant-2 诊断字段 `alignment_failed` / `n_aligned_steps`。
- `train/*` — 每步训练统计：`policy_loss`、`kl`、`actor_lr`、`advantages`、`return`，以及 variant-2 诊断字段 `ursa_v2_adv_pos_frac` / `_neg_frac` / `_zero_frac` / `_abs_mean` / `_oc_normed_std` / `_msp_normed_std` / `_traj_step_count_mean`。
- `eval/*` — held-out 评测：`reward`、`outcome_correct`、`response_length`、`answer_extraction_failed`、`has_drop_moment`、`model_reward`、`step_score_min/mean/last`、`step_count`、`final_reward`、`max_relative_drop`、`answer_tag_present`、`used_answer_fallback`、`used_mathruler`、`reference_supported`。

`MathPRMReward` 输出的全套 per-sample 奖励 metric 文档见 [reward_models.py](reward_models.py) 中 `forward()` 顶部注释。

---

## 6. Paper Eq.9 严格对齐 — variant 2 路径

`run_grpo_math_prm_ursa_8b_variant2.sh` 是 URSA 论文附录 B.1 Eq.9 "variant 2" 严格实现，与 PS-GRPO 并行存在，用于 ablation 对比。实现在 [`ursa_variant2.py`](ursa_variant2.py)，通过幂等 monkey-patch 注册一个新 `--advantage_estimator ursa_variant2`（不修改 `lightrft/`）。

### 公式

```text
A_t^i = r_{s,t}^i · GroupNorm_G(r̄_s^i)     ← process-reward 项
      +              GroupNorm_G(r_o^i)      ← outcome-reward 项
```

其中 `t` 是 **step 索引**（不是 token），`r_{s,t}^i` 为 trajectory `i` 第 `t` 个 step 的 sigmoid PRM 分，`r̄_s^i = mean_t r_{s,t}^i`，`r_o^i ∈ {0,1}` 是 outcome reward，`G` 是 GRPO group size（`n_samples_per_prompt`）。逐 step `A_t^i` 广播到该 step 覆盖的所有 token。**无 cumulative return**，outcome 项保留（不像 Math-Shepherd 风格的 Mode B 那样被丢弃）。

### 数据集 / 启动流程

variant 2 路径要求 manifest 行 label 是 `math_per_step_prm` 而不是 `math_psgrpo`。最简单方法是 sed-relabel PS-GRPO manifest：

```bash
sed 's/"label":[ ]*"math_psgrpo"/"label": "math_per_step_prm"/g' \
    /path/to/math_psgrpo.jsonl \
    > /path/to/math_per_step_prm.jsonl
```

variant 2 启动脚本会自动检测 `PATH_TO_YOUR_MATH_DATASET` 是否指向 psgrpo 路径，若是则自动 swap 到 `*per_step_prm*.jsonl` 兄弟文件，并在训练前 assert 首行 label 是 `math_per_step_prm`。如需自定义路径，设 `PATH_TO_YOUR_MATH_DATASET_VARIANT2`。

```bash
PATH_TO_YOUR_MATH_DATASET_VARIANT2=/path/to/math_per_step_prm.jsonl \
bash examples/math_prm/run_grpo_math_prm_ursa_8b_variant2.sh
```

`--per_step_reward_mode`（`raw` / `group_norm`）只影响**遗留 Math-Shepherd 风格逐 token reward 路径**（`_apply_step_reward_group_norm` 不同聚合方式）；`--advantage_estimator ursa_variant2` 自带 group normalization，不受该 flag 影响。

### 单元测试

`test_ursa_variant2.py` 包含 9 个 AC（acceptance criterion）级单测：与手算 Eq.9 数值等价、K=2/K=4 group 的 GroupNorm 正确性、span 广播、outcome 项非旁路。运行：

```bash
python3 -m unittest examples.math_prm.test_ursa_variant2 -v
```

---

## 7. 实验结果 — 9 天生产 run（PS-GRPO）

8× H100 上 PS-GRPO 配方跑满 9 天的关键指标如下。variant 2 路径并行跑了同样 9 天作对照（W&B `kdwjt4eo`），完整对比见 [PR #53 最终报告 comment](https://github.com/opendilab/LightRFT/pull/53#issuecomment-4608400929)。

| 指标 | baseline (Step 20) | peak | final | Δ vs baseline |
|---|---|---|---|---|
| `eval/outcome_correct` | 0.5952 | **0.6508** (Step 231) | 0.6290 (Step 1008) | **+3.4 pp** |
| `eval/answer_extraction_failed` | 0.028 | 0.018 (~Step 160) | 0.034 | -0.6 pp ↓ |
| `eval/has_drop_moment` | 0.0 | — | 0.0 | (PRM 全程未触发) |
| `eval/response_length` | 400 | 337 (~Step 240) | 377 | -23 ↓ |
| `rollout/alignment_failed` | 0 | — | 0 | 100% step 边界对齐 |
| W&B run | [`kdwjt4eo`](https://wandb.ai/hansbug/LightRFT-URSA8B-Stage3/runs/kdwjt4eo) |

#### eval 轨迹

`eval/outcome_correct` 在 Step 231 见峰 +5.6pp，约 Step 300 出现一次 dip（reward hacking signature）但**自愈**，剩 7 天稳定在 0.60–0.65 区间：

![eval trajectory](assets/exp_20260603/eval_outcome.png)

#### KL + rollout 全局视角

`train/kl` 走出 warmup 后（1e-4 → 1.0 by Step ~200），在 1–100 区间震荡，偶发单 batch >100 spike 总能自愈。`rollout/outcome_correct` 与 `rollout/model_reward` 长期同向变化（无 reward hacking 解耦）：

![KL + rollout](assets/exp_20260603/kl_and_rollout.png)

#### eval 生成质量

`eval/answer_extraction_failed` 在 Step 300 dip 期间短暂飙到 18%（URSA 论文警告的 `†Answer:` 格式漂移信号），之后回稳到 2–5%。`eval/response_length` 与 `eval/step_count` 稳定 — 无 length collapse：

![eval quality](assets/exp_20260603/eval_quality.png)

#### variant 2 路径健康度（W&B run `kdwjt4eo`）

`ursa_v2_adv_pos_frac` 与 `_neg_frac` 长期保持平衡（30–40% / 25–35%）—— GroupNorm 持续产出 signed advantages。`_msp_normed_std` 贴近 1.0。`rollout/alignment_failed` 全程 0：

![variant 2 health](assets/exp_20260603/variant2_health.png)

---

## 8. 目录文件清单

```text
examples/math_prm/
├── README.md                              - 本文档（英文）
├── README_zh.md                           - 本文档（中文）
├── train_colocate.py                      - 训练主入口（由 torchrun 调用）
├── run_grpo_math_prm_ursa_8b.sh           - PS-GRPO 启动脚本（推荐）
├── run_grpo_math_prm_ursa_8b_variant2.sh  - paper Eq.9 严格启动脚本（ablation）
├── reward_models.py                       - MathPRMReward 实现（PS-GRPO）
├── reward_models_utils.py                 - 按 label 选 reward 配方的逻辑
├── ursa_actor.py                          - URSA actor wrapper
├── ursa_variant2.py                       - UrsaVariant2Calculator（paper Eq.9，纯 examples/）
├── math_prm_trainer.py                    - MathPRMSPMDPPOTrainerVL（wandb metric 映射）
├── math_prm_output.py                     - "†Answer:" marker / structured-stop helpers
├── rollout_eos_patch.py                   - FSDP 下可靠 EOS 的 StoppingCriteria 注入
├── test_ursa_variant2.py                  - variant 2 的 9 个 AC 级单测
├── ursa_model/                            - 内置 URSA 模型代码（config / processor / model）
├── tools/
│   ├── prepare_ursa_stage3_manifest.py   - 数据集转换工具
│   └── prepare_ursa_engine_checkpoint.py - Engine-mode checkpoint wrapper
└── assets/
    └── exp_20260603/                      - 9 天生产 run 的 W&B 截图
```

---

## 9. 引用

使用本 example 请引用 URSA 论文：

```bibtex
@article{luo2025ursa,
  title={URSA: Understanding and Verifying Chain-of-Thought Reasoning in Multimodal Mathematics},
  author={Luo, Ruilin and Zheng, Zhuofan and Wang, Yifan and Yu, Yiyao and Ni, Xinzhe and Lin, Zicheng and Zeng, Jin and Yang, Yujiu},
  journal={NeurIPS},
  year={2025}
}
```

---

## License

本 example 与上层 LightRFT 项目使用同一 License（见仓库根目录 `LICENSE`）。
