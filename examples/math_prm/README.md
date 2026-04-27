# Math PRM: GRPO Training with a Process Reward Model

This example trains [URSA-8B](https://huggingface.co/URSA-MATH/URSA-8B) — a multimodal math VLM — with [URSA-8B-RM](https://huggingface.co/URSA-MATH/URSA-RM-8B) as a Process Reward Model (PRM), using the GRPO algorithm with a **PS-GRPO** reward signal as proposed in the [URSA paper (NeurIPS 2025)](https://arxiv.org/abs/2501.04686).

Unlike the rule-based examples under `examples/gsm8k_geo3k/`, the reward here comes from a neural reward model that scores **each reasoning step**, and the final per-trajectory reward depends on *how the step scores evolve* across the response, not just on whether the final answer is right.

## Overview

| Item | Math PRM |
|------|----------|
| Task | Multimodal math reasoning (text + image questions) |
| Modality | Multi-modal (text + image) |
| Actor | URSA-8B (hybrid SAM-B + SigLIP-L vision tower + Qwen2.5-Math-Instruct) |
| Reward Model | URSA-8B-RM (process reward model, step-level scoring) |
| Reward formula | PS-GRPO: `r ∈ {0, 0.5, 1}` (correctness × step-stability) |
| Algorithm | GRPO (group_norm advantage estimator) |
| Rollout engine | Local Hugging Face (vLLM/SGLang URSA support is future work) |

The PS-GRPO reward is computed inside `MathPRMReward` ([reward_models.py](reward_models.py)) and follows the URSA paper:

```text
r =  0                          if outcome_correct == 0
r =  1                          if outcome_correct == 1 and no step-score drop
r =  0.5  ( = 1 - DROP_GAMMA)   if outcome_correct == 1 but a step-score drop occurred
```

A **step-score drop** is detected when any consecutive pair of step scores has a relative drop ≥ `_DROP_THRESHOLD = 0.3`.

---

## 1. Dataset Preprocessing

The training data is `MMathCoT-1M` (Stage 3 split), which needs to be converted into the LightRFT manifest schema.

```bash
python examples/math_prm/tools/prepare_ursa_stage3_manifest.py \
    --output-path /path/to/output/math_psgrpo.jsonl
```

Each row in the converted manifest looks like:

```json
{
  "prompt": "Math question text",
  "images": ["/abs/path/to/image.png"],
  "reference": "Ground-truth answer",
  "label": "math_psgrpo"
}
```

The `label` field is what selects the reward path. Available labels:

| Label | Reward signal |
|---|---|
| `math_psgrpo` | PS-GRPO: `{0, 0.5, 1}` (default for this example) |
| `math_prm` | Pure PRM aggregated step score (continuous in `[0, 1]`) |
| `math_prm_combined` | PRM aggregated score + 0.5 × rule-based correctness |
| `math_rule` | Rule-only baseline `{0, 1}` based on answer match |

For a smoke conversion (32 samples), pass `--max-samples 32`.

---

## 2. Model Checkpoints

You need both the URSA-8B actor and the URSA-8B-RM reward model:

```bash
# Hugging Face IDs
URSA-MATH/URSA-8B       # actor
URSA-MATH/URSA-RM-8B    # reward model
```

Download to a local directory and set the paths in `run_grpo_math_prm_ursa_8b.sh`.

---

## 3. Configure and Run Training

Edit `Part 1: User Configuration` at the top of [run_grpo_math_prm_ursa_8b.sh](run_grpo_math_prm_ursa_8b.sh):

```bash
PATH_TO_YOUR_BASE_MODEL="/path/to/URSA-8B"
PATH_TO_URSA_RM="/path/to/URSA-RM-8B"
PATH_TO_YOUR_MATH_DATASET="/path/to/math_psgrpo.jsonl"
EXPERIMENT_NAME="lightrft-ursa8b-math-prm"
export WANDB_API_KEY="YOUR_WANDB_API_KEY"   # leave empty to disable W&B
```

Then run:

```bash
bash examples/math_prm/run_grpo_math_prm_ursa_8b.sh
```

The default machine target is `1 node × 8 A100 GPUs`. For a different topology, override the standard env vars:

```bash
NNODES=2 GPUS_PER_NODE=8 NODE_RANK=0 \
MASTER_ADDR=10.0.0.1 MASTER_PORT=20092 \
bash examples/math_prm/run_grpo_math_prm_ursa_8b.sh
```

---

## 4. Key Hyperparameters

The launcher uses the URSA-MATH paper's Stage 3 defaults:

| Param | Value | Notes |
|---|---|---|
| `N_SAMPLES` | 8 | Responses sampled per prompt for GRPO |
| `EPISODE` | 10 | Total training episodes |
| `RBS` / `TBS` | 128 / 128 | Rollout / training batch size |
| `KL` | 0.001 | Initial KL coefficient |
| `KL_TARGET` | (off) | If set, switches to AdaptiveKLController |
| `LR` | 1e-6 | Actor learning rate |
| `PROMPT_MAX_LEN` | 1024 | |
| `GENERATE_MAX_LEN` | 3072 | |
| `MAX_SAMPLES` | 15360 | Cap on training subset (paper proxy) |
| `EVAL_HOLDOUT_SIZE` | 500 | A deterministic held-out subset is reserved from `prompt_data` for in-domain eval |

To enable the adaptive KL controller (recommended if you observe the KL drifting), set `KL_TARGET` to a small positive value, e.g. `KL_TARGET=0.5`.

---

## 5. What's Logged

Wandb panels are split into three namespaces:

- `rollout/*` — per-step rollout statistics: `reward`, `outcome_correct`, `model_reward`, `has_drop_moment`, `response_length`.
- `train/*` — per-step training statistics: `policy_loss`, `kl`, `actor_lr`, `advantages`, `return`.
- `eval/*` — evaluation pass on the held-out split: `reward`, `outcome_correct`, `response_length`, `answer_extraction_failed`.

The full per-sample reward metric set emitted by `MathPRMReward` is documented at the top of `forward()` in [reward_models.py](reward_models.py).

---

## 6. Files Under This Directory

```text
examples/math_prm/
├── README.md / README_zh.md      - This guide
├── train_colocate.py             - Main training entry (called by torchrun)
├── run_grpo_math_prm_ursa_8b.sh  - Launcher script
├── reward_models.py              - MathPRMReward implementation (PS-GRPO)
├── reward_models_utils.py        - Reward recipe / mixing logic per label
├── ursa_actor.py                 - URSA-specific actor wrapper
├── math_prm_trainer.py           - MathPRMSPMDPPOTrainerVL (curated wandb metric mapping)
├── math_prm_output.py            - "†Answer:" marker / structured-stop helpers
├── rollout_eos_patch.py          - StoppingCriteria injection for reliable EOS under FSDP
├── ursa_model/                   - Vendored URSA model code (config / processor / model)
└── tools/
    ├── prepare_ursa_stage3_manifest.py     - Dataset conversion tool
    └── prepare_ursa_engine_checkpoint.py   - Engine-mode checkpoint wrapper
```

---

## 7. Citation

If you use this example, please cite the URSA paper:

```bibtex
@article{luo2025ursa,
  title={URSA: Understanding and Verifying Chain-of-Thought Reasoning in Multimodal Mathematics},
  author={Luo, Ruilin and Zheng, Zhuofan and Wang, Yifan and Yu, Yiyao and Ni, Xinzhe and Lin, Zicheng and Zeng, Jin and Yang, Yujiu},
  journal={NeurIPS},
  year={2025}
}
```
