# GRPO Training with GSM8K & Geo3K Datasets

This tutorial walks you through using LightRFT to run GRPO (Group Relative Policy Optimization) training on two math reasoning datasets: **GSM8K** (text-only) and **Geo3K** (multi-modal with geometry diagrams).

## Overview

| Item | GSM8K | Geo3K |
|------|-------|-------|
| Task | Grade-school math word problems | Geometry problem solving |
| Modality | Text-only | Multi-modal (text + image) |
| Source | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | [hiyouga/geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k) |
| Train / Test | 7,473 / 1,319 | ~2,100 / 601 |
| Reward | Pure rule-based (no neural reward model) | Pure rule-based (no neural reward model) |
| Base Model | Qwen2.5-0.5B-Instruct (or larger) | Qwen2.5-VL-7B-Instruct |

Both tasks use a **pure rule-based reward** mechanism:
- **Format reward (10%)**: Checks for `<think>...</think>` reasoning tags followed by `\boxed{}` answer notation.
- **Accuracy reward (90%)**: Verifies the final answer against the ground truth using [mathruler](https://github.com/open-compass/mathruler).

No separate neural reward model is required.

---

## 1. Dataset Preprocessing

Before training, convert the raw HuggingFace datasets into LightRFT-compatible parquet format.

### 1.1 GSM8K Preprocessing

```bash
python examples/gsm8k_geo3k/data_preprocess/gsm8k.py \
    --local_save_dir ~/data/gsm8k
```

The script performs the following steps:

1. Loads `openai/gsm8k` from HuggingFace (or a local path via `--local_dataset_path`).
2. Extracts the numerical answer from the `#### ANSWER` pattern in each solution (commas are stripped, e.g. `1,000` → `1000`).
3. Wraps each question into a chat-structured prompt:

```json
{
  "prompt": [
    {"role": "system", "content": "You FIRST think about the reasoning process step by step ... The final answer MUST BE put in \\boxed{} after the reasoning."},
    {"role": "user", "content": "<original question>"}
  ],
  "extra_info": {
    "label": "gsm8k_rule",
    "reference": "<extracted numerical answer>"
  }
}
```

4. Saves `train.parquet` and `test.parquet` to the specified directory.

### 1.2 Geo3K Preprocessing

```bash
python examples/gsm8k_geo3k/data_preprocess/geo3k.py \
    --local_save_dir ~/data/geo3k
```

The script performs the following steps:

1. Loads `hiyouga/geometry3k` from HuggingFace (or a local path via `--local_dataset_path`).
2. Preserves the geometry diagram images from the dataset.
3. Wraps each problem into a chat-structured prompt with image references:

```json
{
  "prompt": [
    {"role": "system", "content": "You FIRST think about the reasoning process ... The final answer MUST BE put in \\boxed{}."},
    {"role": "user", "content": "<geometry problem text>"}
  ],
  "images": ["<PIL Image>"],
  "extra_info": {
    "label": "geo3k_rule",
    "reference": "<ground truth answer>"
  }
}
```

4. Saves `train.parquet` and `test.parquet` to the specified directory.

### 1.3 Output Format Summary

Both preprocessing scripts produce a unified schema:

| Field | Description |
|-------|-------------|
| `prompt` | Chat-structured list `[{role, content}, ...]` |
| `images` | List of PIL images (Geo3K only; absent for GSM8K) |
| `extra_info.label` | `"gsm8k_rule"` or `"geo3k_rule"` — maps to the reward RECIPE |
| `extra_info.reference` | Ground truth answer string for rule-based evaluation |

---

## 2. Training

### 2.1 GSM8K — Text-Only GRPO Training

```bash
# Edit paths in the script first, then:
bash examples/gsm8k_geo3k/run_grpo_gsm8k_qwen2.5_0.5b.sh
```

Or launch directly with `torchrun`:

```bash
torchrun --nnodes 1 --nproc-per-node 8 \
    examples/gsm8k_geo3k/train_colocate.py \
    --pretrain Qwen/Qwen2.5-0.5B-Instruct \
    --prompt_data /path/to/gsm8k_dataset \
    --text_only \
    --advantage_estimator group_norm \
    --n_samples_per_prompt 5 \
    --num_episodes 30 \
    --train_batch_size 128 \
    --rollout_batch_size 128 \
    --actor_learning_rate 1e-6 \
    --init_kl_coef 0.01 \
    --kl_estimator k3 \
    --use_kl_loss \
    --fsdp --zero_stage 3 --bf16 \
    --flash_attn --gradient_checkpointing \
    --engine_type sglang --engine_tp_size 2 \
    --engine_mem_util 0.6 --enable_engine_sleep \
    --save_path results/gsm8k_grpo \
    --eval_steps 20 --save_steps 20
```

### 2.2 Geo3K — Multi-Modal GRPO Training

```bash
# Edit paths in the script first, then:
bash examples/gsm8k_geo3k/run_grpo_geo3k_qwen2.5_vl_7b.sh
```

Or launch directly with `torchrun`:

```bash
torchrun --nnodes 1 --nproc-per-node 8 \
    examples/gsm8k_geo3k/train_colocate.py \
    --pretrain Qwen/Qwen2.5-VL-7B-Instruct \
    --prompt_data /path/to/geo3k_dataset \
    --mixed_mm_data \
    --images_key images \
    --advantage_estimator group_norm \
    --n_samples_per_prompt 8 \
    --num_episodes 20 \
    --train_batch_size 128 \
    --rollout_batch_size 128 \
    --actor_learning_rate 1e-6 \
    --init_kl_coef 0.01 \
    --kl_estimator k3 \
    --use_kl_loss \
    --fsdp --zero_stage 3 --bf16 \
    --flash_attn --gradient_checkpointing \
    --engine_type sglang --engine_tp_size 2 \
    --engine_mem_util 0.6 --enable_engine_sleep \
    --limit_mm_image_per_prompt 10 \
    --save_path results/geo3k_grpo \
    --eval_steps 20 --save_steps 20
```

### 2.3 Geo3K — LoRA GRPO Training (Parameter-Efficient)

For resource-constrained environments, use LoRA to fine-tune only a small fraction of parameters:

```bash
bash examples/gsm8k_geo3k/run_grpo_geo3k_lora_qwen2.5_vl_7b.sh
```

Key LoRA-specific parameters:

```bash
--lora_rank 128 \
--lora_alpha 256 \
--target_modules all-linear
```

---

## 3. Key Hyperparameters

| Parameter | GSM8K Default | Geo3K Default | Description |
|-----------|:---:|:---:|-------------|
| `--n_samples_per_prompt` | 5 | 8 | Number of rollout samples per prompt (GRPO group size) |
| `--num_episodes` | 30 | 20 | Total training episodes |
| `--train_batch_size` | 128 | 128 | Global training batch size |
| `--rollout_batch_size` | 128 | 128 | Global rollout batch size |
| `--micro_train_batch_size` | 4 | 4 | Per-GPU micro training batch size |
| `--micro_rollout_batch_size` | 4 | 8 | Per-GPU micro rollout batch size |
| `--actor_learning_rate` | 1e-6 | 1e-6 | Actor learning rate |
| `--lr_warmup_ratio` | 0.03 | 0.03 | Learning rate warmup ratio |
| `--init_kl_coef` | 0.01 | 0.01 | KL divergence penalty coefficient |
| `--kl_estimator` | k3 | k3 | KL estimator type |
| `--prompt_max_len` | 1024 | 1024 | Maximum prompt length |
| `--generate_max_len` | 2048 | 2048 | Maximum generation length |
| `--advantage_estimator` | group_norm | group_norm | GRPO advantage estimation |
| `--engine_tp_size` | 2 | 2 | Inference engine tensor parallelism |
| `--l2` | 1e-2 | 1e-2 | L2 regularization weight |
| `--text_only` | Yes | No | Text-only mode (no image processing) |
| `--mixed_mm_data` | No | Yes | Enable multi-modal data processing |

---

## 4. Reward Mechanism

### 4.1 RECIPE Configuration

The reward system is driven by a label-based RECIPE mapping defined in `examples/gsm8k_geo3k/reward_models_utils.py`:

```python
RECIPE = {
    "geo3k_rule": [("geo3k_rule", None, 1.0)],
    "gsm8k_rule": [("gsm8k_rule", None, 1.0)],
}
```

Each sample's `label` field (set during preprocessing) determines which reward function is applied.

### 4.2 Format Reward (10% weight)

Validates the model output follows the required reasoning format:

```
<think> reasoning process here </think> final thought and \boxed{answer}
```

The check uses regex to verify:
1. `<think>...</think>` tags are present.
2. `\boxed{...}` notation is present.
3. The `</think>` closing tag appears **before** `\boxed{}`.

```python
def format_reward_fn(sol: str) -> float:
    think_match = re.search(r'<think>.*?</think>', sol, re.DOTALL)
    boxed_match = re.search(r'\\boxed\{.*?\}', sol, re.DOTALL)
    if think_match and boxed_match:
        return 1.0 if think_match.end() <= boxed_match.start() else 0.0
    return 0.0
```

### 4.3 Accuracy Reward (90% weight)

Extracts the answer from `\boxed{}` and compares it against the ground truth:

```python
def accuracy_reward_fn(sol: str, gt: str) -> float:
    from mathruler.grader import extract_boxed_content, grade_answer
    pred = extract_boxed_content(sol)
    return 1.0 if grade_answer(pred, gt) else 0.0
```

`mathruler.grader` handles numeric equivalence, fraction simplification, and other mathematical normalization.

### 4.4 Combined Reward

```python
final_reward = 0.9 * accuracy_reward + 0.1 * format_reward
```

### 4.5 Response Extraction

Before reward computation, the assistant's response is extracted from the full chat transcript to avoid false positives from system prompt examples:

```python
def extract_response(text: str) -> str:
    # Finds the last <|im_start|>assistant ... <|im_end|> segment
    ...
```

---

## 5. Training Monitoring (W&B)

### 5.1 Enable W&B Logging

Set the following in your training script:

```bash
export WANDB_API_KEY="your_api_key"
export WANDB_PROJECT="LightRFT-Experiments"
export WANDB_MODE="online"  # "offline" for local-only logging
```

### 5.2 Key Metrics

#### rollout/reward

![WandB rollout/reward curve](TODO:_INSERT_IMAGE_LINK_HERE)

**Expected trend**: The `rollout/reward` curve should show a steady upward trend over training steps. In early episodes, the reward typically starts low (around 0.1–0.3) as the model has not yet learned the correct format and reasoning patterns. As training progresses, the reward should gradually increase and stabilize around 0.7–0.9 for GSM8K and 0.5–0.8 for Geo3K. A healthy training run shows smooth, monotonic improvement with minor fluctuations. If the reward plateaus early or drops sharply, consider adjusting the KL coefficient or learning rate.

#### eval/accuracy

![WandB eval/accuracy curve](TODO:_INSERT_IMAGE_LINK_HERE)

**Expected trend**: The `eval/accuracy` curve reflects the model's actual problem-solving ability on the held-out test set. It should correlate with the reward curve but may lag slightly. For GSM8K (Qwen2.5-0.5B), expect accuracy to climb from the base model's level (~30–40%) toward 50–60%+ after full training. For Geo3K (Qwen2.5-VL-7B), accuracy typically improves from ~40% to 60–70%+. The eval curve is noisier than the reward curve due to smaller evaluation sample sizes. Sudden drops may indicate overfitting or KL divergence issues.

### 5.3 Additional Useful Metrics

| Metric | Description |
|--------|-------------|
| `rollout/format_reward` | Format compliance rate |
| `rollout/accuracy_reward` | Answer correctness rate |
| `train/actor_loss` | Actor policy loss (should decrease) |
| `train/kl_divergence` | KL divergence from reference policy (should stay bounded) |
| `train/entropy` | Policy entropy (gradual decrease indicates learning) |

---

## 6. Tips & Troubleshooting

- **OOM**: Reduce `micro_train_batch_size` / `micro_rollout_batch_size`, or lower `--engine_mem_util`.
- **Slow convergence**: Increase `--n_samples_per_prompt` for better GRPO advantage estimation.
- **Format reward stuck at 0**: Verify the system prompt is correctly passed via `--system_prompt` and `--apply_chat_template`.
- **Geo3K image loading errors**: Ensure the preprocessed parquet files contain valid PIL image objects and `--images_key images` is set.
- **LoRA training**: Use `--lora_rank 128 --lora_alpha 256` for a good balance between efficiency and capacity.

---

## Related Resources

- [Supported Algorithms](algorithms.md) — Full algorithm documentation
- [Configuration Reference](configuration.md) — Complete parameter reference
- [Training Strategy Guide](../best_practice/strategy_usage.rst) — FSDP, DeepSpeed, and engine configuration
