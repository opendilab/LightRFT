# GodsMeme GRPO Pipeline

This example adapts LightRFT's vision-language GRPO stack to meme generation.
The policy consumes the already-formatted GodsMeme RL prompt from the dataset, generates the full reasoning-style response, and the reward pipeline renders the final meme text back onto the image before running a pairwise meme judge.

## What is included

- `train_colocate.py`: main GRPO training entry for meme RL.
- `meme_dataset.py`: dataset loader for the prepared GodsMeme RL JSON/JSONL rows.
- `meme_utils.py`: GodsMeme-specific parsing, rendering, and pair aggregation helpers.
- `reward_model.py`: pairwise meme judge wrapper and reward aggregation logic.
- `prompts/generate_meme.txt`: reference policy prompt format.
- `prompts/reward_compare.txt`: pairwise comparison prompt for the meme reward judge.
- `run_meme_grpo.sh`: example launch script.
- `test_reward_model_vllm.py`: optional real-model integration smoke test using vLLM.

## Reward flow

For each rollout group produced from the same prompt:

1. Parse the policy completion and extract the `Text on the Meme` section.
2. Render the generated text onto the base image using the dataset's detection boxes when available.
3. Build pairwise comparisons inside the rollout group.
4. Ask a VLM judge which rendered meme is better.
5. Convert pairwise wins/scores into one scalar reward per sample.
6. Add a small format reward that checks the required GodsMeme response structure.

The reward weights are read from `--reward_pretrain` JSON config:

```text
final_reward = model_reward_weight * pairwise_reward
             + format_reward_weight * format_reward
```

Defaults:

- `model_reward_weight = 1.0`
- `format_reward_weight = 0.1`

## Important rollout constraint

The pairwise reward is computed inside each rollout micro-batch, so keep every GRPO group fully contained in one micro-batch:

```text
micro_rollout_batch_size % n_samples_per_prompt == 0
```

For standard GRPO runs, use `--advantage_estimator group_norm` and set `--n_samples_per_prompt > 1`.

## Dataset expectation

The RL dataset is expected to already be formatted as conversation-style GodsMeme rows:

```json
{
  "id": "sample-001",
  "image": "images/cat.jpg",
  "conversations": [
    {"from": "human", "value": "...GodsMeme prompt... <image>"},
    {"from": "assistant", "value": "...reference meme response..."}
  ],
  "text_loc_info": {
    "loc": [[40, 60, 420, 170], [40, 330, 420, 430]]
  }
}
```

Supported box metadata keys include `detections`, `text_loc_info`, `loc`, `bbox_scale`, and `bbox_normalized`.
If no boxes are available, the renderer falls back to a simple top/bottom banner layout.

## Reward model configuration

`--reward_pretrain` accepts either:

1. A plain Hugging Face model path for the pairwise judge.
2. A JSON object if you want to override judge settings.

Plain path example:

```bash
--reward_pretrain /path/to/Qwen2.5-VL-7B-Instruct
```

JSON example:

```bash
--reward_pretrain '{
  "pairwise": {
    "path": "/path/to/Qwen2.5-VL-7B-Instruct",
    "max_new_tokens": 96,
    "pair_batch_size": 4,
    "max_pairs_per_group": 0,
    "model_reward_weight": 1.0,
    "format_reward_weight": 0.1,
    "reward_prompt_path": "examples/godsmeme/prompts/reward_compare.txt"
  }
}'
```

## Running

Edit the environment variables in `run_meme_grpo.sh`, then launch:

```bash
bash examples/godsmeme/run_meme_grpo.sh
```

## Validation

Lightweight unit tests:

```bash
pytest examples/godsmeme/test_meme_dataset.py examples/godsmeme/test_reward_model.py
```

Optional real-model vLLM smoke test:

```bash
RUN_GODSMEME_VLLM_TEST=1 \
GODSMEME_REWARD_MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct \
GODSMEME_ANNOTATION_PATH=/path/to/train_data.jsonl \
GODSMEME_IMAGE_ROOT=/path/to/image_root \
pytest examples/godsmeme/test_reward_model_vllm.py -k real_data
```
