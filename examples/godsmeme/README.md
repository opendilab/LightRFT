# GodsMeme on LightRFT

This example adapts LightRFT's vision-language GRPO pipeline to meme generation.
The policy sees one source image plus a preformatted GodsMeme prompt, generates a full reasoning-style answer, and is rewarded by a pairwise meme judge that compares rendered meme images within each GRPO group.

## What the training entrypoint actually is

There are two entry layers:

- `examples/godsmeme/run_meme_grpo.sh`: user-facing launcher. Set paths and training knobs here, then run it with `bash`.
- `examples/godsmeme/train_colocate.py`: Python training entry used by `torchrun`. It builds the actor, dataset, reward model, inference engine, and PPO trainer.

If you only want to start training, edit the shell script or override its environment variables. You usually do not need to call `train_colocate.py` directly.

## End-to-end training flow

1. `MemeOnlineRLDataset` loads prebuilt RL rows from `annotation_path` and resolves the source image from `root_dir`.
2. Each row is converted into a multimodal chat prompt with one image placeholder and one text instruction.
3. GRPO samples `n_samples_per_prompt` policy completions for the same prompt.
4. The reward pipeline extracts the `Text on the Meme` section from each completion.
5. The extracted text is rendered back onto the original image by using detection boxes when available, or a simple fallback layout otherwise.
6. A local VLM judge compares candidate meme images pairwise inside the same rollout group.
7. Pairwise scores are aggregated into one scalar reward per sample.
8. A small format reward is added so the policy keeps the expected GodsMeme response structure.
9. LightRFT runs PPO/GRPO updates and saves checkpoints to the configured output directory.

## Directory map

```text
examples/godsmeme/
├── README.md                    # This guide
├── run_meme_grpo.sh             # User-facing launch script
├── train_colocate.py            # Main GRPO training entry
├── meme_dataset.py              # Dataset loader for GodsMeme RL rows
├── reward_model.py              # Pairwise reward judge and reward aggregation
├── meme_utils.py                # Text parsing, rendering, and pair helpers
├── prompts/
│   ├── generate_meme.txt        # Reference policy prompt format
│   └── reward_compare.txt       # Prompt template for the pairwise judge
├── test_meme_dataset.py         # Dataset test scaffold
└── test_reward_model_vllm.py    # Optional reward-model smoke test
```

## Dataset format expected by this example

This example does not build the RL dataset for you. It expects a JSON or JSONL file where each row already looks like a conversation-style GodsMeme training sample.

Minimal example:

```json
{
  "id": "sample-001",
  "image": "images/cat.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "...GodsMeme prompt... <image>"
    },
    {
      "from": "assistant",
      "value": "...reference reasoning and meme text..."
    }
  ],
  "text_loc_info": {
    "loc": [[40, 60, 420, 170], [40, 330, 420, 430]]
  }
}
```

Useful notes:

- `image`, `image_path`, and `img` are accepted as image keys.
- The human message must contain `<image>`.
- The assistant message is treated as the reference output and is also used to infer the expected number of text boxes when box metadata is missing.
- Supported box metadata includes `detections`, `text_loc_info`, `loc`, `bbox_scale`, `bbox_normalized`, and `expected_box_count`.
- If no boxes are available, the renderer falls back to a simple top/bottom style layout.

## Reward design

The reward combines two parts:

```text
final_reward = model_reward_weight * pairwise_reward
             + format_reward_weight * format_reward
```

- `pairwise_reward`: computed by comparing rendered candidate memes from the same rollout group.
- `format_reward`: checks whether the completion keeps the expected GodsMeme answer structure and box count.

Default weights:

- `model_reward_weight = 1.0`
- `format_reward_weight = 0.1`

The default launcher builds `--reward_pretrain` as a JSON blob that points to the local judge model and the comparison prompt template.

The current implementation follows the `HUMOR-RM-Keye-VL` inference pattern directly in `reward_model.py`, loading the Keye-based reward model plus `classification_head.pt` without adding a `llamafactory` runtime dependency. The pairwise judge prompt is kept to the same simple question used in the model README: `Which meme is funnier?`

## Quick start

Edit the paths in `examples/godsmeme/run_meme_grpo.sh`, or override them inline:

```bash
POLICY_MODEL_PATH=/path/to/policy-model \
REWARD_MODEL_PATH=/path/to/reward-model \
ANNOTATION_PATH=/path/to/train_data.jsonl \
IMAGE_ROOT=/path/to/image_root \
bash examples/godsmeme/run_meme_grpo.sh
```

Default outputs:

- checkpoints: `results/<experiment_name>/<run_name>/`
- logs: `rft_logs/<experiment_name>/`

## Important constraints before you launch

- `N_SAMPLES` must be greater than `1` for GRPO with `group_norm`.
- `MICRO_ROLLOUT_BATCH_SIZE % N_SAMPLES == 0` must hold, otherwise one prompt group can be split across micro-batches and pairwise reward becomes invalid.
- Pairwise judge cost grows roughly quadratically with `N_SAMPLES` unless you cap `MAX_PAIRS_PER_GROUP`.
- The current meme reward model does not support `--rm_use_engine`; the judge is loaded directly by `reward_model.py`.
- Each policy prompt uses one source image, so `LIMIT_MM_IMAGE_PER_PROMPT` is set to `1`.
- This example reads `--annotation_path` and `--root_dir`; it does not use the generic `--prompt_data` path for training data.

## Most useful knobs in `run_meme_grpo.sh`

- `POLICY_MODEL_PATH`: actor checkpoint or HF model id.
- `REWARD_MODEL_PATH`: Keye-based reward-model checkpoint path. It should contain `classification_head.pt`.
- `REWARD_MAX_LENGTH`: max sequence length used when scoring rendered meme pairs.
- `ANNOTATION_PATH` / `IMAGE_ROOT`: GodsMeme RL data location.
- `N_SAMPLES`: number of rollouts per prompt. Higher is better for ranking quality but much slower.
- `MAX_PAIRS_PER_GROUP`: limit pairwise comparisons to reduce reward cost.
- `PAIR_BATCH_SIZE`: judge batch size for pairwise evaluation.
- `MICRO_ROLLOUT_BATCH_SIZE`: must be divisible by `N_SAMPLES`.
- `MICRO_TRAIN_BATCH_SIZE`: lower this first if actor training OOMs.
- `ENGINE_TYPE`, `ENGINE_TP`, `ENGINE_MEM_UTIL`: rollout engine settings for the actor.
- `PROMPT_MAX_LEN`, `GENERATE_MAX_LEN`: trim these if prompts or responses are too long for your setup.

## Practical tuning advice

- Start with smaller `N_SAMPLES` such as `4` before scaling to `8`.
- If reward evaluation is the bottleneck, reduce `N_SAMPLES` or set `MAX_PAIRS_PER_GROUP` to a small positive integer.
- If the actor OOMs, first lower `MICRO_TRAIN_BATCH_SIZE`, then `MICRO_ROLLOUT_BATCH_SIZE`, then `ENGINE_MEM_UTIL`.
- If generations are too verbose, reduce `GENERATE_MAX_LEN` and revisit the prompt format in `examples/godsmeme/prompts/generate_meme.txt`.
- If rendered text looks misplaced, verify your box annotations before changing the reward model.

## Optional validation

There are lightweight unit tests for the new reward-model invocation path:

```bash
pytest examples/godsmeme/test_reward_model_vllm.py
```

These tests cover the classifier-head loading helpers and the direct pair-scoring flow without requiring a real reward checkpoint.

## Current limitations

- No data preprocessing script is included in this folder; the RL JSON/JSONL must already be prepared.
- The reward judge is model-based and local, so training cost is much higher than rule-only GRPO examples.
- The reward loader expects the Keye multimodal interface used by `HUMOR-RM-Keye-VL`.
- The fallback renderer is intentionally simple and is mainly a reward-time approximation, not a production meme compositor.
- The policy is optimized for the GodsMeme response template; changing the prompt schema usually requires updating both parsing and reward logic.
