<div align="center">

# Math PRM Training in LightRFT

URSA-MATH Stage 3 reproduction workspace for LightRFT.

</div>

## Scope

This directory is no longer a generic multimodal reward example. It now only keeps the files that are still relevant to the URSA-MATH Stage 3 migration and reproduction path.

Current target:

- actor: `URSA-8B`
- reward model: `URSA-RM-8B`
- reward labels: `math_prm`, `math_psgrpo`, `math_prm_combined`, `math_rule`
- training loop: LightRFT PPO/GRPO stack with local `hf` rollout
- raw dataset: `MMathCoT-1M`

## Runtime Baseline

The runtime baseline is frozen by `/data/LightRFT/Dockerfile`.

- Do not treat package-version changes as the first-line fix.
- Prefer fixing code, schema conversion, prompt formatting, rollout configuration, and reward wiring first.
- The active Stage 3 path in this branch is the local `hf` rollout path; `vllm` / `sglang` experiments are optional and go through the engine-wrapper helper only.

## Directory Map

```text
examples/math_prm/
├── README.md                    # English guide for the current URSA-MATH Stage 3 layout
├── README_zh.md                 # Chinese guide
├── train_colocate.py            # Main LightRFT training entry
├── math_prm_trainer.py          # Example-local trainer wrapper for reduced W&B keys and runtime eval
├── run_grpo_math_prm_ursa_8b.sh # Main Stage 3 launcher
├── ursa_actor.py                # URSA-specific actor wrapper
├── reward_models.py             # Math-only URSA-RM reward implementation
├── reward_models_utils.py       # Math-only reward loading, recipe, and reward aggregation
├── sitecustomize.py             # Local runtime compatibility hook for this example stack
├── tools/                       # Support scripts kept in the slim PR branch
│   ├── __init__.py
│   ├── prepare_ursa_stage3_manifest.py
│   └── prepare_ursa_engine_checkpoint.py
└── ursa_model/                  # Self-contained URSA model code used by actor and PRM loading
```

## What Each Top-Level File Does

### Core training path

- `run_grpo_math_prm_ursa_8b.sh`
  - Main launcher for Stage 3 reproduction.
  - Wires actor path, reward path, dataset path, FSDP setup, rollout settings, and optional W&B.
- `train_colocate.py`
  - Real `torchrun` entry.
  - Builds actor, reference model, reward model, dataset, trainer, and rollout engine.
- `math_prm_trainer.py`
  - Example-local trainer wrapper for math PRM runs.
  - Keeps rollout/train/eval W&B metrics compact and applies runtime eval generation defaults.
- `ursa_actor.py`
  - URSA-specific actor wrapper used to load `UrsaForConditionalGeneration`.

### Reward path

- `reward_models.py`
  - Contains the active `MathPRMReward` implementation only.
  - This file has been trimmed to the URSA-MATH Stage 3 path and no longer carries the old Qwen/SafeWork reward classes.
- `reward_models_utils.py`
  - Contains the active math-only reward loader and recipe logic.
  - Handles `math_prm`, `math_psgrpo`, `math_prm_combined`, and `math_rule`.
- `sitecustomize.py`
  - Local import/runtime compatibility shim for the frozen example environment.

### Self-contained URSA runtime

- `ursa_model/`
  - Local URSA config, processor, image processor, projector, vision towers, and model definitions.
  - This is what lets the current LightRFT path run without importing runtime code directly from the external URSA-MATH repo.

## What Lives Under `tools/`

Everything under `tools/` is support infrastructure, not the main training entry.

- `tools/prepare_ursa_stage3_manifest.py`
  - Converts raw `MMathCoT-1M` Stage 3 jsonl into the LightRFT manifest schema.
- `tools/prepare_ursa_engine_checkpoint.py`
  - Builds a wrapper checkpoint for engine experiments when testing `vllm` / `sglang` loading.

## Active Entry Points

If you only want the current Stage 3 reproduction path, the usual files are:

- `run_grpo_math_prm_ursa_8b.sh`
- `train_colocate.py`
- `math_prm_trainer.py`
- `reward_models.py`
- `reward_models_utils.py`
- `tools/prepare_ursa_stage3_manifest.py`
- `tools/prepare_ursa_engine_checkpoint.py`

## Local Resources

Current machine layout:

```bash
URSA actor:      /home/ubuntu/URSA-MATH/checkpoints/URSA-8B
URSA reward:     /home/ubuntu/URSA-MATH/checkpoints/URSA-RM-8B
MMathCoT-1M raw: /home/ubuntu/URSA-MATH/datasets/URSA-MATH/MMathCoT-1M/train.jsonl
Image root:      /home/ubuntu/URSA-MATH/datasets/URSA-MATH/images
```

Current converted manifest:

```bash
/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_psgrpo.jsonl
```

Current converted manifest summary:

```bash
/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_psgrpo.summary.json
```

## Dataset Preparation

The raw Stage 3 data is not directly consumable by `PromptDatasetVL`.

Raw schema:

```json
{
  "image_url": "...",
  "instruction": "...",
  "output": "..."
}
```

Converted LightRFT schema:

```json
{
  "prompt": "...",
  "images": ["/abs/path/to/image.png"],
  "reference": "...",
  "label": "math_psgrpo"
}
```

Run a smoke conversion:

```bash
python examples/math_prm/tools/prepare_ursa_stage3_manifest.py \
  --max-samples 32 \
  --output-path /data/LightRFT/tmp/ursa_stage3/smoke_manifest.jsonl \
  --summary-path /data/LightRFT/tmp/ursa_stage3/smoke_manifest.summary.json
```

Run the default conversion:

```bash
python examples/math_prm/tools/prepare_ursa_stage3_manifest.py
```

## Training

Expected current-machine values in `examples/math_prm/run_grpo_math_prm_ursa_8b.sh`:

```bash
PATH_TO_YOUR_BASE_MODEL="/home/ubuntu/URSA-MATH/checkpoints/URSA-8B"
PATH_TO_URSA_RM="/home/ubuntu/URSA-MATH/checkpoints/URSA-RM-8B"
PATH_TO_YOUR_MATH_DATASET="/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_psgrpo.jsonl"
EXPECTED_REWARD_LABEL="math_psgrpo"
```

Run training:

```bash
bash examples/math_prm/run_grpo_math_prm_ursa_8b.sh
```

Current default launcher values now follow the explicit Stage 3 settings documented in the local `URSA-MATH` repo where available:

```bash
EPISODE=10
N_SAMPLES=8
RBS=128
TBS=128
MICRO_TRAIN_BATCH_SIZE=4
MICRO_ROLLOUT_BATCH_SIZE=4
LR=1e-6
KL=0.001
PROMPT_MAX_LEN=1024
GENERATE_MAX_LEN=3072
MAX_SAMPLES=15360
```

Notes:

- The paper reports a one-time filtered `20K -> ~15K+` RL set. The exact filtered subset is not present locally, so the launcher keeps the converted manifest path and uses `MAX_SAMPLES=15360` as a scale proxy.
- The paper's default hardware is `32 x H100`; the current machine default remains `1 node x 8 A100`.

## Reward Labels

- `math_prm`
  - Pure PRM reward using `min(step_scores)`.
- `math_psgrpo`
  - PS-GRPO reward computed inside `MathPRMReward`.
- `math_prm_combined`
  - PRM plus explicit rule baseline.
- `math_rule`
  - Rule-only ablation baseline.

## Troubleshooting Shortcuts

- Rebuild the manifest:

```bash
python examples/math_prm/tools/prepare_ursa_stage3_manifest.py
```

- Build the engine-wrapper checkpoint for non-`hf` experiments:

```bash
python examples/math_prm/tools/prepare_ursa_engine_checkpoint.py \
  --source-model-path /path/to/URSA-8B \
  --output-path /path/to/URSA-8B-engine-ready
```
