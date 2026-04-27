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
- `vllm` / `sglang` support for URSA is tracked separately in the migration docs; the active Stage 3 path is the local `hf` rollout path.

## Directory Map

```text
examples/math_prm/
├── README.md                    # English guide for the current URSA-MATH Stage 3 layout
├── README_zh.md                 # Chinese guide
├── URSA_MIGRATION.md            # Temporary migration notes from the original URSA-MATH repo
├── train_colocate.py            # Main LightRFT training entry
├── run_grpo_math_prm_ursa_8b.sh # Main Stage 3 launcher
├── ursa_actor.py                # URSA-specific actor wrapper
├── reward_models.py             # Math-only URSA-RM reward implementation
├── reward_models_utils.py       # Math-only reward loading, recipe, and reward aggregation
├── sitecustomize.py             # Local runtime compatibility hook for this example stack
├── tools/                       # Support scripts, regression checks, smoke runs, and observation tools
│   ├── __init__.py
│   ├── prepare_ursa_stage3_manifest.py
│   ├── prepare_ursa_engine_checkpoint.py
│   ├── prm_infer_score.py
│   ├── check_phase2_alignment.py
│   ├── check_hf_rollout.py
│   ├── check_phase6_script_alignment.py
│   ├── test_phase2_alignment.py
│   ├── run_phase3_smoke.sh
│   ├── run_phase7_observation.sh
│   ├── analyze_phase7_observation.py
│   └── probe_rollout_speed_candidates.py
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

### Data and compatibility tools

- `tools/prepare_ursa_stage3_manifest.py`
  - Converts raw `MMathCoT-1M` Stage 3 jsonl into the LightRFT manifest schema.
- `tools/prepare_ursa_engine_checkpoint.py`
  - Builds a wrapper checkpoint for engine experiments when testing `vllm` / `sglang` loading.
- `tools/prm_infer_score.py`
  - Standalone PRM helper mirrored from URSA-MATH reference logic.

### Regression and validation tools

- `tools/check_phase2_alignment.py`
  - Checks scorer parity against the URSA reference path.
- `tools/check_hf_rollout.py`
  - Minimal local `hf` rollout validation.
- `tools/check_phase6_script_alignment.py`
  - Static checker for current launcher defaults.
- `tools/test_phase2_alignment.py`
  - Regression tests for the active URSA-MATH Stage 3 path.

### Smoke, observation, and profiling

- `tools/run_phase3_smoke.sh`
  - Time-boxed smoke launcher for early-stage training validation.
- `tools/run_phase7_observation.sh`
  - Bounded full-data observation launcher.
- `tools/analyze_phase7_observation.py`
  - Offline analyzer for saved trajectories and observation logs.
- `tools/probe_rollout_speed_candidates.py`
  - Minimal speed probe used to compare rollout-like decode modes without modifying `lightrft/`.

## Active Entry Points

If you only want the current Stage 3 reproduction path, the usual files are:

- `run_grpo_math_prm_ursa_8b.sh`
- `train_colocate.py`
- `reward_models.py`
- `reward_models_utils.py`
- `tools/prepare_ursa_stage3_manifest.py`
- `tools/check_hf_rollout.py`
- `tools/test_phase2_alignment.py`

## Temporary Working Docs

Two kinds of documents still exist only to support the current migration/debugging cycle and are expected to be removed after the work is fully concluded:

- `examples/math_prm/URSA_MIGRATION.md`
  - Temporary migration notes from the original URSA-MATH repo into LightRFT.
- `/data/LightRFT/plan/*`
  - Working notes, phase tracking, failure analyses, and profiling investigations created during the migration.

These are intentionally kept outside the long-term stable training surface. Once the migration is fully closed out and the conclusions have been folded into permanent docs or code comments, they should be deleted.

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

- Validate the local `hf` rollout path:

```bash
python examples/math_prm/tools/check_hf_rollout.py
```

- Run regressions:

```bash
python -m unittest -q examples.math_prm.tools.test_phase2_alignment
```

- Run the Phase 3 smoke script:

```bash
bash examples/math_prm/tools/run_phase3_smoke.sh
```
