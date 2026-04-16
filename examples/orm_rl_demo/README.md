<div align="center">

# ORM RL Demo

Minimal Geo3K-oriented ORM RL demo for LightRFT.

</div>

## Overview

This example is scoped to one runnable path for clarifying the existing ORM RL training flow:
- dataset: Geo3K
- actor: Qwen2.5-VL 7B actor checkpoint
- reward side: one general outcome reward model path
- backend: FSDP training with engine-based reward inference

## Project Structure

```text
orm_rl_demo/
├── train_colocate.py
├── reward_models.py
├── reward_models_utils.py
├── test_reward_models.py
└── run_general_fsdp_qwenvl.sh
```

## Quick Start

The only entry script kept for this demo is:

```bash
export DATA_PATH=/path/to/geo3k
export PRETRAIN_PATH=/path/to/Qwen2.5-VL-7B-Instruct
export REWARD_PRETRAIN_PATHS='{"general":"/path/to/general-reward-model"}'
bash examples/orm_rl_demo/run_general_fsdp_qwenvl.sh
```

The script is a template and does not hardcode cluster-specific or personal paths. Set the dataset and model paths explicitly before running it.

## Demo Flow

This demo is intended to make the ORM RL pipeline easier to inspect:
- the actor generates Geo3K trajectories
- the general ORM path scores those trajectories
- trajectory saving stays enabled for debugging and flow inspection

To avoid rewriting the existing Geo3K dataset files, the demo overrides the dataset label to `geo3k_general` at runtime so the samples are routed through the demo's general-ORM reward mix while keeping the original dataset path unchanged.

## Environment

Environment requirements stay aligned with the repository-level [README_zh.md](../../README_zh.md#环境要求). Refer to the main project document instead of duplicating version constraints here.

## Notes

- The demo intentionally keeps a single shell entrypoint.
- Geo3K reward routing is handled through runtime label override instead of rewriting the dataset itself.
- Runtime paths are provided via environment variables so the example can stay free of cluster-specific or personal information.

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](../../LICENSE) for details.
