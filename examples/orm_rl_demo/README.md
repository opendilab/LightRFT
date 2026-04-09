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
bash examples/orm_rl_demo/run_general_fsdp_qwenvl.sh
```

The script keeps the existing cluster-ready path style and reuses the current Qwen-VL actor / reward-model locations already referenced in this repo.

## Demo Flow

This demo is intended to make the ORM RL pipeline easier to inspect:
- the actor generates Geo3K trajectories
- the general ORM path scores those trajectories
- trajectory saving stays enabled for debugging and flow inspection

To avoid rewriting the existing Geo3K dataset files, the demo overrides the dataset label to `general` at runtime so the samples are routed through the general ORM reward recipe while keeping the original dataset path unchanged.

## Environment

- Python >= 3.8
- CUDA >= 11.8 for GPU training
- 8x A100 (80GB) or similar hardware is recommended for the 72B reward-model setup

## Notes

- The demo intentionally keeps a single shell entrypoint.
- Geo3K reward routing is handled through runtime label override instead of rewriting the dataset itself.
- The current reward-model path is left in the existing cluster-ready style already used by this example directory.

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](../../LICENSE) for details.
