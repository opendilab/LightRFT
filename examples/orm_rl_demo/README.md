<div align="center">

# ORM RL Demo

Minimal example materials for ORM-based RL training in LightRFT.

</div>

## Overview

This example directory is being normalized toward a smaller, more generic `orm_rl_demo` naming scheme.

The current materials focus on:
- multimodal actor training with Qwen2.5-VL style models
- co-located outcome reward model scoring
- FSDP-based training and SGLang / vLLM generation backends
- lightweight trajectory saving for debugging

## Project Structure

```text
orm_rl_demo/
├── train_colocate.py
├── reward_models.py
├── reward_models_utils.py
├── test_reward_models.py
├── run_general_fsdp_qwenvl.sh
├── run_kg_fsdp_qwenvl.sh
└── run_fsdp_deepseek.sh
```

## Quick Start

The primary generic entrypoint in this directory is:

```bash
bash examples/orm_rl_demo/run_general_fsdp_qwenvl.sh
```

This script keeps the existing training flow intact while using the new generic naming.

## Environment

- Python >= 3.8
- CUDA >= 11.8 for GPU training
- 8x A100 (80GB) or similar hardware is recommended for the larger reward-model setups

## Notes

- The current files in this directory still preserve the existing training logic.
- This commit only normalizes naming and path references toward `orm_rl_demo`.
- Further scope reduction can happen later without changing the naming work done here.

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](../../LICENSE) for details.
