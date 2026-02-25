# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About LightRFT

LightRFT is a reinforcement fine-tuning (RFT) framework for LLMs and VLMs, built on top of [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). It supports GRPO, GSPO, GMPO, Dr.GRPO, DAPO, REINFORCE++, CPGD, and FIRE Sampling algorithms, with vLLM/SGLang inference engines and FSDP/DeepSpeed training strategies.

## Common Commands

### Installation
```bash
pip install -r requirements.txt
pip install -e .
pip install -r requirements-dev.txt  # for linting/formatting
```

### Code Formatting & Linting
```bash
make format    # YAPF formatting (line length 120)
make fcheck    # Flake8 linting
```

### Documentation
```bash
make docs       # Build Sphinx HTML docs → docs/build/index.html
make docs-live  # Live preview at http://localhost:8000
```

### Running Tests
Tests are in `lightrft/models/tests/` and use pytest:
```bash
python -m pytest lightrft/models/tests/test_actor_language.py
python -m pytest lightrft/models/tests/test_actor_vl.py
python -m pytest lightrft/models/tests/test_actorvl_fused_linear_logprob.py
```

### Running a Training Example
```bash
# Preprocess dataset first (example for GSM8K):
python examples/data_preprocess/gsm8k_lightrft.py --local_save_dir /path/to/output

# Then launch training (8 GPUs, single node):
bash examples/gsm8k_geo3k/run_grpo_gsm8k_qwen2.5_0.5b.sh

# VLM example (Geo3K):
bash examples/gsm8k_geo3k/run_grpo_geo3k_qwen2.5_vl_7b.sh
```

Training is launched via `torchrun` with `train_colocate.py` inside each example directory.

### Docker
```bash
make dbuild   # Builds opendilab/lightrft:v<VERSION>
make dpush    # Pushes to Docker Hub
```

## Architecture Overview

### Core Training Flow

The typical RLHF training loop works as follows:
1. **Experience generation** (`FastExperienceMaker`): runs policy model via vLLM/SGLang, scores with reward models, computes advantages.
2. **Training** (`SPMDPPOTrainer`): updates the actor using PPO-style policy loss; optionally co-locates reward models on the same GPUs.
3. **Strategy** (`DeepspeedStrategy` / `FSDPV2Strategy`): wraps models for distributed training with a uniform API.

### Key Modules

**`lightrft/trainer/`** — Core training logic:
- `spmd_ppo_trainer.py`: The primary trainer (`SPMDPPOTrainer`, `SPMDPPOTrainerVL`). Extends `PPOTrainer` with SPMD/tensor-parallel support. This is the "entry point" for understanding how training works end-to-end.
- `fast_exp_maker.py`: `FastExperienceMaker` — handles rollout generation via vLLM/SGLang, reward aggregation, and advantage computation. The `generate_samples()` and `_get_return_advs()` methods are the main algorithm extension points.
- `advantage_calculator.py`: Pluggable advantage estimators (GAE, Group Norm/GRPO, RLOO, REINFORCE++, CPGD). Use `get_advantage_calculator()` factory.
- `ppo_trainer.py` / `ppo_trainer_vl.py`: Base PPO trainer (ABC) for LLM and VLM respectively.
- `experience_maker.py`: `NaiveExperienceMaker` base class; `FastExperienceMaker` inherits from this.
- `replay_buffer.py` / `replay_buffer_vl.py`: Experience replay buffers with packing support.

**`lightrft/strategy/`** — Distributed training abstraction:
- `strategy_base.py`: `StrategyBase` ABC with `backward()`, `optimizer_step()`, `save_ckpt()` API.
- `strategy.py`: `get_strategy(args)` factory — picks DeepSpeed or FSDP based on `args.fsdp`.
- `deepspeed/deepspeed.py`: DeepSpeed ZeRO (Stage 1/2/3) strategy.
- `fsdp/fsdpv2.py`: FSDP v2 strategy.
- `config.py`: `StrategyConfig` dataclass (typed access to all strategy params; use `StrategyConfig.from_args(args)` to construct).
- `fake_strategy.py`: `FakeStrategy` for single-process unit testing without distributed setup.

**`lightrft/models/`** — Model wrappers:
- `actor_language.py`: LLM actor wrapping HuggingFace causal LM.
- `actor_vl.py` / `actor_al.py`: VLM and audio-LM actors.
- `actor_modality.py`: `ActorModality` base that both LLM and VLM actors extend.
- `loss.py`: `PolicyLoss` (PPO/GSPO/GMPO/Dr.GRPO/DAPO/Token-Level Policy variants all flow through here), `ValueLoss`, `GPTLMLoss`.
- `srm_vl.py` / `srm_al.py`: Scalar reward model wrappers.
- `grm_vl.py`: Generative reward model (VLM).
- `monkey_patch/`: Patches for distributed training compatibility.

**`lightrft/datasets/`** — Dataset handlers, each implementing a dataset-specific preprocessing interface. `prompts_dataset.py` (LLM) and `prompts_dataset_vl.py` (VLM) are the main training datasets; others are for reward model training.

**`lightrft/utils/`**:
- `cli_args.py`: `add_arguments()` adds engine/FSDP/logging CLI args to any `argparse.ArgumentParser`.
- `remote_rm_utils.py`: Utilities for calling remote reward model HTTP APIs.
- `trajectory_saver.py`: Saves rollout trajectories for analysis.
- `processor.py`: HuggingFace tokenizer/processor wrapper.

### Algorithm Extension Points

| To change... | Edit... |
|---|---|
| Policy loss objective (GRPO/GSPO/GMPO/DAPO/Dr.GRPO) | `lightrft/models/loss.py` → `PolicyLoss.forward()` |
| Advantage estimation method | `lightrft/trainer/advantage_calculator.py` |
| Rollout generation / FIRE sampling | `lightrft/trainer/fast_exp_maker.py` → `generate_samples()` |
| Reward aggregation / normalization | `lightrft/trainer/fast_exp_maker.py` → `_get_return_advs()` |
| Distributed training backend | `lightrft/strategy/` |

### Training Entry Points (examples)

Each example has its own `train_colocate.py`. They share the same general structure:
1. Parse args via `argparse` + `lightrft.utils.cli_args.add_arguments`
2. Build strategy via `get_strategy(args)`
3. Load actor, reference model, reward models
4. Instantiate `SPMDPPOTrainer` (or VL variant)
5. Call `trainer.fit()`

## Commit Style

Follow Conventional Commits: `type(scope): description`
- Common types: `feature`, `fix`, `polish`, `docs`, `style`, `refactor`
- Example: `feature(trainer): add CPGD advantage estimator`

## PR Checklist

Before opening a PR, run:
```bash
make format
make fcheck
```
