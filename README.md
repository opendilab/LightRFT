# LightRFT

<div align="center">

<img src="assets/logo.png" alt="LightRFT Logo" width="600"/>

**Light, Efficient, Omni-modal & Reward-model Driven Reinforcement Fine-Tuning Framework**

[![Version](https://img.shields.io/badge/version-0.1.1-blue.svg)](https://github.com/opendilab/LightRFT)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

English | [简体中文](README_zh.md)

</div>

## Overview

LightRFT (Light Reinforcement Fine-Tuning) is a reinforcement fine-tuning framework for large language models (LLMs) and vision-language or multimodal generative models (VLMs). It provides a structured and extensible workflow for reinforcement learning with verifiable rewards (RLVR), reinforcement learning from human feedback (RLHF), and model-reward-driven policy optimization, covering policy sampling, reward computation, advantage estimation, and policy updates. The repository also includes reward-model training and on-policy distillation workflows.

LightRFT uses `torchrun` and PyTorch distributed communication as its runtime foundation. A unified Strategy interface connects FSDP v2 and DeepSpeed ZeRO training backends with SGLang and vLLM rollout backends. Current code paths and examples include text, image, video, and audio tasks. “Omni-modal” means that the repository contains dedicated model, data, or example paths for these modalities; it does not imply that every model and modality combination works without adaptation.

> This document describes repository version `0.1.1`. Source code and runnable examples define the implemented feature boundary; roadmap items are not treated as released features.

## Contents

- [Design highlights](#design-highlights)
- [Supported algorithm matrix](#supported-algorithm-matrix)
- [Runtime architecture](#runtime-architecture)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Examples and applications](#examples-and-applications)
- [Monitoring, trajectories, and checkpoints](#monitoring-trajectories-and-checkpoints)
- [Repository layout](#repository-layout)
- [Documentation and troubleshooting](#documentation-and-troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Citation, license, and acknowledgements](#citation-license-and-acknowledgements)

## Design highlights

### Unified Strategy abstraction and training–rollout loop

- A common interface connects DeepSpeed/FSDP v2 training backends with SGLang/vLLM rollout backends, reducing backend-specific coupling in the upper-level training workflow.
- Distributed jobs follow a single-program, multiple-data (SPMD) topology launched with `torchrun`; Ray is not required for scheduling, and standard PyTorch distributed tools remain applicable for debugging.
- Training and rollout reuse the same GPU process set by phase. The rollout engine can sleep during policy updates and receives refreshed Actor weights before the next rollout.

LightRFT calls this logical colocation and phase-oriented resource sharing model **Colocate Anything**. See [Runtime Architecture and Resource Reuse](docs/source/best_practice/runtime_architecture.md) for Strategy boundaries, evaluation flow, model placement, and weight synchronization.

### Distributed and parameter-efficient training

- FSDP v2 and DeepSpeed ZeRO stages 1/2/3; DeepSpeed is selected when `--fsdp` is absent.
- BF16, gradient checkpointing, Adam offload, and FSDP CPU offload.
- LoRA, visual-prefix freezing, and sample packing.
- Optional FlashAttention 2 and a fused log-probability path.

### Reward-model-driven optimization

- Rule rewards, custom reward functions, local reward models, and remote reward services.
- Multiple reward sources with task-specific aggregation.
- Training entry points for vision scalar reward models (SRM), vision generative reward models (GRM), and audio SRMs.
- On-policy distillation (OPD) with teacher log-probabilities, either as a distillation-only objective or combined with task rewards.

### Multimodal task paths

- Text (`ActorLanguage`), vision-language (`ActorVL`), and audio-language (`ActorAL`) policy paths.
- Image inputs are handled by the vision-language path; the experience-generation path also handles video fields.
- Examples cover GSM8K text reasoning, Geo3K visual geometry reasoning, video reward-model RL, and audio question answering.

### Experiment tooling

- Weights & Biases and TensorBoard logging.
- Trajectory saving and analysis for repetition, reflection patterns, and policy entropy.
- High-entropy-token annotation and local visualization.
- Distributed checkpoints, optional Hugging Face checkpoints, and conversion utilities.

## Supported algorithm matrix

LightRFT organizes policy optimization, advantage estimation, sampling, and knowledge distillation as composable modules. See the [algorithm guide](docs/source/quick_start/algorithms.md) for principles and detailed configuration.

| Algorithm | Type | Main improvement | Current implementation and entry point | Reference |
|-----------|------|------------------|----------------------------------------|-----------|
| **GRPO** | Policy Optimization | Group-normalized advantage estimation | **Supported**: use `--advantage_estimator group_norm`; requires multiple responses per prompt | [arXiv:2402.03300](https://arxiv.org/pdf/2402.03300) |
| **GSPO (WIP)** | Policy Optimization | Group sequence policy optimization | **Experimental interface**: `--use_gspo` and related options are available while integration is in progress | [arXiv:2507.18071](https://arxiv.org/abs/2507.18071) |
| **GMPO (WIP)** | Policy Optimization | Geometric-mean policy optimization | **In development**: the end-to-end training path is being completed | [arXiv:2507.20673](https://arxiv.org/abs/2507.20673) |
| **Dr.GRPO** | Policy Optimization | Mitigation of length bias | **Supported**: unbiased group-relative optimization reduces length bias and improves token efficiency | [arXiv:2503.20783](https://arxiv.org/abs/2503.20783) |
| **REINFORCE++** | Advantage Estimation | Improved baseline estimation | **Supported**: use `--advantage_estimator reinforce++` for return and advantage estimation | [arXiv:2501.03262](https://arxiv.org/abs/2501.03262) |
| **DAPO** | Policy Optimization | Decoupled clipping and dynamic sampling | **Supported**: includes `--dynamic_sampling`, `--overlong_buffer`, and related training mechanisms | [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) |
| **CPGD** | Advantage Estimation | KL-drift constraint | **Supported**: use `--advantage_estimator cpgd`; `--use_cpg_loss` enables asymmetric clipping | [arXiv:2505.12504](https://arxiv.org/abs/2505.12504) |
| **FIRE Sampling** | Sampling Strategy | High-temperature first-token sampling for greater diversity | **Supported**: configure with `--use_fire` and `--first_token_temperature` | [arXiv:2410.21236](https://arxiv.org/abs/2410.21236) |
| **OPD** | Knowledge Distillation | On-policy teacher–student token-level distillation | **Supported**: reads teacher log-probabilities from `--teacher_model_url` and supports pure or task-reward-hybrid distillation | [Blog](https://thinkingmachines.ai/blog/on-policy-distillation/) |

The main training entry point, `examples/gsm8k_geo3k/train_colocate.py`, also provides the following foundational training paths:

| Method | `--advantage_estimator` | Critic | Description |
|--------|-------------------------|--------|-------------|
| PPO / GAE | `gae` | Required | Computes GAE from value estimates and trains with a value loss |
| REINFORCE | `reinforce` | No | Builds token-level returns from sequence rewards |
| RLOO | `rloo` | No | Uses a leave-one-out group baseline and requires multiple responses per prompt |
| REINFORCE with baseline | `reinforce_baseline` | No | Uses the group mean as the baseline without standard-deviation scaling |

These training paths can be combined with the following stability and efficiency mechanisms:

- **Sample filtering and length control**: `--dynamic_sampling` masks groups with no reward variation, while `--overlong_buffer` adds a length-dependent penalty to overlong responses.
- **[Token-level updates](https://arxiv.org/abs/2506.01939)**: `--high_entropy_token_ratio` restricts policy-gradient updates to a selected fraction of high-entropy tokens; `0.0` disables filtering.
- **Numerical stability**: `--reward_running_norm`, `--reward_clip`, `--advantages_norm`, and `--advantage_clip` control reward normalization, reward clipping, advantage whitening, and advantage clipping.

> **Implementation status:** All algorithms in the matrix are supported except GSPO and GMPO, which remain WIP. WIP entries expose their corresponding designs or experimental interfaces but are not complete training paths in the current release.

## Runtime architecture

A typical LightRFT training cycle is:

```text
data preparation → rollout generation → reward and experience construction
                 → advantage estimation and policy update → weight synchronization
```

The Trainer organizes the iteration, Strategy provides the distributed training and rollout interfaces, and reward components evaluate generated responses. The relationships among Actor, Reference Model, Critic, and rollout policy—and the exact sequence of engine sleep/wake, model reload/offload, and weight synchronization—are documented in [Runtime Architecture and Resource Reuse](docs/source/best_practice/runtime_architecture.md).

## Installation

### Requirements

| Component | Source installation requirement or note |
| --- | --- |
| Python | `>= 3.12` |
| PyTorch | `>= 2.9.1` in `pyproject.toml` |
| GPU | Distributed training requires a CUDA-capable NVIDIA GPU environment |
| Default rollout backend | SGLang `>= 0.5.6.post2` |
| Optional rollout backend | vLLM `>= 0.18.1` |
| Training backend | DeepSpeed `>= 0.18.3`, or PyTorch FSDP v2 |

CUDA, PyTorch, FlashAttention, SGLang, and vLLM have binary compatibility constraints. Select versions compatible with the installed driver and CUDA runtime; the repository Dockerfile is one pinned reference environment.

### Source installation

SGLang is included in the default dependency set:

```bash
git clone https://github.com/opendilab/LightRFT.git
cd LightRFT
pip install -e .
```

Install the optional vLLM backend with:

```bash
pip install -e ".[vllm]"
```

Alternatively, install a compatible vLLM release after the default installation:

```bash
pip install "vllm>=0.18.1"
```

### Docker

Running a GPU container requires Docker and NVIDIA Container Toolkit. The [published example image](https://hub.docker.com/r/opendilab/lightrft) is version `v0.1.0`:

```bash
docker pull opendilab/lightrft:v0.1.0
docker run --gpus all -it --rm \
  --ipc=host \
  -v /path/to/data:/app/data \
  -v /path/to/checkpoints:/app/checkpoints \
  opendilab/lightrft:v0.1.0 /bin/bash
```

Build the repository Dockerfile with:

```bash
make dbuild
make dbuild IMAGE_NAME=your-custom-tag:latest
```

The current Dockerfile starts from `nvcr.io/nvidia/pytorch:25.01-py3` and explicitly installs a PyTorch 2.9.0 CUDA 12.8 wheel, DeepSpeed 0.18.3, vLLM 0.18.1, FlashAttention 2.8.3, and SGLang 0.5.6.post2. Its PyTorch version is lower than the `>=2.9.1` source-package declaration. Verify the intended version set before treating the Dockerfile as a release reference.

### FlashAttention installation

If a FlashAttention source build fails, select a wheel that exactly matches Python, PyTorch, CUDA, and the C++ ABI. For example, the repository Docker environment uses:

```bash
pip install flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

When no matching wheel is available, build from source in an environment with the required compiler toolchain. See the [installation guide](docs/source/installation/index.rst) and [troubleshooting guide](docs/source/best_practice/troubleshooting.md).

## Quick start

The launchers in this repository are training templates. Before running them, review model and dataset paths, GPU count, rollout tensor parallelism, sequence lengths, batch sizes, and logging configuration.

### GSM8K with GRPO

The example uses Qwen2.5-0.5B-Instruct, GSM8K, group-normalized advantages, and rule rewards.

#### 1. Prepare the dataset

```bash
python examples/gsm8k_geo3k/data_preprocess/gsm8k.py \
  --local_save_dir /path/to/data/gsm8k
```

The preprocessing script reads `openai/gsm8k` and writes training and test Parquet files. Each example contains the prompt, reference answer, and the `gsm8k_rule` reward label; the training recipe uses answer-correctness and output-format rules rather than a neural reward model.

#### 2. Review the launcher

Edit `examples/gsm8k_geo3k/run_grpo_gsm8k_qwen2.5_0.5b.sh` and verify at least:

```bash
PATH_TO_YOUR_BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PATH_TO_YOUR_GSM8K_DATASET="/path/to/data/gsm8k"

export NNODES=1
export GPUS_PER_NODE=8
ENGINE_TP=2
```

Also review W&B credentials, master address/port, batch sizes, and sequence lengths. `ENGINE_TP` must divide the total process count.

If W&B is not required, leave `WANDB_API_KEY` empty and remove or adjust the corresponding launcher options. Multi-node execution also requires correct `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT` values.

#### 3. Launch

```bash
# Default SGLang backend
ENGINE_TYPE=sglang \
  bash examples/gsm8k_geo3k/run_grpo_gsm8k_qwen2.5_0.5b.sh

# Optional vLLM backend
ENGINE_TYPE=vllm \
  bash examples/gsm8k_geo3k/run_grpo_gsm8k_qwen2.5_0.5b.sh
```

The launcher uses `torchrun` to start `train_colocate.py`; its default recipe enables FSDP, BF16, FlashAttention, engine sleep/wake, rule rewards, and group-normalized advantages. It is an eight-GPU template. When reducing GPU count, also adjust tensor parallelism, global and micro batch sizes, sequence lengths, or model size.

### Geo3K visual reasoning

Prepare Geo3K, review the model and dataset paths in the launcher, and then run:

```bash
python examples/gsm8k_geo3k/data_preprocess/geo3k.py \
  --local_save_dir /path/to/data/geo3k

ENGINE_TYPE=sglang \
  bash examples/gsm8k_geo3k/run_grpo_geo3k_qwen2.5_vl_7b.sh
```

See the [GSM8K/Geo3K tutorial](docs/source/quick_start/grpo_gsm8k_geo3k_tutorial.md) for the complete workflow.

## Configuration

Model, data, algorithm, distributed-backend, rollout-engine, logging, and checkpoint options are organized in the [configuration guide](docs/source/quick_start/configuration.md). Entry points do not necessarily expose identical arguments, so also inspect the selected launcher and its command-line help:

```bash
python examples/gsm8k_geo3k/train_colocate.py --help
```

For reproducible experiments, use launchers, documentation, and argument parsers from the same repository revision, and treat the selected entry point's `--help` and source implementation as authoritative.

## Examples and applications

| Directory | Modality or task | Purpose |
| --- | --- | --- |
| `examples/gsm8k_geo3k/` | Text and image | GRPO, PPO, LoRA, and rule-reward training |
| `examples/orm_rl_demo/` | Image | Combined format, general model, and accuracy rewards |
| `examples/grm_training/` | Image/video reward | Vision GRM training |
| `examples/grm_vl_rl/` | Video | Policy optimization with a vision reward model |
| `examples/srm_training/` | Image and audio | Vision/audio SRM training |
| `examples/r1_aqa/` | Audio | Audio-question-answering GRPO |
| `examples/on_policy_distillation/` | Text | Teacher service and OPD training |
| `examples/math_benchmarks/` | Text evaluation | Math500, AIME, GPQA, and related benchmarks |
| `examples/entropy_viz/` | Analysis | Local visualization of high-entropy tokens |
| `examples/chat/` | Interactive inference | Check exported model generation |

Example shell files contain cluster-specific paths, ports, and GPU settings and should be treated as templates.

## Monitoring, trajectories, and checkpoints

LightRFT supports Weights & Biases, TensorBoard, trajectory recording and analysis, high-entropy-token visualization, distributed training-state recovery, and Hugging Face-format checkpoints. See the [configuration guide](docs/source/quick_start/configuration.md) for the relevant options, [`lightrft/utils/ckpt_scripts/README.md`](lightrft/utils/ckpt_scripts/README.md) for checkpoint conversion, and `examples/entropy_viz/render_trajectories.html` for local trajectory visualization.

## Repository layout

```text
LightRFT/
├── lightrft/
│   ├── datasets/                 # Text and multimodal datasets
│   ├── evaluation/               # Evaluation and reward functions
│   ├── models/                   # Text, vision, and audio Actors and reward models
│   ├── strategy/
│   │   ├── deepspeed/            # DeepSpeed strategy
│   │   ├── fsdp/                 # FSDP v2 strategy
│   │   ├── sglang_utils/         # SGLang engines and weight synchronization
│   │   └── vllm_utils/           # vLLM engines and weight synchronization
│   ├── trainer/                  # Advantage computation, experience generation, and trainers
│   └── utils/                    # Logging, trajectory, and checkpoint utilities
├── examples/                     # Training, distillation, evaluation, and analysis examples
├── docs/                         # Sphinx documentation
├── tools/                        # Version and Docker helper tools
├── README.md
└── README_zh.md
```

## Documentation and troubleshooting

### Documentation index

- [Installation](docs/source/installation/index.rst)
- [GSM8K/Geo3K tutorial](docs/source/quick_start/grpo_gsm8k_geo3k_tutorial.md)
- [Algorithms](docs/source/quick_start/algorithms.md)
- [Configuration](docs/source/quick_start/configuration.md)
- [Strategy guide](docs/source/best_practice/strategy.rst)
- [Strategy design philosophy](docs/source/best_practice/strategy_design_philosophy.md)
- [Runtime architecture](docs/source/best_practice/runtime_architecture.md)
- [Reward models](docs/source/best_practice/reward_model.md)
- [FAQ](docs/source/best_practice/faq.md)
- [Troubleshooting](docs/source/best_practice/troubleshooting.md)
- [Contributing](docs/source/best_practice/contributing.md)

For rollout-backend, GPU-memory, distributed-initialization, multimodal-data, and training-stability issues, consult the [FAQ](docs/source/best_practice/faq.md) and [troubleshooting guide](docs/source/best_practice/troubleshooting.md).

### Build the documentation locally

```bash
pip install -r requirements-doc.txt
make docs
```

The HTML output is written to `docs/build/html/index.html`. For live preview:

```bash
make docs-live
# Open http://localhost:8000 in a browser
```

## Roadmap

- [v0.1.2 plan](https://github.com/opendilab/LightRFT/issues/28)
- [v0.1.1 plan](https://github.com/opendilab/LightRFT/issues/19)

Roadmap entries describe proposed work and are not guarantees of current functionality.

## Contributing

Issues and pull requests are welcome. The recommended workflow is:

1. Fork the repository and create a feature or documentation branch from `main`.
2. Keep the change scoped and add the necessary tests or documentation checks.
3. Use a [Conventional Commits](https://www.conventionalcommits.org/) style commit message.
4. Push the branch and open a pull request describing the motivation, changes, and validation.

Common repository commit types include `feature`, `fix`, `polish`, `docs`, `style`, and `refactor`. Documentation branch names should contain `doc` when the documentation deployment workflow is required.

Run the development checks with:

```bash
pip install -r requirements-dev.txt
make format   # YAPF
make fcheck   # Flake8
```

See the [contribution guide](docs/source/best_practice/contributing.md) for the repository workflow.

## Citation, license, and acknowledgements

### Citation

If LightRFT supports your research or application, please cite:

```bibtex
@misc{lightrft,
  title={LightRFT: Light, Efficient, Omni-modal & Reward-model Driven Reinforcement Fine-Tuning Framework},
  author={Niu, Yazhe and Pu, Yuan and Shi, Dongxing and Lu, Yudong and Xiong, Yingtong and Ge, Ruijun and Sun, Jiaxuan and Wan, Zunian and Zhang, Shaoang},
  publisher={GitHub},
  howpublished={\url{https://github.com/opendilab/LightRFT}},
  year={2025},
}
```

### License

LightRFT is licensed under the [Apache License 2.0](LICENSE).

### Acknowledgements

LightRFT is based on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), with some files and implementations adapted or reused. The project also builds on or learns from [verl](https://github.com/volcengine/verl), [SGLang](https://github.com/sgl-project/sglang), [vLLM](https://github.com/vllm-project/vllm), [DeepSpeed](https://github.com/microsoft/DeepSpeed), and [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html).

The project is developed in collaboration with colleagues from the System Platform Center and the AI Safety and Trustworthiness Center at Shanghai AI Laboratory.

### Contact

- Issues: [opendilab/LightRFT](https://github.com/opendilab/LightRFT/issues)
- Email: opendilab@pjlab.org.cn
