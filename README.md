# LightRFT

<div align="center">

<img src="assets/logo.png" alt="LightRFT Logo" width="600"/>

**Light, Efficient, Omni-modal & Reward-model Driven Reinforcement Fine-Tuning Framework**

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/opendilab/lightrft)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

English | [简体中文](README_zh.md)

</div>

---

## 📖 Introduction

**LightRFT** (Light Reinforcement Fine-Tuning) is an advanced reinforcement learning fine-tuning framework designed for Large Language Models (LLMs) and Vision-Language Models (VLMs). This framework provides efficient and scalable RLHF (Reinforcement Learning from Human Feedback) and RLVR training capabilities, supporting multiple state-of-the-art algorithms and distributed training strategies.

### ✨ Key Features

- 🚀 **High-Performance Inference Engines**
  - Integrated vLLM and SGLang for efficient sampling and inference
  - FP8 inference optimization for significantly reduced latency and memory usage
  - Flexible engine sleep/wake mechanisms for optimal resource utilization

- 🧠 **Rich Algorithm Ecosystem** 
  - **Policy Optimization**: GRPO, GSPO, GMPO, Dr.GRPO
  - **Advantage Estimation**: REINFORCE++, CPGD
  - **Reward Processing**: Reward Norm/Clip
  - **Sampling Strategy**: FIRE Sampling, Token-Level Policy
  - **Stability Enhancement**: DAPO, select_high_entropy_tokens

- 🔧 **Flexible Training Strategies**
  - FSDP (Fully Sharded Data Parallel) v2 support
  - DeepSpeed ZeRO (Stage 1/2/3) support
  - Gradient checkpointing and mixed precision training (BF16/FP16)
  - Adam Offload and memory optimization techniques

- 🎯 **Innovative Resource Collaboration**
  - **Colocate Anything**: Co-locate reward models with training models to maximize GPU utilization
    - Support multiple reward models for parallel inference on the same device
    - Dynamic memory management with automatic training/inference phase switching
    - Reduced cross-device communication overhead for improved end-to-end training efficiency
  - **Balance Anything** 🚧 (Under Development): Intelligent load balancing system
    - Adaptive task scheduling and resource allocation
    - Automatic load balancing for multi-node training
    - Performance optimization for heterogeneous hardware environments

- 🌐 **Comprehensive Multimodal Support**
  - **Native Vision-Language Model (VLM) Training**
    - Support for mainstream VLMs like Qwen-VL
    - Parallel processing of multimodal image-text data
    - Efficient multimodal tokenization and batching
  - **Multimodal Reward Modeling**
    - Support for multiple visual reward models working in collaboration
    - Joint optimization of image understanding and text generation
  - **Complete Vision-Language Alignment Training Pipeline**
    - Optimized for multimodal RLVR/RLHF training
    - Built-in support for vision-language model fine-tuning

- 📊 **Complete Experimental Toolkit**
  - Weights & Biases (W&B) integration
  - Math capability benchmarking (GSM8K, Geo3K, etc.)
  - Trajectory saving and analysis tools
  - Automatic checkpoint management

---

## 🎯 Supported Algorithms

For detailed algorithm descriptions, implementation details, and usage guide, see [Algorithm Documentation](docs/source/quick_start/algorithms.md).

| Algorithm | Type | Key Improvement | Paper |
|-----------|------|-----------------|-------|
| **GRPO** | Policy Optimization | Group normalized advantage estimation |  [arXiv:2402.03300](https://arxiv.org/pdf/2402.03300)  |
| **GSPO** | Policy Optimization | Generalized surrogate objectives | [arXiv:2507.18071](https://arxiv.org/abs/2507.18071) |
| **GMPO (WIP)** | Policy Optimization | Generalized mirror policy optimization | [arXiv:2507.20673](https://arxiv.org/abs/2507.20673) |
| **Dr.GRPO** | Policy Optimization | Length bias mitigation | [arXiv:2503.20783](https://arxiv.org/abs/2503.20783) |
| **DAPO** | Policy Optimization | Decoupled clip and dynamic sampling policy optimization | [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) |
| **REINFORCE++** | Advantage Estimation | Improved baseline estimation | [arXiv:2501.03262](https://arxiv.org/abs/2501.03262) |
| **CPGD** | Advantage Estimation | KL-based drift constraint | [arXiv:2505.12504](https://arxiv.org/abs/2505.12504) |
| **FIRE Sampling** | Sampling Strategy | Filtering and ranking strategies | [arXiv:2410.21236](https://arxiv.org/abs/2410.21236) |

---

## 🚀 Quick Start

### Requirements

- Python >= 3.10
- CUDA >= 12.8
- PyTorch >= 2.5.1

### Docker Images

TO BE DONE

### Installation

Clone and install LightRFT:

```bash
# Clone the repository
git clone https://github.com/opendilab/LightRFT.git
cd LightRFT

# Install dependencies
pip install -r requirements.txt

# Install LightRFT
pip install -e .
```


## 📚 Usage Guide

### Basic Example: GRPO Training

```bash
# Single node, 8 GPU training example
cd LightRFT

# Run GRPO training (GSM8K math reasoning task)
bash examples/gsm8k_geo3k/run_grpo_gsm8k_qwen2.5_0.5b.sh

# Or run Geo3K geometry problem training (VLM multimodal)
bash examples/gsm8k_geo3k/run_grpo_geo3k_qwen2.5_vl_7b.sh
```

---

## 🏗️ Project Structure

```
LightRFT/
├── lightrft/                      # Core library
│   ├── strategy/                  # Training & inference strategies
│   │   ├── fsdp/                  # FSDP implementation
│   │   ├── deepspeed/             # DeepSpeed implementation
│   │   ├── vllm_utils/            # vLLM utilities
│   │   └── sglang_utils/          # SGLang utilities
│   ├── models/                    # Model definitions
│   │   ├── actor_language.py      # Language model actor
│   │   ├── actor_vl.py            # Vision-language model actor
│   │   └── monkey_patch/          # Model adaptation patches
│   ├── trainer/                   # Trainer implementations
│   │   ├── ppo_trainer.py         # PPO trainer
│   │   ├── ppo_trainer_vl.py      # VLM PPO trainer
│   │   ├── fast_exp_maker.py      # Experience generator
│   │   ├── experience_maker.py    # Base experience generator
│   │   ├── experience_maker_vl.py # VLM experience generator
│   │   └── spmd_ppo_trainer.py    # SPMD PPO trainer
│   ├── datasets/                  # Dataset processing
│   └── utils/                     # Utility functions
│       └── ckpt_scripts/          # Checkpoint processing scripts
│
├── examples/                      # Usage examples
│   ├── gsm8k_geo3k/               # GSM8K/Geo3K math reasoning training examples
│   ├── grm_training/              # Generative reward model training examples
│   ├── srm_training/              # Scalar reward model training examples
│   └── chat/                      # Model dialogue examples
│
├── docs/                          # 📚 Sphinx documentation
│   └── source/
│       ├── installation/          # Installation guides
│       ├── quick_start/           # Quick start & user guides
│       │   ├── algorithms.md      # Algorithm documentation (English)
│       │   ├── algorithms_cn.md   # Algorithm documentation (Chinese)
│       │   └── configuration.md   # Configuration reference
│       └── best_practice/         # Best practices & resources
│           ├── strategy_usage.rst   # Training strategy usage (English)
│           ├── strategy_usage_zh.md # Training strategy usage (Chinese)
│           ├── faq.md              # Frequently asked questions
│           ├── troubleshooting.md  # Troubleshooting guide
│           └── contributing.md     # Contribution guidelines
│
├── assets/                        # Assets
│   └── logo.png                   # Project logo
│
├── results/                       # Training results
├── rft_logs/                      # Training logs
└── README.md                      # Project documentation
```

### 🔑 Key Directory Descriptions

- **`lightrft/`**: LightRFT core library, providing training strategies, model definitions, and trainer implementations
- **`examples/`**: Complete training examples and scripts
  - `gsm8k_geo3k/`: GSM8K and Geo3K math reasoning training examples
  - `grm_training/`: Generative reward model training examples
  - `srm_training/`: Scalar reward model training examples
  - `chat/`: Model dialogue examples
- **`docs/`**: Sphinx documentation with complete user guides and API documentation

---

## ⚙️ Key Configuration Parameters

### Batch Size Configuration

```bash
TBS=128                           # Training batch size
RBS=128                            # Rollout batch size
micro_train_batch_size=1          # Micro batch size per GPU
micro_rollout_batch_size=2        # Rollout micro batch size
```

### Algorithm Parameters

```bash
--advantage_estimator group_norm  # Advantage estimator: group_norm, reinforce, cpgd
--n_samples_per_prompt 8          # Number of samples per prompt
--max_epochs 1                    # Training epochs per episode
--num_episodes 3                  # Total training episodes
--kl_estimator k3                 # KL estimator type
--init_kl_coef 0.001              # KL penalty coefficient
```

### Distributed Training

```bash
--fsdp                            # Enable FSDP
--zero_stage 3                    # DeepSpeed ZeRO Stage
--gradient_checkpointing          # Gradient checkpointing
--adam_offload                    # Adam optimizer offload
--bf16                            # BF16 mixed precision
```

### Inference Engine

```bash
--rm_use_engine                   # Use inference engine (vLLM/SGLang)
--engine_mem_util 0.4             # Engine memory utilization
--engine_tp_size 1                # Engine tensor parallelism degree
--enable_engine_sleep             # Enable engine sleep mechanism
```

---

## 🔧 Troubleshooting


See training scripts for detailed parameter validation logic.

### 1. OOM (Out of Memory)

**Solutions**:
- Reduce `micro_train_batch_size` and `micro_rollout_batch_size`
- Enable `--gradient_checkpointing`
- Lower `--engine_mem_util`
- Use ZeRO Stage 3

### 2. Training Instability

**Solutions**:
- Enable Reward Normalization: `--normalize_reward`
- Lower learning rate
- Use `--advantage_estimator group_norm`
- Try DAPO algorithm

---

---

## 📖 Documentation

### 📚 Complete Documentation Guide

**Quick Start:**
- [Installation Guide](docs/source/installation/index.rst) - Docker images, installation methods, and troubleshooting
- [Supported Algorithms](docs/source/quick_start/algorithms.md) - Comprehensive algorithm guide with implementation details
- [Configuration Reference](docs/source/quick_start/configuration.md) - Complete parameter documentation

**Best Practices:**
- [Training Strategy Usage](docs/source/best_practice/strategy_usage.rst) - FSDP, DeepSpeed, and inference engine configuration
- [FAQ](docs/source/best_practice/faq.md) - Frequently asked questions and solutions
- [Troubleshooting Guide](docs/source/best_practice/troubleshooting.md) - Common issues and debugging
- [Contributing Guide](docs/source/best_practice/contributing.md) - How to contribute to LightRFT

### Build Documentation Locally

Install documentation dependencies:
```bash
pip install -r requirements-doc.txt
```

Generate HTML documentation:
```bash
make docs
# Open docs/build/index.html to view documentation
```

Live documentation preview:
```bash
make docs-live
# Visit http://localhost:8000
```

## 🤝 Contributing

We welcome community contributions! Please follow these steps:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Standards

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Code formatting (YAPF)
yapf -i -r lightrft/

# Code linting (Pylint)
pylint lightrft/
```

---

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

**LightRFT is developed based on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF).** We extend our sincere gratitude to the OpenRLHF team for their excellent work. Some files and implementations in this project are adapted and reused from OpenRLHF.

### Collaboration

This project is developed in collaboration with colleagues from the **System Platform Center** and **Safe and Trustworthy AI Center** at **Shanghai AI Laboratory**. We sincerely thank them for their contributions and support.

### Open Source Dependencies

This project builds upon the following outstanding open-source projects (including but not limited):

- **[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)**, **[verl](https://github.com/volcengine/verl)** - Core RL framework foundation (parts of key components adapted and reused)
- [vLLM](https://github.com/vllm-project/vllm) - High-performance inference engine
- [SGLang](https://github.com/sgl-project/sglang) - Structured generation language runtime
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - Distributed training optimization
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html) - Fully Sharded Data Parallel

Thanks to all contributors and supporters!

---

## 🗓️ RoadMap

We are actively working on the following improvements and features:

### Core Feature Enhancements

- [ ] **Trajectory Functionality Extension**
  - Add more analysis metrics
  - Enhanced trajectory saving and analysis capabilities

- [ ] **Reward Mechanism Refactoring**
  - Refactor rule-based and model-based reward computation
  - Optimize reward dataset processing pipeline

### Algorithm Optimization & Integration

- [ ] **More Algorithm Integration**
  - Entropy-based token selection 
  - GMPO (Generalized Mirror Policy Optimization)
  - GSPO (Generalized Surrogate Policy Optimization)

- [ ] **Advantage Computation Refactoring**
  - Optimize advantage estimation module architecture
  - Unify advantage computation interface across algorithms

- [ ] **Loss-Filter Mechanism Optimization**
  - Refactor loss filtering implementation
  - Complete GSM8K/Geo3K benchmark experiments
  - Document experimental results and analysis



Community contributions and feedback are welcome!

---

## 📮 Contact

For questions or suggestions, please contact us via:

- **Issues**: [GitHub Issues](https://github.com/yourusername/lightrft/issues)
- **Email**: opendilab@pjlab.org.cn


---

<div align="center">

**⭐ If this project helps you, please give us a star!**

Made with ❤️ by LightRFT Team

</div>
