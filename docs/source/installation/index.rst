.. _installation:

==================================
Installation Guide
==================================

This guide provides instructions for installing and setting up LightRFT, a lightweight and high-performance reinforcement learning fine-tuning framework designed for Large Language Models (LLMs) and Vision-Language Models (VLMs).

Requirements
============

Before installing LightRFT, ensure your environment meets the following requirements:

* Python >= 3.8
* CUDA >= 11.8
* PyTorch >= 2.5.1
* CUDA-compatible GPU(s)

Docker Images
=============

TO BE DONE

Installation
============

Standard Installation
----------------------

Clone and install LightRFT:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/opendilab/LightRFT.git
   cd LightRFT

   # Install dependencies
   pip install -r requirements.txt

   # Install LightRFT
   pip install -e .


Documentation Generation (Optional)
====================================

To install dependencies for generating documentation:

.. code-block:: bash

   pip install -r requirements-doc.txt

To generate HTML documentation:

.. code-block:: bash

   make docs

The documentation will be generated in the ``docs/build`` directory. Open ``index.html`` to view it.

For live browser documentation with auto-reload:

.. code-block:: bash

   make docs-live

Project Structure
=================

LightRFT is organized into several key modules:

.. code-block:: text

   LightRFT/
   ├── lightrft/                      # Core library
   │   ├── datasets/                  # Dataset implementations
   │   │   ├── audio_alpaca.py        # Audio dataset
   │   │   ├── grm_dataset.py         # General reward model dataset
   │   │   ├── prompts_dataset.py     # Prompts dataset
   │   │   ├── prompts_dataset_vl.py  # Vision-language prompts dataset
   │   │   ├── sft_dataset.py         # SFT dataset
   │   │   ├── sft_dataset_vl.py      # Vision-language SFT dataset
   │   │   ├── srm_dataset.py         # Safe reward model dataset
   │   │   └── utils.py               # Dataset utilities
   │   ├── models/                    # Model definitions
   │   │   ├── actor_al.py            # Audio-language actor model
   │   │   ├── actor_language.py      # Language actor model
   │   │   ├── actor_vl.py            # Vision-language actor model
   │   │   ├── grm_vl.py              # General reward model (VL)
   │   │   ├── srm_al.py              # Safe reward model (AL)
   │   │   ├── srm_vl.py              # Safe reward model (VL)
   │   │   ├── loss.py                # Loss functions
   │   │   ├── utils.py               # Model utilities
   │   │   └── monkey_patch/          # Model adaptation patches
   │   │       ├── apply.py           # Patch application
   │   │       ├── hf_generate_patch.py  # HuggingFace generate patch
   │   │       ├── llama.py           # LLaMA patches
   │   │       └── qwen.py            # Qwen patches
   │   ├── strategy/                  # Training & inference strategies
   │   │   ├── config.py              # Strategy configuration
   │   │   ├── fake_strategy.py       # Fake strategy for testing
   │   │   ├── strategy.py            # Main strategy implementation
   │   │   ├── strategy_base.py       # Strategy base class
   │   │   ├── deepspeed/             # DeepSpeed implementation
   │   │   │   ├── deepspeed.py       # DeepSpeed strategy
   │   │   │   └── deepspeed_utils.py # DeepSpeed utilities
   │   │   ├── fsdp/                  # FSDP implementation
   │   │   │   ├── fsdp_optimizer.py  # FSDP optimizer
   │   │   │   ├── fsdp_utils.py      # FSDP utilities
   │   │   │   └── fsdpv2.py          # FSDP v2 implementation
   │   │   ├── sglang_utils/          # SGLang utilities
   │   │   │   ├── sglang_engine.py   # SGLang engine
   │   │   │   └── sgl_model_saver.py # SGLang model saver
   │   │   ├── vllm_utils/            # vLLM utilities
   │   │   │   └── vllm_worker_wrap_no_ray.py  # vLLM worker wrapper
   │   │   └── utils/                 # Strategy utilities
   │   │       ├── broadcast_utils.py # Broadcast utilities
   │   │       ├── ckpt_utils.py      # Checkpoint utilities
   │   │       ├── data_utils.py      # Data utilities
   │   │       ├── distributed_util.py  # Distributed utilities
   │   │       ├── optimizer_utils.py # Optimizer utilities
   │   │       ├── parallel_utils.py  # Parallel utilities
   │   │       └── statistic.py       # Statistics utilities
   │   ├── trainer/                   # Trainer implementations
   │   │   ├── experience_maker.py    # Experience generator
   │   │   ├── experience_maker_vl.py # VLM experience generator
   │   │   ├── fast_exp_maker.py      # Fast experience maker
   │   │   ├── grm_trainer_vl.py      # General reward model trainer (VL)
   │   │   ├── kl_controller.py       # KL divergence controller
   │   │   ├── ppo_trainer.py         # PPO trainer
   │   │   ├── ppo_trainer_vl.py      # Vision-language PPO trainer
   │   │   ├── replay_buffer.py       # Replay buffer
   │   │   ├── replay_buffer_utils.py # Replay buffer utilities
   │   │   ├── replay_buffer_vl.py    # Vision-language replay buffer
   │   │   ├── spmd_ppo_trainer.py    # SPMD PPO trainer
   │   │   ├── srm_trainer_al.py      # Safe reward model trainer (AL)
   │   │   ├── srm_trainer_vl.py      # Safe reward model trainer (VL)
   │   │   └── utils.py               # Trainer utilities
   │   └── utils/                     # Utility functions
   │       ├── cli_args.py            # CLI argument parsing
   │       ├── distributed_sampler.py # Distributed sampler
   │       ├── logging_utils.py       # Logging utilities
   │       ├── processor.py           # Data processors
   │       ├── remote_rm_utils.py     # Remote reward model utilities
   │       ├── timer.py               # Timer utilities
   │       ├── trajectory_saver.py    # Trajectory saving utilities
   │       └── utils.py               # General utilities
   │
   ├── examples/                      # Usage examples
   │   ├── chat/                      # Chat model training examples
   │   ├── grm_training/              # General reward model training examples
   │   ├── gsm8k_geo3k/               # GSM8K/Geo3K math reasoning examples
   │   │   ├── data_preprocess/       # Data preprocessing scripts
   │   │   ├── train_colocate.py      # Co-located training script
   │   │   ├── reward_models_utils.py # Reward model utilities
   │   │   ├── run_grpo_gsm8k_qwen2.5_0.5b.sh    # GSM8K training script
   │   │   └── run_grpo_geo3k_qwen2.5_vl_7b.sh   # Geo3K VLM training script
   │   ├── safework_t1/               # Safe and trusted work examples
   │   └── srm_training/              # Safe reward model training examples
   │
   ├── docs/                          # 📚 Sphinx documentation
   │   └── source/
   │       ├── installation/          # Installation guides
   │       ├── quick_start/           # Quick start & user guides
   │       ├── best_practice/         # Best practices & resources
   │       └── api_doc/               # API documentation
   │           ├── datasets/          # Dataset API
   │           ├── models/            # Model API
   │           ├── strategy/          # Strategy API
   │           ├── trainer/           # Trainer API
   │           └── utils/             # Utilities API
   │
   ├── assets/                        # Assets
   │   └── logo.png                   # Project logo
   │
   ├── results/                       # Training results
   ├── rft_logs/                      # Training logs
   ├── requirements.txt               # Python dependencies
   ├── requirements-dev.txt           # Development dependencies
   ├── requirements-doc.txt           # Documentation dependencies
   ├── setup.py                       # Package setup
   └── README.md                      # Project documentation

Key Directory Descriptions
--------------------------

* **lightrft/**: LightRFT core library with five main modules:

  * ``datasets/``: Dataset implementations for prompts, SFT, reward modeling (text, vision-language, audio-language)
  * ``models/``: Actor models (language, vision-language, audio-language), reward models, and loss functions
  * ``strategy/``: Training strategies including FSDP, DeepSpeed, vLLM/SGLang integration
  * ``trainer/``: Trainer implementations for PPO, experience generation, and replay buffers
  * ``utils/``: Utility functions for CLI, logging, distributed training, and trajectory saving

* **examples/**: Complete training examples and scripts

  * ``gsm8k_geo3k/``: GSM8K and Geo3K math reasoning training examples
  * ``grm_training/``: General reward model training examples
  * ``srm_training/``: Safe reward model training examples
  * ``chat/``: Chat model training examples
  * ``safework_t1/``: Safe and trusted work examples

* **docs/**: Sphinx documentation with complete user guides and API documentation

Verification
============

To verify your installation, run a simple test:

.. code-block:: bash

   python -c "import lightrft; print(lightrft)"

You should see the module path without any import errors.

Quick Start Example
===================

After installation, try a basic GRPO training example:

.. code-block:: bash

   # Single node, 8 GPU training example
   cd /path/to/LightRFT

   # Run GRPO training (GSM8K math reasoning task)
   bash examples/gsm8k_geo3k/run_grpo_gsm8k_qwen2.5_0.5b.sh

   # Or run Geo3K geometry problem training (VLM multimodal)
   bash examples/gsm8k_geo3k/run_grpo_geo3k_qwen2.5_vl_7b.sh

Troubleshooting
===============

Common Issues
-------------

**Issue**: CUDA errors or version mismatch

* **Solution**: Ensure CUDA drivers and toolkit version match your PyTorch installation. Check with ``nvcc --version`` and ``python -c "import torch; print(torch.version.cuda)"``

**Issue**: Out of memory errors during training

* **Solution**:

  * Reduce ``micro_train_batch_size`` or ``micro_rollout_batch_size``
  * Enable gradient checkpointing: ``--gradient_checkpointing``
  * Use FSDP with CPU offload: ``--fsdp --fsdp_cpu_offload``
  * Adjust engine memory utilization: ``--engine_mem_util 0.4``

**Issue**: Slow installation of evaluation dependencies

* **Solution**: Use a mirror or proxy for pip:

  .. code-block:: bash

     pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package>

For Additional Support
----------------------

If you encounter issues not covered here:

* Check the project's `GitHub Issues <https://github.com/opendilab/LightRFT/issues>`_
* Review the :doc:`../best_practice/strategy_usage` guide for training configuration
* Consult the example scripts in the ``examples/`` directory

Next Steps
==========

After successful installation:

1. Review the :doc:`../quick_start` guide to understand basic usage
2. Explore :doc:`../best_practice/strategy_usage` for distributed training strategies
3. Check out the ``examples/`` directory for complete training examples
4. Read the algorithm documentation for specific implementation details

Related Documentation
=====================

* :doc:`../quick_start` - Quick start guide
* :doc:`../best_practice/strategy_usage` - Strategy usage guide
* :doc:`../api/index` - API reference
