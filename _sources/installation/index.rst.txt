.. _installation:

==================================
Installation
==================================

This guide provides instructions for installing and setting up LightRFT, a lightweight and high-performance reinforcement learning fine-tuning framework designed for Large Language Models (LLMs) and Vision-Language Models (VLMs).

Requirements
============

Before installing LightRFT, ensure your environment meets the following requirements:

* Python >= 3.10
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

   make live

Project Structure
=================

LightRFT is organized into several key modules:

.. code-block:: text

   LightRFT/
   ├── lightrft/                      # Core library
   │   ├── strategy/                  # Training & inference strategies
   │   │   ├── fsdp/                  # FSDP implementation
   │   │   ├── deepspeed/             # DeepSpeed implementation
   │   │   ├── vllm_utils/            # vLLM utilities
   │   │   ├── sglang_utils/          # SGLang utilities
   │   │   └── utils/                 # Strategy utilities
   │   ├── models/                    # Model definitions
   │   │   ├── actor_al.py            # Audio-language model actor
   │   │   ├── actor_language.py      # Language model actor
   │   │   ├── actor_vl.py            # Vision-language model actor
   │   │   ├── grm_vl.py              # Generative reward model (Vision-Language)
   │   │   ├── srm_al.py              # Scalar reward model (Audio-Language)
   │   │   ├── srm_vl.py              # Scalar reward model (Vision-Language)
   │   │   ├── loss.py                # Loss functions
   │   │   ├── monkey_patch/          # Model adaptation patches for distributed training
   │   │   ├── tests/                 # Model tests
   │   │   └── utils.py               # Model utilities
   │   ├── trainer/                   # Trainer implementations
   │   │   ├── ppo_trainer.py         # LLM PPO trainer
   │   │   ├── ppo_trainer_vl.py      # VLM PPO trainer
   │   │   ├── spmd_ppo_trainer.py    # SPMD PPO trainer Extension (**Core**)
   │   │   ├── grm_trainer_vl.py      # Generative reward model trainer (Vision-Language)
   │   │   ├── srm_trainer_al.py      # Scalar reward model trainer (Audio-Language)
   │   │   ├── srm_trainer_vl.py      # Scalar reward model trainer (Vision-Language)
   │   │   ├── fast_exp_maker.py      # Fast experience generator (**Core**)
   │   │   ├── experience_maker.py    # Base experience generator
   │   │   ├── experience_maker_vl.py # Base experience generator for VLM
   │   │   ├── replay_buffer.py       # Replay buffer
   │   │   ├── replay_buffer_vl.py    # VLM replay buffer
   │   │   ├── replay_buffer_utils.py # Replay buffer utilities
   │   │   ├── kl_controller.py       # KL divergence controller
   │   │   └── utils.py               # Trainer utilities
   │   ├── datasets/                  # Dataset processing
   │   │   ├── audio_alpaca.py        # Audio Alpaca dataset
   │   │   ├── grm_dataset.py         # Generative reward model dataset
   │   │   ├── hpdv3.py               # HPDv3 reward model dataset
   │   │   ├── image_reward_db.py     # Image reward database
   │   │   ├── imagegen_cot_reward.py # Image generation CoT generative reward
   │   │   ├── omnirewardbench.py     # OmniRewardBench dataset
   │   │   ├── process_reward_dataset.py # Reward dataset processing
   │   │   ├── prompts_dataset.py     # LLM Prompts dataset
   │   │   ├── prompts_dataset_vl.py  # Vision-language prompts dataset
   │   │   ├── rapidata.py            # Rapidata reward modeldataset
   │   │   ├── sft_dataset.py         # SFT dataset
   │   │   ├── sft_dataset_vl.py      # VLM SFT dataset
   │   │   ├── srm_dataset.py         # Scalar reward model base dataset
   │   │   └── utils.py               # Dataset utilities
   │   └── utils/                     # Utility functions
   │       ├── ckpt_scripts/          # Checkpoint processing scripts
   │       ├── cli_args.py            # CLI argument parsing
   │       ├── distributed_sampler.py # Distributed sampler
   │       ├── logging_utils.py       # Logging utilities
   │       ├── processor.py           # Data processor for HF model
   │       ├── remote_rm_utils.py     # Remote reward model utilities
   │       ├── timer.py               # Timer utilities
   │       ├── trajectory_saver.py    # Trajectory saver
   │       └── utils.py               # General utilities
   │
   ├── examples/                      # Usage examples
   │   ├── gsm8k_geo3k/               # GSM8K/Geo3K math reasoning training examples
   │   ├── grm_training/              # Generative reward model training examples
   │   ├── srm_training/              # Scalar reward model training examples
   │   ├── chat/                      # Model dialogue examples
   │
   ├── docs/                          # 📚 Sphinx documentation
   │   ├── Makefile                   # Documentation build Makefile
   │   ├── make.bat                   # Documentation build batch file
   │   └── source/                    # Documentation source
   │       ├── _static/               # Static files (CSS, etc.)
   │       ├── api_doc/               # API documentation
   │       ├── best_practice/         # Best practices & resources
   │       ├── installation/          # Installation guides
   │       └── quick_start/           # Quick start & user guides
   │
   ├── assets/                        # Assets
   │   └── logo.png                   # Project logo
   │
   ├── CHANGELOG.md                   # Changelog
   ├── LICENSE                        # License file
   ├── Makefile                       # Project Makefile
   ├── README.md                      # Project documentation (English)
   ├── README_zh.md                   # Project documentation (Chinese)
   ├── requirements.txt               # Python dependencies
   ├── requirements-dev.txt           # Development dependencies
   ├── requirements-doc.txt          # Documentation dependencies
   └── setup.py                       # Package setup script

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
  * ``grm_training/``: Generative reward model training examples
  * ``srm_training/``: Scalar reward model training examples
  * ``chat/``: Model dialogue examples

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
* Consult the example scripts in the `examples <https://github.com/opendilab/LightRFT/tree/main/examples>`_ directory

Next Steps
==========

After successful installation:

1. Review the :doc:`../quick_start/index` guide to understand basic usage
2. Explore :doc:`../best_practice/strategy_usage` for distributed training strategies
3. Check out the `examples <https://github.com/opendilab/LightRFT/tree/main/examples>`_ directory for complete training examples
4. Read the algorithm documentation for specific implementation details

Related Documentation
=====================

* :doc:`../quick_start/index` - Quick start guide
* :doc:`../best_practice/index` - Best practices guide