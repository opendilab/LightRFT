.. LightRFT documentation main file

Welcome to LightRFT's Documentation!
=====================================

.. image:: ./logo.png
   :alt: LightRFT Logo
   :width: 600px
   :align: center

**LightRFT** (Light Reinforcement Fine-Tuning) is a light and efficient reinforcement learning fine-tuning framework designed for Large Language Models (LLMs), Vision-Language Models (VLMs) and other modalities and tasks.
This framework provides efficient and scalable RLHF (Reinforcement Learning from Human Feedback), RLVR (Reinforcement Learning with Verifiable Rewards), and Reward Model training and evaluation capabilities,
supporting multiple state-of-the-art algorithms and distributed training (FSDP, DeepSpeed, etc.) strategies.

Key Features
------------

🚀 **High-Performance Inference Engines**
   * Integrated vLLM and SGLang for efficient sampling and inference
   * FP8 inference optimization for reduced latency and memory usage
   * Flexible engine sleep/wake mechanisms for optimal resource utilization

🧠 **Rich Algorithm Ecosystem**
   * Policy Optimization: GRPO, GSPO, GMPO, Dr.GRPO
   * Advantage Estimation: REINFORCE++, CPGD
   * Reward Processing: Reward Norm/Clip
   * Sampling Strategy: FIRE Sampling, Token-Level Policy
   * Stability Enhancement: Clip Higher, select_high_entropy_tokens

🔧 **Flexible Training Strategies**
   * FSDP (Fully Sharded Data Parallel) support
   * DeepSpeed ZeRO (Stage 1/2/3) support
   * Gradient checkpointing and mixed precision training (BF16/FP16)
   * Adam Offload and memory optimization techniques

🌐 **Comprehensive Multimodal Support**
   * Native Vision-Language Model (VLM) training
   * Support for Qwen-VL, LLaVA, and other mainstream VLMs
   * Multimodal reward modeling with multiple reward models

📊 **Complete Experimental Toolkit**
   * Weights & Biases (W&B) integration
   * Math capability benchmarking (GSM8K, Geo3K, etc.)
   * Trajectory saving and analysis tools
   * Automatic checkpoint management

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation/index
   quick_start/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide & Best Practices

   best_practice/index

.. toctree::
   :maxdepth: 1
   :caption: API Documentation

   api_doc/utils/index
   api_doc/datasets/index
   api_doc/models/index
   api_doc/strategy/index
   api_doc/trainer/index

Quick Links
-----------

* :ref:`installation` - Installation guide
* :ref:`quick_start` - Quick start tutorial
* :doc:`quick_start/algorithms` - Supported algorithms
* :doc:`best_practice/strategy` - Strategy usage guide
* :doc:`quick_start/configuration` - Configuration parameters
* :doc:`best_practice/faq` - Frequently asked questions
* :doc:`best_practice/troubleshooting` - Troubleshooting guide

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
