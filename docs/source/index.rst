.. LightRFT documentation main file

Welcome to LightRFT's Documentation!
=====================================

.. image:: ./logo.png
   :alt: LightRFT Logo
   :width: 600px
   :align: center

**LightRFT** (Light Reinforcement Fine-Tuning) is a lightweight, efficient, and versatile reinforcement learning fine-tuning framework designed for the fine-tuning tasks of Large Language Models (LLMs) and Vision-Language Models (VLMs). Its core advantages include:

* **Comprehensive Multi-paradigm and Multi-modal Training Support**: Native support for RLVR and RLHF training, covering various modalities such as text, image, video, and audio, and supporting the full lifecycle from base models to reward models and reward rules.
* **Unified Strategy Abstraction Layer**: A highly abstract Strategy layer that flexibly controls training (DeepSpeed/FSDPv2) and high-performance inference (vLLM/SGLang) strategies.
* **Easy-to-use and Efficient Multi-model Co-location Paradigm**: Supports flexible multi-model co-location training, enabling scalable algorithm exploration and comparison in large-scale scenarios.

Key Features
------------

🚀 **High-Performance Inference Engines**
   * Integrated vLLM and SGLang for efficient sampling and inference
   * FP8 inference optimization for reduced latency and memory usage (Work in Progress)
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
