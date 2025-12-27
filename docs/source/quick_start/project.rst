======================================================================================
LightRFT Project Overview
======================================================================================

LightRFT is a framework designed for RLHF and RLVR that enables efficient fine-tuning of large language models and vision-language models. The project is structured to support distributed training across multiple GPUs and nodes while maintaining a clean, modular architecture.

High-Level Architecture
======================================================================================

The framework is organized into five main modules, each serving a distinct purpose in the Reinforcement Learning (RL) pipeline.

Core Modules
======================================================================================

Datasets Module
--------------------------------------------------------------------------------------

The Datasets module provides data loading and preprocessing capabilities for various training scenarios including prompts, supervised fine-tuning (SFT), and reward modeling.

Key components:

- Prompts datasets for RL training (text and vision-language)
- SFT datasets for supervised fine-tuning (text and vision-language)
- Reward model datasets (general and safe reward models)
- Audio-language dataset support
- Data preprocessing and tokenization utilities

Models Module
--------------------------------------------------------------------------------------

The Models module defines the neural architectures and adaptations necessary for reinforcement learning with language models and vision-language models. It focuses on making minimal modifications to existing model architectures through a monkey patching approach.

Key components:

- Actor network implementations (language, vision-language, audio-language)
- Reward model implementations (general and safe reward models)
- Loss function definitions for policy optimization
- Non-invasive model modifications via monkey patching
- Support for various model architectures (LLaMA, Qwen, etc.)
- Custom generation methods optimized for RL training

Strategy Module
--------------------------------------------------------------------------------------

The Strategy module implements different approaches to distributed training, allowing the framework to scale efficiently across multiple GPUs and nodes. It provides abstractions for various parallelism techniques and optimization strategies.

Key components:

- DeepSpeed integration for ZeRO-based optimization
- Fully Sharded Data Parallel (FSDP) implementation
- Efficient inference engines (vLLM, SGLang)
- Utilities for distributed tensor operations
- Checkpoint management and broadcasting
- Strategy selection and configuration

Trainer Module
--------------------------------------------------------------------------------------

The Trainer module implements the reinforcement learning algorithms and training loops. It coordinates the interaction between models, strategies, and optimization processes.

Key components:

- PPO (Proximal Policy Optimization) implementation
- Experience generation and collection
- Advantage estimation and return computation
- Policy and value function updates
- Replay buffer management (standard and vision-language)
- KL divergence control
- SPMD (Single Program Multiple Data) training support

Utils Module
--------------------------------------------------------------------------------------

The Utils module provides general-purpose utility functions that support the entire framework, including CLI argument parsing, logging, distributed sampling, and data processing.

Key components:

- Command-line interface argument parsing
- Distributed sampler for data loading
- Logging and monitoring utilities
- Data processors for multimodal inputs
- Remote reward model utilities
- Timer and profiling utilities
- Trajectory saving for analysis

System Workflow
======================================================================================

The LightRFT framework operates through a coordinated workflow:

1. **Initialization**: The system begins by setting up the distributed environment and loading configurations via the Utils module.

2. **Data Loading**: The Datasets module loads and preprocesses training data, creating dataloaders for prompts, SFT, or reward modeling.

3. **Model Preparation**: Pre-trained models are loaded and adapted for RL through strategic monkey patching in the Models module.

4. **Strategy Selection**: Based on configuration, an appropriate distributed training strategy is selected and initialized.

5. **Training Loop**: The Trainer module drives the training process, generating experiences, computing rewards, and updating the policy.

6. **Inference Optimization**: During generation, specialized inference engines (vLLM/SGLang) may be employed for efficiency.

7. **Checkpointing and Evaluation**: The system regularly saves model states and evaluates performance.

Extension Points
======================================================================================

LightRFT is designed with extensibility in mind:

- **New Datasets**: Additional dataset types can be added by extending the base dataset classes in the Datasets module.

- **New Models**: Support for additional model architectures can be added through the monkey patch system or by creating new actor/reward model classes.

- **Alternative Strategies**: New distributed training approaches can be implemented by extending the strategy base class.

- **Custom RL Algorithms**: Different reinforcement learning algorithms can be implemented as new trainer classes.

- **Inference Optimization**: Additional inference engines can be integrated through the strategy module.

- **Custom Utilities**: New utility functions and processors can be added to support specific use cases.

This modular design allows LightRFT to adapt to new research directions, model architectures, and hardware configurations while maintaining a consistent interface for users.
