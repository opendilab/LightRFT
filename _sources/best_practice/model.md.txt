# LightRFT Models Design Document

## Overview

The `lightrft/models` module provides a comprehensive framework for implementing actor models in reinforcement learning scenarios, specifically designed for language model fine-tuning and human feedback integration. This document outlines the design philosophy, architecture, and implementation details of the models package.

## Design Philosophy

### 1. Modular Architecture
The models package follows a modular design approach that separates concerns and promotes code reusability:

- **Actor Base Classes**: Provide foundational functionality for different types of actors
- **Utility Functions**: Common operations and helper functions shared across models
- **Model Patches**: Specialized adaptations for specific model architectures

### 2. Flexibility and Extensibility
The design prioritizes flexibility to support various model types and use cases:

- Support for both text-only and vision-language models
- Configurable optimization strategies (LoRA, quantization, Flash Attention)
- Adaptable to different model architectures and sizes

### 3. Performance Optimization
Built-in optimizations for efficient training and inference:

- Memory-efficient implementations with gradient checkpointing
- Support for distributed training with DeepSpeed and FSDP
- Sample packing for improved batch processing efficiency

## Architecture Components

### Core Classes

#### 1. ActorText
**Purpose**: General-purpose actor for text-only language models

**Key Features**:
- Supports various causal language model architectures
- Configurable LoRA adaptation with auto-detection of target modules
- Flash Attention 2.0 integration for improved performance

**Design Decisions**:
- Generic implementation that works with any HuggingFace causal LM
- Automatic detection of linear modules for LoRA injection
- Flexible generation parameters with post-processing for RL training

#### 2. ActorVL (Vision-Language)
**Purpose**: Specialized actor for vision-language models

**Key Features**:
- Multi-modal input processing (text + vision)
- Support for various VL architectures (LLaVA, Qwen2-VL, InternVL, etc.)
- Image grid processing for different aspect ratios
- Specialized handling for different model types

**Design Decisions**:
- Separate class to handle the complexity of multi-modal inputs
- Model-specific adaptations for different VL architectures
- Flexible pixel value and grid dimension handling

### Utility Functions

#### 1. LoRA Configuration (`apply_lora_configuration`)
**Purpose**: Centralized LoRA setup and configuration

**Design Rationale**:
- Eliminates code duplication across different actor types
- Provides consistent LoRA configuration across the framework

#### 2. Log Probability Computation (`log_probs_from_logits`)
**Purpose**: Efficient computation of log probabilities from model logits

**Design Features**:
- Memory-optimized implementation with row-by-row processing
- Support for different data types (float32, float16, bfloat16)
- Flash Attention integration for improved performance
- Automatic fallback for unsupported configurations

#### 3. Position ID Management (`reset_position_ids`)
**Purpose**: Handle position IDs for packed sequences

**Design Rationale**:
- Essential for sample packing optimization
- Maintains correct positional encoding across concatenated sequences
- Supports variable-length sequences in packed format

**Design Features**:
- Model-architecture-aware detection
- Configurable exclusion of specific modules (vision towers, etc.)
- Support for various model types and architectures

### Model Patches

#### Purpose
The `monkey_patch` directory contains model-specific adaptations and optimizations:

- **Architecture-specific optimizations**: Tailored improvements for specific model architectures
- **Generation method patches**: Enhanced generation capabilities
- **Performance optimizations**: Model-specific performance improvements

## Implementation Details

### 1. Model Initialization Strategy

The models support two initialization patterns:

#### Pattern A: From Pretrained Path
```python
actor = ActorText(
    pretrain_or_model="model_path",
    lora_rank=16,
    use_flash_attention_2=True
)
```

#### Pattern B: From Existing Model
```python
actor = ActorText(
    pretrain_or_model=existing_model,
    packing_samples=True
)
```

**Design Rationale**:
- Supports both training from scratch and fine-tuning existing models
- Enables flexible model deployment scenarios
- Maintains backward compatibility with existing workflows

### 2. Generation and Forward Pass Design

#### Generation Method
- **Input Processing**: Handles various input formats and parameters
- **Model Generation**: Delegates to underlying model with configured parameters
- **Post-processing**: Creates attention masks and action masks for RL training

#### Forward Method
- **Position ID Handling**: Manages positional encoding for different sequence formats
- **Log Probability Computation**: Efficiently computes action probabilities
- **Packed Sequence Support**: Handles multiple sequences in a single batch

### 3. Memory and Performance Optimizations

#### Gradient Checkpointing
- Optional memory-saving technique
- Configurable via `gradient_checkpointing_enable/disable`
- Balances memory usage with computational overhead

#### Sample Packing
- Concatenates multiple sequences for efficient batch processing
- Maintains correct attention patterns through position ID management
- Significantly improves training throughput for variable-length sequences

## Configuration and Customization

### 1. LoRA Configuration
- **Rank and Alpha**: Configurable LoRA dimensions and scaling
- **Target Modules**: Automatic detection with manual override capability
- **Dropout**: Configurable regularization strength

### 2. Attention Mechanisms
- **Flash Attention 2.0**: Optional high-performance attention implementation
- **Fallback Support**: Automatic fallback to standard attention when needed
- **Architecture Compatibility**: Works across different model architectures

### 3. Device and Distributed Training
- **Device Mapping**: Flexible device placement for multi-GPU setups
- **DeepSpeed Integration**: Native support for DeepSpeed ZeRO optimization
- **FSDP Compatibility**: Support for Fully Sharded Data Parallel training

## Error Handling and Robustness

### 1. Graceful Degradation
- Automatic fallback for unsupported features
- Clear error messages for configuration issues
- Compatibility checks for model requirements

### 2. Validation and Assertions
- Input validation for critical parameters
- Assertion checks for incompatible configurations
- Runtime validation of model compatibility

## Conclusion

The LightRFT models package provides a robust, flexible, and efficient foundation for reinforcement learning with language models. The modular design ensures maintainability and extensibility while the comprehensive optimization support enables efficient training and deployment across various hardware configurations and model architectures.

The design prioritizes:
- **Simplicity**: Easy to use and understand
- **Flexibility**: Adaptable to various use cases
- **Performance**: Optimized for efficiency
- **Reliability**: Robust error handling and validation
- **Extensibility**: Easy to add new features and model types

This architecture serves as a solid foundation for current needs while providing a clear path for future enhancements and adaptations.
