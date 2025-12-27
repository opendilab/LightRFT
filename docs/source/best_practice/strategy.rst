Strategy Usage Guide
====================

Overview
--------

LightRFT's strategy module is the distributed training capabilities with additional features for efficient reinforcement learning fine-tuning. The strategy provides a unified interface for managing:

* **Distributed Training Backends**: DeepSpeed ZeRO and FSDP (Fully Sharded Data Parallel)
* **Inference Engine Integration**: vLLM and SGLang for high-throughput generation
* **Memory Optimization**: Optimizer offloading, gradient accumulation, and engine sleep modes
* **Sequence Parallelism**: Efficient handling of long sequences across multiple GPUs

Core API Extensions
-------------------

LightRFT adds the following key methods to the strategy interface:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Purpose
   * - ``setup_inference_engine()``
     - Initialize vLLM or SGLang inference engine
   * - ``update_engine_weights()``
     - Synchronize actor model weights to inference engine
   * - ``gather_and_generate()``
     - Distributed generation with automatic prompt gathering
   * - ``maybe_load_optimizer()``
     - Load optimizer states from CPU (FSDP only)
   * - ``maybe_offload_optimizer()``
     - Offload optimizer states to CPU (FSDP only)
   * - ``wakeup_inference_engine()``
     - Wake up inference engine from sleep mode
   * - ``maybe_sleep_inference_engine()``
     - Put inference engine to sleep to save memory

Creating a Strategy
-------------------

Basic Setup
~~~~~~~~~~~

Use the factory function ``get_strategy()`` to create a strategy instance:

.. code-block:: python

    from lightrft.strategy import get_strategy
    from lightrft.utils import add_arguments

    def train(args):
        # Create strategy (automatically selects DeepSpeed or FSDP based on args)
        strategy = get_strategy(args)

        # Setup inference engine for generation
        strategy.setup_inference_engine(args, engine_type='vllm')

        # Access the engine if needed
        vllm_engine = strategy.inference_engine

        # Create trainer
        trainer = SPMDPPOTrainer(
            strategy=strategy,
            actor=actor,
            critic=critic,
            reward_model=reward_model,
            initial_model=initial_model,
            ema_model=ema_model,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            actor_scheduler=actor_scheduler,
            critic_scheduler=critic_scheduler,
            ...
        )

Strategy Selection
~~~~~~~~~~~~~~~~~~

The strategy type is automatically determined by configuration arguments:

* **FSDP**: Set ``--fsdp`` flag
* **DeepSpeed**: Default when ``--fsdp`` is not set (configurable via ``--zero_stage``)

Using Strategy in Trainers
---------------------------

Standard Training Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The strategy provides standard distributed training operations:

.. code-block:: python

    # Backward pass
    strategy.backward(loss, model, optimizer)

    # Optimizer step with gradient clipping
    strategy.optimizer_step(optimizer, model, scheduler, name="actor")

    # Distributed communication
    averaged_value = strategy.all_reduce(local_value, op="mean")
    gathered_values = strategy.all_gather(local_value)

Memory-Optimized Training
~~~~~~~~~~~~~~~~~~~~~~~~~~

For FSDP-based training, use optimizer offloading to reduce GPU memory:

.. code-block:: python

    def ppo_train(self, global_steps=0):
        torch.cuda.synchronize()
        train_begin = time.time()

        # Load optimizer states from CPU to GPU (FSDP only)
        self.strategy.maybe_load_optimizer(self.actor_optim)

        # Perform training
        train_ret = super().ppo_train(global_steps)

        # Offload optimizer states from GPU to CPU (FSDP only)
        self.strategy.maybe_offload_optimizer(self.actor_optim)

        torch.cuda.synchronize()
        self.strategy.print(f"PPO Train TIMECOST {time.time() - train_begin}")

        # Synchronize actor weights to inference engine
        self.strategy.update_engine_weights(self.actor)

        return train_ret

Engine Weight Synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After training updates, synchronize model weights to the inference engine:

.. code-block:: python

    # Update inference engine with latest actor weights
    strategy.update_engine_weights(actor)

This ensures that the inference engine uses the most recent model parameters for generation.

Using Strategy in Experience Makers
------------------------------------

Text Generation (LLM)
~~~~~~~~~~~~~~~~~~~~~

Use ``gather_and_generate()`` for distributed text generation:

.. code-block:: python

    # Tokenize prompts (without padding for efficiency)
    all_prompt_token_ids = self.tokenize_fn(
        all_prompts,
        self.prompt_max_len,
        padding=False
    )["input_ids"]

    # Generate responses with automatic distribution
    all_outputs = self.strategy.gather_and_generate(
        sampling_params=sampling_params,
        all_prompt_token_ids=all_prompt_token_ids,
        sleep_engine=True  # Automatically sleep engine after generation
    )

    if dist.get_rank(self.vllm_mp_group) == 0:
        self.strategy.print(f"Generated {len(all_outputs)} outputs")

Multimodal Generation (VLM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For vision-language models with images:

.. code-block:: python

    # Generate with multimodal inputs
    all_outputs = self.strategy.gather_and_generate(
        sampling_params=sampling_params,
        all_prompts=all_prompts,        # Text prompts
        all_images=all_images,          # Image data
        images_num=images_num,          # Number of images per prompt
        sleep_engine=True
    )

How ``gather_and_generate()`` Works
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The method performs the following operations:

1. **Gather**: Collects prompts from all ranks within the tensor-parallel group to rank 0

   * Example: With ``world_size=8`` and ``engine_tp_size=4``, ranks [0,1,2,3] gather to rank 0, and ranks [4,5,6,7] gather to rank 4

2. **Generate**: Executes inference using the vLLM/SGLang engine on the gathered prompts

3. **Distribute**: Scatters the generated outputs back to the originating ranks in the same order

4. **Sleep Management**: Automatically handles engine sleep/wake cycles based on the ``sleep_engine`` parameter

.. note::
   Users don't need to manually manage engine sleep states when using this interface.

Required Arguments
------------------

Add LightRFT-specific arguments to your argument parser:

.. code-block:: python

    from lightrft.utils import add_arguments
    import argparse

    # Create parser
    parser = argparse.ArgumentParser()

    # Add LightRFT arguments
    add_arguments(parser)

    # Parse arguments
    args = parser.parse_args()

Key Arguments
~~~~~~~~~~~~~

**Inference Engine Configuration:**

.. code-block:: bash

    --engine_tp_size 4              # Tensor parallelism size for inference engine
    --engine_mem_util 0.85          # GPU memory utilization for KV cache (0.0-1.0)
    --engine_type vllm              # Engine type: 'vllm' or 'sglang'
    --enable_engine_sleep           # Enable engine sleep mode (default: True)
    --disable_engine_sleep          # Disable engine sleep mode

**Distributed Training:**

.. code-block:: bash

    --fsdp                          # Use FSDP instead of DeepSpeed
    --zero_stage 2                  # DeepSpeed ZeRO stage (1, 2, or 3)
    --fsdp_cpu_offload              # Offload FSDP optimizer states to CPU
    --adam_offload                  # Offload Adam optimizer states
    --sp_size 2                     # Sequence parallelism size

**Training Optimization:**

.. code-block:: bash

    --packing_samples               # Pack multiple samples into sequences
    --use_mp_opt                    # Use mixed precision optimizer (FSDP)
    --fused_linear_logprob          # Fused linear layer and logprob computation
    --chunk_size 4096               # Chunk size for fused operations

**Monitoring:**

.. code-block:: bash

    --log_dir ./logs                # Directory for logs and visualizations
    --plot_every 10                 # Plot generation length distribution every N steps

Strategy Implementation Details
--------------------------------

Available Strategies
~~~~~~~~~~~~~~~~~~~~

LightRFT provides two main strategy implementations:

1. **DeepspeedStrategy** (default)

   * Uses DeepSpeed ZeRO for memory-efficient training
   * Configurable ZeRO stages (1, 2, or 3)
   * Supports gradient accumulation and mixed precision
   * Best for: General RLHF training, well-established workflows

2. **FSDPV2Strategy** (when ``--fsdp`` is set)

   * Uses PyTorch's Fully Sharded Data Parallel
   * Supports CPU offloading for optimizer states
   * Native PyTorch implementation with better integration
   * Best for: Maximum memory efficiency, PyTorch-native workflows

Strategy Selection Logic
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # In get_strategy() function
    if args.fsdp:
        strategy = FSDPV2Strategy(...)
    else:
        strategy = DeepspeedStrategy(...)

Engine Sleep/Wake Mechanism
----------------------------

The strategy provides automatic memory management through engine sleep modes:

.. code-block:: python

    # Engine lifecycle management
    strategy.setup_inference_engine(args, engine_type='vllm')  # Creates and wakes engine
    strategy.maybe_sleep_inference_engine()                     # Sleep to save memory
    strategy.wakeup_inference_engine()                          # Wake for generation

.. important::
   When using ``gather_and_generate()`` with ``sleep_engine=True``, the sleep/wake cycle is handled automatically.

Configuration Examples
----------------------

High-Throughput Setup (8 GPUs, DeepSpeed)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Using DeepSpeed ZeRO-2 with large tensor parallelism
    python train.py \
        --zero_stage 2 \
        --engine_tp_size 4 \
        --engine_mem_util 0.9 \
        --enable_engine_sleep \
        --micro_train_batch_size 1 \
        --train_batch_size 128

Memory-Efficient Setup (8 GPUs, FSDP with CPU Offload)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Using FSDP with CPU offloading for maximum memory efficiency
    python train.py \
        --fsdp \
        --fsdp_cpu_offload \
        --use_mp_opt \
        --engine_tp_size 2 \
        --engine_mem_util 0.85 \
        --enable_engine_sleep \
        --micro_train_batch_size 1 \
        --train_batch_size 64

Vision-Language Model Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Training VLMs with multimodal data
    python train_vl.py \
        --fsdp \
        --engine_tp_size 4 \
        --mixed_mm_data \
        --packing_samples \
        --enable_engine_sleep \
        --plot_every 20

Best Practices
--------------

1. Tensor Parallelism Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Set ``engine_tp_size`` to match your model size and GPU count
* For 7B models: ``engine_tp_size=1`` or ``2``
* For 13B-70B models: ``engine_tp_size=4`` or ``8``
* Ensure ``world_size % engine_tp_size == 0``

2. Memory Management
~~~~~~~~~~~~~~~~~~~~

* Enable engine sleep mode for memory-constrained setups: ``--enable_engine_sleep``
* Adjust ``engine_mem_util`` based on available memory (0.5-0.9)
* Use FSDP with CPU offload for maximum memory savings: ``--fsdp --fsdp_cpu_offload``

3. Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Use ``--packing_samples`` for varied sequence lengths
* Enable ``--fused_linear_logprob`` for large vocabulary models
* Set appropriate ``micro_train_batch_size`` to saturate GPU utilization

4. Debugging and Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Use ``--plot_every`` with ``--log_dir`` to track generation length distribution
* Monitor memory with ``strategy.report_memory(prefix="checkpoint_name")``
* Check engine status with ``strategy.inference_engine_status``

Advanced Features
-----------------

Sequence Parallelism
~~~~~~~~~~~~~~~~~~~~

Enable sequence parallelism for very long sequences:

.. code-block:: bash

    # In arguments
    --sp_size 4  # Split sequence across 4 GPUs

The strategy automatically creates sequence-parallel groups and handles communication.

Custom Reward Models
~~~~~~~~~~~~~~~~~~~~

For multiple reward models or remote reward APIs:

.. code-block:: python

    # Multiple reward models
    reward_models = [reward_model_1, reward_model_2, reward_model_3]
    strategy = get_strategy(args)

    # Models are automatically sharded across GPUs
    prepared_rms = [strategy.prepare_model(rm, shard_size=8) for rm in reward_models]

Mixed Precision Training
~~~~~~~~~~~~~~~~~~~~~~~~

Control mixed precision behavior:

.. code-block:: bash

    # Enable BF16 training
    --bf16

    # Use mixed precision optimizer (FSDP)
    --use_mp_opt

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Issue**: Out of memory during generation

* **Solution**: Reduce ``engine_mem_util`` or increase ``engine_tp_size``

**Issue**: Engine not updating with new weights

* **Solution**: Ensure ``update_engine_weights()`` is called after training

**Issue**: Slow generation speed

* **Solution**: Increase ``micro_rollout_batch_size`` or reduce ``engine_tp_size``

**Issue**: FSDP optimizer offload errors

* **Solution**: Verify you're using FSDP strategy (``--fsdp``) and calling offload/load in pairs

API Reference
-------------

For detailed API documentation, see:

* ``lightrft.strategy.strategy_base.StrategyBase`` - Base strategy class
* ``lightrft.strategy.get_strategy()`` - Strategy factory function
* ``lightrft.utils.add_arguments()`` - Argument configuration
