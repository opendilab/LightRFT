"""
Configuration dataclasses for LightRFT strategy module.

This module provides typed configuration objects to replace the use of getattr
for accessing configuration parameters, improving type safety and code clarity.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Dict
import torch
import dataclasses


@dataclass
class StrategyConfig:
    """Base configuration for all training strategies."""

    # Basic training parameters
    # (int): Random seed, defaults to 42
    seed: int = 42
    # (float): Maximum gradient norm for clipping, defaults to 1.0
    max_norm: float = 1.0
    # (int): Micro batch size for training, defaults to 1
    micro_train_batch_size: int = 1
    # (int): Training batch size, defaults to 128
    train_batch_size: int = 128
    # (bool): Use bfloat16 precision, defaults to True
    bf16: bool = True

    # DeepSpeed specific
    # (int): DeepSpeed Zero optimization stage, defaults to 2
    zero_stage: int = 2

    # FSDP specific
    # (bool): Use FSDP (Fully Sharded Data Parallel), defaults to False
    fsdp: bool = False
    # (bool): Enable FSDP CPU offload, defaults to False
    fsdp_cpu_offload: bool = False

    # Common distributed training parameters
    # (bool): Offload Adam optimizer states, defaults to False
    adam_offload: bool = False
    # (int): ZeRO parallel group size, defaults to 1
    zpg: int = 1
    # (Optional[str]): Gradient accumulation data type, defaults to None
    grad_accum_dtype: Optional[str] = None
    # (bool): Overlap communication and computation, defaults to False
    overlap_comm: bool = False
    # (int): Ring attention size, defaults to 1
    ring_attn_size: int = 1

    # Engine and inference parameters
    # (str): Inference engine type, defaults to "vllm"
    engine_type: str = "vllm"
    # (int): Engine tensor parallelism size, defaults to 1
    engine_tp_size: int = 1
    # (bool): Enable engine sleep mode, defaults to False
    enable_engine_sleep: bool = False
    # (int): Local rank for distributed training, defaults to -1
    local_rank: int = -1

    # Sequence parallel parameters
    # (int): Sequence parallelism size, defaults to 1
    sp_size: int = 1

    # Model parameters
    # (float): Actor model learning rate, defaults to 1e-5
    actor_learning_rate: float = 1e-5
    # (float): Critic model learning rate, defaults to 1e-5
    critic_learning_rate: float = 1e-5
    # (tuple): Adam optimizer beta parameters, defaults to (0.9, 0.95)
    adam_betas: tuple = (0.9, 0.95)
    # (float): L2 regularization coefficient, defaults to 0.0
    l2: float = 0.0
    # (float): Learning rate warmup ratio, defaults to 0.03
    lr_warmup_ratio: float = 0.03

    # Training control
    # (bool): Pretrain critic model, defaults to False
    critic_pretrain: bool = False
    # (Optional[str]): Remote reward model URL, defaults to None
    remote_rm_url: Optional[str] = None
    # (Optional[str]): Pretraining data path, defaults to None
    pretrain_data: Optional[str] = None
    # (bool): Use fused linear layer and logprob computation, defaults to False
    fused_linear_logprob: bool = False

    # Reward and advantage processing
    # (bool): Apply running normalization to rewards, defaults to False
    reward_running_norm: bool = False
    # (bool): When reward_running_norm is True, subtract mean during normalization, defaults to False
    reward_running_norm_minus_mean: bool = False
    # (bool): Normalize advantages, defaults to False
    advantages_norm: bool = False
    # (float): Clip advantages to this value, 0 means no clipping, defaults to 0.0
    advantage_clip: float = 0.0
    # (float): Clip rewards to this value, 0 means no clipping, defaults to 0.0
    reward_clip: float = 0.0

    # Experience generation parameters
    # (int): Batch size for micro rollout during experience generation, defaults to 1
    micro_rollout_batch_size: int = 2
    # (int): Number of samples to generate per prompt, defaults to 1
    n_samples_per_prompt: int = 8

    # Overlong sequence handling
    # (bool): Enable overlong sequence buffer penalty, defaults to False
    overlong_buffer: bool = False
    # (int): Buffer length for overlong sequence penalty calculation, defaults to 0
    overlong_buffer_len: int = 1024
    # (float): Penalty factor for overlong sequences, defaults to 1.0
    overlong_buffer_penalty_factor: float = 1.0

    # Dynamic sampling and advantage estimation
    # (bool): Enable dynamic sampling for advantage estimation, defaults to False
    dynamic_sampling: bool = False
    # (str): Advantage estimator method, defaults to "gae"
    advantage_estimator: str = "group_norm"

    # KL loss and estimation
    # (bool): Use KL loss in training, defaults to False
    use_kl_loss: bool = False
    # (str): KL divergence estimator method, defaults to "mean"
    kl_estimator: str = "k3"

    # FSDP specific parameters
    # (bool): Use mixed precision matrix multiplication data, defaults to False
    mixed_mm_data: bool = False
    # (bool): Use model parallel optimizer, defaults to False
    use_mp_opt: bool = False

    # Analysis and monitoring
    # (int): Plot interval steps, defaults to -1
    plot_every: int = -1
    # (bool): Use TensorBoard for logging, defaults to False
    use_tensorboard: bool = False

    # Additional arguments for backward compatibility
    # (Dict[str, Any]): Extra arguments for backward compatibility, defaults to {}
    extra_args: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_args(cls, args_dict) -> 'StrategyConfig':
        """
        Create StrategyConfig from argparse.Namespace or similar object.

        This method provides backward compatibility by extracting parameters
        that were previously accessed via getattr, ensuring smooth migration
        from legacy configuration systems.

        :param args_dict: Configuration arguments object containing training parameters
        :type args_dict: object
        :return: StrategyConfig instance with extracted parameters
        :rtype: StrategyConfig

        Example::

            # From argparse.Namespace
            args = argparse.Namespace(
                seed=42,
                max_norm=1.0,
                micro_train_batch_size=1,
                # ... other parameters
            )
            config = StrategyConfig.from_args(args)

            # From dictionary
            args_dict = {
                'seed': 42,
                'max_norm': 1.0,
                'micro_train_batch_size': 1,
                # ... other parameters
            }
            config = StrategyConfig.from_args(args_dict)
        """
        # Extract all known parameters with their default values
        config = cls()

        # Get all field names from the dataclass (excluding 'extra_args')
        field_names = [field.name for field in dataclasses.fields(cls) if field.name != 'extra_args']

        # Automatically assign values using getattr/hasattr
        for field_name in field_names:
            if hasattr(args_dict, field_name):
                setattr(config, field_name, getattr(args_dict, field_name))

        # Store original args for backward compatibility
        config.extra_args = {k: v for k, v in vars(args_dict).items() if not hasattr(config, k)}

        config.print_config_summary()

        return config

    def print_config_summary(self) -> None:
        """
        Print a summary of the configuration for verification.

        This method shows which parameters were overridden from defaults
        and which are using default values.
        """
        # Only print on rank 0 GPU to avoid duplicate output in distributed training
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        print("=" * 60)
        print("StrategyConfig Configuration Verification Result")
        print("=" * 60)

        # Define default configuration for comparison
        default_config = StrategyConfig()

        print("\nConfiguration Parameters Details:")
        print("-" * 40)

        # Basic Training Parameters
        print("Basic Training Parameters:")
        for attr in ['seed', 'max_norm', 'micro_train_batch_size', 'train_batch_size', 'bf16']:
            current = getattr(self, attr)
            default = getattr(default_config, attr)
            status = "Overridden" if current != default else "Default"
            print(f"  {attr}: {current} ({status})")

        # Distributed Training Parameters
        print("\nDistributed Training Parameters:")
        for attr in [
            'zero_stage', 'fsdp', 'fsdp_cpu_offload', 'adam_offload', 'zpg', 'grad_accum_dtype', 'overlap_comm',
            'ring_attn_size'
        ]:
            current = getattr(self, attr)
            default = getattr(default_config, attr)
            status = "Overridden" if current != default else "Default"
            print(f"  {attr}: {current} ({status})")

        # Engine and Inference Parameters
        print("\nEngine and Inference Parameters:")
        for attr in ['engine_type', 'engine_tp_size', 'enable_engine_sleep', 'local_rank', 'sp_size']:
            current = getattr(self, attr)
            default = getattr(default_config, attr)
            status = "Overridden" if current != default else "Default"
            print(f"  {attr}: {current} ({status})")

        # Model Parameters
        print("\nModel Parameters:")
        for attr in ['actor_learning_rate', 'critic_learning_rate', 'adam_betas', 'l2', 'lr_warmup_ratio']:
            current = getattr(self, attr)
            default = getattr(default_config, attr)
            status = "Overridden" if current != default else "Default"
            print(f"  {attr}: {current} ({status})")

        # Training Control Parameters
        print("\nTraining Control Parameters:")
        for attr in ['critic_pretrain', 'remote_rm_url', 'pretrain_data', 'fused_linear_logprob']:
            current = getattr(self, attr)
            default = getattr(default_config, attr)
            status = "Overridden" if current != default else "Default"
            print(f"  {attr}: {current} ({status})")

        # Reward and Advantage Processing Parameters
        print("\nReward and Advantage Processing Parameters:")
        for attr in [
            'reward_running_norm', 'reward_running_norm_minus_mean', 'advantages_norm', 'advantage_clip', 'reward_clip'
        ]:
            current = getattr(self, attr)
            default = getattr(default_config, attr)
            status = "Overridden" if current != default else "Default"
            print(f"  {attr}: {current} ({status})")

        # Experience Generation Parameters
        print("\nExperience Generation Parameters:")
        for attr in ['micro_rollout_batch_size', 'n_samples_per_prompt']:
            current = getattr(self, attr)
            default = getattr(default_config, attr)
            status = "Overridden" if current != default else "Default"
            print(f"  {attr}: {current} ({status})")

        # Overlong Sequence Handling Parameters
        print("\nOverlong Sequence Handling Parameters:")
        for attr in ['overlong_buffer', 'overlong_buffer_len', 'overlong_buffer_penalty_factor']:
            current = getattr(self, attr)
            default = getattr(default_config, attr)
            status = "Overridden" if current != default else "Default"
            print(f"  {attr}: {current} ({status})")

        # Dynamic Sampling and Advantage Estimation Parameters
        print("\nDynamic Sampling and Advantage Estimation Parameters:")
        for attr in ['dynamic_sampling', 'advantage_estimator']:
            current = getattr(self, attr)
            default = getattr(default_config, attr)
            status = "Overridden" if current != default else "Default"
            print(f"  {attr}: {current} ({status})")

        # KL Loss and Estimation Parameters
        print("\nKL Loss and Estimation Parameters:")
        for attr in ['use_kl_loss', 'kl_estimator']:
            current = getattr(self, attr)
            default = getattr(default_config, attr)
            status = "Overridden" if current != default else "Default"
            print(f"  {attr}: {current} ({status})")

        # FSDP Specific Parameters
        print("\nFSDP Specific Parameters:")
        for attr in ['mixed_mm_data', 'use_mp_opt']:
            current = getattr(self, attr)
            default = getattr(default_config, attr)
            status = "Overridden" if current != default else "Default"
            print(f"  {attr}: {current} ({status})")

        # Analysis and Monitoring Parameters
        print("\nAnalysis and Monitoring Parameters:")
        for attr in ['plot_every', 'use_tensorboard']:
            current = getattr(self, attr)
            default = getattr(default_config, attr)
            status = "Overridden" if current != default else "Default"
            print(f"  {attr}: {current} ({status})")

        # extra_args
        if self.extra_args:
            print("\nExtra Parameters (extra_args):")
            for key, value in self.extra_args.items():
                print(f"  {key}: {value}")
        else:
            print("\nExtra Parameters: None")

        print("\n" + "=" * 60)
        print("Configuration verification completed!")
        print("=" * 60)
