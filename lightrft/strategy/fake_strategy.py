"""
FakeStrategy for testing LightRFT without distributed environment.

This module provides a FakeStrategy class that mimics the behavior of real
distributed training strategies (DeepSpeed, FSDP) but runs in a single process
without actual distributed communication. This is useful for unit testing and
development environments where distributed setup is not available.
"""

import os
import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import List, Tuple, Union
from contextlib import contextmanager

from lightrft.strategy import StrategyConfig, StrategyBase

ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]


class FakeStrategy(StrategyBase):
    """
    Fake strategy for testing without distributed environment.

    This strategy provides the same API as real distributed strategies but
    runs everything in a single process without actual distributed communication.
    It's useful for unit testing and development.

    :param seed: Random seed for reproducibility
    :type seed: int
    :param max_norm: Maximum gradient norm for clipping
    :type max_norm: float
    :param micro_train_batch_size: Batch size for each training step
    :type micro_train_batch_size: int
    :param train_batch_size: Total batch size for training
    :type train_batch_size: int
    :param args: Additional configuration arguments
    :type args: Any
    """
    def __init__(
        self,
        seed: int = 42,
        max_norm: float = 1.0,
        micro_train_batch_size: int = 1,
        train_batch_size: int = 128,
        args=None,
    ) -> None:
        """
        Initialize fake strategy with common parameters.
        """
        super().__init__(seed, max_norm, micro_train_batch_size, train_batch_size, args)

        # Override distributed setup for fake environment
        self.world_size = 1
        self.accumulated_gradient = (
            self.train_batch_size * self.ring_attn_size // self.micro_train_batch_size // self.world_size
        )

        self.print("FakeStrategy Initialized (single process mode)")

    def setup_distributed(self, timeout=None, num_gpu_per_node=8) -> None:
        """
        Fake distributed setup - does nothing in single process mode.

        :param timeout: Maximum time to wait for initialization (ignored)
        :type timeout: timedelta, optional
        :param num_gpu_per_node: Number of GPUs per node (ignored)
        :type num_gpu_per_node: int
        """
        self.set_seed(self.seed)

        # Set local rank to 0 for single process
        if hasattr(self.config, 'local_rank'):
            self.config.local_rank = 0
        else:
            # For backward compatibility with args
            if hasattr(self, 'args') and self.args is not None:
                self.args.local_rank = 0

        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

        self.world_size = 1
        self.print("FakeStrategy: Running in single process mode")

    def create_optimizer(self, model: nn.Module, **kwargs) -> Optimizer:
        """
        Create a standard optimizer for the model.

        :param model: The model to optimize
        :type model: nn.Module
        :param kwargs: Additional optimizer arguments

        :return: The created optimizer
        :rtype: Optimizer
        """
        if hasattr(model, "is_actor") and model.is_actor:
            model = model.model

        # Use AdamW as default optimizer
        return torch.optim.AdamW(model.parameters(), **kwargs)

    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        """
        Prepare models and optimizers - returns them as-is in fake mode.

        :param models_or_model_optim_pairs: Models or (model, optimizer) pairs to prepare

        :return: Prepared models/optimizers (unchanged in fake mode)
        :rtype: Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]
        """
        ret = []
        for arg in models_or_model_optim_pairs:
            ret.append(arg)

        return ret[0] if len(ret) == 1 else ret

    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: Optimizer, **kwargs) -> None:
        """
        Perform backward pass using standard PyTorch.

        :param loss: The loss to backpropagate
        :type loss: torch.Tensor
        :param model: The model
        :type model: nn.Module
        :param optimizer: The optimizer
        :type optimizer: Optimizer
        :param kwargs: Additional arguments
        """
        loss.backward()

    def optimizer_step(self, optimizer: Optimizer, model: nn.Module, scheduler=None, name="model", **kwargs) -> None:
        """
        Take optimizer step using standard PyTorch.

        :param optimizer: The optimizer
        :type optimizer: Optimizer
        :param model: The model
        :type model: nn.Module
        :param scheduler: The learning rate scheduler (optional)
        :param name: Name for logging purposes
        :type name: str
        :param kwargs: Additional arguments
        """
        # Apply gradient clipping if max_norm is set
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)

        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

    def save_ckpt(
        self, model, save_dir: str, tag=None, max_num=3, max_mem=1000, client_state={}, save_latest=True
    ) -> None:
        """
        Save checkpoint using standard PyTorch saving.

        :param model: The model to save
        :param save_dir: Directory to save the checkpoint
        :type save_dir: str
        :param tag: Optional tag for the checkpoint
        :param max_num: Maximum number of checkpoints to keep
        :type max_num: int
        :param max_mem: Maximum memory in MB for checkpoints (ignored)
        :type max_mem: int
        :param client_state: Additional state to save
        :type client_state: dict
        :param save_latest: Whether to save as latest checkpoint
        :type save_latest: bool
        """
        os.makedirs(save_dir, exist_ok=True)

        if tag is None:
            tag = "checkpoint"

        checkpoint_path = os.path.join(save_dir, f"{tag}.pt")

        # Save model state
        if hasattr(model, "state_dict"):
            checkpoint = {'model_state_dict': model.state_dict(), 'client_state': client_state}
            torch.save(checkpoint, checkpoint_path)
            self.print(f"FakeStrategy: Saved checkpoint to {checkpoint_path}")

    def load_ckpt(
        self,
        model,
        load_dir: str,
        tag=None,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
        load_module_only=False,
    ):
        """
        Load checkpoint using standard PyTorch loading.

        :param model: The model to load checkpoint into
        :param load_dir: Directory containing the checkpoint
        :type load_dir: str
        :param tag: Optional specific checkpoint tag to load
        :param load_module_strict: Whether to use strict loading for module states
        :type load_module_strict: bool
        :param load_optimizer_states: Whether to load optimizer states
        :type load_optimizer_states: bool
        :param load_lr_scheduler_states: Whether to load learning rate scheduler states
        :type load_lr_scheduler_states: bool
        :param load_module_only: Whether to load only the module states
        :type load_module_only: bool

        :return: Tuple of (load_path, client_states)
        :rtype: tuple
        """
        if tag is None:
            tag = "checkpoint"

        checkpoint_path = os.path.join(load_dir, f"{tag}.pt")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=load_module_strict)

        client_states = checkpoint.get('client_state', {})

        self.print(f"FakeStrategy: Loaded checkpoint from {checkpoint_path}")
        return checkpoint_path, client_states

    def all_reduce(self, data, op="mean"):
        """
        Fake all-reduce operation - returns data unchanged.

        :param data: Data to be reduced
        :type data: Union[torch.Tensor, dict]
        :param op: Reduction operation (ignored in fake mode)
        :type op: str

        :return: Data unchanged
        :rtype: Union[torch.Tensor, dict]
        """
        return data

    def all_gather(self, data):
        """
        Fake all-gather operation - returns data wrapped in list.

        :param data: Data to be gathered
        :type data: Union[torch.Tensor, dict]

        :return: Data wrapped to mimic gathered result
        :rtype: Union[torch.Tensor, dict]
        """
        if isinstance(data, dict):
            return {k: [v] for k, v in data.items()}
        else:
            return [data]

    @classmethod
    def is_rank_0(cls) -> bool:
        """
        Always returns True in fake mode (single process is rank 0).

        :return: True
        :rtype: bool
        """
        return True

    def get_rank(self) -> int:
        """
        Always returns 0 in fake mode (single process).

        :return: 0
        :rtype: int
        """
        return 0

    def setup_inference_engine(self, args, engine_type="vllm", actor=None):
        """
        Fake inference engine setup - returns None.

        :param args: Configuration arguments
        :type args: argparse.Namespace
        :param engine_type: Type of inference engine (ignored)
        :type engine_type: str
        :param actor: The actor module (ignored)
        :type actor: torch.nn.Module

        :return: None
        :rtype: None
        """
        self.print("FakeStrategy: Inference engine setup skipped")
        return None

    def maybe_sleep_inference_engine(self):
        """
        Fake inference engine sleep - does nothing.
        """
        self.print("FakeStrategy: Inference engine sleep skipped")

    def wakeup_inference_engine(self):
        """
        Fake inference engine wakeup - does nothing.
        """
        self.print("FakeStrategy: Inference engine wakeup skipped")

    def engine_generate_local(self, sampling_params, prompt_token_ids=None, multi_modal_inputs=None):
        """
        Fake generation - returns empty results.

        :param sampling_params: Parameters for generation (ignored)
        :param prompt_token_ids: Prompt token IDs (ignored)
        :param multi_modal_inputs: Multimodal inputs (ignored)

        :return: Empty list
        :rtype: List
        """
        self.print("FakeStrategy: Generation skipped")
        return []

    def gather_and_generate(
        self,
        sampling_params,
        all_prompt_token_ids=None,
        all_prompts=None,
        all_images=None,
        sleep_engine=True,
        images_num=None
    ):
        """
        Fake gather and generate - returns empty results.

        :param sampling_params: Parameters for generation (ignored)
        :param all_prompt_token_ids: All prompt token IDs (ignored)
        :param all_prompts: All prompts (ignored)
        :param all_images: All images (ignored)
        :param sleep_engine: Whether to sleep engine after generation (ignored)
        :type sleep_engine: bool
        :param images_num: Number of images (ignored)

        :return: Empty list
        :rtype: List
        """
        self.print("FakeStrategy: Gather and generate skipped")
        return []

    def update_engine_weights(self, actor):
        """
        Fake engine weight update - does nothing.

        :param actor: The actor model (ignored)
        """
        self.print("FakeStrategy: Engine weight update skipped")

    @contextmanager
    def init_model_context(self):
        """
        Fake model initialization context - does nothing.
        """
        try:
            yield
        finally:
            self.print("FakeStrategy: Model initialization context finished")

    def maybe_offload_optimizer(self, optimizer):
        """
        Fake optimizer offloading - returns optimizer unchanged.

        :param optimizer: The optimizer to potentially offload
        :type optimizer: torch.optim.Optimizer

        :return: The original optimizer
        :rtype: torch.optim.Optimizer
        """
        return optimizer

    def maybe_load_optimizer(self, optimizer, device=torch.cuda.current_device()):
        """
        Fake optimizer loading - returns optimizer unchanged.

        :param optimizer: The optimizer to potentially load
        :type optimizer: torch.optim.Optimizer
        :param device: Target device for loading (ignored)
        :type device: torch.device

        :return: The original optimizer
        :rtype: torch.optim.Optimizer
        """
        return optimizer


def get_fake_strategy(args=None):
    """
    Create and return a FakeStrategy instance.

    This is a convenience function similar to get_strategy() but for fake strategy.

    :param args: Configuration arguments
    :type args: object

    :return: A FakeStrategy instance
    :rtype: FakeStrategy
    """
    config = StrategyConfig.from_args(args) if args is not None else StrategyConfig()

    return FakeStrategy(
        seed=config.seed,
        max_norm=config.max_norm,
        micro_train_batch_size=config.micro_train_batch_size,
        train_batch_size=config.train_batch_size,
        args=args,
    )
