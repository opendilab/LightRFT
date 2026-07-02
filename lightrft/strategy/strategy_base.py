"""
A module for implementing training strategies in deep learning, particularly for RLVR and RLHF.

This module provides base classes and utilities for different training strategies like DeepSpeed and FSDP.
It handles distributed training setup, model/optimizer preparation, checkpointing, and inference engine management.
"""

import math
import os
import re
import random
import time
from loguru import logger
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import deepspeed
import numpy as np
import torch
from easydict import EasyDict
from torch import distributed as dist
from torch import nn, optim
from torch.distributed.device_mesh import init_device_mesh
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers.trainer import get_scheduler

from lightrft.strategy.utils.distributed_util import gather_inputs_object_for_inference, create_sub_group
from lightrft.strategy.utils.broadcast_utils import BroadcastManager
from lightrft.strategy.utils.data_utils import DistributedSampler
from lightrft.strategy.utils.parallel_utils import (
    SPDataProcessor,
    get_sequence_parallel_group,
    set_sequence_parallel_group,
)
from lightrft.strategy.utils.statistic import GenLenAnalyser
from lightrft.strategy.config import StrategyConfig

ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]


class EngineStatus(Enum):
    """
    Enum class for inference engine status.

    :cvar SLEEPED: Engine is in sleep mode
    :cvar WAKEUP: Engine is awake and ready
    """

    SLEEPED = 0
    WAKEUP = 1


class StrategyBase(ABC):
    """
    Base class for training strategies (DeepSpeed and FSDP).

    Provides common functionality for distributed training setup, model preparation,
    optimization, checkpointing, and inference engine management.

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

    def __init__(  # pylint: disable=R0917
        self, seed: int, max_norm: float, micro_train_batch_size: int, train_batch_size: int, args: Optional[Any] = None
    ) -> None:
        """
        Initialize strategy with common parameters.

        :param seed: Random seed for reproducibility
        :type seed: int
        :param max_norm: Maximum gradient norm for clipping
        :type max_norm: float
        :param micro_train_batch_size: Batch size for each training step
        :type micro_train_batch_size: int
        :param train_batch_size: Total batch size for training
        :type train_batch_size: int
        :param args: Additional configuration arguments
        :type args: Any (usually argparse.Namespace)
        """
        self.seed = seed
        self.max_norm = max_norm
        self.micro_train_batch_size = micro_train_batch_size
        self.train_batch_size = train_batch_size
        self.args = args

        # Create config object for typed parameter access
        self.config = StrategyConfig.from_args(args) if args is not None else StrategyConfig()

        # Use config object
        self.adam_offload = self.config.adam_offload
        self.zpg = self.config.zpg
        self.grad_accum_dtype = self.config.grad_accum_dtype
        self.overlap_comm = self.config.overlap_comm

        # inference (rollout) engine related
        self.inference_engine = None
        self.inference_engine_status = EngineStatus.SLEEPED
        self.inference_tokenizer = None
        self.inference_processor = None
        self.broadcast_manager = None
        self.rollout_train_actor = None
        self.rollout_train_actor_is_on_gpu = False
        self.use_separate_hf_rollout_actor = False
        self._separate_hf_rollout_sync_param_pairs = None
        self._separate_hf_rollout_sync_buffer_pairs = None
        self.last_separate_hf_rollout_sync_stats = None

        self.time_steps = defaultdict(int)

        self._profile_step = 0

        # initialize distributed environment
        self.setup_distributed(timeout=timedelta(minutes=60))
        # NOTE: this group is not used by vllm, only used in strategy
        self.engine_mp_group, self.engine_dp_group = create_sub_group(self.config.engine_tp_size)

        # initialize sequence parallel data processor
        self.sp_data_processor = SPDataProcessor()

        self.genlen_analyser = GenLenAnalyser(
            self.engine_dp_group,
            plot_every=self.config.plot_every,
            plot_out_dir=self.config.use_tensorboard,
        )

    def set_seed(self, seed: int) -> None:
        """
        Set random seeds for reproducibility.

        :param seed: The random seed to use
        :type seed: int
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_distributed(self, timeout: Optional[timedelta] = None, num_gpu_per_node: int = 8) -> None:
        """
        Initialize distributed training environment.

        :param timeout: Maximum time to wait for initialization
        :type timeout: timedelta, optional
        :param num_gpu_per_node: Number of GPUs per node
        :type num_gpu_per_node: int
        :raises RuntimeError: If required environment variables are missing
        :raises ValueError: If unsupported engine type is specified
        """
        self.set_seed(self.seed)

        if self.config.local_rank == -1 and "LOCAL_RANK" in os.environ:  # for slurm
            self.config.local_rank = int(os.environ["LOCAL_RANK"])
        elif "RANK" in os.environ:
            rank = int(os.environ["RANK"])
            self.config.local_rank = rank % num_gpu_per_node
        if self.config.local_rank != -1:
            torch.cuda.set_device(self.config.local_rank)
        self.engine_type = self.config.engine_type

        enable_fsdp = self.config.fsdp

        if enable_fsdp:
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            try:
                rank = int(os.environ["RANK"])
                world_size = int(os.environ["WORLD_SIZE"])
            except KeyError as e:
                raise RuntimeError(f"Could not find {e} in the torch environment")

            # initialize the default process group
            host = os.getenv("MASTER_ADDR", "localhost")
            port = os.getenv("MASTER_PORT", "2222")
            init_method = f"tcp://{host}:{port}"
            if rank == 0:
                print(
                    f"Init Distributed Env, init_method:{init_method}, rank:{rank}, world_size:{world_size}, engine_type:{self.config.engine_type}"  # noqa
                )
            # TODO: unify the init_process_group for both vllm and sglang when stable version finished

            if self.config.engine_type in ("vllm", "sglang", "hf"):
                dist.init_process_group(
                    rank=rank,
                    world_size=world_size,
                    # here we set both cpu and cuda as backend, because we need to support
                    # both gpu and cpu training (e.g. FSDP and FSDP with cpu offload)
                    backend="cpu:gloo,cuda:nccl",
                    init_method=init_method,
                    timeout=timeout,
                )
            else:
                raise ValueError(f"Unsupported backend: {self.config.engine_type}")
        else:
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            if self.config.engine_type in ("vllm", "sglang", "hf"):
                deepspeed.init_distributed(dist_backend="nccl", timeout=timeout)
            else:
                raise ValueError(f"Unsupported backend: {self.config.engine_type}")

        self.world_size = dist.get_world_size()
        self.accumulated_gradient = (self.train_batch_size // self.micro_train_batch_size // self.world_size)

        if self.train_batch_size % (self.micro_train_batch_size * self.world_size) != 0:
            raise ValueError(
                f"train_batch_size must be divisible by (micro_train_batch_size * world_size)\n"
                f"  train_batch_size:        {self.train_batch_size}\n"
                f"  micro_train_batch_size:  {self.micro_train_batch_size}\n"
                f"  world_size:              {self.world_size}\n"
                f"  Required: {self.train_batch_size} % ({self.micro_train_batch_size} * {self.world_size}) == 0"
            )
        # initialize sequence parallel
        if self.config.sp_size > 1:
            assert self.world_size % self.config.sp_size == 0, "sp_size should be even divided by world size."
            dp_size = self.world_size // self.config.sp_size
            self.sp_mesh_device = init_device_mesh(
                "cuda", mesh_shape=(dp_size, self.config.sp_size), mesh_dim_names=["dp", "sp"]
            )
            set_sequence_parallel_group(self.sp_mesh_device["sp"].get_group())
            self.print(
                f"Init Sequence Parallel, sp_size:{self.config.sp_size}, \
                local_rank:{dist.get_rank(group=get_sequence_parallel_group())}",
            )

    @abstractmethod
    def create_optimizer(self, model: nn.Module, **kwargs) -> optim.Optimizer:
        """
        Create optimizer for the model.

        :param model: The model to optimize
        :type model: nn.Module
        :param kwargs: Additional optimizer arguments
        :return: The created optimizer
        :rtype: optim.Optimizer
        """
        raise NotImplementedError()

    def prepare(self,
                *models_or_model_optim_pairs: ModelOrModelOptimPair,
                is_rlhf=False) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        """
        Prepare models and optimizers for training.

        :param models_or_model_optim_pairs: Models or (model, optimizer) pairs to prepare
        :param is_rlhf: Whether preparing for RLHF training
        :type is_rlhf: bool
        :return: Prepared models/optimizers
        :rtype: Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]
        """
        raise NotImplementedError()

    @abstractmethod
    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, **kwargs) -> None:
        """
        Perform backward pass.

        :param loss: The loss to backpropagate
        :type loss: torch.Tensor
        :param model: The model
        :type model: nn.Module
        :param optimizer: The optimizer
        :type optimizer: optim.Optimizer
        :param kwargs: Additional arguments
        """
        raise NotImplementedError()

    @abstractmethod
    def optimizer_step(
        self, optimizer: optim.Optimizer, model: nn.Module, scheduler: Any, name: str = "model", **kwargs
    ) -> None:
        """
        Take optimizer step.

        :param optimizer: The optimizer
        :type optimizer: optim.Optimizer
        :param model: The model
        :type model: nn.Module
        :param scheduler: The learning rate scheduler
        :param name: Name for logging purposes
        :type name: str
        :param kwargs: Additional arguments
        """
        raise NotImplementedError()

    def setup_dataloader(
        self,
        replay_buffer,
        batch_size: int,
        pin_memory: bool = False,
        shuffle: bool = True,
        collate_fn: Optional[Callable] = None,
        drop_last: bool = True,
        sampler: Optional[Any] = None,
        consumed_samples: int = 0,
    ) -> DataLoader:
        """
        Set up data loader for training.

        :param replay_buffer: Dataset/replay buffer
        :param batch_size: Batch size
        :type batch_size: int
        :param pin_memory: Whether to pin memory
        :type pin_memory: bool
        :param shuffle: Whether to shuffle data
        :type shuffle: bool
        :param collate_fn: Function to collate samples
        :type collate_fn: Optional[Callable]
        :param drop_last: Whether to drop last incomplete batch
        :type drop_last: bool
        :param sampler: Custom sampler
        :param consumed_samples: Number of samples already consumed
        :type consumed_samples: int
        :return: Configured DataLoader
        :rtype: DataLoader
        """
        if sampler is None:
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
            sampler = DistributedSampler(
                replay_buffer,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle,
                seed=self.seed,
                drop_last=drop_last,
                consumed_samples=consumed_samples,
            )

        return DataLoader(
            replay_buffer,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )

    @abstractmethod
    def save_ckpt(  # pylint: disable=R0917, W0102
        self,
        model: nn.Module,
        save_dir: str,
        tag: Optional[str] = None,
        max_num: int = 3,
        max_mem: int = 1000,
        client_state: Optional[Dict[str, Any]] = None,
        save_latest: bool = True
    ) -> None:
        """
        Save training checkpoint with additional metadata.

        :param model: The model to save
        :param save_dir: Directory to save the checkpoint
        :type save_dir: str
        :param tag: Optional tag for the checkpoint
        :param max_num: Maximum number of checkpoints to keep, defaults to 3
        :type max_num: int
        :param max_mem: Maximum memory in MB for checkpoints, defaults to 1000
        :type max_mem: int
        :param client_state: Additional state to save, defaults to {}
        :type client_state: dict
        :param save_latest: Whether to save as latest checkpoint, defaults to True
        :type save_latest: bool
        """
        raise NotImplementedError()

    @abstractmethod
    def load_ckpt(  # pylint: disable=R0917
        self,
        model: nn.Module,
        load_dir: str,
        tag: Optional[str] = None,
        load_module_strict: bool = True,
        load_optimizer_states: bool = True,
        load_lr_scheduler_states: bool = True,
        load_module_only: bool = False,
    ) -> Any:
        """
        Load training checkpoint with various options.

        :param model: The model to load checkpoint into
        :param load_dir: Directory containing the checkpoint
        :type load_dir: str
        :param tag: Optional specific checkpoint tag to load
        :param load_module_strict: Whether to use strict loading for module states, defaults to True
        :type load_module_strict: bool
        :param load_optimizer_states: Whether to load optimizer states, defaults to True
        :type load_optimizer_states: bool
        :param load_lr_scheduler_states: Whether to load learning rate scheduler states, defaults to True
        :type load_lr_scheduler_states: bool
        :param load_module_only: Whether to load only the module states, defaults to False
        :type load_module_only: bool
        """
        raise NotImplementedError()

    def all_reduce(self,
                   data: Union[torch.Tensor, Dict[str, torch.Tensor]],
                   op: str = "mean") -> Union[torch.Tensor, Dict[str, torch.Tensor], float, int]:
        """
        Perform all-reduce operation across distributed processes.

        :param data: Data to be reduced, can be a tensor or dictionary of tensors
        :type data: Union[torch.Tensor, Dict[str, torch.Tensor]]
        :param op: Reduction operation ('mean', 'max', 'sum')
        :type op: str

        :return: Reduced data in the same format as input
        :rtype: Union[torch.Tensor, Dict[str, torch.Tensor], float, int]
        :raises AssertionError: If op is not one of 'mean', 'max', 'sum'
        """
        assert op in ("mean", "max", "sum")
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_reduce(v, op)
            return ret
        else:
            is_tensor = True
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
                is_tensor = False
            is_cpu_tensor = data.device.type == "cpu"

            if is_cpu_tensor:
                data = data.to(torch.cuda.current_device())
            if op == "mean":
                data /= self.world_size
            dist.all_reduce(data, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM)
            if is_cpu_tensor:
                data = data.cpu()
            return data.item() if not is_tensor else data

    def all_gather(self, data: Union[torch.Tensor, Dict[str,
                                                        torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Gather data from all distributed processes.

        :param data: Data to be gathered, can be a tensor or dictionary of tensors
        :type data: Union[torch.Tensor, dict]

        :return: Gathered data concatenated from all processes
        :rtype: Union[torch.Tensor, dict]
        """
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_gather(v)
            return ret
        else:
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
            is_cpu_tensor = data.device.type == "cpu"

            ret = [torch.zeros_like(data).to(torch.cuda.current_device()) for _ in range(self.world_size)]
            dist.all_gather(ret, data.to(torch.cuda.current_device()))
            return torch.cat(ret).cpu() if is_cpu_tensor else torch.cat(ret)

    @classmethod
    def print(cls, *msg):
        """
        Print messages with timestamp, but only on rank 0.

        :param msg: Messages to print
        """
        current_time = datetime.now()
        time_str = current_time.strftime("%m-%d %H:%M:%S")
        if cls.is_rank_0():
            print(f"[StrategyINFO {time_str}] ", *msg)

    @classmethod
    def is_rank_0(cls) -> bool:
        """
        Check if current process is rank 0.

        :return: True if current process is rank 0
        :rtype: bool
        """
        return dist.get_rank() == 0

    def get_rank(self) -> int:
        """
        Get current process rank.

        :return: Current process rank
        :rtype: int
        """
        return dist.get_rank()

    def unwrap_model(self, model) -> nn.Module:
        """
        Unwrap model from strategy-specific wrappers.

        :param model: Model to unwrap
        :type model: nn.Module

        :return: Unwrapped model
        :rtype: nn.Module
        """
        if hasattr(model, "module"):
            return model.module
        return model

    def prepare_models_and_optimizers(self, actor, critic, reward_models, initial_model, args, max_steps):
        """
        Prepare models, optimizers and schedulers for training.

        :param actor: Actor model
        :type actor: nn.Module
        :param critic: Critic model
        :type critic: nn.Module
        :param reward_models: Reward models
        :type reward_models: nn.Module
        :param initial_model: Initial model for reference
        :type initial_model: nn.Module
        :param args: Training arguments
        :type args: argparse.Namespace
        :param max_steps: Maximum training steps
        :type max_steps: int

        :return: Tuple of prepared models, optimizers, and schedulers
        :rtype: tuple
        """
        setattr(actor, "is_actor", True)

        fsdp_enable = self.config.fsdp

        def _resolve_fsdp_shard_size(preferred_shard_size: int) -> int:
            world_size = max(int(getattr(self, "world_size", 1)), 1)
            if world_size % preferred_shard_size == 0:
                return preferred_shard_size
            candidate = min(preferred_shard_size, world_size)
            while candidate > 1 and world_size % candidate != 0:
                candidate -= 1
            return max(candidate, 1)

        # For FSDP: wrap model first, then create optimizer
        if fsdp_enable:
            actor = self.prepare_model(actor, is_training=True)
            initial_model = self.prepare_model(initial_model)
            if critic is not None:
                critic = self.prepare_model(critic, is_training=True)
            if not self.config.remote_rm_url:
                reward_model_shard_size = _resolve_fsdp_shard_size(preferred_shard_size=8)
                self.print(
                    "Preparing reward model(s) with shard_size="
                    f"{reward_model_shard_size} (world_size={getattr(self, 'world_size', 1)})"
                )
                if isinstance(reward_models, (tuple, list)):
                    reward_models = [
                        self.prepare_model(model, shard_size=reward_model_shard_size) for model in reward_models
                    ]
                else:
                    reward_models = self.prepare_model(reward_models, shard_size=reward_model_shard_size)

        # Configure optimizers
        actor_optim = self.create_optimizer(
            actor, lr=self.config.actor_learning_rate, betas=self.config.adam_betas, weight_decay=self.config.l2
        )

        critic_optim = None
        if self.config.critic_pretrain:
            critic_optim = self.create_optimizer(
                critic, lr=self.config.critic_learning_rate, betas=self.config.adam_betas, weight_decay=self.config.l2
            )

        # Configure schedulers
        actor_scheduler = get_scheduler(
            "cosine_with_min_lr",
            actor_optim,
            num_warmup_steps=math.ceil(max_steps * self.config.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": self.config.actor_learning_rate * 0.1},
        )

        critic_scheduler = None
        if self.config.critic_pretrain:
            critic_scheduler = get_scheduler(
                "cosine_with_min_lr",
                critic_optim,
                num_warmup_steps=math.ceil(max_steps * self.config.lr_warmup_ratio),
                num_training_steps=max_steps,
                scheduler_specific_kwargs={"min_lr": self.config.critic_learning_rate * 0.1},
            )
        self.sync_and_clear_cache()
        # Prepare with strategy if not using FSDP
        if not fsdp_enable:
            return self.prepare(
                (actor, actor_optim, actor_scheduler),
                (critic, critic_optim, critic_scheduler),
                reward_models,
                initial_model,
                is_rlhf=True,
            )
        else:
            return (
                (actor, actor_optim, actor_scheduler),
                (critic, critic_optim, critic_scheduler),
                reward_models,
                initial_model,
            )

    def prepare_reward_model(
        self,
        reward_model: nn.Module,
        args=None,
        max_steps: int = int(1e8),
    ):
        """
        Prepare optimizers and schedulers for reward model training.

        :param reward_models: Reward models
        :type reward_models: nn.Module
        :param args: Training arguments
        :type args: argparse.Namespace
        :param max_steps: Maximum training steps
        :type max_steps: int

        :return: Tuple of prepared model, optimizer, and scheduler
        :rtype: tuple
        """
        fsdp_enable = args.fsdp
        # For FSDP: wrap model first, then create optimizer
        if fsdp_enable:
            reward_model = self.prepare_model(reward_model, is_training=True)

        # Configure optimizers
        reward_model_optim = self.create_optimizer(
            reward_model, lr=args.actor_learning_rate, betas=args.adam_betas, weight_decay=args.l2
        )

        # Configure schedulers
        reward_model_scheduler = get_scheduler(
            "cosine_with_min_lr",
            reward_model_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
        )

        self.sync_and_clear_cache()
        # Prepare with strategy if not using FSDP
        if not fsdp_enable:
            return self.prepare(
                (reward_model, reward_model_optim, reward_model_scheduler),
                is_rlhf=True,
            )
        else:
            # For FSDP: return wrapped model and optimizers
            return reward_model, reward_model_optim, reward_model_scheduler

    @classmethod
    def report_memory(cls, prefix=""):
        """
        Report GPU memory usage statistics.

        :param prefix: Prefix string for the memory report
        :type prefix: str
        """
        usable, total = torch.cuda.mem_get_info()
        used = round((total - usable) / 1e9, 2)
        if torch.distributed.get_rank() == 0:
            print(
                f"MEMORY STATUS: {prefix}, DRIVER_USED={used} GB, "
                f"ALLOCATED={torch.cuda.memory_allocated() / 1e9:.2f} GB"
            )

    def _uses_separate_hf_rollout_actor(self) -> bool:
        return (
            self.inference_engine_type == "hf" and self.use_separate_hf_rollout_actor
            and self.rollout_train_actor is not None and self.inference_engine is not None
            and self.inference_engine is not self.rollout_train_actor
        )

    def _keeps_separate_hf_rollout_actor_on_gpu(self) -> bool:
        return self._uses_separate_hf_rollout_actor() and bool(
            getattr(self.config, "hf_separate_rollout_keep_on_gpu", False)
        )

    def _build_local_hf_rollout_actor_sync_plan(
        self, src_actor: nn.Module, dst_actor: nn.Module
    ) -> Tuple[List[Tuple[str, nn.Parameter, nn.Parameter]], List[Tuple[str, torch.Tensor, torch.Tensor]]]:
        src_params = dict(src_actor.named_parameters())
        dst_params = dict(dst_actor.named_parameters())
        if src_params.keys() != dst_params.keys():
            missing_in_dst = sorted(set(src_params) - set(dst_params))
            missing_in_src = sorted(set(dst_params) - set(src_params))
            raise ValueError(
                "Separate local HF rollout actor parameter mismatch. "
                f"missing_in_dst={missing_in_dst[:8]}, missing_in_src={missing_in_src[:8]}"
            )

        param_pairs = []
        for name, src_param in src_params.items():
            dst_param = dst_params[name]
            if src_param.shape != dst_param.shape:
                raise ValueError(
                    f"Separate local HF rollout actor parameter shape mismatch for {name}: "
                    f"{tuple(src_param.shape)} vs {tuple(dst_param.shape)}"
                )
            param_pairs.append((name, src_param, dst_param))

        src_buffers = dict(src_actor.named_buffers())
        dst_buffers = dict(dst_actor.named_buffers())
        buffer_pairs = []
        for name in sorted(set(src_buffers) & set(dst_buffers)):
            src_buffer = src_buffers[name]
            dst_buffer = dst_buffers[name]
            if src_buffer.shape != dst_buffer.shape:
                raise ValueError(
                    f"Separate local HF rollout actor buffer shape mismatch for {name}: "
                    f"{tuple(src_buffer.shape)} vs {tuple(dst_buffer.shape)}"
                )
            buffer_pairs.append((name, src_buffer, dst_buffer))
        return param_pairs, buffer_pairs

    def _copy_local_hf_rollout_actor_state(self, src_actor: nn.Module, dst_actor: nn.Module) -> None:
        if self._separate_hf_rollout_sync_param_pairs is None or self._separate_hf_rollout_sync_buffer_pairs is None:
            (
                self._separate_hf_rollout_sync_param_pairs,
                self._separate_hf_rollout_sync_buffer_pairs,
            ) = self._build_local_hf_rollout_actor_sync_plan(src_actor, dst_actor)

        for name, src_param, dst_param in self._separate_hf_rollout_sync_param_pairs:
            src_tensor = src_param.detach()
            if src_tensor.device != dst_param.device or src_tensor.dtype != dst_param.dtype:
                src_tensor = src_tensor.to(device=dst_param.device, dtype=dst_param.dtype)
            dst_param.detach().copy_(src_tensor)

        for name, src_buffer_ref, dst_buffer in self._separate_hf_rollout_sync_buffer_pairs:
            src_buffer = src_buffer_ref.detach()
            if src_buffer.device != dst_buffer.device or src_buffer.dtype != dst_buffer.dtype:
                src_buffer = src_buffer.to(device=dst_buffer.device, dtype=dst_buffer.dtype)
            dst_buffer.detach().copy_(src_buffer)

    def _prepare_separate_hf_rollout_actor_for_generation(self) -> None:
        if not self._uses_separate_hf_rollout_actor():
            return
        model = getattr(self.inference_engine, "model", None)
        if isinstance(model, nn.Module):
            model.eval()
        self.inference_engine.eval()

    def _sync_separate_hf_rollout_actor(self, actor: nn.Module) -> None:
        if not self.config.fsdp:
            raise NotImplementedError("Separate local HF rollout actor currently only supports FSDP.")
        if not self._uses_separate_hf_rollout_actor():
            raise RuntimeError("Separate local HF rollout actor is not initialized.")

        keep_on_gpu = self._keeps_separate_hf_rollout_actor_on_gpu()
        sync_t0 = time.time()
        actor_offloaded = False
        offload_actor_t0 = time.time()
        if not keep_on_gpu and self.rollout_train_actor_is_on_gpu:
            self.offload_model(actor)
            self.rollout_train_actor_is_on_gpu = False
            actor_offloaded = True
        offload_actor_s = time.time() - offload_actor_t0

        rollout_offloaded = False
        offload_rollout_t0 = time.time()
        if not keep_on_gpu and self.inference_engine_status != EngineStatus.SLEEPED:
            self.offload_model(self.inference_engine, empty_cache=False)
            rollout_offloaded = True
        offload_rollout_s = time.time() - offload_rollout_t0

        copy_state_t0 = time.time()
        self._copy_local_hf_rollout_actor_state(actor, self.inference_engine)
        copy_state_s = time.time() - copy_state_t0

        reload_rollout_t0 = time.time()
        rollout_reloaded = False
        if keep_on_gpu:
            # `keep_on_gpu=True` skips the wakeup path, so we must explicitly
            # rematerialize the rollout actor here after copying the updated state.
            self.reload_model(self.inference_engine)
            rollout_reloaded = True
        reload_rollout_s = time.time() - reload_rollout_t0

        prepare_t0 = time.time()
        self._prepare_separate_hf_rollout_actor_for_generation()
        prepare_s = time.time() - prepare_t0

        sync_clear_t0 = time.time()
        if keep_on_gpu:
            torch.cuda.synchronize()
            torch.distributed.barrier()
            self.inference_engine_status = EngineStatus.WAKEUP
        else:
            self.inference_engine_status = EngineStatus.SLEEPED
            self.sync_and_clear_cache()
        sync_clear_s = time.time() - sync_clear_t0
        self.last_separate_hf_rollout_sync_stats = {
            "total_s": round(time.time() - sync_t0, 4),
            "keep_on_gpu": keep_on_gpu,
            "actor_offloaded": actor_offloaded,
            "rollout_offloaded": rollout_offloaded,
            "rollout_reloaded": rollout_reloaded,
            "offload_actor_s": round(offload_actor_s, 4),
            "offload_rollout_s": round(offload_rollout_s, 4),
            "copy_state_s": round(copy_state_s, 4),
            "reload_rollout_s": round(reload_rollout_s, 4),
            "prepare_s": round(prepare_s, 4),
            "sync_clear_s": round(sync_clear_s, 4),
        }
        self.print(
            "Finished update engine weights for separate local HF rollout actor",
            self.last_separate_hf_rollout_sync_stats,
        )

    def setup_inference_engine(
        self,
        args,
        engine_type="vllm",
        actor=None,
        rollout_actor=None,
        tokenizer=None,
        processor=None,
    ):
        """
        Initialize and setup the inference engine.

        :param args: Configuration arguments
        :type args: argparse.Namespace
        :param engine_type: Type of inference engine ('vllm', 'sglang', or 'hf')
        :type engine_type: str
        :param actor: The actor module, if passed, will be used to update engine weights
        :type actor: torch.nn.Module

        :return: Initialized inference engine
        :rtype: object
        :raises ValueError: If engine_type is not supported
        """
        self.inference_engine_type = engine_type
        self.inference_tokenizer = tokenizer
        self.inference_processor = processor
        self.rollout_train_actor = None
        self.rollout_train_actor_is_on_gpu = False
        self.use_separate_hf_rollout_actor = False
        self._separate_hf_rollout_sync_param_pairs = None
        self._separate_hf_rollout_sync_buffer_pairs = None
        self.last_separate_hf_rollout_sync_stats = None

        if engine_type == "vllm":
            # Conditional import: vLLM is optional and only imported when explicitly requested
            # Default engine is SGLang. To use vLLM, install with: pip install "LightRFT[vllm]"
            from .vllm_utils import get_vllm_engine_for_rollout
            self.inference_engine = get_vllm_engine_for_rollout(args)
            self.inference_engine_status = EngineStatus.WAKEUP
        elif engine_type == "sglang":
            # Conditional import: SGLang is optional and only imported when explicitly requested
            from .sglang_utils import get_sglang_engine_for_rollout
            self.inference_engine = get_sglang_engine_for_rollout(args)
            self.inference_engine_status = EngineStatus.WAKEUP
        elif engine_type == "hf":
            if actor is None:
                raise ValueError("engine_type='hf' requires the prepared actor to be passed in.")
            if getattr(args, "hf_separate_rollout_actor", False):
                if rollout_actor is None:
                    raise ValueError(
                        "engine_type='hf' with --hf_separate_rollout_actor requires a prepared rollout_actor."
                    )
                self.use_separate_hf_rollout_actor = True
                self.rollout_train_actor = actor
                self.rollout_train_actor_is_on_gpu = True
                self.inference_engine = rollout_actor
                self._prepare_separate_hf_rollout_actor_for_generation()
                self.inference_engine_status = (
                    EngineStatus.WAKEUP if self._keeps_separate_hf_rollout_actor_on_gpu() else EngineStatus.SLEEPED
                )
            else:
                # Local HF mode reuses the actor directly for time-boxed smoke runs.
                self.inference_engine = actor
                self.inference_engine_status = EngineStatus.WAKEUP
        else:
            raise ValueError(f"Unsupported engine type: {engine_type}")

        if actor is not None and (engine_type != "hf" or self.use_separate_hf_rollout_actor):
            self.update_engine_weights(actor)
        self.maybe_sleep_inference_engine()
        return self.inference_engine

    def maybe_sleep_inference_engine(self):
        """
        Put the inference engine to sleep if enabled and available.

        Sleeps the engine to conserve memory when not in use. Only supports vLLM and SGLang engines.
        After sleeping, synchronizes and clears the cache.

        :raises ValueError: If the inference engine type is not supported
        """
        if self.inference_engine is None or self.inference_engine_status == EngineStatus.SLEEPED:
            return
        if self._keeps_separate_hf_rollout_actor_on_gpu():
            self._prepare_separate_hf_rollout_actor_for_generation()
            self.inference_engine_status = EngineStatus.WAKEUP
            return
        if self.inference_engine is not None and (
            self.args.enable_engine_sleep or self._uses_separate_hf_rollout_actor()
        ):
            sleep_t0 = time.time()
            if self.inference_engine_type in ["vllm", "sglang"]:
                self.inference_engine.sleep()
            elif self.inference_engine_type == "hf":
                if self._uses_separate_hf_rollout_actor():
                    self.offload_model(self.inference_engine)
                    if not self.rollout_train_actor_is_on_gpu:
                        self.reload_model(self.rollout_train_actor)
                        self.rollout_train_actor_is_on_gpu = True
                    self.rollout_train_actor.train()
                else:
                    return
            else:
                raise ValueError(f"Unsupported engine type: {self.inference_engine_type}")
            self.inference_engine_status = EngineStatus.SLEEPED

            self.sync_and_clear_cache()
            self.print(f"Sleeped inference engine, TIMECOST {time.time() - sleep_t0}")

    def wakeup_inference_engine(self):
        """
        Wake up the inference engine from sleep state.

        To avoid OOM, we:
            1. sync and clear cache
            2. wakeup engine

        :raises ValueError: If the inference engine type is not supported
        """
        if self.inference_engine is None or self.inference_engine_status == EngineStatus.WAKEUP:
            return
        if self._keeps_separate_hf_rollout_actor_on_gpu():
            self._prepare_separate_hf_rollout_actor_for_generation()
            self.inference_engine_status = EngineStatus.WAKEUP
            return
        self.sync_and_clear_cache()
        wkup_t0 = time.time()

        if self.inference_engine_type in ["vllm", "sglang"]:
            self.inference_engine.wake_up()
        elif self.inference_engine_type == "hf":
            if self._uses_separate_hf_rollout_actor():
                if self.rollout_train_actor_is_on_gpu:
                    self.offload_model(self.rollout_train_actor)
                    self.rollout_train_actor_is_on_gpu = False
                self.reload_model(self.inference_engine)
                self._prepare_separate_hf_rollout_actor_for_generation()
            self.print(f"Finished {self.inference_engine_type} wakeup, TIMECOST {time.time() - wkup_t0}")
            self.inference_engine_status = EngineStatus.WAKEUP
            return
        else:
            raise ValueError(f"Unsupported engine type: {self.inference_engine_type}")
        # torch.cuda.reset_max_memory_allocated()
        self.report_memory("after ppo training, after wakeup inference engine")
        self.print(f"Finished {self.inference_engine_type} wakeup, TIMECOST {time.time() - wkup_t0}")

        self.inference_engine_status = EngineStatus.WAKEUP

    def _sanitize_vllm_sampling_params(self, sampling_params: Any) -> Any:
        """
        Clamp vLLM sampling parameters that must respect the engine context length.

        Newer vLLM releases (>=0.11.0) validate ``truncate_prompt_tokens`` against the engine's
        ``max_model_len``. Some callers already clamp this during ``SamplingParams``
        construction, but this centralized guard keeps generation safe for every
        call path that reaches the strategy layer.

        :param sampling_params: Sampling parameters passed to vLLM.
        :type sampling_params: Any
        :return: Sanitized sampling parameters.
        :rtype: Any
        """
        if sampling_params is None or self.inference_engine_type != "vllm" or self.inference_engine is None:
            return sampling_params

        truncate_prompt_tokens = getattr(sampling_params, "truncate_prompt_tokens", None)
        if truncate_prompt_tokens is None:
            return sampling_params

        model_config = getattr(getattr(self.inference_engine, "llm_engine", None), "model_config", None)
        max_model_len = getattr(model_config, "max_model_len", None)
        if max_model_len is None:
            return sampling_params

        try:
            if truncate_prompt_tokens > max_model_len:
                sampling_params.truncate_prompt_tokens = max_model_len
        except TypeError:
            logger.warning(
                "Skipping vLLM truncate_prompt_tokens validation because values are not directly comparable: "
                "truncate_prompt_tokens={}, max_model_len={}",
                truncate_prompt_tokens,
                max_model_len,
            )

        return sampling_params

    def engine_generate_local(
        self,
        sampling_params: Any,
        prompt_token_ids: Optional[Union[List[int], List[List[int]]]] = None,
        multi_modal_inputs: Optional[List[Dict[str, Any]]] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        images_num: Optional[List[int]] = None,
        videos_num: Optional[List[int]] = None,
    ) -> List[EasyDict]:
        """
        Perform text or multimodal generation using different inference engines based on the input mode.

        :param sampling_params: Parameters used for controlling the generation process (e.g., temperature, top_k).
        :param prompt_token_ids: List of text token IDs.
        :param multi_modal_inputs: A list of dictionaries representing multimodal inputs.
                                   Each dictionary should contain a raw text under the "prompt" key,
                                   and additional modalities (such as images) under the "multi_modal_data" key.
                                   Example:
                                   multi_modal_inputs = [{
                                       "prompt": [...],
                                       "multi_modal_data": {
                                           "image": [...],
                                           "video": [...]
                                       }
                                   }]
        :return: A list of generated outputs in EasyDict format, produced by the selected inference engine.
        :raises ValueError: If both prompt_token_ids and multi_modal_inputs are None.
        :raises ValueError: If both prompt_token_ids and multi_modal_inputs are not None.
        """

        if prompt_token_ids is None and multi_modal_inputs is None:
            raise ValueError("Either prompt_token_ids or multi_modal_inputs must be provided.")

        if self.inference_engine_type != "hf" and prompt_token_ids is not None and multi_modal_inputs is not None:
            raise ValueError("Both prompt_token_ids and multi_modal_inputs can not be provided at the same time.")

        # if inference engine is vllm
        if self.inference_engine_type == "vllm":
            sampling_params = self._sanitize_vllm_sampling_params(sampling_params)
            # For vLLM:
            # - If `prompt_token_ids` is provided, it indicates a pure LLM (text-only) generation.
            # - If `prompts` (i.e., `multi_modal_inputs`) is provided, it indicates a VLM (multimodal) generation.
            if multi_modal_inputs is not None:
                prompt = []
                for item in multi_modal_inputs:
                    prompt_item = {"prompt": item["prompt"]}
                    # Do not forward multimodal prompt_token_ids to vLLM.
                    # vLLM's multimodal processor applies placeholder expansion
                    # itself; passing already-expanded token IDs causes image
                    # placeholders to be expanded a second time.
                    if item.get("multi_modal_data") is not None:
                        prompt_item["multi_modal_data"] = item["multi_modal_data"]
                    prompt.append(prompt_item)
            elif prompt_token_ids is not None:
                if len(prompt_token_ids) > 0 and isinstance(prompt_token_ids[0], int):
                    prompt = [{"prompt_token_ids": prompt_token_ids}]
                else:
                    prompt = [{"prompt_token_ids": token_ids} for token_ids in prompt_token_ids]
            else:
                raise ValueError("Either prompt (multi_modal_inputs) or prompt_token_ids must be provided.")

            vllm_outputs = self.inference_engine.generate(
                sampling_params=sampling_params,
                prompts=prompt,
                use_tqdm=False,
            )
            return [
                EasyDict(
                    prompt_token_ids=output.prompt_token_ids,
                    output_token_ids=output.outputs[0].token_ids,
                ) for output in vllm_outputs
            ]
        elif self.inference_engine_type == "sglang":

            if multi_modal_inputs is not None:  # VLM case
                logger.debug(f"rank {dist.get_rank()} VLM branch")
                prompt = [p["prompt"] for p in multi_modal_inputs]

                # Handle cases where some prompts might not have images
                # Flatten nested list format if needed: [[PIL.Image]] -> [PIL.Image]
                image = [(img[0] if isinstance(img, list) and len(img) > 0 else img)
                         for img in (p.get("multi_modal_data", {}).get("image") for p in multi_modal_inputs)]

                sglang_outputs = self.inference_engine.generate(
                    sampling_params=sampling_params,
                    prompt=prompt,  # skip_tokenizer_init must be False
                    image_data=image,
                )

                # VLM case: prompt_token_ids should be provided separately or extracted from sglang output
                # Since sglang doesn't return prompt_token_ids in VLM mode, we set it to None here
                # and expect the caller to fill it in if needed
                return [
                    EasyDict(
                        prompt_token_ids=None,  # Will be filled by caller if needed
                        output_token_ids=sglang_outputs[i]["output_ids"],
                    ) for i in range(len(sglang_outputs))
                ]
            else:  # text-only case
                logger.debug(f"rank {dist.get_rank()} text-only branch")
                sglang_outputs = self.inference_engine.generate(
                    sampling_params=sampling_params,
                    input_ids=prompt_token_ids,
                    image_data=None,
                )
                # Text-only case: prompt_token_ids is available from input
                return [
                    EasyDict(
                        prompt_token_ids=prompt_token_ids[i],
                        output_token_ids=sglang_outputs[i]["output_ids"],
                    ) for i in range(len(sglang_outputs))
                ]
        elif self.inference_engine_type == "hf":
            if prompt_token_ids is None:
                raise ValueError("Local HF inference requires prompt_token_ids.")

            from lightrft.datasets.utils import zero_pad_sequences

            model = getattr(self.inference_engine, "model", self.inference_engine)
            model_config = getattr(model, "config", None)
            eos_token_id = getattr(self.inference_tokenizer, "eos_token_id", None)
            pad_token_id = getattr(self.inference_tokenizer, "pad_token_id", None)

            if eos_token_id is None and model_config is not None:
                eos_token_id = getattr(model_config, "eos_token_id", None)
            if pad_token_id is None and model_config is not None:
                pad_token_id = getattr(model_config, "pad_token_id", None)

            if isinstance(eos_token_id, (list, tuple)):
                eos_token_id = eos_token_id[0]
            if isinstance(pad_token_id, (list, tuple)):
                pad_token_id = pad_token_id[0]
            if pad_token_id is None:
                pad_token_id = eos_token_id
            if eos_token_id is None:
                raise ValueError("Unable to resolve eos_token_id for local HF inference engine.")

            normalized_prompt_ids = []
            for token_ids in prompt_token_ids:
                if isinstance(token_ids, torch.Tensor):
                    token_ids = token_ids.tolist()
                normalized_prompt_ids.append(token_ids)

            device = torch.cuda.current_device()

            def _prepare_tensor(tensor):
                if tensor is None:
                    return None
                if isinstance(tensor, torch.Tensor) and tensor.numel() == 0:
                    return None
                return tensor.to(device, non_blocking=True)

            top_k = sampling_params.get("top_k", None)
            if top_k is not None and top_k <= 0:
                top_k = None
            temperature = sampling_params.get("temperature", 1.0)
            do_sample = sampling_params.get("do_sample", temperature is None or temperature > 0)

            def _run_local_hf_batch(
                batch_prompt_token_ids,
                batch_pixel_values=None,
                batch_image_grid_thw=None,
                batch_pixel_values_videos=None,
                batch_video_grid_thw=None,
            ):
                prompt_tensors = [torch.tensor(token_ids, dtype=torch.long) for token_ids in batch_prompt_token_ids]
                padded_input_ids = zero_pad_sequences(prompt_tensors, side="left", value=pad_token_id).to(device)
                attention_mask = padded_input_ids.ne(pad_token_id).long()
                prompt_lengths = attention_mask.sum(dim=1).detach().cpu()

                generate_t0 = time.time()
                with torch.no_grad():
                    sequences, attention_mask_out, _ = self.inference_engine.generate(
                        input_ids=padded_input_ids,
                        attention_mask=attention_mask,
                        pixel_values=_prepare_tensor(batch_pixel_values),
                        image_grid_thw=_prepare_tensor(batch_image_grid_thw),
                        pixel_values_videos=_prepare_tensor(batch_pixel_values_videos),
                        video_grid_thw=_prepare_tensor(batch_video_grid_thw),
                        top_k=top_k,
                        top_p=sampling_params.get("top_p", 1.0),
                        temperature=temperature,
                        do_sample=do_sample,
                        max_new_tokens=sampling_params.get("max_new_tokens", 1024),
                        min_new_tokens=sampling_params.get("min_new_tokens", 1),
                        repetition_penalty=sampling_params.get("repetition_penalty", 1.0),
                        no_repeat_ngram_size=sampling_params.get("no_repeat_ngram_size", 0),
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                    )
                elapsed_s = round(time.time() - generate_t0, 4)
                self.print(
                    "Local HF model.generate finished:", {
                        "batch_size": len(batch_prompt_token_ids),
                        "prompt_tokens": [len(token_ids) for token_ids in batch_prompt_token_ids],
                        "elapsed_s": elapsed_s,
                    }
                )

                output_start_idx = padded_input_ids.size(1)
                sequences = sequences.detach().cpu()
                attention_mask_out = attention_mask_out.detach().cpu()

                batch_outputs = []
                for idx in range(sequences.size(0)):
                    total_length = int(attention_mask_out[idx].sum().item())
                    generated_length = max(total_length - int(prompt_lengths[idx].item()), 0)
                    output_end_idx = output_start_idx + generated_length
                    batch_outputs.append(
                        EasyDict(
                            prompt_token_ids=batch_prompt_token_ids[idx],
                            output_token_ids=sequences[idx, output_start_idx:output_end_idx].tolist(),
                        )
                    )
                return batch_outputs, elapsed_s

            max_batch_size = max(int(getattr(self.config, "local_hf_generate_max_batch_size", 0) or 0), 0)
            if max_batch_size > 0 and len(normalized_prompt_ids) > max_batch_size:
                images_prefix = None
                if images_num is not None:
                    images_prefix = [0]
                    for num in images_num:
                        images_prefix.append(images_prefix[-1] + num)
                videos_prefix = None
                if videos_num is not None:
                    videos_prefix = [0]
                    for num in videos_num:
                        videos_prefix.append(videos_prefix[-1] + num)

                def _slice_modal_tensor(tensor, offsets, start, end):
                    if tensor is None:
                        return None
                    if not isinstance(tensor, torch.Tensor):
                        return tensor
                    if tensor.numel() == 0:
                        return tensor
                    if offsets is not None:
                        return tensor[offsets[start]:offsets[end]]
                    return tensor[start:end]

                engine_outputs = []
                for start in range(0, len(normalized_prompt_ids), max_batch_size):
                    end = min(start + max_batch_size, len(normalized_prompt_ids))
                    batch_outputs, chunk_elapsed_s = _run_local_hf_batch(
                        normalized_prompt_ids[start:end],
                        batch_pixel_values=_slice_modal_tensor(pixel_values, images_prefix, start, end),
                        batch_image_grid_thw=_slice_modal_tensor(image_grid_thw, images_prefix, start, end),
                        batch_pixel_values_videos=_slice_modal_tensor(pixel_values_videos, videos_prefix, start, end),
                        batch_video_grid_thw=_slice_modal_tensor(video_grid_thw, videos_prefix, start, end),
                    )
                    engine_outputs.extend(batch_outputs)
                return engine_outputs

            batch_outputs, chunk_elapsed_s = _run_local_hf_batch(
                normalized_prompt_ids,
                batch_pixel_values=pixel_values,
                batch_image_grid_thw=image_grid_thw,
                batch_pixel_values_videos=pixel_values_videos,
                batch_video_grid_thw=video_grid_thw,
            )
            return batch_outputs
        else:
            raise ValueError(f"Unsupported engine type: {self.inference_engine_type}")

    @classmethod
    def _build_multimodal_inputs(
        cls,
        all_prompts,
        all_images,
        images_num,
        all_videos,
        videos_num,
    ):
        """
        Build multimodal inputs for inference engine (vLLM/SGLang).

        This function supports two input formats for images and videos to accommodate
        different data preprocessing approaches:

        Format 1 - Nested List (multi-image/video per prompt already grouped):
            all_images = [[img1_a, img1_b], [img2_a], [img3_a, img3_b, img3_c]]
            images_num = [2, 1, 3]
            -> all_images[i] is directly used as the image list for prompt i

        Format 2 - Flattened List (all images/videos in a single flat list):
            all_images = [img1_a, img1_b, img2_a, img3_a, img3_b, img3_c]
            images_num = [2, 1, 3]
            -> images are sliced based on images_num: [0:2], [2:3], [3:6]

        :param all_prompts: List of text prompts
        :param all_images: Images in nested or flattened format, or None
        :param images_num: Number of images per prompt
        :param all_videos: Videos in nested or flattened format, or None
        :param videos_num: Number of videos per prompt
        :return: List of dicts with 'prompt' and optional 'multi_modal_data' keys
        """
        inputs = []
        img_start_idx = 0
        vid_start_idx = 0
        for i, prompt in enumerate(all_prompts):
            img_num = images_num[i] if images_num is not None else 0
            vid_num = videos_num[i] if videos_num is not None else 0

            # Support two input formats:
            # 1. Nested list: all_images[i] is already a list of images for this prompt
            # 2. Flattened list: all_images is a flat list, slice by img_num
            if all_images is not None:
                if i < len(all_images) and isinstance(all_images[i], list) and len(all_images[i]) == img_num:
                    img_list = all_images[i]
                else:
                    img_list = all_images[img_start_idx:img_start_idx + img_num]
            else:
                img_list = []

            # Same logic for videos
            if all_videos is not None:
                if i < len(all_videos) and isinstance(all_videos[i], list) and len(all_videos[i]) == vid_num:
                    vid_list = all_videos[i]
                else:
                    vid_list = all_videos[vid_start_idx:vid_start_idx + vid_num]
            else:
                vid_list = []

            multi_modal_data = {}
            if len(img_list) > 0 and img_list[0] is not None:
                multi_modal_data["image"] = img_list
            if len(vid_list) > 0 and vid_list[0] is not None:
                multi_modal_data["video"] = vid_list

            if not multi_modal_data:
                # remove the vision start and end tokens for data after apply chat template.
                # Use regex to handle multiple <|image_pad|> tokens (e.g., for high-res images)
                cleaned_prompt = re.sub(r'<\|vision_start\|>(<\|image_pad\|>)+<\|vision_end\|>', '', prompt)
                cleaned_prompt = re.sub(r'<\|vision_start\|>(<\|video_pad\|>)+<\|vision_end\|>', '', cleaned_prompt)
                input_item = {
                    "prompt": cleaned_prompt,
                }
                inputs.append(input_item)
            else:
                input_item = {
                    "prompt": prompt,
                    "multi_modal_data": multi_modal_data,
                }
                inputs.append(input_item)
            img_start_idx += img_num
            vid_start_idx += vid_num
        return inputs

    def gather_and_generate(
        self,
        sampling_params,
        all_prompt_token_ids=None,
        all_prompts=None,
        all_images=None,
        sleep_engine=True,
        images_num=None,
        all_videos=None,
        videos_num=None,
        all_images_pixel_values=None,
        all_videos_pixel_values=None,
        all_images_grid_thw=None,
        all_videos_grid_thw=None,
    ):
        """
        Gather prompts across distributed ranks and perform text/multimodal generation.

        This method coordinates distributed generation by:
        1. Gathering prompts from all ranks within a vLLM tensor parallel group
        2. Performing batched generation using the inference engine
        3. Splitting generated outputs and returning each rank's portion
        4. Optionally putting the inference engine to sleep to conserve memory

        For multimodal inputs, supports flexible input formats:
        - One prompt with one image
        - One prompt with multiple images
        - One prompt with video(s) only (no images)
        - One prompt with one or more videos
        - Mixed image and video inputs

        :param sampling_params: Parameters controlling generation (e.g., temperature, top_k, max_tokens)
        :type sampling_params: Any
        :param all_prompt_token_ids: Token IDs for text-only prompts, defaults to None
        :type all_prompt_token_ids: Optional[List[List[int]]]
        :param all_prompts: Raw text prompts for multimodal generation, defaults to None
        :type all_prompts: Optional[List[str]]
        :param all_images: Images corresponding to prompts for VLM generation, defaults to None
        :type all_images: Optional[List]
        :param sleep_engine: Whether to sleep the inference engine after generation, defaults to True
        :type sleep_engine: bool
        :param images_num: Number of images per prompt (for multi-image scenarios), defaults to None
        :type images_num: Optional[List[int]]
        :param all_videos: Videos corresponding to prompts for video generation, defaults to None
        :type all_videos: Optional[List]
        :param videos_num: Number of videos per prompt, defaults to None
        :type videos_num: Optional[List[int]]

        :return: List of generation outputs for the current rank, each containing prompt_token_ids and output_token_ids
        :rtype: List[EasyDict]
        :raises NotImplementedError: If inference engine is not initialized
        """
        if self.inference_engine is None:
            raise NotImplementedError("Inference engine is not initialized.")
        if self._uses_separate_hf_rollout_actor():
            sleep_engine = True
        self.wakeup_inference_engine()

        # is_multimodal = all_images is not None
        # NOTE: not only check if all_images is None, but also check if it contains non-None elements
        # If all_images is [None, None, ...], any(img is not None for img in all_images) will return False
        # Same logic applies to all_videos
        is_multimodal = (((all_images is not None) and any(img is not None for img in all_images))
                         or ((all_videos is not None) and any(vid is not None for vid in all_videos)))

        if self.inference_engine_type == "hf":
            inputs = all_prompt_token_ids
            assert inputs is not None
            self.print(f"Start VLM gather_and_generate ..., total prompts: {len(inputs)}")
            all_outputs = self.engine_generate_local(
                sampling_params=sampling_params,
                prompt_token_ids=inputs,
                pixel_values=all_images_pixel_values if is_multimodal else None,
                image_grid_thw=all_images_grid_thw if is_multimodal else None,
                pixel_values_videos=all_videos_pixel_values if is_multimodal else None,
                video_grid_thw=all_videos_grid_thw if is_multimodal else None,
                images_num=images_num if is_multimodal else None,
                videos_num=videos_num if is_multimodal else None,
            )
            local_outputs = all_outputs
        else:
            if is_multimodal:
                inputs = self._build_multimodal_inputs(
                    all_prompts=all_prompts,
                    all_images=all_images,
                    images_num=images_num,
                    all_videos=all_videos,
                    videos_num=videos_num,
                )
            else:
                inputs = all_prompt_token_ids
                assert inputs is not None

            inputs = gather_inputs_object_for_inference(input_data=inputs, group=self.engine_mp_group)

            self.print(f"Start VLM gather_and_generate ..., total prompts: {len(inputs)}")

            all_outputs = self.engine_generate_local(
                sampling_params=sampling_params,
                prompt_token_ids=None if is_multimodal else inputs,
                multi_modal_inputs=inputs if is_multimodal else None,
            )

            engine_mp_size = torch.distributed.get_world_size(self.engine_mp_group)
            num_prompts_per_rank = len(all_outputs) // engine_mp_size
            assert len(all_outputs) % engine_mp_size == 0
            cur_rank = torch.distributed.get_rank(self.engine_mp_group)
            local_outputs = all_outputs[cur_rank * num_prompts_per_rank:(cur_rank + 1) * num_prompts_per_rank]

        if is_multimodal and self.inference_engine_type == "sglang":
            # For SGLang VLM case, prompt_token_ids is set to None in engine_generate_local
            # We need to fill it with the actual token_ids here
            for i, output in enumerate(local_outputs):
                if output.prompt_token_ids is None:
                    output.prompt_token_ids = all_prompt_token_ids[i]

        if sleep_engine is True:
            self.maybe_sleep_inference_engine()

        info = self.genlen_analyser.collect(all_outputs, self._profile_step, self.is_rank_0())
        if info is not None:
            self.print(f"step {self._profile_step} generate length: ", info)

        self._profile_step += 1
        self.print(f"Finished gather_and_generate, {len(local_outputs)=}")
        return local_outputs

    def update_engine_weights(self, actor):
        """
        Update the weights of the inference engine from the actor model.

        :param actor: The actor model whose weights will be copied
        """
        if self.inference_engine is None:
            self.print("Skip update engine weights since inference engine is not initialized.")
            return
        if self.inference_engine_type == "hf":
            if self._uses_separate_hf_rollout_actor():
                self._sync_separate_hf_rollout_actor(actor)
            else:
                self.print("Skip update engine weights for local HF engine because it reuses the actor directly.")
            return
        # 1. wakeup engine if sleeped
        self.wakeup_inference_engine()

        # TODO: unify the broadcast manager
        if self.inference_engine_type not in ["vllm", "sglang"]:
            raise NotImplementedError(f"Unsupported engine type: {self.inference_engine_type}")
        if self.broadcast_manager is None:
            self.broadcast_manager = BroadcastManager(actor, self, self.inference_engine)

        self.broadcast_manager.broadcast_to_engine()
        self.print("finished update_engine_weights")
        self.sync_and_clear_cache()

    @classmethod
    def sync_and_clear_cache(cls):
        """
        Synchronize CUDA operations and clear the cache.

        Performs three operations:
        1. CUDA synchronization
        2. Distributed barrier
        3. CUDA cache clearing
        """
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()

    @contextmanager
    def init_model_context(self):
        """
        Context manager for model initialization.

        Currently does nothing by default, used only for DeepSpeed.
        Reports memory usage after completion.
        """
        try:
            # Do nothing by default, only deepspeed
            yield
        finally:
            self.report_memory("Finished init_model_context")

    def maybe_offload_optimizer(self, optimizer):  # pylint: disable=W0613
        """
        Placeholder for FSDP optimizer offloading functionality.
        :param optimizer: The optimizer to potentially offload
        :type optimizer: torch.optim.Optimizer
        """
        self.print("maybe_offload_optimizer not implemented and Skipped")

    def maybe_load_optimizer(self, optimizer, device=torch.cuda.current_device()):  # pylint: disable=W0613
        """
        Placeholder for FSDP optimizer loading functionality.
        :param optimizer: The optimizer to potentially load
        :type optimizer: torch.optim.Optimizer
        :param device: Target device for loading
        :type device: torch.device
        """
        self.print("maybe_load_optimizer not implemented and Skipped")


def is_actor(model):
    """
    Check if a model is an actor model.

    :param model: The model to check
    :return: True if the model is an actor, False otherwise
    :rtype: bool
    """
    return getattr(model, "is_actor", False)
