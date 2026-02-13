"""
Hugging Face FSDP (Fully Sharded Data Parallel) Strategy Module.

This module provides implementations for distributed training using PyTorch's FSDP.
It includes utilities for model wrapping, optimization, checkpointing, and state management
in a distributed training environment. The module supports FSDP v2 strategy,
with special handling for model sharding, mixed precision training, and optimizer state
management.
"""

import os
import shutil
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import distributed as dist
from torch import nn, optim

try:
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        FSDPModule,
        MixedPrecisionPolicy,
        OffloadPolicy,
        fully_shard,
        register_fsdp_forward_method,
    )
except ImportError:
    from torch.distributed._composable.fsdp import (
        fully_shard,
        register_fsdp_forward_method,
        MixedPrecisionPolicy,
        CPUOffloadPolicy,
        OffloadPolicy,
        FSDPModule,
    )

import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.optim import Optimizer
from transformers.trainer_pt_utils import get_module_class_from_name

from lightrft.strategy.strategy_base import StrategyBase, is_actor
from lightrft.strategy.utils.optimizer_utils import group_parameters_for_optimizer_dtensor
from lightrft.strategy.utils.ckpt_utils import find_latest_checkpoint_dir

from .fsdp_optimizer import (
    FSDPadaptOptimizer,
    load_fsdp_optimizer,
    offload_fsdp_optimizer,
)
from .fsdp_utils import is_meta_initialized

ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]

manual_transformer_cls_names_to_wrap = [
    "Embedding",
    "Qwen2VLDecoderLayer",
    "Qwen2VLVisionBlock",
    "Qwen2_5_VLVisionBlock",
    "Qwen2_5_VLDecoderLayer",
    "Qwen2DecoderLayer",
    "LlamaDecoderLayer",  # for DeepSeek-R1-Distill-Llama-70B
    "DeepseekDecoderLayer",
]

vit_transformer_cls_names = [
    "Qwen2VLVisionBlock",
    "Qwen2_5_VLVisionBlock",
]


class FSDPV2Strategy(StrategyBase):
    """
    The strategy for training with PyTorch's Fully Sharded Data Parallel V2.

    This strategy implements model sharding using PyTorch's FSDP to enable training
    of large models across multiple GPUs with memory efficiency.

    :param seed: Random seed for reproducibility.
    :type seed: int

    :param max_norm: Maximum gradient norm for gradient clipping. If 0.0, no clipping is performed.
    :type max_norm: float

    :param micro_train_batch_size: Batch size for a single training step.
    :type micro_train_batch_size: int

    :param train_batch_size: Total batch size for training.
    :type train_batch_size: int

    :param bf16: Whether to use bfloat16 precision.
    :type bf16: bool

    :param args: Additional arguments for the strategy.
    :type args: object
    """

    def __init__(  # pylint: disable=R0917
        self,
        seed: int = 42,
        max_norm: float = 0.0,
        micro_train_batch_size: int = 1,
        train_batch_size: int = 1,
        bf16: bool = True,
        args=None,
    ) -> None:
        """
        Initialize the FSDP V2 strategy.

        :param seed: Random seed for reproducibility
        :type seed: int
        :param max_norm: Maximum gradient norm for gradient clipping. If 0.0, no clipping is performed
        :type max_norm: float
        :param micro_train_batch_size: Batch size for a single training step
        :type micro_train_batch_size: int
        :param train_batch_size: Total batch size for training
        :type train_batch_size: int
        :param bf16: Whether to use bfloat16 precision
        :type bf16: bool
        :param args: Additional arguments for the strategy
        :type args: object
        """
        super().__init__(seed, max_norm, micro_train_batch_size, train_batch_size, args)
        self.bf16 = bf16
        self.mixed_mm_data = self.config.mixed_mm_data
        self.use_naive_opt = not self.config.use_mp_opt

        self.cur_step = defaultdict(int)

        # fsdp cpu offload automatically offloads optimizer
        if self.config.fsdp_cpu_offload:
            self.config.adam_offload = False
            self.print("FSDPV2Strategy fsdp_cpu_offload is True")

        self.print("FSDPV2Strategy Initialized")

    def create_optimizer(self, model, **kwargs) -> Optimizer:
        """
        Create an optimizer for the model with proper parameter grouping.

        Groups parameters by dtype, dtensor shard size, and weight decay to avoid errors
        during gradient clipping and optimization steps.

        :param model: The model for which to create the optimizer
        :type model: torch.nn.Module
        :param kwargs: Additional arguments for the optimizer, including weight_decay
        :return: The created optimizer
        :rtype: torch.optim.Optimizer

        Example::

            >>> optimizer = strategy.create_optimizer(model, lr=1e-4, weight_decay=0.01)
        """
        if is_actor(model):
            model = model.model
        # group params by (dtype, dtensor shard size, weight_dacay) to avoid error in clip_grad and opt.step
        self.grouped_params = group_parameters_for_optimizer_dtensor(model, kwargs["weight_decay"])
        # Convert the grouped parameters into the final format for the optimizer
        optim_params = []
        for (wd_val, _, _), params_list in self.grouped_params.items():
            optim_params.append({
                "params": params_list,
                "weight_decay": wd_val,
            })
        optim = torch.optim.AdamW(optim_params, fused=True, **kwargs)
        self.print(f"Creating optimizer with {self.use_naive_opt=} ")
        if self.use_naive_opt:
            return optim
        else:
            return FSDPadaptOptimizer(optim)

    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, **kwargs) -> None:
        """
        Perform backward pass for the loss.

        :param loss: The loss tensor
        :type loss: torch.Tensor
        :param model: The model
        :type model: torch.nn.Module
        :param optimizer: The optimizer
        :type optimizer: torch.optim.Optimizer
        :param kwargs: Additional arguments
        """
        loss.backward()

    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module,
        scheduler,
        name="model",
        **kwargs,
    ) -> None:
        """
        Perform an optimization step.

        Handles gradient accumulation by only stepping the optimizer and scheduler
        after the specified number of accumulation steps.

        :param optimizer: The optimizer
        :type optimizer: torch.optim.Optimizer
        :param model: The model
        :type model: torch.nn.Module
        :param scheduler: The learning rate scheduler
        :param name: Name identifier for the model
        :type name: str
        :param kwargs: Additional arguments
        """
        self.cur_step[name] += 1
        if self.cur_step[name] == self.accumulated_gradient:
            if is_actor(model):
                model = model.model

            grad_norms = []
            for param_group in self.grouped_params.values():
                grad_norms.append(torch.nn.utils.clip_grad_norm_(param_group, max_norm=self.max_norm))
            # if grad_norm is not finite, skip the update
            if not all(torch.isfinite(grad_norm) for grad_norm in grad_norms):
                print(f"WARN: grad_norm is not finite: {grad_norms}")
                optimizer.zero_grad()
            else:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            self.cur_step[name] = 0

    def prepare_model(self,
                      model,
                      is_training=False,
                      shard_size=-1,
                      reshard_after_forward=True) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        """
        Prepares a model for FSDP training.

        :param model: The model to prepare.
        :type model: torch.nn.Module or None

        :return: The prepared model wrapped with FSDP.
        :rtype: torch.nn.Module

        Example::

            >>> prepared_model = strategy.prepare_model(model)
        """
        def get_auto_shard_size(model, is_training):
            """
            Automatically determine the shard size based on model size and training mode.

            :param model: The model to analyze
            :type model: torch.nn.Module
            :param is_training: Whether the model is being prepared for training
            :type is_training: bool
            :return: The recommended shard size
            :rtype: int
            """
            if is_training:
                return -1
            # Calculate total number of parameters (in billions)
            total_params = sum(p.numel() for p in model.parameters())

            if total_params < 1e10:  # < 10B
                return 1
            elif total_params < 8e10:  # 10B-80B
                return 8
            else:  # ≥ 80B
                return -1

        shard_size = get_auto_shard_size(model, is_training) if shard_size == "auto" else shard_size

        if model is None or not isinstance(model, torch.nn.Module):
            return model
        if hasattr(model, "base_model"):
            # when RM.base_model is an Engine or already wrapped by FSDP, skip fsdp init
            if not isinstance(model.base_model, torch.nn.Module):
                return model
            elif isinstance(model.base_model, FSDPModule):
                return model
        return self._fsdp_init_model(
            model, is_training=is_training, shard_size=shard_size, reshard_after_forward=reshard_after_forward
        )

    @torch.no_grad()
    def _fsdp_init_model(self, model, is_training, shard_size=-1, reshard_after_forward=True):
        """
        Initialize a model with FSDP wrapping.

        Sets up mixed precision, auto-wrapping policy, and sharding strategy for the model.

        :param model: The model to initialize with FSDP
        :type model: torch.nn.Module
        :param is_training: Whether the model is being prepared for training
        :type is_training: bool
        :param shard_size: The shard size for FSDP
        :type shard_size: int
        :param reshard_after_forward: Whether to reshard parameters after forward pass
        :type reshard_after_forward: bool

        :return: The FSDP-wrapped model
        :rtype: torch.nn.Module
        """
        self.report_memory("before FSDP2 wrap model")

        naive_mp_training = self.use_naive_opt and is_training

        model_to_wrap = model.model if is_actor(model) else model

        if isinstance(model_to_wrap, FSDPModule):
            return model

        self.report_memory("before FSDP2 wrap model pos2")

        # this is not sufficient enough, for example, it will only return Qwen2DecoderLayer for qwen2
        default_transformer_cls_names_to_wrap = getattr(model_to_wrap, "_no_split_modules", [])

        # so we add some manual rules
        transformer_cls_names_to_wrap = default_transformer_cls_names_to_wrap
        for cls in manual_transformer_cls_names_to_wrap:
            if cls not in transformer_cls_names_to_wrap:
                transformer_cls_names_to_wrap.append(cls)

        if self.mixed_mm_data:
            # Note:if we have mixed multi-modal data across DP ranks
            # (e.g. some ranks pure text, other ranks contains images)
            # we either keep vision model in full state, or keep it in FSDP's root module.
            # below we keep vit in root module to avoid stuck
            for cls_name in vit_transformer_cls_names:
                transformer_cls_names_to_wrap.remove(cls_name)

        transformer_cls_to_wrap = list()  # noqa
        vit_transformer_cls = list()  # noqa
        for layer_class in transformer_cls_names_to_wrap:
            transformer_cls = get_module_class_from_name(model_to_wrap, layer_class)
            if transformer_cls is not None:
                transformer_cls_to_wrap.append(transformer_cls)

        # Note: in this way, we keep vit in full state by passing no_shard_mesh
        #       this is less memory efficient compared to keep vit in root module
        # vit_transformer_cls = list()
        # for layer_class in vit_transformer_cls_names:
        #     transformer_cls = get_module_class_from_name(model_to_wrap, layer_class)
        #     if transformer_cls is not None:
        #         vit_transformer_cls.append(transformer_cls)

        if len(transformer_cls_to_wrap) == 0:
            self.print("len(transformer_cls_to_wrap)=0", model_to_wrap)
            raise NotImplementedError("len(transformer_cls_to_wrap) == 0, please check the wrapping rules!")

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16 if naive_mp_training else None,
            reduce_dtype=torch.float32 if naive_mp_training else None,
        )
        mesh = None
        world_size = torch.distributed.get_world_size()
        if shard_size != -1:
            assert world_size % shard_size == 0
            mesh = init_device_mesh(
                "cuda", (world_size // shard_size, shard_size), mesh_dim_names=("replicate", "shard")
            )
        else:
            mesh = init_device_mesh("cuda", (1, world_size), mesh_dim_names=("replicate", "shard"))

        no_shard_mesh = init_device_mesh("cuda", (world_size, 1), mesh_dim_names=("replicate", "shard"))  # noqa

        offload_policy = CPUOffloadPolicy() if is_training and self.args.fsdp_cpu_offload else OffloadPolicy()
        fsdp_kwargs = {
            "reshard_after_forward": reshard_after_forward,
            "mp_policy": mp_policy,
            "offload_policy": offload_policy,
            "mesh": mesh,
        }
        # fsdp_kwargs_no_shard = fsdp_kwargs.copy()
        # fsdp_kwargs_no_shard['mesh'] = no_shard_mesh
        # fsdp_kwargs_no_shard['reshard_after_forward'] = True
        # fsdp_kwargs_vit = fsdp_kwargs_no_shard if self.no_shard_vit else fsdp_kwargs

        for cls_to_wrap in transformer_cls_to_wrap:
            for module in model_to_wrap.modules():
                if isinstance(module, cls_to_wrap):
                    # if cls_to_wrap in vit_transformer_cls:
                    #     fully_shard(module, **fsdp_kwargs_vit)
                    fully_shard(module, **fsdp_kwargs)

        if not self.args.fused_linear_logprob:
            # In fused linear logprob implementation, lm_head.weight is directly used to calculate logprob
            # If lm_head is sharded here, it may stuck in actor forward.
            for name, module in model_to_wrap.named_modules():
                if "lm_head" in name:
                    fully_shard(module, **fsdp_kwargs)

        fully_shard(model_to_wrap, **fsdp_kwargs)

        if naive_mp_training:
            # cast model into fp32 to create optimizer with fp32 states
            # https://github.com/pytorch/torchtitan/issues/1133#issuecomment-2824429682
            model_to_wrap = model_to_wrap.to(torch.float32)

        if hasattr(model_to_wrap, "generate"):
            register_fsdp_forward_method(model_to_wrap, "generate")

        if is_meta_initialized(model_to_wrap):
            model.to_empty(device="cuda")

        self.print(f"after _fsdp2_init_model: {model_to_wrap}")
        self.report_memory("after FSDP2 wrap model")
        return model

    @contextmanager
    def init_model_context(self, meta_init=False):
        """
        Context manager for model initialization, for large models it can be initialized on Meta device.

        :param meta_init: if init on meta device
        :type meta_init: bool
        """
        try:
            if meta_init:
                with torch.device("meta"):
                    yield
            else:
                yield
        finally:
            self.report_memory("Finished init_model_context")

    def unwrap_model(self, model) -> nn.Module:
        """
        Unwraps the model from any wrapper modules.

        :param model: The model to unwrap.
        :type model: torch.nn.Module

        :return: The unwrapped model.
        :rtype: torch.nn.Module

        Example::

            >>> unwrapped_model = strategy.unwrap_model(wrapped_model)
        """
        if hasattr(model, "module"):
            return model.module
        else:
            return model

    def save_model(self, model: nn.Module, tokenizer: Any, output_dir: str, **kwargs) -> None:
        """
        Save the model, its configuration, and tokenizer in Hugging Face format.
        In LoRA mode, this saves the adapter. In full mode, this saves the full model.

        This method handles gathering and saving the full model parameters in a distributed setting.
        Only rank 0 process saves the model to disk.

        :param model: The model to save
        :type model: torch.nn.Module
        :param tokenizer: The tokenizer to save
        :param output_dir: Directory to save the model to
        :type output_dir: str
        :param kwargs: Additional arguments to pass to model.save_pretrained
        """
        # Determine the model to save (unwrap ActorVL or similar wrappers)
        actual_model = model.model if is_actor(model) or hasattr(model, "model") else model

        # [Gather Configuration]
        # In this environment, get_model_state_dict uses 'options' and 'StateDictOptions'.
        # We want the full state dict collected to rank 0 (cpu_offload=True to avoid OOM).
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)

        # get_model_state_dict is a collective call, must be called on ALL ranks.
        # It internally interacts with FSDP modules to perform the All-Gather.
        state_dict = get_model_state_dict(actual_model, options=opts)

        if self.is_rank_0():
            os.makedirs(output_dir, exist_ok=True)

            # Use save_pretrained if available (handles HF and PEFT)
            # PEFT's save_pretrained will automatically filter for adapter weights if state_dict is provided
            if hasattr(actual_model, "save_pretrained"):
                actual_model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=True)
            else:
                # Fallback to torch.save
                save_path = os.path.join(output_dir, "pytorch_model.bin")
                torch.save(state_dict, save_path)

            # Save the tokenizer
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)

            self.print(f"Hugging Face model saved to {output_dir}")

        # Ensure all ranks wait for rank 0 to finish saving
        dist.barrier()

    def save_ckpt(
        self,
        model,
        save_dir,
        tag=None,
        max_num=3,
        max_mem=1000,
        client_state={},
        save_latest=True,
        optimizer=None,
        scheduler=None,
    ):  # pylint: disable=R0917,W0102
        """
        Save model checkpoints in a distributed environment with automatic rotation.

        This method saves the sharded model weights to disk and manages the number and total size
        of checkpoints by removing older ones when necessary. It ensures proper synchronization
        between distributed processes.

        :param model: The model to save, typically an FSDP-wrapped model
        :type model: torch.nn.Module

        :param save_dir: Directory where checkpoints will be saved
        :type save_dir: str

        :param optimizer: The optimizer to save
        :type optimizer: torch.optim, optional

        :param scheduler: The scheduler to save
        :type scheduler: torch.lr_scheduler, optional

        :param tag: Subdirectory name for this specific checkpoint
        :type tag: str, optional

        :param max_num: Maximum number of checkpoints to keep
        :type max_num: int, default=3

        :param max_mem: Maximum disk space in GB for all checkpoints
        :type max_mem: int, default=1000

        :param client_state: Additional state to save (not currently used)
        :type client_state: dict, default={}

        :param save_latest: Whether to save a copy as the latest checkpoint (not used)
        :type save_latest: bool, default=True

        Example::

            >>> trainer.save_ckpt(model, "checkpoints", tag="step_1000", max_num=5)
        """
        if self.is_rank_0():
            os.makedirs(save_dir, exist_ok=True)
            while True:
                subdirs = sorted(
                    [(os.path.join(save_dir, d), os.path.getmtime(os.path.join(save_dir, d)))
                     for d in os.listdir(save_dir)
                     if os.path.isdir(os.path.join(save_dir, d))],
                    key=lambda x: x[1],
                )

                if len(subdirs) >= max_num:
                    oldest_dir = subdirs[0][0]
                    if os.path.exists(oldest_dir):
                        shutil.rmtree(oldest_dir)
                        self.print(f"Deleted oldest ckpt {oldest_dir}")
                else:
                    break

        dist.barrier()

        # Determine the model to save (unwrap ActorVL or similar wrappers)
        actual_model = model.model if is_actor(model) or hasattr(model, "model") else model

        # [Sharded State Dict]
        # For DCP, we want the sharded state dict (full_state_dict=False by default)
        opts = StateDictOptions(full_state_dict=False, cpu_offload=False)
        fsdp_state_dict = get_model_state_dict(actual_model, options=opts)

        fp = os.path.join(save_dir, tag)
        os.makedirs(fp, exist_ok=True)
        dcp.save(fsdp_state_dict, checkpoint_id=fp)
        self.print(f"DCP checkpoint saved to {fp}")

        if optimizer is not None:
            opt_base_dir = os.path.join(save_dir, tag, "optim_states")
            os.makedirs(opt_base_dir, exist_ok=True)
            if is_mp_optimizer(optimizer):
                opt_ckpt_path = os.path.join(opt_base_dir, f"_rank{torch.distributed.get_rank()}")
                torch.save(optimizer.state_dict(), opt_ckpt_path)
            else:
                # DCP can only be use with naive optimizer
                fsdp_optim_state_dict = get_optimizer_state_dict(actual_model, optimizer)
                dcp.save(fsdp_optim_state_dict, checkpoint_id=opt_base_dir)

        client_ckpt_path = os.path.join(fp, "client_state.pt")
        torch.save(client_state, client_ckpt_path)
        self.print(f"client_state save to {client_ckpt_path}, content: {client_state}")

        if scheduler is not None:
            sched_ckpt_path = os.path.join(fp, "scheduler_state.pt")
            scheduler_state = scheduler.state_dict()
            torch.save(scheduler_state, sched_ckpt_path)
            self.print(f"scheduler_state save to {sched_ckpt_path}")

    def load_ckpt(  # pylint: disable=R0917
        self,
        model,
        load_dir,
        optimizer=None,
        scheduler=None,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
        **kwargs,
    ):
        """
        Load model checkpoints in a distributed environment.

        This method loads sharded model weights from disk for each distributed process.
        It handles the proper loading of FSDP-sharded state dictionaries.

        :param model: The model to load weights into, typically an FSDP-wrapped model
        :type model: torch.nn.Module

        :param optimizer: The optimizer to load weights into
        :type optimizer: torch.optim, optional

        :param scheduler: The scheduler to load weights into
        :type scheduler: torch.lr_scheduler, optional

        :param load_dir: Directory containing the checkpoints
        :type load_dir: str

        :param load_module_strict: Whether to strictly enforce that the keys in the model state dict match
        :type load_module_strict: bool, default=True

        :param load_optimizer_states: Whether to load optimizer states
        :type load_optimizer_states: bool, default=True

        :param load_lr_scheduler_states: Whether to load learning rate scheduler states
        :type load_lr_scheduler_states: bool, default=True

        :return: A tuple of (load_dir, client_states) where load_dir is the directory from which the
                 checkpoint was loaded and client_states contains additional saved state
        :rtype: tuple

        Example::

            >>> load_dir, client_states = trainer.load_ckpt(model, "checkpoints/step_1000")
        """

        latest_path = find_latest_checkpoint_dir(load_dir)
        self.print(f"Loading DCP checkpoint from {latest_path}")

        fsdp_state_dict = get_model_state_dict(model)

        dcp.load(state_dict=fsdp_state_dict, checkpoint_id=latest_path)
        set_model_state_dict(model, fsdp_state_dict)

        if optimizer is not None and load_optimizer_states:
            opt_ckpt_path = os.path.join(latest_path, "optim_states")
            if not os.path.exists(opt_ckpt_path):
                self.print(f"WARNING: Opt ckpt {opt_ckpt_path} does not exist! Skipping ... ")
            else:
                self.print(f"Loading DCP checkpoint from {opt_ckpt_path}")

                if is_mp_optimizer(optimizer):
                    opt_ckpt_path = os.path.join(latest_path, "optim_states", f"_rank{torch.distributed.get_rank()}")
                    opt_states = torch.load(opt_ckpt_path)
                    optimizer.load_state_dict(opt_states)
                else:
                    # DCP can only be use with naive optimizer
                    fsdp_optim_state_dict = get_optimizer_state_dict(model, optimizer)
                    opt_ckpt_path = os.path.join(latest_path, "optim_states")
                    dcp.load(state_dict=fsdp_optim_state_dict, checkpoint_id=opt_ckpt_path)
                    set_optimizer_state_dict(model, optimizer, fsdp_optim_state_dict)

        if scheduler is not None and load_lr_scheduler_states:
            sched_ckpt_path = os.path.join(latest_path, "scheduler_state.pt")
            if not os.path.exists(sched_ckpt_path):
                self.print(f"WARNING: Scheduler ckpt {sched_ckpt_path} does not exist! Skipping ... ")
            else:
                self.print(f"Loading lr_scheduler_states from {sched_ckpt_path}")
                loaded_scheduler_state = torch.load(sched_ckpt_path)
                scheduler.load_state_dict(loaded_scheduler_state)

        client_states = {}
        client_ckpt_path = os.path.join(latest_path, "client_state.pt")
        if os.path.exists(client_ckpt_path):
            client_states = torch.load(client_ckpt_path)
            self.print(f"Loaded client states: {client_states=}")

        self.sync_and_clear_cache()
        return latest_path, client_states

    def maybe_offload_optimizer(self, optimizer):
        """
        Offload FSDP optimizer states to CPU if adam_offload is enabled.

        :param optimizer: The optimizer to potentially offload
        :type optimizer: torch.optim.Optimizer

        :return: The offloaded optimizer if adam_offload is enabled, otherwise the original optimizer
        :rtype: torch.optim.Optimizer
        """
        if self.args.adam_offload:
            return offload_fsdp_optimizer(optimizer)

    def maybe_load_optimizer(self, optimizer, device=torch.cuda.current_device()):
        """
        Load FSDP optimizer states back to GPU if adam_offload is enabled.

        :param optimizer: The optimizer to potentially load
        :type optimizer: torch.optim.Optimizer
        :param device: The device to load the optimizer to
        :type device: torch.device

        :return: The loaded optimizer if adam_offload is enabled, otherwise the original optimizer
        :rtype: torch.optim.Optimizer
        """
        if self.args.adam_offload:
            return load_fsdp_optimizer(optimizer, device)

    @torch.no_grad()
    def offload_model(self, models, empty_cache: bool = True):
        """
        Offload model(s) to CPU to free GPU memory.

        This method moves the model parameters and buffers to CPU memory, which can be useful
        for memory management during training when certain models are not actively being used.

        :param models: Single model or list/tuple of models to offload
        :type models: torch.nn.Module or list/tuple of torch.nn.Module
        :param empty_cache: Whether to clear CUDA cache after offloading
        :type empty_cache: bool, default=True

        Example::

            >>> strategy.offload_model([actor_model, critic_model])
        """
        def offload_single(model):
            """
            Offload a single model to CPU.

            :param model: The model to offload
            :type model: torch.nn.Module
            """
            if not isinstance(model, torch.nn.Module):
                return
            model.to(torch.device("cpu"))
            # the following code does not work for QwenVL+FSDP2, it will result in some unreleased memory.
            # for param in model.parameters():
            #     param = param.to(torch.device("cpu"), non_blocking=True)
            # for buf in model.buffers():
            #     buf.data = buf.data.to(torch.device("cpu"), non_blocking=True)

        if isinstance(models, (list, tuple)):
            for model in models:
                offload_single(model)
        else:
            offload_single(models)

        if empty_cache:
            self.sync_and_clear_cache()
        self.report_memory("after offload_model")

    @torch.no_grad()
    def reload_model(self, models):
        """
        Reload model(s) from CPU back to GPU.

        This method moves the model parameters and buffers back to GPU memory after they
        have been offloaded to CPU.

        :param models: Single model or list/tuple of models to reload
        :type models: torch.nn.Module or list/tuple of torch.nn.Module

        Example::

            >>> strategy.reload_model([actor_model, critic_model])
        """
        device = torch.cuda.current_device()

        def reload_single(model):
            """
            Reload a single model to GPU.

            :param model: The model to reload
            :type model: torch.nn.Module
            """
            if not isinstance(model, torch.nn.Module):
                return
            model.to(device)
            # the following code does not work for QwenVL+FSDP2, it will result in some unreleased memory.
            # for param in model.parameters():
            #     param.data = param.data.to(device, non_blocking=True)
            # for buf in model.buffers():
            #     buf.data = buf.data.to(device, non_blocking=True)

        if isinstance(models, (list, tuple)):
            for model in models:
                reload_single(model)
        else:
            reload_single(models)
        self.report_memory("after reload_model")


def is_mp_optimizer(optim):
    """
    Check if an optimizer is an instance of FSDPadaptOptimizer.

    This function determines whether the provided optimizer is a model parallel
    optimizer specifically designed for FSDP (Fully Sharded Data Parallel).

    :param optim: The optimizer to check
    :type optim: torch.optim.Optimizer or similar

    :return: True if the optimizer is an instance of FSDPadaptOptimizer, False otherwise
    :rtype: bool

    Example::

        >>> optimizer = FSDPadaptOptimizer(model_parameters, lr=0.01)
        >>> is_mp = is_mp_optimizer(optimizer)
        >>> print(is_mp)
        True
    """
    return isinstance(optim, FSDPadaptOptimizer)
