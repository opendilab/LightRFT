"""
DeepSpeed Strategy Module for lightrft.

This module provides a DeepSpeed-based strategy for training and inference in lightrft.
It handles model initialization, optimization, parameter management, and checkpoint operations
using DeepSpeed's distributed training capabilities. The strategy supports various DeepSpeed
features including ZeRO optimization stages, mixed precision training, and parameter offloading.
"""

import os
import shutil
from contextlib import contextmanager
from typing import List, Tuple, Union

import deepspeed
import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from peft import PeftModel, get_peft_model_state_dict
from torch import distributed as dist
from torch import nn, optim
from torch.optim import Optimizer
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from lightrft.strategy.strategy_base import StrategyBase, is_actor
from lightrft.strategy.utils.optimizer_utils import get_optimizer_grouped_parameters

from .deepspeed_utils import (
    _z3_params_to_fetch,
    get_eval_ds_config,
    get_train_ds_config,
)

ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]


class DeepspeedStrategy(StrategyBase):
    """
    DeepSpeed implementation of the training strategy.

    Modified from https://github.com/OpenRLHF/OpenRLHF, with these changes:
    1. inherits from StrategyBase, add some api
    2. removed ring-attn related code

    :param seed: Random seed for reproducibility
    :type seed: int
    :param max_norm: Maximum gradient norm for gradient clipping (0.0 means no clipping)
    :type max_norm: float
    :param micro_train_batch_size: Batch size for a single GPU/process
    :type micro_train_batch_size: int
    :param train_batch_size: Global batch size across all GPUs/processes
    :type train_batch_size: int
    :param zero_stage: DeepSpeed ZeRO optimization stage (0, 1, 2, or 3)
    :type zero_stage: int
    :param bf16: Whether to use bfloat16 precision
    :type bf16: bool
    :param args: Additional arguments for configuration
    :type args: object
    """

    def __init__(  # pylint: disable=R0917
        self,
        seed: int = 42,
        max_norm: float = 0.0,
        micro_train_batch_size: int = 1,
        train_batch_size: int = 1,
        zero_stage: int = 2,
        bf16: bool = True,
        args=None,
    ) -> None:
        super().__init__(seed, max_norm, micro_train_batch_size, train_batch_size, args)

        self.bf16 = bf16

        self.stage = zero_stage
        # TODO: refactor this mark
        self.is_rlhf = False

        self.print("DeepspeedStrategy Initialized")

    def create_optimizer(self, model, **kwargs) -> Optimizer:
        """
        Create an optimizer for the given model.

        :param model: The model to create an optimizer for
        :type model: nn.Module
        :param kwargs: Additional arguments for the optimizer

        :return: The created optimizer
        :rtype: Optimizer
        """
        if is_actor(model):
            model = model.model
        # Optimizer
        AdamOptimizer = DeepSpeedCPUAdam if self.adam_offload else FusedAdam
        optim_params = get_optimizer_grouped_parameters(model, kwargs["weight_decay"])
        optim = AdamOptimizer(optim_params, **kwargs)
        return optim

    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, **kwargs) -> None:
        """
        Perform backward pass to compute gradients.

        :param loss: The loss tensor to backpropagate
        :type loss: torch.Tensor
        :param model: The model being trained
        :type model: nn.Module
        :param optimizer: The optimizer for the model
        :type optimizer: optim.Optimizer
        :param kwargs: Additional arguments
        """
        if is_actor(model):
            model = model.model
        model.backward(loss)

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

        :param optimizer: The optimizer to step
        :type optimizer: optim.Optimizer
        :param model: The model being trained
        :type model: nn.Module
        :param scheduler: The learning rate scheduler
        :param name: Name identifier for the model
        :type name: str
        :param kwargs: Additional arguments
        """
        if is_actor(model):
            model = model.model
        model.step()

    def unwrap_model(self, model) -> nn.Module:
        """
        Unwrap the model from any wrappers to access the base model.

        :param model: The model to unwrap
        :type model: nn.Module

        :return: The unwrapped model
        :rtype: nn.Module
        """
        if is_actor(model):
            return self.unwrap_model(model.model)
        elif hasattr(model, "module"):
            return model.module
        else:
            return model

    def prepare(self,
                *models_or_model_optim_pairs: ModelOrModelOptimPair,
                is_rlhf=False) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        """
        Prepare models and optimizers for DeepSpeed training.

        Expected input format for RLHF:
        tuple: ((actor, actor_optim, actor_scheduler),
                (critic, critic_optim, critic_scheduler),
                reward_models,
                initial_model)

        :param models_or_model_optim_pairs: Models or (model, optimizer, scheduler) tuples
        :param is_rlhf: Whether this is for RLHF training
        :type is_rlhf: bool

        :return: Prepared models or model-optimizer pairs
        :rtype: Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]
        """
        ret = []
        self.is_rlhf = is_rlhf
        for arg in models_or_model_optim_pairs:
            if isinstance(arg, (tuple, list)):
                if not all(isinstance(item, torch.nn.Module) for item in arg):
                    assert (
                        len(arg) == 3
                    ), f'Expect (model, optimizer, scheduler) pair, got a tuple with size "{len(arg)}"'
                    if arg[0] is not None:
                        ret.append(self._ds_init_train_model(*arg))
                    else:
                        ret.append((None, None, None))
                else:
                    ret.append(self.prepare_reward_models(arg))
            else:
                ret.append(self._ds_init_eval_model(arg))

        return ret[0] if len(ret) == 1 else ret

    def prepare_reward_models(self, reward_models):
        """
        Prepare reward models for DeepSpeed evaluation.

        :param reward_models: List of reward models to prepare
        :type reward_models: List[nn.Module]

        :return: List of prepared reward models
        :rtype: List[nn.Module]
        """
        return [self._ds_init_eval_model(model, param_offload=False) for model in reward_models]

    def _ds_init_train_model(self, model, optim, scheduler):
        """
        Initialize a model for DeepSpeed training.

        :param model: The model to initialize
        :type model: nn.Module
        :param optim: The optimizer for the model
        :type optim: Optimizer
        :param scheduler: The learning rate scheduler

        :return: Tuple of (initialized model, optimizer, scheduler)
        :rtype: Tuple[nn.Module, Optimizer, object]
        """
        is_actor_model = is_actor(model)
        ds_config = self.get_ds_train_config(is_actor_model)

        engine, optim, _, scheduler = deepspeed.initialize(
            model=model.model if is_actor_model else model,
            optimizer=optim,
            lr_scheduler=scheduler,
            config=ds_config,
            args={"local_rank": self.args.local_rank},
            dist_init_required=True,
        )
        if is_actor_model:
            model.model = engine
        else:
            model = engine

        return model, optim, scheduler

    def get_ds_train_config(self, is_actor):
        """
        Get the DeepSpeed configuration for training.

        :param is_actor: Whether the model is an actor model
        :type is_actor: bool

        :return: DeepSpeed configuration dictionary
        :rtype: dict
        """
        # DS Config
        ds_config = get_train_ds_config(
            offload=False,
            adam_offload=self.adam_offload,
            stage=self.stage,
            bf16=self.bf16,
            max_norm=self.max_norm,
            zpg=self.zpg,
            grad_accum_dtype=self.grad_accum_dtype,
            overlap_comm=self.overlap_comm,
        )

        ds_config["train_micro_batch_size_per_gpu"] = self.micro_train_batch_size
        train_batch_size = self.train_batch_size
        # corner case for ptx loss (backward twice)
        if self.is_rlhf and is_actor and self.args.pretrain_data is not None:
            train_batch_size *= 2
        ds_config["train_batch_size"] = train_batch_size * self.ring_attn_size

        return ds_config

    def _ds_init_eval_model(self, model, param_offload=False):
        """
        Initialize a model for DeepSpeed evaluation.

        :param model: The model to initialize
        :type model: nn.Module
        :param param_offload: Whether to offload parameters to CPU
        :type param_offload: bool

        :return: Initialized model
        :rtype: nn.Module
        """
        if not model:
            return model
        is_actor_model = is_actor(model)
        ds_config = self.get_ds_eval_config(offload=param_offload)

        engine, *_ = deepspeed.initialize(
            model=model.model if is_actor_model else model,
            args={"local_rank": self.args.local_rank},
            config=ds_config,
            dist_init_required=True,
        )
        if is_actor_model:
            model.model = engine
        else:
            model = engine
        return model

    def _ds_init_inference_model(self, model):
        """
        Initialize a model for DeepSpeed inference.

        :param model: The model to initialize
        :type model: nn.Module

        :return: DeepSpeed inference engine
        :rtype: deepspeed.InferenceEngine
        """
        ds_engine = deepspeed.init_inference(
            model,
            tensor_parallel={"tp_size": self.world_size},
            dtype=torch.bfloat16,
            checkpoint=None,
            replace_with_kernel_inject=True,
        )
        self.report_memory("after _ds_init_inference_model ")
        return ds_engine

    def get_ds_eval_config(self, offload=False):
        """
        Get the DeepSpeed configuration for evaluation.

        :param offload: Whether to offload parameters to CPU
        :type offload: bool

        :return: DeepSpeed configuration dictionary
        :rtype: dict
        """
        # DS Config
        ds_config = get_eval_ds_config(offload=offload, stage=self.stage if self.stage == 3 else 0, bf16=self.bf16)
        ds_config["train_micro_batch_size_per_gpu"] = self.micro_train_batch_size
        ds_config["train_batch_size"] = self.train_batch_size * self.ring_attn_size

        return ds_config

    @contextmanager
    def init_model_context(self):
        """
        Context manager for initializing a model with DeepSpeed configuration.

        This sets up the HfDeepSpeedConfig for use with Hugging Face models.
        """
        try:
            _dscfg = HfDeepSpeedConfig(self.get_ds_eval_config())  # noqa

            yield
        finally:
            self.report_memory("Finished init_model_context")

    def moving_average(self, model, model_ema, beta=0.992, device="cpu"):
        """
        Update model_ema parameters with exponential moving average of model parameters.

        :param model: Source model
        :type model: nn.Module
        :param model_ema: Target model for EMA
        :type model_ema: nn.Module
        :param beta: EMA decay factor
        :type beta: float
        :param device: Device to perform operations on
        :type device: str
        """
        self.time_steps["ema"] += 1
        if self.time_steps["ema"] % self.accumulated_gradient == 0:
            with torch.no_grad():
                for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                    if param.requires_grad:
                        if self.stage != 3:
                            data = param.data.to(device)
                            param_ema.data.copy_((1 - beta) * data + beta * param_ema.data)
                        else:
                            # TODO: use prefiltering for efficiency
                            params_to_fetch = _z3_params_to_fetch([param, param_ema])
                            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=len(params_to_fetch) > 0):
                                data = param.data.to(device)
                                param_ema.data.copy_((1 - beta) * data + beta * param_ema.data)

    def load_model(  # pylint: disable=R0917
        self,
        model: nn.Module,
        path: str,
        map_location="cpu",
        strict: bool = False,
        key_replace_fn=None,
    ) -> None:
        """
        Load model weights from a file.

        :param model: The model to load weights into
        :type model: nn.Module
        :param path: Path to the saved model weights
        :type path: str
        :param map_location: Device to load the weights to
        :type map_location: str or torch.device
        :param strict: Whether to strictly enforce that the keys in state_dict match the model
        :type strict: bool
        :param key_replace_fn: Function to modify state dict keys
        :type key_replace_fn: callable
        """
        unwrapped_model = self.unwrap_model(model)
        state_dict = torch.load(path, map_location=map_location)
        if key_replace_fn:
            state_dict = key_replace_fn(state_dict)
        unwrapped_model.load_state_dict(state_dict, strict=strict)

    def save_model(self, model: nn.Module, tokenizer, output_dir, **kwargs) -> None:
        """
        Save a model, its configuration, and tokenizer to the specified output directory.

        Handles special cases for DeepSpeed ZeRO-2/3 parameters and PEFT models. For ZeRO parallelism,
        it gathers distributed parameters before saving. For PEFT models, it saves adapter weights
        appropriately based on the DeepSpeed stage.

        :param model: The model to save
        :type model: nn.Module
        :param tokenizer: The tokenizer to save
        :type tokenizer: PreTrainedTokenizer or similar
        :param output_dir: Directory where the model, config, and tokenizer will be saved
        :type output_dir: str
        :param kwargs: Additional arguments to pass to the model's save_pretrained method

        :return: None
        """
        if self.is_rank_0():
            os.makedirs(output_dir, exist_ok=True)

        # save model weights for ZeRO2/3
        model_to_save = self.unwrap_model(model)

        # gather parameters
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            # only gather z3 params
            params_to_fetch = _z3_params_to_fetch([v])
            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=len(params_to_fetch) > 0):
                vv = v.data.cpu()
                if self.is_rank_0():
                    output_state_dict[k] = vv

        if self.is_rank_0():
            state_dict = model_to_save.state_dict()

            # copy named_buffers with `persistent=True`
            for k, v in model_to_save.named_buffers():
                if k not in state_dict:
                    continue
                vv = v.data.cpu()
                output_state_dict[k] = vv

            state_dict_keys = set(state_dict.keys())
            output_state_dict_keys = set(output_state_dict.keys())

            # corner case for tie_word_embeddings, such as Qwen2-0.5B
            if getattr(model_to_save.config, "tie_word_embeddings", False) and "lm_head.weight" in state_dict_keys:
                state_dict_keys.remove("lm_head.weight")

            assert state_dict_keys.issubset(
                output_state_dict_keys
            ), f"mismatch keys {output_state_dict_keys.symmetric_difference(state_dict_keys)}"

            # only save peft weights https://github.com/microsoft/DeepSpeed/issues/4295
            if isinstance(model_to_save, PeftModel):
                model_to_save.save_pretrained(output_dir, **kwargs)
                if self.stage == 3:
                    torch.save(
                        get_peft_model_state_dict(model_to_save, output_state_dict),
                        os.path.join(output_dir, "adapter_model.bin"),
                    )
                    filename = os.path.join(output_dir, "adapter_model.safetensors")
                    if os.path.exists(filename):
                        os.remove(filename)
            else:
                # save model
                model_to_save.save_pretrained(output_dir, state_dict=output_state_dict, **kwargs)

            # save config
            output_config_file = os.path.join(output_dir, "config.json")
            model_to_save.config.to_json_file(output_config_file)
            # save tokenizer
            tokenizer.save_pretrained(output_dir)

            # for models not in AutoModel, copy python module files
            train_from_model_path = model_to_save.config._name_or_path
            if os.path.exists(train_from_model_path):
                for filename in os.listdir(train_from_model_path):
                    if filename.endswith(".py"):
                        shutil.copy(os.path.join(train_from_model_path, filename), os.path.join(output_dir, filename))

    def save_ckpt(
        self,
        model,
        save_dir,
        tag=None,
        max_num=3,
        max_mem=1000,
        client_state={},
        save_latest=True,
        **_kwargs
    ):  # pylint: disable=R0917,W0102
        """
        Save a DeepSpeed model checkpoint with automatic management of checkpoint storage.

        This function manages checkpoint storage by limiting the number of checkpoints and total
        storage size. It automatically removes the oldest checkpoints when limits are exceeded.

        :param model: The DeepSpeed model to save
        :type model: deepspeed.DeepSpeedEngine
        :param save_dir: Directory where checkpoints will be saved
        :type save_dir: str
        :param tag: Optional tag for the checkpoint (e.g., iteration number)
        :type tag: str, optional
        :param max_num: Maximum number of checkpoints to keep
        :type max_num: int, default=3
        :param max_mem: Maximum storage size for checkpoints in GB
        :type max_mem: int, default=1000
        :param client_state: Additional client state to save with the checkpoint
        :type client_state: dict, default={}
        :param save_latest: Whether to save a symbolic link to the latest checkpoint
        :type save_latest: bool, default=True
        :param _kwargs: dict, not used keys, e.g. optimizer, scheduler

        :return: None
        :raises AssertionError: If model is not a DeepSpeedEngine
        """
        assert isinstance(model, deepspeed.DeepSpeedEngine)
        if self.is_rank_0():
            os.makedirs(save_dir, exist_ok=True)
            MAX_SIZE = max_mem * 1024 ** 3  # Convert GB to bytes

            while True:
                subdirs = sorted(
                    [(os.path.join(save_dir, d), os.path.getmtime(os.path.join(save_dir, d)))
                     for d in os.listdir(save_dir)
                     if os.path.isdir(os.path.join(save_dir, d))],
                    key=lambda x: x[1],
                )
                total_size = sum(
                    os.path.getsize(os.path.join(dirpath, f))
                    for subdir, _ in subdirs
                    for dirpath, _, filenames in os.walk(subdir)
                    for f in filenames
                )

                if len(subdirs) >= max_num or total_size > MAX_SIZE:
                    oldest_dir = subdirs[0][0]
                    if os.path.exists(oldest_dir):
                        shutil.rmtree(oldest_dir)
                        self.print(f"Deleted oldest ckpt {oldest_dir}")
                else:
                    break

        dist.barrier()
        model.save_checkpoint(save_dir, tag=tag, client_state=client_state, save_latest=save_latest)

    def load_ckpt(  # pylint: disable=R0917
        self,
        model,
        load_dir,
        tag=None,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
        load_module_only=False,
        **_kwargs,
    ):
        """
        Load a DeepSpeed model checkpoint from the specified directory.

        This function wraps DeepSpeed's checkpoint loading functionality with error handling.

        :param model: The DeepSpeed model to load the checkpoint into
        :type model: deepspeed.DeepSpeedEngine
        :param load_dir: Directory from which to load the checkpoint
        :type load_dir: str
        :param tag: Optional tag to specify which checkpoint to load
        :type tag: str, optional
        :param load_module_strict: Whether to strictly enforce that the keys in the model
                                  state dict match the keys in the checkpoint
        :type load_module_strict: bool, default=True
        :param load_optimizer_states: Whether to load optimizer states from checkpoint
        :type load_optimizer_states: bool, default=True
        :param load_lr_scheduler_states: Whether to load learning rate scheduler states
        :type load_lr_scheduler_states: bool, default=True
        :param load_module_only: Whether to load only the module weights and not optimizer or scheduler states
        :type load_module_only: bool, default=False

        :return: A tuple containing the checkpoint path and loaded states
        :rtype: tuple(str, dict)
        :raises AssertionError: If model is not a DeepSpeedEngine
        :raises Exception: If loading the checkpoint fails
        """
        assert isinstance(model, deepspeed.DeepSpeedEngine)
        load_path, states = model.load_checkpoint(
            load_dir,
            tag,
            load_module_strict=load_module_strict,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
            load_module_only=load_module_only,
        )
        if load_path is None:
            raise Exception(f"[deepspeed] failed to resume from checkpoint {load_dir}")
        return load_path, states
