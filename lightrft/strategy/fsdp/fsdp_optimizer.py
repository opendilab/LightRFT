"""
FSDP Optimizer Module for PyTorch.

This module provides optimizers and utilities for working with PyTorch's Fully Sharded Data Parallel (FSDP)
training. It includes an adapted optimizer for FSDP that handles gradient scaling, clipping, and state
management, as well as utility functions for offloading and loading optimizer states.

The main components include:
- FSDPadaptOptimizer: A wrapper optimizer that handles mixed precision training with FSDP
- Utility functions for optimizer state management and memory optimization
- Support for gradient scaling, clipping, and overflow detection
- Efficient FP16/FP32 parameter conversion and synchronization

Example::

    import torch
    from torch.optim import AdamW
    from lightrft.trainer.fsdp_optimizer import FSDPadaptOptimizer

    # Create base optimizer
    base_optimizer = AdamW(model.parameters(), lr=1e-4)

    # Wrap with FSDP adapter
    fsdp_optimizer = FSDPadaptOptimizer(base_optimizer)

    # Training loop
    for batch in dataloader:
        loss = model(batch)
        fsdp_optimizer.backward(loss)
        success = fsdp_optimizer.step()
        if not success:
            print("Gradient overflow detected, skipping step")
"""

import torch
import torch.distributed as dist
from torch.optim import Optimizer

try:
    from torch.distributed.tensor import DTensor

    DTENSOR_SUPPORTED = True
except (ModuleNotFoundError, ImportError):
    DTENSOR_SUPPORTED = False

from lightrft.utils import get_current_device

from .fsdp_utils import BaseOptimizer, DynamicGradScaler


class FSDPadaptOptimizer(BaseOptimizer):
    """
    Optimizer wrapper for PyTorch FSDP (Fully Sharded Data Parallel).

    This optimizer handles the necessary components for mixed precision training with FSDP:

    - Gradient scaling for numerical stability in mixed precision training
    - Gradient clipping and unscaling to prevent gradient explosion
    - State dictionary management for checkpointing and model saving
    - Efficient FP16/FP32 parameter conversion and synchronization
    - Overflow detection and recovery mechanisms

    The optimizer maintains separate FP16 and FP32 parameter groups where FP16 parameters
    share memory space with the model's FlatParam, while FP32 parameters are used for
    the actual optimization step to maintain numerical precision.

    :param optimizer: The base optimizer to wrap (e.g., AdamW, SGD)
    :type optimizer: torch.optim.Optimizer

    Example::

        import torch
        from torch.optim import AdamW

        base_optimizer = AdamW(model.parameters(), lr=1e-4)
        fsdp_optimizer = FSDPadaptOptimizer(base_optimizer)

        # Training step
        loss = model(batch)
        fsdp_optimizer.backward(loss)
        success = fsdp_optimizer.step()
    """
    def __init__(
        self,
        optimizer: Optimizer,
    ):
        """
        Initialize the FSDP adapted optimizer.

        :param optimizer: The base optimizer to wrap
        :type optimizer: torch.optim.Optimizer
        """
        super().__init__(optim=optimizer)

        # gradient scaler for mixed precision training
        self.grad_scaler = DynamicGradScaler(initial_scale=1.0, growth_factor=1.0, backoff_factor=1.0, max_scale=1.0)

        # gradient clipping threshold
        self._clip_grad_norm = 1.0

        # padding data for computing norm when no gradients are available
        self.padding_grad = torch.zeros([32], dtype=torch.bfloat16, device=get_current_device())
        self.padding_tensor = torch.zeros([32], dtype=torch.bfloat16, device=get_current_device())

        # fp16 and fp32 parameter groups
        # fp16 shares memory space with model.FlatParam, fp32 shares memory space with optim.param_group
        self._fp16_param_groups = dict()
        self._fp32_param_tensor_groups = dict()

        # initialize fp16 and fp32 parameter groups
        for group_idx, param_group in enumerate(self.optim.param_groups):
            group_params = param_group["params"]

            # store reference to fp16 FlatParam storage
            self._fp16_param_groups[group_idx] = group_params

            # create fp32 copies of parameters for optimization
            fp32_tensor_param = [param.data.float() for param in group_params]
            self._fp32_param_tensor_groups[group_idx] = fp32_tensor_param

            # replace optimizer parameter group with fp32 copies
            param_group["params"] = fp32_tensor_param

    @property
    def loss_scale(self):
        """
        Get the current loss scale value used for gradient scaling.

        :return: The current loss scale tensor
        :rtype: torch.Tensor
        """
        return self.grad_scaler.scale

    def backward(self, loss, retain_graph=False):
        """
        Perform backward pass with loss scaling for mixed precision training.

        The loss is scaled to prevent gradient underflow in FP16 computations.

        :param loss: The loss tensor to backpropagate
        :type loss: torch.Tensor
        :param retain_graph: If True, the computation graph will be retained for multiple backward passes
        :type retain_graph: bool

        Example::

            loss = criterion(outputs, targets)
            optimizer.backward(loss)
        """
        loss = self.loss_scale * (loss.float())
        loss.backward(retain_graph=retain_graph)

    def _compute_norm_with_fsdp_flatten(self, group_id, norm_type=2):
        """
        Compute the gradient norm for a parameter group with FSDP flattened parameters.

        This method handles the computation of gradient norms across distributed processes,
        taking into account FSDP's parameter flattening and sharding. It supports both
        regular tensors and DTensor for distributed computation.

        :param group_id: The parameter group ID to compute norm for
        :type group_id: int
        :param norm_type: The type of norm to compute (default: 2 for L2 norm)
        :type norm_type: int

        :return: The computed gradient norm, or -1 if overflow is detected
        :rtype: torch.Tensor
        """
        params = [p for p in self._fp16_param_groups[group_id] if p.untyped_storage().size() != 0]
        gradients = [p.grad for p in params if p.untyped_storage().size() != 0]

        # use padding tensors if no valid parameters/gradients found
        if len(params) <= 0 or len(gradients) <= 0:
            gradients = self.padding_grad
            params = self.padding_tensor

        # compute individual gradient norms (adapted from DeepSpeed)
        grad_norms = []
        for g, p in zip(gradients, params):
            grad_norms.append(g.double().norm(2))

        # compute total norm across all parameters
        if len(grad_norms) == 0:
            # FIX https://github.com/microsoft/DeepSpeed/issues/3564
            # handle edge case when no gradients are available
            total_norm_cuda = torch.tensor(0, dtype=gradients[0].dtype).to(get_current_device()).double()
        else:
            total_norm_cuda = torch.sum(torch.pow(torch.stack(grad_norms), 2))
            # if grad_norm is DTensor, it bahaves like:
            # stack_out_full = stack_out.full_tensor()
            # pow_out_full = torch.pow(stack_out_full, 2)
            # torch.sum(pow_out_full)

        # handle DTensor case for distributed computation
        if DTENSOR_SUPPORTED and isinstance(total_norm_cuda, DTensor):
            # 20250422(sdx): when DTensor enbaled, the output of torch.pow is REPLICATED across FSDP group,
            # and already the sum of suqare, so we should not do reduce-sum again.
            # DTensor output is already replicated across FSDP group
            total_norm_cuda = total_norm_cuda.full_tensor()
        else:
            # reduce across processes in Zero3 group
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.SUM)
            print(f"after reduce {total_norm_cuda=}", flush=True)

        total_norm = total_norm_cuda ** (1.0 / norm_type)

        # check for overflow conditions
        norm_is_inf = total_norm.isinf()
        norm_is_nan = total_norm.isnan()
        inf_or_nan = norm_is_nan.logical_or(norm_is_inf)

        # return -1 if overflow detected, otherwise return the norm
        err = torch.tensor(-1.0, device=get_current_device(), dtype=torch.float)
        total_norm = inf_or_nan * err + inf_or_nan.logical_not() * total_norm

        return total_norm

    def zero_grad(self):
        """
        Set gradients of all FP16 parameters to None.

        This method clears gradients from the FP16 parameter groups that are used
        for the forward pass and gradient computation.
        """
        for _, param_group in self._fp16_param_groups.items():
            for param in param_group:
                param.grad = None

    def step(self):
        """
        Perform a single optimization step with overflow detection and gradient processing.

        This method orchestrates the complete optimization process:
        1. Computes gradient norms for overflow detection
        2. Updates the gradient scaler based on overflow status
        3. Transfers gradients from FP16 to FP32 parameters
        4. Unscales and clips gradients as needed
        5. Performs the optimization step on FP32 parameters
        6. Copies updated FP32 parameters back to FP16 parameters

        :return: True if the optimization step was successful, False if overflow occurred
        :rtype: bool

        Example::

            success = optimizer.step()
            if not success:
                print("Gradient overflow detected, step skipped")
        """
        # compute gradient norms for overflow detection
        found_inf = False
        norm_groups = []
        for group_idx in range(len(self.param_groups)):
            norm_group = self._compute_norm_with_fsdp_flatten(group_idx)
            if norm_group == -1:
                found_inf = True
            norm_groups.append(norm_group)

        # update gradient scaler and handle overflow
        loss_scale = float(self.loss_scale.item())  # backup current scale
        self.grad_scaler.update(found_inf)
        if found_inf:
            print("Overflow occurs, please check it.", flush=True)
            self.zero_grad()
            return False

        # transfer gradients from fp16 to fp32 parameters
        for group_idx in range(len(self.param_groups)):
            if len(self._fp32_param_tensor_groups[group_idx]) <= 0:
                continue
            dtype = self._fp32_param_tensor_groups[group_idx][0].dtype
            fp16_params = [p for p in self._fp16_param_groups[group_idx] if p.untyped_storage().size() != 0]
            grad_fp32 = [p.grad.to(dtype) for p in fp16_params]

            device = self._fp32_param_tensor_groups[group_idx][0].device
            nonzero_fp32 = [p for p in self._fp32_param_tensor_groups[group_idx] if p.untyped_storage().size() != 0]
            for p, g in zip(nonzero_fp32, grad_fp32):
                p.grad = g.to(device)

        # compute global gradient norm and unscale/clip gradients
        scaled_global_grad_norm = torch.linalg.norm(torch.stack(norm_groups))  # pylint: disable=E1102
        self._unscale_and_clip_grads(scaled_global_grad_norm, loss_scale)

        # perform optimization step
        self.optim.step()
        self.zero_grad()

        # copy updated fp32 parameters back to fp16 parameters
        for group_idx in range(len(self._fp16_param_groups)):
            fp16_params = [p for p in self._fp16_param_groups[group_idx] if p.untyped_storage().size() != 0]
            fp32_tensor_params = [
                p for p in self._fp32_param_tensor_groups[group_idx] if p.untyped_storage().size() != 0
            ]
            # release fp32 gradients
            for fp32_param in fp32_tensor_params:
                fp32_param.grad = None
            # update fp16 parameters with fp32 values
            for p, q in zip(fp16_params, fp32_tensor_params):
                p.data.copy_(q)

        return True

    def clip_grad_norm(self, model, max_norm):
        """
        Set gradient clipping norm (actual clipping is performed in the step() method).

        :param model: The model whose gradients will be clipped (unused in current implementation)
        :param max_norm: Maximum norm value for gradient clipping
        :type max_norm: float

        Note:
            The actual gradient clipping is performed internally in the step() method
            using the _unscale_and_clip_grads method.
        """
        # actual clipping is conducted in the step() method
        pass

    #########################
    # utils from hybirdzero #
    #########################

    def _unscale_and_clip_grads(self, total_norm, loss_scale):
        """
        Unscale and clip gradients based on the total norm and loss scale.

        This method combines gradient unscaling (to reverse the loss scaling) with
        gradient clipping to prevent gradient explosion. The combined scale factor
        accounts for both operations.

        :param total_norm: The total gradient norm across all parameters
        :type total_norm: torch.Tensor
        :param loss_scale: The current loss scale used for gradient scaling
        :type loss_scale: float
        """
        # compute combined scale factor for this group
        combined_scale = loss_scale

        if self._clip_grad_norm > 0.0:
            # compute clipping factor (norm is scaled by loss_scale)
            clip = ((total_norm / loss_scale) + 1e-6) / self._clip_grad_norm
            clip = torch.clamp(clip, min=1.0)
            combined_scale = clip * loss_scale

        # apply combined unscaling and clipping to all fp32 parameters
        for _, param in self._fp32_param_tensor_groups.items():
            for p in param:
                if p.untyped_storage().size() != 0:
                    p.grad.data.mul_(1.0 / combined_scale)

    def state_dict(self):
        """
        Get the complete state dictionary for checkpointing.

        The state dictionary includes:
        - Gradient scaler state for loss scaling
        - Base optimizer states (momentum, etc.)
        - FP32 parameter weights for precise restoration

        :return: A dictionary containing all optimizer states and parameters
        :rtype: dict

        Example::

            # Save optimizer state
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, 'checkpoint.pt')
        """
        states = {}

        # save gradient scaler state
        grad_scaler = self.grad_scaler.state_dict()
        states["grad_scaler"] = grad_scaler

        # save base optimizer states
        optim_states = self.optim.state_dict()
        states["base_optim_states"] = optim_states

        # save fp32 parameter weights
        flat_fp32_weights = {}
        for group_idx, param in self._fp32_param_tensor_groups.items():
            flat_fp32_weights[group_idx] = param
        states["flat_fp32_weights"] = flat_fp32_weights

        return states

    def load_state_dict(self, states):
        """
        Load a complete state dictionary from checkpoint.

        This method restores the optimizer to its exact previous state, including
        gradient scaler, optimizer states, and both FP32 and FP16 parameter values.

        :param states: The state dictionary to load
        :type states: dict
        :raises AssertionError: If required state components are missing or parameter counts are inconsistent

        Example::

            # Load optimizer state
            checkpoint = torch.load('checkpoint.pt')
            optimizer.load_state_dict(checkpoint['optimizer'])
        """
        assert "grad_scaler" in states, "Not found grad_scaler state!"
        grad_scaler = states["grad_scaler"]
        self.grad_scaler.load_state_dict(grad_scaler)

        # load base optimizer states
        optim_states = states["base_optim_states"]
        self.optim.load_state_dict(optim_states)

        # load fp32 optimizer weights
        flat_fp32_weights = states["flat_fp32_weights"]
        assert set(flat_fp32_weights.keys()) == set(self._fp32_param_tensor_groups)
        for group_idx, param in flat_fp32_weights.items():
            self_param = self._fp32_param_tensor_groups[group_idx]
            assert len(self_param
                       ) == len(param), f"The number of flat tensor is inconsistent, {len(self_param)} != {len(param)}"
            for p, q in zip(self_param, param):
                p.data.copy_(q.data)

        # synchronize fp16 model weights with loaded fp32 weights
        for group_idx, param in flat_fp32_weights.items():
            fp16_param = self._fp16_param_groups[group_idx]
            fp32_param = self._fp32_param_tensor_groups[group_idx]
            for p, q in zip(fp16_param, fp32_param):
                p.data.copy_(q.data)


@torch.no_grad()
def offload_fsdp_optimizer(optimizer):
    """
    Offload optimizer states from GPU to CPU memory to reduce GPU memory usage.

    This function moves all tensor-based optimizer states (such as momentum buffers,
    variance estimates, etc.) from GPU to CPU memory. This is useful for reducing
    GPU memory pressure during training, especially when using large models or
    when GPU memory is limited.

    :param optimizer: The optimizer whose states should be offloaded to CPU
    :type optimizer: torch.optim.Optimizer

    Example::

        # Offload optimizer states to save GPU memory
        offload_fsdp_optimizer(optimizer)

        # Later, load back when needed
        load_fsdp_optimizer(optimizer)

    Note:
        After offloading, you should call load_fsdp_optimizer before the next
        optimization step to ensure states are available on the correct device.
    """
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)
    torch.cuda.empty_cache()


@torch.no_grad()
def load_fsdp_optimizer(optimizer, device_id=torch.cuda.current_device()):
    """
    Load optimizer states from CPU back to the specified GPU device.

    This function moves all tensor-based optimizer states from CPU memory back to
    the specified GPU device. This is typically used after offload_fsdp_optimizer
    to restore states for the next optimization step.

    :param optimizer: The optimizer whose states should be loaded to GPU
    :type optimizer: torch.optim.Optimizer
    :param device_id: The device ID to load states to (default: current CUDA device)
    :type device_id: int or torch.device

    Example::

        # Load optimizer states back to GPU before optimization
        load_fsdp_optimizer(optimizer, device_id=0)

        # Or use current device
        load_fsdp_optimizer(optimizer)

    Note:
        This function automatically determines the current device using get_current_device()
        to ensure compatibility with distributed training setups.
    """
    if not optimizer.state:
        return
    torch.cuda.empty_cache()
    # Use get_current_device() instead of torch.cuda.current_device() for distributed compatibility
    device_id = get_current_device()
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device_id, non_blocking=True)
