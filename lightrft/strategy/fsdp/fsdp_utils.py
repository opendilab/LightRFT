"""
Gradient scaling and optimization utilities for deep learning.

This module provides tools for gradient handling, norm computation, and optimization in PyTorch.
It includes dynamic gradient scaling for mixed precision training, gradient norm computation,
and base classes for optimizers.
"""

import math
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim import Optimizer

from lightrft.utils import get_current_device

try:
    import amp_C
    from apex.multi_tensor_apply import multi_tensor_applier

    APEX_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    print("The torch implementation for cal_l2norm is slower than apex. Please note this!")
    APEX_AVAILABLE = False

inf = math.inf


def is_meta_initialized(model) -> bool:
    """
    Check if a PyTorch model's parameters are meta-initialized.

    Meta-initialized models contain parameters on a 'meta' device, which are placeholders
    that don't allocate actual memory. These are useful for model initialization without
    memory overhead, commonly used in model parallelism and large model initialization.

    For more information on meta device and meta tensors, see:
    https://docs.pytorch.org/docs/stable/meta.html

    :param model: The PyTorch module to check.
    :type model: torch.nn.Module
    :raises TypeError: if ``model`` is not an instance of :class:`torch.nn.Module`.
    :returns: True if any parameter in the model is on a meta device, False otherwise.
    :rtype: bool

    Example::

        >>> import torch
        >>> model = torch.nn.Linear(10, 1)
        >>> is_meta_initialized(model)  # False for regular model
        False
        >>> with torch.device('meta'):
        ...     meta_model = torch.nn.Linear(10, 1)
        >>> is_meta_initialized(meta_model)  # True for meta model
        True
    """
    for param in model.parameters():
        if hasattr(param, "device") and param.device.type == "meta":
            return True
    return False


def multi_tensor_l2norm_torch(tensor_list, per_tensor):
    """
    Compute L2 norm of multiple tensors using PyTorch.

    This function provides a pure PyTorch implementation for computing L2 norms
    when APEX is not available. It converts all tensors to float32 for computation
    and returns both overall and per-tensor norms.

    :param tensor_list: List of tensors to compute norm for
    :type tensor_list: list[torch.Tensor]
    :param per_tensor: Whether to return per-tensor norms
    :type per_tensor: bool

    :return: Tuple of (overall L2 norm, per-tensor norms)
    :rtype: tuple[torch.Tensor, torch.Tensor]

    Example::

        >>> tensors = [torch.randn(3, 3), torch.randn(2, 2)]
        >>> overall_norm, per_tensor_norms = multi_tensor_l2norm_torch(tensors, True)
        >>> print(overall_norm.shape)  # torch.Size([1])
        >>> print(per_tensor_norms.shape)  # torch.Size([2])
    """
    # Convert tensor_list elements to torch.float32
    tensor_list = [tensor.float() for tensor in tensor_list]
    norms_tensor = torch.stack([torch.norm(tensor, p=2) for tensor in tensor_list])
    l2_norm = torch.norm(norms_tensor, p=2).unsqueeze(0)

    if per_tensor:
        per_tensor_norm = norms_tensor
    else:
        per_tensor_norm = torch.Tensor([]).to(norms_tensor.device)

    return l2_norm, per_tensor_norm


def calc_l2_norm(grads):
    """
    Calculate L2 norm of gradients using optimized implementation when available.

    This function automatically selects the fastest available implementation for
    computing L2 norms. It uses APEX's multi-tensor operations when available
    for better performance, otherwise falls back to PyTorch implementation.

    :param grads: List of gradient tensors
    :type grads: list[torch.Tensor]

    :return: L2 norm of gradients
    :rtype: torch.Tensor

    Example::

        >>> grads = [torch.randn(10, 10).requires_grad_(), torch.randn(5, 5).requires_grad_()]
        >>> norm = calc_l2_norm(grads)
        >>> print(norm.item())  # scalar value
    """
    norm = 0.0
    if len(grads) > 0:
        if APEX_AVAILABLE:
            dummy_overflow_buf = torch.tensor([0], device=get_current_device(), dtype=torch.int32)
            norm, _ = multi_tensor_applier(
                amp_C.multi_tensor_l2norm,
                dummy_overflow_buf,
                [grads],
                False,  # no per-parameter norm
            )
        else:
            norm, _ = multi_tensor_l2norm_torch(grads, False)
    return norm


def calc_lp(grads, norm_type):
    """
    Calculate Lp norm of gradients.

    Computes the p-norm of a list of gradient tensors, where p is specified
    by norm_type. This is useful for gradient clipping and monitoring.

    :param grads: List of gradient tensors
    :type grads: list[torch.Tensor]
    :param norm_type: The p in Lp norm
    :type norm_type: float

    :return: Lp norm of gradients
    :rtype: float

    Example::

        >>> grads = [torch.randn(3, 3), torch.randn(2, 2)]
        >>> l1_norm = calc_lp(grads, 1.0)  # L1 norm
        >>> l2_norm = calc_lp(grads, 2.0)  # L2 norm
    """
    norm = 0.0
    for grad in grads:
        grad_norm = torch.norm(grad, norm_type)
        norm += grad_norm ** norm_type
    return norm


def get_norm(grads, norm_type, enable_cuda_kernels):
    """
    Get norm of gradients with specified norm type.

    This function dispatches to the appropriate norm calculation method based
    on the norm type and whether CUDA kernels are available. It handles special
    cases like infinity norm and optimized L2 norm computation.

    :param grads: List of gradient tensors
    :type grads: list[torch.Tensor]
    :param norm_type: Type of norm to compute (2.0, inf, etc.)
    :type norm_type: float
    :param enable_cuda_kernels: Whether to use CUDA optimized kernels
    :type enable_cuda_kernels: bool

    :return: Norm of gradients
    :rtype: float or torch.Tensor

    Example::

        >>> grads = [torch.randn(3, 3), torch.randn(2, 2)]
        >>> l2_norm = get_norm(grads, 2.0, True)
        >>> inf_norm = get_norm(grads, float('inf'), True)
    """
    if norm_type == inf:
        grad_norm = max(g.data.abs().max() for g in grads)
    elif norm_type == 2.0 and enable_cuda_kernels:
        grad_norm = calc_l2_norm(grads) ** norm_type
    else:
        grad_norm = calc_lp(grads, norm_type)
    return grad_norm


def reduce_grads(gradients, parameters):
    """
    Prepare gradients for norm computation in distributed training.

    This function processes gradients to prepare them for norm computation,
    particularly in distributed training scenarios. It converts gradients to
    float32 for numerical stability during norm calculations.

    :param gradients: List of gradient tensors
    :type gradients: list[torch.Tensor]
    :param parameters: List of parameter tensors
    :type parameters: list[torch.Tensor]

    :return: List of processed gradient tensors
    :rtype: list[torch.Tensor]

    Example::

        >>> grads = [torch.randn(3, 3), torch.randn(2, 2)]
        >>> params = [torch.randn(3, 3), torch.randn(2, 2)]
        >>> processed_grads = reduce_grads(grads, params)
    """
    parallel_grads = []

    for g, _ in zip(gradients, parameters):
        # process all ranks for FSDP parameter group
        parallel_grads.append(g.data.float())

    return parallel_grads


def get_tensor_norm(norm: Union[float, torch.Tensor], move_to_cuda) -> torch.Tensor:
    """
    Convert norm to tensor and move to appropriate device.

    This utility function ensures that norm values are properly converted to
    tensors and placed on the correct device for further computation. It handles
    both scalar and tensor inputs.

    :param norm: Norm value as float or tensor
    :type norm: Union[float, torch.Tensor]
    :param move_to_cuda: Whether to move the tensor to CUDA
    :type move_to_cuda: bool

    :return: Norm as tensor on appropriate device
    :rtype: torch.Tensor

    Example::

        >>> norm_float = 2.5
        >>> norm_tensor = get_tensor_norm(norm_float, True)
        >>> print(norm_tensor.device)  # cuda:0 (if CUDA available)
    """
    if isinstance(norm, float):
        norm = torch.Tensor([norm])
    if move_to_cuda:
        norm = norm.to(get_current_device())
    return norm


def compute_norm(gradients, parameters, norm_type=2):
    """
    Get the norm across distributed environment.

    This function computes gradient norms in a distributed training setting,
    handling device placement and distributed reduction. It's commonly used
    for gradient clipping and monitoring training stability.

    :param gradients: The gradient values
    :type gradients: list[torch.Tensor]
    :param parameters: The parameters each gradient corresponds to
    :type parameters: list[torch.Tensor]
    :param norm_type: Type of the used p-norm. Can be ``'inf'`` for infinity norm
    :type norm_type: float or int

    :return: Total norm of the parameters, need total_norm**(1/norm) before using
    :rtype: float

    Example::

        >>> grads = [param.grad for param in model.parameters() if param.grad is not None]
        >>> params = [param for param in model.parameters() if param.grad is not None]
        >>> total_norm = compute_norm(grads, params)
        >>> print(f"Gradient norm: {total_norm}")
    """

    enable_cuda_kernels = gradients[0].device.type != "cpu"
    # Norm parameters.
    norm_type = float(norm_type)

    tensor_parallel_grads = reduce_grads(gradients, parameters)
    tensor_parallel_norm = get_norm(tensor_parallel_grads, norm_type, enable_cuda_kernels)

    # If norm is type of float, then we convert them into torch.Tensor.
    tensor_parallel_norm = get_tensor_norm(tensor_parallel_norm, enable_cuda_kernels)
    # If grads are on CPU, the norms is also on CPU. Cast them to CUDA tensors
    if not enable_cuda_kernels:
        tensor_parallel_norm = tensor_parallel_norm.to(get_current_device())

    total_norm = tensor_parallel_norm
    """
    Sum across all model-parallel GPUs.
    """
    dist.all_reduce(total_norm, op=dist.ReduceOp.SUM)

    if torch.is_tensor(total_norm):
        total_norm = total_norm.item()

    # Scale.
    if total_norm == float("inf") or total_norm == -float("inf"):
        total_norm = -1

    if math.isnan(total_norm):
        total_norm = -2

    return total_norm


class BaseGradScaler(ABC):
    """
    A base class for the gradient scaler.

    This abstract base class defines the interface for gradient scalers used in
    mixed precision training. Gradient scalers help prevent gradient underflow
    in float16 training by scaling up the loss before backpropagation.

    :param initial_scale: The initial loss scale
    :type initial_scale: float

    Example::

        >>> # Subclass implementation
        >>> class MyGradScaler(BaseGradScaler):
        ...     def update(self, overflow: bool) -> None:
        ...         # Custom update logic
        ...         pass
    """
    def __init__(self, initial_scale: float):
        assert initial_scale > 0
        self._scale = torch.tensor([initial_scale], device=get_current_device(), dtype=torch.float32)

    @property
    def scale(self) -> Tensor:
        """
        Returns the loss scale.

        :return: Current loss scale
        :rtype: torch.Tensor

        Example::

            >>> scaler = DynamicGradScaler(initial_scale=1024.0)
            >>> print(scaler.scale.item())  # 1024.0
        """

        return self._scale

    @property
    def inv_scale(self) -> Tensor:
        """
        Returns the inverse of the loss scale.

        The inverse scale is used to unscale gradients after backpropagation
        to restore their original magnitudes.

        :return: Inverse of current loss scale
        :rtype: torch.Tensor

        Example::

            >>> scaler = DynamicGradScaler(initial_scale=1024.0)
            >>> print(scaler.inv_scale.item())  # 0.0009765625 (1/1024)
        """

        return self._scale.double().reciprocal().float()

    def state_dict(self) -> Dict:
        """
        Returns the states of the gradient scaler as a dict object.

        :return: State dictionary containing scale
        :rtype: Dict

        Example::

            >>> scaler = DynamicGradScaler()
            >>> state = scaler.state_dict()
            >>> print(state.keys())
            dict_keys(['scale'])
        """

        state_dict = dict()
        state_dict["scale"] = self.scale
        return state_dict

    def load_state_dict(self, state_dict: Dict) -> None:
        """
        Load the states of the gradient scaler from a dict object.

        :param state_dict: The states of the gradient scaler
        :type state_dict: Dict

        Example::

            >>> scaler = DynamicGradScaler()
            >>> state = {"scale": torch.tensor([2048.0])}
            >>> scaler.load_state_dict(state)
        """

        self._scale = state_dict["scale"]

    @abstractmethod
    def update(self, overflow: bool) -> None:
        """
        Update the loss scale.

        This abstract method must be implemented by subclasses to define
        how the loss scale should be updated based on overflow detection.

        :param overflow: Whether overflow occurs
        :type overflow: bool
        """

        pass


class DynamicGradScaler(BaseGradScaler):
    """
    A gradient scaler which uses dynamic loss scale.

    This scaler automatically adjusts the loss scale based on gradient overflow
    detection. It increases the scale when training is stable and decreases it
    when overflow occurs, providing automatic mixed precision training support.

    :param initial_scale: The initial loss scale
    :type initial_scale: float
    :param growth_factor: The multiplication factor for increasing loss scale
    :type growth_factor: float
    :param backoff_factor: The multiplication factor for decreasing loss scale
    :type backoff_factor: float
    :param growth_interval: The number of steps to increase loss scale when no overflow occurs
    :type growth_interval: int
    :param min_scale: The minimum loss scale
    :type min_scale: Optional[float]
    :param max_scale: The maximum loss scale
    :type max_scale: Optional[float]
    :param hysteresis: The number of overflows before decreasing loss scale
    :type hysteresis: int
    :param dtype: The data type used for training
    :type dtype: torch.dtype

    Example::

        >>> scaler = DynamicGradScaler(initial_scale=2**16, growth_factor=2.0)
        >>> # In training loop
        >>> for epoch in range(num_epochs):
        ...     for batch in dataloader:
        ...         # Forward pass with scaled loss
        ...         scaled_loss = loss * scaler.scale
        ...         scaled_loss.backward()
        ...         # Check for overflow and update scaler
        ...         overflow = check_overflow(model.parameters())
        ...         scaler.update(overflow)
    """

    def __init__(  # pylint: disable=R0917
        self,
        initial_scale: float = 2**16,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        min_scale: Optional[float] = 1,
        max_scale: Optional[float] = 2**24,
        hysteresis: int = 2,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(initial_scale)
        if min_scale:
            self._min_scale = torch.tensor([min_scale], device=get_current_device(), dtype=torch.float32)
        else:
            self._min_scale = None

        if max_scale:
            self._max_scale = torch.tensor([max_scale], device=get_current_device(), dtype=torch.float32)
        else:
            self._max_scale = None

        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._growth_step = 0
        self._hysteresis = hysteresis
        self._hysteresis_step = 0
        self._dtype = dtype
        self._sanity_checks()

    def _sanity_checks(self) -> None:
        """
        Check if the arguments are correct.

        This method validates all the initialization parameters to ensure they
        are within reasonable ranges and compatible with the specified data type.
        It provides warnings for potentially suboptimal configurations.
        """

        assert self._dtype in [torch.float16, torch.bfloat16, torch.float32]

        if self._min_scale is not None:
            min_scale = self._min_scale.item()
            assert min_scale > 0, "The minimum gradient scale cannot be zero or negative"

            if self._dtype != torch.float16 and min_scale != 1.0:
                print(f"Detect you use {self._dtype}, but min_scale: {min_scale} != 1.0")

        if self._max_scale:
            max_scale = self._max_scale.item()
            assert max_scale > 0, "The maximum gradient scale cannot be zero or negative"

            if self._dtype != torch.float16 and max_scale != 1.0:
                print(f"Detect you use {self._dtype}, but max_scale: {max_scale} != 1.0")

        if self._dtype == torch.float16:
            assert self._growth_factor > 1.0, "The growth factor cannot be equal or smaller than 1"
            assert self._backoff_factor < 1.0 and self._backoff_factor > 0, "The backoff factor must be between 0 and 1"
        else:
            assert self._growth_factor >= 1.0, "The growth factor cannot be smaller than 1"
            assert (
                self._backoff_factor <= 1.0 and self._backoff_factor > 0
            ), "The backoff factor must be between 0 and 1"

            if self._growth_factor != 1.0:
                print(f"Detect you use {self._dtype}, but growth_factor: {self._growth_factor} != 1.0")
            if self._backoff_factor != 1.0:
                print(f"Detect you use {self._dtype}, but backoff_factor: {self._backoff_factor} != 1.0")

        assert self._hysteresis >= 0, "The hysteresis cannot be negative"

    def update(self, overflow: bool) -> None:
        """
        Update the loss scale based on whether overflow occurred.

        This method implements the dynamic scaling algorithm. When overflow occurs,
        it increments the hysteresis counter and resets growth progress. When no
        overflow occurs for a sufficient period, it increases the scale to maximize
        gradient precision.

        :param overflow: Whether overflow occurs
        :type overflow: bool

        Example::

            >>> scaler = DynamicGradScaler()
            >>> # Simulate training steps
            >>> scaler.update(False)  # No overflow, increment growth counter
            >>> scaler.update(True)   # Overflow detected, may decrease scale
        """
        if overflow:
            self._hysteresis_step += 1
            self._growth_step = 0

            if self._hysteresis_step >= self._hysteresis:
                self._backoff_scale()
                print(f"Overflow occurs, the loss scale is adjusted to {self.scale.item()}")
        else:
            self._growth_step += 1
            if self._growth_step == self._growth_interval:
                self._growth_step = 0
                self._hysteresis_step = 0
                self._grow_scale()
                print(
                    f"No overflow for consecutive {self._growth_interval} steps, "
                    f"the loss scale is adjusted to {self.scale.item()}",
                )

    def _backoff_scale(self) -> None:
        """
        Decrease the loss scale when overflow occurs.

        This private method reduces the loss scale by the backoff factor and
        ensures it doesn't go below the minimum scale if specified.
        """

        self._scale = self._scale * self._backoff_factor
        if self._min_scale:
            self._scale = torch.max(self._scale, self._min_scale)

    def _grow_scale(self) -> None:
        """
        Increase the loss scale when no overflow occurs for a period.

        This private method increases the loss scale by the growth factor and
        ensures it doesn't exceed the maximum scale if specified.
        """

        self._scale = self._scale * self._growth_factor
        if self._max_scale:
            self._scale = torch.min(self._scale, self._max_scale)

    def state_dict(self):
        """
        Returns the states of the gradient scaler as a dict object.

        This method provides a complete state dictionary that can be saved
        and restored to maintain training consistency across checkpoints.

        :return: A dictionary containing the current state of the gradient scaler
        :rtype: dict

        Example::

            >>> scaler = DynamicGradScaler()
            >>> scaler_state = scaler.state_dict()
            >>> print(scaler_state.keys())
            dict_keys(['_scale', '_growth_step', '_hysteresis_step'])
        """
        state_dict = dict()
        state_dict["_scale"] = self._scale.item()
        state_dict["_growth_step"] = self._growth_step
        state_dict["_hysteresis_step"] = self._hysteresis_step

        return state_dict

    def load_state_dict(self, state_dict):
        """
        Load the states of the gradient scaler from a dict object.

        This method restores the scaler state from a previously saved state
        dictionary, enabling seamless checkpoint restoration.

        :param state_dict: The states of the gradient scaler
        :type state_dict: dict

        Example::

            >>> scaler = DynamicGradScaler()
            >>> scaler.load_state_dict({
            ...     "_scale": 2048.0,
            ...     "_growth_step": 0,
            ...     "_hysteresis_step": 0
            ... })
        """
        self._scale = self._scale.fill_(state_dict["_scale"])
        self._growth_step = state_dict["_growth_step"]
        self._hysteresis_step = state_dict["_hysteresis_step"]


class BaseOptimizer(Optimizer):
    """
    Base Optimizer class that wraps a PyTorch optimizer.

    This class provides a wrapper around PyTorch optimizers, exposing the same interface
    while allowing for additional functionality like custom backward passes and gradient clipping.
    It serves as a foundation for building more sophisticated optimizers with enhanced features
    for distributed training, gradient scaling, and custom optimization strategies.

    :param optim: The PyTorch optimizer to wrap
    :type optim: torch.optim.Optimizer

    Example::

        >>> import torch.optim as optim
        >>> model = torch.nn.Linear(10, 1)
        >>> pytorch_optimizer = optim.Adam(model.parameters(), lr=0.001)
        >>> wrapped_optimizer = BaseOptimizer(pytorch_optimizer)
        >>> # Use wrapped_optimizer like a regular optimizer
        >>> wrapped_optimizer.zero_grad()
        >>> loss.backward()
        >>> wrapped_optimizer.step()
    """
    def __init__(self, optim: Optimizer):  # pylint: disable=W0231
        self.optim = optim

    @property
    def param_groups(self):
        """
        Access to the parameter groups of the wrapped optimizer.

        Parameter groups allow different sets of parameters to have different
        optimization settings like learning rates, weight decay, etc.

        :return: List of parameter groups
        :rtype: list

        Example::

            >>> optimizer = BaseOptimizer(torch.optim.Adam(model.parameters()))
            >>> print(len(optimizer.param_groups))  # Number of parameter groups
            >>> print(optimizer.param_groups[0]['lr'])  # Learning rate of first group
        """
        return self.optim.param_groups

    @property
    def defaults(self):
        """
        Access to the default parameters of the wrapped optimizer.

        :return: Default parameters dictionary
        :rtype: dict

        Example::

            >>> optimizer = BaseOptimizer(torch.optim.Adam(model.parameters(), lr=0.001))
            >>> print(optimizer.defaults['lr'])  # 0.001
        """
        return self.optim.defaults

    def add_param_group(self, *args, **kwargs):
        """
        Add a parameter group to the optimizer.

        This method allows adding new parameter groups with potentially different
        optimization settings during training.

        :param args: Positional arguments to pass to the wrapped optimizer
        :param kwargs: Keyword arguments to pass to the wrapped optimizer
        :return: Result from the wrapped optimizer's add_param_group method

        Example::

            >>> optimizer = BaseOptimizer(torch.optim.Adam(model.parameters()))
            >>> new_params = [torch.randn(10, requires_grad=True)]
            >>> optimizer.add_param_group({'params': new_params, 'lr': 0.01})
        """
        return self.optim.add_param_group(*args, **kwargs)

    def step(self, *args, **kwargs):
        """
        Perform a single optimization step.

        This method executes one optimization step, updating the model parameters
        based on their gradients and the optimizer's algorithm.

        :param args: Positional arguments to pass to the wrapped optimizer
        :param kwargs: Keyword arguments to pass to the wrapped optimizer
        :return: Result from the wrapped optimizer's step method

        Example::

            >>> optimizer = BaseOptimizer(torch.optim.Adam(model.parameters()))
            >>> loss.backward()
            >>> optimizer.step()
        """
        return self.optim.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        """
        Reset the gradients of all optimized tensors.

        This method clears the gradients of all parameters, which is typically
        done before each backward pass to prevent gradient accumulation.

        :param args: Positional arguments to pass to the wrapped optimizer
        :param kwargs: Keyword arguments to pass to the wrapped optimizer

        Example::

            >>> optimizer = BaseOptimizer(torch.optim.Adam(model.parameters()))
            >>> optimizer.zero_grad()  # Clear gradients
            >>> loss.backward()        # Compute new gradients
            >>> optimizer.step()       # Update parameters
        """
        self.optim.zero_grad(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        """
        Load the optimizer state.

        This method restores the optimizer's internal state from a state dictionary,
        enabling checkpoint restoration and training resumption.

        :param args: Positional arguments to pass to the wrapped optimizer
        :param kwargs: Keyword arguments to pass to the wrapped optimizer

        Example::

            >>> optimizer = BaseOptimizer(torch.optim.Adam(model.parameters()))
            >>> state_dict = torch.load('optimizer_checkpoint.pth')
            >>> optimizer.load_state_dict(state_dict)
        """
        self.optim.load_state_dict(*args, **kwargs)

    def state_dict(self):
        """
        Return the state of the optimizer as a dict.

        This method provides the optimizer's complete state for checkpointing,
        including parameter states and hyperparameters.

        :return: The state of the optimizer
        :rtype: dict

        Example::

            >>> optimizer = BaseOptimizer(torch.optim.Adam(model.parameters()))
            >>> state_dict = optimizer.state_dict()
            >>> torch.save(state_dict, 'optimizer_checkpoint.pth')
        """
        return self.optim.state_dict()

    def backward(self, loss):
        """
        Compute gradients of the loss.

        This method performs backpropagation to compute gradients of the loss
        with respect to the model parameters.

        :param loss: The loss tensor to compute gradients for
        :type loss: torch.Tensor

        Example::

            >>> optimizer = BaseOptimizer(torch.optim.Adam(model.parameters()))
            >>> loss = criterion(output, target)
            >>> optimizer.backward(loss)
        """
        loss.backward()

    def backward_by_grad(self, tensor, grad):
        """
        Compute gradients of the tensor with respect to the provided gradients.

        This method allows for custom gradient computation, useful in scenarios
        like gradient accumulation or custom loss functions.

        :param tensor: The tensor to compute gradients for
        :type tensor: torch.Tensor
        :param grad: The gradients to backpropagate
        :type grad: torch.Tensor

        Example::

            >>> optimizer = BaseOptimizer(torch.optim.Adam(model.parameters()))
            >>> output = model(input)
            >>> custom_grad = torch.randn_like(output)
            >>> optimizer.backward_by_grad(output, custom_grad)
        """
        torch.autograd.backward(tensors=tensor, grad_tensors=grad)

    def clip_grad_norm(self):
        """
        Clip the gradient norm.

        This is a placeholder method that should be implemented by subclasses
        to provide gradient clipping functionality. Gradient clipping helps
        prevent gradient explosion in deep networks.

        Example::

            >>> class MyOptimizer(BaseOptimizer):
            ...     def clip_grad_norm(self):
            ...         torch.nn.utils.clip_grad_norm_(self.param_groups[0]['params'], max_norm=1.0)
        """
        pass
