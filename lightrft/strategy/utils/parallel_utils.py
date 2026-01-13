"""
Sequence Parallelism Utilities for Distributed Training

This module provides utilities for sequence parallelism in distributed training environments.
It includes functions for managing sequence parallel groups, data processing for sequence
parallelism, and operations for tensor splitting, gathering, and all-to-all communication
in sequence parallel contexts.

The module supports:
- Setting and retrieving sequence parallel groups
- Processing data for sequence parallel distribution
- Slicing and padding inputs for sequence parallelism
- Specialized tensor operations for sequence parallel training
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist

_SEQUENCE_PARALLEL_GROUP = None


def set_sequence_parallel_group(group):
    """
    Set the global sequence parallel process group.

    :param group: The process group to use for sequence parallelism.
    :type group: torch.distributed.ProcessGroup
    """
    global _SEQUENCE_PARALLEL_GROUP  # pylint: disable=W0602
    _SEQUENCE_PARALLEL_GROUP = group


def get_sequence_parallel_group():
    """
    Get the current sequence parallel process group.

    :return: The current sequence parallel process group.
    :rtype: torch.distributed.ProcessGroup or None
    """
    global _SEQUENCE_PARALLEL_GROUP  # pylint: disable=W0602  # noqa: F824
    return _SEQUENCE_PARALLEL_GROUP


def get_sequence_parallel_world_size():
    """
    Get the world size of the sequence parallel group.

    :return: The world size of the sequence parallel group, or 1 if no group is set.
    :rtype: int
    """
    global _SEQUENCE_PARALLEL_GROUP  # pylint: disable=W0602  # noqa: F824
    if _SEQUENCE_PARALLEL_GROUP is not None:
        return torch.distributed.get_world_size(_SEQUENCE_PARALLEL_GROUP)
    return 1


def get_sequence_parallel_rank():
    """
    Get the rank of the current process in the sequence parallel group.

    :return: The rank in the sequence parallel group, or 0 if no group is set.
    :rtype: int
    """
    global _SEQUENCE_PARALLEL_GROUP  # pylint: disable=W0602  # noqa: F824
    if _SEQUENCE_PARALLEL_GROUP is not None:
        return torch.distributed.get_rank(_SEQUENCE_PARALLEL_GROUP)
    return 0


class SPDataProcessor:
    """
    A context manager for preprocessing data before conducting sequence parallel operations.

    This class handles the distribution and collection of data across sequence parallel ranks,
    ensuring proper data sharding and gathering for sequence parallel training.
    """
    def __init__(self) -> None:
        """
        Initialize the sequence parallel data processor.

        Sets up the sequence parallel size, group, and rank for data processing.
        """
        self.sp_size = get_sequence_parallel_world_size()
        self.sp_group = get_sequence_parallel_group()
        self.sp_rank = get_sequence_parallel_rank()

    def preprocess(self, data):
        """
        Preprocess data for sequence parallelism by gathering data from all ranks.

        :param data: The data to preprocess.
        :type data: Any

        :return: The preprocessed data, gathered from all ranks if sp_size > 1.
        :rtype: Any
        """
        if self.sp_size <= 1:
            return data
        # gather data for rank in the same dp group
        gathered_data = [None for _ in range(self.sp_size)]
        torch.distributed.all_gather_object(gathered_data, data, group=self.sp_group)
        all_data = [data_item for rank_data in gathered_data for data_item in rank_data]
        del gathered_data
        return all_data

    def postprocess(self, data):
        """
        Postprocess data after sequence parallel operations by distributing data to appropriate ranks.

        :param data: The data to postprocess.
        :type data: Any

        :return: The postprocessed data, distributed to the current rank.
        :rtype: Any
        :raises AssertionError: If the data length is not divisible by sp_size.
        """
        if self.sp_size == 1:
            return data
        assert len(data) % self.sp_size == 0
        local_num = len(data) // self.sp_size
        data = data[self.sp_rank * local_num:(self.sp_rank + 1) * local_num]
        return data


# adapted from https://github.com/volcengine/verl/blob/main/verl/utils/ulysses.py
def sp_slice_and_pad_input(input_ids: torch.Tensor, position_ids: torch.Tensor):
    """
    Pad and slice input_ids to be divisible by sp_size and pad position_ids to be divisible by sp_size.

    Note both input_ids and position_ids will be padded, but only input_ids will be sliced.
    This is the utility of pre-forward for ulysses sequence parallelism.

    :param input_ids: Input tensor with shape [bsz, seqlen].
    :type input_ids: torch.Tensor
    :param position_ids: Position IDs tensor with shape [bsz, seqlen], where bsz must be 1.
    :type position_ids: torch.Tensor

    :return: A tuple containing:
             - Padded and sliced input_ids
             - Padded position_ids
             - Size of padding added
    :rtype: Tuple[torch.Tensor, torch.Tensor, int]
    """
    sp_group = get_sequence_parallel_group()
    sp_size = torch.distributed.get_world_size(sp_group)
    sp_rank = torch.distributed.get_rank(sp_group)

    if position_ids is not None:
        assert position_ids.size(0) == 1
        assert input_ids.size(1) == position_ids.size(1)
    if sp_size <= 1:
        return input_ids, position_ids, 0
    _, total_seq_len = input_ids.shape
    pad_size = (sp_size - total_seq_len % sp_size) % sp_size
    if pad_size > 0:
        input_ids = torch.nn.functional.pad(input_ids, (0, pad_size), value=0)
        if position_ids is not None:
            pad_pos_ids = torch.arange(pad_size, device=position_ids.device).unsqueeze(0)
            position_ids = torch.cat((position_ids, pad_pos_ids), dim=-1)
    # we don't need to slice position ids
    parts = input_ids.size(-1) // sp_size
    slc = [slice(None)] * len(input_ids.shape)
    slc[-1] = slice(sp_rank * parts, (sp_rank + 1) * parts)
    input_ids = input_ids[slc].contiguous()
    return input_ids, position_ids, pad_size


def _split(input_, group, dim=-1):
    """
    Split a tensor along the specified dimension according to the world size.

    :param input_: The input tensor to split.
    :type input_: torch.Tensor
    :param group: The process group for determining world size and rank.
    :type group: torch.distributed.ProcessGroup
    :param dim: The dimension to split along, defaults to -1.
    :type dim: int

    :return: The portion of the input tensor corresponding to the current rank.
    :rtype: torch.Tensor
    :raises AssertionError: If the dimension size is not divisible by world size.
    """
    # skip if only one rank involved
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return input_
    # Split along last dimension.
    dim_size = input_.size(dim)
    assert dim_size % world_size == 0, (
        f"The dimension to split ({dim_size}) is not a multiple of world size ({world_size}), "
        f"cannot split tensor evenly"
    )
    tensor_list = torch.split(input_, dim_size // world_size, dim=dim)
    rank = torch.distributed.get_rank(group=group)
    output = tensor_list[rank].contiguous()
    output = output.detach().clone()
    return output


def _gather(input_, group, dim=-1):
    """
    Gather tensors from all ranks and concatenate them along the specified dimension.

    :param input_: The input tensor from the current rank.
    :type input_: torch.Tensor
    :param group: The process group for gathering.
    :type group: torch.distributed.ProcessGroup
    :param dim: The dimension along which to concatenate the gathered tensors, defaults to -1.
    :type dim: int

    :return: The concatenated tensor from all ranks.
    :rtype: torch.Tensor
    """
    # skip if only one rank involved
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return input_
    # all gather
    rank = torch.distributed.get_rank(group=group)
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    dist.all_gather(tensor_list, input_, group=group)
    # concat
    output = torch.cat(tensor_list, dim=dim).contiguous()
    return output


class _GatherForwardSplitBackward(torch.autograd.Function):
    """
    Custom autograd function that gathers input from model parallel region during forward pass
    and splits gradients during backward pass.

    This function is useful for operations that need to be performed on the full tensor
    in the forward pass but can have their gradients computed in parallel.
    """
    @staticmethod
    def symbolic(input_):
        """
        Symbolic function for torchscript.

        :param input_: The input tensor.
        :type input_: torch.Tensor

        :return: The gathered tensor.
        :rtype: torch.Tensor
        """
        return _gather(input_, group=None)

    @staticmethod
    def forward(ctx, input_, group, dim):
        """
        Forward pass: gather the input tensor from all ranks.

        :param ctx: Context object to save information for backward pass.
        :type ctx: torch.autograd.function.FunctionCtx
        :param input_: The input tensor from the current rank.
        :type input_: torch.Tensor
        :param group: The process group for gathering.
        :type group: torch.distributed.ProcessGroup
        :param dim: The dimension along which to gather.
        :type dim: int

        :return: The gathered tensor from all ranks.
        :rtype: torch.Tensor
        """
        ctx.group = group
        ctx.dim = dim
        return _gather(input_, group, dim)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: split the gradient tensor across ranks.

        :param ctx: Context object with saved information from forward pass.
        :type ctx: torch.autograd.function.FunctionCtx
        :param grad_output: The gradient tensor to split.
        :type grad_output: torch.Tensor

        :return: The split gradient tensor for the current rank, and None for other inputs.
        :rtype: Tuple[torch.Tensor, None, None]
        """
        return _split(grad_output, ctx.group, ctx.dim), None, None


def gather_forward_split_backward_and_unpad(input_, group, dim, padding_size=0, unpad_dim=None):
    """
    Gather tensors in the forward pass, split gradients in the backward pass, and remove padding if needed.

    :param input_: The input tensor from the current rank.
    :type input_: torch.Tensor
    :param group: The process group for gathering.
    :type group: torch.distributed.ProcessGroup
    :param dim: The dimension along which to gather/split.
    :type dim: int
    :param padding_size: Size of padding to remove, defaults to 0.
    :type padding_size: int
    :param unpad_dim: Dimension from which to remove padding, defaults to None.
    :type unpad_dim: int or None

    :return: The gathered (and possibly unpadded) tensor.
    :rtype: torch.Tensor
    """
    gather_input = _GatherForwardSplitBackward.apply(input_, group, dim)
    if padding_size == 0:
        return gather_input
    else:
        slc = [slice(None)] * len(gather_input.shape)
        slc[unpad_dim] = slice(0, -padding_size)
        return gather_input[slc]


# adpated from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/sequence/layer.py
class _SeqAllToAll(torch.autograd.Function):
    """
    Custom autograd function for sequence all-to-all communication.

    This function performs all-to-all communication in the forward pass and handles
    the corresponding backward pass correctly.
    """
    @staticmethod
    def forward(
        ctx,
        group: dist.ProcessGroup,
        scatter_idx: Optional[Union[List[int], int]],
        gather_idx: Optional[Union[List[int], int]],
        *input_: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: perform all-to-all communication.

        :param ctx: Context object to save information for backward pass.
        :type ctx: torch.autograd.function.FunctionCtx
        :param group: The process group for all-to-all communication.
        :type group: dist.ProcessGroup
        :param scatter_idx: Dimension(s) to scatter along.
        :type scatter_idx: Optional[Union[List[int], int]]
        :param gather_idx: Dimension(s) to gather along.
        :type gather_idx: Optional[Union[List[int], int]]
        :param input_: Input tensor(s) to communicate.
        :type input_: torch.Tensor

        :return: The result of all-to-all communication.
        :rtype: torch.Tensor or Tuple[torch.Tensor]
        """
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        seq_world_size = dist.get_world_size(group)
        if dist.get_world_size(group) <= 1:
            if len(input_) == 1:
                return input_[0]
            return input_
        if len(input_) == 1:
            input_list = [t.contiguous() for t in torch.tensor_split(input_[0], seq_world_size, scatter_idx)]
            output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
            # TODO: use all_to_all_single instead
            dist.all_to_all(output_list, input_list, group=group)
            return torch.cat(output_list, dim=gather_idx).contiguous()
        outputs = []
        assert len(scatter_idx) == len(gather_idx)
        assert len(gather_idx) == len(input_)
        for i in range(len(input_)):
            if i == 0:
                input_list = [t.contiguous() for t in torch.tensor_split(input_[i], seq_world_size, scatter_idx[i])]
                output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
                handle_last = dist.all_to_all(output_list, input_list, group=group, async_op=True)
            # conduct the next all2all
            if i + 1 < len(input_):
                input_list_next = [
                    t.contiguous() for t in torch.tensor_split(input_[i + 1], seq_world_size, scatter_idx[i + 1])
                ]
                output_list_next = [torch.empty_like(input_list_next[0]) for _ in range(seq_world_size)]
                handle_next = dist.all_to_all(output_list_next, input_list_next, group=group, async_op=True)
            handle_last.wait()  # pylint: disable=E0606
            outputs.append(torch.cat(output_list, dim=gather_idx[i]).contiguous())
            if i + 1 < len(input_):
                handle_last = handle_next
                input_list = input_list_next
                output_list = output_list_next
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_output: torch.Tensor) -> Tuple[None, torch.Tensor, None, None]:
        """
        Backward pass for the custom autograd function.

        This method handles the gradient computation during backpropagation. If the distributed
        world size is 1 or less, it simply passes through the gradient outputs. Otherwise,
        it applies the sequence all-to-all operation on the gradients.

        :param ctx: The saved context from the forward pass
        :type ctx: torch.autograd.function.FunctionCtx
        :param grad_output: The gradient tensors from the next layer
        :type grad_output: torch.Tensor

        :return: Tuple containing None for non-tensor inputs and the gradient tensors
        :rtype: Tuple[None, torch.Tensor, None, None]
        """
        if dist.get_world_size(ctx.group) <= 1:
            return None, None, None, *grad_output
        res = _SeqAllToAll.apply(ctx.group, ctx.gather_idx, ctx.scatter_idx, *grad_output)
        if len(grad_output) == 1:
            return None, None, None, res
        return None, None, None, *res
