from datetime import timedelta
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.distributed
import torch.distributed as dist
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)


# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
def init_process_group(
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = None,
    pg_options: Optional[Any] = None,
):
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    # NOTE: The pg_options parameter was renamed into backend_options in PyTorch 2.6.0
    # https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # We need to determine the appropriate parameter name based on PyTorch version
    pg_options_param_name = "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


def create_sub_group(group_size: int, backend: str = "nccl") -> Tuple[dist.ProcessGroup, dist.ProcessGroup]:
    """
    Create process subgroups for distributed computing with validation and communication testing.

    This function creates two types of process groups for distributed computing:
    1. Regular groups: processes are grouped consecutively (e.g., [0,1,2,3], [4,5,6,7])
    2. Orthogonal groups: processes are grouped with stride equal to group_size (e.g., [0,4], [1,5], [2,6], [3,7])

    The function also performs communication testing to ensure the groups are working correctly.

    :param group_size: Size of each process subgroup. Must be a divisor of world_size.
    :type group_size: int
    :param backend: Backend for distributed communication ("nccl" for GPU, other options for CPU).
    :type backend: str

    :return: Tuple of (regular process group, orthogonal process group).
    :rtype: Tuple[dist.ProcessGroup, dist.ProcessGroup]
    :raises ValueError: If world_size is not divisible by group_size.

    Example::

        >>> # Create subgroups with size 4 using NCCL backend
        >>> regular_group, orthogonal_group = create_sub_group(4, "nccl")
        >>> # With world_size=8, this creates:
        >>> # Regular groups: [0,1,2,3] and [4,5,6,7]
        >>> # Orthogonal groups: [0,4], [1,5], [2,6], [3,7]
    """
    world_size = dist.get_world_size()
    if world_size % group_size != 0:
        raise ValueError(f"world_size ({world_size}) % group_size ({group_size}) != 0 ")

    num_groups = world_size // group_size

    all_group_ranks = []

    for i in range(num_groups):
        start_rank = i * group_size
        group_ranks = list(range(start_rank, start_rank + group_size))
        all_group_ranks.append(group_ranks)
    group, _ = dist.new_subgroups_by_enumeration(all_group_ranks, backend=backend)

    orthogonal_group_ranks = []
    for i in range(group_size):
        orthogonal_ranks = list(range(i, world_size, group_size))
        orthogonal_group_ranks.append(orthogonal_ranks)
    orthogonal_group, _ = dist.new_subgroups_by_enumeration(orthogonal_group_ranks)

    if dist.get_rank() == 0:
        print(
            f"Finished create TP/PP group, with groupsize={torch.distributed.get_world_size(group)},"
            " start testing comm...",
            flush=True,
        )
    dist.barrier()
    device = "cuda" if backend == "nccl" else "cpu"
    tmp = torch.tensor(1.1, device=device)
    dist.all_reduce(tmp, group=group, op=dist.ReduceOp.SUM)
    dist.barrier()
    assert abs(tmp.item() / dist.get_world_size(group=group) - 1.1) < 1e-4
    if dist.get_rank() == 0:
        print("Finished testing comm!", flush=True)

    return group, orthogonal_group


def all_gather_all_prompt_token_ids(all_prompt_token_ids: List[List[int]], group: dist.ProcessGroup) -> List[List[int]]:
    """
    Collect prompt token_ids across processes with different lengths, handle padding and alignment.

    This function gathers prompt token lists from all processes in the distributed group.
    It handles sequences of different lengths by padding them to the maximum length,
    performing the all-gather operation, and then removing the padding from the results.

    :param all_prompt_token_ids: List of prompt token lists from the current process.
                                Each inner list represents tokens for one prompt.
    :type all_prompt_token_ids: List[List[int]]
    :param group: Distributed communication group for gathering operations.
    :type group: dist.ProcessGroup

    :return: Collected and processed prompt token lists from all processes.
            The padding tokens (-1) are removed from the final result.
    :rtype: List[List[int]]
    :raises AssertionError: If distributed environment is not initialized.

    Example::

        >>> # Gather tokens across processes
        >>> tokens = [[1, 2, 3], [4, 5]]  # Current process tokens
        >>> gathered_tokens = all_gather_all_prompt_token_ids(tokens, process_group)
        >>> # Result contains tokens from all processes in the group
    """
    # Ensure distributed environment is initialized
    assert dist.is_initialized(), "Distributed environment not initialized"

    if torch.distributed.get_world_size(group) == 1:
        return all_prompt_token_ids
    # Get device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Calculate max length in current process
    max_len_local = max(len(tokens) for tokens in all_prompt_token_ids)
    num_prompts = len(all_prompt_token_ids)

    # 2. Synchronize global max length
    max_len_global = torch.tensor(max_len_local, device=device, dtype=torch.long)
    dist.all_reduce(max_len_global, op=dist.ReduceOp.MAX, group=group)
    max_len_global = max_len_global.item()

    # 3. Create padded tensor
    padded_tensor = torch.full((num_prompts, max_len_global), -1, dtype=torch.long, device=device)

    # 4. Fill data into tensor
    for i, tokens in enumerate(all_prompt_token_ids):
        if len(tokens) > 0:
            tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
            padded_tensor[i, :len(tokens)] = tokens_tensor

    # 5. Get communication group info
    world_size = dist.get_world_size(group=group)

    # 6. Execute all-gather operation
    gathered_tensor = torch.zeros((world_size * num_prompts, max_len_global), dtype=torch.long, device=device)
    dist.all_gather_into_tensor(gathered_tensor, padded_tensor, group=group)

    # 7. Convert result format and remove padding
    gathered_list = gathered_tensor.cpu().tolist()
    result = []
    for seq in gathered_list:
        # Remove -1 padding and keep original tokens
        filtered = [token for token in seq if token != -1]
        result.append(filtered)

    return result


def gather_inputs_object_for_inference(input_data: List[Any], group: torch.distributed.ProcessGroup) -> List[Any]:
    """
    All-gather data between inference engine tensor parallel group.

    This function collects data from all processes in the specified process group
    and returns a combined list of all items. It's useful for aggregating distributed
    inputs before processing in a tensor-parallel inference setup. The function
    handles arbitrary Python objects using PyTorch's object gathering mechanism.

    :param input_data: List of objects to be gathered from the current process.
                      Can contain any serializable Python objects.
    :type input_data: List[Any]
    :param group: Inference engine tensor parallel process group that defines the
                  communication context for gathering operations.
    :type group: torch.distributed.ProcessGroup

    :return: Combined list of objects from all processes in the group.
            Items from each process are flattened into a single list.
    :rtype: List[Any]

    Example::

        >>> # Gather inference inputs across tensor parallel processes
        >>> local_inputs = [{"prompt": "Hello"}, {"prompt": "World"}]
        >>> all_inputs = gather_inputs_object_for_inference(local_inputs, tp_group)
        >>> # Result contains inputs from all processes in the tensor parallel group
    """
    if torch.distributed.get_world_size(group) == 1:
        return input_data
    gathered_data = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(gathered_data, input_data, group=group)
    all_data = [data_item for rank_data in gathered_data for data_item in rank_data]
    # delete the reference of gathered_data to avoid unnecessary memory occupation
    del gathered_data
    return all_data
