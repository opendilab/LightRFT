import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

from datasets import interleave_datasets, load_dataset, load_from_disk, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoProcessor, PreTrainedTokenizer, PreTrainedModel, ProcessorMixin
import torch
import torch.distributed as dist


def get_tokenizer(
    pretrain: str, model: PreTrainedModel, padding_side: str = "left", use_fast: bool = True
) -> PreTrainedTokenizer:
    """
    Load and configure a tokenizer for language models.

    :param pretrain: Path or name of the pretrained tokenizer.
    :type pretrain: str
    :param model: Model instance to sync pad_token_id with.
    :type model: transformers.PreTrainedModel
    :param padding_side: Which side to pad on ('left' or 'right'). Defaults to 'left'
        for causal language models to enable efficient batching during generation,
        where padding tokens should be on the left to avoid affecting the generation.
    :type padding_side: str
    :param use_fast: Whether to use fast tokenizer implementation.
    :type use_fast: bool
    :return: Configured tokenizer instance.
    :rtype: transformers.PreTrainedTokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def get_tokenizer_processor_vl(
    pretrain: str,
    model: PreTrainedModel,
    padding_side: str = "left",
    use_fast: bool = True
) -> Tuple[PreTrainedTokenizer, ProcessorMixin]:
    """
    Load and configure tokenizer and processor for vision-language models.

    :param pretrain: Path or name of the pretrained model.
    :type pretrain: str
    :param model: Model instance to sync pad_token_id with.
    :type model: transformers.PreTrainedModel
    :param padding_side: Which side to pad on ('left' or 'right'). Defaults to 'left'
        for causal language models to enable efficient batching during generation,
        where padding tokens should be on the left to avoid affecting the generation.
    :type padding_side: str
    :param use_fast: Whether to use fast tokenizer implementation.
    :type use_fast: bool
    :return: Tuple of (tokenizer, processor).
    :rtype: Tuple[transformers.PreTrainedTokenizer, transformers.ProcessorMixin]
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    processor = AutoProcessor.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)

    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, processor


def blending_datasets(
    datasets: str,
    probabilities: str,
    strategy: Optional[Any] = None,
    seed: int = 42,
    max_count: int = 5000000,
    return_eval: bool = True,
    stopping_strategy: str = "first_exhausted",
    train_split: str = "train",
    eval_split: str = "test",
) -> Union[Dataset, Tuple[Dataset, Dataset]]:
    """
    Load and blend multiple datasets with specified sampling probabilities.

    Supports various dataset formats including local files (.json, .jsonl, .csv),
    HuggingFace datasets, and datasets saved with ``save_to_disk``.

    :param datasets: Comma-separated dataset paths or names (e.g., 'path1,path2').
    :type datasets: str
    :param probabilities: Comma-separated sampling probabilities (e.g., '0.5,0.5').
    :type probabilities: str
    :param strategy: Optional training strategy for distributed logging.
    :type strategy: Optional[Any]
    :param seed: Random seed for reproducible interleaving.
    :type seed: int
    :param max_count: Maximum number of samples to load per dataset.
    :type max_count: int
    :param return_eval: Whether to return evaluation dataset.
    :type return_eval: bool
    :param stopping_strategy: How to handle datasets of different sizes
        ('first_exhausted' or 'all_exhausted').
    :type stopping_strategy: str
    :param train_split: Name of the training split.
    :type train_split: str
    :param eval_split: Name of the evaluation split.
    :type eval_split: str
    :return: Training dataset, or tuple of (train_dataset, eval_dataset) if return_eval=True.
    :rtype: Union[Dataset, Tuple[Dataset, Dataset]]
    """
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    assert len(probabilities) == len(datasets)

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        if strategy:
            strategy.print(f"dataset: {dataset}")

        data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
        dataset = dataset.split("@")[0].strip()
        dataset_basename = os.path.basename(dataset)

        ext = os.path.splitext(dataset)[-1]
        # local python script
        if ext == ".py" or (os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, f"{dataset_basename}.py"))):
            data = load_dataset(dataset, trust_remote_code=True)
            if strategy:
                strategy.print(f"loaded {dataset} with python script")
        # local text file
        elif ext in [".json", ".jsonl", ".csv"]:
            ext = ext.lower().strip(".")
            if ext == "jsonl":
                ext = "json"
            data = load_dataset(ext, data_files=dataset)
            if strategy:
                strategy.print(f"loaded {dataset} with data_files={dataset}")
        # local dataset saved with `datasets.Dataset.save_to_disk`
        elif os.path.isdir(dataset):
            try:
                data = load_from_disk(dataset)
                if strategy:
                    strategy.print(f"loaded {dataset} from disk")
            except Exception as e:
                if strategy:
                    strategy.print(f"failed to load {dataset} from disk: {e}")
                data = load_dataset(dataset, data_dir=data_dir)
                if strategy:
                    strategy.print(f"loaded {dataset} from files")
        # remote/local folder or common file
        else:
            data = load_dataset(dataset, data_dir=data_dir)
            if strategy:
                strategy.print(f"loaded {dataset} from files")

        # ==================== FIX AND OPTIMIZATION START ====================
        # This block is made robust to handle both Dataset and DatasetDict objects.

        # Try to get the specified training split
        if train_split and train_split in data:
            train_data = data[train_split].select(range(min(max_count, len(data[train_split]))))
        else:
            # If the specified split is not found, or if data is a single Dataset
            actual_dataset = None
            if isinstance(data, DatasetDict):
                # If it's a DatasetDict, intelligently use the first available split.
                # This makes the function compatible with datasets that don't have a 'train' split.
                available_splits = list(data.keys())
                if not available_splits:
                    raise ValueError(f"DatasetDict loaded from {dataset} is empty and has no splits.")

                split_name = available_splits[0]
                actual_dataset = data[split_name]
                if strategy:
                    strategy.print(
                        f"WARN: '{train_split}' split not found or not provided. Using the first split: '{split_name}'"
                    )
            elif isinstance(data, Dataset):
                # If it's already a single Dataset, use it directly.
                actual_dataset = data
            else:
                raise TypeError(f"Loaded data from {dataset} is of an unexpected type: {type(data)}")

            train_data = actual_dataset.select(range(min(max_count, len(actual_dataset))))

        train_data_list.append(train_data)
        # ===================== FIX AND OPTIMIZATION END =====================

        if return_eval:
            # Try to get the specified evaluation split
            if eval_split and eval_split in data:
                eval_data = data[eval_split].select(range(min(max_count, len(data[eval_split]))))
            else:
                # Fallback for evaluation data: use a small fraction of the training data.
                # This part is safe because `train_data` is guaranteed to be a `Dataset` object.
                eval_data = train_data.select(range(min(max_count, int(len(train_data) * 0.03))))
                if strategy:
                    strategy.print(
                        f"WARN: '{eval_split}' split not found. Using 3% of the training data for evaluation."
                    )
            eval_data_list.append(eval_data)

    # merge datasets
    if strategy and strategy.is_rank_0():
        print(f"Blending {len(train_data_list)} training datasets...")
        print(train_data_list)

    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    if return_eval:
        if strategy and strategy.is_rank_0():
            print(f"Blending {len(eval_data_list)} evaluation datasets...")
            print(eval_data_list)
        eval_dataset = interleave_datasets(
            eval_data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
        return train_dataset, eval_dataset
    else:
        return train_dataset


def convert_token_to_id(token: str, tokenizer: PreTrainedTokenizer) -> int:
    """
    Convert a string token to its corresponding token ID.

    :param token: Token string to convert.
    :type token: str
    :param tokenizer: Tokenizer instance to use for conversion.
    :type tokenizer: transformers.PreTrainedTokenizer
    :return: Token ID.
    :rtype: int
    :raises ValueError: If token is not a string or encodes to multiple IDs.
    """
    if isinstance(token, str):
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        assert len(token_ids) == 1, f"Token '{token}' encodes to {len(token_ids)} IDs, expected 1"
        return token_ids[0]
    else:
        raise ValueError(f"token should be a string, got {type(token).__name__}")


def print_rank_0(msg: str, *args: Any, **kwargs: Any) -> None:
    """
    Prints message only from rank 0 process in distributed training.

    This function helps avoid duplicate prints in multi-GPU training by
    only printing from the main process (rank 0).

    :param msg: The message to print
    :type msg: str
    :param args: Additional positional arguments to include in the message
    :param kwargs: Additional keyword arguments to include in the message

    Example::

        >>> print_rank_0("Training started", epoch=1)
    """
    if torch.distributed.get_rank() == 0:
        print(f"RANK 0: {msg} {args} {kwargs}", flush=True)


def get_current_device(num_device_per_node: int = 8) -> torch.device:
    """
    Returns the current device (CUDA or NPU).

    This function provides a convenient way to get the current device
    being used by PyTorch, supporting both CUDA (GPU) and NPU.

    :param num_device_per_node: Number of devices per node for distributed training
    :type num_device_per_node: int
    :return: Current device (CUDA or NPU)
    :rtype: torch.device

    Example::

        >>> device = get_current_device()
        >>> model = model.to(device)
    """
    # Check accelerator type from environment variable
    accelerator_type = os.environ.get("ACCELERATOR_TYPE", "gpu").lower()

    if not torch.distributed.is_initialized():
        # Not in distributed mode
        if accelerator_type == "npu":
            try:
                import torch_npu
                return torch.device(f"npu:{torch.npu.current_device()}")
            except (ImportError, RuntimeError):
                return torch.device("npu:0")
        else:
            try:
                # Use get_current_device recursively to ensure proper device detection
                # import torch.cuda
                return torch.device(f"cuda:{torch.cuda.current_device()}")
            except (RuntimeError, AssertionError):
                return torch.device("cuda:0")
    else:
        # In distributed mode
        rank = torch.distributed.get_rank()
        local_rank = rank % num_device_per_node

        if accelerator_type == "npu":
            return torch.device(f"npu:{local_rank}")
        else:
            return torch.device(f"cuda:{local_rank}")


def is_accelerator_available() -> bool:
    """
    Check if any accelerator (CUDA GPU or NPU) is available.

    This function provides a unified way to check for hardware acceleration,
    supporting both NVIDIA CUDA GPUs and Huawei NPUs.

    :return: True if CUDA or NPU is available, False otherwise
    :rtype: bool

    Example::

        >>> if is_accelerator_available():
        ...     model = model.to(get_current_device())
    """
    accelerator_type = os.environ.get("ACCELERATOR_TYPE", "gpu").lower()

    if accelerator_type == "npu":
        try:
            import torch_npu
            return torch_npu.npu.is_available()
        except (ImportError, AttributeError):
            return False
    else:
        return torch.cuda.is_available()


def device_synchronize() -> None:
    """
    Synchronize all streams on the current device (CUDA or NPU).

    This function waits for all kernels in all streams on the current device to complete.
    It's equivalent to torch.cuda.synchronize() for CUDA or torch_npu.npu.synchronize() for NPU.

    Example::

        >>> device_synchronize()  # Wait for all operations to complete
    """
    accelerator_type = os.environ.get("ACCELERATOR_TYPE", "gpu").lower()

    if accelerator_type == "npu":
        try:
            import torch_npu
            torch_npu.npu.synchronize()
        except (ImportError, AttributeError):
            pass
    else:
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def empty_cache() -> None:
    """
    Release all unoccupied cached memory on the current device (CUDA or NPU).

    This function frees up cached memory that is not currently being used,
    which can help reduce memory fragmentation.

    Example::

        >>> empty_cache()  # Free up cached memory
    """
    accelerator_type = os.environ.get("ACCELERATOR_TYPE", "gpu").lower()

    if accelerator_type == "npu":
        try:
            import torch_npu
            torch_npu.npu.empty_cache()
        except (ImportError, AttributeError):
            pass
    else:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def mem_get_info(device: Optional[Union[torch.device, int]] = None) -> Tuple[int, int]:
    """
    Get memory usage information for the current device (CUDA or NPU).

    Returns a tuple of (free_memory, total_memory) in bytes.

    :param device: Device to query (optional, defaults to current device)
    :type device: Optional[Union[torch.device, int]]
    :return: Tuple of (free_memory_bytes, total_memory_bytes)
    :rtype: Tuple[int, int]

    Example::

        >>> free, total = mem_get_info()
        >>> print(f"Free: {free/1e9:.2f} GB, Total: {total/1e9:.2f} GB")
    """
    accelerator_type = os.environ.get("ACCELERATOR_TYPE", "gpu").lower()

    if accelerator_type == "npu":
        try:
            import torch_npu
            return torch_npu.npu.mem_get_info(device)
        except (ImportError, AttributeError, RuntimeError) as e:
            # Fallback: return dummy values if NPU not available
            return (0, 0)
    else:
        if torch.cuda.is_available():
            return torch.cuda.mem_get_info(device)
        else:
            return (0, 0)


def memory_allocated(device: Optional[Union[torch.device, int]] = None) -> int:
    """
    Get the current memory allocated by tensors on the device (CUDA or NPU).

    :param device: Device to query (optional, defaults to current device)
    :type device: Optional[Union[torch.device, int]]
    :return: Memory allocated in bytes
    :rtype: int

    Example::

        >>> allocated = memory_allocated()
        >>> print(f"Allocated: {allocated/1e9:.2f} GB")
    """
    accelerator_type = os.environ.get("ACCELERATOR_TYPE", "gpu").lower()

    if accelerator_type == "npu":
        try:
            import torch_npu
            return torch_npu.npu.memory_allocated(device)
        except (ImportError, AttributeError, RuntimeError):
            return 0
    else:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(device)
        else:
            return 0


def memory_summary(device: Optional[Union[torch.device, int]] = None) -> str:
    """
    Get a human-readable summary of memory allocator state (CUDA or NPU).

    :param device: Device to query (optional, defaults to current device)
    :type device: Optional[Union[torch.device, int]]
    :return: Memory summary string
    :rtype: str

    Example::

        >>> print(memory_summary())
    """
    accelerator_type = os.environ.get("ACCELERATOR_TYPE", "gpu").lower()

    if accelerator_type == "npu":
        try:
            import torch_npu
            return torch_npu.npu.memory_summary(device)
        except (ImportError, AttributeError, RuntimeError):
            return "NPU memory summary not available"
    else:
        if torch.cuda.is_available():
            return torch.cuda.memory_summary(device)
        else:
            return "CUDA not available"


def set_device(device: Union[torch.device, int]) -> None:
    """
    Set the current device (CUDA or NPU).

    :param device: Device to set as current
    :type device: Union[torch.device, int]

    Example::

        >>> set_device(0)  # Set device 0 as current
    """
    accelerator_type = os.environ.get("ACCELERATOR_TYPE", "gpu").lower()

    if accelerator_type == "npu":
        try:
            import torch_npu
            torch_npu.npu.set_device(device)
        except (ImportError, AttributeError):
            pass
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(device)


def manual_seed_all(seed: int) -> None:
    """
    Set the random seed for all devices (CUDA or NPU).

    :param seed: Random seed value
    :type seed: int

    Example::

        >>> manual_seed_all(42)  # Set seed for reproducibility
    """
    accelerator_type = os.environ.get("ACCELERATOR_TYPE", "gpu").lower()

    if accelerator_type == "npu":
        try:
            import torch_npu
            torch_npu.npu.manual_seed_all(seed)
        except (ImportError, AttributeError):
            pass
    else:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def get_torch_profiler(output_file: str,
                       warmup: int = 1,
                       active: int = 1,
                       repeat: int = 1) -> Union[torch.profiler.profile, "DummyProfile"]:
    """
    Creates and returns a PyTorch profiler configured for distributed training.

    This function returns a properly configured PyTorch profiler for the current process.
    For rank 0 process, it returns a full-featured profiler that records CPU and CUDA activities.
    For other ranks, it returns a dummy profiler that does nothing.

    For more details on PyTorch Profiler, see: https://docs.pytorch.org/docs/stable/profiler.html

    :param output_file: Path where profiling results will be saved (only for rank 0)
    :type output_file: str
    :param warmup: Number of steps to wait before profiling starts
    :type warmup: int
    :param active: Number of steps with active profiling
    :type active: int
    :param repeat: Number of times to repeat the profiling cycle
    :type repeat: int

    :return: A PyTorch profiler object or a dummy profiler
    :rtype: torch.profiler.profile or DummyProfile

    Example::

        >>> with get_torch_profiler("./profiler_output", warmup=5, active=10) as prof:
        >>>     for step in range(100):
        >>>         train_step()
        >>>         prof.step()
    """
    from torch.profiler import ProfilerActivity

    if torch.distributed.get_rank() == 0:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=warmup, active=active, repeat=repeat),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(output_file),
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        )
    else:
        prof = DummyProfile()
    return prof


class DummyProfile:
    """
    Dummy Profile class that mimics the PyTorch profiler API but does nothing.

    This class is used as a placeholder for non-rank-0 processes where profiling
    is not needed, allowing the same code to be used across all processes without
    conditional branches.

    Example::

        >>> prof = DummyProfile() if rank != 0 else torch.profiler.profile(...)
        >>> with prof:
        >>>     # code to be profiled
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize a dummy profiler instance.

        :param args: Positional arguments (ignored)
        :param kwargs: Keyword arguments (ignored)
        """
        pass

    def __enter__(self) -> "DummyProfile":
        """
        Context manager entry method.

        :return: Self instance
        :rtype: DummyProfile
        """
        return self

    def __exit__(self, a: Any, b: Any, c: Any) -> None:
        """
        Context manager exit method.

        :param a: Exception type
        :param b: Exception value
        :param c: Exception traceback
        """
        pass

    def start(self) -> None:
        """
        Dummy implementation of the profiler start method.
        Does nothing.
        """
        pass

    def stop(self) -> None:
        """
        Dummy implementation of the profiler stop method.
        Does nothing.
        """
        pass

    def step(self) -> None:
        """
        Dummy implementation of the profiler step method.
        Does nothing.
        """
        pass


def ensure_video_input_available() -> None:
    """
    Ensure ``VideoInput`` is available from ``transformers.image_utils``.

    This function handles compatibility issues across different versions of
    Transformers where ``VideoInput`` may be defined in different modules.

    Version behavior
    ----------------
    * Transformers < 4.52.0:
        ``VideoInput`` is defined in ``transformers.image_utils``, so
        ``from transformers.image_utils import VideoInput`` works.

    * Transformers >= 4.52.0:
        ``VideoInput`` has been moved to ``transformers.video_utils`` and is
        no longer exported from ``transformers.image_utils``. Importing
        ``VideoInput`` from ``transformers.image_utils`` will fail unless we
        manually patch it.

    What this helper does
    ---------------------
    * Tries to import ``VideoInput`` from ``transformers.image_utils``.
    * If that fails (e.g. Transformers >= 4.52.0), it tries to import
      ``VideoInput`` from ``transformers.video_utils`` instead.
    * If both imports fail, it creates a dummy ``VideoInput`` class as a
      fallback.
    * In all non-error cases, it also attaches ``VideoInput`` back onto the
      ``transformers.image_utils`` module so that:

        >>> ensure_video_input_available()
        >>> from transformers.image_utils import VideoInput  # now works for
        ...                                                  # both old and
        ...                                                  # new versions

    This keeps downstream code compatible with projects that expect
    ``transformers.image_utils.VideoInput`` to exist, regardless of the
    installed Transformers version.
    """
    try:
        from transformers.image_utils import VideoInput
    except ImportError:
        try:
            from transformers.video_utils import VideoInput
        except ImportError:

            class VideoInput:
                """
                Placeholder class for VideoInput when transformers doesn't provide it.

                This class serves as a compatibility shim for older Transformers versions
                that don't export VideoInput from transformers.image_utils or
                transformers.video_utils.
                """
                pass

        import transformers
        transformers.image_utils.VideoInput = VideoInput
        sys.modules["transformers.image_utils"].VideoInput = VideoInput


def all_gather_and_flatten(data: Any, group: Optional[dist.ProcessGroup] = None) -> List[Any]:
    """
    Gather data from all processes and flatten the result into a single list.

    :param data: The data to gather from the current process.
    :type data: Any
    :param group: The process group to work on. If None, the default process group is used.
    :type group: ProcessGroup, optional

    :returns: A flattened list containing data from all processes.
    :rtype: List[Any]
    """
    if not dist.is_initialized():
        return data if isinstance(data, list) else [data]

    world_size = dist.get_world_size(group=group)
    gathered_data = [None] * world_size
    dist.all_gather_object(gathered_data, data, group=group)

    flattened_data = []
    for item in gathered_data:
        if isinstance(item, list):
            flattened_data.extend(item)
        else:
            flattened_data.append(item)
    return flattened_data


def all_reduce_dict(metrics_dict: Dict[str, float],
                    op: str = "sum",
                    group: Optional[dist.ProcessGroup] = None) -> Dict[str, float]:
    """
    Perform all-reduce operation on a dictionary of metrics.
    This function converts the dictionary values to a single tensor for efficient reduction.

    :param metrics_dict: Dictionary of metrics to be reduced.
    :type metrics_dict: Dict[str, float]
    :param op: Reduction operation ('sum', 'max', 'min', 'mean').
    :type op: str
    :param group: The process group to work on. If None, the default process group is used.
    :type group: ProcessGroup, optional

    :returns: Reduced dictionary of metrics.
    :rtype: Dict[str, float]
    """
    if not dist.is_initialized():
        return metrics_dict

    keys = sorted(metrics_dict.keys())
    values = [metrics_dict[k] for k in keys]

    # Use the current device if available, otherwise CPU
    device = get_current_device() if is_accelerator_available() else torch.device("cpu")
    tensor = torch.tensor(values, device=device, dtype=torch.float64)

    dist_op_map = {
        "sum": dist.ReduceOp.SUM,
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
        "mean": dist.ReduceOp.SUM,  # Mean is handled by sum then divide
    }
    dist_op = dist_op_map[op.lower()]

    dist.all_reduce(tensor, op=dist_op, group=group)

    if op.lower() == "mean":
        tensor /= dist.get_world_size(group=group)

    reduced_values = tensor.tolist()
    return {k: v for k, v in zip(keys, reduced_values)}
