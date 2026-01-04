import os
import sys
from typing import Any, Dict, List, Optional

from datasets import interleave_datasets, load_dataset, load_from_disk, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoProcessor
import torch
import torch.distributed as dist


def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def get_tokenizer_processor_vl(pretrain, model, padding_side="left", strategy=None, use_fast=True):
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
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    max_count=5000000,
    return_eval=True,
    stopping_strategy="first_exhausted",
    train_split="train",
    eval_split="test",
):
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


def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")


def print_rank_0(msg, *args, **kwargs):
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


def get_current_device(num_device_per_node=8) -> torch.device:
    """
    Returns the current CUDA device.

    This function provides a convenient way to get the current CUDA device
    being used by PyTorch.

    :param num_device_per_node: Number of devices per node for distributed training
    :type num_device_per_node: int
    :return: Current CUDA device
    :rtype: torch.device

    Example::

        >>> device = get_current_device()
        >>> model = model.to(device)
    """
    if not torch.distributed.is_initialized():
        return torch.cuda.current_device()
    return torch.device(f"cuda:{torch.distributed.get_rank() % num_device_per_node}")


def get_torch_profiler(output_file, warmup=1, active=1, repeat=1):
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

    def __enter__(self):
        """
        Context manager entry method.

        :return: Self instance
        :rtype: DummyProfile
        """
        return self

    def __exit__(self, a, b, c):
        """
        Context manager exit method.

        :param a: Exception type
        :param b: Exception value
        :param c: Exception traceback
        """
        pass

    def start(self):
        """
        Dummy implementation of the profiler start method.
        Does nothing.
        """
        pass

    def stop(self):
        """
        Dummy implementation of the profiler stop method.
        Does nothing.
        """
        pass

    def step(self):
        """
        Dummy implementation of the profiler step method.
        Does nothing.
        """
        pass


def ensure_video_input_available():
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
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
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
