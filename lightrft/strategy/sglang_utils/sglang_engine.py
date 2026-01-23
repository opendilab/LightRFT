"""
This module provides a distributed reinforcement learning generation engine for language models.

The RLGenerationEngine class facilitates text generation across multiple processes and nodes,
leveraging PyTorch's distributed capabilities. It supports tensor parallelism for efficient
model execution and provides methods for generating text, updating model weights, and
managing memory resources.

The module is designed to work with different versions of SGLang:
- Supports both old and new Engine import paths
- Compatible with various SGLang API changes across versions

The module is designed to work with the SGLang runtime (srt) system and supports features
such as batch processing, custom sampling parameters, and LoRA fine-tuning.
"""

import os
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist
from sglang.srt.model_executor.model_runner import LocalSerializedTensor

# Try multiple import paths for Engine to ensure compatibility across versions
try:
    # Newer versions (v0.5.0+): Engine is in sglang.srt.entrypoints.engine
    from sglang.srt.entrypoints.engine import Engine
except (ModuleNotFoundError, ImportError):
    try:
        # Some versions: Engine might be in sglang.srt.server
        from sglang.srt.server import Engine
    except (ModuleNotFoundError, ImportError):
        # Fallback: try sglang directly
        from sglang import Engine

from sglang.srt.utils import MultiprocessingSerializer, broadcast_pyobj
from torch.distributed.tensor import DTensor
from lightrft.strategy.utils.distributed_util import gather_inputs_object_for_inference

# This line is necessary, it is used to avoid the memory occupation problem
# This import will implicitly apply the release_memory_occupation and resume_memory_occupation into sglang
# For new versions of sglang (v0.5.6.post2+), the methods are already built-in, so no patching occurs
# For old versions, the methods will be monkey-patched into the Scheduler class
from .sgl_model_saver import release_memory_occupation, resume_memory_occupation  # noqa


class RLGenerationEngine:
    """
    A distributed reinforcement learning generation engine for language models.

    This class manages text generation across multiple processes and nodes using tensor parallelism.
    It wraps the SGLang Engine to provide distributed generation capabilities with efficient memory
    management and weight updating functionality.

    The engine automatically handles distributed coordination, ensuring that only the leader rank
    performs actual generation while broadcasting results to all participating processes. It supports
    various input formats including text prompts, token IDs, and image data.

    :param tp_group_cpu: PyTorch process group for tensor parallelism on CPU
    :type tp_group_cpu: torch.distributed.ProcessGroup
    :param num_gpu_per_node: Number of GPUs per node, defaults to 8
    :type num_gpu_per_node: int
    :param kwargs: Additional arguments to pass to the underlying SGLang Engine

    Example::

        >>> import torch.distributed as dist
        >>> # Initialize distributed environment
        >>> dist.init_process_group("nccl")
        >>> tp_group = dist.new_group()
        >>> engine = RLGenerationEngine(
        ...     tp_group_cpu=tp_group,
        ...     num_gpu_per_node=8,
        ...     model="llama2-7b"
        ... )
    """
    def __init__(
        self,
        tp_group_cpu,
        num_gpu_per_node: int = 8,
        **kwargs,
    ):
        """
        Initialize the RLGenerationEngine with distributed configuration.

        :param tp_group_cpu: PyTorch process group for tensor parallelism on CPU
        :type tp_group_cpu: torch.distributed.ProcessGroup
        :param num_gpu_per_node: Number of GPUs per node, defaults to 8
        :type num_gpu_per_node: int
        :param kwargs: Additional arguments to pass to the underlying SGLang Engine
        """

        self.tp_group_cpu = tp_group_cpu
        self._leader_rank = dist.get_process_group_ranks(group=tp_group_cpu)[0]
        self._tp_rank = dist.get_rank(group=tp_group_cpu)
        self._tp_size = dist.get_world_size(tp_group_cpu)

        num_gpu_per_node = min(dist.get_world_size(), num_gpu_per_node)
        tp_size_per_node = min(self._tp_size, num_gpu_per_node)
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0
        nnodes = max(1, self._tp_size // num_gpu_per_node)
        if self._tp_size > num_gpu_per_node:
            assert self._tp_size % num_gpu_per_node == 0

        if first_rank_in_node:
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            self._engine = Engine(**kwargs, tp_size=self._tp_size, node_rank=node_rank, nnodes=nnodes)
        else:
            self._engine = None

        dist.barrier(group=tp_group_cpu)

    def generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        # The image input. It can be a file name, a url, or base64 encoded string.
        # See also python/sglang/srt/utils.py:load_image.
        image_data: Optional[Union[List[str], str]] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
        gather_inputs=False,
    ) -> Dict:
        """
        Generate text using the language model in a distributed manner.

        This method coordinates text generation across multiple processes, with only the leader
        rank performing actual generation and broadcasting results to all participants. It supports
        various input formats and generation parameters.

        The arguments of this function are the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
        Please refer to `GenerateReqInput` for detailed documentation of each parameter.

        :param prompt: The input prompt(s) for text generation
        :type prompt: Optional[Union[List[str], str]]
        :param sampling_params: Parameters controlling the sampling strategy (temperature, top_p, etc.)
        :type sampling_params: Optional[Union[List[Dict], Dict]]
        :param input_ids: Token IDs for input text (alternative to using prompt)
        :type input_ids: Optional[Union[List[List[int]], List[int]]]
        :param image_data: Image input as file name, URL, or base64 encoded string
        :type image_data: Optional[Union[List[str], str]]
        :param return_logprob: Whether to return log probabilities for generated tokens
        :type return_logprob: Optional[Union[List[bool], bool]]
        :param logprob_start_len: Start position for log probability calculation
        :type logprob_start_len: Optional[Union[List[int], int]]
        :param top_logprobs_num: Number of top log probabilities to return
        :type top_logprobs_num: Optional[Union[List[int], int]]
        :param lora_path: Paths to LoRA weights to apply during generation
        :type lora_path: Optional[List[Optional[str]]]
        :param custom_logit_processor: Custom logit processor for modifying logits
        :type custom_logit_processor: Optional[Union[List[str], str]]
        :param gather_inputs: Whether to gather inputs across all ranks before generation
        :type gather_inputs: bool

        :return: Generation results including generated text and metadata
        :rtype: Dict

        Example::

            >>> result = engine.generate(
            ...     prompt="Translate this to French: Hello, world!",
            ...     sampling_params={"temperature": 0.7, "max_tokens": 50}
            ... )
            >>> print(result["text"])
            >>>
            >>> # Batch generation with different parameters
            >>> results = engine.generate(
            ...     prompt=["Hello", "Goodbye"],
            ...     sampling_params=[{"temperature": 0.5}, {"temperature": 0.9}],
            ...     return_logprob=True
            ... )
        """
        if self._tp_size > 1 and gather_inputs:
            if prompt is not None:
                prompt = gather_inputs_object_for_inference(prompt, group=self.tp_group_cpu)
            if input_ids is not None:
                input_ids = gather_inputs_object_for_inference(input_ids, group=self.tp_group_cpu)
            if image_data is not None:
                image_data = gather_inputs_object_for_inference(image_data, group=self.tp_group_cpu)

        if self._tp_rank == 0:
            output = self._engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                input_ids=input_ids,
                image_data=image_data,
                return_logprob=return_logprob,
                logprob_start_len=logprob_start_len,
                top_logprobs_num=top_logprobs_num,
                lora_path=lora_path,
                custom_logit_processor=custom_logit_processor,
            )
        else:
            output = None

        if self._tp_size > 1:
            global_rank = dist.get_rank()

            try:
                [output] = broadcast_pyobj(
                    data=[output],
                    rank=global_rank,
                    dist_group=self.tp_group_cpu,
                    src=self._leader_rank,
                    force_cpu_device=False,
                )
            except TypeError:
                # Older versions don't support force_cpu_device parameter
                [output] = broadcast_pyobj(
                    data=[output],
                    rank=global_rank,
                    dist_group=self.tp_group_cpu,
                    src=self._leader_rank,
                )

            if gather_inputs:
                num_per_rank = len(output) // self._tp_size
                output = output[self._tp_rank * num_per_rank:(self._tp_rank + 1) * num_per_rank]
        return output

    def update_weights_from_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        flush_cache: bool = False,
        load_format: Optional[str] = None,
    ):
        """
        Update model weights with the provided tensor in a distributed manner.

        This method allows dynamic updating of model parameters during runtime, which is
        particularly useful for reinforcement learning scenarios where model weights need
        to be updated based on training feedback. The method handles distributed coordination
        to ensure all ranks participate in the weight update process.

        :param name: Name of the weight tensor to update (e.g., "model.layers.0.self_attn.q_proj.weight")
        :type name: str
        :param tensor: New weight tensor to replace the existing weights
        :type tensor: torch.Tensor
        :param flush_cache: Whether to flush the KV cache after updating weights
        :type flush_cache: bool
        :param load_format: Format specification for loading the weights
        :type load_format: Optional[str]

        Example::

            >>> # Update a specific layer's weights
            >>> new_weights = torch.randn(768, 768)
            >>> engine.update_weights_from_tensor(
            ...     "model.layers.0.self_attn.q_proj.weight",
            ...     new_weights,
            ...     flush_cache=True
            ... )
        """
        # Most naive implementation, can optimize a lot if it is bottleneck
        serialized_tensor = MultiprocessingSerializer.serialize(_preprocess_tensor_for_update_weights(tensor))

        if self._tp_rank == 0:
            gathered_serialized_tensors = [None for _ in range(self._tp_size)]
        else:
            gathered_serialized_tensors = None
        dist.gather_object(
            obj=serialized_tensor,
            object_gather_list=gathered_serialized_tensors,
            dst=self._leader_rank,
            group=self.tp_group_cpu,
        )

        if self._tp_rank == 0:
            self._engine.update_weights_from_tensor(
                named_tensors=[(
                    name,
                    LocalSerializedTensor(values=gathered_serialized_tensors),
                )],
                load_format=load_format,
                flush_cache=flush_cache,
            )

    def sleep(self, release_weights: bool = False):
        """
        Release memory resources temporarily to free up GPU memory.

        This method releases KV cache and CUDA graph memory to free up GPU resources
        during idle periods. By default, model weights are kept in memory to avoid
        the overhead and risk of saving/restoring them.

        :param release_weights: Whether to also release weights memory. Default is False.
                               Set to True only if you need maximum memory savings and
                               understand the risks (SGLang may not properly restore weights).
        :type release_weights: bool

        Note:
            - By default, only KV cache and CUDA graph are released (recommended)
            - Weights are kept in memory unless explicitly requested
            - After calling sleep(), you must call wake_up() before using the engine again
            - ⚠️ WARNING: Releasing weights may cause generation issues due to SGLang limitations

        Example::

            >>> # Standard usage: keep weights in memory (recommended)
            >>> engine.sleep()
            >>> # ... other operations ...
            >>> engine.wake_up()
            >>>
            >>> # Maximum memory savings: also release weights (use with caution)
            >>> engine.sleep(release_weights=True)
            >>> engine.wake_up(release_weights=True)
        """
        if self._tp_rank == 0:
            # Determine which memory types to release
            if release_weights:
                # Release all memory types (not recommended due to SGLang limitations)
                tags = None  # Will default to GPU_MEMORY_ALL_TYPES in SGLang
            else:
                # Only release KV cache and CUDA graph, keep weights (recommended)
                from sglang.srt.constants import (
                    GPU_MEMORY_TYPE_CUDA_GRAPH,
                    GPU_MEMORY_TYPE_KV_CACHE,
                )
                tags = [GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH]

            # Directly pass tags to engine
            self._engine.release_memory_occupation(tags=tags)

    def wake_up(self, release_weights: bool = False):
        """
        Resume memory occupation after a call to sleep().

        This method should be called before using the engine for generation
        after a previous call to sleep(). It restores the engine to its fully
        operational state with all necessary memory allocations.

        :param release_weights: Must match the value used in sleep(). Default is False.
        :type release_weights: bool

        Important:
            - The release_weights parameter must match what was used in sleep()
            - If you called sleep(release_weights=False), call wake_up(release_weights=False)
            - If you called sleep(release_weights=True), call wake_up(release_weights=True)

        Example::

            >>> # Standard usage (recommended)
            >>> engine.sleep()              # Only KV cache & CUDA graph released
            >>> # ... do other work ...
            >>> engine.wake_up()            # Only KV cache & CUDA graph restored
            >>> result = engine.generate("Hello world")
            >>>
            >>> # If weights were released (use with caution)
            >>> engine.sleep(release_weights=True)
            >>> engine.wake_up(release_weights=True)  # Must match!
        """
        if self._tp_rank == 0:
            # Determine which memory types to resume
            if release_weights:
                # Resume all memory types
                tags = None  # Will default to GPU_MEMORY_ALL_TYPES in SGLang
            else:
                # Only resume KV cache and CUDA graph (recommended)
                from sglang.srt.constants import (
                    GPU_MEMORY_TYPE_CUDA_GRAPH,
                    GPU_MEMORY_TYPE_KV_CACHE,
                )
                tags = [GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH]

            # Directly pass tags to engine
            self._engine.resume_memory_occupation(tags=tags)

    def shutdown(self):
        """
        Shut down the engine and release all resources.

        This method should be called when the engine is no longer needed to
        properly clean up resources including GPU memory, process groups, and
        any background threads. It ensures a clean termination of the engine.

        Example::

            >>> try:
            ...     # Use engine for generation
            ...     results = engine.generate("Hello")
            ... finally:
            ...     engine.shutdown()  # Always clean up resources
        """
        if self._engine is not None:
            self._engine.shutdown()


def _preprocess_tensor_for_update_weights(tensor: torch.Tensor):
    """
    Preprocess tensor for weight updates, handling DTensor conversion.

    This function ensures that tensors are in the correct format for weight updates,
    particularly handling the conversion of distributed tensors (DTensor) to regular
    tensors by gathering all shards into a full tensor.

    :param tensor: Input tensor that may be a DTensor or regular torch.Tensor
    :type tensor: torch.Tensor

    :return: Processed tensor ready for weight update, converted from DTensor if necessary
    :rtype: torch.Tensor

    Example::

        >>> # For DTensor, this will gather all shards
        >>> dtensor = create_distributed_tensor(...)
        >>> full_tensor = _preprocess_tensor_for_update_weights(dtensor)
        >>>
        >>> # For regular tensor, this returns the tensor unchanged
        >>> regular_tensor = torch.randn(100, 100)
        >>> same_tensor = _preprocess_tensor_for_update_weights(regular_tensor)
    """
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor
