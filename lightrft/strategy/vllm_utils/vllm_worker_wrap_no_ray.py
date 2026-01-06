"""
This module provides a wrapper for the vLLM worker that extends its functionality.

The main purpose of this module is to provide a way to update weights of a vLLM worker
model from a source rank. This is particularly useful for distributed training or
inference scenarios where model weights need to be synchronized across multiple workers.
"""

import torch
# vLLM version compatibility notes:
# --------------------------------
# In older versions of vLLM (< 0.13.0), the Worker class is located under:
#     vllm.worker.worker.Worker
#
# In vLLM >= 0.13.0, the Worker implementation was moved to:
#     vllm.v1.worker.gpu_worker.Worker
#
# To maintain compatibility across different vLLM versions, we try importing Worker
# from the new v1 path first (for vllm>=0.13.0). If the import fails (ModuleNotFoundError),
# we fall back to importing from the old path (for vllm<0.13.0).
try:
    from vllm.v1.worker.gpu_worker import Worker
except (ModuleNotFoundError, ImportError):
    try:
        from vllm.worker.worker import Worker
    except (ModuleNotFoundError, ImportError):
        raise ImportError(
            "Could not import Worker from vllm. "
            "Please ensure you have a compatible version of vllm installed. "
            "Supported versions: vllm>=0.6.3 or vllm>=0.13.0"
        )


class WorkerWrap(Worker):
    """
    A wrapper for vLLM worker that extends its functionality.

    This class inherits from vLLM's Worker class and adds the ability to update
    model weights dynamically. This is particularly useful for distributed setups
    where weights need to be broadcast from a source rank to all workers.

    :inherits: vllm.worker.worker.Worker
    """
    def update_weight(self, name, dtype, shape, weight, empty_cache=False):  # pylint: disable=R0917, W0613
        """
        Broadcast weight to all vLLM workers from source rank 0 (actor model).

        This method updates a specific weight tensor in the model. It ensures that
        the data type of the incoming weight matches the model's configured data type
        before loading the weight into the model.

        :param name: The name of the weight tensor to update.
        :type name: str
        :param dtype: The data type of the weight tensor.
        :type dtype: torch.dtype
        :param shape: The shape of the weight tensor.
        :type shape: tuple
        :param weight: The new weight tensor values.
        :type weight: torch.Tensor
        :param empty_cache: Whether to empty CUDA cache after updating weights.
        :type empty_cache: bool

        :raises AssertionError: If the data type of the weight doesn't match the model's configured data type.
        """

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight
        # TODO: should we empty cache if all weights have updated?
        if empty_cache:
            torch.cuda.empty_cache()
