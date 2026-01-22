"""
This module provides memory management utilities for SGLang model execution.

The module implements memory saving functionality by allowing temporary release and restoration
of GPU memory occupied by model weights and states. This is particularly useful in scenarios
where multiple models or processes need to share limited GPU memory resources efficiently.

This module is designed to be compatible with different versions of SGLang:
- For SGLang v0.5.6.post2+: Uses built-in methods from SchedulerUpdateWeightsMixin
- For older versions: Provides backward-compatible monkey patching

The module automatically detects which approach to use based on the SGLang version.
"""

import sglang
import sglang.srt.managers.scheduler as scheduler_module
from sglang.srt.managers.io_struct import (
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
)

# Try to import from new location (v0.5.6.post2+), fallback to old location
try:
    from sglang.srt.managers.scheduler_update_weights_mixin import (
        _export_static_state,
        _import_static_state,
    )
except ImportError:
    from sglang.srt.managers.scheduler import _export_static_state, _import_static_state


def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
    """
    Release memory occupation by stashing model weights and states to CPU memory.

    This method temporarily frees GPU memory by moving model parameters and static states
    to CPU memory. It's designed to be used when the model is temporarily not needed,
    allowing other processes or models to utilize the freed GPU memory.

    Compatible with both old and new SGLang versions by detecting the model runner location.

    The method performs the following operations:
        1. Validates the memory saver adapter
        2. Exports and stashes the model's static state
        3. Clones model parameters to CPU memory if not already done
        4. Pauses the memory saver adapter
        5. Flushes the model cache

    :param recv_req: Request input for releasing memory occupation
    :type recv_req: ReleaseMemoryOccupationReqInput

    :return: Response indicating successful memory release
    :rtype: ReleaseMemoryOccupationReqOutput

    Example::
        >>> scheduler = Scheduler(...)
        >>> req = ReleaseMemoryOccupationReqInput()
        >>> response = scheduler.release_memory_occupation(req)
        >>> # GPU memory is now freed for other uses
    """
    # Get model reference - compatible with different SGLang versions
    # Old version: self.tp_worker.worker.model_runner.model
    # New version (v0.5.6+): self.tp_worker.model_runner.model
    if hasattr(self.tp_worker, 'worker'):
        model = self.tp_worker.worker.model_runner.model
    else:
        model = self.tp_worker.model_runner.model

    self.memory_saver_adapter.check_validity(caller_name="release_memory_occupation")
    self.stashed_model_static_state = _export_static_state(model)

    if not hasattr(self, "stashed_weights"):
        self.stashed_weights = {}
        for name, param in model.named_parameters():
            self.stashed_weights[name] = param.data.detach().clone().cpu()

    self.memory_saver_adapter.pause()
    self.flush_cache()
    return ReleaseMemoryOccupationReqOutput()


def resume_memory_occupation(self, recv_req: ResumeMemoryOccupationReqInput):
    """
    Resume memory occupation by restoring model weights and states from CPU memory.

    This method restores the model to its fully operational state by loading back
    the previously stashed model parameters and static states from CPU memory to GPU.
    It should be called after release_memory_occupation() when the model needs to
    be used again.

    Compatible with both old and new SGLang versions by detecting the model runner location.

    The method performs the following operations:
        1. Validates the memory saver adapter
        2. Resumes the memory saver adapter
        3. Imports the previously stashed static state
        4. Restores model parameters from CPU to GPU
        5. Cleans up temporary static state storage

    :param recv_req: Request input for resuming memory occupation
    :type recv_req: ResumeMemoryOccupationReqInput

    :return: Response indicating successful memory restoration
    :rtype: ResumeMemoryOccupationReqOutput

    Example::
        >>> scheduler = Scheduler(...)
        >>> # After previously calling release_memory_occupation()
        >>> req = ResumeMemoryOccupationReqInput()
        >>> response = scheduler.resume_memory_occupation(req)
        >>> # Model is now ready for inference again
    """
    # Get model reference - compatible with different SGLang versions
    if hasattr(self.tp_worker, 'worker'):
        model = self.tp_worker.worker.model_runner.model
    else:
        model = self.tp_worker.model_runner.model

    self.memory_saver_adapter.check_validity(caller_name="resume_memory_occupation")
    self.memory_saver_adapter.resume()
    _import_static_state(model, self.stashed_model_static_state)
    del self.stashed_model_static_state

    def _import_params(model, static_params):
        """
        Import parameters from stashed weights back to the model.

        :param model: The target model to restore parameters to
        :param static_params: Dictionary of stashed parameter weights
        """
        for name, param in model.named_parameters():
            param.data.copy_(static_params[name])

    _import_params(model, self.stashed_weights)

    return ResumeMemoryOccupationReqOutput()


# Apply monkey patching only if the Scheduler class doesn't already have these methods
# This ensures compatibility with both old versions (need patching) and new versions (already have methods)
if not hasattr(scheduler_module.Scheduler, 'release_memory_occupation'):
    # Old version: need to add the methods via monkey patching
    scheduler_module.Scheduler.release_memory_occupation = release_memory_occupation
    scheduler_module.Scheduler.resume_memory_occupation = resume_memory_occupation
else:
    # New version (v0.5.6.post2+): methods already exist via SchedulerUpdateWeightsMixin
    # No patching needed, but we keep our implementation available for reference
    pass
