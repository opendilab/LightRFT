"""
Python startup hook for URSA rollout-engine subprocesses.

SGLang starts fresh Python worker processes for its scheduler/runtime. Those
workers do not inherit the parent process' in-memory ``AutoConfig`` /
``AutoModel`` registrations, so custom URSA checkpoints still fail to resolve
``model_type='ursa'`` unless we register them again at interpreter startup.

This file is only activated when ``LIGHTRFT_REGISTER_URSA_AUTO_CLASSES=1`` and
the current directory is on ``PYTHONPATH``.
"""

import os


def _maybe_register_ursa() -> None:
    if os.environ.get("LIGHTRFT_REGISTER_URSA_AUTO_CLASSES") != "1":
        return

    try:
        from transformers import (
            AutoConfig,
            AutoModelForTokenClassification,
            AutoModelForVision2Seq,
        )
        from ursa_model import (
            UrsaConfig,
            UrsaForConditionalGeneration,
            UrsaForTokenClassification,
        )
    except Exception:
        return

    AutoConfig.register("ursa", UrsaConfig, exist_ok=True)
    AutoModelForVision2Seq.register(UrsaConfig, UrsaForConditionalGeneration, exist_ok=True)
    AutoModelForTokenClassification.register(UrsaConfig, UrsaForTokenClassification, exist_ok=True)


_maybe_register_ursa()
