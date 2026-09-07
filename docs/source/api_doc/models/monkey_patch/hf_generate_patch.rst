Hugging Face generation patch
=============================

``lightrft.models.monkey_patch.hf_generate_patch`` replaces Transformers'
stopping-criteria builder with a variant that omits the EOS criterion. It is an
explicit monkey patch: importing the module alone does not modify
``GenerationMixin``; callers must invoke
``apply_monkey_patch_to_generation_mixin()``.

The implementation is tied to the Transformers generation API noted in the
source file and should be revalidated when that dependency is upgraded.

.. literalinclude:: ../../../../../lightrft/models/monkey_patch/hf_generate_patch.py
   :language: python
   :pyobject: apply_monkey_patch_to_generation_mixin
