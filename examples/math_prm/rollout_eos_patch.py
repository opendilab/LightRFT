"""
Math PRM rollout EOS patch — keeps the fix local to examples/math_prm/.

Background
----------
On 8-GPU FSDP rollouts, historical attempts to terminate URSA generation
through a ``LogitsProcessor`` that nudges the eos logit up were unreliable:
logs showed the processor firing hundreds of times (``forced_eos_rows=291``
per batch-of-4) while every sample still ran the full ``max_new_tokens``
(``mean_length=511.8`` / 512). The "logits nudge → sampled token →
``EosTokenCriteria``" handshake does not close under FSDP's numerical
regime, and even on single card it is only probabilistic.

Fix
---
Install a ``StoppingCriteria`` directly on the rollout actor's underlying
HF model. HF's sample loop calls ``stopping_criteria(input_ids, scores)``
*after* each new token is appended, and ANDs the returned mask into
``unfinished_sequences``. When we return True for a row, HF marks it
finished immediately — no sampling, no logit tricks, no numerical edge.

The criteria also exposes an ``eos_token_id`` attribute so HF's
``has_eos_stopping_criteria`` detection (``utils.py:2735``) treats our
signal as EOS-equivalent and enables the post-EOS pad-fill path at
``utils.py:2835`` for rows we have marked done.

Shape
-----
This module is self-contained under ``examples/math_prm/`` and is installed
from ``train_colocate.py`` via ``install_math_prm_rollout_eos_patch``. The
install helper wraps ``rollout_actor.model.generate`` so every generate
call gets a fresh criteria injected. Since math_prm's training loop only
ever runs math_prm batches, unconditional injection is correct — on
non-math content the criteria simply never sees ``†Answer:`` and its
``done`` mask stays all-False (pure runtime no-op).

No changes to ``lightrft/`` are required.
"""

from __future__ import annotations

import functools
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import torch
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

from math_prm_output import MATH_PRM_ANSWER_MARKER, should_stop_math_prm_response_text


class StructuredAnswerStoppingCriteria(StoppingCriteria):
    """
    Terminate URSA-math_prm rollout generation when a fully-formed
    ``†Answer: <answer>`` line has been emitted.

    Key properties:

    - Exposes ``eos_token_id`` as an attribute so HF's internal
      ``has_eos_stopping_criteria`` detection (``utils.py:2735``) treats this
      criteria as EOS-equivalent, which enables the post-EOS pad-fill path
      (``utils.py:2835``). Without that attr, rows we mark done would keep
      getting non-pad filler tokens written into their slots, and
      ``process_sequences`` — which derives attention-mask from
      ``seq.ne(eos_token_id) & seq.ne(pad_token_id)`` — would still count
      those positions as real content.
    - Checks only every ``check_interval`` tokens to amortise CPU
      ``batch_decode`` cost (matching the existing LogitsProcessor cadence).
    - Done bits are *sticky*: once set, the criteria re-asserts them on
      every subsequent call, including between gated checks. This is
      critical — HF's sample loop ANDs our return into
      ``unfinished_sequences`` (``utils.py:2842``), so if we ever returned
      False for a row we had previously stopped, HF would un-stop it.
    """

    def __init__(self, tokenizer, prompt_length: int, eos_token_id: int):
        self.tokenizer = tokenizer
        self.prompt_length = int(prompt_length)
        self.eos_token_id = int(eos_token_id)
        self.check_interval = 4
        self.marker_scan_max_tokens = 192
        self.answer_tail_max_tokens = 128
        self.answer_marker_token_ids = tuple(
            int(token_id) for token_id in tokenizer.encode(MATH_PRM_ANSWER_MARKER, add_special_tokens=False)
        )
        self._marker_seen: Optional[List[bool]] = None
        self._done: Optional[torch.Tensor] = None
        self._stats: Dict[str, float] = defaultdict(float)

    def _ensure_state(self, batch_size: int, device) -> None:
        if self._marker_seen is None or len(self._marker_seen) != batch_size:
            self._marker_seen = [False] * batch_size
        if self._done is None or self._done.numel() != batch_size:
            self._done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        elif self._done.device != device:
            self._done = self._done.to(device)

    def _scan_row_for_answer_marker(self, row_token_ids: torch.Tensor) -> bool:
        marker_token_ids = self.answer_marker_token_ids
        if not marker_token_ids:
            return MATH_PRM_ANSWER_MARKER in self.tokenizer.decode(row_token_ids, skip_special_tokens=False)

        token_ids = row_token_ids.tolist()
        marker_len = len(marker_token_ids)
        if len(token_ids) < marker_len:
            return False

        search_start = max(0, len(token_ids) - self.marker_scan_max_tokens)
        token_ids = token_ids[search_start:]
        last_start = len(token_ids) - marker_len + 1
        for start_idx in range(max(last_start, 0)):
            if tuple(token_ids[start_idx:start_idx + marker_len]) == marker_token_ids:
                return True
        return False

    def _decode_rows(self, row_token_ids: torch.Tensor) -> List[str]:
        decode_t0 = time.time()
        texts = self.tokenizer.batch_decode(row_token_ids, skip_special_tokens=False)
        self._stats["decode_time_s"] += time.time() - decode_t0
        self._stats["decoded_rows"] += len(texts)
        return texts

    def get_debug_stats(self) -> Optional[Dict[str, Union[int, float]]]:
        if self._stats["calls"] <= 0:
            return None
        return {
            "calls": int(self._stats["calls"]),
            "gated_checks": int(self._stats["gated_checks"]),
            "marker_scan_rows": int(self._stats["marker_scan_rows"]),
            "marker_hits": int(self._stats["marker_hits"]),
            "answer_tail_rows": int(self._stats["answer_tail_rows"]),
            "decoded_rows": int(self._stats["decoded_rows"]),
            "stopped_rows": int(self._stats["stopped_rows"]),
            "decode_time_s": round(float(self._stats["decode_time_s"]), 4),
        }

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        self._stats["calls"] += 1
        batch_size = input_ids.size(0)
        self._ensure_state(batch_size, input_ids.device)

        if input_ids.size(1) <= self.prompt_length:
            return self._done.clone()

        generated_length = input_ids.size(1) - self.prompt_length
        # Always return the sticky done mask, even on non-gated-check steps,
        # so HF cannot flip previously-stopped rows back to unfinished.
        if generated_length % self.check_interval != 0:
            return self._done.clone()
        self._stats["gated_checks"] += 1

        unresolved_rows = [
            idx for idx in range(batch_size)
            if not self._marker_seen[idx] and not bool(self._done[idx].item())
        ]
        if unresolved_rows:
            scan_start = max(self.prompt_length, input_ids.size(1) - self.marker_scan_max_tokens)
            scan_ids = input_ids[unresolved_rows, scan_start:].detach().cpu()
            self._stats["marker_scan_rows"] += len(unresolved_rows)
            matched_row_indices = []
            matched_scan_ids = []
            for row_idx, row_token_ids in zip(unresolved_rows, scan_ids):
                if self._scan_row_for_answer_marker(row_token_ids):
                    self._marker_seen[row_idx] = True
                    matched_row_indices.append(row_idx)
                    matched_scan_ids.append(row_token_ids)

            if matched_row_indices:
                self._stats["marker_hits"] += len(matched_row_indices)
                matched_scan_ids = torch.stack(matched_scan_ids)
                scan_texts = self._decode_rows(matched_scan_ids)
                for row_idx, text in zip(matched_row_indices, scan_texts):
                    if should_stop_math_prm_response_text(text):
                        self._done[row_idx] = True

        marker_rows = [
            idx for idx in range(batch_size)
            if self._marker_seen[idx] and not bool(self._done[idx].item())
        ]
        if marker_rows:
            tail_start = max(self.prompt_length, input_ids.size(1) - self.answer_tail_max_tokens)
            tail_ids = input_ids[marker_rows, tail_start:].detach().cpu()
            self._stats["answer_tail_rows"] += len(marker_rows)
            tail_texts = self._decode_rows(tail_ids)
            for row_idx, text in zip(marker_rows, tail_texts):
                if should_stop_math_prm_response_text(text):
                    self._done[row_idx] = True

        self._stats["stopped_rows"] = int(self._done.sum().item())
        return self._done.clone()


def install_math_prm_rollout_eos_patch(rollout_actor, tokenizer, eos_token_id: int) -> None:
    """
    Wrap ``rollout_actor.model.generate`` so that every generate call gets a
    fresh ``StructuredAnswerStoppingCriteria`` injected into its
    ``stopping_criteria`` kwarg.

    This is only installed from the math_prm example's ``train_colocate.py``
    on the dedicated rollout actor that is used exclusively for math_prm
    batches, so unconditional injection is correct and keeps the patch
    self-contained without any reliance on lightrft-side signals.

    For non-math batches the criteria simply never sees ``†Answer:`` in the
    decoded tail, so its ``done`` mask stays all-False and the patch is a
    no-op at runtime.

    Idempotent: a second install call is a no-op.
    """
    model = rollout_actor.model
    if getattr(model, "_math_prm_rollout_eos_patch_installed", False):
        return

    orig_generate = model.generate

    @functools.wraps(orig_generate)
    def patched_generate(*args: Any, **kwargs: Any):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is not None and hasattr(input_ids, "size"):
            prompt_length = int(input_ids.size(1))
            new_criteria = StructuredAnswerStoppingCriteria(
                tokenizer=tokenizer,
                prompt_length=prompt_length,
                eos_token_id=int(eos_token_id),
            )
            existing = kwargs.get("stopping_criteria")
            if existing is None:
                kwargs["stopping_criteria"] = StoppingCriteriaList([new_criteria])
            else:
                # Be conservative — if caller already provided criteria,
                # prepend ours rather than dropping theirs.
                kwargs["stopping_criteria"] = StoppingCriteriaList([new_criteria, *existing])

            # HF auto-enables `synced_gpus=True` under FSDP (see
            # generation/utils.py:2218), but each rank here runs an independent
            # local-HF generate on its own prompt slice: reshard_after_forward
            # is False on the rollout actor so there are no per-step
            # collectives. Leaving synced_gpus on causes the loop to `continue`
            # past the input_ids append at utils.py:2838 once this rank's rows
            # are all done — combined with URSA's prefill-vs-decode branching
            # in modeling_ursa.py:279 (takes prefill when
            # `input_ids.shape[1] != 1`), the stale input_ids triggers an
            # IndexError in `_merge_input_ids_with_image_features`. Force it
            # off so each rank's generate loop exits cleanly when its own
            # stopping criteria fire.
            kwargs.setdefault("synced_gpus", False)

        return orig_generate(*args, **kwargs)

    model.generate = patched_generate
    model._math_prm_rollout_eos_patch_installed = True
