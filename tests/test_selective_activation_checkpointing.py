# [MODULE DOC]
# Tests for trl.models.selective_activation_checkpointing (SAC): the save/recompute policy and its effect on real
# training with gradient checkpointing.
# Main contents: a dispatch-mode counter of ops the policy would save, a policy-only test class (no accelerator),
# and an accelerator-gated class comparing SAC with full checkpointing on the tiny Qwen2 model.
# Import-time side effects: none.

# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import CheckpointPolicy
from transformers import AutoModelForCausalLM
from transformers.testing_utils import torch_device

from trl.models.selective_activation_checkpointing import (
    _aten_attention_ops,
    _build_policy_fn,
    enable_selective_activation_checkpointing,
)

from .testing_utils import TrlTestCase, require_torch_accelerator


class _SDPACounter(TorchDispatchMode):
    # [CLASS DOC]
    # HELPER: TorchDispatchMode that counts dispatched ops for which the SAC policy returns MUST_SAVE.
    # State: count (int, starts 0) and _policy_fn built from the registered aten attention ops.
    # Used as a context manager around forward+backward to count attention kernel executions.
    """Counts dispatcher calls the SAC policy would save, not just anything named `scaled_dot_product` - e.g. the
    `_scaled_dot_product_attention_math` fallback (older transformers, explicit causal mask) still has that name but
    isn't in the policy's save list, so it must not count as "the policy has something to save"."""

    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Initializes the dispatch mode, zeroes count and builds the policy function.
        # Inputs: none. Outputs: none. Exceptions: none. Side effects: sets attributes.
        super().__init__()
        self.count = 0
        self._policy_fn = _build_policy_fn(_aten_attention_ops())

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # [FUNCTION DOC]
        # HELPER: Increments count when the policy marks func as MUST_SAVE, then executes func with args and kwargs
        # (kwargs defaulting to an empty dict) and returns its result.
        # Inputs: func, types, args, kwargs. Outputs: the op result. Exceptions: propagates op errors.
        # Side effects: mutates count.
        if self._policy_fn(None, func) == CheckpointPolicy.MUST_SAVE:
            self.count += 1
        return func(*args, **(kwargs or {}))


class TestSelectiveActivationCheckpointingPolicy(TrlTestCase):
    # [CLASS DOC]
    # Unit tests for the policy built by _build_policy_fn against aten ops and name strings; needs no model.
    """Exercises the SAC policy directly, against real and fake ops, so it needs no accelerator or model."""

    def test_saves_aten_sdpa_ops(self):
        # [FUNCTION DOC]
        # Asserts at least one aten SDPA op is registered and the policy returns MUST_SAVE for each of them.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: none.
        aten_attention_ops = _aten_attention_ops()
        assert aten_attention_ops, "At least one SDPA backend should be registered under torch.ops.aten"
        policy_fn = _build_policy_fn(aten_attention_ops)
        for op in aten_attention_ops:
            assert policy_fn(None, op) == CheckpointPolicy.MUST_SAVE

    def test_saves_flash_attn_ops_by_name(self):
        # [FUNCTION DOC]
        # Asserts that with an empty aten-op set the policy still returns MUST_SAVE for a flash-attn op name string
        # (substring matching on names). Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        # flash-attn's custom kernels live under hashed namespaces (e.g. `_flash_attn2_cuda_f12afc9::varlen_fwd`),
        # so they can only be recognized by substring, not by identity like the aten ops above.
        policy_fn = _build_policy_fn(set())
        assert policy_fn(None, "flash_attn2_cuda::varlen_fwd") == CheckpointPolicy.MUST_SAVE

    def test_recomputes_everything_else(self):
        # [FUNCTION DOC]
        # Asserts the policy returns PREFER_RECOMPUTE for a non-attention op (aten mm).
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        policy_fn = _build_policy_fn(_aten_attention_ops())
        assert policy_fn(None, torch.ops.aten.mm.default) == CheckpointPolicy.PREFER_RECOMPUTE


@require_torch_accelerator
class TestSelectiveActivationCheckpointing(TrlTestCase):
    # [CLASS DOC]
    # End-to-end SAC tests on the tiny Qwen2 model with SDPA attention; requires an accelerator.
    # Class attribute model_id names the model. _forward_backward is the shared runner.
    model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

    def _forward_backward(self, selective):
        # [FUNCTION DOC]
        # HELPER: Loads the model with sdpa attention on torch_device, sets train mode, optionally enables SAC, enables
        # non-reentrant gradient checkpointing, seeds 42, runs forward with labels on random 2x32 input ids and backward
        # under an _SDPACounter.
        # Inputs: selective (bool, whether to enable SAC). Outputs: tuple of the model (with gradients) and the count.
        # Exceptions: propagates model errors. Side effects: loads a model, sets the global torch seed.
        model = AutoModelForCausalLM.from_pretrained(self.model_id, attn_implementation="sdpa").to(torch_device)
        model.train()
        if selective:
            enable_selective_activation_checkpointing(model)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        torch.manual_seed(42)
        input_ids = torch.randint(0, model.config.vocab_size, (2, 32), device=torch_device)
        with _SDPACounter() as counter:
            model(input_ids=input_ids, labels=input_ids).loss.backward()
        return model, counter.count

    def test_matches_full_checkpointing(self):
        # [FUNCTION DOC]
        # Asserts every parameter gradient with SAC matches full checkpointing (rtol 1e-4, atol 1e-5).
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: loads two models.
        """SAC must produce the same gradients as full checkpointing."""
        model_full, _ = self._forward_backward(selective=False)
        model_sac, _ = self._forward_backward(selective=True)
        for p_full, p_sac in zip(model_full.parameters(), model_sac.parameters(), strict=True):
            torch.testing.assert_close(p_sac.grad, p_full.grad, rtol=1e-4, atol=1e-5)

    def test_skips_attention_recompute(self):
        # [FUNCTION DOC]
        # Asserts full checkpointing counts exactly num_hidden_layers more saved-attention dispatches than SAC (one
        # recompute per layer avoided). Skips if the backend dispatches no countable fused SDPA op.
        # Inputs/Outputs: none. Exceptions: AssertionError; pytest skip. Side effects: loads models.
        """Full checkpointing dispatches SDPA 3 times per layer (forward, recompute, backward), SAC only 2."""
        model, count_full = self._forward_backward(selective=False)
        if count_full == 0:
            pytest.skip("this attention backend doesn't dispatch a fused SDPA op to count")
        _, count_sac = self._forward_backward(selective=True)
        assert count_full - count_sac == model.config.num_hidden_layers

    def test_idempotent(self):
        # [FUNCTION DOC]
        # Asserts that enabling SAC twice leaves the same gradient_checkpointing_enable wrapper object (no stacking).
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: loads a model.
        """Enabling SAC twice on the same model must not stack wrappers."""
        model = AutoModelForCausalLM.from_pretrained(self.model_id, attn_implementation="sdpa").to(torch_device)
        enable_selective_activation_checkpointing(model)
        once_wrapped = model.gradient_checkpointing_enable
        enable_selective_activation_checkpointing(model)
        assert model.gradient_checkpointing_enable is once_wrapped
