# [MODULE DOC]
# Tests for trl.models.activation_offloading: the OffloadActivations context manager (saved-tensor hooks that move
# autograd activations to CPU and bring them back for backward) and NoOpManager (temporarily disables offloading).
# Covers gradient equivalence with/without offloading (plain and PEFT models), NoOpManager, min_offload_size,
# storage deduplication of views (only in streams mode), protection against reused allocator addresses, clearing
# of stale tracker state between forwards, and parameter-storage filtering. All tests require an accelerator.
# Import-time side effects: imports peft (LoraConfig, get_peft_model) only if available.

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

import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.testing_utils import torch_device
from transformers.utils import is_peft_available

from trl.models import activation_offloading as activation_offloading_module
from trl.models.activation_offloading import NoOpManager, OffloadActivations

from .testing_utils import TrlTestCase, require_peft, require_torch_accelerator


if is_peft_available():
    from peft import LoraConfig, get_peft_model


class TestActivationOffloading(TrlTestCase):
    # [CLASS DOC]
    # Test case for activation offloading; each test builds its own model/context. Inherits TrlTestCase.
    @require_torch_accelerator
    @require_peft
    def test_offloading_with_peft_models(self) -> None:
        # [FUNCTION DOC]
        # Checks gradients of a LoRA-wrapped tiny Qwen2 (r 8, alpha 16, dropout 0.1) are the same with and without
        # offloading.
        # Setup: random token input (2, 10); seed 42 forward/backward without offloading, store trainable grads, clear
        # grads, seed 42 forward under OffloadActivations (after update_model_params), backward outside the context.
        # Assertions: each stored grad is close (rtol 1e-4, atol 1e-5) to the new grad of the same parameter.
        # KNOWN ISSUE: the comparison is written as a tuple of the assert_close call and a message string, so the message is
        # never used (assert_close still raises on mismatch).
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: allocates on the accelerator.
        """Test that activation offloading works with PEFT models."""
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=8,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)
        inp = torch.randint(0, 100, (2, 10), device=torch_device)

        # First forward-backward pass without offloading
        torch.manual_seed(42)
        loss = model(inp, labels=inp).loss
        loss.backward()

        # Store gradients - only from trainable parameters
        grads_original = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grads_original.append((name, param.grad.clone()))

        # Reset gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None

        # Second forward-backward pass with offloading
        torch.manual_seed(42)
        offload_ctx = OffloadActivations()
        offload_ctx.update_model_params(model)
        with offload_ctx:
            loss_c = model(inp, labels=inp).loss
        loss_c.backward()

        # Compare gradients - only trainable parameters
        for name_orig, grad_orig in grads_original:
            for name_param, param in model.named_parameters():
                if name_param == name_orig and param.requires_grad and param.grad is not None:
                    (
                        torch.testing.assert_close(grad_orig, param.grad, rtol=1e-4, atol=1e-5),
                        (f"Gradient mismatch for {name_orig}"),
                    )

    @require_torch_accelerator
    def test_noop_manager_with_offloading(self):
        # [FUNCTION DOC]
        # Checks that inside an OffloadActivations context, a forward/backward wrapped in NoOpManager yields the same
        # gradients (rtol 1e-4, atol 1e-5) as one with offloading active, both seeded 42 on the tiny Qwen2 model.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: allocates on the accelerator.
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)
        inp = torch.randint(0, 100, (2, 10), device=torch_device)

        # Run with offloading but disable for specific section
        offload_ctx = OffloadActivations()
        offload_ctx.update_model_params(model)
        with offload_ctx:
            # First forward-backward with normal offloading
            torch.manual_seed(42)
            out1 = model(inp, labels=inp)
            out1.loss.backward()
            grads1 = [p.grad.clone() for p in model.parameters()]

            # Reset grads
            for p in model.parameters():
                p.grad = None

            # Second forward-backward with NoOpManager
            with NoOpManager():
                torch.manual_seed(42)
                out2 = model(inp, labels=inp)
                out2.loss.backward()

            grads2 = [p.grad.clone() for p in model.parameters()]

        # Gradients should match as NoOpManager should have prevented offloading
        for g1, g2 in zip(grads1, grads2, strict=True):
            torch.testing.assert_close(g1, g2, rtol=1e-4, atol=1e-5)

    @require_torch_accelerator
    def test_min_offload_size(self):
        # [FUNCTION DOC]
        # Smoke test: a two-layer Linear stack (5 to 5, 5 to 1000) runs forward and backward inside
        # OffloadActivations with min_offload_size 1000, mixing offloaded and non-offloaded activations. Passes if no error.
        # Inputs/Outputs: none. Exceptions: any runtime error fails the test. Side effects: accelerator allocations.
        """Test that tensors smaller than min_offload_size aren't offloaded"""
        model = nn.Sequential(
            nn.Linear(5, 5),  # Small layer that shouldn't be offloaded
            nn.Linear(5, 1000),  # Large layer that should be offloaded
        ).to(torch_device)

        inp = torch.randn(2, 5, device=torch_device)

        with OffloadActivations(min_offload_size=1000):
            out = model(inp)
            out.sum().backward()

        # The test passes if no errors occur, as we're mainly testing
        # that the logic handles both offloaded and non-offloaded tensors

    @require_torch_accelerator
    def test_real_hf_model(self):
        # [FUNCTION DOC]
        # Checks on tiny Qwen2 that the loss and every parameter gradient are identical (default tolerances) between a
        # baseline forward/backward and one where forward and backward both run inside OffloadActivations (seed 42 each).
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: accelerator allocations.
        """Test with an actual HuggingFace model"""
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)

        # Create small input
        inp = torch.randint(0, 100, (2, 10), device=torch_device)

        # Baseline without offloading
        torch.manual_seed(42)
        out1 = model(inp, labels=inp).loss
        out1.backward()
        grads1 = [p.grad.clone() for p in model.parameters()]

        # Reset grads
        for p in model.parameters():
            p.grad = None

        # With offloading
        offload_ctx = OffloadActivations()
        offload_ctx.update_model_params(model)
        with offload_ctx:
            torch.manual_seed(42)
            out2 = model(inp, labels=inp).loss
            out2.backward()

        grads2 = [p.grad.clone() for p in model.parameters()]

        # Check outputs and gradients match
        torch.testing.assert_close(out1, out2)
        for g1, g2 in zip(grads1, grads2, strict=True):
            torch.testing.assert_close(g1, g2)

    @require_torch_accelerator
    def test_tensor_deduplication(self):
        # [FUNCTION DOC]
        # Checks that saved tensors sharing storage are deduplicated.
        # Setup: nested ModelWithViews whose forward saves a flat view and a transpose of the same linear output;
        # OffloadActivations with min_offload_size 1 and parameters registered; input (10, 100) requiring grad.
        # Assertions: tensor_id counter positive; tracker has at least one offloaded (modified true) and one deduplicated
        # (modified false) entry; number of unique offloaded storages is less than total tensor ids; backward succeeds.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: accelerator allocations.
        """Test that deduplication works correctly for tensors sharing storage"""

        class ModelWithViews(nn.Module):
            # [CLASS DOC]
            # HELPER: Tiny module with one Linear(100, 100) whose forward produces two views of the same output.
            def __init__(self):
                # [FUNCTION DOC]
                # HELPER: Initializes nn.Module and creates the Linear(100, 100) layer.
                # Inputs: none. Outputs: none. Exceptions: none. Side effects: registers the submodule.
                super().__init__()
                self.linear = nn.Linear(100, 100)

            def forward(self, x):
                # [FUNCTION DOC]
                # HELPER: Applies the linear layer, then returns the sum of a flattened view plus the sum of its transpose.
                # Inputs: x - tensor (N, 100). Outputs: scalar tensor. Exceptions: none. Side effects: none.
                out = self.linear(x)
                view1 = out.view(-1)
                view2 = out.transpose(0, 1)
                return view1.sum() + view2.sum()

        model = ModelWithViews().to(torch_device)
        offload_ctx = OffloadActivations(min_offload_size=1)
        offload_ctx.update_model_params(model)

        x = torch.randn(10, 100, device=torch_device, requires_grad=True)
        with offload_ctx:
            loss = model(x)

        total_tensor_ids = offload_ctx.tensor_id
        assert total_tensor_ids > 0, "Should have created tensor IDs"

        # modified=True means offloaded to CPU, modified=False means kept on GPU (deduplicated)
        deduplicated_count = sum(1 for _, modified, _, _, _ in offload_ctx.tracker.values() if not modified)
        offloaded_count = sum(1 for _, modified, _, _, _ in offload_ctx.tracker.values() if modified)

        assert offloaded_count > 0, "Should have offloaded at least one tensor"
        assert deduplicated_count > 0, "Should have deduplicated at least one tensor (view)"

        unique_storages_offloaded = len(offload_ctx.storage_to_tensor_id)
        assert unique_storages_offloaded < total_tensor_ids, (
            f"Deduplication should result in fewer storages ({unique_storages_offloaded}) "
            f"than total tensors ({total_tensor_ids})"
        )

        loss.backward()

    @require_torch_accelerator
    def test_reused_storage_key_after_stash_reap_is_not_deduplicated(self):
        # [FUNCTION DOC]
        # Checks that when a storage key is reused by a new tensor after the earlier one was reaped from the forward stash,
        # the new tensor is not mistaken for a live view.
        # Setup: three 4x4 tensors; the module-level _get_unique_tensor_key is temporarily replaced by fake_storage_key so
        # the first and third tensors share a key; OffloadActivations without pinned memory, with streams,
        # min_offload_size 1 and max_fwd_stash_size 1; each tensor saved via the SaveTensor Function. The original key
        # function is restored in a finally block.
        # Assertions: all 3 tracker entries are offloaded, none deduplicated; backward succeeds.
        # Inputs/Outputs: none. Exceptions: assertion failure.
        # Side effects: temporarily monkeypatches the activation_offloading module function.
        """A reused allocator address should not be treated as a live view."""

        class SaveTensor(torch.autograd.Function):
            # [CLASS DOC]
            # HELPER: Custom autograd Function that saves its input for backward and returns the sum, so the saved tensor goes
            # through the offloading pack/unpack hooks.
            @staticmethod
            def forward(ctx, tensor):
                # [FUNCTION DOC]
                # HELPER: Saves tensor on ctx for backward and returns its sum.
                # Inputs: ctx, tensor. Outputs: scalar tensor. Exceptions: none. Side effects: stores tensor on ctx.
                ctx.save_for_backward(tensor)
                return tensor.sum()

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC]
                # HELPER: Retrieves the saved tensor and returns a gradient of ones shaped like it, scaled by grad_output.
                # Inputs: ctx, grad_output. Outputs: gradient tensor. Exceptions: none. Side effects: none.
                (tensor,) = ctx.saved_tensors
                return torch.ones_like(tensor) * grad_output

        first = torch.randn(4, 4, device=torch_device, requires_grad=True)
        filler = torch.randn(4, 4, device=torch_device, requires_grad=True)
        reused = torch.randn(4, 4, device=torch_device, requires_grad=True)

        def fake_storage_key(tensor):
            # [FUNCTION DOC]
            # HELPER: Replacement storage-key function: returns a shared fixed key (with dtype) for the first and reused
            # tensors, and a key based on the object id (with dtype) for any other tensor.
            # Inputs: tensor. Outputs: tuple key. Exceptions: none. Side effects: none.
            if id(tensor) in {id(first), id(reused)}:
                return ("reused-storage-key", tensor.dtype)
            return (id(tensor), tensor.dtype)

        offload_ctx = OffloadActivations(
            use_pin_memory=False,
            use_streams=True,
            min_offload_size=1,
            max_fwd_stash_size=1,
        )

        original_get_unique_tensor_key = activation_offloading_module._get_unique_tensor_key
        activation_offloading_module._get_unique_tensor_key = fake_storage_key
        try:
            with offload_ctx:
                loss = SaveTensor.apply(first) + SaveTensor.apply(filler) + SaveTensor.apply(reused)
        finally:
            activation_offloading_module._get_unique_tensor_key = original_get_unique_tensor_key

        offloaded_count = sum(1 for _, modified, _, _, _ in offload_ctx.tracker.values() if modified)
        deduplicated_count = sum(1 for _, modified, _, _, _ in offload_ctx.tracker.values() if not modified)

        assert offloaded_count == 3
        assert deduplicated_count == 0

        loss.backward()

    @require_torch_accelerator
    def test_stale_tracker_state_is_cleared_between_forwards(self):
        # [FUNCTION DOC]
        # Checks that tracker entries from an unused graph branch do not accumulate across steps.
        # Setup: nested ModelWithUnusedBranch returning sums of a used and an unused Linear(8, 8); OffloadActivations without
        # pinned memory or streams, min_offload_size 1. Three iterations of zeroing grads, forward under the context,
        # backward on the used output only, recording the tracker length.
        # Assertion: all three recorded lengths are equal.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: accelerator allocations.
        """Test that tensors from unused graph branches don't accumulate across steps."""

        class ModelWithUnusedBranch(nn.Module):
            # [CLASS DOC]
            # HELPER: Module with two Linear(8, 8) layers, used and unused, whose outputs are both returned as sums.
            def __init__(self):
                # [FUNCTION DOC]
                # HELPER: Initializes nn.Module and creates the used and unused Linear(8, 8) layers.
                # Inputs: none. Outputs: none. Exceptions: none. Side effects: registers submodules.
                super().__init__()
                self.used = nn.Linear(8, 8)
                self.unused = nn.Linear(8, 8)

            def forward(self, x):
                # [FUNCTION DOC]
                # HELPER: Returns a tuple (sum of used(x), sum of unused(x)).
                # Inputs: x - tensor (N, 8). Outputs: tuple of two scalars. Exceptions: none. Side effects: none.
                return self.used(x).sum(), self.unused(x).sum()

        model = ModelWithUnusedBranch().to(torch_device)
        offload_ctx = OffloadActivations(use_pin_memory=False, use_streams=False, min_offload_size=1)
        offload_ctx.update_model_params(model)
        inp = torch.randn(4, 8, device=torch_device)

        tracker_counts = []
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            with offload_ctx:
                loss, _ = model(inp)
            loss.backward()
            tracker_counts.append(len(offload_ctx.tracker))

        assert tracker_counts == [tracker_counts[0]] * len(tracker_counts)

    @require_torch_accelerator
    def test_parameter_filtering(self):
        # [FUNCTION DOC]
        # Checks update_model_params records the untyped storage data pointers of all parameters of a two-layer Linear
        # model: param_storages is non-empty and equals the set of parameter storage pointers.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: accelerator allocations.
        """Test that model parameters are filtered during offloading"""
        model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 10)).to(torch_device)
        offload_ctx = OffloadActivations()
        offload_ctx.update_model_params(model)

        assert len(offload_ctx.param_storages) > 0, "Should have tracked parameter storages"

        param_ptrs = {p.data.untyped_storage().data_ptr() for p in model.parameters()}
        assert offload_ctx.param_storages == param_ptrs, "Tracked storages should match parameter storages"

    @require_torch_accelerator
    def test_tensor_deduplication_only_happens_in_streams_mode(self):
        # [FUNCTION DOC]
        # Checks that two saved views of the same 64x64 base tensor are both offloaded when use_streams is False (flags
        # true, true) but the second is deduplicated when use_streams is True (flags true, false).
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: accelerator allocations.
        """Two saved views of the same storage are deduplicated (one re-offload skipped) only when use_streams=True."""

        class SaveTensor(torch.autograd.Function):
            # [CLASS DOC]
            # HELPER: Custom autograd Function that saves its input for backward and returns the sum, so the saved tensor goes
            # through the offloading pack/unpack hooks.
            @staticmethod
            def forward(ctx, tensor):
                # [FUNCTION DOC]
                # HELPER: Saves tensor on ctx for backward and returns its sum.
                # Inputs: ctx, tensor. Outputs: scalar tensor. Exceptions: none. Side effects: stores tensor on ctx.
                ctx.save_for_backward(tensor)
                return tensor.sum()

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC]
                # HELPER: Retrieves the saved tensor and returns a gradient of ones shaped like it, scaled by grad_output.
                # Inputs: ctx, grad_output. Outputs: gradient tensor. Exceptions: none. Side effects: none.
                (tensor,) = ctx.saved_tensors
                return torch.ones_like(tensor) * grad_output

        def offload_two_views(use_streams: bool) -> list[bool]:
            # [FUNCTION DOC]
            # HELPER: Creates a random 64x64 base tensor requiring grad, takes a flat view and a transpose, saves both via
            # SaveTensor inside an OffloadActivations context (min_offload_size 1, given use_streams), collects the "modified"
            # flag of every tracker entry, then runs backward.
            # Inputs: use_streams - bool. Outputs: list of bool (true means offloaded, false means deduplicated).
            # Exceptions: none expected. Side effects: accelerator allocations, runs backward.
            base = torch.randn(64, 64, device=torch_device, requires_grad=True)
            view1 = base.view(-1)
            view2 = base.transpose(0, 1)
            offload_ctx = OffloadActivations(use_streams=use_streams, min_offload_size=1)
            with offload_ctx:
                loss = SaveTensor.apply(view1) + SaveTensor.apply(view2)
            modified_flags = [modified for _, modified, _, _, _ in offload_ctx.tracker.values()]
            loss.backward()
            return modified_flags

        assert offload_two_views(use_streams=False) == [True, True]
        assert offload_two_views(use_streams=True) == [True, False]
