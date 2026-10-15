# [MODULE DOC]
# Tests for the Triton/PyTorch kernels in trl.kernels: the fused selective log-softmax + entropy kernel, the chunked log-prob autograd function ChunkedLogProbFunction (and its fp32 addmm helper), and the chunked divergence autograd function ChunkedDivergenceFunction.
# Each kernel is compared against a plain PyTorch reference (values and gradients) across dtypes, temperatures, masks, logit scaling/softcapping, bias and frozen parameters, and under torch.compile.
# All test classes require a torch accelerator and Triton (gated with require_torch_accelerator and require_triton).
# Import-time side effects: kernel symbols are imported only when Triton is available.

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

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from transformers.testing_utils import torch_device
from transformers.utils import is_triton_available

from .testing_utils import require_torch_accelerator, require_triton


if is_triton_available():
    from trl.kernels import ChunkedLogProbFunction, selective_log_softmax_and_entropy
    from trl.kernels.chunked_divergence import ChunkedDivergenceFunction
    from trl.kernels.chunked_logprob import _addmm_fp32


def reference(logits, index, temperature, row_mask):
    # [FUNCTION DOC]
    # HELPER: Plain PyTorch reference for selective_log_softmax_and_entropy.
    # Behavior: cast logits to float32 and divide by temperature; take log-softmax over the last (vocab) dimension; gather the log-prob of the token in index for each position; compute entropy as minus the sum of p times log p over the vocab; if row_mask is given, zero both outputs where the mask is False.
    # Inputs: logits (batch, seq, vocab), index (batch, seq) token ids, temperature (float), row_mask (bool tensor of shape (batch, seq) or None).
    # Outputs: tuple (selected_logprobs, entropy), each float32 of shape (batch, seq).
    # Exceptions: none beyond tensor shape errors. Side effects: none.
    logits = logits.float() / temperature
    logprobs = logits.log_softmax(-1)
    selected_logprobs = logprobs.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    entropy = -(logprobs.exp() * logprobs).sum(-1)
    if row_mask is not None:
        selected_logprobs = selected_logprobs.masked_fill(~row_mask, 0.0)
        entropy = entropy.masked_fill(~row_mask, 0.0)
    return selected_logprobs, entropy


@require_torch_accelerator
@require_triton
class TestLogProbEntropy:
    # [CLASS DOC]
    # Tests selective_log_softmax_and_entropy (fused Triton kernel) against the reference helper, for forward values, gradients and torch.compile compatibility.
    # No state; plain pytest class. Requires an accelerator and Triton.
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        ("logprob_weight", "entropy_weight", "use_mask"),
        [(2, None, True), (None, 0.5, True), (2, 0.5, True), (2, 0.5, False)],
    )
    def test_forward_and_backward(self, dtype, logprob_weight, entropy_weight, use_mask):
        # [FUNCTION DOC]
        # Tests forward values and backward gradients of selective_log_softmax_and_entropy against the reference.
        # Setup: vocab size 2053 (two full Triton blocks plus a partial one); random logits of shape (2, 5, vocab) in the given dtype with the first 1024 columns shifted up by 20 and columns 1024-2047 shifted down by 20 to stress online normalization; logits and indices sliced on the sequence dimension (positions 1-3) to produce non-contiguous views; one index forced into the partial last block; temperature 0.7; an optional sliced boolean row mask.
        # Then overwrites one logprob output element with zero (to verify the outputs are not aliased with saved backward state), builds a loss from weighted logprob and/or entropy sums depending on logprob_weight / entropy_weight, and backpropagates. The same is done on a detached clone with the reference.
        # Assertions: logprobs and entropy close to reference (tolerance 1e-5), gradient of the base logits close to reference (tolerance 1e-3).
        # Inputs: dtype (float32/float16/bfloat16), logprob_weight, entropy_weight (each a number or None), use_mask (bool) from parametrize. Outputs: none. Exceptions: AssertionError on mismatch. Side effects: none.
        # Cross two full Triton blocks and leave a partial final block.
        vocab_size = 2053
        base_logits = torch.randn(2, 5, vocab_size, device=torch_device, dtype=dtype)
        base_logits[..., :1024].add_(20)
        # Exercise online normalization across blocks with sharply different maxima.
        base_logits[..., 1024:2048].sub_(20)
        base_logits.requires_grad_()
        # TRL slices sequence logits and token IDs without making them contiguous.
        logits = base_logits[:, 1:4]
        index = torch.randint(vocab_size, (2, 5), device=torch_device)[:, 1:4]
        index[0, 0] = vocab_size - 1  # select a target from the partial block
        temperature = 0.7
        # Slice the mask too, matching the non-contiguous views passed by TRL.
        sliced_mask = torch.tensor([[1, 1, 0, 1, 1], [1, 0, 1, 1, 1]], device=torch_device, dtype=torch.bool)[:, 1:4]
        row_mask = sliced_mask if use_mask else None

        logprobs, entropy = selective_log_softmax_and_entropy(
            logits, index, temperature=temperature, row_mask=row_mask
        )
        # Mutating exposed outputs must not corrupt the private statistics saved for backward.
        logprobs[0, 2] = 0.0
        loss = 0.0
        if logprob_weight is not None:
            loss = loss + logprob_weight * logprobs.sum()
        if entropy_weight is not None:
            loss = loss + entropy_weight * entropy.sum()
        loss.backward()
        actual_grad = base_logits.grad.clone()

        reference_logits = base_logits.detach().clone().requires_grad_()
        reference_logprobs, reference_entropy = reference(reference_logits[:, 1:4], index, temperature, row_mask)
        reference_logprobs[0, 2] = 0.0
        reference_loss = 0.0
        if logprob_weight is not None:
            reference_loss = reference_loss + logprob_weight * reference_logprobs.sum()
        if entropy_weight is not None:
            reference_loss = reference_loss + entropy_weight * reference_entropy.sum()
        reference_loss.backward()

        torch.testing.assert_close(logprobs, reference_logprobs, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(entropy, reference_entropy, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(actual_grad, reference_logits.grad, rtol=1e-3, atol=1e-3)

    def test_torch_compile_fullgraph(self):
        # [FUNCTION DOC]
        # Tests that selective_log_softmax_and_entropy works under torch.compile with fullgraph enabled.
        # Setup: bfloat16 logits (2, 3, 257) requiring grad, random indices, a row mask with some False entries; a local loss sums logprobs plus 0.1 times entropy at temperature 0.8; compiled with fullgraph and backpropagated.
        # Assertions: logits gradient exists, is all finite, and is exactly zero for masked-out rows.
        # Inputs: none. Outputs: none. Exceptions: AssertionError; torch compile errors if graph breaks. Side effects: none.
        logits = torch.randn(2, 3, 257, device=torch_device, dtype=torch.bfloat16, requires_grad=True)
        index = torch.randint(257, (2, 3), device=torch_device)
        row_mask = torch.tensor([[1, 0, 1], [0, 1, 1]], device=torch_device, dtype=torch.bool)

        def loss(logits, index, row_mask):
            # [FUNCTION DOC]
            # HELPER: Loss closure compiled in the test.
            # Behavior: computes logprobs and entropy with selective_log_softmax_and_entropy at temperature 0.8 using the given row mask, returns the sum of logprobs plus 0.1 times entropy.
            # Inputs: logits, index, row_mask tensors. Outputs: scalar tensor. Exceptions: none. Side effects: none.
            logprobs, entropy = selective_log_softmax_and_entropy(logits, index, temperature=0.8, row_mask=row_mask)
            return (logprobs + 0.1 * entropy).sum()

        torch.compile(loss, fullgraph=True)(logits, index, row_mask).backward()

        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert torch.count_nonzero(logits.grad[~row_mask]) == 0


@require_torch_accelerator
@require_triton
class TestChunkedLogProbFunction:
    # [CLASS DOC]
    # Tests ChunkedLogProbFunction, the autograd function that computes per-token log-probs (plus optional entropy, log-sum of squared probs, mean logits and is-top1 flags) from hidden states and an LM head weight without materializing the full logits.
    # Class attributes: N=64 tokens, H=32 hidden size, V=128 vocab, CHUNK_SIZE=32 vocab chunk size.
    # Uses the helper _reference_logprobs_and_entropy as ground truth for values and gradients. Also tests the _addmm_fp32 helper.
    N, H, V = 64, 32, 128
    CHUNK_SIZE = 32

    def test_torch_compile(self):
        # [FUNCTION DOC]
        # Tests that ChunkedLogProbFunction gives the same hidden-state gradient eagerly and under torch.compile (Inductor passes float arguments as fp64).
        # Setup: seed 42; random hidden (requires grad), weight, labels; a local loss applies the function with temperature 0.7, softcapping 30.0 and logit scale 0.5 and sums logprobs and entropy.
        # Assertions: compiled gradient equals the eager gradient.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: none.
        # Inductor passes the float arguments (temperature, logit scale, softcapping) as fp64
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, device=torch_device, requires_grad=True)
        weight = torch.randn(self.V, self.H, device=torch_device)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        def loss(hidden):
            # [FUNCTION DOC]
            # HELPER: Loss closure used for eager and compiled runs.
            # Behavior: applies ChunkedLogProbFunction to the given hidden states with the captured weight and labels, no bias, temperature 0.7, chunk size CHUNK_SIZE, softcapping 30.0, logit scale 0.5; returns sum of logprobs plus sum of entropy.
            # Inputs: hidden (N, H). Outputs: scalar tensor. Exceptions: none. Side effects: none.
            logprobs, entropy, *_ = ChunkedLogProbFunction.apply(
                hidden, weight, None, labels, 0.7, self.CHUNK_SIZE, 30.0, 0.5
            )
            return logprobs.sum() + entropy.sum()

        loss(hidden).backward()
        expected_grad, hidden.grad = hidden.grad, None
        torch.compile(loss)(hidden).backward()

        torch.testing.assert_close(hidden.grad, expected_grad)

    def _reference_logprobs_and_entropy(
        self, hidden, weight, labels, temperature, bias=None, logit_scale=1.0, final_logit_softcapping=None
    ):
        # [FUNCTION DOC]
        # HELPER: Plain PyTorch reference for ChunkedLogProbFunction outputs.
        # Behavior: logits are hidden times transposed weight, plus bias if given; cast to float32 and multiply by logit_scale; if final_logit_softcapping is set, apply tanh softcapping (divide by cap, tanh, multiply by cap); divide by temperature. Then log-softmax over vocab, gather log-prob at labels, and compute entropy as minus the sum of p times log p.
        # Inputs: hidden (N, H), weight (V, H), labels (N,), temperature (float), bias (V,) optional, logit_scale (float, defaults to 1.0), final_logit_softcapping (float, optional).
        # Outputs: tuple (logprobs, entropy), float32 tensors of shape (N,).
        # Exceptions: none. Side effects: none.
        logits = hidden @ weight.t()
        if bias is not None:
            logits = logits + bias
        logits = logits.to(torch.float32) * logit_scale
        if final_logit_softcapping is not None:
            logits = torch.tanh(logits / final_logit_softcapping) * final_logit_softcapping
        logits = logits / temperature  # [N, V]
        log_p = F.log_softmax(logits, dim=-1)
        logprobs = log_p.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        p = torch.softmax(logits, dim=-1)
        entropy = -(p * log_p).sum(dim=-1)
        return logprobs, entropy

    # CPU tensors stand in for an accelerator without Triton (e.g. Trainium): TRL does not target CPU training
    @pytest.mark.parametrize("device", [torch_device, "cpu"])
    @pytest.mark.parametrize("temperature", [1.0, 0.7])
    def test_forward(self, device, temperature):
        # [FUNCTION DOC]
        # Tests forward values of ChunkedLogProbFunction and that it processes tokens in row chunks of at most TOKEN_CHUNK_SIZE.
        # Setup: on the accelerator or on CPU (CPU stands in for an accelerator without Triton); patches trl.kernels.chunked_logprob.TOKEN_CHUNK_SIZE to 17 and wraps torch.mm in that module to record the row count of each chunk.
        # Assertions: logprobs and entropy close to the reference (1e-5); maximum chunk size is at most 17 and the last chunk has 13 rows (64 = 3 times 17 plus 13).
        # Inputs: device, temperature from parametrize. Outputs: none. Exceptions: AssertionError. Side effects: patches are restored on exit.
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, device=device)
        weight = torch.randn(self.V, self.H, device=device)
        labels = torch.randint(0, self.V, (self.N,), device=device)
        chunk_rows = []
        torch_mm = torch.mm

        def record_chunk(input, mat2, *, out=None):
            # [FUNCTION DOC]
            # HELPER: Side effect for the patched torch.mm that records chunk sizes.
            # Behavior: appends the row count of input to the enclosing chunk_rows list and delegates to the original torch.mm (with out if given).
            # Inputs: input, mat2 tensors, keyword-only out. Outputs: the matrix product. Exceptions: none. Side effects: mutates chunk_rows.
            chunk_rows.append(input.size(0))
            return torch_mm(input, mat2, out=out)

        with (
            patch("trl.kernels.chunked_logprob.TOKEN_CHUNK_SIZE", 17),
            patch("trl.kernels.chunked_logprob.torch.mm", side_effect=record_chunk),
        ):
            logprobs_chunked, entropy_chunked, *_ = ChunkedLogProbFunction.apply(
                hidden, weight, None, labels, temperature, self.CHUNK_SIZE
            )
        logprobs_ref, entropy_ref = self._reference_logprobs_and_entropy(hidden, weight, labels, temperature)

        torch.testing.assert_close(logprobs_chunked, logprobs_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(entropy_chunked, entropy_ref, atol=1e-5, rtol=1e-5)
        assert max(chunk_rows) <= 17
        assert chunk_rows[-1] == 13

    def test_log_sum_sq_probs(self):
        # [FUNCTION DOC]
        # Tests the third output (log of the sum of squared probabilities) of ChunkedLogProbFunction.
        # Setup: random hidden, weight, labels; temperature 0.7.
        # Assertions: output is close to logsumexp of twice the logits minus twice logsumexp of the logits, with logits being hidden times transposed weight divided by temperature.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: none.
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, device=torch_device)
        weight = torch.randn(self.V, self.H, device=torch_device)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        _, _, log_sum_sq_probs, _, _ = ChunkedLogProbFunction.apply(hidden, weight, None, labels, 0.7, self.CHUNK_SIZE)
        logits = (hidden @ weight.t()) / 0.7

        expected = torch.logsumexp(2 * logits, dim=-1) - 2 * torch.logsumexp(logits, dim=-1)
        torch.testing.assert_close(log_sum_sq_probs, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("outputs", [(), ("entropy",), ("is_top1", "mean_logits")])
    def test_outputs_subset(self, outputs):
        # [FUNCTION DOC]
        # Tests the outputs argument of ChunkedLogProbFunction: outputs not requested are None, requested ones (and log_probs, always returned) match a full run, and gradients are unchanged.
        # Setup: run the function with all outputs and backprop the logprob sum, saving hidden/weight gradients; reset grads; run again with only the requested outputs (softcapping None, logit scale 1.0) and backprop.
        # Assertions: for each of the five outputs (log_probs, entropy, log_sum_sq_probs, mean_logits, is_top1), equal to the full run if requested or log_probs, otherwise None; hidden and weight gradients equal those of the full run.
        # Inputs: outputs (tuple of names) from parametrize. Outputs: none. Exceptions: AssertionError. Side effects: none.
        # Outputs that are not requested are `None`, and the others, with the gradients, are unchanged
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, device=torch_device, requires_grad=True)
        weight = torch.randn(self.V, self.H, device=torch_device, requires_grad=True)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        full = ChunkedLogProbFunction.apply(hidden, weight, None, labels, 0.7, self.CHUNK_SIZE)
        full[0].sum().backward()
        grads = hidden.grad.clone(), weight.grad.clone()
        hidden.grad = weight.grad = None
        subset = ChunkedLogProbFunction.apply(hidden, weight, None, labels, 0.7, self.CHUNK_SIZE, None, 1.0, outputs)
        subset[0].sum().backward()

        names = ("log_probs", "entropy", "log_sum_sq_probs", "mean_logits", "is_top1")
        for name, x, y in zip(names, full, subset, strict=True):
            if name == "log_probs" or name in outputs:
                torch.testing.assert_close(y, x)
            else:
                assert y is None
        torch.testing.assert_close(hidden.grad, grads[0])
        torch.testing.assert_close(weight.grad, grads[1])

    def test_mean_logits(self):
        # [FUNCTION DOC]
        # Tests the fourth output (mean logits) of ChunkedLogProbFunction.
        # Assertions: equals the mean over vocab of hidden times transposed weight, divided by temperature 0.7, within 1e-5.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: none.
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, device=torch_device)
        weight = torch.randn(self.V, self.H, device=torch_device)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        _, _, _, mean_logits, _ = ChunkedLogProbFunction.apply(hidden, weight, None, labels, 0.7, self.CHUNK_SIZE)

        torch.testing.assert_close(mean_logits, (hidden @ weight.t()).mean(-1) / 0.7, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize(
        ("temperature", "logit_scale", "final_logit_softcapping"),
        [(1.0, 1.0, None), (0.7, 0.5, None), (0.7, 1.0, 3.0)],
    )
    @pytest.mark.parametrize("entropy_weight", [0.0, 0.5])
    def test_backward_frozen_head(self, temperature, logit_scale, final_logit_softcapping, entropy_weight):
        # [FUNCTION DOC]
        # Tests hidden-state gradients when the LM head is frozen (weight does not require grad, as with a PEFT adapter).
        # Setup: function applied with given temperature, softcapping and logit scale; loss is twice the logprob sum plus entropy_weight times the entropy sum; same loss computed with the reference.
        # Assertions: hidden gradients close to reference (1e-4).
        # Inputs: temperature, logit_scale, final_logit_softcapping, entropy_weight from parametrize. Outputs: none. Exceptions: AssertionError. Side effects: none.
        # Only the hidden states need a gradient, as with a PEFT adapter
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, device=torch_device, requires_grad=True)
        weight = torch.randn(self.V, self.H, device=torch_device)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        logprobs, entropy, *_ = ChunkedLogProbFunction.apply(
            hidden, weight, None, labels, temperature, self.CHUNK_SIZE, final_logit_softcapping, logit_scale
        )
        (2.0 * logprobs + entropy_weight * entropy).sum().backward()
        grad = hidden.grad.clone()

        hidden.grad = None
        logprobs_ref, entropy_ref = self._reference_logprobs_and_entropy(
            hidden,
            weight,
            labels,
            temperature,
            logit_scale=logit_scale,
            final_logit_softcapping=final_logit_softcapping,
        )
        (2.0 * logprobs_ref + entropy_weight * entropy_ref).sum().backward()

        torch.testing.assert_close(grad, hidden.grad, atol=1e-4, rtol=1e-4)

    def test_is_top1_matches_argmax_with_ties(self):
        # [FUNCTION DOC]
        # Tests that the is_top1 output breaks exact ties the same way argmax does (first index wins).
        # Setup: the second half of the weight rows duplicates the first half, producing exact logit ties; labels alternate between the argmax and its duplicate half a vocab later (clamped to V-1).
        # Assertions: is_top1 equals the boolean comparison of argmax with labels.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: none.
        # Duplicated head rows make exact ties; `argmax` picks the first one, and so must `is_top1`
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, device=torch_device)
        weight = torch.randn(self.V, self.H, device=torch_device)
        weight[self.V // 2 :] = weight[: self.V - self.V // 2]
        argmax = (hidden @ weight.t()).argmax(-1)
        # Half the labels are the argmax, half its duplicate further down the vocabulary
        labels = torch.where(torch.arange(self.N, device=torch_device) % 2 == 0, argmax, argmax + self.V // 2)
        labels = labels.clamp(max=self.V - 1)

        *_, is_top1 = ChunkedLogProbFunction.apply(hidden, weight, None, labels, 1.0, self.CHUNK_SIZE)

        torch.testing.assert_close(is_top1, argmax == labels)

    # CPU tensors stand in for an accelerator without Triton (e.g. Trainium): TRL does not target CPU training
    @pytest.mark.parametrize("device", [torch_device, "cpu"])
    @pytest.mark.parametrize(
        ("logit_scale", "final_logit_softcapping"),
        [
            (0.5, None),  # models that scale but don't softcap, e.g. MPT
            (1.0, 30.0),  # models that softcap but don't scale, e.g. Gemma 2
            (0.5, 30.0),  # both, applied in that order
        ],
    )
    def test_logit_scale_and_softcapping(self, device, logit_scale, final_logit_softcapping):
        # [FUNCTION DOC]
        # Tests logit scaling and final logit softcapping in ChunkedLogProbFunction (scale applied before softcapping).
        # Setup: on accelerator or CPU; temperature 0.7; combinations of scale only, softcap only, and both.
        # Assertions: logprobs and entropy close to the reference with the same scale/softcap (1e-5).
        # Inputs: device, logit_scale, final_logit_softcapping from parametrize. Outputs: none. Exceptions: AssertionError. Side effects: none.
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, device=device)
        weight = torch.randn(self.V, self.H, device=device)
        labels = torch.randint(0, self.V, (self.N,), device=device)

        logprobs, entropy, *_ = ChunkedLogProbFunction.apply(
            hidden, weight, None, labels, 0.7, self.CHUNK_SIZE, final_logit_softcapping, logit_scale
        )
        logprobs_ref, entropy_ref = self._reference_logprobs_and_entropy(
            hidden, weight, labels, 0.7, logit_scale=logit_scale, final_logit_softcapping=final_logit_softcapping
        )

        torch.testing.assert_close(logprobs, logprobs_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(entropy, entropy_ref, atol=1e-5, rtol=1e-5)

    # CPU tensors stand in for an accelerator without Triton (e.g. Trainium): TRL does not target CPU training
    @pytest.mark.parametrize("device", [torch_device, "cpu"])
    @pytest.mark.parametrize("temperature", [1.0, 0.7])
    def test_backward(self, device, temperature):
        # [FUNCTION DOC]
        # Tests gradients of hidden and weight through the logprob output of ChunkedLogProbFunction in float32.
        # Setup: on accelerator or CPU; backprop the logprob sum through the chunked function, save grads, reset, backprop the reference logprob sum.
        # Assertions: hidden and weight gradients close to reference (1e-5).
        # Inputs: device, temperature from parametrize. Outputs: none. Exceptions: AssertionError. Side effects: none.
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, requires_grad=True, device=device)
        weight = torch.randn(self.V, self.H, requires_grad=True, device=device)
        labels = torch.randint(0, self.V, (self.N,), device=device)

        # Chunked backward
        logprobs_chunked, *_ = ChunkedLogProbFunction.apply(hidden, weight, None, labels, temperature, self.CHUNK_SIZE)
        logprobs_chunked.sum().backward()
        grad_hidden_chunked = hidden.grad.clone()
        grad_weight_chunked = weight.grad.clone()

        hidden.grad = None
        weight.grad = None

        # Reference backward
        logprobs_ref, _ = self._reference_logprobs_and_entropy(hidden, weight, labels, temperature)
        logprobs_ref.sum().backward()

        torch.testing.assert_close(grad_hidden_chunked, hidden.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(grad_weight_chunked, weight.grad, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("temperature", [1.0, 0.7])
    def test_backward_bfloat16(self, temperature):
        # [FUNCTION DOC]
        # Same as test_backward but with bfloat16 hidden and weight on the accelerator, with looser tolerance (1e-2).
        # Inputs: temperature from parametrize. Outputs: none. Exceptions: AssertionError. Side effects: none.
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, dtype=torch.bfloat16, requires_grad=True, device=torch_device)
        weight = torch.randn(self.V, self.H, dtype=torch.bfloat16, requires_grad=True, device=torch_device)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        # Chunked backward
        logprobs_chunked, *_ = ChunkedLogProbFunction.apply(hidden, weight, None, labels, temperature, self.CHUNK_SIZE)
        logprobs_chunked.sum().backward()
        grad_hidden_chunked = hidden.grad.clone()
        grad_weight_chunked = weight.grad.clone()

        hidden.grad = None
        weight.grad = None

        # Reference backward
        logprobs_ref, _ = self._reference_logprobs_and_entropy(hidden, weight, labels, temperature)
        logprobs_ref.sum().backward()

        torch.testing.assert_close(grad_hidden_chunked, hidden.grad, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(grad_weight_chunked, weight.grad, atol=1e-2, rtol=1e-2)

    def test_backward_bfloat16_hidden_float32_weight(self):
        # [FUNCTION DOC]
        # Tests mixed dtypes: bfloat16 hidden states with a float32 weight.
        # Setup: chunked function at temperature 1.0; reference computed with the weight cast to the hidden dtype.
        # Assertions: logprobs, hidden gradient and weight gradient close to reference (1e-2).
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: none.
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, dtype=torch.bfloat16, requires_grad=True, device=torch_device)
        weight = torch.randn(self.V, self.H, dtype=torch.float32, requires_grad=True, device=torch_device)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        logprobs, *_ = ChunkedLogProbFunction.apply(hidden, weight, None, labels, 1.0, self.CHUNK_SIZE)
        logprobs.sum().backward()
        grad_hidden = hidden.grad.clone()
        grad_weight = weight.grad.clone()

        hidden.grad = None
        weight.grad = None
        reference, _ = self._reference_logprobs_and_entropy(hidden, weight.to(hidden.dtype), labels, 1.0)
        reference.sum().backward()

        torch.testing.assert_close(logprobs, reference, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(grad_hidden, hidden.grad, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(grad_weight, weight.grad, atol=1e-2, rtol=1e-2)

    def test_backward_uses_autocast_hidden_for_weight_gradient(self):
        # [FUNCTION DOC]
        # Tests that under autocast the weight gradient is computed from the autocast (bfloat16) hidden states, not the original float32 ones.
        # Setup: float32 hidden values exactly representable in bfloat16, and a perturbed copy (plus 1e-4) that rounds to the same bfloat16 values (asserted); two identical weights; the function is run on both under bfloat16 autocast and backpropagated.
        # Assertions: the two logprob outputs are exactly equal and the two weight gradients are exactly equal.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: none.
        torch.manual_seed(42)
        hidden = (
            torch.linspace(1.0, 2.0, self.N * self.H, device=torch_device)
            .reshape(self.N, self.H)
            .to(torch.bfloat16)
            .float()
        )
        perturbed_hidden = hidden + 1e-4
        assert torch.equal(hidden.to(torch.bfloat16), perturbed_hidden.to(torch.bfloat16))
        hidden.requires_grad_()
        perturbed_hidden.requires_grad_()
        weight = torch.randn(self.V, self.H, requires_grad=True, device=torch_device)
        perturbed_weight = weight.detach().clone().requires_grad_()
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        # Autocast gives both inputs identical projected values. Their weight gradients must therefore also match;
        # using the original fp32 hidden states in backward would make the gradients depend on the discarded bits.
        with torch.autocast(torch_device, dtype=torch.bfloat16):
            logprobs, *_ = ChunkedLogProbFunction.apply(hidden, weight, None, labels, 1.0, self.CHUNK_SIZE)
            perturbed_logprobs, *_ = ChunkedLogProbFunction.apply(
                perturbed_hidden, perturbed_weight, None, labels, 1.0, self.CHUNK_SIZE
            )
        logprobs.sum().backward()
        perturbed_logprobs.sum().backward()

        torch.testing.assert_close(logprobs, perturbed_logprobs, rtol=0, atol=0)
        torch.testing.assert_close(weight.grad, perturbed_weight.grad, rtol=0, atol=0)

    @pytest.mark.parametrize("temperature", [1.0, 0.7])
    def test_backward_entropy(self, temperature):
        # [FUNCTION DOC]
        # Tests gradients through the entropy output alone of ChunkedLogProbFunction.
        # Assertions: hidden and weight gradients close to reference entropy gradients (1e-4).
        # Inputs: temperature from parametrize. Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Backprop through the `entropy` output alone (as opposed to `logprobs`, covered above)."""
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, requires_grad=True, device=torch_device)
        weight = torch.randn(self.V, self.H, requires_grad=True, device=torch_device)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        # Chunked backward
        _, entropy_chunked, *_ = ChunkedLogProbFunction.apply(
            hidden, weight, None, labels, temperature, self.CHUNK_SIZE
        )
        entropy_chunked.sum().backward()
        grad_hidden_chunked = hidden.grad.clone()
        grad_weight_chunked = weight.grad.clone()

        hidden.grad = None
        weight.grad = None

        # Reference backward
        _, entropy_ref = self._reference_logprobs_and_entropy(hidden, weight, labels, temperature)
        entropy_ref.sum().backward()

        torch.testing.assert_close(grad_hidden_chunked, hidden.grad, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(grad_weight_chunked, weight.grad, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("temperature", [1.0, 0.7])
    def test_backward_combined(self, temperature):
        # [FUNCTION DOC]
        # Tests that gradients from logprobs and entropy accumulate (not overwrite) when both are used.
        # Setup: TOKEN_CHUNK_SIZE patched to 17 so multiple row chunks are used; loss is twice logprob sum plus half entropy sum; reference computed the same way.
        # Assertions: hidden and weight gradients close to reference (1e-4).
        # Inputs: temperature from parametrize. Outputs: none. Exceptions: AssertionError. Side effects: patch restored on exit.
        """Backprop through `logprobs` and `entropy` together, to catch the gradients overwriting each other
        instead of accumulating."""
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, requires_grad=True, device=torch_device)
        weight = torch.randn(self.V, self.H, requires_grad=True, device=torch_device)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        # Chunked backward
        with patch("trl.kernels.chunked_logprob.TOKEN_CHUNK_SIZE", 17):
            logprobs_chunked, entropy_chunked, *_ = ChunkedLogProbFunction.apply(
                hidden, weight, None, labels, temperature, self.CHUNK_SIZE
            )
            (2.0 * logprobs_chunked + 0.5 * entropy_chunked).sum().backward()
        grad_hidden_chunked = hidden.grad.clone()
        grad_weight_chunked = weight.grad.clone()

        hidden.grad = None
        weight.grad = None

        # Reference backward
        logprobs_ref, entropy_ref = self._reference_logprobs_and_entropy(hidden, weight, labels, temperature)
        (2.0 * logprobs_ref + 0.5 * entropy_ref).sum().backward()

        torch.testing.assert_close(grad_hidden_chunked, hidden.grad, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(grad_weight_chunked, weight.grad, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_bias(self, dtype):
        # [FUNCTION DOC]
        # Tests ChunkedLogProbFunction with an LM head bias.
        # Setup: hidden, weight and bias all requiring grad in the given dtype; temperature 0.7; loss is twice logprob sum plus half entropy sum, compared to the reference with bias.
        # Assertions: logprobs, entropy and the gradients of hidden, weight and bias close to reference (atol 5e-2 / rtol 2e-2 for bfloat16, 1e-4 otherwise).
        # Inputs: dtype from parametrize. Outputs: none. Exceptions: AssertionError. Side effects: none.
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, dtype=dtype, requires_grad=True, device=torch_device)
        weight = torch.randn(self.V, self.H, dtype=dtype, requires_grad=True, device=torch_device)
        bias = torch.randn(self.V, dtype=dtype, requires_grad=True, device=torch_device)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        logprobs_chunked, entropy_chunked, *_ = ChunkedLogProbFunction.apply(
            hidden, weight, bias, labels, 0.7, self.CHUNK_SIZE
        )
        (2.0 * logprobs_chunked + 0.5 * entropy_chunked).sum().backward()
        chunked_grads = hidden.grad.clone(), weight.grad.clone(), bias.grad.clone()

        hidden.grad = weight.grad = bias.grad = None
        logprobs_ref, entropy_ref = self._reference_logprobs_and_entropy(hidden, weight, labels, 0.7, bias)
        (2.0 * logprobs_ref + 0.5 * entropy_ref).sum().backward()

        atol, rtol = (5e-2, 2e-2) if dtype == torch.bfloat16 else (1e-4, 1e-4)
        torch.testing.assert_close(logprobs_chunked, logprobs_ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(entropy_chunked, entropy_ref, atol=atol, rtol=rtol)
        for actual, expected in zip(chunked_grads, (hidden.grad, weight.grad, bias.grad), strict=True):
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def test_backward_skips_frozen_parameter_gradients(self):
        # [FUNCTION DOC]
        # Tests that backward does not allocate full-size gradient buffers for a frozen weight and bias.
        # Setup: only hidden requires grad; torch.zeros is wrapped with a spy during backward.
        # Assertions: no torch.zeros call used the weight shape or the bias shape; hidden gradient exists.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: torch.zeros patch restored on exit.
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, requires_grad=True, device=torch_device)
        weight = torch.randn(self.V, self.H, device=torch_device)
        bias = torch.randn(self.V, device=torch_device)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        logprobs, *_ = ChunkedLogProbFunction.apply(hidden, weight, bias, labels, 1.0, self.CHUNK_SIZE)
        with patch.object(torch, "zeros", wraps=torch.zeros) as mock_zeros:
            logprobs.sum().backward()

        # A frozen LM head should not allocate full-vocabulary gradient buffers. These shapes are distinct from the
        # per-token accumulators and the hidden-state gradient allocated by the same backward pass.
        allocated_shapes = [call.args[0] for call in mock_zeros.call_args_list]
        assert weight.shape not in allocated_shapes
        assert bias.shape not in allocated_shapes
        assert hidden.grad is not None

    @pytest.mark.parametrize("requires_grad", [(True, False, False), (False, True, False), (False, False, True)])
    def test_backward_with_partially_frozen_inputs(self, requires_grad):
        # [FUNCTION DOC]
        # Tests backward when only one of hidden, weight, bias requires grad.
        # Setup: run the chunked function and the reference on detached clones with the same requires_grad flags; backprop logprob sums.
        # Assertions: for each input whose reference gradient exists, the chunked gradient is close to it (1e-5).
        # Inputs: requires_grad (3-tuple of bools) from parametrize. Outputs: none. Exceptions: AssertionError. Side effects: none.
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, requires_grad=requires_grad[0], device=torch_device)
        weight = torch.randn(self.V, self.H, requires_grad=requires_grad[1], device=torch_device)
        bias = torch.randn(self.V, requires_grad=requires_grad[2], device=torch_device)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        logprobs, *_ = ChunkedLogProbFunction.apply(hidden, weight, bias, labels, 1.0, self.CHUNK_SIZE)
        logprobs.sum().backward()
        chunked_grads = hidden.grad, weight.grad, bias.grad

        hidden_ref = hidden.detach().clone().requires_grad_(requires_grad[0])
        weight_ref = weight.detach().clone().requires_grad_(requires_grad[1])
        bias_ref = bias.detach().clone().requires_grad_(requires_grad[2])
        logprobs_ref, _ = self._reference_logprobs_and_entropy(hidden_ref, weight_ref, labels, 1.0, bias_ref)
        logprobs_ref.sum().backward()

        for actual, expected in zip(chunked_grads, (hidden_ref.grad, weight_ref.grad, bias_ref.grad), strict=True):
            if expected is not None:
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_addmm_fp32(self, dtype):
        # [FUNCTION DOC]
        # Tests the _addmm_fp32 helper, which adds the product of two matrices into a float32 accumulator in place, accumulating in fp32.
        # Setup: a (64, 256) and b (256, 32) in the given dtype, a float32 accumulator (64, 32); expected value computed in float64.
        # Assertions: accumulator after the call is close to the expected sum (atol 1e-3, rtol 1e-4).
        # Inputs: dtype from parametrize. Outputs: none. Exceptions: AssertionError. Side effects: none.
        # Accumulates the product in fp32, without rounding it to `dtype` first
        torch.manual_seed(42)
        a = torch.randn(64, 256, device=torch_device, dtype=dtype)
        b = torch.randn(256, 32, device=torch_device, dtype=dtype)
        acc = torch.randn(64, 32, device=torch_device)
        expected = acc.double() + a.double() @ b.double()

        _addmm_fp32(acc, a, b)

        torch.testing.assert_close(acc.double(), expected, atol=1e-3, rtol=1e-4)


@require_torch_accelerator
@require_triton
class TestChunkedDivergenceFunction:
    # [CLASS DOC]
    # Tests ChunkedDivergenceFunction, the autograd function computing a chunked student/teacher divergence from hidden states and LM heads.
    # No state; contains a torch.compile consistency test.
    @pytest.mark.parametrize("beta", [0.0, 0.7, 1.0])
    def test_torch_compile(self, beta):
        # [FUNCTION DOC]
        # Tests that ChunkedDivergenceFunction gives the same student hidden gradient eagerly and under torch.compile (Inductor passes float arguments as fp64).
        # Setup: student hidden (64, 32) requiring grad with weight (128, 32); teacher hidden (64, 48) with weight (128, 48); no biases.
        # Assertions: compiled gradient equals the eager gradient.
        # Inputs: beta (0.0, 0.7, 1.0) from parametrize. Outputs: none. Exceptions: AssertionError. Side effects: none.
        # Inductor passes the float arguments (beta, temperature, logit scales, softcapping) as fp64
        torch.manual_seed(42)
        student_hidden = torch.randn(64, 32, device=torch_device, requires_grad=True)
        student_weight = torch.randn(128, 32, device=torch_device)
        teacher_hidden = torch.randn(64, 48, device=torch_device)
        teacher_weight = torch.randn(128, 48, device=torch_device)

        def loss(student_hidden):
            # [FUNCTION DOC]
            # HELPER: Loss closure used for eager and compiled runs.
            # Behavior: applies ChunkedDivergenceFunction to student hidden/weight and teacher hidden/weight (no biases) with the captured beta and the positional float arguments 0.7, 0.5, 30.0, and returns the sum of the divergence output.
            # Inputs: student_hidden (64, 32). Outputs: scalar tensor. Exceptions: none. Side effects: none.
            divergence, _ = ChunkedDivergenceFunction.apply(
                student_hidden, student_weight, None, teacher_hidden, teacher_weight, None, beta, 0.7, 0.5, 30.0
            )
            return divergence.sum()

        loss(student_hidden).backward()
        expected_grad, student_hidden.grad = student_hidden.grad, None
        torch.compile(loss)(student_hidden).backward()

        torch.testing.assert_close(student_hidden.grad, expected_grad)
