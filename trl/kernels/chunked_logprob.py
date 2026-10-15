# [MODULE DOC]
# Module trl.kernels.chunked_logprob: memory-efficient per-token log-probabilities of target tokens (and
# optional entropy, log of sum of squared probs, mean logit, is-top1 flag) computed from hidden states and the
# LM head in tiles of TOKEN_CHUNK_SIZE tokens by chunk_size vocab entries, never materializing [N, V] logits.
# Main contents: shared primitives also imported by chunked_divergence (_jit, _use_triton, _addmm_fp32,
# _transform, _torch_transform, _BLOCK_SIZE, _MM_OUT_DTYPE), Triton kernels _forward_kernel/_backward_kernel
# with torch fallbacks _forward_tile/_backward_tile, and ChunkedLogProbFunction (torch.autograd.Function).
# Import-time side effects: imports triton when available; sets TOKEN_CHUNK_SIZE 4096, OPTIONAL_OUTPUTS,
# _BLOCK_SIZE 1024, _MM_OUT_DTYPE (torch 2.8.0 or later) and caches _TRITON_AVAILABLE.

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

# Python < 3.14 evaluates annotations at definition time, and `tl` (for `tl.constexpr`) only exists with Triton
from __future__ import annotations

import math

import torch
from packaging.version import Version
from transformers.utils import is_triton_available


if is_triton_available():
    import triton
    import triton.language as tl


# Tokens per projection tile; the caller sets the vocabulary width with `chunk_size`
TOKEN_CHUNK_SIZE = 4096
# Per-token outputs besides the log-probabilities, each computed only when requested
OPTIONAL_OUTPUTS = ("entropy", "log_sum_sq_probs", "mean_logits", "is_top1")
_BLOCK_SIZE = 1024
# `torch.addmm(..., out_dtype=torch.float32)` from bf16/fp16 inputs, CUDA only
_MM_OUT_DTYPE = Version(torch.__version__) >= Version("2.8.0")


def _jit(fn):
    # [FUNCTION DOC]
    # HELPER: Decorator applying triton.jit when Triton is available, else returning the function unchanged.
    # Inputs: fn. Outputs: jitted kernel or fn. Exceptions: none. Side effects: none.
    # Triton ships with PyTorch on Linux only; without it, the tiles are folded in torch
    return triton.jit(fn) if is_triton_available() else fn


# Read once at import: Dynamo warns when it traces a call to the `lru_cache`-wrapped `is_triton_available`
_TRITON_AVAILABLE = is_triton_available()


def _use_triton(device: torch.device) -> bool:
    # [FUNCTION DOC]
    # HELPER: True when Triton is available and the device type is cuda or xpu.
    # Inputs: device. Outputs: bool. Exceptions: none. Side effects: none.
    return _TRITON_AVAILABLE and device.type in ("cuda", "xpu")


def _addmm_fp32(acc: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
    # [FUNCTION DOC]
    # HELPER: Accumulate the matrix product of a and b into the fp32 accumulator acc without first rounding the
    # product to the inputs' dtype: on CUDA with half/bf16 inputs and torch 2.8 or later use addmm with float32
    # output into acc; otherwise upcast both operands and addmm in place.
    # Inputs: acc (fp32), a, b. Outputs: none. Exceptions: shape errors. Side effects: mutates acc.
    """
    `acc += a @ b` for an fp32 `acc`, without rounding `a @ b` to the inputs' dtype first.

    Rounding every vocabulary chunk's product to bf16 before summing them biases training (entropy drifts lower over a
    few dozen steps), even though each step's gradient error looks the same size as with full logits.
    """
    if _MM_OUT_DTYPE and acc.device.type == "cuda" and a.dtype in (torch.float16, torch.bfloat16):
        torch.addmm(acc, a, b, out_dtype=torch.float32, out=acc)
    else:
        acc.addmm_(a.float(), b.float())


@_jit
def _transform(z, logit_scale, softcap, inv_t, HAS_SOFTCAP: tl.constexpr):
    # [FUNCTION DOC]
    # HELPER: Triton device function mapping raw logits to transformed logits: multiply by logit_scale, apply
    # softcap times tanh(z / softcap) (expressed via sigmoid) when HAS_SOFTCAP, then multiply by inv_t; scalars
    # cast to fp32 first.
    # Inputs: z block, logit_scale, softcap, inv_t, HAS_SOFTCAP. Outputs: transformed block. Exceptions: none.
    # Side effects: none.
    # Before torch 2.15, torch.compile passes Python float arguments as fp64; keep the math in fp32
    logit_scale = tl.cast(logit_scale, tl.float32)
    softcap = tl.cast(softcap, tl.float32)
    inv_t = tl.cast(inv_t, tl.float32)
    z = z * logit_scale
    if HAS_SOFTCAP:
        # tanh(x) = 2 * sigmoid(2x) - 1
        z = softcap * (2 * tl.sigmoid(2 * z / softcap) - 1)
    return z * inv_t


@_jit
def _forward_kernel(
    mm_ptr,
    mm_stride,
    targets_ptr,
    max_ptr,
    sum_exp_ptr,
    x_sum_exp_ptr,
    sq_sum_exp_ptr,
    z_sum_ptr,
    max_before_target_ptr,
    target_logit_ptr,
    vocab_start,
    n_cols,
    logit_scale,
    softcap,
    inv_t,
    HAS_SOFTCAP: tl.constexpr,
    HAS_ENTROPY: tl.constexpr,
    HAS_LOG_SUM_SQ_PROBS: tl.constexpr,
    HAS_MEAN_LOGITS: tl.constexpr,
    HAS_IS_TOP1: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # [FUNCTION DOC]
    # HELPER: Triton kernel, one program per token row, folding one vocab tile into running statistics.
    # Behavior: load the row's running max, sum of exps, and optional accumulators; the target's local column is
    # target minus vocab_start. Per BLOCK_SIZE block: transform logits; add to z_sum if HAS_MEAN_LOGITS; mask
    # invalid columns to negative infinity; if HAS_IS_TOP1 update the max over columns strictly before the target
    # (so ties with earlier tokens are not top-1); online-logsumexp update of max and sum; with HAS_ENTROPY
    # accumulate exp-weighted logits; with HAS_LOG_SUM_SQ_PROBS accumulate squared exps (rescaled by the squared
    # factor). Store updated stats, and if the target falls inside this tile store its transformed logit.
    # Inputs: tile pointer/stride, targets, stat pointers, vocab_start, n_cols, transform params, flags,
    # BLOCK_SIZE. Outputs: none. Exceptions: none. Side effects: writes stat buffers.
    # One program per token: fold one vocabulary chunk of its logits into the running online-logsumexp statistics
    row = tl.program_id(0).to(tl.int64)
    row_ptr = mm_ptr + row * mm_stride
    m = tl.load(max_ptr + row)
    local = tl.load(targets_ptr + row) - vocab_start
    max_before_target = tl.load(max_before_target_ptr + row)
    s = tl.load(sum_exp_ptr + row)
    xs = tl.load(x_sum_exp_ptr + row)
    sq = tl.load(sq_sum_exp_ptr + row)
    zs = tl.load(z_sum_ptr + row)
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        z = tl.load(row_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        z = _transform(z, logit_scale, softcap, inv_t, HAS_SOFTCAP)
        if HAS_MEAN_LOGITS:
            zs += tl.sum(tl.where(mask, z, 0.0), axis=0)
        z = tl.where(mask, z, -float("inf"))
        if HAS_IS_TOP1:
            # `argmax` returns the first maximum, so a tie with an earlier token does not count as a top-1 prediction
            max_before_target = tl.maximum(
                max_before_target, tl.max(tl.where(offsets < local, z, -float("inf")), axis=0)
            )
        new_m = tl.maximum(m, tl.max(z, axis=0))
        rescale = tl.exp(m - new_m)
        e = tl.where(mask, tl.exp(z - new_m), 0.0)
        s = s * rescale + tl.sum(e, axis=0)
        if HAS_ENTROPY:
            xs = xs * rescale + tl.sum(e * tl.where(mask, z, 0.0), axis=0)
        if HAS_LOG_SUM_SQ_PROBS:
            sq = sq * rescale * rescale + tl.sum(e * e, axis=0)
        m = new_m
    tl.store(max_ptr + row, m)
    tl.store(sum_exp_ptr + row, s)
    if HAS_ENTROPY:
        tl.store(x_sum_exp_ptr + row, xs)
    if HAS_LOG_SUM_SQ_PROBS:
        tl.store(sq_sum_exp_ptr + row, sq)
    if HAS_MEAN_LOGITS:
        tl.store(z_sum_ptr + row, zs)
    if HAS_IS_TOP1:
        tl.store(max_before_target_ptr + row, max_before_target)

    if (local >= 0) & (local < n_cols):
        z = tl.load(row_ptr + local).to(tl.float32)
        tl.store(target_logit_ptr + row, _transform(z, logit_scale, softcap, inv_t, HAS_SOFTCAP))


@_jit
def _backward_kernel(
    mm_ptr,
    mm_stride,
    targets_ptr,
    log_z_ptr,
    entropy_ptr,
    grad_logprobs_ptr,
    grad_entropy_ptr,
    vocab_start,
    n_cols,
    logit_scale,
    softcap,
    inv_t,
    HAS_SOFTCAP: tl.constexpr,
    HAS_LOGPROB_GRAD: tl.constexpr,
    HAS_ENTROPY_GRAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # [FUNCTION DOC]
    # HELPER: Triton kernel, grid (rows, column blocks), overwriting the logits tile with the gradient with respect
    # to the raw projection output.
    # Behavior: recompute transformed logits (keeping tanh value for softcap), log p and p. Gradient with respect
    # to the transformed logits: with HAS_LOGPROB_GRAD add upstream grad times (one-hot target minus p); with
    # HAS_ENTROPY_GRAD subtract upstream grad times p times (log p plus entropy). Multiply by inv_t, the softcap
    # derivative when enabled and logit_scale; store in the tile's dtype.
    # Inputs: tile pointer/stride, targets, log_z, entropy, grad pointers, vocab_start, n_cols, transform params,
    # flags, BLOCK_SIZE. Outputs: none. Exceptions: none. Side effects: overwrites the tile.
    # Overwrite one block of the logits tile with the gradient of the loss with respect to the projection output
    # Before torch 2.15, torch.compile passes Python float arguments as fp64; keep the math in fp32
    logit_scale = tl.cast(logit_scale, tl.float32)
    softcap = tl.cast(softcap, tl.float32)
    inv_t = tl.cast(inv_t, tl.float32)
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    ptrs = mm_ptr + row * mm_stride + offsets
    y = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32) * logit_scale
    if HAS_SOFTCAP:
        t = 2 * tl.sigmoid(2 * y / softcap) - 1
        z = softcap * t * inv_t
    else:
        z = y * inv_t
    log_p = z - tl.load(log_z_ptr + row)
    p = tl.exp(log_p)

    grad = tl.zeros((BLOCK_SIZE,), tl.float32)
    if HAS_LOGPROB_GRAD:
        # d(log p_target) / dz_j = 1[j = target] - p_j
        g = tl.load(grad_logprobs_ptr + row)
        is_target = (offsets + vocab_start) == tl.load(targets_ptr + row)
        grad += g * (is_target.to(tl.float32) - p)
    if HAS_ENTROPY_GRAD:
        # d(entropy) / dz_j = -p_j * (log p_j + entropy)
        g = tl.load(grad_entropy_ptr + row)
        grad -= g * p * (log_p + tl.load(entropy_ptr + row))
    grad = grad * inv_t
    if HAS_SOFTCAP:
        grad = grad * (1 - t * t)
    grad = grad * logit_scale
    tl.store(ptrs, grad.to(mm_ptr.dtype.element_ty), mask=mask)


def _torch_transform(z, logit_scale, softcap, inv_t, HAS_SOFTCAP):
    # [FUNCTION DOC]
    # HELPER: Torch equivalent of _transform (scale, optional softcap tanh, times inv_t).
    # Inputs: z tensor and the same params. Outputs: tensor. Exceptions: none. Side effects: none.
    z = z * logit_scale
    if HAS_SOFTCAP:
        z = softcap * torch.tanh(z / softcap)
    return z * inv_t


def _forward_tile(
    tile,
    tile_stride,
    targets,
    running_max,
    sum_exp,
    x_sum_exp,
    sq_sum_exp,
    z_sum,
    max_before_target,
    target_logit,
    vocab_start,
    n_cols,
    logit_scale,
    softcap,
    inv_t,
    HAS_SOFTCAP,
    HAS_ENTROPY,
    HAS_LOG_SUM_SQ_PROBS,
    HAS_MEAN_LOGITS,
    HAS_IS_TOP1,
    BLOCK_SIZE,
):
    # [FUNCTION DOC]
    # HELPER: Torch fallback of _forward_kernel over the whole tile (vectorized across rows): same statistics
    # updates done in place; the target logit is gathered with a clamped index and written only for rows whose
    # target lies in this tile. tile_stride and BLOCK_SIZE are unused.
    # Inputs: as _forward_kernel (tensors). Outputs: none. Exceptions: none. Side effects: mutates stat tensors.
    # `_forward_kernel` in torch, for devices without Triton: the whole tile at once instead of one program per token
    z = _torch_transform(tile.float(), logit_scale, softcap, inv_t, HAS_SOFTCAP)
    local = targets - vocab_start
    if HAS_MEAN_LOGITS:
        z_sum += z.sum(-1)
    if HAS_IS_TOP1:
        # `argmax` returns the first maximum, so a tie with an earlier token does not count as a top-1 prediction
        before_target = torch.arange(n_cols, device=z.device) < local.unsqueeze(-1)
        max_before_target.copy_(torch.maximum(max_before_target, z.masked_fill(~before_target, -math.inf).amax(-1)))
    new_m = torch.maximum(running_max, z.amax(-1))
    rescale = torch.exp(running_max - new_m)
    e = torch.exp(z - new_m.unsqueeze(-1))
    sum_exp.mul_(rescale).add_(e.sum(-1))
    if HAS_ENTROPY:
        x_sum_exp.mul_(rescale).add_((e * z).sum(-1))
    if HAS_LOG_SUM_SQ_PROBS:
        sq_sum_exp.mul_(rescale * rescale).add_((e * e).sum(-1))
    running_max.copy_(new_m)
    in_tile = (local >= 0) & (local < n_cols)
    target_z = z.gather(-1, local.clamp(0, n_cols - 1).unsqueeze(-1)).squeeze(-1)
    target_logit.copy_(torch.where(in_tile, target_z, target_logit))


def _backward_tile(
    tile,
    tile_stride,
    targets,
    log_z,
    entropy,
    grad_logprobs,
    grad_entropy,
    vocab_start,
    n_cols,
    logit_scale,
    softcap,
    inv_t,
    HAS_SOFTCAP,
    HAS_LOGPROB_GRAD,
    HAS_ENTROPY_GRAD,
    BLOCK_SIZE,
):
    # [FUNCTION DOC]
    # HELPER: Torch fallback of _backward_kernel over the whole tile; the result is copied into tile.
    # Inputs: as _backward_kernel (tensors). Outputs: none. Exceptions: none. Side effects: overwrites tile.
    # `_backward_kernel` in torch, for devices without Triton: the whole tile at once instead of one block per program
    y = tile.float() * logit_scale
    if HAS_SOFTCAP:
        t = torch.tanh(y / softcap)
        z = softcap * t * inv_t
    else:
        z = y * inv_t
    log_p = z - log_z.unsqueeze(-1)
    p = torch.exp(log_p)

    grad = torch.zeros_like(z)
    if HAS_LOGPROB_GRAD:
        # d(log p_target) / dz_j = 1[j = target] - p_j
        is_target = torch.arange(vocab_start, vocab_start + n_cols, device=z.device) == targets.unsqueeze(-1)
        grad += grad_logprobs.unsqueeze(-1) * (is_target.float() - p)
    if HAS_ENTROPY_GRAD:
        # d(entropy) / dz_j = -p_j * (log p_j + entropy)
        grad -= grad_entropy.unsqueeze(-1) * p * (log_p + entropy.unsqueeze(-1))
    grad = grad * inv_t
    if HAS_SOFTCAP:
        grad = grad * (1 - t * t)
    tile.copy_(grad * logit_scale)


class ChunkedLogProbFunction(torch.autograd.Function):
    # [CLASS DOC]
    # Autograd function computing target log-probs of hidden times weight transposed (plus bias), with temperature,
    # optional final-logit softcapping and logit scale, and optional extra outputs selected by name from
    # OPTIONAL_OUTPUTS. Gradients flow to hidden, weight and bias through logprobs and entropy; the other extras are
    # non-differentiable.
    """
    Per-token log-probabilities of `hidden @ weight.T`, without materializing the `[N, V]` logits, and optionally the
    entropy, `log(sum_v p_v^2)`, the mean logit and whether the target is the argmax (`outputs`, `None` when not
    requested).

    The projection runs in cuBLAS on `[TOKEN_CHUNK_SIZE, chunk_size]` tiles; a Triton kernel (torch ops on devices
    without Triton) folds each tile into online-logsumexp statistics in one pass. The backward recomputes each tile,
    turns it into the logits gradient in place, and accumulates the gradient GEMMs in fp32.
    """

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,  # [N, H]
        weight: torch.Tensor,  # [V, H]
        bias: torch.Tensor | None,  # [V]
        targets: torch.Tensor,  # [N]
        temperature: float,
        chunk_size: int,
        final_logit_softcapping: float | None = None,
        logit_scale: float = 1.0,
        outputs: tuple[str, ...] = OPTIONAL_OUTPUTS,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        # [FUNCTION DOC]
        # Behavior: disable grad materialization; compute dtype is the autocast dtype if enabled else hidden's dtype.
        # Allocate fp32 per-token stats (running max and max-before-target at negative infinity, others zero) and one
        # reusable tile buffer in the compute dtype. For each token chunk and vocab chunk, project with mm into the
        # tile (plus bias) and fold with the Triton kernel or torch fallback. Then log Z = max + log(sum); logprobs =
        # target logit minus log Z; entropy = log Z minus weighted-logit sum over sum (if requested); log_sum_sq_probs
        # = log(sq sum) minus 2 log(sum); mean_logits = z_sum over vocab size; is_top1 = target logit at least the max
        # and strictly above every earlier column. Save hidden, weight, bias, targets, log Z and entropy (log Z as a
        # placeholder when entropy is absent) plus compute dtype, chunk size and kernel args; mark the extras other
        # than entropy non-differentiable.
        # Inputs: ctx, hidden [N, H], weight [V, H], bias [V] or None, targets [N], temperature, chunk_size,
        # final_logit_softcapping (None), logit_scale (1.0), outputs (default all optional outputs).
        # Outputs: (logprobs, entropy, log_sum_sq_probs, mean_logits, is_top1), unrequested ones None.
        # Exceptions: none explicit. Side effects: none beyond allocations.
        # entropy is often computed for logging only (no grad required); without this, autograd would
        # materialize its incoming gradient as zeros and backward would waste compute on a no-op term
        ctx.set_materialize_grads(False)
        device = hidden.device
        N = hidden.shape[0]
        vocab = weight.shape[0]
        compute_dtype = (
            torch.get_autocast_dtype(device.type) if torch.is_autocast_enabled(device.type) else hidden.dtype
        )
        kernel_args = {
            "logit_scale": logit_scale,
            "softcap": final_logit_softcapping or 1.0,
            "inv_t": 1 / temperature,
            "HAS_SOFTCAP": final_logit_softcapping is not None,
        }
        output_flags = {f"HAS_{name.upper()}": name in outputs for name in OPTIONAL_OUTPUTS}

        running_max = torch.full((N,), float("-inf"), device=device, dtype=torch.float32)
        sum_exp = torch.zeros((N,), device=device, dtype=torch.float32)
        x_sum_exp = torch.zeros((N,), device=device, dtype=torch.float32)
        sq_sum_exp = torch.zeros((N,), device=device, dtype=torch.float32)
        z_sum = torch.zeros((N,), device=device, dtype=torch.float32)
        max_before_target = torch.full((N,), float("-inf"), device=device, dtype=torch.float32)
        target_logit = torch.zeros((N,), device=device, dtype=torch.float32)
        mm_buf = torch.empty((min(N, TOKEN_CHUNK_SIZE), chunk_size), device=device, dtype=compute_dtype)

        for token_start in range(0, N, TOKEN_CHUNK_SIZE):
            token_end = min(token_start + TOKEN_CHUNK_SIZE, N)
            n = token_end - token_start
            hidden_chunk = hidden[token_start:token_end].to(compute_dtype)
            for start in range(0, vocab, chunk_size):
                end = min(start + chunk_size, vocab)
                tile = mm_buf[:n, : end - start]
                torch.mm(hidden_chunk, weight[start:end].to(compute_dtype).t(), out=tile)
                if bias is not None:
                    tile.add_(bias[start:end].to(compute_dtype))
                fold = _forward_kernel[(n,)] if _use_triton(device) else _forward_tile
                fold(
                    tile,
                    tile.stride(0),
                    targets[token_start:token_end],
                    running_max[token_start:token_end],
                    sum_exp[token_start:token_end],
                    x_sum_exp[token_start:token_end],
                    sq_sum_exp[token_start:token_end],
                    z_sum[token_start:token_end],
                    max_before_target[token_start:token_end],
                    target_logit[token_start:token_end],
                    start,
                    end - start,
                    BLOCK_SIZE=_BLOCK_SIZE,
                    **output_flags,
                    **kernel_args,
                )

        log_z = running_max + torch.log(sum_exp)
        logprobs = target_logit - log_z
        entropy = log_z - x_sum_exp / sum_exp if "entropy" in outputs else None
        log_sum_sq_probs = torch.log(sq_sum_exp) - 2 * torch.log(sum_exp) if "log_sum_sq_probs" in outputs else None
        mean_logits = z_sum / vocab if "mean_logits" in outputs else None
        is_top1 = (target_logit >= running_max) & (target_logit > max_before_target) if "is_top1" in outputs else None

        # Without entropy there is no entropy gradient, so `log_z` only fills its slot
        ctx.save_for_backward(hidden, weight, bias, targets, log_z, log_z if entropy is None else entropy)
        ctx.compute_dtype = compute_dtype
        ctx.chunk_size = chunk_size
        ctx.kernel_args = kernel_args
        ctx.mark_non_differentiable(*(x for x in (log_sum_sq_probs, mean_logits, is_top1) if x is not None))
        return logprobs, entropy, log_sum_sq_probs, mean_logits, is_top1

    @staticmethod
    def backward(ctx, grad_logprobs: torch.Tensor | None, grad_entropy: torch.Tensor | None, *_):  # type: ignore
        # [FUNCTION DOC]
        # Behavior: import maybe_gather_lm_head_ctx lazily (circular import) and run inside it. Allocate fp32 gradient
        # accumulators as needed by needs_input_grad. For every tile, recompute the projection, overwrite it with the
        # logits gradient (Triton or fallback, flags set by which upstream grads are present; missing grads are
        # passed as placeholders), then accumulate hidden grad += tile times weight chunk, weight grad += tile
        # transposed times hidden chunk, bias grad += column sums. Return gradients cast to input dtypes and None for
        # the six non-tensor/target inputs.
        # Inputs: ctx, grad_logprobs (or None), grad_entropy (or None), ignored grads for the other outputs.
        # Outputs: tuple of 9 gradients. Exceptions: none explicit.
        # Side effects: may gather a sharded LM head temporarily.
        # `trl.trainer.utils` imports this module, so import from it at call time
        from ..trainer.utils import maybe_gather_lm_head_ctx

        hidden, weight, bias, targets, log_z, entropy = ctx.saved_tensors
        compute_dtype = ctx.compute_dtype
        chunk_size = ctx.chunk_size
        needs_hidden_grad, needs_weight_grad, needs_bias_grad = ctx.needs_input_grad[:3]
        N = hidden.shape[0]
        with maybe_gather_lm_head_ctx(weight, bias):
            vocab = weight.shape[0]
            # Always accumulate in fp32, even when the inputs are not
            grad_hidden = (
                torch.zeros(hidden.shape, device=hidden.device, dtype=torch.float32) if needs_hidden_grad else None
            )
            grad_weight = (
                torch.zeros(weight.shape, device=weight.device, dtype=torch.float32) if needs_weight_grad else None
            )
            grad_bias = torch.zeros(bias.shape, device=bias.device, dtype=torch.float32) if needs_bias_grad else None
            mm_buf = torch.empty((min(N, TOKEN_CHUNK_SIZE), chunk_size), device=hidden.device, dtype=compute_dtype)
            grad_logprobs = grad_logprobs.float().contiguous() if grad_logprobs is not None else None
            grad_entropy = grad_entropy.float().contiguous() if grad_entropy is not None else None

            for token_start in range(0, N, TOKEN_CHUNK_SIZE):
                token_end = min(token_start + TOKEN_CHUNK_SIZE, N)
                n = token_end - token_start
                hidden_chunk = hidden[token_start:token_end].to(compute_dtype)
                for start in range(0, vocab, chunk_size):
                    end = min(start + chunk_size, vocab)
                    w_chunk = weight[start:end].to(compute_dtype)
                    tile = mm_buf[:n, : end - start]
                    torch.mm(hidden_chunk, w_chunk.t(), out=tile)
                    if bias is not None:
                        tile.add_(bias[start:end].to(compute_dtype))
                    fold = (
                        _backward_kernel[(n, triton.cdiv(end - start, _BLOCK_SIZE))]
                        if _use_triton(hidden.device)
                        else _backward_tile
                    )
                    fold(
                        tile,
                        tile.stride(0),
                        targets[token_start:token_end],
                        log_z[token_start:token_end],
                        entropy[token_start:token_end],
                        grad_logprobs[token_start:token_end] if grad_logprobs is not None else log_z,
                        grad_entropy[token_start:token_end] if grad_entropy is not None else log_z,
                        start,
                        end - start,
                        HAS_LOGPROB_GRAD=grad_logprobs is not None,
                        HAS_ENTROPY_GRAD=grad_entropy is not None,
                        BLOCK_SIZE=_BLOCK_SIZE,
                        **ctx.kernel_args,
                    )
                    if grad_hidden is not None:
                        _addmm_fp32(grad_hidden[token_start:token_end], tile, w_chunk)
                    if grad_weight is not None:
                        _addmm_fp32(grad_weight[start:end], tile.t(), hidden_chunk)
                    if grad_bias is not None:
                        grad_bias[start:end] += tile.sum(dim=0, dtype=torch.float32)

        return (
            grad_hidden.to(hidden.dtype) if grad_hidden is not None else None,
            grad_weight.to(weight.dtype) if grad_weight is not None else None,
            grad_bias.to(bias.dtype) if grad_bias is not None else None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
