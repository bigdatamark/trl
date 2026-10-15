# [MODULE DOC]
# Module trl.kernels.chunked_divergence: memory-efficient per-token divergence (forward KL, reverse KL or
# generalized JSD) between a student and a teacher next-token distribution, plus the student entropy,
# computed directly from hidden states and LM-head weights in [TOKEN_CHUNK_SIZE, VOCAB_CHUNK_SIZE] tiles so
# neither model's full [N, V] logits is ever materialized. Used by distillation trainers.
# Main contents: Triton kernels (_stats_kernel, _jsd_kernel, _backward_kernel) with torch fallbacks
# (_stats_tile, _jsd_tile, _backward_tile) selected by _use_triton; projection helpers _compute_dtype and
# _project; and ChunkedDivergenceFunction (torch.autograd.Function). Reuses _transform/_torch_transform
# (logit scale, optional softcap tanh, temperature), _jit, _BLOCK_SIZE, _MM_OUT_DTYPE and _addmm_fp32 from
# chunked_logprob.
# Import-time side effects: imports triton when available; defines TOKEN_CHUNK_SIZE 4096 and VOCAB_CHUNK_SIZE
# 32768; kernels are JIT-wrapped (compiled lazily).

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
from transformers.utils import is_triton_available

from ..trainer.utils import maybe_gather_lm_head_ctx
from .chunked_logprob import _BLOCK_SIZE, _MM_OUT_DTYPE, _addmm_fp32, _jit, _torch_transform, _transform, _use_triton


if is_triton_available():
    import triton
    import triton.language as tl


# The projections run on `[TOKEN_CHUNK_SIZE, VOCAB_CHUNK_SIZE]` tiles, so neither model's logits exist in full
TOKEN_CHUNK_SIZE = 4096
VOCAB_CHUNK_SIZE = 32768


@_jit
def _stats_kernel(
    s_ptr,
    s_stride,
    t_ptr,
    t_stride,
    s_max_ptr,
    s_sum_ptr,
    s_x_sum_ptr,
    t_max_ptr,
    t_sum_ptr,
    t_weighted_gap_ptr,
    s_weighted_gap_ptr,
    n_cols,
    s_scale,
    s_softcap,
    t_scale,
    t_softcap,
    inv_t,
    S_HAS_SOFTCAP: tl.constexpr,
    T_HAS_SOFTCAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # [FUNCTION DOC]
    # HELPER: Triton kernel, one program per token row, folding one vocab tile of student and teacher logits into
    # running online-logsumexp statistics.
    # Behavior: load the running stats for the row; iterate over the tile in BLOCK_SIZE blocks; transform both
    # logit blocks (scale, optional softcap, divide by temperature); gap is teacher minus student transformed
    # logit (0 outside the mask). For the student, update the running max, rescale and accumulate the exp-sum,
    # the exp-weighted sum of logits (for entropy) and the exp-weighted negative gap (for reverse KL). For the
    # teacher, update max, exp-sum and exp-weighted gap (for forward KL). Store all stats back.
    # Inputs: pointers/strides of the two tiles, pointers to seven per-row float32 stat vectors, n_cols, scale and
    # softcap per model, inv_t, softcap flags and BLOCK_SIZE (compile-time).
    # Outputs: none (stats updated in place). Exceptions: none. Side effects: writes the stat buffers.
    # One program per token: fold one vocabulary chunk of both models' logits into online-logsumexp statistics
    row = tl.program_id(0).to(tl.int64)
    s_row = s_ptr + row * s_stride
    t_row = t_ptr + row * t_stride
    s_m = tl.load(s_max_ptr + row)
    s_s = tl.load(s_sum_ptr + row)
    s_xs = tl.load(s_x_sum_ptr + row)
    t_m = tl.load(t_max_ptr + row)
    t_s = tl.load(t_sum_ptr + row)
    t_gap = tl.load(t_weighted_gap_ptr + row)
    s_gap = tl.load(s_weighted_gap_ptr + row)
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        zs = _transform(
            tl.load(s_row + offsets, mask=mask, other=0.0).to(tl.float32), s_scale, s_softcap, inv_t, S_HAS_SOFTCAP
        )
        zt = _transform(
            tl.load(t_row + offsets, mask=mask, other=0.0).to(tl.float32), t_scale, t_softcap, inv_t, T_HAS_SOFTCAP
        )
        gap = tl.where(mask, zt - zs, 0.0)

        new_s_m = tl.maximum(s_m, tl.max(tl.where(mask, zs, -float("inf")), axis=0))
        s_rescale = tl.exp(s_m - new_s_m)
        es = tl.where(mask, tl.exp(zs - new_s_m), 0.0)
        s_s = s_s * s_rescale + tl.sum(es, axis=0)
        s_xs = s_xs * s_rescale + tl.sum(es * tl.where(mask, zs, 0.0), axis=0)
        s_gap = s_gap * s_rescale - tl.sum(es * gap, axis=0)
        s_m = new_s_m

        new_t_m = tl.maximum(t_m, tl.max(tl.where(mask, zt, -float("inf")), axis=0))
        t_rescale = tl.exp(t_m - new_t_m)
        et = tl.where(mask, tl.exp(zt - new_t_m), 0.0)
        t_s = t_s * t_rescale + tl.sum(et, axis=0)
        t_gap = t_gap * t_rescale + tl.sum(et * gap, axis=0)
        t_m = new_t_m
    tl.store(s_max_ptr + row, s_m)
    tl.store(s_sum_ptr + row, s_s)
    tl.store(s_x_sum_ptr + row, s_xs)
    tl.store(t_max_ptr + row, t_m)
    tl.store(t_sum_ptr + row, t_s)
    tl.store(t_weighted_gap_ptr + row, t_gap)
    tl.store(s_weighted_gap_ptr + row, s_gap)


@_jit
def _jsd_kernel(
    s_ptr,
    s_stride,
    t_ptr,
    t_stride,
    s_log_z_ptr,
    t_log_z_ptr,
    jsd_ptr,
    kl_student_ptr,
    n_cols,
    beta,
    s_scale,
    s_softcap,
    t_scale,
    t_softcap,
    inv_t,
    S_HAS_SOFTCAP: tl.constexpr,
    T_HAS_SOFTCAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # [FUNCTION DOC]
    # HELPER: Triton kernel for the second pass of the generalized JSD, one program per row.
    # Behavior: cast float arguments to fp32. Using precomputed log-normalizers, compute log student and teacher
    # probabilities per block, the log mixture as a stable logsumexp of log(1 - beta) plus log p_s and log(beta)
    # plus log p_t, then accumulate beta times p_t times (log p_t minus log m) plus (1 - beta) times the student
    # term into jsd, and the student term p_s times (log p_s minus log m) into kl_student.
    # Inputs: tile pointers/strides, s_log_z and t_log_z pointers, jsd and kl_student pointers, n_cols, beta,
    # transform parameters, flags, BLOCK_SIZE. Outputs: none. Exceptions: none.
    # Side effects: accumulates into jsd and kl_student.
    # Second pass of the generalized JSD, which needs both normalizers to form the mixture m = (1 - beta) p_s + beta p_t
    # Before torch 2.15, torch.compile passes Python float arguments as fp64; keep the math in fp32
    beta = tl.cast(beta, tl.float32)
    s_scale = tl.cast(s_scale, tl.float32)
    s_softcap = tl.cast(s_softcap, tl.float32)
    t_scale = tl.cast(t_scale, tl.float32)
    t_softcap = tl.cast(t_softcap, tl.float32)
    inv_t = tl.cast(inv_t, tl.float32)
    row = tl.program_id(0).to(tl.int64)
    s_row = s_ptr + row * s_stride
    t_row = t_ptr + row * t_stride
    s_log_z = tl.load(s_log_z_ptr + row)
    t_log_z = tl.load(t_log_z_ptr + row)
    jsd = tl.load(jsd_ptr + row)
    kl_student = tl.load(kl_student_ptr + row)
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        zs = _transform(
            tl.load(s_row + offsets, mask=mask, other=0.0).to(tl.float32), s_scale, s_softcap, inv_t, S_HAS_SOFTCAP
        )
        zt = _transform(
            tl.load(t_row + offsets, mask=mask, other=0.0).to(tl.float32), t_scale, t_softcap, inv_t, T_HAS_SOFTCAP
        )
        log_ps = zs - s_log_z
        log_pt = zt - t_log_z
        a = log_ps + tl.log(1 - beta)
        b = log_pt + tl.log(beta)
        top = tl.maximum(a, b)
        log_m = top + tl.log(tl.exp(a - top) + tl.exp(b - top))
        ps = tl.where(mask, tl.exp(log_ps), 0.0)
        pt = tl.where(mask, tl.exp(log_pt), 0.0)
        kl_s = ps * tl.where(mask, log_ps - log_m, 0.0)
        jsd += tl.sum(beta * pt * tl.where(mask, log_pt - log_m, 0.0) + (1 - beta) * kl_s, axis=0)
        kl_student += tl.sum(kl_s, axis=0)
    tl.store(jsd_ptr + row, jsd)
    tl.store(kl_student_ptr + row, kl_student)


@_jit
def _backward_kernel(
    s_ptr,
    s_stride,
    t_ptr,
    t_stride,
    s_log_z_ptr,
    t_log_z_ptr,
    aux_ptr,
    grad_ptr,
    n_cols,
    beta,
    s_scale,
    s_softcap,
    t_scale,
    t_softcap,
    inv_t,
    MODE: tl.constexpr,
    S_HAS_SOFTCAP: tl.constexpr,
    T_HAS_SOFTCAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # [FUNCTION DOC]
    # HELPER: Triton kernel, grid (rows, column blocks), overwriting the student tile with the gradient of the
    # per-token divergence with respect to the raw projection output.
    # Behavior: recompute student transformed logits keeping the tanh value for softcap, the teacher logits, and
    # log-probabilities. Gradient with respect to the student's transformed logit: MODE 0 (forward KL) p_s minus
    # p_t; MODE 1 (reverse KL) p_s times (log p_s minus log p_t minus aux), aux being the per-token KL; MODE 2 (JSD)
    # (1 - beta) times p_s times (log p_s minus log m minus aux), aux being KL(p_s, m). Multiply by the upstream
    # per-token gradient and inv_t, by the softcap derivative (1 minus tanh squared) when enabled, and by the
    # student scale; store in the tile's dtype.
    # Inputs: as _jsd_kernel plus aux_ptr, grad_ptr and MODE. Outputs: none. Exceptions: none.
    # Side effects: overwrites the student tile.
    # Overwrite one block of the student tile with the gradient of the divergence with respect to the projection output
    # Before torch 2.15, torch.compile passes Python float arguments as fp64; keep the math in fp32
    beta = tl.cast(beta, tl.float32)
    s_scale = tl.cast(s_scale, tl.float32)
    s_softcap = tl.cast(s_softcap, tl.float32)
    t_scale = tl.cast(t_scale, tl.float32)
    t_softcap = tl.cast(t_softcap, tl.float32)
    inv_t = tl.cast(inv_t, tl.float32)
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    s_ptrs = s_ptr + row * s_stride + offsets
    y = tl.load(s_ptrs, mask=mask, other=0.0).to(tl.float32) * s_scale
    if S_HAS_SOFTCAP:
        th = 2 * tl.sigmoid(2 * y / s_softcap) - 1
        zs = s_softcap * th * inv_t
    else:
        zs = y * inv_t
    zt = _transform(
        tl.load(t_ptr + row * t_stride + offsets, mask=mask, other=0.0).to(tl.float32),
        t_scale,
        t_softcap,
        inv_t,
        T_HAS_SOFTCAP,
    )
    log_ps = zs - tl.load(s_log_z_ptr + row)
    log_pt = zt - tl.load(t_log_z_ptr + row)
    ps = tl.exp(log_ps)
    if MODE == 0:
        # forward KL(p_t || p_s)
        grad = ps - tl.exp(log_pt)
    elif MODE == 1:
        # reverse KL(p_s || p_t), `aux` is the per-token KL
        grad = ps * (log_ps - log_pt - tl.load(aux_ptr + row))
    else:
        # generalized JSD, `aux` is the per-token KL(p_s || m)
        a = log_ps + tl.log(1 - beta)
        b = log_pt + tl.log(beta)
        top = tl.maximum(a, b)
        log_m = top + tl.log(tl.exp(a - top) + tl.exp(b - top))
        grad = (1 - beta) * ps * (log_ps - log_m - tl.load(aux_ptr + row))
    grad = grad * tl.load(grad_ptr + row) * inv_t
    if S_HAS_SOFTCAP:
        grad = grad * (1 - th * th)
    grad = grad * s_scale
    tl.store(s_ptrs, grad.to(s_ptr.dtype.element_ty), mask=mask)


def _stats_tile(
    s_tile,
    s_stride,
    t_tile,
    t_stride,
    s_max,
    s_sum,
    s_x_sum,
    t_max,
    t_sum,
    t_weighted_gap,
    s_weighted_gap,
    n_cols,
    s_scale,
    s_softcap,
    t_scale,
    t_softcap,
    inv_t,
    S_HAS_SOFTCAP,
    T_HAS_SOFTCAP,
    BLOCK_SIZE,
):
    # [FUNCTION DOC]
    # HELPER: Torch fallback of _stats_kernel operating on the whole tile at once (vectorized over rows), updating
    # the stat tensors in place with the same online-logsumexp recurrences. Stride, n_cols and BLOCK_SIZE are
    # accepted for signature compatibility and unused.
    # Inputs: same as _stats_kernel with tensors instead of pointers. Outputs: none.
    # Exceptions: none. Side effects: mutates the stat tensors.
    # `_stats_kernel` in torch, for devices without Triton: the whole tile at once instead of one program per token
    zs = _torch_transform(s_tile, s_scale, s_softcap, inv_t, S_HAS_SOFTCAP)
    zt = _torch_transform(t_tile, t_scale, t_softcap, inv_t, T_HAS_SOFTCAP)
    gap = zt - zs

    new_s_m = torch.maximum(s_max, zs.amax(-1))
    s_rescale = torch.exp(s_max - new_s_m)
    es = torch.exp(zs - new_s_m.unsqueeze(-1))
    s_sum.mul_(s_rescale).add_(es.sum(-1))
    s_x_sum.mul_(s_rescale).add_((es * zs).sum(-1))
    s_weighted_gap.mul_(s_rescale).sub_((es * gap).sum(-1))
    s_max.copy_(new_s_m)

    new_t_m = torch.maximum(t_max, zt.amax(-1))
    t_rescale = torch.exp(t_max - new_t_m)
    et = torch.exp(zt - new_t_m.unsqueeze(-1))
    t_sum.mul_(t_rescale).add_(et.sum(-1))
    t_weighted_gap.mul_(t_rescale).add_((et * gap).sum(-1))
    t_max.copy_(new_t_m)


def _jsd_tile(
    s_tile,
    s_stride,
    t_tile,
    t_stride,
    s_log_z,
    t_log_z,
    jsd,
    kl_student,
    n_cols,
    beta,
    s_scale,
    s_softcap,
    t_scale,
    t_softcap,
    inv_t,
    S_HAS_SOFTCAP,
    T_HAS_SOFTCAP,
    BLOCK_SIZE,
):
    # [FUNCTION DOC]
    # HELPER: Torch fallback of _jsd_kernel on the whole tile; uses logaddexp for the mixture and accumulates into
    # jsd and kl_student in place. Unused args kept for signature compatibility.
    # Inputs: as _jsd_kernel (tensors). Outputs: none. Exceptions: none. Side effects: mutates jsd, kl_student.
    # `_jsd_kernel` in torch, for devices without Triton: the whole tile at once instead of one program per token
    log_ps = _torch_transform(s_tile, s_scale, s_softcap, inv_t, S_HAS_SOFTCAP) - s_log_z.unsqueeze(-1)
    log_pt = _torch_transform(t_tile, t_scale, t_softcap, inv_t, T_HAS_SOFTCAP) - t_log_z.unsqueeze(-1)
    log_m = torch.logaddexp(log_ps + math.log(1 - beta), log_pt + math.log(beta))
    kl_s = torch.exp(log_ps) * (log_ps - log_m)
    jsd += (beta * torch.exp(log_pt) * (log_pt - log_m) + (1 - beta) * kl_s).sum(-1)
    kl_student += kl_s.sum(-1)


def _backward_tile(
    s_tile,
    s_stride,
    t_tile,
    t_stride,
    s_log_z,
    t_log_z,
    aux,
    grad_divergence,
    n_cols,
    beta,
    s_scale,
    s_softcap,
    t_scale,
    t_softcap,
    inv_t,
    MODE,
    S_HAS_SOFTCAP,
    T_HAS_SOFTCAP,
    BLOCK_SIZE,
):
    # [FUNCTION DOC]
    # HELPER: Torch fallback of _backward_kernel on the whole tile, with the same three MODE formulas; the result is
    # copied into s_tile. Unused args kept for signature compatibility.
    # Inputs: as _backward_kernel (tensors). Outputs: none. Exceptions: none. Side effects: overwrites s_tile.
    # `_backward_kernel` in torch, for devices without Triton: the whole tile at once instead of one block per program
    y = s_tile * s_scale
    if S_HAS_SOFTCAP:
        th = torch.tanh(y / s_softcap)
        zs = s_softcap * th * inv_t
    else:
        zs = y * inv_t
    zt = _torch_transform(t_tile, t_scale, t_softcap, inv_t, T_HAS_SOFTCAP)
    log_ps = zs - s_log_z.unsqueeze(-1)
    log_pt = zt - t_log_z.unsqueeze(-1)
    ps = torch.exp(log_ps)
    if MODE == 0:
        # forward KL(p_t || p_s)
        grad = ps - torch.exp(log_pt)
    elif MODE == 1:
        # reverse KL(p_s || p_t), `aux` is the per-token KL
        grad = ps * (log_ps - log_pt - aux.unsqueeze(-1))
    else:
        # generalized JSD, `aux` is the per-token KL(p_s || m)
        log_m = torch.logaddexp(log_ps + math.log(1 - beta), log_pt + math.log(beta))
        grad = (1 - beta) * ps * (log_ps - log_m - aux.unsqueeze(-1))
    grad = grad * grad_divergence.unsqueeze(-1) * inv_t
    if S_HAS_SOFTCAP:
        grad = grad * (1 - th * th)
    s_tile.copy_(grad * s_scale)


def _compute_dtype(hidden: torch.Tensor) -> torch.dtype:
    # [FUNCTION DOC]
    # HELPER: The dtype matmuls should run in: the autocast dtype of the tensor's device type when autocast is
    # enabled there, else the tensor's own dtype.
    # Inputs: hidden (tensor). Outputs: torch.dtype. Exceptions: none. Side effects: none.
    device_type = hidden.device.type
    return torch.get_autocast_dtype(device_type) if torch.is_autocast_enabled(device_type) else hidden.dtype


def _project(tile, hidden_chunk, weight, bias, start, end, dtype):
    # [FUNCTION DOC]
    # HELPER: Compute one logits tile in fp32: hidden_chunk times the transposed weight rows start to end.
    # Behavior: cast the weight slice to dtype; on CUDA with half/bf16 dtype and _MM_OUT_DTYPE support, use mm with
    # float32 output written into tile; otherwise upcast both operands to float32 and mm into tile. Add the bias
    # slice (fp32) when given.
    # Inputs: tile (preallocated fp32 [n, end - start]), hidden_chunk, weight [V, H], bias (optional), start, end,
    # dtype. Outputs: none. Exceptions: shape errors. Side effects: writes tile.
    # FP32 projection output avoids rounding logits before computing the divergence.
    w = weight[start:end].to(dtype).t()
    if _MM_OUT_DTYPE and tile.device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        torch.mm(hidden_chunk, w, out_dtype=torch.float32, out=tile)
    else:
        torch.mm(hidden_chunk.float(), w.float(), out=tile)
    if bias is not None:
        tile.add_(bias[start:end].float())


class ChunkedDivergenceFunction(torch.autograd.Function):
    # [CLASS DOC]
    # Autograd function returning per-token divergence (differentiable for the student only) and student entropy
    # (non-differentiable). beta 0 is forward KL(teacher, student), 1 reverse KL(student, teacher), in between the
    # generalized JSD. Supports per-model logit scale and final-logit softcapping and a shared temperature.
    """
    Per-token divergence between the student's and the teacher's next-token distributions, and the student's entropy,
    computed from both models' hidden states and LM heads without materializing either `[N, V]` logits.

    `beta` selects the divergence: `0.0` is the forward KL(teacher || student), `1.0` the reverse KL(student ||
    teacher), and anything in between the generalized JSD with mixture `(1 - beta) * p_student + beta * p_teacher`. The
    gradient flows to the student only; the entropy has none.
    """

    @staticmethod
    def forward(
        ctx,
        student_hidden: torch.Tensor,  # [N, H_s]
        student_weight: torch.Tensor,  # [V, H_s]
        student_bias: torch.Tensor | None,
        teacher_hidden: torch.Tensor,  # [N, H_t]
        teacher_weight: torch.Tensor,  # [V, H_t]
        teacher_bias: torch.Tensor | None,
        beta: float,
        temperature: float = 1.0,
        student_logit_scale: float = 1.0,
        student_final_logit_softcapping: float | None = None,
        teacher_logit_scale: float = 1.0,
        teacher_final_logit_softcapping: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC]
        # Behavior: disable grad materialization. Compute dtypes per model and kernel args (softcap defaults to 1.0
        # when disabled, inv_t is 1 over temperature). Allocate per-token fp32 stats (max initialized to negative
        # infinity) and two reusable fp32 tile buffers. Inside lm-head gather contexts (for ZeRO-3 sharded heads),
        # loop over token chunks; for each vocab chunk project both tiles and fold stats (Triton or torch). Compute
        # log-normalizers for the chunk; if 0 < beta < 1 do a second vocab pass with the JSD fold. Afterwards: entropy
        # = student log Z minus the exp-weighted mean logit; divergence for beta 0 is teacher-weighted mean gap minus
        # teacher log Z plus student log Z (aux = divergence); for beta 1 the student-weighted mean negative gap minus
        # student log Z plus teacher log Z (aux = divergence); otherwise jsd with aux = kl_student. Save inputs,
        # log-normalizers and aux for backward, plus dtypes, beta, mode (0, 1, 2) and kernel args; mark entropy
        # non-differentiable.
        # Inputs: ctx, student_hidden [N, H_s], student_weight [V, H_s], student_bias, teacher_hidden [N, H_t],
        # teacher_weight [V, H_t], teacher_bias, beta, temperature (1.0), student_logit_scale (1.0),
        # student_final_logit_softcapping (None), teacher_logit_scale (1.0), teacher_final_logit_softcapping (None).
        # Outputs: (divergence [N], entropy [N]) in fp32.
        # Exceptions: none explicit. Side effects: may gather sharded LM-head weights temporarily.
        ctx.set_materialize_grads(False)
        device = student_hidden.device
        N = student_hidden.shape[0]
        s_dtype, t_dtype = _compute_dtype(student_hidden), _compute_dtype(teacher_hidden)
        kernel_args = {
            "s_scale": student_logit_scale,
            "s_softcap": student_final_logit_softcapping or 1.0,
            "t_scale": teacher_logit_scale,
            "t_softcap": teacher_final_logit_softcapping or 1.0,
            "inv_t": 1 / temperature,
            "S_HAS_SOFTCAP": student_final_logit_softcapping is not None,
            "T_HAS_SOFTCAP": teacher_final_logit_softcapping is not None,
        }

        def zeros():
            # [FUNCTION DOC]
            # HELPER: Return a new fp32 zero vector of length N on the input device. Inputs: none. Outputs: tensor.
            # Exceptions: none. Side effects: allocation.
            return torch.zeros((N,), device=device, dtype=torch.float32)

        s_max, t_max = zeros() - math.inf, zeros() - math.inf
        s_sum, s_x_sum, t_sum, t_weighted_gap, s_weighted_gap = zeros(), zeros(), zeros(), zeros(), zeros()
        jsd, kl_student = zeros(), zeros()
        rows = min(N, TOKEN_CHUNK_SIZE)
        s_buf = torch.empty((rows, VOCAB_CHUNK_SIZE), device=device, dtype=torch.float32)
        t_buf = torch.empty((rows, VOCAB_CHUNK_SIZE), device=device, dtype=torch.float32)
        is_jsd = 0.0 < beta < 1.0

        with (
            maybe_gather_lm_head_ctx(student_weight, student_bias),
            maybe_gather_lm_head_ctx(teacher_weight, teacher_bias),
        ):
            vocab = student_weight.shape[0]
            for token_start in range(0, N, TOKEN_CHUNK_SIZE):
                token_end = min(token_start + TOKEN_CHUNK_SIZE, N)
                n = token_end - token_start
                h_s = student_hidden[token_start:token_end].to(s_dtype)
                h_t = teacher_hidden[token_start:token_end].to(t_dtype)
                sl = slice(token_start, token_end)
                for start in range(0, vocab, VOCAB_CHUNK_SIZE):
                    end = min(start + VOCAB_CHUNK_SIZE, vocab)
                    s_tile, t_tile = s_buf[:n, : end - start], t_buf[:n, : end - start]
                    _project(s_tile, h_s, student_weight, student_bias, start, end, s_dtype)
                    _project(t_tile, h_t, teacher_weight, teacher_bias, start, end, t_dtype)
                    fold = _stats_kernel[(n,)] if _use_triton(device) else _stats_tile
                    fold(
                        s_tile,
                        s_tile.stride(0),
                        t_tile,
                        t_tile.stride(0),
                        s_max[sl],
                        s_sum[sl],
                        s_x_sum[sl],
                        t_max[sl],
                        t_sum[sl],
                        t_weighted_gap[sl],
                        s_weighted_gap[sl],
                        end - start,
                        BLOCK_SIZE=_BLOCK_SIZE,
                        **kernel_args,
                    )
                s_log_z = s_max[sl] + torch.log(s_sum[sl])
                t_log_z = t_max[sl] + torch.log(t_sum[sl])
                if is_jsd:
                    for start in range(0, vocab, VOCAB_CHUNK_SIZE):
                        end = min(start + VOCAB_CHUNK_SIZE, vocab)
                        s_tile, t_tile = s_buf[:n, : end - start], t_buf[:n, : end - start]
                        _project(s_tile, h_s, student_weight, student_bias, start, end, s_dtype)
                        _project(t_tile, h_t, teacher_weight, teacher_bias, start, end, t_dtype)
                        fold = _jsd_kernel[(n,)] if _use_triton(device) else _jsd_tile
                        fold(
                            s_tile,
                            s_tile.stride(0),
                            t_tile,
                            t_tile.stride(0),
                            s_log_z,
                            t_log_z,
                            jsd[sl],
                            kl_student[sl],
                            end - start,
                            beta,
                            BLOCK_SIZE=_BLOCK_SIZE,
                            **kernel_args,
                        )

        s_log_z = s_max + torch.log(s_sum)
        t_log_z = t_max + torch.log(t_sum)
        entropy = s_log_z - s_x_sum / s_sum
        if beta == 0.0:
            divergence = t_weighted_gap / t_sum - t_log_z + s_log_z
            aux = divergence
        elif beta == 1.0:
            divergence = s_weighted_gap / s_sum - s_log_z + t_log_z
            aux = divergence
        else:
            divergence = jsd
            aux = kl_student

        ctx.save_for_backward(
            student_hidden,
            student_weight,
            student_bias,
            teacher_hidden,
            teacher_weight,
            teacher_bias,
            s_log_z,
            t_log_z,
            aux,
        )
        ctx.dtypes = s_dtype, t_dtype
        ctx.beta = beta
        ctx.mode = 0 if beta == 0.0 else 1 if beta == 1.0 else 2
        ctx.kernel_args = kernel_args
        ctx.mark_non_differentiable(entropy)
        return divergence, entropy

    @staticmethod
    def backward(ctx, grad_divergence: torch.Tensor | None, _):  # type: ignore
        # [FUNCTION DOC]
        # Behavior: return all None if the divergence received no gradient. Otherwise, inside lm-head gather contexts,
        # allocate fp32 accumulators for the student hidden, weight and bias gradients as required by
        # needs_input_grad; for each token chunk and vocab chunk, re-project both tiles, overwrite the student tile with
        # the per-logit gradient (Triton grid of rows by column blocks, or torch fallback), cast to the student compute
        # dtype and accumulate: hidden grad += tile times weight slice, weight grad += tile transposed times hidden,
        # bias grad += column sums. Return the gradients cast to the input dtypes and None for the nine remaining
        # inputs (teacher inputs and scalars get no gradient).
        # Inputs: ctx, grad_divergence ([N] or None), ignored entropy grad.
        # Outputs: tuple of 12 gradients. Exceptions: none explicit. Side effects: temporary gathers, allocations.
        (
            student_hidden,
            student_weight,
            student_bias,
            teacher_hidden,
            teacher_weight,
            teacher_bias,
            s_log_z,
            t_log_z,
            aux,
        ) = ctx.saved_tensors
        s_dtype, t_dtype = ctx.dtypes
        needs_hidden_grad, needs_weight_grad, needs_bias_grad = ctx.needs_input_grad[:3]
        N = student_hidden.shape[0]
        grad_hidden = grad_weight = grad_bias = None
        if grad_divergence is None:
            return (None,) * 12
        grad_divergence = grad_divergence.float().contiguous()
        with (
            maybe_gather_lm_head_ctx(student_weight, student_bias),
            maybe_gather_lm_head_ctx(teacher_weight, teacher_bias),
        ):
            vocab = student_weight.shape[0]
            # Always accumulate in fp32, even when the inputs are not
            if needs_hidden_grad:
                grad_hidden = torch.zeros(student_hidden.shape, device=student_hidden.device, dtype=torch.float32)
            if needs_weight_grad:
                grad_weight = torch.zeros(student_weight.shape, device=student_weight.device, dtype=torch.float32)
            if needs_bias_grad:
                grad_bias = torch.zeros(student_bias.shape, device=student_bias.device, dtype=torch.float32)
            rows = min(N, TOKEN_CHUNK_SIZE)
            s_buf = torch.empty((rows, VOCAB_CHUNK_SIZE), device=student_hidden.device, dtype=torch.float32)
            t_buf = torch.empty((rows, VOCAB_CHUNK_SIZE), device=student_hidden.device, dtype=torch.float32)
            for token_start in range(0, N, TOKEN_CHUNK_SIZE):
                token_end = min(token_start + TOKEN_CHUNK_SIZE, N)
                n = token_end - token_start
                sl = slice(token_start, token_end)
                h_s = student_hidden[sl].to(s_dtype)
                h_t = teacher_hidden[sl].to(t_dtype)
                for start in range(0, vocab, VOCAB_CHUNK_SIZE):
                    end = min(start + VOCAB_CHUNK_SIZE, vocab)
                    s_tile, t_tile = s_buf[:n, : end - start], t_buf[:n, : end - start]
                    _project(s_tile, h_s, student_weight, student_bias, start, end, s_dtype)
                    _project(t_tile, h_t, teacher_weight, teacher_bias, start, end, t_dtype)
                    fold = (
                        _backward_kernel[(n, triton.cdiv(end - start, _BLOCK_SIZE))]
                        if _use_triton(student_hidden.device)
                        else _backward_tile
                    )
                    fold(
                        s_tile,
                        s_tile.stride(0),
                        t_tile,
                        t_tile.stride(0),
                        s_log_z[sl],
                        t_log_z[sl],
                        aux[sl],
                        grad_divergence[sl],
                        end - start,
                        ctx.beta,
                        MODE=ctx.mode,
                        BLOCK_SIZE=_BLOCK_SIZE,
                        **ctx.kernel_args,
                    )
                    grad_tile = s_tile.to(s_dtype)
                    if grad_hidden is not None:
                        _addmm_fp32(grad_hidden[sl], grad_tile, student_weight[start:end].to(s_dtype))
                    if grad_weight is not None:
                        _addmm_fp32(grad_weight[start:end], grad_tile.t(), h_s)
                    if grad_bias is not None:
                        grad_bias[start:end] += s_tile.sum(dim=0, dtype=torch.float32)

        return (
            grad_hidden.to(student_hidden.dtype) if grad_hidden is not None else None,
            grad_weight.to(student_weight.dtype) if grad_weight is not None else None,
            grad_bias.to(student_bias.dtype) if grad_bias is not None else None,
            *(None,) * 9,
        )
