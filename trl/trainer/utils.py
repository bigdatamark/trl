# [MODULE DOC]
# Shared helper utilities used by the TRL trainers (trl.trainer.*) and some experimental code.
# Main contents: distributed env helpers (MASTER_ADDR/PORT), tensor helpers (pad, flush_left, nanstd, nanmin,
# nanmax, split_tensor_dict, shuffle_sequence_dict, pixel-value split/unsplit for VLMs), log-prob/entropy helpers
# (selective_log_softmax, entropy_from_logits, selective_log_softmax_and_entropy, with optional Triton kernels),
# the fused LM head patch (add_fused_lm_head and FusedCausalLMOutput), samplers (RepeatSampler and its streaming
# counterpart repeat_iterable_dataset), model/config helpers (create_model_from_path, get_quantization_config,
# get_peft_config, get_config_model_id, hash_module, use_adapter, disable_dropout_in_model), logging helpers
# (model cards, Comet/Trackio URLs, rich completion tables), async event-loop helpers and MFU/FLOPs estimation.
# Import-time side effects: conditional imports of comet_ml, the Triton fused kernel, peft and rich when
# available; creates a module logger; defines module constants (chunk size for the fused head, peak FLOPs table).

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

import asyncio
import copy
import functools
import hashlib
import importlib.resources as pkg_resources
import inspect
import json
import os
import random
import re
import socket
import threading
import types
from collections.abc import Callable, Mapping, Sequence, Sized
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from importlib.metadata import version
from itertools import accumulate
from typing import TypeVar

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, PartialState
from accelerate.logging import get_logger
from datasets import IterableDataset
from huggingface_hub import ModelCard, ModelCardData
from packaging.version import Version
from torch.distributed.tensor import DTensor
from torch.utils.data import Sampler
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
    is_comet_available,
    is_trackio_available,
)
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from transformers.utils import ModelOutput, is_peft_available, is_rich_available, is_triton_available

from ..kernels import ChunkedLogProbFunction as _ChunkedLogProbFunction
from ..kernels.chunked_logprob import _use_triton
from ..trainer.model_config import ModelConfig


if is_comet_available():
    import comet_ml


if is_triton_available():
    from ..kernels import selective_log_softmax_and_entropy as _fused_logprob_entropy


if is_peft_available():
    from peft import LoraConfig, PeftConfig, PeftModel


if is_rich_available():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text


logger = get_logger(__name__)


def _is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    # [FUNCTION DOC]
    # HELPER: Checks whether a TCP port can be bound.
    # Behavior: opens an IPv4 stream socket with SO_REUSEADDR and tries to bind host:port.
    # Inputs: port, host (default 127.0.0.1). Outputs: True if bind succeeds, False on OSError.
    # Exceptions: none (OSError swallowed). Side effects: briefly binds then closes a socket.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def _find_free_port() -> int:
    # [FUNCTION DOC]
    # HELPER: Returns a free port, preferring 29500, 23456, 12355, 12345 in that order, else an OS-assigned ephemeral port.
    # Inputs: none. Outputs: int port. Exceptions: OSError if the ephemeral bind fails. Side effects: temporary sockets.
    candidates = (29500, 23456, 12355, 12345)
    for p in candidates:
        if _is_port_free(p):
            return p
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def ensure_master_addr_port(addr: str | None = None, port: int | None = None) -> None:
    # [FUNCTION DOC]
    # Behavior: sets MASTER_ADDR to the existing env value, else addr, else "localhost". For the port: if port is None
    # and MASTER_PORT env is set to something other than empty/"0"/"auto", parses it as int (ignored if not numeric).
    # Then sets MASTER_PORT to a free port when port is None or 0, else to port.
    # Inputs: addr, port (optional). Outputs: none.
    # Exceptions: none. Side effects: mutates os.environ; probes sockets.
    """
    Ensure `MASTER_ADDR`/`MASTER_PORT` are set safely.

    - Respects existing environment variables.
    - Defaults `MASTER_ADDR` to localhost if unset.
    - Chooses a free TCP port if `MASTER_PORT` is unset to avoid collisions.
    - If `MASTER_PORT` is set to `"0"` or `"auto"`, it is resolved to a free port.
    """
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR") or addr or "localhost"

    env_port = os.environ.get("MASTER_PORT", "").strip().lower()
    if port is None and env_port not in {"", "0", "auto"}:
        try:
            port = int(env_port)
        except ValueError:
            pass

    os.environ["MASTER_PORT"] = str(_find_free_port() if port in (None, 0) else port)


def pad(
    tensors: list[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = "right",
    pad_to_multiple_of: int | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: stacks tensors of varying shapes into one tensor of shape (count, max shape per dim), filled with
    # padding_value. The first dimension is rounded up to a multiple of pad_to_multiple_of when given. Each tensor is
    # placed at the start (right padding) or end (left padding) along the first dimension and at the start along the
    # others. Uses the first tensor's dtype and device.
    # Inputs: tensors list, padding_value (0), padding_side ("right"), pad_to_multiple_of.
    # Outputs: padded tensor. Exceptions: ValueError for an invalid padding_side. Side effects: none.
    """
    Pads a list of tensors to the same shape along the first dimension.

    Args:
        tensors (`list[torch.Tensor]`):
            List of input tensors to pad.
        padding_value (`int`):
            Value to use for padding. Default is 0.
        padding_side (`str`):
            Side on which to add padding. Must be 'left' or 'right'. Default is 'right'.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

    Returns:
        `torch.Tensor`:
            A single tensor containing the padded tensors.

    Examples:
    ```python
    >>> import torch

    >>> pad([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
    tensor([[1, 2, 3],
            [4, 5, 0]])

    >>> pad([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])])
    tensor([[[1, 2],
            [3, 4]],
            [[5, 6],
            [0, 0]]])
    ```
    """
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Apply pad_to_multiple_of to the first (sequence) dimension
    if pad_to_multiple_of is not None:
        remainder = output_shape[0] % pad_to_multiple_of
        if remainder != 0:
            output_shape[0] += pad_to_multiple_of - remainder

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        if padding_side == "left":
            seq_start = output_shape[0] - t.shape[0]
        elif padding_side == "right":
            seq_start = 0
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        # Define the slices
        seq_slice = slice(seq_start, seq_start + t.shape[0])
        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output


def disable_dropout_in_model(model: torch.nn.Module) -> None:
    # [FUNCTION DOC]
    # Behavior: sets p to 0 on every torch Dropout module in the model.
    # Inputs: model. Outputs: none. Exceptions: none. Side effects: mutates the model's dropout modules.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def maybe_gather_lm_head_ctx(*params: torch.nn.Parameter):
    # [FUNCTION DOC]
    # Behavior: returns a nullcontext unless DeepSpeed ZeRO-3 is enabled; then deduplicates the given parameters by
    # identity, drops None and already-available ones, and returns a deepspeed GatheredParameters context over the
    # rest (nullcontext if none remain).
    # Inputs: params (parameters or None). Outputs: a context manager.
    # Exceptions: ImportError if deepspeed is missing under ZeRO-3. Side effects: gathering occurs when the context is entered.
    """
    Context manager that allgathers ZeRO-3 partitioned `lm_head` weight/bias for a fused loss.

    Fused losses (e.g. Liger) read `lm_head.weight` directly and hand it to the kernel without ever calling the
    `lm_head` module. Under DeepSpeed ZeRO-3 every parameter is sharded to `numel 0` and only re-materialized inside
    its owning module's forward hook, so the head's gather hook never fires and the kernel receives an empty weight.
    This gathers the given parameters for the duration of a direct access, so the matmul sees the full weight. Fused
    losses compute and stash their gradients during the forward; custom autograd functions that recompute during the
    backward must enter this context again. The backward's gradient reduction relies on the ZeRO-3 pre-forward hooks
    being armed, so the caller must run the loss inside the model's forward (e.g. via `_ForwardRedirection`).

    Returns a null context when ZeRO-3 is not enabled, or when the parameters are already gathered — with tied
    embeddings `embed_tokens` keeps the weight `AVAILABLE`, and re-partitioning it on exit would collide with
    `embed_tokens`' active-submodule tracking.

    Args:
        *params (`torch.nn.Parameter`):
            Parameters to gather (e.g. `lm_head.weight` and `lm_head.bias`). `None` values are ignored.
    """
    from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

    if not is_deepspeed_zero3_enabled():
        return nullcontext()

    import deepspeed
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    # Deduplicate by identity: with a shared reference (e.g. PEFT with no separate `ref_model`, or tied embeddings)
    # the same parameter is passed more than once, and ZeRO-3 mishandles duplicate entries in the gather list.
    to_gather = {id(p): p for p in params if p is not None and p.ds_status != ZeroParamStatus.AVAILABLE}
    if not to_gather:
        return nullcontext()
    return deepspeed.zero.GatheredParameters(list(to_gather.values()))


def get_callable_name(func: Callable) -> str:
    # [FUNCTION DOC]
    # Behavior: unwraps nested functools.partial objects and returns the function's name, or the class name for
    # objects without a name attribute (callable instances).
    # Inputs: func. Outputs: str. Exceptions: none. Side effects: none.
    """
    Return a display name for a callable, supporting the picklable reward forms: module-level functions,
    [`functools.partial`](https://docs.python.org/3/library/functools.html#functools.partial) (unwrapped to the wrapped
    function's name), and callable class instances (which fall back to their class name).
    """
    while isinstance(func, functools.partial):
        func = func.func
    return getattr(func, "__name__", type(func).__name__)


def is_async_callable(func: Callable) -> bool:
    # [FUNCTION DOC]
    # Behavior: unwraps partials, then returns true if the function or its call method is a coroutine function.
    # Inputs: func. Outputs: bool. Exceptions: none. Side effects: none.
    """
    Return whether calling `func` returns a coroutine, for the same forms as [`get_callable_name`]: module-level
    functions, [`functools.partial`](https://docs.python.org/3/library/functools.html#functools.partial) (unwrapped to
    the wrapped callable), and callable class instances (which carry the `async` on their `__call__`).

    `inspect.iscoroutinefunction` alone covers only the first two: it inspects the instance itself, not its `__call__`,
    so an async callable class reads as synchronous and its coroutine is never awaited.
    """
    while isinstance(func, functools.partial):
        func = func.func
    return inspect.iscoroutinefunction(func) or inspect.iscoroutinefunction(func.__call__)


def get_quantization_config(model_args: ModelConfig) -> BitsAndBytesConfig | None:
    # [FUNCTION DOC]
    # Behavior: builds a BitsAndBytesConfig from ModelConfig: 4-bit (compute dtype = model dtype, quant type, nested
    # quant, quant storage) when load_in_4bit; 8-bit when load_in_8bit; otherwise None.
    # Inputs: model_args. Outputs: BitsAndBytesConfig or None. Exceptions: from BitsAndBytesConfig. Side effects: none.
    if model_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model_args.dtype,  # For consistency with model weights, we use the same value as `dtype`
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
            bnb_4bit_quant_storage=model_args.bnb_4bit_quant_storage,
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config


def get_peft_config(model_args: ModelConfig) -> "PeftConfig | None":
    # [FUNCTION DOC]
    # Behavior: returns None if use_peft is False; requires peft; builds a LoraConfig from ModelConfig (task type, r,
    # target modules, alpha, dropout, bias "none", rslora, dora, modules_to_save), passing target_parameters only when set.
    # Inputs: model_args. Outputs: LoraConfig or None.
    # Exceptions: ValueError when peft is not installed. Side effects: none.
    if model_args.use_peft is False:
        return None

    if not is_peft_available():
        raise ValueError(
            "You need to have PEFT library installed in your environment, make sure to install `peft`. "
            "Make sure to run `pip install -U peft`."
        )

    # `target_parameters` was added in PEFT 0.17.0, so only pass it when the user asked for it
    lora_config_kwargs = {}
    if model_args.lora_target_parameters is not None:
        lora_config_kwargs["target_parameters"] = model_args.lora_target_parameters

    peft_config = LoraConfig(
        task_type=model_args.lora_task_type,
        r=model_args.lora_r,
        target_modules=model_args.lora_target_modules,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        use_rslora=model_args.use_rslora,
        use_dora=model_args.use_dora,
        modules_to_save=model_args.lora_modules_to_save,
        **lora_config_kwargs,
    )

    return peft_config


def generate_model_card(
    base_model: str | None,
    model_name: str,
    hub_model_id: str,
    dataset_name: str | None,
    tags: list[str],
    wandb_url: str | None,
    trackio_url: str | None,
    trainer_name: str,
    trainer_citation: str | None = None,
    template_file: str | None = None,
    paper_title: str | None = None,
    paper_id: str | None = None,
    comet_url: str | None = None,
) -> ModelCard:
    # [FUNCTION DOC]
    # Behavior: builds ModelCardData (library transformers, tags prefixed with generated_from_trainer) and renders the
    # template from trl/templates (default lm_model_card.md) with run URLs, trainer info, paper info and installed
    # versions of trl, transformers, torch, datasets and tokenizers.
    # Inputs: see signature. Outputs: huggingface_hub ModelCard.
    # Exceptions: PackageNotFoundError if a version lookup fails; template errors. Side effects: reads the template file.
    """
    Generate a [`~huggingface_hub.ModelCard`] from a template.

    Args:
        base_model (`str` or `None`):
            Base model name.
        model_name (`str`):
            Model name.
        hub_model_id (`str`):
            Hub model ID as `username/model_id`.
        dataset_name (`str` or `None`):
            Dataset name.
        tags (`list[str]`):
            Tags.
        wandb_url (`str` or `None`):
            Weights & Biases run URL.
        trackio_url (`str` or `None`):
            Trackio Space URL.
        trainer_name (`str`):
            Trainer name.
        trainer_citation (`str`, *optional*):
            Trainer citation as a BibTeX entry.
        template_file (`str`, *optional*):
            Template file name located in the `trl/templates` directory. Defaults to `lm_model_card.md`.
        paper_title (`str`, *optional*):
            Paper title.
        paper_id (`str`, *optional*):
            ArXiv paper ID as `YYMM.NNNNN`.
        comet_url (`str`, *optional*):
            Comet experiment URL.

    Returns:
        [`~huggingface_hub.ModelCard`]:
            A ModelCard object.
    """
    card_data = ModelCardData(
        base_model=base_model,
        datasets=dataset_name,
        library_name="transformers",
        licence="license",
        model_name=model_name,
        tags=["generated_from_trainer", *tags],
    )
    template_file = template_file or "lm_model_card.md"
    card = ModelCard.from_template(
        card_data,
        template_path=str(pkg_resources.files("trl").joinpath(f"templates/{template_file}")),
        base_model=base_model,
        model_name=model_name,
        hub_model_id=hub_model_id,
        dataset_name=dataset_name,
        wandb_url=wandb_url,
        trackio_url=trackio_url,
        comet_url=comet_url,
        trainer_name=trainer_name,
        trainer_citation=trainer_citation,
        paper_title=paper_title,
        paper_id=paper_id,
        trl_version=version("trl"),
        transformers_version=version("transformers"),
        pytorch_version=version("torch"),
        datasets_version=version("datasets"),
        tokenizers_version=version("tokenizers"),
    )
    return card


def get_comet_experiment_url() -> str | None:
    # [FUNCTION DOC]
    # Behavior: returns the running Comet experiment URL, or None when comet is unavailable or no experiment runs.
    # Inputs: none. Outputs: str or None. Exceptions: none. Side effects: none.
    """
    If Comet integration is enabled, return the URL of the current Comet experiment; otherwise, return `None`.
    """
    if not is_comet_available():
        return None

    if comet_ml.get_running_experiment() is not None:
        return comet_ml.get_running_experiment().url

    return None


def get_trackio_space_url() -> str | None:
    # [FUNCTION DOC]
    # Behavior: if trackio is installed and has a current run with a Space id, returns the Space URL (slash replaced
    # by dash in the id) with project and run name query parameters; otherwise None.
    # Inputs: none. Outputs: str or None. Exceptions: none. Side effects: none.
    """
    If Trackio integration is enabled, return the URL of the current Trackio Space; otherwise, return `None`.
    """
    if not is_trackio_available():
        return None

    from trackio import context_vars

    run = context_vars.current_run.get()
    if run is None:
        return None
    space_id = run._space_id
    if space_id is None:
        return None
    space_id = space_id.replace("/", "-")
    project = run.project
    name = run.name
    return f"https://{space_id}.hf.space?project={project}&runs={name}&sidebar=collapsed"


def log_table_to_comet_experiment(name: str, table: pd.DataFrame) -> None:
    # [FUNCTION DOC]
    # Behavior: logs the DataFrame as a table file named name to the running Comet experiment, if one exists.
    # Inputs: name, table. Outputs: none.
    # Exceptions: ModuleNotFoundError if comet is not installed. Side effects: network logging to Comet.
    """
    If Comet integration is enabled logs a table to the Comet experiment if it is currently running.

    Args:
        name (`str`):
            Table name.
        table (`pandas.DataFrame`):
            The Pandas DataFrame containing the table to log.
    """
    if not is_comet_available():
        raise ModuleNotFoundError("The comet-ml is not installed. Please install it first: pip install comet-ml")

    experiment = comet_ml.get_running_experiment()
    if experiment is not None:
        experiment.log_table(tabular_data=table, filename=name)


def flush_left(mask: torch.Tensor, *tensors: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
    # [FUNCTION DOC]
    # Behavior: for 2-D mask and same-shape tensors, rotates each row left by the position of its first nonzero mask
    # value (cyclic gather), then truncates all columns starting at the first column whose mask is zero in every row.
    # Works on clones.
    # Inputs: mask, tensors. Outputs: the flushed mask alone, or a tuple of mask and tensors.
    # Exceptions: none. Side effects: none.
    """
    Shift non-zero elements in the mask and corresponding tensors to the left.

    This function operates on a binary mask and any number of additional tensors with the same dimensions as the mask.
    For each row, non-zero values are shifted to the leftmost positions. Then, columns that contain only zeros across
    all rows are truncated from the mask and tensors. Visually, this operation can be represented as follows:

    ```
    [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
     [0, x, x, x, 0, 0]]       [x, x, x, 0]]
    ```

    Args:
        mask (`torch.Tensor`):
            2D tensor (binary mask) with shape `(N, M)`.
        *tensors (`torch.Tensor`):
            One or more 2D tensors with the same shape as `mask`. These tensors will be processed alongside `mask`,
            with non-zero values shifted and excess zero columns truncated in the same manner.

    Returns:
        `torch.Tensor`:
            Updated binary mask with non-zero values flushed to the left and trailing zero columns removed.
        `*torch.Tensor`
            Updated tensors, processed in the same way as the mask.

    Example:
    ```python
    >>> mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 0, 0]])
    >>> tensor = torch.tensor([[9, 9, 2, 3, 4], [9, 5, 6, 9, 9]])
    >>> new_mask, new_tensor = flush_left(mask, tensor)
    >>> print(new_mask)
    tensor([[1, 1, 1],
            [1, 1, 0]])

    >>> print(new_tensor)
    tensor([[2, 3, 4],
            [5, 6, 0]])
    ```
    """
    _, M = mask.shape

    # Create copy of mask and tensors
    mask_copy = mask.clone()
    tensors = [t.clone() for t in tensors]

    # Shift non-zero values to the left
    first_non_zero = mask_copy.argmax(dim=1)
    pos = torch.arange(M, device=mask_copy.device).unsqueeze(0)
    idx_roll = (pos + first_non_zero.unsqueeze(1)) % M
    mask_roll = mask_copy.gather(1, idx_roll)
    rolled_tensors = [t.gather(1, idx_roll) for t in tensors]

    # Truncate trailing columns that are all zeros in mask_roll
    col_sums = mask_roll.sum(dim=0)
    empty_cols = col_sums == 0
    first_empty_col = int(empty_cols.to(torch.int8).argmax()) if empty_cols.any() else M
    flushed_mask = mask_roll[:, :first_empty_col]
    flushed_tensors = [t[:, :first_empty_col] for t in rolled_tensors]

    if not flushed_tensors:
        return flushed_mask
    return flushed_mask, *flushed_tensors


def _supports_trl_loss_kernel(
    logits: torch.Tensor, index: torch.Tensor | None = None, row_mask: torch.Tensor | None = None
) -> bool:
    # [FUNCTION DOC]
    # HELPER: Decides whether the Triton log-prob/entropy kernel can handle the inputs.
    # Behavior: true when logits are on cuda/xpu, fp16/bf16/fp32, 1 to 3 dims, contiguous last dim, nonempty vocab;
    # if index is given it must match the logits leading shape, be int32/int64 and on the same device; if row_mask is
    # given, index must exist and row_mask must match its shape, be bool/int32/int64 and on the same device.
    # Inputs: logits, index, row_mask. Outputs: bool. Exceptions: none. Side effects: none.
    supported = (
        logits.device.type in ("cuda", "xpu")
        and logits.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and 1 <= logits.ndim <= 3
        and logits.stride(-1) == 1
        and logits.shape[-1] > 0
    )
    if index is not None:
        supported = (
            supported
            and index.shape == logits.shape[:-1]
            and index.dtype in (torch.int32, torch.int64)
            and index.device == logits.device
        )
    if row_mask is not None:
        supported = (
            supported
            and index is not None
            and row_mask.shape == index.shape
            and row_mask.dtype in (torch.bool, torch.int32, torch.int64)
            and row_mask.device == logits.device
        )
    return supported


def selective_log_softmax(
    logits: torch.Tensor,
    index: torch.Tensor,
    temperature: float = 1.0,
    row_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: gathers log-softmax values at index without materializing the full log-softmax when possible.
    # Uses the fused Triton kernel when available and supported. Otherwise divides logits by temperature, unsqueezes
    # an index of rank one less than logits, and for fp32/fp64 subtracts a per-row logsumexp from gathered logits;
    # for other dtypes computes log_softmax row by row and gathers. Squeezes back and zeroes rows where row_mask is 0.
    # Inputs: logits, index, temperature (1.0), row_mask. Outputs: tensor shaped like index.
    # Exceptions: ValueError when temperature is not positive. Side effects: none.
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    # for index with shape (...):
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    # for index with shape (..., K):
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(..., K)` or `(...)`, specifying the positions to gather from the log-softmax
            output. When the last case is used, `K` log-probabilities are gathered per position (e.g. for top-K)
        temperature (`float`, *optional*, defaults to `1.0`):
            Temperature used to scale the logits.
        row_mask (`torch.Tensor`, *optional*):
            Mask with the same shape as `index`. Rows containing zero are skipped and return zero log-probability.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if is_triton_available() and _supports_trl_loss_kernel(logits, index, row_mask):
        logprobs, _ = _fused_logprob_entropy(logits, index, temperature=temperature, row_mask=row_mask)
        return logprobs

    if temperature != 1.0:
        logits = logits / temperature
    squeeze = index.ndim == logits.ndim - 1
    if squeeze:
        index = index.unsqueeze(-1)

    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values.unsqueeze(-1)  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index, strict=True):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)

    if squeeze:
        per_token_logps = per_token_logps.squeeze(-1)
    if row_mask is not None:
        per_token_logps = per_token_logps.masked_fill(row_mask == 0, 0.0)

    return per_token_logps


def entropy_from_logits(logits: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: uses the Triton kernel (with a dummy zero index) when supported; otherwise flattens leading dims,
    # computes entropy chunk by chunk (chunk_size rows) from log_softmax, and reshapes to the leading shape.
    # Inputs: logits, chunk_size (128). Outputs: entropy tensor. Exceptions: none. Side effects: none.
    """
    Compute the Shannon entropy (in nats) for each row of *logits* in a memory-efficient way.

    Instead of materializing the full softmax for all rows at once, the logits are flattened to shape (N, num_classes),
    where N is the product of all leading dimensions. Computation is then performed in chunks of size `chunk_size`
    along this flattened dimension, reducing peak memory usage. The result is reshaped back to match the input's
    leading dimensions.

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`. Entropy is taken along the last axis; all leading dimensions
            are preserved in the output.
        chunk_size (`int`, *optional*, defaults to `128`):
            Number of rows from the flattened logits to process per iteration. Smaller values reduce memory usage at
            the cost of more iterations.

    Returns:
        `torch.Tensor`:
            Entropy values with shape `logits.shape[:-1]`.
    """
    if is_triton_available() and _supports_trl_loss_kernel(logits):
        index = torch.zeros(logits.shape[:-1], device=logits.device, dtype=torch.long)
        _, entropy = _fused_logprob_entropy(logits, index)
        return entropy

    original_shape = logits.shape[:-1]  # all dims except num_classes
    num_classes = logits.shape[-1]

    # Flatten all leading dimensions into one
    flat_logits = logits.reshape(-1, num_classes)

    entropies = []
    for chunk in flat_logits.split(chunk_size, dim=0):
        logps = F.log_softmax(chunk, dim=-1)
        chunk_entropy = -(torch.exp(logps) * logps).sum(-1)
        entropies.append(chunk_entropy)

    entropies = torch.cat(entropies, dim=0)
    return entropies.reshape(original_shape)


def strip_images_from_messages(messages):
    # [FUNCTION DOC]
    # Behavior: returns non-list input unchanged; for a list of messages, replaces each image content block with a
    # block holding only its type, copying modified messages (originals untouched).
    # Inputs: messages. Outputs: stripped conversation. Exceptions: none. Side effects: none.
    """
    Drop the image objects from a conversation, keeping its structure.

    Each `{"type": "image", "image": <PIL.Image>}` content block becomes `{"type": "image"}`, so the conversation can
    be logged or serialized. Messages with string content are returned unchanged.

    Args:
        messages (`list[dict]` or `str`):
            Conversation to strip. Anything other than a list is returned as is.

    Returns:
        `list[dict]` or `str`: The conversation without the image objects.
    """
    if not isinstance(messages, list):
        return messages
    result = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            content = [{"type": block["type"]} if block.get("type") == "image" else block for block in content]
            msg = {**msg, "content": content}
        result.append(msg)
    return result


def selective_log_softmax_and_entropy(
    logits: torch.Tensor,
    index: torch.Tensor,
    entropy_requires_grad: bool = True,
    temperature: float = 1.0,
    row_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC]
    # Behavior: computes selected log-probs and entropy together. Triton path when supported (entropy detached if
    # entropy_requires_grad is false). Otherwise scales by temperature, calls selective_log_softmax (with row_mask) and
    # entropy_from_logits (under no_grad if gradients not required), zeroing entropy on masked rows.
    # Inputs: logits, index, entropy_requires_grad (True), temperature (1.0), row_mask.
    # Outputs: tuple of log-probs and entropy. Exceptions: ValueError for non-positive temperature. Side effects: none.
    """Compute selected log-probabilities and entropy in one pass over the logits.

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Indices of shape `(...)` selecting one log-probability per row.
        entropy_requires_grad (`bool`, *optional*, defaults to `True`):
            Whether gradients should flow through the entropy output.
        temperature (`float`, *optional*, defaults to `1.0`):
            Temperature used to scale the logits.
        row_mask (`torch.Tensor`, *optional*):
            Mask with the same shape as `index`. Rows containing zero are skipped and return zero outputs.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`:
            The selected log-probabilities and per-row entropies.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if is_triton_available() and _supports_trl_loss_kernel(logits, index, row_mask):
        logprobs, entropy = _fused_logprob_entropy(logits, index, temperature=temperature, row_mask=row_mask)
        return logprobs, entropy if entropy_requires_grad else entropy.detach()
    if temperature != 1.0:
        logits = logits / temperature
    logprobs = selective_log_softmax(logits, index, row_mask=row_mask)
    if entropy_requires_grad:
        entropy = entropy_from_logits(logits)
    else:
        with torch.no_grad():
            entropy = entropy_from_logits(logits)
    if row_mask is not None:
        entropy = entropy.masked_fill(row_mask == 0, 0.0)
    return logprobs, entropy


def print_prompt_completions_sample(
    prompts: list,
    completions: list,
    rewards: dict[str, list[float]],
    advantages: list[float] | None,
    step: int,
    num_samples: int = None,
    extra: dict[str, list] | None = None,
) -> None:
    # [FUNCTION DOC]
    # Behavior: prints a rich table inside a panel titled with the step: columns Prompt, Completion, one per reward,
    # Advantage (if given), then extra columns. If num_samples is at least the count, shows all; if not positive,
    # prints nothing; otherwise samples that many random rows. Rewards and advantages formatted with 2 decimals.
    # Inputs: prompts, completions, rewards, advantages, step, num_samples, extra. Outputs: none.
    # Exceptions: ImportError when rich is missing. Side effects: console output; uses Python random.
    """
    Print out a sample of model completions to the console with multiple reward metrics.

    This function creates a nicely formatted table showing prompt-completion pairs, useful for monitoring model outputs
    during training. It requires the `rich` library to be installed.

    Args:
        prompts (`list`):
            List of prompts. Can be either strings or lists of messages.
        completions (`list`):
            List of completions corresponding to the prompts. Can be either strings or lists of messages.
        rewards (`dict[str, list[float]]`):
            Dictionary where keys are reward names and values are lists of rewards.
        advantages (`list[float]` or `None`):
            List of advantages corresponding to the prompts and completions. If `None`, the advantage column is omitted
            (e.g. for distillation, which has no advantages).
        step (`int`):
            Current training step number, used in the output title.
        num_samples (`int`, *optional*):
            Number of random samples to display. If `None` (default), all items will be displayed.
        extra (`dict[str, list]`, *optional*):
            Additional columns to display after the advantage column. Keys are column names and values are lists of
            per-completion data (strings or any value convertible to string). Typically populated via `log_extra` in
            reward functions. If `None` (default), no extra columns are shown.

    Example:
    ```python
    >>> from trl.trainer.utils import print_prompt_completions_sample

    >>> prompts = ["The sky is", "The sun is"]
    >>> completions = [" blue.", " in the sky."]
    >>> rewards = {"Correctness": [0.123, 0.456], "Format": [0.789, 0.101]}
    >>> advantages = [0.987, 0.654]
    >>> extra = {"source": ["dataset_A", "dataset_B"]}
    >>> print_prompt_completions_sample(prompts, completions, rewards, advantages, 42, extra=extra)
    ╭────────────────────────────────── Step 42 ───────────────────────────────────╮
    │ ┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓ │
    │ ┃ Prompt     ┃ Completion   ┃ Correctness ┃ Format ┃ Advantage ┃ source    ┃ │
    │ ┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩ │
    │ │ The sky is │  blue.       │        0.12 │   0.79 │      0.99 │ dataset_A │ │
    │ ├────────────┼──────────────┼─────────────┼────────┼───────────┼───────────┤ │
    │ │ The sun is │  in the sky. │        0.46 │   0.10 │      0.65 │ dataset_B │ │
    │ └────────────┴──────────────┴─────────────┴────────┴───────────┴───────────┘ │
    ╰──────────────────────────────────────────────────────────────────────────────╯
    ```
    """
    if not is_rich_available():
        raise ImportError(
            "The function `print_prompt_completions_sample` requires the `rich` library. Please install it with "
            "`pip install rich`."
        )
    console = Console()
    table = Table(show_header=True, header_style="bold white", expand=True)

    extra = extra or {}

    # Add columns
    table.add_column("Prompt", style="bright_yellow")
    table.add_column("Completion", style="bright_green")
    for reward_name in rewards.keys():
        table.add_column(reward_name, style="bold cyan", justify="right")
    if advantages is not None:
        table.add_column("Advantage", style="bold magenta", justify="right")
    for extra_name in extra.keys():
        table.add_column(extra_name, style="bright_white")

    def format_entry(entry) -> Text:
        # [FUNCTION DOC]
        # HELPER: Renders a cell. For a list of message dicts: role in upper case, reasoning_content/thinking in dim
        # italics, content text (image blocks shown as [IMAGE]), and tool calls as name(arg=value, ...) with JSON-string
        # arguments parsed; messages separated by blank lines. Anything else is rendered with str.
        # Inputs: entry. Outputs: rich Text. Exceptions: JSON decode errors for malformed tool arguments. Side effects: none.
        t = Text()
        if isinstance(entry, list) and all(isinstance(m, dict) for m in entry):
            for j, msg in enumerate(entry):
                role = msg.get("role", "")
                t.append(f"{role.upper()}\n", style="bold red")
                if "reasoning_content" in msg or "thinking" in msg:
                    reasoning = msg.get("reasoning_content") or msg.get("thinking")
                    if reasoning:
                        t.append(reasoning, style="italic dim white")
                        t.append("\n")
                if "content" in msg:
                    content = msg["content"]
                    if isinstance(content, list):
                        # VLM format: content is a list of typed blocks, e.g.
                        # [{"type": "image", ...}, {"type": "text", "text": "..."}]
                        for block in content:
                            if block.get("type") == "text":
                                t.append(block["text"])
                            elif block.get("type") == "image":
                                t.append("[IMAGE]", style="bold cyan")
                    elif isinstance(content, str):
                        t.append(content)
                if "tool_calls" in msg:
                    for k, tc in enumerate(msg["tool_calls"]):
                        if k > 0:
                            t.append("\n")
                        fn = tc.get("function", {})
                        raw = fn.get("arguments", {})
                        args = raw if isinstance(raw, dict) else json.loads(raw)
                        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                        t.append(f"{fn.get('name', '?')}({args_str})")
                if j < len(entry) - 1:
                    t.append("\n\n")
        else:
            t.append(str(entry))
        return t

    # Some basic input validation
    if num_samples is not None:
        if num_samples >= len(prompts):
            num_samples = None
        elif num_samples <= 0:
            return

    # Subsample data if num_samples is specified
    if num_samples is not None:
        indices = random.sample(range(len(prompts)), num_samples)
        prompts = [prompts[i] for i in indices]
        completions = [completions[i] for i in indices]
        rewards = {key: [val[i] for i in indices] for key, val in rewards.items()}
        if advantages is not None:
            advantages = [advantages[i] for i in indices]
        extra = {key: [val[i] for i in indices] for key, val in extra.items()}

    for i in range(len(prompts)):
        reward_values = [f"{rewards[key][i]:.2f}" for key in rewards.keys()]  # 2 decimals
        extra_values = [format_entry(extra[key][i]) for key in extra.keys()]
        row = [format_entry(prompts[i]), format_entry(completions[i]), *reward_values]
        if advantages is not None:
            row.append(f"{advantages[i]:.2f}")
        row.extend(extra_values)
        table.add_row(*row)
        table.add_section()  # Adds a separator between rows

    panel = Panel(table, expand=False, title=f"Step {step}", border_style="bold white")
    console.print(panel)


class RepeatSampler(Sampler):
    # [CLASS DOC]
    # torch Sampler yielding dataset indices so each prompt appears mini_repeat_count times consecutively, in batches
    # of batch_size unique indices, each batch repeated repeat_count times. Incomplete trailing batches are dropped.
    # State: data_source, mini_repeat_count, batch_size, repeat_count, num_samples, shuffle, seed, and a private
    # torch Generator when shuffling (seeded if seed given). Used by GRPO/RLOO-style trainers to keep prompt groups
    # together across processes.
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the dataset.
        seed (`int`, *optional*):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        # [FUNCTION DOC]
        # Behavior: stores the arguments, records the dataset length, and creates a local Generator (manually seeded when
        # seed is given) if shuffle is true.
        # Inputs: data_source, mini_repeat_count, batch_size (1), repeat_count (1), shuffle (True), seed.
        # Outputs: none. Exceptions: TypeError if data_source has no length. Side effects: none.
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        # [FUNCTION DOC]
        # Behavior: permutes indices with the generator (or keeps order), chunks into batch_size groups, drops a short last
        # chunk, and for each chunk yields, repeat_count times, every index mini_repeat_count times in a row.
        # Inputs: none. Outputs: generator of ints. Exceptions: none. Side effects: advances the generator state.
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        else:
            indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Behavior: full batches times batch_size times mini_repeat_count times repeat_count.
        # Inputs: none. Outputs: int. Exceptions: none. Side effects: none.
        return (self.num_samples // self.batch_size) * self.batch_size * self.mini_repeat_count * self.repeat_count


def repeat_iterable_dataset(
    dataset: IterableDataset, mini_repeat_count: int, batch_size: int = 1, repeat_count: int = 1
):
    # [FUNCTION DOC]
    # Behavior: streaming equivalent of RepeatSampler without shuffling: applies a batched map with batch_size records
    # per batch that drops incomplete batches and emits each batch repeat_count times with every record repeated
    # mini_repeat_count times consecutively (deep copies).
    # Inputs: dataset, mini_repeat_count, batch_size (1), repeat_count (1). Outputs: mapped IterableDataset.
    # Exceptions: none of its own. Side effects: none (lazy).
    """
    Streaming counterpart of [`RepeatSampler`] for [`~datasets.IterableDataset`].

    An [`~datasets.IterableDataset`] cannot be indexed, so a sampler cannot be attached to it. Instead of reordering
    indices, this reorders the *stream* itself with [`~datasets.IterableDataset.map`], producing records in exactly the
    same order that [`RepeatSampler`] yields indices for a map-style dataset with the same arguments. Because the
    transform stays chained to `dataset`, `set_epoch` propagates to an upstream [`~datasets.IterableDataset.shuffle`]
    and the order is reshuffled every epoch, as [`RepeatSampler`] does.

    Shuffling is intentionally left out: it is handled upstream via [`~datasets.IterableDataset.shuffle`] (buffered
    shuffling), the streaming equivalent of the full permutation done by [`RepeatSampler`].

    Args:
        dataset (`datasets.IterableDataset`):
            Dataset to stream from.
        mini_repeat_count (`int`):
            Number of times to repeat each record per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique records per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full batch.

    Returns:
        `datasets.IterableDataset`: Dataset yielding the records of `dataset`, repeated and reordered.

    Example:
    ```python
    >>> from datasets import IterableDataset

    >>> dataset = IterableDataset.from_generator(lambda: ({"x": i} for i in range(7)))
    >>> repeated = repeat_iterable_dataset(dataset, mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> [record["x"] for record in repeated]
    [0, 0, 1, 1, 2, 2,
     0, 0, 1, 1, 2, 2,
     0, 0, 1, 1, 2, 2,
     0, 0, 1, 1, 2, 2,
     3, 3, 4, 4, 5, 5,
     3, 3, 4, 4, 5, 5,
     3, 3, 4, 4, 5, 5,
     3, 3, 4, 4, 5, 5]
    ```
    """

    def repeat_batch(batch: dict) -> dict:
        # [FUNCTION DOC]
        # HELPER: Map function: given a column-to-list batch, returns empty columns if shorter than batch_size, otherwise
        # the repeated records as described, transposed back to columns.
        # Inputs: batch dict. Outputs: dict of lists. Exceptions: none. Side effects: none.
        # `batch` maps each column to a list of `batch_size` values (fewer for the trailing batch, which is dropped to
        # match RepeatSampler). Transpose it back to records, repeating them as RepeatSampler repeats indices. Records
        # are deep-copied so repeats stay independent, matching the map-style path where the dataset re-materializes a
        # fresh object per access.
        keys = list(batch)
        if len(batch[keys[0]]) < batch_size:
            return {key: [] for key in keys}
        repeated = {key: [] for key in keys}
        for _ in range(repeat_count):
            for values in zip(*(batch[key] for key in keys), strict=True):
                for _ in range(mini_repeat_count):
                    for key, value in zip(keys, values, strict=True):
                        repeated[key].append(copy.deepcopy(value))
        return repeated

    return dataset.map(repeat_batch, batched=True, batch_size=batch_size)


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor, dim: int | tuple[int, ...] | None = None, keepdim: bool = False) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: NaN-ignoring sample standard deviation: nanmean, nanmean of squared deviations, Bessel correction
    # count/(count-1) (NaN when count is at most 1), square root; then removes reduced dims unless keepdim (all dims
    # squeezed when dim is None; negative dims normalized for tuples).
    # Inputs: tensor, dim, keepdim. Outputs: tensor. Exceptions: none. Side effects: none.
    """
    Compute the standard deviation of a tensor, ignoring NaNs.

    Args:
        tensor (`torch.Tensor`):
            Input tensor.
        dim (`int` or `tuple[int, ...]`, *optional*):
            Dimension(s) to reduce. Defaults to all dimensions.
        keepdim (`bool`, *optional*, defaults to `False`):
            Whether to keep reduced dimensions.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    # Compute variance ignoring NaNs
    mean = torch.nanmean(tensor, dim=dim, keepdim=True)
    variance = torch.nanmean((tensor - mean) ** 2, dim=dim, keepdim=True)
    count = torch.sum(~torch.isnan(tensor), dim=dim, keepdim=True)  # count of non-NaN values
    correction = count / (count - 1)
    correction = torch.where(count > 1, correction, torch.full_like(correction, float("nan")))
    variance *= correction  # Bessel's correction
    std = torch.sqrt(variance)
    if keepdim:
        return std
    if dim is None:
        return std.squeeze()
    if isinstance(dim, int):
        return std.squeeze(dim)
    dims = [(d if d >= 0 else d + std.ndim) for d in dim]
    for d in sorted(dims, reverse=True):
        std = std.squeeze(d)
    return std


def split_tensor_dict(
    tensor_dict: dict[str, torch.Tensor | None], num_chunks: int
) -> list[dict[str, torch.Tensor | None]]:
    # [FUNCTION DOC]
    # Behavior: splits each value along the first dimension into num_chunks equal chunks of size (first non-None
    # value length floor-divided by num_chunks); lists and tensors with dims are sliced, 0-d tensors are copied into
    # every chunk, None stays None.
    # Inputs: tensor_dict, num_chunks. Outputs: list of dicts.
    # Exceptions: StopIteration if every value is None.
    # Side effects: none.
    # KNOWN ISSUE: the chunk size is read from the shape of the first non-None value, so a dict whose first non-None
    # value is a list raises AttributeError although lists are otherwise supported; trailing rows are dropped when the
    # length is not divisible by num_chunks.
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.

    Example:
    ```python
    >>> x = torch.arange(12).reshape(6, 2)
    >>> y = torch.arange(6).reshape(6, 1)
    >>> tensor_dict = {"x": x, "y": y}
    >>> split_tensor_dict(tensor_dict, 3)
    [
        {"x": tensor([[0, 1], [2, 3]]), "y": tensor([[0], [1]])},
        {"x": tensor([[4, 5], [6, 7]]), "y": tensor([[2], [3]])},
        {"x": tensor([[ 8,  9], [10, 11]]), "y": tensor([[4], [5]])}
    ]
    ```
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    chunks = []
    for i in range(num_chunks):
        chunk_dict = {}
        for key, tensor in tensor_dict.items():
            if tensor is not None and (isinstance(tensor, list) or tensor.ndim > 0):
                chunk_dict[key] = tensor[i * chunk_size : (i + 1) * chunk_size]
            elif tensor is not None and tensor.ndim == 0:
                chunk_dict[key] = tensor
            else:
                chunk_dict[key] = None
        chunks.append(chunk_dict)
    return chunks


def shuffle_sequence_dict(seq_dict: dict[str, Sequence | None]) -> dict[str, Sequence | None]:
    # [FUNCTION DOC]
    # Behavior: draws one random permutation (global torch RNG) of the length of the first non-None value and applies
    # it to every tensor with at least one dim and every list; None and 0-d tensors pass through.
    # Inputs: seq_dict. Outputs: new dict. Exceptions: StopIteration if all values are None. Side effects: consumes global RNG.
    """
    Shuffles all sequence-like values in a dictionary along the first dimension in unison.

    Example:
    ```python
    >>> x = torch.arange(6).reshape(3, 2)
    >>> y = ["a", "b", "c"]
    >>> seq_dict = {"x": x, "y": y}
    >>> shuffle_sequence_dict(seq_dict)
    {'x': tensor([[2, 3],
                  [0, 1],
                  [4, 5]]),
     'y': ['b', 'a', 'c']}
    ```
    """
    # Determine batch size from the first non-None sequence
    batch_size = len(next(v for v in seq_dict.values() if v is not None))
    permutation = torch.randperm(batch_size)

    def permute(v: Sequence | None) -> Sequence | None:
        # [FUNCTION DOC]
        # HELPER: Applies the shared permutation to one value (None and scalars unchanged, tensors indexed, sequences rebuilt as lists).
        # Inputs: v. Outputs: permuted value. Exceptions: none. Side effects: none.
        if v is None:
            return None
        if isinstance(v, torch.Tensor) and v.ndim == 0:
            return v
        if isinstance(v, torch.Tensor) and v.ndim >= 1:
            return v[permutation]
        return [v[i] for i in permutation]

    return {key: permute(val) for key, val in seq_dict.items()}


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: minimum over non-NaN values of a 1-D tensor; a NaN scalar (same dtype/device) if all are NaN.
    # Inputs: tensor. Outputs: 0-d tensor. Exceptions: RuntimeError on an empty tensor. Side effects: none.
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: maximum over non-NaN values of a 1-D tensor; NaN if all are NaN.
    # Inputs: tensor. Outputs: 0-d tensor. Exceptions: RuntimeError on an empty tensor. Side effects: none.
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


def identity(x):
    # [FUNCTION DOC]
    # HELPER: Returns its argument unchanged; used as a no-op data collator.
    # Inputs: x. Outputs: x. Exceptions: none. Side effects: none.
    """Do we really need docs for this?"""
    return x


def split_pixel_values_by_grid(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    # [FUNCTION DOC]
    # Behavior: if pixel_values and num_images are present, splits flat per-batch image tensors into per-sample lists
    # so rows can be shuffled/split with their samples:
    # - image_grid_thw: rows per image = product of grid dims (must sum to pixel row count); pixel_values split by
    #   per-sample row sums and image_grid_thw split by num_images.
    # - image_position_ids: pixel_values and position ids split by num_images.
    # - spatial_shapes: pixel_values, pixel_attention_mask and spatial_shapes split by num_tiles.
    # - num_tiles only: pixel_values split by num_tiles.
    # - otherwise, if pixel rows equal total images, split pixel_values (and image_sizes) by num_images; else unchanged.
    # Inputs: batch dict. Outputs: new dict (or the same batch).
    # Exceptions: ValueError on a grid/row count mismatch. Side effects: none.
    """
    Splits `batch["pixel_values"]` into a list of tensors, one per sample, based on `batch["num_images"]`.

    For models with `image_grid_thw` (e.g. Qwen), the grid dimensions determine how many rows of `pixel_values` belong
    to each image. For models with `image_position_ids` instead (e.g. Gemma), `pixel_values` is indexed directly by
    image count. For models with `spatial_shapes` (e.g. LFM2-VL) or with `num_tiles` alone (e.g. InternVL),
    tile-indexed tensors are split using `num_tiles`. Models with none of these store `pixel_values` either flat, one
    row per image (e.g. LLaVA), in which case it is split by image count, or already padded to one row per sample (e.g.
    Idefics), in which case it is left as is.
    """
    if "pixel_values" not in batch or "num_images" not in batch:
        return batch

    num_images = batch["num_images"]
    pixel_values = batch["pixel_values"]  # [total, feature_dim]

    if "image_grid_thw" in batch:
        lengths = batch["image_grid_thw"].prod(-1).tolist()  # [num_images]
        if sum(lengths) != pixel_values.size(0):
            raise ValueError(
                f"Mismatch: sum(lengths) = {sum(lengths)} != pixel_values.size(0) = {pixel_values.size(0)}"
            )

        boundaries = [0, *accumulate(num_images)]
        image_grid_thw = batch["image_grid_thw"]  # [total, 3]
        sections = [sum(lengths[boundaries[i] : boundaries[i + 1]]) for i in range(len(num_images))]
        split_pixel_values = list(torch.split(pixel_values, sections, dim=0))
        split_image_grid_thw = list(torch.split(image_grid_thw, num_images, dim=0))
        return {**batch, "pixel_values": split_pixel_values, "image_grid_thw": split_image_grid_thw}

    if "image_position_ids" in batch:
        image_position_ids = batch["image_position_ids"]  # [total]
        split_pixel_values = list(torch.split(pixel_values, num_images, dim=0))
        split_image_position_ids = list(torch.split(image_position_ids, num_images, dim=0))
        return {**batch, "pixel_values": split_pixel_values, "image_position_ids": split_image_position_ids}

    if "spatial_shapes" in batch:
        num_tiles = batch["num_tiles"]
        pixel_attention_mask = batch["pixel_attention_mask"]
        spatial_shapes = batch["spatial_shapes"]
        split_pixel_values = list(torch.split(pixel_values, num_tiles, dim=0))
        split_pixel_attention_mask = list(torch.split(pixel_attention_mask, num_tiles, dim=0))
        split_spatial_shapes = list(torch.split(spatial_shapes, num_tiles, dim=0))
        return {
            **batch,
            "pixel_values": split_pixel_values,
            "pixel_attention_mask": split_pixel_attention_mask,
            "spatial_shapes": split_spatial_shapes,
        }

    if "num_tiles" in batch:
        num_tiles = batch["num_tiles"]
        return {**batch, "pixel_values": list(torch.split(pixel_values, num_tiles, dim=0))}

    # Models without extra metadata store pixel_values either flat, one row per image (e.g. LLaVA), or already
    # padded to one row per sample (e.g. Idefics, SmolVLM). Only the flat layout needs splitting.
    if pixel_values.size(0) != sum(num_images):
        return batch

    batch = {**batch, "pixel_values": list(torch.split(pixel_values, num_images, dim=0))}
    if "image_sizes" in batch:
        batch = {**batch, "image_sizes": list(torch.split(batch["image_sizes"], num_images, dim=0))}
    return batch


def unsplit_pixel_values_by_grid(batch: dict[str, torch.Tensor | list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    # [FUNCTION DOC]
    # Behavior: concatenates along the first dim any of pixel_values, image_grid_thw, image_position_ids,
    # pixel_attention_mask, spatial_shapes and image_sizes that are lists, returning a new dict.
    # Inputs: batch. Outputs: dict. Exceptions: none. Side effects: none.
    """
    Opposite of `split_pixel_values_by_grid`. Merges a list of tensors in `batch["pixel_values"]` back into a single
    tensor along the first dimension.
    """
    pixel_values = batch.get("pixel_values")
    if isinstance(pixel_values, list):
        merged = torch.cat(pixel_values, dim=0)
        batch = {**batch, "pixel_values": merged}

    image_grid_thw = batch.get("image_grid_thw")
    if isinstance(image_grid_thw, list):
        merged = torch.cat(image_grid_thw, dim=0)
        batch = {**batch, "image_grid_thw": merged}

    image_position_ids = batch.get("image_position_ids")
    if isinstance(image_position_ids, list):
        merged = torch.cat(image_position_ids, dim=0)
        batch = {**batch, "image_position_ids": merged}

    pixel_attention_mask = batch.get("pixel_attention_mask")
    if isinstance(pixel_attention_mask, list):
        merged = torch.cat(pixel_attention_mask, dim=0)
        batch = {**batch, "pixel_attention_mask": merged}

    spatial_shapes = batch.get("spatial_shapes")
    if isinstance(spatial_shapes, list):
        merged = torch.cat(spatial_shapes, dim=0)
        batch = {**batch, "spatial_shapes": merged}

    image_sizes = batch.get("image_sizes")
    if isinstance(image_sizes, list):
        merged = torch.cat(image_sizes, dim=0)
        batch = {**batch, "image_sizes": merged}

    return batch


TListOrMapping = TypeVar("TListOrMapping", list, Mapping)


# This function is intentionally not used internally. It is provided as a utility for users whose datasets contain
# `None` values inserted by tabular backends (e.g., Arrow/Parquet) for missing keys in nested structures. This
# situation arises when loading datasets created before `datasets` v4.7.0 (which introduced the Json dtype), or when
# datasets created after that version were saved without using the Json feature. In both cases, users can apply this
# function via `dataset = dataset.with_transform(remove_none_values)` before training to strip the spurious `None`
# values. See the migration guide for more details.
def remove_none_values(example: TListOrMapping) -> TListOrMapping:
    # [FUNCTION DOC]
    # Behavior: recursively drops None-valued keys from mappings; lists keep their None items but recurse into nested
    # dicts/lists. Intended for users via with_transform, not used internally.
    # Inputs: example (list or mapping). Outputs: cleaned structure.
    # Exceptions: TypeError for other input types. Side effects: none.
    """
    Recursively removes entries with `None` values from a nested structure (list or dictionary).

    Args:
        example (`list` or `Mapping`):
            Input nested structure (list or dictionary) from which to remove `None`.

    Examples:
    ```python
    >>> dataset = dataset.with_transform(remove_none_values)
    ```
    ```python
    >>> [
    ...     {
    ...         "a": {"aa": None, "ab": 1},
    ...         "b": "my_string",
    ...     }
    ... ]
    >>> remove_none_values(example)
    [{'a': {'ab': 1}, 'b': 'my_string'}]
    ```
    """
    if isinstance(example, list):
        return [remove_none_values(value) if isinstance(value, (dict, list)) else value for value in example]
    elif isinstance(example, Mapping):
        return {
            key: remove_none_values(value) if isinstance(value, (dict, list)) else value
            for key, value in example.items()
            if value is not None
        }
    else:
        raise TypeError("Input must be a list or a dictionary.")


def create_model_from_path(
    model_id: str, architecture: _BaseAutoModelClass | None = None, **kwargs
) -> PreTrainedModel:
    # [FUNCTION DOC]
    # Behavior: normalizes dtype (default "float32"; strings bfloat16/float16/float32 become torch dtypes; torch
    # dtypes, "auto" and None pass); rejects use_kernels on transformers 5.2-5.3; defaults device_map to None on
    # CPU/MPS and "auto" otherwise; infers the architecture class from the config's first architecture name in
    # transformers, else from auto_map (image-text-to-text preferred over causal LM), else AutoModelForCausalLM; loads
    # with from_pretrained; ties weights on transformers 5.3.0.
    # Inputs: model_id, architecture (optional), kwargs for from_pretrained.
    # Outputs: PreTrainedModel.
    # Exceptions: ValueError for invalid dtype or the kernels/version combo; loading errors.
    # Side effects: downloads config and weights; mutates kwargs.
    """
    Create a model from a given path using the specified initialization arguments.

    Args:
        model_id (`str`):
            Path to the model. Can be either a local directory or a model identifier from the Hugging Face Hub.
        architecture (`_BaseAutoModelClass`, *optional*):
            Model architecture class to instantiate. The model is initialized using the `from_pretrained` method of
            this class. If `None`, the architecture will be inferred from the model's configuration.
        kwargs (`dict`):
            Initialization keyword arguments to pass to the model's `from_pretrained` method. When `'dtype'` is
            specified, it can be either a `torch.dtype` or one of the strings: `'bfloat16'`, `'float16'`, `'float32'`,
            or `'auto'`. If not explicitly set, `dtype` defaults to `'float32'`.

    Returns:
        [`~transformers.PreTrainedModel`]:
            The instantiated model.
    """
    dtype = kwargs.get("dtype", "float32")
    if isinstance(dtype, torch.dtype) or dtype == "auto" or dtype is None:
        pass  # dtype is already a torch.dtype or "auto" or None
    elif isinstance(dtype, str) and dtype in ["bfloat16", "float16", "float32"]:
        kwargs["dtype"] = getattr(torch, dtype)
    else:
        raise ValueError(
            "Invalid `dtype` passed to the config. Expected either 'auto' or a string representing "
            f"a valid `torch.dtype` (e.g., 'float32'), but got {dtype}."
        )
    # transformers 5.2/5.3 pin a kernels<0.11 that fails to load the Hub kernels (fixed in 5.4.0, transformers#44887)
    if kwargs.get("use_kernels") and Version("5.2.0") <= Version(transformers.__version__) < Version("5.4.0"):
        raise ValueError(
            "`use_kernels=True` requires transformers>=5.4.0: 5.2.0 and 5.3.0 fail to load the Hub kernels."
        )
    # Respect CPU-only execution: device_map="auto" dispatches the model to the GPU even when the user requested
    # use_cpu=True, which later splits models across devices (e.g. a teacher placed on CPU vs. a student on GPU).
    # On MPS, "auto" segfaults when casting bf16 weights to float32 (huggingface/transformers#48029).
    if "device_map" not in kwargs:
        kwargs["device_map"] = None if PartialState().device.type in ("cpu", "mps") else "auto"
    if architecture is None:
        # Best effort to infer architecture from config, but we fall back to AutoModelForCausalLM if we can't find it
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=kwargs.get("trust_remote_code", False))
        architecture = getattr(transformers, config.architectures[0], None)
        if architecture is None:
            # Remote-code checkpoint: the architecture name lives in the dynamic module, not in
            # `transformers`. Pick the most specific auto class declared in `config.auto_map`.
            auto_map = config.auto_map or {}
            for candidate in (AutoModelForImageTextToText, AutoModelForCausalLM):
                if candidate.__name__ in auto_map:
                    architecture = candidate
                    break
            else:
                architecture = AutoModelForCausalLM
    model = architecture.from_pretrained(model_id, **kwargs)
    # transformers 5.3.0 skips tying remote-code tied embeddings, leaving them on meta (transformers#44469)
    if Version(transformers.__version__) == Version("5.3.0"):
        model.tie_weights()
    return model


def hash_module(module: torch.nn.Module) -> str:
    # [FUNCTION DOC]
    # HELPER: SHA-256 over the module state dict in sorted key order, hashing each tensor's dtype string and raw
    # bytes (bf16 and fp8 converted to fp32 first) on CPU.
    # Inputs: module. Outputs: hex digest str. Exceptions: none. Side effects: copies tensors to CPU.
    h = hashlib.sha256()
    for _, tensor in sorted(module.state_dict().items()):
        tensor = tensor.cpu()
        h.update(str(tensor.dtype).encode())
        if tensor.dtype in [torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2]:
            tensor = tensor.to(torch.float32)
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def get_config_model_id(config: PretrainedConfig) -> str:
    # [FUNCTION DOC]
    # Behavior: returns the config's _name_or_path, or empty string if absent.
    # Inputs: config. Outputs: str. Exceptions: none. Side effects: none.
    """
    Retrieve the model identifier from a given model configuration.

    Args:
        config ([`~transformers.PreTrainedConfig`]):
            Configuration from which to extract the model identifier.

    Returns:
        `str`:
            The model identifier associated with the model configuration.
    """
    return getattr(config, "_name_or_path", "")


@contextmanager
def global_then_local_main_first():
    # [FUNCTION DOC]
    # Behavior: context manager entering PartialState main_process_first and then local_main_process_first, so the
    # global main runs first, then each node's local main, then the rest.
    # Inputs: none. Outputs: yields None. Exceptions: none. Side effects: distributed barriers.
    """
    Context manager that lets the global main process run the block first, then the local main of each node, then
    everyone else. Both scopes of `PartialState.main_process_first`, one after the other.

    Work that writes to a cache only has to happen once per cache the processes can read. The global main goes first,
    which is enough when the cache is shared. The local mains then go first, so a cache on node-local disk costs one
    pass per node rather than one per process.
    """
    state = PartialState()
    with state.main_process_first(), state.local_main_process_first():
        yield


@contextmanager
def use_adapter(model: "PeftModel", adapter_name: str | None):
    # [FUNCTION DOC]
    # Behavior: context manager; with adapter_name None disables all adapters for the block; otherwise sets the given
    # adapter active and restores the previous active adapter afterwards (even on error).
    # Inputs: model, adapter_name. Outputs: yields None.
    # Exceptions: ImportError when peft is not installed. Side effects: temporarily changes model adapter state.
    """
    Context manager to temporarily set and reset the active adapter in a PEFT model.

    Args:
        model ([`~peft.PeftModel`]):
            PEFT model to manage.
        adapter_name (`str` or `None`):
            Name of the adapter to set as active. If `None`, the context manager will disable all adapters.

    Example:
    ```python
    >>> from trl.trainer.utils import use_adapter
    >>> from peft import AutoPeftModelForCausalLM
    >>> import torch

    >>> model = AutoPeftModelForCausalLM.from_pretrained("path/to/model")
    >>> input_ids = torch.tensor([[1, 2, 3]])
    >>> with use_adapter(model, "adapter_name"):
    ...     outputs = model(input_ids)
    ```
    """

    if not is_peft_available():
        raise ImportError(
            "You're trying to use a PEFT adapter but PEFT is not installed. Please install it with `pip install peft`."
        )
    if adapter_name is None:
        with model.disable_adapter():
            yield
    else:
        previous_adapter = model.active_adapter
        model.set_adapter(adapter_name)
        try:
            yield
        finally:
            model.set_adapter(previous_adapter)


def start_event_loop_in_daemon(
    name: str | None = None,
) -> tuple[threading.Thread, asyncio.AbstractEventLoop, threading.Event]:
    # [FUNCTION DOC]
    # Behavior: creates a new asyncio loop and a daemon thread that sets it as current, signals a ready Event and runs
    # the loop forever; starts the thread.
    # Inputs: name (thread name). Outputs: tuple of thread, loop, ready event.
    # Exceptions: none. Side effects: starts a background thread.
    """
    This function creates a new daemon thread that runs the provided event loop.

    Args:
        name (`str`, *optional*):
            Name of the thread. If `None`, the default thread naming will be used.

    Returns:
        `threading.Thread`:
            The thread running the event loop.
        `asyncio.AbstractEventLoop`:
            The event loop being run in the thread.
        `threading.Event`:
            An event that is set when the loop is ready.
    """
    loop = asyncio.new_event_loop()
    loop_ready_event = threading.Event()

    def run_loop():
        # [FUNCTION DOC]
        # HELPER: Thread target: sets the loop as current, signals readiness, and runs the loop until stopped.
        # Inputs: none (closure). Outputs: none. Exceptions: none. Side effects: blocks the thread running the loop.
        asyncio.set_event_loop(loop)
        loop_ready_event.set()
        loop.run_forever()

    thread = threading.Thread(target=run_loop, name=name, daemon=True)
    thread.start()
    return thread, loop, loop_ready_event


def shutdown_event_loop_in_daemon(
    thread: threading.Thread | None,
    loop: asyncio.AbstractEventLoop | None,
) -> None:
    # [FUNCTION DOC]
    # Behavior: no-op if loop or thread is None; otherwise schedules loop stop thread-safely and joins the thread with a 5 s timeout.
    # Inputs: thread, loop. Outputs: none. Exceptions: none. Side effects: stops the loop; the loop is not closed.
    """
    Shutdown an asyncio event loop running in a separate thread.

    This function stops the event loop and waits for the associated thread to finish execution.

    Args:
        thread (`threading.Thread`):
            The thread running the event loop.
        loop (`asyncio.AbstractEventLoop`):
            The asyncio event loop to shut down.
    """
    if loop is None or thread is None:
        return
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)


_CHUNKED_LOGPROB_CHUNK_SIZE = 32768


@dataclass
class FusedCausalLMOutput(ModelOutput):
    # [CLASS DOC]
    # ModelOutput dataclass returned by a model patched by add_fused_lm_head when called with fused_lm_head enabled.
    # Fields (all default None): loss, log_probs, entropy, log_sum_sq_probs, mean_logits, is_top1, label_mask,
    # aux_loss. Per-token fields are zero on ignored labels and None when not requested.
    """
    Output of a model given a fused LM head with [`add_fused_lm_head`] and called with `fused_lm_head=True`. Every
    per-token field is zero where the label is `-100`. Per-token fields not listed in `outputs` are `None`.

    Args:
        loss (`torch.Tensor`):
            Negative log-likelihood under the temperature-scaled distribution, summed over the non-ignored tokens and
            divided by `num_items_in_batch` (their count when not passed), plus the MoE load-balancing loss weighted by
            the config's `router_aux_loss_coef` when router logits are requested. At `temperature=1.0`, this is the
            loss the model's own forward computes.
        log_probs (`torch.Tensor` of shape `(batch, seq_len - 1)`, or `(batch, seq_len)` with `shift_labels`):
            Log-probability of each next-token label.
        entropy (`torch.Tensor`, same shape as `log_probs`):
            Entropy of the next-token distribution.
        log_sum_sq_probs (`torch.Tensor`, same shape as `log_probs`):
            `log(sum_v p_v^2)` of the next-token distribution, without gradient.
        mean_logits (`torch.Tensor`, same shape as `log_probs`):
            Mean of the temperature-scaled next-token logits over the vocabulary, without gradient.
        is_top1 (`torch.Tensor`, same shape as `log_probs`):
            Whether the label is the most likely next token.
        label_mask (`torch.Tensor`, same shape as `log_probs`):
            Whether the label is not `-100`. Prompt-learning PEFT pads the labels, so this can count one more token per
            sequence than the caller's labels.
        aux_loss (`torch.Tensor`, *optional*):
            MoE load-balancing loss, when called with `output_router_logits=True`.
    """

    loss: torch.Tensor | None = None
    log_probs: torch.Tensor | None = None
    entropy: torch.Tensor | None = None
    log_sum_sq_probs: torch.Tensor | None = None
    mean_logits: torch.Tensor | None = None
    is_top1: torch.Tensor | None = None
    label_mask: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None


def add_fused_lm_head(
    model: PreTrainedModel,
    temperature: float = 1.0,
    cast_lm_head_to_fp32: bool = False,
    outputs: tuple[str, ...] = ("log_probs",),
) -> None:
    # [FUNCTION DOC]
    # Behavior: patches model.forward (bound method, signature preserved for generate) so that with
    # fused_lm_head enabled it computes per-token label statistics in chunks without full logits. Precomputes from
    # the text config: final_logit_softcapping and a logit scale (logit_scale, else output_multiplier, else
    # lm_head_multiplier, else logits_scaling inverted except for hyperclovax where it multiplies; default 1.0), and
    # the backbone attribute ("model" for VLMs on transformers<5, else "base_model").
    # Inputs: model, temperature (1.0), cast_lm_head_to_fp32 (False), outputs (which per-token fields to compute).
    # Outputs: none. Exceptions: none at patch time. Side effects: replaces model.forward.
    """
    Add a fused LM head to `model`: `model(..., labels=labels, fused_lm_head=True)` returns per-token log-probabilities
    instead of logits, without materializing the `(batch, seq_len, vocab)` logits.

    With `fused_lm_head=True`, the forward runs the backbone and projects through the LM head, in tiles, only the
    positions whose next-token label is not `-100`, and returns a [`FusedCausalLMOutput`]. Pre-shifted `shift_labels`,
    as passed under context or sequence parallelism, are scored without shifting. Without it, the forward is the
    original one, so generation is unchanged. Add the head before wrapping the model with PEFT.

    Args:
        model ([`~transformers.PreTrainedModel`]):
            Causal language model to patch.
        temperature (`float`, *optional*, defaults to `1.0`):
            Temperature the logits are divided by.
        cast_lm_head_to_fp32 (`bool`, *optional*, defaults to `False`):
            Whether to run the LM head projection in float32, outside autocast.
        outputs (`tuple[str, ...]`, *optional*, defaults to `("log_probs",)`):
            Per-token fields of [`FusedCausalLMOutput`] the kernel computes, among `"log_probs"`, `"entropy"`,
            `"log_sum_sq_probs"`, `"mean_logits"` and `"is_top1"`. The others are `None`. `log_probs` (and so `loss`)
            is always computed; each extra field costs a little on every call.
    """
    original_forward = model.forward
    text_config = model.config.get_text_config()
    final_logit_softcapping = getattr(text_config, "final_logit_softcapping", None)
    # `logit_scale` is None on models that don't scale (e.g. MPT); read that as unscaled (1.0). A real 0.0 is kept
    # as-is. Muse Glimmer applies the same pre-softcap multiplier as `output_multiplier`.
    logit_scale = getattr(text_config, "logit_scale", None)
    if logit_scale is None:
        logit_scale = getattr(text_config, "output_multiplier", None)
    # Falcon-H1 multiplies the logits by `lm_head_multiplier`. Granite and MiniCPM3 divide them by
    # `logits_scaling`, while HyperCLOVA X multiplies them by it.
    if logit_scale is None:
        logit_scale = getattr(text_config, "lm_head_multiplier", None)
    if logit_scale is None and getattr(text_config, "logits_scaling", None) is not None:
        logits_scaling = text_config.logits_scaling
        logit_scale = logits_scaling if text_config.model_type == "hyperclovax" else 1 / logits_scaling
    logit_scale = 1.0 if logit_scale is None else logit_scale
    # Before transformers 5, vision-language models expose their language backbone as `model` rather than through
    # `base_model_prefix`.
    backbone_attr = "base_model"
    if text_config is not model.config and Version(transformers.__version__) < Version("5.0.0"):
        backbone_attr = "model"

    # Keep the original signature: `generate` validates its model kwargs against it
    @functools.wraps(type(model).forward)
    def _fused_forward(
        self, *args, fused_lm_head=False, labels=None, shift_labels=None, num_items_in_batch=None, **kwargs
    ):
        # [FUNCTION DOC]
        # HELPER: Replacement forward.
        # Behavior: without fused_lm_head, forwards labels/shift_labels/num_items_in_batch (if given) to the original
        # forward. With it: disables cache, requests router logits if the config asks, runs the backbone, and aligns hidden
        # states with labels (drop last position and first label, or use shift_labels as is). Moves both to the LM head
        # device, builds the valid-label mask, optionally upcasts to fp32 with autocast disabled, materializes DTensor
        # weights (FSDP2), and under the ZeRO-3 gather context runs the chunked log-prob autograd function (chunk size
        # 32768) either on only valid positions (Triton devices) or on all positions with labels clamped to 0. Results are
        # scattered/masked back to the label shape. loss = negative sum of log-probs divided by num_items_in_batch or the
        # valid count (min 1). With router logits, adds the config's router_aux_loss_coef times the Mixtral
        # load-balancing loss (expert counts from the model or text config depending on transformers version and gpt_oss
        # workaround), using attention_mask when present.
        # Inputs: self, args, fused_lm_head, labels, shift_labels, num_items_in_batch, kwargs.
        # Outputs: original model output, or FusedCausalLMOutput.
        # Exceptions: propagated from backbone. Side effects: none beyond computation.
        if not fused_lm_head:
            if labels is not None:
                kwargs["labels"] = labels
            if shift_labels is not None:
                kwargs["shift_labels"] = shift_labels
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            return original_forward(*args, **kwargs)

        # The backbone and the LM head are looked up on each call, since FSDP and PEFT replace them after patching
        kwargs["use_cache"] = False
        # MoE models: like the model's own forward, request router logits when the config asks for them
        if getattr(text_config, "output_router_logits", False):
            kwargs.setdefault("output_router_logits", True)
        backbone_outputs = getattr(self, backbone_attr)(*args, **kwargs)
        if shift_labels is None:
            hidden_states = backbone_outputs.last_hidden_state[:, :-1]
            labels = labels[:, 1:]
        else:
            hidden_states = backbone_outputs.last_hidden_state
            labels = shift_labels
        lm_head = self.get_output_embeddings()
        # With the model split across devices (`device_map`), the backbone can end on another device than the head
        hidden_states = hidden_states.to(lm_head.weight.device)
        labels = labels.to(lm_head.weight.device)
        mask = labels != -100
        autocast_ctx = nullcontext()
        if cast_lm_head_to_fp32:
            hidden_states = hidden_states.float()
            autocast_ctx = torch.autocast(hidden_states.device.type, enabled=False)

        weight, bias = lm_head.weight, lm_head.bias
        # FSDP2 exposes sharded parameters as DTensors, while the backbone output is a regular tensor. Gather the
        # head once before splitting tokens so every projection uses compatible tensor types.
        if isinstance(weight, DTensor):
            weight = weight.full_tensor()
            if bias is not None:
                bias = bias.full_tensor()
        # On Triton devices, only the valid positions are scored. Elsewhere (e.g. Neuron), the number of valid positions
        # would change the tile shapes on every batch and trigger a recompilation, so every position is scored instead.
        compact = _use_triton(hidden_states.device)
        with autocast_ctx, maybe_gather_lm_head_ctx(weight, bias):
            per_token = _ChunkedLogProbFunction.apply(
                hidden_states[mask] if compact else hidden_states.reshape(-1, hidden_states.size(-1)),
                weight,
                bias,
                labels[mask] if compact else labels.clamp(min=0).reshape(-1),
                temperature,
                _CHUNKED_LOGPROB_CHUNK_SIZE,
                final_logit_softcapping,
                logit_scale,
                outputs,
            )
        if compact:
            # `masked_scatter` keeps the output connected to the model even when no label is valid. This lets an
            # all-masked microbatch contribute a differentiable zero instead of failing in `backward()`.
            per_token = (None if x is None else x.new_zeros(mask.shape).masked_scatter(mask, x) for x in per_token)
        else:
            per_token = (None if x is None else x.view(mask.shape).masked_fill(~mask, 0) for x in per_token)
        log_probs, entropy, log_sum_sq_probs, mean_logits, is_top1 = per_token
        loss = -log_probs.sum() / (mask.sum().clamp(min=1) if num_items_in_batch is None else num_items_in_batch)

        aux_loss = None
        if kwargs.get("output_router_logits"):
            # Mixtral's load-balancing loss is the one every MoE family pulls in through the modular system.
            from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func

            if Version(transformers.__version__) < Version("5.0.0"):
                num_experts = self.num_experts
                num_experts_per_tok = self.num_experts_per_tok
            else:
                # Upstream bug AttributeError: 'GptOssConfig' object has no attribute 'num_experts'; see #5754
                if text_config.model_type == "gpt_oss" and Version("5.0.0") <= Version(
                    transformers.__version__
                ) < Version("5.6.0"):
                    num_experts = self.num_experts
                else:
                    num_experts = text_config.num_experts
                num_experts_per_tok = text_config.num_experts_per_tok
            # Padding-free packs all real tokens into a single row, so `attention_mask` is None and every token counts.
            aux_loss = load_balancing_loss_func(
                backbone_outputs.router_logits, num_experts, num_experts_per_tok, kwargs.get("attention_mask")
            )
            loss = loss + getattr(text_config, "router_aux_loss_coef", 0.0) * aux_loss

        return FusedCausalLMOutput(
            loss=loss,
            log_probs=log_probs,
            entropy=entropy,
            log_sum_sq_probs=log_sum_sq_probs,
            mean_logits=mean_logits,
            is_top1=is_top1,
            label_mask=mask,
            aux_loss=aux_loss,
        )

    model.forward = types.MethodType(_fused_forward, model)


def compute_flops_per_token(config: PretrainedConfig, seq_len: int) -> int:
    # [FUNCTION DOC]
    # Behavior: estimates training FLOPs per token as 3 times forward FLOPs. Per layer: attention projections
    # (2*h*(q + 2 kv) dims, output projection) plus non-causal attention score terms (4 * heads * head_dim * seq_len);
    # dense MLP 6*h*intermediate; for MoE configs (num_experts_per_tok present) layers whose index is divisible by
    # decoder_sparse_step use routed experts (6*h*moe_intermediate per active expert plus 2*h*num_experts router, with
    # num_experts from num_local_experts on transformers>=5.1) and others use the dense MLP. Adds 2*V*h for the LM head.
    # head_dim falls back to hidden size over heads.
    # Inputs: config, seq_len. Outputs: int. Exceptions: AttributeError for configs missing fields. Side effects: none.
    """
    Estimate training FLOPs per token for a causal language model (forward + backward).

    Supports dense and MoE architectures. Backward is assumed to cost 2× the forward pass, so total training FLOPs = 3
    × forward FLOPs. The attention-score term uses the non-causal convention (every token attends to the full
    `seq_len`, matching PaLM / Megatron / nanoGPT); pass the resulting MFU through [`adjusted_mfu`] for the Llama /
    DeepSpeed Ulysses causal-corrected convention.

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        seq_len (`int`):
            The sequence length used for training.

    Returns:
        `int`: Estimated training FLOPs per token.
    """
    h = config.hidden_size
    L = config.num_hidden_layers
    V = config.vocab_size
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    # `head_dim` is optional on configs: Llama and Mistral declare it, Qwen2 doesn't. Derive it when missing, like
    # transformers' own modeling code does.
    head_dim = getattr(config, "head_dim", None) or h // n_heads

    # Attention: Q/K/V/O projections + attention score (Q·Kᵀ and attn·V).
    qkv_flops = 2 * h * (n_heads * head_dim + 2 * n_kv_heads * head_dim)
    o_proj_flops = 2 * n_heads * head_dim * h
    attn_score_flops = 2 * 2 * n_heads * head_dim * seq_len
    attn_flops = qkv_flops + o_proj_flops + attn_score_flops

    # MoE dispatch: `num_experts_per_tok` is the canonical MoE marker — present on Mixtral,
    # Qwen3-MoE, DeepSeek-V2, etc.; absent on dense configs.
    num_experts_per_tok = getattr(config, "num_experts_per_tok", None)
    if num_experts_per_tok is None:
        mlp_flops = 2 * 3 * h * config.intermediate_size
        total_layer_flops = L * (attn_flops + mlp_flops)
    else:
        # Routed experts (gate + up + down, 3 matmuls each) + router.
        if Version(transformers.__version__) >= Version("5.1.0"):
            num_experts = config.num_local_experts
        else:
            num_experts = config.num_experts
        moe_mlp_flops = num_experts_per_tok * 2 * 3 * h * config.moe_intermediate_size
        moe_mlp_flops += 2 * h * num_experts
        dense_mlp_flops = 2 * 3 * h * config.intermediate_size  # interspersed dense layers
        sparse_step = config.decoder_sparse_step
        total_layer_flops = sum(
            attn_flops + (moe_mlp_flops if layer_idx % sparse_step == 0 else dense_mlp_flops) for layer_idx in range(L)
        )

    # Embedding is a lookup (no FLOPs); lm_head is a 2*V*h matmul, tied or not.
    lm_head_flops = 2 * V * h

    forward_flops = total_layer_flops + lm_head_flops
    return 3 * forward_flops


# Theoretical dense accelerator throughput. Values and sources follow TorchTitan's BF16 peak-FLOPs lookup, extended
# with the additional NVIDIA GPUs offered by Hugging Face Jobs. More specific names must precede their prefixes.
_PEAK_FLOPS_BY_DEVICE = (
    # NVIDIA
    ("GB300", {"bfloat16": 2.5e15}),
    ("GB200", {"bfloat16": 2.5e15}),
    ("B300", {"bfloat16": 2.25e15}),
    ("B200", {"bfloat16": 2.25e15}),
    ("H100 NVL", {"float16": 835e12, "bfloat16": 835e12}),
    ("H100 PCIe", {"float16": 756e12, "bfloat16": 756e12}),
    ("H100", {"float16": 989e12, "bfloat16": 989e12}),
    ("H200 NVL", {"float16": 835e12, "bfloat16": 835e12}),
    ("H200", {"float16": 989e12, "bfloat16": 989e12}),
    ("H20", {"float16": 148e12, "bfloat16": 148e12}),
    ("RTX PRO 6000", {"float16": 500e12, "bfloat16": 500e12}),
    ("A100", {"float16": 312e12, "bfloat16": 312e12}),
    ("A6000", {"float16": 154.85e12, "bfloat16": 154.85e12}),
    ("A10G", {"float16": 125e12, "bfloat16": 125e12}),
    ("A10", {"float16": 125e12, "bfloat16": 125e12}),
    ("L40S", {"float16": 362e12, "bfloat16": 362e12}),
    ("L4", {"float16": 121e12, "bfloat16": 121e12}),
    ("T4", {"float16": 65e12}),
    # AMD
    ("MI355X", {"bfloat16": 2500e12}),
    ("MI325X", {"bfloat16": 1300e12}),
    ("MI300X", {"bfloat16": 1300e12}),
    ("MI250X", {"bfloat16": 191.5e12}),
)


def get_peak_flops(device_name: str, dtype: str) -> float | None:
    # [FUNCTION DOC]
    # Behavior: case-insensitive whole-word search of the device name against the ordered peak-FLOPs table (specific
    # names first); returns the dtype entry or None.
    # Inputs: device_name, dtype. Outputs: float or None. Exceptions: none. Side effects: none.
    """
    Get the theoretical dense accelerator peak FLOPs for a device and dtype.

    Args:
        device_name (`str`):
            Device name as returned by the accelerator runtime.
        dtype (`str`):
            Floating-point dtype used by the model's matrix multiplications.

    Returns:
        `float` or `None`: Peak FLOPs, or `None` when the device or dtype is not in the lookup table.
    """
    device_name = device_name.casefold()
    for model_name, peak_flops_by_dtype in _PEAK_FLOPS_BY_DEVICE:
        if re.search(rf"\b{re.escape(model_name.casefold())}\b", device_name):
            return peak_flops_by_dtype.get(dtype)
    return None


def get_peak_flops_per_device(accelerator: Accelerator, dtype: str) -> float | None:
    # [FUNCTION DOC]
    # Behavior: reads the CUDA device name for the accelerator device and looks up peak FLOPs; logs an info message
    # when unknown.
    # Inputs: accelerator, dtype. Outputs: float or None.
    # Exceptions: errors from torch.cuda on non-CUDA devices. Side effects: info log.
    """
    Resolve the theoretical dense peak FLOPs for the local training device.

    Args:
        accelerator ([`~accelerate.Accelerator`]):
            Accelerator managing the training devices.
        dtype (`str`):
            Configured model dtype.

    Returns:
        `float` or `None`: Local device peak FLOPs, or `None` if the device or precision is unsupported.
    """
    device_name = torch.cuda.get_device_name(accelerator.device)
    peak_flops = get_peak_flops(device_name, dtype)
    if peak_flops is None:
        logger.info(
            "MFU metrics are disabled because the peak FLOPs are unknown for the local training device or "
            "precision. Throughput and timing metrics are still reported."
        )
    return peak_flops


def compute_mfu(
    flops_per_token: int,
    tokens_per_second: float,
    world_size: int,
    peak_flops_per_device: float,
) -> float:
    # [FUNCTION DOC]
    # Behavior: returns 100 * flops_per_token * tokens_per_second / (peak_flops_per_device * world_size).
    # Inputs: flops_per_token, tokens_per_second, world_size, peak_flops_per_device. Outputs: float percent.
    # Exceptions: ZeroDivisionError if peak or world size is zero. Side effects: none.
    """
    Compute Model FLOPs Utilization (MFU) as a percentage.

    The caller is responsible for correcting `tokens_per_second` for any parallelism dimension that causes the
    trainer's token counter to over-count (e.g. context parallelism, sequence parallelism, tensor parallelism — every
    rank in those dims sees the same input tokens).

    Args:
        flops_per_token (`int`):
            Training FLOPs per token (from [`compute_flops_per_token`]).
        tokens_per_second (`float`):
            Aggregate tokens per second across all devices, after any parallelism corrections.
        world_size (`int`):
            Number of devices (GPUs).
        peak_flops_per_device (`float`):
            Theoretical dense peak FLOPs per device for the training precision.

    Returns:
        `float`: MFU as a percentage (0-100).
    """
    return 100 * (flops_per_token * tokens_per_second) / (peak_flops_per_device * world_size)


def adjusted_mfu(mfu: float, config: PretrainedConfig, seq_len: int) -> float:
    # [FUNCTION DOC]
    # Behavior: removes half of the attention-score FLOPs (layers * 3 * 2 * heads * head_dim * seq_len) from the
    # per-token total and rescales mfu by the ratio of corrected to full FLOPs.
    # Inputs: mfu, config, seq_len. Outputs: float percent. Exceptions: none. Side effects: none.
    """
    Apply a causal-masking correction to an MFU computed with [`compute_flops_per_token`].

    [`compute_flops_per_token`] uses the non-causal attention convention (every token treated as attending to the full
    `seq_len`, matching PaLM / Megatron / nanoGPT). With causal masking, only half of the attention-score FLOPs (`Q·Kᵀ`
    and `attn·V`) are actually performed. This function subtracts that half from the per-token total and rescales `mfu`
    accordingly. Use it to compare against reports that follow the Llama 2/3 / DeepSpeed Ulysses convention.

    Args:
        mfu (`float`):
            MFU as a percentage, computed via [`compute_mfu`] (i.e., using the non-causal [`compute_flops_per_token`]).
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        seq_len (`int`):
            The sequence length used for training.

    Returns:
        `float`: Causal-corrected MFU as a percentage.
    """
    flops_full = compute_flops_per_token(config, seq_len)
    # Half of the attention-score FLOPs (Q·Kᵀ and attn·V), per layer, ×3 for fwd+bwd.
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    half_attn_score = config.num_hidden_layers * 3 * 2 * config.num_attention_heads * head_dim * seq_len
    return mfu * (flops_full - half_attn_score) / flops_full
