# [MODULE DOC]
# Module trl.models.utils: model-wrapping utilities used by trainers: unwrapping DDP/DeepSpeed/FSDP models for
# generation (with ZeRO-3 parameter gathering and generation-config override), preparing auxiliary
# (reference/teacher/reward) models for DeepSpeed or FSDP inference, forward redirection through wrappers,
# temporarily disabling gradient checkpointing, and a deprecated create_reference_model alias.
# Main contents: remove_hooks / add_hooks / get_all_parameters / iter_params (ZeRO-3 hook management),
# _unwrap_model_for_generation, _override_model_generation_config, unwrap_model_for_generation,
# prepare_deepspeed, prepare_fsdp, _ForwardRedirection, disable_gradient_checkpointing, create_reference_model.
# Import-time side effects: imports trl.experimental.utils with the experimental warning suppressed; imports
# FSDPModule from the torch-version-appropriate path and get_parameters_from_modules for accelerate 1.11.0 or
# later.

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

import itertools
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import accelerate
import torch
import torch.nn as nn
import transformers
from accelerate import Accelerator
from packaging.version import Version
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig, PreTrainedModel

from ..import_utils import suppress_experimental_warning


with suppress_experimental_warning():
    from ..experimental.utils import create_reference_model as _create_reference_model


if Version(torch.__version__) >= Version("2.6.0"):
    from torch.distributed.fsdp import FSDPModule
else:  # the FSDP2 API was public only from torch 2.6
    from torch.distributed._composable.fsdp import FSDPModule

if Version(accelerate.__version__) >= Version("1.11.0"):
    from accelerate.utils.fsdp_utils import get_parameters_from_modules

if TYPE_CHECKING:
    from deepspeed.runtime.engine import DeepSpeedEngine
    from torch.nn import Module
    from torch.nn.parallel.distributed import DistributedDataParallel


def remove_hooks(model: "DeepSpeedEngine") -> None:
    # [FUNCTION DOC]
    # HELPER: Remove DeepSpeed ZeRO-3 parameter-offload hooks so a gathered model can run generation freely.
    # Behavior: no-op if the engine has no optimizer attribute yet. The offload object is optimizer.parameter_offload
    # if present, else the optimizer; raise if optimizer is None. Clear ds_active_sub_modules of every parameter
    # (including external ones) of its module, remove all forward and backward hooks and reset the hook lists.
    # Inputs: model (DeepSpeedEngine). Outputs: none. Exceptions: RuntimeError when optimizer is None.
    # Side effects: mutates DeepSpeed hook state.
    """Removes the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if not hasattr(model, "optimizer"):  # before the first training step, the model has no optimizer
        return
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    else:
        raise RuntimeError("The model optimizer is None, which is not yet supported.")

    for param in iter_params(optimizer_offload.module, recurse=True):
        param.ds_active_sub_modules.clear()

    for hook in optimizer_offload.forward_hooks:
        hook.remove()
    for hook in optimizer_offload.backward_hooks:
        hook.remove()

    optimizer_offload.forward_hooks = []
    optimizer_offload.backward_hooks = []


def get_all_parameters(sub_module, recurse=False):
    # [FUNCTION DOC]
    # HELPER: Chain a module's named parameters (optionally recursive) with its DeepSpeed external parameters.
    # Inputs: sub_module, recurse (default False). Outputs: iterator of (name, param).
    # Exceptions: AttributeError when the module lacks ds_external_parameters. Side effects: none.
    return itertools.chain(sub_module.named_parameters(recurse=recurse), sub_module.ds_external_parameters())


def iter_params(module, recurse=False):
    # [FUNCTION DOC]
    # HELPER: List of parameters from get_all_parameters (names dropped).
    # Inputs: module, recurse (default False). Outputs: list of parameters. Exceptions: as above. Side effects: none.
    return [param for _, param in get_all_parameters(module, recurse)]


def add_hooks(model: "DeepSpeedEngine") -> None:
    # [FUNCTION DOC]
    # HELPER: Re-register DeepSpeed ZeRO-3 hooks after generation.
    # Behavior: same no-op / offload-object resolution / error as remove_hooks; if the offload object has a
    # param_coordinator (stage 3) and its trace is valid, invalidate it (prevents stale traces after generation);
    # then re-register the DeepSpeed module hooks on its module.
    # Inputs: model (DeepSpeedEngine). Outputs: none. Exceptions: RuntimeError when optimizer is None.
    # Side effects: mutates DeepSpeed hook and trace state.
    """Adds the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if not hasattr(model, "optimizer"):  # before the first training step, the model has no optimizer
        return
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    else:
        raise RuntimeError("The model optimizer is None, which is not yet supported.")

    # Invalidate parameter coordinator trace to prevent stale state
    # after generation forward passes (fixes ZeRO-3 + GKD compatibility)
    if hasattr(optimizer_offload, "param_coordinator"):  # param_coordinator only exists in ZeRO stage 3
        coordinator = optimizer_offload.param_coordinator
        # Only invalidate if trace is not already invalid
        if not coordinator.is_invalid_trace():
            coordinator._invalidate_trace()

    optimizer_offload._register_deepspeed_module(optimizer_offload.module)


@contextmanager
def _unwrap_model_for_generation(
    model: "DistributedDataParallel | DeepSpeedEngine",
    accelerator: "Accelerator",
    gather_deepspeed3_params: bool = True,
    gradient_checkpointing_kwargs: dict | None = None,
):
    # [FUNCTION DOC]
    # HELPER: Context manager yielding an unwrapped model ready for generate.
    # Behavior: unwrap via the accelerator; if gradient checkpointing is on, disable it. Under ZeRO-3 either yield the
    # unwrapped model directly (gather_deepspeed3_params False) or, inside deepspeed GatheredParameters over all
    # parameters, remove hooks, yield, and re-add hooks. Otherwise yield the unwrapped model. Afterwards re-enable
    # gradient checkpointing with gradient_checkpointing_kwargs if it was on.
    # KNOWN ISSUE: there is no try/finally around the yields, so if the body raises, ZeRO-3 hooks are not re-added and
    # gradient checkpointing is not re-enabled.
    # Inputs: model, accelerator, gather_deepspeed3_params (True), gradient_checkpointing_kwargs (optional).
    # Outputs: yields the unwrapped model.
    # Exceptions: as remove_hooks/add_hooks. Side effects: toggles checkpointing, gathers ZeRO-3 params, hooks.
    """
    Context manager to unwrap distributed or accelerated models for generation tasks.

    Args:
        model (`DistributedDataParallel | DeepSpeedEngine`):
            Model to be unwrapped.
        accelerator ([`~accelerate.Accelerator`]):
            Accelerator instance managing the model.
        gather_deepspeed3_params (`bool`, *optional*, defaults to `True`):
            Whether to gather weights for DeepSpeed ZeRO Stage 3 models. If `False`, skips parameter gathering, which
            can be more memory-efficient but may lead to slower generation times.
        gradient_checkpointing_kwargs (`dict`, *optional*):
            Additional kwargs for gradient checkpointing enabling.

    Yields:
        Unwrapped model.

    Example:
    ```python
    >>> with _unwrap_model_for_generation(model, accelerator) as unwrapped_model:
    ...     generated_outputs = unwrapped_model.generate(input_ids)
    ```
    """
    unwrapped_model = accelerator.unwrap_model(model)
    is_gradient_checkpointing = unwrapped_model.is_gradient_checkpointing
    if is_gradient_checkpointing:
        unwrapped_model.gradient_checkpointing_disable()
    from ..distributed import DistributedBackend

    if DistributedBackend(accelerator).is_zero3:
        if not gather_deepspeed3_params:
            yield accelerator.unwrap_model(model)
        else:
            import deepspeed

            with deepspeed.zero.GatheredParameters(model.parameters()):
                remove_hooks(model)
                yield accelerator.unwrap_model(model)
                add_hooks(model)
    else:
        yield unwrapped_model
    if is_gradient_checkpointing:
        unwrapped_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)


@contextmanager
def _override_model_generation_config(model, generation_kwargs=None):
    # [FUNCTION DOC]
    # HELPER: Context manager temporarily replacing the model's generation_config with one updated by
    # generation_kwargs (workaround for transformers config merging before v5).
    # Behavior: on transformers 5.0.0 or later, or without generation_kwargs, or when the model has no
    # generation_config, yield the model and return. Otherwise use the PEFT base model if available, build a new
    # GenerationConfig from the original's dict updated with generation_kwargs, install it, yield, and restore the
    # original in finally.
    # KNOWN ISSUE: the early-exit branch yields the model while the override branch yields None, so the yielded
    # value is inconsistent (current callers ignore it).
    # Inputs: model, generation_kwargs (optional dict). Outputs: yields model or None.
    # Exceptions: GenerationConfig validation errors. Side effects: temporarily mutates model.generation_config.
    """
    Context manager to temporarily override a model's generation_config with training config.

    This works around transformers' config merging logic that would otherwise overwrite values matching global defaults
    with model-specific values (see upstream issue transformers#42762; fixed in transformers v5 by PR
    `transformers#42702`).

    By temporarily setting the model's generation_config to match the passed generation_config, we avoid the conflict.

    The model's original generation_config is preserved outside this context, ensuring that saved/pushed models retain
    their intended inference behavior.

    Args:
        model ([`~transformers.PreTrainedModel`]):
            The model (typically unwrapped_model) whose generation_config to temporarily override.
        generation_kwargs (`dict`):
            Generation kwargs to be used to override model's generation config.
    """
    if (
        # Issue fixed in transformers v5 by PR transformers#42702
        Version(transformers.__version__) >= Version("5.0.0")
        or generation_kwargs is None
        or not hasattr(model, "generation_config")
    ):
        yield model
        return
    # If it is a PEFT model, override the underlying base model
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    # Keep original model generation_config
    original_config = model.generation_config
    # Create training-specific generation config from the model's original generation config
    # Then overwrite it with the training-specific generation kwargs
    generation_config = GenerationConfig.from_dict(model.generation_config.to_dict())
    generation_config.update(**generation_kwargs)
    model.generation_config = generation_config
    try:
        yield
    finally:
        model.generation_config = original_config


@contextmanager
def unwrap_model_for_generation(
    model: "DistributedDataParallel | DeepSpeedEngine",
    accelerator: "Accelerator",
    gather_deepspeed3_params: bool = True,
    generation_kwargs: dict | None = None,
    gradient_checkpointing_kwargs: dict | None = None,
):
    # [FUNCTION DOC]
    # Public context manager combining _unwrap_model_for_generation and _override_model_generation_config; yields
    # the unwrapped model.
    # Inputs: model, accelerator, gather_deepspeed3_params (True), generation_kwargs (optional),
    # gradient_checkpointing_kwargs (optional). Outputs: yields the unwrapped model.
    # Exceptions: as the two helpers. Side effects: as the two helpers.
    """
    Context manager to unwrap distributed or accelerated models for generation tasks.

    This function unwraps distributed models (FSDP, DeepSpeed) and optionally overrides the model's generation_config
    temporarily during generation. This is useful for applying training-specific generation parameters without
    permanently modifying the model's original generation_config.

    Args:
        model (`DistributedDataParallel | DeepSpeedEngine`):
            Model to be unwrapped.
        accelerator ([`~accelerate.Accelerator`]):
            Accelerator instance managing the model.
        gather_deepspeed3_params (`bool`, *optional*, defaults to `True`):
            Whether to gather weights for DeepSpeed ZeRO Stage 3 models. If `False`, skips parameter gathering, which
            can be more memory-efficient but may lead to slower generation times.
        generation_kwargs (`dict`, *optional*):
            If provided, temporarily overrides the model's generation_config during generation. The original config is
            automatically restored when exiting the context. This is useful for using different generation parameters
            during training vs. inference.
        gradient_checkpointing_kwargs (`dict`, *optional*):
            Additional kwargs for gradient checkpointing enabling.

    Yields:
        Unwrapped model with optionally overridden generation_config.
    """
    with (
        _unwrap_model_for_generation(
            model,
            accelerator,
            gather_deepspeed3_params=gather_deepspeed3_params,
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        ) as unwrapped_model,
        _override_model_generation_config(unwrapped_model, generation_kwargs=generation_kwargs),
    ):
        yield unwrapped_model


def prepare_deepspeed(model: "Module", accelerator: "Accelerator"):
    # [FUNCTION DOC]
    # Wrap an auxiliary model with a DeepSpeed inference engine using the accelerator's DeepSpeed config.
    # Behavior: deep-copy the plugin config and read the ZeRO stage. For ZeRO-3 with a known hidden size (max of
    # hidden_sizes or hidden_size) set reduce_bucket_size to hidden size squared, param persistence threshold to 10
    # times hidden size and prefetch bucket size to 0.9 times hidden size squared. Drop optimizer, scheduler and
    # offload_optimizer blocks; set stage 0 when not ZeRO-3. Initialize with deepspeed.initialize and set eval mode.
    # Inputs: model (may be None), accelerator. Outputs: DeepSpeedEngine.
    # Exceptions: ImportError without deepspeed; KeyError if zero_optimization is missing. Side effects: DeepSpeed
    # initialization (distributed sharding, memory allocation).
    """Prepares the model for DeepSpeed inference or evaluation by initializing it with the appropriate configuration.

    Adapted from accelerate:
    https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    """
    import deepspeed  # local import (instead of top-level) to avoid DS init interfering with other backends (like vllm): https://github.com/deepspeedai/DeepSpeed/issues/7252

    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
    stage = config_kwargs["zero_optimization"]["stage"]

    if model is not None:
        hidden_size = (
            max(model.config.hidden_sizes)
            if getattr(model.config, "hidden_sizes", None)
            else getattr(model.config, "hidden_size", None)
        )
        if hidden_size is not None and stage == 3:
            # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache
            # @ step 0: expected module 1, but got module 0`
            # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
            config_kwargs.update(
                {
                    "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                    "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                    "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                }
            )

    # No optimizer is passed below, so drop the training-only blocks (as transformers does for inference-mode
    # deepspeed_init). Otherwise DeepSpeed builds a CPUAdam on GPU params from `optimizer` + `offload_optimizer`, or
    # an LR scheduler with no optimizer to attach to from `scheduler`, and the run dies before training.
    config_kwargs.pop("optimizer", None)
    config_kwargs.pop("scheduler", None)
    config_kwargs["zero_optimization"].pop("offload_optimizer", None)
    # If ZeRO-3 is used, we shard both the active and reference model.
    # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO
    # disabled (stage 0)
    if stage != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model


def prepare_fsdp(model, accelerator: Accelerator) -> FSDP | FSDPModule:
    # [FUNCTION DOC]
    # Wrap an auxiliary model with FSDP for inference unless already wrapped.
    # Behavior: FSDP1: set the plugin's auto-wrap policy on the model and wrap with FSDP using the plugin's
    # sharding, offload, wrap, mixed precision, prefetch, orig-params, init, ignored-modules and limit-gather
    # settings on the accelerator device. FSDP2: compute ignored params (accelerate 1.11.0 or later, else warn and
    # use None), build fully_shard kwargs (reshard_after_forward, offload policy, mixed precision policy defaulting
    # to an empty MixedPrecisionPolicy, device sub-mesh over the FSDP dims when a mesh exists, ignored_params on
    # torch 2.7.0 or later) and call fully_shard in place. Other versions raise. Set eval mode.
    # Inputs: model, accelerator. Outputs: the wrapped (or already wrapped) model.
    # Exceptions: ValueError for unsupported FSDP versions. Side effects: shards the model; may warn.
    # Check if the model is already a FSDP model due to `Manual Wrapping` and if so, don't wrap it again
    if not isinstance(model, (FSDP, FSDPModule)):
        fsdp_plugin = accelerator.state.fsdp_plugin
        if fsdp_plugin.fsdp_version == 1:
            accelerator.state.fsdp_plugin.set_auto_wrap_policy(model)
            kwargs = {
                "sharding_strategy": fsdp_plugin.sharding_strategy or fsdp_plugin.reshard_after_forward,
                "cpu_offload": fsdp_plugin.cpu_offload,
                "auto_wrap_policy": fsdp_plugin.auto_wrap_policy,
                "mixed_precision": fsdp_plugin.mixed_precision_policy,
                "sync_module_states": fsdp_plugin.sync_module_states,
                "backward_prefetch": fsdp_plugin.backward_prefetch,
                "forward_prefetch": fsdp_plugin.forward_prefetch,
                "use_orig_params": fsdp_plugin.use_orig_params,
                "param_init_fn": fsdp_plugin.param_init_fn,
                "ignored_modules": fsdp_plugin.ignored_modules,
                "limit_all_gathers": fsdp_plugin.limit_all_gathers,
                "device_id": accelerator.device,
            }
            model = FSDP(model, **kwargs)
        elif fsdp_plugin.fsdp_version == 2:
            if Version(torch.__version__) >= Version("2.6.0"):
                from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
            else:
                from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard

            mesh = getattr(accelerator, "torch_device_mesh", None)
            if Version(accelerate.__version__) >= Version("1.11.0"):
                ignored_params = get_parameters_from_modules(fsdp_plugin.ignored_modules, model, accelerator.device)
            else:
                warnings.warn(
                    "FSDP version 2 is being used with accelerate version < 1.11.0, which may lead to incorrect "
                    "handling of ignored modules. Please upgrade accelerate to v1.11.0 or later for proper support."
                )
                ignored_params = None
            fsdp2_kwargs = {
                "reshard_after_forward": fsdp_plugin.reshard_after_forward,
                "offload_policy": fsdp_plugin.cpu_offload,
                # `fully_shard` doesn't accept `None` in case of `MixedPrecisionPolicy`
                "mp_policy": fsdp_plugin.mixed_precision_policy or MixedPrecisionPolicy(),
                "mesh": mesh[tuple(accelerator.parallelism_config.fsdp_dim_names)] if mesh is not None else None,
            }
            # `ignored_params` is only supported in torch >= 2.7.0
            if Version(torch.__version__) >= Version("2.7.0"):
                fsdp2_kwargs["ignored_params"] = ignored_params
            fully_shard(model, **fsdp2_kwargs)
        else:
            raise ValueError(f"FSDP version {fsdp_plugin.fsdp_version} is not supported.")
    model.eval()
    return model


class _ForwardRedirection:
    # [CLASS DOC]
    # Utility (from PyTorch Lightning) routing a method call on an inner module through the wrapper's forward, so
    # wrapper hooks (FSDP/DDP gathering) run around it. Stateless; subclasses may override the two hook methods.
    """Implements the `forward-redirection`.

    Taken from Pytorch-lightning:
    https://github.com/Lightning-AI/pytorch-lightning/blob/02311d03fb982560246eead7c08104481fac9579/src/lightning/pytorch/strategies/strategy.py#L602

    A method call to a wrapped module gets rerouted through the wrapper's `forward` method instead.

    """

    def __call__(
        self, wrapper_module: nn.Module, original_module: nn.Module, method: Callable, *args: Any, **kwargs: Any
    ):
        # [FUNCTION DOC]
        # Behavior: temporarily replace original_module.forward with a function that restores the real forward, calls
        # method with the given arguments, and runs on_after_inner_forward; then call wrapper_module with args/kwargs,
        # run on_after_outer_forward and return the wrapper output.
        # Inputs: wrapper_module, original_module, method, args, kwargs. Outputs: wrapper output (method's result).
        # Exceptions: propagates method errors (forward stays patched if the wrapper fails before calling it).
        # Side effects: temporarily patches original_module.forward.
        """Reroutes a method call through the `wrapper_module`'s `forward` method.

        Args:
            wrapper_module: The module that has `original_module` wrapped.
            original_module: The module that was wrapped inside `wrapper_module`.
            method: The method that should be called on the `original_module` after inputs get
                redirected through the `wrapper_module`'s `forward` method.
            *args: The positional arguments to the `method`. They will get passed to a patched
                `forward` method instead.
            **kwargs: The keyword arguments to the `method`. They will get passed to a patched
                `forward` method instead.

        """
        original_forward = original_module.forward

        def wrapped_forward(*_args: Any, **_kwargs: Any) -> Any:
            # [FUNCTION DOC]
            # HELPER: Patched forward: restore the original forward, call method with the received args, run the inner hook
            # and return the result. Inputs: args, kwargs. Outputs: method result. Exceptions: method errors.
            # Side effects: restores original_module.forward.
            # Unpatch ourselves immediately before calling the method `method_name`
            # because itself may want to call the real `forward`
            original_module.forward = original_forward  # type: ignore[method-assign]
            # Call the actual method e.g. `.training_step(...)`
            out = method(*_args, **_kwargs)
            self.on_after_inner_forward(wrapper_module, original_module)
            return out

        # Patch the original_module's forward so we can redirect the arguments back to the real method
        original_module.forward = wrapped_forward  # type: ignore[method-assign]

        wrapper_output = wrapper_module(*args, **kwargs)
        self.on_after_outer_forward(wrapper_module, original_module)
        return wrapper_output

    def on_after_inner_forward(self, wrapper_module: nn.Module, original_module: nn.Module) -> None:
        # [FUNCTION DOC]
        # HELPER: Hook after the inner method call; no-op by default.
        # Inputs: wrapper_module, original_module. Outputs: none. Exceptions: none. Side effects: none.
        pass

    def on_after_outer_forward(self, wrapper_module: nn.Module, original_module: nn.Module) -> None:
        # [FUNCTION DOC]
        # HELPER: Hook after the wrapper forward; no-op by default.
        # Inputs: wrapper_module, original_module. Outputs: none. Exceptions: none. Side effects: none.
        pass


@contextmanager
def disable_gradient_checkpointing(model: PreTrainedModel, gradient_checkpointing_kwargs: dict | None = None):
    # [FUNCTION DOC]
    # Context manager turning gradient checkpointing off for the block and back on afterwards (in finally, with
    # gradient_checkpointing_kwargs passed positionally) if it was enabled.
    # Inputs: model, gradient_checkpointing_kwargs (optional). Outputs: yields None. Exceptions: none.
    # Side effects: toggles the model's checkpointing state.
    """
    Temporarily disable gradient checkpointing, restoring the previous state afterward.

    Args:
        model (`PreTrainedModel`):
            Model for which to temporarily disable gradient checkpointing.
        gradient_checkpointing_kwargs (`dict`, *optional*):
            Additional kwargs for gradient checkpointing enabling.
    """
    was_enabled = model.is_gradient_checkpointing
    if was_enabled:
        model.gradient_checkpointing_disable()
    try:
        yield
    finally:
        if was_enabled:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)


def create_reference_model(
    model: nn.Module, num_shared_layers: int | None = None, pattern: str | None = None
) -> nn.Module:
    # [FUNCTION DOC]
    # Deprecated alias: emit a FutureWarning pointing to trl.experimental.utils and delegate to its
    # create_reference_model with the same arguments.
    # Inputs: model, num_shared_layers (optional), pattern (optional). Outputs: frozen reference model.
    # Exceptions: as the experimental function. Side effects: FutureWarning.
    warnings.warn(
        "The `create_reference_model` function is now located in `trl.experimental.utils`. Please update your "
        "imports to `from trl.experimental.utils import create_reference_model`. This import path will be removed in "
        "TRL 1.0.0.",
        FutureWarning,
        stacklevel=2,
    )
    return _create_reference_model(model, num_shared_layers=num_shared_layers, pattern=pattern)
