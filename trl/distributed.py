# [MODULE DOC]
# Small abstraction over the distributed backend (DeepSpeed ZeRO vs FSDP) used by trainers when they need full
# parameters (e.g. merging adapters, syncing weights to vLLM, generation).
# Contents: DistributedBackend. Import-time side effects: none (deepspeed / torch FSDP imported lazily).

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

"""Distributed training backend abstraction."""

from contextlib import contextmanager


class DistributedBackend:
    # [CLASS DOC]
    # Detects the backend once from an accelerate Accelerator and exposes no-op-when-unneeded context managers.
    # State: zero_stage (DeepSpeed ZeRO stage, 0 when DeepSpeed is not used) and fsdp_version (1, 2, or None when FSDP
    # is not used). Properties is_zero3 / is_fsdp derive from them; gather_params handles ZeRO-3 and summon_full_params
    # handles FSDP v1.
    """Abstracts distributed backend specifics (DeepSpeed ZeRO, FSDP) behind a uniform API.

    Detects the active backend once at construction from ``accelerator.state``, then provides context managers that are
    no-ops on backends where the operation is not needed.

    Args:
        accelerator ([`~accelerate.Accelerator`]):
            The accelerator instance managing the distributed state.

    Example:

    ```python
    >>> dist = DistributedBackend(accelerator)
    >>> with dist.gather_params(list(model.parameters())):
    ...     model.merge_adapter()
    >>> with dist.summon_full_params(model, recurse=False):
    ...     outputs = model.generate(inputs)
    ```
    """

    def __init__(self, accelerator):
        # [FUNCTION DOC]
        # Reads accelerator.state.deepspeed_plugin and fsdp_plugin (if present) to set zero_stage and fsdp_version.
        # Inputs: accelerator. Outputs: none. Exceptions: AttributeError if accelerator lacks a state.
        # Side effects: sets attributes.
        ds_plugin = accelerator.state.deepspeed_plugin
        fsdp_plugin = getattr(accelerator.state, "fsdp_plugin", None)
        self.zero_stage = ds_plugin.zero_stage if ds_plugin else 0
        self.fsdp_version = getattr(fsdp_plugin, "fsdp_version", None) if fsdp_plugin else None

    @property
    def is_zero3(self) -> bool:
        # [FUNCTION DOC]
        # Property: True when zero_stage equals 3. Inputs: none. Outputs: bool. Exceptions: none. Side effects: none.
        """Whether DeepSpeed ZeRO Stage 3 is active."""
        return self.zero_stage == 3

    @property
    def is_fsdp(self) -> bool:
        # [FUNCTION DOC]
        # Property: True when any FSDP version is active (fsdp_version not None). Outputs: bool. Exceptions/side effects: none.
        """Whether FSDP (any version) is active."""
        return self.fsdp_version is not None

    @contextmanager
    def gather_params(self, params):
        # [FUNCTION DOC]
        # Context manager: under ZeRO-3 enters deepspeed.zero.GatheredParameters over params so they are materialized in the
        # block; otherwise just yields.
        # Inputs: params (iterable of parameters). Outputs: yields nothing.
        # Exceptions: ImportError if deepspeed is missing under ZeRO-3. Side effects: temporary all-gather of parameters.
        """Gather sharded parameters under DeepSpeed ZeRO-3; no-op otherwise.

        Args:
            params (iterable of `torch.nn.Parameter`):
                Parameters to gather.
        """
        if self.is_zero3:
            import deepspeed

            with deepspeed.zero.GatheredParameters(params):
                yield
        else:
            yield

    @contextmanager
    def summon_full_params(self, module, **kwargs):
        # [FUNCTION DOC]
        # Context manager: for FSDP v1 enters FullyShardedDataParallel.summon_full_params(module, kwargs); for FSDP v2 or no
        # FSDP just yields.
        # Inputs: module, kwargs forwarded (e.g. recurse, writeback). Outputs: yields nothing.
        # Exceptions: FSDP errors propagate. Side effects: temporarily unshards parameters.
        """Materialize full FSDP v1 parameters; no-op for FSDP v2 or non-FSDP backends.

        FSDP v2 parameters are always accessible and require no explicit materialization.

        Args:
            module (`torch.nn.Module`):
                The FSDP-wrapped module.
            **kwargs:
                Forwarded to ``FSDP.summon_full_params`` (e.g. ``recurse``, ``writeback``).
        """
        if self.fsdp_version == 1:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            with FSDP.summon_full_params(module, **kwargs):
                yield
        else:
            yield
