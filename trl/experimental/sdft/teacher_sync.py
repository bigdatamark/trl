# [MODULE DOC]
# Teacher-synchronization utilities for SDFT/self-distillation trainers.
# Contents: is_pure_lora_training (detects LoRA-only training), SyncTeacherModelCallback (EMA or hard sync of a separate
# teacher model, reusing SyncRefModelCallback), and PEFTAdapterEMACallback (keeps an EMA copy of the PEFT adapter as a
# second "teacher" adapter on the same model). Import-time side effects: creates a logger; peft is imported lazily.

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

import logging

import torch
from accelerate.utils import is_peft_model
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from ...trainer.callbacks import SyncRefModelCallback


logger = logging.getLogger(__name__)


def is_pure_lora_training(model, accelerator=None) -> bool:
    # [FUNCTION DOC]
    # HELPER: Reports whether the model is a PEFT model whose active adapter is LoRA and all trainable parameters are LoRA
    # parameters (names containing "lora_").
    # Behavior: returns False for non-PEFT models; unwraps via the accelerator if given; uses the active adapter (default
    # "default"); compares the last dotted component of the peft type string with "LORA".
    # Inputs: model, accelerator (optional). Outputs: bool.
    # Exceptions: AttributeError if the adapter config is missing. Side effects: none.
    # KNOWN ISSUE: the PEFT check runs before unwrapping, so a model wrapped by DDP/DeepSpeed may be reported as non-PEFT.
    """Return `True` when the active adapter is LoRA and every trainable parameter is a LoRA parameter."""
    if not is_peft_model(model):
        return False

    if accelerator is not None:
        model = accelerator.unwrap_model(model)

    adapter_name = model.active_adapter
    if adapter_name is None:
        adapter_name = "default"
    adapter_config = model.peft_config.get(adapter_name)
    peft_type = adapter_config.peft_type
    if peft_type is None or str(peft_type).split(".")[-1] != "LORA":
        return False

    for name, param in model.named_parameters():
        if param.requires_grad and "lora_" not in name:
            return False
    return True


class SyncTeacherModelCallback(SyncRefModelCallback):
    # [CLASS DOC]
    # Callback that periodically updates a separate teacher model from the student by mixing weights.
    # Subclass of SyncRefModelCallback, storing the teacher as ref_model and reusing its sync_target_model (blend with
    # alpha: target = (1 - alpha) * target + alpha * student, handling ZeRO-3 gathering).
    """Synchronize an EMA teacher model with the student model on each configured sync step."""

    def __init__(self, teacher_model, accelerator=None):
        # [FUNCTION DOC]
        # Stores the teacher model and optional accelerator via the parent constructor.
        # Inputs: teacher_model, accelerator (optional). Outputs: none. Side effects: none.
        super().__init__(ref_model=teacher_model, accelerator=accelerator)

    def on_step_end(self, args, state, control, **kwargs):
        # [FUNCTION DOC]
        # On every step whose global_step is a multiple of args.teacher_sync_steps, unwraps the student (when an accelerator is
        # set) and blends it into the teacher with args.teacher_update_rate.
        # Inputs: args (needs teacher_sync_steps and teacher_update_rate), state, control, kwargs with "model".
        # Outputs: none. Exceptions: KeyError if model missing. Side effects: mutates teacher weights.
        model = kwargs["model"]
        if self.ref_model is not None and state.global_step % args.teacher_sync_steps == 0:
            if self.accelerator:
                model = self.accelerator.unwrap_model(model)
            self.sync_target_model(model, self.ref_model, args.teacher_update_rate)


class PEFTAdapterEMACallback(TrainerCallback):
    # [CLASS DOC]
    # Callback maintaining an EMA teacher as a second PEFT adapter on the student model.
    # State: model, teacher_adapter_name ("teacher"), update_rate (0.05), sync_steps (1), accelerator, shadow_weights
    # (dict of EMA tensors keyed by PEFT state-dict keys), teacher_adapter_config, _initialized.
    # Flow: on_train_begin (or first sync) adds a zero-initialized teacher adapter; on_step_end updates shadow weights
    # from the student adapter and loads them into the teacher adapter.
    """
    Callback that maintains an EMA copy of PEFT adapter weights for use as a teacher model in self-distillation.

    The callback creates a secondary adapter ("teacher") with zero-initialized weights and maintains shadow weights
    that are updated via exponential moving average: `teacher_weight = (1-α) * teacher_weight + α * student_weight`

    Usage:
        ```python
        >>> trainer.add_callback(
        ...     PEFTAdapterEMACallback(
        ...         model=model,
        ...         teacher_adapter_name="teacher",
        ...         update_rate=0.05,
        ...     )
        ... )
        ```
    """

    def __init__(
        self,
        model,
        teacher_adapter_name: str = "teacher",
        update_rate: float = 0.05,
        sync_steps: int = 1,
        accelerator=None,
    ):
        # [FUNCTION DOC]
        # Stores configuration; no adapter is created yet.
        # Inputs: model, teacher_adapter_name, update_rate, sync_steps, accelerator. Outputs: none. Side effects: none.
        self.model = model
        self.teacher_adapter_name = teacher_adapter_name
        self.update_rate = update_rate
        self.sync_steps = sync_steps
        self.accelerator = accelerator
        self.shadow_weights: dict[str, torch.Tensor] | None = None
        self.teacher_adapter_config = None
        self._initialized = False

    def _get_student_state_dict(self):
        # [FUNCTION DOC]
        # HELPER: Returns the active adapter's PEFT state dict (keys without adapter name) of the (unwrapped) model.
        # Inputs: none. Outputs: dict of tensors. Side effects: imports peft.
        """Get student adapter state dict using PEFT keys (without adapter name)."""
        from peft import get_peft_model_state_dict

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(self.model)
        else:
            model = self.model
        return get_peft_model_state_dict(model)

    def _initialize_teacher_adapter(self):
        # [FUNCTION DOC]
        # HELPER: Creates the teacher adapter once.
        # Behavior: no-op if already initialized; unwraps the model; takes the active adapter (default "default") config;
        # adds a new adapter with the same config, sets it active, loads all-zero tensors shaped like the student state, then
        # restores the original active adapter; shadow weights start as zeros. Logs at info.
        # Inputs: none. Outputs: none. Side effects: mutates the model's adapters and active adapter temporarily.
        # KNOWN ISSUE: the docstring says the teacher is "initialized from student adapter", but the weights are zeros
        # (so for LoRA the teacher starts as the base model), and EMA starts from zero rather than the student weights.
        """Create teacher adapter with zero weights initialized from student adapter."""
        from peft import get_peft_model_state_dict, set_peft_model_state_dict

        if self._initialized:
            return

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(self.model)
        else:
            model = self.model

        adapter_name = model.active_adapter
        if adapter_name is None:
            adapter_name = "default"

        self.teacher_adapter_config = model.peft_config.get(adapter_name)

        student_state = get_peft_model_state_dict(model)

        teacher_state = {k: torch.zeros_like(v) for k, v in student_state.items()}

        model.add_adapter(self.teacher_adapter_name, self.teacher_adapter_config)

        model.set_adapter(self.teacher_adapter_name)
        set_peft_model_state_dict(model, teacher_state, adapter_name=self.teacher_adapter_name)

        model.set_adapter(adapter_name)

        self.shadow_weights = {k: v.clone().zero_() for k, v in teacher_state.items()}

        self._initialized = True
        logger.info(f"Initialized PEFT adapter EMA teacher with adapter name: {self.teacher_adapter_name}")

    @torch.no_grad()
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # [FUNCTION DOC]
        # Every sync_steps steps: initializes the teacher adapter if needed, picks up the accelerator from kwargs if missing,
        # updates each shadow tensor with (1 - update_rate) * shadow + update_rate * student, and loads the shadow weights into
        # the teacher adapter (temporarily activating it, then restoring the previous active adapter). Runs without grad.
        # Inputs: args, state, control, kwargs. Outputs: none. Side effects: mutates shadow weights and teacher adapter.
        if state.global_step % self.sync_steps != 0:
            return

        if not self._initialized:
            self._initialize_teacher_adapter()

        if self.shadow_weights is None:
            return

        if self.accelerator is None and "accelerator" in kwargs:
            self.accelerator = kwargs["accelerator"]

        student_state = self._get_student_state_dict()

        for key, student_param in student_state.items():
            if key in self.shadow_weights:
                shadow = self.shadow_weights[key]
                shadow.data = (1 - self.update_rate) * shadow.data + self.update_rate * student_param.data

        from peft import set_peft_model_state_dict

        if self.accelerator is not None:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
        else:
            unwrapped_model = self.model

        original_adapter = unwrapped_model.active_adapter
        unwrapped_model.set_adapter(self.teacher_adapter_name)
        set_peft_model_state_dict(unwrapped_model, self.shadow_weights, adapter_name=self.teacher_adapter_name)
        unwrapped_model.set_adapter(original_adapter)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # [FUNCTION DOC]
        # Picks up the accelerator from kwargs if not set and initializes the teacher adapter.
        # Inputs: args, state, control, kwargs. Outputs: none. Side effects: adds the teacher adapter.
        if self.accelerator is None and "accelerator" in kwargs:
            self.accelerator = kwargs["accelerator"]
        self._initialize_teacher_adapter()
