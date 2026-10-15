# [MODULE DOC]
# Module trl.experimental.sdpo.teacher_sync: utilities for keeping the SDPO self-distillation teacher in sync
# with the student during training.
# Main contents: is_pure_lora_training (detects LoRA-only training so the teacher can be an extra adapter),
# SyncTeacherModelCallback (EMA sync of a full separate teacher model, built on SyncRefModelCallback) and
# PEFTAdapterEMACallback (EMA teacher kept as a second PEFT adapter on the same base model).
# Used by the SDPO trainer to choose and register the teacher-sync strategy.
# Import-time side effects: creates a module-level logger; peft is imported lazily inside methods.

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
    # HELPER: Decide whether the model is trained purely through a LoRA adapter.
    # Behavior: return False if the model is not a PEFT model. Unwrap via accelerator when one is given. Read the
    # active adapter name (fall back to "default" when None), fetch its config from peft_config, and return False
    # unless the peft_type's last dotted component is "LORA". Then return False if any parameter that requires
    # grad does not contain "lora_" in its name; otherwise True.
    # Inputs: model (any model, possibly accelerate-wrapped), accelerator (optional, used only to unwrap).
    # Outputs: bool.
    # Exceptions: AttributeError if the active adapter name has no entry in peft_config (config is None).
    # Side effects: none.
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
    # Trainer callback that keeps a separate full teacher model as an EMA of the student.
    # Inherits SyncRefModelCallback and reuses its ref_model slot (holding the teacher) and its sync_target_model
    # mixing routine; only changes the trigger condition and the config fields read (teacher_sync_steps and
    # teacher_update_rate on the training args instead of the ref-model fields).
    """Synchronize an EMA teacher model with the student model on each configured sync step."""

    def __init__(self, teacher_model, accelerator=None):
        # [FUNCTION DOC]
        # Behavior: forward teacher_model as ref_model and accelerator to SyncRefModelCallback.__init__.
        # Inputs: teacher_model (the teacher module), accelerator (optional Accelerator).
        # Outputs: none. Exceptions: none. Side effects: stores the attributes on self via the parent.
        super().__init__(ref_model=teacher_model, accelerator=accelerator)

    def on_step_end(self, args, state, control, **kwargs):
        # [FUNCTION DOC]
        # Behavior: at each optimizer step end, if a teacher exists and global_step is a multiple of
        # args.teacher_sync_steps, take the student from kwargs["model"] (unwrapped through the accelerator when set)
        # and blend it into the teacher with sync_target_model using args.teacher_update_rate as the mixing weight.
        # Inputs: args (training config with teacher_sync_steps / teacher_update_rate), state (global_step), control,
        # kwargs (must contain "model").
        # Outputs: none.
        # Exceptions: KeyError if "model" is missing from kwargs; ZeroDivisionError if teacher_sync_steps is 0.
        # Side effects: mutates teacher parameters in place.
        model = kwargs["model"]
        if self.ref_model is not None and state.global_step % args.teacher_sync_steps == 0:
            if self.accelerator:
                model = self.accelerator.unwrap_model(model)
            self.sync_target_model(model, self.ref_model, args.teacher_update_rate)


class PEFTAdapterEMACallback(TrainerCallback):
    # [CLASS DOC]
    # Trainer callback maintaining an EMA teacher as an additional PEFT adapter (named teacher_adapter_name) on
    # the student model, so no second model copy is needed.
    # State: model, teacher_adapter_name, update_rate (EMA alpha), sync_steps, accelerator, shadow_weights
    # (dict of PEFT state-dict keys to fp tensors holding the EMA, None until initialized), teacher_adapter_config,
    # _initialized flag.
    # Flow: on_train_begin creates the teacher adapter (zero weights) via _initialize_teacher_adapter; on_step_end
    # every sync_steps steps updates shadow_weights with the student's adapter weights and loads them into the
    # teacher adapter. Inherits transformers TrainerCallback.
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
        # Behavior: store the constructor arguments; set shadow_weights and teacher_adapter_config to None and
        # _initialized to False. No adapter is created here.
        # Inputs: model (PEFT model, may be accelerate-wrapped), teacher_adapter_name (default "teacher"),
        # update_rate (default 0.05), sync_steps (default 1), accelerator (optional).
        # Outputs: none. Exceptions: none. Side effects: none beyond attribute assignment.
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
        # HELPER: Return the PEFT state dict of the currently active (student) adapter.
        # Behavior: unwrap self.model through the accelerator if set, then call peft get_peft_model_state_dict, which
        # returns keys without the adapter name.
        # Inputs: none. Outputs: dict of parameter name to tensor.
        # Exceptions: ImportError if peft is not installed. Side effects: none.
        """Get student adapter state dict using PEFT keys (without adapter name)."""
        from peft import get_peft_model_state_dict

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(self.model)
        else:
            model = self.model
        return get_peft_model_state_dict(model)

    def _initialize_teacher_adapter(self):
        # [FUNCTION DOC]
        # HELPER: Create the teacher adapter once.
        # Behavior: no-op when already initialized. Unwrap the model; take the active adapter name (default
        # "default") and its config; build a zero tensor for every key in the student state dict; add a new adapter
        # named teacher_adapter_name with the same config; switch to it, load the zero state into it, switch back to
        # the student adapter. Set shadow_weights to zeroed clones of the teacher state, mark initialized and log info.
        # Note the teacher therefore starts at zero, not as a copy of the student; EMA moves it toward the student.
        # Inputs: none. Outputs: none.
        # Exceptions: ImportError without peft; errors from add_adapter if the name already exists.
        # Side effects: adds an adapter to the model, toggles the active adapter, sets shadow_weights/_initialized,
        # logs.
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
        # Behavior (under no_grad): return unless global_step is a multiple of sync_steps. Initialize lazily if
        # needed; return if shadow_weights is still None. Adopt kwargs["accelerator"] if no accelerator is set. For
        # every student state key present in shadow_weights, set shadow to (1 - update_rate) times shadow plus
        # update_rate times student. Then on the unwrapped model, remember the active adapter, activate the teacher
        # adapter, load shadow_weights into it, and restore the original adapter.
        # Inputs: args, state (global_step), control, kwargs (optional "accelerator").
        # Outputs: none.
        # Exceptions: ZeroDivisionError if sync_steps is 0; ImportError without peft.
        # Side effects: mutates shadow_weights and the teacher adapter weights; may set self.accelerator.
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
        # Behavior: adopt kwargs["accelerator"] if none set, then create the teacher adapter (idempotent).
        # Inputs: args, state, control, kwargs. Outputs: none. Exceptions: as _initialize_teacher_adapter.
        # Side effects: may add the teacher adapter to the model and set self.accelerator.
        if self.accelerator is None and "accelerator" in kwargs:
            self.accelerator = kwargs["accelerator"]
        self._initialize_teacher_adapter()
