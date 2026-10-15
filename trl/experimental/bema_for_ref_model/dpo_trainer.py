# [MODULE DOC]
# Experimental DPOTrainer that exposes the reference model to callbacks, enabling BEMACallback to update it.
# Main content: DPOTrainer subclass of the stable trl DPOTrainer. No import-time side effects.

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

from ...trainer.dpo_trainer import DPOTrainer as _DPOTrainer
from .callback import CallbackHandlerWithRefModel


class DPOTrainer(_DPOTrainer):
    # [CLASS DOC]
    # Stable DPOTrainer whose callback_handler is replaced by CallbackHandlerWithRefModel after construction; all other
    # behavior is inherited.
    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Behavior: runs the parent constructor, then rebuilds callback_handler from the existing handler's callbacks plus
        # model, ref_model, processing_class, optimizer and lr_scheduler.
        # Inputs: same arguments as the stable DPOTrainer. Outputs: none.
        # Exceptions: those of the parent constructor.
        # Side effects: replaces self.callback_handler.
        super().__init__(*args, **kwargs)
        # Replace with a new one that calls the events with the reference model
        self.callback_handler = CallbackHandlerWithRefModel(
            self.callback_handler.callbacks,
            self.model,
            self.ref_model,
            self.processing_class,
            self.optimizer,
            self.lr_scheduler,
        )
