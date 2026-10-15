# [MODULE DOC]
# Smoke test for trl.trainer.callbacks.RichProgressCallback used with a plain transformers Trainer.
# Main contents: a one-parameter dummy model and a test class gated by require_rich.
# Import-time side effects: none.

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
import torch.nn as nn
from datasets import Dataset
from transformers import Trainer, TrainingArguments

from trl.trainer.callbacks import RichProgressCallback

from .testing_utils import TrlTestCase, require_rich


class DummyModel(nn.Module):
    # [CLASS DOC]
    # HELPER: Minimal torch module with a single scalar parameter a (initialized to 1.0) used as a Trainer model.
    # Construction: no arguments. Methods: forward multiplies input by a.
    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Initializes the module and registers scalar parameter a = 1.0.
        # Inputs: none. Outputs: none. Exceptions: none. Side effects: registers a parameter.
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # [FUNCTION DOC]
        # HELPER: Returns a times x.
        # Inputs: x (tensor or float). Outputs: tensor product. Exceptions: none. Side effects: none.
        # Note: ignores the y column; the Trainer takes the first element of the returned tensor as the loss.
        return self.a * x


@require_rich
class TestRichProgressCallback(TrlTestCase):
    # [CLASS DOC]
    # Tests that training and frequent evaluation run end to end with RichProgressCallback attached (no assertions,
    # success means no exception). Skipped if rich is unavailable. Inherits TrlTestCase (provides tmp_dir).
    def setup_method(self):
        # [FUNCTION DOC]
        # HELPER: Builds the dummy model, a 5-row train dataset and a 101-row eval dataset of {x: 1.0, y: 2.0}.
        # Inputs: none. Outputs: none. Exceptions: none. Side effects: sets instance attributes.
        self.dummy_model = DummyModel()
        self.dummy_train_dataset = Dataset.from_list([{"x": 1.0, "y": 2.0}] * 5)
        self.dummy_val_dataset = Dataset.from_list([{"x": 1.0, "y": 2.0}] * 101)

    def test_rich_progress_callback_logging(self):
        # [FUNCTION DOC]
        # Runs Trainer.train for 4 epochs with batch size 2, eval and logging every step, tqdm disabled, no saving or
        # reporting, and the RichProgressCallback attached. Asserts nothing explicitly; passes if no exception.
        # Inputs: none. Outputs: none. Exceptions: any training error. Side effects: writes into tmp_dir.
        training_args = TrainingArguments(
            output_dir=self.tmp_dir,
            per_device_eval_batch_size=2,
            per_device_train_batch_size=2,
            num_train_epochs=4,
            eval_strategy="steps",
            eval_steps=1,
            logging_strategy="steps",
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            disable_tqdm=True,
        )
        callbacks = [RichProgressCallback()]
        trainer = Trainer(
            model=self.dummy_model,
            train_dataset=self.dummy_train_dataset,
            eval_dataset=self.dummy_val_dataset,
            args=training_args,
            callbacks=callbacks,
        )

        trainer.train()
