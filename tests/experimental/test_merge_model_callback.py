# [MODULE DOC]
# Tests for the experimental MergeModelCallback (trl.experimental.merge_model_callback).
# Purpose: verify that the callback, attached to a DPOTrainer, writes a merged model into a "merged"
# subfolder of saved checkpoints. Contents: one test class, skipped unless mergekit is installed.
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


import os

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

from trl import DPOConfig, DPOTrainer
from trl.experimental.merge_model_callback import MergeConfig, MergeModelCallback

from ..testing_utils import TrlTestCase, require_mergekit


@require_mergekit
class TestMergeModelCallback(TrlTestCase):
    # [CLASS DOC]
    # End-to-end tests of MergeModelCallback with a tiny Qwen2 model trained by DPO on the "zen"
    # standard_preference dataset. Inherits TrlTestCase (provides self.tmp_dir). Gated by require_mergekit.
    # State: self.model, self.tokenizer, self.dataset built per test in setup_method.
    def setup_method(self):
        # [FUNCTION DOC]
        # HELPER: Per-test fixture loading the tiny Qwen2 causal LM (float32), its tokenizer, and the zen
        # standard_preference train split.
        # Inputs: none. Outputs: none (sets self.model, self.tokenizer, self.dataset).
        # Exceptions: propagates hub download/load errors. Side effects: network/cache access.
        self.model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", dtype="float32"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        self.dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

    def test_callback(self):
        # [FUNCTION DOC]
        # Tests that with the default MergeConfig and default merge_at_every_checkpoint (False), training one
        # epoch with save_steps=1 produces a "merged" directory inside the last checkpoint.
        # Setup: DPOConfig with step saving every step, no reporting. Assertion: merged dir exists in the
        # checkpoint found by get_last_checkpoint.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure. Side effects: writes checkpoints to tmp_dir.
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            num_train_epochs=1,
            report_to="none",
            save_strategy="steps",
            save_steps=1,
        )
        config = MergeConfig()
        merge_callback = MergeModelCallback(config)
        trainer = DPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            processing_class=self.tokenizer,
            callbacks=[merge_callback],
        )
        trainer.train()
        last_checkpoint = get_last_checkpoint(self.tmp_dir)
        merged_path = os.path.join(last_checkpoint, "merged")
        assert os.path.isdir(merged_path), "Merged folder does not exist in the last checkpoint."

    def test_every_checkpoint(self):
        # [FUNCTION DOC]
        # Tests merge_at_every_checkpoint=True: after training with save_steps=1, every "checkpoint-*"
        # directory in tmp_dir must contain a "merged" subdirectory.
        # Inputs/Outputs: none. Exceptions: AssertionError naming the offending checkpoint.
        # Side effects: writes checkpoints and merged models to tmp_dir.
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            num_train_epochs=1,
            report_to="none",
            save_strategy="steps",
            save_steps=1,
        )
        config = MergeConfig()
        merge_callback = MergeModelCallback(config, merge_at_every_checkpoint=True)
        trainer = DPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            processing_class=self.tokenizer,
            callbacks=[merge_callback],
        )
        trainer.train()

        checkpoints = sorted(
            [os.path.join(self.tmp_dir, cp) for cp in os.listdir(self.tmp_dir) if cp.startswith("checkpoint-")]
        )

        for checkpoint in checkpoints:
            merged_path = os.path.join(checkpoint, "merged")
            assert os.path.isdir(merged_path), f"Merged folder does not exist in checkpoint {checkpoint}."
