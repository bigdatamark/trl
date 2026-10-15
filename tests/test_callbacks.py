# [MODULE DOC]
# Tests for the TRL Trainer callbacks LogCompletionsCallback (logs generated completions for eval prompts to
# Weights & Biases or Comet) and BEMACallback (bias-corrected EMA of weights, saved under output_dir/bema).
# Each test runs a plain transformers Trainer on tiny Qwen2 and zen datasets with the callback attached.
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

import json
import os
from unittest.mock import call, patch

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, Trainer, TrainingArguments

from trl import BEMACallback, LogCompletionsCallback

from .testing_utils import TrlTestCase, require_comet, require_wandb


class TestLogCompletionsCallback(TrlTestCase):
    # [CLASS DOC]
    # Tests LogCompletionsCallback with real tracking backends. State from setup_method: model, tokenizer, tokenized
    # prompt-only dataset (8 train rows) and a GenerationConfig with max_length 32. Inherits TrlTestCase.
    def setup_method(self):
        # [FUNCTION DOC]
        # HELPER: Per-test fixture: loads tiny Qwen2 model and tokenizer, copies the tokenizer pad id into the model config,
        # loads zen standard_prompt_only, keeps 8 train rows, tokenizes prompts (pad/truncate to 16, labels equal input
        # ids) and creates the generation config.
        # Inputs: none. Outputs: none. Exceptions: load errors propagate. Side effects: sets instance attributes.
        self.model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        self.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        # `Trainer` realigns the configs at train time, so mirror the pad token as the TRL trainers do at init.
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")
        dataset["train"] = dataset["train"].select(range(8))

        def tokenize_function(examples):
            # [FUNCTION DOC]
            # HELPER: Batched map function tokenizing the "prompt" column to max_length 16 with padding and truncation and
            # adding labels as a copy of input_ids. Inputs: examples - batch dict. Outputs: tokenizer BatchEncoding with labels.
            # Exceptions: none. Side effects: none.
            out = self.tokenizer(examples["prompt"], padding="max_length", max_length=16, truncation=True)
            out["labels"] = out["input_ids"].copy()
            return out

        self.dataset = dataset.map(tokenize_function, batched=True)

        self.generation_config = GenerationConfig(max_length=32)

    @require_wandb
    def test_basic_wandb(self):
        # [FUNCTION DOC]
        # Trains with eval every 2 steps (batch size 2) reporting to wandb and the callback logging 2 prompts, then reads
        # the completions table JSON from the wandb run directory.
        # Assertions: columns include step, prompt and completion; the first test prompt appears in the first data row.
        # Requires wandb. Inputs/Outputs: none. Exceptions: assertion failure.
        # Side effects: creates a wandb run and local files.
        import wandb

        training_args = TrainingArguments(
            output_dir=self.tmp_dir,
            eval_strategy="steps",
            eval_steps=2,  # evaluate every 2 steps
            per_device_train_batch_size=2,  # 8 samples in total so 4 batches of 2 per epoch
            per_device_eval_batch_size=2,
            report_to="wandb",
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            processing_class=self.tokenizer,
        )
        completions_callback = LogCompletionsCallback(trainer, self.generation_config, num_prompts=2)
        trainer.add_callback(completions_callback)
        trainer.train()

        # Get the current run
        completions_path = wandb.run.summary.completions["path"]
        json_path = os.path.join(wandb.run.dir, completions_path)
        with open(json_path) as f:
            completions = json.load(f)

        # Check that the columns are correct
        assert "step" in completions["columns"]
        assert "prompt" in completions["columns"]
        assert "completion" in completions["columns"]

        # Check that the prompt is in the log
        assert self.dataset["test"][0]["prompt"] in completions["data"][0]

    @require_comet
    def test_basic_comet(self):
        # [FUNCTION DOC]
        # Trains with eval every 2 steps reporting to comet_ml with the callback (2 prompts), ends the running experiment to
        # flush, then queries the Comet API for dataframe assets.
        # Assertions: the experiment exists; number of logged tables equals half of (train plus test rows) plus one; every
        # table file is named completions.csv. Requires comet_ml.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: network calls to Comet.
        import comet_ml

        training_args = TrainingArguments(
            output_dir=self.tmp_dir,
            eval_strategy="steps",
            eval_steps=2,  # evaluate every 2 steps
            per_device_train_batch_size=2,  # 8 samples in total so 4 batches of 2 per epoch
            per_device_eval_batch_size=2,
            report_to="comet_ml",
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            processing_class=self.tokenizer,
        )
        completions_callback = LogCompletionsCallback(trainer, self.generation_config, num_prompts=2)
        trainer.add_callback(completions_callback)
        trainer.train()

        # close experiment to make sure all pending data are flushed
        experiment = comet_ml.get_running_experiment()
        assert experiment is not None
        experiment.end()

        # get experiment assets and check that all required tables was logged
        steps = len(self.dataset["train"]) + len(self.dataset["test"])
        tables_logged = int(steps / 2) + 1  # +1 to include zero step

        api_experiment = comet_ml.APIExperiment(previous_experiment=experiment.id)
        tables = api_experiment.get_asset_list("dataframe")
        assert tables is not None
        assert len(tables) == tables_logged
        assert all(table["fileName"] == "completions.csv" for table in tables)


class TestBEMACallback(TrlTestCase):
    # [CLASS DOC]
    # Tests BEMACallback. State from setup_method: tiny Qwen2 model and tokenizer, zen standard_language_modeling
    # dataset tokenized to max length 17 with labels. Inherits TrlTestCase.
    def setup_method(self):
        # [FUNCTION DOC]
        # HELPER: Per-test fixture: loads model/tokenizer, mirrors pad token id into the model config, tokenizes the text
        # column (padding to 17, labels copy input ids, text column removed).
        # Inputs: none. Outputs: none. Exceptions: load errors propagate. Side effects: sets instance attributes.
        self.model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        self.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        # `Trainer` realigns the configs at train time, so mirror the pad token as the TRL trainers do at init.
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling")

        def tokenize_function(examples, tokenizer):
            # [FUNCTION DOC]
            # HELPER: Batched map function tokenizing "text" with padding to max_length 17 and adding labels as a copy of
            # input_ids. Inputs: examples - batch dict; tokenizer. Outputs: BatchEncoding. Exceptions/Side effects: none.
            out = tokenizer(examples["text"], padding="max_length", max_length=17)
            out["labels"] = out["input_ids"].copy()
            return out

        self.dataset = dataset.map(
            tokenize_function, fn_kwargs={"tokenizer": self.tokenizer}, remove_columns=["text"], batched=True
        )

    def test_model_saved(self):
        # [FUNCTION DOC]
        # Checks BEMACallback(update_freq 2) writes a loadable model to output_dir/bema after training.
        # Setup: TrainingArguments with report_to none (default 3 epochs, batch size 8) and the callback attached to a
        # plain Trainer on the language-modeling train split.
        # Assertions: the bema directory exists and AutoModelForCausalLM loads from it.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: trains and writes checkpoints.
        """Test that BEMACallback saves the BEMA model."""
        training_args = TrainingArguments(output_dir=self.tmp_dir, report_to="none")
        bema_callback = BEMACallback(update_freq=2)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            processing_class=self.tokenizer,
            callbacks=[bema_callback],
        )
        trainer.train()

        # Check that the BEMA model was saved and can be loaded
        bema_path = os.path.join(self.tmp_dir, "bema")
        assert os.path.isdir(bema_path), "BEMA directory was not created"
        AutoModelForCausalLM.from_pretrained(bema_path)

    def test_update_frequency_0(self):
        # [FUNCTION DOC]
        # Checks update_freq 2 calls _update_bema_weights at steps 2, 4, 6, 8.
        # KNOWN ISSUE: the inline comment lists the updates as 2, 4, 5, 8, which disagrees with the asserted 2, 4, 6, 8.
        # Setup: _update_bema_weights is patched with a mock on the callback instance; a plain Trainer with default args
        # (3 epochs, batch size 8, 17 samples, 9 optimizer steps) trains with the callback attached.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: trains in self.tmp_dir.
        """Test that BEMA callback respects the update frequency."""
        training_args = TrainingArguments(output_dir=self.tmp_dir, report_to="none")
        bema_callback = BEMACallback(update_freq=2)

        with patch.object(bema_callback, "_update_bema_weights") as mock_update:
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset["train"],
                processing_class=self.tokenizer,
                callbacks=[bema_callback],
            )

            trainer.train()

            # Total 9 steps (17 samples, batch size 8, 3 epochs).
            # BEMA starts after step 0 and updates every 2 steps → updates at 2, 4, 5, 8
            assert mock_update.call_args_list == [call(2), call(4), call(6), call(8)]

    def test_update_frequency_1(self):
        # [FUNCTION DOC]
        # Checks update_freq 3 calls _update_bema_weights at steps 3, 6, 9.
        # Setup: _update_bema_weights is patched with a mock on the callback instance; a plain Trainer with default args
        # (3 epochs, batch size 8, 17 samples, 9 optimizer steps) trains with the callback attached.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: trains in self.tmp_dir.
        """Test that BEMA callback respects the update frequency."""
        training_args = TrainingArguments(output_dir=self.tmp_dir, report_to="none")
        bema_callback = BEMACallback(update_freq=3)

        with patch.object(bema_callback, "_update_bema_weights") as mock_update:
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset["train"],
                processing_class=self.tokenizer,
                callbacks=[bema_callback],
            )

            trainer.train()

            # Total 9 steps (17 samples, batch size 8, 3 epochs).
            # BEMA starts after step 0 and updates every 3 steps → updates at 3, 6, 9
            assert mock_update.call_args_list == [call(3), call(6), call(9)]

    def test_update_frequency_2(self):
        # [FUNCTION DOC]
        # Checks update_freq 2 with update_after 3 calls _update_bema_weights at steps 5, 7, 9.
        # Setup: _update_bema_weights is patched with a mock on the callback instance; a plain Trainer with default args
        # (3 epochs, batch size 8, 17 samples, 9 optimizer steps) trains with the callback attached.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: trains in self.tmp_dir.
        """Test that BEMA callback respects the update frequency."""
        training_args = TrainingArguments(output_dir=self.tmp_dir, report_to="none")
        bema_callback = BEMACallback(update_freq=2, update_after=3)

        with patch.object(bema_callback, "_update_bema_weights") as mock_update:
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset["train"],
                processing_class=self.tokenizer,
                callbacks=[bema_callback],
            )

            trainer.train()

            # Total 9 steps (17 samples, batch size 8, 3 epochs).
            # BEMA starts after step 3 and updates every 2 steps → updates at 5, 7, 9
            assert mock_update.call_args_list == [call(5), call(7), call(9)]

    def test_bias_power_zero(self):
        # [FUNCTION DOC]
        # Smoke test: training completes with BEMACallback(update_freq 2, bias_power 0.0).
        # Setup: TrainingArguments with report_to none (default 3 epochs, batch size 8) and the callback attached to a
        # plain Trainer on the language-modeling train split.
        # Inputs/Outputs: none. Exceptions: any error fails. Side effects: trains in self.tmp_dir.
        """Test that BEMACallback works with bias_power=0.0 (maximum, undecayed bias-correction)."""
        training_args = TrainingArguments(output_dir=self.tmp_dir, report_to="none")
        bema_callback = BEMACallback(update_freq=2, bias_power=0.0)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            processing_class=self.tokenizer,
            callbacks=[bema_callback],
        )
        trainer.train()

    def test_no_ema(self):
        # [FUNCTION DOC]
        # Smoke test: training completes with BEMACallback(update_freq 2, ema_power 0.0), i.e. no EMA component.
        # Setup: TrainingArguments with report_to none (default 3 epochs, batch size 8) and the callback attached to a
        # plain Trainer on the language-modeling train split.
        # Inputs/Outputs: none. Exceptions: any error fails. Side effects: trains in self.tmp_dir.
        """Test that BEMACallback works without EMA updates."""
        training_args = TrainingArguments(output_dir=self.tmp_dir, report_to="none")
        bema_callback = BEMACallback(update_freq=2, ema_power=0.0)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            processing_class=self.tokenizer,
            callbacks=[bema_callback],
        )
        trainer.train()
