# [MODULE DOC]
# Tests for the experimental TPO (Triple Preference Optimization) trainer in trl.experimental.tpo.
# Main contents: a helper that synthesizes the required "reference" (gold) completion column from "chosen";
# tests for DataCollatorForTriplePreference padding/mask layout with and without the reference branch; and
# TPOTrainer integration tests (training, eval dataset variants, evaluate on raw data, trust_remote_code, loss
# types, conversational and implicit-prompt data, tpo_alpha 0, missing/mismatched reference errors, seeded PEFT
# init, PEFT training, pad token syncing). Uses trl-internal-testing/zen datasets and tiny models.
# Import-time side effects: imports peft.LoraConfig only if PEFT is available.

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

import pytest
import torch
from datasets import DatasetDict, IterableDatasetDict, load_dataset
from transformers.utils import is_peft_available

from trl.experimental.tpo import TPOConfig, TPOTrainer
from trl.experimental.tpo.tpo_trainer import DataCollatorForTriplePreference

from ..testing_utils import TrlTestCase, require_peft


if is_peft_available():
    from peft import LoraConfig


def _add_reference_column(example):
    # [FUNCTION DOC]
    # HELPER: Dataset map function that sets example["reference"] to a copy of example["chosen"] (same object).
    # Inputs: example - dict row with a "chosen" key. Outputs: the same dict, mutated.
    # Exceptions: KeyError if "chosen" missing. Side effects: mutates the input row.
    """Synthesize a `reference` (gold) completion for tests by reusing the chosen completion."""
    example["reference"] = example["chosen"]
    return example


class TestDataCollatorForTriplePreference(TrlTestCase):
    # [CLASS DOC]
    # Unit tests for DataCollatorForTriplePreference on hand-built tokenized examples (pad_token_id 0).
    # Inherits TrlTestCase. No state.
    def test_padding_and_masks(self):
        # [FUNCTION DOC]
        # Checks the default collator stacks prompt+chosen rows for all examples, then prompt+rejected, then
        # prompt+reference (3 times batch size rows), right-padded with 0 to the longest row.
        # Assertions: output keys are exactly input_ids, attention_mask, completion_mask; input_ids match the expected
        # six rows; attention_mask is 1 on real tokens; completion_mask is 1 only on completion tokens.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        collator = DataCollatorForTriplePreference(pad_token_id=0)
        examples = [
            {"prompt_ids": [1, 2, 3], "chosen_ids": [4, 5], "rejected_ids": [6], "reference_ids": [7, 8]},
            {"prompt_ids": [9, 10], "chosen_ids": [11], "rejected_ids": [12, 13], "reference_ids": [14]},
        ]
        result = collator(examples)

        expected_input_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5],  # prompt + chosen (example 1)
                [9, 10, 11, 0, 0],  # prompt + chosen (example 2, padded)
                [1, 2, 3, 6, 0],  # prompt + rejected (example 1, padded)
                [9, 10, 12, 13, 0],  # prompt + rejected (example 2, padded)
                [1, 2, 3, 7, 8],  # prompt + reference (example 1)
                [9, 10, 14, 0, 0],  # prompt + reference (example 2, padded)
            ]
        )
        expected_attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0],
            ]
        )
        expected_completion_mask = torch.tensor(
            [
                [0, 0, 0, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 1, 0, 0],
            ]
        )

        assert set(result.keys()) == {"input_ids", "attention_mask", "completion_mask"}
        torch.testing.assert_close(result["input_ids"], expected_input_ids)
        torch.testing.assert_close(result["attention_mask"], expected_attention_mask)
        torch.testing.assert_close(result["completion_mask"], expected_completion_mask)

    def test_exclude_reference(self):
        # [FUNCTION DOC]
        # Checks include_reference False yields only the chosen and rejected rows (2 times batch size, shape (4, 5)), with
        # the same ordering and padding, and the same three output keys.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        # When `include_reference=False`, the collator only emits the chosen/rejected halves so the per-step
        # compute/memory cost matches DPO's `DataCollatorForPreference`. This is the layout used by
        # `TPOTrainer` when `tpo_alpha=0.0`.
        collator = DataCollatorForTriplePreference(pad_token_id=0, include_reference=False)
        examples = [
            {"prompt_ids": [1, 2, 3], "chosen_ids": [4, 5], "rejected_ids": [6], "reference_ids": [7, 8]},
            {"prompt_ids": [9, 10], "chosen_ids": [11], "rejected_ids": [12, 13], "reference_ids": [14]},
        ]
        result = collator(examples)

        expected_input_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5],  # prompt + chosen (example 1)
                [9, 10, 11, 0, 0],  # prompt + chosen (example 2, padded)
                [1, 2, 3, 6, 0],  # prompt + rejected (example 1, padded)
                [9, 10, 12, 13, 0],  # prompt + rejected (example 2, padded)
            ]
        )
        assert result["input_ids"].shape == (4, 5)  # 2 * B rows, no reference branch
        torch.testing.assert_close(result["input_ids"], expected_input_ids)
        assert set(result.keys()) == {"input_ids", "attention_mask", "completion_mask"}


class TestTPOTrainer(TrlTestCase):
    # [CLASS DOC]
    # Integration tests for TPOTrainer with the tiny Qwen2 model (and others where noted).
    # Inherits TrlTestCase (self.tmp_dir). Each test loads a zen preference dataset and adds the reference column.
    def test_train(self):
        # [FUNCTION DOC]
        # Trains TPOTrainer on the standard_preference train split (learning_rate 0.1, other defaults).
        # Assertions: final log_history entry has a non-None train_loss and every parameter snapshot taken before
        # training differs from the parameter after training.
        # Inputs: none (self.tmp_dir). Outputs: none. Exceptions: assertion failure.
        # Side effects: downloads/loads dataset and tiny model; trains and writes to self.tmp_dir.
        # Get the dataset and synthesize a reference (gold) completion
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        dataset = dataset.map(_add_reference_column)

        training_args = TPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = TPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize(
        "eval_dataset_type",
        [
            "dataset",
            "iterable_dataset",
            "dataset_dict",
            "iterable_dataset_dict",
            "dict_of_dataset",
            "dict_of_iterable_dataset",
            "none",
        ],
    )
    def test_init_with_eval_dataset(self, eval_dataset_type):
        # [FUNCTION DOC]
        # Parametrized over eval dataset kinds: Dataset, IterableDataset, DatasetDict, IterableDatasetDict, plain dict of
        # Dataset, plain dict of IterableDataset, or None. Builds the eval set from the zen test split (streaming when the
        # kind contains "iterable"), with reference column added; constructs the trainer.
        # Assertions: None stays None; dict-like eval sets keep keys data1/data2 and each split's first row has
        # prompt_ids (tokenized independently); single datasets have prompt_ids in their first row.
        # Inputs: eval_dataset_type - parametrized string. Outputs: none. Exceptions: assertion failure.
        # Side effects: loads datasets/model.
        # Get the dataset and synthesize a reference (gold) completion
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        train_dataset = train_dataset.map(_add_reference_column)

        if eval_dataset_type == "none":
            eval_dataset = None
        else:
            streaming = "iterable" in eval_dataset_type
            eval_split = load_dataset(
                "trl-internal-testing/zen", "standard_preference", split="test", streaming=streaming
            )
            eval_split = eval_split.map(_add_reference_column)
            if eval_dataset_type in ("dataset", "iterable_dataset"):
                eval_dataset = eval_split
            elif eval_dataset_type in ("dataset_dict", "iterable_dataset_dict"):
                dataset_dict_cls = IterableDatasetDict if streaming else DatasetDict
                eval_dataset = dataset_dict_cls({"data1": eval_split, "data2": eval_split})
            else:  # "dict_of_dataset" or "dict_of_iterable_dataset"
                eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = TPOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = TPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        if eval_dataset_type == "none":
            assert trainer.eval_dataset is None
        elif isinstance(trainer.eval_dataset, dict):
            assert set(trainer.eval_dataset.keys()) == {"data1", "data2"}
            # Each split was tokenized independently.
            assert "prompt_ids" in next(iter(trainer.eval_dataset["data1"]))
            assert "prompt_ids" in next(iter(trainer.eval_dataset["data2"]))
        else:
            assert "prompt_ids" in next(iter(trainer.eval_dataset))

    def test_evaluate_with_raw_dataset(self):
        # [FUNCTION DOC]
        # Checks evaluate accepts an unprocessed dataset (the raw train split with reference column) passed directly
        # and returns metrics with a non-None eval_loss (regression for issue 6115).
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: loads model/dataset, runs evaluation.
        # `evaluate` should accept the same (unprocessed) dataset types as the trainer, e.g. a held-out test set
        # passed directly to `evaluate`, mirroring DPO/KTO. See https://github.com/huggingface/trl/issues/6115.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        dataset = dataset.map(_add_reference_column)

        training_args = TPOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = TPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        metrics = trainer.evaluate(eval_dataset=dataset)
        assert metrics["eval_loss"] is not None

    def test_trust_remote_code(self):
        # [FUNCTION DOC]
        # Checks loading tiny-RemoteForCausalLM raises ValueError mentioning "custom code" without trust_remote_code,
        # and succeeds with trust_remote_code True, producing a model whose class name is RemoteForCausalLM.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: loads remote-code model.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        dataset = dataset.map(_add_reference_column)
        model_id = "trl-internal-testing/tiny-RemoteForCausalLM"

        with pytest.raises(ValueError, match="custom code"):
            TPOTrainer(
                model=model_id,
                args=TPOConfig(output_dir=self.tmp_dir, report_to="none"),
                train_dataset=dataset,
            )

        trainer = TPOTrainer(
            model=model_id,
            args=TPOConfig(output_dir=self.tmp_dir, report_to="none", trust_remote_code=True),
            train_dataset=dataset,
        )
        assert type(trainer.model).__name__ == "RemoteForCausalLM"

    @pytest.mark.parametrize("loss_type", ["sigmoid", "hinge", "ipo", "tpo-l"])
    def test_train_loss_types(self, loss_type):
        # [FUNCTION DOC]
        # Parametrized over loss_type sigmoid, hinge, ipo, tpo-l: trains on the train split with evaluation on the test
        # split every 3 steps (learning_rate 0.1).
        # Assertions: final log_history entry has a non-None train_loss and every parameter snapshot taken before
        # training differs from the parameter after training.
        # Inputs: none (self.tmp_dir). Outputs: none. Exceptions: assertion failure.
        # Side effects: downloads/loads dataset and tiny model; trains and writes to self.tmp_dir.
        # Get the dataset and synthesize a reference (gold) completion
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference")
        dataset = dataset.map(_add_reference_column)

        training_args = TPOConfig(
            output_dir=self.tmp_dir,
            loss_type=loss_type,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
            eval_strategy="steps",
            eval_steps=3,
        )
        trainer = TPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_conversational(self):
        # [FUNCTION DOC]
        # Trains on the conversational_preference train split (learning_rate 0.1).
        # Assertions: final log_history entry has a non-None train_loss and every parameter snapshot taken before
        # training differs from the parameter after training.
        # Inputs: none (self.tmp_dir). Outputs: none. Exceptions: assertion failure.
        # Side effects: downloads/loads dataset and tiny model; trains and writes to self.tmp_dir.
        # Get the dataset and synthesize a reference (gold) completion
        dataset = load_dataset("trl-internal-testing/zen", "conversational_preference", split="train")
        dataset = dataset.map(_add_reference_column)

        training_args = TPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            report_to="none",
        )
        trainer = TPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_without_nll(self):
        # [FUNCTION DOC]
        # Checks tpo_alpha 0.0: the default data collator is DataCollatorForTriplePreference with include_reference
        # False, collating two train rows gives 4 rows (chosen and rejected only), then trains (learning_rate 0.1).
        # Assertions: final log_history entry has a non-None train_loss and every parameter snapshot taken before
        # training differs from the parameter after training.
        # Inputs: none (self.tmp_dir). Outputs: none. Exceptions: assertion failure.
        # Side effects: downloads/loads dataset and tiny model; trains and writes to self.tmp_dir.
        # Setting tpo_alpha=0.0 disables the NLL term, skips the corresponding cross-entropy, and also drops the
        # reference branch from the collated batch so the model doesn't pay the extra forward-pass cost.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        dataset = dataset.map(_add_reference_column)

        training_args = TPOConfig(
            output_dir=self.tmp_dir,
            tpo_alpha=0.0,
            learning_rate=0.1,
            report_to="none",
        )
        trainer = TPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        # The default collator should drop the reference branch entirely when `tpo_alpha=0.0`.
        assert isinstance(trainer.data_collator, DataCollatorForTriplePreference)
        assert trainer.data_collator.include_reference is False

        # Verify the collated batch is 2 * per_device_train_batch_size (chosen + rejected only), not 3 * B.
        batch = trainer.data_collator(list(trainer.train_dataset.select(range(2))))
        assert batch["input_ids"].shape[0] == 4  # 2 branches * 2 examples

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_implicit_prompt(self):
        # [FUNCTION DOC]
        # Regression test for implicit-prompt data (no prompt column; prompt embedded in chosen, rejected and the
        # synthesized reference): trains on standard_implicit_prompt_preference (learning_rate 0.1) to make sure prompt
        # extraction also strips the reference.
        # Assertions: final log_history entry has a non-None train_loss and every parameter snapshot taken before
        # training differs from the parameter after training.
        # Inputs: none (self.tmp_dir). Outputs: none. Exceptions: assertion failure.
        # Side effects: downloads/loads dataset and tiny model; trains and writes to self.tmp_dir.
        # Implicit-prompt variant: no `prompt` column, the prompt is embedded in `chosen`/`rejected` and (for TPO)
        # also in `reference`. Regression test for the `extract_prompt` bug where the reference column was left
        # untouched, silently doubling the prompt in the reference branch.
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Synthesize a reference column that shares the same implicit prompt as chosen/rejected
        dataset = dataset.map(_add_reference_column)

        training_args = TPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            report_to="none",
        )
        trainer = TPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_implicit_prompt_mismatched_reference_raises(self):
        # [FUNCTION DOC]
        # Checks that with implicit-prompt data whose reference does not share the chosen/rejected prompt prefix,
        # constructing TPOTrainer raises ValueError mentioning "implicit prompt".
        # Setup: reference overwritten by a nested map function with an unrelated string.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: loads model/dataset.
        # When the dataset has no `prompt` column and the `reference` completion does not share the implicit
        # prompt prefix of `chosen`/`rejected`, the trainer must raise a clear error rather than silently
        # corrupting the reference branch.
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        def _set_unrelated_reference(example):
            # [FUNCTION DOC]
            # HELPER: Map function setting example["reference"] to a fixed unrelated sentence without the prompt prefix.
            # Inputs: example dict. Outputs: the mutated dict. Exceptions: none. Side effects: mutates the row.
            example["reference"] = "unrelated completion without the shared prompt prefix."
            return example

        dataset = dataset.map(_set_unrelated_reference)

        training_args = TPOConfig(output_dir=self.tmp_dir, report_to="none")
        with pytest.raises(ValueError, match="implicit prompt"):
            TPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=dataset,
            )

    def test_missing_reference_column_raises(self):
        # [FUNCTION DOC]
        # Checks that a preference dataset without a "reference" column makes TPOTrainer raise ValueError matching
        # "reference". Inputs/Outputs: none. Exceptions: assertion failure. Side effects: loads model/dataset.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = TPOConfig(output_dir=self.tmp_dir, report_to="none")
        with pytest.raises(ValueError, match="reference"):
            TPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=dataset,
            )

    @require_peft
    def test_peft_init_is_seeded(self):
        # [FUNCTION DOC]
        # Checks LoRA adapter initialization is deterministic given the config seed regardless of global RNG state.
        # Setup: for global seeds 0 and 1, set torch's global seed, build a trainer with a default LoraConfig and clone all
        # lora_A parameters. Requires PEFT.
        # Assertions: some lora_A params exist and each one is identical across the two trainers.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: alters the global torch RNG; loads models.
        # Two trainers with the same seed start from the same adapter weights
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        dataset = dataset.map(_add_reference_column)
        adapters = []
        for global_seed in range(2):
            torch.manual_seed(global_seed)  # a different global RNG state, as in two separate runs
            trainer = TPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=TPOConfig(output_dir=self.tmp_dir, report_to="none"),
                train_dataset=dataset,
                peft_config=LoraConfig(),
            )
            adapters.append({n: p.clone() for n, p in trainer.model.named_parameters() if "lora_A" in n})

        assert adapters[0]
        for n, param in adapters[0].items():
            assert torch.equal(param, adapters[1][n]), f"Parameter {n} differs between the two trainers."

    @require_peft
    def test_train_with_peft(self):
        # [FUNCTION DOC]
        # Trains with a default LoraConfig (learning_rate 0.1). Requires PEFT.
        # Assertions: train_loss logged; every parameter whose name contains "lora" changed after training.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: loads model/dataset, trains.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        dataset = dataset.map(_add_reference_column)

        training_args = TPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            report_to="none",
        )
        trainer = TPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(),
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        for n, param in previous_trainable_params.items():
            if "lora" in n:
                new_param = trainer.model.get_parameter(n)
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_pad_token_id_synced_with_model_config(self):
        # [FUNCTION DOC]
        # Checks that with tiny-MistralForCausalLM-0.2 (tokenizer without a pad token, so eos is used) the trainer sets a
        # non-None pad_token_id on the processing class and copies it into model.config and model.generation_config.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: loads model/dataset.
        # This model's tokenizer has no pad token, so the trainer falls back to the eos token. The model configs must
        # follow: otherwise `Trainer` realigns them at train time and reports it as a change the user did not make.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        dataset = dataset.map(_add_reference_column)

        training_args = TPOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = TPOTrainer(
            model="trl-internal-testing/tiny-MistralForCausalLM-0.2", args=training_args, train_dataset=dataset
        )

        pad_token_id = trainer.processing_class.pad_token_id
        assert pad_token_id is not None
        assert trainer.model.config.pad_token_id == pad_token_id
        assert trainer.model.generation_config.pad_token_id == pad_token_id
