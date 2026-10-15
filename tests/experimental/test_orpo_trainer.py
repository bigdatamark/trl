# [MODULE DOC]
# Tests for the experimental ORPOTrainer / ORPOConfig (trl.experimental.orpo), Odds Ratio Preference
# Optimization. Covers remote-code gating, training for decoder-only (tiny Qwen2) and
# encoder-decoder (tiny T5) models, response truncation, eval-dataset tokenization,
# processing_class auto-loading, seeded/LoRA PEFT training, and compute_metrics. Uses "zen"
# preference datasets. Import-time side effects: none.

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
from datasets import DatasetDict, load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from trl.experimental.orpo import ORPOConfig, ORPOTrainer

from ..testing_utils import TrlTestCase, require_peft


class TestORPOTrainer(TrlTestCase):
    # [CLASS DOC]
    # Integration test suite for ORPOTrainer; inherits TrlTestCase (self.tmp_dir). setup_method loads a
    # tiny Qwen2 model/tokenizer (pad token set to eos) and a tiny T5 model/tokenizer shared by tests.
    def setup_method(self):
        # [FUNCTION DOC]
        # HELPER: Per-test fixture. Sets self.model_id, self.model (float32 Qwen2), self.tokenizer with
        # pad_token set to eos_token, and self.t5_model / self.t5_tokenizer (float32 tiny T5).
        # Inputs/Outputs: none. Exceptions: load errors propagate. Side effects: network/cache access.
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype="float32")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # get t5 as seq2seq example:
        model_id = "trl-internal-testing/tiny-T5ForConditionalGeneration"
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype="float32")
        self.t5_tokenizer = AutoTokenizer.from_pretrained(model_id)

    def test_trust_remote_code(self):
        # [FUNCTION DOC]
        # Tests that a remote-code model id raises ValueError matching "custom code" when
        # trust_remote_code is unset, and loads as RemoteForCausalLM when trust_remote_code=True.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure. Side effects: downloads models/datasets; writes to tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        model_id = "trl-internal-testing/tiny-RemoteForCausalLM"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        with pytest.raises(ValueError, match="custom code"):
            ORPOTrainer(
                model=model_id,
                args=ORPOConfig(output_dir=self.tmp_dir, report_to="none"),
                processing_class=tokenizer,
                train_dataset=dataset,
            )

        trainer = ORPOTrainer(
            model=model_id,
            args=ORPOConfig(output_dir=self.tmp_dir, report_to="none", trust_remote_code=True),
            processing_class=tokenizer,
            train_dataset=dataset,
        )
        assert type(trainer.model).__name__ == "RemoteForCausalLM"

    @pytest.mark.parametrize(
        "name, config_name",
        [
            ("qwen", "standard_preference"),
            ("t5", "standard_implicit_prompt_preference"),
            ("qwen", "conversational_preference"),
        ],
    )
    def test_orpo_trainer(self, name, config_name):
        # [FUNCTION DOC]
        # Parametrized over (model name qwen/t5, dataset config). Trains 3 steps (batch 2, lr 0.9, beta 0.1,
        # step eval); for t5 sets is_encoder_decoder. Asserts a train_loss is logged and every parameter with
        # nonzero sum changed.
        # Inputs: name, config_name (str). Outputs: none. Exceptions: AssertionError on failure. Side effects: downloads models/datasets; writes to tmp_dir.
        training_args = ORPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            remove_unused_columns=False,
            gradient_accumulation_steps=1,
            learning_rate=9e-1,
            eval_strategy="steps",
            beta=0.1,
            report_to="none",
        )

        dataset = load_dataset("trl-internal-testing/zen", config_name)

        if name == "qwen":
            model = self.model
            tokenizer = self.tokenizer
        elif name == "t5":
            model = self.t5_model
            tokenizer = self.t5_tokenizer
            training_args.is_encoder_decoder = True

        trainer = ORPOTrainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:  # ignore 0 biases
                assert not torch.equal(param, new_param)

    def test_orpo_trainer_truncation(self):
        # [FUNCTION DOC]
        # Tests that with max_length=8 each tokenized chosen/rejected sequence equals the untruncated one cut
        # to max(8, prompt length): prompts are never truncated; completions fill the remaining budget.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure. Side effects: downloads models/datasets; writes to tmp_dir.
        # Each response is truncated to `max_length` minus the prompt length. The prompt is never truncated, so a prompt
        # longer than `max_length` is kept whole with an empty completion.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        training_args = ORPOConfig(output_dir=self.tmp_dir, report_to="none")
        full = ORPOTrainer(
            model=self.model, args=training_args, processing_class=self.tokenizer, train_dataset=dataset
        ).train_dataset
        training_args = ORPOConfig(output_dir=self.tmp_dir, max_length=8, report_to="none")
        truncated = ORPOTrainer(
            model=self.model, args=training_args, processing_class=self.tokenizer, train_dataset=dataset
        ).train_dataset

        for full_row, truncated_row in zip(full, truncated, strict=True):
            length = max(8, len(full_row["prompt_input_ids"]))
            assert truncated_row["chosen_input_ids"] == full_row["chosen_input_ids"][:length]
            assert truncated_row["rejected_input_ids"] == full_row["rejected_input_ids"][:length]

    @pytest.mark.parametrize(
        "eval_dataset_type",
        [
            "dataset",
            "dataset_dict",
            "none",
        ],
    )
    def test_init_with_eval_dataset(self, eval_dataset_type):
        # [FUNCTION DOC]
        # Parametrized over eval_dataset_type dataset, dataset_dict, none. Tests None stays None; a
        # DatasetDict becomes a dict with keys data1/data2 whose splits each contain chosen_input_ids; a
        # single Dataset is tokenized (has chosen_input_ids).
        # Inputs: eval_dataset_type (str). Outputs: none. Exceptions: AssertionError on failure. Side effects: downloads models/datasets; writes to tmp_dir.
        # ORPO tokenizes the eval dataset at init by calling `.map()` directly on it, so streaming (iterable) and
        # plain dict-of-datasets eval datasets are not yet supported; only `Dataset` and `DatasetDict` are tested.
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        if eval_dataset_type == "none":
            eval_dataset = None
        else:
            eval_split = load_dataset("trl-internal-testing/zen", "standard_preference", split="test")
            if eval_dataset_type == "dataset":
                eval_dataset = eval_split
            else:  # "dataset_dict"
                eval_dataset = DatasetDict({"data1": eval_split, "data2": eval_split})

        training_args = ORPOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = ORPOTrainer(
            model=self.model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        if eval_dataset_type == "none":
            assert trainer.eval_dataset is None
        elif isinstance(trainer.eval_dataset, dict):
            assert set(trainer.eval_dataset.keys()) == {"data1", "data2"}
            # Each split was tokenized independently.
            assert "chosen_input_ids" in next(iter(trainer.eval_dataset["data1"]))
            assert "chosen_input_ids" in next(iter(trainer.eval_dataset["data2"]))
        else:
            assert "chosen_input_ids" in next(iter(trainer.eval_dataset))

    def test_orpo_trainer_processing_class_autoloaded(self):
        # [FUNCTION DOC]
        # Tests that omitting processing_class auto-loads one from the model, and that 3 training steps then
        # log a train_loss.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure. Side effects: downloads models/datasets; writes to tmp_dir.
        # processing_class is documented as optional: when omitted it should be
        # auto-loaded from the model, consistent with DPOTrainer / RewardTrainer.
        training_args = ORPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            remove_unused_columns=False,
            beta=0.1,
            report_to="none",
        )
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference")
        trainer = ORPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
        )
        assert trainer.processing_class is not None
        trainer.train()
        assert trainer.state.log_history[-1]["train_loss"] is not None

    @require_peft
    def test_peft_init_is_seeded(self):
        # [FUNCTION DOC]
        # Requires peft. Builds two trainers with a default LoraConfig after setting different global torch
        # seeds; asserts lora_A parameters exist and are identical across the two, i.e. adapter init is
        # seeded from the config seed rather than the global RNG state.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure. Side effects: downloads models/datasets; writes to tmp_dir.
        from peft import LoraConfig

        # Two trainers with the same seed start from the same adapter weights
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        adapters = []
        for global_seed in range(2):
            torch.manual_seed(global_seed)  # a different global RNG state, as in two separate runs
            trainer = ORPOTrainer(
                model=self.model_id,
                args=ORPOConfig(output_dir=self.tmp_dir, report_to="none"),
                processing_class=self.tokenizer,
                train_dataset=dataset,
                peft_config=LoraConfig(),
            )
            adapters.append({n: p.clone() for n, p in trainer.model.named_parameters() if "lora_A" in n})

        assert adapters[0]
        for n, param in adapters[0].items():
            assert torch.equal(param, adapters[1][n]), f"Parameter {n} differs between the two trainers."

    @pytest.mark.parametrize(
        "config_name",
        [
            "standard_preference",
            "standard_implicit_prompt_preference",
            "conversational_preference",
            "conversational_implicit_prompt_preference",
        ],
    )
    @require_peft
    def test_orpo_trainer_with_lora(self, config_name):
        # [FUNCTION DOC]
        # Requires peft. Parametrized over four preference dataset configs. Trains 3 steps with LoRA (r 16,
        # alpha 32, dropout 0.05, grad accumulation 4, beta 0.1); asserts train_loss logged and every LoRA
        # parameter with nonzero sum changed.
        # Inputs: config_name (str). Outputs: none. Exceptions: AssertionError on failure. Side effects: downloads models/datasets; writes to tmp_dir.
        from peft import LoraConfig

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        training_args = ORPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            remove_unused_columns=False,
            gradient_accumulation_steps=4,
            learning_rate=9e-1,
            eval_strategy="steps",
            beta=0.1,
            report_to="none",
        )

        dataset = load_dataset("trl-internal-testing/zen", config_name)

        trainer = ORPOTrainer(
            model=self.model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            peft_config=lora_config,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            if "lora" in n:
                new_param = trainer.model.get_parameter(n)
                if param.sum() != 0:  # ignore 0 biases
                    assert not torch.equal(param, new_param)

    def test_compute_metrics(self):
        # [FUNCTION DOC]
        # Loads its own tiny Qwen2 model/tokenizer (pad = eos). Tests that a compute_metrics callback
        # returning test 0.0 is used during step-wise eval (eval_steps 1): the second-to-last log entry has
        # eval_test equal to 0.0.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure. Side effects: downloads models/datasets; writes to tmp_dir.
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", dtype="float32")
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        tokenizer.pad_token = tokenizer.eos_token

        dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

        def dummy_compute_metrics(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Ignores its arguments and returns a dict with key test mapped to 0.0.
            # Inputs: any. Outputs: dict. Exceptions: none. Side effects: none.
            return {"test": 0.0}

        training_args = ORPOConfig(
            output_dir=self.tmp_dir,
            remove_unused_columns=False,
            per_device_train_batch_size=2,
            do_eval=True,
            eval_strategy="steps",
            eval_steps=1,
            per_device_eval_batch_size=2,
            report_to="none",
        )

        trainer = ORPOTrainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=dummy_compute_metrics,
        )

        trainer.train()

        assert trainer.state.log_history[-2]["eval_test"] == 0.0
