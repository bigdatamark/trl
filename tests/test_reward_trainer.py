# [MODULE DOC]
# Tests for trl RewardTrainer, RewardConfig and DataCollatorForPreference (trl.trainer.reward_trainer).
# Main contents: collator unit tests and end-to-end trainer tests on tiny models from trl-internal-testing with
# the zen and toolcall preference datasets. PEFT imports (LoraConfig, get_peft_model) happen only if peft is
# installed. Optional-dependency tests are gated by require_* decorators.
# Import-time side effects: conditional peft import.

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
import pathlib

import pytest
import torch
from datasets import DatasetDict, IterableDatasetDict, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_peft_available

from trl import RewardConfig, RewardTrainer
from trl.trainer.reward_trainer import DataCollatorForPreference

from .testing_utils import (
    TrlTestCase,
    require_bitsandbytes,
    require_peft,
    require_peft_target_parameters,
    require_torch_accelerator,
)


if is_peft_available():
    from peft import LoraConfig, get_peft_model


class TestDataCollatorForPreference(TrlTestCase):
    # [CLASS DOC]
    # Tests DataCollatorForPreference: it stacks all chosen sequences then all rejected sequences into one batch,
    # right-pads with pad_token_id (optionally to a multiple of pad_to_multiple_of), builds attention_mask, and
    # returns a margin tensor when examples have a margin key.
    def test_basic_padding(self):
        # [FUNCTION DOC]
        # Asserts two examples give input_ids rows chosen1, chosen2, rejected1, rejected2 padded with 0 to length 3,
        # and the matching 1/0 attention_mask. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Test basic padding functionality without completion masks."""
        collator = DataCollatorForPreference(pad_token_id=0)
        examples = [
            {"chosen_ids": [1, 2, 3], "rejected_ids": [4, 5]},
            {"chosen_ids": [6, 7], "rejected_ids": [8]},
        ]

        result = collator(examples)

        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [6, 7, 0], [4, 5, 0], [8, 0, 0]]))
        torch.testing.assert_close(
            result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 0]])
        )

    def test_pad_to_multiple_of(self):
        # [FUNCTION DOC]
        # Asserts that with pad_to_multiple_of 4 the same batch is padded to length 4.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Test padding to multiple of specified value."""
        collator = DataCollatorForPreference(pad_token_id=0, pad_to_multiple_of=4)
        examples = [
            {"chosen_ids": [1, 2, 3], "rejected_ids": [4, 5]},
            {"chosen_ids": [6, 7], "rejected_ids": [8]},
        ]

        result = collator(examples)

        torch.testing.assert_close(
            result["input_ids"], torch.tensor([[1, 2, 3, 0], [6, 7, 0, 0], [4, 5, 0, 0], [8, 0, 0, 0]])
        )
        torch.testing.assert_close(
            result["attention_mask"], torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]])
        )

    def test_single_example(self):
        # [FUNCTION DOC]
        # Asserts a single example yields two rows (chosen, rejected) padded to length 3 with matching mask.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Test collator with a single example."""
        collator = DataCollatorForPreference(pad_token_id=0)
        examples = [{"chosen_ids": [1, 2, 3], "rejected_ids": [4, 5]}]

        result = collator(examples)

        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))

    def test_different_pad_token_id(self):
        # [FUNCTION DOC]
        # Asserts pad_token_id 999 is used for padding while the attention mask is unchanged.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Test with different pad token ID."""
        collator = DataCollatorForPreference(pad_token_id=999)
        examples = [
            {"chosen_ids": [1, 2, 3], "rejected_ids": [4, 5]},
            {"chosen_ids": [6, 7], "rejected_ids": [8]},
        ]

        result = collator(examples)

        torch.testing.assert_close(
            result["input_ids"], torch.tensor([[1, 2, 3], [6, 7, 999], [4, 5, 999], [8, 999, 999]])
        )
        torch.testing.assert_close(
            result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 0]])
        )

    def test_collate_with_margin(self):
        # [FUNCTION DOC]
        # Asserts padding as in the basic test plus a margin tensor of 0.1 and 0.2 in example order.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        collator = DataCollatorForPreference(pad_token_id=0)
        examples = [
            {"chosen_ids": [1, 2, 3], "rejected_ids": [4, 5], "margin": 0.1},
            {"chosen_ids": [6, 7], "rejected_ids": [8], "margin": 0.2},
        ]

        result = collator(examples)

        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [6, 7, 0], [4, 5, 0], [8, 0, 0]]))
        torch.testing.assert_close(
            result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 0]])
        )
        torch.testing.assert_close(result["margin"], torch.tensor([0.1, 0.2]))


class TestRewardTrainer(TrlTestCase):
    # [CLASS DOC]
    # End-to-end RewardTrainer tests. Each uses self.tmp_dir as output_dir and report_to none.
    # Common "standard train check": snapshot all named parameters, call train, assert the last log entry has a
    # train_loss, and assert every parameter differs from its snapshot.
    # Common "PEFT train check": like the standard check, but parameters whose names (prefixed base_model.model.)
    # belong to the base model must be unchanged and other non-base_layer parameters must change.
    # Inherits TrlTestCase. Tests load models and datasets from the Hub (cached).
    def test_raises_error_when_model_num_labels_not_one(self):
        # [FUNCTION DOC]
        # Asserts passing a sequence-classification model loaded with the default 2 labels raises ValueError mentioning
        # that reward models require num_labels 1. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Test that RewardTrainer raises ValueError when model doesn't have num_labels=1."""
        model = AutoModelForSequenceClassification.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            dtype="float32",
            # num_labels=2,  # Defaults to 2 num_labels for causal models
        )

        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        training_args = RewardConfig(output_dir=self.tmp_dir, report_to="none")
        with pytest.raises(ValueError, match=r"reward models require `num_labels=1`"):
            RewardTrainer(model=model, args=training_args, train_dataset=dataset)

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            "trl-internal-testing/tiny-Qwen3MoeForCausalLM",
            "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
        ],
    )
    def test_train(self, model_id):
        # [FUNCTION DOC]
        # Runs the standard train check for a model id string (Qwen2, Qwen3 MoE, Llama tiny models).
        # Inputs: model_id (str, parametrized). Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        training_args = RewardConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = RewardTrainer(model=model_id, args=training_args, train_dataset=dataset)

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
        ],
    )
    def test_evaluate_with_eval_dataset(self, eval_dataset_type):
        # [FUNCTION DOC]
        # Asserts evaluate accepts a raw, unprepared eval dataset passed directly (regression for issue 6115) in six
        # forms: Dataset, IterableDataset, DatasetDict, IterableDatasetDict, dict of Dataset, dict of IterableDataset.
        # Single datasets must yield eval_loss; dict forms must yield eval_data1_loss and eval_data2_loss.
        # Inputs: eval_dataset_type (str, parametrized). Outputs: none. Exceptions: AssertionError. Side effects: none.
        # `evaluate` accepts a raw (unprepared) dataset passed directly, not only a preprocessed `eval_dataset` set
        # at init. See https://github.com/huggingface/trl/issues/6115.
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")
        streaming = "iterable" in eval_dataset_type
        eval_split = load_dataset(
            "trl-internal-testing/zen", "standard_implicit_prompt_preference", split="test", streaming=streaming
        )
        if eval_dataset_type in ("dataset", "iterable_dataset"):
            eval_dataset = eval_split
        elif eval_dataset_type in ("dataset_dict", "iterable_dataset_dict"):
            dataset_dict_cls = IterableDatasetDict if streaming else DatasetDict
            eval_dataset = dataset_dict_cls({"data1": eval_split, "data2": eval_split})
        else:  # "dict_of_dataset" or "dict_of_iterable_dataset"
            eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = RewardConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=train_dataset
        )

        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        if eval_dataset_type in ("dataset", "iterable_dataset"):
            assert metrics["eval_loss"] is not None
        else:
            assert metrics["eval_data1_loss"] is not None
            assert metrics["eval_data2_loss"] is not None

    def test_trust_remote_code(self):
        # [FUNCTION DOC]
        # Asserts loading a remote-code model id without trust_remote_code raises ValueError mentioning custom code, and
        # with trust_remote_code True the model class is RemoteForSequenceClassification.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: may download remote code.
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")
        model_id = "trl-internal-testing/tiny-RemoteForCausalLM"

        with pytest.raises(ValueError, match="custom code"):
            RewardTrainer(
                model=model_id,
                args=RewardConfig(output_dir=self.tmp_dir, report_to="none"),
                train_dataset=dataset,
            )

        trainer = RewardTrainer(
            model=model_id,
            args=RewardConfig(output_dir=self.tmp_dir, report_to="none", trust_remote_code=True),
            train_dataset=dataset,
        )
        assert type(trainer.model).__name__ == "RemoteForSequenceClassification"

    @pytest.mark.parametrize(
        "config_name",
        [
            "standard_preference",
            "conversational_preference",
            "standard_implicit_prompt_preference",
            "conversational_implicit_prompt_preference",
        ],
    )
    def test_train_dataset_types(self, config_name):
        # [FUNCTION DOC]
        # Runs the standard train check on each zen preference config (standard/conversational, explicit/implicit
        # prompt). Inputs: config_name (str, parametrized). Outputs: none. Exceptions: AssertionError.
        # Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", config_name, split="train")

        training_args = RewardConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = RewardTrainer(
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

    def test_train_model(self):
        # [FUNCTION DOC]
        # Runs the standard train check with an already instantiated sequence-classification model (num_labels 1,
        # float32). Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        model = AutoModelForSequenceClassification.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            num_labels=1,  # required for reward models
            dtype="float32",
        )

        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        training_args = RewardConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = RewardTrainer(model=model, args=training_args, train_dataset=dataset)

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_from_sequence_classification_model(self):
        # [FUNCTION DOC]
        # Runs the standard train check with a model id that is already a sequence-classification checkpoint.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        training_args = RewardConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
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

    def test_train_model_dtype(self):
        # [FUNCTION DOC]
        # Trains with model_init_kwargs dtype float16 and learning rate 0.1, then asserts every non-layernorm parameter
        # is float16 and changed (layernorm skipped because it is flaky on CI).
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        training_args = RewardConfig(
            output_dir=self.tmp_dir,
            model_init_kwargs={"dtype": torch.float16},
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            # For some reasonn model.layers.0.input_layernorm.weight doesn't change in GitHub Actions but does
            # locally. We ignore this parameter for now
            if "layernorm" in n:
                continue
            new_param = trainer.model.get_parameter(n)
            # Check the torch dtype
            assert new_param.dtype == torch.float16
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_peft
    def test_train_dense_with_peft_config(self):
        # [FUNCTION DOC]
        # Runs the PEFT train check with a model id and a default LoraConfig (requires peft); base parameter names come
        # from a separately loaded copy of the model. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: writes tmp_dir.
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForSequenceClassification.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        training_args = RewardConfig(output_dir=self.tmp_dir, report_to="none")

        trainer = RewardTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(),
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "base_layer" not in n:  # We expect the peft params to be different (except for the base layer)
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_peft
    @require_bitsandbytes
    def test_train_peft_and_quantization(self):
        # [FUNCTION DOC]
        # With peft and bitsandbytes, passes a 4-bit NF4 double-quant BitsAndBytesConfig (bf16 compute) and LoRA; asserts
        # the base model is loaded in 4 bit, trains with lr 0.1, and asserts LoRA parameters are bfloat16 and changed
        # (base params unchecked since bnb mutates biases). Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        training_args = RewardConfig(output_dir=self.tmp_dir, learning_rate=0.1, report_to="none")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",  # identifier, so that the trainer quantizes it
            args=training_args,
            train_dataset=dataset,
            quantization_config=quantization_config,
            peft_config=LoraConfig(),
        )

        # Check that the trainer applied the quantization config when loading the model
        assert trainer.model.base_model.model.is_loaded_in_4bit

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed, and that they are cast to bfloat16, as recommended by the QLoRA
        # paper. The base model params are not checked: bitsandbytes casts the biases of a Linear4bit in-place during
        # the forward pass, so some of them change in a way that is unrelated to training.
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if "lora" in n:  # We expect the peft params to be different
                assert param.dtype == torch.bfloat16, f"Parameter {n} is not in bfloat16."
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_peft
    @require_bitsandbytes
    def test_train_peft_dora_and_quantization(self):
        # [FUNCTION DOC]
        # Same as the 4-bit LoRA test but with DoRA; additionally asserts lora_magnitude_vector parameters stay float32
        # and change, other LoRA parameters are bfloat16 and change. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        training_args = RewardConfig(output_dir=self.tmp_dir, learning_rate=0.1, report_to="none")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",  # identifier, so that the trainer quantizes it
            args=training_args,
            train_dataset=dataset,
            quantization_config=quantization_config,
            peft_config=LoraConfig(use_dora=True),
        )

        # Check that the trainer applied the quantization config when loading the model
        assert trainer.model.base_model.model.is_loaded_in_4bit

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed, and that they are cast to bfloat16, as recommended by the QLoRA
        # paper. The base model params are not checked: bitsandbytes casts the biases of a Linear4bit in-place during
        # the forward pass, so some of them change in a way that is unrelated to training.
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            # The DoRA magnitude vector is the exception: it must stay in float32, since its optimizer updates can be
            # smaller than bfloat16 can represent, which would otherwise silently freeze it, see #7268
            if "lora_magnitude_vector" in n:
                assert param.dtype == torch.float32, f"Parameter {n} is not in float32."
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."
            elif "lora" in n:  # We expect the peft params to be different
                assert param.dtype == torch.bfloat16, f"Parameter {n} is not in bfloat16."
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_peft_target_parameters
    def test_train_moe_with_peft_config(self):
        # [FUNCTION DOC]
        # Runs the PEFT train check on tiny GPT-OSS MoE with LoRA targeting the score module and the expert parameters
        # down_proj and gate_up_proj (requires peft target_parameters support).
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        model_id = "trl-internal-testing/tiny-GptOssForCausalLM"
        model = AutoModelForSequenceClassification.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        training_args = RewardConfig(output_dir=self.tmp_dir, report_to="none")

        trainer = RewardTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(
                target_modules=["score"], target_parameters=["mlp.experts.down_proj", "mlp.experts.gate_up_proj"]
            ),
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "base_layer" not in n:  # We expect the peft params to be different (except for the base layer)
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_peft
    def test_peft_init_is_seeded(self):
        # [FUNCTION DOC]
        # Builds two trainers with LoRA after setting different global torch seeds and asserts their lora_A weights are
        # non-empty and identical, i.e. adapter init uses the config seed. Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: changes the global torch seed.
        # Two trainers with the same seed start from the same adapter weights
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        adapters = []
        for global_seed in range(2):
            torch.manual_seed(global_seed)  # a different global RNG state, as in two separate runs
            trainer = RewardTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=RewardConfig(output_dir=self.tmp_dir, report_to="none"),
                train_dataset=dataset,
                peft_config=LoraConfig(),
            )
            adapters.append({n: p.clone() for n, p in trainer.model.named_parameters() if "lora_A" in n})

        assert adapters[0]
        for n, param in adapters[0].items():
            assert torch.equal(param, adapters[1][n]), f"Parameter {n} differs between the two trainers."

    @require_peft
    def test_train_peft_model(self):
        # [FUNCTION DOC]
        # Runs the PEFT train check with a model already wrapped by get_peft_model (num_labels 1, float32).
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=1,  # required for reward models
            dtype="float32",
        )

        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        lora_config = LoraConfig()
        model = get_peft_model(model, lora_config)

        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        training_args = RewardConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = RewardTrainer(model=model, args=training_args, train_dataset=dataset)

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "base_layer" not in n:  # We expect the peft params to be different (except for the base layer)
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    # In practice, this test is the same as `test_train_dense_with_peft_config`, since gradient checkpointing is
    # enabled by default in `RewardTrainer`. We keep it as a regression guard: if the default ever changes, we still
    # explicitly test PEFT + gradient checkpointing, which has caused issues in the past.
    @require_peft
    def test_train_with_peft_config_and_gradient_checkpointing(self):
        # [FUNCTION DOC]
        # Runs the PEFT train check with gradient_checkpointing explicitly True (regression guard; it is the default).
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForSequenceClassification.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        training_args = RewardConfig(output_dir=self.tmp_dir, gradient_checkpointing=True, report_to="none")

        trainer = RewardTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(),
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "base_layer" not in n:  # We expect the peft params to be different (except for the base layer)
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize("use_reentrant", [True, False])
    @require_peft
    def test_train_with_peft_config_and_gradient_checkpointing_reentrant(self, use_reentrant):
        # [FUNCTION DOC]
        # Runs the PEFT train check on the sequence-classification checkpoint with gradient checkpointing and the given
        # use_reentrant kwarg. Inputs: use_reentrant (bool, parametrized). Outputs: none. Exceptions: AssertionError.
        # Side effects: writes tmp_dir.
        model_id = "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"
        model = AutoModelForSequenceClassification.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        training_args = RewardConfig(
            output_dir=self.tmp_dir,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": use_reentrant},
            report_to="none",
        )

        trainer = RewardTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(),
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "base_layer" not in n:  # We expect the peft params to be different (except for the base layer)
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize(
        "chosen_column,rejected_column,expect_deprecation_warning",
        [
            ("chosen_ids", "rejected_ids", False),
            ("chosen_input_ids", "rejected_input_ids", True),
        ],
    )
    def test_train_with_pretokenized_data(self, chosen_column, rejected_column, expect_deprecation_warning):
        # [FUNCTION DOC]
        # Tokenizes chosen and rejected into the given column names and drops the text columns. For the legacy names
        # chosen_input_ids/rejected_input_ids asserts a FutureWarning about v1 is raised. Asserts the trainer dataset uses
        # chosen_ids/rejected_ids and not the legacy names, then runs the standard train check.
        # Inputs: chosen_column, rejected_column (str), expect_deprecation_warning (bool), parametrized.
        # Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        def tokenize_example(example):
            # [FUNCTION DOC]
            # HELPER: Maps an example to the chosen and rejected token id lists under the parametrized column names.
            # Inputs: example (dict). Outputs: dict of two id lists. Exceptions: none. Side effects: none.
            return {
                chosen_column: tokenizer(example["chosen"]).input_ids,
                rejected_column: tokenizer(example["rejected"]).input_ids,
            }

        # Apply tokenization
        tokenized_dataset = dataset.map(tokenize_example, remove_columns=["chosen", "rejected"])

        training_args = RewardConfig(output_dir=self.tmp_dir, report_to="none")
        if expect_deprecation_warning:
            with pytest.warns(FutureWarning, match=r"will not be supported in v1"):
                trainer = RewardTrainer(model=model_id, args=training_args, train_dataset=tokenized_dataset)
        else:
            trainer = RewardTrainer(model=model_id, args=training_args, train_dataset=tokenized_dataset)

        assert "chosen_ids" in trainer.train_dataset.column_names
        assert "rejected_ids" in trainer.train_dataset.column_names
        assert "chosen_input_ids" not in trainer.train_dataset.column_names
        assert "rejected_input_ids" not in trainer.train_dataset.column_names

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_with_iterable_dataset(self):
        # [FUNCTION DOC]
        # Runs the standard train check on a streaming dataset with max_steps 3.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset(
            "trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train", streaming=True
        )

        training_args = RewardConfig(output_dir=self.tmp_dir, max_steps=3, report_to="none")
        trainer = RewardTrainer(
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

    def test_train_with_chat_template_kwargs(self):
        # [FUNCTION DOC]
        # Sets a Qwen-like chat template that capitalizes role names when role_capital is true and adds a per-example
        # chat_template_kwargs column alternating role_capital false/true. Asserts the trainer keeps the template and
        # that the first two tokenized chosen and rejected sequences start with the lowercase then uppercase system
        # prompt ids; then runs the standard train check. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_implicit_prompt_preference", split="train")

        training_args = RewardConfig(output_dir=self.tmp_dir, report_to="none")

        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        # The following template is a simplified version of the Qwen chat template, where an additional argument
        # `role_capital` is used to control the capitalization of roles.
        tokenizer.chat_template = '{%- if messages[0]["role"] == "system" -%}    {{ "<|im_start|>" + ("SYSTEM" if role_capital else "system") + "\\n" + messages[0]["content"] + "<|im_end|>\\n" }}{%- else -%}    {{ "<|im_start|>" + ("SYSTEM" if role_capital else "system") + "\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n" }}{%- endif -%}{%- for message in messages -%}    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) -%}        {{ "<|im_start|>" + (message.role.upper() if role_capital else message.role) + "\\n" + message.content + "<|im_end|>\\n" }}    {%- elif message.role == "assistant" -%}        {{ "<|im_start|>" + ("ASSISTANT" if role_capital else "assistant") }}        {%- if message.content -%}            {{ "\\n" + message.content }}        {%- endif -%}        {{ "<|im_end|>\\n" }}    {%- elif message.role == "tool" -%}        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") -%}            {{ "<|im_start|>" + ("USER" if role_capital else "user") }}        {%- endif -%}        {{ "\\n<tool_response>\\n" + message.content + "\\n</tool_response>" }}        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") -%}            {{ "<|im_end|>\\n" }}        {%- endif -%}    {%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}    {{ "<|im_start|>" + ("ASSISTANT" if role_capital else "assistant") + "\\n" }}{%- endif -%}'

        dataset = dataset.add_column(
            "chat_template_kwargs", [{"role_capital": bool(i % 2)} for i in range(len(dataset))]
        )
        assert "chat_template_kwargs" in dataset.features

        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        assert trainer.processing_class.chat_template == tokenizer.chat_template

        for i in range(2):
            role = "SYSTEM" if i else "system"
            system_prompt = (
                f"<|im_start|>{role}\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>"
            )
            system_prompt_ids = trainer.processing_class(system_prompt)["input_ids"]
            assert trainer.train_dataset[i]["chosen_ids"][: len(system_prompt_ids)] == system_prompt_ids
            assert trainer.train_dataset[i]["rejected_ids"][: len(system_prompt_ids)] == system_prompt_ids

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_with_set_chat_template_from_model(self):
        # [FUNCTION DOC]
        # Uses chat_template_path naming a Hub model (Qwen3-4B) to give tiny GPT-NeoX a template and runs the standard
        # train check, skipping gpt_neox.final_layer_norm.bias (zero gradient under the mean-free loss).
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_preference", split="train")

        training_args = RewardConfig(output_dir=self.tmp_dir, chat_template_path="Qwen/Qwen3-4B", report_to="none")
        # trl-internal-testing/tiny-GPTNeoXForCausalLM doesn't have a chat template set by default
        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-GPTNeoXForCausalLM",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            # RewardTrainer uses a mean-free loss that cancels uniform shifts in output scores. Since GPT-NeoX models
            # include a final LayerNorm, its bias consistently receives zero gradient and remains unchanged, so we skip
            # this parameter.
            if n == "gpt_neox.final_layer_norm.bias":
                continue
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_with_set_chat_template_from_path(self, lazy_shared_datadir):
        # [FUNCTION DOC]
        # Like the previous test but with a local template.jinja from lazy_shared_datadir; additionally asserts the
        # checkpoint-9 directory contains chat_template.jinja identical to the source file.
        # Inputs: lazy_shared_datadir (fixture path). Outputs: none. Exceptions: AssertionError.
        # Side effects: writes checkpoints to tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_preference", split="train")

        training_args = RewardConfig(
            output_dir=self.tmp_dir,
            chat_template_path=str(lazy_shared_datadir / "template.jinja"),
            report_to="none",
        )
        # trl-internal-testing/tiny-GPTNeoXForCausalLM doesn't have a chat template set by default
        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-GPTNeoXForCausalLM",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            # RewardTrainer uses a mean-free loss that cancels uniform shifts in output scores. Since GPT-NeoX models
            # include a final LayerNorm, its bias consistently receives zero gradient and remains unchanged, so we skip
            # this parameter.
            if n == "gpt_neox.final_layer_norm.bias":
                continue
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

        # Check that the template saved in the output directory is the same as the one used for training
        template_path = pathlib.Path(self.tmp_dir) / "checkpoint-9" / "chat_template.jinja"
        assert template_path.exists(), f"Chat template not found at {template_path}"

        with open(template_path) as f:
            template_content = f.read()
        with open(training_args.chat_template_path) as f:
            original_template_content = f.read()
        assert template_content == original_template_content, "Chat template content does not match the original"

    def test_train_toolcall_data(self):
        # [FUNCTION DOC]
        # Runs the standard train check on the toolcall preference dataset with batch size 2 and max_length 512.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/toolcall", "preference", split="train")

        training_args = RewardConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,  # toolcall sequences are longer than standard data, reduce batch size to avoid OOM
            max_length=512,  # toolcall sequences are longer than standard data, limit length to avoid OOM
            report_to="none",
        )
        trainer = RewardTrainer(
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

    def test_train_toolcall_data_as_json(self):
        # [FUNCTION DOC]
        # Same as the toolcall test but first decodes the tools column from JSON strings into lists of dicts.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        # Tabular backends (Arrow/Parquet) can insert `None` for missing keys in nested structures.
        # If `tools` is stored as a list of dicts and examples use different dict schemas, nulls may
        # be introduced and break tool processing. This test ensures we also support `tools` provided
        # as a list of dicts.
        dataset = load_dataset("trl-internal-testing/toolcall", "preference", split="train")

        def convert_to_json(example):
            # [FUNCTION DOC]
            # HELPER: Returns a dict replacing tools with its JSON-decoded value. Inputs: example (dict).
            # Outputs: dict. Exceptions: json decode errors. Side effects: none.
            return {"tools": json.loads(example["tools"])}

        dataset = dataset.map(convert_to_json)

        training_args = RewardConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,  # toolcall sequences are longer than standard data, reduce batch size to avoid OOM
            max_length=512,  # toolcall sequences are longer than standard data, limit length to avoid OOM
            report_to="none",
        )
        trainer = RewardTrainer(
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

    def test_train_with_eval(self):
        # [FUNCTION DOC]
        # Trains with eval every 3 steps and asserts the first log entry has eval_loss.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference")

        training_args = RewardConfig(output_dir=self.tmp_dir, eval_strategy="steps", eval_steps=3, report_to="none")
        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        trainer.train()

        assert trainer.state.log_history[0]["eval_loss"] is not None

    def test_train_with_multiple_eval_dataset(self):
        # [FUNCTION DOC]
        # Trains with a dict of two eval datasets and asserts the third-last and second-last log entries contain
        # eval_data1_loss and eval_data2_loss. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference")

        training_args = RewardConfig(output_dir=self.tmp_dir, eval_strategy="steps", eval_steps=3, report_to="none")
        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset={"data1": dataset["test"], "data2": dataset["test"]},
        )
        trainer.train()

        assert trainer.state.log_history[-3]["eval_data1_loss"] is not None
        assert trainer.state.log_history[-2]["eval_data2_loss"] is not None

    @pytest.mark.parametrize("train_dataset_type", ["dataset", "iterable_dataset", "none", "unsupported_dataset_dict"])
    def test_init_with_train_dataset(self, train_dataset_type):
        # [FUNCTION DOC]
        # Checks train_dataset validation: None raises ValueError (train_dataset is required), a DatasetDict raises
        # TypeError (must be Dataset or IterableDataset); Dataset and streaming IterableDataset (max_steps 3) are
        # tokenized so the first example has chosen_ids.
        # Inputs: train_dataset_type (str, parametrized). Outputs: none. Exceptions: AssertionError.
        # Side effects: none.
        streaming = "iterable" in train_dataset_type
        if train_dataset_type == "none":
            train_dataset = None
        else:
            train_dataset = load_dataset(
                "trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train", streaming=streaming
            )
            if train_dataset_type == "unsupported_dataset_dict":
                # `DatasetDict` is representative of any unsupported type here; not exhaustive
                train_dataset = DatasetDict({"train": train_dataset})

        # Iterable (streaming) datasets have no length, so `max_steps` is required.
        training_args = RewardConfig(output_dir=self.tmp_dir, max_steps=3 if streaming else -1, report_to="none")

        if train_dataset_type == "none":
            with pytest.raises(ValueError, match="`train_dataset` is required"):
                RewardTrainer(
                    model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                    args=training_args,
                    train_dataset=train_dataset,
                )
        elif train_dataset_type == "unsupported_dataset_dict":
            with pytest.raises(TypeError, match="`train_dataset` must be a `Dataset` or `IterableDataset`"):
                RewardTrainer(
                    model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                    args=training_args,
                    train_dataset=train_dataset,
                )
        else:
            trainer = RewardTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=train_dataset
            )
            assert "chosen_ids" in next(iter(trainer.train_dataset))

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
        # Checks eval_dataset preparation for None, single datasets and dict-like containers: None stays None; dict
        # results keep keys data1 and data2, each tokenized with chosen_ids; single datasets are tokenized.
        # Inputs: eval_dataset_type (str, parametrized). Outputs: none. Exceptions: AssertionError. Side effects: none.
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        if eval_dataset_type == "none":
            eval_dataset = None
        else:
            streaming = "iterable" in eval_dataset_type
            eval_split = load_dataset(
                "trl-internal-testing/zen", "standard_implicit_prompt_preference", split="test", streaming=streaming
            )
            if eval_dataset_type in ("dataset", "iterable_dataset"):
                eval_dataset = eval_split
            elif eval_dataset_type in ("dataset_dict", "iterable_dataset_dict"):
                dataset_dict_cls = IterableDatasetDict if streaming else DatasetDict
                eval_dataset = dataset_dict_cls({"data1": eval_split, "data2": eval_split})
            else:  # "dict_of_dataset" or "dict_of_iterable_dataset"
                eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = RewardConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = RewardTrainer(
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
            assert "chosen_ids" in next(iter(trainer.eval_dataset["data1"]))
            assert "chosen_ids" in next(iter(trainer.eval_dataset["data2"]))
        else:
            assert "chosen_ids" in next(iter(trainer.eval_dataset))

    def test_train_with_compute_metrics(self):
        # [FUNCTION DOC]
        # Trains with eval every 3 steps and a compute_metrics returning my_metric 0.123; asserts the second-last log
        # entry has eval_my_metric 0.123. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference")

        def dummy_compute_metrics(eval_pred):
            # [FUNCTION DOC]
            # HELPER: Ignores eval_pred and returns a dict with my_metric 0.123. Exceptions: none. Side effects: none.
            return {"my_metric": 0.123}

        training_args = RewardConfig(
            output_dir=self.tmp_dir,
            eval_strategy="steps",
            eval_steps=3,
            report_to="none",
        )
        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=dummy_compute_metrics,
        )

        trainer.train()

        assert trainer.state.log_history[-2]["eval_my_metric"] == 0.123

    # In practice, this test is the same as `test_train`, since gradient checkpointing is enabled by default in
    # `RewardTrainer`. We keep it as a regression guard: if the default ever changes, we still explicitly test gradient
    # checkpointing, which has caused issues in the past.
    def test_train_with_gradient_checkpointing(self):
        # [FUNCTION DOC]
        # Runs the standard train check with gradient_checkpointing explicitly True (regression guard).
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        training_args = RewardConfig(output_dir=self.tmp_dir, gradient_checkpointing=True, report_to="none")
        trainer = RewardTrainer(
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

    @pytest.mark.parametrize("use_reentrant", [True, False])
    def test_train_with_gradient_checkpointing_reentrant(self, use_reentrant):
        # [FUNCTION DOC]
        # Runs the standard train check on the sequence-classification checkpoint with the given use_reentrant.
        # Inputs: use_reentrant (bool, parametrized). Outputs: none. Exceptions: AssertionError.
        # Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        training_args = RewardConfig(
            output_dir=self.tmp_dir,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": use_reentrant},
            report_to="none",
        )
        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
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

    @require_torch_accelerator
    def test_train_with_fp16(self):
        # [FUNCTION DOC]
        # Runs the standard train check with fp16 True; requires an accelerator.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        training_args = RewardConfig(output_dir=self.tmp_dir, fp16=True, report_to="none")
        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_tag_added(self):
        # [FUNCTION DOC]
        # Asserts a trainer built with default args adds reward-trainer and trl to model.model_tags.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: default output dir may be created.
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            train_dataset=dataset,
        )

        for tag in ["reward-trainer", "trl"]:
            assert tag in trainer.model.model_tags

    @require_peft
    def test_tag_added_peft(self):
        # [FUNCTION DOC]
        # Same tag assertion with a LoRA peft_config (requires peft).
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: as above.
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            train_dataset=dataset,
            peft_config=LoraConfig(),
        )

        for tag in ["reward-trainer", "trl"]:
            assert tag in trainer.model.model_tags

    def test_train_with_margin(self):
        # [FUNCTION DOC]
        # Adds a margin column (length of chosen text) and runs the standard train check.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        def add_margin(example):
            # [FUNCTION DOC]
            # HELPER: Returns a dict with margin equal to the character length of chosen. Exceptions: none.
            # Side effects: none.
            # dummy margin based on the length of the chosen summary
            return {"margin": len(example["chosen"])}

        dataset = dataset.map(add_margin)

        training_args = RewardConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = RewardTrainer(
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

    def test_train_with_center_rewards_coefficient(self):
        # [FUNCTION DOC]
        # Runs the standard train check with center_rewards_coefficient 0.01.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        training_args = RewardConfig(output_dir=self.tmp_dir, center_rewards_coefficient=0.01, report_to="none")
        trainer = RewardTrainer(
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

    def test_eos_token_id_synced_with_model_config(self):
        # [FUNCTION DOC]
        # With eos_token set to the im_start token, asserts the tokenizer eos id is 151644 and the model config
        # eos_token_id equals it. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        # The trainer sets the requested eos token on the tokenizer. The model config must follow: otherwise
        # `Trainer` realigns it at train time and reports it as a change the user did not make.
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        training_args = RewardConfig(output_dir=self.tmp_dir, eos_token="<|im_start|>", report_to="none")
        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        eos_token_id = trainer.processing_class.eos_token_id
        assert eos_token_id == 151644  # <|im_start|>
        assert trainer.model.config.eos_token_id == eos_token_id
