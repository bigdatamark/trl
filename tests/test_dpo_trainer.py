# [MODULE DOC]
# Tests for trl.trainer.dpo_trainer: the text and vision preference data collators and DPOTrainer (Direct Preference Optimization) end to end.
# Main contents: TestDataCollatorForPreference, TestDataCollatorForVisionPreference, TestDPOTrainer (text models: training across architectures, loss types, reference model handling, PEFT/quantization, dataset types, evaluation, metrics, init errors), TestDPOTrainerVLM (vision-language models).
# Tests use tiny models and datasets from the trl-internal-testing Hub org (network access or HF cache required); TrlTestCase supplies a temporary output directory as self.tmp_dir.
# Import-time side effects: PEFT symbols imported only when PEFT is installed.

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
import transformers
from datasets import Dataset, DatasetDict, IterableDatasetDict, load_dataset
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.testing_utils import torch_device
from transformers.utils import is_peft_available

from trl import DPOConfig, DPOTrainer
from trl.trainer.dpo_trainer import DataCollatorForPreference, DataCollatorForVisionPreference

from .testing_utils import (
    TrlTestCase,
    is_ampere_or_newer,
    is_bf16_supported,
    require_bitsandbytes,
    require_kernels,
    require_liger_kernel,
    require_peft,
    require_peft_target_parameters,
    require_torch_accelerator,
    require_vision,
)


if is_peft_available():
    import peft
    from peft import LoraConfig, PromptTuningConfig, get_peft_model
    from peft.utils import TaskType


class TestDataCollatorForPreference(TrlTestCase):
    # [CLASS DOC]
    # Tests DataCollatorForPreference, which turns tokenized prompt/chosen/rejected examples into one batch where chosen sequences come first and rejected sequences second.
    # Each test builds a collator with pad token id 0 and checks the exact padded tensors. Inherits TrlTestCase.
    def test_padding_and_masks(self):
        # [FUNCTION DOC]
        # Tests padding and masks for two examples of different lengths.
        # Assertions: output keys are exactly input_ids, attention_mask, completion_mask; input_ids stack prompt+chosen for each example then prompt+rejected, right-padded with 0 to the longest row; attention mask is 1 on real tokens; completion mask is 1 only on chosen/rejected tokens.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: none.
        collator = DataCollatorForPreference(pad_token_id=0)
        examples = [
            {"prompt_ids": [1, 2, 3], "chosen_ids": [4, 5], "rejected_ids": [6]},
            {"prompt_ids": [7, 8], "chosen_ids": [9, 10], "rejected_ids": [11, 12, 13]},
        ]
        result = collator(examples)

        expected_input_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5],  # prompt + chosen (example 1)
                [7, 8, 9, 10, 0],  # prompt + chosen (example 2, padded)
                [1, 2, 3, 6, 0],  # prompt + rejected (example 1, padded)
                [7, 8, 11, 12, 13],  # prompt + rejected (example 2)
            ]
        )
        expected_attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
            ]
        )
        expected_completion_mask = torch.tensor(
            [
                [0, 0, 0, 1, 1],  # chosen completion (example 1)
                [0, 0, 1, 1, 0],  # chosen completion (example 2, padded)
                [0, 0, 0, 1, 0],  # rejected completion (example 1, padded)
                [0, 0, 1, 1, 1],  # rejected completion (example 2)
            ]
        )

        assert set(result.keys()) == {"input_ids", "attention_mask", "completion_mask"}
        torch.testing.assert_close(result["input_ids"], expected_input_ids)
        torch.testing.assert_close(result["attention_mask"], expected_attention_mask)
        torch.testing.assert_close(result["completion_mask"], expected_completion_mask)

    def test_optional_reference_logps(self):
        # [FUNCTION DOC]
        # Tests that ref_chosen_logps and ref_rejected_logps fields in examples are collated into float tensors and added to the output keys.
        # Assertions: key set includes both ref fields; values equal the per-example floats in order.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: none.
        collator = DataCollatorForPreference(pad_token_id=0)
        examples = [
            {
                "prompt_ids": [1, 2],
                "chosen_ids": [3],
                "rejected_ids": [4],
                "ref_chosen_logps": 0.1,
                "ref_rejected_logps": 0.2,
            },
            {
                "prompt_ids": [5],
                "chosen_ids": [6, 7],
                "rejected_ids": [8, 9],
                "ref_chosen_logps": 0.3,
                "ref_rejected_logps": 0.4,
            },
        ]
        result = collator(examples)

        expected_ref_chosen_logps = torch.tensor([0.1, 0.3])
        expected_ref_rejected_logps = torch.tensor([0.2, 0.4])

        assert set(result.keys()) == {
            "input_ids",
            "attention_mask",
            "completion_mask",
            "ref_chosen_logps",
            "ref_rejected_logps",
        }
        torch.testing.assert_close(result["ref_chosen_logps"], expected_ref_chosen_logps)
        torch.testing.assert_close(result["ref_rejected_logps"], expected_ref_rejected_logps)

    def test_with_pad_to_multiple_of(self):
        # [FUNCTION DOC]
        # Tests pad_to_multiple_of=5: sequences of length 2 and 4 are padded to length 5.
        # Assertions: key set and exact padded input_ids.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: none.
        collator = DataCollatorForPreference(pad_token_id=0, pad_to_multiple_of=5)
        examples = [
            {"prompt_ids": [1], "chosen_ids": [2], "rejected_ids": [3]},
            {"prompt_ids": [4, 5], "chosen_ids": [6, 7], "rejected_ids": [8, 9]},
        ]
        result = collator(examples)

        expected_input_ids = torch.tensor(
            [
                [1, 2, 0, 0, 0],  # prompt + chosen (example 1, padded to multiple of 5)
                [4, 5, 6, 7, 0],  # prompt + chosen (example 2)
                [1, 3, 0, 0, 0],  # prompt + rejected (example 1, padded to multiple of 5)
                [4, 5, 8, 9, 0],  # prompt + rejected (example 2)
            ]
        )

        assert set(result.keys()) == {"input_ids", "attention_mask", "completion_mask"}
        torch.testing.assert_close(result["input_ids"], expected_input_ids)


class TestDataCollatorForVisionPreference(TrlTestCase):
    # [CLASS DOC]
    # Tests DataCollatorForVisionPreference, which applies a processor to image + conversational preference examples.
    # Inherits TrlTestCase.
    @pytest.mark.skipif(
        Version(transformers.__version__) < Version("5.3.0"),
        reason="mm_token_type_ids are returned by default since transformers-5.3.0 (see transformers#43972)",
    )
    @require_vision
    def test_mm_token_type_ids_shape(self):
        # [FUNCTION DOC]
        # Regression test: when the processor returns mm_token_type_ids (transformers 5.3.0 or later, e.g. Qwen2.5-VL), the collator must extend it for the completion part so its shape matches input_ids.
        # Setup: tiny Qwen2.5-VL processor, one example with a 16x16 RGB image, a user prompt and chosen/rejected assistant replies. Skipped on older transformers; requires vision deps.
        # Assertions: mm_token_type_ids is in the output and has the same shape as input_ids.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: downloads the processor.
        # Regression test: when the processor returns mm_token_type_ids (e.g. Qwen2.5-VL after
        # transformers#43972), the collator must concatenate it with zeros for the completion part
        # so that its shape matches input_ids. Without the fix this raises an IndexError in the model.
        from PIL import Image
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained("trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration")
        collator = DataCollatorForVisionPreference(processor)
        image = Image.new("RGB", (16, 16))
        examples = [
            {
                "images": [image],
                "prompt": [{"role": "user", "content": "What is this?"}],
                "chosen": [{"role": "assistant", "content": "A red square."}],
                "rejected": [{"role": "assistant", "content": "A blue circle."}],
            }
        ]
        output = collator(examples)
        assert "mm_token_type_ids" in output
        assert output["mm_token_type_ids"].shape == output["input_ids"].shape, (
            f"mm_token_type_ids shape {output['mm_token_type_ids'].shape} != "
            f"input_ids shape {output['input_ids'].shape}"
        )


class TestDPOTrainer(TrlTestCase):
    # [CLASS DOC]
    # End-to-end tests for DPOTrainer on text-only causal LMs.
    # Recurring pattern ("standard train check"): load the zen standard_preference train split, build DPOConfig with output_dir self.tmp_dir, learning rate 0.1 (tiny gradients) and report_to none, snapshot all named parameters, train, assert the last log entry has a train_loss, and assert every parameter changed. PEFT variants instead assert base parameters unchanged and adapter parameters changed (excluding base_layer, and the "ref" adapter where present).
    # Other tests check init-time validation errors, evaluation paths, reference model behaviour, dataset preprocessing and logged metrics. Inherits TrlTestCase.
    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Cohere2ForCausalLM",
            pytest.param(
                "trl-internal-testing/tiny-Glm4MoeForCausalLM",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.0.0"),
                    reason="GLM4 tokenizer requires transformers>=5.0.0",
                ),
            ),
            "trl-internal-testing/tiny-GptOssForCausalLM",
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            "trl-internal-testing/tiny-Qwen3MoeForCausalLM",
            pytest.param(
                "trl-internal-testing/tiny-NemotronHForCausalLM-nano",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.7.0"),
                    reason="Nemotron 3 gradient checkpointing requires transformers>=5.7.0 (see transformers#45625)",
                ),
            ),
            pytest.param(
                "trl-internal-testing/tiny-Olmo3ForCausalLM",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("4.57.0"),
                    reason="Olmo 3 requires transformers>=4.57.0",
                ),
            ),
            "trl-internal-testing/tiny-Lfm2ForCausalLM",
            pytest.param(
                "trl-internal-testing/tiny-Lfm2ForCausalLM-2.5",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.0.0"),
                    reason="LFM2.5 tokenizer requires transformers>=5.0.0",
                ),
            ),
        ],
    )
    def test_train(self, model_id):
        # [FUNCTION DOC]
        # Standard train check across many tiny architectures (Cohere2, GLM4-MoE, GPT-OSS, Qwen2.5, Qwen3-MoE, Nemotron-H, Olmo3, LFM2, LFM2.5), some skipped on older transformers.
        # Extra assertion: if the trainer enabled the MoE auxiliary loss, the last log contains aux_loss.
        # Inputs: model_id (Hub id) from parametrize. Outputs: none. Exceptions: AssertionError. Side effects: downloads models, writes to self.tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = DPOTrainer(model=model_id, args=training_args, train_dataset=dataset)

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # MoE models log the load-balancing auxiliary loss (on by default)
        if trainer.aux_loss_enabled:
            assert trainer.state.log_history[-1]["aux_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_liger_kernel
    def test_train_with_liger(self):
        # [FUNCTION DOC]
        # Standard train check with use_liger_kernel=True on Qwen2.5 (requires liger kernel).
        # Extra assertions: building the trainer emits a FutureWarning that use_liger_kernel is deprecated, and liger_kernel_config has fused_linear_cross_entropy set to False (it would replace the fused LM head forward).
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(output_dir=self.tmp_dir, learning_rate=0.1, use_liger_kernel=True, report_to="none")
        with pytest.warns(FutureWarning, match="`use_liger_kernel=True` is deprecated"):
            trainer = DPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=dataset,
            )
        # Liger's fused linear cross-entropy would replace the forward that carries the fused LM head
        assert trainer.args.liger_kernel_config["fused_linear_cross_entropy"] is False

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize("precompute_ref_log_probs", [False, True])
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
    def test_evaluate_with_eval_dataset(self, eval_dataset_type, precompute_ref_log_probs):
        # [FUNCTION DOC]
        # Tests that evaluate accepts a raw, unprepared eval dataset passed directly (issue 6115).
        # Setup: eval dataset given as a Dataset, IterableDataset, DatasetDict, IterableDatasetDict, or plain dict of either (two keys data1/data2, same test split); trainer built without eval_dataset.
        # Branches: streaming with precompute_ref_log_probs must raise ValueError about IterableDataset not being supported, then return; single dataset yields eval_loss; dict types yield eval_data1_loss and eval_data2_loss.
        # Inputs: eval_dataset_type, precompute_ref_log_probs from parametrize. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        # `evaluate` accepts a raw (unprepared) dataset passed directly, not only a preprocessed `eval_dataset` set
        # at init. See https://github.com/huggingface/trl/issues/6115.
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        streaming = "iterable" in eval_dataset_type
        eval_split = load_dataset("trl-internal-testing/zen", "standard_preference", split="test", streaming=streaming)
        if eval_dataset_type in ("dataset", "iterable_dataset"):
            eval_dataset = eval_split
        elif eval_dataset_type in ("dataset_dict", "iterable_dataset_dict"):
            dataset_dict_cls = IterableDatasetDict if streaming else DatasetDict
            eval_dataset = dataset_dict_cls({"data1": eval_split, "data2": eval_split})
        else:  # "dict_of_dataset" or "dict_of_iterable_dataset"
            eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = DPOConfig(
            output_dir=self.tmp_dir, precompute_ref_log_probs=precompute_ref_log_probs, report_to="none"
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=train_dataset
        )

        if streaming and precompute_ref_log_probs:
            with pytest.raises(ValueError, match="precompute_ref_log_probs.*not supported with IterableDataset"):
                trainer.evaluate(eval_dataset=eval_dataset)
            return

        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        if eval_dataset_type in ("dataset", "iterable_dataset"):
            assert metrics["eval_loss"] is not None
        else:
            assert metrics["eval_data1_loss"] is not None
            assert metrics["eval_data2_loss"] is not None

    def test_evaluate_precompute_ref_log_probs_after_training_raises(self):
        # [FUNCTION DOC]
        # Tests that with full fine-tuning, precompute_ref_log_probs=True and no ref model, evaluating a newly passed dataset works before training but raises after training (the policy no longer equals the reference).
        # Setup: max_steps 1. Assertions: evaluate before training gives eval_loss; after training, evaluate with the dataset raises ValueError mentioning that reference log-probs cannot be computed.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        # Full fine-tuning with `precompute_ref_log_probs=True` and no `ref_model` uses `self.model` as the reference.
        # That's valid only before training; a dataset passed to `evaluate()` afterwards can't get a correct reference.
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        eval_dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="test")
        training_args = DPOConfig(
            output_dir=self.tmp_dir, max_steps=1, precompute_ref_log_probs=True, report_to="none"
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=train_dataset
        )

        # Before training the reference is available, so evaluating a new dataset works.
        assert trainer.evaluate(eval_dataset=eval_dataset)["eval_loss"] is not None

        trainer.train()

        with pytest.raises(ValueError, match="Cannot compute reference log-probs for a dataset passed to"):
            trainer.evaluate(eval_dataset=eval_dataset)

    @pytest.mark.parametrize("eval_dataset_type", ["dataset", "dataset_dict", "dict_of_dataset"])
    def test_evaluate_precompute_ref_log_probs_at_init_after_training(self, eval_dataset_type):
        # [FUNCTION DOC]
        # Regression test for the guard above: an eval_dataset given at init has its reference log-probs precomputed once and stored, so evaluate with no arguments after training must not raise.
        # Setup: eval dataset as Dataset, DatasetDict or dict; max_steps 1; precompute_ref_log_probs=True.
        # Assertions: eval_loss (or eval_data1_loss and eval_data2_loss) present after training.
        # Inputs: eval_dataset_type from parametrize. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        # Regression for the guard above: an `eval_dataset` set at init has its reference log-probs precomputed once
        # (against the untrained reference) and stored, so no-arg `evaluate()` reuses those stored values and must not
        # raise after training, even with full fine-tuning and `precompute_ref_log_probs=True`.
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        eval_split = load_dataset("trl-internal-testing/zen", "standard_preference", split="test")
        if eval_dataset_type == "dataset":
            eval_dataset = eval_split
        elif eval_dataset_type == "dataset_dict":
            eval_dataset = DatasetDict({"data1": eval_split, "data2": eval_split})
        else:  # "dict_of_dataset"
            eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = DPOConfig(
            output_dir=self.tmp_dir, max_steps=1, precompute_ref_log_probs=True, report_to="none"
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        trainer.train()

        metrics = trainer.evaluate()
        if eval_dataset_type == "dataset":
            assert metrics["eval_loss"] is not None
        else:
            assert metrics["eval_data1_loss"] is not None
            assert metrics["eval_data2_loss"] is not None

    def test_trust_remote_code(self):
        # [FUNCTION DOC]
        # Tests trust_remote_code: a Hub model with custom code fails without it and loads with it.
        # Assertions: ValueError mentioning custom code when trust_remote_code is unset; with trust_remote_code=True the model class name is RemoteForCausalLM.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: downloads and executes remote model code.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        model_id = "trl-internal-testing/tiny-RemoteForCausalLM"

        with pytest.raises(ValueError, match="custom code"):
            DPOTrainer(
                model=model_id,
                args=DPOConfig(output_dir=self.tmp_dir, report_to="none"),
                train_dataset=dataset,
            )

        trainer = DPOTrainer(
            model=model_id,
            args=DPOConfig(output_dir=self.tmp_dir, report_to="none", trust_remote_code=True),
            train_dataset=dataset,
        )
        assert type(trainer.model).__name__ == "RemoteForCausalLM"

    @pytest.mark.parametrize(
        "config_name",
        [
            "standard_preference",
            "conversational_preference",
            "standard_implicit_prompt_preference",
            "conversational_implicit_prompt_preference",
        ],
    )
    def test_train_dataset_format(self, config_name):
        # [FUNCTION DOC]
        # Standard train check on Qwen2.5 for the four preference dataset formats (standard / conversational, explicit / implicit prompt).
        # Inputs: config_name from parametrize. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", config_name, split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    # Special case for harmony
    def test_train_gpt_oss(self):
        # [FUNCTION DOC]
        # Standard train check on tiny GPT-OSS with the harmony preference dataset (special chat format).
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        dataset = load_dataset("trl-internal-testing/harmony", "preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-GptOssForCausalLM", args=training_args, train_dataset=dataset
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
        # Standard train check with an already-instantiated float32 model object instead of a model id.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            dtype="float32",
        )

        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = DPOTrainer(model=model, args=training_args, train_dataset=dataset)

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize(
        "loss_type",
        [
            "sigmoid",
            "hinge",
            "ipo",
            "exo_pair",
            "nca_pair",
            "robust",
            "bco_pair",
            "sppo_hard",
            "aot",
            "aot_unpaired",
            "apo_zero",
            "apo_down",
            "discopop",
            "sft",
            "sigmoid_norm",
        ],
    )
    def test_train_loss_types(self, loss_type):
        # [FUNCTION DOC]
        # Standard train check for each supported single loss_type (sigmoid, hinge, ipo, exo_pair, nca_pair, robust, bco_pair, sppo_hard, aot, aot_unpaired, apo_zero, apo_down, discopop, sft, sigmoid_norm), with evaluation every 3 steps on the test split; label_smoothing 1e-3 for exo_pair, else 0.
        # Inputs: loss_type from parametrize. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            loss_type=loss_type,
            label_smoothing=1e-3 if loss_type == "exo_pair" else 0.0,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
            eval_strategy="steps",
            eval_steps=3,
        )
        trainer = DPOTrainer(
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

    def test_train_multi_loss_types(self):
        # [FUNCTION DOC]
        # Standard train check with a list of loss types (sigmoid, bco_pair, sft), the combination used by MPO.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            loss_type=["sigmoid", "bco_pair", "sft"],  # this specific combination is used in MPO
            report_to="none",
        )
        trainer = DPOTrainer(
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

    def test_train_with_wpo(self):
        # [FUNCTION DOC]
        # Standard train check with use_weighting=True (WPO weighting).
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
            use_weighting=True,
        )
        trainer = DPOTrainer(
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

    def test_train_with_ld(self):
        # [FUNCTION DOC]
        # Standard train check with ld_alpha=0.5 (LD-DPO length desensitization).
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
            ld_alpha=0.5,
        )
        trainer = DPOTrainer(
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
        "f_divergence_type",
        ["reverse_kl", "forward_kl", "js_divergence", "alpha_divergence"],
    )
    def test_train_with_f_divergence(self, f_divergence_type):
        # [FUNCTION DOC]
        # Standard train check for each f_divergence_type (reverse_kl, forward_kl, js_divergence, alpha_divergence).
        # Inputs: f_divergence_type from parametrize. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
            f_divergence_type=f_divergence_type,
        )
        trainer = DPOTrainer(
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

    @pytest.mark.parametrize("loss_type", ["apo_zero", "apo_down", "aot", "aot_unpaired", "nca_pair", "sppo_hard"])
    def test_init_fails_with_f_divergence_and_unsupported_loss(self, loss_type):
        # [FUNCTION DOC]
        # Tests that a non-default f_divergence_type (js_divergence) combined with a loss not based on the chosen-rejected reward difference fails at init.
        # Assertions: DPOTrainer construction raises ValueError mentioning the supported loss types.
        # Inputs: loss_type (apo_zero, apo_down, aot, aot_unpaired, nca_pair, sppo_hard) from parametrize. Outputs: none. Exceptions: AssertionError. Side effects: none.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        # These losses are not built on the chosen-rejected reward difference, so a non-default value is rejected.
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            loss_type=loss_type,
            f_divergence_type="js_divergence",
            report_to="none",
        )

        with pytest.raises(ValueError, match="is only supported for the following loss types"):
            DPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=dataset,
            )

    def test_train_with_explicit_ref_model(self):
        # [FUNCTION DOC]
        # Standard train check with an explicit ref_model (same checkpoint loaded separately in float32).
        # Extra assertion: every reference model parameter still equals the initial policy parameter (ref model is frozen).
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        # When specifying a ref model, it's usually because we want it to be a different checkpoint, but for testing
        # purposes we will just just use the same checkpoint
        ref_model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", dtype="float32"
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            ref_model=ref_model,
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
            new_ref_param = trainer.ref_model.get_parameter(n)
            torch.testing.assert_close(param, new_ref_param, msg=f"Reference model parameter {n} has changed.")

    def test_train_with_sync_ref_model(self):
        # [FUNCTION DOC]
        # Standard train check with sync_ref_model=True and ref_model_sync_steps=2.
        # Extra assertions: trainer has a ref_model, and every reference parameter changed after training (synced from the policy).
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            sync_ref_model=True,
            ref_model_sync_steps=2,  # reduce sync steps to ensure a sync happens
            report_to="none",
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}
        assert trainer.ref_model is not None
        previous_ref_params = {n: param.clone() for n, param in trainer.ref_model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."
            new_ref_param = trainer.ref_model.get_parameter(n)
            assert not torch.equal(previous_ref_params[n], new_ref_param), f"Ref Parameter {n} has not changed."

    def test_ref_model_like_model_instance(self):
        # [FUNCTION DOC]
        # Tests that when the policy is passed as a model instance (bfloat16, eager attention), the auto-created reference model copies its dtype and attention implementation.
        # Assertions: ref_model dtype is bfloat16 and its config attention implementation is eager.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: none.
        # A policy passed as an instance has no `model_init_kwargs`: the reference is built with its dtype and attention
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", dtype=torch.bfloat16, attn_implementation="eager"
        )
        training_args = DPOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = DPOTrainer(model=model, args=training_args, train_dataset=dataset)

        assert trainer.ref_model.dtype == torch.bfloat16
        assert trainer.ref_model.config._attn_implementation == "eager"

    def test_train_model_dtype(self):
        # [FUNCTION DOC]
        # Standard train check with model_init_kwargs dtype float16.
        # Extra assertion: each checked parameter is float16. Parameters with layernorm in the name are skipped (they do not change reliably in CI).
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            model_init_kwargs={"dtype": torch.float16},
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
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
    def test_peft_init_is_seeded(self):
        # [FUNCTION DOC]
        # Tests that LoRA adapter initialization is seeded by the trainer: two trainers built after different global torch seeds have identical lora_A weights.
        # Assertions: the collected lora_A dict is non-empty and every tensor matches between the two trainers.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: mutates global torch RNG state.
        # Two trainers with the same seed start from the same adapter weights
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        adapters = []
        for global_seed in range(2):
            torch.manual_seed(global_seed)  # a different global RNG state, as in two separate runs
            trainer = DPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=DPOConfig(output_dir=self.tmp_dir, report_to="none"),
                train_dataset=dataset,
                peft_config=LoraConfig(),
            )
            adapters.append({n: p.clone() for n, p in trainer.model.named_parameters() if "lora_A" in n})

        assert adapters[0]
        for n, param in adapters[0].items():
            assert torch.equal(param, adapters[1][n]), f"Parameter {n} differs between the two trainers."

    @require_peft
    def test_train_dense_with_peft_config_lora(self):
        # [FUNCTION DOC]
        # PEFT train check on dense Qwen2.5 with a default LoraConfig passed as peft_config (learning rate 1.0).
        # Assertions: base model parameters (prefixed base_model.model.) unchanged; other parameters except base_layer changed.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=1.0,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )

        trainer = DPOTrainer(
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

    @require_peft_target_parameters
    def test_train_moe_with_peft_config(self):
        # [FUNCTION DOC]
        # PEFT train check on MoE GPT-OSS with LoRA targeting expert parameters (mlp.experts.down_proj, mlp.experts.gate_up_proj) via target_parameters.
        # Assertions: base parameters unchanged, adapter parameters (non base_layer) changed.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        model_id = "trl-internal-testing/tiny-GptOssForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )

        trainer = DPOTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(target_parameters=["mlp.experts.down_proj", "mlp.experts.gate_up_proj"]),
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
    def test_train_peft_model(self):
        # [FUNCTION DOC]
        # PEFT train check with a model already wrapped by get_peft_model (learning rate 1.0).
        # Assertions: base parameters unchanged; parameters other than base_layer and the ref adapter changed.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")

        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        lora_config = LoraConfig()
        model = get_peft_model(model, lora_config)

        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=1.0,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = DPOTrainer(model=model, args=training_args, train_dataset=dataset)

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "base_layer" not in n and "ref" not in n:  # and the peft params to be different (except base and ref)
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_peft_target_parameters
    def test_train_moe_peft_model(self):
        # [FUNCTION DOC]
        # Regression test (issue 5222): PEFT-wrapped MoE model with target_parameters LoRA.
        # Extra assertion: on PEFT older than 0.20.0 no "ref" adapter is created (only one adapter supported with target_parameters); on newer PEFT a "ref" adapter exists. Then the PEFT train check, excluding base_layer and ref adapter parameters.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        # Regression test for https://github.com/huggingface/trl/issues/5222. Before PEFT 0.20.0, only one adapter per
        # model was supported when the LoRA config uses `target_parameters` (see peft#3340, fixed in peft#3350), so no
        # "ref" adapter could be created and the reference log probs were computed with adapters disabled instead.
        model_id = "trl-internal-testing/tiny-GptOssForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        lora_config = LoraConfig(target_parameters=["mlp.experts.down_proj", "mlp.experts.gate_up_proj"])
        model = get_peft_model(model, lora_config)

        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = DPOTrainer(model=model, args=training_args, train_dataset=dataset)

        if Version(peft.__version__) < Version("0.20.0"):
            assert "ref" not in trainer.model.peft_config
        else:
            assert "ref" in trainer.model.peft_config

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "base_layer" not in n and "ref" not in n:  # and the peft params to be different (except base and ref)
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    # In practice, this test is the same as `test_train_dense_with_peft_config_lora`, since gradient checkpointing is
    # enabled by default in `DPOTrainer`. We keep it as a regression guard: if the default ever changes, we still
    # explicitly test PEFT + gradient checkpointing, which has caused issues in the past.
    @require_peft
    def test_train_with_peft_config_and_gradient_checkpointing(self):
        # [FUNCTION DOC]
        # PEFT train check with LoraConfig and gradient_checkpointing=True explicitly (regression guard; it is also the default).
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            gradient_checkpointing=True,
            report_to="none",
        )

        trainer = DPOTrainer(
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

    @pytest.mark.skipif(not is_bf16_supported(), reason="test requires bf16 support")
    def test_ref_logps_use_mixed_precision(self):
        # [FUNCTION DOC]
        # Tests that reference log-prob computation enters autocast itself, so under bf16 training the backbone runs in bfloat16.
        # Setup: bf16=True, batch size 2; one batch from the train dataloader prepared via _prepare_inputs; a forward hook on the first q_proj module records output dtypes while compute_ref_log_probs runs. Skipped without bf16 support.
        # Assertions: exactly one recorded dtype, bfloat16; both ref logps tensors finite.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: none.
        # Reference precomputation runs outside `compute_loss`, so it has to enter autocast itself. Without that the
        # backbone would silently run in fp32 under bf16 training.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            bf16=True,
            per_device_train_batch_size=2,
            report_to="none",
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )
        inputs = trainer._prepare_inputs(next(iter(trainer.get_train_dataloader())))
        projection = next(module for name, module in trainer.model.named_modules() if name.endswith("q_proj"))
        projection_dtypes = []

        def record_projection_dtype(_module, _args, output):
            # [FUNCTION DOC]
            # HELPER: Forward hook recording the dtype of the module output into projection_dtypes.
            # Inputs: module, args (unused), output tensor. Outputs: None. Exceptions: none. Side effects: appends to the enclosing list.
            projection_dtypes.append(output.dtype)

        with projection.register_forward_hook(record_projection_dtype):
            ref_chosen_logps, ref_rejected_logps = trainer.compute_ref_log_probs(trainer.model, inputs)

        assert projection_dtypes == [torch.bfloat16]
        assert torch.isfinite(ref_chosen_logps).all() and torch.isfinite(ref_rejected_logps).all()

    def test_train_with_precomputed_ref_logps(self):
        # [FUNCTION DOC]
        # Tests one training step with precompute_ref_log_probs=True (batch size 2, max_steps 1).
        # Assertions: last log has train_loss.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=1,
            precompute_ref_log_probs=True,
            report_to="none",
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    @require_peft
    def test_peft_lm_head_raises(self):
        # [FUNCTION DOC]
        # Tests that a LoRA adapter on lm_head is rejected at init, since the chunked projection reads lm_head.weight directly and would never train it.
        # Setup: LoraConfig targeting q_proj, v_proj, lm_head; ensure_weight_tying=True added only for PEFT versions from 0.19.0 up to before 0.19.2.dev0 to silence a weight-tying warning.
        # Assertions: construction raises ValueError mentioning lm_head.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: none.
        # The chunked projection reads `lm_head.weight` directly, so a LoRA adapter on `lm_head` is silently
        # ignored and never trained. The trainer must fail fast instead of training a silently-frozen head.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        training_args = DPOConfig(output_dir=self.tmp_dir, report_to="none")
        # `ensure_weight_tying=True` silences PEFT's weight-tying warning fired when lm_head is in the adapter on a
        # model with untied embeddings; once tying respects `tie_word_embeddings`, the flag itself warns that no tied
        # modules were found, so it must only be set on the affected range.
        # - Introduced in PEFT 0.19.0 (peft#2879); fixed on main, unreleased as of 0.19.2.dev0 (peft#3171)
        needs_ensure_weight_tying = Version("0.19.0") <= Version(peft.__version__) < Version("0.19.2.dev0")
        lora_config = LoraConfig(
            target_modules=["q_proj", "v_proj", "lm_head"],
            **({"ensure_weight_tying": True} if needs_ensure_weight_tying else {}),
        )
        with pytest.raises(ValueError, match="lm_head"):
            DPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=dataset,
                peft_config=lora_config,
            )

    @require_peft
    def test_train_peft_prompt_tuning(self):
        # [FUNCTION DOC]
        # Train check with prompt tuning (PromptTuningConfig, causal LM, 8 virtual tokens).
        # Assertions: train_loss logged; parameters whose name contains prompt_encoder changed.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            peft_config=PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=8),
        )
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the prompt embeddings have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if "prompt_encoder" in n:
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize("iterable_as", ["train", "eval", "eval_dict", "eval_iterable_dataset_dict"])
    def test_precompute_ref_log_probs_raises_for_iterable_dataset(self, iterable_as):
        # [FUNCTION DOC]
        # Tests that precompute_ref_log_probs=True is rejected for streaming datasets (caching is index based).
        # Setup: the streaming dataset is placed as train dataset, eval dataset, inside a dict eval dataset, or inside an IterableDatasetDict; max_steps 1 so the iterable train case reaches the check.
        # Assertions: construction raises ValueError about IterableDataset not being supported.
        # Inputs: iterable_as from parametrize. Outputs: none. Exceptions: AssertionError. Side effects: none.
        # `precompute_ref_log_probs=True` caches reference log-probs by index, which requires random access and is
        # therefore incompatible with a streaming `IterableDataset` — whether passed directly or nested in a dict or
        # `IterableDatasetDict` as the eval dataset.
        map_dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        iterable_dataset = load_dataset(
            "trl-internal-testing/zen", "standard_preference", split="train", streaming=True
        )

        train_dataset, eval_dataset = map_dataset, None
        if iterable_as == "train":
            train_dataset = iterable_dataset
        elif iterable_as == "eval":
            eval_dataset = iterable_dataset
        elif iterable_as == "eval_dict":
            eval_dataset = {"eval": iterable_dataset}
        elif iterable_as == "eval_iterable_dataset_dict":
            eval_dataset = IterableDatasetDict({"eval": iterable_dataset})

        # `max_steps` lets the iterable-train case reach the precompute check instead of failing earlier on the
        # missing dataset length required by the learning-rate scheduler.
        training_args = DPOConfig(
            output_dir=self.tmp_dir, max_steps=1, report_to="none", precompute_ref_log_probs=True
        )
        with pytest.raises(ValueError, match="precompute_ref_log_probs.*not supported with IterableDataset"):
            DPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )

    def test_train_with_iterable_dataset(self):
        # [FUNCTION DOC]
        # Standard train check with a streaming (iterable) train dataset and max_steps 3.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train", streaming=True)

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            max_steps=3,
            report_to="none",
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_kernels
    @pytest.mark.skipif(
        not is_ampere_or_newer() and torch_device != "xpu",
        reason="Flash Attention 2 requires Ampere or newer GPU, or XPU",
    )
    def test_train_padding_free(self):
        # [FUNCTION DOC]
        # Tests padding_free=True training with the kernels-community flash-attn2 attention and bf16.
        # Assertions: last log has train_loss. Requires kernels and an Ampere-or-newer GPU or XPU.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            padding_free=True,
            model_init_kwargs={"attn_implementation": "kernels-community/flash-attn2"},
            bf16=True,  # flash_attention_2 only supports bf16 and fp16
            report_to="none",
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.skipif(
        Version(transformers.__version__) < Version("4.57.0"), reason="Olmo 3 requires transformers>=4.57.0"
    )
    def test_tokenize_prompt_not_prefix_of_conversation(self):
        # [FUNCTION DOC]
        # Tests tokenization when the tokenized prompt is not a prefix of the tokenized full conversation (Olmo-3 Think template appends a think tag to the generation prompt).
        # Assertions: for each example and for chosen and rejected, decoding the processed completion ids gives the assistant content followed by the EOS token. Skipped on transformers older than 4.57.0.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: none.
        # The Olmo-3 Think template ends the generation prompt with `<think>`, which the full conversation doesn't
        # have, so the tokenized prompt is not a prefix of the tokenized prompt+completion
        dataset = load_dataset("trl-internal-testing/zen", "conversational_preference", split="train")

        training_args = DPOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Olmo3ForCausalLM",
            args=training_args,
            train_dataset=dataset,
        )

        # The chosen and rejected completions are the whole assistant turns
        tokenizer = trainer.processing_class
        for example, processed in zip(dataset, trainer.train_dataset, strict=True):
            for key in ("chosen", "rejected"):
                assert tokenizer.decode(processed[f"{key}_ids"]) == example[key][0]["content"] + tokenizer.eos_token

    def test_train_with_chat_template_kwargs(self):
        # [FUNCTION DOC]
        # Tests per-example chat_template_kwargs: a custom simplified Qwen template uses a role_capital flag to uppercase role names.
        # Setup: tokenizer with the custom template; dataset gets a chat_template_kwargs column alternating role_capital False/True.
        # Assertions: column exists; trainer keeps the template; the first two processed prompts start with the default system prompt using "system" or "SYSTEM" respectively; training logs train_loss.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )

        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        # The following template is a simplified version of the Qwen chat template, where an additional argument
        # `role_capital` is used to control the capitalization of roles.
        tokenizer.chat_template = '{%- if messages[0]["role"] == "system" -%}    {{ "<|im_start|>" + ("SYSTEM" if role_capital else "system") + "\\n" + messages[0]["content"] + "<|im_end|>\\n" }}{%- else -%}    {{ "<|im_start|>" + ("SYSTEM" if role_capital else "system") + "\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n" }}{%- endif -%}{%- for message in messages -%}    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) -%}        {{ "<|im_start|>" + (message.role.upper() if role_capital else message.role) + "\\n" + message.content + "<|im_end|>\\n" }}    {%- elif message.role == "assistant" -%}        {{ "<|im_start|>" + ("ASSISTANT" if role_capital else "assistant") }}        {%- if message.content -%}            {{ "\\n" + message.content }}        {%- endif -%}        {{ "<|im_end|>\\n" }}    {%- elif message.role == "tool" -%}        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") -%}            {{ "<|im_start|>" + ("USER" if role_capital else "user") }}        {%- endif -%}        {{ "\\n<tool_response>\\n" + message.content + "\\n</tool_response>" }}        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") -%}            {{ "<|im_end|>\\n" }}        {%- endif -%}    {%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}    {{ "<|im_start|>" + ("ASSISTANT" if role_capital else "assistant") + "\\n" }}{%- endif -%}'

        dataset = dataset.add_column(
            "chat_template_kwargs", [{"role_capital": bool(i % 2)} for i in range(len(dataset))]
        )
        assert "chat_template_kwargs" in dataset.features

        trainer = DPOTrainer(
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
            assert trainer.train_dataset[i]["prompt_ids"][: len(system_prompt_ids)] == system_prompt_ids

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_fully_truncated_completion_examples_dropped(self):
        # [FUNCTION DOC]
        # Tests that with keep_start truncation and max_length 6, an example whose prompt alone fills max_length (no completion tokens left) is dropped during preparation.
        # Assertions: one example remains and its prompt ids equal the tokenization of "Hi".
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: none.
        """With `keep_start` truncation, an example whose prompt alone fills `max_length` loses every completion token
        in the collator, so it's dropped during dataset preparation."""
        dataset = Dataset.from_dict(
            {
                "prompt": ["Hi", "This is a very long prompt that fills up the whole max_length budget on its own"],
                "chosen": [" there", " yes"],
                "rejected": [" bye", " no"],
            }
        )

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            max_length=6,
            truncation_mode="keep_start",
            report_to="none",
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        # Only the short-prompt example survives.
        assert len(trainer.train_dataset) == 1
        assert trainer.train_dataset[0]["prompt_ids"] == trainer.processing_class("Hi")["input_ids"]

    def test_train_toolcall_data(self):
        # [FUNCTION DOC]
        # Tests training on tool-call preference data (batch size 2, max_length 512).
        # Assertions: last log has train_loss.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        dataset = load_dataset("trl-internal-testing/toolcall", "preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=2,  # toolcall sequences are longer than standard data, reduce batch size to avoid OOM
            max_length=512,  # toolcall sequences are longer than standard data, limit length to avoid OOM
            report_to="none",
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
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
        # Tests training with an eval dataset (eval strategy set in the config).
        # Assertions: the first log entry has eval_loss.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

        training_args = DPOConfig(output_dir=self.tmp_dir, eval_strategy="steps", eval_steps=3, report_to="none")
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        trainer.train()

        assert trainer.state.log_history[0]["eval_loss"] is not None

    def test_train_with_multiple_eval_dataset(self):
        # [FUNCTION DOC]
        # Tests training with a dict of two eval datasets (data1, data2).
        # Assertions: third-to-last log has eval_data1_loss, second-to-last has eval_data2_loss.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

        training_args = DPOConfig(output_dir=self.tmp_dir, eval_strategy="steps", eval_steps=3, report_to="none")
        trainer = DPOTrainer(
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
        # Tests train_dataset validation and preprocessing at init.
        # Branches: None raises ValueError that train_dataset is required; a DatasetDict raises TypeError that it must be a Dataset or IterableDataset; a Dataset or streaming IterableDataset is tokenized so its first example has prompt_ids.
        # Inputs: train_dataset_type from parametrize. Outputs: none. Exceptions: AssertionError. Side effects: none.
        streaming = "iterable" in train_dataset_type
        if train_dataset_type == "none":
            train_dataset = None
        else:
            train_dataset = load_dataset(
                "trl-internal-testing/zen", "standard_preference", split="train", streaming=streaming
            )
            if train_dataset_type == "unsupported_dataset_dict":
                # `DatasetDict` is representative of any unsupported type here; not exhaustive
                train_dataset = DatasetDict({"train": train_dataset})

        # Iterable (streaming) datasets have no length, so `max_steps` is required.
        training_args = DPOConfig(output_dir=self.tmp_dir, max_steps=3 if streaming else -1, report_to="none")

        if train_dataset_type == "none":
            with pytest.raises(ValueError, match="`train_dataset` is required"):
                DPOTrainer(
                    model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                    args=training_args,
                    train_dataset=train_dataset,
                )
        elif train_dataset_type == "unsupported_dataset_dict":
            with pytest.raises(TypeError, match="`train_dataset` must be a `Dataset` or `IterableDataset`"):
                DPOTrainer(
                    model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                    args=training_args,
                    train_dataset=train_dataset,
                )
        else:
            trainer = DPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=train_dataset
            )
            assert "prompt_ids" in next(iter(trainer.train_dataset))

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
        # Tests eval_dataset preprocessing at init for None, Dataset, IterableDataset, (Iterable)DatasetDict and dict of datasets.
        # Assertions: None stays None; dict-like results keep keys data1 and data2 and each split is tokenized (has prompt_ids); a single dataset is tokenized.
        # Inputs: eval_dataset_type from parametrize. Outputs: none. Exceptions: AssertionError. Side effects: none.
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        if eval_dataset_type == "none":
            eval_dataset = None
        else:
            streaming = "iterable" in eval_dataset_type
            eval_split = load_dataset(
                "trl-internal-testing/zen", "standard_preference", split="test", streaming=streaming
            )
            if eval_dataset_type in ("dataset", "iterable_dataset"):
                eval_dataset = eval_split
            elif eval_dataset_type in ("dataset_dict", "iterable_dataset_dict"):
                dataset_dict_cls = IterableDatasetDict if streaming else DatasetDict
                eval_dataset = dataset_dict_cls({"data1": eval_split, "data2": eval_split})
            else:  # "dict_of_dataset" or "dict_of_iterable_dataset"
                eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = DPOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = DPOTrainer(
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

    def test_train_with_compute_metrics(self):
        # [FUNCTION DOC]
        # Tests a user compute_metrics callback during evaluation every 3 steps.
        # Assertions: training emits a FutureWarning about return_outputs=True; the callback sees predictions whose last dimension equals the model vocab size (full logits); second-to-last log has eval_my_metric equal to 0.123.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

        def dummy_compute_metrics(eval_pred):
            # [FUNCTION DOC]
            # HELPER: Dummy compute_metrics callback.
            # Behavior: asserts predictions last dimension equals the model vocab size, returns a dict with my_metric 0.123.
            # Inputs: eval_pred (EvalPrediction). Outputs: dict. Exceptions: AssertionError. Side effects: none.
            # The logits are the full-vocabulary logits, as before the fused LM head
            assert eval_pred.predictions.shape[-1] == trainer.model.config.vocab_size
            return {"my_metric": 0.123}

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            eval_strategy="steps",
            eval_steps=3,
            report_to="none",
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=dummy_compute_metrics,
        )

        with pytest.warns(FutureWarning, match="`return_outputs=True`"):
            trainer.train()

        assert trainer.state.log_history[-2]["eval_my_metric"] == 0.123

    def test_logits_metrics_match_full_logits(self):
        # [FUNCTION DOC]
        # Tests that logits/chosen, logits/rejected and mean_token_accuracy metrics (from the fused LM head kernel) match values computed from full model logits.
        # Setup: model in eval mode; one prepared train batch; compute_loss under no_grad records metrics in trainer._metrics eval; full logits recomputed, shifted (drop last position, drop first mask/label column), split in half into chosen/rejected.
        # Assertions: mean chosen and rejected logits over completion tokens match the last metric values (relative 1e-4); token accuracy of chosen argmax vs labels matches mean_token_accuracy.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: appends metrics to trainer state.
        # `logits/*` and `mean_token_accuracy` come from the fused LM head's kernel; check them against the full logits
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        training_args = DPOConfig(output_dir=self.tmp_dir, per_device_train_batch_size=4, report_to="none")
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )
        trainer.model.eval()
        inputs = trainer._prepare_inputs(next(iter(trainer.get_train_dataloader())))

        with torch.no_grad():
            trainer.compute_loss(trainer.model, inputs)
            logits = trainer.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits

        logits = logits[:, :-1].float()
        mask = inputs["completion_mask"][:, 1:].bool()
        chosen_logits, rejected_logits = logits.chunk(2)
        chosen_mask, rejected_mask = mask.chunk(2)
        chosen_labels, _ = inputs["input_ids"][:, 1:].chunk(2)
        metrics = trainer._metrics["eval"]
        assert metrics["logits/chosen"][-1] == pytest.approx(chosen_logits[chosen_mask].mean().item(), rel=1e-4)
        assert metrics["logits/rejected"][-1] == pytest.approx(rejected_logits[rejected_mask].mean().item(), rel=1e-4)
        accuracy = (chosen_logits.argmax(-1) == chosen_labels)[chosen_mask].float().mean().item()
        assert metrics["mean_token_accuracy"][-1] == pytest.approx(accuracy)

    # In practice, this test is the same as `test_train`, since gradient checkpointing is enabled by default in
    # `DPOTrainer`. We keep it as a regression guard: if the default ever changes, we still explicitly test gradient
    # checkpointing, which has caused issues in the past.
    def test_train_with_gradient_checkpointing(self):
        # [FUNCTION DOC]
        # Tests training with gradient_checkpointing=True explicitly (regression guard; default is already on).
        # Assertions: last log has train_loss.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            gradient_checkpointing=True,
            report_to="none",
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
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
        # Tests training with fp16=True on an accelerator.
        # Assertions: last log has train_loss.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            fp16=True,
            report_to="none",
        )
        trainer = DPOTrainer(
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
        # Tests that the trainer adds the dpo and trl tags to model.model_tags.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: none.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            train_dataset=dataset,
        )

        for tag in ["dpo", "trl"]:
            assert tag in trainer.model.model_tags

    @require_peft
    def test_tag_added_peft(self):
        # [FUNCTION DOC]
        # Tests that dpo and trl tags are added to a PEFT-wrapped model too.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: none.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            train_dataset=dataset,
            peft_config=LoraConfig(),
        )

        for tag in ["dpo", "trl"]:
            assert tag in trainer.model.model_tags

    @require_peft
    @require_bitsandbytes
    def test_train_peft_and_quantization(self):
        # [FUNCTION DOC]
        # Tests QLoRA: model id with a 4-bit nf4 BitsAndBytesConfig (double quant, bf16 compute) passed as quantization_config plus a LoraConfig.
        # Assertions: base model is loaded in 4 bit; after training train_loss and mean_token_accuracy logged; lora parameters changed and are bfloat16. Base parameters not checked (bitsandbytes casts biases in place).
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir. Requires PEFT and bitsandbytes.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(output_dir=self.tmp_dir, learning_rate=0.1, report_to="none")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        trainer = DPOTrainer(
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
        assert trainer.state.log_history[-1]["mean_token_accuracy"] is not None

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
        # Same as the QLoRA test but with DoRA (use_dora=True).
        # Extra assertion: the lora_magnitude_vector parameters stay float32 (bf16 would freeze tiny updates), other lora parameters are bfloat16.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(output_dir=self.tmp_dir, learning_rate=0.1, report_to="none")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        trainer = DPOTrainer(
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
        assert trainer.state.log_history[-1]["mean_token_accuracy"] is not None

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

    def test_pad_token_id_synced_with_model_config(self):
        # [FUNCTION DOC]
        # Tests that when the tokenizer has no pad token (tiny Mistral) and falls back to EOS, the model config and generation config pad_token_id are synced to it.
        # Assertions: tokenizer pad id is set and equals both config values.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: none.
        # This model's tokenizer has no pad token, so the trainer falls back to the eos token. The model configs must
        # follow: otherwise `Trainer` realigns them at train time and reports it as a change the user did not make.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-MistralForCausalLM-0.2", args=training_args, train_dataset=dataset
        )

        pad_token_id = trainer.processing_class.pad_token_id
        assert pad_token_id is not None
        assert trainer.model.config.pad_token_id == pad_token_id
        assert trainer.model.generation_config.pad_token_id == pad_token_id


@require_vision
class TestDPOTrainerVLM(TrlTestCase):
    # [CLASS DOC]
    # End-to-end tests for DPOTrainer with vision-language models (requires vision deps). Uses the zen-image conversational preference dataset and max_length None (truncation can drop image tokens). Inherits TrlTestCase.
    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
            pytest.param(
                "trl-internal-testing/tiny-Gemma4ForConditionalGeneration",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.5.0"),
                    reason="Gemma4 models were introduced in transformers-5.5.0",
                ),
            ),
            # "trl-internal-testing/tiny-Idefics2ForConditionalGeneration",  high memory peak, skipped for now
            # "trl-internal-testing/tiny-Idefics3ForConditionalGeneration",  high memory peak, skipped for now
            pytest.param(
                "trl-internal-testing/tiny-Lfm2VlForConditionalGeneration-2.5",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.0.0"),
                    reason="LFM2.5-VL requires transformers>=5.0.0",
                ),
            ),
            "trl-internal-testing/tiny-LlavaForConditionalGeneration",
            "trl-internal-testing/tiny-LlavaNextForConditionalGeneration",
            pytest.param(
                "trl-internal-testing/tiny-MuseGlimmerForConditionalGeneration",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.15.0"),
                    reason="Muse Glimmer was introduced in transformers-5.15.0",
                ),
            ),
            "trl-internal-testing/tiny-Qwen2VLForConditionalGeneration",
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
            # "trl-internal-testing/tiny-SmolVLMForConditionalGeneration", seems not to support bf16 properly
            pytest.param(
                "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration",
                marks=[
                    pytest.mark.skipif(
                        Version(transformers.__version__) < Version("4.57.0"),
                        reason="Qwen3-VL series were introduced in transformers-4.57.0",
                    ),
                ],
            ),
            pytest.param(
                "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration-NoThink",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.2.0"),
                    reason="Qwen3.5 models were introduced in transformers-5.2.0",
                ),
            ),
            pytest.param(
                "trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.2.0"),
                    reason="Qwen3.5 models were introduced in transformers-5.2.0",
                ),
            ),
            pytest.param(
                "trl-internal-testing/tiny-MuseSparkForConditionalGeneration",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.20.0"),
                    reason="Muse Spark Open was introduced in transformers-5.20.0",
                ),
            ),
        ],
    )
    def test_train_vlm(self, model_id):
        # [FUNCTION DOC]
        # Train check across tiny VLMs (Gemma3/4, LFM2.5-VL, LLaVA, LLaVA-Next, Muse Glimmer, Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5, Qwen3.5-MoE, Muse Spark), batch size 2, some skipped on older transformers.
        # Assertions: train_loss logged; all parameters changed, except: for LLaVA models the last vision encoder layer and post_layernorm must stay unchanged; for Muse Spark the audio_tower must stay unchanged.
        # Inputs: model_id from parametrize. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            max_length=None,  # for VLMs, truncating can remove image tokens, leading to errors
            per_device_train_batch_size=2,  # VLM training is memory intensive, reduce batch size to avoid OOM
            report_to="none",
        )
        trainer = DPOTrainer(model=model_id, args=training_args, train_dataset=dataset)

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            # LLaVA & LLaVA-Next: vision_feature_layer=-2 leaves the last encoder layer (layers.1) and
            # post_layernorm (pooler-only path) without gradient by design. Assert they stay frozen — if they
            # ever start training, the feature-selection plumbing has likely regressed.
            if model_id in (
                "trl-internal-testing/tiny-LlavaForConditionalGeneration",
                "trl-internal-testing/tiny-LlavaNextForConditionalGeneration",
            ) and ("encoder.layers.1" in n or "post_layernorm" in n):
                assert torch.equal(param, new_param), f"Param {n} expected frozen by LLaVA design, but changed"
            # Muse Spark Open: no dataset here feeds `audio_tower`, so it gets no gradient. Assert it stays frozen.
            elif model_id == "trl-internal-testing/tiny-MuseSparkForConditionalGeneration" and "audio_tower" in n:
                assert torch.equal(param, new_param), f"Param {n} expected frozen, but changed"
            else:
                assert not torch.equal(param, new_param), f"Param {n} is not updated"

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
        ],
    )
    @pytest.mark.xfail(
        Version(transformers.__version__) < Version("4.57.0"),
        reason="Mixing text-only and image+text examples is only supported in transformers >= 4.57.0",
        strict=False,
    )
    def test_train_vlm_multi_image(self, model_id):
        # [FUNCTION DOC]
        # Tests training Qwen2.5-VL on multi-image preference data with batch size 1 (xfail before transformers 4.57.0).
        # Assertions: train_loss logged.
        # Inputs: model_id from parametrize. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen-multi-image", "conversational_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            max_length=None,  # for VLMs, truncating can remove image tokens, leading to errors
            per_device_train_batch_size=1,  # VLM training is memory intensive, reduce batch size to avoid OOM
            report_to="none",
        )
        trainer = DPOTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Param {n} is not updated"

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
        ],
    )
    @pytest.mark.parametrize(
        "dataset_config",
        ["conversational_preference", "standard_preference"],
    )
    def test_train_vlm_text_only_data(self, model_id, dataset_config):
        # [FUNCTION DOC]
        # Tests training a VLM on text-only data (conversational and standard).
        # Assertions: ddp_find_unused_parameters was enabled (vision tower gets no gradient); train_loss logged; parameters under model.visual unchanged, others changed.
        # Inputs: model_id, dataset_config from parametrize. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", dataset_config, split="train")

        training_args = DPOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = DPOTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
        )

        # The vision tower gets no gradients, so DDP must search for unused parameters
        assert trainer.args.ddp_find_unused_parameters

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n.startswith("model.visual"):
                torch.testing.assert_close(param, new_param, rtol=1e-12, atol=1e-12, msg=f"Param {n} is updated")
            else:
                assert not torch.equal(param, new_param), f"Param {n} is not updated"

    def test_train_vlm_with_max_length(self):
        # [FUNCTION DOC]
        # Regression test (issue 5283): with max_length 37, mm_token_type_ids must be truncated along with input_ids (one completion token cut, image tokens kept).
        # Assertions: train_loss logged.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: writes to self.tmp_dir.
        # Regression test for #5283: mm_token_type_ids must be truncated alongside input_ids when max_length is set,
        # otherwise a shape mismatch crashes the model forward pass.
        # max_length=37 truncates 1 completion token (total_len=38) while keeping all image tokens (prompt_len=34) safe.
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_preference", split="train")
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            max_length=37,  # total_len=38, prompt_len=34 — truncates completion, not image tokens
            per_device_train_batch_size=2,
            report_to="none",
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()
        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_train_vlm_keep_end_raises(self):
        # [FUNCTION DOC]
        # Regression test (issue 5285): truncation_mode keep_end with a VLM must fail at init.
        # Assertions: building the config warns that keep_end is deprecated; building the trainer raises ValueError that keep_end is not supported for VLMs.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: none.
        # Regression test for #5285: keep_end with a VLM must raise at init time, not silently corrupt training.
        # Image tokens live at the start of the sequence (in the prompt); keep_end would drop them.
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_preference", split="train")
        with pytest.warns(FutureWarning, match="keep_end.*deprecated"):
            training_args = DPOConfig(
                output_dir=self.tmp_dir,
                max_length=32,
                truncation_mode="keep_end",
                report_to="none",
            )
        with pytest.raises(ValueError, match="truncation_mode='keep_end' is not supported for vision-language models"):
            DPOTrainer(
                model="trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
                args=training_args,
                train_dataset=dataset,
            )

    def test_vision_dataset_with_text_model_raises(self):
        # [FUNCTION DOC]
        # Tests that an image dataset with a text-only model raises ValueError about needing a vision-language model.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: none.
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_preference", split="train")
        training_args = DPOConfig(output_dir=self.tmp_dir, report_to="none")
        with pytest.raises(ValueError, match="vision-related.*vision-language model"):
            DPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=dataset,
            )

    def test_precompute_ref_log_probs_raises_for_vision(self):
        # [FUNCTION DOC]
        # Tests that precompute_ref_log_probs with a vision dataset raises ValueError (not supported for vision datasets).
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: none.
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_preference", split="train")
        training_args = DPOConfig(output_dir=self.tmp_dir, report_to="none", precompute_ref_log_probs=True)
        with pytest.raises(ValueError, match="precompute_ref_log_probs.*not supported for vision datasets"):
            DPOTrainer(
                model="trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
                args=training_args,
                train_dataset=dataset,
            )

    def test_pad_token_synced_with_model_config_vision(self):
        # [FUNCTION DOC]
        # Tests that a requested pad_token (deprecated config option, warns FutureWarning) reaches the processor tokenizer, the text config and the generation config for a vision dataset.
        # Assertions: all three pad_token_id values equal the id of the fim_pad token.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: none.
        # A vision dataset takes the other collator branch, which used to skip the pad token handling entirely, so
        # the requested pad token reached neither the tokenizer nor the model configs.
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_preference", split="train")

        with pytest.warns(FutureWarning, match="`pad_token` is deprecated"):
            training_args = DPOConfig(
                output_dir=self.tmp_dir,
                max_length=None,  # for VLMs, truncating can remove image tokens, leading to errors
                pad_token="<|fim_pad|>",
                report_to="none",
            )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
            args=training_args,
            train_dataset=dataset,
        )

        pad_token_id = trainer.processing_class.tokenizer.convert_tokens_to_ids("<|fim_pad|>")
        assert trainer.processing_class.tokenizer.pad_token_id == pad_token_id
        assert trainer.model.config.get_text_config().pad_token_id == pad_token_id
        assert trainer.model.generation_config.pad_token_id == pad_token_id
