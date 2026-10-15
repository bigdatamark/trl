# [MODULE DOC]
# Tests for trl SFTTrainer, SFTConfig and DataCollatorForLanguageModeling (trl.trainer.sft_trainer).
# Main contents: collator unit tests (padding, padding-free, packing, position ids) and a large TestSFTTrainer
# class covering model loading, dataset formats, losses (nll, dft), MoE auxiliary loss, PEFT variants,
# packing, completion/assistant-only loss, chat templates, tool calls, kernels, and VLM tests.
# Uses tiny models from trl-internal-testing and zen datasets. PEFT symbols are imported only if peft is
# installed; optional-dependency tests are gated by require_* decorators.
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
import transformers
from datasets import Dataset, DatasetDict, IterableDatasetDict, load_dataset
from packaging.version import Version
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from transformers.testing_utils import torch_device
from transformers.trainer_pt_utils import LabelSmoother
from transformers.utils import is_peft_available

from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from .testing_utils import (
    TrlTestCase,
    ignore_warnings,
    is_ampere_or_newer,
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
    from peft import (
        LoraConfig,
        PrefixTuningConfig,
        PromptEncoderConfig,
        PromptTuningConfig,
        TaskType,
        get_peft_model,
    )


class TestDataCollatorForLanguageModeling(TrlTestCase):
    # [CLASS DOC]
    # Tests DataCollatorForLanguageModeling. In padded mode it right-pads input_ids with pad_token_id, returns
    # attention_mask and labels (given labels or input_ids, with -100 on padding) and optionally position_ids. In
    # padding-free mode it concatenates all examples into one row and returns position_ids restarting at each
    # document, labels with -100 on each document's first token, cumulative sequence lengths (cu_seq_lens_q/k),
    # max lengths and seq_idx. Also tests the static helper get_position_ids_from_packed_seq_lengths.
    def test_basic_padding(self):
        # [FUNCTION DOC]
        # Asserts padded input_ids, attention_mask and labels with -100 on padding, and exactly these three keys.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Test basic padding."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)
        examples = [{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}, {"input_ids": [4, 5], "labels": [4, 5]}]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3], [4, 5, -100]]))

    def test_without_labels(self):
        # [FUNCTION DOC]
        # Asserts that without a labels key, labels default to input_ids with -100 on padding.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """When no labels are provided, they default to the input IDs (padding excluded)."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)
        examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3], [4, 5, -100]]))

    def test_provided_labels_not_reconstructed_from_masks(self):
        # [FUNCTION DOC]
        # Asserts that provided labels are used as is: completion_mask and assistant_masks columns are ignored and do not
        # add -100 entries; output keys are only input_ids, attention_mask, labels.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Provided labels are used as is, never rebuilt from the mask columns. Here the labels match the input IDs
        (every token trainable); the masks would introduce -100 if they were consumed, so the labels staying equal to
        the input IDs proves the masks are ignored."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)
        examples = [
            {"input_ids": [1, 2, 3], "labels": [1, 2, 3], "completion_mask": [0, 0, 1], "assistant_masks": [0, 0, 1]},
            {"input_ids": [4, 5], "labels": [4, 5], "completion_mask": [0, 1], "assistant_masks": [0, 1]},
        ]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3], [4, 5, -100]]))

    def test_return_position_ids(self):
        # [FUNCTION DOC]
        # Asserts that with return_position_ids, padded output also has position_ids counting 0..n-1 per row with 0 on
        # padding. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Padded mode with return_position_ids: position IDs are returned alongside the attention mask."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0, return_position_ids=True)
        examples = [{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}, {"input_ids": [4, 5], "labels": [4, 5]}]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "attention_mask", "position_ids", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 1, 2], [0, 1, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3], [4, 5, -100]]))

    def test_return_position_ids_packed(self):
        # [FUNCTION DOC]
        # Asserts that with seq_lengths in padded mode, position_ids restart at each packed document boundary.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Padded mode with return_position_ids on packed examples: position IDs reset at document boundaries."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0, return_position_ids=True)
        examples = [{"input_ids": [1, 2, 3, 4, 5], "seq_lengths": [3, 2]}, {"input_ids": [6, 7], "seq_lengths": [2]}]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "attention_mask", "position_ids", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3, 4, 5], [6, 7, 0, 0, 0]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 1, 2, 0, 1], [0, 1, 0, 0, 0]]))

    def test_padding_free_mode(self):
        # [FUNCTION DOC]
        # Asserts padding-free output: one concatenated row, restarting position_ids, labels with -100 at document starts,
        # cu_seq_lens_q of 0, 3, 5 (int32), max_length_q 3, and the full padding-free key set.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Test padding-free mode where sequences are concatenated."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=True)
        examples = [{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}, {"input_ids": [4, 5], "labels": [4, 5]}]

        result = collator(examples)

        assert set(result.keys()) == {
            "input_ids",
            "position_ids",
            "labels",
            "cu_seq_lens_q",
            "cu_seq_lens_k",
            "max_length_q",
            "max_length_k",
            "seq_idx",
        }
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3, 4, 5]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 1, 2, 0, 1]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[-100, 2, 3, -100, 5]]))
        torch.testing.assert_close(result["cu_seq_lens_q"], torch.tensor([0, 3, 5], dtype=torch.int32))
        assert result["max_length_q"] == 3

    def test_padding_free_without_labels(self):
        # [FUNCTION DOC]
        # Same expectations as the padding-free test when examples have no labels.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Padding-free mode without labels: labels default to the input IDs (document starts masked)."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=True)
        examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]

        result = collator(examples)

        assert set(result.keys()) == {
            "input_ids",
            "position_ids",
            "labels",
            "cu_seq_lens_q",
            "cu_seq_lens_k",
            "max_length_q",
            "max_length_k",
            "seq_idx",
        }
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3, 4, 5]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 1, 2, 0, 1]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[-100, 2, 3, -100, 5]]))
        torch.testing.assert_close(result["cu_seq_lens_q"], torch.tensor([0, 3, 5], dtype=torch.int32))
        assert result["max_length_q"] == 3

    def test_packing(self):
        # [FUNCTION DOC]
        # Asserts padding-free collation of packed examples with seq_lengths splits documents correctly: position_ids
        # restart per document, labels mask each document start, cu_seq_lens_q 0, 3, 6, 10, 11, max_length_q 4, and
        # seq_idx enumerates the four documents. No attention_mask is returned.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Test that when using packing with position_ids, attention_mask is dropped with fa2."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=True)

        # Simulate packed sequences with position_ids that restart (typical of BFD packing)
        examples = [
            {"input_ids": [1, 2, 3, 4, 5, 6], "seq_lengths": [3, 3], "labels": [1, 2, 3, 4, 5, 6]},
            {"input_ids": [7, 8, 9, 10, 11], "seq_lengths": [4, 1], "labels": [7, 8, 9, 10, 11]},
        ]

        result = collator(examples)

        assert set(result.keys()) == {
            "input_ids",
            "position_ids",
            "labels",
            "cu_seq_lens_q",
            "cu_seq_lens_k",
            "max_length_q",
            "max_length_k",
            "seq_idx",
        }
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[-100, 2, 3, -100, 5, 6, -100, 8, 9, 10, -100]]))
        torch.testing.assert_close(result["cu_seq_lens_q"], torch.tensor([0, 3, 6, 10, 11], dtype=torch.int32))
        assert result["max_length_q"] == 4
        torch.testing.assert_close(
            result["seq_idx"], torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3]], dtype=torch.int32)
        )

    def test_pad_to_multiple_of(self):
        # [FUNCTION DOC]
        # Asserts padded mode with pad_to_multiple_of 4 pads rows to length 4 with matching mask and -100 labels.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Test padding to multiple of specified value."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0, pad_to_multiple_of=4)
        examples = [{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}, {"input_ids": [4, 5], "labels": [4, 5]}]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3, -100], [4, 5, -100, -100]]))

    def test_pad_to_multiple_of_and_padding_free(self):
        # [FUNCTION DOC]
        # Asserts padding-free mode with pad_to_multiple_of 4 pads the single row to length 8; the padding positions get
        # position 0 and label -100 and each padding token appears as its own one-token sequence in cu_seq_lens_q
        # (0, 3, 5, 6, 7, 8); max_length_q 3. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Test padding to multiple of specified value."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=True, pad_to_multiple_of=4)
        examples = [{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}, {"input_ids": [4, 5], "labels": [4, 5]}]

        result = collator(examples)

        assert set(result.keys()) == {
            "input_ids",
            "position_ids",
            "labels",
            "cu_seq_lens_q",
            "cu_seq_lens_k",
            "max_length_q",
            "max_length_k",
            "seq_idx",
        }
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 1, 2, 0, 1, 0, 0, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[-100, 2, 3, -100, 5, -100, -100, -100]]))
        torch.testing.assert_close(result["cu_seq_lens_q"], torch.tensor([0, 3, 5, 6, 7, 8], dtype=torch.int32))
        assert result["max_length_q"] == 3

    def test_custom_position_ids_but_no_padding_free(self):
        # [FUNCTION DOC]
        # Asserts seq_lengths are ignored in padded mode without return_position_ids: only input_ids, attention_mask and
        # labels are returned. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Test that custom position_ids are ignored if padding_free is False."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)
        examples = [
            {"input_ids": [1, 2, 3], "seq_lengths": [1, 2], "labels": [1, 2, 3]},
            {"input_ids": [4, 5], "seq_lengths": [2], "labels": [4, 5]},
        ]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3], [4, 5, -100]]))

    def test_single_example(self):
        # [FUNCTION DOC]
        # Asserts a single example is returned unpadded with all-ones mask and labels equal to input_ids.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Test collator with a single example."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)
        examples = [{"input_ids": [1, 2, 3, 4], "labels": [1, 2, 3, 4]}]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3, 4]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1, 1]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3, 4]]))

    def test_different_pad_token_id(self):
        # [FUNCTION DOC]
        # Asserts padding uses pad_token_id 999 while labels still use -100.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Test with different pad token ID."""
        collator = DataCollatorForLanguageModeling(pad_token_id=999)
        examples = [{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}, {"input_ids": [4, 5], "labels": [4, 5]}]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 999]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3], [4, 5, -100]]))

    def test_single_example_single_doc(self):
        # [FUNCTION DOC]
        # Asserts get_position_ids_from_packed_seq_lengths on one example with one document of length 5 returns a list
        # with one tensor 0..4. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        batch_seq_lengths = [[5]]
        result = DataCollatorForLanguageModeling.get_position_ids_from_packed_seq_lengths(batch_seq_lengths)
        assert len(result) == 1
        assert torch.equal(result[0], torch.arange(5))

    def test_single_example_multiple_docs(self):
        # [FUNCTION DOC]
        # Asserts lengths 3 and 2 in one example give positions 0, 1, 2, 0, 1. Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: none.
        batch_seq_lengths = [[3, 2]]
        result = DataCollatorForLanguageModeling.get_position_ids_from_packed_seq_lengths(batch_seq_lengths)
        assert len(result) == 1
        # First sequence: 0, 1, 2; second sequence: 0, 1
        assert torch.equal(result[0], torch.tensor([0, 1, 2, 0, 1]))

    def test_multiple_examples(self):
        # [FUNCTION DOC]
        # Asserts two examples (lengths 2+2 and 3) give two tensors 0, 1, 0, 1 and 0, 1, 2.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        batch_seq_lengths = [[2, 2], [3]]
        result = DataCollatorForLanguageModeling.get_position_ids_from_packed_seq_lengths(batch_seq_lengths)
        assert len(result) == 2
        assert torch.equal(result[0], torch.tensor([0, 1, 0, 1]))
        assert torch.equal(result[1], torch.arange(3))


class TestSFTTrainer(TrlTestCase):
    # [CLASS DOC]
    # End-to-end SFTTrainer tests. Most use self.tmp_dir as output_dir and report_to none with the tiny Qwen2 model
    # and zen datasets. Common "standard train check": snapshot named parameters, train, assert the last log entry
    # has train_loss and every parameter changed. Common "PEFT train check": base-model parameters (prefixed
    # base_model.model.) unchanged and other non-base_layer parameters changed. Inherits TrlTestCase.
    def test_init_with_training_arguments(self):
        # [FUNCTION DOC]
        # Asserts the trainer accepts a plain transformers TrainingArguments instead of SFTConfig.
        # Inputs/Outputs: none. Exceptions: construction errors. Side effects: none.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")
        args = TrainingArguments(output_dir=self.tmp_dir, report_to="none")
        SFTTrainer(model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=args, train_dataset=dataset)

    def test_init_auto_processing_class_uses_model_revision(self):
        # [FUNCTION DOC]
        # With model_init_kwargs revision pinned, asserts the auto-loaded tokenizer comes from that revision (its chat
        # template is 2558 characters long, unlike main). Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: downloads a specific revision.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=SFTConfig(
                output_dir=self.tmp_dir,
                report_to="none",
                model_init_kwargs={"revision": "8913f5819566"},
            ),
            train_dataset=dataset,
        )
        # This revision's chat template is 2558 chars; the one on `main` is 2507. Comparing the length is enough to
        # catch the tokenizer being loaded from the default branch instead of the pinned revision.
        assert len(trainer.processing_class.chat_template) == 2558

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Cohere2ForCausalLM",
            "trl-internal-testing/tiny-FalconMambaForCausalLM",
            pytest.param(
                "trl-internal-testing/tiny-Glm4MoeForCausalLM",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.0.0"),
                    reason="GLM4 tokenizer requires transformers>=5.0.0",
                ),
            ),
            "trl-internal-testing/tiny-GptOssForCausalLM",
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            "trl-internal-testing/tiny-Qwen2ForCausalLM-R1-Distill",
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
        # Runs the standard train check across many causal LMs (Cohere2, FalconMamba, GLM4 MoE, GPT-OSS, Qwen2,
        # R1-Distill, Qwen3 MoE, Nemotron-H, Olmo3, LFM2/2.5; some skipped on old transformers); if aux_loss_enabled also
        # asserts aux_loss is logged. Inputs: model_id (str, parametrized). Outputs: none. Exceptions: AssertionError.
        # Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(model=model_id, args=training_args, train_dataset=dataset)

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

    @pytest.mark.parametrize(
        "config_name",
        [
            "standard_language_modeling",
            "conversational_language_modeling",
            "standard_prompt_completion",
            "conversational_prompt_completion",
        ],
    )
    def test_train_dataset_format(self, config_name):
        # [FUNCTION DOC]
        # Runs the standard train check on standard/conversational language-modeling and prompt-completion datasets.
        # Inputs: config_name (str, parametrized). Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", config_name, split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_trust_remote_code(self):
        # [FUNCTION DOC]
        # Asserts a remote-code model id raises ValueError mentioning custom code without trust_remote_code and loads as
        # RemoteForCausalLM with it. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: may download code.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")
        model_id = "trl-internal-testing/tiny-RemoteForCausalLM"

        with pytest.raises(ValueError, match="custom code"):
            SFTTrainer(
                model=model_id,
                args=SFTConfig(output_dir=self.tmp_dir, report_to="none"),
                train_dataset=dataset,
            )

        trainer = SFTTrainer(
            model=model_id,
            args=SFTConfig(output_dir=self.tmp_dir, report_to="none", trust_remote_code=True),
            train_dataset=dataset,
        )
        assert type(trainer.model).__name__ == "RemoteForCausalLM"

    # Special case for harmony
    def test_train_gpt_oss(self):
        # [FUNCTION DOC]
        # Runs the standard train check on tiny GPT-OSS with the harmony language-modeling dataset.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/harmony", "language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
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
        # Runs the standard train check with a float32 model instance. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: writes tmp_dir.
        model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            dtype="float32",
        )

        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset)

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_dft_loss_matches_reference(self):
        # [FUNCTION DOC]
        # With loss_type dft, calls compute_loss on random 2x8 ids whose first 3 label positions are -100 and compares it
        # (tolerance 1e-4) with a reference: shift logits and labels by one, take per-token log-probabilities of the
        # labels, weight each negative log-probability by the detached token probability, average over unmasked tokens.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")
        training_args = SFTConfig(output_dir=self.tmp_dir, loss_type="dft", report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )
        input_ids = torch.randint(0, trainer.model.config.vocab_size, (2, 8), device=trainer.model.device)
        labels = input_ids.masked_fill(torch.arange(8, device=input_ids.device) < 3, -100)
        inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": labels}

        loss = trainer.compute_loss(trainer.model, dict(inputs))

        # DFT (https://huggingface.co/papers/2508.05629) weighs each token's NLL by its detached probability
        with torch.no_grad():
            logits = trainer.model(input_ids=input_ids, attention_mask=inputs["attention_mask"]).logits[:, :-1]
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        logprobs = logits.float().log_softmax(-1).gather(-1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        expected = (-logprobs.exp() * logprobs)[mask].sum() / mask.sum()
        torch.testing.assert_close(loss, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("num_items_in_batch", [None, 0])
    def test_dft_loss_without_trainable_tokens(self, num_items_in_batch):
        # [FUNCTION DOC]
        # With dft loss and all labels -100, asserts compute_loss returns exactly 0.0 and backward yields only finite
        # gradients, for num_items_in_batch None and 0. Inputs: num_items_in_batch (parametrized). Outputs: none.
        # Exceptions: AssertionError. Side effects: populates parameter gradients.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")
        training_args = SFTConfig(output_dir=self.tmp_dir, loss_type="dft", report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )
        input_ids = torch.randint(0, trainer.model.config.vocab_size, (2, 8), device=trainer.model.device)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": torch.full_like(input_ids, -100),
        }

        loss = trainer.compute_loss(trainer.model, inputs, num_items_in_batch=num_items_in_batch)
        loss.backward()

        assert loss.item() == 0.0
        assert all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None)

    def test_train_dft_loss(self):
        # [FUNCTION DOC]
        # Runs the standard train check with dft loss, lr 0.1 and eval every 3 steps.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling")

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            loss_type="dft",
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
            eval_strategy="steps",
            eval_steps=3,
        )
        trainer = SFTTrainer(
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

    def test_train_with_formatting_func(self):
        # [FUNCTION DOC]
        # Runs the standard train check on a preference dataset converted to text by a formatting_func.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        # Dummy formatting function
        def formatting_prompts_func(example):
            # [FUNCTION DOC]
            # HELPER: Returns a two-line string with headers Chosen and Rejected followed by the example's chosen and
            # rejected texts. Inputs: example (dict). Outputs: str. Exceptions: KeyError if fields missing. Side effects: none.
            chosen, rejected = example["chosen"], example["rejected"]
            return f"### Chosen: {chosen}\n### Rejected: {rejected}"

        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            formatting_func=formatting_prompts_func,
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
        # Trains with model_init_kwargs dtype float16 (lr 0.1); asserts every non-layernorm parameter is float16 and
        # changed. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            model_init_kwargs={"dtype": torch.float16},
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = SFTTrainer(
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
        # Builds two LoRA trainers after different global torch seeds and asserts identical non-empty lora_A weights.
        # Requires peft. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: changes the global seed.
        # Two trainers with the same seed start from the same adapter weights
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")
        adapters = []
        for global_seed in range(2):
            torch.manual_seed(global_seed)  # a different global RNG state, as in two separate runs
            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=SFTConfig(output_dir=self.tmp_dir, report_to="none"),
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
        # Runs the PEFT train check with a model id and default LoraConfig. Requires peft. Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: writes tmp_dir.
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")

        trainer = SFTTrainer(
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
        "peft_type",
        [
            "prompt_tuning",
            "prefix_tuning",
            "prompt_encoder",
        ],
    )
    @require_peft
    def test_train_with_peft_config_prompt_tuning(self, peft_type):
        # [FUNCTION DOC]
        # For prompt tuning, prefix tuning (xfail on peft 0.17.1 or older) and prompt encoder (p-tuning), each with 4
        # virtual tokens, bf16 and gradient checkpointing off: trains and asserts base parameters (prefixed
        # base_model.) are unchanged and all other parameters changed. Requires peft.
        # Inputs: peft_type (str, parametrized). Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.{n}" for n, _ in model.named_parameters()]

        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        # Initialize the trainer, p-tuning doesn't support gradient checkpointing
        training_args = SFTConfig(bf16=False, output_dir=self.tmp_dir, report_to="none", gradient_checkpointing=False)
        if peft_type == "prompt_tuning":
            peft_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=4,
                tokenizer_name_or_path="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            )
        elif peft_type == "prefix_tuning":
            if Version(peft.__version__) <= Version("0.17.1"):
                pytest.xfail(
                    "Prefix tuning with device_map='auto' is broken in peft 0.17.1 and below. See "
                    "https://github.com/huggingface/peft/issues/2821"
                )
            peft_config = PrefixTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=4,
            )
        elif peft_type == "prompt_encoder":
            peft_config = PromptEncoderConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=4,
                encoder_hidden_size=model.config.hidden_size,  # This will be overwritten below
            )
        trainer = SFTTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
            peft_config=peft_config,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            else:  # We expect the peft params to be different
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_peft_target_parameters
    def test_train_moe_with_peft_config(self):
        # [FUNCTION DOC]
        # Runs the PEFT train check on tiny GPT-OSS with LoRA target_parameters on expert down_proj and gate_up_proj.
        # Requires peft target_parameters support. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: writes tmp_dir.
        model_id = "trl-internal-testing/tiny-GptOssForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")

        trainer = SFTTrainer(
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

    @pytest.mark.parametrize(
        "model_id, expect_coef, expect_aux_loss",
        [
            # MoE whose forward returns an auxiliary loss: the architecture's own coefficient is applied
            ("trl-internal-testing/tiny-Qwen3MoeForCausalLM", 0.001, True),
            # MoE that balances its experts with a router bias: it declares no coefficient, so the term stays off
            ("trl-internal-testing/tiny-DeepseekV3ForCausalLM", 0.0, False),
            # Dense model: no coefficient to inherit
            ("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", 0.0, False),
        ],
    )
    def test_router_aux_loss_coef_defaults_to_the_architecture(self, model_id, expect_coef, expect_aux_loss):
        # [FUNCTION DOC]
        # Asserts router_aux_loss_coef and aux_loss_enabled default from the architecture: Qwen3 MoE gives 0.001 and
        # enabled; DeepseekV3 (router-bias balancing) and dense Qwen2 give 0.0 and disabled.
        # Inputs: model_id, expect_coef, expect_aux_loss (parametrized). Outputs: none. Exceptions: AssertionError.
        # Side effects: none.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train[:2]")
        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(model=model_id, args=training_args, train_dataset=dataset)

        assert trainer.router_aux_loss_coef == expect_coef
        assert trainer.aux_loss_enabled == expect_aux_loss

    def test_router_aux_loss_coef_fails_without_router_logits(self):
        # [FUNCTION DOC]
        # Asserts setting router_aux_loss_coef 0.5 on a dense model raises ValueError (not a Mixture-of-Experts model).
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train[:2]")
        training_args = SFTConfig(output_dir=self.tmp_dir, router_aux_loss_coef=0.5, report_to="none")

        # Dense model: no `output_router_logits` on its config, so there is nothing to compute the term from
        with pytest.raises(ValueError, match="not a Mixture-of-Experts model"):
            SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
            )

    def test_router_aux_loss_coef_explicit_value_overrides_the_architecture(self):
        # [FUNCTION DOC]
        # Asserts an explicit router_aux_loss_coef 0.5 on Qwen3 MoE is used and aux loss is enabled.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train[:2]")
        training_args = SFTConfig(output_dir=self.tmp_dir, router_aux_loss_coef=0.5, report_to="none")

        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen3MoeForCausalLM", args=training_args, train_dataset=dataset
        )

        assert trainer.router_aux_loss_coef == 0.5
        assert trainer.aux_loss_enabled

    def test_aux_loss_does_not_scale_with_gradient_accumulation(self):
        # [FUNCTION DOC]
        # Uses four copies of one example. Measures the router weight update of the first Qwen3 MoE layer after one SGD
        # step (lr 1.0, no clipping) with and without the auxiliary loss (coef 1.0 vs 0.0); the difference isolates the
        # aux-loss contribution. Asserts that contribution with batch 4 and no accumulation matches batch 2 with 2
        # accumulation steps (rtol 1e-3, atol 1e-6). Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: builds and trains four trainers in tmp_dir.
        # Four copies of one example: every micro-batch has the same aux loss as the full batch
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train").select([0] * 4)

        def router_update(batch_size, gradient_accumulation_steps, router_aux_loss_coef):
            # [FUNCTION DOC]
            # HELPER: Builds a trainer with the given per-device batch size, gradient accumulation steps and
            # router_aux_loss_coef (max_steps 1, plain SGD, lr 1.0, max_grad_norm 0), trains one step and returns the change
            # in the first layer's router gate weight. Inputs: batch_size, gradient_accumulation_steps (int),
            # router_aux_loss_coef (float). Outputs: tensor. Exceptions: none expected. Side effects: writes tmp_dir.
            training_args = SFTConfig(
                output_dir=self.tmp_dir,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_steps=1,
                optim="sgd",
                learning_rate=1.0,  # with plain SGD and no clipping, the update is the gradient
                max_grad_norm=0.0,
                router_aux_loss_coef=router_aux_loss_coef,
                report_to="none",
            )
            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen3MoeForCausalLM", args=training_args, train_dataset=dataset
            )
            router = trainer.model.model.layers[0].mlp.gate.weight
            previous = router.detach().clone()
            trainer.train()
            return router.detach() - previous

        # The update is linear in the gradient, so the difference is the aux loss's own contribution
        aux_update = router_update(4, 1, 1.0) - router_update(4, 1, 0.0)
        accumulated_aux_update = router_update(2, 2, 1.0) - router_update(2, 2, 0.0)
        torch.testing.assert_close(accumulated_aux_update, aux_update, rtol=1e-3, atol=1e-6)

    @require_peft
    def test_train_peft_model(self):
        # [FUNCTION DOC]
        # Runs the PEFT train check with a model already wrapped by get_peft_model (default LoRA). Requires peft.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")

        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        lora_config = LoraConfig()
        model = get_peft_model(model, lora_config)

        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset)

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

    # In practice, this test is the same as `test_train_dense_with_peft_config_lora`, since gradient checkpointing is
    # enabled by default in `SFTTrainer`. We keep it as a regression guard: if the default ever changes, we still
    # explicitly test PEFT + gradient checkpointing, which has caused issues in the past.
    @require_peft
    def test_train_with_peft_config_and_gradient_checkpointing(self):
        # [FUNCTION DOC]
        # Runs the PEFT train check with LoRA and gradient_checkpointing explicitly True (regression guard).
        # Requires peft. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, gradient_checkpointing=True, report_to="none")

        trainer = SFTTrainer(
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
        # Runs the PEFT train check with LoRA, gradient checkpointing and the given use_reentrant kwarg. Requires peft.
        # Inputs: use_reentrant (bool, parametrized). Outputs: none. Exceptions: AssertionError.
        # Side effects: writes tmp_dir.
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": use_reentrant},
            report_to="none",
        )

        trainer = SFTTrainer(
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

    @require_liger_kernel
    def test_train_with_liger(self):
        # [FUNCTION DOC]
        # Requires liger-kernel. Asserts use_liger_kernel True emits a deprecation FutureWarning and disables
        # fused_linear_cross_entropy in liger_kernel_config, then runs the standard train check.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, use_liger_kernel=True, report_to="none")
        with pytest.warns(FutureWarning, match="`use_liger_kernel=True` is deprecated"):
            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
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

    def test_train_with_non_chatml_conversational_data(self):
        # [FUNCTION DOC]
        # Renames conversational messages to a conversations column with from/value keys and runs the standard train
        # check. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

        # Rename role/content to from/value to ensure SFT works with non-chatML conversational data
        def rename_fields(example: list[dict]):
            # [FUNCTION DOC]
            # HELPER: Converts an example's messages (role/content) into a conversations list with from/value keys.
            # Inputs: example (dict). Outputs: dict. Exceptions: KeyError if fields missing. Side effects: none.
            return {"conversations": [{"from": m["role"], "value": m["content"]} for m in example["messages"]]}

        dataset = dataset.map(rename_fields, remove_columns="messages")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_with_pretokenized_data(self):
        # [FUNCTION DOC]
        # Tokenizes the text column (dropping it) and runs the standard train check on the pre-tokenized dataset.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        def tokenize_example(example):
            # [FUNCTION DOC]
            # HELPER: Returns the tokenizer output (input_ids, attention_mask) for the example text.
            # Inputs: example (dict). Outputs: tokenizer encoding. Exceptions: none. Side effects: none.
            return tokenizer(example["text"])

        # Apply tokenization
        tokenized_dataset = dataset.map(tokenize_example, remove_columns=["text"])

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(model=model_id, args=training_args, train_dataset=tokenized_dataset)

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_dataset_with_transform_requires_skip_prepare_dataset(self):
        # [FUNCTION DOC]
        # Asserts a dataset with an on-the-fly with_transform raises ValueError advising skip_prepare_dataset and a
        # trainer-ready dataset. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        dataset = Dataset.from_dict({"text": ["hello world"]})

        def add_suffix(batch):
            # [FUNCTION DOC]
            # HELPER: Batch transform appending an AUG marker to each text. Inputs: batch (dict). Outputs: the mutated batch.
            # Exceptions: none. Side effects: mutates the batch.
            batch["text"] = [text + " <AUG>" for text in batch["text"]]
            return batch

        dataset = dataset.with_transform(add_suffix)
        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")

        with pytest.raises(
            ValueError,
            match=r"Dataset\.with_transform\(\).*skip_prepare_dataset.*trainer-ready",
        ):
            SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
            )

    def test_padding_free_without_packing_and_max_length_raises(self):
        # [FUNCTION DOC]
        # Asserts padding_free True with max_length 16 and no packing raises ValueError (max_length is not enforced).
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train[:2]")
        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            max_length=16,
            padding_free=True,
            report_to="none",
        )

        with pytest.raises(ValueError, match="`max_length` is not enforced"):
            SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
            )

    def test_train_with_iterable_dataset(self):
        # [FUNCTION DOC]
        # Runs the standard train check on a streaming dataset with max_steps 3.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train", streaming=True)

        training_args = SFTConfig(output_dir=self.tmp_dir, max_steps=3, report_to="none")
        trainer = SFTTrainer(
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
        # Requires kernels and an Ampere-or-newer GPU or XPU. Runs the standard train check with padding_free, no
        # max_length, bf16 and the kernels-community flash-attn2 attention implementation.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            padding_free=True,
            max_length=None,  # padding-free without packing doesn't enforce max_length
            model_init_kwargs={"attn_implementation": "kernels-community/flash-attn2"},
            bf16=True,  # flash_attention_2 only supports bf16 and fp16
            report_to="none",
        )
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize("packing_strategy", ["bfd", "wrapped"])
    @ignore_warnings(message="You are using packing, but the attention implementation is not.*", category=UserWarning)
    @ignore_warnings(message="Padding-free training is enabled, but the attention.*", category=UserWarning)
    def test_train_packing(self, packing_strategy):
        # [FUNCTION DOC]
        # Runs the standard train check with packing, max_length 10, for the bfd and wrapped packing strategies
        # (attention-implementation warnings ignored). Inputs: packing_strategy (str, parametrized). Outputs: none.
        # Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(
            output_dir=self.tmp_dir, packing=True, packing_strategy=packing_strategy, max_length=10, report_to="none"
        )
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @ignore_warnings(message="You are using packing, but the attention implementation is not.*", category=UserWarning)
    @ignore_warnings(message="Padding-free training is enabled, but the attention.*", category=UserWarning)
    def test_eval_packing(self):
        # [FUNCTION DOC]
        # With packing and max_length 64 for both splits, asserts the packed train set holds 17 sequences in 3 rows and
        # eval holds 2 sequences in 1 row, with every row's total length at most 64.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling")

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            packing=True,
            max_length=64,
            report_to="none",
        )
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        # Check the number of sequences in train and eval datasets
        num_train_seqs = sum(len(x) for x in trainer.train_dataset["seq_lengths"])
        num_eval_seqs = sum(len(x) for x in trainer.eval_dataset["seq_lengths"])
        assert num_train_seqs == 17  # we should still have 17 seqs
        assert num_eval_seqs == 2  # we should still have 2 seqs

        # Check that all sequences are shorter than the max length
        assert all(sum(x) <= 64 for x in trainer.train_dataset["seq_lengths"])
        assert all(sum(x) <= 64 for x in trainer.eval_dataset["seq_lengths"])

        # Check the number of sequences in train and eval datasets
        assert len(trainer.train_dataset["input_ids"]) == 3  # w/ this dataset, we end up with 46 seqs
        assert len(trainer.eval_dataset["input_ids"]) == 1  # w/ this dataset, we end up with 6 seqs

    @ignore_warnings(message="You are using packing, but the attention implementation is not.*", category=UserWarning)
    @ignore_warnings(message="Padding-free training is enabled, but the attention.*", category=UserWarning)
    def test_only_train_packing(self):
        # [FUNCTION DOC]
        # With packing but eval_packing False, asserts train is packed (17 sequences in 3 rows, each at most 64 tokens)
        # while eval has no seq_lengths column and keeps 2 rows. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: none.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling")

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            packing=True,
            eval_packing=False,
            max_length=64,
            report_to="none",
        )
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        # Check the number of sequences in train dataset
        num_train_seqs = sum(len(x) for x in trainer.train_dataset["seq_lengths"])
        assert num_train_seqs == 17  # we should still have 17 seqs

        # We expect eval dataset not having "seq_lengths" as eval_packing is False
        assert "seq_lengths" not in trainer.eval_dataset

        # Check that all sequences are shorter than the max length
        assert all(sum(x) <= 64 for x in trainer.train_dataset["seq_lengths"])

        # Check the number of sequences in train and eval datasets
        assert len(trainer.train_dataset["input_ids"]) == 3  # w/ this dataset, we end up with 46 seqs
        assert len(trainer.eval_dataset["input_ids"]) == 2  # w/ this dataset, we end up with 6 seqs

    @pytest.mark.skipif(
        Version(transformers.__version__) < Version("4.57.0"), reason="Olmo 3 requires transformers>=4.57.0"
    )
    def test_tokenize_prompt_not_prefix_of_conversation(self):
        # [FUNCTION DOC]
        # With tiny Olmo-3 (transformers 4.57.0 or newer) whose generation prompt is not a prefix of the conversation,
        # asserts the trainable labels of each example decode to the assistant content plus eos.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        # The Olmo-3 Think template ends the generation prompt with `<think>`, which the full conversation doesn't
        # have, so the tokenized prompt is not a prefix of the tokenized prompt+completion
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Olmo3ForCausalLM",
            args=training_args,
            train_dataset=dataset,
        )

        # The completion is the whole assistant turn
        tokenizer = trainer.processing_class
        for example, processed in zip(dataset, trainer.train_dataset, strict=True):
            completion_ids = [label for label in processed["labels"] if label != -100]
            assert tokenizer.decode(completion_ids) == example["completion"][0]["content"] + tokenizer.eos_token

    def test_train_with_chat_template_kwargs(self):
        # [FUNCTION DOC]
        # Sets a Qwen-like template capitalizing roles when role_capital is true and adds an alternating
        # chat_template_kwargs column; asserts the template is kept and the first two tokenized examples start with the
        # lowercase then uppercase system prompt ids, then runs the standard train check.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")

        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        # The following template is a simplified version of the Qwen chat template, where an additional argument
        # `role_capital` is used to control the capitalization of roles.
        tokenizer.chat_template = '{%- if messages[0]["role"] == "system" -%}    {{ "<|im_start|>" + ("SYSTEM" if role_capital else "system") + "\\n" + messages[0]["content"] + "<|im_end|>\\n" }}{%- else -%}    {{ "<|im_start|>" + ("SYSTEM" if role_capital else "system") + "\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n" }}{%- endif -%}{%- for message in messages -%}    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) -%}        {{ "<|im_start|>" + (message.role.upper() if role_capital else message.role) + "\\n" + message.content + "<|im_end|>\\n" }}    {%- elif message.role == "assistant" -%}        {{ "<|im_start|>" + ("ASSISTANT" if role_capital else "assistant") }}        {%- if message.content -%}            {{ "\\n" + message.content }}        {%- endif -%}        {{ "<|im_end|>\\n" }}    {%- elif message.role == "tool" -%}        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") -%}            {{ "<|im_start|>" + ("USER" if role_capital else "user") }}        {%- endif -%}        {{ "\\n<tool_response>\\n" + message.content + "\\n</tool_response>" }}        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") -%}            {{ "<|im_end|>\\n" }}        {%- endif -%}    {%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}    {{ "<|im_start|>" + ("ASSISTANT" if role_capital else "assistant") + "\\n" }}{%- endif -%}'

        dataset = dataset.add_column(
            "chat_template_kwargs", [{"role_capital": bool(i % 2)} for i in range(len(dataset))]
        )
        assert "chat_template_kwargs" in dataset.features

        trainer = SFTTrainer(
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
            assert trainer.train_dataset[i]["input_ids"][: len(system_prompt_ids)] == system_prompt_ids

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_assistant_only(self):
        # [FUNCTION DOC]
        # Runs the standard train check on tiny Qwen3 with assistant_only_loss. Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, assistant_only_loss=True, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_dataset_preparation_builds_labels_for_assistant_only_loss(self):
        # [FUNCTION DOC]
        # With assistant_only_loss, asserts the prepared dataset has a labels column equal in length to input_ids, equal to
        # input_ids or -100 at each position, and containing both trainable and masked positions in every example.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Dataset preparation must bake the assistant masks into a labels column."""
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, assistant_only_loss=True, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM", args=training_args, train_dataset=dataset
        )

        assert "labels" in trainer.train_dataset.column_names
        for example in trainer.train_dataset:
            labels, input_ids = example["labels"], example["input_ids"]
            assert len(labels) == len(input_ids)
            # Labels are input_ids with non-assistant tokens masked to -100.
            assert all(label == -100 or label == token_id for label, token_id in zip(labels, input_ids, strict=True))
            assert any(label != -100 for label in labels)  # assistant tokens contribute to the loss
            assert any(label == -100 for label in labels)  # non-assistant tokens are masked

    @pytest.mark.skipif(
        Version(transformers.__version__) < Version("5.18.0.dev0"),
        reason="Processors return correct assistant masks for images since transformers 5.18.0",
    )
    @require_vision
    def test_train_vlm_assistant_only(self):
        # [FUNCTION DOC]
        # Requires vision and transformers 5.18.0.dev0 or newer. Runs the standard train check on tiny Qwen2-VL with
        # assistant_only_loss, batch 1 and no max_length. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_language_modeling", split="train")

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            assistant_only_loss=True,
            per_device_train_batch_size=1,  # VLM training is memory intensive, reduce batch size to avoid OOM
            max_length=None,  # for VLMs, truncating can remove image tokens, leading to errors
            report_to="none",
        )
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2VLForConditionalGeneration",
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

    @pytest.mark.skipif(
        Version(transformers.__version__) < Version("5.18.0.dev0"),
        reason="Processors return correct assistant masks for images since transformers 5.18.0",
    )
    @require_vision
    def test_data_collator_builds_labels_for_vlm_assistant_only_loss(self):
        # [FUNCTION DOC]
        # Requires vision and transformers 5.18.0.dev0 or newer. Collates two image examples and asserts that, on
        # attended positions, labels equal input_ids or -100 with both trainable and masked tokens present.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """The vision data collator must mask the non-assistant tokens in the labels."""
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, assistant_only_loss=True, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2VLForConditionalGeneration",
            args=training_args,
            train_dataset=dataset,
        )

        batch = trainer.data_collator([dataset[0], dataset[1]])
        for labels, input_ids, attention_mask in zip(
            batch["labels"], batch["input_ids"], batch["attention_mask"], strict=True
        ):
            labels, input_ids = labels[attention_mask == 1], input_ids[attention_mask == 1]
            # Labels are input_ids with non-assistant tokens masked to -100.
            assert all(label == -100 or label == token_id for label, token_id in zip(labels, input_ids, strict=True))
            assert any(label != -100 for label in labels)  # assistant tokens contribute to the loss
            assert any(label == -100 for label in labels)  # non-assistant tokens are masked

    @pytest.mark.skipif(
        Version(transformers.__version__) >= Version("5.18.0.dev0"),
        reason="Assistant-only loss on vision datasets is supported since transformers 5.18.0",
    )
    @require_vision
    def test_vlm_assistant_only_loss_requires_transformers_5_18(self):
        # [FUNCTION DOC]
        # Only on transformers older than 5.18.0.dev0: asserts assistant_only_loss with an image dataset raises
        # ValueError requiring transformers 5.18.0 or newer. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: none.
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, assistant_only_loss=True, report_to="none")
        with pytest.raises(ValueError, match="requires transformers>=5.18.0"):
            SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2VLForConditionalGeneration",
                args=training_args,
                train_dataset=dataset,
            )

    def test_fully_masked_examples_dropped_after_truncation(self):
        # [FUNCTION DOC]
        # With max_length 3, asserts an example whose assistant tokens are all truncated away is dropped and the other
        # keeps labels 6, 7, 8. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        # Example 0's assistant tokens all lie beyond `max_length=3`, so keep_start truncation leaves it fully masked;
        # example 1 keeps a trainable token and survives.
        dataset = Dataset.from_list(
            [
                {"input_ids": [1, 2, 3, 4, 5], "assistant_masks": [0, 0, 0, 1, 1]},
                {"input_ids": [6, 7, 8], "assistant_masks": [1, 1, 1]},
            ]
        )

        training_args = SFTConfig(output_dir=self.tmp_dir, max_length=3, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        assert trainer.train_dataset[:]["labels"] == [[6, 7, 8]]

    def test_dataset_truncated_to_max_length(self):
        # [FUNCTION DOC]
        # Asserts with max_length 4 every prepared example has at most 4 input_ids and labels.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Dataset preparation truncates every example to `max_length`."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, max_length=4, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        for example in trainer.train_dataset:
            assert len(example["input_ids"]) <= 4
            assert len(example["labels"]) <= 4

    @pytest.mark.parametrize("dataset_text_field", ["text", "my_column"])
    def test_dataset_preparation_adds_eos_to_dataset_text_field(self, dataset_text_field):
        # [FUNCTION DOC]
        # Asserts the eos token is appended (and trained on) when the text column is named text or a custom
        # dataset_text_field. Inputs: dataset_text_field (str, parametrized). Outputs: none. Exceptions: AssertionError.
        # Side effects: none.
        """Dataset preparation must append EOS to the text column named by `dataset_text_field`, not only `"text"`."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")
        if dataset_text_field != "text":
            dataset = dataset.rename_column("text", dataset_text_field)

        training_args = SFTConfig(output_dir=self.tmp_dir, dataset_text_field=dataset_text_field, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        eos_token_id = trainer.processing_class.eos_token_id
        for example in trainer.train_dataset:
            assert example["input_ids"][-1] == eos_token_id
            assert example["labels"][-1] == eos_token_id  # EOS is trained on

    def test_dataset_preparation_builds_labels_for_completion_only(self):
        # [FUNCTION DOC]
        # Asserts a prompt-completion dataset (completion_only_loss defaulting to true) gets labels equal to input_ids or
        # -100, with both prompt (masked) and completion (trainable) tokens present.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Dataset preparation must bake the completion mask into a labels column when completion_only_loss
        resolves to True (the default for prompt-completion datasets)."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_completion", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        assert "labels" in trainer.train_dataset.column_names
        for example in trainer.train_dataset:
            labels, input_ids = example["labels"], example["input_ids"]
            assert len(labels) == len(input_ids)
            # Labels are input_ids with prompt tokens masked to -100.
            assert all(label == -100 or label == token_id for label, token_id in zip(labels, input_ids, strict=True))
            assert any(label != -100 for label in labels)  # completion tokens contribute to the loss
            assert any(label == -100 for label in labels)  # prompt tokens are masked

    def test_dataset_preparation_respects_existing_labels(self):
        # [FUNCTION DOC]
        # Asserts a user-provided labels column is kept unchanged even when assistant_masks is also present.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """A user-provided labels column must be taken as is, even when mask columns are also present."""
        dataset = Dataset.from_list(
            [
                {"input_ids": [1, 2, 3, 4], "labels": [1, -100, 3, -100], "assistant_masks": [1, 1, 1, 1]},
                {"input_ids": [5, 6], "labels": [-100, 6], "assistant_masks": [1, 1]},
            ]
        )

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        assert trainer.train_dataset[:]["labels"] == [[1, -100, 3, -100], [-100, 6]]

    def test_dataset_preparation_builds_labels_for_pretokenized_with_masks(self):
        # [FUNCTION DOC]
        # Asserts pre-tokenized examples with assistant_masks but no labels get labels with -100 where the mask is 0.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Pre-tokenized datasets that carry mask columns but no labels must get labels built at preparation."""
        dataset = Dataset.from_list(
            [
                {"input_ids": [1, 2, 3, 4], "assistant_masks": [0, 0, 1, 1]},
                {"input_ids": [5, 6, 7], "assistant_masks": [0, 1, 0]},
            ]
        )

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        assert trainer.train_dataset[:]["labels"] == [[-100, -100, 3, 4], [-100, 6, -100]]

    def test_packing_carries_labels(self):
        # [FUNCTION DOC]
        # With packing (max_length 64) on a prompt-completion dataset, asserts labels are packed (same length as
        # input_ids), completion_mask is removed, and some labels remain trainable.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """With packing enabled, the labels column must be packed alongside input_ids (replacing the masks)."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_completion", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, packing=True, max_length=64, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        column_names = trainer.train_dataset.column_names
        assert "labels" in column_names
        assert "completion_mask" not in column_names
        for example in trainer.train_dataset:
            assert len(example["labels"]) == len(example["input_ids"])
        # The packed dataset must still contain trainable (non -100) labels
        assert any(label != -100 for example in trainer.train_dataset for label in example["labels"])

    def test_skip_prepare_dataset_with_masks_but_no_labels_raises(self):
        # [FUNCTION DOC]
        # Asserts skip_prepare_dataset with assistant_masks but no labels raises ValueError asking to add a labels
        # column. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """With `skip_prepare_dataset=True`, labels are not built at preparation time and the collator doesn't
        consume the mask columns; such datasets must be rejected instead of silently training on the full sequence."""
        dataset = Dataset.from_list([{"input_ids": [1, 2, 3, 4], "assistant_masks": [0, 0, 1, 1]} for _ in range(2)])

        training_args = SFTConfig(
            output_dir=self.tmp_dir, dataset_kwargs={"skip_prepare_dataset": True}, report_to="none"
        )
        with pytest.raises(ValueError, match="Add a 'labels' column"):
            SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
            )

    def test_train_completion_only(self):
        # [FUNCTION DOC]
        # Runs the standard train check on tiny Qwen3 with completion_only_loss and conversational prompt-completion
        # data. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, completion_only_loss=True, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_completion_only_harmony(self):
        # [FUNCTION DOC]
        # Runs the standard train check on tiny GPT-OSS with completion_only_loss and harmony prompt-completion data.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/harmony", "prompt_completion", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, completion_only_loss=True, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-GptOssForCausalLM", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_assistant_only_and_completion_only(self):
        # [FUNCTION DOC]
        # Appends a user turn and another assistant turn to each completion, then runs the standard train check on tiny
        # Qwen3 with both assistant_only_loss and completion_only_loss. Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")

        # To test this case, we need to add user messages in the completion (they'll be masked in the loss)
        def add_to_completion(example):
            # [FUNCTION DOC]
            # HELPER: Appends the first prompt message and then the first completion message to the completion list.
            # Inputs: example (dict). Outputs: the mutated example. Exceptions: none. Side effects: mutates the example.
            example["completion"].append(example["prompt"][0])
            example["completion"].append(example["completion"][0])
            return example

        dataset = dataset.map(add_to_completion)

        training_args = SFTConfig(
            output_dir=self.tmp_dir, assistant_only_loss=True, completion_only_loss=True, report_to="none"
        )
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_assistant_only_iterable_dataset(self):
        # [FUNCTION DOC]
        # Runs the standard train check with assistant_only_loss on a streaming dataset (max_steps 3).
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset(
            "trl-internal-testing/zen", "conversational_language_modeling", split="train", streaming=True
        )

        training_args = SFTConfig(output_dir=self.tmp_dir, assistant_only_loss=True, max_steps=3, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_with_set_chat_template_from_model(self):
        # [FUNCTION DOC]
        # Uses chat_template_path naming Qwen3-4B to give tiny GPT-NeoX a chat template and runs the standard train
        # check. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, chat_template_path="Qwen/Qwen3-4B", report_to="none")
        # trl-internal-testing/tiny-GPTNeoXForCausalLM doesn't have a chat template set by default
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-GPTNeoXForCausalLM", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_with_set_chat_template_from_path(self, lazy_shared_datadir):
        # [FUNCTION DOC]
        # Uses a local template.jinja via chat_template_path, runs the standard train check, then asserts checkpoint-9
        # contains chat_template.jinja identical to the source. Inputs: lazy_shared_datadir (fixture path).
        # Outputs: none. Exceptions: AssertionError. Side effects: writes checkpoints to tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            chat_template_path=str(lazy_shared_datadir / "template.jinja"),
            report_to="none",
        )
        # trl-internal-testing/tiny-GPTNeoXForCausalLM doesn't have a chat template set by default
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-GPTNeoXForCausalLM", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
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
        # Runs the standard train check on the toolcall language-modeling dataset (batch 2, max_length 512).
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/toolcall", "language_modeling", split="train")

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,  # toolcall sequences are longer than standard data, reduce batch size to avoid OOM
            max_length=512,  # toolcall sequences are longer than standard data, limit length to avoid OOM
            report_to="none",
        )
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
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
        # Same as the toolcall test after decoding the tools column from JSON strings into lists of dicts.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        # Tabular backends (Arrow/Parquet) can insert `None` for missing keys in nested structures.
        # If `tools` is stored as a list of dicts and examples use different dict schemas, nulls may
        # be introduced and break tool processing. This test ensures we also support `tools` provided
        # as a list of dicts.
        dataset = load_dataset("trl-internal-testing/toolcall", "language_modeling", split="train")

        def convert_to_json(example):
            # [FUNCTION DOC]
            # HELPER: Returns a dict replacing tools with its JSON-decoded value. Inputs: example (dict). Outputs: dict.
            # Exceptions: json decode errors. Side effects: none.
            return {"tools": json.loads(example["tools"])}

        dataset = dataset.map(convert_to_json)

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,  # toolcall sequences are longer than standard data, reduce batch size to avoid OOM
            max_length=512,  # toolcall sequences are longer than standard data, limit length to avoid OOM
            report_to="none",
        )
        trainer = SFTTrainer(
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
        # Trains with eval every 3 steps and asserts the first log entry has eval_loss.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling")

        training_args = SFTConfig(output_dir=self.tmp_dir, eval_strategy="steps", eval_steps=3, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        trainer.train()

        assert trainer.state.log_history[0]["eval_loss"] is not None

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
        # Asserts evaluate accepts a raw eval dataset in six container forms (regression for issue 6115), yielding
        # eval_loss or eval_data1_loss and eval_data2_loss. Inputs: eval_dataset_type (str, parametrized).
        # Outputs: none. Exceptions: AssertionError. Side effects: none.
        # `evaluate` accepts a raw (unprepared) dataset passed directly, not only a preprocessed `eval_dataset` set
        # at init. See https://github.com/huggingface/trl/issues/6115.
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")
        streaming = "iterable" in eval_dataset_type
        eval_split = load_dataset(
            "trl-internal-testing/zen", "standard_language_modeling", split="test", streaming=streaming
        )
        if eval_dataset_type in ("dataset", "iterable_dataset"):
            eval_dataset = eval_split
        elif eval_dataset_type in ("dataset_dict", "iterable_dataset_dict"):
            dataset_dict_cls = IterableDatasetDict if streaming else DatasetDict
            eval_dataset = dataset_dict_cls({"data1": eval_split, "data2": eval_split})
        else:  # "dict_of_dataset" or "dict_of_iterable_dataset"
            eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=train_dataset
        )

        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        if eval_dataset_type in ("dataset", "iterable_dataset"):
            assert metrics["eval_loss"] is not None
        else:
            assert metrics["eval_data1_loss"] is not None
            assert metrics["eval_data2_loss"] is not None

    def test_train_with_metric_for_best_model(self):
        # [FUNCTION DOC]
        # Smoke test: trains with eval every 3 steps and metric_for_best_model set to eval_mean_token_accuracy (a metric
        # SFTTrainer adds itself); passes if training does not fail looking the metric up in evaluate results.
        # Inputs/Outputs: none. Exceptions: any training error (e.g. KeyError). Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling")

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            eval_strategy="steps",
            eval_steps=3,
            # It's important to use a key that SFTTrainer adds itself (not one from the base Trainer), since the bug is
            # that trainer-specific metrics don't reach the dict returned by `evaluate()`.
            metric_for_best_model="eval_mean_token_accuracy",
            report_to="none",
        )
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        trainer.train()

    def test_train_with_multiple_eval_dataset(self):
        # [FUNCTION DOC]
        # Trains with two named eval datasets and asserts the third-last and second-last log entries contain
        # eval_data1_loss and eval_data2_loss. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling")

        training_args = SFTConfig(output_dir=self.tmp_dir, eval_strategy="steps", eval_steps=3, report_to="none")
        trainer = SFTTrainer(
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
        # TypeError (must be Dataset or IterableDataset); Dataset and streaming dataset (max_steps 3) are tokenized so the
        # first example has input_ids. Inputs: train_dataset_type (str, parametrized). Outputs: none.
        # Exceptions: AssertionError. Side effects: none.
        streaming = "iterable" in train_dataset_type
        if train_dataset_type == "none":
            train_dataset = None
        else:
            train_dataset = load_dataset(
                "trl-internal-testing/zen", "standard_language_modeling", split="train", streaming=streaming
            )
            if train_dataset_type == "unsupported_dataset_dict":
                # `DatasetDict` is representative of any unsupported type here; not exhaustive
                train_dataset = DatasetDict({"train": train_dataset})

        # Iterable (streaming) datasets have no length, so `max_steps` is required.
        training_args = SFTConfig(output_dir=self.tmp_dir, max_steps=3 if streaming else -1, report_to="none")

        if train_dataset_type == "none":
            with pytest.raises(ValueError, match="`train_dataset` is required"):
                SFTTrainer(
                    model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                    args=training_args,
                    train_dataset=train_dataset,
                )
        elif train_dataset_type == "unsupported_dataset_dict":
            with pytest.raises(TypeError, match="`train_dataset` must be a `Dataset` or `IterableDataset`"):
                SFTTrainer(
                    model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                    args=training_args,
                    train_dataset=train_dataset,
                )
        else:
            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=train_dataset
            )
            assert "input_ids" in next(iter(trainer.train_dataset))

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
        # Checks eval_dataset preparation: None stays None; dict-like inputs keep keys data1 and data2 each tokenized
        # with input_ids; single datasets are tokenized. Inputs: eval_dataset_type (str, parametrized). Outputs: none.
        # Exceptions: AssertionError. Side effects: none.
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        if eval_dataset_type == "none":
            eval_dataset = None
        else:
            streaming = "iterable" in eval_dataset_type
            eval_split = load_dataset(
                "trl-internal-testing/zen", "standard_language_modeling", split="test", streaming=streaming
            )
            if eval_dataset_type in ("dataset", "iterable_dataset"):
                eval_dataset = eval_split
            elif eval_dataset_type in ("dataset_dict", "iterable_dataset_dict"):
                dataset_dict_cls = IterableDatasetDict if streaming else DatasetDict
                eval_dataset = dataset_dict_cls({"data1": eval_split, "data2": eval_split})
            else:  # "dict_of_dataset" or "dict_of_iterable_dataset"
                eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
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
            assert "input_ids" in next(iter(trainer.eval_dataset["data1"]))
            assert "input_ids" in next(iter(trainer.eval_dataset["data2"]))
        else:
            assert "input_ids" in next(iter(trainer.eval_dataset))

    def test_train_with_compute_metrics(self):
        # [FUNCTION DOC]
        # Trains with eval every 3 steps and a compute_metrics that asserts predictions carry full-vocabulary logits and
        # returns my_metric 0.123; asserts training emits a FutureWarning about return_outputs True and the second-last
        # log entry has eval_my_metric 0.123. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling")

        def dummy_compute_metrics(eval_pred):
            # [FUNCTION DOC]
            # HELPER: Asserts the last dimension of eval_pred.predictions equals the model vocab size (closure over trainer)
            # and returns a dict with my_metric 0.123. Exceptions: AssertionError. Side effects: none.
            # The logits are the full-vocabulary logits, as before the fused LM head
            assert eval_pred.predictions.shape[-1] == trainer.model.config.vocab_size
            return {"my_metric": 0.123}

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            eval_strategy="steps",
            eval_steps=3,
            report_to="none",
        )
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=dummy_compute_metrics,
        )

        with pytest.warns(FutureWarning, match="`return_outputs=True`"):
            trainer.train()

        assert trainer.state.log_history[-2]["eval_my_metric"] == 0.123

    def test_train_with_compute_loss_func(self):
        # [FUNCTION DOC]
        # Trains 2 steps with a custom compute_loss_func (expects a FutureWarning that it is deprecated); asserts the
        # function received full-vocabulary logits every time and that mean_token_accuracy was logged.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")
        vocab_sizes = []

        def compute_loss_func(outputs, labels, num_items_in_batch=None):
            # [FUNCTION DOC]
            # HELPER: Records the logits vocabulary size, then returns token-level cross entropy between logits without the
            # last position and labels without the first position, ignoring -100. Inputs: outputs (model output), labels
            # (tensor), num_items_in_batch (unused). Outputs: scalar loss tensor. Exceptions: none.
            # Side effects: appends to vocab_sizes.
            # The outputs are the model's own, with the full logits, as before the fused LM head
            vocab_sizes.append(outputs.logits.shape[-1])
            logits = outputs.logits[:, :-1].flatten(0, 1)
            return torch.nn.functional.cross_entropy(logits, labels[:, 1:].flatten(), ignore_index=-100)

        training_args = SFTConfig(output_dir=self.tmp_dir, max_steps=2, logging_steps=1, report_to="none")
        with pytest.warns(FutureWarning, match="`compute_loss_func` is deprecated"):
            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=dataset,
                compute_loss_func=compute_loss_func,
            )
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert vocab_sizes and all(v == trainer.model.config.vocab_size for v in vocab_sizes)
        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert trainer.state.log_history[-2]["mean_token_accuracy"] is not None
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_predict_returns_logits(self):
        # [FUNCTION DOC]
        # Asserts predict (with a FutureWarning about return_outputs True) returns predictions whose last dimension is
        # the vocab size. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="test")
        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        with pytest.warns(FutureWarning, match="`return_outputs=True`"):
            predictions = trainer.predict(trainer.train_dataset)

        assert predictions.predictions.shape[-1] == trainer.model.config.vocab_size

    def test_chunked_nll_is_deprecated_alias_of_nll(self):
        # [FUNCTION DOC]
        # Asserts loss_type chunked_nll emits a deprecation FutureWarning and is normalized to nll.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        with pytest.warns(FutureWarning, match="`loss_type='chunked_nll'` is deprecated"):
            training_args = SFTConfig(output_dir=self.tmp_dir, loss_type="chunked_nll", report_to="none")
        assert training_args.loss_type == "nll"

    def test_label_smoothing(self):
        # [FUNCTION DOC]
        # With label_smoothing_factor 0.1 (deprecated, FutureWarning expected) and the model in eval mode, asserts
        # compute_loss on a collated batch of 4 equals transformers LabelSmoother with epsilon 0.1 on the full logits
        # with shifted labels. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: sets model to eval mode.
        # Label smoothing goes through `Trainer`'s label smoother, on the full logits
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")
        training_args = SFTConfig(output_dir=self.tmp_dir, label_smoothing_factor=0.1, report_to="none")
        with pytest.warns(FutureWarning, match="`label_smoothing_factor` is deprecated"):
            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
            )
        trainer.model.eval()  # no dropout, so the two forward passes match
        batch = trainer.data_collator([trainer.train_dataset[i] for i in range(4)])
        inputs = {k: v.to(trainer.model.device) for k, v in batch.items()}

        loss = trainer.compute_loss(trainer.model, inputs)

        expected = LabelSmoother(epsilon=0.1)(trainer.model(**inputs), inputs["labels"], shift_labels=True)
        torch.testing.assert_close(loss, expected)

    # In practice, this test is the same as `test_train`, since gradient checkpointing is enabled by default in
    # `SFTTrainer`. We keep it as a regression guard: if the default ever changes, we still explicitly test gradient
    # checkpointing, which has caused issues in the past.
    def test_train_with_gradient_checkpointing(self):
        # [FUNCTION DOC]
        # Runs the standard train check with gradient_checkpointing explicitly True (regression guard).
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, gradient_checkpointing=True, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize(
        "gradient_checkpointing_kwargs",
        [
            {"use_reentrant": True},
            {"use_reentrant": False},
            {"selective": True},
            {"selective": False},
        ],
    )
    def test_train_with_gradient_checkpointing_kwargs(self, gradient_checkpointing_kwargs):
        # [FUNCTION DOC]
        # Runs the standard train check with gradient checkpointing and kwargs use_reentrant True/False or selective
        # True/False. Inputs: gradient_checkpointing_kwargs (dict, parametrized). Outputs: none.
        # Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
            report_to="none",
        )
        trainer = SFTTrainer(
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
        # Runs the standard train check with fp16; requires an accelerator. Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, fp16=True, report_to="none")
        trainer = SFTTrainer(
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
    def test_train_with_activation_offloading(self):
        # [FUNCTION DOC]
        # Runs the standard train check with activation_offloading; requires an accelerator. Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, activation_offloading=True, report_to="none")
        trainer = SFTTrainer(
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
        # Asserts a trainer with default args adds sft and trl to model.model_tags. Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: none.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            train_dataset=dataset,
        )

        for tag in ["sft", "trl"]:
            assert tag in trainer.model.model_tags

    @require_peft
    def test_tag_added_peft(self):
        # [FUNCTION DOC]
        # Same tag assertion with a LoRA peft_config (requires peft). Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: none.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            train_dataset=dataset,
            peft_config=LoraConfig(),
        )

        for tag in ["sft", "trl"]:
            assert tag in trainer.model.model_tags

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
    @require_vision
    def test_train_vlm(self, model_id):
        # [FUNCTION DOC]
        # Requires vision. Trains each listed VLM (Gemma3/4, LFM2.5-VL, LLaVA, LLaVA-Next, Muse Glimmer, Qwen2/2.5/3-VL,
        # Qwen3.5 dense and MoE, Muse Spark; newer ones skipped on old transformers) with batch 1 and no max_length,
        # asserting train_loss is logged and every parameter changed except: LLaVA vision encoder layers.1 and
        # post_layernorm stay unchanged, and Muse Spark audio_tower stays unchanged.
        # Inputs: model_id (str, parametrized). Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_language_modeling", split="train")

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,  # VLM training is memory intensive, reduce batch size to avoid OOM
            max_length=None,  # for VLMs, truncating can remove image tokens, leading to errors
            report_to="none",
        )
        trainer = SFTTrainer(model=model_id, args=training_args, train_dataset=dataset)

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
    @require_vision
    def test_train_vlm_multi_image(self, model_id):
        # [FUNCTION DOC]
        # Requires vision; xfail (non-strict) before transformers 4.57.0. On the multi-image prompt-completion dataset
        # with Qwen2.5-VL, asserts ddp_find_unused_parameters is enabled, then runs the standard train check.
        # Inputs: model_id (str, parametrized). Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset(
            "trl-internal-testing/zen-multi-image", "conversational_prompt_completion", split="train"
        )

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,  # VLM training is memory intensive, reduce batch size to avoid OOM
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            max_length=None,  # for VLMs, truncating can remove image tokens, leading to errors
            report_to="none",
        )
        trainer = SFTTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
        )
        # Some samples have no image: their batches leave the vision tower without gradients
        assert trainer.args.ddp_find_unused_parameters

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
            # Special case for Gemma, as it uses token_type_ids, and we need to ensure they are properly in the collator:
            "trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
            pytest.param(
                "trl-internal-testing/tiny-Gemma4ForConditionalGeneration",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.5.0"),
                    reason="Gemma4 models were introduced in transformers-5.5.0",
                ),
            ),
        ],
    )
    @require_vision
    def test_train_vlm_prompt_completion(self, model_id):
        # [FUNCTION DOC]
        # Requires vision. Runs the standard train check on image prompt-completion data for Qwen2.5-VL, Gemma3 and Gemma4
        # (the Gemma models exercise token_type_ids collation). Inputs: model_id (str, parametrized). Outputs: none.
        # Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_completion", split="train")

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,  # VLM training is memory intensive, reduce batch size to avoid OOM
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            max_length=None,  # for VLMs, truncating can remove image tokens, leading to errors
            report_to="none",
        )
        trainer = SFTTrainer(
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
        [
            "conversational_language_modeling",
            "conversational_prompt_completion",
            "standard_language_modeling",  # Regression test for #5334
            "standard_prompt_completion",
        ],
    )
    @require_vision
    def test_train_vlm_text_only_data(self, model_id, dataset_config):
        # [FUNCTION DOC]
        # Requires vision. Trains Qwen2.5-VL on text-only zen data in four formats; asserts ddp_find_unused_parameters,
        # train_loss logged, parameters under model.visual unchanged (tolerance 1e-12) and all others changed.
        # Inputs: model_id, dataset_config (str, parametrized). Outputs: none. Exceptions: AssertionError.
        # Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", dataset_config, split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
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

    @ignore_warnings(message="You are using packing, but the attention implementation is not.*", category=UserWarning)
    @require_vision
    def test_train_vlm_text_only_data_packing(self):
        # [FUNCTION DOC]
        # Requires vision. Regression for issue 6545: packing on a VLM checkpoint with text-only data must work; trains
        # Qwen2.5-VL with packing and asserts visual parameters unchanged and others changed.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        # Packing is incompatible with on-the-fly image processing, not with VLMs. A text-only dataset goes through
        # the regular text pipeline, so packing must be available on a VLM checkpoint too. Regression test for #6545.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            packing=True,
            report_to="none",
        )
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
            args=training_args,
            train_dataset=dataset,
        )

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

    @require_vision
    def test_vision_dataset_with_text_model_raises(self):
        # [FUNCTION DOC]
        # Requires vision. Asserts an image dataset with a text-only model raises ValueError mentioning vision-related
        # data and a vision-language model. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_language_modeling", split="train")
        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        with pytest.raises(ValueError, match="vision-related.*vision-language model"):
            SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=dataset,
            )

    @require_peft
    def test_prompt_tuning(self):
        # [FUNCTION DOC]
        # Trains with a prompt encoder (p-tuning, 8 virtual tokens) peft_config; asserts train_loss and
        # mean_token_accuracy are logged, base_model parameters unchanged, prompt_encoder parameters changed, and raises
        # ValueError on any other parameter name. Requires peft. Inputs/Outputs: none.
        # Exceptions: AssertionError, ValueError. Side effects: writes tmp_dir.
        """Test that SFT works with Prompt Tuning."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            peft_config=PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=8),
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert trainer.state.log_history[-1]["mean_token_accuracy"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if "base_model" in n:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "prompt_encoder" in n:  # We expect the peft params to be different
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."
            else:
                raise ValueError(f"Unexpected parameter {n} in model: {trainer.model}")

    @require_peft
    @require_bitsandbytes
    def test_train_peft_and_quantization(self):
        # [FUNCTION DOC]
        # With peft and bitsandbytes, loads the model id in 4-bit NF4 double-quant (bf16 compute) with LoRA and lr 0.1;
        # asserts 4-bit loading, train_loss and mean_token_accuracy logged, and LoRA parameters bfloat16 and changed.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, learning_rate=0.1, report_to="none")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        trainer = SFTTrainer(
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
        # Same as the 4-bit LoRA test with DoRA: magnitude vectors stay float32 and change, other LoRA parameters are
        # bfloat16 and change. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, learning_rate=0.1, report_to="none")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        trainer = SFTTrainer(
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

    @require_peft
    def test_prompt_tuning_peft_model(self):
        # [FUNCTION DOC]
        # Same assertions as test_prompt_tuning but passing a model already wrapped with a prompt-encoder PEFT config.
        # Requires peft. Inputs/Outputs: none. Exceptions: AssertionError, ValueError. Side effects: writes tmp_dir.
        """Test that SFT works with Prompt Tuning and a pre-converted PeftModel"""
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", dtype="float32")
        model = get_peft_model(model, PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=8))

        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset)

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert trainer.state.log_history[-1]["mean_token_accuracy"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if "base_model" in n:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "prompt_encoder" in n:  # We expect the peft params to be different
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."
            else:
                raise ValueError(f"Unexpected parameter {n} in model: {trainer.model}")

    def test_pad_token_id_synced_with_model_config(self):
        # [FUNCTION DOC]
        # With tiny Mistral (no pad token, eos used), asserts the tokenizer pad id is set and equals model.config and
        # model.generation_config pad_token_id. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        # This model's tokenizer has no pad token, so the trainer falls back to the eos token. The model configs must
        # follow: otherwise `Trainer` realigns them at train time and reports it as a change the user did not make.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-MistralForCausalLM-0.2", args=training_args, train_dataset=dataset
        )

        pad_token_id = trainer.processing_class.pad_token_id
        assert pad_token_id is not None
        assert trainer.model.config.pad_token_id == pad_token_id
        assert trainer.model.generation_config.pad_token_id == pad_token_id

    @require_vision
    def test_pad_token_id_synced_with_model_config_vision(self):
        # [FUNCTION DOC]
        # Requires vision. With Qwen2.5-VL and an image dataset, asserts the processor tokenizer pad id is set and equals
        # the text config pad_token_id and generation_config pad_token_id. Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: none.
        # A vision dataset takes the other collator branch, which used to skip the pad token handling entirely.
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, max_length=None, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
            args=training_args,
            train_dataset=dataset,
        )

        pad_token_id = trainer.processing_class.tokenizer.pad_token_id
        assert pad_token_id is not None
        assert trainer.model.config.get_text_config().pad_token_id == pad_token_id
        assert trainer.model.generation_config.pad_token_id == pad_token_id

    @pytest.mark.parametrize(
        "generation_eos_token_id, expected_eos_token_ids",
        [
            ([151645, 151643], [151644, 151645, 151643]),  # a list of ids, as Qwen and Llama ship
            (151645, [151644, 151645]),  # a single id, as Mistral and GPT-2 ship
            (None, [151644]),  # no id at all
            ([151644, 151645], [151644, 151645]),  # a list that already holds the requested token
        ],
    )
    def test_eos_token_id_synced_with_model_config(self, generation_eos_token_id, expected_eos_token_ids):
        # [FUNCTION DOC]
        # With eos_token set to the im_start token (id 151644) and the model's generation eos given as a list, a single
        # id, None, or a list already containing 151644, asserts the tokenizer eos id is 151644, model.config eos equals
        # it, and generation_config eos becomes the expected list (requested id first, then existing ids, without
        # duplicates). Inputs: generation_eos_token_id, expected_eos_token_ids (parametrized). Outputs: none.
        # Exceptions: AssertionError. Side effects: none.
        # The trainer sets the requested eos token on the tokenizer. The model configs must follow: otherwise
        # `Trainer` realigns them at train time and reports it as a change the user did not make. The generation
        # config holds the eos token as a list, as a single id, or not at all, so cover the three shapes.
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", dtype="float32")
        model.generation_config.eos_token_id = generation_eos_token_id

        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, eos_token="<|im_start|>", report_to="none")
        trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset)

        eos_token_id = trainer.processing_class.eos_token_id
        assert eos_token_id == 151644  # <|im_start|>
        assert trainer.model.config.eos_token_id == eos_token_id
        # The model's own eos tokens are kept, since any of them halts generation
        assert trainer.model.generation_config.eos_token_id == expected_eos_token_ids
