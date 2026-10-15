# [MODULE DOC]
# Tests for the helpers in trl.experimental.utils: DataCollatorForChatML, truncate_dataset, truncate_right,
# prepare_peft_model (QLoRA dtype handling) and create_reference_model (frozen copies and shared layers).
# Each helper gets its own TrlTestCase subclass; real tiny models/tokenizers from trl-internal-testing are used.
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
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.experimental.utils import (
    DataCollatorForChatML,
    create_reference_model,
    prepare_peft_model,
    truncate_dataset,
    truncate_right,
)

from ..testing_utils import TrlTestCase, require_bitsandbytes, require_peft, require_torch_accelerator


class TestDataCollatorForChatML(TrlTestCase):
    # [CLASS DOC]
    # Tests DataCollatorForChatML on the zen conversational_language_modeling dataset with the tiny Qwen2 tokenizer.
    # State set per test in setup_method: tokenizer, bos/eos ids, ignore_index (-100), max_length (1024),
    # messages_key ("messages"), examples (dataset rows as list) and the collator. Inherits TrlTestCase.
    def setup_method(self):
        # [FUNCTION DOC]
        # HELPER: Per-test fixture: loads the tiny Qwen2 tokenizer (pad falls back to eos), records bos/eos ids (defaults 1
        # and 2 if missing), ignore_index -100, max_length 1024, messages key, loads the zen dataset rows and builds the
        # collator with tokenizer, max_length and ignore_index.
        # Inputs: none. Outputs: none. Exceptions: load errors propagate. Side effects: sets instance attributes.
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Define token IDs
        self.bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 1
        self.eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 2
        # Token ID for "true", the last assistant's response in the example:
        self.ignore_index = -100
        self.max_length = 1024
        self.messages_key = "messages"

        # Example input
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")
        self.examples = dataset.to_list()

        # Initialize the data collator
        self.collator = DataCollatorForChatML(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            ignore_index=self.ignore_index,
        )

    def test_data_collator_for_chatml(self):
        # [FUNCTION DOC]
        # Checks the collator output for the first example.
        # Assertions: keys input_ids, attention_mask, labels, prompts, prompt_attention_mask exist; the last message is
        # from the assistant; decoded input_ids contain the final assistant reply while decoded prompts do not; labels for
        # the first prompt-length positions equal ignore_index; the non-ignored labels after the prompt, up to and including
        # the first im_end token, equal the tokenization of the reply plus eos; the decoded input does not end with an
        # assistant generation prompt.
        # KNOWN ISSUE: the final labels-equal-reply assertion is duplicated verbatim (harmless); the setup comment about a
        # "true" token id is stale and unrelated to the code beneath it.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        # Process the data
        data = self.collator(self.examples)

        # Verify basic shapes and types
        assert "input_ids" in data
        assert "attention_mask" in data
        assert "labels" in data
        assert "prompts" in data
        assert "prompt_attention_mask" in data

        # Decode input_ids and labels for verification
        input_ids = data["input_ids"][0].tolist()
        labels = data["labels"][0].tolist()
        prompt_only = data["prompts"][0].tolist()

        # Get the last assistant's response for comparison
        last_message = self.examples[0][self.messages_key][-1]
        assert last_message["role"] == "assistant", "Last message should be from assistant"
        last_assistant_response = last_message["content"]

        # Verify that input_ids contain both prompt and response
        decoded_input = self.tokenizer.decode(input_ids)
        assert last_assistant_response in decoded_input, "Input should contain assistant's response"

        # Verify that prompts only contain the conversation up to the last response
        decoded_prompt = self.tokenizer.decode(prompt_only)
        assert last_assistant_response not in decoded_prompt, "Prompt should not contain assistant's response"

        # Verify labels are -100 for non-assistant parts
        prompt_length = len(prompt_only)
        assert all(label == self.ignore_index for label in labels[:prompt_length]), (
            "Labels should be ignore_index for prompt tokens"
        )

        # Verify labels match assistant response after prompt
        # Add a filter to remove any trailing tokens after the first <|im_end|>
        last_assistant_response_with_end = last_assistant_response + self.tokenizer.eos_token
        last_assistant_response_tokens = self.tokenizer.encode(
            last_assistant_response_with_end, add_special_tokens=False
        )

        response_labels = []
        for label in labels[prompt_length:]:
            if label == self.ignore_index:
                continue
            response_labels.append(label)
            if label == self.tokenizer.convert_tokens_to_ids("<|im_end|>"):
                break
        assert response_labels == last_assistant_response_tokens, "Labels should match assistant response tokens"

        # Verify there isn't a generation prompt at the end
        generation_prompt = "<|im_start|>assistant"
        assert not decoded_input.strip().endswith(generation_prompt), (
            f"Input should not end with generation prompt '{generation_prompt}'"
        )

        assert response_labels == last_assistant_response_tokens, "Labels should match assistant response tokens"


class TestTruncateExamples(TrlTestCase):
    # [CLASS DOC]
    # Tests truncate_dataset, which truncates list columns (input_ids, attention_mask) to max_length. Inherits TrlTestCase.
    def test_with_dataset(self):
        # [FUNCTION DOC]
        # Checks a numpy-formatted (float32) Dataset is truncated to length 2 per row (shorter rows untouched) and that
        # the dataset format is preserved. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        examples = {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
        }
        dataset = Dataset.from_dict(examples)
        dataset = dataset.with_format("numpy", dtype="float32")
        format = dataset.format
        max_length = 2
        expected_output = {
            "input_ids": [[1, 2], [4, 5], [8]],
            "attention_mask": [[0, 1], [0, 0], [1]],
        }
        dataset = truncate_dataset(dataset, max_length)
        assert dataset.to_dict() == expected_output
        assert format == dataset.format

    def test_with_iterable_dataset(self):
        # [FUNCTION DOC]
        # Checks truncation of a numpy-formatted IterableDataset: the batched unformatted output equals the truncated
        # expected dict and the private _formatting attribute is preserved. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        examples = {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
        }
        dataset = Dataset.from_dict(examples).to_iterable_dataset()
        dataset = dataset.with_format("numpy")
        formatting = dataset._formatting
        max_length = 2
        expected_output = {
            "input_ids": [[1, 2], [4, 5], [8]],
            "attention_mask": [[0, 1], [0, 0], [1]],
        }
        dataset = truncate_dataset(dataset, max_length)
        num_examples = len(examples[next(iter(examples))])
        assert next(iter(dataset.with_format(None).batch(batch_size=num_examples))) == expected_output
        assert formatting == dataset._formatting

    def test_with_extra_column(self):
        # [FUNCTION DOC]
        # Checks that a non-tokenized extra string column passes through unchanged while token columns are truncated
        # to length 2. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        examples = {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
            "my_column": ["a", "b", "c"],
        }
        dataset = Dataset.from_dict(examples)
        max_length = 2
        expected_output = {
            "input_ids": [[1, 2], [4, 5], [8]],
            "attention_mask": [[0, 1], [0, 0], [1]],
            "my_column": ["a", "b", "c"],
        }
        dataset = truncate_dataset(dataset, max_length)
        assert dataset.to_dict() == expected_output


class TestTruncateRight(TrlTestCase):
    # [CLASS DOC]
    # Tests truncate_right, which cuts each row after its first stop token, pads the rest and returns a mask.
    # Inherits TrlTestCase.
    def test_truncates_after_the_first_of_any_stop_token(self):
        # [FUNCTION DOC]
        # Checks with stop tokens 2 and 3 and pad 0 that each row is cut after whichever stop token appears first
        # (row 1 on 2, row 2 on 3), tail filled with 0 and mask 1 up to and including the stop token. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        # The second row stops on 3, a stop token the first row does not use
        input_ids = torch.tensor([[5, 2, 7, 8], [5, 3, 7, 8]])
        output_ids, mask = truncate_right(input_ids, [2, 3], pad_token_id=0)
        assert output_ids.tolist() == [[5, 2, 0, 0], [5, 3, 0, 0]]
        assert mask.tolist() == [[1, 1, 0, 0], [1, 1, 0, 0]]

    def test_keeps_sequences_without_a_stop_token(self):
        # [FUNCTION DOC]
        # Checks a row without any stop token is returned unchanged with an all-ones mask. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        input_ids = torch.tensor([[5, 7, 8, 9]])
        output_ids, mask = truncate_right(input_ids, [2, 3], pad_token_id=0)
        assert output_ids.tolist() == [[5, 7, 8, 9]]
        assert mask.tolist() == [[1, 1, 1, 1]]


class TestPreparePeftModel(TrlTestCase):
    # [CLASS DOC]
    # Tests prepare_peft_model. Inherits TrlTestCase.
    @require_peft
    @require_bitsandbytes
    @require_torch_accelerator
    def test_qlora_bf16_yields_uniform_dtype(self):
        # [FUNCTION DOC]
        # Checks that a 4-bit nf4 (bf16 compute) tiny Qwen2 model prepared with a LoRA config (r 8 on q/k/v/o proj) and
        # bf16 TrainingArguments has no float32 parameters left. Requires PEFT, bitsandbytes and an accelerator.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: loads a quantized model on the accelerator.
        import torch
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments

        model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
            ),
            dtype=torch.bfloat16,
        )
        args = TrainingArguments(output_dir=self.tmp_dir, bf16=True, report_to=[])
        peft_config = LoraConfig(r=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
        model = prepare_peft_model(model, peft_config, args)

        fp32 = [name for name, param in model.named_parameters() if param.dtype == torch.float32]
        assert fp32 == [], f"expected no float32 params after prepare_peft_model, got e.g. {fp32[:5]}"


class TestReferenceModel(TrlTestCase):
    # [CLASS DOC]
    # Tests create_reference_model on tiny GPT2 (2 layers). State from setup_method: model, test_input (one
    # 4-token row), AdamW optimizer with lr 1, layer_format naming the attention c_attn weight of a layer.
    # Inherits TrlTestCase.
    def setup_method(self):
        # [FUNCTION DOC]
        # HELPER: Per-test fixture loading tiny GPT2, a fixed input, an AdamW optimizer (lr 1) and the parameter name template.
        # Inputs: none. Outputs: none. Exceptions: load errors propagate. Side effects: sets instance attributes.
        self.model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-GPT2LMHeadModel")
        self.test_input = torch.tensor([[0, 1, 2, 3]])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1)
        self.layer_format = "transformer.h.{layer}.attn.c_attn.weight"

    def test_independent_reference_models_are_equal(self):
        # [FUNCTION DOC]
        # Checks a fully independent reference model: before one optimizer step on the source model (LM loss on the
        # test input) both layers match the reference; after the step the reference's layers are unchanged while both
        # source layers changed. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        layer_0 = self.layer_format.format(layer=0)
        layer_1 = self.layer_format.format(layer=1)

        ref_model = create_reference_model(self.model)

        first_layer_before = self.model.get_parameter(layer_0).data.clone()
        last_layer_before = self.model.get_parameter(layer_1).data.clone()  # the model only has 2 layers

        first_ref_layer_before = ref_model.get_parameter(layer_0).data.clone()
        last_ref_layer_before = ref_model.get_parameter(layer_1).data.clone()

        output = self.model(input_ids=self.test_input, labels=self.test_input)
        output.loss.backward()
        self.optimizer.step()

        first_layer_after = self.model.get_parameter(layer_0).data.clone()
        last_layer_after = self.model.get_parameter(layer_1).data.clone()

        first_ref_layer_after = ref_model.get_parameter(layer_0).data.clone()
        last_ref_layer_after = ref_model.get_parameter(layer_1).data.clone()

        # before optimization ref and model are identical
        assert (first_layer_before == first_ref_layer_before).all()
        assert (last_layer_before == last_ref_layer_before).all()

        # ref model stays identical after optimization
        assert (first_ref_layer_before == first_ref_layer_after).all()
        assert (last_ref_layer_before == last_ref_layer_after).all()

        # optimized model changes
        assert not (first_layer_before == first_layer_after).all()
        assert not (last_layer_before == last_layer_after).all()

    def test_shared_layers(self):
        # [FUNCTION DOC]
        # Checks num_shared_layers 1: after an optimizer step, the reference stays identical to its initial values, the
        # source model's first (shared, frozen) layer is unchanged, and its second layer changed. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        layer_0 = self.layer_format.format(layer=0)
        layer_1 = self.layer_format.format(layer=1)

        ref_model = create_reference_model(self.model, num_shared_layers=1)

        first_layer_before = self.model.get_parameter(layer_0).data.clone()
        second_layer_before = self.model.get_parameter(layer_1).data.clone()

        first_ref_layer_before = ref_model.get_parameter(layer_0).data.clone()
        second_ref_layer_before = ref_model.get_parameter(layer_1).data.clone()

        output = self.model(input_ids=self.test_input, labels=self.test_input)
        output.loss.backward()
        self.optimizer.step()

        first_layer_after = self.model.get_parameter(layer_0).data.clone()
        second_layer_after = self.model.get_parameter(layer_1).data.clone()

        first_ref_layer_after = ref_model.get_parameter(layer_0).data.clone()
        second_ref_layer_after = ref_model.get_parameter(layer_1).data.clone()

        # before optimization ref and model are identical
        assert (first_layer_before == first_ref_layer_before).all()
        assert (second_layer_before == second_ref_layer_before).all()

        # ref model stays identical after optimization
        assert (first_ref_layer_before == first_ref_layer_after).all()
        assert (second_ref_layer_before == second_ref_layer_after).all()

        # first layer of optimized model stays the same
        assert (first_layer_before == first_layer_after).all()

        # other layers in optimized model change
        assert not (second_layer_before == second_layer_after).all()

    def test_shared_layers_share_memory(self):
        # [FUNCTION DOC]
        # Checks the shared first layer of the reference has the same data pointer as the source (no deepcopy, issue
        # 2904) while the unshared second layer has a different pointer. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        # Shared layers must reference the same storage as the source model, not a `deepcopy` duplicate,
        # so they are held in memory only once (see issue #2904).
        layer_0 = self.layer_format.format(layer=0)
        layer_1 = self.layer_format.format(layer=1)

        ref_model = create_reference_model(self.model, num_shared_layers=1)

        # the shared layer points at the same storage as the source model
        assert ref_model.get_parameter(layer_0).data_ptr() == self.model.get_parameter(layer_0).data_ptr()
        # an unshared layer is an independent copy
        assert ref_model.get_parameter(layer_1).data_ptr() != self.model.get_parameter(layer_1).data_ptr()
