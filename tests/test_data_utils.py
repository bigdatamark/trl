# [MODULE DOC]
# Tests for trl.data_utils: prepare_multimodal_messages, is_conversational, is_conversational_from_value,
# apply_chat_template / maybe_apply_chat_template (incl. tools, chat_template_kwargs and the Harmony format of
# GPT-OSS), unpair_preference_dataset / maybe_unpair_preference_dataset, extract_prompt / maybe_extract_prompt,
# pack_dataset (wrapped and bfd strategies) and maybe_convert_to_chatml.
# Main contents: one test class per utility using hand-written example dicts and small in-memory Datasets, plus
# many tiny tokenizers for chat template application.
# Import-time side effects: imports PIL.Image only if vision is available.

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

import textwrap
from time import strftime

import pytest
import transformers
from datasets import Dataset, DatasetDict
from packaging.version import Version
from transformers import AutoProcessor, AutoTokenizer, is_vision_available

from trl.data_utils import (
    apply_chat_template,
    extract_prompt,
    is_conversational,
    is_conversational_from_value,
    maybe_apply_chat_template,
    maybe_convert_to_chatml,
    maybe_extract_prompt,
    maybe_unpair_preference_dataset,
    pack_dataset,
    prepare_multimodal_messages,
    unpair_preference_dataset,
)

from .testing_utils import TrlTestCase, require_vision


if is_vision_available():
    from PIL import Image


@require_vision
class TestPrepareMultimodalMessages:
    # [CLASS DOC]
    # Tests prepare_multimodal_messages, which converts string message contents into lists of typed blocks (text,
    # image) and inserts the provided images into the first user message. Requires vision. Plain pytest class.
    def test_basic_user_assistant_conversation(self):
        # [FUNCTION DOC]
        # Checks a user/assistant conversation with one image becomes block lists: user gets the image block followed by
        # the text block, assistant gets a text block. Uses 10x10 PIL RGB images. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        """Test basic conversation with user and assistant messages."""
        messages = [
            {"role": "user", "content": "What color is the sky?"},
            {"role": "assistant", "content": "It is blue."},
        ]
        image = Image.new("RGB", (10, 10), color="blue")
        messages = prepare_multimodal_messages(messages, images=[image])

        expected = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": "What color is the sky?"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "It is blue."}],
            },
        ]

        assert messages == expected

    def test_first_user_message_gets_image(self):
        # [FUNCTION DOC]
        # Checks only the first user message receives the image block; the later user message gets only a text block. Uses 10x10 PIL RGB images. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        """Test that only the first user message gets an image."""
        messages = [
            {"role": "user", "content": "What color is the sky?"},
            {"role": "assistant", "content": "It is blue."},
            {"role": "user", "content": "How about the grass?"},
        ]

        image = Image.new("RGB", (10, 10), color="blue")
        messages = prepare_multimodal_messages(messages, images=[image])

        expected = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": "What color is the sky?"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "It is blue."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "How about the grass?"}],
            },
        ]

        assert messages == expected

    def test_multiple_images(self):
        # [FUNCTION DOC]
        # Checks three images are all inserted, in order, before the text in the first user message. Uses 10x10 PIL RGB images. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        """Test that multiple images are added to the first user message."""
        messages = [
            {"role": "user", "content": "What color is the sky?"},
            {"role": "assistant", "content": "It is blue."},
        ]
        images = [Image.new("RGB", (10, 10), color=color) for color in ["red", "green", "blue"]]
        messages = prepare_multimodal_messages(messages, images=images)

        expected = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": images[0]},
                    {"type": "image", "image": images[1]},
                    {"type": "image", "image": images[2]},
                    {"type": "text", "text": "What color is the sky?"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "It is blue."}],
            },
        ]

        assert messages == expected

    def test_system_message_transformation(self):
        # [FUNCTION DOC]
        # Checks a string system message becomes a single text block while the user message gets the image. Uses 10x10 PIL RGB images. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        """Test that system messages are properly transformed."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What color is the sky?"},
        ]

        image = Image.new("RGB", (10, 10), color="blue")
        messages = prepare_multimodal_messages(messages, images=[image])

        expected = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant"}],
            },
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": "What color is the sky?"}],
            },
        ]

        assert messages == expected

    def test_already_prepared_messages_unchanged(self):
        # [FUNCTION DOC]
        # Checks already block-formatted messages keep their structure, with the placeholder image block of the user
        # message filled with the provided image object. Uses 10x10 PIL RGB images.
        # KNOWN ISSUE: the docstring says messages with list content are "not modified", but the test expects the image
        # placeholder to be filled in with the provided image.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        """Test that messages with list content are not modified."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant"}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is the sky?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "It is blue."}]},
        ]

        image = Image.new("RGB", (10, 10), color="blue")
        messages = prepare_multimodal_messages(messages, images=[image])

        expected = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant"}],
            },
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": "What color is the sky?"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "It is blue."}],
            },
        ]

        assert messages == expected

    def test_mixed_prepared_and_unprepared_messages(self):
        # [FUNCTION DOC]
        # Checks a mix of string and block contents: strings are converted, the existing assistant block list is kept, and
        # only the first user message gets the image. Uses 10x10 PIL RGB images. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        """Test handling of mixed prepared and unprepared messages."""
        messages = [
            {"role": "user", "content": "What color is the sky?"},
            {"role": "assistant", "content": [{"type": "text", "text": "It is blue."}]},
            {"role": "user", "content": "What about the grass?"},
        ]

        image = Image.new("RGB", (10, 10), color="blue")
        messages = prepare_multimodal_messages(messages, images=[image])

        expected = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": "What color is the sky?"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "It is blue."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "What about the grass?"}],
            },
        ]

        assert messages == expected

    def test_message_with_tool_calling_turns(self):
        # [FUNCTION DOC]
        # Checks with no images that the assistant tool-call message (no content) is left as is, the tool message's string
        # content becomes a text block (name kept), and the other messages become text blocks. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        """Test that both the assistant tool call and the tool role turns messages are properly transformed."""
        messages = [
            {"role": "user", "content": "What's the weather like in New York?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "tool",
                        "function": {"name": "get_current_weather", "arguments": {"location": "New York"}},
                    }
                ],
            },
            {"role": "tool", "name": "get_current_weather", "content": "22.0"},
            {"role": "assistant", "content": "The current weather in New York is 22.0 degrees Celsius."},
        ]

        messages = prepare_multimodal_messages(messages)

        expected = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What's the weather like in New York?"}],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "tool",
                        "function": {"name": "get_current_weather", "arguments": {"location": "New York"}},
                    }
                ],
            },
            {"role": "tool", "name": "get_current_weather", "content": [{"type": "text", "text": "22.0"}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "The current weather in New York is 22.0 degrees Celsius."}],
            },
        ]

        assert messages == expected

    def test_prepared_image_blocks_without_new_images(self):
        # [FUNCTION DOC]
        # Checks that with no images passed, an existing image block carrying an image payload is preserved, and the string
        # assistant content is converted to a text block. Uses 10x10 PIL RGB images. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        """Test that existing image payloads are preserved when no new images are provided."""
        image = Image.new("RGB", (10, 10), color="blue")
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": "What color is the sky?"}],
            },
            {"role": "assistant", "content": "It is blue."},
        ]

        messages = prepare_multimodal_messages(messages)

        expected = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": "What color is the sky?"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "It is blue."}]},
        ]

        assert messages == expected


class TestIsConversational(TrlTestCase):
    # [CLASS DOC]
    # Tests is_conversational over class-level lists of examples: conversational examples cover language modeling,
    # prompt-only, prompt-completion, preference (explicit/implicit prompt, with tool calls and tools), unpaired
    # preference, and Harmony variants with system messages and thinking; non-conversational examples are their
    # plain-string counterparts. Inherits TrlTestCase.
    # fmt: off
    conversational_examples = [
        {  # Language modeling
            "messages": [
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "It is blue."},
            ],
        },
        {  # Prompt-only
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
        },
        {  # Prompt-completion
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
            "completion": [{"role": "assistant", "content": "It is blue."}],
        },
        {  # Preference
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
            "chosen": [{"role": "assistant", "content": "It is blue."}],
            "rejected": [{"role": "assistant", "content": "It is green."}],
        },
        {  # Preference with implicit prompt
            "chosen": [
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "It is blue."},
            ],
            "rejected": [
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "It is green."},
            ],
        },
        {  # Preference with tool calls
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
            "chosen": [
                {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "get_color", "arguments": {"what": "sky"}}}]},
                {"role": "tool", "name": "get_color", "content": "blue"},
                {"role": "assistant", "content": "It is blue."},
            ],
            "rejected": [
                {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "get_color", "arguments": {"what": "tree"}}}]},
                {"role": "tool", "name": "get_color", "content": "green"},
                {"role": "assistant", "content": "It is green."},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "description": "Gets the color.",
                        "name": "get_color",
                        "parameters": {"properties": {"what": {"description": "What to get the color of.", "type": "string"}}, "required": ["what"], "type": "object"},
                        "return": {"description": "The color.", "type": "string"},
                    },
                },
            ],
        },
        {  # Unpaired preference
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
            "completion": [{"role": "assistant", "content": "It is blue."}],
            "label": True,
        },
        {  # Language modeling with harmony
            "messages": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "thinking": "The user asks the color of the sky...", "content": "It is blue."},
            ],
        },
        {  # Prompt-only with harmony
            "prompt": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
            ],
        },
        {  # Prompt-completion with harmony
            "prompt": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
            ],
            "completion": [
                {"role": "assistant", "thinking": "The user asks the color of the sky...", "content": "It is blue."},
            ],
        },
        {  # Preference with harmony
            "prompt": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
            ],
            "chosen": [
                {"role": "assistant", "thinking": "The user asks the color of the sky...", "content": "It is blue."},
            ],
            "rejected": [
                {"role": "assistant", "thinking": "The user asks the color of the tree...", "content": "It is green."},
            ],
        },
        {  # Preference with implicit prompt and harmony
            "chosen": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "thinking": "The user asks the color of the sky...", "content": "It is blue."},
            ],
            "rejected": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "thinking": "The user asks the color of the tree...", "content": "It is green."},
            ],
        },
        {  # Unpaired preference with harmony
            "prompt": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
            ],
            "completion": [
                {"role": "assistant", "thinking": "The user asks the color of the sky...", "content": "It is blue."},
            ],
            "label": True,
        },
    ]
    # fmt: on

    non_conversational_examples = [
        {"prompt": "The sky is", "completion": " blue."},
        {"text": "The sky is blue."},
        {"prompt": "The sky is"},
        {"prompt": "The sky is", "chosen": " blue.", "rejected": " green."},
        {"prompt": "The sky is", "completion": " blue.", "label": True},
    ]

    @pytest.mark.parametrize("example", conversational_examples)
    def test_conversational(self, example):
        # [FUNCTION DOC]
        # Parametrized over conversational_examples: asserts is_conversational is truthy.
        # Inputs: example. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        assert is_conversational(example)

    @pytest.mark.parametrize("example", non_conversational_examples)
    def test_non_conversational(self, example):
        # [FUNCTION DOC]
        # Parametrized over non_conversational_examples: asserts is_conversational is falsy.
        # Inputs: example. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        assert not is_conversational(example)


class TestIsConversationalFromValue(TrlTestCase):
    # [CLASS DOC]
    # Tests is_conversational_from_value, which detects ShareGPT-style "conversations" with from/value keys.
    # Inherits TrlTestCase.
    def test_positive_1(self):
        # [FUNCTION DOC]
        # Asserts True for a "conversations" list of from/value dicts. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        example = {
            "conversations": [
                {"from": "user", "value": "What color is the sky?"},
                {"from": "assistant", "value": "It is blue."},
            ],
        }
        assert is_conversational_from_value(example)

    def test_negative_1(self):
        # [FUNCTION DOC]
        # Asserts False for a ChatML "messages" list of role/content dicts. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        example = {
            "messages": [
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "It is blue."},
            ],
        }
        assert not is_conversational_from_value(example)

    def test_negative_2(self):
        # [FUNCTION DOC]
        # Asserts False for a plain text example. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        example = {"text": "The sky is blue."}
        assert not is_conversational_from_value(example)


class TestApplyChatTemplate(TrlTestCase):
    # [CLASS DOC]
    # Tests apply_chat_template / maybe_apply_chat_template over a class-level list of ~30 tiny tokenizers (some
    # version-gated) and lists of conversational and non-conversational examples for each dataset type.
    # Inherits TrlTestCase.
    tokenizers = [
        "trl-internal-testing/tiny-CohereForCausalLM",
        "trl-internal-testing/tiny-Cohere2ForCausalLM",
        "trl-internal-testing/tiny-DeepseekV3ForCausalLM",
        "trl-internal-testing/tiny-DeepseekV3ForCausalLM-0528",
        "trl-internal-testing/tiny-FalconMambaForCausalLM",
        "trl-internal-testing/tiny-Gemma2ForCausalLM",
        "trl-internal-testing/tiny-GemmaForCausalLM",
        "trl-internal-testing/tiny-GptOssForCausalLM",
        pytest.param(
            "trl-internal-testing/tiny-Glm4MoeForCausalLM",
            marks=pytest.mark.skipif(
                Version(transformers.__version__) < Version("5.0.0"),
                reason="GLM4 tokenizer requires transformers>=5.0.0",
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
        "trl-internal-testing/tiny-LlamaForCausalLM-3.1",
        "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
        "trl-internal-testing/tiny-LlamaForCausalLM-3",
        "trl-internal-testing/tiny-MistralForCausalLM-0.1",
        "trl-internal-testing/tiny-MistralForCausalLM-0.2",
        pytest.param(
            "trl-internal-testing/tiny-NemotronHForCausalLM-nano",
            marks=pytest.mark.skipif(
                Version(transformers.__version__) < Version("5.3.0"),
                reason="Nemotron 3 tokenizer requires transformers>=5.3.0",
            ),
        ),
        pytest.param(
            "trl-internal-testing/tiny-NemotronHForCausalLM-super",
            marks=pytest.mark.skipif(
                Version(transformers.__version__) < Version("5.3.0"),
                reason="Nemotron 3 tokenizer requires transformers>=5.3.0",
            ),
        ),
        pytest.param(
            "trl-internal-testing/tiny-NemotronHForCausalLM-ultra",
            marks=pytest.mark.skipif(
                Version(transformers.__version__) < Version("5.3.0"),
                reason="Nemotron 3 tokenizer requires transformers>=5.3.0",
            ),
        ),
        pytest.param(
            "trl-internal-testing/tiny-NemotronHForCausalLM-3.5-lightning",
            marks=pytest.mark.skipif(
                Version(transformers.__version__) < Version("5.3.0"),
                reason="Nemotron 3.5 tokenizer requires transformers>=5.3.0",
            ),
        ),
        pytest.param(
            "trl-internal-testing/tiny-Olmo3ForCausalLM",
            marks=pytest.mark.skipif(
                Version(transformers.__version__) < Version("4.57.0"),
                reason="Olmo 3 requires transformers>=4.57.0",
            ),
        ),
        "trl-internal-testing/tiny-Phi3ForCausalLM-3",
        "trl-internal-testing/tiny-Phi3ForCausalLM-3.5",
        "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        "trl-internal-testing/tiny-Qwen3ForCausalLM",
        "trl-internal-testing/tiny-Qwen3ForCausalLM-Instruct-2507",
        pytest.param(
            "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration-NoThink",
            marks=pytest.mark.skipif(
                Version(transformers.__version__) < Version("5.0.0"),
                reason="Qwen3.5 tokenizer requires transformers>=5.0.0",
            ),
        ),
        pytest.param(
            "trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6",
            marks=pytest.mark.skipif(
                Version(transformers.__version__) < Version("5.0.0"),
                reason="Qwen3.5 tokenizer requires transformers>=5.0.0",
            ),
        ),
        pytest.param(
            "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration-3.8",
            marks=pytest.mark.skipif(
                Version(transformers.__version__) < Version("5.0.0"),
                reason="Qwen3.5 tokenizer requires transformers>=5.0.0",
            ),
        ),
    ]

    conversational_examples = [
        {  # Language modeling
            "messages": [
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "It is blue."},
            ],
        },
        {  # Prompt-only
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
        },
        {  # Prompt-completion
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
            "completion": [{"role": "assistant", "content": "It is blue."}],
        },
        {  # Preference
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
            "chosen": [{"role": "assistant", "content": "It is blue."}],
            "rejected": [{"role": "assistant", "content": "It is green."}],
        },
        {  # Preference with implicit prompt
            "chosen": [
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "It is blue."},
            ],
            "rejected": [
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "It is green."},
            ],
        },
        {  # Unpaired preference
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
            "completion": [{"role": "assistant", "content": "It is blue."}],
            "label": True,
        },
    ]

    non_conversational_examples = [
        {"text": "The sky is blue."},  # Language modeling
        {"prompt": "The sky is"},  # Prompt-only
        {"prompt": "The sky is", "completion": " blue."},  # Prompt-completion
        {"prompt": "The sky is", "chosen": " blue.", "rejected": " green."},  # Preference
        {"chosen": "The sky is blue.", "rejected": "The sky is green."},  # Preference with implicit prompt
        {"prompt": "The sky is", "completion": " blue.", "label": True},  # Unpaired preference
    ]

    @pytest.mark.parametrize("example", conversational_examples)
    @pytest.mark.parametrize("tokenizer_id", tokenizers)
    def test_apply_chat_template(self, tokenizer_id, example):
        # [FUNCTION DOC]
        # Parametrized over tokenizers x conversational examples: applies apply_chat_template. Shared assertions: result is a dict; each of prompt/chosen/rejected/completion present in the input is a string in the output; a messages input yields a string "text" key; a label is kept as the same bool.
        # Inputs: tokenizer_id, example. Outputs: none. Exceptions: assertion failure. Side effects: loads tokenizers from the Hub.
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        result = apply_chat_template(example, tokenizer)

        # Checking if the result is a dictionary
        assert isinstance(result, dict)

        # The chat template should be applied to the following keys
        for key in ["prompt", "chosen", "rejected", "completion"]:
            if key in example:
                assert key in result
                assert isinstance(result[key], str)

        # Exception for messages, the key is "text" once the chat template is applied
        if "messages" in example:
            assert "text" in result
            assert isinstance(result["text"], str)

        # The label should be kept
        if "label" in example:
            assert "label" in result
            assert isinstance(result["label"], bool)
            assert result["label"] == example["label"]

    # both conversational and non-conversational examples
    @pytest.mark.parametrize("example", conversational_examples + non_conversational_examples)
    @pytest.mark.parametrize("tokenizer_id", tokenizers)
    def test_maybe_apply_chat_template(self, tokenizer_id, example):
        # [FUNCTION DOC]
        # Parametrized over tokenizers x (conversational plus non-conversational) examples: applies
        # maybe_apply_chat_template (non-conversational examples pass through). Shared assertions: result is a dict; each of prompt/chosen/rejected/completion present in the input is a string in the output; a messages input yields a string "text" key; a label is kept as the same bool.
        # Inputs: tokenizer_id, example. Outputs: none. Exceptions: assertion failure. Side effects: loads tokenizers from the Hub.
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        result = maybe_apply_chat_template(example, tokenizer)

        # Checking if the result is a dictionary
        assert isinstance(result, dict)

        # The chat template should be applied to the following keys
        for key in ["prompt", "chosen", "rejected", "completion"]:
            if key in example:
                assert key in result
                assert isinstance(result[key], str)

        # Exception for messages, the key is "text" once the chat template is applied
        if "messages" in example:
            assert "text" in result
            assert isinstance(result["text"], str)

        # The label should be kept
        if "label" in example:
            assert "label" in result
            assert isinstance(result["label"], bool)
            assert result["label"] == example["label"]

    def test_apply_chat_template_with_chat_template_kwargs(self):
        # [FUNCTION DOC]
        # Checks per-example chat_template_kwargs (enable_thinking False) are forwarded: the tiny Qwen3 prompt renders as the
        # user turn plus an assistant header followed by an empty think block. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: loads tokenizers from the Hub.
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM")

        example = {
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
            # with this tokenizer, when you pass enable_thinking=False, it will add "<think>\n\n</think>\n\n"
            "chat_template_kwargs": {"enable_thinking": False},
        }
        result = apply_chat_template(example, tokenizer)

        # docstyle-ignore
        expected = textwrap.dedent("""\
        <|im_start|>user
        What color is the sky?<|im_end|>
        <|im_start|>assistant
        <think>

        </think>

        """)

        assert result["prompt"] == expected

    def test_apply_chat_template_preference_round_trips_with_generation_prompt_insertion(self):
        # [FUNCTION DOC]
        # Regression test: with a custom template whose generation prompt appends an opening think tag, a preference
        # example where only chosen starts with a think block must satisfy prompt plus chosen equal to the full render of
        # prompt+chosen, and likewise for rejected (completions sliced against the final common-prefix prompt).
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: loads tokenizers from the Hub.
        # Regression test: when the generation prompt inserts extra tokens (e.g. an opening `<think>` tag, as in
        # DeepSeek-R1-style templates), the prompt is reduced to the common prefix of the rendered prompt+chosen and
        # prompt+rejected strings. The completions must be sliced against the *final* prompt; slicing `chosen` before
        # the prompt shrinks again against `rejected` used to silently drop the leading `<think>` tag from `chosen`.
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM")
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'] + ': ' + message['content'] + '\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ 'assistant: <think>' }}{% endif %}"
        )

        example = {
            "prompt": [{"role": "user", "content": "What is 2+2?"}],
            # chosen keeps its reasoning trace, rejected doesn't: the common prefix of the two rendered strings ends
            # before the `<think>` inserted by the generation prompt
            "chosen": [{"role": "assistant", "content": "<think>2+2=4</think>It is 4."}],
            "rejected": [{"role": "assistant", "content": "It is 4."}],
        }
        result = apply_chat_template(example, tokenizer)

        expected_chosen = tokenizer.apply_chat_template(example["prompt"] + example["chosen"], tokenize=False)
        expected_rejected = tokenizer.apply_chat_template(example["prompt"] + example["rejected"], tokenize=False)
        assert result["prompt"] + result["chosen"] == expected_chosen
        assert result["prompt"] + result["rejected"] == expected_rejected

    def test_apply_chat_template_with_tools(self):
        # [FUNCTION DOC]
        # Checks passing tools (a nested get_current_temperature function) to apply_chat_template with the Llama 3.2
        # processing class puts the tool name into the rendered prompt, and passing tools None leaves it out.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: loads tokenizers from the Hub.
        tokenizer = AutoProcessor.from_pretrained("trl-internal-testing/tiny-LlamaForCausalLM-3.2")

        # Define dummy test tools
        def get_current_temperature(location: str):
            # [FUNCTION DOC]
            # HELPER: Dummy tool whose docstring defines its schema; always returns 22.0.
            # Inputs: location - str (unused). Outputs: float. Exceptions/Side effects: none.
            """
            Gets the temperature at a given location.

            Args:
                location: The location to get the temperature for
            """
            return 22.0

        # Define test case
        test_case = {
            "prompt": [
                {"content": "What's the temperature in London?", "role": "user"},
            ]
        }
        # Test with tools
        result_with_tools = apply_chat_template(test_case, tokenizer, tools=[get_current_temperature])

        # Verify tools are included in the output
        assert "get_current_temperature" in result_with_tools["prompt"]

        # Test without tools
        result_without_tools = apply_chat_template(test_case, tokenizer, tools=None)

        # Verify tools are not included in the output
        assert "get_current_temperature" not in result_without_tools["prompt"]


class TestApplyChatTemplateHarmony(TrlTestCase):
    # [CLASS DOC]
    # Tests apply_chat_template output strings for the Harmony format (GPT-OSS) for each dataset type, including
    # analysis (thinking) and final channels. Inherits TrlTestCase.
    def test_language_modeling(self):
        # [FUNCTION DOC]
        # Messages example: asserts output["text"] equals the full Harmony rendering ending with the analysis channel
        # (thinking) and the final channel content. Uses the tiny GPT-OSS tokenizer with reasoning_effort "low" and model_identity "You are HuggingGPT."; the expected
        # Harmony header contains the identity, knowledge cutoff 2024-06, today's date (strftime), reasoning level, valid
        # channels line and the system message turned into developer instructions. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: loads tokenizers from the Hub.
        messages = {
            "messages": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "thinking": "The user asks the color of the sky...", "content": "It is blue."},
            ],
        }
        output = apply_chat_template(
            messages,
            processing_class=AutoTokenizer.from_pretrained("trl-internal-testing/tiny-GptOssForCausalLM"),
            reasoning_effort="low",
            model_identity="You are HuggingGPT.",
        )

        # docstyle-ignore
        expected = textwrap.dedent(f"""\
        <|start|>system<|message|>You are HuggingGPT.
        Knowledge cutoff: 2024-06
        Current date: {strftime("%Y-%m-%d")}

        Reasoning: low

        # Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions

        Respond in a friendly manner.

        <|end|><|start|>user<|message|>What color is the sky?<|end|><|start|>assistant<|channel|>analysis<|message|>The user asks the color of the sky...<|end|><|start|>assistant<|channel|>final<|message|>It is blue.<|return|>""")

        assert output["text"] == expected

    def test_prompt_only(self):
        # [FUNCTION DOC]
        # Prompt-only example: asserts the prompt ends with the assistant start marker (generation prompt). Uses the tiny GPT-OSS tokenizer with reasoning_effort "low" and model_identity "You are HuggingGPT."; the expected
        # Harmony header contains the identity, knowledge cutoff 2024-06, today's date (strftime), reasoning level, valid
        # channels line and the system message turned into developer instructions. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: loads tokenizers from the Hub.
        messages = {
            "prompt": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
            ],
        }
        output = apply_chat_template(
            messages,
            processing_class=AutoTokenizer.from_pretrained("trl-internal-testing/tiny-GptOssForCausalLM"),
            reasoning_effort="low",
            model_identity="You are HuggingGPT.",
        )

        # docstyle-ignore
        expected = textwrap.dedent(f"""\
        <|start|>system<|message|>You are HuggingGPT.
        Knowledge cutoff: 2024-06
        Current date: {strftime("%Y-%m-%d")}

        Reasoning: low

        # Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions

        Respond in a friendly manner.

        <|end|><|start|>user<|message|>What color is the sky?<|end|><|start|>assistant""")

        assert output["prompt"] == expected

    def test_prompt_completion(self):
        # [FUNCTION DOC]
        # Prompt-completion example: asserts the prompt string and that the completion starts at the analysis channel and
        # contains the final channel answer. Uses the tiny GPT-OSS tokenizer with reasoning_effort "low" and model_identity "You are HuggingGPT."; the expected
        # Harmony header contains the identity, knowledge cutoff 2024-06, today's date (strftime), reasoning level, valid
        # channels line and the system message turned into developer instructions. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: loads tokenizers from the Hub.
        messages = {
            "prompt": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
            ],
            "completion": [
                {"role": "assistant", "thinking": "The user asks the color of the sky...", "content": "It is blue."},
            ],
        }
        output = apply_chat_template(
            messages,
            processing_class=AutoTokenizer.from_pretrained("trl-internal-testing/tiny-GptOssForCausalLM"),
            reasoning_effort="low",
            model_identity="You are HuggingGPT.",
        )

        # docstyle-ignore
        expected_prompt = textwrap.dedent(f"""\
        <|start|>system<|message|>You are HuggingGPT.
        Knowledge cutoff: 2024-06
        Current date: {strftime("%Y-%m-%d")}

        Reasoning: low

        # Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions

        Respond in a friendly manner.

        <|end|><|start|>user<|message|>What color is the sky?<|end|><|start|>assistant""")
        expected_completion = "<|channel|>analysis<|message|>The user asks the color of the sky...<|end|><|start|>assistant<|channel|>final<|message|>It is blue.<|return|>"

        assert output["prompt"] == expected_prompt
        assert output["completion"] == expected_completion

    def test_preference(self):
        # [FUNCTION DOC]
        # Preference example: asserts prompt, chosen and rejected strings (each completion has its own analysis and final
        # channels). Uses the tiny GPT-OSS tokenizer with reasoning_effort "low" and model_identity "You are HuggingGPT."; the expected
        # Harmony header contains the identity, knowledge cutoff 2024-06, today's date (strftime), reasoning level, valid
        # channels line and the system message turned into developer instructions. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: loads tokenizers from the Hub.
        messages = {
            "prompt": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
            ],
            "chosen": [
                {"role": "assistant", "thinking": "The user asks the color of the sky...", "content": "It is blue."},
            ],
            "rejected": [
                {"role": "assistant", "thinking": "The user asks the color of the tree...", "content": "It is green."},
            ],
        }
        output = apply_chat_template(
            messages,
            processing_class=AutoTokenizer.from_pretrained("trl-internal-testing/tiny-GptOssForCausalLM"),
            reasoning_effort="low",
            model_identity="You are HuggingGPT.",
        )

        # docstyle-ignore
        expected_prompt = textwrap.dedent(f"""\
        <|start|>system<|message|>You are HuggingGPT.
        Knowledge cutoff: 2024-06
        Current date: {strftime("%Y-%m-%d")}

        Reasoning: low

        # Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions

        Respond in a friendly manner.

        <|end|><|start|>user<|message|>What color is the sky?<|end|><|start|>assistant""")
        expected_chosen = "<|channel|>analysis<|message|>The user asks the color of the sky...<|end|><|start|>assistant<|channel|>final<|message|>It is blue.<|return|>"
        expected_rejected = "<|channel|>analysis<|message|>The user asks the color of the tree...<|end|><|start|>assistant<|channel|>final<|message|>It is green.<|return|>"

        assert output["prompt"] == expected_prompt
        assert output["chosen"] == expected_chosen
        assert output["rejected"] == expected_rejected

    def test_preference_with_implicit_prompt(self):
        # [FUNCTION DOC]
        # Implicit-prompt preference: asserts chosen and rejected each contain the full rendering including system/developer
        # and user parts. Uses the tiny GPT-OSS tokenizer with reasoning_effort "low" and model_identity "You are HuggingGPT."; the expected
        # Harmony header contains the identity, knowledge cutoff 2024-06, today's date (strftime), reasoning level, valid
        # channels line and the system message turned into developer instructions. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: loads tokenizers from the Hub.
        messages = {
            "chosen": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "thinking": "The user asks the color of the sky...", "content": "It is blue."},
            ],
            "rejected": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "thinking": "The user asks the color of the tree...", "content": "It is green."},
            ],
        }
        output = apply_chat_template(
            messages,
            processing_class=AutoTokenizer.from_pretrained("trl-internal-testing/tiny-GptOssForCausalLM"),
            reasoning_effort="low",
            model_identity="You are HuggingGPT.",
        )

        # docstyle-ignore
        expected_chosen = textwrap.dedent(f"""\
        <|start|>system<|message|>You are HuggingGPT.
        Knowledge cutoff: 2024-06
        Current date: {strftime("%Y-%m-%d")}

        Reasoning: low

        # Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions

        Respond in a friendly manner.

        <|end|><|start|>user<|message|>What color is the sky?<|end|><|start|>assistant<|channel|>analysis<|message|>The user asks the color of the sky...<|end|><|start|>assistant<|channel|>final<|message|>It is blue.<|return|>""")

        # docstyle-ignore
        expected_rejected = textwrap.dedent(f"""\
        <|start|>system<|message|>You are HuggingGPT.
        Knowledge cutoff: 2024-06
        Current date: {strftime("%Y-%m-%d")}

        Reasoning: low

        # Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions

        Respond in a friendly manner.

        <|end|><|start|>user<|message|>What color is the sky?<|end|><|start|>assistant<|channel|>analysis<|message|>The user asks the color of the tree...<|end|><|start|>assistant<|channel|>final<|message|>It is green.<|return|>""")

        assert output["chosen"] == expected_chosen
        assert output["rejected"] == expected_rejected

    def test_unpaired_preference(self):
        # [FUNCTION DOC]
        # Unpaired preference: asserts prompt and completion strings and that label stays truthy. Uses the tiny GPT-OSS tokenizer with reasoning_effort "low" and model_identity "You are HuggingGPT."; the expected
        # Harmony header contains the identity, knowledge cutoff 2024-06, today's date (strftime), reasoning level, valid
        # channels line and the system message turned into developer instructions. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: loads tokenizers from the Hub.
        messages = {
            "prompt": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
            ],
            "completion": [
                {"role": "assistant", "thinking": "The user asks the color of the sky...", "content": "It is blue."},
            ],
            "label": True,
        }
        output = apply_chat_template(
            messages,
            processing_class=AutoTokenizer.from_pretrained("trl-internal-testing/tiny-GptOssForCausalLM"),
            reasoning_effort="low",
            model_identity="You are HuggingGPT.",
        )

        # docstyle-ignore
        expected_prompt = textwrap.dedent(f"""\
        <|start|>system<|message|>You are HuggingGPT.
        Knowledge cutoff: 2024-06
        Current date: {strftime("%Y-%m-%d")}

        Reasoning: low

        # Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions

        Respond in a friendly manner.

        <|end|><|start|>user<|message|>What color is the sky?<|end|><|start|>assistant""")
        expected_completion = "<|channel|>analysis<|message|>The user asks the color of the sky...<|end|><|start|>assistant<|channel|>final<|message|>It is blue.<|return|>"

        assert output["prompt"] == expected_prompt
        assert output["completion"] == expected_completion
        assert output["label"]


class TestUnpairPreferenceDataset(TrlTestCase):
    # [CLASS DOC]
    # Tests unpair_preference_dataset / maybe_unpair_preference_dataset. Class attributes: paired_dataset (2 rows of
    # prompt/chosen/rejected) and unpaired_dataset (4 rows: chosen rows labelled True first, then rejected rows labelled
    # False). Inherits TrlTestCase.
    paired_dataset = Dataset.from_dict(
        {
            "prompt": ["The sky is", "The sun is"],
            "chosen": [" blue.", " in the sky."],
            "rejected": [" green.", " in the sea."],
        }
    )

    unpaired_dataset = Dataset.from_dict(
        {
            "prompt": ["The sky is", "The sun is", "The sky is", "The sun is"],
            "completion": [" blue.", " in the sky.", " green.", " in the sea."],
            "label": [True, True, False, False],
        }
    )

    def test_unpair_preference_dataset(self):
        # [FUNCTION DOC]
        # Asserts unpairing paired_dataset equals unpaired_dataset. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        # Test that a paired dataset is correctly converted to unpaired
        unpaired_dataset = unpair_preference_dataset(self.paired_dataset)
        assert unpaired_dataset.to_dict() == self.unpaired_dataset.to_dict(), (
            "The paired dataset should be converted to unpaired."
        )

    def test_unpair_preference_dataset_extra_columns(self):
        # [FUNCTION DOC]
        # Asserts an extra column is duplicated for the chosen and rejected halves (1, 2, 1, 2). Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        # Test that extra columns are preserved and duplicated for chosen and rejected rows
        paired_dataset = Dataset.from_dict(
            {
                "prompt": ["The sky is", "The sun is"],
                "chosen": [" blue.", " in the sky."],
                "rejected": [" green.", " in the sea."],
                "extra": [1, 2],
            }
        )
        unpaired_dataset = unpair_preference_dataset(paired_dataset)
        expected = {**self.unpaired_dataset.to_dict(), "extra": [1, 2, 1, 2]}
        assert unpaired_dataset.to_dict() == expected

    def test_unpair_preference_dataset_iterable(self):
        # [FUNCTION DOC]
        # Asserts unpairing an IterableDataset yields rows equal to the unpaired_dataset rows. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        # Test that an IterableDataset with extra columns is correctly unpaired
        paired_dataset = self.paired_dataset.to_iterable_dataset()
        unpaired_dataset = unpair_preference_dataset(paired_dataset)
        assert list(unpaired_dataset) == [
            dict(zip(self.unpaired_dataset.column_names, vals, strict=False))
            for vals in zip(*self.unpaired_dataset.to_dict().values(), strict=False)
        ]

    def test_unpair_preference_dataset_iterable_extra_columns(self):
        # [FUNCTION DOC]
        # Asserts an IterableDataset with an extra column yields unpaired rows with the extra column duplicated. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        # Test that an IterableDataset with extra columns preserves and duplicates them
        paired_iterable = Dataset.from_dict(
            {
                "prompt": ["The sky is", "The sun is"],
                "chosen": [" blue.", " in the sky."],
                "rejected": [" green.", " in the sea."],
                "extra": [1, 2],
            }
        ).to_iterable_dataset()
        unpaired_dataset = unpair_preference_dataset(paired_iterable)
        expected = {**self.unpaired_dataset.to_dict(), "extra": [1, 2, 1, 2]}
        assert list(unpaired_dataset) == [
            dict(zip(expected.keys(), vals, strict=False)) for vals in zip(*expected.values(), strict=False)
        ]

    def test_unpair_preference_dataset_dict(self):
        # [FUNCTION DOC]
        # Asserts a DatasetDict is unpaired split by split. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        # Test that a paired dataset dict is correctly converted to unpaired
        paired_dataset_dict = DatasetDict({"abc": self.paired_dataset})
        unpaired_dataset_dict = unpair_preference_dataset(paired_dataset_dict)
        assert unpaired_dataset_dict["abc"].to_dict() == self.unpaired_dataset.to_dict(), (
            "The paired dataset should be converted to unpaired."
        )

    def test_maybe_unpair_preference_dataset(self):
        # [FUNCTION DOC]
        # Asserts maybe_unpair_preference_dataset unpairs a paired Dataset. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        # Test that a paired dataset is correctly converted to unpaired with maybe_unpair_preference_dataset
        unpaired_dataset = maybe_unpair_preference_dataset(self.paired_dataset)
        assert unpaired_dataset.to_dict() == self.unpaired_dataset.to_dict(), (
            "The paired dataset should be converted to unpaired."
        )

    def test_maybe_unpair_preference_dataset_dict(self):
        # [FUNCTION DOC]
        # Asserts maybe_unpair_preference_dataset unpairs a paired DatasetDict. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        # Test that a paired dataset dict is correctly converted to unpaired with maybe_unpair_preference_dataset
        paired_dataset_dict = DatasetDict({"abc": self.paired_dataset})
        unpaired_dataset_dict = maybe_unpair_preference_dataset(paired_dataset_dict)
        assert unpaired_dataset_dict["abc"].to_dict() == self.unpaired_dataset.to_dict(), (
            "The paired dataset should be converted to unpaired."
        )

    def test_maybe_unpair_preference_dataset_already_paired(self):
        # [FUNCTION DOC]
        # Asserts an already unpaired Dataset is returned unchanged.
        # KNOWN ISSUE: the name and comment say "already paired" while the input is the already unpaired dataset. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        # Test that a paired dataset remains unchanged with maybe_unpair_preference_dataset
        unpaired_dataset = maybe_unpair_preference_dataset(self.unpaired_dataset)
        assert unpaired_dataset.to_dict() == self.unpaired_dataset.to_dict(), (
            "The unpaired dataset should remain unchanged."
        )

    def test_maybe_unpair_preference_dataset_dict_already_paired(self):
        # [FUNCTION DOC]
        # Asserts an already unpaired DatasetDict is returned unchanged.
        # KNOWN ISSUE: the name and comment say "already paired" while the input is already unpaired. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        # Test that a paired dataset dict remains unchanged with maybe_unpair_preference_dataset
        unpaired_dataset_dict = maybe_unpair_preference_dataset(DatasetDict({"abc": self.unpaired_dataset}))
        assert unpaired_dataset_dict["abc"].to_dict() == self.unpaired_dataset.to_dict(), (
            "The unpaired dataset should remain unchanged."
        )


class TestExtractPrompt(TrlTestCase):
    # [CLASS DOC]
    # Tests extract_prompt / maybe_extract_prompt, which move the shared leading part of chosen and rejected into a
    # "prompt" field. Class attributes hold implicit and explicit examples in conversational and standard form.
    # Inherits TrlTestCase.
    example_implicit_prompt_conversational = {
        "chosen": [
            {"role": "user", "content": "What color is the sky?"},
            {"role": "assistant", "content": "It is blue."},
        ],
        "rejected": [
            {"role": "user", "content": "What color is the sky?"},
            {"role": "assistant", "content": "It is green."},
        ],
    }

    example_explicit_prompt_conversational = {
        "prompt": [
            {"role": "user", "content": "What color is the sky?"},
        ],
        "chosen": [
            {"role": "assistant", "content": "It is blue."},
        ],
        "rejected": [
            {"role": "assistant", "content": "It is green."},
        ],
    }

    example_implicit_prompt_standard = {
        "chosen": "The sky is blue.",
        "rejected": "The sky is green.",
    }

    example_explicit_prompt_standard = {
        "prompt": "The sky is",
        "chosen": " blue.",
        "rejected": " green.",
    }

    def test_extract_prompt_conversational(self):
        # [FUNCTION DOC]
        # Asserts extract_prompt on the implicit conversational example yields the explicit form (user turn as prompt). Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        # Test that the prompt is correctly extracted from the dataset
        example_extracted_prompt = extract_prompt(self.example_implicit_prompt_conversational)
        assert example_extracted_prompt == self.example_explicit_prompt_conversational, (
            "The prompt is not correctly extracted from the dataset."
        )

    def test_maybe_extract_prompt_conversational(self):
        # [FUNCTION DOC]
        # Same as above via maybe_extract_prompt. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        # Test that the prompt is correctly extracted from the dataset with maybe_extract_prompt
        example_extracted_prompt = maybe_extract_prompt(self.example_implicit_prompt_conversational)
        assert example_extracted_prompt == self.example_explicit_prompt_conversational, (
            "The prompt is not correctly extracted from the dataset."
        )

    def test_maybe_extract_prompt_conversational_already_explicit(self):
        # [FUNCTION DOC]
        # Asserts maybe_extract_prompt leaves an explicit conversational example unchanged. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        # Test that the prompt remains unchanged with maybe_extract_prompt
        example_extracted_prompt = maybe_extract_prompt(self.example_explicit_prompt_conversational)
        assert example_extracted_prompt == self.example_explicit_prompt_conversational, (
            "The prompt should remain unchanged."
        )

    def test_extract_prompt_standard(self):
        # [FUNCTION DOC]
        # Asserts extract_prompt on "The sky is blue."/"The sky is green." gives prompt "The sky is" and completions " blue."
        # /" green.". Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        # Test that the prompt is correctly extracted from the dataset
        example_extracted_prompt = extract_prompt(self.example_implicit_prompt_standard)
        assert example_extracted_prompt == self.example_explicit_prompt_standard, (
            "The prompt is not correctly extracted from the dataset."
        )

    def test_maybe_extract_prompt_standard(self):
        # [FUNCTION DOC]
        # Same as above via maybe_extract_prompt. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        # Test that the prompt is correctly extracted from the dataset with maybe_extract_prompt
        example_extracted_prompt = maybe_extract_prompt(self.example_implicit_prompt_standard)
        assert example_extracted_prompt == self.example_explicit_prompt_standard, (
            "The prompt is not correctly extracted from the dataset."
        )

    def test_maybe_extract_prompt_standard_already_explicit(self):
        # [FUNCTION DOC]
        # Asserts maybe_extract_prompt leaves an explicit standard example unchanged. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        # Test that the prompt remains unchanged with maybe_extract_prompt
        example_extracted_prompt = maybe_extract_prompt(self.example_explicit_prompt_standard)
        assert example_extracted_prompt == self.example_explicit_prompt_standard, "The prompt should remain unchanged."


class TestPackDatasetWrapped(TrlTestCase):
    # [CLASS DOC]
    # Tests pack_dataset with strategy "wrapped" (concatenate all sequences and cut into fixed-length chunks).
    # Inherits TrlTestCase.
    def test_with_dataset(self):
        # [FUNCTION DOC]
        # Packs three rows into length-3 chunks (last chunk shorter) for input_ids and attention_mask, and asserts the
        # numpy/float32 dataset format is preserved. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        examples = {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
        }
        dataset = Dataset.from_dict(examples)
        dataset = dataset.with_format("numpy", dtype="float32")
        format = dataset.format
        seq_length = 3
        expected_output = {
            "input_ids": [[1, 2, 3], [4, 5, 6], [7, 8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1], [1, 1]],
        }
        dataset = pack_dataset(dataset, seq_length, strategy="wrapped")
        assert dataset.to_dict() == expected_output
        assert format == dataset.format

    def test_with_iterable_dataset(self):
        # [FUNCTION DOC]
        # Same packing on a numpy-formatted IterableDataset; asserts the batched output and the private _formatting are
        # preserved. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        examples = {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
        }
        dataset = Dataset.from_dict(examples).to_iterable_dataset()
        dataset = dataset.with_format("numpy")
        formatting = dataset._formatting
        seq_length = 3
        expected_output = {
            "input_ids": [[1, 2, 3], [4, 5, 6], [7, 8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1], [1, 1]],
        }
        dataset = pack_dataset(dataset, seq_length, strategy="wrapped")
        num_examples = len(examples[next(iter(examples))])
        assert next(iter(dataset.with_format(None).batch(batch_size=num_examples))) == expected_output
        assert formatting == dataset._formatting

    def test_with_multiple_map_batches(self):
        # [FUNCTION DOC]
        # Packs six length-2 rows to length 4 with map batch_size 3: each map batch is packed independently, giving
        # chunks of 4 then 2 per batch (regression for batches starting at non-zero offsets). Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        # `map` passes each batch as a slice, so every batch but the first starts at a non-zero offset
        examples = {
            "input_ids": [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]],
            "attention_mask": [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]],
        }
        dataset = Dataset.from_dict(examples)
        seq_length = 4
        expected_output = {
            "input_ids": [[1, 2, 3, 4], [5, 6], [7, 8, 9, 10], [11, 12]],
            "attention_mask": [[0, 1, 1, 0], [0, 1], [1, 0, 0, 1], [1, 0]],
        }
        dataset = pack_dataset(dataset, seq_length, strategy="wrapped", map_kwargs={"batch_size": 3})
        assert dataset.to_dict() == expected_output


class TestPackDatasetBfd(TrlTestCase):
    # [CLASS DOC]
    # Tests pack_dataset with best-fit-decreasing strategies: "bfd" (overlong sequences truncated) and "bfd_split"
    # (overlong sequences split into pieces), both adding a seq_lengths column. Inherits TrlTestCase.
    def test_with_dataset(self):
        # [FUNCTION DOC]
        # Packs rows of lengths 3, 4, 1 into bins of 4: the length-4 row alone, then 3 plus 1, with seq_lengths (4) and
        # (3, 1); checks seq_lengths appears in the output format's columns and compares formats.
        # KNOWN ISSUE: the variable named expected_format is the packed dataset's own format; seq_lengths is removed from it
        # in place before comparing the original format with the packed dataset's format, so the final check relies on
        # that in-place removal and is not an independent expectation.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        examples = {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
        }
        dataset = Dataset.from_dict(examples)
        dataset = dataset.with_format("numpy", dtype="float32")
        format = dataset.format
        seq_length = 4
        expected_output = {
            "input_ids": [[4, 5, 6, 7], [1, 2, 3, 8]],
            "seq_lengths": [[4], [3, 1]],
        }
        dataset = pack_dataset(dataset, seq_length, strategy="bfd")
        expected_format = dataset.format
        assert dataset.to_dict() == expected_output
        assert "seq_lengths" in expected_format["columns"]
        expected_format["columns"].remove("seq_lengths")
        assert format == dataset.format

    def test_with_iterable_dataset(self):
        # [FUNCTION DOC]
        # Same bfd packing on a numpy-formatted IterableDataset; asserts the output batch and the _formatting attribute. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        examples = {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
        }
        dataset = Dataset.from_dict(examples).to_iterable_dataset()
        dataset = dataset.with_format("numpy")
        formatting = dataset._formatting
        seq_length = 4
        expected_output = {
            "input_ids": [[4, 5, 6, 7], [1, 2, 3, 8]],
            "seq_lengths": [[4], [3, 1]],
        }
        dataset = pack_dataset(dataset, seq_length, strategy="bfd")
        num_examples = len(examples[next(iter(examples))])
        assert next(iter(dataset.with_format(None).batch(batch_size=num_examples))) == expected_output
        assert formatting == dataset._formatting

    def test_with_overlong_0(self):
        # [FUNCTION DOC]
        # bfd_split with bin 4: a length-5 row is split into 4 and 1; result bins (1-4), (8-11), (6, 7, 5, 12) with
        # seq_lengths (4), (4), (2, 1, 1). Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        examples = {
            "input_ids": [[1, 2, 3, 4, 5], [6, 7], [8, 9, 10, 11], [12]],
        }
        dataset = Dataset.from_dict(examples)
        seq_length = 4
        expected_output = {
            "input_ids": [[1, 2, 3, 4], [8, 9, 10, 11], [6, 7, 5, 12]],
            "seq_lengths": [[4], [4], [2, 1, 1]],
        }
        dataset = pack_dataset(dataset, seq_length, strategy="bfd_split")
        assert dataset.to_dict() == expected_output

    def test_with_overlong_two_coluns(self):
        # [FUNCTION DOC]
        # bfd_split with two parallel columns: overlong rows split identically in both columns; asserts the five resulting
        # bins and seq_lengths. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        examples = {
            "col1": [[1, -2, 3, -4, 5, -6], [7, -8, 9], [-10, 11, -12], [13, -14, 15, -16]],
            "col2": [[-1, 2, -3, 4, -5, 6], [-7, 8, -9], [10, -11, 12], [-13, 14, -15, 16]],
        }
        dataset = Dataset.from_dict(examples)
        seq_length = 4
        expected_output = {
            "col1": [[1, -2, 3, -4], [13, -14, 15, -16], [7, -8, 9], [-10, 11, -12], [5, -6]],
            "col2": [[-1, 2, -3, 4], [-13, 14, -15, 16], [-7, 8, -9], [10, -11, 12], [-5, 6]],
            "seq_lengths": [[4], [4], [3], [3], [2]],
        }
        dataset = pack_dataset(dataset, seq_length, strategy="bfd_split")
        assert dataset.to_dict() == expected_output

    def test_with_non_power_of_2(self):
        # [FUNCTION DOC]
        # bfd_split with bin 5: asserts bins (1-5), (7-10 plus 6), (11-13) and seq_lengths (5), (4, 1), (3). Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        examples = {
            "input_ids": [[1, 2, 3, 4, 5], [6], [7, 8, 9, 10], [11, 12, 13]],
        }
        dataset = Dataset.from_dict(examples)
        seq_length = 5
        expected_output = {
            "input_ids": [[1, 2, 3, 4, 5], [7, 8, 9, 10, 6], [11, 12, 13]],
            "seq_lengths": [[5], [4, 1], [3]],
        }
        dataset = pack_dataset(dataset, seq_length, strategy="bfd_split")
        assert dataset.to_dict() == expected_output

    def test_default_no_split(self):
        # [FUNCTION DOC]
        # bfd (no split) with bin 4: the overflow token 5 of the length-5 row is dropped; bins (1-4), (8-11), (6, 7, 12). Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        """Test default 'bfd' strategy for SFT datasets (truncates overflow)."""
        examples = {
            "input_ids": [[1, 2, 3, 4, 5], [6, 7], [8, 9, 10, 11], [12]],
        }
        dataset = Dataset.from_dict(examples)
        seq_length = 4
        # With default 'bfd' strategy, overflow tokens are discarded
        expected_output = {
            "input_ids": [[1, 2, 3, 4], [8, 9, 10, 11], [6, 7, 12]],
            "seq_lengths": [[4], [4], [2, 1]],
        }
        dataset = pack_dataset(dataset, seq_length, strategy="bfd")
        assert dataset.to_dict() == expected_output

    def test_with_empty_sequences(self):
        # [FUNCTION DOC]
        # bfd_split with empty rows: empty sequences are dropped; bins (3, 4, 5, 6) and (1, 2). Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        examples = {
            "input_ids": [[1, 2], [], [3, 4, 5], [], [6]],
        }
        dataset = Dataset.from_dict(examples)
        seq_length = 4
        expected_output = {
            "input_ids": [[3, 4, 5, 6], [1, 2]],
            "seq_lengths": [[3, 1], [2]],
        }
        dataset = pack_dataset(dataset, seq_length, strategy="bfd_split")
        assert dataset.to_dict() == expected_output


class TestMaybeConvertToChatML(TrlTestCase):
    # [CLASS DOC]
    # Tests maybe_convert_to_chatml, which converts from/value messages to role/content. Inherits TrlTestCase.
    def test_with_conversations_key(self):
        # [FUNCTION DOC]
        # Asserts a "conversations" key is renamed to "messages" and its entries converted to role/content. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        # Particular case where the key is "conversations": we rename it to "messages"
        example = {
            "conversations": [
                {"from": "user", "value": "What color is the sky?"},
                {"from": "assistant", "value": "It is blue."},
            ]
        }
        expected_output = {
            "messages": [
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "It is blue."},
            ]
        }
        assert maybe_convert_to_chatml(example) == expected_output

    def test_without_conversations_key(self):
        # [FUNCTION DOC]
        # Asserts prompt/completion from/value lists are converted in place without renaming keys. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        # Same as before, but we don't rename the keys
        example = {
            "prompt": [{"from": "user", "value": "What color is the sky?"}],
            "completion": [{"from": "assistant", "value": "It is blue."}],
        }
        expected_output = {
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
            "completion": [{"role": "assistant", "content": "It is blue."}],
        }
        assert maybe_convert_to_chatml(example) == expected_output

    def test_not_conversional(self):
        # [FUNCTION DOC]
        # Asserts a plain text example is returned unchanged. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        # When not needed, the example should remain unchanged
        example = {"text": "The sky is blue."}
        assert maybe_convert_to_chatml(example) == example

    def test_already_chatml(self):
        # [FUNCTION DOC]
        # Asserts an example already in role/content format is returned unchanged. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        # When the example is already in ChatML format, it should remain unchanged
        example = {
            "messages": [
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "It is blue."},
            ]
        }
        assert maybe_convert_to_chatml(example) == example
