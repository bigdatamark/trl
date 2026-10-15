# [MODULE DOC]
# Example training script: QLoRA SFT teaching tool calling to CohereLabs/tiny-aya-global on bebechien/SimpleToolCalling,
# using a custom tool-aware chat template file (tiny_aya_chat_template.jinja next to this script) that is saved with
# the tokenizer. Contents: TOOLS (two function schemas: search_knowledge_base and search_google, each taking a query
# string), create_conversation and main. Importing has no side effects; running the file calls main.

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

# /// script
# dependencies = [
#     "trl[peft]",
#     "bitsandbytes",
#     "kernels",
#     "trackio",
# ]
# ///

"""
Teach tool calling to CohereLabs/tiny-aya-global using SFT with QLoRA on the bebechien/SimpleToolCalling dataset.

The model used in this script does not have native tool-calling support. We extend its existing Jinja2 chat template to
serialize tool schemas into the system preamble and render tool calls as structured <tool_call> XML inside the model's
native <|START_RESPONSE|> / <|END_RESPONSE|> delimiters. The modified template is saved with the tokenizer, so
inference only requires loading the tokenizer from the output directory and calling apply_chat_template with
tools=TOOLS — no manual system-prompt construction needed.

Example:

    python examples/sft_tool_calling/sft_tiny_aya_tool_calling.py
"""

import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from trl import SFTConfig, SFTTrainer


# These are the tool schemas that are used in the dataset
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search internal company documents, policies and project data.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "query string"}},
                "required": ["query"],
            },
            "return": {"type": "string"},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_google",
            "description": "Search public information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "query string"}},
                "required": ["query"],
            },
            "return": {"type": "string"},
        },
    },
]


def create_conversation(sample):
    # [FUNCTION DOC]
    # HELPER: Map function converting a raw row into a prompt-completion tool-calling example.
    # Behavior: prompt is one user message with user_content; completion is one assistant message whose tool_calls
    # holds a single function call named tool_name with arguments parsed from the tool_arguments JSON string; tools is
    # the TOOLS list.
    # Inputs: sample (dict with user_content, tool_name, tool_arguments). Outputs: dict with prompt, completion, tools.
    # Exceptions: KeyError on missing fields; JSON decode errors for malformed arguments. Side effects: none.
    return {
        "prompt": [{"role": "user", "content": sample["user_content"]}],
        "completion": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": sample["tool_name"],
                            "arguments": json.loads(sample["tool_arguments"]),
                        },
                    }
                ],
            },
        ],
        "tools": TOOLS,
    }


def main():
    # [FUNCTION DOC]
    # Runs the example end to end.
    # Behavior: loads the train split, maps it with create_conversation removing all original columns, splits it 50/50
    # with shuffling (only the train half is used; no eval); loads the model with sdpa attention, float16, hub kernels
    # and 4-bit NF4 double-quantized bitsandbytes (float16 compute); LoRA r 32 / alpha 32 on attention and MLP
    # projections; SFTConfig with batch 1, gradient accumulation 4, the custom chat template path, 5 warmup steps,
    # learning rate 2e-4, paged 8-bit AdamW, logging every step to trackio (space named after output dir), max length
    # 1024, activation offloading and push_to_hub enabled; trains, saves to output_dir, pushes to the Hub and calls end.
    # Inputs: none. Outputs: none. Exceptions: propagated (Hub push requires credentials).
    # Side effects: downloads, GPU training, file writes, Hub uploads, trackio logging.
    model_id = "CohereLabs/tiny-aya-global"
    dataset_name = "bebechien/SimpleToolCalling"
    output_dir = "tiny-aya-global-tool-calling-SFT"

    # Load and format dataset
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.map(create_conversation, remove_columns=dataset.features)
    dataset = dataset.train_test_split(test_size=0.5, shuffle=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="sdpa",
        dtype=torch.float16,
        use_kernels=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )

    # Configure LoRA
    peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Train
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        # Use the tool-aware chat template
        chat_template_path=str(Path(__file__).parent / "tiny_aya_chat_template.jinja"),
        warmup_steps=5,
        learning_rate=2e-4,
        optim="paged_adamw_8bit",
        logging_steps=1,
        report_to="trackio",
        trackio_space_id=output_dir,
        max_length=1024,
        activation_offloading=True,
        push_to_hub=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        peft_config=peft_config,
    )
    trainer.train()

    # Save model and tokenizer (tokenizer carries the updated chat template)
    trainer.save_model(output_dir)
    trainer.push_to_hub(dataset_name=dataset_name)

    trainer.end()


if __name__ == "__main__":
    main()
