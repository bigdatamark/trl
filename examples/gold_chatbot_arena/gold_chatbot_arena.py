# [MODULE DOC]
# Example script: GOLD (cross-tokenizer on-policy distillation) of a small student (e.g. LFM2.5-1.2B or Gemma 4) from
# a Qwen3-4B teacher on a raw-text or conversational dataset. Under the main guard: parses ScriptArguments/GOLDConfig/
# ModelConfig, sets student model_init_kwargs (use_cache off with gradient checkpointing, optional quantization),
# defaults the teacher tokenizer to the teacher model when ULD loss is used, builds unquantized teacher init kwargs
# merged with user overrides, loads the student tokenizer (pad defaults to eos), splits raw "text" datasets into
# prompt/completion via split_text_for_gold and drops empty pairs, picks an eval split (test split, then
# "validation", then "dev"), trains GOLDTrainer with optional PEFT and a LogCompletionsCallback when evaluating,
# saves, optionally pushes and calls trainer.end().
# Import-time side effects: creates a module logger.

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
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "peft",
#     "trackio",
# ]
# ///

# docstyle-ignore
"""
# Recommended small student: LFM2.5-1.2B
python examples/gold_chatbot_arena/gold_chatbot_arena.py \
    --model_name_or_path LiquidAI/LFM2.5-1.2B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen3-4B \
    --dataset_name nvidia/Nemotron-Pretraining-Specialized-v1.1 \
    --dataset_config Nemotron-Pretraining-Formal-Logic \
    --use_uld_loss \
    --lmbda 0.0 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --output_dir gold-lfm2.5-1.2b \
    --num_train_epochs 1 \
    --push_to_hub

# Recommended Gemma 4 student. Gemma 4 uses SentencePiece, so cross-tokenizer
# ULD requires the positional alignment path (`use_extended_uld=False`). This
# script uses Gemma 4 in text-only mode; use gold_qwen3_vl.py for multimodal training.
python examples/gold_chatbot_arena/gold_chatbot_arena.py \
    --model_name_or_path google/gemma-4-E4B-it \
    --teacher_model_name_or_path Qwen/Qwen3-4B \
    --dataset_name nvidia/Nemotron-Pretraining-Specialized-v1.1 \
    --dataset_config Nemotron-Pretraining-Formal-Logic \
    --use_uld_loss \
    --no_use_extended_uld \
    --lmbda 0.0 \
    --learning_rate 2e-4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --output_dir gold-gemma-4-e4b \
    --num_train_epochs 1 \
    --push_to_hub \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 16
"""

import logging

from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold.gold_config import GOLDConfig
from trl.experimental.gold.gold_trainer import GOLDTrainer


logger = logging.getLogger(__name__)


def split_text_for_gold(example, tokenizer, max_length):
    # [FUNCTION DOC]
    # HELPER: Splits raw text into a prompt (first ~75% of tokens) and completion (rest).
    # Behavior: tokenizes without special tokens with offset mapping; when max_length is set keeps only the last
    # max_length tokens; empty token list yields empty prompt and completion. Split index is max(1, floor(0.75 * n));
    # the prompt is the original text from the first kept token's start to the split token's start, completion from there
    # to the last token's end (if split index equals n the completion is empty).
    # Inputs: example with "text", tokenizer (fast, supporting offsets), max_length or None.
    # Outputs: dict with "prompt" and "completion" strings.
    # Exceptions: tokenizers without offset mapping support raise. Side effects: none.
    """Split a raw language-modeling example into a 75% prompt and 25% completion."""
    text = example["text"]
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    if max_length is not None:
        input_ids = input_ids[-max_length:]
        offsets = offsets[-max_length:]
    if not input_ids:
        return {"prompt": "", "completion": ""}
    split_idx = max(1, int(len(input_ids) * 0.75))
    text_start = offsets[0][0]
    text_end = offsets[-1][1]
    split_offset = offsets[split_idx][0] if split_idx < len(offsets) else text_end
    return {
        "prompt": text[text_start:split_offset],
        "completion": text[split_offset:text_end],
    }


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # Model & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    if training_args.teacher_tokenizer_name_or_path is None and training_args.use_uld_loss:
        training_args.teacher_tokenizer_name_or_path = training_args.teacher_model_name_or_path

    # The teacher is deliberately left unquantized: quantizing it degrades the reference distribution the student is
    # trained to match. To quantize it anyway, override the default below with
    # `--teacher_model_init_kwargs '{"quantization_config": {...}}'`.
    teacher_model_kwargs = dict(
        revision=training_args.teacher_model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
        use_cache=True,
    )
    if training_args.teacher_model_init_kwargs is not None:
        teacher_model_kwargs.update(training_args.teacher_model_init_kwargs)
    training_args.teacher_model_init_kwargs = teacher_model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    dataset_sample = next(iter(dataset[script_args.dataset_train_split]))
    if "text" in dataset_sample and "prompt" not in dataset_sample and "messages" not in dataset_sample:
        dataset = dataset.map(
            split_text_for_gold,
            fn_kwargs={"tokenizer": tokenizer, "max_length": training_args.max_length},
            remove_columns=["text"],
            num_proc=training_args.dataset_num_proc,
            desc="Splitting raw text into GOLD prompts and completions",
        )
        dataset = dataset.filter(
            lambda example: bool(example["prompt"] and example["completion"]),
            num_proc=training_args.dataset_num_proc,
            desc="Removing examples that are too short for GOLD",
        )

    ################
    # Training
    ################
    eval_dataset = None
    if training_args.eval_strategy != "no":
        if script_args.dataset_test_split in dataset:
            eval_dataset = dataset[script_args.dataset_test_split]
        elif "validation" in dataset:
            eval_dataset = dataset["validation"]
        elif "dev" in dataset:
            eval_dataset = dataset["dev"]

    trainer = GOLDTrainer(
        model=model_args.model_name_or_path,
        teacher_model=training_args.teacher_model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length, do_sample=True, temperature=training_args.temperature
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.end()
