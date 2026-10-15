# [MODULE DOC]
# Example training script: SFT of Gemma 3 vision models on single-image (HuggingFaceH4/llava-instruct-mix-vsft) or
# multi-image (FanqingM/MMIU-Benchmark) data. For MMIU, image zip archives are downloaded and extracted into the
# current directory and rows are converted to system/user/assistant messages with embedded PIL images.
# Contents: process_vision_info, format_data, prepare_dataset, main. Importing has no side effects.

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
#     "Pillow>=9.4.0",
#     "trackio",
#     "kernels",
# ]
# ///

"""
Train Gemma 3 on the HuggingFaceH4/llava-instruct-mix-vsft dataset (single-image).

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/sft_gemma3_vision/sft_gemma3_vision.py \
    --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path google/gemma-3-4b-it \
    --per_device_train_batch_size 1 \
    --output_dir Gemma-3-4B-SFT-MMIU \
    --dtype bfloat16 \
    --use_peft \
    --lora_target_modules all-linear \
    --attn_implementation eager

Train Gemma 3 on the FanqingM/MMIU-Benchmark dataset (multi-image).

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/sft_gemma3_vision/sft_gemma3_vision.py \
    --dataset_name FanqingM/MMIU-Benchmark \
    --dataset_train_split test \
    --model_name_or_path google/gemma-3-4b-it \
    --per_device_train_batch_size 1 \
    --output_dir Gemma-3-4B-SFT-MMIU \
    --dtype bfloat16 \
    --use_peft \
    --lora_target_modules all-linear \
    --attn_implementation eager
"""

import io
import os
import zipfile

import torch
from datasets import DatasetDict, load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
from transformers import AutoModelForImageTextToText

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
    get_quantization_config,
)


# For multi-image example
def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    # [FUNCTION DOC]
    # HELPER: Collects RGB PIL images from chat messages.
    # Behavior: for each message, treats non-list content as a one-element list; for each dict element having an image
    # key or type image, takes the image value (or the element itself), and if not None decodes its bytes field into a
    # PIL image converted to RGB.
    # Inputs: messages (list of dict). Outputs: list of PIL images. Exceptions: KeyError/TypeError if an image entry lacks
    # bytes (e.g. already a PIL image); PIL decode errors. Side effects: none.
    # KNOWN ISSUE: unused in this script (dead code), despite the comment marking it for the multi-image example.
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        for element in content:
            if isinstance(element, dict) and ("image" in element or element.get("type") == "image"):
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                if image is not None:
                    image = Image.open(io.BytesIO(image["bytes"]))
                    image_inputs.append(image.convert("RGB"))
    return image_inputs


def format_data(samples: dict[str, any]) -> dict[str, list]:
    # [FUNCTION DOC]
    # HELPER: Batched map function converting MMIU rows into conversational messages.
    # Behavior: for each row index, opens every path in input_image_path (relative to the cwd, i.e. the extracted zip
    # folders), converts to RGB and wraps it as an image content part; failures are printed and skipped. Builds a system
    # message from context, a user message of all images followed by the question text, and an assistant message with
    # output.
    # Inputs: samples (batched dict with question, input_image_path, context, output). Outputs: dict with messages list.
    # Exceptions: none raised for unreadable images (caught); KeyError on missing columns. Side effects: file reads,
    # printing.
    formatted_samples = {"messages": []}
    for cont in range(len(samples["question"])):
        images = []
        for img_path in samples["input_image_path"][cont]:
            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append({"type": "image", "image": image})
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue

        formatted_samples["messages"].append(
            [
                {"role": "system", "content": [{"type": "text", "text": samples["context"][cont]}]},
                {"role": "user", "content": images + [{"type": "text", "text": samples["question"][cont]}]},
                {"role": "assistant", "content": [{"type": "text", "text": samples["output"][cont]}]},
            ]
        )
    return formatted_samples


# For multi-image example
def prepare_dataset(dataset: DatasetDict, dataset_name: str) -> DatasetDict:
    # [FUNCTION DOC]
    # HELPER: Downloads and extracts the dataset repo's zip archives, then reformats rows with format_data.
    # Behavior: lists repo files, downloads every .zip, extracts each into a folder named after the zip path without the
    # extension (created relative to the cwd), then maps format_data batched (batch size 4, 16 processes).
    # Inputs: dataset (DatasetDict), dataset_name (str). Outputs: the mapped DatasetDict.
    # Exceptions: Hub/zip errors propagate. Side effects: network downloads, directory creation and file extraction in cwd.
    all_files = list_repo_files(dataset_name, repo_type="dataset")
    zip_files = [f for f in all_files if f.endswith(".zip")]

    for zip_filename in zip_files:
        zip_path = hf_hub_download(repo_id=dataset_name, filename=zip_filename, repo_type="dataset")
        extract_folder = zip_filename.replace(".zip", "")
        os.makedirs(extract_folder, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_folder)

    dataset = dataset.map(format_data, batched=True, batch_size=4, num_proc=16)
    return dataset


def main():
    # [FUNCTION DOC]
    # Parses arguments and runs SFT.
    # Behavior: parses ScriptArguments, SFTConfig and ModelConfig; sets max_length None (no truncation); builds model
    # kwargs (revision, attention, dtype resolved to torch dtype unless auto/None, quantization only when set) and loads
    # AutoModelForImageTextToText; loads the dataset and runs prepare_dataset only for FanqingM/MMIU-Benchmark; builds
    # SFTTrainer (eval split only when eval enabled, optional PEFT); trains, saves, optionally pushes, calls end.
    # Inputs: CLI args. Outputs: none. Exceptions: propagated. Side effects: downloads, training, file writes.
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.max_length = None

    ################
    # Model
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForImageTextToText.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    if script_args.dataset_name == "FanqingM/MMIU-Benchmark":
        dataset = prepare_dataset(dataset, script_args.dataset_name)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.end()


if __name__ == "__main__":
    main()
