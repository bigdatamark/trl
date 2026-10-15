# [MODULE DOC]
# Command-line training script for DistillationTrainer (runnable directly, via "trl distillation", or as a PEP 723 script).
# Contents: main (builds kwargs, loads data, trains, saves/pushes) and make_parser (TrlParser over
# ScriptArguments-style args, DistillationConfig, ModelConfig, DatasetMixtureConfig).
# The module docstring holds example CLI invocations. Import-time side effects: imports argparse only (heavy
# imports are deferred into main / make_parser); under __main__ it parses args/config and runs main.

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
#     "trl",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

"""
# Full training
```bash
python trl/scripts/distillation.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name trl-lib/ultrafeedback-prompt \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --output_dir Qwen2.5-0.5B-Distillation
```

# LoRA
```bash
python trl/scripts/distillation.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name trl-lib/ultrafeedback-prompt \
    --learning_rate 2.0e-4 \
    --num_train_epochs 1 \
    --output_dir Qwen2.5-0.5B-Distillation \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
```
"""

import argparse


def main(script_args, training_args, model_args, dataset_args):
    # [FUNCTION DOC]
    # Behavior:
    # - Requires training_args.teacher_model_name_or_path.
    # - Student quantization goes through the trainer's quantization_config argument, not model_init_kwargs.
    # - Builds teacher_model_init_kwargs the same way (teacher_model_revision, no quantization by default so the
    #   teacher distribution is exact) merged with user teacher_model_init_kwargs.
    # - Builds model_init_kwargs from model_args (model_revision as revision, attn_implementation, dtype) plus
    #   training_args.trust_remote_code, overridden by any user model_init_kwargs, and writes it back to
    #   training_args.model_init_kwargs.
    # - Dataset: if dataset_args.datasets is set, uses get_dataset (warning that dataset_name is ignored when both
    #   are set); else loads script_args.dataset_name with dataset_config and streaming; else error.
    # - Instantiates DistillationTrainer with the model id, train split dataset_train_split, eval split dataset_test_split only
    #   when eval_strategy is not "no", quantization config from get_quantization_config and peft config from
    #   get_peft_config.
    # - Trains (resume_from_checkpoint), saves to output_dir, pushes to the Hub if push_to_hub (dataset_name as
    #   model card dataset), and calls trainer.end() on transformers 5.18.0 or newer.
    # Inputs: script_args, training_args (DistillationConfig), model_args (ModelConfig), dataset_args (DatasetMixtureConfig).
    # Outputs: none.
    # Exceptions: ValueError when neither datasets nor dataset_name is given; ValueError when no
    # teacher model is set; loading/training errors propagate.
    # Side effects: mutates training_args fields, downloads models/datasets, writes checkpoints and the final model
    # to output_dir, may push to the Hub, prints status via accelerator.print, logs a warning.
    import transformers
    from accelerate.logging import get_logger
    from datasets import load_dataset
    from packaging.version import Version

    from trl import DistillationTrainer, get_dataset, get_peft_config, get_quantization_config

    logger = get_logger(__name__)

    if training_args.teacher_model_name_or_path is None:
        raise ValueError(
            "A teacher model is required for distillation training. Set it with `--teacher_model_name_or_path`."
        )

    quantization_config = get_quantization_config(model_args)
    # The student's quantization is passed via the trainer's `quantization_config` argument (below), so it must NOT
    # also be set in `model_init_kwargs` — the trainer rejects that combination.
    model_init_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=training_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )
    if training_args.model_init_kwargs is not None:
        model_init_kwargs.update(training_args.model_init_kwargs)
    training_args.model_init_kwargs = model_init_kwargs

    # The teacher is deliberately left unquantized: quantizing it degrades the reference distribution the student is
    # trained to match. To quantize it anyway, override the default below with
    # `--teacher_model_init_kwargs '{"quantization_config": {...}}'`.
    teacher_model_init_kwargs = dict(
        revision=training_args.teacher_model_revision,
        trust_remote_code=training_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )
    if training_args.teacher_model_init_kwargs is not None:
        teacher_model_init_kwargs.update(training_args.teacher_model_init_kwargs)
    training_args.teacher_model_init_kwargs = teacher_model_init_kwargs

    # Load the dataset
    if dataset_args.datasets and script_args.dataset_name:
        logger.warning(
            "Both `datasets` and `dataset_name` are provided. The `datasets` argument will be used to load the "
            "dataset and `dataset_name` will be ignored."
        )
        dataset = get_dataset(dataset_args)
    elif dataset_args.datasets and not script_args.dataset_name:
        dataset = get_dataset(dataset_args)
    elif not dataset_args.datasets and script_args.dataset_name:
        dataset = load_dataset(
            script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
        )
    else:
        raise ValueError("Either `datasets` or `dataset_name` must be provided.")

    # Initialize the Distillation trainer
    trainer = DistillationTrainer(
        model=model_args.model_name_or_path,
        teacher_model=training_args.teacher_model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        quantization_config=quantization_config,
        peft_config=get_peft_config(model_args),
    )

    # Train the model
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Log training complete
    trainer.accelerator.print("✅ Training completed.")

    # Save and push to Hub
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"🤗 Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}.")

    # Finish the trackers and destroy the process group
    if Version(transformers.__version__) >= Version("5.18.0"):
        trainer.end()


def make_parser(subparsers: argparse._SubParsersAction | None = None, prog: str | None = None):
    # [FUNCTION DOC]
    # Behavior: builds the dataclass tuple (ScriptArguments, DistillationConfig, ModelConfig, DatasetMixtureConfig); if subparsers is given,
    # registers a "distillation" subcommand on it (used by the trl CLI), otherwise creates a standalone TrlParser with prog.
    # Inputs: subparsers (optional argparse subparsers action), prog (optional program name).
    # Outputs: the parser / subparser.
    # Exceptions: none expected. Side effects: registers a subcommand when subparsers is given.
    from trl import DatasetMixtureConfig, DistillationConfig, ModelConfig, ScriptArguments, TrlParser

    dataclass_types = (ScriptArguments, DistillationConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser(
            "distillation", help="Run the distillation training script", dataclass_types=dataclass_types
        )
    else:
        parser = TrlParser(dataclass_types, prog=prog)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, dataset_args = parser.parse_args_and_config(fail_with_unknown_args=False)
    main(script_args, training_args, model_args, dataset_args)
