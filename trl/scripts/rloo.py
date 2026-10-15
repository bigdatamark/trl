# [MODULE DOC]
# Command-line training script for RLOOTrainer (runnable directly, via "trl rloo", or as a PEP 723 script).
# Contents: main (builds kwargs, loads data, trains, saves/pushes) and make_parser (TrlParser over
# ScriptArguments-style args, RLOOConfig, ModelConfig, DatasetMixtureConfig). Also defines RLOOScriptArguments.
# Import-time side effects: imports argparse, importlib, os, sys (heavy
# imports deferred into main / make_parser, though trl.ScriptArguments is imported at module level); under __main__ it parses args/config and runs main.

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

import argparse
import importlib
import os
import sys
from dataclasses import dataclass, field

from trl import ScriptArguments


@dataclass
class RLOOScriptArguments(ScriptArguments):
    # [CLASS DOC]
    # ScriptArguments subclass (dataclass) adding reward options for the RLOO script: reward_model_name_or_path
    # (str or None, default None) and reward_funcs (list of str or None, default None: registry names or dotted
    # import paths). Consumed by main; no methods.
    """
    Script arguments for the RLOO training script.

    Args:
        reward_model_name_or_path (`str`, *optional*):
            Reward model id of a pretrained model hosted inside a model repo on huggingface.co or local path to a
            directory containing model weights saved using [`~transformers.PreTrainedModel.save_pretrained`].
        reward_funcs (`list[str]`, *optional*):
            Reward functions to use. Supported values are:
                - `"accuracy_reward"`
                - `"reasoning_accuracy_reward"`
                - `"think_format_reward"`
                - `"get_soft_overlong_punishment"` (used values are `max_completion_len=1280`, `soft_punish_cache=256`)
                - any dotted import path (e.g., `'my_lib.rewards.custom_reward'`).
    """

    reward_model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "Reward model id of a pretrained model hosted inside a model repo on huggingface.co or "
            "local path to a directory containing model weights saved using `PreTrainedModel.save_pretrained`."
        },
    )
    reward_funcs: list[str] | None = field(
        default=None,
        metadata={
            "help": "Reward functions to use. Supported values are: `accuracy_reward`, `reasoning_accuracy_reward`, `think_format_reward`, "
            "`get_soft_overlong_punishment` (used values are `max_completion_len=1280`, `soft_punish_cache=256`), or "
            "any dotted import path (e.g., `'my_lib.rewards.custom_reward'`)."
        },
    )


def main(script_args, training_args, model_args, dataset_args):
    # [FUNCTION DOC]
    # Behavior:
    # - Builds reward_funcs: the reward model path (if reward_model_name_or_path) first, then for each name in
    #   script_args.reward_funcs: a registry entry (accuracy_reward, reasoning_accuracy_reward,
    #   think_format_reward, get_soft_overlong_punishment instantiated with max_completion_len 1280 and
    #   soft_punish_cache 256), else a dotted path imported after inserting the current working directory at the
    #   front of sys.path, else error.
    # - Builds model_init_kwargs from model_args (model_revision as revision, attn_implementation, dtype) plus
    #   training_args.trust_remote_code, overridden by any user model_init_kwargs, and writes it back to
    #   training_args.model_init_kwargs.
    # - Dataset: if dataset_args.datasets is set, uses get_dataset (warning that dataset_name is ignored when both
    #   are set); else loads script_args.dataset_name with dataset_config and streaming; else error.
    # - Instantiates RLOOTrainer with the model id, train split dataset_train_split, eval split dataset_test_split only
    #   when eval_strategy is not "no", quantization config from get_quantization_config and peft config from
    #   get_peft_config.
    #   Passes reward_funcs to the trainer.
    # - Trains (resume_from_checkpoint), saves to output_dir, pushes to the Hub if push_to_hub (dataset_name as
    #   model card dataset), and calls trainer.end() on transformers 5.18.0 or newer.
    # Inputs: script_args, training_args (RLOOConfig), model_args (ModelConfig), dataset_args (DatasetMixtureConfig).
    # Outputs: none.
    # Exceptions: ValueError when neither datasets nor dataset_name is given; ValueError for an unknown reward function name;
    # ImportError/AttributeError for bad dotted paths. Also inserts cwd into sys.path once per dotted reward; loading/training errors propagate.
    # Side effects: mutates training_args fields, downloads models/datasets, writes checkpoints and the final model
    # to output_dir, may push to the Hub, prints status via accelerator.print, logs a warning.
    import transformers
    from accelerate.logging import get_logger
    from datasets import load_dataset
    from packaging.version import Version

    from trl import RLOOTrainer, get_dataset, get_peft_config, get_quantization_config
    from trl.rewards import (
        accuracy_reward,
        get_soft_overlong_punishment,
        reasoning_accuracy_reward,
        think_format_reward,
    )

    logger = get_logger(__name__)

    reward_funcs_registry = {
        "accuracy_reward": accuracy_reward,
        "reasoning_accuracy_reward": reasoning_accuracy_reward,
        "think_format_reward": think_format_reward,
        "get_soft_overlong_punishment": get_soft_overlong_punishment(max_completion_len=1280, soft_punish_cache=256),
    }

    # Get the reward models and functions
    reward_funcs = []
    if script_args.reward_model_name_or_path:
        reward_funcs.append(script_args.reward_model_name_or_path)

    if script_args.reward_funcs:
        for func_name in script_args.reward_funcs:
            if func_name in reward_funcs_registry:
                reward_funcs.append(reward_funcs_registry[func_name])
            elif "." in func_name:
                module_path, func_name = func_name.rsplit(".", 1)
                sys.path.insert(0, os.getcwd())
                module = importlib.import_module(module_path)
                reward_func = getattr(module, func_name)
                reward_funcs.append(reward_func)
            else:
                raise ValueError(
                    f"Could not load reward function '{func_name}'. Expected one of "
                    f"{list(reward_funcs_registry.keys())} or a valid import path."
                )

    model_init_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=training_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )
    if training_args.model_init_kwargs is not None:
        model_init_kwargs.update(training_args.model_init_kwargs)
    training_args.model_init_kwargs = model_init_kwargs

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

    # Initialize the RLOO trainer
    trainer = RLOOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        quantization_config=get_quantization_config(model_args),
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
    # Behavior: builds the dataclass tuple (RLOOScriptArguments, RLOOConfig, ModelConfig, DatasetMixtureConfig); if subparsers is given,
    # registers a "rloo" subcommand on it (used by the trl CLI), otherwise creates a standalone TrlParser with prog.
    # Inputs: subparsers (optional argparse subparsers action), prog (optional program name).
    # Outputs: the parser / subparser.
    # Exceptions: none expected. Side effects: registers a subcommand when subparsers is given.
    from trl import DatasetMixtureConfig, ModelConfig, RLOOConfig, TrlParser

    dataclass_types = (RLOOScriptArguments, RLOOConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("rloo", help="Run the RLOO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types, prog=prog)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, dataset_args = parser.parse_args_and_config(fail_with_unknown_args=False)
    main(script_args, training_args, model_args, dataset_args)
