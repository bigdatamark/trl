# [MODULE DOC]
# Example training script: RLOO with LoRA and colocated vLLM generation on AI-MO/NuminaMath-TIR, training
# Qwen/Qwen3-0.6B with think_format_reward and accuracy_reward. All hyperparameters are hard-coded (no CLI parsing).
# Contents: main and its nested make_conversation helper. Importing has no side effects; running calls main.

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
#     "trl[vllm,peft]",
#     "math-verify",
#     "latex2sympy2_extended",
#     "trackio",
#     "kernels",
# ]
# ///

"""
NuminaMath example: RLOO on math dataset with vLLM.

  export TRACKIO_PROJECT="RLOO-NuminaMath-TIR"
  accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml examples/rloo_math/rloo_math.py

For TL;DR or other datasets with a reward model, use the generic script:
  python -m trl.scripts.rloo --dataset_name trl-lib/tldr --reward_model_name_or_path ... --model_name_or_path ...
"""

import torch
from datasets import load_dataset
from peft import LoraConfig

from trl import RLOOConfig, RLOOTrainer
from trl.rewards import accuracy_reward, think_format_reward


def main():
    # [FUNCTION DOC]
    # Runs the whole example end to end.
    # Behavior: loads the first 5% of the train and test splits; maps rows to a system+user conversational prompt and
    # removes the messages/problem columns; builds an RLOOConfig (output_dir Qwen3-0.6B-RLOO, bfloat16 model init,
    # learning rate 1e-5, completion logging of 2 samples, max completion length 1024, gradient accumulation 2, 8 steps
    # per generation, vLLM colocate mode at 0.5 GPU memory utilization, a fixed run name); builds RLOOTrainer with a
    # default LoraConfig; trains, saves to output_dir, always pushes to the Hub with the dataset name, and calls end.
    # Inputs: none. Outputs: none.
    # Exceptions: propagated from datasets, vLLM, training or Hub upload (push fails without credentials).
    # Side effects: downloads, GPU training, file writes, Hub upload, experiment logging.
    # Dataset
    train_dataset, eval_dataset = load_dataset("AI-MO/NuminaMath-TIR", split=["train[:5%]", "test[:5%]"])

    SYSTEM_PROMPT = (
        "A conversation between user and assistant. The user asks a question, and the assistant solves it. The "
        "assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think></think> tags, i.e., <think>\nThis is my "
        "reasoning.\n</think>\nThis is my answer."
    )

    def make_conversation(example):
        # [FUNCTION DOC]
        # HELPER: Map function turning a row into a prompt of a system message (SYSTEM_PROMPT) and a user message holding the
        # problem. Inputs: example (dict with problem). Outputs: dict with key prompt. Exceptions: KeyError if problem is
        # missing. Side effects: none.
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    train_dataset = train_dataset.map(make_conversation, remove_columns=["messages", "problem"])
    eval_dataset = eval_dataset.map(make_conversation, remove_columns=["messages", "problem"])

    # Training
    training_args = RLOOConfig(
        output_dir="Qwen3-0.6B-RLOO",
        model_init_kwargs={"dtype": torch.bfloat16},
        learning_rate=1e-5,
        log_completions=True,
        num_completions_to_print=2,
        max_completion_length=1024,
        gradient_accumulation_steps=2,
        steps_per_generation=8,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.5,
        run_name="Qwen3-0.6B-RLOO-NuminaMath-TIR",
    )

    trainer = RLOOTrainer(
        model="Qwen/Qwen3-0.6B",
        args=training_args,
        reward_funcs=[think_format_reward, accuracy_reward],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=LoraConfig(),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    trainer.push_to_hub(dataset_name="AI-MO/NuminaMath-TIR")

    trainer.end()


if __name__ == "__main__":
    main()
