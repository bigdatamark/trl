# [MODULE DOC]
# Example script: AsyncGRPO on GSM8K with Qwen3-0.6B against a vLLM server on localhost:8000 (default), rewarded by
# accuracy_reward (needs math_verify). Main contents: format_sample and main. Not part of the trl package.
# Import-time side effects: none beyond imports.

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
#     "math-verify",
#     "latex2sympy2_extended",
#     "trackio",
# ]
# ///

"""
pip install math_verify

CUDA_VISIBLE_DEVICES=1 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B \
    --max-model-len 2048 \
    --logprobs-mode processed_logprobs \
    --weight-transfer-config '{"backend":"nccl"}'

CUDA_VISIBLE_DEVICES=0 accelerate launch examples/async_grpo_math/async_grpo_math.py
"""

from datasets import load_dataset

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.rewards import accuracy_reward


def format_sample(sample):
    # [FUNCTION DOC]
    # HELPER: Converts a GSM8K row into a prompt plus reference solution.
    # Behavior: prompt is a single user message with the question; solution is the text after the last "####" marker
    # in the answer, stripped.
    # Inputs: sample with "question" and "answer". Outputs: dict with "prompt" and "solution".
    # Exceptions: KeyError on missing columns. Side effects: none.
    return {
        "prompt": [{"role": "user", "content": sample["question"]}],
        "solution": sample["answer"].split("####")[-1].strip(),
    }


def main() -> None:
    # [FUNCTION DOC]
    # Behavior: loads and maps GSM8K train; builds AsyncGRPOConfig (output_dir async_grpo_gsm8k, no saving, batch 16,
    # grad accumulation 2, completion length 1024, thinking disabled, 200 steps, lr 1e-5, trackio, completion logging);
    # trains AsyncGRPOTrainer on Qwen/Qwen3-0.6B with accuracy_reward, then calls trainer.end().
    # Inputs: none. Outputs: none. Exceptions: propagates download/connection/training errors.
    # Side effects: network downloads, vLLM traffic, trackio logging.
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

    config = AsyncGRPOConfig(
        output_dir="async_grpo_gsm8k",
        save_strategy="no",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        max_completion_length=1024,
        chat_template_kwargs={"enable_thinking": False},
        max_steps=200,
        learning_rate=1e-5,
        report_to="trackio",
        trackio_space_id="async-grpo-gsm8k",
        project="async-grpo-gsm8k",
        log_completions=True,
    )
    trainer = AsyncGRPOTrainer(
        model="Qwen/Qwen3-0.6B",
        args=config,
        train_dataset=dataset,
        reward_funcs=accuracy_reward,
    )
    trainer.train()

    trainer.end()


if __name__ == "__main__":
    main()
