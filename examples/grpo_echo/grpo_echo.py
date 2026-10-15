# [MODULE DOC]
# Example script: minimal GRPO with an OpenEnv Echo environment hosted on a HF Space: the model calls an echo tool and
# is rewarded by the environment (which favors longer messages). Main contents: parse_args, reward_func, main (defines
# EchoToolEnv inline). Import-time side effects: none beyond imports (requires echo_env).

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
#     "openenv-echo-env @ git+https://huggingface.co/spaces/qgallouedec/echo_env",
# ]
# ///


"""
Simple script to run GRPO training with OpenEnv's Echo environment. The environment echoes back the message
sent to it and rewards longer completions.

Setup (Option A - Install from HF Space, recommended):

```sh
uv pip install git+https://huggingface.co/spaces/qgallouedec/echo_env
```

Setup (Option B - Clone OpenEnv repo, for development):

```sh
git clone https://github.com/huggingface/OpenEnv.git
cd OpenEnv/envs/echo_env
uv pip install -e .
```

Usage:

```sh
python examples/grpo_echo/grpo_echo.py
python examples/grpo_echo/grpo_echo.py --model Qwen/Qwen2.5-0.5B-Instruct --env-host https://qgallouedec-echo-env.hf.space
```
"""

import argparse

from datasets import Dataset
from echo_env import EchoEnv
from echo_env.models import EchoAction

from trl import GRPOConfig, GRPOTrainer


def parse_args():
    # [FUNCTION DOC]
    # Behavior: argparse for model (Qwen/Qwen3-0.6B) and env host URL (public echo Space).
    # Inputs: command line. Outputs: Namespace. Exceptions: SystemExit on bad args. Side effects: may print usage.
    parser = argparse.ArgumentParser(description="Run GRPO training with Echo environment.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model to use for training.",
    )
    parser.add_argument(
        "--env-host",
        type=str,
        default="https://qgallouedec-echo-env.hf.space",
        help="URL for the Echo environment HF Space.",
    )
    return parser.parse_args()


def reward_func(environments, **kwargs):
    # [FUNCTION DOC]
    # Behavior: returns each environment's reward from its last echo call (0 if none).
    # Inputs: environments, kwargs ignored. Outputs: list of floats. Exceptions: none. Side effects: none.
    return [env.reward for env in environments]


def main():
    # [FUNCTION DOC]
    # Behavior: parses args; builds a 5-row dataset of user prompts asking to echo various phrases; defines EchoToolEnv;
    # trains GRPOTrainer with reward_func and environment_factory EchoToolEnv (thinking disabled, completion logging,
    # logging every 2 steps, 1 printed completion); calls trainer.end().
    # Inputs: command line. Outputs: none. Exceptions: propagates network/training errors.
    # Side effects: remote environment traffic, GPU training, logging, output files.
    args = parse_args()

    dataset = Dataset.from_dict(
        {
            "prompt": [
                [{"role": "user", "content": "Try to echo 'Hello World!' in the environment."}],
                [{"role": "user", "content": "Make the environment echo 'Goodbye World!'"}],
                [{"role": "user", "content": "Can you ask the environment to echo 'TRL is great!'?"}],
                [{"role": "user", "content": "What happens if you ask the environment to echo 'I love RLHF!'?"}],
                [{"role": "user", "content": "Try to make the environment echo 'OpenEnv is awesome!'"}],
            ],
        }
    )

    class EchoToolEnv:
        # [CLASS DOC]
        # Per-rollout environment holding an EchoEnv client (for the closed-over args.env_host) and the latest reward. reset
        # clears the reward; echo is the single tool. No base class.
        def __init__(self):
            # [FUNCTION DOC]
            # Behavior: creates the EchoEnv client and sets reward 0. Inputs: none. Outputs: none.
            # Exceptions: client errors. Side effects: creates a client.
            self.env = EchoEnv(base_url=args.env_host)
            self.reward = 0.0

        def reset(self, **kwargs) -> None | str:
            # [FUNCTION DOC]
            # Behavior: sets reward 0 and returns None (no extra observation appended to the prompt). The remote env is not reset.
            # Inputs: kwargs ignored. Outputs: None. Exceptions: none. Side effects: mutates reward.
            self.reward = 0.0
            return None

        def echo(self, message: str) -> str:
            # [FUNCTION DOC]
            # Tool: sends an EchoAction with message, stores the observation's reward and returns the echoed message.
            # Inputs: message. Outputs: string. Exceptions: network errors. Side effects: remote step, mutates reward.
            """
            Echo the message back from the environment.

            Args:
                message: The message to echo

            Returns:
                The echoed message.
            """
            observation = self.env.step(EchoAction(message=message))
            self.reward = observation.observation.reward
            return observation.observation.echoed_message

    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=dataset,
        reward_funcs=reward_func,
        args=GRPOConfig(
            chat_template_kwargs={"enable_thinking": False},
            log_completions=True,
            logging_steps=2,
            num_completions_to_print=1,
        ),
        environment_factory=EchoToolEnv,
    )
    trainer.train()

    trainer.end()


if __name__ == "__main__":
    main()
