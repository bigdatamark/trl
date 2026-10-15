# [PACKAGE DOC]
# Registry of TRL CLI subcommands. Exports Command (abstract base) and get_commands. Also imports EnvCommand,
# TrainingCommand and VllmServeCommand. Side effects on import: none beyond importing those lightweight modules.

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

from .base import Command
from .env import EnvCommand
from .training import TrainingCommand
from .vllm_serve import VllmServeCommand


def get_commands() -> list[Command]:
    # [FUNCTION DOC]
    # Returns fresh instances of all top-level commands, in help order: training commands distillation, dpo, then env,
    # then training commands grpo, kto, reward, rloo, sft, then vllm-serve.
    # Inputs: none. Outputs: list of Command. Exceptions: none. Side effects: none.
    """Return all registered top-level TRL CLI commands."""
    return [
        TrainingCommand("distillation"),
        TrainingCommand("dpo"),
        EnvCommand(),
        TrainingCommand("grpo"),
        TrainingCommand("kto"),
        TrainingCommand("reward"),
        TrainingCommand("rloo"),
        TrainingCommand("sft"),
        VllmServeCommand(),
    ]


__all__ = ["Command", "get_commands"]
