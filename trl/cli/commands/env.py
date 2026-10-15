# [MODULE DOC]
# Defines the "trl env" CLI subcommand that prints environment/version information. Import-time side effects: none.

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

from argparse import Namespace

from .base import Command, CommandContext


class EnvCommand(Command):
    # [CLASS DOC]
    # Command subclass named "env". Registers a plain subparser and, when run, prints environment info via
    # trl.scripts.env.print_env.
    """CLI command that prints TRL environment information."""

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes the base Command with name "env" and help text "Print the environment information".
        # Inputs: none. Outputs: none. Exceptions: none. Side effects: none.
        super().__init__(name="env", help_text="Print the environment information")

    def register(self, subparsers) -> None:
        # [FUNCTION DOC]
        # Adds the "env" subparser with the help text. Inputs: subparsers. Outputs: none. Exceptions: none.
        # Side effects: mutates the subparsers action.
        subparsers.add_parser(self.name, help=self.help_text)

    def run(self, args: Namespace, context: CommandContext) -> int:
        # [FUNCTION DOC]
        # Lazily imports print_env and calls it, returning exit code 0. Inputs: args, context (unused).
        # Outputs: 0. Exceptions: import errors propagate. Side effects: prints to stdout.
        from ...scripts.env import print_env

        print_env()
        return 0
