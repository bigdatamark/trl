# [MODULE DOC]
# Base abstractions for TRL CLI subcommands: CommandContext (raw argv carried to commands) and the abstract Command
# class each subcommand implements (register its argparse subparser, then run).
# Import-time side effects: none.

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

from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass


@dataclass(slots=True)
class CommandContext:
    # [CLASS DOC]
    # Slotted dataclass holding argv (the full list of CLI tokens after the program name) so commands can re-parse their
    # own arguments independently of the top-level parser.
    """Context shared by CLI commands during execution."""

    argv: list[str]

    def argv_after(self, token: str) -> list[str]:
        # [FUNCTION DOC]
        # Returns the tokens following the first occurrence of token in argv, or an empty list if token is absent.
        # Inputs: token (str). Outputs: list of str. Exceptions: none. Side effects: none.
        """
        Return CLI tokens after the first occurrence of `token`.

        Parameters:
            token (`str`):
                Subcommand name as it appears in `argv`.
        """
        try:
            index = self.argv.index(token)
        except ValueError:
            return []
        return self.argv[index + 1 :]


class Command(ABC):
    # [CLASS DOC]
    # Abstract base class for a CLI subcommand. State: name (subcommand name) and help_text (help string). Subclasses
    # implement register (add a subparser) and run (execute and return an exit code). main builds the parser by calling
    # register on each command and dispatches to run for the selected one.
    """
    Base command definition for the TRL CLI.

    Parameters:
        name (`str`):
            Subcommand name exposed by the CLI.
        help_text (`str`):
            Short description displayed in help output.
    """

    def __init__(self, name: str, help_text: str):
        # [FUNCTION DOC]
        # Stores name and help_text. Inputs: name (str), help_text (str). Outputs: none. Exceptions: none.
        # Side effects: sets attributes.
        self.name = name
        self.help_text = help_text

    @abstractmethod
    def register(self, subparsers) -> None:
        # [FUNCTION DOC]
        # Abstract: add this command's parser to the argparse subparsers action. Inputs: subparsers. Outputs: none.
        # Exceptions: none (no implementation). Side effects: defined by subclasses.
        """Register this command parser in the subparser collection."""

    @abstractmethod
    def run(self, args: Namespace, context: CommandContext) -> int:
        # [FUNCTION DOC]
        # Abstract: execute the command. Inputs: args (parsed top-level Namespace), context (CommandContext).
        # Outputs: int exit code. Exceptions: none here. Side effects: defined by subclasses.
        """Execute the command."""
