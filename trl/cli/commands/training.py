# [MODULE DOC]
# Defines TrainingCommand, the generic CLI subcommand ("trl sft", "trl dpo", ...) that parses a training script's
# arguments, separates accelerate options from script options, and launches trl/scripts/<name>.py with accelerate.
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

import importlib
from argparse import Namespace

from .base import Command, CommandContext


def _subtract_subsequence(lst: list[str], subseq: list[str]) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Removes an ordered subsequence from a list.
    # Behavior: walk lst once while tracking the next expected element of subseq; when an item equals it, skip the item
    # and advance to the next expected element; otherwise keep the item. Elements of subseq not found in order leave the
    # rest of lst untouched.
    # Inputs: lst, subseq (lists of str). Outputs: new list. Exceptions: none. Side effects: none.
    """Return lst with the ordered subsequence subseq removed."""
    sub_iter = iter(subseq)
    current = next(sub_iter, None)
    result = []
    for item in lst:
        if current is not None and item == current:
            current = next(sub_iter, None)
        else:
            result.append(item)
    return result


class TrainingCommand(Command):
    # [CLASS DOC]
    # Command subclass parametrized by a script name. The script module trl.scripts.<name> must expose make_parser(prog)
    # returning a TrlParser. Help is delegated to that parser (the subparser is registered with add_help=False).
    """
    Generic CLI command that launches a training script with accelerate.

    The script `trl/scripts/<name>.py` must expose a `make_parser()` function.

    Parameters:
        name (`str`):
            CLI subcommand name (e.g. `"dpo"`).
    """

    def __init__(self, name: str):
        # [FUNCTION DOC]
        # Initializes the base Command with the given name and help text "Run the <name> training script".
        # Inputs: name (str). Outputs: none. Exceptions: none. Side effects: none.
        super().__init__(name=name, help_text=f"Run the {name} training script")

    def register(self, subparsers) -> None:
        # [FUNCTION DOC]
        # Adds a subparser for this command without argparse's own -h handling. Inputs: subparsers. Outputs: none.
        # Exceptions: none. Side effects: mutates subparsers.
        subparsers.add_parser(self.name, help=self.help_text, add_help=False)

    def run(self, args: Namespace, context: CommandContext) -> int:
        # [FUNCTION DOC]
        # Parses and launches the training script.
        # Behavior: import trl.scripts.<name>; take argv tokens after the command name; build the script parser; parse them
        # together with any config file, returning separately the leftover strings from the config file and from the CLI
        # (leftovers are treated as accelerate launch options). Resolve --accelerate_config in the combined leftovers into
        # launch_args; the script args are the original CLI tokens with the CLI leftovers removed (as an ordered subsequence).
        # Then call launch_training_script with "<name>.py".
        # Inputs: args (unused), context (CommandContext). Outputs: 0.
        # Exceptions: SystemExit from -h or parse errors; ValueError from resolve_accelerate_config_argument; launch errors.
        # Side effects: runs training.
        from ..accelerate_config import resolve_accelerate_config_argument
        from ..accelerate_launcher import launch_training_script

        module = importlib.import_module(f"...scripts.{self.name}", package=__package__)
        all_args = context.argv_after(self.name)
        parser = module.make_parser(prog=f"trl {self.name}")

        # Handles -h (exits). Returns config_remaining and cli_remaining separately.
        # cli_remaining is an ordered subsequence of all_args; config_remaining is not.
        *_, config_remaining, cli_remaining = parser.parse_args_and_config(
            all_args, return_remaining_strings=True, separate_remaining_strings=True
        )
        launch_args = resolve_accelerate_config_argument(config_remaining + cli_remaining)
        training_script_args = _subtract_subsequence(all_args, cli_remaining)

        launch_training_script(
            script_name=f"{self.name}.py",
            launch_args=launch_args,
            training_script_args=training_script_args,
        )
        return 0
