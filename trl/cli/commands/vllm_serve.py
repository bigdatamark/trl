# [MODULE DOC]
# Defines the "trl vllm-serve" CLI subcommand that starts TRL's vLLM generation server. Import-time side effects: none.

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


class VllmServeCommand(Command):
    # [CLASS DOC]
    # Command subclass named "vllm-serve"; delegates argument parsing and help to trl.scripts.vllm_serve.make_parser and
    # runs trl.scripts.vllm_serve.main.
    """CLI command for serving TRL models with vLLM."""

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes the base Command with name "vllm-serve" and help "Serve a model with vLLM".
        # Inputs: none. Outputs: none. Exceptions: none. Side effects: none.
        super().__init__(name="vllm-serve", help_text="Serve a model with vLLM")

    def register(self, subparsers) -> None:
        # [FUNCTION DOC]
        # Adds the subparser with add_help=False so -h reaches the script parser. Inputs: subparsers. Outputs: none.
        # Exceptions: none. Side effects: mutates subparsers.
        subparsers.add_parser(self.name, help=self.help_text, add_help=False)

    def run(self, args: Namespace, context: CommandContext) -> int:
        # [FUNCTION DOC]
        # Lazily imports the vllm_serve script, parses argv tokens after "vllm-serve" (with optional config file) keeping
        # unrecognized strings, and calls the script's main with the parsed args and those extra strings (forwarded to vLLM).
        # Inputs: args (unused), context. Outputs: 0 after the server exits.
        # Exceptions: SystemExit from parsing; server errors propagate. Side effects: starts a blocking vLLM server.
        from ...scripts.vllm_serve import main as vllm_serve_main
        from ...scripts.vllm_serve import make_parser as make_vllm_serve_parser

        parser = make_vllm_serve_parser(prog="trl vllm-serve")
        # Unrecognized arguments are forwarded to `vllm serve`, so that any vLLM option can be used.
        script_args, extra_args = parser.parse_args_and_config(
            args=context.argv_after(self.name), return_remaining_strings=True
        )
        vllm_serve_main(script_args, extra_args)
        return 0
