# [MODULE DOC]
# HELPER: Minimal deterministic Open Reward Standard (ORS) environment used as a subprocess spawn target by the
# openreward adapter tests.
# Contents: TRAIN_TASKS (four tasks with ids echo-0..echo-3 and targets hello, world, trl, openreward), pydantic
# models EchoTaskSpec, EchoParams, HintParams, and EchoEnvironment. When run as __main__, serves EchoEnvironment
# with openreward's Server on 127.0.0.1 at the port given by the PORT env var (KeyError if unset).
# Inputs: none at import. Outputs: none.
# Exceptions: ImportError if openreward or pydantic is missing.
# Side effects: none at import; starts a blocking HTTP server only when executed as a script.

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

"""Spawn target for `tests/experimental/test_openreward.py`.

Implements a minimal, deterministic Open Reward Standard environment so the adapter can be exercised end-to-end without
hitting the network. The same definition is also published as a Hugging Face Space at
`trl-internal-testing/openreward-echo-env` for the optional integration test.

The model is given a target string and must call `echo(text=...)` with exactly that string. Reward is `1.0` on match,
`0.0` otherwise; the episode finishes on a correct echo.
"""

from openreward.environments import (
    Environment,
    JSONObject,
    ListToolsOutput,
    Server,
    TextBlock,
    ToolOutput,
    ToolSpec,
    tool,
)
from pydantic import BaseModel


TRAIN_TASKS: list[JSONObject] = [
    {"id": "echo-0", "target": "hello"},
    {"id": "echo-1", "target": "world"},
    {"id": "echo-2", "target": "trl"},
    {"id": "echo-3", "target": "openreward"},
]


class EchoTaskSpec(BaseModel):
    # [CLASS DOC]
    # HELPER: Pydantic model validating a task spec: id (str), target (str) and failure_reward (float, default 0.0)
    # returned on a wrong echo. No methods.
    id: str
    target: str
    failure_reward: float = 0.0


class EchoParams(BaseModel):
    # [CLASS DOC]
    # HELPER: Pydantic model for the echo tool arguments: text (str). No methods.
    text: str


class HintParams(BaseModel):
    # [CLASS DOC]
    # HELPER: Empty pydantic model used as the hint tool's argument schema. No fields or methods.
    """Parameters for the optional hint tool (session-only via ``list_task_tools``)."""

    pass


class EchoEnvironment(Environment):
    # [CLASS DOC]
    # HELPER: ORS Environment subclass where the model must call the echo tool with the task's target string.
    # State: config (EchoTaskSpec parsed from the task spec). Class methods enumerate splits/tasks; get_prompt gives
    # the instruction; echo is the shared scoring tool; hint is a session-only tool advertised via list_task_tools.
    """Tiny deterministic ORS env: echo the target string to win."""

    def __init__(self, task_spec: JSONObject = {}, secrets: dict[str, str] = {}):  # noqa: B006 - signature dictated by openreward.environments.Environment
        # [FUNCTION DOC]
        # HELPER: Initialize the base Environment with task_spec and validate it into self.config.
        # Inputs: task_spec (dict, default empty), secrets (dict, ignored). Outputs: none.
        # Exceptions: pydantic ValidationError if id or target is missing (including with the empty default).
        # Side effects: sets self.config.
        super().__init__(task_spec)
        self.config = EchoTaskSpec.model_validate(task_spec)

    @classmethod
    def list_splits(cls) -> list[str]:
        # [FUNCTION DOC]
        # HELPER: Return the available splits, always a single-element list containing "train".
        # Inputs: none. Outputs: list of str. Exceptions: none. Side effects: none.
        return ["train"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        # [FUNCTION DOC]
        # HELPER: Return TRAIN_TASKS (the shared module list, not a copy) for split "train".
        # Inputs: split (str). Outputs: list of task dicts. Exceptions: ValueError for any other split. Side effects: none.
        if split != "train":
            raise ValueError(f"unknown split: {split}")
        return TRAIN_TASKS

    def get_prompt(self) -> list[TextBlock]:
        # [FUNCTION DOC]
        # HELPER: Return a one-element list with a text block telling the model to call echo with the configured target.
        # Inputs: none. Outputs: list of TextBlock. Exceptions: none. Side effects: none.
        return [
            TextBlock(
                type="text",
                text=f"Call the `echo` tool with text='{self.config.target}' to win.",
            )
        ]

    def list_task_tools(self) -> ListToolsOutput:
        # [FUNCTION DOC]
        # HELPER: Advertise the task-specific hint tool: returns ListToolsOutput with one ToolSpec named hint whose input
        # schema is HintParams' JSON schema. Inputs: none. Exceptions: none. Side effects: none.
        """Expose ``hint`` only on GET /task_tools (merged into ``session.list_tools()``)."""
        return ListToolsOutput(
            tools=[
                ToolSpec(
                    name="hint",
                    description="Cheap hint for tests exercising ORS /task_tools discovery.",
                    input_schema=HintParams.model_json_schema(),
                )
            ]
        )

    @tool
    async def echo(self, params: EchoParams) -> ToolOutput:
        # [FUNCTION DOC]
        # HELPER: Shared ORS tool scoring the submission.
        # Behavior: compares params.text to config.target exactly; returns a ToolOutput with one text block ("match", or
        # "no match (got <repr of text>)"), reward 1.0 on match else config.failure_reward, and finished set on match.
        # Inputs: params (EchoParams). Outputs: ToolOutput. Exceptions: none. Side effects: none.
        """Submit a string. Reward 1.0 + finished if it matches the target.

        Args:
            text: The string to echo back.
        """
        correct = params.text == self.config.target
        return ToolOutput(
            blocks=[
                TextBlock(
                    type="text",
                    text="match" if correct else f"no match (got {params.text!r})",
                )
            ],
            reward=1.0 if correct else self.config.failure_reward,
            finished=correct,
        )

    @tool(shared=False)
    async def hint(self, params: HintParams) -> ToolOutput:
        # [FUNCTION DOC]
        # HELPER: Non-shared (session-only) tool returning a ToolOutput with a single text block suggesting to call echo;
        # no reward or finished flag set. Inputs: params (HintParams, unused). Exceptions: none. Side effects: none.
        """Session-advertised helper (spec comes from ``list_task_tools``)."""
        return ToolOutput(blocks=[TextBlock(text="try echo(text=...)")])


if __name__ == "__main__":
    import os

    Server([EchoEnvironment]).run(host="127.0.0.1", port=int(os.environ["PORT"]))
