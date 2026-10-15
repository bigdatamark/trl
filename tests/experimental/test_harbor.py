# [MODULE DOC]
# Tests for the Harbor x TRL integration (trl.experimental.harbor) that need no Harbor sandbox:
# agent resolution (_resolve_agent), HarborSpec dataset building from an on-disk task suite, the
# environment factory, and the outcome reward function. Defines a helper that writes fake task dirs.
# Import-time side effects: none (harbor itself is imported lazily by the library).

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

"""Tests for the Harbor x TRL integration that don't need a running Harbor sandbox.

`harbor` is imported lazily (only when an env is *started*), so spec construction, agent resolution, dataset building,
and the reward function are all testable without `harbor` / a sandbox backend.
"""

from pathlib import Path

import pytest

from trl.experimental.harbor import AGENTS, HarborBashEnv, HarborEnv, HarborSpec
from trl.experimental.harbor._spec import _outcome_reward_func, _resolve_agent

from ..testing_utils import TrlTestCase


def _write_task(tasks_dir: Path, task_id: str, gold: str, difficulty: int) -> None:
    # [FUNCTION DOC]
    # HELPER: Creates a fake Harbor task directory tasks_dir/task_id containing empty environment and
    # tests subdirs, an instruction.md ("Solve task <id>."), and a task.toml with a [task] name and a
    # [metadata] table (gold_answer, reward_mode_initial exact_short, difficulty_level,
    # kaggle_dataset_name "owner/<id>"). The TOML is built by joining a list of lines with newlines.
    # Inputs: tasks_dir (Path), task_id (str), gold (str), difficulty (int). Outputs: None.
    # Exceptions: FileExistsError if the task dir already exists. Side effects: writes files.
    d = tasks_dir / task_id
    (d / "environment").mkdir(parents=True)
    (d / "tests").mkdir()
    (d / "instruction.md").write_text(f"Solve task {task_id}.")
    # Built from a joined list (not a triple-quoted block) so doc-builder doesn't reflow the TOML.
    lines = [
        "[task]",
        f'name = "{task_id}"',
        "[metadata]",
        f'gold_answer = "{gold}"',
        'reward_mode_initial = "exact_short"',
        f"difficulty_level = {difficulty}",
        f'kaggle_dataset_name = "owner/{task_id}"',
    ]
    (d / "task.toml").write_text("\n".join(lines))


class TestResolveAgent(TrlTestCase):
    # [CLASS DOC]
    # Tests _resolve_agent, which maps an agent spec (builtin name, class, "module:Class" import path,
    # or "file.py:Class" path) to a HarborEnv subclass. Inherits TrlTestCase for self.tmp_dir.
    def test_builtin_name(self):
        # [FUNCTION DOC]
        # Tests that "bash" resolves to HarborBashEnv and that AGENTS["bash"] is HarborBashEnv.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure. Side effects: none.
        assert _resolve_agent("bash") is HarborBashEnv
        assert AGENTS["bash"] is HarborBashEnv

    def test_class_passthrough(self):
        # [FUNCTION DOC]
        # Tests that passing the HarborBashEnv class returns it unchanged.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure. Side effects: none.
        assert _resolve_agent(HarborBashEnv) is HarborBashEnv

    def test_import_path(self):
        # [FUNCTION DOC]
        # Tests that the import path "trl.experimental.harbor:HarborBashEnv" resolves to HarborBashEnv.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure. Side effects: none.
        assert _resolve_agent("trl.experimental.harbor:HarborBashEnv") is HarborBashEnv

    def test_file_path(self):
        # [FUNCTION DOC]
        # Tests file-path resolution: writes my_harness.py in tmp_dir defining MyEnv (a HarborEnv subclass
        # with a run_cmd tool), resolves "<path>:MyEnv", and asserts the result subclasses HarborEnv and is
        # named MyEnv.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes a file and imports it.
        path = Path(self.tmp_dir) / "my_harness.py"
        path.write_text(
            "from trl.experimental.harbor import HarborEnv\n"
            "class MyEnv(HarborEnv):\n"
            "    def run_cmd(self, command: str) -> str:\n"
            "        'Run a command.\\n\\nArgs:\\n    command: cmd.'\n"
            "        return self._exec(command)\n"
        )
        cls = _resolve_agent(f"{path}:MyEnv")
        assert issubclass(cls, HarborEnv) and cls.__name__ == "MyEnv"

    def test_unknown_name_raises(self):
        # [FUNCTION DOC]
        # Tests that an unknown bare name raises ValueError.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure. Side effects: none.
        with pytest.raises(ValueError):
            _resolve_agent("not-a-harness")

    def test_non_harborenv_raises(self):
        # [FUNCTION DOC]
        # Tests that an import path to a class that is not a HarborEnv subclass (HarborSpec) raises TypeError.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure. Side effects: none.
        with pytest.raises(TypeError):
            _resolve_agent("trl.experimental.harbor:HarborSpec")  # not a HarborEnv subclass


class TestHarborSpecDataset(TrlTestCase):
    # [CLASS DOC]
    # Tests HarborSpec over a two-task fake suite (0001_a gold alpha difficulty 0; 0002_b gold beta
    # difficulty 3): dataset columns, num_tasks / indices selection, and environment_factory.
    def _suite(self) -> str:
        # [FUNCTION DOC]
        # HELPER: Creates tmp_dir/tasks with the two fake tasks via _write_task and returns tmp_dir as str.
        # Inputs: none. Outputs: str suite root. Exceptions: FileExistsError if called twice per test.
        # Side effects: writes files under tmp_dir.
        tasks = Path(self.tmp_dir) / "tasks"
        tasks.mkdir()
        _write_task(tasks, "0001_a", "alpha", 0)
        _write_task(tasks, "0002_b", "beta", 3)
        return str(self.tmp_dir)

    def test_train_dataset_columns_and_metadata(self):
        # [FUNCTION DOC]
        # Tests train_dataset has 2 rows; row 0 has a single empty user-message prompt, task_dir ending in
        # 0001_a, task_index 0, gold_answer alpha; row 1 has difficulty_level 3.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes the fake suite.
        ds = HarborSpec(self._suite()).train_dataset
        assert len(ds) == 2
        assert ds[0]["prompt"] == [{"role": "user", "content": ""}]  # env appends instruction at reset
        assert ds[0]["task_dir"].endswith("0001_a")
        assert ds[0]["task_index"] == 0
        assert ds[0]["gold_answer"] == "alpha"
        assert ds[1]["difficulty_level"] == 3

    def test_num_tasks_cap(self):
        # [FUNCTION DOC]
        # Tests num_tasks=1 limits the dataset to 1 row.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes the fake suite.
        ds = HarborSpec(self._suite(), num_tasks=1).train_dataset
        assert len(ds) == 1

    def test_indices_selection(self):
        # [FUNCTION DOC]
        # Tests indices=[1] yields exactly one row, the 0002_b task.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes the fake suite.
        ds = HarborSpec(self._suite(), indices=[1]).train_dataset
        assert len(ds) == 1 and ds[0]["task_dir"].endswith("0002_b")

    def test_num_tasks_and_indices_mutually_exclusive(self):
        # [FUNCTION DOC]
        # Tests that passing both num_tasks and indices raises ValueError.
        # Inputs/Outputs: none. Exceptions: AssertionError if not raised. Side effects: writes the fake suite.
        with pytest.raises(ValueError):
            HarborSpec(self._suite(), num_tasks=1, indices=[0])

    def test_environment_factory_returns_fresh_envs(self):
        # [FUNCTION DOC]
        # Tests that environment_factory with agent "bash" returns a new HarborBashEnv instance per call.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes the fake suite.
        factory = HarborSpec(self._suite(), agent="bash").environment_factory
        e1, e2 = factory(), factory()
        assert isinstance(e1, HarborBashEnv) and e1 is not e2


class TestRewardFunc(TrlTestCase):
    # [CLASS DOC]
    # Tests _outcome_reward_func and the default reward of a fresh HarborBashEnv.
    def test_outcome_reward_reads_env_reward(self):
        # [FUNCTION DOC]
        # Tests that given env objects (local stub class with a reward attribute), the function returns
        # their reward values in order ([1.0, 0.0]).
        # Inputs/Outputs: none. Exceptions: AssertionError on failure. Side effects: none.
        class _Env:
            # [CLASS DOC]
            # HELPER: Minimal stand-in environment exposing only a reward attribute.
            def __init__(self, r):
                # [FUNCTION DOC]
                # HELPER: Stores r as self.reward. Inputs: r (float). Outputs: none. Exceptions: none.
                # Side effects: sets attribute.
                self.reward = r

        assert _outcome_reward_func([_Env(1.0), _Env(0.0)]) == [1.0, 0.0]

    def test_outcome_reward_uses_environment_reward_when_passed(self):
        # [FUNCTION DOC]
        # Tests that when environment_reward (list of floats) is passed with no envs, it is returned as is.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure. Side effects: none.
        # AsyncGRPOTrainer captures rewards in its rollout worker and passes them as a list, with no
        # live env instances. The reward func must use them directly.
        assert _outcome_reward_func(environment_reward=[0.25, 0.75]) == [0.25, 0.75]

    def test_fresh_env_reward_is_zero_without_backend(self):
        # [FUNCTION DOC]
        # Tests that a never-reset HarborBashEnv exposes bash and reset as class functions (found via
        # inspect.getmembers on the type) and that its reward is 0.0 without starting a backend or
        # importing harbor.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure. Side effects: none.
        # The trainer discovers tool methods on the environment's class (`inspect.getmembers` on the
        # instance would evaluate `reward`). A fresh env (never `reset`) must expose its tools and return
        # 0.0 from `reward` WITHOUT starting the Harbor backend or importing `harbor` (not installed in
        # the trainer env).
        import inspect

        env = HarborBashEnv()
        names = {n for n, _ in inspect.getmembers(type(env), predicate=inspect.isfunction)}
        assert {"bash", "reset"} <= names
        assert env.reward == 0.0
