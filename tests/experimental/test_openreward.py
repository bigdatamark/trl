# [MODULE DOC]
# End-to-end tests for trl.experimental.openreward (OpenRewardSpec) against a real local ORS echo
# server (_openreward_echo_env.py run as a subprocess); no mocks. Covers lazy construction, dataset
# derivation (num_tasks / indices / metadata), tool binding on rollout environments, reset/echo
# reward semantics, the reward function over environments, session isolation, and task-tool
# discovery options. The echo env's tasks have targets hello, world, trl, ... in order.
# Import-time side effects: computes module path constants only.

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

"""Tests for `trl.experimental.openreward`.

A class-scoped fixture spawns ``_openreward_echo_env.py`` as a uvicorn subprocess on a free port and points the
openreward SDK at it via the ``OPENREWARD_API_URL`` / ``OPENREWARD_SESSION_URL`` overrides. Tests then exercise the
adapter end-to-end against real HTTP — no mocks, no network.

The same env definition is published at ``trl-internal-testing/openreward-echo-env`` if you want to point at the hosted
Space directly.
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

from trl.experimental.openreward import OpenRewardSpec

from ..testing_utils import TrlTestCase, require_openreward


_HERE = Path(__file__).parent
_ECHO_ENV_SCRIPT = _HERE / "_openreward_echo_env.py"


def _free_port() -> int:
    # [FUNCTION DOC]
    # HELPER: Returns a free localhost TCP port by binding a socket to port 0 and reading the assigned
    # port, then closing it. Inputs: none. Outputs: int. Exceptions: OSError on socket failure.
    # Side effects: briefly opens a socket.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="class")
def echo_env_url():
    # [FUNCTION DOC]
    # HELPER: Class-scoped pytest fixture. Starts the echo env script with the current interpreter on a
    # free port (PORT env var, output discarded), polls GET /health every 0.2s for up to 30s until 200.
    # Then saves and overrides OPENREWARD_API_URL and OPENREWARD_SESSION_URL to the URL and sets
    # OPENREWARD_API_KEY to "test" if unset; yields the base URL. Teardown restores the saved env vars
    # (removing ones that were unset), terminates the process, and kills it if it does not exit in 5s.
    # Inputs: none. Outputs: yields str URL.
    # Exceptions: RuntimeError if the server is not ready in 30s (process terminated first).
    # Side effects: subprocess, os.environ mutation.
    """Spawn the echo env on a free port; tear down on teardown."""
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(_ECHO_ENV_SCRIPT)],
        env={**os.environ, "PORT": str(port)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30.0
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=1.0)
            if r.status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(0.2)
    else:
        proc.terminate()
        raise RuntimeError(f"echo env did not become ready at {url}")

    # The openreward SDK by default rewrites base_url into api.<host> /
    # sessions.<host>; for a single-host self-hosted server these env vars
    # bypass that two-subdomain layout.
    saved = {k: os.environ.get(k) for k in ("OPENREWARD_API_URL", "OPENREWARD_SESSION_URL", "OPENREWARD_API_KEY")}
    os.environ["OPENREWARD_API_URL"] = url
    os.environ["OPENREWARD_SESSION_URL"] = url
    os.environ.setdefault("OPENREWARD_API_KEY", "test")

    yield url

    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    proc.terminate()
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()


@require_openreward
@pytest.mark.usefixtures("echo_env_url")
class TestOpenRewardSpec(TrlTestCase):
    # [CLASS DOC]
    # Tests the public OpenRewardSpec surface against the echo server. Requires the openreward package;
    # uses the echo_env_url class fixture. Inherits TrlTestCase. Most tests build a spec with
    # env_name "echoenvironment" and use environment_factory to create rollout envs.
    """Exercises the public `OpenRewardSpec` surface against a real ORS server."""

    def test_construction_is_lazy(self, echo_env_url):
        # [FUNCTION DOC]
        # Tests construction does no HTTP and stores _target (the URL), _is_url True, and _num_tasks 2.
        # Inputs: echo_env_url (str fixture). Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: HTTP calls to the local echo server; opened sessions are closed via env._close().
        # Construction must not perform any HTTP — `train_dataset` /
        # `environment_factory` are `cached_property` and only fire on access.
        spec = OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=2)
        # Touching only private attributes should not have hit the network.
        assert spec._target == echo_env_url
        assert spec._is_url is True
        assert spec._num_tasks == 2

    def test_train_dataset_derives_from_env(self, echo_env_url):
        # [FUNCTION DOC]
        # Tests train_dataset with num_tasks=2 has 2 rows with columns prompt, task_index, and target
        # (metadata folded in); row 0 task_index 0 target hello, row 1 target world.
        # Inputs: echo_env_url (str fixture). Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: HTTP calls to the local echo server; opened sessions are closed via env._close().
        spec = OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=2)
        ds = spec.train_dataset
        assert len(ds) == 2
        assert "prompt" in ds.column_names
        assert "task_index" in ds.column_names
        # Per-task metadata folded in (id, target) when include_metadata=True.
        assert "target" in ds.column_names
        assert ds[0]["task_index"] == 0
        assert ds[0]["target"] == "hello"
        assert ds[1]["target"] == "world"

    def test_train_dataset_with_indices(self, echo_env_url):
        # [FUNCTION DOC]
        # Tests indices=[0, 2] selects tasks with targets hello and trl.
        # Inputs: echo_env_url (str fixture). Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: HTTP calls to the local echo server; opened sessions are closed via env._close().
        spec = OpenRewardSpec(echo_env_url, env_name="echoenvironment", indices=[0, 2])
        ds = spec.train_dataset
        assert [row["target"] for row in ds] == ["hello", "trl"]

    def test_num_tasks_and_indices_are_mutually_exclusive(self, echo_env_url):
        # [FUNCTION DOC]
        # Tests passing both num_tasks and indices raises ValueError.
        # Inputs: echo_env_url (str fixture). Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: HTTP calls to the local echo server; opened sessions are closed via env._close().
        with pytest.raises(ValueError):
            OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=2, indices=[0])

    def test_environment_factory_returns_rollout_env_with_bound_tools(self, echo_env_url):
        # [FUNCTION DOC]
        # Tests a rollout env has a callable echo tool annotated text str returning str (shared tool) and a
        # callable hint tool (task-scoped tool).
        # Inputs: echo_env_url (str fixture). Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: HTTP calls to the local echo server; opened sessions are closed via env._close().
        spec = OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=2)
        env = spec.environment_factory()
        # Shared + task-scoped ORS tools (ORS /tools vs /task_tools) are both bound for GRPO.
        assert callable(env.echo)
        sig = env.echo.__annotations__
        assert sig["text"] is str
        assert sig["return"] is str
        assert callable(env.hint)

    def test_discover_task_tools_false_skips_task_scoped_binding(self, echo_env_url):
        # [FUNCTION DOC]
        # Tests discover_task_tools=False binds echo but not hint.
        # Inputs: echo_env_url (str fixture). Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: HTTP calls to the local echo server; opened sessions are closed via env._close().
        spec = OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=1, discover_task_tools=False)
        env = spec.environment_factory()
        assert callable(env.echo)
        assert not hasattr(env, "hint")

    def test_reset_returns_prompt_and_opens_session(self, echo_env_url):
        # [FUNCTION DOC]
        # Tests reset with dataset row 0 returns a prompt mentioning echo and hello.
        # Inputs: echo_env_url (str fixture). Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: HTTP calls to the local echo server; opened sessions are closed via env._close().
        spec = OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=1)
        env = spec.environment_factory()
        prompt = env.reset(**spec.train_dataset[0])
        assert "echo" in prompt and "hello" in prompt
        env._close()

    def test_correct_echo_returns_match_with_reward_and_finished(self, echo_env_url):
        # [FUNCTION DOC]
        # Tests echo with the correct text returns output containing match, sets reward 1.0 and finished True.
        # Inputs: echo_env_url (str fixture). Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: HTTP calls to the local echo server; opened sessions are closed via env._close().
        spec = OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=1)
        env = spec.environment_factory()
        env.reset(**spec.train_dataset[0])
        out = env.echo(text="hello")
        assert "match" in out
        assert env.reward == 1.0
        assert env.finished is True
        env._close()

    def test_wrong_echo_returns_no_match_with_zero_reward(self, echo_env_url):
        # [FUNCTION DOC]
        # Tests echo with a wrong text returns "no match", reward 0.0, finished False.
        # Inputs: echo_env_url (str fixture). Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: HTTP calls to the local echo server; opened sessions are closed via env._close().
        spec = OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=1)
        env = spec.environment_factory()
        env.reset(**spec.train_dataset[0])
        out = env.echo(text="goodbye")
        assert "no match" in out
        assert env.reward == 0.0
        assert env.finished is False
        env._close()

    def test_reward_func_reads_last_non_null_per_environment(self, echo_env_url):
        # [FUNCTION DOC]
        # Tests reward_funcs over two envs (one correct echo, one wrong) returns [1.0, 0.0].
        # Inputs: echo_env_url (str fixture). Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: HTTP calls to the local echo server; opened sessions are closed via env._close().
        spec = OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=2)
        env_a = spec.environment_factory()
        env_b = spec.environment_factory()
        env_a.reset(**spec.train_dataset[0])
        env_b.reset(**spec.train_dataset[1])
        env_a.echo(text="hello")  # match → reward=1.0
        env_b.echo(text="oops")  # no match → reward=0.0
        rewards = spec.reward_funcs(environments=[env_a, env_b])
        assert rewards == [1.0, 0.0]
        env_a._close()
        env_b._close()

    def test_factory_produces_isolated_sessions(self, echo_env_url):
        # [FUNCTION DOC]
        # Tests that rewarding one env (1.0) leaves a second env's reward as None.
        # Inputs: echo_env_url (str fixture). Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: HTTP calls to the local echo server; opened sessions are closed via env._close().
        # GRPO opens N concurrent envs; mutating one must not leak into another.
        spec = OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=2)
        env_a = spec.environment_factory()
        env_b = spec.environment_factory()
        env_a.reset(**spec.train_dataset[0])
        env_b.reset(**spec.train_dataset[1])
        env_a.echo(text="hello")
        assert env_a.reward == 1.0
        assert env_b.reward is None  # untouched
        env_a._close()
        env_b._close()

    @pytest.mark.parametrize(
        "calls, failure_reward, expected_history, expected_reward",
        [
            ([], 0.0, [], None),
            (["hint"], 0.0, [None], None),
            (["missing_tool"], 0.0, [None], None),
            (["wrong"], 0.0, [0.0], 0.0),
            (["wrong"], -1.0, [-1.0], -1.0),
            (["correct"], 0.0, [1.0], 1.0),
            (["wrong", "hint"], 0.0, [0.0, None], 0.0),
            (["wrong", "missing_tool"], -1.0, [-1.0, None], -1.0),
            (["wrong", "hint", "correct"], -1.0, [-1.0, None, 1.0], 1.0),
        ],
    )
    def test_reward_func_distinguishes_missing_and_observed_rewards(
        self, echo_env_url, calls, failure_reward, expected_history, expected_reward
    ):
        # [FUNCTION DOC]
        # Parametrized over sequences of calls (wrong / correct / hint / missing_tool) and failure_reward.
        # Resets with an explicit task_spec (target hello, given failure_reward), performs the calls (a
        # missing tool must return a string starting with "Error:"), then asserts env.rewards equals the
        # expected per-call history (None for non-scoring calls) and reward_funcs returns the last non-None
        # reward or None.
        # Inputs: echo_env_url, calls (list[str]), failure_reward (float), expected_history (list),
        # expected_reward (float or None). Outputs: none. Exceptions: AssertionError.
        # Side effects: HTTP; env closed in finally.
        spec = OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=1)
        env = spec.environment_factory()
        try:
            env.reset(task_spec={"id": "reward-test", "target": "hello", "failure_reward": failure_reward})
            for call in calls:
                if call == "wrong":
                    env.echo(text="wrong")
                elif call == "correct":
                    env.echo(text="hello")
                elif call == "hint":
                    env.hint()
                else:
                    assert env._call_ors_tool("missing_tool", {}).startswith("Error:")
            assert env.rewards == expected_history
            assert spec.reward_funcs(environments=[env]) == [expected_reward]
        finally:
            env._close()

    @pytest.mark.parametrize("failure_reward", [0.0, -1.0])
    def test_reward_func_reset_clears_previous_outcome(self, echo_env_url, failure_reward):
        # [FUNCTION DOC]
        # Parametrized over failure_reward 0.0 and -1.0. After a wrong echo the reward equals
        # failure_reward; after reset(task_index=0) rewards is empty and reward_funcs returns [None].
        # Inputs: echo_env_url, failure_reward (float). Outputs: none. Exceptions: AssertionError.
        # Side effects: HTTP; env closed in finally.
        spec = OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=1)
        env = spec.environment_factory()
        try:
            env.reset(task_spec={"id": "reward-test", "target": "hello", "failure_reward": failure_reward})
            env.echo(text="wrong")
            assert spec.reward_funcs(environments=[env]) == [failure_reward]
            env.reset(task_index=0)
            assert env.rewards == []
            assert spec.reward_funcs(environments=[env]) == [None]
        finally:
            env._close()

    def test_reward_func_mixed_scoring_coverage(self, echo_env_url):
        # [FUNCTION DOC]
        # Tests three envs (untouched, wrong, correct) yield rewards [None, 0.0, 1.0].
        # Inputs: echo_env_url (str fixture). Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: HTTP calls to the local echo server; opened sessions are closed via env._close().
        spec = OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=1)
        envs = [spec.environment_factory() for _ in range(3)]
        try:
            for env in envs:
                env.reset(task_index=0)
            envs[1].echo(text="wrong")
            envs[2].echo(text="hello")
            assert spec.reward_funcs(environments=envs) == [None, 0.0, 1.0]
        finally:
            for env in envs:
                env._close()

    def test_metadata_does_not_overwrite_reserved_columns(self, echo_env_url):
        # [FUNCTION DOC]
        # Tests the prompt column remains a list of message dicts whose first role is user and task_index
        # is an int, i.e. metadata never overwrites reserved columns.
        # Inputs: echo_env_url (str fixture). Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: HTTP calls to the local echo server; opened sessions are closed via env._close().
        # If a task spec ever shipped a `prompt` key, the metadata loop must
        # not clobber our chat-format `prompt` column. Same for `task_index`.
        # We assert the shape directly — the echo env's task spec doesn't
        # currently have either, but the guard is what we're testing.
        spec = OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=1)
        ds = spec.train_dataset
        # `prompt` is a list-of-message-dicts, not a string from task spec.
        assert isinstance(ds[0]["prompt"], list)
        assert ds[0]["prompt"][0]["role"] == "user"
        # `task_index` is the int we set, not anything from the spec.
        assert isinstance(ds[0]["task_index"], int)

    def test_task_tools_discovery_index_probes_single_task(self, echo_env_url):
        # [FUNCTION DOC]
        # Tests task_tools_discovery_index=0 with num_tasks=4 still binds echo and hint and keeps 4 rows.
        # Inputs: echo_env_url (str fixture). Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: HTTP calls to the local echo server; opened sessions are closed via env._close().
        # task_tools_discovery_index=0 tells the spec to probe only task 0 for
        # tool discovery (ORS /task_tools), regardless of how many tasks are in
        # the dataset. This is the Toolathlon pattern: all tasks expose the same
        # meta-tools, so probing one is sufficient and avoids N sessions.
        spec = OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=4, task_tools_discovery_index=0)
        env = spec.environment_factory()
        # Shared tool (echo) and task-specific tool (hint) are both bound.
        assert callable(env.echo)
        assert callable(env.hint)
        # Dataset still has 4 tasks even though discovery only probed index 0.
        assert len(spec.train_dataset) == 4
        env._close()

    def test_task_tools_discovery_index_with_indices(self, echo_env_url):
        # [FUNCTION DOC]
        # Tests indices [0..3] with task_tools_discovery_index=2 still binds hint and keeps 4 rows.
        # Inputs: echo_env_url (str fixture). Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: HTTP calls to the local echo server; opened sessions are closed via env._close().
        # When indices= is set AND task_tools_discovery_index is set, discovery
        # uses only the explicit discovery index (not all indices for probing).
        spec = OpenRewardSpec(
            echo_env_url, env_name="echoenvironment", indices=[0, 1, 2, 3], task_tools_discovery_index=2
        )
        env = spec.environment_factory()
        # Task tools are still discovered (via index 2).
        assert callable(env.hint)
        assert len(spec.train_dataset) == 4
        env._close()

    def test_two_specs_get_isolated_rollout_subclasses(self, echo_env_url):
        # [FUNCTION DOC]
        # Tests envs from two separate specs have distinct types (per-spec rollout subclasses) and both
        # have a callable echo.
        # Inputs: echo_env_url (str fixture). Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: HTTP calls to the local echo server; opened sessions are closed via env._close().
        # Two specs (potentially against different envs with different tool
        # sets) must each produce rollout instances with their own subclass,
        # so neither side's bound tools clobber or shadow the other's.
        spec_a = OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=1)
        spec_b = OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=1)
        env_a = spec_a.environment_factory()
        env_b = spec_b.environment_factory()
        # Each rollout is a distinct subclass of _RolloutEnvironment.
        assert type(env_a) is not type(env_b)
        # Both subclasses still get their own `echo` method.
        assert callable(env_a.echo) and callable(env_b.echo)
        env_a._close()
        env_b._close()
