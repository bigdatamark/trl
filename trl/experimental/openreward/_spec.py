# [MODULE DOC]
# User-facing OpenRewardSpec for plugging an OpenReward Standard (ORS) environment into GRPOTrainer.
# Contents: small helpers normalizing SDK outputs (_tool_specs_from_session_list_tools, _to_spec), the default
# outcome-only reward function, and OpenRewardSpec which lazily builds a training dataset, an environment factory
# producing _RolloutEnvironment instances, and exposes the reward function.
# Relies on environment.py for the lazy SDK import and rollout adapter. Import-time side effects: creates a logger;
# the openreward SDK is not imported until a property is accessed.

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

"""User-facing spec object for the OpenReward × TRL integration.

The user constructs **one** ``OpenRewardSpec`` (a thin specification holding the env target + a few options) and reads
three properties off of it — ``.train_dataset``, ``.environment_factory``, ``.reward_funcs`` — each of which plugs
directly into the matching ``GRPOTrainer`` kwarg:

```python
>>> from trl import GRPOConfig, GRPOTrainer
>>> from trl.experimental.openreward import OpenRewardSpec

>>> spec = OpenRewardSpec("Eigent/SETA", num_tasks=64)

>>> trainer = GRPOTrainer(
...     model="Qwen/Qwen3-4B",
...     args=GRPOConfig(num_generations=2, max_steps=5, max_tool_calling_iterations=20),
...     train_dataset=spec.train_dataset,
...     environment_factory=spec.environment_factory,
...     reward_funcs=spec.reward_funcs,
... )
>>> trainer.train()
```

Backed by the official ``openreward`` SDK (lazy-imported); install with ``pip install trl[openreward]``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from functools import cached_property
from typing import Any

from .environment import _import_openreward, _RolloutEnvironment, _spec_to_dict


logger = logging.getLogger(__name__)


def _tool_specs_from_session_list_tools(out: Any) -> list[Any]:
    # [FUNCTION DOC]
    # HELPER: Converts the result of a list_tools call (an iterable of ToolSpec) to a plain list.
    # Inputs: out (iterable). Outputs: list. Exceptions: TypeError if not iterable. Side effects: none.
    """Normalise ``session.list_tools()`` / ``environment.list_tools()`` to a flat list.

    The client SDK (``openreward.api``) returns ``list[ToolSpec]`` directly.
    """
    return list(out)


def _to_spec(task) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Returns a task's spec dict: the input itself when it is a dict, otherwise its task_spec attribute.
    # Inputs: task (SDK Task or dict). Outputs: dict. Exceptions: AttributeError for other objects. Side effects: none.
    """Normalise an SDK ``Task`` (or already-a-dict) to its task_spec dict."""
    if isinstance(task, dict):
        return task
    return task.task_spec


def _outcome_only_reward_func(environments, **_) -> list[float | None]:
    # [FUNCTION DOC]
    # HELPER: Default reward function: returns each environment's reward attribute (the last non-null step reward, or None
    # when no step produced a reward, as maintained by _RolloutEnvironment).
    # Inputs: environments (list of rollout environments), other keyword arguments ignored.
    # Outputs: list of float or None, one per environment. Exceptions: none. Side effects: none.
    """Default reward function: last non-null reward in each rollout's trajectory.

    Suitable for sparse-outcome envs (e.g. SETA, where only `submit_solution` returns a non-null reward). Override by
    passing a different callable to ``reward_funcs=``. Returns `None` when no non-null reward was observed, so an
    unscored rollout is not treated as a scored zero.
    """
    return [env.reward for env in environments]


class OpenRewardSpec:
    # [CLASS DOC]
    # Specification object holding an ORS target and options; the trainer-facing objects are derived lazily.
    # State: private copies of constructor options plus _is_url (target contains "://"). Cached properties _sdk_client
    # (OpenReward client) and _sdk_env (SDK Environment) are shared by train_dataset and environment_factory, so the SDK is
    # contacted only once per spec.
    # Flow: train_dataset lists tasks into rows; environment_factory discovers tools (_discovered_tool_specs) and returns a
    # closure creating per-rollout environments; reward_funcs returns the module-level default reward function.
    # No base class.
    """Single spec object that wires an ORS environment into a TRL trainer.

    Args:
        target (`str`):
            Either an openreward.ai catalog name (`"Eigent/SETA"`) or a URL pointing at any ORS server
            (`"https://you-seta.hf.space"`, `"http://localhost:8080"`). Auto-detected by the presence of `://` in the
            string.
        num_tasks (`int`, *optional*):
            Cap on the number of tasks pulled into the dataset. ``None`` uses every task the env exposes.
        split (`str`, *optional*, defaults to `"train"`):
            Which split's task list to draw from.
        indices (`list[int]`, *optional*):
            Specific task indices to train on. Mutually exclusive with ``num_tasks``. Useful for debugging or
            curriculum subsets.
        api_key (`str`, *optional*):
            ``OPENREWARD_API_KEY`` override. Only used when ``target`` is a catalog name.
        secrets (`dict[str, str]`, *optional*):
            Per-session secrets forwarded to ``env.session(secrets=)``.
        env_name (`str`, *optional*):
            Override for the env name to look up on the server. Rarely needed.
        include_metadata (`bool`, *optional*, defaults to `True`):
            Fold per-task metadata (`difficulty`, `category`, `tags`, ...) into the dataset rows so reward funcs can
            read them via TRL's ``inputs`` argument.
        discover_task_tools (`bool`, *optional*, defaults to `True`):
            If ``True``, opens a short-lived ORS session and uses ``session.list_tools()`` so task-specific tools
            (``GET …/task_tools`` per ORS — e.g. ``@tool(shared=False)`` and ``list_task_tools()``) are bound for GRPO.
            If probe fails, falls back to ``environment.list_tools()`` only. Set ``False`` to skip the extra session
            (shared tools only / offline quirks).
        task_tools_discovery_index (`int`, *optional*):
            Task index used only for the discovery session when set; **overrides** multi-index probing below. When
            omitted and ``indices=`` is set, discovery opens one probe session per **distinct** entry in ``indices``
            (sorted) and **merges** tool specs by name so task-specific tools from every listed task are bound. When
            omitted and ``num_tasks`` / full-list mode is used, probes task ``0`` only. Ignored when
            ``discover_task_tools=False``.
    """

    def __init__(
        self,
        target: str,
        *,
        num_tasks: int | None = None,
        split: str = "train",
        indices: list[int] | None = None,
        api_key: str | None = None,
        secrets: dict[str, str] | None = None,
        env_name: str | None = None,
        include_metadata: bool = True,
        discover_task_tools: bool = True,
        task_tools_discovery_index: int | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Stores options without any network access.
        # Inputs: target (catalog name or URL), keyword-only num_tasks, split ("train"), indices, api_key, secrets, env_name,
        # include_metadata (True), discover_task_tools (True), task_tools_discovery_index.
        # Outputs: none. Exceptions: ValueError when both num_tasks and indices are given.
        # Side effects: sets private attributes.
        if num_tasks is not None and indices is not None:
            raise ValueError("Provide num_tasks or indices, not both.")

        self._target = target
        self._is_url = "://" in target
        self._num_tasks = num_tasks
        self._split = split
        self._indices = indices
        self._api_key = api_key
        self._secrets = secrets
        self._env_name = env_name
        self._include_metadata = include_metadata
        self._discover_task_tools = discover_task_tools
        self._task_tools_discovery_index = task_tools_discovery_index

    # ── public surface ──────────────────────────────────────────────

    @cached_property
    def train_dataset(self):
        # [FUNCTION DOC]
        # Cached property building a datasets.Dataset from the environment's tasks.
        # Behavior: explicit indices fetch each task by index; num_tasks fetches the first min(num_tasks, total) tasks by index;
        # otherwise list_tasks for the split, falling back to per-index fetches if listing raises. Rows have "prompt" (a single
        # user message with empty content) and "task_index". With include_metadata, adds one column per key found in any dict
        # spec (sorted, excluding reserved prompt/task_index), None where missing.
        # Inputs: none. Outputs: Dataset.
        # Exceptions: SDK/network errors; ImportError if openreward missing.
        # Side effects: network calls; caches the result and the SDK client/env.
        """A `datasets.Dataset` derived from the env's task list.

        Plugs directly into TRL's ``train_dataset=`` slot. Built lazily on first access. Has at minimum:
          - `prompt`: empty user message (TRL appends the env's prompt).
          - `task_index`: int passed to the adapter's `reset()`.
          - per-task metadata columns (when `include_metadata=True`).
        """
        from datasets import Dataset

        env = self._sdk_env

        # When the user asked for a specific subset (`num_tasks` cap or
        # explicit `indices`), fetch only those — `list_tasks` returns
        # the whole split (1376 entries for SETA, etc.) and is slow on
        # the platform. Per-index `get_task` is fast and cheap.
        if self._indices is not None:
            indexes = list(self._indices)
            task_specs = [_to_spec(env.get_task(self._split, i)) for i in indexes]
        elif self._num_tasks is not None:
            n_total = env.num_tasks(self._split)
            n = min(self._num_tasks, n_total)
            indexes = list(range(n))
            task_specs = [_to_spec(env.get_task(self._split, i)) for i in indexes]
        else:
            # No cap — fetch everything in one shot.
            try:
                tasks = env.list_tasks(self._split)
            except Exception:  # noqa: BLE001
                n = env.num_tasks(self._split)
                tasks = [env.get_task(self._split, i) for i in range(n)]
            task_specs = [_to_spec(t) for t in tasks]
            indexes = list(range(len(task_specs)))

        rows: dict[str, list[Any]] = {
            "prompt": [[{"role": "user", "content": ""}] for _ in indexes],
            "task_index": indexes,
        }

        if self._include_metadata and task_specs:
            metadata_keys: set[str] = set()
            for spec in task_specs:
                if isinstance(spec, dict):
                    metadata_keys.update(spec.keys())
            # Never let a task-spec key overwrite our reserved row columns
            # (e.g. an env that exposes a `prompt` task-spec field would
            # otherwise replace our chat-format prompt with a raw string).
            metadata_keys -= rows.keys()
            for key in sorted(metadata_keys):
                rows[key] = [s.get(key) if isinstance(s, dict) else None for s in task_specs]

        return Dataset.from_dict(rows)

    @cached_property
    def environment_factory(self) -> Callable[[], _RolloutEnvironment]:
        # [FUNCTION DOC]
        # Cached property returning a zero-argument factory of rollout environments.
        # Behavior: discovers tool specs once, creates a fresh subclass of _RolloutEnvironment unique to this spec (named with
        # the spec id) so bound tool methods of different specs do not collide, and prepares constructor kwargs: split,
        # effective secrets, env_name, pre-built client, env and tool specs, plus base_url (URL target) or name and api_key
        # (catalog target); None values are dropped.
        # Inputs: none. Outputs: callable returning a new instance of that subclass.
        # Exceptions: SDK/network errors during discovery. Side effects: network calls, caching.
        """Zero-arg callable that returns a fresh ``_RolloutEnvironment``.

        Plugs directly into TRL's ``environment_factory=`` slot. TRL calls this once per rollout at trainer
        construction time, so each rollout has an isolated ORS session. Reuses the spec's already-discovered SDK env +
        tool specs to skip per-env HTTP at trainer init.
        """
        # Pre-fetch tool specs once at the spec level. ORS exposes shared tools on GET /tools and the full
        # shared+task-specific set on GET /task_tools (session-scoped). GRPO inspects methods before reset(), so we
        # must bind every tool the episode will see — use session.list_tools() when discover_task_tools=True.
        env = self._sdk_env
        tool_specs = self._discovered_tool_specs(env)
        client = self._sdk_client

        # Each spec gets its own `_RolloutEnvironment` subclass so two specs
        # for different envs (different tool sets) never clobber each other's
        # bound methods on the shared parent class.
        rollout_cls = type(f"_RolloutEnvironment_{id(self):x}", (_RolloutEnvironment,), {})

        kwargs = {
            "split": self._split,
            "secrets": self._effective_secrets(),
            "env_name": self._env_name,
            "_client": client,
            "_env": env,
            "_tool_specs": tool_specs,
        }
        if self._is_url:
            kwargs["base_url"] = self._target
        else:
            kwargs["name"] = self._target
            if self._api_key:
                kwargs["api_key"] = self._api_key

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        def _make() -> _RolloutEnvironment:
            # [FUNCTION DOC]
            # HELPER: Factory closure: constructs a new instance of the per-spec rollout subclass with the prepared kwargs.
            # Inputs: none. Outputs: a fresh rollout environment (binds tools on first construction; reuses client/env).
            # Exceptions: those of the rollout constructor. Side effects: none beyond construction.
            return rollout_cls(**kwargs)

        return _make

    @property
    def reward_funcs(self) -> Callable[..., list[float | None]]:
        # [FUNCTION DOC]
        # Returns the module-level _outcome_only_reward_func (stable identity, picklable).
        # Inputs: none. Outputs: the callable. Exceptions: none. Side effects: none.
        """Default outcome-only reward function (last non-null reward per rollout).

        Plugs directly into TRL's ``reward_funcs=`` slot. Stable identity — module-level function, picklable for
        multi-process workers. Returns `None` for rollouts with no observed non-null reward.
        """
        return _outcome_only_reward_func

    # ── internals ───────────────────────────────────────────────────

    def _effective_secrets(self) -> dict[str, str] | None:
        # [FUNCTION DOC]
        # HELPER: Per-session secrets to forward. Returns user secrets if provided, or None/secrets for URL targets; for catalog
        # targets without secrets, returns a dict with "api_key" set to the api_key option or OPENREWARD_API_KEY env var, or
        # None if neither exists.
        # Inputs: none. Outputs: dict or None. Exceptions: none. Side effects: reads environment variables.
        """Auto-forward ``OPENREWARD_API_KEY`` as the conventional ``api_key``
        per-session secret on platform mode (most envs need it)."""
        if self._secrets is not None or self._is_url:
            return self._secrets
        key = self._api_key or os.environ.get("OPENREWARD_API_KEY")
        return {"api_key": key} if key else None

    def _task_tools_probe_indices(self) -> list[int]:
        # [FUNCTION DOC]
        # HELPER: Task indices for tool discovery: the explicit discovery index if set, else the sorted distinct indices option,
        # else task 0.
        # Inputs: none. Outputs: list of ints. Exceptions: none. Side effects: none.
        """Pick task indices for ``session.list_tools()`` discovery (ORS ``/task_tools``)."""
        if self._task_tools_discovery_index is not None:
            return [self._task_tools_discovery_index]
        if self._indices is not None:
            return sorted(set(self._indices))
        return [0]

    def _discovered_tool_specs(self, env):
        # [FUNCTION DOC]
        # HELPER: Determines the tool specs to bind on rollout environments.
        # Behavior: fetches shared tools; returns them when discovery is disabled, when num_tasks fails (warning) or is zero.
        # Filters probe indices to the valid range (warning for dropped ones; shared tools if none remain). For each index,
        # opens a short session with effective secrets, lists its tools, and merges by name keeping the first spec (warning if a
        # later spec with the same name differs); failed probes are logged and skipped. Returns shared tools if every probe
        # failed.
        # Inputs: env (SDK Environment). Outputs: list of ToolSpec (or dict) objects.
        # Exceptions: errors from the initial shared list_tools call propagate.
        # Side effects: network calls, opens/closes sessions, warning logs.
        """Return ToolSpec list for binding on `_RolloutEnvironment`.

        Prefer ``session.list_tools()`` (ORS ``/task_tools``) so task-specific tools match what the OpenReward SDK
        exposes during rollouts. Fall back to ``environment.list_tools()`` on failure or when disabled.
        """
        shared = env.list_tools()
        shared = _tool_specs_from_session_list_tools(shared)
        if not self._discover_task_tools:
            return shared

        try:
            n = env.num_tasks(self._split)
        except Exception as e:  # noqa: BLE001
            logger.warning("OpenRewardSpec: num_tasks failed (%s); using shared tools only.", e)
            return shared

        if n <= 0:
            return shared

        probe_indices = self._task_tools_probe_indices()
        bad = [idx for idx in probe_indices if idx < 0 or idx >= n]
        if bad:
            logger.warning(
                "OpenRewardSpec: task tool probe index(es) %s out of range for split=%r (num_tasks=%s); ignoring them.",
                bad,
                self._split,
                n,
            )
        probe_indices = [idx for idx in probe_indices if 0 <= idx < n]
        if not probe_indices:
            logger.warning("OpenRewardSpec: no valid task indices for tool discovery; using shared tools only.")
            return shared

        secrets = self._effective_secrets()
        merged: list[Any] = []
        seen: dict[str, Any] = {}
        any_probe_ok = False

        for idx in probe_indices:
            try:
                with env.session(split=self._split, index=idx, secrets=secrets) as sess:
                    specs = _tool_specs_from_session_list_tools(sess.list_tools())
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "OpenRewardSpec: session tool discovery failed for task_index=%s (%s); skipping.",
                    idx,
                    e,
                )
                continue
            any_probe_ok = True
            for spec in specs:
                d = _spec_to_dict(spec)
                name = d["name"]
                if name not in seen:
                    seen[name] = spec
                    merged.append(spec)
                elif _spec_to_dict(seen[name]) != d:
                    logger.warning(
                        "OpenRewardSpec: tool %r has differing schemas across probe tasks; keeping first.",
                        name,
                    )

        if not any_probe_ok:
            logger.warning("OpenRewardSpec: all task tool probes failed; using shared tools only.")
            return shared
        return merged

    @cached_property
    def _sdk_client(self):
        # [FUNCTION DOC]
        # HELPER: Cached OpenReward client. Passes api_key option, else OPENREWARD_API_KEY env var for catalog targets; passes
        # base_url for URL targets.
        # Inputs: none. Outputs: openreward.OpenReward instance. Exceptions: ImportError if SDK missing.
        # Side effects: lazy SDK import; reads environment variables.
        """The shared ``openreward.OpenReward`` client used by `dataset` and `factory`."""
        openreward = _import_openreward()
        kwargs: dict[str, Any] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        elif not self._is_url and "OPENREWARD_API_KEY" in os.environ:
            kwargs["api_key"] = os.environ["OPENREWARD_API_KEY"]
        if self._is_url:
            kwargs["base_url"] = self._target
        return openreward.OpenReward(**kwargs)

    @cached_property
    def _sdk_env(self):
        # [FUNCTION DOC]
        # HELPER: Cached SDK Environment handle, looked up by env_name (or "env") for URL targets and by the catalog name
        # otherwise.
        # Inputs: none. Outputs: SDK Environment. Exceptions: SDK/network errors. Side effects: network call.
        """The shared SDK ``Environment`` handle."""
        if self._is_url:
            # Self-hosted single-env URL — pass any name; the SDK redirects.
            # Fallback to "env" matches `_RolloutEnvironment.__init__`.
            target = self._env_name or "env"
        else:
            target = self._target
        return self._sdk_client.environments.get(target)
