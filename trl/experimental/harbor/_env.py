# [MODULE DOC]
# Harbor-backed environment classes for GRPOTrainer's environment_factory.
# Main contents: HarborEnv (base env owning a Harbor sandbox and verifier, driven synchronously through a private
# asyncio loop on a daemon thread), HarborBashEnv (single bash tool harness), the _NO_REWARD sentinel, the bash prompt
# suffix, and AGENTS (name-to-class registry used by HarborSpec).
# Import-time side effects: none; harbor is imported lazily inside the async lifecycle methods.

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

"""Harbor-backed environments for `GRPOTrainer(environment_factory=...)`.

A `HarborEnv` wraps a Harbor sandbox + verifier. TRL drives the rollout loop: it calls the env's tool methods during
generation and reads `env.reward` afterwards. The pluggable "base agent" is the harness — the set of tool methods the
env exposes + how it submits. `HarborBashEnv` is the single-`bash`-tool harness (submit by writing
`/workdir/answer.txt`); subclass `HarborEnv` to add your own.

Harbor's API is async and its sandbox client is bound to the event loop it was created on, so each env owns one loop
and drives start/exec/verify/stop through it synchronously (TRL's tool loop is sync). `harbor` is imported lazily, so
importing this module does not require it installed (install `trl[harbor]`, which needs Python >= 3.12).
"""

import asyncio
import tempfile
import threading
import uuid
from pathlib import Path


_NO_REWARD = object()  # sentinel: reward not computed yet (0.0 is a valid reward)


class HarborEnv:
    # [CLASS DOC]
    # Base TRL environment wrapping one Harbor task sandbox at a time.
    # State: _environment_type (sandbox backend string), _loop and _loop_thread (dedicated event loop running forever on a
    # daemon thread), _env (Harbor environment or None), _task (Harbor Task), _paths (TrialPaths in a temp dir), _reward
    # (cached verifier result or the _NO_REWARD sentinel), class attribute PROMPT_SUFFIX appended to instructions.
    # Flow per rollout: reset(task_dir) starts the sandbox and returns the instruction; subclass tool methods call _exec;
    # the reward property runs the verifier once lazily; __del__ tears down. Subclasses define tools and may override
    # _setup.
    """Base TRL environment backed by a Harbor sandbox + verifier.

    Subclasses define the tool methods (the harness). The lifecycle TRL drives per rollout: `reset(task_dir)` (start
    the task's sandbox, return its instruction) -> tool methods (exec into the sandbox) -> `reward` (run the verifier
    once, lazily, after the rollout).

    Args:
        environment_type (`str`, *optional*, defaults to `"docker"`):
            Harbor sandbox backend, passed through to Harbor (`"docker"`, `"e2b"`, `"daytona"`, ...).
    """

    #: Extra guidance appended to the task instruction by the harness subclass.
    PROMPT_SUFFIX: str = ""

    def __init__(self, environment_type: str = "docker"):
        # [FUNCTION DOC]
        # Behavior: stores the backend type, creates a new event loop and starts it on a daemon thread, and initialises
        # sandbox state to empty with the reward unset.
        # Inputs: environment_type (str, default "docker").
        # Outputs: none. Exceptions: none.
        # Side effects: starts a background thread.
        self._environment_type = environment_type
        # Harbor's async sandbox client is bound to the loop it was created on, so we run that loop on a
        # dedicated daemon thread and submit coroutines to it via `run_coroutine_threadsafe`. This works
        # whether the caller is on a plain thread (GRPOTrainer drives tools from the main thread) or
        # already inside a running event loop (AsyncGRPOTrainer's rollout worker calls tool methods from
        # its own loop, where `loop.run_until_complete` would raise "another loop is already running").
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()
        self._env = None  # harbor BaseEnvironment for the current task
        self._task = None
        self._paths = None
        self._reward = _NO_REWARD

    def _run(self, coro):
        # [FUNCTION DOC]
        # HELPER: Submits a coroutine to the env's loop thread and blocks until it completes, returning its result.
        # Inputs: coro (coroutine). Outputs: the coroutine result.
        # Exceptions: re-raises the coroutine's exception.
        # Side effects: those of the coroutine.
        """Run a coroutine on this env's loop (which lives on its own thread) and block for the result."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def reset(self, task_dir: str | None = None, **kwargs) -> str:
        # [FUNCTION DOC]
        # Behavior: requires task_dir; starts the task sandbox (stopping any previous one), clears the cached reward, and
        # returns the task instruction concatenated with PROMPT_SUFFIX.
        # Inputs: task_dir (str path from the dataset row), **kwargs (other row fields, ignored).
        # Outputs: str prompt text.
        # Exceptions: ValueError if task_dir is None; Harbor errors from sandbox startup.
        # Side effects: creates and starts a sandbox, temp trial dir; resets _reward.
        if task_dir is None:
            raise ValueError("HarborEnv.reset requires `task_dir` (provided by the dataset row).")
        instruction = self._run(self._start(task_dir))
        self._reward = _NO_REWARD
        return instruction + self.PROMPT_SUFFIX

    def _exec(self, command: str, timeout: int = 180) -> str:
        # [FUNCTION DOC]
        # HELPER: Runs a shell command in the current sandbox.
        # Behavior: executes with the given timeout, concatenates stdout and stderr (None treated as empty), truncates to
        # 8000 characters plus a truncation marker, and returns a placeholder containing the return code if output is empty.
        # Inputs: command (str), timeout (int seconds, default 180).
        # Outputs: str.
        # Exceptions: AttributeError if called before reset (no sandbox); Harbor exec errors.
        # Side effects: runs a command in the sandbox.
        """Run a shell command in the sandbox; return combined stdout+stderr (truncated to 8k)."""
        result = self._run(self._env.exec(command, timeout_sec=timeout))
        out = (result.stdout or "") + (result.stderr or "")
        if len(out) > 8000:
            out = out[:8000] + "\n... [truncated]"
        return out or f"(empty output, rc={result.return_code})"

    @property
    def reward(self) -> float:
        # [FUNCTION DOC]
        # Behavior: property. Returns 0.0 if no sandbox exists (never reset). Otherwise, on first access runs the verifier and
        # caches the float; later accesses return the cached value until the next reset.
        # Inputs: none. Outputs: float.
        # Exceptions: verifier/Harbor errors.
        # Side effects: runs the verifier in the sandbox once; sets _reward.
        # Submission = the agent wrote /workdir/answer.txt during the rollout; the verifier reads it.
        # Computed once, lazily, on first read (TRL reads this after the rollout via reward_funcs).
        # A fresh env that was never `reset` has no sandbox/task to verify — return 0.0 without invoking
        # the verifier, which would start the Harbor backend and import `harbor`. The trainer discovers
        # tool methods on the class, so it never reads this property; the guard is for user code that
        # touches a fresh env directly.
        if self._env is None:
            return 0.0
        if self._reward is _NO_REWARD:
            self._reward = self._run(self._verify())
        return self._reward

    # ── harbor lifecycle (async, run on this env's loop) ────────────────────

    async def _start(self, task_dir: str) -> str:
        # [FUNCTION DOC]
        # HELPER: Async sandbox startup for a task.
        # Behavior: lazily imports Harbor; stops the previous sandbox; loads the Task from task_dir; creates TrialPaths in a
        # new temp dir prefixed "harbor_trl_"; builds the environment via Harbor's EnvironmentFactory with the configured
        # backend type, the task's environment dir, short name, a random session id and the task's environment config; starts
        # it without forcing a rebuild; re-uploads Dockerfile COPY files; runs the task healthcheck; creates /workdir and
        # /home/user/input; runs the harness _setup hook.
        # Inputs: task_dir (str). Outputs: the task instruction (str).
        # Exceptions: ImportError if harbor is missing; Harbor startup errors.
        # Side effects: sets _task, _paths, _env; creates a temp directory (never deleted); starts a sandbox.
        from harbor.environments.factory import EnvironmentFactory
        from harbor.models.task.task import Task
        from harbor.models.trial.config import EnvironmentConfig as TrialEnvironmentConfig
        from harbor.models.trial.paths import TrialPaths

        await self._stop()  # tear down the previous task's sandbox
        self._task = Task(task_dir=Path(task_dir))
        self._paths = TrialPaths(trial_dir=Path(tempfile.mkdtemp(prefix="harbor_trl_")))
        self._env = EnvironmentFactory.create_environment_from_config(
            config=TrialEnvironmentConfig(type=self._environment_type),
            environment_dir=self._task.paths.environment_dir,
            environment_name=self._task.short_name,
            session_id=uuid.uuid4().hex,
            trial_paths=self._paths,
            task_env_config=self._task.config.environment,
        )
        await self._env.start(force_build=False)
        await self._upload_build_files()  # some sandbox builds (e.g. E2B from_dockerfile) drop COPY'd files
        await self._env.run_healthcheck()  # task pre-agent hook (e.g. pull data into /home/user/input)
        await self._env.exec("mkdir -p /workdir /home/user/input")
        await self._setup()  # harness-specific sandbox prep (e.g. start a Jupyter kernel)
        return self._task.instruction

    async def _upload_build_files(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Recreates files copied by the task Dockerfile in the sandbox (needed for backends that drop COPY files).
        # Behavior: if the environment Dockerfile exists, for each line starting with COPY (case-insensitive) and not using
        # --from: drop flag tokens, require at least a source and a destination; for each source that is a local file under
        # the environment dir, compute the remote path (the destination itself for a single source not ending in "/", else
        # destination directory plus source basename), upload to a random /tmp path, then as root create the parent directory
        # and move it into place. Non-file sources (dirs, globs) are skipped.
        # Inputs: none (uses _task and _env). Outputs: none.
        # Exceptions: Harbor upload/exec errors.
        # Side effects: file uploads and shell commands in the sandbox.
        """Replicate the task Dockerfile's `COPY` directives into the sandbox.

        E2B's remote `from_dockerfile` build honors `RUN` steps but silently drops files `COPY`'d from the build
        context, which breaks healthchecks that run those files (e.g. a data-pull script). We re-create them at
        runtime: `upload_file` writes as the sandbox `user`, so we stage each file in a user-writable tmp path and `mv`
        it into place as root (destinations like `/opt` are root-owned). Idempotent. Handles the common ``COPY <src>
        <dst>`` form; flags / globs / ``--from`` are skipped.
        """
        dockerfile = self._task.paths.environment_dir / "Dockerfile"
        if not dockerfile.exists():
            return
        for line in dockerfile.read_text().splitlines():
            s = line.strip()
            if not s.upper().startswith("COPY ") or "--from" in s:
                continue
            parts = [p for p in s[len("COPY ") :].split() if not p.startswith("--")]
            if len(parts) < 2:
                continue
            *srcs, dst = parts
            for src in srcs:
                local = self._task.paths.environment_dir / src
                if not local.is_file():
                    continue
                remote = dst if (len(srcs) == 1 and not dst.endswith("/")) else dst.rstrip("/") + "/" + Path(src).name
                parent = remote.rsplit("/", 1)[0] or "/"
                tmp = "/tmp/" + uuid.uuid4().hex
                await self._env.upload_file(local, tmp)
                await self._env.exec(f"mkdir -p {parent} && mv {tmp} {remote}", user="root")

    async def _setup(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Harness-specific hook run after the sandbox is ready on each reset. Default does nothing.
        # Inputs: none. Outputs: none. Exceptions: none. Side effects: none (overrides may upload files or start services).
        """Harness-specific sandbox preparation, run once per `reset` after the sandbox is up.

        Override to upload helper files (`await self._env.upload_file(...)`) or start servers in the sandbox. The
        default is a no-op (the bash harness needs nothing beyond the base setup).
        """

    async def _verify(self) -> float:
        # [FUNCTION DOC]
        # HELPER: Runs the Harbor verifier for the current task.
        # Behavior: lazily imports Harbor; empties (creating) the verifier directory for the sandbox OS with chmod; builds the
        # verifier from a default VerifierConfig with the task, trial paths, environment and the task's verifier env
        # overrides (None if empty); awaits verification; returns the "reward" entry, else the first reward value, else 0.0,
        # as float.
        # Inputs: none. Outputs: float.
        # Exceptions: Harbor verifier errors.
        # Side effects: runs tests in the sandbox.
        from harbor.models.trial.config import VerifierConfig
        from harbor.models.trial.paths import EnvironmentPaths
        from harbor.verifier.factory import VerifierFactory

        # Pre-create the verifier dir (test.sh redirects stdout there; the shell can't mkdir the parent).
        env_paths = EnvironmentPaths.for_os(self._env.os)
        await self._env.empty_dirs([env_paths.verifier_dir], chmod=True)
        # Carry the task's [verifier].env (e.g. expected-answer / judge-model settings) into the verifier,
        # mirroring Harbor's trial runner (`override_env=`). A default trial `VerifierConfig()` is otherwise
        # correct here — the task verifier has no trial-level import_path/kwargs to forward.
        verifier = VerifierFactory.create_verifier_from_config(
            VerifierConfig(),
            task=self._task,
            trial_paths=self._paths,
            environment=self._env,
            override_env=self._task.config.verifier.env or None,
        )
        result = await verifier.verify()
        rewards = result.rewards or {}
        return float(rewards.get("reward", next(iter(rewards.values()), 0.0)))

    async def _stop(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Stops and deletes the current sandbox if any, always resetting _env to None (even if stop fails).
        # Inputs: none. Outputs: none.
        # Exceptions: propagates stop errors after clearing _env.
        # Side effects: sandbox deletion.
        if self._env is not None:
            try:
                await self._env.stop(delete=True)
            finally:
                self._env = None

    def __del__(self):
        # [FUNCTION DOC]
        # Behavior: best-effort teardown on garbage collection: stop the sandbox through the loop, swallowing any exception,
        # and finally schedule the loop to stop.
        # Inputs: none. Outputs: none. Exceptions: none (all swallowed except from scheduling the loop stop).
        # Side effects: sandbox deletion; stops the background loop (thread is daemon and then exits).
        try:
            self._run(self._stop())
        except Exception:  # noqa: BLE001 — best-effort teardown
            pass
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)


_BASH_PROMPT_SUFFIX = (
    "\n\nYou have a single `bash` tool: run a shell command in the sandbox and get its stdout+stderr. "
    "The dataset files are in /home/user/input/. Python 3 + pandas + numpy + scikit-learn are "
    "preinstalled. **Submit your final answer by writing it to /workdir/answer.txt via the `bash` "
    'tool**, e.g. `echo -n "<value>" > /workdir/answer.txt`. Stating the answer in prose does NOT submit '
    "it; only writing the file counts. Keep the answer short, and do not end your turn without submitting."
)


class HarborBashEnv(HarborEnv):
    # [CLASS DOC]
    # Harness with a single bash tool; PROMPT_SUFFIX tells the model to submit by writing /workdir/answer.txt and lists
    # preinstalled tools. Registered as "bash" in AGENTS.
    """Single-`bash`-tool harness; submit by writing `/workdir/answer.txt`."""

    PROMPT_SUFFIX = _BASH_PROMPT_SUFFIX

    def bash(self, command: str) -> str:
        # [FUNCTION DOC]
        # Behavior: tool method exposed to the model; runs the command via _exec with the default 180 s timeout and returns
        # the combined output. Its docstring is the tool description shown to the model.
        # Inputs: command (str). Outputs: str.
        # Exceptions: those of _exec.
        # Side effects: command execution in the sandbox.
        """
        Run a shell command in the sandbox and return its combined stdout+stderr. The shell is non-stateful between
        calls. Use it to explore files (ls, head, cat), run Python (`python3 -c "..."`), and submit the answer (`echo
        -n "<value>" > /workdir/answer.txt`).

        Args:
            command: The shell command to run.

        Returns:
            The command's combined stdout and stderr.
        """
        return self._exec(command)


#: Built-in harnesses, selectable by name in `HarborSpec(agent=...)`. Pass a `HarborEnv` subclass (or an
#: import path / file path resolving to one) for a custom harness.
AGENTS: dict[str, type[HarborEnv]] = {"bash": HarborBashEnv}
