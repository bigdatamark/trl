# [MODULE DOC]
# Jupyter-style Harbor harness: JupyterEnv, a HarborEnv subclass that gives the model a stateful Python kernel tool
# and a shell tool inside the Harbor sandbox. Kernel state lives in kernel_server.py (uploaded to /opt and run in the
# background on 127.0.0.1:8765); each cell is sent through run_cell.py with base64-encoded code. Submission is by
# writing /workdir/answer.txt. Module constants: _HERE (this directory) and _JUPYTER_PROMPT_SUFFIX (instructions
# appended to task prompts). Import-time side effects: imports trl.experimental.harbor.

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

"""Jupyter-style base agent — a stateful-kernel Harbor harness.

A custom [`~trl.experimental.harbor.HarborEnv`] subclass exposing a stateful Python kernel
(variables/imports persist across cells) plus a shell tool. Point a trainer at it with
``HarborSpec(dataset, agent="examples/grpo_harbor/harnesses/jupyter/env.py:JupyterEnv")``.

The kernel is a tiny HTTP server (`kernel_server.py`, uploaded to /opt/ and started in `_setup`); each
cell runs via `python3 /opt/run_cell.py` (see `run_cell.py`). Submission is by writing
``/workdir/answer.txt`` (same verifier as the bash harness).
"""

import asyncio
import base64
import json
import shlex
from pathlib import Path

from trl.experimental.harbor import HarborEnv


_HERE = Path(__file__).parent

_JUPYTER_PROMPT_SUFFIX = (
    "\n\nYou are a data-analysis agent with a **stateful Python kernel**: variables, imports, and "
    "side-effects persist across `add_and_execute_code_cell` calls. Dataset files are in "
    "/home/user/input/. Use `execute_shell_command` for shell (pip install, ls). **Submit your final "
    "answer by writing it to /workdir/answer.txt** (e.g. "
    "`add_and_execute_code_cell(code=\"open('/workdir/answer.txt','w').write(str(ans))\")`). Keep it "
    "short; do not end your turn without submitting."
)


class JupyterEnv(HarborEnv):
    # [CLASS DOC]
    # HarborEnv subclass (base class manages the sandbox in self._env, the event loop in self._loop and provides _exec for
    # shell commands and the reset/reward plumbing). Overrides PROMPT_SUFFIX with the Jupyter instructions and the _setup
    # hook to install the kernel; exposes two tools to the model: add_and_execute_code_cell (via _run_cell) and
    # execute_shell_command (via the base _exec).
    """Stateful-Jupyter-kernel harness over a Harbor sandbox."""

    PROMPT_SUFFIX = _JUPYTER_PROMPT_SUFFIX

    async def _setup(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Async sandbox setup hook.
        # Behavior: uploads kernel_server.py and run_cell.py into /opt, installs curl with apt-get if missing (120 s timeout),
        # starts the kernel server detached with nohup/setsid logging to /tmp/kernel.log, then polls up to 30 times (0.5 s
        # apart) with curl for an HTTP 200 on port 8765, returning on success.
        # Inputs: none. Outputs: none.
        # Exceptions: RuntimeError including the kernel log if it never binds; sandbox exec/upload errors.
        # Side effects: file uploads, package install and a background process in the sandbox.
        # Upload the kernel server + cell runner, ensure curl, start the kernel, wait for it to bind.
        await self._env.upload_file(_HERE / "kernel_server.py", "/opt/kernel_server.py")
        await self._env.upload_file(_HERE / "run_cell.py", "/opt/run_cell.py")
        await self._env.exec("which curl >/dev/null 2>&1 || apt-get install -y curl", timeout_sec=120)
        await self._env.exec(
            "nohup setsid python3 /opt/kernel_server.py >/tmp/kernel.log 2>&1 < /dev/null &", timeout_sec=30
        )
        for _ in range(30):
            r = await self._env.exec("curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8765/", timeout_sec=5)
            if (r.stdout or "").strip() == "200":
                return
            await asyncio.sleep(0.5)
        log = await self._env.exec("cat /tmp/kernel.log", timeout_sec=5)
        raise RuntimeError(f"kernel_server failed to bind 127.0.0.1:8765\n--- kernel.log ---\n{log.stdout}")

    def _run_cell(self, code: str) -> str:
        # [FUNCTION DOC]
        # HELPER: Runs code in the kernel synchronously on the env's event loop.
        # Behavior: base64-encodes the code, executes run_cell.py with it (shell-quoted, 180 s timeout); if stdout is empty
        # returns a bracketed message with return code and stderr; otherwise parses stdout as JSON and returns its "output"
        # field (empty string if absent), or a bracketed "unparseable" message with the first 500 chars on decode failure.
        # Inputs: code string. Outputs: string. Exceptions: sandbox exec errors. Side effects: executes code in the sandbox.
        b64 = base64.b64encode(code.encode()).decode()
        result = self._loop.run_until_complete(
            self._env.exec(f"python3 /opt/run_cell.py --code-b64 {shlex.quote(b64)}", timeout_sec=180)
        )
        raw = (result.stdout or "").strip()
        if not raw:
            return f"[run_cell empty stdout, rc={result.return_code}, stderr={result.stderr or ''}]"
        try:
            return str(json.loads(raw).get("output", ""))
        except json.JSONDecodeError:
            return f"[run_cell unparseable: {raw[:500]}]"

    def add_and_execute_code_cell(self, code: str) -> str:
        # [FUNCTION DOC]
        # Tool: executes Python code in the persistent kernel and returns its textual output (delegates to _run_cell).
        # Inputs: code. Outputs: string. Exceptions: as _run_cell. Side effects: kernel state mutation in the sandbox.
        """
        Execute Python code in the stateful kernel. Variables, imports, and side-effects persist across
        calls. Use this for all computation.

        Args:
            code: The Python code to execute.

        Returns:
            The textual output of the executed cell.
        """
        return self._run_cell(code)

    def execute_shell_command(self, command: str) -> str:
        # [FUNCTION DOC]
        # Tool: runs a shell command in the sandbox via the base class _exec and returns combined output; not shared with
        # kernel state. Inputs: command. Outputs: string. Exceptions: as base _exec. Side effects: sandbox command execution.
        """
        Run a shell command in the sandbox (pip install, ls, etc.). Not stateful with the Python kernel.

        Args:
            command: The shell command to run.

        Returns:
            The command's combined stdout and stderr.
        """
        return self._exec(command)
