# [MODULE DOC]
# CLI script run inside the Harbor sandbox (python3 /opt/run_cell.py --code-b64 ...) that forwards one base64-encoded
# code cell to kernel_server.py on 127.0.0.1:8765 and writes the server's JSON reply to stdout. Constant PORT 8765.
# KNOWN ISSUE: the docstring says the agent base64-decodes the output on the host; actually the output is plain
# JSON that JupyterEnv._run_cell parses with json.
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

"""CLI client for the kernel_server, invoked by the Harbor agent via env.exec.

Usage (inside the container):
  python3 /opt/run_cell.py --code-b64 <base64-encoded-python-source>

Writes the kernel server's response payload (already JSON-encoded) to stdout.
The agent base64-decodes it on the host side. Base64 sidesteps shell escaping
for code containing quotes, newlines, etc.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.request


PORT = 8765


def main() -> int:
    # [FUNCTION DOC]
    # Behavior: parses --code-b64 (required) and --timeout (default 120 s); decodes the code as UTF-8, POSTs it as JSON
    # to the kernel server, and writes the decoded response body to stdout returning 0. Any exception writes a JSON
    # object with an "[run_cell err]" output and ok false and returns 1.
    # Inputs: command line. Outputs: exit code int.
    # Exceptions: SystemExit on bad args; binascii/Unicode errors from decoding the argument propagate.
    # Side effects: HTTP request, stdout write.
    p = argparse.ArgumentParser()
    p.add_argument("--code-b64", required=True)
    p.add_argument("--timeout", type=int, default=120)
    args = p.parse_args()

    code = base64.b64decode(args.code_b64).decode("utf-8")
    payload = json.dumps({"code": code}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            sys.stdout.write(r.read().decode("utf-8", errors="replace"))
        return 0
    except Exception as exc:
        sys.stdout.write(json.dumps({"output": f"[run_cell err] {exc}", "ok": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
