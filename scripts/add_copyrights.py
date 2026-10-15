# [MODULE DOC]
# Repository maintenance script (run in CI / pre-commit) that ensures every git-tracked Python file starts with the
# exact HuggingFace Apache-2.0 copyright header, prepending it where missing.
# Contents: COPYRIGHT_HEADER (year range ends at the current year, computed at import), get_tracked_python_files,
# check_and_add_copyright and main. Import-time side effects: none beyond computing the header string.

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

import os
import subprocess
import sys
from datetime import datetime


COPYRIGHT_HEADER = f"""# Copyright 2020-{datetime.now().year} The HuggingFace Team. All rights reserved.
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
"""


def get_tracked_python_files():
    # [FUNCTION DOC]
    # HELPER: Lists tracked Python files by running git ls-files in the current directory and keeping paths ending in .py.
    # Inputs: none. Outputs: list of relative path strings; empty list if git fails.
    # Exceptions: CalledProcessError is caught and printed. Side effects: spawns a git subprocess, may print an error.
    """Get a list of all tracked Python files using git."""
    try:
        # Get the list of all tracked files from Git
        result = subprocess.run(["git", "ls-files"], stdout=subprocess.PIPE, text=True, check=True)
        # Split the result by lines to get individual file paths
        files = result.stdout.splitlines()
        # Filter only Python files
        py_files = [f for f in files if f.endswith(".py")]
        return py_files
    except subprocess.CalledProcessError as e:
        print(f"Error fetching tracked files: {e}")
        return []


def check_and_add_copyright(file_path):
    # [FUNCTION DOC]
    # HELPER: Ensures one file begins with COPYRIGHT_HEADER.
    # Behavior: if the path is not a file, prints a skip message and returns None. Reads the file (utf-8); if its text
    # starts with the header exactly, returns True. Otherwise prints a modify message, rewrites the file as header plus a
    # blank line plus the original content and returns False.
    # Inputs: file_path (str). Outputs: True, False, or None (missing file).
    # Exceptions: I/O errors propagate. Side effects: may rewrite the file; printing.
    # KNOWN ISSUE: the missing-file path returns None (docstring implies a check result), which main treats as falsy, so a
    # tracked-but-deleted file makes the script report missing copyrights and exit 1.
    """Check if the file contains a copyright notice, and add it if missing."""
    if not os.path.isfile(file_path):
        print(f"[SKIP] {file_path} does not exist.")
        return

    with open(file_path, encoding="utf-8") as f:
        content = f.readlines()

    # Check if the exact copyright header exists
    if "".join(content).startswith(COPYRIGHT_HEADER):
        return True

    # If no copyright notice was found, prepend the header
    print(f"[MODIFY] Adding copyright to {file_path}.")
    with open(file_path, "w", encoding="utf-8") as f:
        # Write the copyright header followed by the original content
        f.write(COPYRIGHT_HEADER + "\n" + "".join(content))
    return False


def main():
    # [FUNCTION DOC]
    # Checks all tracked Python files and sets the exit status.
    # Behavior: gets tracked files; if none, prints a message and returns (exit 0). Runs check_and_add_copyright on every
    # file (all files are processed, not short-circuited); exits 1 with a failure message if any result is falsy,
    # else exits 0 with a success message.
    # Inputs: none. Outputs: none (always terminates via sys.exit unless no files). Exceptions: SystemExit.
    # Side effects: may rewrite files; printing.
    """Main function to check and add copyright for all tracked Python files."""
    py_files = get_tracked_python_files()
    if not py_files:
        print("No Python files are tracked in the repository.")
        return

    print(f"Checking {len(py_files)} Python files for copyright notice...")

    have_copyright = [check_and_add_copyright(file_path) for file_path in py_files]
    if not all(have_copyright):
        print("❌ Some files were missing the required copyright and have been updated.")
        sys.exit(1)
    else:
        print("✅ All files have the required copyright.")
        sys.exit(0)


if __name__ == "__main__":
    main()
