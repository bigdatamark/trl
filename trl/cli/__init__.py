# [PACKAGE DOC]
# The trl command-line interface package. Exports main (the entry point behind the "trl" console script).
# Submodules: main (parser + dispatch), commands (subcommand classes), accelerate_config and accelerate_launcher
# (helpers for launching training scripts through accelerate).
# Side effects on import: imports trl.cli.main and the command classes; heavy script modules are imported only when a
# command runs.

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

from .main import main


__all__ = ["main"]
