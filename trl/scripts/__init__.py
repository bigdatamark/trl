# [PACKAGE DOC]
# Package trl.scripts: command-line entry scripts (sft, dpo, grpo, kto, reward, rloo, distillation, env,
# vllm_serve) plus shared script utilities.
# Exports (lazily, from utils): DatasetMixtureConfig, ScriptArguments, TrlParser, get_dataset, init_zero_verbose.
# Lazy imports: outside TYPE_CHECKING the package's sys.modules entry is replaced with a _LazyModule over
# _import_structure, so utils (and its heavy deps) load only on first attribute access.
# Side effects on import: replaces its own sys.modules entry.

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

from typing import TYPE_CHECKING

from .._lazy_module import _LazyModule


_import_structure = {
    "utils": ["DatasetMixtureConfig", "ScriptArguments", "TrlParser", "get_dataset", "init_zero_verbose"],
}

if TYPE_CHECKING:
    from .utils import DatasetMixtureConfig, ScriptArguments, TrlParser, get_dataset, init_zero_verbose
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
