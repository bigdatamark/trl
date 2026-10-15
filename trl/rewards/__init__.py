# [PACKAGE DOC]
# Package trl.rewards: ready-made reward functions for online RL trainers (e.g. GRPOTrainer, RLOOTrainer).
# Exports: accuracy_reward, reasoning_accuracy_reward, get_cosine_scaled_reward (from accuracy_rewards);
# think_format_reward (from format_rewards); get_repetition_penalty_reward, get_soft_overlong_punishment
# (from other_rewards).
# Lazy imports: at runtime the module object in sys.modules is replaced by a _LazyModule built from
# _import_structure, so submodules (and their optional deps such as math_verify) load only on first attribute
# access. Under TYPE_CHECKING the names are imported eagerly for static analysis.
# Side effects on import: replaces its own sys.modules entry; nothing else.

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

import sys
from typing import TYPE_CHECKING

from .._lazy_module import _LazyModule


_import_structure = {
    "accuracy_rewards": ["accuracy_reward", "get_cosine_scaled_reward", "reasoning_accuracy_reward"],
    "format_rewards": ["think_format_reward"],
    "other_rewards": ["get_repetition_penalty_reward", "get_soft_overlong_punishment"],
}


if TYPE_CHECKING:
    from .accuracy_rewards import accuracy_reward, get_cosine_scaled_reward, reasoning_accuracy_reward
    from .format_rewards import think_format_reward
    from .other_rewards import get_repetition_penalty_reward, get_soft_overlong_punishment


else:
    sys.modules[__name__] = _LazyModule(__name__, __file__, _import_structure, module_spec=__spec__)
