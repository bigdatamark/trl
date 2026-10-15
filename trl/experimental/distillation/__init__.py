# [PACKAGE DOC]
# Deprecated experimental import path for distillation; the trainer was promoted to the stable API.
# Exports DistillationConfig and DistillationTrainer, thin subclasses of trl.trainer's classes that emit a
# FutureWarning (removal in v2.0.0) when instantiated. Unlike other packages, the classes are defined directly here.
# Side effects on import: imports trl.trainer's distillation classes; no warning is emitted until instantiation.

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

import warnings
from dataclasses import dataclass

from ...trainer import DistillationConfig as _DistillationConfig
from ...trainer import DistillationTrainer as _DistillationTrainer


__all__ = ["DistillationConfig", "DistillationTrainer"]


@dataclass
class DistillationConfig(_DistillationConfig):
    # [CLASS DOC]
    # Deprecated alias dataclass of the stable DistillationConfig; adds no fields.
    # Construction: same as the parent. Inheritance: subclasses trl DistillationConfig.
    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: emits a FutureWarning (stacklevel 3) about the deprecated import path, then runs the parent post-init.
        # Inputs: none. Outputs: none.
        # Exceptions: whatever the parent validation raises.
        # Side effects: warning.
        warnings.warn(
            "This import path is deprecated and will be removed in v2.0.0. "
            "The `DistillationConfig` has been promoted to the stable API. "
            "Update your imports to `from trl import DistillationConfig`.",
            FutureWarning,
            stacklevel=3,
        )
        super().__post_init__()


@dataclass
class DistillationTrainer(_DistillationTrainer):
    # [CLASS DOC]
    # Deprecated alias of the stable DistillationTrainer; behavior identical apart from the warning.
    # KNOWN ISSUE: the class is decorated with dataclass although it is a trainer, not a config. With no annotated fields
    # the explicit __init__ is kept, but dataclass also adds a generated eq method and sets the hash method to None,
    # making instances unhashable, unlike the parent trainer.
    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Behavior: emits a FutureWarning (stacklevel 2), then forwards all arguments to the parent constructor.
        # Inputs: any positional/keyword args of the stable DistillationTrainer. Outputs: none.
        # Exceptions: whatever the parent constructor raises.
        # Side effects: warning; full trainer construction.
        warnings.warn(
            "This import path is deprecated and will be removed in v2.0.0. "
            "The `DistillationTrainer` has been promoted to the stable API. "
            "Update your imports to `from trl import DistillationTrainer`.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
