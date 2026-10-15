# [PACKAGE DOC]
# Deprecated compatibility shim package for KTO under trl.experimental.
# Exports KTOConfig and KTOTrainer, thin subclasses of the stable trl.trainer KTOConfig/KTOTrainer
# that emit a FutureWarning (removal planned for v2.0.0) when instantiated, pointing users to "from trl import ...".
# No lazy imports; importing the package imports the stable KTO trainer/config eagerly. No other side effects on import.

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

from ...trainer import KTOConfig as _KTOConfig
from ...trainer import KTOTrainer as _KTOTrainer


__all__ = ["KTOConfig", "KTOTrainer"]


@dataclass
class KTOConfig(_KTOConfig):
    # [CLASS DOC]
    # Deprecated alias dataclass of the stable KTOConfig; adds no fields.
    # Behavior: identical to the parent except a FutureWarning is raised in __post_init__.
    # Inputs: same constructor arguments as the stable KTOConfig.
    # Side effects: warning on construction.
    def __post_init__(self):
        # [FUNCTION DOC]
        # Emits a FutureWarning (stacklevel 3, so it points at the user's construction call) about the deprecated import path,
        # then delegates to the parent __post_init__ for normal validation/defaulting.
        # Inputs: none beyond self. Outputs: none.
        # Exceptions: whatever the parent __post_init__ raises.
        # Side effects: warning emitted; self mutated by the parent post-init.
        warnings.warn(
            "This import path is deprecated and will be removed in v2.0.0. "
            "The `KTOConfig` has been promoted to the stable API. "
            "Update your imports to `from trl import KTOConfig`.",
            FutureWarning,
            stacklevel=3,
        )
        super().__post_init__()


@dataclass
class KTOTrainer(_KTOTrainer):
    # [CLASS DOC]
    # Deprecated alias of the stable KTOTrainer; adds no behavior except a FutureWarning on construction.
    # Inputs: same arguments as the stable KTOTrainer.
    # KNOWN ISSUE: the class is decorated with dataclass, which is unnecessary for a trainer; since __init__ is defined explicitly
    # it is not replaced, but dataclass may still generate repr/eq methods from any inherited class annotations.
    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Emits a FutureWarning (stacklevel 2) about the deprecated import path and forwards all positional and keyword
        # arguments unchanged to the stable KTOTrainer constructor.
        # Inputs: any args/kwargs accepted by the stable KTOTrainer. Outputs: none.
        # Exceptions: whatever the parent constructor raises.
        # Side effects: warning emitted; full trainer initialization by the parent.
        warnings.warn(
            "This import path is deprecated and will be removed in v2.0.0. "
            "The `KTOTrainer` has been promoted to the stable API. "
            "Update your imports to `from trl import KTOTrainer`.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
