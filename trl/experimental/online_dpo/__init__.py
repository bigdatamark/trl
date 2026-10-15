# [PACKAGE DOC]
# Experimental package for Online DPO (online Direct Preference Optimization: generate completions on the fly, rank them with a judge or reward model, apply DPO-style loss).
# Re-exports OnlineDPOConfig (training configuration dataclass) and OnlineDPOTrainer (the trainer) from its submodules; __all__ lists both.
# Imports are eager (no lazy loading); importing the package imports the trainer module and its heavy dependencies
# (torch, transformers). No other side effects on import.

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

from .online_dpo_config import OnlineDPOConfig
from .online_dpo_trainer import OnlineDPOTrainer


__all__ = ["OnlineDPOConfig", "OnlineDPOTrainer"]
