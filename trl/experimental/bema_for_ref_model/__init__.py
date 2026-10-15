# [PACKAGE DOC]
# Experimental BEMA-as-reference-model package: lets the BEMA (bias-corrected EMA) weights periodically
# overwrite the DPO reference model.
# Exports BEMACallback (extends the stable BEMA callback with reference-model updates) and DPOTrainer (stable DPO
# trainer whose callback handler passes ref_model to callbacks).
# Imports eagerly (no lazy loading); no side effects beyond importing the submodules.

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

from .callback import BEMACallback
from .dpo_trainer import DPOTrainer
