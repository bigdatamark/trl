# [PACKAGE DOC]
# Package trl.experimental.server_distillation: experimental on-policy distillation where teacher
# log-probabilities come from a remote server instead of a locally loaded teacher model.
# Exports: ServerDistillationConfig (training config), ServerDistillationTrainer (the trainer) and
# build_teacher_request_inputs (helper that builds the token payload sent to the teacher server).
# Imports are eager (no lazy loading); importing pulls in the config and trainer modules and their dependencies.
# Side effects on import: none beyond those of the submodules.

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

from .server_distillation_config import ServerDistillationConfig
from .server_distillation_trainer import ServerDistillationTrainer, build_teacher_request_inputs


__all__ = ["ServerDistillationConfig", "ServerDistillationTrainer", "build_teacher_request_inputs"]
