# [PACKAGE DOC]
# Package of one-off scripts that build tiny randomly initialized models for TRL's unit tests and publish them
# under the trl-internal-testing Hub organization. Each script is run as a module (python -m ...) and does all its work
# at import time. Subpackages group scripts by head type (for_causal_lm, for_conditional_generation,
# for_block_diffusion, for_sequence_classification); shared helpers live in _common.
# Exports: nothing. Side effects on import: none (the file is empty apart from the license).

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

