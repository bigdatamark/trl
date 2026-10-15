# [MODULE DOC]
# HELPER: Defines RemoteConfig, a LlamaConfig under the custom model_type "remote", used by the remote-code tiny model.
# Import-time side effects: none.

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

from transformers import LlamaConfig


class RemoteConfig(LlamaConfig):
    # [CLASS DOC]
    # HELPER: LlamaConfig subclass whose only change is the class attribute model_type set to "remote", so the
    # saved config.json is not recognized by built-in transformers auto classes.
    # Construction inputs: same as LlamaConfig. Methods: inherited. Side effects: none.
    model_type = "remote"
