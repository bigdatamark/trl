# [PACKAGE DOC]
# HELPER: Holds the custom configuration and modeling source files for the tiny-RemoteForCausalLM Hub repo.
# Imported as a subpackage by remote_for_causal_lm.py; the files are uploaded alongside the weights (via
# register_for_auto_class) so the model can only be loaded with trust_remote_code.
# Exports nothing itself; importing it has no side effects.

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

# Source files for the tiny-RemoteForCausalLM Hub repo. Loaded as a sub-package
# by ../remote_for_causal_lm.py and pushed verbatim to the Hub so consumers can
# load the model with `trust_remote_code=True`.
