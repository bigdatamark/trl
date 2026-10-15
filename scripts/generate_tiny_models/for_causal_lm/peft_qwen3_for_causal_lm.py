# [MODULE DOC]
# Script that wraps the already-published trl-internal-testing/tiny-Qwen3ForCausalLM in a LoRA adapter and pushes
# the adapter to trl-internal-testing/tiny-<PEFT wrapper class name> (e.g. tiny-PeftModelForCausalLM).
# Behavior (import time): checks the transformers floor, seeds RNGs with the default seed, loads the base model with
# its stored dtype, applies get_peft_model with a default LoraConfig but init_lora_weights False (so lora_B is
# non-zero and the adapter is not the identity), runs smoke_test without tokenizer, then push_to_hub with no
# tokenizer and no generation config.
# Inputs: --create-pr CLI flag. Outputs: none. Exceptions: version mismatch / smoke test RuntimeError; Hub errors.
# Side effects: Hub download of the base model and upload of adapter files.

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

from peft import LoraConfig, get_peft_model
from transformers import Qwen3ForCausalLM

from .._common import check_transformers_version, push_to_hub, set_seed, smoke_test


check_transformers_version()

set_seed()

BASE = "trl-internal-testing/tiny-Qwen3ForCausalLM"

model = Qwen3ForCausalLM.from_pretrained(BASE, dtype="auto")
# Otherwise lora_B is zeroed and the adapter is the identity.
model = get_peft_model(model, LoraConfig(init_lora_weights=False))
smoke_test(model, None)
push_to_hub(model, None, None, "tiny")
