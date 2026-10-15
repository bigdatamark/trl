# [MODULE DOC]
# Script that builds a tiny randomly initialized LlavaNextForConditionalGeneration mirroring llava-hf/llava-v1.6-mistral-7b-hf and pushes it to trl-internal-testing/tiny-LlavaNextForConditionalGeneration.
# Behavior (all at import time, run as a module): checks the transformers floor from pyproject.toml is installed exactly, seeds RNGs, loads the reference AutoProcessor and GenerationConfig,
# builds a scaled-down config (mostly 2 layers, hidden size 8-16, intermediate size 32), instantiates the model in bfloat16,
# runs smoke_test, check_dtype_pattern and print_config_diff against the reference, then calls push_to_hub.
# Details: Saved in bfloat16 (float16 triggers a CUDA assert during generation). Text sub-config dtype is reset to None to undo an upstream config mistake.
# Inputs: CLI flags --create-pr / --full-diff (parsed by _common). Outputs: none returned.
# Exceptions: RuntimeError on transformers version mismatch or failed smoke test; Hub/network errors propagate.
# Side effects: downloads reference tokenizer/config from the Hub, prints diagnostics, creates/updates a Hub repo (or skips if it exists).
#

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

# Original model dtype is float16, but it triggers CUDA device-side assert on generation (see GH-4741),
# so this tiny model is saved in bfloat16.
# Upstream hotfix: llava-hf/llava-v1.6-mistral-7b-hf mistakenly sets text_config.dtype to "bfloat16"
# (see https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/discussions/46), which we clear here.

import torch
from transformers import AutoConfig, AutoProcessor, GenerationConfig, LlavaNextForConditionalGeneration

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    print_config_diff,
    push_to_hub,
    set_seed,
    smoke_test,
)


check_transformers_version()

set_seed()

MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

processor = AutoProcessor.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)

text_config = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "layer_types": None,
    "dtype": None,  # hotfix for upstream text_config.dtype = "bfloat16"
}
vision_config = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "num_attention_heads": 4,
}

config = AutoConfig.from_pretrained(MODEL_ID, text_config=text_config, vision_config=vision_config)
model = LlavaNextForConditionalGeneration(config).to(dtype=torch.bfloat16)
smoke_test(model, processor)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, processor, generation_config, "tiny")
