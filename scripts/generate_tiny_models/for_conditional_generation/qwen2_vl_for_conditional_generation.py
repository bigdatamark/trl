# [MODULE DOC]
# Script that builds and publishes a tiny randomly initialized Qwen2VLForConditionalGeneration checkpoint for tests.
# Reference checkpoint: Qwen/Qwen2-VL-2B-Instruct.
# Behavior: loads the reference processor and generation config; builds a config with a 2-layer, hidden size 16,
# 4-head (2 KV-head) text model with default mrope rope_scaling (sections 1,1,0) and layer_types cleared, a
# 2-block vision tower (hidden 16, embed_dim 64, 4 heads) and a separate root-level rope_scaling dict;
# instantiates the model in bfloat16, runs smoke test, dtype check and config diff, then pushes as tiny.
# Inputs: none programmatic; command-line flags are parsed inside the shared push_to_hub helper of the
# _common module (e.g. --create-pr). Requires network access to the Hugging Face Hub and write credentials.
# Outputs: none returned; prints a config diff vs the reference checkpoint and status messages.
# Exceptions: propagates errors from the version check, Hub downloads, the smoke test and the dtype check.
# Side effects (all at import/run time, since the whole script is top-level code): checks the installed
# transformers version, seeds RNGs, downloads the reference config/processor from the Hub, and uploads
# the tiny model to the trl-internal-testing organization unless the repo already exists.
# Contents: no functions or classes; meant to be run as a module (python -m) from the generate_tiny_models package.

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

# Note: two distinct rope_scaling dict objects (root and text_config). See GH-4101 and transformers#41020.

import torch
from transformers import AutoConfig, AutoProcessor, GenerationConfig, Qwen2VLForConditionalGeneration

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

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

processor = AutoProcessor.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)

text_config = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "layer_types": None,
    "rope_scaling": {"type": "default", "mrope_section": [1, 1, 0], "rope_type": "default"},
}
vision_config = {
    "hidden_size": 16,
    "num_heads": 4,
    "embed_dim": 64,
    "depth": 2,
}

config = AutoConfig.from_pretrained(
    MODEL_ID,
    text_config=text_config,
    vision_config=vision_config,
    rope_scaling={"type": "default", "mrope_section": [1, 1, 0], "rope_type": "default"},
)
model = Qwen2VLForConditionalGeneration(config).to(dtype=torch.bfloat16)
smoke_test(model, processor)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, processor, generation_config, "tiny")
