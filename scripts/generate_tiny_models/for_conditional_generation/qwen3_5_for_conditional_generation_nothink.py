# [MODULE DOC]
# Script that builds and publishes a tiny randomly initialized Qwen3_5ForConditionalGeneration (thinking disabled by default) checkpoint for tests.
# Reference checkpoint: Qwen/Qwen3.5-0.8B, whose chat template defaults to non-thinking.
# Behavior: requires transformers 5.2.0; loads only the processor (no generation config exists upstream); builds
# a 2-layer text model (hidden 16, 4 heads, 2 KV heads) with layer types linear then full attention and
# full_attention_interval 2, and a 2-block vision tower (hidden 16, 4 heads, intermediate 32, out 16);
# instantiates in bfloat16, then casts A_log and the norm weight of every linear-attention layer back to
# float32 to match upstream; runs smoke test, dtype check, config diff and pushes with suffix NoThink and no
# generation config.
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

# Notes:
# - Qwen3.5 auto-builds layer_types from num_hidden_layers with default interval 4, so tiny models
#   (2 layers) end up all-linear-attention, which breaks dynamic cache. Force one full-attention layer.
# - The vision config expects `depth`/`num_heads` (not `num_hidden_layers`/`num_attention_heads`).
# - Qwen3.5 has no published generation_config on the Hub yet.
# - Qwen3.5 keeps some linear-attn weights in float32; we cast them back after the bfloat16 conversion.
# - Sourced from Qwen/Qwen3.5-0.8B because its bundled tokenizer defaults to thinking disabled.
#   Switching to a >=4B Qwen3.5 release will flip that default.

import torch
from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    print_config_diff,
    push_to_hub,
    set_seed,
    smoke_test,
)


TRANSFORMERS_VERSION = "5.2.0"
check_transformers_version(TRANSFORMERS_VERSION)

set_seed()

MODEL_ID = "Qwen/Qwen3.5-0.8B"

processor = AutoProcessor.from_pretrained(MODEL_ID)

text_config = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "layer_types": ["linear_attention", "full_attention"],
    "full_attention_interval": 2,
}
vision_config = {
    "hidden_size": 16,
    "depth": 2,
    "num_heads": 4,
    "intermediate_size": 32,
    "out_hidden_size": 16,
}

config = AutoConfig.from_pretrained(MODEL_ID, text_config=text_config, vision_config=vision_config)
model = Qwen3_5ForConditionalGeneration(config).to(dtype=torch.bfloat16)

# Restore float32 for linear-attn weights that the upstream model keeps in fp32.
for i, layer_type in enumerate(config.text_config.layer_types):
    if layer_type == "linear_attention":
        linear_attn = model.model.language_model.layers[i].linear_attn
        linear_attn.A_log.data = linear_attn.A_log.data.float()
        linear_attn.norm.weight.data = linear_attn.norm.weight.data.float()

smoke_test(model, processor)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, processor, None, "tiny", "NoThink")
