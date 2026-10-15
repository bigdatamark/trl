# [MODULE DOC]
# Script that builds a tiny randomly initialized Gemma3ForConditionalGeneration mirroring google/gemma-3-4b-it and pushes it to trl-internal-testing/tiny-Gemma3ForConditionalGeneration.
# Behavior (all at import time, run as a module): checks the transformers floor from pyproject.toml is installed exactly, seeds RNGs, loads the reference AutoProcessor and GenerationConfig,
# builds a scaled-down config (mostly 2 layers, hidden size 8-16, intermediate size 32), instantiates the model in bfloat16,
# runs smoke_test, check_dtype_pattern and print_config_diff against the reference, then calls push_to_hub.
# Details: Text: 2 layers (one sliding, one full attention), hidden 16. Vision: 2 layers, hidden 16, image_size 224 so 256 patches match mm_tokens_per_image (projector pooling becomes identity); the image processor size is also set to 224x224.
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

import torch
from transformers import AutoConfig, AutoProcessor, Gemma3ForConditionalGeneration, GenerationConfig

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

MODEL_ID = "google/gemma-3-4b-it"

processor = AutoProcessor.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)

# Gemma3 SigLIP defaults to image_size=896 → 4,096 patches. With production-scale intermediate_size
# (~4,304), the vision FFN saves [batch, 4,096, 4,304] activations for backprop (~134 MB/layer).
# Use image_size=224 → 256 patches (matching mm_tokens_per_image=256, so the projector's AvgPool2d
# gets kernel_size=1, i.e. identity) and scale down intermediate_size to keep activations tiny.
text_config = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    # One of each attention type. Deriving the pattern from num_hidden_layers=2 gives two
    # sliding layers (the reference is 5:1 over 34 layers), so the global RoPE path would never run.
    "layer_types": ["sliding_attention", "full_attention"],
    "intermediate_size": 32,
}
vision_config = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "embed_dim": 64,
    "intermediate_size": 32,
    "image_size": 224,
}
processor.image_processor.size = {"height": 224, "width": 224}

config = AutoConfig.from_pretrained(MODEL_ID, text_config=text_config, vision_config=vision_config)
model = Gemma3ForConditionalGeneration(config).to(dtype=torch.bfloat16)
smoke_test(model, processor)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, processor, generation_config, "tiny")
