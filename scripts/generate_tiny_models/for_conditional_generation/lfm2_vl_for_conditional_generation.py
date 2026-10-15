# [MODULE DOC]
# Script that builds a tiny randomly initialized Lfm2VlForConditionalGeneration mirroring LiquidAI/LFM2.5-VL-3B and pushes it to trl-internal-testing/tiny-Lfm2VlForConditionalGeneration-2.5.
# Behavior (all at import time, run as a module): checks pinned transformers 5.0.0 is installed exactly, seeds RNGs, loads the reference AutoProcessor and GenerationConfig,
# builds a scaled-down config (mostly 2 layers, hidden size 8-16, intermediate size 32), instantiates the model in bfloat16, re-initializes weights via init_weights_tiny_model,
# runs smoke_test, check_dtype_pattern and print_config_diff against the reference, then calls push_to_hub.
# Details: Text: 2 layers (conv then full_attention), hidden 8. Vision: 2 layers, hidden 16. Projector hidden size 32. Weights are re-initialized with init_weights_tiny_model.
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
from transformers import AutoConfig, AutoProcessor, GenerationConfig, Lfm2VlForConditionalGeneration

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    init_weights_tiny_model,
    print_config_diff,
    push_to_hub,
    set_seed,
    smoke_test,
)


# LFM2.5-VL ships a `TokenizersBackend` tokenizer, which was introduced in transformers 5.0.0. This is above TRL's
# transformers floor, so unlike the other tiny models this one can't be loaded by the floor CI job, and the tests
# using it are skipped there.
check_transformers_version("5.0.0")

set_seed()

MODEL_ID = "LiquidAI/LFM2.5-VL-3B"

processor = AutoProcessor.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)

text_config = {
    "num_hidden_layers": 2,
    "hidden_size": 8,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 32,
    # LFM2 interleaves short convolution layers with full attention layers. The reference spells the pattern out with
    # `layer_types` rather than `full_attn_idxs`; keep one of each so both layer types are exercised.
    "layer_types": ["conv", "full_attention"],
}
vision_config = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "intermediate_size": 32,
}

config = AutoConfig.from_pretrained(
    MODEL_ID, text_config=text_config, vision_config=vision_config, projector_hidden_size=32
)
model = Lfm2VlForConditionalGeneration(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)
smoke_test(model, processor)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, processor, generation_config, "tiny", "2.5")
