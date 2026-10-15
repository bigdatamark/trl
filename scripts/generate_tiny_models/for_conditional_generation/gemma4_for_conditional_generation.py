# [MODULE DOC]
# Script that builds a tiny randomly initialized Gemma4ForConditionalGeneration mirroring google/gemma-4-E2B-it and pushes it to trl-internal-testing/tiny-Gemma4ForConditionalGeneration.
# Behavior (all at import time, run as a module): checks pinned transformers 5.5.0 is installed exactly, seeds RNGs, loads the reference AutoProcessor and GenerationConfig,
# builds a scaled-down config (mostly 2 layers, hidden size 8-16, intermediate size 32), instantiates the model in bfloat16,
# runs smoke_test, check_dtype_pattern and print_config_diff against the reference, then calls push_to_hub.
# Details: Loads the reference config and mutates text/vision sub-configs in place (dict overrides break rope validation). Image budget is 70 soft tokens (630 patches as position embedding size), set on the processor and image processor too. Text uses one sliding and one full layer, no KV-shared layers, global head dim 8, per-layer input size 16; audio_config is dropped.
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

# Gemma4 rope validation fails when passing text_config as a dict through AutoConfig,
# so the config is loaded first and then mutated in place.

import torch
from transformers import AutoConfig, AutoProcessor, Gemma4ForConditionalGeneration, GenerationConfig

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    print_config_diff,
    push_to_hub,
    set_seed,
    smoke_test,
)


TRANSFORMERS_VERSION = "5.5.0"  # PR transformers#45192 was backported to 5.5.0 with commit 5135e5e
check_transformers_version(TRANSFORMERS_VERSION)

set_seed()

MODEL_ID = "google/gemma-4-E2B-it"

processor = AutoProcessor.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)

# Gemma4 image processor uses aspect-ratio-preserving resizing, not a fixed image size. max_soft_tokens controls
# the output token budget and must be one of (70, 140, 280, 560, 1120). The smallest value (70) gives
# max_patches = 70 × pooling_kernel_size² = 70 × 9 = 630, so position_embedding_size must be at least 630.
# intermediate_size mirrors Gemma3: without it the production value (text: 6144, vision: 3072) is inherited, causing
# training activations [batch, patches, intermediate_size] to dominate GPU memory and OOM in CI.
IMAGE_TOKENS = 70  # minimum supported max_soft_tokens
MAX_PATCHES = IMAGE_TOKENS * 3**2  # 630

text_config = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 32,
}
vision_config = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "embed_dim": 64,
    "intermediate_size": 32,
    "position_embedding_size": MAX_PATCHES,  # 630
    "default_output_length": IMAGE_TOKENS,  # 70
}

processor.image_seq_length = IMAGE_TOKENS  # top-level Gemma4Processor attribute (serialized to processor_config.json)
processor.image_processor.image_seq_length = IMAGE_TOKENS  # nested Gemma4ImageProcessor attribute
processor.image_processor.max_soft_tokens = IMAGE_TOKENS

config = AutoConfig.from_pretrained(MODEL_ID)
for k, v in text_config.items():
    setattr(config.text_config, k, v)
for k, v in vision_config.items():
    setattr(config.vision_config, k, v)
config.text_config.layer_types = ["sliding_attention", "full_attention"]
config.text_config.num_kv_shared_layers = 0
config.text_config.global_head_dim = 8
config.text_config.hidden_size_per_layer_input = 16
config.audio_config = None

model = Gemma4ForConditionalGeneration(config).to(dtype=torch.bfloat16)
smoke_test(model, processor)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, processor, generation_config, "tiny")
