# [MODULE DOC]
# Script that builds and publishes a tiny randomly initialized Qwen3VLForConditionalGeneration checkpoint for tests.
# Reference checkpoint: Qwen/Qwen3-VL-2B-Instruct.
# Behavior: requires transformers 4.57.0; loads processor and generation config; builds a 2-layer text model
# (hidden 16, 4 heads, 2 KV heads, interleaved mrope with sections 2,2,2, no layer_types key) and a 2-block
# vision tower (hidden 16, 4 heads, out 16, deepstack at block 1); instantiates in bfloat16, runs smoke test,
# dtype check, config diff and pushes as tiny.
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
# - "layer_types" is intentionally omitted from text_config: qwen3_vl's modeling code checks
#   `hasattr(config, "layer_types")` and uses a different path when absent
#   (see transformers/models/qwen3_vl/modeling_qwen3_vl.py).
# - mrope_section needs 3 elements (for dim, offset in enumerate((1, 2), start=1): mrope_section[dim]).

import torch
from transformers import AutoConfig, AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    print_config_diff,
    push_to_hub,
    set_seed,
    smoke_test,
)


TRANSFORMERS_VERSION = "4.57.0"
check_transformers_version(TRANSFORMERS_VERSION)

set_seed()

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

processor = AutoProcessor.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)

text_config = {
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "rope_scaling": {"mrope_interleaved": True, "mrope_section": [2, 2, 2], "rope_type": "default"},
}
vision_config = {
    # Real Qwen3-VL has depth=24 with deepstack at layers [5, 11, 17]. With depth=2 here, those
    # indexes are unreachable, so the deepstack mergers were instantiated but never invoked.
    # Pick an index inside [0, depth).
    "deepstack_visual_indexes": [1],
    "hidden_size": 16,
    "num_heads": 4,
    "depth": 2,
    "out_hidden_size": 16,
}

config = AutoConfig.from_pretrained(MODEL_ID, text_config=text_config, vision_config=vision_config)
model = Qwen3VLForConditionalGeneration(config).to(dtype=torch.bfloat16)
smoke_test(model, processor)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, processor, generation_config, "tiny")
