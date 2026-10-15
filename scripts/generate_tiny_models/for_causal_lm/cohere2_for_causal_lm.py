# [MODULE DOC]
# Script that builds a tiny randomly initialized Cohere2ForCausalLM mirroring CohereLabs/tiny-aya-earth and pushes it to trl-internal-testing/tiny-Cohere2ForCausalLM.
# Behavior (all at import time, run as a module): checks the transformers floor from pyproject.toml is installed exactly, seeds RNGs, loads the reference AutoTokenizer and GenerationConfig,
# builds a scaled-down config (mostly 2 layers, hidden size 8-16, intermediate size 32), instantiates the model in bfloat16, re-initializes weights via init_weights_tiny_model,
# runs smoke_test, check_dtype_pattern and print_config_diff against the reference, then calls push_to_hub.
# Details: Config forces one sliding and one full attention layer (so the full-attention NoPE path runs) and copies the reference's Cohere-specific flags (hybrid cache, parallel block, gated activation, embedding sharing, logit_scale 1.0, rope_theta 50000).
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
from transformers import AutoTokenizer, Cohere2Config, Cohere2ForCausalLM, GenerationConfig

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    init_weights_tiny_model,
    print_config_diff,
    push_to_hub,
    set_seed,
    smoke_test,
)


check_transformers_version()

set_seed()

MODEL_ID = "CohereLabs/tiny-aya-earth"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)
config = Cohere2Config(
    vocab_size=262144,
    hidden_size=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=32,
    # One of each attention type. Deriving the pattern from num_hidden_layers=2 gives two sliding
    # layers (the reference is 3:1), so the full-attention NoPE path would never run.
    layer_types=["sliding_attention", "full_attention"],
    logit_scale=1.0,
    rope_theta=50000,
    bos_token_id=2,
    eos_token_id=3,
    cache_implementation="hybrid",
    layer_switch=4,
    order_of_interleaved_layers="local_attn_first",
    position_embedding_type="rope_gptj",
    rotary_pct=1.0,
    use_embedding_sharing=True,
    use_gated_activation=True,
    use_parallel_block=True,
    use_parallel_embedding=False,
    use_qk_norm=False,
)
model = Cohere2ForCausalLM(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)
smoke_test(model, tokenizer)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, tokenizer, generation_config, "tiny")
