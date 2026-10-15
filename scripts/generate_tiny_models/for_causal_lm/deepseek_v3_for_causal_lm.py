# [MODULE DOC]
# Script that builds a tiny randomly initialized DeepseekV3ForCausalLM mirroring deepseek-ai/DeepSeek-R1 and pushes it to trl-internal-testing/tiny-DeepseekV3ForCausalLM.
# Behavior (all at import time, run as a module): checks the transformers floor from pyproject.toml is installed exactly, seeds RNGs, loads the reference AutoTokenizer and GenerationConfig,
# builds a scaled-down config (mostly 2 layers, hidden size 8-16, intermediate size 32), instantiates the model in bfloat16, re-initializes weights via init_weights_tiny_model,
# runs smoke_test, check_dtype_pattern and print_config_diff against the reference, then calls push_to_hub.
# Details: Config uses 2 attention heads and 2 key/value heads (MLA requires them equal), a yarn rope_scaling block copied from the reference, sigmoid scoring with noaux_tc top-k routing and one next-token-prediction layer.
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
from transformers import AutoTokenizer, DeepseekV3Config, DeepseekV3ForCausalLM, GenerationConfig

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

MODEL_ID = "deepseek-ai/DeepSeek-R1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)
config = DeepseekV3Config(
    vocab_size=129280,
    hidden_size=8,
    # DeepSeek-V3 uses MLA, not GQA: kv_b_proj always reconstructs K/V with the full
    # num_attention_heads, so num_key_value_heads must equal num_attention_heads (i.e.
    # num_key_value_groups == 1). A mismatch (e.g. 4/2) is silently ignored by the KV
    # projection but makes SDPA repeat_kv the already-full K/V heads, crashing with a
    # head-count mismatch (see transformers#46960).
    num_attention_heads=2,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=32,
    max_position_embeddings=163840,
    rope_scaling={
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "factor": 40.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    },
    ep_size=1,
    moe_layer_freq=1,
    num_nextn_predict_layers=1,
    scoring_func="sigmoid",
    topk_method="noaux_tc",
)
model = DeepseekV3ForCausalLM(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)
smoke_test(model, tokenizer)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, tokenizer, generation_config, "tiny")
