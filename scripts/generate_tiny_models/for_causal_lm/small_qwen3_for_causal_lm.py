# [MODULE DOC]
# Script that builds a tiny randomly initialized Qwen3ForCausalLM mirroring Qwen/Qwen3-4B and pushes it to trl-internal-testing/small-Qwen3ForCausalLM.
# Behavior (all at import time, run as a module): checks the transformers floor from pyproject.toml is installed exactly, seeds RNGs, loads the reference AutoTokenizer and GenerationConfig,
# builds a scaled-down config (mostly 2 layers, hidden size 8-16, intermediate size 32), instantiates the model in bfloat16,
# runs smoke_test, check_dtype_pattern and print_config_diff against the reference, then calls push_to_hub.
# Details: Uses hidden size 128 (head dim 32, as vLLM requires) with tied embeddings; does not call init_weights_tiny_model.
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

# Slightly bigger than the "tiny" variant: vLLM requires hidden_size // num_attention_heads = 32.

import torch
from transformers import AutoTokenizer, GenerationConfig, Qwen3Config, Qwen3ForCausalLM

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

MODEL_ID = "Qwen/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)
config = Qwen3Config(
    vocab_size=151936,
    hidden_size=128,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=32,
    tie_word_embeddings=True,
    max_position_embeddings=40960,
    rope_theta=1000000,
    max_window_layers=36,
    bos_token_id=151643,
    eos_token_id=151645,
)
model = Qwen3ForCausalLM(config).to(dtype=torch.bfloat16)
smoke_test(model, tokenizer)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, tokenizer, generation_config, "small")
