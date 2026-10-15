# [MODULE DOC]
# Script that builds and publishes a tiny randomly initialized LlamaForSequenceClassification reward-model style checkpoint.
# Reference checkpoint: meta-llama/Llama-3.2-1B-Instruct.
# Behavior: loads tokenizer and generation config; builds a config with num_labels 1, hidden size 16, 4 attention
# heads, 2 KV heads, 2 hidden layers and intermediate size 32;
# instantiates in bfloat16, re-initializes weights with the shared init_weights_tiny_model helper, runs the
# smoke test, dtype check and config diff, then pushes with prefix tiny and suffix 3.2.
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

import torch
from transformers import AutoConfig, AutoTokenizer, GenerationConfig, LlamaForSequenceClassification

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

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)
config = AutoConfig.from_pretrained(
    MODEL_ID,
    num_labels=1,
    hidden_size=16,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=32,
)
model = LlamaForSequenceClassification(config).to(dtype=torch.bfloat16)
init_weights_tiny_model(model)
smoke_test(model, tokenizer)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, tokenizer, generation_config, "tiny", "3.2")
