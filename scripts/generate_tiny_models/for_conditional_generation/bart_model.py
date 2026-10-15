# [MODULE DOC]
# Script that builds a tiny BartModel (encoder-decoder base model, no LM head) from facebook/bart-base and pushes it to
# trl-internal-testing/tiny-BartModel.
# Behavior (import time): checks the transformers floor, seeds RNGs, loads the reference tokenizer and full config,
# only reduces d_model to 24 (layer counts and FFN sizes stay at the reference values), instantiates in float32, runs
# check_dtype_pattern and print_config_diff (no smoke test, no weight re-init), then push_to_hub without a generation
# config.
# Inputs: --create-pr / --full-diff. Outputs: none. Exceptions: version mismatch RuntimeError; Hub errors.
# Side effects: Hub downloads, printing, Hub upload.

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
from transformers import AutoConfig, AutoTokenizer, BartModel

from .._common import check_dtype_pattern, check_transformers_version, print_config_diff, push_to_hub, set_seed


check_transformers_version()

set_seed()

MODEL_ID = "facebook/bart-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
config = AutoConfig.from_pretrained(MODEL_ID)
config.d_model = 24
model = BartModel(config).to(dtype=torch.float32)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, tokenizer, None, "tiny")
