# [MODULE DOC]
# Script that builds and publishes a tiny randomly initialized T5ForConditionalGeneration checkpoint for tests.
# Reference checkpoint: google/flan-t5-small.
# Behavior: loads tokenizer, generation config and the reference config unchanged except d_model set to 24;
# instantiates in float32, runs the dtype check and config diff (no smoke test), and pushes as tiny with the
# tokenizer.
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
from transformers import AutoConfig, AutoTokenizer, GenerationConfig, T5ForConditionalGeneration

from .._common import check_dtype_pattern, check_transformers_version, print_config_diff, push_to_hub, set_seed


check_transformers_version()

set_seed()

MODEL_ID = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)
config = AutoConfig.from_pretrained(MODEL_ID)
config.d_model = 24
model = T5ForConditionalGeneration(config).to(dtype=torch.float32)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, tokenizer, generation_config, "tiny")
