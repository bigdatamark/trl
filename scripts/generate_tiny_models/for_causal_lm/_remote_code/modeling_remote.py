# [MODULE DOC]
# HELPER: Model classes for the remote-code tiny model: thin Llama subclasses bound to RemoteConfig (base model,
# causal LM head and sequence-classification head). Uploaded verbatim to the Hub with the weights.
# Import-time side effects: none.

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

import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaForSequenceClassification, LlamaModel, LlamaPreTrainedModel

from .configuration_remote import RemoteConfig


class RemoteModel(LlamaModel):
    # [CLASS DOC]
    # HELPER: LlamaModel subclass with config_class set to RemoteConfig; no other changes.
    config_class = RemoteConfig


class RemoteForCausalLM(LlamaForCausalLM):
    # [CLASS DOC]
    # HELPER: LlamaForCausalLM subclass with config_class set to RemoteConfig; no other changes.
    config_class = RemoteConfig


class RemoteForSequenceClassification(LlamaForSequenceClassification):
    # [CLASS DOC]
    # HELPER: LlamaForSequenceClassification subclass with config_class set to RemoteConfig. Overrides __init__ so the
    # backbone is a RemoteModel built directly instead of through AutoModel.from_config (which would re-trigger the
    # trust_remote_code prompt).
    config_class = RemoteConfig

    def __init__(self, config):
        # [FUNCTION DOC]
        # HELPER: Builds the classifier without calling the Llama parent initializers.
        # Behavior: calls LlamaPreTrainedModel initializer directly (skipping LlamaForSequenceClassification's), stores
        # num_labels from config, creates self.model as RemoteModel(config) and self.score as a bias-free linear layer from
        # hidden_size to num_labels, then calls post_init to initialize weights.
        # Inputs: config (RemoteConfig). Outputs: none. Exceptions: none raised here.
        # Side effects: registers submodules and initializes their weights.
        # The parent's MRO calls `AutoModel.from_config(config)`, which would re-trigger the
        # trust_remote_code prompt for `RemoteConfig`. Wire `RemoteModel` in directly instead.
        LlamaPreTrainedModel.__init__(self, config)
        self.num_labels = config.num_labels
        self.model = RemoteModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.post_init()
