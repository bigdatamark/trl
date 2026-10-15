# [MODULE DOC]
# Tests for trl.models.utils: prepare_deepspeed config sanitizing, disable_gradient_checkpointing, and
# unwrap_model_for_generation restoring gradient-checkpointing kwargs.
# Main contents: one module-level parametrized test and two test classes using the tiny Qwen2 causal LM.
# Import-time side effects: none beyond importing torch, accelerate, transformers and trl.

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

import types
from unittest.mock import patch

import pytest
import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM
from transformers.testing_utils import torch_device

from trl.import_utils import is_deepspeed_available
from trl.models.utils import disable_gradient_checkpointing, prepare_deepspeed, unwrap_model_for_generation


@pytest.mark.skipif(not is_deepspeed_available(), reason="deepspeed is not installed")
@pytest.mark.parametrize("stage", [1, 2, 3])
def test_prepare_deepspeed_strips_optimizer_for_cpu_offload(stage):
    # [FUNCTION DOC]
    # Tests that prepare_deepspeed removes training-only DeepSpeed config blocks for eval-only models (ZeRO stages 1-3).
    # Setup: skipped without deepspeed; builds a config with bf16, ZeRO stage with CPU optimizer offload, an AdamW
    # optimizer block and a WarmupLR scheduler block; wraps it in a namespace-based fake accelerator; patches
    # deepspeed.initialize with a fake that records the received config.
    # Assertions: the captured config has no optimizer, no scheduler, and no offload_optimizer under zero_optimization.
    # Inputs: stage (int, parametrized 1/2/3). Outputs: none. Exceptions: AssertionError on failure.
    # Side effects: temporarily patches deepspeed.initialize.
    # Note: uses SimpleNamespace stand-ins, which the repository test guidelines discourage (left as is).
    pytest.importorskip("deepspeed")
    # prepare_deepspeed initializes eval-only models without an optimizer. If the deep-copied config still carries the
    # training-only blocks, DeepSpeed builds a CPUAdam on GPU params (`optimizer` + `offload_optimizer: cpu`) or an LR
    # scheduler with no optimizer (`scheduler`), and the run dies before training. They must be dropped before
    # `deepspeed.initialize`. Intercept the call to check the config it would receive.
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": stage, "offload_optimizer": {"device": "cpu"}},
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-5}},
        "scheduler": {
            "type": "WarmupLR",
            "params": {"warmup_min_lr": 0, "warmup_max_lr": 1e-5, "warmup_num_steps": 10},
        },
    }
    accelerator = types.SimpleNamespace(
        state=types.SimpleNamespace(deepspeed_plugin=types.SimpleNamespace(deepspeed_config=ds_config))
    )

    captured = {}

    def fake_initialize(model, config):
        # [FUNCTION DOC]
        # HELPER: Stand-in for deepspeed.initialize that stores config in the captured dict.
        # Inputs: model, config. Outputs: a 4-tuple whose first item is a namespace with a no-op eval, the rest None.
        # Exceptions: none. Side effects: mutates the enclosing captured dict.
        captured["config"] = config
        return types.SimpleNamespace(eval=lambda: None), None, None, None

    with patch("deepspeed.initialize", fake_initialize):
        prepare_deepspeed(model=None, accelerator=accelerator)

    assert "optimizer" not in captured["config"]
    assert "scheduler" not in captured["config"]
    assert "offload_optimizer" not in captured["config"]["zero_optimization"]


class TestDisableGradientCheckpointing:
    # [CLASS DOC]
    # Tests the disable_gradient_checkpointing context manager on the tiny Qwen2 model: it must turn checkpointing
    # off inside the context and restore the original state on exit. No shared state or setup.
    def test_when_disabled(self):
        # [FUNCTION DOC]
        # Tests that a model with checkpointing off stays off before, inside and after the context.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: downloads/loads the tiny model.
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        assert model.is_gradient_checkpointing is False
        with disable_gradient_checkpointing(model):
            assert model.is_gradient_checkpointing is False
        assert model.is_gradient_checkpointing is False

    def test_when_enabled(self):
        # [FUNCTION DOC]
        # Tests that a model with checkpointing enabled reports it off inside the context and on again after exit.
        # Inputs: none. Outputs: none. Exceptions: AssertionError. Side effects: loads the tiny model.
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        model.gradient_checkpointing_enable()
        assert model.is_gradient_checkpointing is True
        with disable_gradient_checkpointing(model):
            assert model.is_gradient_checkpointing is False
        assert model.is_gradient_checkpointing is True


class TestUnwrapModelForGeneration:
    # [CLASS DOC]
    # Tests unwrap_model_for_generation restoring gradient checkpointing with the original kwargs after the context.
    @pytest.mark.parametrize("use_reentrant", [True, False])
    def test_restores_gradient_checkpointing_kwargs(self, use_reentrant):
        # [FUNCTION DOC]
        # Tests that after exiting unwrap_model_for_generation, checkpointing is re-enabled with the given use_reentrant.
        # Setup: tiny Qwen2 on torch_device in train mode, checkpointing enabled with use_reentrant; enter and exit the
        # context with a plain Accelerator and the same kwargs; compute LM loss on random 1x8 input ids.
        # Assertions: with reentrant checkpointing, autograd.grad over parameters raises RuntimeError mentioning .grad();
        # with non-reentrant it succeeds (distinguishes which mode was restored).
        # Inputs: use_reentrant (bool, parametrized). Outputs: none. Exceptions: AssertionError/RuntimeError on failure.
        # Side effects: loads model onto device.
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5").to(torch_device)
        model.train()  # checkpointing only runs in training mode
        model.gradient_checkpointing_enable({"use_reentrant": use_reentrant})
        with unwrap_model_for_generation(
            model, Accelerator(), gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
        ):
            pass

        input_ids = torch.randint(0, model.config.vocab_size, (1, 8), device=torch_device)
        loss = model(input_ids=input_ids, labels=input_ids).loss
        # Only non-reentrant checkpointing supports `torch.autograd.grad`, so this tells which one is enabled
        if use_reentrant:
            with pytest.raises(RuntimeError, match=r"\.grad\(\)"):
                torch.autograd.grad(loss, list(model.parameters()))
        else:
            torch.autograd.grad(loss, list(model.parameters()))
