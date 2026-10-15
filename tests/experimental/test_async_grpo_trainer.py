# [MODULE DOC]
# Tests for the experimental AsyncGRPOTrainer and its async rollout worker: end-to-end training with stub
# rollout/weight-transfer components, batching/collation and tree-packed rows, metric reduction, the rollout
# loop's multi-turn sample building and drift reconciliation, checkpoint/resume of rollout state, LoRA helper
# functions, and PEFT adapter syncing against an in-process fake vLLM server.
# Contents: dummy_reward_func, stubs (_StubRolloutWorker, _StubWeightTransfer), FakeVLLMServer /
# FakeVLLMTransport with the fake_vllm fixture, many data-building helpers, and Test* classes.
# GPU-only tests use the small-* Qwen2.5 model since the trainer loads with Flash Attention (head size multiple
# of 8).
# Side effects at import: none beyond imports.

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

import asyncio
import itertools
import json
import math
import multiprocessing as mp
import os
import queue
import threading
import time
from collections import OrderedDict, defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
import requests
import torch
from accelerate import PartialState
from datasets import Dataset, load_dataset
from requests.adapters import BaseAdapter
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.testing_utils import torch_device

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo.async_grpo_trainer import (
    DataCollatorForRollout,
    FixedCountBatcher,
    RolloutWorkerProtocol,
    TokenBudgetBatcher,
    _iter_vllm_named_params,
    _reduce_metric,
    round_lora_rank,
    save_lora_adapter,
    select_adapter_sync,
    validate_lora_for_vllm_sync,
)
from trl.experimental.async_grpo.async_rollout_worker import (
    AsyncRolloutWorker,
    DriftKind,
    RolloutGroup,
    RolloutSample,
    TrainingSequence,
    TurnRecord,
    _AsyncRolloutLoop,
    _chain_to_sequences,
    _common_prefix_len,
    _SampleBuilder,
)
from trl.experimental.async_grpo.packing import TREE_ATTENTION, TreeRow
from trl.trainer.base_trainer import _BaseTrainer

from ..testing_utils import (
    TrlTestCase,
    is_ampere_or_newer,
    require_peft,
    require_response_parsing,
    require_torch_accelerator,
    require_vllm,
)


# The trainer loads the model with Flash Attention, which requires a `head_size` multiple of 8. Hence the `small-*`
# model (`head_size=32`) below, rather than the usual `tiny-*` one (`head_size=2`).


def dummy_reward_func(completions, **kwargs):
    # [FUNCTION DOC]
    # HELPER: Pseudo-random reward per completion: Python hash of the first message's content modulo 100, divided by
    # 100, giving a value in [0, 0.99].
    # Inputs: completions (list of message lists), kwargs ignored. Outputs: list of float.
    # Exceptions: none. Side effects: none. Note: string hashing is salted per process, so values differ between runs.
    return [float(hash(c[0]["content"]) % 100) / 100.0 for c in completions]


class _StubRolloutWorker:
    # [CLASS DOC]
    # HELPER: In-process stand-in for the async GRPO rollout worker implementing RolloutWorkerProtocol without vLLM.
    # State: rollout_buffer and metrics_queue (queues; metrics stays empty), samples per sync, model version (starts
    # 0), fork_k (copies per generation) and an infinite sample iterator. start and update_model_version each enqueue
    # samples_per_weight_sync samples; stop and check_health do nothing.
    """Minimal rollout worker stub for testing the trainer in isolation."""

    def __init__(
        self, tokenizer, dataset, num_generations: int = 8, samples_per_weight_sync: int = 10, fork_k: int = 1
    ):
        # [FUNCTION DOC]
        # HELPER: Store settings and build the sample iterator.
        # Inputs: tokenizer, dataset (conversational prompt/completion), num_generations (int, default 8),
        # samples_per_weight_sync (int, default 10), fork_k (int, default 1). Outputs: none. Side effects: none.
        self.rollout_buffer = queue.Queue()
        self.metrics_queue = queue.Queue()  # drained by the trainer in `log()`; this stub measures nothing
        self._samples_per_weight_sync = samples_per_weight_sync
        self._model_version = 0
        self._fork_k = fork_k
        self._sample_iter = self._make_sample_iter(tokenizer, dataset, num_generations)

    def _make_sample_iter(self, tokenizer, dataset, num_generations):
        # [FUNCTION DOC]
        # HELPER: Infinite generator of RolloutSample objects grouped per prompt.
        # Behavior: cycles the dataset with a running group id; builds num_generations completions by appending " <idx>"
        # to the reference completion text; tokenizes the prompt (with generation prompt) and each prompt plus completion;
        # scores completions with dummy_reward_func and standardizes rewards (subtract mean, divide by population std) to
        # get advantages. For each generation yields fork_k identical samples with input ids, completion mask (0 prompt,
        # 1 completion), old log-probs (0 for prompt, -0.5 for completion), advantage, current model version, group id,
        # and metrics reward and reward_std.
        # Inputs: tokenizer, dataset, num_generations. Outputs: generator.
        # Exceptions: tokenizer errors. Side effects: none. Advantages are NaN if all rewards in a group are equal.
        for group_id, row in enumerate(itertools.cycle(dataset)):
            completions = [
                [{"role": "assistant", "content": f"{row['completion'][0]['content']} {idx}"}]
                for idx in range(num_generations)
            ]
            prompt_completions = [row["prompt"] + completion for completion in completions]
            prompt_ids = tokenizer.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True, return_dict=False
            )
            prompt_completion_ids = tokenizer.apply_chat_template(
                prompt_completions, tokenize=True, add_generation_prompt=False, return_dict=False
            )
            rewards = np.array(dummy_reward_func(completions))
            advantages = (rewards - rewards.mean()) / rewards.std()
            for idx in range(num_generations):
                completion_ids = prompt_completion_ids[idx][len(prompt_ids) :]
                sample = RolloutSample(
                    prompt=row["prompt"],
                    completion=completions[idx],
                    input_ids=np.array(prompt_ids + completion_ids, dtype=np.int64),
                    completion_mask=np.array([0] * len(prompt_ids) + [1] * len(completion_ids), dtype=np.int64),
                    old_log_probs=np.array([0.0] * len(prompt_ids) + [-0.5] * len(completion_ids), dtype=np.float32),
                    advantage=float(advantages[idx]),
                    model_version=self._model_version,
                    group_id=group_id,
                    metrics={"reward": float(rewards[idx]), "reward_std": float(rewards.std())},
                )
                # fork_k rows per generation, all sharing this group_id: mimics message-mode forking a
                # conversation into several training rows. Only the shared group_id matters for the epoch
                # count, so identical duplicates are enough here (row shapes are covered by the reconciler tests).
                for _ in range(self._fork_k):
                    yield sample

    def _fill_queue(self):
        # [FUNCTION DOC]
        # HELPER: Enqueue the next samples_per_weight_sync samples into rollout_buffer. Side effects: mutates the queue.
        for _ in range(self._samples_per_weight_sync):
            self.rollout_buffer.put(next(self._sample_iter))

    def start(self):
        # [FUNCTION DOC]
        # HELPER: Fill the buffer once with an initial batch. Side effects: mutates the queue.
        self._fill_queue()

    def update_model_version(self, model_version):
        # [FUNCTION DOC]
        # HELPER: Record model_version and enqueue another batch tagged with it. Inputs: model_version (int).
        self._model_version = model_version
        self._fill_queue()

    def stop(self):
        # [FUNCTION DOC]
        # HELPER: No-op stop. Inputs/Outputs/Exceptions/Side effects: none.
        pass

    def check_health(self, stale_after_s):
        # [FUNCTION DOC]
        # HELPER: No-op health check. Inputs: stale_after_s (ignored). Outputs: none.
        pass


class _StubWeightTransfer:
    # [CLASS DOC]
    # HELPER: No-op weight transfer stand-in; every lifecycle method does nothing except send_weights, which drains
    # the parameter iterator so the trainer's generator runs to completion.
    """No-op weight transfer for testing the trainer without a real vLLM server."""

    def init_weight_transfer(self):
        # [FUNCTION DOC]
        # HELPER: No-op. Inputs/Outputs/Exceptions/Side effects: none.
        pass

    def pause(self):
        # [FUNCTION DOC]
        # HELPER: No-op. Inputs/Outputs/Exceptions/Side effects: none.
        pass

    def send_weights(self, iterator):
        # [FUNCTION DOC]
        # HELPER: Consume and discard every element of the given iterator. Inputs: iterator. Outputs: None.
        # Exceptions: whatever the iterator raises. Side effects: advances the iterator.
        for _ in iterator:  # drain the param stream like the real client does
            pass

    def resume(self):
        # [FUNCTION DOC]
        # HELPER: No-op. Inputs/Outputs/Exceptions/Side effects: none.
        pass

    def destroy(self):
        # [FUNCTION DOC]
        # HELPER: No-op. Inputs/Outputs/Exceptions/Side effects: none.
        pass


class FakeVLLMServer:
    # [CLASS DOC]
    # HELPER: In-memory model of the vLLM server endpoints used by VLLMClient, with LoRA adapter state.
    # Attributes: url (class constant "http://fake-vllm"), lora_config (None means LoRA disabled), data_parallel_size,
    # max_model_len, dtype string, loaded (ordered adapter name to path, oldest first), evicted (names dropped by LRU),
    # paused flag, on_load callback, requests (every path asked, in order). dispatch routes a request; _load
    # implements adapter loading. Paired with FakeVLLMTransport.
    """The state a vLLM server has and a mocked dict does not, behind the eight paths [`VLLMClient`] uses.

    Configure it after construction (`fake.lora_config = ...`) and before the trainer is built; the trainer probes
    `/server_info` in its constructor. Read `loaded`, `evicted` and `paused` afterwards to see what the trainer did
    from the server's side.

    Attributes:
        lora_config (`dict` or `None`):
            What `/server_info` reports under `lora_config`. `None` is a server started without `--enable-lora`, which
            also makes `/v1/load_lora_adapter` a 404, as it is on the real server.
        data_parallel_size (`int`):
            What `/server_info` reports under `parallel_config`.
        loaded (`OrderedDict[str, str]`):
            Adapter name to path, oldest first. This is the LRU order vLLM's `LRUCacheLoRAModelManager` evicts in.
        evicted (`list[str]`):
            Adapters silently dropped because a load arrived with `loaded` already at `max_loras`. A real server does
            exactly this rather than reject the load, which is why `select_adapter_sync` sizes `--max-loras` for it.
        paused (`bool`):
            Whether `/pause` has been called more recently than `/resume`.
        requests (`list[str]`):
            Every path this server was asked, in order.
        on_load (`callable`, *optional*):
            Called as `on_load(name, path)` the moment a load is accepted, before the response is returned — the one
            place a test can observe the trainer's state *at* load time (e.g. that `model_version` has not moved yet).
    """

    url = "http://fake-vllm"  # a scheme and a host, never bound to anything

    def __init__(self, *, lora_config=None, data_parallel_size=1, max_model_len=2048, dtype="torch.bfloat16"):
        # [FUNCTION DOC]
        # HELPER: Store the keyword-only settings (lora_config None, data_parallel_size 1, max_model_len 2048, dtype
        # "torch.bfloat16" by default) and initialize empty loaded/evicted/requests, paused False, on_load None.
        # Outputs: none. Exceptions: none. Side effects: none.
        self.lora_config = lora_config
        self.data_parallel_size = data_parallel_size
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.loaded: OrderedDict[str, str] = OrderedDict()
        self.evicted: list[str] = []
        self.paused = False
        self.on_load = None
        self.requests: list[str] = []  # every path asked of this server, in order

    def dispatch(self, method: str, path: str, query: dict, body) -> tuple[int, object]:
        # [FUNCTION DOC]
        # HELPER: Answer one HTTP request by path, returning a status code and a JSON-able payload.
        # Behavior: records the path; /health gives 200 empty; /server_info gives the vllm_config with lora_config,
        # parallel config (data parallel size, tensor parallel 1, world size 1) and model dtype; /v1/models lists "base"
        # with max_model_len plus every loaded adapter; /get_world_size returns data_parallel_size unless the
        # include_dp query is "false" (then 1); /pause and /resume toggle paused; /v1/load_lora_adapter delegates to
        # _load (load_inplace defaults False); /v1/unload_lora_adapter is 404 without LoRA, else removes the name (if
        # present) and returns "Success"; anything else is 404.
        # Inputs: method (str), path (str), query (dict of lists), body (parsed JSON or None).
        # Outputs: tuple of status code and payload. Exceptions: KeyError if a load/unload body lacks required keys.
        # Side effects: mutates requests, paused and loaded.
        """Answer one request. Returns `(status_code, json_payload)`."""
        self.requests.append(path)
        if path == "/health":
            return 200, {}
        if path == "/server_info":
            return 200, {
                "vllm_config": {
                    "lora_config": self.lora_config,
                    "parallel_config": {
                        "data_parallel_size": self.data_parallel_size,
                        "tensor_parallel_size": 1,
                        "world_size": 1,
                    },
                    "model_config": {"dtype": self.dtype},
                }
            }
        if path == "/v1/models":
            return 200, {
                "data": [{"id": "base", "max_model_len": self.max_model_len}, *({"id": name} for name in self.loaded)]
            }
        if path == "/get_world_size":
            include_dp = query.get("include_dp", ["true"])[0].lower() != "false"
            return 200, {"world_size": self.data_parallel_size if include_dp else 1}
        if path == "/pause":
            self.paused = True
            return 200, {}
        if path == "/resume":
            self.paused = False
            return 200, {}
        if path == "/v1/load_lora_adapter":
            return self._load(body["lora_name"], body["lora_path"], body.get("load_inplace", False))
        if path == "/v1/unload_lora_adapter":
            if self.lora_config is None:
                return 404, {"detail": "Not Found"}
            self.loaded.pop(body["lora_name"], None)
            return 200, "Success"
        return 404, {"detail": f"no route for {method} {path}"}

    def _load(self, name: str, path: str, load_inplace: bool) -> tuple[int, object]:
        # [FUNCTION DOC]
        # HELPER: Emulate vLLM's runtime LoRA load.
        # Behavior: 404 when LoRA is disabled; 400 InvalidUserInput if the name is already loaded and load_inplace is
        # false; 404 if the path has no adapter_config.json file; when adding a new name at max_loras capacity, evicts
        # the oldest adapter into evicted; then moves/inserts the name at the end of loaded, calls on_load if set and
        # returns 200 with a success string.
        # Inputs: name (str), path (str), load_inplace (bool). Outputs: tuple of status and payload.
        # Exceptions: whatever on_load raises. Side effects: reads the filesystem; mutates loaded and evicted.
        # Without `--enable-lora` (and `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`) the endpoint is not registered at all.
        if self.lora_config is None:
            return 404, {"detail": "Not Found"}
        # `_check_load_lora_adapter_request`: a repeated name is rejected unless the caller asked to load in place.
        if name in self.loaded and not load_inplace:
            return 400, {
                "error": {
                    "message": f"The lora adapter '{name}' has already been loaded. If you want to load the adapter "
                    "in place, set 'load_inplace' to True.",
                    "type": "InvalidUserInput",
                }
            }
        # `add_lora` validates eagerly: the engine reads the directory before the request is answered, so a path the
        # server cannot see is a 404 here, not a 200 followed by an engine death.
        if not os.path.isfile(os.path.join(path, "adapter_config.json")):
            return 404, {"error": {"message": f"Loading lora {name} failed: No adapter found for {path}"}}
        # `LRUCacheLoRAModelManager.add_adapter`: at capacity the oldest adapter is removed, silently.
        if name not in self.loaded and len(self.loaded) >= self.lora_config["max_loras"]:
            self.evicted.append(self.loaded.popitem(last=False)[0])
        self.loaded.pop(name, None)  # re-insert at the end: a load touches the LRU order
        self.loaded[name] = path
        if self.on_load is not None:
            self.on_load(name, path)
        return 200, f"Success: LoRA adapter '{name}' added successfully."


class FakeVLLMTransport(BaseAdapter):
    # [CLASS DOC]
    # HELPER: requests transport adapter (BaseAdapter subclass) that routes requests to a FakeVLLMServer instead of
    # the network. State: server.
    """A `requests` transport that delivers to a [`FakeVLLMServer`] instead of a socket."""

    def __init__(self, server: FakeVLLMServer):
        # [FUNCTION DOC]
        # HELPER: Initialize the base adapter and keep a reference to server. Outputs: none. Side effects: none.
        super().__init__()
        self.server = server

    def send(self, request, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Convert a prepared request into a server dispatch call and wrap the result as a requests Response.
        # Behavior: splits the URL, JSON-decodes the body if present, dispatches with the parsed query, and builds a
        # Response with the status, URL, request, JSON content-type header, JSON-encoded payload and utf-8 encoding.
        # Inputs: request (PreparedRequest), kwargs ignored. Outputs: requests Response.
        # Exceptions: JSON decode errors for non-JSON bodies. Side effects: server state changes via dispatch.
        url = urlsplit(request.url)
        body = json.loads(request.body) if request.body else None
        status, payload = self.server.dispatch(request.method, url.path, parse_qs(url.query), body)
        response = requests.Response()
        response.status_code = status
        response.url = request.url
        response.request = request
        response.headers["Content-Type"] = "application/json"
        response._content = json.dumps(payload).encode()
        response.encoding = "utf-8"
        return response

    def close(self):
        # [FUNCTION DOC]
        # HELPER: No-op close. Inputs/Outputs/Exceptions/Side effects: none.
        pass


@pytest.fixture
def fake_vllm(monkeypatch) -> FakeVLLMServer:
    # [FUNCTION DOC]
    # HELPER: Pytest fixture that routes all requests Session calls for URLs starting with FakeVLLMServer.url to a
    # fresh FakeVLLMServer, leaving other URLs on the real adapter.
    # Inputs: monkeypatch. Outputs: the FakeVLLMServer instance (configure before building the trainer).
    # Exceptions: none. Side effects: patches requests.Session.get_adapter for the test duration.
    """A [`FakeVLLMServer`] that every `requests` call to `FakeVLLMServer.url` reaches, for this test only.

    The one patch in this module. It swaps the transport for one URL, leaves every other request untouched, and is
    undone by `monkeypatch` when the test ends.
    """
    server = FakeVLLMServer()
    transport = FakeVLLMTransport(server)
    real_get_adapter = requests.Session.get_adapter

    def get_adapter(self, url):
        # [FUNCTION DOC]
        # HELPER: Return the fake transport for fake-server URLs, otherwise the original get_adapter result.
        # Inputs: self (Session), url (str). Outputs: transport adapter. Exceptions: those of the real lookup.
        return transport if url.startswith(FakeVLLMServer.url) else real_get_adapter(self, url)

    monkeypatch.setattr(requests.Session, "get_adapter", get_adapter)
    return server


@pytest.mark.skipif(
    not is_ampere_or_newer() and torch_device != "xpu" and torch.version.hip is None,
    reason="Flash Attention requires Ampere or newer GPU, XPU, or ROCm",
)
class TestAsyncGRPOTrainer(TrlTestCase):
    # [CLASS DOC]
    # End-to-end AsyncGRPOTrainer tests with stub rollout worker and weight transfer on the small Qwen2.5 model and
    # zen conversational_prompt_completion (GPU-gated: Ampere+, XPU or ROCm). Inherits TrlTestCase for tmp_dir.
    def test_init_minimal(self):
        # [FUNCTION DOC]
        # Constructs the trainer with only model id, reward function, dataset and the stubs; asserts nothing beyond not
        # raising.
        # Test that AsyncGRPOTrainer can be instantiated with only model, reward_model and train_dataset
        model_id = "trl-internal-testing/small-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")
        AsyncGRPOTrainer(
            model=model_id,
            reward_funcs=dummy_reward_func,
            train_dataset=dataset,
            rollout_worker=_StubRolloutWorker(AutoTokenizer.from_pretrained(model_id), dataset, num_generations=3),
            weight_transfer=_StubWeightTransfer(),
        )

    def test_train(self):
        # [FUNCTION DOC]
        # Trains (lr 0.1, batch 3, 3 generations, completion length 8, token budget 64, 5 s timeout) and asserts
        # train_loss is logged and every parameter changed.
        model_id = "trl-internal-testing/small-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")

        training_args = AsyncGRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            token_budget=64,  # set explicitly; the stub worker has no real vLLM server to query for max_model_len
            vllm_server_timeout=5.0,  # short timeout so test fails fast if queue runs dry
            report_to="none",
        )
        trainer = AsyncGRPOTrainer(
            model=model_id,
            reward_funcs=dummy_reward_func,  # unused: the stub pre-computes rewards, but the trainer requires this argument
            args=training_args,
            train_dataset=dataset,
            rollout_worker=_StubRolloutWorker(AutoTokenizer.from_pretrained(model_id), dataset, num_generations=3),
            weight_transfer=_StubWeightTransfer(),
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_resume_from_checkpoint(self):
        # [FUNCTION DOC]
        # Trains 2 steps saving each step (ignore_data_skip forced True), asserts checkpoint-1 has trainer_state.json,
        # then a second trainer with max_steps 3 resumes from it without error (also with ignore_data_skip True).
        # Checks that ignore_data_skip is True and that resume doesn't crash. The stub worker is not an
        # AsyncRolloutWorker, so the checkpoint-write and resume-read paths stay inert here — those are
        # covered by TestRolloutStateCheckpoint.
        model_id = "trl-internal-testing/small-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        training_args = AsyncGRPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,
            num_generations=3,
            max_steps=2,
            save_steps=1,
            max_completion_length=8,
            token_budget=64,
            vllm_server_timeout=5.0,
            report_to="none",
        )

        # First run: train for 2 steps, which saves a checkpoint at step 1.
        trainer = AsyncGRPOTrainer(
            model=model_id,
            reward_funcs=dummy_reward_func,
            args=training_args,
            train_dataset=dataset,
            rollout_worker=_StubRolloutWorker(tokenizer, dataset, num_generations=3),
            weight_transfer=_StubWeightTransfer(),
        )
        assert trainer.args.ignore_data_skip is True
        trainer.train()

        # Second run: resume from the step-1 checkpoint.
        checkpoint_dir = os.path.join(self.tmp_dir, "checkpoint-1")
        assert os.path.isfile(os.path.join(checkpoint_dir, "trainer_state.json"))

        training_args2 = AsyncGRPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,
            num_generations=3,
            max_steps=3,
            max_completion_length=8,
            token_budget=64,
            vllm_server_timeout=5.0,
            report_to="none",
        )
        trainer2 = AsyncGRPOTrainer(
            model=model_id,
            reward_funcs=dummy_reward_func,
            args=training_args2,
            train_dataset=dataset,
            rollout_worker=_StubRolloutWorker(tokenizer, dataset, num_generations=3),
            weight_transfer=_StubWeightTransfer(),
        )
        assert trainer2.args.ignore_data_skip is True
        trainer2.train(resume_from_checkpoint=checkpoint_dir)

    def test_dataset_required_without_environment(self):
        # [FUNCTION DOC]
        # Constructing without train_dataset or environment raises ValueError "`train_dataset` is required".
        # The data has to come from somewhere: an external `train_dataset`, or an environment that owns it. With
        # neither, construction fails fast.
        training_args = AsyncGRPOConfig(output_dir=self.tmp_dir, max_steps=5, report_to="none")
        with pytest.raises(ValueError, match="`train_dataset` is required"):
            AsyncGRPOTrainer(
                model="trl-internal-testing/small-Qwen2ForCausalLM-2.5",
                reward_funcs=dummy_reward_func,
                args=training_args,
                weight_transfer=_StubWeightTransfer(),
            )

    def test_environment_owned_data_requires_max_steps(self):
        # [FUNCTION DOC]
        # With an environment factory but no dataset and max_steps unset, construction raises ValueError mentioning
        # max_steps.
        # When the environment owns the data and no external `train_dataset` is provided, `max_steps` must set the
        # training length. The guard fires before the rollout worker is built, so no worker/dataset is needed.
        class DummyEnvironment:
            # [CLASS DOC]
            # HELPER: Minimal environment class whose reset returns a fixed instruction string.
            def reset(self, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Return "Guess the 5-letter word." regardless of kwargs. Side effects: none.
                return "Guess the 5-letter word."

        args = AsyncGRPOConfig(output_dir=self.tmp_dir, report_to="none")  # max_steps unset
        with pytest.raises(ValueError, match="max_steps"):
            AsyncGRPOTrainer(
                model="trl-internal-testing/small-Qwen2ForCausalLM-2.5",
                reward_funcs=dummy_reward_func,
                args=args,
                environment_factory=DummyEnvironment,
                weight_transfer=_StubWeightTransfer(),
            )

    def test_multiple_environments_without_dataset_raises(self):
        # [FUNCTION DOC]
        # A dict of two environment factories without a train_dataset raises ValueError "requires a `train_dataset`".
        # Multiple environments (a `dict` factory) need a `train_dataset` with an `environment` column to route each
        # example. Without one, there is no per-rollout selector, so construction fails fast with a clear message.
        class EnvA:
            # [CLASS DOC]
            # HELPER: Placeholder environment class with a no-op reset.
            def reset(self, **kwargs):
                # [FUNCTION DOC]
                # HELPER: No-op reset returning None.
                ...

        class EnvB:
            # [CLASS DOC]
            # HELPER: Second placeholder environment class with a no-op reset.
            def reset(self, **kwargs):
                # [FUNCTION DOC]
                # HELPER: No-op reset returning None.
                ...

        args = AsyncGRPOConfig(output_dir=self.tmp_dir, max_steps=1, report_to="none")
        with pytest.raises(ValueError, match="requires a `train_dataset`"):
            AsyncGRPOTrainer(
                model="trl-internal-testing/small-Qwen2ForCausalLM-2.5",
                reward_funcs=dummy_reward_func,
                args=args,
                environment_factory={"a": EnvA, "b": EnvB},
                weight_transfer=_StubWeightTransfer(),
            )


def _vision_parameter_names(model) -> set[str]:
    # [FUNCTION DOC]
    # HELPER: Collect the fully qualified names of parameters belonging to a VLM's vision tower.
    # Behavior: finds every submodule that is a PreTrainedModel whose config object is identically the model's
    # vision_config, and returns "<module name>.<parameter name>" for each of its parameters.
    # Inputs: model (VLM PreTrainedModel). Outputs: set of str.
    # Exceptions: AttributeError if the config has no vision_config. Side effects: none.
    """Names of the parameters belonging to a vision-language model's vision tower.

    Located through the vision config, since module names differ across architectures (`visual` for Qwen-VL,
    `vision_tower` for Gemma 3, `vision_model` for SmolVLM).
    """
    vision_config = model.config.vision_config
    return {
        f"{module_name}.{parameter_name}"
        for module_name, module in model.named_modules()
        if isinstance(module, PreTrainedModel) and module.config is vision_config
        for parameter_name, _ in module.named_parameters()
    }


@pytest.mark.skipif(
    not is_ampere_or_newer() and torch_device != "xpu" and torch.version.hip is None,
    reason="Flash Attention requires Ampere or newer GPU, XPU, or ROCm",
)
@pytest.mark.parametrize(
    "model_id",
    [
        pytest.param("trl-internal-testing/tiny-Qwen3VLForConditionalGeneration", id="qwen3_vl"),
    ],
)
class TestAsyncGRPOTrainerVLM(TrlTestCase):
    # [CLASS DOC]
    # Text-only training on vision-language checkpoints (parametrized; currently tiny Qwen3-VL). The vision tower is
    # loaded but must be frozen and excluded from weight sync. GPU-gated.
    """Text-only training on vision-language checkpoints. The vision tower is loaded (so parameter names match the
    vLLM server) but frozen, since the dataset carries no images."""

    def _trainer(self, model_id, **config_kwargs):
        # [FUNCTION DOC]
        # HELPER: Build an AsyncGRPOTrainer for model_id with the shared test config (lr 0.1, batch 3, 3 generations,
        # completion length 8, token budget 64, 5 s timeout) plus overrides, a stub worker and stub transfer.
        # Inputs: model_id (str), config overrides. Outputs: AsyncGRPOTrainer.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")
        training_args = AsyncGRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            # `TokenBudgetBatcher` only closes a row once the next sample no longer fits, and these checkpoints
            # render a `zen` row to ~36 tokens: at 256 no row ever closes and training blocks on an empty queue.
            token_budget=64,
            vllm_server_timeout=5.0,  # short timeout so test fails fast if queue runs dry
            report_to="none",
            **config_kwargs,
        )
        return AsyncGRPOTrainer(
            model=model_id,
            reward_funcs=dummy_reward_func,  # unused: the stub pre-computes rewards, but the trainer requires it
            args=training_args,
            train_dataset=dataset,
            rollout_worker=_StubRolloutWorker(AutoTokenizer.from_pretrained(model_id), dataset, num_generations=3),
            weight_transfer=_StubWeightTransfer(),
        )

    def test_vision_tower_is_frozen(self, model_id):
        # [FUNCTION DOC]
        # Vision parameters exist, are all frozen, trainable parameters exist and are disjoint from them; after
        # create_optimizer, the optimizer's parameter set (by data pointer) equals the trainable parameters.
        trainer = self._trainer(model_id)

        vision = _vision_parameter_names(trainer.model)
        frozen = {n for n, p in trainer.model.named_parameters() if not p.requires_grad}
        trainable = {n for n, p in trainer.model.named_parameters() if p.requires_grad}
        assert vision
        assert vision <= frozen
        assert trainable and not (trainable & vision)

        # Frozen parameters get no optimizer state: the optimizer only sees the text tower.
        trainer.create_optimizer()
        optimized = {p.data_ptr() for group in trainer.optimizer.param_groups for p in group["params"]}
        assert optimized == {p.data_ptr() for n, p in trainer.model.named_parameters() if n in trainable}

    def test_weight_sync_streams_the_text_tower_only(self, model_id):
        # [FUNCTION DOC]
        # The streamed weight names are non-empty, equal the trainable parameter names, include model.language_model.*
        # names and none containing .visual. or .vision_tower.
        # The whole point of loading the full VLM: parameter names must match the vLLM server, which serves the
        # `*ForConditionalGeneration` architecture (`model.language_model.*`), not the text tower alone
        # (`model.*`). Frozen vision weights never change, so they are not streamed at all.
        trainer = self._trainer(model_id)
        streamed = dict(trainer._streaming_iter())

        assert streamed
        assert streamed.keys() == {n for n, p in trainer.model.named_parameters() if p.requires_grad}
        assert any(name.startswith("model.language_model.") for name in streamed)
        assert not any(".visual." in name or ".vision_tower." in name for name in streamed)

    def test_train(self, model_id):
        # [FUNCTION DOC]
        # Skips MoE ids (unsupported FLOPs computation). Trains and asserts train_loss is logged, vision parameters are
        # unchanged and every trainable non-vision parameter changed.
        if "Moe" in model_id:
            # `compute_flops_per_token` reads `num_local_experts`, `intermediate_size` and `decoder_sparse_step`,
            # none of which exist on `Qwen3_5MoeTextConfig`. Text-only Qwen3.5-MoE hits this too.
            pytest.skip("compute_flops_per_token does not support the Qwen3.5-MoE config shape")
        trainer = self._trainer(model_id)
        previous_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        vision = _vision_parameter_names(trainer.model)
        for n, param in previous_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in vision:
                assert torch.equal(param, new_param), f"Vision-tower parameter {n} has changed."
            elif new_param.requires_grad:
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."


def _checkpoint_trainer(output_dir, dataset_start_index=0):
    # [FUNCTION DOC]
    # HELPER: Build a default-config AsyncGRPOTrainer (small Qwen2.5, zen prompt/completion, stubs, 3 generations),
    # then replace rollout_worker with a MagicMock specced as AsyncRolloutWorker whose _loop_kwargs holds
    # dataset_start_index so the checkpoint/resume paths run.
    # Inputs: output_dir (str), dataset_start_index (int, default 0). Outputs: trainer. Side effects: downloads.
    model_id = "trl-internal-testing/small-Qwen2ForCausalLM-2.5"
    dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")
    trainer = AsyncGRPOTrainer(
        model=model_id,
        reward_funcs=dummy_reward_func,
        args=AsyncGRPOConfig(output_dir=output_dir, report_to="none"),
        train_dataset=dataset,
        rollout_worker=_StubRolloutWorker(AutoTokenizer.from_pretrained(model_id), dataset, num_generations=3),
        weight_transfer=_StubWeightTransfer(),
    )
    # The checkpoint paths only run for an `AsyncRolloutWorker`, which requires vLLM
    trainer.rollout_worker = MagicMock(spec=AsyncRolloutWorker)
    trainer.rollout_worker._loop_kwargs = {"dataset_start_index": dataset_start_index}
    return trainer


class TestRolloutStartIndex(TrlTestCase):
    # [CLASS DOC]
    # Tests the rollout loop dataset iterator's start index.
    def test_rollout_loop_skips_to_start_index(self):
        # [FUNCTION DOC]
        # With a 10-row dataset, 2 generations and start index 3, the first row yielded by _repeat_iterator is "row_3".
        dataset = Dataset.from_dict({"prompt": [f"row_{i}" for i in range(10)]})
        loop = _rollout_loop(dataset=dataset, num_generations=2, dataset_start_index=3)
        it = loop._repeat_iterator()
        _group_id, row = next(it)
        assert row["prompt"] == "row_3"


@pytest.mark.skipif(
    not is_ampere_or_newer() and torch_device != "xpu" and torch.version.hip is None,
    reason="Flash Attention requires Ampere or newer GPU, XPU, or ROCm",
)
class TestRolloutStateCheckpoint(TrlTestCase):
    # [CLASS DOC]
    # Tests writing rollout_state.json at checkpoint time and reading it on resume, with base trainer methods patched
    # out. GPU-gated.
    """Prompt-index checkpoint/resume logic, on a real trainer whose rollout worker is stood in (it needs vLLM)."""

    def _trainer_at_step_5(self, trained_groups, dataset_start_index=10):
        # [FUNCTION DOC]
        # HELPER: Build a checkpoint trainer (default start index 10), set global_step 5 and _trained_groups.
        # Outputs: trainer.
        trainer = _checkpoint_trainer(self.tmp_dir, dataset_start_index)
        trainer.state.global_step = 5
        trainer._trained_groups = trained_groups
        return trainer

    def test_save_checkpoint_writes_rollout_state(self):
        # [FUNCTION DOC]
        # Trained groups 0..4 with start index 10: saved prompt_index is 15.
        trainer = self._trainer_at_step_5({0, 1, 2, 3, 4}, dataset_start_index=10)

        with patch.object(_BaseTrainer, "_save_checkpoint"):
            trainer._save_checkpoint(trainer.model, None)

        with open(os.path.join(self.tmp_dir, "checkpoint-5", "rollout_state.json")) as f:
            data = json.load(f)
        assert data["prompt_index"] == 15  # dataset_start_index(10) + first_untrained(5)

    def test_save_checkpoint_writes_rollout_state_before_the_hub_push(self):
        # [FUNCTION DOC]
        # The rollout state file already exists when the patched base _save_checkpoint (Hub push point) runs, once.
        # `super()._save_checkpoint` is what uploads the checkpoint folder under `hub_strategy="checkpoint"`, so the
        # file has to exist by the time it runs. Writing it from an `on_save` callback would not: `Trainer` fires
        # `on_save` only after `_save_checkpoint` returns, leaving the Hub copy without it.
        trainer = self._trainer_at_step_5({0, 1})
        written_before_super = []

        def record(*_args, **_kwargs):
            # [FUNCTION DOC]
            # HELPER: Append whether checkpoint-5/rollout_state.json exists to the enclosing list; ignores arguments.
            written_before_super.append(
                os.path.isfile(os.path.join(self.tmp_dir, "checkpoint-5", "rollout_state.json"))
            )

        with patch.object(_BaseTrainer, "_save_checkpoint", side_effect=record):
            trainer._save_checkpoint(trainer.model, None)

        assert written_before_super == [True]

    def test_save_checkpoint_skips_holes_left_by_stale_drops(self):
        # [FUNCTION DOC]
        # Trained groups missing id 2 with start index 0: saved prompt_index is 2 (first untrained group).
        # Group 2 was never trained (all of its rollouts were dropped as stale), so the cursor stops there: those
        # prompts get re-generated on resume instead of being silently skipped.
        trainer = self._trainer_at_step_5({0, 1, 3, 4, 5}, dataset_start_index=0)

        with patch.object(_BaseTrainer, "_save_checkpoint"):
            trainer._save_checkpoint(trainer.model, None)

        with open(os.path.join(self.tmp_dir, "checkpoint-5", "rollout_state.json")) as f:
            assert json.load(f)["prompt_index"] == 2

    def test_inner_training_loop_sets_dataset_start_index_from_file(self):
        # [FUNCTION DOC]
        # A checkpoint state with prompt_index 77 and model_version 42 sets the worker's dataset_start_index to 77,
        # _groups_before_resume to 77, model_version to 42, and calls update_model_version once with 42.
        checkpoint_dir = os.path.join(self.tmp_dir, "checkpoint-10")
        os.makedirs(checkpoint_dir)
        with open(os.path.join(checkpoint_dir, "rollout_state.json"), "w") as f:
            json.dump({"prompt_index": 77, "model_version": 42}, f)

        trainer = _checkpoint_trainer(self.tmp_dir)

        with patch.object(_BaseTrainer, "_inner_training_loop", return_value=None):
            trainer._inner_training_loop(resume_from_checkpoint=checkpoint_dir)

        assert trainer.rollout_worker._loop_kwargs["dataset_start_index"] == 77
        # Epochs are counted in prompts trained, so a resumed run has to pick that count up too, or it would train
        # `num_train_epochs` more passes on top of the ones already done.
        assert trainer._groups_before_resume == 77
        # Adapter names are derived from `model_version`, so restarting it at 0 would republish a different adapter
        # under a name a still-running vLLM server already holds, and strand the previous run's directories outside
        # the eviction window.
        assert trainer.model_version == 42
        trainer.rollout_worker.update_model_version.assert_called_once_with(42)

    def test_inner_training_loop_defaults_model_version_for_old_checkpoints(self):
        # [FUNCTION DOC]
        # A state file without model_version resets a preset model_version of 7 to 0.
        # Checkpoints written before `model_version` was serialized must still resume, restarting numbering at 0.
        checkpoint_dir = os.path.join(self.tmp_dir, "checkpoint-10")
        os.makedirs(checkpoint_dir)
        with open(os.path.join(checkpoint_dir, "rollout_state.json"), "w") as f:
            json.dump({"prompt_index": 5}, f)

        trainer = _checkpoint_trainer(self.tmp_dir)
        trainer.model_version = 7  # a version the resume has to reset, not the 0 `__init__` starts from

        with patch.object(_BaseTrainer, "_inner_training_loop", return_value=None):
            trainer._inner_training_loop(resume_from_checkpoint=checkpoint_dir)

        assert trainer.model_version == 0


def _rollout_loop(**kwargs):
    # [FUNCTION DOC]
    # HELPER: Construct an _AsyncRolloutLoop with test defaults (constructor only sets up state; no vLLM).
    # Behavior: uses the tiny Qwen3-MoE tokenizer and model name, zen conversational_prompt_only dataset,
    # dummy_reward_func, eos id from the tokenizer, multiprocessing queues for rollout buffer, exceptions and metrics,
    # shared int model version 0, shared float heartbeat 0.0 and a failed event; caller kwargs override any of these.
    # Inputs: kwargs for _AsyncRolloutLoop. Outputs: _AsyncRolloutLoop.
    # Exceptions: TypeError for missing required args. Side effects: downloads tokenizer and dataset; creates mp
    # primitives and (inside the loop) an asyncio event loop that callers should close.
    # `_AsyncRolloutLoop.__init__` only sets up state (no vLLM connection / generation happens here).
    model_id = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    kwargs = {
        "model_name": model_id,
        "dataset": load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train"),
        "reward_funcs": [dummy_reward_func],
        "processing_class": tokenizer,
        "eos_token_ids": [tokenizer.eos_token_id],
        "rollout_buffer": mp.Queue(),
        "model_version_value": mp.Value("i", 0),
        "heartbeat_value": mp.Value("d", 0.0),
        "failed_event": mp.Event(),
        "exception_info_queue": mp.Queue(),
        "metrics_queue": mp.Queue(),
        **kwargs,
    }
    return _AsyncRolloutLoop(**kwargs)


class TestAsyncRolloutWorkerEnvironments(TrlTestCase):
    # [CLASS DOC]
    # Tests the rollout loop's environment and tool wiring (no vLLM): per-environment tool exposure, safe tool
    # discovery, off-event-loop environment calls with timing metrics, and unknown-environment errors.
    """Unit tests for the rollout worker's environment/tool wiring (no vLLM required)."""

    def _make_loop(self, environment_factory, dataset=None):
        # [FUNCTION DOC]
        # HELPER: Build a rollout loop with the given environment factory (class or dict of classes), 2 generations and at
        # most 4 in-flight tasks, using the zen prompt-only dataset unless one is given. Outputs: _AsyncRolloutLoop.
        if dataset is None:
            dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        return _rollout_loop(
            dataset=dataset, environment_factory=environment_factory, num_generations=2, max_inflight_tasks=4
        )

    def test_multiple_environments_expose_only_their_own_tools(self):
        # [FUNCTION DOC]
        # With two environments (counter with increment, echo with shout): the loop is multi-environment, each
        # environment's tool list contains only its own method, the union of tools has both, and the reuse pool holds one
        # probe instance per environment. Closes the asyncio loop afterwards.
        class CounterEnvironment:
            # [CLASS DOC]
            # HELPER: Test environment with a no-op reset and one tool, increment.
            def reset(self, **kwargs):
                # [FUNCTION DOC]
                # HELPER: No-op reset.
                ...

            def increment(self, step: int) -> int:
                # [FUNCTION DOC]
                # HELPER: Tool that returns step unchanged (docstring provides the tool schema). Inputs: step (int).
                """Increment the counter.

                Args:
                    step: Value to add.

                Returns:
                    The updated value.
                """
                return step

        class EchoEnvironment:
            # [CLASS DOC]
            # HELPER: Test environment with a no-op reset and one tool, shout.
            def reset(self, **kwargs):
                # [FUNCTION DOC]
                # HELPER: No-op reset.
                ...

            def shout(self, text: str) -> str:
                # [FUNCTION DOC]
                # HELPER: Tool returning text upper-cased. Inputs: text (str). Outputs: str.
                """Shout the text.

                Args:
                    text: Text to shout.

                Returns:
                    The text in upper case.
                """
                return text.upper()

        loop = self._make_loop({"counter": CounterEnvironment, "echo": EchoEnvironment})
        try:
            assert loop._multi_environment is True
            # Each environment exposes only its own tool, used to render that example's prompt schema.
            assert [tool.__name__ for tool in loop._env_tools["counter"]] == ["increment"]
            assert [tool.__name__ for tool in loop._env_tools["echo"]] == ["shout"]
            # `self.tools` is the union, used only to decide whether a training chat template is needed.
            assert sorted(tool.__name__ for tool in loop.tools) == ["increment", "shout"]
            # The probe instances seed the reuse pool, so they are not wasted.
            assert len(loop._environment_pool["counter"]) == 1
            assert len(loop._environment_pool["echo"]) == 1
        finally:
            loop._loop.close()

    def test_tool_discovery_does_not_evaluate_properties(self):
        # [FUNCTION DOC]
        # An environment with a property that raises when accessed: discovery lists only echo as a tool (single
        # environment keyed by None) and as the union, proving the property was never evaluated.
        # The init-time probe lists an environment's tool methods. `inspect.getmembers` calls `getattr` on every name
        # before applying the predicate, so listing the instance would evaluate this property; listing the class
        # leaves it inert.
        class PropertyEnvironment:
            # [CLASS DOC]
            # HELPER: Environment with a no-op reset, a reward property that raises RuntimeError, and an echo tool.
            def reset(self, **kwargs):
                # [FUNCTION DOC]
                # HELPER: No-op reset.
                ...

            @property
            def reward(self) -> float:
                # [FUNCTION DOC]
                # HELPER: Property that always raises RuntimeError, to detect accidental evaluation during tool discovery.
                raise RuntimeError("`reward` must not be evaluated while discovering tools")

            def echo(self, text: str) -> str:
                # [FUNCTION DOC]
                # HELPER: Tool returning text unchanged. Inputs: text (str). Outputs: str.
                """Echo the text back.

                Args:
                    text: Text to echo.

                Returns:
                    The text, unchanged.
                """
                return text

        loop = self._make_loop(PropertyEnvironment)
        try:
            assert [tool.__name__ for tool in loop._env_tools[None]] == ["echo"]
            assert [tool.__name__ for tool in loop.tools] == ["echo"]
        finally:
            loop._loop.close()

    @require_response_parsing
    def test_environment_calls_run_off_the_event_loop_and_are_timed(self):
        # [FUNCTION DOC]
        # Requires response parsing. Runs the full rollout loop with a slow environment (each of reset, the step tool and
        # get_reward sleeps 20 ms) and a fake generator that emits a step tool call, then a final answer once a tool
        # response is in the prompt; get_reward stops the loop. Drains the metrics queue, summing rate pairs, and asserts
        # reset ran on non-main threads and that the mean of env/reset_s, env/reward_s and tools/step/latency_s is at
        # least 0.02 s.
        reset_threads = []

        class SlowEnvironment:
            # [CLASS DOC]
            # HELPER: Environment whose reset, step and get_reward each sleep 20 ms; reset records its thread.
            def reset(self, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Append the current thread to reset_threads and sleep 20 ms. Side effects: mutates reset_threads.
                reset_threads.append(threading.current_thread())
                time.sleep(0.02)

            def step(self) -> str:
                # [FUNCTION DOC]
                # HELPER: Tool that sleeps 20 ms and returns "ok".
                """
                Take a step.
                """
                time.sleep(0.02)
                return "ok"

            def get_reward(self):
                # [FUNCTION DOC]
                # HELPER: Sleep 20 ms, schedule setting the loop's stop event thread-safely, and return 1.0.
                # Side effects: stops the rollout loop.
                time.sleep(0.02)
                loop._loop.call_soon_threadsafe(loop._stop_event.set)
                return 1.0

        loop = self._make_loop(SlowEnvironment)
        tool_call = loop.tokenizer.encode(_TOOL_CALL.replace('"t"', '"step"'), add_special_tokens=False)
        final = loop.tokenizer.encode(_FINAL, add_special_tokens=False)

        async def _generate_one_turn(prompt_ids):
            # [FUNCTION DOC]
            # HELPER: Async fake generation: returns the final-answer tokens if the decoded prompt contains a tool response,
            # else the step tool-call tokens, with log-prob -0.1 per token. Outputs: tuple of ids and log-probs.
            ids = final if "<tool_response>" in loop.tokenizer.decode(prompt_ids) else tool_call
            return ids, [-0.1] * len(ids)

        loop._generate_one_turn = _generate_one_turn
        PartialState()
        try:
            loop._loop.run_until_complete(loop._run_loops(stop_event=loop._stop_event))
        finally:
            loop._loop.close()

        rates = defaultdict(lambda: [0.0, 0.0])
        while True:
            try:
                payload = loop._metrics_queue.get(timeout=1)
            except queue.Empty:
                break
            for key, value in payload.items():
                if isinstance(value, tuple):
                    rates[key][0] += value[0]
                    rates[key][1] += value[1]
        assert reset_threads and all(t is not threading.main_thread() for t in reset_threads)
        for key in ("env/reset_s", "env/reward_s", "tools/step/latency_s"):
            assert rates[key][0] / rates[key][1] >= 0.02

    def test_unknown_environment_raises(self):
        # [FUNCTION DOC]
        # A dataset whose environment column is "unknown" with only a counter environment configured: running the
        # generate loop raises ValueError "not among the environments".
        # An example whose `environment` field doesn't match any configured environment should fail with a clear error
        # rather than a bare KeyError mid-rollout. The check fires before any generation, so no vLLM is needed here.
        class CounterEnvironment:
            # [CLASS DOC]
            # HELPER: Test environment with a no-op reset and one tool, increment.
            def reset(self, **kwargs):
                # [FUNCTION DOC]
                # HELPER: No-op reset.
                ...

            def increment(self, step: int) -> int:
                # [FUNCTION DOC]
                # HELPER: Tool returning step unchanged. Inputs: step (int).
                """Increment the counter.

                Args:
                    step: Value to add.

                Returns:
                    The updated value.
                """
                return step

        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        dataset = dataset.map(lambda example: {"environment": "unknown"})

        loop = self._make_loop({"counter": CounterEnvironment}, dataset=dataset)
        try:
            with pytest.raises(ValueError, match="not among the environments"):
                loop._loop.run_until_complete(loop._generate_loop(stop_event=loop._stop_event))
        finally:
            loop._loop.close()


_TOKEN_IDS = itertools.count(1)


def _rollout_sample(length: int, advantage: float = 0.0, reward: float = 0.0) -> dict:
    # [FUNCTION DOC]
    # HELPER: Build a synthetic rollout sample dict with globally unique token ids.
    # Behavior: draws length ids from the module counter _TOKEN_IDS (so no two samples share a prefix); completion
    # mask is 0 for the first token and 1 after; old log-probs are zeros; group_id is the first token id; metrics hold
    # reward. Inputs: length (int), advantage (float, default 0.0), reward (float, default 0.0).
    # Outputs: dict. Exceptions: none. Side effects: advances the module-level counter.
    # First token is a prompt token (completion_mask 0); the rest are completion tokens.
    input_ids = np.array([next(_TOKEN_IDS) for _ in range(length)], dtype=np.int64)
    return {
        "input_ids": input_ids,
        "completion_mask": np.array([0] + [1] * (length - 1), dtype=np.int64),
        "old_log_probs": np.zeros(length, dtype=np.float32),
        "advantage": advantage,
        "group_id": int(input_ids[0]),
        "metrics": {"reward": reward},
    }


def _tree_row(samples: list[dict]) -> TreeRow:
    # [FUNCTION DOC]
    # HELPER: Build a TreeRow by adding each sample in order. Inputs: samples (list of dict). Outputs: TreeRow.
    row = TreeRow()
    for sample in samples:
        row.add(sample)
    return row


def _micro_batches(batcher):
    # [FUNCTION DOC]
    # HELPER: Generator converting a batcher's micro-batches of TreeRows into lists of each row's sample lists.
    # Inputs: batcher (iterable of row lists). Outputs: generator of list of lists of dict.
    return ([row.samples for row in rows] for rows in batcher)


def _row_tokens(group: list[dict]) -> int:
    # [FUNCTION DOC]
    # HELPER: Total number of input tokens across the samples of a group. Inputs: group (list of dict). Outputs: int.
    return sum(len(sample["input_ids"]) for sample in group)


class TestPackingAwareBatching(TrlTestCase):
    # [CLASS DOC]
    # CPU tests of FixedCountBatcher, TokenBudgetBatcher and DataCollatorForRollout with TreeRow packing: balance by
    # attention cost, budget enforcement, oversized-sample dropping, padding/position/label/index tensors, and
    # collator metrics.
    """Packing-aware dynamic batching is pure scheduling/tensorization, so these run without a GPU."""

    def test_balance_partitions_all_samples_into_non_empty_rows(self):
        # [FUNCTION DOC]
        # Lengths 5,1,1,1,1 over 2 processes in one micro-batch: 2 non-empty rows and every sample placed exactly once.
        samples = [_rollout_sample(length) for length in (5, 1, 1, 1, 1)]
        groups = next(_micro_batches(FixedCountBatcher(iter(samples), num_processes=2, microbatch_size=len(samples))))

        assert len(groups) == 2
        assert all(len(group) > 0 for group in groups)
        # Every sample is placed exactly once.
        flat = [sample for group in groups for sample in group]
        assert sorted(len(s["input_ids"]) for s in flat) == sorted(len(s["input_ids"]) for s in samples)

    def test_balance_equalizes_attention_cost(self):
        # [FUNCTION DOC]
        # Lengths 3,3,2,2 over 2 rows: both rows have equal TreeRow load.
        # [3, 3, 2, 2] across two rows packs as [3, 2] | [3, 2] -> equal attention cost, not equal counts by accident.
        samples = [_rollout_sample(length) for length in (3, 3, 2, 2)]
        groups = next(_micro_batches(FixedCountBatcher(iter(samples), num_processes=2, microbatch_size=len(samples))))

        loads = [_tree_row(group).load for group in groups]
        assert loads[0] == loads[1]

    def test_balance_prefers_attention_cost_over_count(self):
        # [FUNCTION DOC]
        # Lengths 4,3,2,1: rows split as 1 and 3 samples with equal loads of 10 causal score pairs.
        # One long sample alone balances attention cost better than splitting by count: [4] | [3, 2, 1].
        samples = [_rollout_sample(length) for length in (4, 3, 2, 1)]
        groups = next(_micro_batches(FixedCountBatcher(iter(samples), num_processes=2, microbatch_size=len(samples))))

        counts = sorted(len(group) for group in groups)
        assert counts == [1, 3]
        loads = sorted(_tree_row(group).load for group in groups)
        assert loads == [10, 10]  # causal score pairs: [4] -> 10, [3, 2, 1] -> 6 + 3 + 1

    def test_token_budget_batcher_respects_budget_and_fills_every_row(self):
        # [FUNCTION DOC]
        # 100 length-3 samples, 2 processes, budget 8: first 5 micro-batches have 2 non-empty rows within budget.
        source = (_rollout_sample(3) for _ in range(100))
        batcher = TokenBudgetBatcher(source, num_processes=2, token_budget=8, metrics=defaultdict(list))

        micro_batches = list(itertools.islice(_micro_batches(batcher), 5))
        assert len(micro_batches) == 5
        for groups in micro_batches:
            assert len(groups) == 2
            assert all(len(group) > 0 for group in groups)  # no rank forwards zero tokens
            assert all(_row_tokens(group) <= 8 for group in groups)  # peak memory bounded

    def test_token_budget_batcher_sizes_rows_dynamically(self):
        # [FUNCTION DOC]
        # Budget 8: length-5 samples pack one per row and length-2 samples pack four per row.
        # Long samples pack few per row, short samples pack many — same budget, different counts.
        long_batcher = TokenBudgetBatcher(
            (_rollout_sample(5) for _ in range(100)), num_processes=2, token_budget=8, metrics=defaultdict(list)
        )
        short_batcher = TokenBudgetBatcher(
            (_rollout_sample(2) for _ in range(100)), num_processes=2, token_budget=8, metrics=defaultdict(list)
        )

        long_mb = next(_micro_batches(long_batcher))
        short_mb = next(_micro_batches(short_batcher))

        assert all(len(group) == 1 for group in long_mb)  # 5 + 5 > 8 -> one per row
        assert all(len(group) == 4 for group in short_mb)  # 2 * 4 = 8 -> four per row

    def test_token_budget_batcher_drops_oversized_sample(self):
        # [FUNCTION DOC]
        # Initializes PartialState (for logging); a leading length-12 sample exceeds budget 8 and is dropped: the first 5
        # micro-batches have 2 non-empty rows and no sample longer than 8.
        # A sample longer than the whole budget (12 > 8) fits in no row, so it is dropped, never emptying a row.
        PartialState()  # the drop path logs via accelerate's logger, which needs an initialized state
        source = (_rollout_sample(n) for n in ([12] + [3] * 60))
        batcher = TokenBudgetBatcher(source, num_processes=2, token_budget=8, metrics=defaultdict(list))
        for groups in itertools.islice(_micro_batches(batcher), 5):
            assert len(groups) == 2
            assert all(len(group) > 0 for group in groups)  # every row stays non-empty
            lengths = [len(sample["input_ids"]) for group in groups for sample in group]
            assert all(length <= 8 for length in lengths)  # the oversized sample (12) was dropped

    def test_fixed_count_batcher_yields_balanced_fixed_count_micro_batches(self):
        # [FUNCTION DOC]
        # Cycling lengths 4,3,2,1 with micro-batch size 4: 3 micro-batches each with 2 non-empty rows totalling 4 samples.
        source = (_rollout_sample(length) for length in itertools.cycle((4, 3, 2, 1)))
        batcher = FixedCountBatcher(source, num_processes=2, microbatch_size=4)

        micro_batches = list(itertools.islice(_micro_batches(batcher), 3))
        assert len(micro_batches) == 3
        for groups in micro_batches:
            assert len(groups) == 2
            assert all(len(group) > 0 for group in groups)  # no rank forwards zero tokens
            assert sum(len(group) for group in groups) == 4  # fixed sample count per micro-batch

    def test_collator_pads_unequal_rows(self):
        # [FUNCTION DOC]
        # One sample per row (lengths 3 and 2): asserts right padding of input_ids, attention mask, position ids reset per
        # sequence and zero-padded, shift labels with -100 at ends and padding, token_index with -1 for padding,
        # per-completion-token advantages (0 for padding), segment ids, global trained/forwarded token counts (3, 5),
        # reward metric mean 0.375 and packing ratio pair (5, 5).
        # The planner hands the collator a pre-partitioned micro-batch, wrapped in a length-1 list by the dataloader.
        collator = DataCollatorForRollout(pad_token_id=0, num_processes=2)
        a = _rollout_sample(3, advantage=1.0, reward=0.5)
        b = _rollout_sample(2, advantage=-1.0, reward=0.25)
        groups = [[a], [b]]

        batch = collator([[_tree_row(group) for group in groups]])

        assert batch["input_ids"].tolist() == [
            a["input_ids"].tolist(),
            b["input_ids"].tolist() + [0],
        ]  # b right-padded with pad_token_id
        assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
        assert batch["position_ids"].tolist() == [[0, 1, 2], [0, 1, 0]]  # resets per sequence, 0-padded
        assert batch["shift_labels"].tolist() == [
            a["input_ids"][1:].tolist() + [-100],
            b["input_ids"][1:].tolist() + [-100, -100],
        ]
        assert batch["token_index"].tolist() == [[0, 1], [0, -1]]  # -1 marks inter-rank padding
        assert batch["advantages"].tolist() == [[1.0, 1.0], [-1.0, 0.0]]
        assert batch["segment_id"].tolist() == [[0, 0], [0, 0]]  # one sample per row
        assert batch["global_n_tokens"].tolist() == [3.0, 3.0]  # a: 2 + b: 1 completion tokens
        assert batch["global_n_forward_tokens"].tolist() == [5.0, 5.0]  # every token forwarded, prompts included
        # Per-sample rewards are aggregated here on rank 0 rather than broadcast with the batch and reduced back.
        assert collator.metrics["reward"] == [0.375]  # mean over the whole micro-batch: (0.5 + 0.25) / 2
        assert collator.metrics["batch/packing_ratio"] == [(5, 5)]  # nothing shared, every raw token forwarded

    def test_collator_packs_multiple_samples_per_row(self):
        # [FUNCTION DOC]
        # Two samples per row with equal row lengths: concatenated input ids, position ids resetting at each sequence,
        # concatenated shift labels, token_index of completion positions, advantages and segment ids per completion token,
        # global counts (6, 10), reward mean 0.5, samples_per_row 2 and pad fraction (0, 10).
        collator = DataCollatorForRollout(pad_token_id=0, num_processes=2)
        a = _rollout_sample(3, advantage=1.0, reward=0.5)
        c = _rollout_sample(2, advantage=2.0, reward=0.25)
        b = _rollout_sample(2, advantage=-1.0, reward=0.75)
        d = _rollout_sample(3, advantage=-2.0, reward=0.5)
        groups = [[a, c], [b, d]]  # both rows pack to 5 tokens -> no inter-rank padding

        batch = collator([[_tree_row(group) for group in groups]])

        assert batch["input_ids"].tolist() == [
            a["input_ids"].tolist() + c["input_ids"].tolist(),
            b["input_ids"].tolist() + d["input_ids"].tolist(),
        ]
        assert batch["position_ids"].tolist() == [[0, 1, 2, 0, 1], [0, 1, 0, 1, 2]]  # resets at each sequence start
        labels = [s["input_ids"][1:].tolist() + [-100] for s in (a, c, b, d)]
        assert batch["shift_labels"].tolist() == [labels[0] + labels[1], labels[2] + labels[3]]
        assert batch["token_index"].tolist() == [[0, 1, 3], [0, 2, 3]]
        assert batch["advantages"].tolist() == [[1.0, 1.0, 2.0], [-1.0, -2.0, -2.0]]
        assert batch["segment_id"].tolist() == [[0, 0, 1], [0, 1, 1]]
        assert batch["global_n_tokens"].tolist() == [6.0, 6.0]  # a:2 + c:1 + b:1 + d:2 completion tokens
        assert batch["global_n_forward_tokens"].tolist() == [10.0, 10.0]  # 3 + 2 + 2 + 3
        assert collator.metrics["reward"] == [0.5]  # (0.5 + 0.25 + 0.75 + 0.5) / 4
        assert collator.metrics["batch/samples_per_row"] == [2.0]
        assert collator.metrics["batch/pad_frac"] == [(0, 10)]  # rows pack equal -> no inter-rank padding

    def test_collator_handles_a_ragged_metric_key_set(self):
        # [FUNCTION DOC]
        # Mixes a sample with an extra tools/call_frequency metric and one without, in both orders: reward averages to
        # 0.5 over both and the tool metric is [4.0], averaged only over samples that carry it.
        # A micro-batch mixes samples that carry `tools/*` with samples that do not. Both orderings matter: a first
        # sample without the keys must not drop them, one with them must not KeyError on the rest.
        collator = DataCollatorForRollout(pad_token_id=0, num_processes=2)
        tooled = _rollout_sample(2, reward=1.0)
        tooled["metrics"] = {"reward": 1.0, "tools/call_frequency": 4.0}
        plain = _rollout_sample(2, reward=0.0)

        for groups in ([[plain], [tooled]], [[tooled], [plain]]):
            collator.metrics.clear()
            collator([[_tree_row(group) for group in groups]])
            assert collator.metrics["reward"] == [0.5]  # (1.0 + 0.0) / 2, over both samples
            assert collator.metrics["tools/call_frequency"] == [4.0]  # only the sample that carries it


def _tree_sample(tokens: list[int], n_prompt: int, advantage: float = 0.0, group_id: int = 0):
    # [FUNCTION DOC]
    # HELPER: Build a sample dict from explicit tokens with n_prompt prompt tokens (mask 0, old log-prob 0) followed by
    # completion tokens (mask 1, old log-prob -0.5), plus advantage, group_id and reward metric 0.0.
    # Inputs: tokens (list of int), n_prompt (int), advantage (float, default 0), group_id (int, default 0).
    # Outputs: dict. Side effects: none.
    return {
        "input_ids": np.array(tokens, dtype=np.int64),
        "completion_mask": np.array([0] * n_prompt + [1] * (len(tokens) - n_prompt), dtype=np.int64),
        "old_log_probs": np.array([0.0] * n_prompt + [-0.5] * (len(tokens) - n_prompt), dtype=np.float32),
        "advantage": advantage,
        "group_id": group_id,
        "metrics": {"reward": 0.0},
    }


class TestTreePacking(TrlTestCase):
    # [CLASS DOC]
    # CPU tests of TreeRow prefix-sharing packing and of the batchers' tree-aware planning. Class attribute ROWS holds
    # three samples sharing a 3-token prompt, two of which also share two completion tokens.
    """Tree packing is CPU bookkeeping up to the attention call, so everything but the forward runs without a GPU."""

    ROWS = [
        _tree_sample([101, 102, 103, 201, 202], n_prompt=3, advantage=1.0),
        _tree_sample([101, 102, 103, 301, 302], n_prompt=3, advantage=-1.0),
        _tree_sample([101, 102, 103, 201, 202, 401], n_prompt=3, advantage=0.5),
    ]

    def test_shared_prefixes_are_forwarded_once(self):
        # [FUNCTION DOC]
        # Packing ROWS forwards shared prefixes once: asserts the exact packed input ids and position ids (positions follow
        # the depth in the prefix tree, with branch points repeated).
        row = _tree_row(self.ROWS).pack()

        assert row["input_ids"].tolist() == [101, 102, 103, 201, 202, 202, 401, 103, 301, 302]
        assert row["position_ids"].tolist() == [0, 1, 2, 3, 4, 4, 5, 2, 3, 4]

    def test_one_packed_position_feeds_several_loss_terms(self):
        # [FUNCTION DOC]
        # Asserts the packed shift labels (-100 where no next token is trained), token_index (a packed position can be
        # referenced by several samples' loss terms), per-term advantages and segment ids.
        row = _tree_row(self.ROWS).pack()

        assert row["shift_labels"].tolist() == [-100, -100, 201, 202, -100, 401, -100, 301, 302, -100]
        assert row["token_index"].tolist() == [2, 3, 7, 8, 2, 3, 5]
        assert row["advantages"].tolist() == [1.0, 1.0, -1.0, -1.0, 0.5, 0.5, 0.5]
        assert row["segment_id"].tolist() == [0, 0, 1, 1, 2, 2, 2]

    def test_every_row_sees_exactly_its_own_causal_mask(self):
        # [FUNCTION DOC]
        # For each query segment in the cumulative sequence-length arrays, takes its key path through tree_kv_index and
        # checks every query position sees a prefix ending in itself; the set of visible token tuples equals the set of
        # all prefixes of the original samples.
        row = _tree_row(self.ROWS).pack()

        seen = []
        for q_start, q_end, k_start, k_end in zip(
            row["tree_cu_seqlens_q"][:-1],
            row["tree_cu_seqlens_q"][1:],
            row["tree_cu_seqlens_k"][:-1],
            row["tree_cu_seqlens_k"][1:],
            strict=True,
        ):
            path = row["tree_kv_index"][k_start:k_end]
            for q in range(q_start, q_end):
                visible = path[: len(path) - (q_end - q) + 1]
                assert visible[-1] == q
                seen.append(tuple(row["input_ids"][visible].tolist()))
        prefixes = {tuple(s["input_ids"][: i + 1]) for s in self.ROWS for i in range(len(s["input_ids"]))}

        assert set(seen) == prefixes

    def test_collator_pads_tree_rows_and_keeps_the_segments_aligned(self):
        # [FUNCTION DOC]
        # Collates a tree row (ROWS) and a single-sample row over 2 processes: asserts padded input ids and attention mask,
        # cumulative q/k sequence lengths padded with -1, per-row max sequence lengths, the second row's shift labels, and
        # packing ratio (21, 15).
        collator = DataCollatorForRollout(pad_token_id=0, num_processes=2)
        groups = [self.ROWS, [_tree_sample([101, 102, 103, 201, 202], n_prompt=3)]]

        batch = collator([[_tree_row(group) for group in groups]])

        assert batch["input_ids"].tolist() == [
            [101, 102, 103, 201, 202, 202, 401, 103, 301, 302],
            [101, 102, 103, 201, 202, 0, 0, 0, 0, 0],
        ]
        assert batch["attention_mask"].tolist() == [[1] * 10, [1] * 5 + [0] * 5]
        assert batch["tree_cu_seqlens_q"].tolist() == [[0, 2, 4, 5, 7, 10], [0, 5, -1, -1, -1, -1]]
        assert batch["tree_cu_seqlens_k"].tolist() == [[0, 2, 6, 11, 17, 22], [0, 5, -1, -1, -1, -1]]
        assert batch["tree_max_seqlens"].tolist() == [[3, 6], [5, 5]]
        assert batch["shift_labels"][1].tolist() == [-100, -100, 201, 202] + [-100] * 6
        assert collator.metrics["batch/packing_ratio"] == [(21, 15)]  # 21 raw tokens forwarded as 15

    def test_cost_counts_unique_tokens_and_surviving_score_pairs(self):
        # [FUNCTION DOC]
        # The TreeRow for ROWS has 10 unique tokens and a load of 38 causal score pairs.
        row = _tree_row(self.ROWS)
        assert (len(row.tokens), row.load) == (10, 15 + 12 + 11)  # depths 1..5, then 3..5 and 5..6 for the branches

    def test_groups_share_a_common_prefix(self):
        # [FUNCTION DOC]
        # Two samples from different groups sharing tokens 1,2,3 still share that prefix when packed.
        rows = [_tree_sample([1, 2, 3, last], n_prompt=1, group_id=g) for g, last in ((0, 4), (1, 5))]
        packed = _tree_row(rows).pack()

        assert packed["input_ids"].tolist() == [1, 2, 3, 4, 3, 5]

    def _stream(self, n_groups, rows_per_group=2):
        # [FUNCTION DOC]
        # HELPER: Generator of rows_per_group (default 2) samples per group for n_groups groups; each sample is a 6-token
        # group-specific prompt plus 2 row-specific completion tokens, with group_id set.
        """Groups of rows over their own 6-token prompt: 8 raw tokens each, but only 3 novel after the first."""
        for g in range(n_groups):
            for i in range(rows_per_group):
                yield _tree_sample([1000 * g + t for t in range(6)] + [100 + i, 200 + i], n_prompt=6, group_id=g)

    def test_budget_bounds_unique_tokens_so_more_rows_fit(self):
        # [FUNCTION DOC]
        # With budget 22 counted in unique tokens, the first micro-batch holds 8 samples in total.
        tree = next(_micro_batches(TokenBudgetBatcher(self._stream(50), 2, 22, defaultdict(list))))

        assert sum(len(group) for group in tree) == 8  # two 11-unique-token groups per row, of two rows each

    def test_planner_reads_one_sample_of_lookahead(self):
        # [FUNCTION DOC]
        # Wraps the source in a counting generator; after one micro-batch with budget 20 the planner has read exactly the
        # number of placed samples plus one.
        reads = 0

        def counted(source):
            # [FUNCTION DOC]
            # HELPER: Generator re-yielding source samples while incrementing the enclosing reads counter.
            nonlocal reads
            for sample in source:
                reads += 1
                yield sample

        batcher = TokenBudgetBatcher(counted(self._stream(50)), 2, 20, defaultdict(list))
        micro_batch = next(_micro_batches(batcher))

        assert reads == sum(len(group) for group in micro_batch) + 1

    def test_a_group_stays_in_one_row_so_its_rows_share(self):
        # [FUNCTION DOC]
        # Over 5 micro-batches with budget 20, all samples of a group land in a single row.
        batcher = TokenBudgetBatcher(self._stream(50), 2, 20, defaultdict(list))

        for micro_batch in itertools.islice(_micro_batches(batcher), 5):
            rows_by_group = defaultdict(set)
            for i, group in enumerate(micro_batch):
                for sample in group:
                    rows_by_group[sample["group_id"]].add(i)
            assert all(len(rows) == 1 for rows in rows_by_group.values())

    def test_rows_of_identical_samples_still_close_a_micro_batch(self):
        # [FUNCTION DOC]
        # With an endless stream of identical samples and budget 20, a micro-batch is produced with non-empty rows whose
        # trained (completion) token count is at most 20.
        source = (_tree_sample(list(range(10)), n_prompt=6) for _ in range(10_000))
        batcher = TokenBudgetBatcher(source, 2, 20, defaultdict(list))

        micro_batch = next(_micro_batches(batcher))

        assert all(len(group) > 0 for group in micro_batch)
        assert all(sum(sum(s["completion_mask"]) for s in group) <= 20 for group in micro_batch)

    def test_a_split_group_can_be_rejoined_on_either_row(self):
        # [FUNCTION DOC]
        # A crafted stream of one group whose samples must be split across the 2 rows (budget 14) followed by an oversized
        # sample of another group: the first micro-batch places all 5 group samples.
        prompt = list(range(1, 4))
        stream = [
            _tree_sample(prompt + list(range(10, 10 + n)), len(prompt) + n - t, group_id=0)
            for n, t in ((3, 3), (6, 6), (4, 4), (5, 5), (7, 4))
        ]
        stream.append(_tree_sample(list(range(20, 34)), n_prompt=7, group_id=1))  # fits in neither row: closes
        batcher = TokenBudgetBatcher(iter(stream), 2, 14, defaultdict(list))

        micro_batch = next(_micro_batches(batcher))

        assert sum(len(group) for group in micro_batch) == 5

    def test_a_lone_group_fills_every_row_of_a_fixed_count_micro_batch(self):
        # [FUNCTION DOC]
        # FixedCountBatcher with 8 processes and micro-batch size 8 over groups of 8 samples: every row is non-empty in the
        # first 2 micro-batches.
        source = (_tree_sample(list(range(6)) + [100 + i], n_prompt=6, group_id=i // 8) for i in range(16))
        batcher = FixedCountBatcher(source, num_processes=8, microbatch_size=8)

        for micro_batch in itertools.islice(_micro_batches(batcher), 2):
            assert all(len(group) > 0 for group in micro_batch)

    def test_a_group_is_split_rather_than_let_a_rank_starve(self):
        # [FUNCTION DOC]
        # An endless single group with budget 10 on 2 rows: the first 3 micro-batches have no empty row.
        source = (_tree_sample(list(range(6)) + [100 + i, 200 + i], n_prompt=6, group_id=0) for i in range(100))
        batcher = TokenBudgetBatcher(source, 2, 10, defaultdict(list))

        for micro_batch in itertools.islice(_micro_batches(batcher), 3):
            assert all(len(group) > 0 for group in micro_batch)


def _sequence_packed_batch(samples: list[dict]) -> dict[str, torch.Tensor]:
    # [FUNCTION DOC]
    # HELPER: Build the reference sequence-packed batch (no prefix sharing) for the same samples.
    # Behavior: packs each sample alone as a TreeRow, concatenates every packed field along the sequence dimension
    # with a leading batch dim of 1; offsets token_index by each sample's start and segment ids by sample index; sets
    # tree_kv_index to all positions in order, both cumulative sequence-length tensors to the sample offsets (int32),
    # max sequence lengths to the longest sample, an all-ones attention mask, and global trained/forwarded token
    # counts and mean sequence length.
    # Inputs: samples (list of dict). Outputs: dict of tensors. Exceptions: none. Side effects: none.
    """Main's sequence packing: the samples end to end, `position_ids` restarting at each one, nothing shared."""
    rows = [_tree_row([sample]).pack() for sample in samples]
    offsets = list(itertools.accumulate((row["input_ids"].numel() for row in rows), initial=0))
    batch = {key: torch.cat([row[key] for row in rows])[None] for key in rows[0]}
    batch["token_index"] = torch.cat(
        [row["token_index"] + offset for row, offset in zip(rows, offsets, strict=False)]
    )[None]
    batch["segment_id"] = torch.cat([row["segment_id"] + i for i, row in enumerate(rows)])[None]
    batch["tree_kv_index"] = torch.arange(offsets[-1])[None]
    batch["tree_cu_seqlens_q"] = batch["tree_cu_seqlens_k"] = torch.tensor([offsets], dtype=torch.int32)
    longest = max(len(sample["input_ids"]) for sample in samples)
    batch["tree_max_seqlens"] = torch.tensor([[longest, longest]])
    batch["attention_mask"] = torch.ones_like(batch["input_ids"])
    batch["global_n_tokens"] = torch.tensor([float(batch["token_index"].numel())])
    batch["global_n_forward_tokens"] = torch.tensor([float(offsets[-1])])
    batch["mean_seq_len"] = torch.tensor([offsets[-1] / len(samples)])
    return batch


@require_torch_accelerator
@pytest.mark.parametrize(
    ("model_id", "model_init_kwargs"),
    [
        pytest.param("trl-internal-testing/small-Qwen2ForCausalLM-2.5", {}, id="qwen2"),
        pytest.param(
            "trl-internal-testing/tiny-Gemma2ForCausalLM",
            {"head_dim": 32, "sliding_window": 16, "ignore_mismatched_sizes": True},
            id="gemma2-sliding-window-softcap",
            marks=pytest.mark.skipif(
                torch.version.hip is not None, reason="The AITER kernel used on ROCm has no softcap"
            ),
        ),
        pytest.param("trl-internal-testing/tiny-Qwen3MoeForCausalLM", {}, id="qwen3-moe"),
        pytest.param("trl-internal-testing/tiny-Qwen3VLForConditionalGeneration", {}, id="qwen3-vl"),
    ],
)
class TestTreePackingParity(TrlTestCase):
    # [CLASS DOC]
    # Accelerator-only parity test across models (small Qwen2.5; tiny Gemma2 with sliding window and softcap, skipped
    # on ROCm; tiny Qwen3-MoE; tiny Qwen3-VL): tree-packed loss and gradients must match the sequence-packed reference.
    """A tree-packed micro-batch must give the GRPO loss and gradients of main's sequence packing of the same samples."""

    def _samples(self):
        # [FUNCTION DOC]
        # HELPER: Build 12 samples: for 2 groups, a 64-token group prompt plus 6 completions of 24 tokens (4 distinct, 2
        # sharing the first 12 tokens of the first), advantages from -2.5 to 2.5 by index, and old log-probs all -12.
        # Outputs: list of dict.
        samples = []
        for group_id in range(2):
            prompt = [1 + 100 * group_id + i for i in range(64)]
            completions = [[1000 + 100 * i + j for j in range(24)] for i in range(4)]
            completions += [completions[0][:12] + [2000 + 100 * i + j for j in range(12)] for i in range(2)]
            for i, completion in enumerate(completions):
                sample = _tree_sample(prompt + completion, len(prompt), advantage=i - 2.5, group_id=group_id)
                sample["old_log_probs"] = np.full(len(sample["input_ids"]), -12.0, dtype=np.float32)
                samples.append(sample)
        return samples

    def test_loss_and_grads_match_sequence_packing(self, model_id, model_init_kwargs):
        # [FUNCTION DOC]
        # Builds a trainer in bfloat16 with router aux loss off, moves the model to the device, and computes the loss
        # and gradients twice: tree attention on the collated tree row, and a flash-attention kernel (AITER on ROCm,
        # flash-attn3 otherwise) on the sequence-packed batch. Asserts packing ratio (1056, 378), losses close within
        # 1e-3, same gradient keys, and relative gradient difference norm below 5 percent.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")
        trainer = AsyncGRPOTrainer(
            model=model_id,
            reward_funcs=dummy_reward_func,
            args=AsyncGRPOConfig(
                output_dir=self.tmp_dir,
                model_init_kwargs={"dtype": torch.bfloat16, **model_init_kwargs},
                router_aux_loss_coef=0.0,
                report_to="none",
            ),
            train_dataset=dataset,
            rollout_worker=_StubRolloutWorker(AutoTokenizer.from_pretrained(model_id), dataset, num_generations=3),
            weight_transfer=_StubWeightTransfer(),
        )
        trainer.current_gradient_accumulation_steps = 1
        model = trainer.model.to(torch_device)
        samples = self._samples()
        collator = DataCollatorForRollout(pad_token_id=0)
        runs = [
            (TREE_ATTENTION, collator([[_tree_row(samples)]])),
            (
                "kernels-community/aiter-flash-attn@v2" if torch.version.hip else "kernels-community/flash-attn3",
                _sequence_packed_batch(samples),
            ),
        ]

        losses, grads = [], []
        for attn_implementation, batch in runs:
            model.set_attn_implementation(attn_implementation)
            model.zero_grad()
            loss = trainer.compute_loss(model, {key: value.to(torch_device) for key, value in batch.items()})
            loss.backward()
            losses.append(loss.detach().float())
            grads.append({name: p.grad.float() for name, p in model.named_parameters() if p.grad is not None})

        assert collator.metrics["batch/packing_ratio"] == [(1056, 378)]
        torch.testing.assert_close(losses[0], losses[1], atol=1e-3, rtol=1e-3)
        assert grads[0].keys() == grads[1].keys()
        packed_grads, sequence_grads = (torch.cat([g.flatten() for g in run.values()]) for run in grads)
        assert (packed_grads - sequence_grads).norm() < 5e-2 * sequence_grads.norm()


class TestTreePackingRefusesUnsupportedModels(TrlTestCase):
    # [CLASS DOC]
    # Tests that the trainer refuses architectures tree packing cannot handle.
    @pytest.mark.parametrize(
        ("model_id", "match"),
        [
            pytest.param(
                "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration-Think", "linear_attention", id="qwen35"
            ),
            pytest.param(
                "trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6", "linear_attention", id="qwen36-moe"
            ),
            pytest.param("trl-internal-testing/tiny-GptOssForCausalLM", "attention sinks", id="gpt-oss"),
        ],
    )
    def test_init_raises(self, model_id, match):
        # [FUNCTION DOC]
        # Parametrized: Qwen3.5 and Qwen3.6-MoE (linear attention) and GPT-OSS (attention sinks) raise ValueError with the
        # matching message at construction.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")
        with pytest.raises(ValueError, match=match):
            AsyncGRPOTrainer(
                model=model_id,
                reward_funcs=dummy_reward_func,
                args=AsyncGRPOConfig(output_dir=self.tmp_dir, report_to="none"),
                train_dataset=dataset,
                rollout_worker=_StubRolloutWorker(AutoTokenizer.from_pretrained(model_id), dataset, num_generations=3),
                weight_transfer=_StubWeightTransfer(),
            )


def _finalize(turns, rollout_id="r0", fork_threshold=1024):
    # [FUNCTION DOC]
    # HELPER: Run _chain_to_sequences on turn records and return only the training rows (drop the tally).
    # Inputs: turns (list of TurnRecord), rollout_id (default "r0"), fork_threshold (default 1024).
    # Outputs: list of TrainingSequence.
    rows, _tally = _chain_to_sequences(turns, rollout_id, fork_threshold)
    return rows


class TestRolloutWorkerProtocol(TrlTestCase):
    # [CLASS DOC]
    # Guards that the stub worker keeps up with RolloutWorkerProtocol.
    def test_stub_worker_exposes_every_protocol_attribute(self):
        # [FUNCTION DOC]
        # Builds the stub with no tokenizer and empty dataset (iterator is lazy) and asserts every annotated protocol
        # attribute exists on it.
        # A stub that falls behind the protocol breaks training, and only the GPU-gated `test_train` would notice.
        stub = _StubRolloutWorker(None, [])
        missing = [name for name in RolloutWorkerProtocol.__annotations__ if not hasattr(stub, name)]
        assert not missing


class TestMetricReduction(TrlTestCase):
    # [CLASS DOC]
    # Tests _reduce_metric: shape and name select the reduction.
    """The value's shape and the key's suffix are the whole reduction API; nothing is registered or configured."""

    def test_gauge_is_meaned(self):
        # [FUNCTION DOC]
        # A plain gauge reduces to the mean (3.0).
        assert _reduce_metric("perf/fwd_bwd_s", [1.0, 2.0, 6.0]) == 3.0

    def test_counter_is_summed(self):
        # [FUNCTION DOC]
        # A _total counter reduces to the sum (3.0).
        assert _reduce_metric("sample/dropped_stale_total", [1.0, 1.0, 1.0]) == 3.0

    def test_extrema(self):
        # [FUNCTION DOC]
        # _max/_min suffixes and max/min words take the extreme; a name merely containing "max" letters is averaged.
        assert _reduce_metric("sample/forwarded_tokens_max", [3.0, 9.0, 5.0]) == 9.0
        assert _reduce_metric("clip_ratio/low_min", [3.0, 9.0, 5.0]) == 3.0
        # A `max`/`min` WORD, not a suffix, so the upstream spellings reduce correctly too.
        assert _reduce_metric("completions/max_length", [3.0, 9.0, 5.0]) == 9.0
        assert _reduce_metric("completions/min_length", [3.0, 9.0, 5.0]) == 3.0
        # ... and a name that merely contains those letters is still a gauge.
        assert _reduce_metric("perf/maximal_s", [2.0, 4.0]) == 3.0

    def test_rate_is_sum_over_sum_not_mean_of_ratios(self):
        # [FUNCTION DOC]
        # Rate pairs reduce to total numerator over total denominator (200/101), not the mean of ratios.
        # The two windows have rates 1 tok/s and 100 tok/s. Meaning the ratios would give 50.5; the honest rate is
        # total tokens over total seconds. This is the defect that made `training_tok/s` unusable.
        assert _reduce_metric("perf/forwarded_tok_s_e2e", [(100.0, 100.0), (100.0, 1.0)]) == pytest.approx(200 / 101)

    def test_rate_with_zero_denominator_is_nan(self):
        # [FUNCTION DOC]
        # A rate with zero total denominator reduces to NaN.
        assert math.isnan(_reduce_metric("perf/forwarded_tok_s_e2e", [(0.0, 0.0)]))


class TestWorkerMetricPush(TrlTestCase):
    # [CLASS DOC]
    # Tests _push_metrics of the rollout loop: merged payload and non-blocking behavior.
    """The worker's payload has the same shape as the trainer's sink, so draining it is an append."""

    def test_counters_and_rates_ride_along_and_reset(self):
        # [FUNCTION DOC]
        # A counter of 2 and rate (3.0, 2) are merged with the pushed gauge into one payload; the next empty push yields an
        # empty payload (deltas reset).
        loop = _rollout_loop()
        loop._counters["tools/search/call_total"] += 2
        loop._rates["tools/latency_s"][0] += 3.0
        loop._rates["tools/latency_s"][1] += 2
        loop._push_metrics({"rollout/score_s": 0.5})

        payload = loop._metrics_queue.get(timeout=5)
        assert payload == {
            "rollout/score_s": 0.5,
            "tools/search/call_total": 2.0,
            "tools/latency_s": (3.0, 2.0),
        }
        # Counters carry deltas: whatever was pushed must not be pushed again.
        loop._push_metrics({})
        assert loop._metrics_queue.get(timeout=5) == {}

    def test_push_never_blocks_when_the_trainer_stops_draining(self):
        # [FUNCTION DOC]
        # With a size-1 metrics queue, 50 pushes complete without blocking.
        loop = _rollout_loop(metrics_queue=mp.Queue(maxsize=1))
        for _ in range(50):
            loop._push_metrics({"rollout/score_s": 1.0})  # drops instead of blocking generation


class TestToolExecution(TrlTestCase):
    # [CLASS DOC]
    # Tests _execute_tool_calls of the rollout loop: sync/async/failing/unknown tools, ordering, counters, and that
    # sync tools run off the event loop.
    @staticmethod
    def _call(name, **arguments):
        # [FUNCTION DOC]
        # HELPER: Static method building a tool-call dict of type "function" with the given name and keyword arguments.
        # Outputs: dict.
        return {"type": "function", "function": {"name": name, "arguments": arguments}}

    def test_every_tool_shape_runs_and_calls_keep_their_order(self):
        # [FUNCTION DOC]
        # Executes calls to a sync tool, an async tool, a raising async tool and an unknown tool. Asserts 4 calls and 2
        # failures, response messages in call order with the expected contents (error text containing the exception
        # message, "unknown tool"), the failing tool's failure counter is 1, the unknown-name counter is 1, and 3 latency
        # samples were recorded.
        def sync_tool(value: int) -> str:
            # [FUNCTION DOC]
            # HELPER: Return "sync:<value>".
            return f"sync:{value}"

        async def async_tool(value: int) -> str:
            # [FUNCTION DOC]
            # HELPER: Yield to the event loop once, then return "async:<value>".
            await asyncio.sleep(0)
            return f"async:{value}"

        async def failing_tool(value: int) -> str:
            # [FUNCTION DOC]
            # HELPER: Always raise RuntimeError "boom:<value>".
            raise RuntimeError(f"boom:{value}")

        loop = _rollout_loop(max_inflight_tasks=4)
        tool_dict = {
            "sync_tool": sync_tool,
            "async_tool": async_tool,
            "failing_tool": failing_tool,
        }
        calls = [
            self._call("sync_tool", value=1),
            self._call("async_tool", value=2),
            self._call("failing_tool", value=3),
            self._call("missing_tool", value=4),
        ]
        messages, n_calls, n_failures = asyncio.run(loop._execute_tool_calls(calls, tool_dict))

        assert n_calls == 4
        assert n_failures == 2
        assert [m["name"] for m in messages] == ["sync_tool", "async_tool", "failing_tool", "missing_tool"]
        assert messages[0]["content"] == "sync:1"
        assert messages[1]["content"] == "async:2"
        assert "boom:3" in messages[2]["content"]
        assert "unknown tool" in messages[3]["content"]
        assert loop._counters["tools/failing_tool/failure_total"] == 1
        assert loop._counters["tools/unknown_name_total"] == 1
        assert loop._rates["tools/latency_s"][1] == 3

    def test_slow_sync_tool_does_not_block_the_event_loop(self):
        # [FUNCTION DOC]
        # Runs a 0.5 s sleeping sync tool alongside a ticker task that increments every 10 ms; asserts the tool result is
        # "done" and the ticker advanced more than twice (the tool ran in a thread).
        def slow_tool() -> str:
            # [FUNCTION DOC]
            # HELPER: Sleep 0.5 s and return "done".
            time.sleep(0.5)
            return "done"

        loop = _rollout_loop(max_inflight_tasks=4)

        async def scenario():
            # [FUNCTION DOC]
            # HELPER: Async scenario: start the ticker, execute the slow tool call, cancel the ticker, and return the messages
            # and tick count.
            ticks = 0

            async def ticker():
                # [FUNCTION DOC]
                # HELPER: Endless async loop incrementing the enclosing ticks counter every 10 ms until cancelled.
                nonlocal ticks
                while True:
                    ticks += 1
                    await asyncio.sleep(0.01)

            task = asyncio.create_task(ticker())
            messages, _, _ = await loop._execute_tool_calls([self._call("slow_tool")], {"slow_tool": slow_tool})
            task.cancel()
            return messages, ticks

        messages, ticks = asyncio.run(scenario())
        assert messages[0]["content"] == "done"
        assert ticks > 2  # inline, the ticker would have advanced once at most


class TestReconciler(TrlTestCase):
    # [CLASS DOC]
    # Tests the multi-turn reconciler (_chain_to_sequences, _SampleBuilder, _common_prefix_len): how consecutive turn
    # records are merged into training rows depending on token drift between a turn's held tokens and the next
    # turn's re-tokenized prompt (CLEAN keeps one row, REALIGN masks drifted tokens, FORK starts a new row).
    @pytest.mark.parametrize("mask", [None, [1, 0, 1], [0, 0, 0]])
    def test_completion_masks_are_generic(self, mask):
        # [FUNCTION DOC]
        # Parametrized over no mask, a partial mask and an all-zero mask on a single turn: all-zero yields no rows;
        # otherwise the row's completion mask is prompt zeros followed by the given mask (all ones by default), input ids
        # are prompt plus completion, and the last three old log-probs equal the turn's.
        rows = _finalize([TurnRecord([1, 2], [3, 4, 5], [-0.1, -0.2, -0.3], mask)])
        if mask == [0, 0, 0]:
            assert rows == []
        else:
            assert rows[0].completion_mask == [0, 0] + (mask if mask is not None else [1, 1, 1])
            assert rows[0].input_ids == [1, 2, 3, 4, 5]
            assert rows[0].old_log_probs[-3:] == [-0.1, -0.2, -0.3]

    def test_common_prefix_len(self):
        # [FUNCTION DOC]
        # Checks _common_prefix_len for identical lists, a strict prefix, an empty list, and with chunk size 2 both the
        # whole-chunk fast path and the per-token fallback.
        assert _common_prefix_len([1, 2, 3], [1, 2, 3]) == 3  # identical
        assert _common_prefix_len([1, 2], [1, 2, 3, 4]) == 2  # old is a prefix of new
        assert _common_prefix_len([], [1, 2, 3]) == 0  # empty
        # chunk=2 exercises both the whole-chunk fast path and the per-token fallback
        assert _common_prefix_len([1, 2, 3, 4, 5], [1, 2, 3, 9, 5], chunk=2) == 3
        assert _common_prefix_len([1, 2, 3, 4], [1, 2, 3, 4, 5], chunk=2) == 4

    def test_single_turn(self):
        # [FUNCTION DOC]
        # One turn yields one row with concatenated ids, mask 0 for prompt and 1 for completion, zero prompt log-probs and
        # rollout id "r0".
        rows = _finalize([TurnRecord([1, 2, 3], [10, 11], [-0.1, -0.2])])
        assert len(rows) == 1
        assert rows[0].input_ids == [1, 2, 3, 10, 11]
        assert rows[0].completion_mask == [0, 0, 0, 1, 1]
        assert rows[0].old_log_probs == [0.0, 0.0, 0.0, -0.1, -0.2]
        assert rows[0].rollout_id == "r0"

    def test_clean_chain_stays_one_row(self):
        # [FUNCTION DOC]
        # When turn 2's prompt extends turn 1's prompt plus generation plus tool tokens, the result is one row with tool
        # tokens masked 0 and both generations masked 1, with matching log-probs.
        # Turn 2's re-tokenized prompt starts with the held tokens (gen [10,11] + tool [20,21]) -> CLEAN.
        turn1 = TurnRecord([1, 2, 3], [10, 11], [-0.1, -0.2])
        turn2 = TurnRecord([1, 2, 3, 10, 11, 20, 21], [30, 31], [-0.3, -0.4])
        rows = _finalize([turn1, turn2])
        assert len(rows) == 1
        assert rows[0].input_ids == [1, 2, 3, 10, 11, 20, 21, 30, 31]
        assert rows[0].completion_mask == [0, 0, 0, 1, 1, 0, 0, 1, 1]  # prompt=0 gen=1 tool=0 gen=1
        assert rows[0].old_log_probs == [0.0, 0.0, 0.0, -0.1, -0.2, 0.0, 0.0, -0.3, -0.4]

    def test_rewrite_forks_into_two_rows(self):
        # [FUNCTION DOC]
        # Divergence inside turn 1's answer with fork threshold 2 yields two rows: turn 1 trained in row 1, and row 2 with
        # turn 2's prompt as context (all 0) and its generation trained.
        # Divergence inside turn 1's answer + a turn >= fork_threshold -> FORK. Every generated token is
        # trained in exactly one row (turn 1's tokens are context in row 2).
        turn1 = TurnRecord([1, 2, 3], [10, 11, 12, 13], [-0.2] * 4)
        turn2 = TurnRecord([1, 2, 3, 10, 99, 88, 77], [30, 31, 32], [-0.2] * 3)
        rows = _finalize([turn1, turn2], fork_threshold=2)
        assert len(rows) == 2
        assert rows[0].input_ids == [1, 2, 3, 10, 11, 12, 13]
        assert rows[0].completion_mask == [0, 0, 0, 1, 1, 1, 1]
        assert rows[1].input_ids == [1, 2, 3, 10, 99, 88, 77, 30, 31, 32]
        assert rows[1].completion_mask == [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

    def test_fork_when_divergence_precedes_last_response(self):
        # [FUNCTION DOC]
        # After appending a clean turn, classifying a next turn whose prompt diverges before the last response start
        # returns FORK with 4 drift tokens, regardless of the large threshold.
        # matched < last_response_start_idx -> FORK regardless of threshold (distinct from the length trigger).
        builder = _SampleBuilder(fork_threshold=1024)
        builder.append_turn(
            TurnRecord([1, 2, 3], [10, 11], [-0.2] * 2), DriftKind.CLEAN
        )  # last_response_start_idx == 3
        # 5 held tokens, only the first matches -> 4 tokens of drift, reported alongside the kind.
        assert builder.classify_token_drift(TurnRecord([1, 9, 3, 10, 11], [30], [-0.2] * 1)) == (DriftKind.FORK, 4)

    def test_drift_tally_counts_transitions(self):
        # [FUNCTION DOC]
        # A single-turn chain gives an all-zero tally; a 3-turn chain (one clean, one forked transition) gives 2
        # transitions, 1 clean, 1 fork and drift_max 4.
        # The tally is what makes `fork_threshold_tokens` tunable: it reports the drift the threshold is compared
        # against. It counts turn TRANSITIONS, so a single-turn rollout has none.
        turn1 = TurnRecord([1, 2, 3], [10, 11, 12, 13], [-0.2] * 4)
        clean = TurnRecord([1, 2, 3, 10, 11, 12, 13], [20], [-0.2] * 1)
        forked = TurnRecord([1, 2, 3, 10, 99, 88, 77, 20], [30], [-0.2] * 1)

        _rows, tally = _chain_to_sequences([turn1], "r0", 2)
        assert tally == {"clean": 0, "realign": 0, "fork": 0, "transitions": 0, "drift_tokens": 0, "drift_max": 0}

        _rows, tally = _chain_to_sequences([turn1, clean, forked], "r0", 2)
        assert (tally["transitions"], tally["clean"], tally["fork"]) == (2, 1, 1)
        assert tally["drift_max"] == 4  # the forking transition invalidated 4 held tokens

    def test_tail_wobble_realigns_to_context(self):
        # [FUNCTION DOC]
        # A re-rendered last generated token (short drift) realigns: one row where only the drifted token is masked 0 with
        # log-prob 0, keeping the matched part trained.
        # Last generated token re-renders (11 -> 12): a short wobble in the last answer -> REALIGN. Only the drifted
        # token is masked to context; the matched part of the answer (10) keeps its training signal and logprob.
        turn1 = TurnRecord([1, 2, 3], [10, 11], [-0.1, -0.2])
        turn2 = TurnRecord([1, 2, 3, 10, 12], [30, 31], [-0.3, -0.4])
        rows = _finalize([turn1, turn2])
        assert len(rows) == 1
        assert rows[0].input_ids == [1, 2, 3, 10, 12, 30, 31]
        assert rows[0].completion_mask == [0, 0, 0, 1, 0, 1, 1]
        assert rows[0].old_log_probs == [0.0, 0.0, 0.0, -0.1, 0.0, -0.3, -0.4]

    def test_large_drift_forks_even_when_next_turn_is_short(self):
        # [FUNCTION DOC]
        # With threshold 3, a 6-token drift from a template dropping turn 1's answer forks into two rows even though turn
        # 2's generation is a single token; turn 1's answer stays fully trained.
        # The Qwen3 case: turn 1 is a long trained answer (a reasoning block); the template drops it on turn 2, so
        # the prompt diverges at the answer start (large drift) while turn 2's own answer is tiny. The decision must
        # key on the drift (6 tokens invalidated), not the incoming turn's length (1) -> FORK, so turn 1 keeps its
        # training signal. (fork_threshold sits between the two: gating on output length would wrongly REALIGN.)
        turn1 = TurnRecord([1, 2, 3], [10, 11, 12, 13, 14, 15], [-0.2] * 6)  # long trained answer
        turn2 = TurnRecord([1, 2, 3, 99, 88], [30], [-0.2] * 1)  # answer dropped by the template; short next turn
        rows = _finalize([turn1, turn2], fork_threshold=3)
        assert len(rows) == 2
        assert rows[0].input_ids == [1, 2, 3, 10, 11, 12, 13, 14, 15]
        assert rows[0].completion_mask == [0, 0, 0, 1, 1, 1, 1, 1, 1]  # long answer stays trained
        assert rows[1].input_ids == [1, 2, 3, 99, 88, 30]
        assert rows[1].completion_mask == [0, 0, 0, 0, 0, 1]


# The vLLM server's turns, opened by Qwen3's empty thinking block: a tool call continues the loop, an answer ends it
_TOOL_CALL = '<think>\n\n</think>\n\n<tool_call>\n{"name": "t", "arguments": {}}\n</tool_call><|im_end|>'
_FINAL = "<think>\n\n</think>\n\ndone<|im_end|>"


def t() -> str:
    # [FUNCTION DOC]
    # HELPER: Trivial tool used by the rollout-loop tests; returns "ok" (docstring provides its schema).
    # Inputs: none. Outputs: str. Exceptions: none. Side effects: none.
    """
    Return ok.
    """
    return "ok"


def _run(*, turns, max_iters=None):
    # [FUNCTION DOC]
    # HELPER: Drive _AsyncRolloutLoop._generate_one with the real tokenizer, chat template and response parser, faking
    # only vLLM generation.
    # Behavior: builds a loop with tool t and the given max_tool_calling_iterations, pre-tokenizes each scripted turn
    # text, replaces _generate_one_turn with an async fake that pops the next scripted ids and returns log-prob -0.1
    # per token, and runs _generate_one for a single "hi" user message with tool t.
    # Inputs: turns (list of str, keyword-only), max_iters (int or None). Outputs: the 6-tuple from _generate_one
    # (completion, completion ids, sequences, call count, failure count, rollout reward).
    # Exceptions: IndexError if the loop asks for more turns than scripted. Side effects: downloads a tokenizer.
    """
    Drive `_AsyncRolloutLoop._generate_one` with the real tokenizer, chat template and response parser, scripting only
    the vLLM server, which generates the text of `turns` one turn after the other.
    """
    loop = _rollout_loop(tools=[t], max_tool_calling_iterations=max_iters)
    outputs = [loop.tokenizer.encode(text, add_special_tokens=False) for text in turns]

    async def _generate_one_turn(prompt_ids):
        # [FUNCTION DOC]
        # HELPER: Async fake returning the next scripted token ids (popped from outputs) and -0.1 log-prob per token.
        # Side effects: mutates outputs.
        ids = outputs.pop(0)
        return ids, [-0.1] * len(ids)

    loop._generate_one_turn = _generate_one_turn

    # _generate_one returns (completion, completion_ids, sequences, n_calls, n_failures, rollout_reward).
    return asyncio.run(loop._generate_one([{"role": "user", "content": "hi"}], {"t": t}, [t]))


@require_response_parsing
class TestRolloutLoop(TrlTestCase):
    # [CLASS DOC]
    # Tests multi-turn generation of the rollout loop (requires response parsing support) using scripted Qwen3-style
    # turns: _TOOL_CALL (an empty think block then a call to tool t) and _FINAL (empty think block then "done").
    def test_single_turn_no_tool_call(self):
        # [FUNCTION DOC]
        # A single final turn gives one sequence ending with the completion ids, mask 1 only over the completion,
        # log-probs -0.1 there, a single assistant message and no tool calls or failures.
        completion, completion_ids, sequences, n_calls, n_failures, _ = _run(turns=[_FINAL])
        assert len(sequences) == 1
        n = len(completion_ids)
        assert sequences[0].input_ids[-n:] == completion_ids
        assert sequences[0].completion_mask == [0] * (len(sequences[0].input_ids) - n) + [1] * n
        assert sequences[0].old_log_probs[-n:] == [-0.1] * n
        assert [m["role"] for m in completion] == ["assistant"]
        assert n_calls == 0 and n_failures == 0

    def test_clean_two_turns_stay_one_row(self):
        # [FUNCTION DOC]
        # Tool call then final answer: one sequence whose mask sum equals the completion id count, messages assistant,
        # tool (content "ok"), assistant, one call and no failures.
        # The training chat template is prefix-preserving, so turn 2's re-tokenized prompt extends turn 1 -> one row
        completion, completion_ids, sequences, n_calls, n_failures, _ = _run(turns=[_TOOL_CALL, _FINAL])
        assert len(sequences) == 1
        assert sum(sequences[0].completion_mask) == len(completion_ids)  # every generated token is trained once
        assert [m["role"] for m in completion] == ["assistant", "tool", "assistant"]
        assert completion[1]["content"] == "ok"
        assert n_calls == 1 and n_failures == 0

    def test_max_tool_calling_iterations_caps_turns(self):
        # [FUNCTION DOC]
        # With max iterations 0, a tool-call turn ends the rollout: one sequence, one assistant message, zero calls.
        # max_iters=0: even though turn 1 is a tool call, the loop breaks before executing it.
        completion, _, sequences, n_calls, _, _ = _run(turns=[_TOOL_CALL], max_iters=0)
        assert len(sequences) == 1
        assert [m["role"] for m in completion] == ["assistant"]  # no tool message appended
        assert n_calls == 0


def two_reward(completions, **kwargs):
    # [FUNCTION DOC]
    # HELPER: Deterministic reward for two-conversation groups: always returns [1.0, 3.0]. Inputs ignored.
    # Deterministic per-conversation reward: two conversations get rewards 1.0 and 3.0.
    return [1.0, 3.0]


class AsyncTwoReward:
    # [CLASS DOC]
    # HELPER: Async callable class equivalent of two_reward, to test that coroutine-returning callables are awaited.
    # No state.
    # Same reward as `two_reward`, as an async callable class: the picklable form the docs recommend, and the one
    # `inspect.iscoroutinefunction` does not see as asynchronous on its own.
    async def __call__(self, completions, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Async; returns [1.0, 3.0] regardless of inputs.
        return [1.0, 3.0]


def _group(completions_sequences, completions_ids):
    # [FUNCTION DOC]
    # HELPER: Build a RolloutGroup for n conversations: "hi" user prompts, assistant completions "c<i>", given
    # completion ids and per-conversation training sequences, zero tool counts, model_version 7, group_id 0, and no
    # environment or rollout rewards. Inputs: completions_sequences, completions_ids. Outputs: RolloutGroup.
    n = len(completions_sequences)
    return RolloutGroup(
        prompts=[[{"role": "user", "content": "hi"}] for _ in range(n)],
        reward_kwargs={},
        completions=[[{"role": "assistant", "content": f"c{i}"}] for i in range(n)],
        completions_ids=completions_ids,
        completions_sequences=completions_sequences,
        tool_call_counts=[0] * n,
        tool_failure_counts=[0] * n,
        model_version=7,
        group_id=0,
        env_rewards=[None] * n,
        rollout_rewards=[None] * n,
    )


class TestScoreGroupOptionThree(TrlTestCase):
    # [CLASS DOC]
    # Tests _score_group: one reward and advantage per conversation, stamped onto every training row of that
    # conversation (including forks), with per-row metrics.
    @pytest.fixture(autouse=True)
    def _init_accelerate_state(self):
        # [FUNCTION DOC]
        # HELPER: Autouse fixture initializing accelerate's PartialState, needed by the scorer's logger.
        # _score_group logs via accelerate's logger, which requires PartialState() to be initialized
        # (the real worker does this in _child_main before running the loop).
        PartialState()

    def test_one_advantage_per_conversation_stamped_on_every_row(self):
        # [FUNCTION DOC]
        # Conversation 0 has one row, conversation 1 two forked rows; rewards 1 and 3 give advantages -1 and +1. Asserts 3
        # samples, advantages -1, +1, +1 (fork rows identical), samples map one-to-one onto source sequences, model
        # version 7 everywhere.
        # conv 0: one row; conv 1: two rows (a fork).
        seq_a = TrainingSequence([1, 2, 3, 10, 11], [0, 0, 0, 1, 1], [0, 0, 0, -0.1, -0.2], "c0")
        seq_b1 = TrainingSequence([1, 2, 3, 20, 21], [0, 0, 0, 1, 1], [0, 0, 0, -0.3, -0.4], "c1")
        seq_b2 = TrainingSequence([1, 2, 3, 20, 99, 30], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, -0.5], "c1")
        group = _group([[seq_a], [seq_b1, seq_b2]], completions_ids=[[10, 11], [20, 21, 30]])

        samples = asyncio.run(_rollout_loop(reward_funcs=[two_reward])._score_group(group))

        # 1 row for conv 0 + 2 rows for conv 1 = 3 samples.
        assert len(samples) == 3

        # rewards [1, 3] -> mean 2, std 1 -> advantages [-1, +1].
        assert samples[0].advantage == pytest.approx(-1.0)
        assert samples[1].advantage == pytest.approx(1.0)
        assert samples[2].advantage == pytest.approx(1.0)
        # The fork's two rows carry the identical (conversation-level) advantage — no split.
        assert samples[1].advantage == samples[2].advantage

        # Each sample maps 1:1 onto its source TrainingSequence.
        assert (
            samples[0].input_ids.tolist() == seq_a.input_ids
            and samples[0].completion_mask.tolist() == seq_a.completion_mask
        )
        assert samples[0].old_log_probs.tolist() == pytest.approx(seq_a.old_log_probs)
        assert samples[1].input_ids.tolist() == seq_b1.input_ids
        assert samples[2].input_ids.tolist() == seq_b2.input_ids

        assert all(s.model_version == 7 for s in samples)

    def test_async_callable_class_reward_func_is_awaited(self):
        # [FUNCTION DOC]
        # With AsyncTwoReward, per-function reward metrics are 1.0 and 3.0 and advantages -1 and +1.
        seq_a = TrainingSequence([1, 2, 10], [0, 0, 1], [0, 0, -0.1], "c0")
        seq_b = TrainingSequence([1, 2, 20], [0, 0, 1], [0, 0, -0.2], "c1")
        group = _group([[seq_a], [seq_b]], completions_ids=[[10], [20]])

        samples = asyncio.run(_rollout_loop(reward_funcs=[AsyncTwoReward()])._score_group(group))

        # Run as a synchronous function, the reward is an un-awaited coroutine and never a number.
        assert samples[0].metrics["rewards/AsyncTwoReward"] == 1.0
        assert samples[1].metrics["rewards/AsyncTwoReward"] == 3.0
        assert samples[0].advantage == pytest.approx(-1.0)
        assert samples[1].advantage == pytest.approx(1.0)

    def test_metrics_are_per_conversation_and_independent(self):
        # [FUNCTION DOC]
        # Reward, reward_std (1.0) and per-function reward metrics are per conversation, and the fork's two rows have
        # distinct metrics dict objects.
        seq_a = TrainingSequence([1, 2, 10], [0, 0, 1], [0, 0, -0.1], "c0")
        seq_b1 = TrainingSequence([1, 2, 20], [0, 0, 1], [0, 0, -0.2], "c1")
        seq_b2 = TrainingSequence([1, 2, 20, 30], [0, 0, 0, 1], [0, 0, 0, -0.3], "c1")
        group = _group([[seq_a], [seq_b1, seq_b2]], completions_ids=[[10], [20, 30]])

        samples = asyncio.run(_rollout_loop(reward_funcs=[two_reward])._score_group(group))

        assert samples[0].metrics["reward"] == 1.0
        assert samples[1].metrics["reward"] == 3.0 and samples[2].metrics["reward"] == 3.0
        assert samples[0].metrics["reward_std"] == pytest.approx(1.0)
        assert samples[1].metrics["rewards/two_reward"] == 3.0
        # The fork's two rows must not share a metrics dict (the score loop mutates it per sample).
        assert samples[1].metrics is not samples[2].metrics

    def test_all_none_reward_conversation_is_unscorable(self):
        # [FUNCTION DOC]
        # A conversation whose reward is None gets advantage 0 and NaN reward; the remaining lone scorable conversation
        # is zero-centered (advantage 0) with reward 2.0.
        # A conversation for which every reward func returns None gets advantage 0 and NaN reward.
        def maybe_none(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Reward returning [None, 2.0] regardless of inputs.
            return [None, 2.0]

        seq_a = TrainingSequence([1, 2, 10], [0, 0, 1], [0, 0, -0.1], "c0")
        seq_b = TrainingSequence([1, 2, 20], [0, 0, 1], [0, 0, -0.2], "c1")
        group = _group([[seq_a], [seq_b]], completions_ids=[[10], [20]])

        samples = asyncio.run(_rollout_loop(reward_funcs=[maybe_none])._score_group(group))

        assert len(samples) == 2
        assert samples[0].advantage == 0.0  # unscorable -> advantage 0
        assert math.isnan(samples[0].metrics["reward"])
        assert samples[1].advantage == 0.0  # only one scorable row -> zero-centered
        assert samples[1].metrics["reward"] == 2.0


@pytest.mark.skipif(
    not is_ampere_or_newer() and torch_device != "xpu" and torch.version.hip is None,
    reason="Flash Attention requires Ampere or newer GPU, XPU, or ROCm",
)
class TestEpochStop(TrlTestCase):
    # [CLASS DOC]
    # End-to-end tests (GPU-gated) that num_train_epochs counts distinct prompt groups, independent of how many rows
    # each conversation forks into.
    """`num_train_epochs` stops after N full passes over the PROMPTS, counted as distinct group_ids.

    The point of the fix is that this is fork-independent: a conversation that message-mode forks into several training
    rows still counts as one prompt-group, so the number of epochs does not depend on the fork rate (only the number of
    optimizer steps does). Driven end-to-end through the real trainer (stub worker + real forward/backward), which is
    what actually regresses if the wiring breaks.
    """

    def _train(self, fork_k, num_train_epochs=2):
        # [FUNCTION DOC]
        # HELPER: Train small Qwen2.5 with num_train_epochs (default 2), batch 3, 3 generations, completion length 8,
        # fixed-count batching and a stub worker emitting fork_k copies per generation.
        # Inputs: fork_k (int), num_train_epochs (int). Outputs: tuple of (trained trainer, dataset size).
        model_id = "trl-internal-testing/small-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")
        args = AsyncGRPOConfig(
            output_dir=self.tmp_dir,
            num_train_epochs=num_train_epochs,  # epoch-driven: no explicit max_steps
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            token_budget=-1,  # FixedCountBatcher: deterministic samples/step, so groups accrue predictably
            vllm_server_timeout=5.0,
            report_to="none",
        )
        worker = _StubRolloutWorker(AutoTokenizer.from_pretrained(model_id), dataset, num_generations=3, fork_k=fork_k)
        trainer = AsyncGRPOTrainer(
            model=model_id,
            reward_funcs=dummy_reward_func,
            args=args,
            train_dataset=dataset,
            rollout_worker=worker,
            weight_transfer=_StubWeightTransfer(),
        )
        trainer.train()
        return trainer, len(dataset)

    def test_epoch_stop_is_fork_independent(self):
        # [FUNCTION DOC]
        # Runs with fork_k 1 and 3: both train between 2x and 3x (exclusive) the number of prompts in distinct groups, and
        # the forked run takes strictly more optimizer steps.
        no_fork, num_prompts = self._train(fork_k=1)
        forked, _ = self._train(fork_k=3)

        # Both stop after num_train_epochs=2 full passes over the prompts, i.e. ~2 x num_prompts distinct
        # groups, plus at most one micro-batch of overshoot — regardless of the fork rate. (HF's own
        # state.epoch is meaningless here: it's global_step/max_steps over an infinite IterableDataset,
        # so we judge epochs by distinct prompt-groups trained, which is what the callback targets.)
        for trainer in (no_fork, forked):
            assert 2 * num_prompts <= len(trainer._trained_groups) < 3 * num_prompts

        # Forks add rows, hence fixed-size optimizer steps: 3x rows per conversation must take strictly more
        # steps for the same 2 epochs. If forks leaked into the epoch count, the forked run would instead
        # stop in FEWER prompt-passes (the pre-fix bug).
        assert forked.state.global_step > no_fork.state.global_step

    def test_epoch_stop_saves_the_final_step(self):
        # [FUNCTION DOC]
        # After an epoch-driven stop, the output dir contains exactly one checkpoint, at the final global step.
        trainer, _ = self._train(fork_k=1)

        # The run stops long before the default `save_steps=500`, so this checkpoint comes from the epoch stop alone.
        assert os.listdir(self.tmp_dir) == [f"checkpoint-{trainer.state.global_step}"]


@require_peft
class TestValidateLoraForVLLMSync(TrlTestCase):
    # [CLASS DOC]
    # Tests validate_lora_for_vllm_sync on real PEFT models built over tiny Qwen2.5 (requires peft).
    model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

    def _peft_model(self, peft_config):
        # [FUNCTION DOC]
        # HELPER: Load the tiny model and wrap it with get_peft_model using the given config. Outputs: PeftModel.
        from peft import get_peft_model

        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        return get_peft_model(model, peft_config)

    def _validate(self, **lora_kwargs):
        # [FUNCTION DOC]
        # HELPER: Build a LoraConfig (default target modules q_proj and v_proj) from kwargs, wrap the model, and return the
        # validator's result (the adapter config). Exceptions: ValueError from the validator.
        from peft import LoraConfig

        lora_kwargs.setdefault("target_modules", ["q_proj", "v_proj"])
        return validate_lora_for_vllm_sync(self._peft_model(LoraConfig(**lora_kwargs)))

    def test_plain_lora_is_accepted(self):
        # [FUNCTION DOC]
        # A rank-16 plain LoRA validates and the returned config has r 16.
        config = self._validate(r=16)
        assert config.r == 16

    def test_several_active_adapters_rejected(self):
        # [FUNCTION DOC]
        # Two simultaneously active adapters raise ValueError "single adapter".
        from peft import LoraConfig

        model = self._peft_model(LoraConfig(target_modules=["q_proj"]))
        model.add_adapter("ref", LoraConfig(target_modules=["q_proj"]))
        model.base_model.set_adapter(["default", "ref"])
        with pytest.raises(ValueError, match="single adapter"):
            validate_lora_for_vllm_sync(model)

    def test_non_lora_adapter_rejected(self):
        # [FUNCTION DOC]
        # A prompt-tuning adapter raises ValueError "LoRA adapters only" (before any LoRA-only attribute is read).
        # Guards the order of the checks: every LoRA-only field is read after this one, so a prompt-tuning config
        # must be rejected here rather than blowing up on a missing attribute.
        from peft import PromptTuningConfig, TaskType

        model = self._peft_model(PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=4))
        with pytest.raises(ValueError, match="LoRA adapters only"):
            validate_lora_for_vllm_sync(model)

    # One case per thing a LoRA adapter cannot carry to vLLM. All of them fall back to merged sync.
    @pytest.mark.parametrize(
        ("lora_kwargs", "match"),
        [
            ({"modules_to_save": ["lm_head"]}, "modules_to_save"),
            ({"use_dora": True}, "DoRA"),
            ({"bias": "all"}, "bias"),
            ({"target_modules": None, "target_parameters": ["mlp.gate_proj.weight"]}, "is_3d_lora_weight"),
            ({"target_modules": ["q_proj", "lm_head"]}, "head or the embeddings"),
            ({"target_modules": ["q_proj", "embed_tokens"]}, "head or the embeddings"),
        ],
    )
    def test_unservable_adapter_rejected(self, lora_kwargs, match):
        # [FUNCTION DOC]
        # Parametrized: modules_to_save, DoRA, bias, 3D target_parameters, and targeting lm_head or embed_tokens each raise
        # ValueError with the matching message.
        with pytest.raises(ValueError, match=match):
            self._validate(**lora_kwargs)

    def test_regex_target_modules_do_not_crash(self):
        # [FUNCTION DOC]
        # A regex string target_modules validates and is returned unchanged.
        # `target_modules` may be a regex string rather than a collection; iterating it character by character would
        # both miss the check and never raise.
        config = self._validate(target_modules=".*(q_proj|v_proj)$")
        assert config.target_modules == ".*(q_proj|v_proj)$"


class TestRoundLoraRank(TrlTestCase):
    # [CLASS DOC]
    # Tests round_lora_rank (round up to the next vLLM-supported LoRA rank).
    @pytest.mark.parametrize(("adapter_rank", "expected"), [(1, 1), (4, 8), (8, 8), (16, 16), (33, 64), (512, 512)])
    def test_rounds_up_to_a_supported_rank(self, adapter_rank, expected):
        # [FUNCTION DOC]
        # Parametrized: 1 to 1, 4 to 8, 8 to 8, 16 to 16, 33 to 64, 512 to 512.
        assert round_lora_rank(adapter_rank) == expected

    def test_above_the_largest_supported_rank_raises(self):
        # [FUNCTION DOC]
        # Rank 513 raises ValueError mentioning 512.
        # A bare `StopIteration` here would surface as an unrelated RuntimeError deep in the trainer.
        with pytest.raises(ValueError, match="512"):
            round_lora_rank(513)


@require_peft
class TestSaveLoraAdapter(TrlTestCase):
    # [CLASS DOC]
    # Tests save_lora_adapter writes a flat, PEFT-loadable adapter directory and can overwrite an existing one
    # (requires peft).
    def _model_and_env(self, is_main_process=True):
        # [FUNCTION DOC]
        # HELPER: Build a CPU LoRA model over tiny Qwen2.5 (q_proj, v_proj) and a lightweight accelerator stand-in with the
        # test device and the given is_main_process flag (default True). Outputs: tuple of model and accelerator.
        from peft import LoraConfig, get_peft_model

        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        model = get_peft_model(model, LoraConfig(target_modules=["q_proj", "v_proj"]))
        # The model stays on CPU while the accelerator reports the runner's device, which is the `fsdp_offload_params`
        # shape: `save_lora_adapter` has to move each shard to the device before gathering and bring the result back
        # before writing. On a GPU runner that is a real round trip; hardcoding `cpu` here made it a no-op, so a
        # tensor left on the wrong device could never have failed this test.
        accelerator = SimpleNamespace(device=torch.device(torch_device), is_main_process=is_main_process)
        return model, accelerator

    def test_writes_a_flat_loadable_adapter_directory(self):
        # [FUNCTION DOC]
        # Saves to trl-policy-v1: adapter_config.json and adapter_model.safetensors are at the root; reloading with
        # PeftModel gives non-empty LoRA parameters equal to the originals.
        model, accelerator = self._model_and_env()
        dest = os.path.join(self.tmp_dir, "trl-policy-v1")

        save_lora_adapter(model, accelerator, "default", dest)

        # vLLM reads `adapter_config.json` at the root of the path it is handed.
        assert os.path.isfile(os.path.join(dest, "adapter_config.json"))
        assert os.path.isfile(os.path.join(dest, "adapter_model.safetensors"))

        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        reloaded = PeftModel.from_pretrained(base, dest)
        saved = {n: p for n, p in reloaded.named_parameters() if "lora_" in n}
        assert saved  # the adapter is not empty
        for name, param in saved.items():
            torch.testing.assert_close(param, model.get_parameter(name))

    def test_republishing_over_an_existing_directory_does_not_collide(self):
        # [FUNCTION DOC]
        # Saves, renames the config to stale.json, saves again to the same directory: the new config exists and the stale
        # file is gone (destination cleared before publishing).
        # The version counter restarts with the process, so a second run sharing an output directory republishes
        # `v1` over the first run's. `os.rename` onto a non-empty directory raises, so the destination is cleared.
        model, accelerator = self._model_and_env()
        dest = os.path.join(self.tmp_dir, "trl-policy-v1")
        save_lora_adapter(model, accelerator, "default", dest)
        os.rename(os.path.join(dest, "adapter_config.json"), os.path.join(dest, "stale.json"))

        save_lora_adapter(model, accelerator, "default", dest)

        assert os.path.isfile(os.path.join(dest, "adapter_config.json"))
        assert not os.path.exists(os.path.join(dest, "stale.json"))

    def test_the_published_directory_holds_only_servable_adapters(self):
        # [FUNCTION DOC]
        # After saving into a namespace directory, it contains only trl-policy-v1 (no staging leftovers) and its
        # adapter_config.json has peft_type LORA.
        # The parent of `dest_dir` is an adapter namespace: vLLM's filesystem resolver serves `<dir>/<adapter_name>`
        # by name. Staging must not leave a half-written sibling in it that looks like an adapter.
        model, accelerator = self._model_and_env()
        namespace = os.path.join(self.tmp_dir, "vllm_lora")
        save_lora_adapter(model, accelerator, "default", os.path.join(namespace, "trl-policy-v1"))

        entries = os.listdir(namespace)
        assert entries == ["trl-policy-v1"]
        # vLLM keys on the directory basename and reads `adapter_config.json` from its root.
        config = json.loads(open(os.path.join(namespace, "trl-policy-v1", "adapter_config.json")).read())
        assert config["peft_type"] == "LORA"

    def test_non_main_processes_write_nothing(self):
        # [FUNCTION DOC]
        # With is_main_process False, saving (which still gathers) creates no destination directory.
        # Non-writing ranks still materialize the adapter: that all-gather is a collective.
        model, accelerator = self._model_and_env(is_main_process=False)
        dest = os.path.join(self.tmp_dir, "trl-policy-v1")

        save_lora_adapter(model, accelerator, "default", dest)

        assert not os.path.exists(dest)


class TestRolloutRequestModel(TrlTestCase):
    # [CLASS DOC]
    # Tests the model name the rollout loop requests from vLLM (_request_model).
    def _loop(self, lora_name):
        # [FUNCTION DOC]
        # HELPER: Build a rollout loop for model "Qwen/Qwen3-4B" with the given lora_name. Outputs: _AsyncRolloutLoop.
        return _rollout_loop(model_name="Qwen/Qwen3-4B", lora_name=lora_name)

    def test_dense_runs_request_the_base_model(self):
        # [FUNCTION DOC]
        # Without a LoRA name the request model is the base model name.
        assert self._loop(None)._request_model == "Qwen/Qwen3-4B"

    def test_adapter_runs_request_the_versioned_adapter(self):
        # [FUNCTION DOC]
        # With lora_name "trl-policy" the request model is "trl-policy-v0", and becomes "trl-policy-v3" after setting the
        # shared model version value to 3.
        # In vLLM's API an adapter *is* a model name: naming the base model would silently serve the base model.
        loop = self._loop("trl-policy")
        assert loop._request_model == "trl-policy-v0"
        loop._model_version_value.value = 3
        assert loop._request_model == "trl-policy-v3"


# A LoRA-capable server, as reported by `/server_info`. `max_loras` covers the `max_staleness + 1` servable policy
# versions plus the one being loaded, for the tests' default `max_staleness=1`.
SERVER_LORA_CONFIG = {"max_lora_rank": 32, "max_loras": 3}


def _server_info(lora_config, data_parallel_size=1):
    # [FUNCTION DOC]
    # HELPER: Build the subset of a /server_info payload read by select_adapter_sync: lora_config and a parallel
    # config with data_parallel_size (default 1). Outputs: dict.
    """The two fields of `/server_info` that `select_adapter_sync` reads, shaped as the server returns them."""
    return {"lora_config": lora_config, "parallel_config": {"data_parallel_size": data_parallel_size}}


@require_peft
class TestSelectAdapterSync(TrlTestCase):
    # [CLASS DOC]
    # Tests select_adapter_sync as a pure function of a server_info dict, a real PeftModel over tiny Qwen2.5 on CPU and
    # an AsyncGRPOConfig (requires peft). Covers mode choice, warnings, capacity errors, and two strict-xfail proposed
    # behavior changes.
    """The sync-mode decision is a pure function of the server's report and the model, so it is tested as one.

    Its inputs here are the real types — a `/server_info` dict and a `PeftModel` — not doubles for them. No GPU: the
    tiny model loads on CPU and nothing is trained.
    """

    model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

    def _peft_model(self, **lora_kwargs):
        # [FUNCTION DOC]
        # HELPER: Wrap the tiny model with a LoraConfig (defaults q_proj/v_proj, rank 8) built from kwargs. Outputs:
        # PeftModel.
        from peft import LoraConfig, get_peft_model

        lora_kwargs.setdefault("target_modules", ["q_proj", "v_proj"])
        lora_kwargs.setdefault("r", 8)
        return get_peft_model(AutoModelForCausalLM.from_pretrained(self.model_id), LoraConfig(**lora_kwargs))

    def _args(self, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Build an AsyncGRPOConfig in tmp_dir with reporting off and max_staleness defaulting to 1.
        kwargs.setdefault("max_staleness", 1)
        return AsyncGRPOConfig(output_dir=self.tmp_dir, report_to="none", **kwargs)

    def test_lora_server_selects_adapter_sync(self):
        # [FUNCTION DOC]
        # A LoRA-capable server with a plain adapter selects adapter sync (True).
        assert select_adapter_sync(_server_info(SERVER_LORA_CONFIG), self._peft_model(), self._args()) is True

    def test_plain_server_selects_merged_sync_with_a_warning(self, caplog):
        # [FUNCTION DOC]
        # A server without LoRA selects merged sync (False) and logs a warning mentioning --enable-lora.
        with caplog.at_level("WARNING"):
            assert select_adapter_sync(_server_info(None), self._peft_model(), self._args()) is False
        assert "--enable-lora" in caplog.text

    def test_unservable_adapter_falls_back_with_a_warning(self, caplog):
        # [FUNCTION DOC]
        # An adapter with modules_to_save falls back to merged sync with a warning mentioning modules_to_save.
        with caplog.at_level("WARNING"):
            decision = select_adapter_sync(
                _server_info(SERVER_LORA_CONFIG), self._peft_model(modules_to_save=["lm_head"]), self._args()
            )
        assert decision is False
        assert "modules_to_save" in caplog.text

    def test_data_parallel_server_falls_back_with_a_warning(self, caplog):
        # [FUNCTION DOC]
        # A data-parallel size 2 server falls back to merged sync with a warning mentioning --data-parallel-size.
        with caplog.at_level("WARNING"):
            decision = select_adapter_sync(
                _server_info(SERVER_LORA_CONFIG, data_parallel_size=2), self._peft_model(), self._args()
            )
        assert decision is False
        assert "--data-parallel-size" in caplog.text

    def test_rank_above_the_servers_capacity_raises(self):
        # [FUNCTION DOC]
        # Adapter rank 64 against max_lora_rank 32 raises ValueError suggesting --max-lora-rank 64.
        with pytest.raises(ValueError, match="--max-lora-rank 64"):
            select_adapter_sync(_server_info(SERVER_LORA_CONFIG), self._peft_model(r=64), self._args())

    def test_rank_pattern_counts_toward_the_rank_bound(self):
        # [FUNCTION DOC]
        # A rank_pattern lifting q_proj to 64 also raises ValueError suggesting --max-lora-rank 64.
        # `rank_pattern` can lift one module above `r`; the server has to hold the largest, not the base.
        with pytest.raises(ValueError, match="--max-lora-rank 64"):
            select_adapter_sync(
                _server_info(SERVER_LORA_CONFIG), self._peft_model(r=8, rank_pattern={"q_proj": 64}), self._args()
            )

    def test_too_few_adapter_slots_for_max_staleness_raises(self):
        # [FUNCTION DOC]
        # max_loras 2 with max_staleness 1 raises ValueError suggesting --max-loras 3 (max_staleness + 2).
        # `max_staleness + 1` versions stay servable and the next one is loaded before the oldest is unloaded, so the
        # server has to hold `max_staleness + 2`. One fewer and vLLM evicts a servable policy silently, every sync.
        with pytest.raises(ValueError, match="--max-loras 3"):
            select_adapter_sync(
                _server_info({"max_lora_rank": 32, "max_loras": 2}), self._peft_model(), self._args(max_staleness=1)
            )

    # --- behaviour changes proposed after the DP=2 runs; strict xfail so flipping the code flips these ------------

    @pytest.mark.xfail(
        strict=True,
        reason="Proposed: vLLM's internal DP load balancer fans `add_lora` out to every replica (DPLBAsyncMPClient), "
        "and the merged path this falls back to cannot work on an `--enable-lora` server. Pending decision.",
    )
    def test_data_parallel_server_should_use_adapter_sync(self):
        # [FUNCTION DOC]
        # Strict xfail documenting a proposed change: a data-parallel LoRA server should select adapter sync.
        assert (
            select_adapter_sync(
                _server_info(SERVER_LORA_CONFIG, data_parallel_size=2), self._peft_model(), self._args()
            )
            is True
        )

    @pytest.mark.xfail(
        strict=True,
        reason="Proposed: merged sync cannot run against an `--enable-lora` server (the LoRA wrappers move base "
        "weights to `base_layer.weight`, which the NCCL receive path cannot resolve), so an adapter the server "
        "cannot serve should raise here rather than fall back into a guaranteed crash. Pending decision.",
    )
    def test_unservable_adapter_on_a_lora_server_should_raise(self):
        # [FUNCTION DOC]
        # Strict xfail documenting a proposed change: an unservable adapter on a LoRA server should raise instead of
        # falling back.
        with pytest.raises(ValueError, match="modules_to_save"):
            select_adapter_sync(
                _server_info(SERVER_LORA_CONFIG), self._peft_model(modules_to_save=["lm_head"]), self._args()
            )


@pytest.mark.skipif(
    not is_ampere_or_newer() and torch_device != "xpu" and torch.version.hip is None,
    reason="Flash Attention requires Ampere or newer GPU, XPU, or ROCm",
)
@require_peft
class TestAsyncGRPOTrainerPeft(TrlTestCase):
    # [CLASS DOC]
    # PEFT/LoRA tests of the trainer against FakeVLLMServer through the real VLLMClient (GPU-gated, requires peft).
    # Covers LoRA training, head-targeting refusal, adapter sync mode selection, adapter publishing and LRU window
    # management, the merged-weights fallback and cleanup on errors. Module constant SERVER_LORA_CONFIG reports
    # max_lora_rank 32 and max_loras 3. Class attribute model_id is small Qwen2.5.
    """The trainer against a `FakeVLLMServer`: its real `VLLMClient` sends real requests, the wire is faked.

    Nothing here is patched. The trainer probes, loads, pauses, resumes and unloads through `VLLMClient` exactly as it
    does in production, and the assertions read what the server saw — `fake.loaded`, `fake.evicted`, `fake.paused`.
    """

    model_id = "trl-internal-testing/small-Qwen2ForCausalLM-2.5"

    def _lora_config(self, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Build a peft LoraConfig with defaults target_modules q_proj and v_proj and rank 8, overridable by kwargs.
        # Outputs: LoraConfig. Side effects: imports peft.
        from peft import LoraConfig

        kwargs.setdefault("target_modules", ["q_proj", "v_proj"])
        kwargs.setdefault("r", 8)
        return LoraConfig(**kwargs)

    def _build(
        self,
        fake: FakeVLLMServer,
        peft_config,
        lora_config=None,
        data_parallel_size=1,
        weight_transfer=None,
        rollout_worker="stub",
        **config_kwargs,
    ):
        # [FUNCTION DOC]
        # HELPER: Configure the fake server (lora_config, data_parallel_size) and build a PEFT trainer against it.
        # Behavior: loads the zen dataset and tokenizer; defaults max_staleness to 1; config uses lr 0.1, batch 3, 3
        # generations, completion length 8, token budget 64, vllm_server_base_url pointing at the fake, 5 s timeout. The
        # rollout worker is a fresh stub (3 generations) when "stub" is passed, otherwise passed through (None makes the trainer build a
        # real AsyncRolloutWorker). weight_transfer None lets the trainer own weight sync and probe the server.
        # Inputs: fake, peft_config, lora_config, data_parallel_size, weight_transfer, rollout_worker, config overrides.
        # Outputs: AsyncGRPOTrainer. Exceptions: propagates trainer construction errors.
        # Side effects: mutates the fake server's configuration.
        """Build a trainer against `fake`, configured to look like a server started with the given flags.

        With `weight_transfer=None` the trainer owns weight sync, which is what makes it probe the server and pick a
        mode. Pass a `weight_transfer` to exercise the send itself — an injected backend deliberately keeps adapter
        sync off. `rollout_worker="stub"` injects `_StubRolloutWorker`; `None` lets the trainer build the real
        `AsyncRolloutWorker`, which is cheap until `start()` and useful for reading what it was told.
        """
        fake.lora_config = lora_config
        fake.data_parallel_size = data_parallel_size
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        config_kwargs.setdefault("max_staleness", 1)
        args = AsyncGRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            token_budget=64,  # set explicitly so the batcher does not size itself from the server's max_model_len
            vllm_server_base_url=fake.url,
            vllm_server_timeout=5.0,
            report_to="none",
            **config_kwargs,
        )
        if rollout_worker == "stub":
            rollout_worker = _StubRolloutWorker(tokenizer, dataset, num_generations=3)
        return AsyncGRPOTrainer(
            model=self.model_id,
            reward_funcs=dummy_reward_func,
            args=args,
            train_dataset=dataset,
            peft_config=peft_config,
            rollout_worker=rollout_worker,
            weight_transfer=weight_transfer,
        )

    def test_train_peft_config(self, fake_vllm):
        # [FUNCTION DOC]
        # Trains with LoRA and a stub transfer: train_loss is logged, every LoRA parameter changed, and base parameters
        # stayed within an absolute tolerance of 1e-5 (merge/unmerge drift only).
        trainer = self._build(fake_vllm, self._lora_config(), weight_transfer=_StubWeightTransfer())
        previous_params = {n: p.clone() for n, p in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        for name, param in previous_params.items():
            new_param = trainer.model.get_parameter(name)
            if "lora_" in name:
                assert not torch.equal(param, new_param), f"Adapter parameter {name} has not changed."
            else:
                # Base weights are frozen, but every sync merges the adapter in and unmerges it out, which is not
                # bit-exact: the targeted `base_layer` weights drift by a float ulp. A gradient step at this
                # learning rate would move them by orders of magnitude more.
                torch.testing.assert_close(
                    param, new_param, rtol=0, atol=1e-5, msg=f"Base parameter {name} has changed."
                )

    @pytest.mark.parametrize("target_modules", [["q_proj", "lm_head"], r".*_head$"])
    def test_lora_on_the_head_is_refused(self, fake_vllm, target_modules):
        # [FUNCTION DOC]
        # Parametrized over a target list including lm_head and a regex matching names ending in _head: construction
        # raises ValueError mentioning lm_head.
        # The chunked log-probability path reads `lm_head.weight`, which on a PEFT-wrapped head is the base weight:
        # the adapter delta would be silently dropped from every trainer logprob. The regex case is why the check is
        # on the module and not on `target_modules` — that string never contains "lm_head".
        with pytest.raises(ValueError, match="lm_head"):
            self._build(fake_vllm, self._lora_config(target_modules=target_modules), lora_config=SERVER_LORA_CONFIG)

    def test_an_injected_weight_transfer_keeps_adapter_sync_off(self, fake_vllm):
        # [FUNCTION DOC]
        # With an injected weight transfer on a LoRA-capable server, _lora_sync is False and the fake server received no
        # requests.
        # An injected backend owns weight sync and may be a no-op that disables it; adapter sync must not run behind
        # its back, and the server is not consulted.
        trainer = self._build(
            fake_vllm, self._lora_config(), lora_config=SERVER_LORA_CONFIG, weight_transfer=_StubWeightTransfer()
        )
        assert trainer._lora_sync is False
        assert fake_vllm.requests == []

    def test_a_misconfigured_server_raises_at_init(self, fake_vllm):
        # [FUNCTION DOC]
        # LoRA rank 16 against a server with max_lora_rank 8 raises ValueError mentioning --max-lora-rank at init.
        # The probe runs on rank 0 only, so its error is broadcast and raised on every rank: raising on rank 0 alone
        # would leave the other ranks waiting in the collective that shares the sync mode.
        with pytest.raises(ValueError, match="--max-lora-rank"):
            self._build(fake_vllm, self._lora_config(r=16), lora_config={"max_lora_rank": 8, "max_loras": 3})

    @require_vllm  # `AsyncRolloutWorker.__init__` refuses to build without vLLM installed
    @pytest.mark.parametrize(("lora_config", "expected"), [(SERVER_LORA_CONFIG, "trl-policy"), (None, None)])
    def test_the_rollout_worker_is_told_the_adapter_name(self, fake_vllm, lora_config, expected):
        # [FUNCTION DOC]
        # Requires vLLM. Parametrized: with a LoRA-capable server the real worker's lora_name kwarg is "trl-policy";
        # without LoRA support it is None.
        # In vLLM's API an adapter *is* a model name: a rollout request that does not name it is served by the base
        # model, silently, with plausible rewards. The real worker is built here (it spawns nothing until `start()`)
        # to read the `lora_name` the trainer hands it.
        trainer = self._build(fake_vllm, self._lora_config(), lora_config=lora_config, rollout_worker=None)
        assert trainer.rollout_worker._loop_kwargs["lora_name"] == expected

    # --- adapter sync, as the server sees it -----------------------------------------------------------------------

    def test_adapter_sync_loads_the_adapter_before_bumping_the_version(self, fake_vllm):
        # [FUNCTION DOC]
        # Records model_version at load time via the fake's on_load hook and calls _sync_weight once. Asserts the version
        # was 0 at load and is 1 after (also on the stub worker), the only loaded adapter is trl-policy-v1 whose path is
        # absolute and contains adapter_config.json, and the server is not left paused.
        trainer = self._build(fake_vllm, self._lora_config(), lora_config=SERVER_LORA_CONFIG)
        version_at_load = []
        fake_vllm.on_load = lambda name, path: version_at_load.append(trainer.model_version)

        trainer._sync_weight()

        # The rollout worker derives the adapter it requests from `model_version`, so the version must move only
        # once the adapter it names exists on the server.
        assert version_at_load == [0]
        assert trainer.model_version == 1
        assert trainer.rollout_worker._model_version == 1
        assert list(fake_vllm.loaded) == ["trl-policy-v1"]
        path = fake_vllm.loaded["trl-policy-v1"]
        assert os.path.isfile(os.path.join(path, "adapter_config.json"))
        # The server resolves the path in its own process, with its own cwd, possibly on another host. vLLM reads a
        # path it cannot resolve as a Hub repo id, so a relative one dies inside the engine.
        assert os.path.isabs(path)
        assert fake_vllm.paused is False  # paused for the publish, resumed after

    @pytest.mark.parametrize(
        ("max_staleness", "max_loras", "servable_after_five"),
        [(1, 3, ["trl-policy-v4", "trl-policy-v5"]), (4, 6, [f"trl-policy-v{v}" for v in range(1, 6)])],
    )
    def test_adapter_sync_keeps_max_staleness_plus_one_versions_servable(
        self, fake_vllm, max_staleness, max_loras, servable_after_five
    ):
        # [FUNCTION DOC]
        # Parametrized over (max_staleness, max_loras, expected loaded names): after 5 syncs, the server holds exactly the
        # last max_staleness + 1 adapter versions, nothing was LRU-evicted, and on disk the adapter directories for
        # versions greater than 5 minus (max_staleness + 1) minus 1 still exist while older ones were deleted.
        # Regression: an early version evicted `v{N-2}` regardless of `max_staleness`, deleting an adapter that
        # in-flight requests could still name; vLLM resolves `lora_path` lazily inside the engine, so the missing
        # directory killed the engine with an `HFValidationError` rather than 404ing. `N-2` happens to equal
        # `N-(max_staleness+1)` at `max_staleness=1`, which is why the second row exists.
        trainer = self._build(
            fake_vllm,
            self._lora_config(),
            lora_config={"max_lora_rank": 32, "max_loras": max_loras},
            max_staleness=max_staleness,
        )

        for _ in range(5):
            trainer._sync_weight()

        assert list(fake_vllm.loaded) == servable_after_five
        # The server never had to evict one itself: the `--max-loras` bound left room for the load-then-unload.
        assert fake_vllm.evicted == []
        # The files lag the unload by one sync, so one more directory than the servable set survives on disk.
        newest_deleted = 5 - (max_staleness + 1) - 1
        for version in range(1, 6):
            exists = os.path.isdir(os.path.join(trainer._lora_dir, f"trl-policy-v{version}"))
            assert exists is (version > newest_deleted), f"trl-policy-v{version}"

    # --- merged fallback ---------------------------------------------------------------------------------------

    def test_merged_sync_sends_base_names_only(self, fake_vllm):
        # [FUNCTION DOC]
        # Without server LoRA support (merged fallback) and with lm_head in modules_to_save, records the names sent by
        # a recording transfer: non-empty, none contains PEFT/wrapper markers (lora_, base_model.model., .base_layer,
        # original_module, modules_to_save, _checkpoint_wrapped_module.), and names are unique.
        sent = []

        class _RecordingWeightTransfer(_StubWeightTransfer):
            # [CLASS DOC]
            # HELPER: _StubWeightTransfer subclass whose send_weights records parameter names into the enclosing sent list.
            def send_weights(self, iterator):
                # [FUNCTION DOC]
                # HELPER: Consume the (name, tensor) iterator, appending each name to sent. Side effects: mutates sent.
                sent.extend(name for name, _ in iterator)

        trainer = self._build(
            fake_vllm, self._lora_config(modules_to_save=["lm_head"]), weight_transfer=_RecordingWeightTransfer()
        )
        trainer._sync_weight()

        assert sent
        forbidden = (
            "lora_",
            "base_model.model.",
            ".base_layer",
            "original_module",
            "modules_to_save",
            "_checkpoint_wrapped_module.",
        )
        for name in sent:
            assert not any(token in name for token in forbidden), name
        # `modules_to_save` keeps a trainable copy and the frozen original side by side. Only the copy is sent, under
        # the base checkpoint's name — sending both would repeat that name and desync the packed transfer.
        assert len(sent) == len(set(sent))

    @require_vllm  # the merged path builds the real `WeightTransferClient`, whose constructor requires vLLM
    def test_manifest_and_streaming_iter_agree_element_for_element(self, fake_vllm):
        # [FUNCTION DOC]
        # Requires vLLM. With the real WeightTransferClient (no network init), the manifest names equal the names from
        # _streaming_iter and from _iter_vllm_named_params, and names, shapes and dtype names have equal lengths.
        # The server sizes its receive buffers from the manifest, so a name present in one and not the other
        # desyncs the transfer. The real `WeightTransferClient` is built here; it only touches the network on
        # `init_weight_transfer`, which is not called.
        trainer = self._build(fake_vllm, self._lora_config(modules_to_save=["lm_head"]), lora_config=None)
        manifest = trainer.weight_transfer._weight_update_info

        assert manifest["names"] == [name for name, _ in trainer._streaming_iter()]
        assert len(manifest["names"]) == len(manifest["shapes"]) == len(manifest["dtype_names"])
        assert manifest["names"] == [name for name, _ in _iter_vllm_named_params(trainer.model)]

    def test_adapters_are_unmerged_when_the_send_raises(self, fake_vllm):
        # [FUNCTION DOC]
        # With a transfer that records whether any adapter is merged and then raises RuntimeError, _sync_weight
        # propagates the error; adapters were merged during the send and are unmerged afterwards.
        merged_during_send = []

        class _RaisingWeightTransfer(_StubWeightTransfer):
            # [CLASS DOC]
            # HELPER: _StubWeightTransfer subclass whose send_weights records the merged state and raises.
            def send_weights(self, iterator):
                # [FUNCTION DOC]
                # HELPER: Append whether any adapter of the enclosing trainer's model is merged, then raise RuntimeError
                # "transfer blew up". Inputs: iterator (not consumed). Side effects: mutates merged_during_send.
                merged_during_send.append(_any_adapter_merged(trainer.model))
                raise RuntimeError("transfer blew up")

        trainer = self._build(fake_vllm, self._lora_config(), weight_transfer=_RaisingWeightTransfer())
        with pytest.raises(RuntimeError, match="transfer blew up"):
            trainer._sync_weight()

        # Leaving the adapter merged would silently train merged weights from the next step on.
        assert merged_during_send == [True]
        assert not _any_adapter_merged(trainer.model)


def _any_adapter_merged(model) -> bool:
    # [FUNCTION DOC]
    # HELPER: Return True if any module of model is a PEFT BaseTunerLayer whose merged flag is set.
    # Inputs: model (nn.Module). Outputs: bool. Exceptions: ImportError without peft. Side effects: none.
    from peft.tuners.tuners_utils import BaseTunerLayer

    return any(isinstance(module, BaseTunerLayer) and module.merged for module in model.modules())
