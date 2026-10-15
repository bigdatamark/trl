# [MODULE DOC]
# Tests for the experimental AsyncDistillationTrainer and its async rollout worker: packing-aware batching and
# collation, metric reduction, teacher log-prob parsing and multi-teacher routing, the narrow top-1/actual-token
# support used for beta > 0, the sparse generalized JSD loss, per-teacher stats, end-to-end training with stub
# rollout/weight-transfer components, epoch stopping, rollout-state checkpointing and PEFT adapter syncing
# against an in-process fake vLLM server.
# Contents: synthetic data helpers (_make_teacher_topk, _rollout_sample, _position, _rollout_loop,
# _scoring_loop, _checkpoint_trainer, _any_adapter_merged), stubs (_StubRolloutWorker, _StubWeightTransfer),
# the FakeVLLMServer / FakeVLLMTransport pair with the fake_vllm fixture, and the Test* classes.
# Constants: TEACHER_TOP_K is 4 (teacher candidates per position; rows carry one extra slot for the actual token).
# GPU-only tests use the small-* Qwen3 model since the trainer loads with Flash Attention (head size multiple of 8).
# Side effects at import: none beyond imports (imports vLLM-free trainer internals).

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
from collections import OrderedDict, defaultdict
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
import requests
import torch
from datasets import Dataset, load_dataset
from requests.adapters import BaseAdapter
from transformers import AutoTokenizer, TrainerControl
from transformers.testing_utils import torch_device

from trl.experimental.async_distillation import AsyncDistillationConfig, AsyncDistillationTrainer
from trl.experimental.async_distillation.async_distillation_trainer import (
    DataCollatorForRollout,
    FixedCountBatcher,
    RolloutWorkerProtocol,
    TokenBudgetBatcher,
    _add_tail_bucket,
    _balance_by_squared_length,
    _chunked_jsd_loss,
    _EpochStopCallback,
    _iter_vllm_named_params,
    _jsd_divergence,
    _narrow_top1_actual_support,
    _reduce_metric,
)
from trl.experimental.async_distillation.async_rollout_worker import (
    AsyncRolloutWorker,
    RolloutSample,
    _AsyncRolloutLoop,
    _parse_teacher_logprobs_at_position,
)
from trl.experimental.server_distillation.server_distillation_trainer import (
    _jsd_divergence as _reference_jsd_divergence,
)
from trl.trainer.base_trainer import _BaseTrainer

from ..testing_utils import TrlTestCase, is_ampere_or_newer, require_peft, require_vllm


TEACHER_TOP_K = 4

# The trainer loads the model with Flash Attention, which requires a `head_size` multiple of 8. Hence the `small-*`
# model (`head_size=32`) below, rather than the usual `tiny-*` one (`head_size=2`).


def _make_teacher_topk(
    vocab_size: int, actual_token_ids: list[int], top_k: int, seed: int
) -> tuple[list[list[int]], list[list[float]]]:
    # [FUNCTION DOC]
    # HELPER: Build synthetic teacher top-k candidates per completion position, shaped like teacher scoring output.
    # Behavior: seeds a CPU torch Generator with seed; for each actual token id draws random logits over vocab_size,
    # takes log-softmax and the top_k ids/log-probs; if the actual id is not among them, overwrites the last
    # candidate with the actual id and its log-prob (so the actual token is always present).
    # Inputs: vocab_size (int), actual_token_ids (list of int), top_k (int), seed (int).
    # Outputs: tuple of (ids per position, log-probs per position), each a list of lists of length top_k.
    # Exceptions: none. Side effects: none.
    """Build synthetic per-position teacher candidates, shaped like what `_score_with_teacher` returns.

    Always includes the actual/realized token, matching the real server's guarantee that it's reported regardless of
    rank; otherwise `_narrow_top1_actual_support`'s `beta != 0.0` path would never find it among random candidates and
    silently mask every position out of the loss.
    """
    generator = torch.Generator().manual_seed(seed)
    ids, logprobs = [], []
    for actual_id in actual_token_ids:
        logits = torch.randn(vocab_size, generator=generator)
        log_probs = torch.log_softmax(logits, dim=-1)
        top_logprobs, top_ids = log_probs.topk(top_k)
        top_ids_list, top_logprobs_list = top_ids.tolist(), top_logprobs.tolist()
        if actual_id not in top_ids_list:
            top_ids_list[-1] = actual_id
            top_logprobs_list[-1] = log_probs[actual_id].item()
        ids.append(top_ids_list)
        logprobs.append(top_logprobs_list)
    return ids, logprobs


class _StubRolloutWorker:
    # [CLASS DOC]
    # HELPER: In-process stand-in for the async rollout worker implementing RolloutWorkerProtocol without vLLM.
    # State: rollout_buffer (queue of RolloutSample), metrics_queue (always empty), samples-per-sync count, current
    # model version, vocab size, and an infinite sample iterator over the dataset. start and update_model_version
    # each push samples_per_weight_sync samples into the buffer; stop and check_health do nothing.
    """Minimal rollout worker stub for testing the trainer in isolation, bypassing real vLLM/network."""

    def __init__(self, tokenizer, dataset, vocab_size: int, samples_per_weight_sync: int = 10):
        # [FUNCTION DOC]
        # HELPER: Store settings, create the two queues, set model version 0 and build the sample iterator.
        # Inputs: tokenizer, dataset (conversational prompt/completion rows), vocab_size (int), samples_per_weight_sync
        # (int, default 10). Outputs: none. Exceptions: none. Side effects: none.
        self.rollout_buffer = queue.Queue()
        self.metrics_queue = queue.Queue()  # drained by the trainer in `log()`; this stub measures nothing
        self._samples_per_weight_sync = samples_per_weight_sync
        self._model_version = 0
        self._vocab_size = vocab_size
        self._sample_iter = self._make_sample_iter(tokenizer, dataset)

    def _make_sample_iter(self, tokenizer, dataset):
        # [FUNCTION DOC]
        # HELPER: Infinite generator of RolloutSample objects cycling over the dataset.
        # Behavior: for each row (index idx counts across cycles) tokenizes the prompt with a generation prompt and the
        # prompt plus completion without it; completion ids are the suffix beyond the prompt length. Builds teacher
        # arrays of shape (total length, TEACHER_TOP_K + 1) filled with -1 ids / -inf log-probs, filling completion rows'
        # first TEACHER_TOP_K columns from _make_teacher_topk (seeded with idx). Yields a sample with input ids,
        # completion mask (0 for prompt, 1 for completion), teacher arrays, the current model version, teacher_id
        # "default", prompt_id idx and empty metrics.
        # Inputs: tokenizer, dataset. Outputs: generator. Exceptions: tokenizer errors. Side effects: none.
        for idx, row in enumerate(itertools.cycle(dataset)):
            completion = row["completion"]
            prompt_ids = tokenizer.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True, return_dict=False
            )
            prompt_completion_ids = tokenizer.apply_chat_template(
                row["prompt"] + completion, tokenize=True, add_generation_prompt=False, return_dict=False
            )
            completion_ids = prompt_completion_ids[len(prompt_ids) :]
            teacher_topk_ids, teacher_topk_logprobs = _make_teacher_topk(
                self._vocab_size, completion_ids, TEACHER_TOP_K, seed=idx
            )
            full_teacher_ids = np.full((len(prompt_ids) + len(completion_ids), TEACHER_TOP_K + 1), -1, dtype=np.int64)
            full_teacher_ids[len(prompt_ids) :, :TEACHER_TOP_K] = teacher_topk_ids
            full_teacher_logprobs = np.full(full_teacher_ids.shape, -np.inf, dtype=np.float32)
            full_teacher_logprobs[len(prompt_ids) :, :TEACHER_TOP_K] = teacher_topk_logprobs
            yield RolloutSample(
                prompt=row["prompt"],
                completion=completion,
                input_ids=np.array(prompt_ids + completion_ids, dtype=np.int64),
                completion_mask=np.array([0] * len(prompt_ids) + [1] * len(completion_ids), dtype=np.int64),
                teacher_topk_ids=full_teacher_ids,
                teacher_topk_logprobs=full_teacher_logprobs,
                model_version=self._model_version,
                teacher_id="default",
                prompt_id=idx,  # `itertools.cycle` keeps counting across epochs, like the real `_repeat_iterator`
                metrics={},
            )

    def _fill_queue(self):
        # [FUNCTION DOC]
        # HELPER: Put the next samples_per_weight_sync samples into rollout_buffer.
        # Inputs/Outputs: none. Exceptions: none. Side effects: mutates the queue.
        for _ in range(self._samples_per_weight_sync):
            self.rollout_buffer.put(next(self._sample_iter))

    def start(self):
        # [FUNCTION DOC]
        # HELPER: Fill the buffer once with an initial batch of samples. Side effects: mutates the queue.
        self._fill_queue()

    def update_model_version(self, version):
        # [FUNCTION DOC]
        # HELPER: Record the new model version and enqueue another batch of samples tagged with it.
        # Inputs: version (int). Outputs: none. Side effects: mutates state and queue.
        self._model_version = version
        self._fill_queue()

    def stop(self):
        # [FUNCTION DOC]
        # HELPER: No-op stop. Inputs/Outputs/Exceptions/Side effects: none.
        pass

    def check_health(self, stale_after_s):
        # [FUNCTION DOC]
        # HELPER: No-op health check (always healthy). Inputs: stale_after_s (ignored). Outputs: none.
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


def _rollout_sample(length: int, top_k: int = TEACHER_TOP_K, teacher_id: str = "default", prompt_id: int = 0) -> dict:
    # [FUNCTION DOC]
    # HELPER: Build a synthetic rollout sample dict of the given length for batching/collation tests.
    # Behavior: input ids are 0..length-1; completion mask is 0 for the first (prompt) token and 1 after; teacher ids
    # are -1 everywhere except completion rows, whose first top_k columns are 0..top_k-1; teacher log-probs are
    # -inf except completion rows, set to minus the candidate index minus 0.1. Includes teacher_id, prompt_id and
    # empty metrics.
    # Inputs: length (int), top_k (int, default TEACHER_TOP_K), teacher_id (str, default "default"), prompt_id (int,
    # default 0). Outputs: dict of numpy arrays and scalars. Exceptions: none. Side effects: none.
    # First token is a prompt token (completion_mask 0, empty teacher candidates); the rest are completion tokens.
    teacher_topk_ids = np.full((length, top_k + 1), -1, dtype=np.int64)
    teacher_topk_ids[1:, :top_k] = np.arange(top_k)
    teacher_topk_logprobs = np.full((length, top_k + 1), -np.inf, dtype=np.float32)
    teacher_topk_logprobs[1:, :top_k] = -np.arange(top_k) - 0.1
    return {
        "input_ids": np.arange(length, dtype=np.int64),
        "completion_mask": np.array([0] + [1] * (length - 1), dtype=np.int64),
        "teacher_topk_ids": teacher_topk_ids,
        "teacher_topk_logprobs": teacher_topk_logprobs,
        "teacher_id": teacher_id,
        "prompt_id": prompt_id,
        "metrics": {},
    }


class TestPackingAwareBatching:
    # [CLASS DOC]
    # CPU-only tests for the batchers, the squared-length balancer and DataCollatorForRollout, focusing on the
    # teacher-candidate padding and the metrics the collator records. Uses _rollout_sample inputs. No setup state.
    """Packing/collation is pure scheduling/tensorization (ported from async_grpo's batchers, unchanged), so these
    run without a GPU. The teacher-candidate padding in the collator is new and is the focus here.
    """

    def test_fixed_count_batcher_yields_balanced_fixed_count_micro_batches(self):
        # [FUNCTION DOC]
        # FixedCountBatcher over an endless source of lengths 4,3,2,1 with 2 processes and micro-batch size 4: the first
        # 3 micro-batches each have 2 non-empty groups totalling 4 samples.
        source = (_rollout_sample(length) for length in itertools.cycle((4, 3, 2, 1)))
        batcher = FixedCountBatcher(source, num_processes=2, microbatch_size=4)

        micro_batches = list(itertools.islice(iter(batcher), 3))
        assert len(micro_batches) == 3
        for groups in micro_batches:
            assert len(groups) == 2
            assert all(len(group) > 0 for group in groups)
            assert sum(len(group) for group in groups) == 4

    def test_token_budget_batcher_respects_budget_and_fills_every_row(self):
        # [FUNCTION DOC]
        # TokenBudgetBatcher over 100 length-3 samples with 2 processes and budget 8: the first 5 micro-batches each have
        # 2 non-empty groups and no group exceeds 8 tokens.
        source = (_rollout_sample(3) for _ in range(100))
        batcher = TokenBudgetBatcher(source, num_processes=2, token_budget=8, metrics=defaultdict(list))

        micro_batches = list(itertools.islice(iter(batcher), 5))
        assert len(micro_batches) == 5
        for groups in micro_batches:
            assert len(groups) == 2
            assert all(len(group) > 0 for group in groups)
            assert all(sum(len(s["input_ids"]) for s in group) <= 8 for group in groups)

    def test_balance_equalizes_squared_length(self):
        # [FUNCTION DOC]
        # _balance_by_squared_length on lengths 3,3,2,2 into 2 groups yields equal sums of squared lengths.
        samples = [_rollout_sample(length) for length in (3, 3, 2, 2)]
        groups = _balance_by_squared_length(samples, num_groups=2)
        loads = [sum(len(s["input_ids"]) ** 2 for s in group) for group in groups]
        assert loads[0] == loads[1]

    def test_collator_pads_teacher_candidates_and_masks_prompt_positions(self):
        # [FUNCTION DOC]
        # Collates one length-3 sample (1 process): teacher arrays have shape 1 by 3 by (TEACHER_TOP_K + 1); the prompt
        # position is all -1 ids and -inf log-probs; completion position 1 keeps ids 0..TEACHER_TOP_K-1 and the extra
        # slot stays -1.
        collator = DataCollatorForRollout(pad_token_id=0, num_processes=1)
        a = _rollout_sample(3)  # input_ids [0, 1, 2]; position 0 is prompt (no candidates)
        batch = collator([[[a]]])

        width = (
            TEACHER_TOP_K + 1
        )  # +1 for the sentinel-padded slot (no tail bucket added here; that's compute_loss's job)
        assert batch["teacher_topk_ids"].shape == (1, 3, width)
        assert batch["teacher_topk_logprobs"].shape == (1, 3, width)
        # Prompt position (index 0): every candidate slot is the -1/-inf sentinel.
        assert batch["teacher_topk_ids"][0, 0].tolist() == [-1] * width
        assert torch.isinf(batch["teacher_topk_logprobs"][0, 0]).all()
        # Completion positions (index 1, 2): the real top_k candidates survive, the extra slot is padding.
        assert batch["teacher_topk_ids"][0, 1, :TEACHER_TOP_K].tolist() == list(range(TEACHER_TOP_K))
        assert batch["teacher_topk_ids"][0, 1, TEACHER_TOP_K] == -1

    def test_collator_pads_unequal_rows(self):
        # [FUNCTION DOC]
        # Collates samples of length 3 (prompt_id 7) and 2 (prompt_id 8) on 2 processes. Asserts right padding with
        # pad id 0 for input_ids, attention and completion masks; global token counts (3 trained, 5 forwarded) and mean
        # sequence length 2.5 broadcast per row; the collator's metrics (forwarded/trained token means, masked token
        # fraction as a (2, 5) pair, samples per row, max row tokens, row imbalance 9/6.5, pad fraction (1, 6), no
        # row-fill metric without a budget); and prompts_trained equals {7, 8}.
        collator = DataCollatorForRollout(pad_token_id=0, num_processes=2)
        a = _rollout_sample(3, prompt_id=7)  # input_ids [0, 1, 2]
        b = _rollout_sample(2, prompt_id=8)  # input_ids [0, 1]

        batch = collator([[[a], [b]]])

        assert batch["input_ids"].tolist() == [[0, 1, 2], [0, 1, 0]]  # row b right-padded with pad_token_id
        assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
        assert batch["completion_mask"].tolist() == [[0, 1, 1], [0, 1, 0]]
        assert batch["global_n_tokens"].tolist() == [3.0, 3.0]  # a: 2 + b: 1 completion tokens
        assert batch["global_n_forward_tokens"].tolist() == [5.0, 5.0]  # every token forwarded, prompts included
        assert batch["mean_seq_len"].tolist() == [2.5, 2.5]

        # Sample and packing metrics are aggregated here on rank 0, not broadcast with the batch and reduced back.
        assert collator.metrics["sample/forwarded_tokens_mean"] == [2.5]
        assert collator.metrics["sample/trained_tokens_mean"] == [1.5]
        assert collator.metrics["batch/masked_token_frac"] == [(2, 5)]  # the two prompt tokens earn no gradient
        assert collator.metrics["batch/samples_per_row"] == [1.0]
        assert collator.metrics["batch/row_tokens_max"] == [3.0]
        assert collator.metrics["batch/row_imbalance"] == [9 / 6.5]  # Σ Lᵢ² of 9 and 4, against their mean
        assert collator.metrics["batch/pad_frac"] == [(1, 6)]  # the one slot row b was padded with
        assert "batch/row_fill_frac" not in collator.metrics  # no budget passed, so nothing to fill

        # The prompts behind this micro-batch, accumulated for `_EpochStopCallback` to count epochs with.
        assert collator.prompts_trained == {7, 8}

    def test_collator_packs_teacher_id_per_token_for_mopd(self):
        # [FUNCTION DOC]
        # With a teacher_id_to_idx mapping, packing a "math" length-3 sample and a "code" length-2 sample into one row
        # yields per-token teacher_id_idx 0,0,0,1,1; without a mapping the key is absent.
        # Packing concatenates samples from different teachers into one row, so compute_loss can only attribute a
        # token's jsd/teacher_entropy to a teacher if the id is broadcast across that sample's own tokens here.
        collator = DataCollatorForRollout(pad_token_id=0, num_processes=1, teacher_id_to_idx={"math": 0, "code": 1})
        batch = collator([[[_rollout_sample(3, teacher_id="math"), _rollout_sample(2, teacher_id="code")]]])

        assert batch["teacher_id_idx"].tolist() == [[0, 0, 0, 1, 1]]

        single = DataCollatorForRollout(pad_token_id=0, num_processes=1)
        assert "teacher_id_idx" not in single([[[_rollout_sample(3)]]])


class TestMetricReduction:
    # [CLASS DOC]
    # Tests _reduce_metric: reduction is chosen by value shape (pairs are rates) and key name (total suffix sums,
    # max/min as a word take extremes, others average).
    """The value's shape and the key's suffix are the whole reduction API; nothing is registered or configured."""

    @pytest.mark.parametrize(
        ("key", "values", "expected"),
        [
            ("perf/fwd_bwd_s", [1.0, 2.0, 6.0], 3.0),  # gauge -> window mean
            ("sample/dropped_stale_total", [1.0, 1.0, 1.0], 3.0),  # counter -> sum of the pushed deltas
            ("sample/forwarded_tokens_max", [3.0, 9.0, 5.0], 9.0),
            # A `max`/`min` WORD, not a suffix, so the upstream spellings reduce correctly too...
            ("completions/max_length", [3.0, 9.0, 5.0], 9.0),
            ("completions/min_length", [3.0, 9.0, 5.0], 3.0),
            # ... while a name that merely contains those letters stays a gauge.
            ("perf/maximal_s", [2.0, 4.0], 3.0),
            # Rates reduce as Σnum / Σden. These two windows ran at 1 and 100 tok/s; meaning the ratios would report
            # 50.5, the defect that made `training_tok/s` unusable.
            ("perf/forwarded_tok_s_wall_clock", [(100.0, 100.0), (100.0, 1.0)], 200 / 101),
        ],
    )
    def test_reduces_by_shape_and_name(self, key, values, expected):
        # [FUNCTION DOC]
        # Parametrized cases: gauge mean, total-counter sum, _max suffix, upstream max_length/min_length words, a name
        # merely containing "max" letters stays a mean, and a rate reduces as sum of numerators over sum of denominators.
        # Asserts the reduced value approximately equals expected.
        assert _reduce_metric(key, values) == pytest.approx(expected)

    def test_rate_with_zero_denominator_is_nan(self):
        # [FUNCTION DOC]
        # A rate whose only pair is (0, 0) reduces to NaN.
        assert math.isnan(_reduce_metric("perf/forwarded_tok_s_wall_clock", [(0.0, 0.0)]))


class TestRolloutWorkerProtocol:
    # [CLASS DOC]
    # Guards that the test stub stays in sync with RolloutWorkerProtocol.
    def test_stub_worker_exposes_every_protocol_attribute(self):
        # [FUNCTION DOC]
        # Builds _StubRolloutWorker with no tokenizer and an empty dataset (the sample generator is lazy) and asserts
        # every annotated attribute of RolloutWorkerProtocol exists on it.
        # A stub that falls behind the protocol breaks training, and only the GPU-gated trainer test would notice.
        stub = _StubRolloutWorker(tokenizer=None, dataset=[], vocab_size=8)
        assert not [name for name in RolloutWorkerProtocol.__annotations__ if not hasattr(stub, name)]


def _position(*entries):
    # [FUNCTION DOC]
    # HELPER: Build one vLLM prompt_logprobs entry: maps the string form of each token id to a dict with its logprob
    # and rank. Inputs: entries as (token_id, logprob, rank) triples. Outputs: dict. Exceptions: none (ValueError if
    # an entry is not a triple). Side effects: none.
    """A vLLM `prompt_logprobs[i]` entry: `{token_id: {"logprob": ..., "rank": ...}}`, JSON keys are strings."""
    return {str(token_id): {"logprob": logprob, "rank": rank} for token_id, logprob, rank in entries}


class TestParseTeacherLogprobs:
    # [CLASS DOC]
    # Tests _parse_teacher_logprobs_at_position: candidates sorted by rank, NaN log-probs dropped, empty or None
    # positions give empty lists, and an out-of-top-k realized token is kept.
    """vLLM's `prompt_logprobs` mapping is unordered; parsing sorts it by rank and drops NaNs."""

    @pytest.mark.parametrize(
        ("position", "expected_ids", "expected_logprobs"),
        [
            (_position((9, -2.0, 3), (5, -0.1, 1), (1, -1.0, 2)), [5, 1, 9], [-0.1, -1.0, -2.0]),
            # The realized token is reported whatever its rank, on top of the requested top-k. Truncating back to k
            # (as `VLLMClient.get_sequence_logprobs` does for the sync trainers) would drop it here, zeroing the
            # teacher signal at that position for beta=1.0.
            (_position((5, -0.1, 1), (1, -1.0, 2), (9, -8.0, 4242)), [5, 1, 9], [-0.1, -1.0, -8.0]),
            # NaN reflects an invalid score, not a padding slot, so the candidate is dropped rather than sentinelled.
            (_position((5, -0.1, 1), (1, float("nan"), 2), (9, -2.0, 3)), [5, 9], [-0.1, -2.0]),
            # A position vLLM scored nothing for yields the same empty pair as a masked-out prompt position.
            (None, [], []),
            ({}, [], []),
        ],
    )
    def test_parses_position(self, position, expected_ids, expected_logprobs):
        # [FUNCTION DOC]
        # Parametrized over positions (unordered ranks, an extra high-rank realized token, a NaN candidate, None, empty
        # dict). Asserts the parsed ids equal expected and log-probs approximately equal expected.
        ids, logprobs = _parse_teacher_logprobs_at_position(position)
        assert ids == expected_ids
        assert logprobs == pytest.approx(expected_logprobs)


def _rollout_loop(**kwargs):
    # [FUNCTION DOC]
    # HELPER: Construct an _AsyncRolloutLoop with test defaults (constructor only sets up state, no connections).
    # Behavior: uses the given processing_class or loads the small Qwen2.5 tokenizer; defaults model_name "test", a
    # one-prompt dataset, eos ids from the tokenizer, multiprocessing queues for rollout buffer, exceptions and
    # metrics, shared integer model version 0, shared float heartbeat 0.0 and a failed event; caller kwargs override.
    # Inputs: keyword arguments for _AsyncRolloutLoop. Outputs: _AsyncRolloutLoop.
    # Exceptions: TypeError for missing required args. Side effects: may download a tokenizer; creates mp primitives.
    # `_AsyncRolloutLoop.__init__` only sets up state (no vLLM or teacher connection happens here).
    tokenizer = kwargs.get("processing_class") or AutoTokenizer.from_pretrained(
        "trl-internal-testing/small-Qwen2ForCausalLM-2.5"
    )
    kwargs = {
        "model_name": "test",
        "dataset": Dataset.from_dict({"prompt": ["hi"]}),
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


def _scoring_loop(tokenizer, teacher_server_urls):
    # [FUNCTION DOC]
    # HELPER: Build a rollout loop configured for teacher scoring (given teacher URLs, top-k TEACHER_TOP_K, timeout
    # 30) and set teacher_model_names so each teacher id maps to "<id>-model" (normally resolved at run time).
    # Inputs: tokenizer, teacher_server_urls (dict of id to URL). Outputs: _AsyncRolloutLoop. Side effects: none.
    loop = _rollout_loop(
        processing_class=tokenizer,
        teacher_server_urls=teacher_server_urls,
        teacher_top_k=TEACHER_TOP_K,
        request_timeout=30,
    )
    # Normally resolved from each teacher server's /v1/models in _run_loops, which these tests never reach
    loop.teacher_model_names = {teacher_id: f"{teacher_id}-model" for teacher_id in teacher_server_urls}
    return loop


ONE_TEACHER = {"default": "http://default:8001"}
TWO_TEACHERS = {"math": "http://math:8002", "code": "http://code:8003"}


class TestWorkerMetrics:
    # [CLASS DOC]
    # Tests the rollout loop's metric pushing: payload shape, delta reset of counters and rates, non-blocking pushes
    # and per-rollout completion/teacher metrics. Module constants ONE_TEACHER and TWO_TEACHERS give URL maps.
    """The worker's payload has the same shape as the trainer's sink, so draining it in `log()` is an append."""

    def _loop(self, teacher_server_urls=ONE_TEACHER, maxsize=0):
        # [FUNCTION DOC]
        # HELPER: Build a rollout loop with the given teacher URLs (default ONE_TEACHER) and a metrics queue of the given
        # maxsize (0 means unbounded). Outputs: _AsyncRolloutLoop.
        return _rollout_loop(teacher_server_urls=teacher_server_urls, metrics_queue=mp.Queue(maxsize=maxsize))

    def test_counters_and_rates_ride_along_and_reset(self):
        # [FUNCTION DOC]
        # Increments a counter by 2 and a rate by (3.0, 2), pushes a gauge, and asserts the queued payload merges the
        # gauge, the counter as float and the rate pair; a second empty push yields an empty payload (deltas reset).
        loop = self._loop()
        loop._counters["rollout/vllm_retry_total"] += 2
        loop._rates["rollout/score_s"][0] += 3.0
        loop._rates["rollout/score_s"][1] += 2
        loop._push_metrics({"rollout/inflight": 4.0})

        assert loop._metrics_queue.get(timeout=5) == {
            "rollout/inflight": 4.0,
            "rollout/vllm_retry_total": 2.0,
            "rollout/score_s": (3.0, 2.0),
        }
        # Counters and rates carry deltas, so what one push reported must not ride along on the next.
        loop._push_metrics({})
        assert loop._metrics_queue.get(timeout=5) == {}

    def test_push_never_blocks_when_the_trainer_stops_draining(self):
        # [FUNCTION DOC]
        # With a metrics queue of maxsize 1, 50 pushes complete without blocking (extra payloads are dropped).
        loop = self._loop(maxsize=1)
        for _ in range(50):
            loop._push_metrics({"rollout/inflight": 1.0})  # drops rather than stalling generation

    @pytest.mark.parametrize(
        ("teacher_server_urls", "ends_on_eos", "clipped", "per_teacher"),
        [
            (ONE_TEACHER, True, 0.0, False),  # ends on eos: the model stopped on its own
            (TWO_TEACHERS, False, 1.0, True),  # ends mid-sentence: cut off by max_completion_length
        ],
    )
    def test_rollout_push_reports_the_completion_and_the_teacher_call(
        self, teacher_server_urls, ends_on_eos, clipped, per_teacher
    ):
        # [FUNCTION DOC]
        # Parametrized: one teacher with an eos-terminated completion (not clipped, no per-teacher series) versus two
        # teachers with a non-eos completion (clipped, per-teacher series). Pushes rollout metrics for a 3-token
        # completion with score time 0.25 and duration 1.5 and asserts duration, mean length pair (3, 1), max length 3,
        # clipped-ratio pair, score-time pair, and presence of teacher_score_s/math only for multiple teachers.
        loop = self._loop(teacher_server_urls)
        completion_ids = [7, 8, loop.eos_token_ids[0] if ends_on_eos else 9]
        loop._push_rollout_metrics(completion_ids=completion_ids, teacher_id="math", score_s=0.25, duration_s=1.5)
        payload = loop._metrics_queue.get(timeout=5)

        assert payload["rollout/duration_s"] == 1.5
        assert payload["completions/mean_length"] == (3.0, 1.0)
        assert payload["completions/max_length"] == 3.0
        assert payload["completions/clipped_ratio"] == (clipped, 1.0)
        assert payload["rollout/score_s"] == (0.25, 1.0)
        # A per-teacher latency series only exists under MOPD; with one teacher the blended mean is the whole story.
        assert ("teacher_score_s/math" in payload) is per_teacher


class TestMultiTeacherRouting:
    # [CLASS DOC]
    # Tests teacher routing in _generate_and_score_one: one teacher scores all rows; with several teachers the row's
    # teacher_id selects URL and served model name; unroutable rows raise. Uses fakes for generation and HTTP post.
    """A single configured teacher scores every row regardless of `teacher_id` (plain on-policy distillation).

    Multiple teachers enable MOPD: each row's `teacher_id` column selects among them; a missing or unmapped
    `teacher_id` is a configuration error, not a silent fallback. Each request must carry the URL *and* the served
    model id of the teacher it routes to, since `/v1/completions` names the model it addresses and every teacher serves
    its own.
    """

    def _run(self, teacher_server_urls, *teacher_ids):
        # [FUNCTION DOC]
        # HELPER: Score one row per teacher id (None means the row has no teacher_id column) and capture requests.
        # Behavior: builds a scoring loop with the small Qwen2.5 tokenizer, replaces _generate_one_turn with an async fake
        # returning the single completion token 7, and _post with an async fake that records (base URL, payload) and
        # returns one candidate (token 7, logprob -0.5, rank 1) per prompt token; runs _generate_and_score_one per row
        # via asyncio.run.
        # Inputs: teacher_server_urls (dict), teacher_ids (variadic). Outputs: tuple of (samples list, requests list).
        # Exceptions: propagates ValueError from routing. Side effects: may download a tokenizer.
        """Score one row per `teacher_ids` entry (`None` = no `teacher_id` column) and return the requests sent."""
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/small-Qwen2ForCausalLM-2.5")
        loop = _scoring_loop(tokenizer, teacher_server_urls)
        requests = []

        async def fake_generate_one_turn(prompt_ids):
            # [FUNCTION DOC]
            # HELPER: Async fake generation returning the completion token list containing only 7. Inputs: prompt_ids.
            return [7]

        async def fake_post(base_url, path, payload, timeout):
            # [FUNCTION DOC]
            # HELPER: Async fake HTTP post: records the base URL and payload into the enclosing requests list and returns a
            # choices payload with one identical prompt_logprobs entry per prompt token.
            # Inputs: base_url, path, payload, timeout. Outputs: dict. Side effects: appends to requests.
            requests.append((base_url, payload))
            # vLLM scores every token of the sequence it was handed, not just the completion region.
            return {"choices": [{"prompt_logprobs": [_position((7, -0.5, 1))] * len(payload["prompt"])}]}

        loop._generate_one_turn = fake_generate_one_turn
        loop._post = fake_post

        rows = [
            {"prompt": [{"role": "user", "content": "hi"}], **({} if tid is None else {"teacher_id": tid})}
            for tid in teacher_ids
        ]
        samples = [asyncio.run(loop._generate_and_score_one(i, row)) for i, row in enumerate(rows)]
        return samples, requests

    @pytest.mark.parametrize("teacher_id", ["whatever", None])
    def test_single_teacher_scores_every_row(self, teacher_id):
        # [FUNCTION DOC]
        # Parametrized over an arbitrary teacher id and no teacher id: with one teacher the single request goes to its
        # URL with model "default-model" and the sample's teacher_id is "default".
        samples, requests = self._run(ONE_TEACHER, teacher_id)
        assert [url for url, _ in requests] == ["http://default:8001"]
        assert [payload["model"] for _, payload in requests] == ["default-model"]
        assert [sample.teacher_id for sample in samples] == ["default"]

    def test_each_row_is_addressed_to_its_own_teacher(self):
        # [FUNCTION DOC]
        # With two teachers, rows tagged code then math go to their respective URLs and model names in order, and the
        # samples carry those teacher ids.
        samples, requests = self._run(TWO_TEACHERS, "code", "math")
        assert [url for url, _ in requests] == ["http://code:8003", "http://math:8002"]
        assert [payload["model"] for _, payload in requests] == ["code-model", "math-model"]
        # The routed id rides along on the sample so the collator can tag every token with it (MOPD metrics).
        assert [sample.teacher_id for sample in samples] == ["code", "math"]

    def test_scores_only_the_completion_region(self):
        # [FUNCTION DOC]
        # The request sends the full input ids with prompt_logprobs equal to TEACHER_TOP_K; in the sample, prompt rows of
        # teacher ids are all -1, completion rows have 7 in the first slot and -1 elsewhere.
        (sample,), ((_, payload),) = self._run(ONE_TEACHER, None)
        n_prompt = len(sample.input_ids) - int(sample.completion_mask.sum())
        assert payload["prompt"] == sample.input_ids.tolist()
        assert payload["prompt_logprobs"] == TEACHER_TOP_K
        assert (sample.teacher_topk_ids[:n_prompt] == -1).all()
        assert (sample.teacher_topk_ids[n_prompt:, 0] == 7).all()
        assert (sample.teacher_topk_ids[n_prompt:, 1:] == -1).all()

    @pytest.mark.parametrize(("teacher_id", "match"), [(None, "teacher_id=None"), ("unknown", "teacher_id='unknown'")])
    def test_multi_teacher_raises_on_unroutable_row(self, teacher_id, match):
        # [FUNCTION DOC]
        # Parametrized over a missing and an unknown teacher_id with two teachers: ValueError mentioning the offending id.
        with pytest.raises(ValueError, match=match):
            self._run(TWO_TEACHERS, teacher_id)


class TestBetaRange:
    # [CLASS DOC]
    # Tests AsyncDistillationConfig validation of beta to the closed range 0 to 1.
    """beta must be a valid generalized-JSD interpolation coefficient in [0.0, 1.0], matching
    `DistillationTrainer`/`GKDConfig`'s own bounds; values outside that range are never meaningful.
    """

    @pytest.mark.parametrize("beta", [-0.1, 1.1, 2.0])
    def test_out_of_range_beta_raises(self, beta):
        # [FUNCTION DOC]
        # Parametrized over -0.1, 1.1 and 2.0: constructing the config raises ValueError "beta must be in [0.0, 1.0]".
        with pytest.raises(ValueError, match=r"beta must be in \[0.0, 1.0\]"):
            AsyncDistillationConfig(output_dir="/tmp/unused", report_to="none", beta=beta)


class TestNarrowTop1ActualSupport:
    # [CLASS DOC]
    # Tests _narrow_top1_actual_support, which for 0 < beta < 1 narrows the teacher candidates to two slots (teacher
    # top-1, actual token) and for beta 1.0 to a single actual-token slot, with a validity mask; plus checks that the
    # narrow support composes with _add_tail_bucket and _jsd_divergence. All inputs are small hand-built tensors of
    # shape batch by positions by candidates.
    """`beta != 0.0` narrows the teacher-reported support to just the candidates the divergence needs, mirroring
    `ServerDistillationTrainer`'s `beta > 0` path exactly, including its `beta == 1.0` special case: for `0 < beta <
    1`, the support is the teacher's own top-1 token plus the completion's actual/realized token (the only two
    identities the wire protocol guarantees a teacher logprob for); at `beta == 1.0` (pure reverse KL, purely
    student-weighted), the teacher's top-1 token is dropped entirely and only the actual token remains.
    """

    def test_top1_and_actual_both_valid_and_distinct(self):
        # [FUNCTION DOC]
        # Top-1 token 5 and actual token 9 present at index 2 with beta 0.5: narrow ids are 5 and 9, log-probs -0.1 and
        # -2.0, both slots valid.
        # position 0: top-1 is token 5, actual (realized) token is 9, found at index 2 of the reported candidates.
        teacher_topk_ids = torch.tensor([[[5, 3, 9, -1]]])
        teacher_topk_logprobs = torch.tensor([[[-0.1, -1.0, -2.0, float("-inf")]]])
        actual_token_ids = torch.tensor([[9]])
        valid_mask_wide = teacher_topk_ids != -1

        narrow_ids, narrow_logps, mask = _narrow_top1_actual_support(
            teacher_topk_ids, teacher_topk_logprobs, actual_token_ids, valid_mask_wide, beta=0.5
        )
        assert narrow_ids.tolist() == [[[5, 9]]]
        assert narrow_logps.squeeze().tolist() == pytest.approx([-0.1, -2.0])
        assert mask.tolist() == [[[True, True]]]

    def test_actual_token_same_as_top1_is_deduplicated(self):
        # [FUNCTION DOC]
        # Actual token equals the top-1 (5): ids are 5 and 5 with the second slot masked out.
        # The realized token IS the teacher's top-1 choice: the second (actual) slot must be masked out, not
        # double-counted, since it's the same token as the first (top-1) slot.
        teacher_topk_ids = torch.tensor([[[5, 3, 9, -1]]])
        teacher_topk_logprobs = torch.tensor([[[-0.1, -1.0, -2.0, float("-inf")]]])
        actual_token_ids = torch.tensor([[5]])
        valid_mask_wide = teacher_topk_ids != -1

        narrow_ids, _, mask = _narrow_top1_actual_support(
            teacher_topk_ids, teacher_topk_logprobs, actual_token_ids, valid_mask_wide, beta=0.5
        )
        assert narrow_ids.tolist() == [[[5, 5]]]
        assert mask.tolist() == [[[True, False]]]

    def test_actual_token_missing_from_reported_candidates(self):
        # [FUNCTION DOC]
        # Actual token 42 not among candidates: ids are 5 and 42 with the actual slot masked out.
        # The realized token never appears among the teacher's reported candidates at all (e.g. a low-probability
        # token the server didn't report); the actual slot must be masked out, not silently given a wrong logprob.
        teacher_topk_ids = torch.tensor([[[5, 3, -1, -1]]])
        teacher_topk_logprobs = torch.tensor([[[-0.1, -1.0, float("-inf"), float("-inf")]]])
        actual_token_ids = torch.tensor([[42]])
        valid_mask_wide = teacher_topk_ids != -1

        narrow_ids, _, mask = _narrow_top1_actual_support(
            teacher_topk_ids, teacher_topk_logprobs, actual_token_ids, valid_mask_wide, beta=0.5
        )
        assert narrow_ids.tolist() == [[[5, 42]]]
        assert mask.tolist() == [[[True, False]]]

    def test_top1_missing_but_actual_present(self):
        # [FUNCTION DOC]
        # Top-1 slot invalid (-1) but actual token 9 present: mask is False for top-1 and True for actual.
        # No usable teacher signal at all for the top-1 slot (e.g. every candidate was NaN server-side), but the
        # actual token still happens to be present among the (otherwise-invalid) reported candidates.
        teacher_topk_ids = torch.tensor([[[-1, 9, -1, -1]]])
        teacher_topk_logprobs = torch.tensor([[[float("-inf"), -2.0, float("-inf"), float("-inf")]]])
        actual_token_ids = torch.tensor([[9]])
        valid_mask_wide = teacher_topk_ids != -1

        _, _, mask = _narrow_top1_actual_support(
            teacher_topk_ids, teacher_topk_logprobs, actual_token_ids, valid_mask_wide, beta=0.5
        )
        assert mask.tolist() == [[[False, True]]]

    def test_beta_one_drops_teacher_top1_entirely(self):
        # [FUNCTION DOC]
        # beta 1.0: support width is 1, the id is the actual token 9 with log-prob -2.0, and it is valid.
        # Pure reverse KL is purely student-weighted: the teacher's top-1 token (5) must not appear in the support
        # at all, even though it's valid and distinct from the actual token (9) - only width-1 support remains.
        teacher_topk_ids = torch.tensor([[[5, 3, 9, -1]]])
        teacher_topk_logprobs = torch.tensor([[[-0.1, -1.0, -2.0, float("-inf")]]])
        actual_token_ids = torch.tensor([[9]])
        valid_mask_wide = teacher_topk_ids != -1

        narrow_ids, narrow_logps, mask = _narrow_top1_actual_support(
            teacher_topk_ids, teacher_topk_logprobs, actual_token_ids, valid_mask_wide, beta=1.0
        )
        assert narrow_ids.shape[-1] == 1
        assert narrow_ids.tolist() == [[[9]]]
        assert narrow_logps.squeeze().item() == pytest.approx(-2.0)
        assert mask.tolist() == [[[True]]]

    def test_beta_one_actual_token_missing(self):
        # [FUNCTION DOC]
        # beta 1.0 with actual token 42 absent: the single slot id is 42 and is masked out.
        # Pure reverse KL, but the actual token isn't among the teacher's reported candidates: the lone support
        # slot must be masked out (no usable teacher signal at all), not silently wrong.
        teacher_topk_ids = torch.tensor([[[5, 3, -1, -1]]])
        teacher_topk_logprobs = torch.tensor([[[-0.1, -1.0, float("-inf"), float("-inf")]]])
        actual_token_ids = torch.tensor([[42]])
        valid_mask_wide = teacher_topk_ids != -1

        narrow_ids, _, mask = _narrow_top1_actual_support(
            teacher_topk_ids, teacher_topk_logprobs, actual_token_ids, valid_mask_wide, beta=1.0
        )
        assert narrow_ids.tolist() == [[[42]]]
        assert mask.tolist() == [[[False]]]

    def test_narrow_support_feeds_add_tail_bucket_and_jsd_divergence_without_error(self):
        # [FUNCTION DOC]
        # beta 0.5 on two positions: gathers random student log-probs (vocab 32) at the narrow ids, sets masked slots to
        # -inf on both sides, adds tail buckets and computes the JSD. Asserts all values are finite and every per-token
        # sum over the support is at least -1e-5.
        # End-to-end sanity: the narrow support's shape/dtype/mask conventions must be exactly what
        # `_add_tail_bucket`/`_jsd_divergence` (shared with the beta=0.0 path) expect.
        teacher_topk_ids = torch.tensor([[[5, 3, 9, -1], [7, -1, -1, -1]]])
        teacher_topk_logprobs = torch.tensor(
            [[[-0.1, -1.0, -2.0, float("-inf")], [-0.05, float("-inf"), float("-inf"), float("-inf")]]]
        )
        actual_token_ids = torch.tensor([[9, 7]])
        valid_mask_wide = teacher_topk_ids != -1

        narrow_ids, teacher_logps, mask = _narrow_top1_actual_support(
            teacher_topk_ids, teacher_topk_logprobs, actual_token_ids, valid_mask_wide, beta=0.5
        )
        student_logps_full = torch.log_softmax(torch.randn(1, 2, 32), dim=-1)
        student_logps = student_logps_full.gather(-1, narrow_ids.clamp_min(0))
        neg_inf = torch.full((), float("-inf"))
        student_logps = torch.where(mask, student_logps, neg_inf)
        teacher_logps = torch.where(mask, teacher_logps, neg_inf)

        student_full, support_mask = _add_tail_bucket(student_logps, mask)
        teacher_full, _ = _add_tail_bucket(teacher_logps, mask)
        jsd = _jsd_divergence(student_full, teacher_full, beta=0.5, support_mask=support_mask)
        assert torch.isfinite(jsd).all()
        # Individual per-slot terms can be negative (e.g. student_prob(x) * (log student(x) - log teacher(x)) when
        # student(x) < teacher(x)); only the per-token sum across the support dimension (the actual KL/JSD) is
        # guaranteed non-negative (Gibbs' inequality).
        assert (jsd.sum(dim=-1) >= -1e-5).all()

    def test_narrow_support_beta_one_feeds_add_tail_bucket_and_jsd_divergence_without_error(self):
        # [FUNCTION DOC]
        # Same pipeline as the beta 0.5 variant but with beta 1.0 (width-1 support): finite values and per-token sums at
        # least -1e-5.
        # Same end-to-end sanity check, but for the width-1 beta==1.0 support.
        teacher_topk_ids = torch.tensor([[[5, 3, 9, -1], [7, -1, -1, -1]]])
        teacher_topk_logprobs = torch.tensor(
            [[[-0.1, -1.0, -2.0, float("-inf")], [-0.05, float("-inf"), float("-inf"), float("-inf")]]]
        )
        actual_token_ids = torch.tensor([[9, 7]])
        valid_mask_wide = teacher_topk_ids != -1

        narrow_ids, teacher_logps, mask = _narrow_top1_actual_support(
            teacher_topk_ids, teacher_topk_logprobs, actual_token_ids, valid_mask_wide, beta=1.0
        )
        student_logps_full = torch.log_softmax(torch.randn(1, 2, 32), dim=-1)
        student_logps = student_logps_full.gather(-1, narrow_ids.clamp_min(0))
        neg_inf = torch.full((), float("-inf"))
        student_logps = torch.where(mask, student_logps, neg_inf)
        teacher_logps = torch.where(mask, teacher_logps, neg_inf)

        student_full, support_mask = _add_tail_bucket(student_logps, mask)
        teacher_full, _ = _add_tail_bucket(teacher_logps, mask)
        jsd = _jsd_divergence(student_full, teacher_full, beta=1.0, support_mask=support_mask)
        assert torch.isfinite(jsd).all()
        # Individual per-slot terms can be negative (e.g. student_prob(x) * (log student(x) - log teacher(x)) when
        # student(x) < teacher(x)); only the per-token sum across the support dimension (the actual KL/JSD) is
        # guaranteed non-negative (Gibbs' inequality).
        assert (jsd.sum(dim=-1) >= -1e-5).all()

    def test_padded_positions_keep_forward_and_backward_finite(self):
        # [FUNCTION DOC]
        # Ragged 2-sample batch where one position is entirely -1/-inf. For beta 0.5 and 1.0 runs the narrow support,
        # tail bucket and JSD on student log-probs derived from a leaf tensor requiring grad; asserts the JSD is finite,
        # then sums per-token JSD only over positions with any valid slot, backpropagates and asserts finite gradients.
        # Adapted from ServerDistillationTrainer's own `TestServerReverseKLPaddingMask` test: a ragged batch where
        # one sample's completion is shorter than another's, so some positions are all -inf sentinels (masked out
        # via has_teacher_signal in compute_loss, but this checks the loss math itself, given already-neutralized
        # padding, never leaks -inf/NaN into the backward pass).
        teacher_topk_ids = torch.tensor([[[5, 3, 9, -1], [-1, -1, -1, -1]], [[5, 3, 9, -1], [7, 2, 4, -1]]])
        teacher_topk_logprobs = torch.tensor(
            [
                [[-0.1, -1.0, -2.0, float("-inf")], [float("-inf")] * 4],
                [[-0.1, -1.0, -2.0, float("-inf")], [-0.05, -1.2, -2.5, float("-inf")]],
            ]
        )
        actual_token_ids = torch.tensor([[9, 0], [9, 7]])
        valid_mask_wide = teacher_topk_ids != -1

        for beta in (0.5, 1.0):
            narrow_ids, teacher_logps, mask = _narrow_top1_actual_support(
                teacher_topk_ids, teacher_topk_logprobs, actual_token_ids, valid_mask_wide, beta=beta
            )
            raw_student = torch.randn(2, 2, 32, requires_grad=True)
            student_log_probs_full = torch.log_softmax(raw_student, dim=-1)
            student_logps = student_log_probs_full.gather(-1, narrow_ids.clamp_min(0))
            neg_inf = torch.full((), float("-inf"))
            student_logps = torch.where(mask, student_logps, neg_inf)
            teacher_logps = torch.where(mask, teacher_logps, neg_inf)

            student_full, support_mask = _add_tail_bucket(student_logps, mask)
            teacher_full, _ = _add_tail_bucket(teacher_logps, mask)
            jsd = _jsd_divergence(student_full, teacher_full, beta=beta, support_mask=support_mask)
            assert torch.isfinite(jsd).all(), f"beta={beta}"

            # The fully-masked position (sample 0, position 1) must not contribute -inf/NaN gradient once summed
            # and backpropagated, mirroring compute_loss's has_teacher_signal exclusion of such positions.
            has_teacher_signal = mask.any(dim=-1)
            loss = jsd.sum(dim=-1)[has_teacher_signal].sum()
            loss.backward()
            assert torch.isfinite(raw_student.grad).all(), f"beta={beta}"


class TestSparseGeneralizedJSDMatchesFullVocabReference:
    # [CLASS DOC]
    # Checks the sparse tail-bucket JSD equals the dense reference _jsd_divergence from ServerDistillationTrainer
    # when the "top-k" support is the full vocabulary.
    """Compares our sparse top-k pipeline (`_add_tail_bucket` + `_jsd_divergence`) against a plain dense-vocabulary
    generalized-JSD computation — `ServerDistillationTrainer`'s own `_jsd_divergence`, called with `support_mask=None`
    so it takes its dense `F.kl_div` branch instead of the masked/sparse one. Feeding our pipeline the *full*
    vocabulary as its "top-k" support (`top_k = vocab_size`) leaves ~0 residual mass for the tail bucket, so the two
    must match up to that negligible epsilon, not just approximately.
    """

    def test_forward_kl_matches_full_vocab_generalized_jsd_loss(self):
        # [FUNCTION DOC]
        # With seed 0, batch 2, length 5, vocab 32 random logits: computes the reference dense JSD at beta 0 and averages
        # over tokens; computes the sparse path by sorting teacher log-probs over the whole vocab, gathering student
        # log-probs, adding tail buckets with an all-true mask and averaging; asserts the two are close.
        torch.manual_seed(0)
        batch, seq_len, vocab_size = 2, 5, 32
        student_logits = torch.randn(batch, seq_len, vocab_size)
        teacher_logits = torch.randn(batch, seq_len, vocab_size)

        student_log_probs_full = torch.log_softmax(student_logits, dim=-1)
        teacher_log_probs_full = torch.log_softmax(teacher_logits, dim=-1)

        reference_jsd = _reference_jsd_divergence(student_log_probs_full, teacher_log_probs_full, beta=0.0)
        reference = reference_jsd.sum() / (batch * seq_len)  # batchmean, matching this test's own reduction below

        # Simulate what the async worker/collator produce, but with the full vocabulary as the "top-k" support so
        # nothing is truncated (see class docstring for why this is a valid dense reference).
        teacher_topk_logprobs, teacher_topk_ids = teacher_log_probs_full.topk(vocab_size, dim=-1)

        student_support_logps = student_log_probs_full.gather(-1, teacher_topk_ids)
        valid_mask = torch.ones_like(teacher_topk_ids, dtype=torch.bool)

        student_log_probs, support_mask = _add_tail_bucket(student_support_logps, valid_mask)
        teacher_log_probs, _ = _add_tail_bucket(teacher_topk_logprobs, valid_mask)

        jsd = _jsd_divergence(student_log_probs, teacher_log_probs, beta=0.0, support_mask=support_mask)
        ours = jsd.sum() / max(jsd.shape[0] * jsd.shape[1], 1)  # batchmean, matching the reference's reduction

        torch.testing.assert_close(ours, reference)


class TestPerTeacherStats:
    # [CLASS DOC]
    # Tests the per-teacher statistics returned by _chunked_jsd_loss (count, JSD sum, teacher entropy sum per teacher)
    # for packed rows split into chunks. Class constants: 10 positions, hidden size 8, vocab 32, 2 teachers.
    """MOPD's per-teacher breakdown: `_chunked_jsd_loss` returns a `(count, jsd_sum, teacher_entropy_sum)` triple per
    teacher, which `compute_loss` turns into `teacher_jsd/{id}`. Packing concatenates several teachers' tokens into one
    row and then splits that row into fixed-size chunks, so the partition has to survive a teacher's positions being
    spread across chunk boundaries.
    """

    NUM_POSITIONS, HIDDEN_SIZE, VOCAB_SIZE, NUM_TEACHERS = 10, 8, 32, 2

    def _run(self, teacher_id_idx, chunk_size):
        # [FUNCTION DOC]
        # HELPER: Run _chunked_jsd_loss on seeded random hidden states and LM head weights with synthetic teacher
        # candidates from _make_teacher_topk (all valid), positional settings matching beta 0 with entropy tracking on,
        # and the given per-token teacher index tensor and chunk size.
        # Inputs: teacher_id_idx (tensor or None), chunk_size (int). Outputs: the loss function's result tuple (last item
        # is the flattened per-teacher stats). Exceptions: none expected. Side effects: none.
        generator = torch.Generator().manual_seed(0)
        hidden = torch.randn(self.NUM_POSITIONS, self.HIDDEN_SIZE, generator=generator)
        lm_head_weight = torch.randn(self.VOCAB_SIZE, self.HIDDEN_SIZE, generator=generator)
        actual_token_ids = torch.randint(self.VOCAB_SIZE, (self.NUM_POSITIONS,), generator=generator).tolist()
        ids, logprobs = _make_teacher_topk(self.VOCAB_SIZE, actual_token_ids, TEACHER_TOP_K, seed=0)
        target_ids = torch.tensor(ids)

        return _chunked_jsd_loss(
            hidden,
            lm_head_weight,
            None,
            1.0,
            None,
            target_ids,
            torch.tensor(logprobs),
            torch.ones_like(target_ids, dtype=torch.bool),
            0.0,
            1.0,
            True,
            teacher_id_idx,
            self.NUM_TEACHERS,
            chunk_size,
        )

    @pytest.mark.parametrize("chunk_size", [3, 256])
    def test_stats_partition_the_batch_by_teacher(self, chunk_size):
        # [FUNCTION DOC]
        # Parametrized over chunk sizes 3 and 256: per-teacher counts equal the bincount of teacher indices, and
        # per-teacher JSD and entropy sums add up to the overall sums.
        teacher_id_idx = torch.tensor([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
        _, jsd_sum, _, teacher_entropy_sum, per_teacher_stats = self._run(teacher_id_idx, chunk_size)

        counts, jsd_sums, entropy_sums = per_teacher_stats.view(self.NUM_TEACHERS, 3).T
        torch.testing.assert_close(counts, torch.bincount(teacher_id_idx).float())
        torch.testing.assert_close(jsd_sums.sum(), jsd_sum)
        torch.testing.assert_close(entropy_sums.sum(), teacher_entropy_sum)

    def test_chunking_does_not_move_tokens_between_teachers(self):
        # [FUNCTION DOC]
        # Per-teacher stats with chunk size 3 equal those computed in a single chunk.
        teacher_id_idx = torch.tensor([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
        chunked = self._run(teacher_id_idx, 3)[-1]
        whole = self._run(teacher_id_idx, self.NUM_POSITIONS)[-1]

        torch.testing.assert_close(chunked, whole)

    def test_single_teacher_reports_no_breakdown(self):
        # [FUNCTION DOC]
        # With no teacher index tensor the per-teacher stats are all zero.
        per_teacher_stats = self._run(None, 3)[-1]

        assert not per_teacher_stats.any()


@pytest.mark.skipif(
    not is_ampere_or_newer() and torch_device != "xpu" and torch.version.hip is None,
    reason="Flash Attention requires Ampere or newer GPU, XPU, or ROCm",
)
class TestAsyncDistillationTrainer(TrlTestCase):
    # [CLASS DOC]
    # End-to-end trainer tests (skipped without Ampere+ CUDA, XPU or ROCm since Flash Attention is used), using
    # _StubRolloutWorker and _StubWeightTransfer instead of vLLM, on zen conversational_prompt_completion.
    # Inherits TrlTestCase for tmp_dir.
    def test_init_minimal(self):
        # [FUNCTION DOC]
        # Constructs the trainer with only model id, dataset and the two stubs; asserts nothing beyond not raising.
        model_id = "trl-internal-testing/small-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        AsyncDistillationTrainer(
            model=model_id,
            train_dataset=dataset,
            rollout_worker=_StubRolloutWorker(tokenizer, dataset, vocab_size=len(tokenizer)),
            weight_transfer=_StubWeightTransfer(),
        )

    def test_train(self):
        # [FUNCTION DOC]
        # Trains small Qwen2.5 (lr 0.1, batch 3, top-k 4, completion length 8, 5 s server timeout, token budget disabled)
        # and asserts train_loss is logged and every parameter changed.
        model_id = "trl-internal-testing/small-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        training_args = AsyncDistillationConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,
            teacher_top_k=TEACHER_TOP_K,
            max_completion_length=8,
            vllm_server_timeout=5.0,  # short timeout so test fails fast if queue runs dry
            token_budget=-1,  # disable token budgeting (no real vLLM server to query max_model_len from)
            report_to="none",
        )
        trainer = AsyncDistillationTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
            rollout_worker=_StubRolloutWorker(tokenizer, dataset, vocab_size=len(tokenizer)),
            weight_transfer=_StubWeightTransfer(),
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_resume_from_checkpoint(self):
        # [FUNCTION DOC]
        # With tiny Qwen3, trains 2 steps saving every step, asserting ignore_data_skip is forced True and checkpoint-1
        # has trainer_state.json; then builds a new trainer with max_steps 3 and resumes from checkpoint-1 without error.
        # Checks that ignore_data_skip is True and that resume doesn't crash. The stub worker is not an
        # AsyncRolloutWorker, so the checkpoint-write and resume-read paths stay inert here — those are covered by
        # TestRolloutStateCheckpoint.
        # Qwen3 rather than the Qwen2.5 model the tests above use: that one has head_dim=2, which flash-attn rejects
        # ("head_size should be a multiple of 8") on some of its kernels, and this test happens to select one of them.
        model_id = "trl-internal-testing/tiny-Qwen3ForCausalLM"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        def build_trainer(**overrides):
            # [FUNCTION DOC]
            # HELPER: Build an AsyncDistillationTrainer with the shared test config (batch 3, top-k 4, completion length 8,
            # token budget disabled, 5 s timeout) plus overrides, fresh stubs and the tiny Qwen3 model.
            # Inputs: config field overrides. Outputs: AsyncDistillationTrainer.
            args = AsyncDistillationConfig(
                output_dir=self.tmp_dir,
                per_device_train_batch_size=3,
                teacher_top_k=TEACHER_TOP_K,
                max_completion_length=8,
                token_budget=-1,  # FixedCountBatcher, so a step consumes a fixed, predictable number of samples
                vllm_server_timeout=5.0,
                report_to="none",
                **overrides,
            )
            return AsyncDistillationTrainer(
                model=model_id,
                args=args,
                train_dataset=dataset,
                rollout_worker=_StubRolloutWorker(tokenizer, dataset, vocab_size=len(tokenizer)),
                weight_transfer=_StubWeightTransfer(),
            )

        # First run: train for 2 steps, which saves a checkpoint at step 1.
        trainer = build_trainer(max_steps=2, save_steps=1)
        assert trainer.args.ignore_data_skip is True
        trainer.train()

        checkpoint_dir = os.path.join(self.tmp_dir, "checkpoint-1")
        assert os.path.isfile(os.path.join(checkpoint_dir, "trainer_state.json"))

        # Second run: resume from the step-1 checkpoint.
        build_trainer(max_steps=3).train(resume_from_checkpoint=checkpoint_dir)

    @pytest.mark.parametrize("beta", [0.25, 0.5, 0.75, 1.0])
    def test_train_with_nonzero_beta(self, beta):
        # [FUNCTION DOC]
        # Parametrized over beta 0.25, 0.5, 0.75, 1.0: trains small Qwen2.5 and asserts at least one log record has
        # grad_norm and that grad_norm and loss are finite in every such record.
        # Adapted from ServerDistillationTrainer's own `test_reverse_kl_finite_grad_with_ragged_batch`: a real
        # training run through the narrow top-1 + actual-token support path (including the beta==1.0 special case),
        # checking grad_norm/loss stay finite throughout, not just that training completes.
        model_id = "trl-internal-testing/small-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        training_args = AsyncDistillationConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            teacher_top_k=TEACHER_TOP_K,
            max_completion_length=8,
            vllm_server_timeout=5.0,
            token_budget=-1,
            beta=beta,
            report_to="none",
        )
        trainer = AsyncDistillationTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
            rollout_worker=_StubRolloutWorker(tokenizer, dataset, vocab_size=len(tokenizer)),
            weight_transfer=_StubWeightTransfer(),
        )
        trainer.train()

        log_history = [rec for rec in trainer.state.log_history if "grad_norm" in rec]
        assert log_history, "Expected at least one grad_norm log entry during training"
        for record in log_history:
            assert math.isfinite(record["grad_norm"]), f"grad_norm={record['grad_norm']} leaked -inf into backward"
            assert math.isfinite(record["loss"])


def _checkpoint_trainer(output_dir, dataset_start_index=0):
    # [FUNCTION DOC]
    # HELPER: Build a default-config AsyncDistillationTrainer (small Qwen2.5, zen prompt/completion, stubs) and then
    # replace its rollout_worker with a MagicMock specced as AsyncRolloutWorker whose _loop_kwargs holds the given
    # dataset_start_index, so checkpoint/resume code paths (which only run for a real worker type) execute.
    # Inputs: output_dir (str), dataset_start_index (int, default 0). Outputs: the trainer.
    # Exceptions: none expected. Side effects: downloads model/dataset.
    model_id = "trl-internal-testing/small-Qwen2ForCausalLM-2.5"
    dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    trainer = AsyncDistillationTrainer(
        model=model_id,
        args=AsyncDistillationConfig(output_dir=output_dir, report_to="none"),
        train_dataset=dataset,
        rollout_worker=_StubRolloutWorker(tokenizer, dataset, vocab_size=len(tokenizer)),
        weight_transfer=_StubWeightTransfer(),
    )
    # The checkpoint paths only run for an `AsyncRolloutWorker`, which requires vLLM
    trainer.rollout_worker = MagicMock(spec=AsyncRolloutWorker)
    trainer.rollout_worker._loop_kwargs = {"dataset_start_index": dataset_start_index}
    return trainer


@pytest.mark.skipif(
    not is_ampere_or_newer() and torch_device != "xpu" and torch.version.hip is None,
    reason="Flash Attention requires Ampere or newer GPU, XPU, or ROCm",
)
class TestEpochStop(TrlTestCase):
    # [CLASS DOC]
    # Tests _EpochStopCallback: training stops once the number of distinct trained prompt ids plus the prompts trained
    # before a resume reaches the target. GPU-gated like the trainer tests.
    """`num_train_epochs` stops after N full passes over the PROMPTS, counted as distinct prompt ids.

    The count spans a resume: the worker restarts `prompt_id` at 0, so the prompts trained before the checkpoint ride
    in `_prompts_before_resume` and are added back here, or a resumed run would train `num_train_epochs` more passes on
    top of the ones already done.
    """

    @pytest.mark.parametrize(
        ("trained", "before_resume", "should_stop"),
        [
            ({0, 1, 2}, 0, False),
            ({0, 1, 2, 3}, 0, True),
            ({0, 1}, 2, True),  # a resumed run reaches the target counting the checkpoint's prompts
            (set(), 4, True),  # ... and reaches it having trained nothing of its own
        ],
    )
    def test_stops_once_the_prompt_target_is_reached(self, trained, before_resume, should_stop):
        # [FUNCTION DOC]
        # Parametrized over (trained ids, prompts before resume, expected stop) with a target of 4 prompts: sets the
        # trainer's _trained_prompts and _prompts_before_resume, calls on_step_end with a fresh TrainerControl, and
        # asserts should_training_stop and should_save both equal the expected value.
        trainer = _checkpoint_trainer(self.tmp_dir)
        trainer._trained_prompts = trained
        trainer._prompts_before_resume = before_resume
        control = TrainerControl()

        _EpochStopCallback(trainer, target_prompts=4).on_step_end(trainer.args, None, control)

        assert control.should_training_stop is should_stop
        assert control.should_save is should_stop  # the final step is saved, as when training reaches `max_steps`


class TestRolloutStartIndex(TrlTestCase):
    # [CLASS DOC]
    # Tests that the rollout loop's dataset iterator honors dataset_start_index.
    def test_rollout_loop_skips_to_start_index(self):
        # [FUNCTION DOC]
        # With a 10-row dataset and start index 3, the first row yielded by _repeat_iterator has prompt "row_3".
        loop = _rollout_loop(
            dataset=Dataset.from_dict({"prompt": [f"row_{i}" for i in range(10)]}), dataset_start_index=3
        )
        _prompt_id, row = next(loop._repeat_iterator())
        assert row["prompt"] == "row_3"


@pytest.mark.skipif(
    not is_ampere_or_newer() and torch_device != "xpu" and torch.version.hip is None,
    reason="Flash Attention requires Ampere or newer GPU, XPU, or ROCm",
)
class TestRolloutStateCheckpoint(TrlTestCase):
    # [CLASS DOC]
    # Tests saving rollout_state.json in checkpoints and restoring prompt index / model version on resume, using
    # _checkpoint_trainer and patching the base trainer's checkpoint and inner-loop methods. GPU-gated.
    """Prompt-index checkpoint/resume logic, on a real trainer whose rollout worker is stood in (it needs vLLM)."""

    def _trainer_at_step_5(self, trained_prompts, dataset_start_index=0):
        # [FUNCTION DOC]
        # HELPER: Build a checkpoint trainer with the given start index, set global_step to 5 and _trained_prompts to the
        # given set. Outputs: the trainer.
        trainer = _checkpoint_trainer(self.tmp_dir, dataset_start_index)
        trainer.state.global_step = 5
        trainer._trained_prompts = trained_prompts
        return trainer

    @pytest.mark.parametrize(
        ("trained_prompts", "dataset_start_index", "expected"),
        [
            ({0, 1, 2, 3, 4}, 10, 15),  # dataset_start_index(10) + first_untrained(5)
            # Prompt 2 was never trained (its rollout was dropped as stale), so the cursor stops there: it gets
            # re-generated on resume instead of being silently skipped.
            ({0, 1, 3, 4, 5}, 0, 2),
        ],
    )
    def test_save_checkpoint_writes_rollout_state(self, trained_prompts, dataset_start_index, expected):
        # [FUNCTION DOC]
        # Parametrized: the saved prompt_index equals the start index plus the first untrained prompt id (10 + 5 = 15 for
        # contiguous ids; 2 when prompt 2 is missing). The base _save_checkpoint is patched out; reads
        # checkpoint-5/rollout_state.json.
        trainer = self._trainer_at_step_5(trained_prompts, dataset_start_index=dataset_start_index)

        with patch.object(_BaseTrainer, "_save_checkpoint"):
            trainer._save_checkpoint(trainer.model, None)

        with open(os.path.join(self.tmp_dir, "checkpoint-5", "rollout_state.json")) as f:
            assert json.load(f)["prompt_index"] == expected

    def test_save_checkpoint_writes_rollout_state_before_the_hub_push(self):
        # [FUNCTION DOC]
        # Patches the base _save_checkpoint with a recorder that checks whether rollout_state.json already exists; asserts
        # it did exist when the base method (which pushes to the Hub) ran, exactly once.
        # `super()._save_checkpoint` is what uploads the checkpoint folder under `hub_strategy="checkpoint"`, so the
        # file has to exist by the time it runs. Writing it from an `on_save` callback would not: `Trainer` fires
        # `on_save` only after `_save_checkpoint` returns, leaving the Hub copy without it.
        trainer = self._trainer_at_step_5({0, 1})
        written_before_super = []

        def record(*_args, **_kwargs):
            # [FUNCTION DOC]
            # HELPER: Append whether checkpoint-5/rollout_state.json exists to the enclosing list; ignores its arguments.
            written_before_super.append(
                os.path.isfile(os.path.join(self.tmp_dir, "checkpoint-5", "rollout_state.json"))
            )

        with patch.object(_BaseTrainer, "_save_checkpoint", side_effect=record):
            trainer._save_checkpoint(trainer.model, None)

        assert written_before_super == [True]

    def test_inner_training_loop_sets_dataset_start_index_from_file(self):
        # [FUNCTION DOC]
        # Writes a checkpoint-10 rollout_state.json with prompt_index 77 and model_version 42, calls _inner_training_loop
        # with the base implementation patched out, and asserts: the worker's dataset_start_index is 77,
        # _prompts_before_resume is 77, _trained_prompts is reset to empty, model_version is 42, and the worker's
        # update_model_version was called once with 42.
        checkpoint_dir = os.path.join(self.tmp_dir, "checkpoint-10")
        os.makedirs(checkpoint_dir)
        with open(os.path.join(checkpoint_dir, "rollout_state.json"), "w") as f:
            json.dump({"prompt_index": 77, "model_version": 42}, f)

        trainer = _checkpoint_trainer(self.tmp_dir)
        trainer._trained_prompts = {1, 2, 3}  # a prior run's ids, which the reset has to drop

        with patch.object(_BaseTrainer, "_inner_training_loop", return_value=None):
            trainer._inner_training_loop(resume_from_checkpoint=checkpoint_dir)

        assert trainer.rollout_worker._loop_kwargs["dataset_start_index"] == 77
        # Epochs are counted in prompts trained, so a resumed run has to pick that count up too, or it would train
        # `num_train_epochs` more passes on top of the ones already done.
        assert trainer._prompts_before_resume == 77
        # The worker restarts `prompt_id` at 0, so ids left over from an earlier run would collide with this one's.
        assert trainer._trained_prompts == set()
        # Adapter names are derived from `model_version`, so restarting it at 0 would republish a different adapter
        # under a name a still-running vLLM server already holds, and strand the previous run's directories outside
        # the eviction window.
        assert trainer.model_version == 42
        trainer.rollout_worker.update_model_version.assert_called_once_with(42)

    def test_inner_training_loop_defaults_model_version_for_old_checkpoints(self):
        # [FUNCTION DOC]
        # A rollout_state.json without model_version resets a preset model_version of 7 back to 0 on resume.
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


@pytest.mark.skipif(
    not is_ampere_or_newer() and torch_device != "xpu" and torch.version.hip is None,
    reason="Flash Attention requires Ampere or newer GPU, XPU, or ROCm",
)
@require_peft
class TestAsyncDistillationTrainerPeft(TrlTestCase):
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
        # Behavior: loads the zen dataset and tokenizer; defaults max_staleness to 1; config uses lr 0.1, batch 3, top-k
        # 4, completion length 8, token budget disabled, vllm_server_base_url pointing at the fake, 5 s timeout. The
        # rollout worker is a fresh stub when "stub" is passed, otherwise passed through (None makes the trainer build a
        # real AsyncRolloutWorker). weight_transfer None lets the trainer own weight sync and probe the server.
        # Inputs: fake, peft_config, lora_config, data_parallel_size, weight_transfer, rollout_worker, config overrides.
        # Outputs: AsyncDistillationTrainer. Exceptions: propagates trainer construction errors.
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
        args = AsyncDistillationConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,
            teacher_top_k=TEACHER_TOP_K,
            max_completion_length=8,
            token_budget=-1,  # disable token budgeting (no real vLLM server to query max_model_len from)
            vllm_server_base_url=fake.url,
            vllm_server_timeout=5.0,
            report_to="none",
            **config_kwargs,
        )
        if rollout_worker == "stub":
            rollout_worker = _StubRolloutWorker(tokenizer, dataset, vocab_size=len(tokenizer))
        return AsyncDistillationTrainer(
            model=self.model_id,
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
        # The chunked divergence path reads `lm_head.weight`, which on a PEFT-wrapped head is the base weight: the
        # adapter delta would be silently dropped from every student logprob. The regex case is why the check is on
        # the module and not on `target_modules` — that string never contains "lm_head".
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
        # model, silently, and distilled against the teacher as if it were the student. The real worker is built here
        # (it spawns nothing until `start()`) to read the `lora_name` the trainer hands it.
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
        # In-flight requests may still name any of the last `max_staleness + 1` versions; vLLM resolves `lora_path`
        # lazily inside the engine, so deleting one of those directories kills the engine with an `HFValidationError`
        # rather than 404ing.
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
