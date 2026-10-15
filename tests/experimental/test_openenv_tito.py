# [MODULE DOC]
# Tests for the OpenEnv harness integration of the async GRPO rollout worker
# (trl.experimental.async_grpo.openenv_harness plus _chain_to_sequences and _SampleBuilder from
# async_rollout_worker). Theme: captured token contexts ("token in, token out") and eligible sampled
# tokens must survive trainer ingestion exactly: loss masks, logprobs and rewritten prompt tails are
# preserved; invalid captures fail loudly via CaptureContractError and the worker failure channel;
# transport failures produce empty rollouts; sessions are always closed; tool metrics are reported.
# Contains small entry builders, a fake Session, and a make_loop fixture.
# Import-time side effects: skips the whole module unless openenv.core.harness.training is
# importable. KNOWN ISSUE: relies heavily on MagicMock, SimpleNamespace and monkeypatched __init__ /
# internals, contrary to the repository test guideline of exercising real objects.

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

"""Captured contexts and eligible sampled tokens must survive trainer ingestion exactly."""

import asyncio
import itertools
import queue
import threading
import time
from collections import defaultdict
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from accelerate import PartialState


pytest.importorskip("openenv.core.harness.training")

from openenv.core.harness import HarnessRolloutResult, ToolResult, ToolTraceEntry, TrainingTrace

from trl.experimental.async_grpo import openenv_harness
from trl.experimental.async_grpo.async_rollout_worker import _chain_to_sequences, _SampleBuilder


def entry(prompt, completion, logprobs, mask=None):
    # [FUNCTION DOC]
    # HELPER: Builds a training-trace entry dict with prompt_token_ids, completion_token_ids,
    # per_token_logps, and loss_mask (default: zeros over the prompt then ones over the completion).
    # Inputs: prompt, completion (lists of int), logprobs (list or None), mask (optional list).
    # Outputs: dict. Exceptions: TypeError if mask is None and prompt/completion lack len.
    # Side effects: none.
    return {
        "prompt_token_ids": prompt,
        "completion_token_ids": completion,
        "per_token_logps": logprobs,
        "loss_mask": mask if mask is not None else [0] * len(prompt) + [1] * len(completion),
    }


def convert(entries):
    # [FUNCTION DOC]
    # HELPER: Sets metadata node_id to the row index as a string on each entry (mutating them), builds a
    # TrainingTrace from the entries, and converts it to turns with _turns_from_training_trace.
    # Inputs: entries (list of dict). Outputs: list of turns.
    # Exceptions: ValueError from trace validation/conversion. Side effects: mutates entries.
    for index, row in enumerate(entries):
        row.setdefault("metadata", {})["node_id"] = str(index)
    return openenv_harness._turns_from_training_trace(TrainingTrace.from_entries(entries))


def test_lossless_capture_preserves_rewritten_tail_and_partial_mask():
    # [FUNCTION DOC]
    # Tests two turns where the second prompt rewrites a token of the first completion (5 becomes 99):
    # _chain_to_sequences yields 2 rows without realignment, row 0 keeps tokens and the partial mask
    # 0,0,1,0,1 with its logprobs at the tail, row 1 has the rewritten context and a single supervised
    # token, and total supervised tokens is 3.
    # Inputs: none. Outputs: none. Exceptions: AssertionError on failure. Side effects: none.
    entries = [
        entry([1, 2], [3, 4, 5], [-0.1, -0.2, -0.3], [0, 0, 1, 0, 1]),
        entry([1, 2, 3, 4, 99, 6], [7], [-0.4], [0, 0, 0, 0, 0, 0, 1]),
    ]
    turns = convert(entries)
    rows, tally = _chain_to_sequences(turns, "rollout", 0)
    assert len(rows) == 2 and tally["realign"] == 0
    assert rows[0].input_ids == [1, 2, 3, 4, 5]
    assert rows[0].completion_mask == [0, 0, 1, 0, 1]
    assert rows[0].old_log_probs[-3:] == [-0.1, -0.2, -0.3]
    assert rows[1].input_ids == [1, 2, 3, 4, 99, 6, 7]
    assert rows[1].completion_mask == [0, 0, 0, 0, 0, 0, 1]
    assert sum(sum(row.completion_mask) for row in rows) == 3


@pytest.mark.parametrize("logprobs", [[], None, [float("nan")], [float("inf")], [0.1], [True]])
def test_missing_or_invalid_sample_logprobs_cannot_be_filled_with_zeros(logprobs):
    # [FUNCTION DOC]
    # Parametrized over empty, None, NaN, inf, positive, and boolean logprobs: converting an entry with
    # them raises ValueError (never silently zero-filled).
    # Inputs: logprobs. Outputs: none. Exceptions: AssertionError on failure. Side effects: none.
    with pytest.raises(ValueError):
        convert([entry([1], [2], logprobs)])


@pytest.mark.parametrize("mask,logprobs", [(1, None), (1, []), ([1, 0], [-0.1]), (0, [])])
def test_builder_rejects_misaligned_arrays_before_mutation(mask, logprobs):
    # [FUNCTION DOC]
    # Parametrized over mismatched mask/logprobs combos: _SampleBuilder append raises ValueError and the
    # builder's tokens, loss_mask, logprobs remain empty (validation before mutation).
    # Inputs: mask, logprobs. Outputs: none. Exceptions: AssertionError on failure. Side effects: none.
    builder = _SampleBuilder(fork_threshold=0)
    with pytest.raises(ValueError):
        builder._append([2], mask=mask, logprobs=logprobs)
    assert builder.tokens == builder.loss_mask == builder.logprobs == []


@pytest.mark.parametrize("mask", [[1, 1], [0], [0, 2], [0, True]])
def test_consumer_rejects_invalid_full_masks(mask):
    # [FUNCTION DOC]
    # Parametrized over invalid full loss masks (supervised prompt token, wrong length, value 2, boolean):
    # conversion raises ValueError.
    # Inputs: mask. Outputs: none. Exceptions: AssertionError on failure. Side effects: none.
    with pytest.raises(ValueError):
        convert([entry([1], [2], [-0.1], mask)])


@pytest.mark.parametrize(
    "adapter,lossless,threshold", [(None, True, 0), (None, False, 1024), (SimpleNamespace(), True, 1024)]
)
def test_loop_owning_capture_defaults_to_lossless_reconciliation(monkeypatch, adapter, lossless, threshold):
    # [FUNCTION DOC]
    # Parametrized over (adapter, lossless_capture, expected fork threshold). Monkeypatches the base
    # _AsyncRolloutLoop init with a minimal one that records fork_threshold_tokens (default 1024) and
    # sampling fields. Asserts _HarnessRolloutLoop sets _fork_threshold_tokens to 0 when lossless and
    # loop-owning (no adapter), otherwise 1024; and that the session factory is called with no args
    # when an adapter is given, else with sampling equal to training_sampling at temperature 0.8.
    # Shuts down the session pool afterwards.
    # Inputs: monkeypatch, adapter, lossless, threshold. Outputs: none. Exceptions: AssertionError.
    # Side effects: temporary monkeypatch; thread pool created and shut down.
    def init(self, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Replacement base __init__ setting _fork_threshold_tokens from kwargs (default 1024),
        # max_tool_calling_iterations 8, temperature 0.8, top_p 1.0, top_k -1, min_p 0.0,
        # repetition_penalty 1.0, max_tokens 32, empty reward_func_names, max_inflight_tasks 1.
        # Inputs: self, kwargs. Outputs: none. Exceptions: none. Side effects: sets attributes.
        self._fork_threshold_tokens = kwargs.get("fork_threshold_tokens", 1024)
        self.max_tool_calling_iterations = 8
        self.temperature = 0.8
        self.top_p = 1.0
        self.top_k = -1
        self.min_p = 0.0
        self.repetition_penalty = 1.0
        self.max_tokens = 32
        self.reward_func_names = []
        self.max_inflight_tasks = 1

    monkeypatch.setattr(openenv_harness._AsyncRolloutLoop, "__init__", init)
    factory = MagicMock(return_value=SimpleNamespace())
    loop = openenv_harness._HarnessRolloutLoop(
        harness_session_factory=factory, harness_adapter=adapter, lossless_capture=lossless
    )
    try:
        assert loop._fork_threshold_tokens == threshold
        if adapter is not None:
            factory.assert_called_once_with()
        else:
            factory.assert_called_once_with(sampling=openenv_harness.training_sampling({"temperature": 0.8}))
    finally:
        loop._session_pool.shutdown()


@pytest.fixture
def make_loop():
    # [FUNCTION DOC]
    # HELPER: Pytest fixture yielding a factory that builds real _HarnessRolloutLoop instances with a
    # MagicMock tokenizer (eos/pad 0), a one-row dataset, no reward funcs, fresh queues/events, model
    # name "test", max_inflight_tasks 8, temperature defaulting to 0.8. Initializes accelerate
    # PartialState first. Teardown shuts down each loop's session pool (waiting) and closes its asyncio
    # loop. Inputs: none. Outputs: yields the make callable.
    # Exceptions: propagates construction errors. Side effects: threads, event loops.
    PartialState()
    loops = []

    def make(factory=None, *, factory_builder=None, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Constructs a loop; the session factory is factory_builder if given, else a callable
        # returning factory regardless of kwargs. Extra kwargs are forwarded; the loop is tracked for
        # teardown. Inputs: factory (optional), factory_builder (optional, keyword), kwargs.
        # Outputs: _HarnessRolloutLoop. Exceptions: propagates constructor errors.
        # Side effects: appends to the fixture's loop list.
        kwargs.setdefault("temperature", 0.8)
        tokenizer = MagicMock(eos_token_id=0, pad_token_id=0)
        loop = openenv_harness._HarnessRolloutLoop(
            harness_session_factory=factory_builder or (lambda **kwargs: factory),
            model_name="test",
            dataset=[{"prompt": [{"role": "user", "content": "task"}]}],
            reward_funcs=[],
            processing_class=tokenizer,
            eos_token_ids=[0],
            rollout_buffer=queue.Queue(),
            model_version_value=SimpleNamespace(value=0),
            heartbeat_value=SimpleNamespace(value=0.0),
            failed_event=threading.Event(),
            exception_info_queue=queue.Queue(),
            metrics_queue=queue.Queue(),
            max_inflight_tasks=8,
            **kwargs,
        )
        loops.append(loop)
        return loop

    yield make
    for loop in loops:
        loop._session_pool.shutdown(wait=True)
        loop._loop.close()


def captured_entry():
    # [FUNCTION DOC]
    # HELPER: Returns a valid captured entry (prompt [1], completion [2, 3], logprobs -0.1/-0.2, mask
    # 0,1,0) with a user "task" request, an assistant "answer" response, and metadata node_id 0 plus
    # sampling_params from training_sampling at temperature 0.8.
    # Inputs: none. Outputs: dict. Exceptions: none. Side effects: none.
    row = entry([1], [2, 3], [-0.1, -0.2], [0, 1, 0])
    row.update(
        request={"messages": [{"role": "user", "content": "task"}]},
        response={"choices": [{"message": {"role": "assistant", "content": "answer"}}]},
        metadata={"node_id": "0", "sampling_params": openenv_harness.training_sampling({"temperature": 0.8})},
    )
    return row


class Session:
    # [CLASS DOC]
    # HELPER: Fake harness session returning a single-entry TrainingTrace from a stored row, completing
    # immediately, verifying with env_reward 0.25, and recording close via a threading Event.
    def __init__(self, row):
        # [FUNCTION DOC]
        # HELPER: Stores row and creates the closed Event. Inputs: row (dict). Outputs: none.
        # Exceptions: none. Side effects: sets attributes.
        self.row = row
        self.closed = threading.Event()

    def wait_for_completion(self):
        # [FUNCTION DOC]
        # HELPER: Returns 0 immediately. Inputs: none. Outputs: int. Exceptions: none. Side effects: none.
        return 0

    def fetch_training_trace(self):
        # [FUNCTION DOC]
        # HELPER: Copies the row, forcing metadata node_id "0" (merging existing metadata), and returns a
        # TrainingTrace built from it. Inputs: none. Outputs: TrainingTrace.
        # Exceptions: validation errors from TrainingTrace. Side effects: none.
        row = {**self.row, "metadata": {**self.row.get("metadata", {}), "node_id": "0"}}
        return TrainingTrace.from_entries([row])

    def verify(self, completion):
        # [FUNCTION DOC]
        # HELPER: Returns an object with env_reward 0.25. Inputs: completion (ignored). Outputs:
        # SimpleNamespace. Exceptions: none. Side effects: none.
        return SimpleNamespace(env_reward=0.25)

    def close(self):
        # [FUNCTION DOC]
        # HELPER: Sets the closed Event. Inputs: none. Outputs: None. Exceptions: none.
        # Side effects: event set.
        self.closed.set()


@pytest.mark.parametrize("temperature", [0.3, 0.8, 1.2])
def test_worker_supplies_factory_sampling_before_session_start(make_loop, monkeypatch, temperature):
    # [FUNCTION DOC]
    # Parametrized over temperatures. Requires harbor_env.harness. Patches HarborSession construction
    # and the factory client, builds a HarborSessionFactory (initial temperature 1.0) via make_loop with
    # the given temperature, and asserts the factory's sampling was replaced by the training policy
    # before any session; seeds the factory task tables, runs one session (row without metadata), and
    # asserts the session was constructed with that policy, the result is non-empty, and the session
    # was closed.
    # Inputs: make_loop, monkeypatch, temperature. Outputs: none. Exceptions: AssertionError.
    # Side effects: monkeypatches harness module attributes.
    harness = pytest.importorskip("harbor_env.harness")
    policy = openenv_harness.training_sampling({"temperature": temperature})
    row = captured_entry()
    # Token ingestion no longer depends on redundant policy metadata.
    row.pop("metadata")
    session = Session(row)
    construct_session = MagicMock(return_value=session)
    monkeypatch.setattr(harness, "HarborSession", construct_session)
    monkeypatch.setattr(harness.HarborSessionFactory, "new_client", lambda self: MagicMock())
    builder = partial(harness.HarborSessionFactory, "http://unused", sampling={"temperature": 1.0})
    loop = make_loop(factory_builder=builder, temperature=temperature)
    assert loop._factory.sampling == policy
    loop._factory._tasks = [{"instruction": "task", "index": 0}]
    loop._factory._by_instruction = {harness.instruction_id("task"): 0}
    result, _ = loop._run_session([{"role": "user", "content": "task"}])
    assert construct_session.call_args.kwargs["sampling"] == policy
    assert result != loop._EMPTY_ROLLOUT
    assert session.closed.is_set()


@pytest.mark.parametrize("field,value", [("prompt_token_ids", []), ("per_token_logps", []), ("loss_mask", [1, 1, 1])])
def test_invalid_capture_reaches_worker_failure_channel(make_loop, field, value):
    # [FUNCTION DOC]
    # Parametrized over corrupting prompt_token_ids, per_token_logps, or loss_mask. Running the loop
    # raises CaptureContractError; the failed event is set; the exception queue holds name
    # CaptureContractError, a message, and a traceback mentioning fetch_training_trace; the session is
    # closed; and nothing reached the rollout buffer.
    # Inputs: make_loop, field, value. Outputs: worker threads run. Exceptions: AssertionError on failure. Side effects: worker threads run.
    row = captured_entry()
    row[field] = value
    session = Session(row)
    loop = make_loop(SimpleNamespace(create=lambda *args, **kwargs: session))
    with pytest.raises(openenv_harness.CaptureContractError):
        loop.run()
    assert loop._failed_event.is_set()
    name, message, traceback = loop._exception_info_queue.get_nowait()
    assert name == "CaptureContractError" and message and "fetch_training_trace" in traceback
    assert session.closed.is_set()
    assert loop.rollout_buffer.empty()


def test_producer_validation_error_is_fatal(make_loop):
    # [FUNCTION DOC]
    # Tests a ValueError from fetch_training_trace (fatal validation finding) becomes a
    # CaptureContractError matching "fatal validation" from _run_session, and the session is closed.
    # Inputs: make_loop fixture. Outputs: none. Exceptions: AssertionError on failure.
    # Side effects: runs the loop's worker threads/event loop; loops cleaned up by the fixture.
    session = Session(captured_entry())
    session.fetch_training_trace = MagicMock(
        side_effect=ValueError("cannot train a capture with fatal validation findings")
    )
    loop = make_loop(SimpleNamespace(create=lambda *args, **kwargs: session))
    with pytest.raises(openenv_harness.CaptureContractError, match="fatal validation"):
        loop._run_session([])
    assert session.closed.is_set()


@pytest.mark.parametrize("phase", ["create", "wait_for_completion", "fetch_training_trace", "verify"])
def test_transport_failures_remain_unscorable(make_loop, phase):
    # [FUNCTION DOC]
    # Parametrized over failing phase create, wait_for_completion, fetch_training_trace, verify (each
    # raising ConnectionError). _run_session returns the empty rollout with None metrics, and the
    # session is closed whenever it was created.
    # Inputs: make_loop, phase. Outputs: none. Exceptions: AssertionError. Side effects: loop threads.
    session = Session(captured_entry())
    factory = SimpleNamespace(create=lambda *args, **kwargs: session)
    target = factory if phase == "create" else session
    setattr(target, phase, MagicMock(side_effect=ConnectionError("sandbox unavailable")))
    loop = make_loop(factory)
    assert loop._run_session([]) == (loop._EMPTY_ROLLOUT, None)
    if phase != "create":
        assert session.closed.is_set()


def test_timeout_preserves_verifier_reward_and_tokens(make_loop):
    # [FUNCTION DOC]
    # Tests a TimeoutError in wait_for_completion still yields the verifier reward 0.25, the captured
    # row (tokens 1,2,3 and mask 0,1,0), and a loop_exhausted metric.
    # Inputs: make_loop fixture. Outputs: none. Exceptions: AssertionError on failure.
    # Side effects: runs the loop's worker threads/event loop; loops cleaned up by the fixture.
    session = Session(captured_entry())
    session.wait_for_completion = MagicMock(side_effect=TimeoutError("agent budget"))
    loop = make_loop(SimpleNamespace(create=lambda *args, **kwargs: session))
    result, metrics = loop._run_session([])
    assert result[-1] == 0.25
    assert result[2][0].input_ids == [1, 2, 3]
    assert result[2][0].completion_mask == [0, 1, 0]
    assert metrics["loop_exhausted"]


def test_capture_error_closes_other_inflight_sessions(make_loop):
    # [FUNCTION DOC]
    # Runs two sessions concurrently: a blocking one waiting up to 3s for its own close, and a broken one
    # (empty prompt ids) that waits until the first has started. Asserts run raises
    # CaptureContractError, both sessions are closed, the blocking wait was released by close (True),
    # and the stop event is set.
    # Inputs: make_loop fixture. Outputs: none. Exceptions: AssertionError on failure.
    # Side effects: runs the loop's worker threads/event loop; loops cleaned up by the fixture.
    started = threading.Event()
    blocking = Session(captured_entry())
    released = []

    def wait():
        # [FUNCTION DOC]
        # HELPER: Signals started, then blocks up to 3s on the blocking session's closed event, appending
        # whether it was set. Inputs: none. Outputs: None. Exceptions: none.
        # Side effects: sets event, appends to released.
        started.set()
        released.append(blocking.closed.wait(3))

    blocking.wait_for_completion = wait
    broken = Session(captured_entry())
    broken.row["prompt_token_ids"] = []
    broken.wait_for_completion = lambda: started.wait(3)
    sessions = iter([blocking, broken])
    loop = make_loop(SimpleNamespace(create=lambda *args, **kwargs: next(sessions)))
    with pytest.raises(openenv_harness.CaptureContractError):
        loop.run()
    assert blocking.closed.is_set() and broken.closed.is_set()
    assert released == [True]
    assert loop._stop_event.is_set()


def test_concurrent_correctness_metrics_stay_on_event_loop(make_loop):
    # [FUNCTION DOC]
    # Replaces the loop's _rates with a defaultdict subclass asserting access only from the creating
    # thread, then runs 32 concurrent _generate_one calls with a reward function adding 0.3 to the env
    # reward. Asserts each result has reward 0.55 and rows, and the 32 metric payloads' correctness
    # counts sum to 8.0 correct of 32 total.
    # Inputs: make_loop fixture. Outputs: none. Exceptions: AssertionError on failure.
    # Side effects: runs the loop's worker threads/event loop; loops cleaned up by the fixture.
    owner = threading.get_ident()

    class Rates(defaultdict):
        # [CLASS DOC]
        # HELPER: defaultdict that asserts every item lookup happens on the owner (test) thread.
        def __getitem__(self, key):
            # [FUNCTION DOC]
            # HELPER: Asserts the current thread is the owner, then delegates to defaultdict lookup.
            # Inputs: key. Outputs: value. Exceptions: AssertionError from a foreign thread.
            # Side effects: may insert a default.
            assert threading.get_ident() == owner, "metric accumulator accessed from a session thread"
            return super().__getitem__(key)

    loop = make_loop(
        SimpleNamespace(create=lambda *args, **kwargs: Session(captured_entry())),
        rollout_reward_fn=lambda outcome: outcome.env_reward + 0.3,
    )
    loop._rates = Rates(lambda: [0.0, 0.0])

    async def run():
        # [FUNCTION DOC]
        # HELPER: Async; gathers 32 concurrent _generate_one calls with group ids 0..31.
        # Inputs: none. Outputs: list of results. Exceptions: propagates. Side effects: loop metrics queued.
        return await asyncio.gather(*(loop._generate_one([], {}, [], group_id=i) for i in range(32)))

    results = asyncio.run(run())
    assert all(result[-1] == 0.55 and result[2] for result in results)
    metrics = [loop._metrics_queue.get_nowait() for _ in results]
    assert sum(item["rollout/correctness_mean"][0] for item in metrics) == 8.0
    assert sum(item["rollout/correctness_mean"][1] for item in metrics) == 32


def test_white_box_tool_metrics_reach_the_queue_per_tool(make_loop, monkeypatch):
    # [FUNCTION DOC]
    # Tests white-box adapter mode: parse_response is patched to return a turn with bash and edit tool
    # calls then an empty turn; a local Adapter runs both tools (edit fails) with 20ms sleeps between.
    # Asserts usage positions 3-4 equal (2, 1), per-tool call totals of 1, an edit failure but no bash
    # failure, total tool latency count 2 with at least 40ms, and per-tool latencies recorded.
    # Inputs: make_loop, monkeypatch. Outputs: none. Exceptions: AssertionError.
    # Side effects: monkeypatch; sleeps.
    """The adapter runs the tools between two model turns; that gap is logged as their latency, by name."""
    call = {"type": "function", "function": {"name": "bash", "arguments": {}}}
    replies = iter([{"role": "assistant", "tool_calls": [call, {**call, "function": {"name": "edit"}}]}, {}])
    monkeypatch.setattr(openenv_harness, "parse_response", lambda *args, **kwargs: next(replies))

    class Adapter:
        # [CLASS DOC]
        # HELPER: Minimal white-box harness adapter driving two model steps and producing a tool trace.
        def run_white_box(self, step, session, limits):
            # [FUNCTION DOC]
            # HELPER: Calls step once, for each returned tool call sleeps 20ms and records a ToolTraceEntry
            # (error "boom" for edit), calls step again, and returns a HarnessRolloutResult with a final
            # assistant message and the trace. Inputs: step, session, limits. Outputs: HarnessRolloutResult.
            # Exceptions: propagates step errors. Side effects: sleeps.
            trace = []
            for tc in step([{"role": "user", "content": "task"}], [], {}).response.tool_calls:
                time.sleep(0.02)
                trace.append(ToolTraceEntry(tc.name, tc.args, ToolResult(error="boom" if tc.name == "edit" else None)))
            step([{"role": "user", "content": "task"}], [], {})
            return HarnessRolloutResult(messages=[{"role": "assistant", "content": "done"}], tool_trace=trace)

    loop = make_loop(
        SimpleNamespace(create=lambda *args, **kwargs: Session(captured_entry())), harness_adapter=Adapter()
    )
    loop.tokenizer.apply_chat_template = MagicMock(return_value=[1])

    async def generate_one_turn(prompt_ids):
        # [FUNCTION DOC]
        # HELPER: Async stub returning completion ids 2,3 and logprobs -0.1,-0.2. Inputs: prompt_ids.
        # Outputs: tuple of lists. Exceptions: none. Side effects: none.
        return [2, 3], [-0.1, -0.2]

    loop._generate_one_turn = generate_one_turn
    result = loop._loop.run_until_complete(loop._generate_one([], {}, [], group_id=0))
    assert result[3:5] == (2, 1)
    payload = loop._metrics_queue.get_nowait()
    assert payload["tools/bash/call_total"] == 1 and payload["tools/edit/call_total"] == 1
    assert payload["tools/edit/failure_total"] == 1 and "tools/bash/failure_total" not in payload
    assert payload["tools/latency_s"][1] == 2 and payload["tools/latency_s"][0] >= 0.04
    assert payload["tools/bash/latency_s"][1] == 1 and payload["tools/edit/latency_s"][0] >= 0.02


def test_loop_owning_tool_counts_resolve_names_through_call_ids(make_loop):
    # [FUNCTION DOC]
    # Tests loop-owning capture where tool messages carry only tool_call_id: the failing edit result
    # (content starting with Traceback) is attributed via call ids. Asserts usage (1, 1), bash call
    # total 1, edit failure total 1, and no latency metric.
    # Inputs: make_loop fixture. Outputs: none. Exceptions: AssertionError on failure.
    # Side effects: runs the loop's worker threads/event loop; loops cleaned up by the fixture.
    """Captured tool results carry `tool_call_id`, not a name; failures still land on the right tool."""
    row = captured_entry()
    row["request"]["messages"] = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "a", "function": {"name": "bash"}}, {"id": "b", "function": {"name": "edit"}}],
        },
        {"role": "tool", "tool_call_id": "a", "content": "ok"},
        {"role": "tool", "tool_call_id": "b", "content": "Traceback: boom"},
    ]
    row["response"]["choices"][0]["message"]["tool_calls"] = [{"id": "c", "function": {"name": "bash"}}]
    loop = make_loop(SimpleNamespace(create=lambda *args, **kwargs: Session(row)))
    result = loop._loop.run_until_complete(loop._generate_one([], {}, [], group_id=0))
    assert result[3:5] == (1, 1)
    payload = loop._metrics_queue.get_nowait()
    assert payload["tools/bash/call_total"] == 1 and payload["tools/edit/failure_total"] == 1
    assert "tools/latency_s" not in payload


def test_harness_worker_produces_scored_training_samples(make_loop):
    # [FUNCTION DOC]
    # Runs the loop's main _run_loops until 8 rows are buffered (5s timeout) with sessions alternating
    # rewards 0 and 1. Asserts all rows share group 0, keep tokens 1,2,3 / mask 0,1,0, have old logprobs
    # 0.0,-0.1,-0.2 (prompt padded with 0.0), and advantages straddle zero.
    # Inputs: make_loop fixture. Outputs: none. Exceptions: AssertionError on failure.
    # Side effects: runs the loop's worker threads/event loop; loops cleaned up by the fixture.
    indices = itertools.count()

    def create(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Session factory returning a fake Session whose verify yields reward alternating 0.0 and
        # 1.0 by creation order. Inputs: args, kwargs (ignored). Outputs: Session. Exceptions: none.
        # Side effects: advances the counter.
        reward = float(next(indices) % 2)
        session = Session(captured_entry())
        session.verify = lambda completion: SimpleNamespace(env_reward=reward)
        return session

    loop = make_loop(SimpleNamespace(create=create))

    async def run():
        # [FUNCTION DOC]
        # HELPER: Async driver: starts _run_loops as a task, waits for the buffer to reach 8, then sets the
        # stop event and awaits the runner (5s timeouts each). Inputs: none. Outputs: None.
        # Exceptions: TimeoutError; runner exceptions are re-raised. Side effects: sets stop event.
        runner = asyncio.create_task(loop._run_loops(loop._stop_event))

        async def wait_for_group():
            # [FUNCTION DOC]
            # HELPER: Async poll every 10ms until the rollout buffer has 8 items, awaiting the runner early if it
            # finished (to surface its exception). Inputs: none. Outputs: None. Exceptions: runner errors.
            # Side effects: none.
            while loop.rollout_buffer.qsize() < 8:
                if runner.done():
                    await runner
                await asyncio.sleep(0.01)

        try:
            await asyncio.wait_for(wait_for_group(), timeout=5)
        finally:
            loop._stop_event.set()
            await asyncio.wait_for(runner, timeout=5)

    loop._loop.run_until_complete(run())
    rows = [loop.rollout_buffer.get_nowait() for _ in range(8)]
    assert {row.group_id for row in rows} == {0}
    assert all(row.input_ids == [1, 2, 3] and row.completion_mask == [0, 1, 0] for row in rows)
    assert all(row.old_log_probs == [0.0, -0.1, -0.2] for row in rows)
    assert min(row.advantage for row in rows) < 0 < max(row.advantage for row in rows)


def test_harbor_producer_defaults_and_partial_masks_round_trip():
    # [FUNCTION DOC]
    # Requires harbor_env.harness. Tests a HarborSessionFactory's sampling normalizes to the training
    # policy (top_k -1 maps to 0, repetition_penalty 1.0) and that a Harbor turn with partial mask
    # converts via to_training_trace and _chain_to_sequences preserving tokens, mask 0,0,1,0,1, and
    # tail logprobs.
    # Inputs: none. Outputs: none. Exceptions: AssertionError on failure. Side effects: none.
    harness = pytest.importorskip("harbor_env.harness")
    from openenv.harbor.contract import to_training_trace
    from openenv.harbor.models import HarborRolloutResult, HarborTurn

    factory = harness.HarborSessionFactory("http://unused", sampling={"temperature": 0.8, "top_p": 1.0, "top_k": -1})
    result = HarborRolloutResult(
        turns=[
            HarborTurn(
                turn=0,
                node_id="call-0",
                prompt_token_ids=[10, 11],
                completion_token_ids=[12, 13, 14],
                per_token_logps=[-0.1, -0.2, -0.3],
                loss_mask=[0, 0, 1, 0, 1],
                request_messages=[{"role": "user", "content": "task"}],
                sampling_params=factory.sampling,
            )
        ]
    )
    policy = openenv_harness.training_sampling(
        {"temperature": 0.8, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}
    )
    trace = to_training_trace(result)
    assert factory.sampling == policy
    turns = openenv_harness._turns_from_training_trace(trace)
    rows, _ = _chain_to_sequences(turns, "test", 0)
    assert rows[0].input_ids == [10, 11, 12, 13, 14]
    assert rows[0].completion_mask == [0, 0, 1, 0, 1]
    assert rows[0].old_log_probs[-3:] == [-0.1, -0.2, -0.3]


def test_reconciliation_error_reaches_worker_failure_channel(make_loop, monkeypatch):
    # [FUNCTION DOC]
    # Patches _chain_to_sequences to raise ValueError; running the loop raises CaptureContractError
    # matching the message, sets the failed event, closes the session, and leaves the buffer empty.
    # Inputs: make_loop, monkeypatch. Outputs: none. Exceptions: AssertionError. Side effects: monkeypatch.
    session = Session(captured_entry())
    loop = make_loop(SimpleNamespace(create=lambda *args, **kwargs: session))
    monkeypatch.setattr(
        openenv_harness, "_chain_to_sequences", MagicMock(side_effect=ValueError("invalid training sequence"))
    )
    with pytest.raises(openenv_harness.CaptureContractError, match="invalid training sequence"):
        loop.run()
    assert loop._failed_event.is_set()
    assert session.closed.is_set()
    assert loop.rollout_buffer.empty()


@pytest.mark.parametrize("reward", [None, 0.0, 1.0])
def test_harbor_session_to_worker_preserves_masks_and_outcome(make_loop, reward):
    # [FUNCTION DOC]
    # Parametrized over reward None/0.0/1.0. Requires harbor_env.harness. A real HarborSession wraps a
    # mocked env returning a JSON round-tripped result with a trainable turn, a non-trainable turn, an
    # auxiliary turn, and a discarded turn. Asserts the reward passes through, usage counts 2 agent calls,
    # only one training row is produced (tokens 10,11,12, mask 0,1,0, old logprobs 0.0,-0.1,-0.2), and
    # the owned env is closed once.
    # Inputs: make_loop, reward. Outputs: none. Exceptions: AssertionError. Side effects: loop threads.
    HarborSession = pytest.importorskip("harbor_env.harness").HarborSession
    from openenv.harbor.models import HarborRolloutResult, HarborTurn

    turns = [
        HarborTurn(
            turn=0,
            node_id="a",
            prompt_token_ids=[10],
            completion_token_ids=[11, 12],
            per_token_logps=[-0.1, -0.2],
            loss_mask=[0, 1, 0],
            request_messages=[{"role": "user", "content": "task"}],
            tool_calls=[{"name": "bash", "arguments": "{}"}],
        ),
        HarborTurn(
            turn=1,
            node_id="b",
            prompt_token_ids=[20],
            completion_token_ids=[21],
            per_token_logps=[-0.3],
            loss_mask=[0, 0],
            trainable=False,
            request_messages=[{"role": "user", "content": "task"}],
            tool_calls=[{"name": "bash", "arguments": "{}"}],
        ),
        HarborTurn(turn=2, node_id="title", role="auxiliary"),
        HarborTurn(turn=3, node_id="retry", discarded=True),
    ]
    result = HarborRolloutResult.model_validate_json(HarborRolloutResult(turns=turns, reward=reward).model_dump_json())
    env = MagicMock()
    env.run_rollout.return_value = result
    session = HarborSession(
        env=env,
        split="tasks",
        task_index=0,
        instruction="task",
        harness="opencode",
        sandbox="docker",
        llm_url="http://unused",
        model="test",
        owns_env=True,
    )
    loop = make_loop(SimpleNamespace(create=lambda *args, **kwargs: session))
    actual, _ = loop._run_session([])
    assert actual[-1] == reward
    assert actual[3] == 2  # Usage includes the zero-masked agent call.
    assert len(actual[2]) == 1
    assert actual[2][0].input_ids == [10, 11, 12]
    assert actual[2][0].completion_mask == [0, 1, 0]
    assert actual[2][0].old_log_probs == [0.0, -0.1, -0.2]
    env.close.assert_called_once()


def test_shared_prefix_is_supervised_once_across_both_branches():
    # [FUNCTION DOC]
    # Builds a RolloutGraph with root, left, and right turns and two agent sequences sharing root; after
    # conversion and _chain_to_sequences the supervised tokens across rows are exactly 2, 3, 4 (shared
    # prefix supervised once).
    # Inputs: none. Outputs: none. Exceptions: AssertionError on failure. Side effects: none.
    from openenv.core.harness.capture.contract import to_training_trace
    from openenv.core.harness.capture.graph import RolloutGraph, TurnNode

    graph = RolloutGraph()
    for node_id, prompt, output in [("root", [1], [2]), ("left", [1, 2], [3]), ("right", [1, 2], [4])]:
        graph.add_turn(TurnNode(node_id=node_id, prompt_ids=prompt, sampled_ids=output, sampled_logprobs=[-0.1]))
    document = {
        "sequences": [
            {"role": "agent", "node_ids": ["root", "left"], "loss_mask": [0, 1, 1]},
            {"role": "agent", "node_ids": ["root", "right"], "loss_mask": [0, 1, 1]},
        ]
    }
    turns = openenv_harness._turns_from_training_trace(to_training_trace(graph, document))
    rows, _ = _chain_to_sequences(turns, "rollout", 0)
    supervised = [
        token for row in rows for token, mask in zip(row.input_ids, row.completion_mask, strict=True) if mask
    ]
    assert supervised == [2, 3, 4]


def test_legacy_session_fails_before_starting_agent(make_loop):
    # [FUNCTION DOC]
    # Tests a legacy session exposing fetch_proxy_trace but no fetch_training_trace raises
    # CaptureContractError mentioning fetch_training_trace before wait_for_completion is called, and
    # close is called once.
    # Inputs: make_loop fixture. Outputs: none. Exceptions: AssertionError on failure.
    # Side effects: runs the loop's worker threads/event loop; loops cleaned up by the fixture.
    session = SimpleNamespace(wait_for_completion=MagicMock(), fetch_proxy_trace=MagicMock(), close=MagicMock())

    # Sessions must be hashable because the worker tracks them for shutdown.
    class LegacySession:
        # [CLASS DOC]
        # HELPER: Hashable legacy-style session class whose methods are the shared mocks; lacks
        # fetch_training_trace.
        wait_for_completion = session.wait_for_completion
        fetch_proxy_trace = session.fetch_proxy_trace
        close = session.close

    loop = make_loop(SimpleNamespace(create=lambda *args, **kwargs: LegacySession()))
    with pytest.raises(openenv_harness.CaptureContractError, match="fetch_training_trace"):
        loop._run_session([])
    session.wait_for_completion.assert_not_called()
    session.close.assert_called_once()
