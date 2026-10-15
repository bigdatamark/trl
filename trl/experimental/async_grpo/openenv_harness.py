# [MODULE DOC]
# OpenEnv harness integration for AsyncGRPO: a rollout worker whose per-rollout generation is delegated to an OpenEnv
# ResourceSessionFactory (e.g. Harbor sandboxes) instead of TRL's own tool-calling turn loop.
# Two modes: adapter mode (a HarnessAdapter runs the agent loop and calls back into TRL to sample each turn from vLLM)
# and loop-owning mode (the external agent samples via its own client; TRL fetches a token-level TrainingTrace
# afterwards). Contents: CaptureContractError, HarnessRolloutOutcome, conversion helpers, _HarnessRolloutLoop,
# trace helpers, HarnessRolloutWorker. Requires the openenv package at import time.
# Import-time side effects: imports openenv (ImportError if missing); creates module logger.

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

from __future__ import annotations

import asyncio
import functools
import itertools
import json
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, cast

from accelerate.logging import get_logger
from openenv.core.harness import (
    HarnessAdapter,
    HarnessRunLimits,
    ModelStepResult,
    ResourceSessionFactory,
    TraceEntry,
    TrainableSession,
    TrainingTrace,
)
from openenv.core.harness.capture.upstream import training_sampling
from openenv.core.llm_client import LLMResponse, ToolCall

from ...chat_template_utils import parse_response
from .async_rollout_worker import (
    AsyncRolloutWorker,
    TurnRecord,
    _AsyncRolloutLoop,
    _chain_to_sequences,
)


logger = get_logger(__name__)

Message = dict[str, Any]


class CaptureContractError(ValueError):
    # [CLASS DOC]
    # ValueError subclass raised when captured tokens/masks/sampling metadata from a session are unusable for training.
    # Unlike other session errors (which make a rollout unscorable), it propagates and stops the worker.
    # Inputs: message. Side effects: none.
    """Captured tokens, masks or sampling metadata do not satisfy the training contract."""


@dataclass
class HarnessRolloutOutcome:
    # [CLASS DOC]
    # Dataclass passed to a user rollout_reward_fn: env_reward (float or None from session.verify), completion
    # (message list), trace (TraceEntry list; empty in adapter mode), tool_call_count, tool_failure_count,
    # tool_calls_by_name, timed_out (loop-owning agent timed out).
    env_reward: float | None
    completion: list[Message]
    trace: list[TraceEntry]
    tool_call_count: int
    tool_failure_count: int
    tool_calls_by_name: dict[str, int]
    timed_out: bool


def _tools_to_schema(tools: list) -> list[dict] | None:
    # [FUNCTION DOC]
    # HELPER: Maps OpenEnv/MCP Tool objects (name, description, input_schema) to OpenAI function-schema dicts of type
    # "function". Returns None for an empty or missing list.
    # Inputs: tools list. Outputs: list of dicts or None. Exceptions: AttributeError on malformed tools. Side effects: none.
    """Convert OpenEnv `Tool`s (MCP spec) into the OpenAI function schema `apply_chat_template` expects."""
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.input_schema},
        }
        for t in tools
    ]


def _msg_to_llm_response(msg: Message) -> LLMResponse:
    # [FUNCTION DOC]
    # HELPER: Converts a parsed assistant message dict to an OpenEnv LLMResponse.
    # Behavior: for each tool call takes the nested "function" dict (or the call itself), parses string arguments as JSON
    # (invalid JSON becomes an empty dict), uses the call id or "call_<index>"; content defaults to an empty string.
    # Inputs: msg dict. Outputs: LLMResponse. Exceptions: KeyError if a tool call lacks a name. Side effects: none.
    """Convert a `parse_response` assistant message into the `LLMResponse` the harness adapter expects."""
    tool_calls = []
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function", tc)
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        tool_calls.append(ToolCall(id=tc.get("id") or f"call_{i}", name=fn["name"], args=args))
    return LLMResponse(content=msg.get("content") or "", tool_calls=tool_calls)


class _HarnessRolloutLoop(_AsyncRolloutLoop):
    # [CLASS DOC]
    # Subclass of _AsyncRolloutLoop whose _generate_one runs one OpenEnv session on a thread pool and returns the same
    # tuple shape as the base loop (completion messages, completion ids, sequences, tool call count, tool failure count,
    # reward). Class attributes: _provides_rollout_reward True (reward comes from the rollout, not reward functions),
    # _EMPTY_ROLLOUT sentinel (empty data, reward None => excluded from group baseline).
    # State: _adapter, _factory (session factory), _limits (HarnessRunLimits), _rollout_reward_fn, _session_pool
    # (ThreadPoolExecutor sized max(1, max_inflight_tasks)), _live_sessions (sessions to close on stop).
    # Appends "harness_reward" to reward_func_names.
    """`_AsyncRolloutLoop` whose `_generate_one` drives an OpenEnv session instead of TRL's own turn loop."""

    _provides_rollout_reward = True

    # Unscorable rollouts are excluded from the group baseline.
    _EMPTY_ROLLOUT = ([], [], [], 0, 0, None)

    def __init__(
        self,
        *,
        harness_session_factory: Callable[..., ResourceSessionFactory],
        harness_adapter: HarnessAdapter | None = None,
        rollout_reward_fn: Callable[[HarnessRolloutOutcome], float | None] | None = None,
        lossless_capture: bool = True,
        **loop_kwargs,
    ):
        # [FUNCTION DOC]
        # Behavior: in loop-owning mode with lossless_capture, forces fork_threshold_tokens to 0 (any rewrite forks a row) before
        # calling the base constructor. In loop-owning mode builds the factory with sampling settings (temperature, top_p,
        # top_k, min_p, repetition_penalty, dropping None values) wrapped by training_sampling; in adapter mode calls the factory
        # without arguments. Sets run limits: max_turns = max_tool_calling_iterations or 8, sampling temperature and max_tokens.
        # Creates the session thread pool and live-session set.
        # Inputs: harness_session_factory (callable), harness_adapter (optional), rollout_reward_fn (optional),
        # lossless_capture (bool, default True), remaining kwargs forwarded to the base loop.
        # Outputs: none. Exceptions: from factory or base constructor. Side effects: thread pool creation, factory creation.
        # Fork rewritten captures so reconciliation preserves eligible sampled tokens.
        if harness_adapter is None and lossless_capture:
            loop_kwargs["fork_threshold_tokens"] = 0
        super().__init__(**loop_kwargs)
        # An adapter lets TRL sample each turn; otherwise the agent owns the loop.
        self._adapter = harness_adapter
        if harness_adapter is None:
            sampling_kwargs = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,
            }
            sampling_kwargs = {key: value for key, value in sampling_kwargs.items() if value is not None}
            self._factory = harness_session_factory(sampling=training_sampling(sampling_kwargs))
        else:
            self._factory = harness_session_factory()
        self._limits = HarnessRunLimits(
            max_turns=self.max_tool_calling_iterations if self.max_tool_calling_iterations is not None else 8,
            sampling={"temperature": self.temperature, "max_tokens": self.max_tokens},
        )
        self._rollout_reward_fn = rollout_reward_fn
        self.reward_func_names.append("harness_reward")

        self._session_pool = ThreadPoolExecutor(
            max_workers=max(1, self.max_inflight_tasks), thread_name_prefix="harness-session"
        )
        self._live_sessions: set = set()

    async def _generate_one(self, prompt, tool_dict, tools, group_id=0):
        # [FUNCTION DOC]
        # Runs _run_session on the session pool from the event loop and folds its metrics into the loop's counters.
        # Behavior: if metrics exist, adds correctness to the rollout/correctness_mean rate (sum, count), per-tool call and
        # failure counters (tools/<name>/call_total, failure_total), overall and per-tool latency rates, then forwards the
        # remaining metrics to _push_rollout_metrics. tools and tool_dict arguments are ignored (the session owns tools).
        # Inputs: prompt, tool_dict, tools, group_id (default 0). Outputs: rollout tuple.
        # Exceptions: CaptureContractError propagates. Side effects: metric state mutation.
        # OpenEnv sessions are synchronous; metrics stay on the event loop.
        loop = asyncio.get_running_loop()
        result, metrics = await loop.run_in_executor(self._session_pool, self._run_session, prompt, group_id)
        if metrics is not None:
            correctness = metrics.pop("correctness", None)
            if correctness is not None:
                self._rates["rollout/correctness_mean"][0] += correctness
                self._rates["rollout/correctness_mean"][1] += 1.0
            for name, count in metrics.pop("tool_calls_by_name").items():
                self._counters[f"tools/{name}/call_total"] += count
            for name, count in metrics.pop("tool_failures_by_name").items():
                self._counters[f"tools/{name}/failure_total"] += count
            for name, elapsed in metrics.pop("tool_latency"):
                self._rates["tools/latency_s"][0] += elapsed
                self._rates["tools/latency_s"][1] += 1
                self._rates[f"tools/{name}/latency_s"][0] += elapsed
                self._rates[f"tools/{name}/latency_s"][1] += 1
            self._push_rollout_metrics(**metrics)
        return result

    async def _run_loops(self, stop_event) -> None:
        # [FUNCTION DOC]
        # Wraps the base _run_loops: starts a task that waits for stop_event and then closes all live sessions in threads;
        # in a finally block sets stop_event, awaits that task and shuts down the session pool (waiting for threads).
        # Inputs: stop_event (asyncio.Event). Outputs: none. Exceptions: propagates base errors after cleanup.
        # Side effects: closes sessions, stops thread pool.
        async def _close_live_sessions_on_stop() -> None:
            # [FUNCTION DOC]
            # HELPER: Awaits stop_event, then closes a snapshot of live sessions concurrently via asyncio.to_thread.
            # Inputs/Outputs: none. Exceptions: none (close errors logged). Side effects: session closes.
            await stop_event.wait()

            def _close(session):
                # [FUNCTION DOC]
                # HELPER: Calls session.close, logging a warning with traceback on any exception.
                # Inputs: session. Outputs: none. Exceptions: none. Side effects: closes session, logging.
                try:
                    session.close()
                except Exception:
                    logger.warning("closing in-flight harness session on stop failed", exc_info=True)

            await asyncio.gather(*(asyncio.to_thread(_close, session) for session in list(self._live_sessions)))

        close_task = asyncio.create_task(_close_live_sessions_on_stop())
        try:
            await super()._run_loops(stop_event)
        finally:
            # Unblock other sessions before joining their threads, including on capture errors.
            stop_event.set()
            await close_task
            await asyncio.to_thread(self._session_pool.shutdown, wait=True)

    def _run_session(self, prompt, group_id=0):
        # [FUNCTION DOC]
        # Runs one rollout synchronously on a pool thread.
        # Behavior: creates a session with seed = group_id (all generations of a group pick the same task) and a random hex
        # episode id; creation failure returns _EMPTY_ROLLOUT and no metrics. Registers the session as live; returns empty if the
        # stop event is already set.
        # Adapter mode: runs adapter.run_white_box with _sample_turn bound to local turns/turn_times lists; completion is the
        # result messages; tool_call_count from metrics "tool_calls" (default trace length); counts calls and errors per tool
        # from tool_trace; per-tool latency = gap between a turn's reply and next request divided evenly among that turn's
        # tool names (last turn not timed).
        # Loop-owning mode: session must be a TrainableSession (else CaptureContractError); waits for completion (TimeoutError
        # sets timed_out and continues), fetches the training trace, converts turns and trace entries (conversion errors become
        # CaptureContractError); completion, call counts and failure estimates come from the trace helpers.
        # Then verifies the completion, computes reward via rollout_reward_fn(outcome) or env_reward, converts turns to training
        # sequences with _chain_to_sequences (errors -> CaptureContractError), and builds metrics (turns, sequences,
        # completion_ids, tally, loop_exhausted, duration_s, per-tool dicts, latency; correctness = env_reward only when a
        # custom reward fn is used and env_reward exists).
        # Inputs: prompt, group_id (default 0). Outputs: (rollout tuple, metrics dict or None).
        # Exceptions: CaptureContractError re-raised; any other exception is logged and yields _EMPTY_ROLLOUT.
        # Side effects: session lifecycle (always discarded from live set and closed in finally), network, logging.
        """Return a rollout and its metrics from a pool thread.

        Session failures are unscorable. Invalid training captures stop the worker.
        """
        t_dispatch = time.monotonic()
        rollout_id = uuid.uuid4().hex
        # Every generation in a group must select the same task.
        seed = group_id
        try:
            session = self._factory.create(prompt, seed=seed, episode_id=rollout_id)
        except Exception:
            logger.warning("harness session create failed; scoring rollout as unscorable", exc_info=True)
            return self._EMPTY_ROLLOUT, None
        self._live_sessions.add(session)
        timed_out = False
        trace: list[TraceEntry] = []
        tool_calls_by_name: dict[str, int] = {}
        tool_failures_by_name: dict[str, int] = {}
        tool_latency: list[tuple[str, float]] = []
        try:
            if self._stop_event.is_set():
                return self._EMPTY_ROLLOUT, None
            if self._adapter is not None:
                turns: list[TurnRecord] = []
                turn_times: list[tuple[float, float, list[str]]] = []
                result = self._adapter.run_white_box(
                    functools.partial(self._sample_turn, turns, turn_times), session, self._limits
                )
                completion = result.messages
                tool_call_count = int(result.metrics.get("tool_calls", len(result.tool_trace)))
                for entry in result.tool_trace:
                    tool_calls_by_name[entry.tool_name] = tool_calls_by_name.get(entry.tool_name, 0) + 1
                    if entry.result.error is not None:
                        tool_failures_by_name[entry.tool_name] = tool_failures_by_name.get(entry.tool_name, 0) + 1
                # A turn's tool calls run between its reply and the next request, so that gap (harness overhead
                # included) is their latency. The trailing turn has no next request and is not timed.
                tool_latency = [
                    (name, (t_request - t_reply) / len(names))
                    for (_, t_reply, names), (t_request, _, _) in itertools.pairwise(turn_times)
                    for name in names
                ]
            else:
                if not isinstance(session, TrainableSession):
                    raise CaptureContractError("loop-owning training requires OpenEnv fetch_training_trace()")
                loop_session = cast(TrainableSession, session)
                try:
                    loop_session.wait_for_completion()
                except TimeoutError:
                    logger.warning("harness agent timed out; training captured turns, timed_out flagged")
                    timed_out = True
                try:
                    captured = loop_session.fetch_training_trace()
                    turns = _turns_from_training_trace(captured)
                    entries = captured.to_trace_entries()
                    trace = entries
                except (ValueError, TypeError, KeyError) as exc:
                    raise CaptureContractError(str(exc)) from exc
                completion = _messages_from_trace(entries)
                tool_calls_by_name = _tool_call_counts_by_name(entries)
                tool_call_count = sum(tool_calls_by_name.values())
                tool_failures_by_name = _tool_failures_by_name(entries)
            tool_failure_count = sum(tool_failures_by_name.values())
            verify = session.verify(completion)
            env_reward = float(verify.env_reward) if verify.env_reward is not None else None
            outcome = HarnessRolloutOutcome(
                env_reward=env_reward,
                completion=completion,
                trace=trace,
                tool_call_count=tool_call_count,
                tool_failure_count=tool_failure_count,
                tool_calls_by_name=tool_calls_by_name,
                timed_out=timed_out,
            )
            reward = self._rollout_reward_fn(outcome) if self._rollout_reward_fn else env_reward
            try:
                sequences, tally = _chain_to_sequences(turns, rollout_id, self._fork_threshold_tokens)
            except (ValueError, TypeError, KeyError) as exc:
                raise CaptureContractError(str(exc)) from exc
            completion_ids = [tid for turn in turns for tid in turn.output_ids]
            metrics = dict(
                turns=len(turns),
                sequences=len(sequences),
                completion_ids=completion_ids,
                tally=tally,
                loop_exhausted=timed_out,
                duration_s=time.monotonic() - t_dispatch,
                tool_calls_by_name=tool_calls_by_name,
                tool_failures_by_name=tool_failures_by_name,
                tool_latency=tool_latency,
            )
            if self._rollout_reward_fn is not None and env_reward is not None:
                metrics["correctness"] = env_reward
            return (completion, completion_ids, sequences, tool_call_count, tool_failure_count, reward), metrics
        except CaptureContractError:
            raise
        except Exception:
            logger.warning("harness rollout failed; scoring as unscorable", exc_info=True)
            return self._EMPTY_ROLLOUT, None
        finally:
            self._live_sessions.discard(session)
            try:
                session.close()
            except Exception:
                logger.warning("harness session close failed", exc_info=True)

    def _sample_turn(
        self, turns: list[TurnRecord], turn_times: list[tuple[float, float, list[str]]], messages, tools, sampling
    ) -> ModelStepResult:
        # [FUNCTION DOC]
        # OpenEnv ModelStep callback used in adapter mode (called on a pool thread).
        # Behavior: records request time; converts tools to schema; tokenizes messages with the chat template (generation
        # prompt added, chat_template and chat_template_kwargs applied); submits _generate_one_turn to the loop's event loop and
        # blocks on the result (token ids, logprobs); appends a TurnRecord; parses the reply with parse_response (prefix =
        # prompt ids) and converts to LLMResponse; appends (request time, reply time, tool names) to turn_times.
        # Inputs: turns, turn_times (mutable accumulators), messages, tools, sampling (unused).
        # Outputs: ModelStepResult with response, prompt_ids, completion_ids, logprobs.
        # Exceptions: generation/parse errors propagate. Side effects: appends to the two lists, vLLM request.
        """OpenEnv `ModelStep`: sample one assistant turn against vLLM and record a `TurnRecord` into `turns`.

        `turn_times` gets `(request time, reply time, requested tool names)` per turn, for the tool latency metrics.
        """
        t_request = time.monotonic()
        tools = _tools_to_schema(tools)
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            chat_template=self.chat_template,
            **self.chat_template_kwargs,
        )
        # ModelStep is sync on a pool thread; bridge the async vLLM POST onto the loop's event loop.
        turn_ids, logprobs = asyncio.run_coroutine_threadsafe(self._generate_one_turn(prompt_ids), self._loop).result()
        turns.append(TurnRecord(prompt_ids, turn_ids, logprobs))
        response = _msg_to_llm_response(parse_response(self.tokenizer, turn_ids, prefix=prompt_ids, tools=tools))
        turn_times.append((t_request, time.monotonic(), [tc.name for tc in response.tool_calls]))
        return ModelStepResult(response=response, prompt_ids=prompt_ids, completion_ids=turn_ids, logprobs=logprobs)


def _turns_from_training_trace(trace: TrainingTrace) -> list[TurnRecord]:
    # [FUNCTION DOC]
    # HELPER: Builds TurnRecords from an OpenEnv TrainingTrace: prompt ids, completion ids, per-token logprobs and the loss
    # mask with the prompt part removed (so it aligns with completion tokens).
    # Inputs: trace. Outputs: list of TurnRecord. Exceptions: AttributeError on malformed trace. Side effects: none.
    """Map OpenEnv's validated full masks to TRL's completion masks."""
    return [
        TurnRecord(
            list(turn.prompt_token_ids),
            list(turn.completion_token_ids),
            list(turn.per_token_logps),
            list(turn.loss_mask[len(turn.prompt_token_ids) :]),
        )
        for turn in trace.turns
    ]


def _tool_call_counts_by_name(entries: list[TraceEntry]) -> dict[str, int]:
    # [FUNCTION DOC]
    # HELPER: For every trace entry, reads the first choice's message tool_calls from the response (missing pieces treated
    # as empty) and counts calls per function name (skipping unnamed).
    # Inputs: entries. Outputs: dict name -> count. Exceptions: none for missing keys. Side effects: none.
    """Count tool calls by name across the selected agent turns."""
    counts: dict[str, int] = {}
    for entry in entries:
        for tc in (entry.get("response", {}).get("choices") or [{}])[0].get("message", {}).get("tool_calls") or []:
            name = (tc.get("function") or tc).get("name")
            if name:
                counts[name] = counts.get(name, 0) + 1
    return counts


def _tool_failures_by_name(entries: list[TraceEntry]) -> dict[str, int]:
    # [FUNCTION DOC]
    # HELPER: Heuristic failure count per tool from the last entry's request messages.
    # Behavior: maps tool call id -> tool name from assistant tool_calls; for each "tool" role message resolves its name
    # (message name or via tool_call_id), deduplicates on (name, first 200 chars of content), and counts it as a failure
    # if the lowercased content contains "error", "failed", "traceback" or "exception".
    # Inputs: entries. Outputs: dict name -> failure count (name may be None). Exceptions: none. Side effects: none.
    """Estimate failures per tool from the last request's tool-result text, deduplicated by name and content."""
    messages = (entries[-1]["request"] if entries else {}).get("messages") or []
    # A tool result names its call id, not its tool; the assistant message that issued the call has the name.
    names = {
        tc.get("id"): (tc.get("function") or tc).get("name") for m in messages for tc in m.get("tool_calls") or []
    }
    seen, failures = set(), {}
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        name = msg.get("name") or names.get(msg.get("tool_call_id"))
        key = (name, str(msg.get("content"))[:200])
        if key in seen:
            continue
        seen.add(key)
        text = str(msg.get("content") or "").lower()
        if any(w in text for w in ("error", "failed", "traceback", "exception")):
            failures[name] = failures.get(name, 0) + 1
    return failures


def _messages_from_trace(entries: list[TraceEntry]) -> list[Message]:
    # [FUNCTION DOC]
    # HELPER: Returns the last entry's request messages plus an assistant message with the first choice's content (empty
    # string if no choices). Empty entries return an empty list. Tool calls of the final reply are not included.
    # Inputs: entries. Outputs: list of message dicts. Exceptions: KeyError if the last entry lacks "request".
    # Side effects: none.
    """Return the last selected request's messages followed by its assistant reply."""
    if not entries:
        return []
    last = entries[-1]
    choices = (last.get("response") or {}).get("choices") or []
    content = choices[0]["message"].get("content", "") if choices else ""
    return list(last["request"].get("messages") or []) + [{"role": "assistant", "content": content}]


class HarnessRolloutWorker(AsyncRolloutWorker):
    # [CLASS DOC]
    # AsyncRolloutWorker subclass that only swaps the loop class to _HarnessRolloutLoop (_loop_cls); harness constructor
    # kwargs are forwarded through the base worker to the loop. Inject via AsyncGRPOTrainer(rollout_worker=...).
    """AsyncGRPO rollout worker that drives an OpenEnv `ResourceSessionFactory`.

    Pass `harness_session_factory` as a callable returning an OpenEnv `ResourceSessionFactory`, for example
    `functools.partial(HarborSessionFactory, server_url, ...)`. In loop-owning mode, the worker calls it with
    `sampling=` from its training policy. With `harness_adapter`, it calls it without arguments and samples turns
    itself. Inject the worker via `AsyncGRPOTrainer(rollout_worker=...)`.
    """

    _loop_cls = _HarnessRolloutLoop
