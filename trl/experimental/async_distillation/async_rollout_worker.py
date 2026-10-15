# [MODULE DOC]
# Rollout generation for AsyncDistillationTrainer, run in a spawned CPU-only child process. For each dataset row the
# child samples one completion from the student's vLLM server, teacher-forces it through a teacher vLLM server
# (routed per row when several teachers are configured) to obtain sparse top-k per-position logprobs, and pushes a
# RolloutSample onto a multiprocessing queue for the trainer. Ported from the async_grpo worker without groups,
# rewards, tools or advantages.
# Contents: RolloutSample, child bootstrap helpers (_scrub_child_env, _spawn_stop_watcher, _child_main),
# _parse_teacher_logprobs_at_position, _AsyncRolloutLoop, AsyncRolloutWorker.
# Import-time side effects: imports aiohttp and defines _RETRYABLE_HTTP_ERRORS when available; module logger.

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
import math
import multiprocessing as mp
import os
import pickle
import queue
import random
import threading
import time
import traceback
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from multiprocessing.queues import Queue as MPQueue
from multiprocessing.sharedctypes import Synchronized as MPValue
from multiprocessing.synchronize import Event as MPEvent
from typing import Any, TypeAlias

import numpy as np
from accelerate.logging import get_logger
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from ...import_utils import is_aiohttp_available, is_vllm_available
from ...trainer.utils import print_prompt_completions_sample


if is_aiohttp_available():
    import aiohttp

    _RETRYABLE_HTTP_ERRORS = (aiohttp.ClientError, asyncio.TimeoutError, TimeoutError, ConnectionResetError)


logger = get_logger(__name__)

Messages: TypeAlias = list[dict[str, str]]


@dataclass(slots=True)
class RolloutSample:
    # [CLASS DOC]
    # Slotted dataclass for one training row (see docstring): prompt/completion messages, input_ids and completion_mask
    # (int64, prompt positions 0), teacher_topk_ids (int64, rows of width teacher_top_k + 1 padded with -1) and
    # teacher_topk_logprobs (float32, padded with negative infinity) for every input position (prompt rows all padding),
    # model_version at dispatch, metrics (currently empty), teacher_id, prompt_id (row counter across epochs), enqueued_at.
    """One training row: a student on-policy completion plus the teacher's sparse per-position scoring.

    Unlike [`~trl.experimental.async_grpo.async_rollout_worker.RolloutSample`], there is no `advantage` (no
    group-relative baseline: distillation has no policy-gradient term), no reward metrics, and no student-side
    generation-time logprobs (nothing in `compute_loss` reads them today; the student side of the divergence is
    computed fresh from the live model). `teacher_topk_ids` / `teacher_topk_logprobs` carry, per completion position,
    the teacher's top-`teacher_top_k` candidate tokens (plus the realized token if it fell outside that set), see
    `_parse_teacher_logprobs_at_position`. Rows are padded to a fixed width with id `-1` / logprob `-inf`, masked out
    in `compute_loss`.
    """

    prompt: Messages
    completion: Messages
    input_ids: np.ndarray
    completion_mask: np.ndarray
    teacher_topk_ids: np.ndarray  # (len(input_ids), teacher_top_k + 1), padded with -1
    teacher_topk_logprobs: np.ndarray  # (len(input_ids), teacher_top_k + 1), padded with -inf
    model_version: int
    metrics: dict[str, float]
    teacher_id: str  # which teacher_server_urls entry scored this sample (see _resolve_teacher_server_url)
    prompt_id: int  # index of the dataset row this came from, counted across epochs; drives the epoch-driven stop
    enqueued_at: float | None = None


# Env vars the child must drop so accelerate's `PartialState()` initialises in
# single-process mode instead of trying to join the parent's process group.
_CHILD_ENV_TO_STRIP = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCH_FR_DUMP_TEMP_FILE",
    "NCCL_DEBUG_FILE",
)


def _scrub_child_env() -> None:
    # [FUNCTION DOC]
    # HELPER: Hides GPUs from the child (CUDA_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES empty) and removes distributed and
    # torchelastic environment variables listed in _CHILD_ENV_TO_STRIP.
    # Inputs/Outputs: none. Exceptions: none. Side effects: mutates os.environ.
    # The child has no business touching CUDA; any library that imports torch
    # and lazily probes devices would race the parent's allocator.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # On ROCm, an empty CUDA_VISIBLE_DEVICES is ignored when HIP_VISIBLE_DEVICES or ROCR_VISIBLE_DEVICES is set.
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    for k in _CHILD_ENV_TO_STRIP:
        os.environ.pop(k, None)


def _spawn_stop_watcher(rollout_loop: "_AsyncRolloutLoop", stop_event: MPEvent) -> None:
    # [FUNCTION DOC]
    # HELPER: Starts daemon thread "distillation-mp-stop-watcher" that waits for the multiprocessing stop event and then
    # sets the loop's asyncio stop event via call_soon_threadsafe (RuntimeError from a closed loop ignored).
    # Inputs: rollout_loop, stop_event. Outputs: none. Side effects: thread creation.
    # Daemon thread that translates the parent's mp.Event into the child's
    # asyncio.Event so _run_loops breaks out of its gather.
    def _watch():
        # [FUNCTION DOC]
        # HELPER: Thread body: waits on stop_event, then schedules the asyncio stop event set; ignores RuntimeError.
        # Inputs/Outputs: none. Side effects: sets the event.
        stop_event.wait()
        try:
            rollout_loop._loop.call_soon_threadsafe(rollout_loop._stop_event.set)
        except RuntimeError:
            # Loop already closed (run() returned before stop fired). Nothing to do.
            pass

    threading.Thread(target=_watch, daemon=True, name="distillation-mp-stop-watcher").start()


def _child_main(
    loop_kwargs: dict[str, Any],
    samples_queue: MPQueue,
    model_version_value: MPValue,
    stop_event: MPEvent,
    child_ready_event: MPEvent,
    heartbeat_value: MPValue,
    failed_event: MPEvent,
    exception_info_queue: MPQueue,
    metrics_queue: MPQueue,
) -> None:
    # [FUNCTION DOC]
    # HELPER: Child process entry point: scrubs env, initializes accelerate PartialState, builds _AsyncRolloutLoop from
    # loop_kwargs plus shared IPC objects, sets the ready event, starts the stop watcher and runs the loop; exceptions are
    # printed and re-raised.
    # Inputs: loop_kwargs and the shared queues/values/events. Outputs: none. Exceptions: re-raised loop errors.
    # Side effects: env mutation.
    _scrub_child_env()
    # `accelerate.logging.get_logger` requires `PartialState()` to have been called.
    from accelerate.state import PartialState

    PartialState()

    rollout_loop = _AsyncRolloutLoop(
        **loop_kwargs,
        rollout_buffer=samples_queue,
        model_version_value=model_version_value,
        heartbeat_value=heartbeat_value,
        failed_event=failed_event,
        exception_info_queue=exception_info_queue,
        metrics_queue=metrics_queue,
    )
    child_ready_event.set()
    _spawn_stop_watcher(rollout_loop, stop_event)
    try:
        rollout_loop.run()
    except Exception:
        traceback.print_exc()
        raise


def _parse_teacher_logprobs_at_position(position_logprobs: dict[str, dict] | None) -> tuple[list[int], list[float]]:
    # [FUNCTION DOC]
    # HELPER: Converts one position of vLLM prompt_logprobs (token id string -> dict with logprob and rank) into parallel
    # lists of int token ids and logprobs sorted by ascending rank, dropping NaN logprobs. Empty/None input or all-NaN
    # entries give two empty lists.
    # Inputs: position_logprobs (dict or None). Outputs: (ids, logprobs). Exceptions: KeyError on missing rank/logprob.
    # Side effects: none.
    """Parse one completion position's entry of a vLLM `prompt_logprobs` response.

    vLLM reports `{token_id: {"logprob": ..., "rank": ...}}` per position, holding the top-`teacher_top_k` candidates
    plus the realized/actual token when that one fell outside the top-k, in unspecified order. Sorting by `rank` puts
    the teacher's own top-1 candidate first, which is the ordering `_narrow_top1_actual_support` indexes into. Keeping
    the whole mapping (rather than truncating back to `teacher_top_k`, the way
    [`~trl.generation.vllm_client.VLLMClient.get_sequence_logprobs`] does for the sync trainers) is what guarantees the
    realized token is always present, at any rank. A NaN logprob is dropped rather than coerced to a sentinel, since it
    reflects a genuinely invalid score, not a padding slot; a position vLLM scored no candidate for at all (`None`,
    e.g. the sequence's very first token) yields empty lists, masked out in `compute_loss`.
    """
    if not position_logprobs:
        return [], []
    items = sorted(position_logprobs.items(), key=lambda item: item[1]["rank"])
    parsed = [(int(tid), entry["logprob"]) for tid, entry in items if not math.isnan(entry["logprob"])]
    if not parsed:
        return [], []
    ids, lps = zip(*parsed, strict=True)
    return list(ids), list(lps)


class _AsyncRolloutLoop:
    # [CLASS DOC]
    # Asyncio generate-and-score loop in the child (see docstring). One task per sample (_generate_and_score_one), up to
    # max_inflight_tasks concurrently, pushes samples directly to rollout_buffer. State: tokenizer, eos ids, shared IPC
    # objects, metric accumulators (_counters, _rates), sampling params, student server URL, teacher_server_urls (default
    # {"default": "http://localhost:8001"}, trailing slashes stripped), teacher_top_k, teacher_temperature,
    # teacher_model_names (resolved at startup), counters for tokens/samples, aiohttp session, private event loop and stop
    # event.
    """Asyncio generate-and-score loop. Lives entirely inside the spawned child process.

    Structurally this is [`~trl.experimental.async_grpo.async_rollout_worker._AsyncRolloutLoop`] with the
    group-relative-advantage machinery removed (distillation has no group baseline: each prompt yields exactly one
    on-policy sample) and the reward scoring step replaced by a teacher-scoring HTTP call. Because there is no
    grouping, generation and scoring happen in a single task per sample (`_generate_and_score_one`) rather than two
    loops connected by an internal queue, concurrency across in-flight samples still comes from running up to
    `max_inflight_tasks` such tasks at once, exactly as in the GRPO worker.

    Every server it talks to is a plain `vllm serve`, reached through the same OpenAI-compatible `/v1/completions`
    endpoint, in two different modes: the student's server samples a real completion (`_generate_one_turn`), and one or
    more teachers' servers teacher-force that completion and report their own per-position distribution over it
    (`_score_with_teacher`, no new tokens generated). With multiple teachers configured (`teacher_server_urls`),
    `_resolve_teacher_server_url` routes each row to its `teacher_id`'s server (MOPD: multi-teacher on-policy
    distillation). Pushes scored [`RolloutSample`]s into the shared `mp.Queue` (`rollout_buffer`); reads the bumped
    policy version from the shared `mp.Value` (`model_version_value`).
    """

    def __init__(
        self,
        *,
        model_name: str,
        dataset: Dataset,
        processing_class: PreTrainedTokenizerBase,
        eos_token_ids: list[int],
        rollout_buffer: MPQueue,
        model_version_value: MPValue,
        heartbeat_value: MPValue,
        failed_event: MPEvent,
        exception_info_queue: MPQueue,
        metrics_queue: MPQueue,
        max_inflight_tasks: int = 128,
        queue_maxsize: int = 0,
        vllm_server_url: str = "http://localhost:8000",
        lora_name: str | None = None,
        teacher_server_urls: dict[str, str] | None = None,
        teacher_top_k: int = 8,
        teacher_temperature: float = 1.0,
        max_tokens: int = 32,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float | None = None,
        repetition_penalty: float = 1.0,
        request_timeout: int = 120,
        chat_template_kwargs: dict[str, Any] | None = None,
        log_completions: bool = False,
        log_completions_steps: int = 100,
        num_completions_to_print: int | None = None,
        dataset_start_index: int = 0,
    ):
        # [FUNCTION DOC]
        # Stores configuration; starts the dataset iterator at dataset_start_index modulo length when positive; normalizes
        # server URLs; creates accumulators, a new event loop and stop event. Unlike the GRPO loop there is no tool/env setup
        # and no response-schema handling.
        # Inputs: keyword args per signature. Outputs: none. Exceptions: ZeroDivisionError for an empty dataset with a positive
        # start index. Side effects: event loop creation.
        self.model_name = model_name
        self.lora_name = lora_name
        self.dataset = dataset
        if dataset_start_index > 0:
            start = dataset_start_index % len(dataset)
            self._dataset_iter = iter(dataset.select(range(start, len(dataset))))
        else:
            self._dataset_iter = iter(dataset)
        self.tokenizer = processing_class
        self.eos_token_ids = eos_token_ids
        self.rollout_buffer = rollout_buffer  # shared mp.Queue
        self._model_version_value = model_version_value  # shared mp.Value
        self._heartbeat_value = heartbeat_value  # shared mp.Value('d'); wall-clock seconds
        self._failed_event = failed_event  # shared mp.Event
        self._exception_info_queue = exception_info_queue  # shared mp.Queue(maxsize=1)
        self._metrics_queue = metrics_queue  # shared mp.Queue; drained by the trainer in `log()`
        # Metric accumulators
        self._counters: dict[str, float] = defaultdict(float)
        self._rates: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        self._pushed_completion_tokens = 0
        self._pushed_at = time.monotonic()

        self.max_inflight_tasks = max_inflight_tasks
        self.queue_maxsize = queue_maxsize
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.request_timeout = request_timeout
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.log_completions = log_completions
        self.log_completions_steps = log_completions_steps
        self.num_completions_to_print = num_completions_to_print
        self.vllm_server_url = vllm_server_url.rstrip("/")
        teacher_server_urls = teacher_server_urls or {"default": "http://localhost:8001"}
        self.teacher_server_urls = {k: v.rstrip("/") for k, v in teacher_server_urls.items()}
        self.teacher_top_k = teacher_top_k
        self.teacher_temperature = teacher_temperature

        # Served model id per teacher, resolved from each teacher server's `/v1/models` in `_run_loops`, since
        # `/v1/completions` names the model it is addressing and a teacher serves a different one than the student.
        self.teacher_model_names: dict[str, str] = {}

        self._total_completion_tokens = 0
        self._total_samples_scored = 0
        self._generation_start_time: float | None = None
        self.session: aiohttp.ClientSession | None = None

        self._loop = asyncio.new_event_loop()
        self._stop_event = asyncio.Event()

    @property
    def model_version(self) -> int:
        # [FUNCTION DOC]
        # Current policy version read from the shared value, as int. No inputs, exceptions or side effects.
        return int(self._model_version_value.value)

    @property
    def _request_model(self) -> str:
        # [FUNCTION DOC]
        # HELPER: Student model name for requests: model_name, or "<lora_name>-v<model_version>" when LoRA sync is active.
        # Inputs: none. Outputs: str. Side effects: none.
        # In vLLM's API an adapter *is* a model name: naming the base model while an adapter is loaded silently
        # serves the base model, so the published version has to be part of the request. `model_version` reads the
        # shared `mp.Value`, so a sync is picked up without extra IPC.
        if self.lora_name is None:
            return self.model_name
        return f"{self.lora_name}-v{self.model_version}"

    def run(self) -> None:
        # [FUNCTION DOC]
        # Runs _run_loops on the private loop. On any BaseException: best-effort puts (type name, message, traceback) on the
        # exception queue, sets the failed event, logs, re-raises. Always closes the loop.
        # Inputs/Outputs: none. Exceptions: re-raised. Side effects: IPC signalling.
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_loops(stop_event=self._stop_event))
        except BaseException as e:
            # Push pickle-safe exception info to the parent before setting _failed_event, so a
            # reader that sees the event is guaranteed to also see the info on the queue.
            info = (type(e).__name__, str(e), traceback.format_exc())
            try:
                self._exception_info_queue.put_nowait(info)
            except Exception:
                pass  # queue full (parent hasn't drained a prior failure), best-effort put
            self._failed_event.set()
            logger.exception(f"Worker process failed: {e}")
            raise
        finally:
            self._loop.close()

    async def _run_loops(self, stop_event: asyncio.Event) -> None:
        # [FUNCTION DOC]
        # Opens an aiohttp session (connection limit twice max_inflight_tasks, since each task makes student and teacher
        # calls), logs sampling config, resolves teacher model names, then runs the generate loop.
        # Inputs: stop_event. Outputs: none. Exceptions: propagates. Side effects: network.
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=2 * self.max_inflight_tasks)) as session:
            self.session = session
            logger.info(
                f"vllm worker started: max_inflight_tasks={self.max_inflight_tasks}, temperature={self.temperature}, "
                f"top_p={self.top_p}, top_k={self.top_k}, min_p={self.min_p}, "
                f"repetition_penalty={self.repetition_penalty}"
            )
            await self._resolve_teacher_model_names()
            await self._generate_loop(stop_event=stop_event)

    async def _resolve_teacher_model_names(self) -> None:
        # [FUNCTION DOC]
        # HELPER: For each teacher, GETs /v1/models (outer retry up to 30 attempts) and records the first model id in
        # teacher_model_names; logs it. Serves as the teacher readiness wait.
        # Inputs: none. Outputs: none. Exceptions: HTTP errors after retries; KeyError/IndexError on malformed JSON.
        # Side effects: network, logging.
        """Ask every teacher server which model it serves, so `_score_with_teacher` can name it in its request.

        Doubles as the readiness wait for the teachers: the retry budget is the same generous one the generation and
        scoring calls use, so a teacher still loading its weights when the worker starts is waited out rather than
        failing the run.
        """
        for teacher_id, url in self.teacher_server_urls.items():
            models = await self._retry(
                lambda url=url: self._get(url, "/v1/models", self.request_timeout),
                max_attempts=30,
                label=f"teacher {teacher_id} /v1/models",
            )
            self.teacher_model_names[teacher_id] = models["data"][0]["id"]
            logger.info(f"teacher {teacher_id!r} at {url} serves {self.teacher_model_names[teacher_id]}")

    async def _generate_loop(self, stop_event: asyncio.Event) -> None:
        # [FUNCTION DOC]
        # Main loop: each iteration writes the wall-clock heartbeat, fills free slots with _generate_and_score_one tasks for the
        # next (prompt_id, row), sleeps 10 ms if nothing is in flight (returning when stopping), else waits up to 0.1 s for
        # completions. For each finished task: frees slot, re-raises its exception, counts completion tokens and samples, pushes
        # rollout/inflight and windowed rollout/generated_tok_s, prints a completion sample every log_completions_steps samples
        # when log_completions is on, then put_nowait's the sample (stamping enqueued_at), polling every 0.1 s while full
        # (returns if stopping) and pushing rollout/backpressure_s if it blocked. Finally cancels and awaits in-flight tasks.
        # Inputs: stop_event. Outputs: none. Exceptions: task errors propagate. Side effects: IPC, metrics, logging.
        inflight_tasks: dict[asyncio.Task, int] = {}
        free_slots = set(range(self.max_inflight_tasks))
        work_iter = self._repeat_iterator()

        self._generation_start_time = time.monotonic()
        try:
            while True:
                self._heartbeat_value.value = time.time()
                while free_slots and not stop_event.is_set():
                    prompt_id, row = next(work_iter)
                    slot = free_slots.pop()
                    task = asyncio.create_task(self._generate_and_score_one(prompt_id, row))
                    inflight_tasks[task] = slot

                if not inflight_tasks:
                    if stop_event.is_set():
                        return
                    await asyncio.sleep(0.01)
                    continue

                done, _ = await asyncio.wait(inflight_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=0.1)
                if not done:
                    continue

                for task in done:
                    slot = inflight_tasks.pop(task)
                    free_slots.add(slot)
                    if task.exception() is not None:
                        raise task.exception()
                    sample = task.result()
                    self._total_completion_tokens += sum(sample.completion_mask)
                    self._total_samples_scored += 1
                    now = time.monotonic()
                    self._push_metrics(
                        {
                            "rollout/inflight": float(len(inflight_tasks)),
                            # Windowed, not cumulative. A cumulative average cannot show a generation stall.
                            "rollout/generated_tok_s": (
                                float(self._total_completion_tokens - self._pushed_completion_tokens),
                                now - self._pushed_at,
                            ),
                        }
                    )
                    self._pushed_completion_tokens = self._total_completion_tokens
                    self._pushed_at = now

                    # Each scored sample is its own single-element batch (no group-relative multi-sample scoring the
                    # way GRPO has), so unlike GRPO's per-group call, `num_completions_to_print` has nothing to
                    # subsample here without this gate — printing on every sample would flood logs over a long run.
                    if self.log_completions and self._total_samples_scored % self.log_completions_steps == 0:
                        print_prompt_completions_sample(
                            prompts=[sample.prompt],
                            completions=[sample.completion],
                            rewards={},
                            advantages=None,  # no group-relative advantage in distillation
                            step=self._total_samples_scored,
                            num_samples=self.num_completions_to_print,
                        )

                    t_blocked = None
                    while True:
                        try:
                            sample.enqueued_at = time.time()
                            self.rollout_buffer.put_nowait(sample)
                            break
                        except queue.Full:
                            if stop_event.is_set():
                                return
                            if t_blocked is None:
                                t_blocked = time.monotonic()
                                logger.info(
                                    f"rollout buffer full (maxsize={self.queue_maxsize}), "
                                    "waiting for trainer to consume..."
                                )
                            await asyncio.sleep(0.1)
                    if t_blocked is not None:
                        self._push_metrics({"rollout/backpressure_s": time.monotonic() - t_blocked})
        finally:
            for task in inflight_tasks:
                task.cancel()
            if inflight_tasks:
                await asyncio.gather(*inflight_tasks, return_exceptions=True)

    async def _retry(self, coro_factory: Callable[[], Awaitable], *, label: str, max_attempts: int = 1):
        # [FUNCTION DOC]
        # HELPER: Awaits coro_factory up to max_attempts (default 1), retrying only _RETRYABLE_HTTP_ERRORS with sleep
        # min(2 ** min(attempt, 4), 16) seconds, incrementing rollout/vllm_retry_total and warning each retry.
        # Inputs: coro_factory, label, max_attempts. Outputs: coroutine result. Exceptions: last retryable error or any other.
        # Side effects: sleep, metrics, logging.
        """Retry an aiohttp coroutine on transport errors with bounded exponential backoff, counting the retries.

        Ported from [`~trl.experimental.async_grpo.async_rollout_worker._AsyncRolloutLoop._retry`]. Every retried call
        in this loop targets a vLLM server — the student's or a teacher's — so the counter is named for the dependency
        rather than for the transport.
        """
        for attempt in range(max_attempts):
            try:
                return await coro_factory()
            except _RETRYABLE_HTTP_ERRORS as e:
                if attempt >= max_attempts - 1:
                    raise
                self._counters["rollout/vllm_retry_total"] += 1
                sleep = min(2 ** min(attempt, 4), 16)
                logger.warning(
                    f"{label} failed ({type(e).__name__}: {e}); retry {attempt + 1}/{max_attempts} in {sleep}s"
                )
                await asyncio.sleep(sleep)

    def _push_metrics(self, values: dict[str, float | tuple[float, float]]) -> None:
        # [FUNCTION DOC]
        # HELPER: Sends given values plus accumulated counters and rates ((num, den) tuples) to the metrics queue without
        # blocking (dropped when full) and clears the accumulators.
        # Inputs: values dict. Outputs: none. Exceptions: none. Side effects: IPC write, clears accumulators.
        """Send metrics to the trainer, along with the counters and rates accumulated since the last push."""
        payload = dict(values)
        payload.update(self._counters)
        payload.update({key: (num, den) for key, (num, den) in self._rates.items()})
        self._counters.clear()
        self._rates.clear()
        try:
            self._metrics_queue.put_nowait(payload)
        except queue.Full:
            pass  # never block generation

    def _push_rollout_metrics(
        self, *, completion_ids: list[int], teacher_id: str, score_s: float, duration_s: float
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Accumulates completions/mean_length and rollout/score_s rates, teacher_score_s/<teacher_id> when several
        # teachers exist, and completions/clipped_ratio (last token not EOS/pad) for non-empty completions; then pushes
        # rollout/duration_s and completions/max_length and min_length.
        # Inputs: keyword completion_ids, teacher_id, score_s, duration_s. Outputs: none. Side effects: metrics.
        """One rollout: what the student generated, and the teacher call that scored it.

        A rollout yields exactly one training sample here — one turn, no group baseline, no re-tokenization forking —
        so unlike [`~trl.experimental.async_grpo.async_rollout_worker._AsyncRolloutLoop._push_rollout_metrics`] there
        is no `rollout/samples_per_rollout`, `rollout/turns_*`, drift tally or tool tally to report.
        """
        completion_tokens = len(completion_ids)
        self._rates["completions/mean_length"][0] += completion_tokens
        self._rates["completions/mean_length"][1] += 1
        self._rates["rollout/score_s"][0] += score_s
        self._rates["rollout/score_s"][1] += 1
        if len(self.teacher_server_urls) > 1:
            # MOPD: one slow teacher throttles only the rollouts routed to it, which the blended mean above hides.
            # Keyed the way the trainer names `teacher_jsd/{teacher_id}`, so the per-teacher series group together.
            self._rates[f"teacher_score_s/{teacher_id}"][0] += score_s
            self._rates[f"teacher_score_s/{teacher_id}"][1] += 1
        if completion_ids:
            # NOTE(@aminediro):
            # Truncation is read off the same way [`GRPOTrainer`] and [`RLOOTrainer`] define
            # `completions/clipped_ratio`: a completion that does not end on EOS (or pad) was cut off by `max_tokens`
            # rather than finishing. Deliberately NOT vLLM's `finish_reason`, so the metric means the same thing here
            eos_and_pad = (*self.eos_token_ids, self.tokenizer.pad_token_id)
            self._rates["completions/clipped_ratio"][0] += completion_ids[-1] not in eos_and_pad
            self._rates["completions/clipped_ratio"][1] += 1
        self._push_metrics(
            {
                "rollout/duration_s": duration_s,
                "completions/max_length": float(completion_tokens),
                "completions/min_length": float(completion_tokens),
            }
        )

    def _repeat_iterator(self) -> Iterator[tuple[int, dict[str, Any]]]:
        # [FUNCTION DOC]
        # HELPER: Infinite generator yielding (prompt_id, row) once per row, restarting the dataset in order when exhausted;
        # prompt_id increments per yield.
        # Inputs: none. Outputs: iterator. Exceptions: StopIteration-derived RuntimeError for an empty dataset.
        # Side effects: replaces _dataset_iter.
        prompt_id = 0
        while True:
            try:
                row = next(self._dataset_iter)
            except StopIteration:
                self._dataset_iter = iter(self.dataset)
                row = next(self._dataset_iter)
            # One sample per row, so this is `AsyncGRPOTrainer`'s `group_id` with `num_generations` fixed at 1.
            yield prompt_id, row
            prompt_id += 1

    def _resolve_teacher_server_url(self, row: dict[str, Any]) -> tuple[str, str]:
        # [FUNCTION DOC]
        # HELPER: With one teacher returns its (id, url) regardless of row; otherwise uses row "teacher_id" and returns
        # (id, url). Inputs: row. Outputs: tuple. Exceptions: ValueError when the id is missing or unknown. Side effects: none.
        """Pick which teacher server scores `row` (MOPD: multi-teacher on-policy distillation).

        A single configured teacher scores every row regardless of any `teacher_id` column (plain on-policy
        distillation). With multiple teachers, `row["teacher_id"]` selects among them; a missing or unmapped
        `teacher_id` is a configuration error (raises), not a silent misroute to the wrong teacher. Returns
        `(teacher_id, server_url)`; the resolved id is carried on the `RolloutSample` so `compute_loss` can break down
        `jsd`/`entropy`/`teacher_entropy` per teacher.
        """
        if len(self.teacher_server_urls) == 1:
            ((teacher_id, url),) = self.teacher_server_urls.items()
            return teacher_id, url
        teacher_id = row.get("teacher_id")
        if teacher_id not in self.teacher_server_urls:
            raise ValueError(
                f"Example has `teacher_id={teacher_id!r}`, which is not among the teachers passed to "
                f"`teacher_server_urls`. Expected one of: {list(self.teacher_server_urls)}."
            )
        return teacher_id, self.teacher_server_urls[teacher_id]

    async def _generate_and_score_one(self, prompt_id: int, row: dict[str, Any]) -> RolloutSample:
        # [FUNCTION DOC]
        # Produces one RolloutSample.
        # Behavior: snapshots model_version before generating; tokenizes row "prompt" with the chat template (generation prompt
        # added, chat_template_kwargs); samples completion ids from the student; decodes them (special tokens kept) into a
        # single assistant message; resolves the teacher and scores prompt+completion (timed). Builds input_ids, completion_mask
        # (0 for prompt, 1 for completion) and full-length teacher arrays filled at completion positions with the candidate
        # lists. Pushes rollout metrics.
        # Inputs: prompt_id, row. Outputs: RolloutSample (metrics empty).
        # Exceptions: KeyError if row lacks prompt; ValueError for teacher routing; HTTP errors; ValueError/IndexError if the
        # teacher returned more candidates than teacher_top_k + 1 or fewer positions than expected.
        # Side effects: network, metrics.
        model_version = self.model_version
        t_dispatch = time.monotonic()
        prompt = row["prompt"]
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt, return_dict=False, add_generation_prompt=True, **self.chat_template_kwargs
        )
        completion_ids = await self._generate_one_turn(prompt_ids)
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)
        completion = [{"role": "assistant", "content": completion_text}]

        teacher_id, teacher_server_url = self._resolve_teacher_server_url(row)
        t_score = time.monotonic()
        teacher_topk_ids, teacher_topk_logprobs = await self._score_with_teacher(
            prompt_ids, completion_ids, teacher_server_url, self.teacher_model_names[teacher_id]
        )
        score_s = time.monotonic() - t_score

        input_ids = np.asarray(prompt_ids + completion_ids, dtype=np.int64)
        completion_mask = np.asarray([0] * len(prompt_ids) + [1] * len(completion_ids), dtype=np.int64)
        full_teacher_ids = np.full((len(input_ids), self.teacher_top_k + 1), -1, dtype=np.int64)
        full_teacher_logprobs = np.full((len(input_ids), self.teacher_top_k + 1), -np.inf, dtype=np.float32)
        for i, (cand_ids, cand_lps) in enumerate(zip(teacher_topk_ids, teacher_topk_logprobs, strict=True)):
            full_teacher_ids[len(prompt_ids) + i, : len(cand_ids)] = cand_ids
            full_teacher_logprobs[len(prompt_ids) + i, : len(cand_lps)] = cand_lps

        self._push_rollout_metrics(
            completion_ids=completion_ids,
            teacher_id=teacher_id,
            score_s=score_s,
            duration_s=time.monotonic() - t_dispatch,
        )
        return RolloutSample(
            prompt=prompt,
            completion=completion,
            input_ids=input_ids,
            completion_mask=completion_mask,
            teacher_topk_ids=full_teacher_ids,
            teacher_topk_logprobs=full_teacher_logprobs,
            model_version=model_version,
            teacher_id=teacher_id,
            prompt_id=prompt_id,
            metrics={},
        )

    async def _generate_one_turn(self, prompt_ids: list[int]) -> list[int]:
        # [FUNCTION DOC]
        # HELPER: POSTs /v1/completions to the student server (model _request_model, token prompt, max_tokens, temperature,
        # top_p, top_k, repetition_penalty, n 1, return_token_ids, random 63-bit seed, min_p when set; no logprobs requested)
        # with outer retry of 30 attempts.
        # Inputs: prompt_ids. Outputs: list of completion token ids.
        # KNOWN ISSUE: _post already retries 3 times internally, so retries compound (up to 90 attempts) and the retry counter
        # counts both layers; same pattern as the async_grpo worker.
        # Exceptions: HTTP errors; KeyError on malformed response. Side effects: network.
        payload = {
            "model": self._request_model,
            "prompt": prompt_ids,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "n": 1,
            "return_token_ids": True,
            # Unseeded requests draw their sampling seed from an RNG that every data-parallel vLLM engine seeds
            # identically, so samples of the same prompt spread across engines can decode the same text. A unique seed
            # per request keeps them independent.
            "seed": random.getrandbits(63),
        }
        if self.min_p is not None:
            payload["min_p"] = self.min_p
        output = await self._retry(
            lambda: self._post(self.vllm_server_url, "/v1/completions", payload, self.request_timeout),
            max_attempts=30,
            label="student vllm /v1/completions",
        )
        return output["choices"][0]["token_ids"]

    async def _score_with_teacher(
        self, prompt_ids: list[int], completion_ids: list[int], teacher_server_url: str, teacher_model_name: str
    ) -> tuple[list[list[int]], list[list[float]]]:
        # [FUNCTION DOC]
        # HELPER: POSTs /v1/completions to the teacher with the full prompt+completion, max_tokens 1, teacher_temperature and
        # prompt_logprobs = teacher_top_k (outer retry 30 attempts). Parses prompt_logprobs from index len(prompt_ids) to the
        # end with _parse_teacher_logprobs_at_position.
        # Inputs: prompt_ids, completion_ids, teacher_server_url, teacher_model_name.
        # Outputs: (list of candidate id lists, list of logprob lists), one per completion position.
        # Exceptions: HTTP errors; KeyError on malformed response. Side effects: network.
        """Teacher-force the student's completion through a teacher and read back its per-position distribution.

        `max_tokens=1` generates a single throwaway token (the teacher is never sampled from), and
        `prompt_logprobs=teacher_top_k` makes the server score every position of the sequence it was handed, reporting
        the top-`teacher_top_k` candidates plus the realized token when that one fell outside the top-k. This is the
        request [`~trl.generation.vllm_client.VLLMClient.get_sequence_logprobs`] issues for the synchronous
        server-teacher trainers; the response is parsed here rather than through that client because it truncates each
        position back to exactly `top_logprobs` entries and reports the realized token in separate fields, whereas the
        loss here wants one rank-sorted list per position (see `_parse_teacher_logprobs_at_position`).

        `temperature` reaches the reported logprobs only on a server started with `--logprobs-mode processed_logprobs`,
        and a `teacher_top_k` above vLLM's default cap of 20 additionally needs `--max-logprobs -1`; see
        `AsyncDistillationConfig.teacher_server_urls` for the full `vllm serve` command.
        """
        payload = {
            "model": teacher_model_name,
            "prompt": prompt_ids + completion_ids,
            "max_tokens": 1,
            "temperature": self.teacher_temperature,
            "prompt_logprobs": self.teacher_top_k,
        }
        output = await self._retry(
            lambda: self._post(teacher_server_url, "/v1/completions", payload, self.request_timeout),
            max_attempts=30,
            label="teacher /v1/completions",
        )
        # `prompt_logprobs[i]` scores `sequence[i]` conditioned on `sequence[:i]`, so the row spans the whole sequence
        # and the completion region starts at `len(prompt_ids)`.
        prompt_logprobs = output["choices"][0]["prompt_logprobs"]
        topk_ids, topk_logprobs = [], []
        for position in range(len(prompt_ids), len(prompt_logprobs)):
            ids, lps = _parse_teacher_logprobs_at_position(prompt_logprobs[position])
            topk_ids.append(ids)
            topk_logprobs.append(lps)
        return topk_ids, topk_logprobs

    async def _get(self, base_url: str, path: str, timeout: float, max_retries: int = 3) -> dict:
        # [FUNCTION DOC]
        # HELPER: GET base_url + path with a total timeout, raise_for_status and return JSON, retried up to max_retries
        # (default 3). Inputs: base_url, path, timeout, max_retries. Outputs: dict. Exceptions: aiohttp errors.
        # Side effects: network.
        client_timeout = aiohttp.ClientTimeout(total=timeout)

        async def _do_get():
            # [FUNCTION DOC]
            # HELPER: Single GET attempt on the shared session returning parsed JSON. Exceptions: aiohttp errors.
            async with self.session.get(f"{base_url}{path}", timeout=client_timeout) as response:
                response.raise_for_status()
                return await response.json()

        return await self._retry(_do_get, label=f"GET {base_url}{path}", max_attempts=max_retries)

    async def _post(self, base_url: str, path: str, payload: dict, timeout: float, max_retries: int = 3) -> dict:
        # [FUNCTION DOC]
        # HELPER: POST JSON payload to base_url + path with a total timeout, raise_for_status, return the JSON (empty dict if
        # empty), retried up to max_retries (default 3).
        # Inputs: base_url, path, payload, timeout, max_retries. Outputs: dict. Exceptions: aiohttp errors. Side effects: network.
        client_timeout = aiohttp.ClientTimeout(total=timeout)

        async def _do_post():
            # [FUNCTION DOC]
            # HELPER: Single POST attempt returning parsed JSON or an empty dict. Exceptions: aiohttp errors.
            async with self.session.post(f"{base_url}{path}", json=payload, timeout=client_timeout) as response:
                response.raise_for_status()
                content = await response.json()
                return content if content else {}

        return await self._retry(_do_post, label=f"POST {base_url}{path}", max_attempts=max_retries)


class AsyncRolloutWorker:
    # [CLASS DOC]
    # Parent-side controller of the distillation rollout child (see docstring); same lifecycle and IPC primitives as the
    # async_grpo AsyncRolloutWorker (spawn context, rollout_buffer, model version value, stop/ready/failed events,
    # heartbeat, exception queue, metrics_queue of maxsize 4096), but always runs this module's _AsyncRolloutLoop.
    """Parent-side controller: spawns a child process running `_AsyncRolloutLoop`.

    Ported from [`~trl.experimental.async_grpo.async_rollout_worker.AsyncRolloutWorker`], same lifecycle contract
    (start/stop/check_health/update_model_version), same picklability validation, same `CUDA_VISIBLE_DEVICES=""` child
    isolation. No `reward_funcs`/`tools`/`environment_factory` to forward: distillation's only external calls are the
    vLLM servers already carried by `loop_kwargs` (`vllm_server_url`, `teacher_server_urls`).
    """

    def __init__(
        self,
        *,
        queue_maxsize: int = 0,
        child_ready_timeout: int = 300,
        **loop_kwargs: Any,
    ):
        # [FUNCTION DOC]
        # Creates IPC primitives, stores loop kwargs (adding queue_maxsize and defaulting dataset_start_index to 0).
        # Inputs: queue_maxsize (default 0), child_ready_timeout (default 300 s), loop kwargs. Outputs: none.
        # Exceptions: ImportError if vLLM or aiohttp missing. Side effects: IPC object creation.
        if not is_vllm_available():
            raise ImportError("vLLM is required to use AsyncRolloutWorker. Install it with: pip install vllm")
        if not is_aiohttp_available():
            raise ImportError("aiohttp is not installed. Please install it with `pip install aiohttp`.")
        ctx = mp.get_context("spawn")
        self._mp_ctx = ctx
        self.rollout_buffer = ctx.Queue(maxsize=queue_maxsize)
        self._model_version_value = ctx.Value("i", 0)
        self._stop_event_mp = ctx.Event()
        self._child_ready_event = ctx.Event()
        # Liveness state shared with the child. Wall-clock seconds because monotonic() is per-process.
        self._heartbeat_value = ctx.Value("d", 0.0)
        self._failed_event = ctx.Event()
        self._exception_info_queue = ctx.Queue(maxsize=1)
        # Metrics the child measures, drained by the trainer in `log()`. Bounded and dropped-on-full in the child, so a
        # trainer that stops draining can never block generation.
        self.metrics_queue = ctx.Queue(maxsize=4096)
        # Forwarded verbatim to _AsyncRolloutLoop in the child. queue_maxsize is also
        # forwarded, the child reads it for "rollout buffer full" log lines.
        loop_kwargs["queue_maxsize"] = queue_maxsize
        loop_kwargs.setdefault("dataset_start_index", 0)
        self._loop_kwargs = loop_kwargs
        self._child_ready_timeout = child_ready_timeout
        self._process: mp.Process | None = None

    @property
    def model_version(self) -> int:
        # [FUNCTION DOC]
        # Reads the shared model version as int. No side effects.
        return int(self._model_version_value.value)

    @model_version.setter
    def model_version(self, value: int) -> None:
        # [FUNCTION DOC]
        # Setter: writes int(value) to the shared value under its lock. Side effects: visible to the child.
        # NOTE Read/write ops like += are not atomic with mp.Value
        with self._model_version_value.get_lock():
            self._model_version_value.value = int(value)

    def update_model_version(self, model_version: int) -> None:
        # [FUNCTION DOC]
        # Sets model_version via the property. Inputs: model_version (int). Outputs: none.
        self.model_version = model_version

    def start(self) -> None:
        # [FUNCTION DOC]
        # No-op with warning if already started. Resets heartbeat, checks kwargs pickle (TypeError otherwise), spawns daemon
        # process "distillation-rollout-worker-child" running _child_main, then polls the ready event every second, raising
        # RuntimeError if the child dies or child_ready_timeout passes.
        # Inputs/Outputs: none. Side effects: process creation, logging.
        if self._process is not None:
            logger.warning("AsyncRolloutWorker.start() called but child process is already running; ignoring.")
            return
        # Reset so spawn-import latency (~tens of seconds) doesn't immediately trip check_health.
        self._heartbeat_value.value = time.time()
        try:
            pickle.dumps(self._loop_kwargs)
        except (pickle.PicklingError, AttributeError, TypeError) as e:
            raise TypeError(
                "AsyncRolloutWorker forwards its constructor kwargs to a spawned child process, so they must be "
                "picklable. Lambdas and closures are not: use a module-level function, functools.partial, or a "
                "callable class instance instead."
            ) from e
        self._process = self._mp_ctx.Process(
            target=_child_main,
            args=(
                self._loop_kwargs,
                self.rollout_buffer,
                self._model_version_value,
                self._stop_event_mp,
                self._child_ready_event,
                self._heartbeat_value,
                self._failed_event,
                self._exception_info_queue,
                self.metrics_queue,
            ),
            name="distillation-rollout-worker-child",
            daemon=True,
        )
        self._process.start()
        logger.info(
            f"AsyncRolloutWorker spawned child pid={self._process.pid}; "
            f"waiting up to {self._child_ready_timeout}s for the ready signal"
        )
        # spawn re-imports torch+transformers+trl+vllm in the child, slow on cold launch. Poll
        # liveness so an early crash surfaces immediately instead of after the full timeout.
        deadline = time.monotonic() + self._child_ready_timeout
        while not self._child_ready_event.wait(timeout=1.0):
            if not self._process.is_alive():
                exit_code = self._process.exitcode
                self._process = None
                raise RuntimeError(
                    f"AsyncRolloutWorker child exited during init (exitcode={exit_code}). "
                    "Check the child's stderr for the traceback."
                )
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"AsyncRolloutWorker child did not signal ready within {self._child_ready_timeout}s."
                )
        logger.info("AsyncRolloutWorker child is ready")

    def check_health(self, stale_after_s: float) -> None:
        # [FUNCTION DOC]
        # Raises RuntimeError if the failed event is set (chaining a RuntimeError built from queued exception info if any), or
        # if the heartbeat is older than stale_after_s. Inputs: stale_after_s. Outputs: none.
        # Side effects: consumes queued exception info.
        """Raise if the child crashed or hasn't ticked the heartbeat within `stale_after_s`."""
        if self._failed_event.is_set():
            try:
                type_name, msg, tb = self._exception_info_queue.get_nowait()
                cause = RuntimeError(f"{type_name}: {msg}\n{tb}")
            except queue.Empty:
                cause = None
            raise RuntimeError("Rollout worker child has failed; see chained exception.") from cause
        age = time.time() - self._heartbeat_value.value
        if age > stale_after_s:
            raise RuntimeError(f"Rollout worker heartbeat stale: {age:.0f}s > {stale_after_s:.0f}s; child is hung.")

    def stop(self) -> None:
        # [FUNCTION DOC]
        # No-op without a process; else sets the stop event and, if the process was started, joins 15 s, terminates and joins
        # 5 s, then kills if still alive; clears _process. Inputs/Outputs: none. Side effects: process termination, logging.
        if self._process is None:
            return
        logger.info("Stopping AsyncRolloutWorker child process...")
        self._stop_event_mp.set()
        # If start() raised before Process.start() returned (e.g. pickle failure during spawn),
        # _popen is None and .join() would assert, skip cleanly.
        if self._process._popen is not None:
            self._process.join(timeout=15)
            if self._process.is_alive():
                logger.warning("Child did not exit within 15s; terminating.")
                self._process.terminate()
                self._process.join(timeout=5)
                if self._process.is_alive():
                    self._process.kill()
        self._process = None
