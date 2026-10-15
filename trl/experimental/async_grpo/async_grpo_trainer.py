# [MODULE DOC]
# AsyncGRPOTrainer and its data pipeline: asynchronous GRPO where a rollout worker (default: AsyncRolloutWorker child
# process talking to a vLLM server) produces scored samples into a queue while the trainer consumes them, and weights
# are synced back to vLLM every weight_sync_steps steps (NCCL full weights, or LoRA adapter files when possible).
# Contents: _reduce_metric (logging-window reduction), protocols for pluggable rollout workers and weight transfer,
# callbacks (step interval, optimizer timing, train begin, epoch stop), log_rollout_traces (trackio traces), the input
# pipeline (RolloutQueueDataset -> FixedCountBatcher or TokenBudgetBatcher planners -> DataCollatorForRollout packer
# using TreeRow tree packing), LoRA sync helpers (_iter_vllm_named_params, round_lora_rank,
# validate_lora_for_vllm_sync, save_lora_adapter, select_adapter_sync) and the trainer itself.
# Import-time side effects: imports peft and trackio when available; module logger; type aliases RewardFunc,
# MetricValue, EnvironmentFactory; constant VLLM_LORA_RANKS.

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


import contextvars
import itertools
import json
import math
import os
import queue
import shutil
import textwrap
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from multiprocessing.queues import Queue as MPQueue
from typing import Any, Protocol

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import broadcast_object_list, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from torch.distributed._tensor import DTensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, SaveStrategy
from transformers.utils import is_peft_available

from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import (
    add_fused_lm_head,
    compute_flops_per_token,
    compute_mfu,
    create_model_from_path,
    get_config_model_id,
    get_peak_flops_per_device,
    is_trackio_available,
    nanmax,
    nanmin,
    pad,
)
from .async_grpo_config import AsyncGRPOConfig
from .async_rollout_worker import AsyncRolloutWorker, RolloutSample
from .packing import TREE_ATTENTION, TreeRow, register_tree_attention
from .vllm_client import VLLMClient
from .weight_transfer import WeightTransferClient


logger = get_logger(__name__)

if is_peft_available():
    from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model

if is_trackio_available():
    import trackio

# A reward function is a callable that returns a list of floats (the rewards). The callable receives prompts,
# completions, and additional arguments from the trainer (refer to the trainer's source for details). To ensure forward
# compatibility, it should accept **kwargs.
RewardFunc = Callable[..., list[float]]

# One logged value is either a float or a `(numerator, denominator)` pair; see `_reduce_metric`.
MetricValue = float | tuple[float, float]


def _reduce_metric(key: str, values: list[MetricValue]) -> float:
    # [FUNCTION DOC]
    # HELPER: Reduces one logging window of values for key (rules in docstring).
    # Behavior: tuple values -> sum of numerators / sum of denominators (NaN when the denominator sums to 0). Otherwise the
    # last path component split on "_" is checked: contains "total" -> sum, "max" -> max, "min" -> min, else mean.
    # Inputs: key (str), values (non-empty list, uniform shape). Outputs: float.
    # Exceptions: IndexError on empty values. Side effects: none.
    """Reduce one logging window into the number that gets logged.

    The value's shape and the key's suffix declare the reduction, so nothing has to be registered or configured:

    - `(numerator, denominator)` pairs are a rate, reduced as Σnum / Σden. A rate is never stored *as* a rate, which is
      what makes a mean-of-ratios impossible — the defect that made `training_tok/s` unusable.
    - a `total` word in the name is a counter, summed over the window (producers push deltas).
    - a `max` or `min` word is an extremum. Matching whole words rather than a suffix keeps the upstream spellings
      (`completions/max_length`, `clip_ratio/high_max`) working alongside this trainer's own (`row_tokens_max`).
    - anything else is a gauge, and its window mean is meaningful.

    A key must always be logged with the same shape, since the reduction is picked from the first value seen.
    """
    if isinstance(values[0], tuple):
        numerator, denominator = (sum(v) for v in zip(*values, strict=True))
        return numerator / denominator if denominator else float("nan")
    words = key.split("/")[-1].split("_")
    if "total" in words:
        return sum(values)
    if "max" in words:
        return max(values)
    if "min" in words:
        return min(values)
    return sum(values) / len(values)


class _SupportsReset(Protocol):
    # [CLASS DOC]
    # Typing protocol for environments: must expose reset accepting keyword args and returning a str or None.
    def reset(self, **kwargs) -> str | None:
        # [FUNCTION DOC]
        # Protocol stub: resets the environment for a new rollout given the dataset row fields; returns an optional
        # observation appended to the prompt. Inputs: kwargs. Outputs: str or None. Side effects: implementation-defined.
        ...


EnvironmentFactory = Callable[[], _SupportsReset]


class RolloutWorkerProtocol(Protocol):
    # [CLASS DOC]
    # Structural interface for a pluggable rollout worker (see docstring): attributes rollout_buffer and metrics_queue,
    # methods start, stop, update_model_version, check_health. AsyncRolloutWorker satisfies it.
    """Interface a rollout worker must implement to be passed as `rollout_worker` to
    [`experimental.async_grpo.AsyncGRPOTrainer`].

    The default [`AsyncRolloutWorker`] spawns a CUDA-free child process and scores completions with the trainer's
    `reward_funcs`. Implement this protocol to plug in a custom rollout/scoring backend instead — for example, one that
    runs reward models on their own GPUs.

    Attributes:
        rollout_buffer (`queue.Queue` or `multiprocessing.queues.Queue`):
            Queue the trainer drains; the worker pushes scored `RolloutSample`s onto it. The two queue types are
            structurally identical (`get` / `put_nowait` / `qsize`) but nominally unrelated, so both are allowed: the
            default [`AsyncRolloutWorker`] runs its loop in a spawned process and uses `multiprocessing.Queue`, while
            an in-process worker uses `queue.Queue`.
        metrics_queue (`queue.Queue` or `multiprocessing.queues.Queue`):
            Queue the trainer drains in `log()` for metrics the worker measured itself. Each item is one dict shaped
            like the trainer's metric sink — `{key: float}` for a gauge or a counter, `{key: (numerator, denominator)}`
            for a rate — so draining it is an append. A worker that measures nothing exposes an empty queue.
    """

    rollout_buffer: queue.Queue | MPQueue
    metrics_queue: queue.Queue | MPQueue

    def start(self) -> None:
        # [FUNCTION DOC]
        # Protocol stub: begin producing rollouts (called once on train begin after the initial sync). No return.
        """Begin producing rollouts. Called once on train begin, after the initial weight sync."""
        ...

    def stop(self) -> None:
        # [FUNCTION DOC]
        # Protocol stub: stop the worker and release resources (called on train end). No return.
        """Stop the worker and release its resources. Called on train end."""
        ...

    def update_model_version(self, model_version: int) -> None:
        # [FUNCTION DOC]
        # Protocol stub: receive the new live policy version (int) after a weight sync. No return.
        """Tell the worker which policy version is now live, so it can tag or discard stale samples."""
        ...

    def check_health(self, stale_after_s: float) -> None:
        # [FUNCTION DOC]
        # Protocol stub: raise if the worker crashed or produced nothing within stale_after_s seconds. No return.
        """Raise if the worker has crashed or stopped producing within `stale_after_s` seconds."""
        ...


class WeightTransferProtocol(Protocol):
    # [CLASS DOC]
    # Structural interface for a pluggable weight-sync backend (see docstring): init_weight_transfer, pause, send_weights,
    # resume, destroy. WeightTransferClient satisfies it; a no-op implementation disables trainer-side sync.
    """Interface a weight-sync backend must implement to be passed as `weight_transfer` to
    [`experimental.async_grpo.AsyncGRPOTrainer`].

    The default [`WeightTransferClient`] streams the trainer's weights into the vLLM server over NCCL. Implement this
    protocol to plug in a different sync mechanism, or pass a no-op implementation to disable trainer-side weight sync
    (e.g. when a custom `rollout_worker` updates the policy itself).
    """

    def init_weight_transfer(self) -> None:
        # [FUNCTION DOC]
        # Protocol stub: one-time setup before the first sync (e.g. NCCL group). No return.
        """Set up the transfer (e.g. the NCCL group). Called once on train begin, before the first sync."""
        ...

    def pause(self) -> None:
        # [FUNCTION DOC]
        # Protocol stub: pause the inference server before weights are swapped. No return.
        """Pause the inference server before weights are swapped in."""
        ...

    def send_weights(self, iterator: Iterator[tuple[str, torch.Tensor]]) -> None:
        # [FUNCTION DOC]
        # Protocol stub: stream (name, tensor) pairs from iterator to the server. No return.
        """Stream `(name, tensor)` pairs from `iterator` to the inference server."""
        ...

    def resume(self) -> None:
        # [FUNCTION DOC]
        # Protocol stub: resume the inference server after the update. No return.
        """Resume the inference server after the weights are updated."""
        ...

    def destroy(self) -> None:
        # [FUNCTION DOC]
        # Protocol stub: release transfer resources at train end. No return.
        """Release transfer resources. Called on train end."""
        ...


class StepIntervalCallback(TrainerCallback):
    # [CLASS DOC]
    # TrainerCallback calling fn (no arguments) at the end of every optimizer step whose global_step is a multiple of
    # every_n_steps. Used for periodic weight sync and per-step metric flushing.
    """
    A callback that calls a function every N optimization steps.
    """

    def __init__(self, fn, every_n_steps: int):
        # [FUNCTION DOC]
        # Stores fn and every_n_steps. Inputs: fn (callable), every_n_steps (int). Outputs: none. Side effects: attributes.
        self.fn = fn
        self.every_n_steps = every_n_steps

    def on_step_end(self, _args, state, _control, **_kwargs):
        # [FUNCTION DOC]
        # Calls fn when state.global_step modulo every_n_steps is 0. Inputs: args, state, control, kwargs (only state used).
        # Outputs: none. Exceptions: from fn; ZeroDivisionError if every_n_steps is 0. Side effects: those of fn.
        if state.global_step % self.every_n_steps == 0:
            self.fn()


class _OptimizerTimeCallback(TrainerCallback):
    # [CLASS DOC]
    # HELPER: TrainerCallback measuring optimizer.step duration into trainer._step_optimizer_s (flushed per step by
    # _log_step_metrics). State: _trainer, _t0 (perf_counter at pre-step).
    """Times the optimizer step, which `training_step` cannot see.

    Without it, clip + step + zero_grad land in the same unmeasured gap as the rollout-queue wait and get read as
    starvation. Covers `optimizer.step()` only, not the gradient clipping that precedes it.
    """

    def __init__(self, trainer: "AsyncGRPOTrainer"):
        # [FUNCTION DOC]
        # HELPER: Stores the trainer and sets _t0 to None. Inputs: trainer. Outputs: none. Side effects: attributes.
        self._trainer = trainer
        self._t0 = None

    def on_pre_optimizer_step(self, _args, _state, _control, **_kwargs):
        # [FUNCTION DOC]
        # HELPER: Records perf_counter into _t0. Inputs: callback args (unused). Outputs: none. Side effects: sets _t0.
        self._t0 = time.perf_counter()

    def on_optimizer_step(self, _args, _state, _control, **_kwargs):
        # [FUNCTION DOC]
        # HELPER: Adds elapsed time since _t0 to trainer._step_optimizer_s.
        # Inputs: callback args (unused). Outputs: none. Exceptions: TypeError if _t0 is None. Side effects: trainer state.
        self._trainer._step_optimizer_s += time.perf_counter() - self._t0


class _TrainBeginCallback(TrainerCallback):
    # [CLASS DOC]
    # HELPER: One-shot on_train_begin setup (guarded by _fired): main process initializes weight transfer, all ranks run
    # the initial _sync_weight, then main process starts the rollout worker.
    """Idempotent train-begin setup: NCCL group setup + cold weight sync to vLLM, then start the rollout worker.
    The weight sync must complete before the worker starts, which the ordering here guarantees.
    """

    def __init__(self, trainer: "AsyncGRPOTrainer"):
        # [FUNCTION DOC]
        # HELPER: Stores trainer and sets _fired False. Inputs: trainer. Outputs: none. Side effects: attributes.
        self._trainer = trainer
        self._fired = False

    def on_train_begin(self, _args, _state, _control, **_kwargs):
        # [FUNCTION DOC]
        # HELPER: Returns if already fired; else marks fired, then on the main process calls weight_transfer
        # .init_weight_transfer (if a backend exists), calls trainer._sync_weight on every rank (collective), and starts
        # the rollout worker on the main process.
        # Inputs: callback args (unused). Outputs: none. Exceptions: from setup/sync/start. Side effects: network, process
        # spawn, model_version increment.
        if self._fired:
            return
        self._fired = True
        if self._trainer.accelerator.is_main_process and self._trainer.weight_transfer is not None:
            self._trainer.weight_transfer.init_weight_transfer()
        self._trainer._sync_weight()
        if self._trainer.accelerator.is_main_process and self._trainer.rollout_worker is not None:
            self._trainer.rollout_worker.start()


class _EpochStopCallback(TrainerCallback):
    # [CLASS DOC]
    # HELPER: Stops training once trained distinct prompt groups (_groups_before_resume plus size of _trained_groups)
    # reach target_groups; the boolean is sum-reduced across ranks so all ranks stop together.
    """Stop after `num_train_epochs` full passes over the prompt dataset.

    An epoch is counted in distinct prompt-groups actually trained (accumulated in the collator, which runs on the main
    process just before the model forward). This is fork-independent: all generations of a prompt and all forked rows
    of a conversation share one `group_id`, so a conversation forking into many rows still counts once. Only the main
    process collates (`dispatch_batches=True`), so the stop decision is reduced across ranks to keep data-parallel
    workers in lockstep.
    """

    def __init__(self, trainer: "AsyncGRPOTrainer", target_groups: int):
        # [FUNCTION DOC]
        # HELPER: Stores trainer and target group count. Inputs: trainer, target_groups (int). Outputs: none.
        self._trainer = trainer
        self._target = target_groups

    def on_step_end(self, args, _state, control, **_kwargs):
        # [FUNCTION DOC]
        # HELPER: Computes whether the target is reached on this rank, sum-reduces a 0/1 tensor across ranks; if any rank
        # reached it, sets control.should_training_stop and, under the STEPS save strategy, control.should_save.
        # Inputs: args, state (unused), control. Outputs: none. Exceptions: collective errors. Side effects: mutates control.
        acc = self._trainer.accelerator
        trained = self._trainer._groups_before_resume + len(self._trainer._trained_groups)
        reached = torch.tensor(int(trained >= self._target), device=acc.device)
        if int(acc.reduce(reached, reduction="sum").item()) >= 1:
            control.should_training_stop = True
            # `max_steps` is only a safety ceiling here, so `DefaultFlowCallback` never reaches its end-of-training save.
            # Save the final step the same way, or the steps since the last multiple of `save_steps` are lost.
            if args.save_strategy == SaveStrategy.STEPS:
                control.should_save = True


def log_rollout_traces(samples: list[RolloutSample], step: int, report_to: list[str], max_traces: int = 16) -> None:
    # [FUNCTION DOC]
    # Logs up to max_traces samples as trackio.Trace objects (messages = prompt + completion; metadata = sample metrics,
    # advantage, group_id, model_version, prompt/completion token counts from completion_mask) under key "rollouts" at
    # step. No-op if samples is empty, "trackio" not in report_to, or trackio unavailable.
    # Inputs: samples, step, report_to, max_traces (default 16). Outputs: none.
    # Exceptions: none (all errors logged as warnings). Side effects: trackio logging.
    """Log rollout samples to trackio as inspectable traces (prompt + completion + reward/advantage per sample).

    Call from rank 0 during training, where the HF trackio callback has already initialised the run; the traces then
    show up under the run's Traces tab so rollouts can be read directly instead of grepping logs. No-op unless trackio
    is the active logging backend (installed and listed in `report_to`). Best-effort: a trackio hiccup must never break
    training.

    Args:
        samples (`list[RolloutSample]`):
            Consumed rollout samples to log; the first `max_traces` are recorded.
        step (`int`):
            Step value the traces are logged under (e.g. the policy version, so the UI groups by policy).
        report_to (`list[str]`):
            The training args' `report_to`; logging happens only when it contains `"trackio"`.
        max_traces (`int`, *optional*, defaults to `16`):
            Maximum number of traces to log per call.
    """
    if not samples or "trackio" not in report_to or not is_trackio_available():
        return
    try:
        traces = [
            trackio.Trace(
                messages=list(sample.prompt) + list(sample.completion),
                metadata={
                    **sample.metrics,  # reward, reward_std, rewards/<func>, tools/call_frequency, tools/failure_frequency
                    "advantage": float(sample.advantage),
                    "group_id": int(sample.group_id),
                    "model_version": int(sample.model_version),
                    "prompt_tokens": len(sample.input_ids) - int(sample.completion_mask.sum()),
                    "completion_tokens": int(sample.completion_mask.sum()),
                },
            )
            for sample in samples[:max_traces]
        ]
        trackio.log({"rollouts": traces}, step=step)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"rollout trace logging skipped: {type(e).__name__}: {e}")


class RolloutQueueDataset(torch.utils.data.IterableDataset):
    # [CLASS DOC]
    # IterableDataset (main process only) that pulls RolloutSamples from the worker queue, drops stale ones, records
    # queue/staleness metrics into the trainer's shared metric sink and yields sample dicts. Also samples traces for
    # trackio: for policy versions divisible by _trace_log_interval (8), buffers _traces_per_log (8) samples once per
    # version and hands them to a daemon thread via a bounded queue (maxsize 4, drop on full).
    # State: queue, model_version_fn, check_health_fn, stale_after_s, metrics, wait_s (accumulated blocking time, reset by
    # the trainer), max_staleness, poll_interval_s, report_to, trace buffer/queue state.
    def __init__(
        self,
        rollout_queue,
        model_version_fn,
        check_health_fn,
        stale_after_s,
        metrics,
        max_staleness=3,
        poll_interval_s=5.0,
        report_to=None,
    ):
        # [FUNCTION DOC]
        # Stores arguments (max_staleness default 3, poll_interval_s default 5.0, report_to default empty list), initializes
        # trace state and starts the "rollout-trace-logger" daemon thread running _drain_traces.
        # Inputs: rollout_queue, model_version_fn, check_health_fn, stale_after_s, metrics, max_staleness, poll_interval_s,
        # report_to. Outputs: none. Exceptions: none. Side effects: starts a thread.
        self.queue = rollout_queue
        self.model_version_fn = model_version_fn
        self.check_health_fn = check_health_fn
        self.stale_after_s = stale_after_s
        # The trainer's metric sink, shared by reference. This dataset lives in the main process (`num_workers=0`,
        # `dispatch_batches=True`), which is also the only process where the queue wait is real, so its metrics need no
        # communication at all — they are appended straight into the sink the trainer reduces in `log()`.
        self.metrics = metrics
        # Blocking-get time, accumulated here and flushed by `training_step` at the optimizer-step boundary — the only
        # place that knows when a step's worth of waiting is done.
        self.wait_s = 0.0
        self.max_staleness = max_staleness
        self.poll_interval_s = poll_interval_s
        self.report_to = report_to or []
        self._trace_buf: list = []
        # Traces exist to be read, not to cover the run: a handful per sampled policy version is enough to eyeball
        # what the generator produced. Both numbers are therefore per *policy version* — `_traces_per_log` samples,
        # once every `_trace_log_interval` versions. Flushing every `_trace_log_interval` *samples* instead fired
        # ~60x per optimizer step at 484 samples/step, and since trackio writes a `metrics` row for every `log()`
        # call — empty, because the payload is all traces — the real per-step metrics ended up buried under ~60x
        # their own number of `{}` rows, which is what made the dashboard slow to query.
        self._traces_per_log = 8
        self._trace_log_interval = 8
        self._last_traced_version = -1
        # Log traces off the training path: __iter__ enqueues, a daemon thread drains. Bounded + drop-on-full so a
        # slow trackio backend never blocks sample delivery.
        self._trace_queue: queue.Queue = queue.Queue(maxsize=4)
        threading.Thread(target=self._drain_traces, name="rollout-trace-logger", daemon=True).start()

    def _drain_traces(self):
        # [FUNCTION DOC]
        # HELPER: Infinite loop: takes (samples, step, context) from the trace queue and runs log_rollout_traces inside the
        # captured contextvars context (so trackio's run is visible).
        # Inputs/Outputs: none. Exceptions: none expected (trace logging swallows errors). Side effects: trackio logging.
        while True:
            samples, step, ctx = self._trace_queue.get()
            # Replay the main-thread context captured at enqueue: trackio's run lives in a ContextVar this daemon
            # thread wouldn't otherwise inherit, so trackio.log() would raise "call init() first".
            ctx.run(log_rollout_traces, samples, step=step, report_to=self.report_to)

    def __iter__(self):
        # [FUNCTION DOC]
        # Infinite generator. For each sample: blocks on queue.get with poll_interval_s timeout, calling check_health_fn on
        # every timeout (never returns on empty); accumulates wait into wait_s and logs waits over 1 s. Computes staleness =
        # current version - sample version; if above max_staleness logs, appends sample/dropped_stale_total and skips.
        # Otherwise appends sample/rollout_queue_size, sample/time_in_queue_s (if enqueued_at set), staleness mean/max, handles
        # trace sampling, and yields a dict with input_ids, completion_mask, old_log_probs, advantage, group_id, metrics.
        # Inputs: none. Outputs: iterator of dicts. Exceptions: RuntimeError from check_health_fn.
        # Side effects: metric sink appends, logging, trace queue writes.
        while True:
            t0 = time.time()
            while True:
                try:
                    sample = self.queue.get(timeout=self.poll_interval_s)
                    break
                except queue.Empty:
                    # Returning here would broadcast None through accelerate's dispatch loop.
                    self.check_health_fn(self.stale_after_s)
            now = time.time()
            wait_s = now - t0
            self.wait_s += wait_s
            if wait_s > 1.0:
                logger.info(f"waited {wait_s:.1f}s for sample (qsize={self.queue.qsize()})")

            version = self.model_version_fn()
            staleness = version - sample.model_version
            if staleness > self.max_staleness:
                logger.info(f"dropping stale sample (staleness={staleness}, max={self.max_staleness})")
                self.metrics["sample/dropped_stale_total"].append(1.0)
                continue  # drop stale, pull next

            # Three different views of the same queue, and they must not be confused with each other:
            #   `sample/rollout_queue_size`  how many scored samples are in it, counted where the TRAINER consumes
            #                                them rather than where the worker filled it
            #   `sample/time_in_queue_s`     how long THIS sample sat in it — its own idle time, and the seconds
            #                                half of its off-policyness
            #   `perf/rollout_wait_s`        how long the TRAINER sat blocked because the queue was empty
            # An empty queue with the trainer waiting is generation-bound; a full queue with no wait is trainer-bound
            # (and then `rollout/backpressure_s` is what generation lost to it).
            self.metrics["sample/rollout_queue_size"].append(float(self.queue.qsize()))
            if sample.enqueued_at is not None:
                self.metrics["sample/time_in_queue_s"].append(now - sample.enqueued_at)
            # Freshness
            self.metrics["sample/staleness_mean"].append(float(staleness))
            self.metrics["sample/staleness_max"].append(float(staleness))

            if version % self._trace_log_interval == 0 and version != self._last_traced_version:
                self._trace_buf.append(sample)
                if len(self._trace_buf) >= self._traces_per_log:
                    try:
                        # Capture the main-thread context (holds trackio's run) so the drain thread can replay it.
                        self._trace_queue.put_nowait((self._trace_buf, version, contextvars.copy_context()))
                    except queue.Full:
                        pass
                    self._trace_buf = []
                    self._last_traced_version = version

            yield {
                "input_ids": sample.input_ids,
                "completion_mask": sample.completion_mask,
                "old_log_probs": sample.old_log_probs,
                "advantage": sample.advantage,
                "group_id": sample.group_id,
                "metrics": sample.metrics,  # per-sample rewards; aggregated by the collator, never sent to the model
            }


class FixedCountBatcher(torch.utils.data.IterableDataset):
    # [CLASS DOC]
    # Planner IterableDataset emitting micro-batches of a fixed sample count partitioned into num_processes TreeRows by
    # LPT on attention-cost load (see docstring). State: dataset, num_processes, microbatch_size.
    """Fixed-count batcher (the planner) wrapping [`RolloutQueueDataset`].

    Buffers `microbatch_size` (= `per_device_train_batch_size × num_processes`) samples, then partitions them across
    the `num_processes` rows (one per DP rank) balanced by attention cost so no rank straggles at the per-micro-batch
    all-reduce. The sample count is fixed, so this does not bound peak memory — use [`TokenBudgetBatcher`] for that.

    A group's samples stay together so they share their prefix, and groups are placed costliest-first into the least
    loaded row (LPT scheduling); attention is superlinear in length, so equal token counts would not equalize
    wall-time. A micro-batch has fewer groups than rows as soon as `num_generations` exceeds
    `per_device_train_batch_size`, so the largest group is halved until every rank gets one: a rank forwarding nothing
    would desync FSDP/EP collectives. With `microbatch_size >= num_processes` such a group always exists.

    Args:
        dataset ([`RolloutQueueDataset`]):
            Source yielding single rollout-sample dicts.
        num_processes (`int`):
            Number of DP ranks; the number of rows (one per rank) in each micro-batch.
        microbatch_size (`int`):
            Number of samples buffered into each micro-batch before it is partitioned and emitted.
    """

    def __init__(self, dataset: "RolloutQueueDataset", num_processes: int, microbatch_size: int):
        # [FUNCTION DOC]
        # Stores dataset, num_processes and microbatch_size. Outputs: none. Side effects: none.
        self.dataset = dataset
        self.num_processes = num_processes
        self.microbatch_size = microbatch_size

    def __iter__(self):
        # [FUNCTION DOC]
        # Buffers microbatch_size samples, groups them by group_id (insertion order), repeatedly splits the largest group into
        # two halves until there are at least num_processes groups, builds a TreeRow per group to price its load, then assigns
        # groups in descending load to the currently least loaded output row (adding all its samples). Yields the list of
        # num_processes TreeRows and resets the buffer. Leftover samples at source end are never emitted (source is infinite).
        # Inputs: none. Outputs: iterator of lists of TreeRow. Exceptions: none expected. Side effects: consumes the source.
        batch = []
        for sample in self.dataset:
            batch.append(sample)
            if len(batch) < self.microbatch_size:
                continue
            groups = {}
            for queued in batch:
                groups.setdefault(queued["group_id"], []).append(queued)
            groups = list(groups.values())
            while len(groups) < self.num_processes:
                largest = groups.pop(max(range(len(groups)), key=lambda i: len(groups[i])))
                groups += [largest[: len(largest) // 2], largest[len(largest) // 2 :]]
            priced = []
            for group in groups:
                priced.append(TreeRow())
                for queued in group:
                    priced[-1].add(queued)
            rows = [TreeRow() for _ in range(self.num_processes)]
            for group in sorted(priced, key=lambda row: row.load, reverse=True):
                row = min(rows, key=lambda row: row.load)
                for queued in group.samples:
                    row.add(queued)
            yield rows
            batch = []


class TokenBudgetBatcher(torch.utils.data.IterableDataset):
    # [CLASS DOC]
    # Planner IterableDataset forming micro-batches of num_processes TreeRows under a per-row unique-token and trained-token
    # budget (see docstring). State: dataset, num_processes, token_budget, metrics (shared sink).
    """Token-budgeted dynamic batcher (the planner) wrapping [`RolloutQueueDataset`].

    Keeps `num_processes` open rows (one per DP rank) and pulls single samples from the source one at a time, dropping
    each into the row with the smallest running attention cost that still fits within `token_budget` forwarded tokens.
    When the next sample fits in no row, the current micro-batch is emitted — a list of `num_processes` groups, already
    partitioned per rank — and a fresh one is started with that sample. The number of samples per row is therefore
    dynamic: short samples pack many per row, long ones pack few, while every row stays within `token_budget` tokens.
    This bounds peak memory independently of `per_device_train_batch_size` and keeps the rows cost-balanced so no rank
    straggles at the per-micro-batch all-reduce.

    The budget counts *unique* tokens, so it fits more samples than their raw length suggests: a sample costs only what
    it does not already share with its row. It also bounds the row's loss terms, which packing does not shrink —
    otherwise rows that share everything would add no tokens and the row would never close. A sample's standalone
    length upper-bounds both, so the drop rule below stays correct.

    Every emitted micro-batch has all `num_processes` rows non-empty (a rank forwarding zero tokens would desync
    FSDP/EP collectives): a micro-batch is only closed once every row holds at least one sample. A sample longer than
    `token_budget` fits in no row, so it is dropped with a warning; set `token_budget` >= the vLLM server's
    `max_model_len` (the cap on prompt + completion) to avoid dropping samples.

    Args:
        dataset ([`RolloutQueueDataset`]):
            Source yielding single rollout-sample dicts.
        num_processes (`int`):
            Number of DP ranks; the number of rows (one per rank) in each micro-batch.
        token_budget (`int`):
            Maximum tokens forwarded in a single row (one rank's forward).
        metrics (`dict`):
            The trainer's metric sink, appended to when a sample is dropped for exceeding the budget.
    """

    def __init__(self, dataset: "RolloutQueueDataset", num_processes: int, token_budget: int, metrics: dict):
        # [FUNCTION DOC]
        # Stores dataset, num_processes, token_budget and metrics. Outputs: none. Side effects: none.
        self.dataset = dataset
        self.num_processes = num_processes
        self.token_budget = token_budget
        self.metrics = metrics  # the trainer's sink, for the drop counter below

    def __iter__(self):
        # [FUNCTION DOC]
        # Starts with num_processes empty TreeRows. For each sample: if its raw length exceeds token_budget, warns, appends
        # batch/dropped_oversize_total and skips. Otherwise loops: ask _place for a row; if found, add and move on; if None,
        # yield the current rows and start fresh empty rows, then retry (an empty row always fits).
        # Inputs: none. Outputs: iterator of lists of TreeRow (all non-empty). Exceptions: none expected.
        # Side effects: metric appends, logging.
        rows = [TreeRow() for _ in range(self.num_processes)]
        for sample in self.dataset:
            n = len(sample["input_ids"])
            if n > self.token_budget:
                # Longer than the whole budget: fits in no row, so drop it (placing it would overshoot the budget
                # or force an empty row that desyncs FSDP/EP collectives).
                logger.warning(
                    f"Dropping a rollout sample of {n} tokens that exceeds token_budget={self.token_budget}. "
                    "Raise token_budget to avoid dropping samples."
                )
                self.metrics["batch/dropped_oversize_total"].append(1.0)
                continue
            while True:
                placed_row = self._place(rows, sample)
                if placed_row is not None:
                    placed_row.add(sample)
                    break
                # No more place in rows
                yield rows
                rows = [TreeRow() for _ in range(self.num_processes)]

    def _place(self, rows: list[TreeRow], sample: dict[str, Any]) -> TreeRow | None:
        # [FUNCTION DOC]
        # HELPER: Chooses a row for sample. First the least loaded fitting row among rows already containing its group_id;
        # if none fits and the sample has home rows and every row is non-empty, returns None (close micro-batch); otherwise
        # the least loaded fitting row overall (None if none fit).
        # Inputs: rows, sample. Outputs: TreeRow or None. Exceptions: none. Side effects: none.
        """Pick the row `sample` belongs in, or `None` when the micro-batch has to close first.

        A sample is cheapest in a row that already holds its `group_id`, so those rows get first refusal — all of them,
        since an earlier split can leave one group on several rows. When none of them has room the sample can still be
        split off, but only to keep a rank from forwarding nothing; otherwise closing the micro-batch is cheaper, since
        the sample then keeps its prefix in a fresh row.
        """

        def least_loaded(candidates: list[TreeRow]) -> TreeRow | None:
            # [FUNCTION DOC]
            # HELPER: Among candidates that fit the sample within token_budget, returns the one with minimal load, or None.
            # Inputs: candidates. Outputs: TreeRow or None. Side effects: none.
            return min(
                (row for row in candidates if row.fits(sample, self.token_budget)),
                key=lambda row: row.load,
                default=None,
            )

        home = [row for row in rows if sample["group_id"] in row.groups]
        placed = least_loaded(home)
        if placed is not None:
            return placed
        if home and all(row.samples for row in rows):
            return None
        return least_loaded(rows)


class _EmptyIterableDataset(torch.utils.data.IterableDataset):
    # [CLASS DOC]
    # HELPER: IterableDataset yielding nothing, used on non-main ranks where the dispatching dataloader never iterates.
    """Placeholder for non-rank-0 processes. Never actually iterated."""

    def __iter__(self):
        # [FUNCTION DOC]
        # HELPER: Returns an empty iterator. Inputs/outputs trivial; no exceptions or side effects.
        return iter([])


@dataclass
class DataCollatorForRollout(DataCollatorMixin):
    # [CLASS DOC]
    # Dataclass collator (the packer) turning one pre-partitioned micro-batch (list of num_processes TreeRows) into a
    # rectangular padded batch (see docstring). Fields: pad_token_id, num_processes (default 1), return_tensors "pt",
    # groups_trained (shared set of trained group ids, used for epoch counting), metrics (shared sink), token_budget
    # (0 when batching by count). Inherits DataCollatorMixin (dispatches to torch_call).
    """
    Padding-free collator (the packer) for rollout samples. Packs a micro-batch into `num_processes` rows (one per DP
    rank) and, for each row, the flat list of loss terms taken over it: which packed position predicts which target,
    with what old log-probability, advantage and originating completion. Rows are padded only to the longest row, so
    the batch stays rectangular for `DataLoaderDispatcher` to scatter row `i` -> rank `i`; this inter-rank padding is
    stripped per-rank in `compute_loss`.

    Each row is a [`~packing.TreeRow`], so a token shared by several of its samples is forwarded once.

    The micro-batch arrives already partitioned into `num_processes` rows by the upstream planner
    ([`FixedCountBatcher`] or [`TokenBudgetBatcher`]) — which balances each row's attention cost to avoid stragglers at
    the gradient all-reduce — so the collator only packs and tensorizes the given rows.

    Args:
        pad_token_id (`int`):
            Token id used to pad `input_ids`.
        num_processes (`int`, *optional*, defaults to `1`):
            Number of DP ranks; the micro-batch is packed into this many rows.
        metrics (`dict[str, list]`, *optional*):
            The trainer's metric sink, appended to with this micro-batch's sample and packing metrics.
        token_budget (`int`, *optional*, defaults to `0`):
            Per-row token cap of the planner, or `0` when batching by fixed sample count.
    """

    pad_token_id: int
    num_processes: int = 1
    return_tensors: str = "pt"
    # Distinct prompt-group ids, it counts exactly the prompt-groups that get trained
    groups_trained: set[int] = field(default_factory=set)
    # The trainer's metric sink, shared by reference (like `groups_trained`).
    metrics: dict[str, list] = field(default_factory=lambda: defaultdict(list))
    # Per-row token cap of the planner,
    token_budget: int = 0

    def torch_call(self, examples: list[Any]) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Packs each row with TreeRow.pack and pads every key across rows to the longest (padding: pad_token_id for input_ids,
        # -100 for shift_labels, -1 for token_index and tree index/cu_seqlens keys, 0 otherwise); adds attention_mask (1 for
        # real positions). Adds group ids to groups_trained. Adds per-rank-repeated tensors of length num_processes:
        # global_n_tokens (total loss terms), global_n_forward_tokens (total packed tokens), mean_seq_len (packed tokens per
        # sample). Logs metrics via _log_metrics.
        # Inputs: examples (length-1 list holding the rows). Outputs: dict of tensors.
        # Exceptions: ValueError if examples does not have exactly one element. Side effects: groups_trained and metrics updates.
        # The dataloader uses batch_size=1 over a planner that pre-partitions each micro-batch into `num_processes`
        # rows, so `examples` is a length-1 list holding that single micro-batch (one row per rank).
        (rows,) = examples

        packed = [row.pack() for row in rows]
        padding = {
            "input_ids": self.pad_token_id,
            "shift_labels": -100,
            "token_index": -1,
            "tree_kv_index": -1,
            "tree_cu_seqlens_q": -1,
            "tree_cu_seqlens_k": -1,
        }
        batch = {key: pad([p[key] for p in packed], padding_value=padding.get(key, 0)) for key in packed[0]}
        batch["attention_mask"] = pad([torch.ones_like(p["input_ids"]) for p in packed], padding_value=0)

        all_examples = [example for row in rows for example in row.samples]
        self.groups_trained.update(example["group_id"] for example in all_examples)

        # Repeated per rank so that DataLoaderDispatcher (dispatch_batches=True) slices correctly on dim=0
        n_trained_tokens = sum(p["token_index"].numel() for p in packed)
        n_forward_tokens = sum(p["input_ids"].numel() for p in packed)
        batch["global_n_tokens"] = torch.full((self.num_processes,), float(n_trained_tokens))
        batch["global_n_forward_tokens"] = torch.full((self.num_processes,), float(n_forward_tokens))
        batch["mean_seq_len"] = torch.full((self.num_processes,), n_forward_tokens / len(all_examples))

        self._log_metrics(rows=rows, all_examples=all_examples, padded=batch["attention_mask"])
        return batch

    def _log_metrics(self, rows: list[TreeRow], all_examples: list[dict[str, Any]], padded: torch.Tensor) -> None:
        # [FUNCTION DOC]
        # HELPER: Appends to the metric sink: per-sample forwarded/trained token mean and max, batch/masked_token_frac rate,
        # nan-aware mean of every per-sample metric key (union of keys), samples_per_row, row token mean/max, row_imbalance
        # (max load / mean load), packing_ratio (raw tokens, unique tokens), row_fill_frac when token_budget is set, and
        # pad_frac (padded positions over total) from the attention mask.
        # Inputs: rows, all_examples, padded (attention mask). Outputs: none. Exceptions: ZeroDivisionError on empty input or
        # zero total load. Side effects: metric sink.
        """Append this micro-batch's sample and packing metrics straight into the trainer's sink.

        Rank 0 holds the whole micro-batch, so the per-sample rewards can be aggregated here instead of being packed
        into NaN-padded tensors, broadcast to every rank and reduced back — which is what this trainer used to do to
        compute a number rank 0 already had.
        """
        # A training row's tokens split in two: those the loss is taken over (`completion_mask == 1`) and those merely
        # forwarded — the prompt, tool results, and any tail a realign demoted to context. FLOPs are spent on both, so
        # both are worth a number, and their ratio is what says how much of the forward earns no gradient.
        forwarded = [len(example["input_ids"]) for example in all_examples]
        trained = [int(example["completion_mask"].sum()) for example in all_examples]
        self.metrics["sample/forwarded_tokens_mean"].append(sum(forwarded) / len(forwarded))
        self.metrics["sample/forwarded_tokens_max"].append(float(max(forwarded)))
        self.metrics["sample/trained_tokens_mean"].append(sum(trained) / len(trained))
        self.metrics["batch/masked_token_frac"].append((sum(forwarded) - sum(trained), sum(forwarded)))

        # Per-sample rewards, nan-aware: a reward func may return None for an unscorable sample, and a sample for which
        # every func returned None carries NaN rather than a misleading 0.
        # Union of keys, not the first sample's: `tools/*` is stamped per group, so a micro-batch mixes samples that
        # have those keys with samples that do not. Each key averages over the samples that carry it.
        keys = dict.fromkeys(key for example in all_examples for key in example["metrics"])
        for key in keys:
            values = [example["metrics"][key] for example in all_examples if key in example["metrics"]]
            valid = [v for v in values if not math.isnan(v)]
            self.metrics[key].append(sum(valid) / len(valid) if valid else float("nan"))

        row_tokens = [len(row.tokens) for row in rows]
        loads = [row.load for row in rows]
        self.metrics["batch/samples_per_row"].append(len(all_examples) / len(rows))
        self.metrics["batch/row_tokens_mean"].append(sum(row_tokens) / len(row_tokens))
        self.metrics["batch/row_tokens_max"].append(float(max(row_tokens)))
        self.metrics["batch/row_imbalance"].append(max(loads) / (sum(loads) / len(loads)))
        self.metrics["batch/packing_ratio"].append((sum(forwarded), sum(row_tokens)))
        if self.token_budget:
            self.metrics["batch/row_fill_frac"].append((sum(row_tokens), self.token_budget * len(row_tokens)))
        # Inter-rank padding, added so the batch is rectangular for the dispatcher. It costs broadcast bytes only:
        # `compute_loss` strips it before the forward, so no FLOPs are spent on it.
        self.metrics["batch/pad_frac"].append((padded.numel() - int(padded.sum()), padded.numel()))


def _iter_vllm_named_params(model: torch.nn.Module) -> Iterator[tuple[str, torch.nn.Parameter]]:
    # [FUNCTION DOC]
    # HELPER: Yields (vLLM checkpoint name, parameter) for parameters sent to vLLM in merged sync.
    # Behavior: skips frozen params unless the model is PEFT; strips "module." prefix and "_checkpoint_wrapped_module.",
    # then "base_model.model." prefix and ".base_layer"; under PEFT skips names containing the adapter prefix (e.g. lora_);
    # skips "original_module" entries and removes "modules_to_save.default.". Parameters yielded as-is (possibly DTensor).
    # Inputs: model. Outputs: iterator. Exceptions: none. Side effects: none.
    """Yield `(name, param)` for every parameter the vLLM server receives, under its base-checkpoint name.

    The weight manifest built once in `__init__` and the stream sent by `_streaming_iter` must agree
    element-for-element — the server sizes its receive buffers from the manifest — so both walk this generator rather
    than each re-deriving the name cleanup. Parameters are yielded as-is: `DTensor.shape` is the global shape, so
    building the manifest costs no all-gather.
    """
    is_peft = is_peft_model(model)
    for name, param in model.named_parameters():
        # Frozen parameters (a VLM's vision tower) never change, so they are never sent. Under PEFT the filter is
        # inverted: the trainable parameters are the adapter tensors vLLM never sees, and it is the frozen base
        # weights that move once the adapter has been merged into them.
        if not is_peft and not param.requires_grad:
            continue
        # DDP/FSDP1 wrapping and gradient checkpointing, avoids vllm module not exist error
        name = name.removeprefix("module.").replace("_checkpoint_wrapped_module.", "")
        # When using PEFT, we need to recover the original parameter name
        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
        # Skip PEFT layers: they don't exist in vLLM, and they are merged already.
        if is_peft and model.prefix in name:
            continue
        # When module to save, remove its prefix and discard the original module
        if "original_module" in name:
            continue
        name = name.replace("modules_to_save.default.", "")
        yield name, param


# Ranks vLLM builds stacked LoRA buffers for. Any other value crashes engine construction.
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


def round_lora_rank(adapter_rank: int) -> int:
    # [FUNCTION DOC]
    # Returns the smallest value in VLLM_LORA_RANKS at least adapter_rank.
    # Inputs: adapter_rank (int). Outputs: int. Exceptions: ValueError if above 512. Side effects: none.
    """
    Round a LoRA rank up to the smallest `max_lora_rank` vLLM accepts.

    `max_lora_rank` is a capacity bound rather than the served rank, so an `r=4` adapter is served correctly under `8`.

    Args:
        adapter_rank (`int`):
            Rank of the adapter to serve.

    Returns:
        `int`:
            Smallest value in `VLLM_LORA_RANKS` greater than or equal to `adapter_rank`.

    Examples:

    ```python
    >>> round_lora_rank(4)
    8
    ```
    """
    for rank in VLLM_LORA_RANKS:
        if rank >= adapter_rank:
            return rank
    raise ValueError(
        f"LoRA rank {adapter_rank} is larger than the largest rank vLLM can serve ({VLLM_LORA_RANKS[-1]}). Lower `r` "
        f"(and `rank_pattern`) to sync the adapter to vLLM, or sync merged weights instead."
    )


def validate_lora_for_vllm_sync(model: "PeftModel") -> "LoraConfig":
    # [FUNCTION DOC]
    # Checks that the single active adapter is a plain LoRA servable by vLLM and returns its config.
    # Rejects (ValueError): not exactly one active adapter, non-LoraConfig, modules_to_save, use_dora, bias other than
    # "none", target_parameters, or target_modules (string treated as one entry) mentioning lm_head or embed_tokens.
    # Inputs: model (PeftModel). Outputs: LoraConfig. Side effects: none.
    """
    Return the active adapter's config, checking it can be served as a plain vLLM LoRA adapter.

    vLLM's LoRA manager only knows about `lora_A` / `lora_B` deltas on a fixed set of linear layers. Anything else a
    PEFT config can express — fully-trained modules, DoRA magnitudes, trained biases, fused-MoE parameter LoRA — has no
    home in the adapter checkpoint vLLM loads, and is silently ignored (or worse, misread) at serving time. Every such
    config is rejected here so the caller can fall back to syncing merged weights, which handles all of them.

    Args:
        model ([`~peft.PeftModel`]):
            PEFT model whose active adapter is checked.

    Returns:
        [`~peft.LoraConfig`]:
            Config of the single active adapter.
    """
    if len(model.active_adapters) != 1:
        raise ValueError(
            f"Adapter-only vLLM sync serves a single adapter, but {len(model.active_adapters)} are active "
            f"({model.active_adapters}). Activate one adapter, or sync merged weights instead."
        )
    peft_config = model.peft_config[model.active_adapters[0]]
    if not isinstance(peft_config, LoraConfig):
        raise ValueError(
            f"Adapter-only vLLM sync supports LoRA adapters only, got `{type(peft_config).__name__}`. Sync merged "
            f"weights instead."
        )
    if peft_config.modules_to_save:
        raise ValueError(
            f"`modules_to_save={peft_config.modules_to_save}` cannot ride in a LoRA adapter: those modules are "
            f"fully trained, not a low-rank delta, so vLLM would keep serving the base checkpoint's copy. Sync "
            f"merged weights instead."
        )
    if peft_config.use_dora:
        raise ValueError(
            "DoRA adapters are not servable by vLLM: the magnitude vector has no slot in the adapter format. Sync "
            "merged weights instead."
        )
    if peft_config.bias != "none":
        raise ValueError(
            f"`bias='{peft_config.bias}'` trains bias terms, which are not part of a LoRA adapter and would be "
            f"dropped on the way to vLLM. Use `bias='none'`, or sync merged weights instead."
        )
    if peft_config.target_parameters:
        raise ValueError(
            f"`target_parameters={peft_config.target_parameters}` (fused-MoE parameter LoRA) needs the adapter to be "
            f"declared with `is_3d_lora_weight=True` against a server started with "
            f"`--enable-mixed-moe-lora-format`. vLLM does not inspect the checkpoint, so a wrong declaration loads "
            f"the weights into the wrong stacked buffers and silently produces garbage. Sync merged weights instead."
        )
    target_modules = peft_config.target_modules or []
    if isinstance(target_modules, str):  # a regex matching module names, rather than a collection of them
        target_modules = [target_modules]
    if any("lm_head" in module or "embed_tokens" in module for module in target_modules):
        raise ValueError(
            f"`target_modules={peft_config.target_modules}` targets the head or the embeddings. vLLM restricts LoRA "
            f"on those layers and drops the head delta from `prompt_logprobs` entirely "
            f"(https://github.com/vllm-project/vllm/issues/51594), so the trainer and the server would disagree on "
            f"every logprob. Sync merged weights instead."
        )
    return peft_config


def save_lora_adapter(model: "PeftModel", accelerator: Accelerator, adapter_name: str, dest_dir: str) -> None:
    # [FUNCTION DOC]
    # Collective adapter save (all ranks call it).
    # Behavior: selects state_dict entries containing "lora_" and ".<adapter_name>."; moves CPU shards to the accelerator
    # device, materializes DTensors with full_tensor, stores detached CPU copies. Non-main ranks return there. Main rank:
    # recreates a hidden temp dir ".<basename>.tmp" next to dest_dir, save_pretrained with only that adapter and the
    # materialized state dict (safetensors), picks the saved directory (temp root for "default", else its adapter
    # subdirectory), deletes dest_dir and renames into place, then removes the temp dir.
    # Inputs: model, accelerator, adapter_name, dest_dir. Outputs: none.
    # Exceptions: OSError from filesystem ops; collective errors. Side effects: GPU all-gathers, files written/deleted.
    """
    Save one PEFT adapter to `dest_dir`, gathering sharded parameters as needed.

    This is a collective: every process must call it, not only the main one, because materializing a sharded parameter
    all-gathers across ranks.

    `dest_dir` is named after the adapter and holds `adapter_config.json` at its root, the layout vLLM's filesystem
    LoRA resolver expects when it serves `<cache_dir>/<adapter_name>`. It is published with an atomic rename, so a
    server reading it over a shared filesystem never observes a half-flushed adapter.

    Args:
        model ([`~peft.PeftModel`]):
            PEFT model holding the adapter.
        accelerator ([`~accelerate.Accelerator`]):
            Accelerator whose device sharded parameters are gathered on.
        adapter_name (`str`):
            Name of the adapter to save.
        dest_dir (`str`):
            Directory the adapter is published to. Replaced if it already exists.
    """
    state_dict = {
        name: param for name, param in model.state_dict().items() if "lora_" in name and f".{adapter_name}." in name
    }
    materialized = {}
    for name, param in state_dict.items():
        # With `fsdp_offload_params`, the local shard lives on CPU; the all-gather needs it on device.
        if param.is_cpu:
            param = param.to(accelerator.device)
        if isinstance(param, DTensor):  # FSDP2
            param = param.full_tensor()
        materialized[name] = param.detach().cpu()

    if not accelerator.is_main_process:
        return
    # Staged under a dot-prefixed name: `dest_dir`'s parent is an adapter namespace, so a plain `<name>.tmp` sibling
    # would be a half-written adapter sitting in it, and a crash mid-save would leak one there permanently.
    tmp_dir = os.path.join(os.path.dirname(dest_dir), f".{os.path.basename(dest_dir)}.tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    model.save_pretrained(tmp_dir, selected_adapters=[adapter_name], state_dict=materialized, safe_serialization=True)
    # PEFT nests every adapter but `"default"` in a subdirectory, while vLLM expects `adapter_config.json` at the root
    # of the path it is handed.
    saved_dir = tmp_dir if adapter_name == "default" else os.path.join(tmp_dir, adapter_name)
    shutil.rmtree(dest_dir, ignore_errors=True)
    os.rename(saved_dir, dest_dir)  # atomic within one filesystem
    shutil.rmtree(tmp_dir, ignore_errors=True)


def select_adapter_sync(server_info: dict, model: "PeftModel", args: AsyncGRPOConfig) -> bool:
    # [FUNCTION DOC]
    # Decides adapter-only sync (True) vs merged sync (False).
    # Behavior: False with a warning when server lora_config is None, or data_parallel_size above 1, or
    # validate_lora_for_vllm_sync fails. Computes adapter rank as max of r and rank_pattern values; raises if the server
    # max_lora_rank is lower (suggesting round_lora_rank); raises if server max_loras is below max_staleness + 2.
    # Otherwise True.
    # Inputs: server_info (dict), model, args. Outputs: bool.
    # Exceptions: ValueError for rank/slot misconfiguration; KeyError on malformed server_info. Side effects: logging.
    """
    Decide whether weight syncs push the adapter alone instead of merged base weights.

    A pure function of what the server reported and what is about to be trained, so it can be tested with the dict
    `/server_info` returns rather than with a server. Anything vLLM's adapter format cannot express falls back to the
    merged path with a warning. A server that is merely misconfigured — too low a rank, too few adapter slots — raises
    instead, rather than silently syncing 100x the bytes its launch line asked for.

    Args:
        server_info (`dict`):
            The server's resolved `vllm_config`, as returned by [`VLLMClient.get_server_info`]. Only `lora_config` and
            `parallel_config.data_parallel_size` are read.
        model ([`~peft.PeftModel`]):
            The PEFT-wrapped model about to be trained.
        args ([`AsyncGRPOConfig`]):
            The training arguments; `max_staleness` sets how many adapter versions the server must hold at once.

    Returns:
        `bool`:
            Whether syncs should push only the adapter.
    """
    # `lora_config` is `None` unless the server was started with `--enable-lora`, making it the capability probe.
    server_lora_config = server_info["lora_config"]
    if server_lora_config is None:
        logger.warning(
            "Training a PEFT model against a vLLM server started without `--enable-lora`: every weight sync has "
            "to merge the adapter into the base model and push the full weights. Restart the server with "
            "`--enable-lora --max-lora-rank <r>` and `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1` in its environment to "
            "push only the adapter instead (~1% of the bytes)."
        )
        return False

    # `/v1/load_lora_adapter` reaches only the data-parallel replica that answered it, leaving the others on
    # the base model. The merged path has no such problem, so hand data-parallel servers to it.
    if server_info["parallel_config"]["data_parallel_size"] > 1:
        logger.warning(
            "Falling back to merged-weight vLLM sync: the server runs with `--data-parallel-size > 1`, where a "
            "loaded adapter reaches only the replica that received the request."
        )
        return False

    try:
        peft_config = validate_lora_for_vllm_sync(model)
    except ValueError as error:
        logger.warning(f"Falling back to merged-weight vLLM sync. {error}")
        return False

    # `rank_pattern` can lift individual modules above the base `r`.
    adapter_rank = max([peft_config.r, *peft_config.rank_pattern.values()])
    if server_lora_config["max_lora_rank"] < adapter_rank:
        raise ValueError(
            f"The adapter has rank {adapter_rank}, but the vLLM server was started with `--max-lora-rank "
            f"{server_lora_config['max_lora_rank']}`. Restart it with `--max-lora-rank "
            f"{round_lora_rank(adapter_rank)}`."
        )
    # `max_staleness` bounds how many policies back a sample may still be consumed from, so `max_staleness + 1`
    # adapters can be named by in-flight requests at once. Each sync loads the new version before unloading the one
    # that just fell out of that window, so the server briefly holds one more — and vLLM does not reject the extra
    # load, it silently evicts the oldest still-servable adapter.
    min_loras = args.max_staleness + 2
    if server_lora_config["max_loras"] < min_loras:
        raise ValueError(
            f"The vLLM server was started with `--max-loras {server_lora_config['max_loras']}`, but "
            f"`max_staleness={args.max_staleness}` lets requests still name {min_loras - 1} distinct policy "
            f"versions, and each sync loads the next one before unloading the oldest. Restart the server with "
            f"`--max-loras {min_loras}`, or lower `max_staleness`."
        )
    return True


class AsyncGRPOTrainer(_BaseTrainer):
    # [CLASS DOC]
    # Asynchronous GRPO trainer built on _BaseTrainer (transformers Trainer). Main process owns the rollout worker,
    # rollout queue, vLLM client and weight transfer; all ranks train on tree-packed rows dispatched from rank 0.
    # Loss: token-level clipped PPO surrogate with asymmetric epsilon (no KL/ref model), normalized by global trained token
    # count per rank and gradient accumulation (loss_is_scaled_for_ga True), plus optional MoE aux loss.
    # Key state: epsilon_low/high, temperature, eos_token_ids, router_aux_loss_coef/aux_loss_enabled, _is_vlm,
    # _metrics sink per mode, per-step accumulators (_step_*), model_version, LoRA sync state (_lora_sync, _lora_name
    # "trl-policy", _lora_dir under output_dir/.vllm_lora, _adapter_name), epoch stop bookkeeping (_trained_groups,
    # _groups_before_resume, _epoch_stop_groups), rollout_worker, rollout_queue, weight_transfer, vllm_client.
    # Flow: __init__ builds model/worker/callbacks; on train begin initial sync and worker start; get_train_dataloader
    # builds the queue pipeline; compute_loss per micro-batch; callbacks sync weights and flush metrics; log reduces
    # metrics; _save_checkpoint and _inner_training_loop handle resume state and teardown.
    # KNOWN ISSUE: the class docstring says extra train_dataset columns are ignored, but the rollout worker forwards every
    # non-prompt column to reward functions as reward kwargs; it also says the default scheduler is linear with warmup
    # while AsyncGRPOConfig defaults lr_scheduler_type to constant.
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
    Models](https://huggingface.co/papers/2402.03300). This trainer is the asynchronous version of GRPO, where
    generation is offloaded to an external vLLM server that runs asynchronously alongside training, decoupling rollout
    from the gradient update loop.

    Example:

    ```python
    >>> from trl.experimental.async_grpo import AsyncGRPOTrainer
    >>> from trl.rewards import accuracy_reward
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    >>> trainer = AsyncGRPOTrainer(
    ...     model="Qwen/Qwen2.5-0.5B-Instruct",
    ...     reward_funcs=accuracy_reward,
    ...     train_dataset=dataset,
    ... )
    >>> trainer.train()
    ```

    Args:
        model (`str`):
            Model to be trained. Must be a string, being the *model id* of a pretrained model hosted inside a model
            repo on huggingface.co, or a path to a *directory* containing model weights saved using
            [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
            with the architecture declared in its config: [`~transformers.AutoModelForImageTextToText`] for a
            vision-language checkpoint (whose vision tower is loaded but frozen, see [Vision-language
            models](async_grpo_trainer#vision-language-models)), [`~transformers.AutoModelForCausalLM`] otherwise. The
            model name is also used to identify the model on the vLLM server used for generation.
        reward_funcs (`RewardFunc | list[RewardFunc]`, *optional*):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. May be omitted when the reward is supplied
            by the environment through `environment_factory` (see below). Can be either:

            - A single reward function: The function is provided with the prompts and the generated completions, plus
              any additional columns in the dataset. It should return a list of rewards. Reward functions can be either
              synchronous or asynchronous and can also return `None` when the reward is not applicable to those
              samples. This is useful for multi-task training where different reward functions apply to different types
              of samples. When a reward function returns `None` for a sample, that reward function is excluded from the
              reward calculation for that sample. For more details, see [Using a custom reward
              function](#using-a-custom-reward-function).
            - A list of reward functions, where each item is a reward function as described above. Rewards from all
              functions are summed.

            Unlike [`GRPOTrainer`], rewards are computed in a spawned child process, so each reward function (along
            with `tools` and `environment_factory`) must be picklable: use a module-level function,
            `functools.partial`, or a callable class instance — lambdas and closures will fail at startup. Tools are
            registered under their `__name__`, so they must be module-level functions. The child process also runs with
            `CUDA_VISIBLE_DEVICES=""`, so a GPU-backed reward model runs on CPU (slow), not the trainer's GPU.
        args ([`experimental.async_grpo.AsyncGRPOConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`], *optional*):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset are
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).

            May be omitted only when an `environment_factory` is provided and the environment owns (or procedurally
            generates) the data, returning the prompt from its `reset()` method. In that case, `max_steps` must be set
            to define the training length.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`]. A
            padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
            `tokenizer.eos_token` will be used as the default.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of `AdamW` on your
            model and a scheduler given by [`~transformers.get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped. When the vLLM server was
            started with `--enable-lora`, a LoRA adapter is synced to it as an adapter rather than as merged weights;
            see the LoRA section of the AsyncGRPO documentation.
        tools (list of `Callable`, *optional*):
            A list of callable tool functions (sync or async) that the model can invoke during generation. Each tool
            should be a standard Python function with properly type-hinted arguments and return values, and a
            Google-style docstring describing its purpose, arguments, and return value. For more details, see:
            https://huggingface.co/docs/transformers/en/chat_extras#passing-tools. The model uses the function's name,
            type hints, and docstring to determine how to call it. Ensure that the model's chat template supports tool
            use and that it has been fine-tuned for tool calling.
        environment_factory (`EnvironmentFactory` or `dict[str, EnvironmentFactory]`, *optional*):
            A callable that creates and returns an environment instance, or a dictionary mapping environment names to
            such callables. The environment class should define methods that can be invoked as tools during generation.
            Each method should comply with the same requirements as the `tools` described above. The environment must
            also implement a callable `reset` method that can be used to reset state between generations. The `reset`
            method should return either `None` or a string: when it returns a string, that string is appended to the
            last user message before generation. The environment may also define a `get_reward` method taking no
            argument and returning a `float`: when present, the environment owns the reward, and `get_reward` is called
            once per completed rollout to score it from the environment's internal state. It acts as an additional
            reward source (with weight 1, logged under the environment's class name) alongside `reward_funcs`, which
            then becomes optional.

            With a single callable, every example uses the same environment, with one instance per rollout so their
            interactions stay isolated. With a dictionary, each example must carry an `environment` field selecting its
            environment by name, and only that environment's tools are exposed in its prompt — letting a single run mix
            tasks (e.g. a coding environment and a game). This feature is experimental and may change or be removed at
            any time without prior notice.
        rollout_worker (`RolloutWorkerProtocol`, *optional*):
            Custom rollout worker implementing [`RolloutWorkerProtocol`]. If `None`, a default [`AsyncRolloutWorker`]
            is created, which spawns a CUDA-free child process and scores completions with the trainer's
            `reward_funcs`. Pass a custom worker to plug in a different rollout/scoring backend instead — for example,
            one that runs reward models on their own GPUs.
        weight_transfer (`WeightTransferProtocol`, *optional*):
            Custom weight-sync backend implementing [`WeightTransferProtocol`]. If `None`, a default
            [`WeightTransferClient`] is created that streams the trainer's weights into the config's vLLM server over
            NCCL. This is independent of `rollout_worker`: a custom rollout worker still gets weight sync. Pass a no-op
            implementation to disable trainer-side weight sync.
    """

    loss_is_scaled_for_ga = True

    _tag_names = ["trl", "async-grpo"]
    _name = "AsyncGRPO"
    _paper = {
        "title": "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
        "id": "2402.03300",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{shao2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }"""),
    }

    def __init__(
        self,
        model: str,
        reward_funcs: RewardFunc | list[RewardFunc] | None = None,
        args: AsyncGRPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: "PeftConfig | None" = None,
        tools: list[Callable] | None = None,
        environment_factory: EnvironmentFactory | dict[str, EnvironmentFactory] | None = None,
        rollout_worker: RolloutWorkerProtocol | None = None,
        weight_transfer: WeightTransferProtocol | None = None,
    ):
        # [FUNCTION DOC]
        # Behavior, in order: default args named "<model basename>-AsyncGRPO"; sets epsilons (epsilon_high falls back to
        # epsilon) and temperature; registers tree attention; seeds; loads the model from path with attn_implementation tree
        # attention and model_init_kwargs defaults (trust_remote_code, dtype). Rejects use_liger_kernel, layer types other than
        # full/sliding attention, and attention sinks. Resolves router_aux_loss_coef (config value or 0.0 when unset; raises if
        # set on a non-MoE model) and aux_loss_enabled. For VLMs freezes everything except the text model and output
        # embeddings. Adds the fused lm head (log_probs and entropy at the training temperature). Loads tokenizer if needed,
        # sets pad token to eos when missing and mirrors pad id to model configs; collects eos ids (generation config plus the
        # tokenizer's). Applies PEFT (validating type and not already PEFT), rejects adapters on lm_head, enables input grads
        # under gradient checkpointing. Normalizes reward_funcs to a list. Without a dataset requires a single
        # environment_factory and positive max_steps and builds a placeholder dataset of empty user prompts
        # (per_device_train_batch_size times gradient_accumulation_steps rows). Calls the base constructor.
        # Then: peak FLOPs for MFU; epoch-driven stop when max_steps is unset and the dataset has a length (target groups =
        # ceil(epochs x prompts); max_steps set to a ceiling assuming max_tool_calling_iterations+1 or 32 rows per
        # conversation); auto max_inflight_tasks = max(max_staleness, 1) x samples per step; forces ignore_data_skip True
        # (warning if it was False); initializes metric and step state. Main process creates VLLMClient. For PEFT without an
        # injected weight_transfer, rank 0 runs _init_lora_sync and the result (or exception) is broadcast and raised on all.
        # Main process: weight_transfer = injected, else None for LoRA sync, else WeightTransferClient with a manifest of
        # names/dtypes/shapes from _iter_vllm_named_params; rollout_worker = injected or AsyncRolloutWorker built from args;
        # rollout_queue from the worker. Other ranks set these to None. Adds callbacks: optimizer timing, train begin,
        # weight sync every weight_sync_steps, step metrics every step, and epoch stop when applicable.
        # Inputs: model (str path/id), reward_funcs, args, train_dataset, processing_class, callbacks, optimizers,
        # peft_config, tools, environment_factory, rollout_worker, weight_transfer.
        # Outputs: none.
        # Exceptions: NotImplementedError (liger), ValueError (unsupported architecture, MoE coef, PEFT conflicts, missing
        # dataset requirements, LoRA server misconfig), ImportError/TypeError (peft). Side effects: model loading, network
        # probing of vLLM, broadcast collectives, mutates args (max_steps, max_inflight_tasks, ignore_data_skip).
        # Args
        if args is None:
            model_name = model.split("/")[-1]
            args = AsyncGRPOConfig(f"{model_name}-AsyncGRPO")

        # Training arguments
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        self.temperature = args.temperature

        register_tree_attention()

        # Model
        # PEFT initializes the adapter weights randomly, so set_seed must be done before creating the model to ensure
        # reproducibility.
        set_seed(args.seed)
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)
        model_init_kwargs.setdefault("dtype", args.dtype)
        model_revision = model_init_kwargs.get("revision")
        model = create_model_from_path(
            model,
            device_map=None,
            attn_implementation=TREE_ATTENTION,
            **model_init_kwargs,
        )

        if args.use_liger_kernel:
            raise NotImplementedError("`use_liger_kernel` is not supported yet.")

        text_config = model.config.get_text_config()
        layer_types = set(getattr(text_config, "layer_types", None) or []) - {"full_attention", "sliding_attention"}
        # Check tree attention compat
        if layer_types:
            raise ValueError(
                f"{type(model).__name__} has {sorted(layer_types)} layers. Tree packing only routes attention layers "
                "along each sample's own path; a recurrent or convolutional layer would scan across samples."
            )
        if any(name.endswith(".sinks") for name, _ in model.named_parameters()):
            raise ValueError(
                f"{type(model).__name__} uses attention sinks, which the FA3 kernel used by tree packing does not "
                "support."
            )

        # MoE load-balancing auxiliary loss. `output_router_logits` in the config means the model returns its router
        # logits; `router_aux_loss_coef` is the architecture's own coefficient, 0.0 when the config declares none.
        coef = args.router_aux_loss_coef
        self.router_aux_loss_coef = getattr(text_config, "router_aux_loss_coef", 0.0) if coef is None else coef
        if coef and not hasattr(text_config, "output_router_logits"):
            raise ValueError(
                f"`router_aux_loss_coef` is set to {coef} but {type(model).__name__} is not a Mixture-of-Experts model "
                f"that returns its router logits, so there is no auxiliary loss to weight."
            )
        self.aux_loss_enabled = hasattr(text_config, "output_router_logits") and self.router_aux_loss_coef != 0.0

        self._is_vlm = text_config is not model.config
        if self._is_vlm:
            # Train the text model only. It is located through the text config, since module names differ across
            # architectures (`model.language_model` for Qwen-VL and Gemma 3, `model.text_model` for SmolVLM).
            text_model = next(
                module
                for module in model.modules()
                if isinstance(module, PreTrainedModel) and module is not model and module.config is text_config
            )
            model.requires_grad_(False)
            text_model.requires_grad_(True)
            model.get_output_embeddings().requires_grad_(True)

        add_fused_lm_head(model, temperature=self.temperature, outputs=("log_probs", "entropy"))

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(
                get_config_model_id(model.config), revision=model_revision, trust_remote_code=args.trust_remote_code
            )
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token
        # The model must agree with the tokenizer on the pad token from construction, so mirror it onto the model
        # configs.
        model.config.get_text_config().pad_token_id = processing_class.pad_token_id
        model.generation_config.pad_token_id = processing_class.pad_token_id

        # Every eos id the model declares (e.g. an end-of-turn token) ends a completion, not only the tokenizer's
        eos_token_ids = model.generation_config.eos_token_id
        if eos_token_ids is None:
            eos_token_ids = []
        elif isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        if processing_class.eos_token_id not in eos_token_ids:
            eos_token_ids = [processing_class.eos_token_id, *eos_token_ids]
        self.eos_token_ids = eos_token_ids

        # PEFT. Placed after `add_fused_lm_head`, which reads the bare `lm_head` and would otherwise have to
        # traverse `base_model.model` to find it.
        if peft_config is not None:
            if not is_peft_available():
                raise ImportError(
                    "You passed `peft_config` but the `peft` library is not installed. "
                    "Install it with `pip install trl[peft]`."
                )
            if not isinstance(peft_config, PeftConfig):
                raise TypeError(
                    f"`peft_config` must be a `peft.PeftConfig` instance (e.g. `peft.LoraConfig`), "
                    f"got {type(peft_config).__name__}."
                )
            if is_peft_model(model):
                raise ValueError(
                    "You passed a `PeftModel` instance together with a `peft_config` to the trainer. Please first merge "
                    "and unload the existing adapter, save the resulting base model, and then pass that base model along "
                    "with the new `peft_config` to the trainer."
                )
            # Unlike `GRPOTrainer`, no `autocast_adapter_dtype=False` branch (it works around a DeepSpeed ZeRO-3
            # dtype mismatch, and AsyncGRPO is FSDP2-only) and no "ref" adapter (there is no reference model).
            model = get_peft_model(model, peft_config)

        # `add_fused_lm_head` computes logits from `lm_head.weight` directly. On a PEFT-wrapped head that is the
        # base layer's weight, so the adapter delta is never applied: the trainer scores a policy that does not exist
        # while the server serves the real one, and `ratio` is wrong on every token with nothing raised. Checked on
        # the module rather than on `target_modules`, so a regex that happens to match the head is caught too.
        # `SFTTrainer` refuses the same configuration.
        if is_peft_model(model):
            from peft.tuners.tuners_utils import BaseTunerLayer

            if isinstance(model.get_output_embeddings(), BaseTunerLayer):
                raise ValueError(
                    "`AsyncGRPOTrainer` does not support a PEFT adapter on `lm_head`: its chunked log-probability "
                    "computation reads the head's base weight and would silently ignore the adapter. Remove `lm_head` "
                    "from `target_modules`."
                )

        # NOTE: See https://github.com/huggingface/transformers/issues/42489
        if is_peft_model(model) and args.gradient_checkpointing:
            model.enable_input_require_grads()

        # Reward functions
        if reward_funcs is None:
            reward_funcs = []
        elif not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]

        if train_dataset is None:
            # A dataset is optional when an environment owns the data and returns the prompt from `reset()`; then
            # `max_steps` sets the length. Build a placeholder dataset of empty prompts to feed the rollout worker,
            # which cycles it indefinitely (so its length only needs to be non-degenerate).
            if environment_factory is None:
                raise ValueError("`train_dataset` is required unless an `environment_factory` is provided.")
            if isinstance(environment_factory, dict):
                raise ValueError(
                    "A `dict` `environment_factory` (multiple environments) requires a `train_dataset` with an "
                    "`environment` column to route each example to its environment. Provide a dataset, or pass a "
                    "single environment factory."
                )
            if args.max_steps <= 0:
                raise ValueError(
                    "When training without a `train_dataset` (the environment owns the data and returns the prompt "
                    "from `reset()`), `max_steps` must be set to a positive value to define the training length. Set "
                    "it via `AsyncGRPOConfig(max_steps=...)`."
                )
            num_placeholder_rows = args.per_device_train_batch_size * args.gradient_accumulation_steps
            train_dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": ""}]] * num_placeholder_rows})

        # Initialize the Trainer
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        precision = self.accelerator.mixed_precision
        dtype = {
            "bf16": "bfloat16",
            "fp16": "float16",
            "no": str(self.model.dtype).removeprefix("torch."),
        }.get(precision, precision)
        self._peak_flops_per_device = get_peak_flops_per_device(self.accelerator, dtype)

        # Epoch handling: stop after num_train_epochs full passes over the PROMPT dataset, counted as distinct
        # prompt-groups trained (fork-independent).
        self._trained_groups: set[int] = set()
        # Tracks restart to match `num_train_epochs`
        self._groups_before_resume = 0
        self._epoch_stop_groups: int | None = None
        samples_per_step = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.accelerator.num_processes
        )
        if self.args.max_steps <= 0 and train_dataset is not None and hasattr(train_dataset, "__len__"):
            # Fork-independent stop: num_train_epochs full passes over the prompts.
            self._epoch_stop_groups = math.ceil(self.args.num_train_epochs * len(train_dataset))
            # max_steps is a generous safety ceiling; with the default constant LR its exact value doesn't matter.
            max_rows_per_conv = (
                self.args.max_tool_calling_iterations + 1 if self.args.max_tool_calling_iterations is not None else 32
            )
            samples_per_epoch = len(train_dataset) * self.args.num_generations * max_rows_per_conv
            self.args.max_steps = math.ceil(self.args.num_train_epochs * samples_per_epoch / samples_per_step)
            logger.info(
                f"Epoch-driven stop: {self._epoch_stop_groups} prompt-groups "
                f"({self.args.num_train_epochs} epochs x {len(train_dataset)} prompts); "
                f"max_steps={self.args.max_steps} is a safety ceiling."
            )

        # Infer max_inflight_tasks when not explicitly set. Generating more samples than the trainer can consume
        # before they become stale is wasteful. The useful upper bound is max_staleness * samples_per_step, floored
        # at samples_per_step so max_staleness=0 (a valid, strict discard policy) can't also zero out the rollout
        # loop's own scheduling capacity and hang the trainer forever on an empty queue.
        if self.args.max_inflight_tasks < 0:
            self.args.max_inflight_tasks = max(self.args.max_staleness, 1) * samples_per_step
            logger.info(
                f"max_inflight_tasks set to {self.args.max_inflight_tasks} "
                f"(max_staleness={self.args.max_staleness} × samples_per_step={samples_per_step})"
            )

        # The base Trainer's data-skip replay loop doesn't apply to AsyncGRPO's live rollout queue;
        # force it off regardless of what the user passed.
        if not self.args.ignore_data_skip:
            logger.warning(
                "`ignore_data_skip` is forced to `True` for AsyncGRPO because the base Trainer's skip-and-replay "
                "loop does not apply to a live rollout queue."
            )
        self.args.ignore_data_skip = True

        # The metric sink. Values are floats, or `(numerator, denominator)` pairs for rates
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._current_train_step_time = 0.0
        self._last_step_end_time = None
        self._rollout_dataset = None
        # Accumulated across one optimizer step's micro-batches, flushed by `_log_step_metrics`.
        self._step_forward_s = 0.0
        self._step_optimizer_s = 0.0
        self._step_microbatches = 0
        self._step_forward_tokens = 0.0
        self._step_trained_tokens = 0.0
        self._step_seq_len_weighted = 0.0
        self._step_samples = 0.0
        self._last_groups_trained = 0
        self.model_version = 0
        # Adapter-only vLLM sync is derived from a PEFT model plus a server started with `--enable-lora`, rather
        # than configured. Rank 0 probes and broadcasts the answer below.
        self._lora_sync = False
        self._lora_name = "trl-policy"
        # Absolute: the path is resolved by the *server's* process, which has its own working directory and may not
        # be on this machine. vLLM reads a path it cannot resolve as a Hub repo id, failing deep inside the engine.
        self._lora_dir = os.path.abspath(os.path.join(self.args.output_dir, ".vllm_lora"))
        # Captured once, so a trainer that later activates another adapter does not ship that one as the policy.
        self._adapter_name = model.active_adapters[0] if is_peft_model(model) else None
        # Weight sync and the token-budget query target the vLLM server from the config; the client is the single
        # place that talks to it, independent of how rollouts are produced.
        self.vllm_client = (
            VLLMClient(self.args.vllm_server_base_url, self.args.vllm_server_timeout)
            if self.accelerator.is_main_process
            else None
        )
        # Rank 0 probes the server for the sync mode, which every rank must agree on: one arm runs a collective
        # adapter save, the other a collective parameter gather, and a split decision hangs both. A failed probe is
        # broadcast in its place and raised on every rank before rank 0 builds anything else, since raising on rank 0
        # alone would leave the others waiting in this collective.
        if weight_transfer is None and is_peft_model(model):
            lora_sync = None
            if self.accelerator.is_main_process:
                try:
                    lora_sync = self._init_lora_sync(model)
                except Exception as error:
                    lora_sync = error
            lora_sync = broadcast_object_list([lora_sync], from_process=0)[0]
            if isinstance(lora_sync, Exception):
                raise lora_sync
            self._lora_sync = lora_sync
        # Create worker and queue on rank 0
        if self.accelerator.is_main_process:
            if weight_transfer is not None:
                # Injected backend (e.g. a no-op stub in tests, or a custom sync mechanism). It owns weight sync
                # entirely, so the server is not probed and adapter sync stays off.
                self.weight_transfer = weight_transfer
            else:
                if self._lora_sync:
                    # The adapter reaches the server as a directory path over HTTP, so there is no NCCL transfer
                    # group to build and no manifest to collect.
                    self.weight_transfer = None
                else:
                    # Collect weight metadata once — names/dtypes/shapes are fixed for the lifetime of training.
                    # DTensor.shape returns the global shape without triggering any all-gather.
                    weight_names, weight_dtype_names, weight_shapes = [], [], []
                    for name, param in _iter_vllm_named_params(model):
                        weight_names.append(name)
                        weight_dtype_names.append(str(param.dtype).split(".")[-1])
                        weight_shapes.append(list(param.shape))
                    self.weight_transfer = WeightTransferClient(
                        vllm_client=self.vllm_client,
                        weight_update_info={
                            "names": weight_names,
                            "dtype_names": weight_dtype_names,
                            "shapes": weight_shapes,
                        },
                        weight_sync_timeout=self.args.weight_sync_timeout,
                    )

            if rollout_worker is not None:
                # Use the injected worker (e.g. a stub in tests). The queue is owned by the worker.
                self.rollout_worker = rollout_worker
            else:
                self.rollout_worker = AsyncRolloutWorker(
                    model_name=get_config_model_id(model.config),
                    lora_name=self._lora_name if self._lora_sync else None,
                    dataset=train_dataset,
                    reward_funcs=reward_funcs,
                    processing_class=processing_class,
                    eos_token_ids=self.eos_token_ids,
                    tools=tools,
                    environment_factory=environment_factory,
                    num_generations=self.args.num_generations,
                    max_inflight_tasks=self.args.max_inflight_tasks,
                    queue_maxsize=self.args.queue_maxsize,
                    vllm_server_url=self.args.vllm_server_base_url,
                    max_tokens=self.args.max_completion_length,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    top_k=self.args.top_k,
                    min_p=self.args.min_p,
                    repetition_penalty=self.args.repetition_penalty,
                    request_timeout=self.args.request_timeout,
                    chat_template_kwargs=self.args.chat_template_kwargs,
                    max_tool_calling_iterations=self.args.max_tool_calling_iterations,
                    log_completions=self.args.log_completions,
                    num_completions_to_print=self.args.num_completions_to_print,
                    fork_threshold_tokens=self.args.fork_threshold_tokens,
                )
            # TODO(@aminediro): decide if this is returned by the worker or common API that is passed to the worker later.
            self.rollout_queue = self.rollout_worker.rollout_buffer
        else:
            self.rollout_queue = None
            self.rollout_worker = None
            self.weight_transfer = None

        # Add callbacks. Cold weight sync + worker start on train begin, then periodic weight syncs.
        self.add_callback(_OptimizerTimeCallback(self))
        self.add_callback(_TrainBeginCallback(self))
        self.add_callback(StepIntervalCallback(self._sync_weight, self.args.weight_sync_steps))
        self.add_callback(StepIntervalCallback(self._log_step_metrics, 1))
        if self._epoch_stop_groups is not None:
            self.add_callback(_EpochStopCallback(self, self._epoch_stop_groups))

    def _init_lora_sync(self, model: "PeftModel") -> bool:
        # [FUNCTION DOC]
        # HELPER: Main process only: waits for the server, fetches server info and returns select_adapter_sync's decision,
        # logging the adapter naming and directory when enabled.
        # Inputs: model. Outputs: bool. Exceptions: TimeoutError, ValueError, HTTP errors. Side effects: network, logging.
        """Probe the server and decide the sync mode. Main process only; the decision itself is [`select_adapter_sync`]."""
        self.vllm_client.wait_for_server_ready()
        lora_sync = select_adapter_sync(self.vllm_client.get_server_info(), model, self.args)
        if lora_sync:
            logger.info(
                f"Adapter-only vLLM sync enabled: syncs publish the '{self._adapter_name}' adapter as "
                f"'{self._lora_name}-v{{N}}' from {self._lora_dir}, which the vLLM server must be able to read."
            )
        return lora_sync

    def get_train_dataloader(self) -> DataLoader:
        # [FUNCTION DOC]
        # Builds the training dataloader. Main process: RolloutQueueDataset (stored as _rollout_dataset), resolves token_budget
        # from the server's max_model_len when None, wraps in TokenBudgetBatcher if the budget is positive else
        # FixedCountBatcher with per_device_train_batch_size x num_processes samples. Other ranks: _EmptyIterableDataset.
        # Returns accelerator.prepare of a DataLoader with batch_size 1, num_workers 0 and a DataCollatorForRollout sharing the
        # trainer's groups set and metric sink (token_budget clamped to at least 0).
        # Inputs: none. Outputs: DataLoader. Exceptions: server errors. Side effects: may mutate args.token_budget.
        num_processes = self.accelerator.num_processes
        if self.accelerator.is_main_process:
            dataset = RolloutQueueDataset(
                rollout_queue=self.rollout_queue,
                model_version_fn=lambda: self.model_version,
                check_health_fn=self.rollout_worker.check_health,
                stale_after_s=self.args.heartbeat_stale_after_s,
                metrics=self._metrics["train"],
                max_staleness=self.args.max_staleness,
                report_to=self.args.report_to,
            )
            # Kept so `_log_step_metrics` can flush the queue wait at the optimizer-step boundary, which is the only
            # place that knows a step's worth of waiting is over.
            self._rollout_dataset = dataset
            # Default the token budget to the vLLM server's max_model_len (the cap on prompt + completion), so no
            # rollout sample can exceed it. Wait for the server like weight sync does, so a still-loading vLLM doesn't
            # fail training here.
            if self.args.token_budget is None:
                self.vllm_client.wait_for_server_ready()
                self.args.token_budget = self.vllm_client.get_max_model_len()
                logger.info(f"token_budget unset; defaulting to vLLM max_model_len={self.args.token_budget}")
            # TokenBudgetBatcher caps each row at `token_budget` tokens (dynamic count, bounds peak memory);
            # FixedCountBatcher uses a fixed `per_device_train_batch_size × num_processes` samples per micro-batch.
            if self.args.token_budget > 0:
                dataset = TokenBudgetBatcher(dataset, num_processes, self.args.token_budget, self._metrics["train"])
            else:
                dataset = FixedCountBatcher(
                    dataset, num_processes, self.args.per_device_train_batch_size * num_processes
                )
        else:
            dataset = _EmptyIterableDataset()

        # Each planner item is one complete micro-batch (`num_processes` pre-packed rows), so the dataloader pulls them
        # one at a time (batch_size=1) and the collator tensorizes each into a rectangular `(num_processes, T_max)`
        # batch that DataLoaderDispatcher scatters row `i` -> rank `i`.
        return self.accelerator.prepare(
            DataLoader(
                dataset,
                batch_size=1,
                collate_fn=DataCollatorForRollout(
                    self.processing_class.pad_token_id,
                    num_processes,
                    groups_trained=self._trained_groups,
                    metrics=self._metrics["train"],
                    # `or 0` because only rank 0 fills an unset budget from the vLLM server above; the other ranks
                    # construct the collator (and never use it) while `token_budget` is still `None`.
                    token_budget=max(self.args.token_budget or 0, 0),
                ),
                num_workers=0,
                # NOTE(@aminediro):
                # dispatch_batches = True for DataLoader whose underlying dataset is an IterableDataset
                # dataloader prepared by the Accelerator is only iterated through on the main process a
            )
        )

    def _set_signature_columns_if_needed(self):
        # [FUNCTION DOC]
        # HELPER: Sets _signature_columns (if unset) to the packed batch keys produced by the collator so the base Trainer
        # does not drop them. Inputs/Outputs: none. Side effects: sets attribute.
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids"
        if self._signature_columns is None:
            self._signature_columns = [
                "input_ids",
                "attention_mask",
                "position_ids",
                "shift_labels",
                "token_index",
                "old_log_probs",
                "advantages",
                "segment_id",
                "global_n_tokens",
                "global_n_forward_tokens",
                "mean_seq_len",
                "tree_kv_index",
                "tree_cu_seqlens_q",
                "tree_cu_seqlens_k",
                "tree_max_seqlens",
            ]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # [FUNCTION DOC]
        # Per micro-batch on each rank (one packed row).
        # Behavior: strips inter-rank padding using attention_mask (sequence keys) and non-negative filters (token_index and
        # tree index keys), unsqueezing to batch 1; takes max seqlens from tree_max_seqlens. Forwards with fused_lm_head,
        # shift_labels, tree kwargs and output_router_logits when aux loss is enabled; gathers log_probs and entropy at
        # token_index (timed). Loss: ratio = exp(logp - old logp); per-term loss = negative min(ratio x A, clamp(ratio,
        # 1 - epsilon_low, 1 + epsilon_high) x A); summed, divided by max(global_n_tokens / world_size, 1) and by
        # current_gradient_accumulation_steps; adds router_aux_loss_coef x aux_loss / GA steps when enabled.
        # Metrics (no grad, all-reduced sums over terms): ratio, kl (k3 estimator), entropy, clip_ratio low_mean/high_mean/
        # region_mean (low: ratio below 1-eps with negative advantage; high: above 1+eps with positive advantage); per-sample
        # low clip fraction nan-min and high nan-max gathered across ranks (clip_ratio/low_min, high_max); aux_loss mean across
        # ranks. Accumulates step counters (forward tokens, trained tokens, length-weighted mean length, samples, forward time).
        # Inputs: model, inputs (collator batch slice), return_outputs and num_items_in_batch (ignored).
        # Outputs: scalar loss tensor (outputs never returned).
        # Exceptions: collective/forward errors. Side effects: metric sink appends, collectives, step accumulators.
        mask_bool = inputs["attention_mask"].bool()
        selected = inputs["token_index"] >= 0
        input_ids = inputs["input_ids"][mask_bool].unsqueeze(0)
        position_ids = inputs["position_ids"][mask_bool].unsqueeze(0)
        shift_labels = inputs["shift_labels"][mask_bool].unsqueeze(0)
        token_index = inputs["token_index"][selected]
        old_log_probs = inputs["old_log_probs"][selected]
        advantages = inputs["advantages"][selected]
        segment_id = inputs["segment_id"][selected]
        tree_kwargs = {
            key: inputs[key][inputs[key] >= 0] for key in ("tree_kv_index", "tree_cu_seqlens_q", "tree_cu_seqlens_k")
        }
        tree_kwargs["tree_max_seqlen_q"], tree_kwargs["tree_max_seqlen_k"] = inputs["tree_max_seqlens"][0].tolist()

        forward_start = time.time()
        # MoE models: request router logits so the forward returns the load-balancing loss
        router_kwargs = {"output_router_logits": True} if self.aux_loss_enabled else {}
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            shift_labels=shift_labels,
            fused_lm_head=True,
            **router_kwargs,
            **tree_kwargs,
        )
        log_probs = outputs.log_probs.flatten()[token_index]
        entropy = outputs.entropy.flatten()[token_index]
        self._last_forward_time_s = time.time() - forward_start

        log_ratio = log_probs - old_log_probs
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # DDP/FSDP averages gradients across ranks (world_size).
        # To get correct per-token normalization we scale by 1/tokens_per_rank
        # = world_size / global_n_tokens, so after DDP averaging the effective
        loss = per_token_loss.sum()
        global_n_tokens = inputs["global_n_tokens"][0]
        world_size = self.accelerator.num_processes
        tokens_per_rank = (global_n_tokens / world_size).clamp(min=1.0)
        loss = loss / tokens_per_rank.to(torch.float32)
        # For DAPO, we would scale like this instead:
        # loss = loss / max(per_token_loss.size(0), 1)
        loss = loss / self.current_gradient_accumulation_steps

        # The policy loss above is scaled for gradient accumulation (HF auto-scaling is off here), so scale aux too
        if self.aux_loss_enabled:
            aux_loss = outputs.aux_loss
            loss = loss + self.router_aux_loss_coef * aux_loss / self.current_gradient_accumulation_steps

        with torch.no_grad():
            local_ratio_sum = coef_1.sum()
            local_count = torch.full_like(local_ratio_sum, float(log_probs.numel()))
            # Approx KL: http://joschu.net/blog/kl-approx.html
            local_kl_sum = ((coef_1 - 1) - log_ratio).sum()
            local_entropy_sum = entropy.sum()

            # Compute the clipped probability ratios
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped
            local_low_clip_sum = is_low_clipped.float().sum()
            local_high_clip_sum = is_high_clipped.float().sum()
            local_region_clip_sum = is_region_clipped.float().sum()

            # Batch all-reduce: [ratio_sum, kl_sum, entropy_sum, low_clip_sum, high_clip_sum, region_clip_sum, count]
            stats = torch.stack(
                [
                    local_ratio_sum,
                    local_kl_sum,
                    local_entropy_sum,
                    local_low_clip_sum,
                    local_high_clip_sum,
                    local_region_clip_sum,
                    local_count,
                ]
            )
            stats = self.accelerator.reduce(stats, reduction="sum")
            (
                global_ratio_sum,
                global_kl_sum,
                global_entropy_sum,
                global_low_clip_sum,
                global_high_clip_sum,
                global_region_clip_sum,
                global_count,
            ) = stats.unbind(0)
            self._metrics["train"]["ratio"].append((global_ratio_sum / global_count).item())
            self._metrics["train"]["kl"].append((global_kl_sum / global_count).item())
            self._metrics["train"]["entropy"].append((global_entropy_sum / global_count).item())
            self._metrics["train"]["clip_ratio/low_mean"].append((global_low_clip_sum / global_count).item())
            self._metrics["train"]["clip_ratio/high_mean"].append((global_high_clip_sum / global_count).item())
            self._metrics["train"]["clip_ratio/region_mean"].append((global_region_clip_sum / global_count).item())

            num_seq = int(segment_id.max()) + 1  # completions in this rank's row

            def seg_sum(vals):  # per-completion segment sum over the flat selection
                # [FUNCTION DOC]
                # HELPER: Sums vals per sample index (segment_id) into a zero vector of length num_seq via index-add.
                # Inputs: vals tensor. Outputs: tensor of length num_seq. Side effects: none.
                return torch.zeros(num_seq, device=vals.device).index_add_(0, segment_id, vals)

            seq_tokens = seg_sum(torch.ones_like(coef_1, dtype=torch.float32))
            seq_low = seg_sum(is_low_clipped.float())
            seq_high = seg_sum(is_high_clipped.float())
            per_seq_low = seq_low / seq_tokens  # NaN for a completion with no valid tokens; ignored by nan-aware min
            per_seq_high = seq_high / seq_tokens
            gathered_low_min = self.accelerator.gather(nanmin(per_seq_low))
            gathered_high_max = self.accelerator.gather(nanmax(per_seq_high))
            self._metrics["train"]["clip_ratio/low_min"].append(nanmin(gathered_low_min).item())
            self._metrics["train"]["clip_ratio/high_max"].append(nanmax(gathered_high_max).item())

            if self.aux_loss_enabled:
                gathered_aux = self.accelerator.reduce(aux_loss.detach().to(torch.float32), reduction="sum")
                self._metrics["train"]["aux_loss"].append((gathered_aux / world_size).item())

        # Per-step accounting, accumulated across the micro-batches of one optimizer step and flushed in
        # `training_step`. The counts are batch-wide (the collator broadcasts one value per rank), so they are read off
        # rank-local inputs without a collective. Sample rewards and packing metrics are NOT gathered here — rank 0
        # already logged them in the collator.
        n_forward_tokens = float(inputs["global_n_forward_tokens"][0])
        mean_seq_len = float(inputs["mean_seq_len"][0])
        self._step_forward_tokens += n_forward_tokens
        self._step_trained_tokens += float(global_n_tokens)
        self._step_seq_len_weighted += mean_seq_len * n_forward_tokens
        self._step_samples += n_forward_tokens / mean_seq_len
        self._step_forward_s += self._last_forward_time_s
        return loss

    def training_step(self, model, inputs, num_items_in_batch):
        # [FUNCTION DOC]
        # Wraps the base training_step, counting micro-batches and accumulating wall time into _current_train_step_time.
        # Inputs: model, inputs, num_items_in_batch. Outputs: base loss output. Side effects: step accumulators.
        time_before = time.perf_counter()
        output = super().training_step(model, inputs, num_items_in_batch)
        self._step_microbatches += 1
        self._current_train_step_time += time.perf_counter() - time_before
        return output

    def _log_step_metrics(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Called after each optimizer step. Appends perf/fwd_bwd_s, fwd_s, optimizer_s, step_s (time since previous
        # call; skipped on first), batch/microbatches, forwarded/trained tokens and samples per step. Main process also flushes
        # perf/rollout_wait_s (resetting the dataset's wait_s) and batch/groups_per_step. When tokens were forwarded, computes
        # FLOPs per token at the mean sequence length and logs forwarded_tok_s and MFU on fwd/bwd time and on wall-clock step
        # time (plus trained_tok_s wall clock); MFU only when peak FLOPs are known. Resets all step accumulators.
        # Inputs/Outputs: none. Exceptions: ZeroDivisionError if fwd_bwd_s is 0 with tokens. Side effects: metric sink.
        """Flush one optimizer step's worth of accounting: the time budget, what the batch held, and throughput.

        Called from `on_step_end`, i.e. after `optimizer.step()`, so `perf/optimizer_s` covers this step.
        """
        time_after = time.perf_counter()
        metrics = self._metrics["train"]
        fwd_bwd_s = self._current_train_step_time
        # The first step has no predecessor, so there is nothing to measure between yet.
        step_s = time_after - self._last_step_end_time if self._last_step_end_time is not None else None

        metrics["perf/fwd_bwd_s"].append(fwd_bwd_s)
        metrics["perf/fwd_s"].append(self._step_forward_s)
        metrics["perf/optimizer_s"].append(self._step_optimizer_s)
        metrics["batch/microbatches_per_step"].append(float(self._step_microbatches))
        metrics["batch/forwarded_tokens_per_step"].append(self._step_forward_tokens)
        metrics["batch/trained_tokens_per_step"].append(self._step_trained_tokens)
        metrics["batch/samples_per_step"].append(self._step_samples)
        if step_s is not None:
            metrics["perf/step_s"].append(step_s)

        if self.accelerator.is_main_process:
            metrics["perf/rollout_wait_s"].append(self._rollout_dataset.wait_s)
            self._rollout_dataset.wait_s = 0.0
            metrics["batch/groups_per_step"].append(float(len(self._trained_groups) - self._last_groups_trained))
            self._last_groups_trained = len(self._trained_groups)

        # Throughput and MFU are reported on TWO bases, because for async RL one number cannot answer both questions.
        ## `_fwd_bwd` divides by `perf/fwd_bwd_s`: the compute alone.
        ## `_wall_clock` divides by `perf/step_s`: the whole step, rollout waits included  and says what fraction of the allocation actually became training.
        if self._step_forward_tokens > 0:
            mean_seq_len = self._step_seq_len_weighted / self._step_forward_tokens
            flops_per_token = compute_flops_per_token(self.model.config.get_text_config(), int(mean_seq_len))
            world_size = self.accelerator.num_processes
            metrics["perf/forwarded_tok_s_fwd_bwd"].append((self._step_forward_tokens, fwd_bwd_s))
            if self._peak_flops_per_device is not None:
                metrics["perf/mfu_fwd_bwd"].append(
                    compute_mfu(
                        flops_per_token, self._step_forward_tokens / fwd_bwd_s, world_size, self._peak_flops_per_device
                    )
                )
            if step_s is not None:
                metrics["perf/forwarded_tok_s_wall_clock"].append((self._step_forward_tokens, step_s))
                metrics["perf/trained_tok_s_wall_clock"].append((self._step_trained_tokens, step_s))
                if self._peak_flops_per_device is not None:
                    metrics["perf/mfu_wall_clock"].append(
                        compute_mfu(
                            flops_per_token,
                            self._step_forward_tokens / step_s,
                            world_size,
                            self._peak_flops_per_device,
                        )
                    )

        self._last_step_end_time = time_after
        self._current_train_step_time = 0.0
        self._step_forward_s = 0.0
        self._step_optimizer_s = 0.0
        self._step_microbatches = 0
        self._step_forward_tokens = 0.0
        self._step_trained_tokens = 0.0
        self._step_seq_len_weighted = 0.0
        self._step_samples = 0.0

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        # [FUNCTION DOC]
        # Main process drains the worker's metrics_queue into the train sink (even in eval mode). Reduces each key of the
        # current mode's sink with _reduce_metric after dropping NaN floats (None if nothing left), prefixes "eval_" in eval
        # mode, merges into logs, calls the base log and clears that mode's sink.
        # Inputs: logs (dict), start_time. Outputs: none. Side effects: logging backends, clears sink.
        mode = "train" if self.model.training else "eval"
        if self.accelerator.is_main_process and self.rollout_worker:
            while True:
                try:
                    for key, value in self.rollout_worker.metrics_queue.get_nowait().items():
                        # NOTE(@aminediro): we might be filling train metrics dict even in eval mode
                        self._metrics["train"][key].append(value)
                except queue.Empty:
                    break

        metrics = {}
        for key, val in self._metrics[mode].items():
            valid = [v for v in val if isinstance(v, tuple) or not math.isnan(v)]
            metrics[key] = _reduce_metric(key, valid) if valid else None

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs.update(metrics)
        super().log(logs, start_time)
        self._metrics[mode].clear()

    def _streaming_iter(self):
        # [FUNCTION DOC]
        # HELPER: Generator over _iter_vllm_named_params of the unwrapped model yielding (name, full tensor): DTensors are
        # all-gathered via full_tensor (a collective, so every rank must iterate), others detached; moved to the accelerator
        # device when needed.
        # Inputs: none. Outputs: iterator. Side effects: collectives, temporary GPU memory.
        # Iterate parameters one at a time. For FSDP2 (DTensor), full_tensor() all-gathers just this parameter across
        # FSDP ranks, then frees it once the generator advances — avoiding materializing the full model in memory.
        device = self.accelerator.device
        for name, param in _iter_vllm_named_params(self.accelerator.unwrap_model(self.model)):
            full = param.full_tensor() if isinstance(param, DTensor) else param.detach()
            if full.device != device:
                full = full.to(device)
            yield name, full

    def _sync_weight(self):
        # [FUNCTION DOC]
        # Runs _sync_weight_lora or _sync_weight_merged and logs perf/weight_sync_s and its pause, barrier and transfer phases.
        # Inputs/Outputs: none. Exceptions: from the sync path. Side effects: metrics, logging, server state, model_version.
        """Publish the current policy to vLLM, then bump the version the rollout worker requests.

        Dispatches to the sync mode picked at init, logging `perf/weight_sync_*` for both here so they stay comparable.
        """
        t0 = time.time()
        if self._lora_sync:
            t_pause, t_barrier, t_transfer = self._sync_weight_lora(t0)
        else:
            t_pause, t_barrier, t_transfer = self._sync_weight_merged(t0)
        weight_sync_s = time.time() - t0
        # log the three phases  of weight sync
        self._metrics["train"]["perf/weight_sync_s"].append(weight_sync_s)
        self._metrics["train"]["perf/weight_sync_pause_s"].append(t_pause - t0)
        self._metrics["train"]["perf/weight_sync_barrier_s"].append(t_barrier - t_pause)
        self._metrics["train"]["perf/weight_sync_transfer_s"].append(t_transfer - t_barrier)
        logger.info(f"Weight sync: done. Total {weight_sync_s:.1f}s")

    def _sync_weight_merged(self, t0: float) -> tuple[float, float, float]:
        # [FUNCTION DOC]
        # Main pauses the server via weight_transfer (if any); barrier; merges the PEFT adapter if present; main streams
        # weights through weight_transfer.send_weights while other ranks drain _streaming_iter for the collectives (main
        # without a backend also drains); unmerges in finally; barrier; main resumes, increments model_version and notifies the
        # worker.
        # Inputs: t0. Outputs: (t_pause, t_barrier, t_transfer) timestamps. Exceptions: transfer errors.
        # Side effects: network/NCCL, temporary adapter merge, model_version.
        """Fold the adapter, if any, into the base weights and stream every parameter to vLLM over NCCL.

        Args:
            t0 (`float`):
                When the sync started, for the phase log lines.

        Returns:
            `tuple[float, float, float]`:
                The pause, barrier and transfer phase marks, which [`_sync_weight`] turns into metrics.
        """
        # Under DDP `self.model` is the wrapper, which exposes none of the PEFT API used below.
        model = self.accelerator.unwrap_model(self.model)

        logger.info("Weight sync: pausing vLLM...")
        if self.accelerator.is_main_process and self.weight_transfer:
            self.weight_transfer.pause()
        t_pause = time.time()
        logger.info(f"Weight sync: pause took {t_pause - t0:.1f}s, waiting for all ranks...")

        self.accelerator.wait_for_everyone()
        t_barrier = time.time()

        logger.info(f"Weight sync: transferring weights... (barrier took {t_barrier - t_pause:.1f}s)")
        # vLLM only knows the base checkpoint's parameters, so the adapter is folded into them for the send. The
        # `finally` is not optional: leaving it merged would train merged weights from the next step on.
        if is_peft_model(model):
            model.merge_adapter()
        try:
            if self.accelerator.is_main_process and self.weight_transfer:
                self.weight_transfer.send_weights(self._streaming_iter())
            else:
                # Non-rank-0 processes must still participate in full_tensor() collectives for FSDP2.
                for _ in self._streaming_iter():
                    pass
        finally:
            if is_peft_model(model):
                model.unmerge_adapter()
        t_transfer = time.time()

        self.accelerator.wait_for_everyone()

        logger.info(f"Weight sync: resuming vLLM... (transfer took {t_transfer - t_barrier:.1f}s)")
        if self.accelerator.is_main_process:
            if self.weight_transfer:
                self.weight_transfer.resume()
            self.model_version += 1
            if self.rollout_worker:
                self.rollout_worker.update_model_version(self.model_version)
        return t_pause, t_barrier, t_transfer

    def _sync_weight_lora(self, t0: float) -> tuple[float, float, float]:
        # [FUNCTION DOC]
        # Publishes version N+1 as adapter "<lora_name>-v<N+1>" under _lora_dir. Main pauses the server; barrier; all ranks
        # call save_lora_adapter; barrier; main loads the adapter on the server, resumes, unloads version N+1-(max_staleness+1)
        # when positive, deletes the directory of the version before that when above 1, sets model_version and notifies the
        # worker.
        # Inputs: t0. Outputs: (t_pause, t_barrier, t_transfer). Exceptions: HTTP/filesystem errors.
        # Side effects: files, server adapter registry, model_version.
        """Publish the trained adapter to vLLM under a fresh versioned name, moving no base weight.

        Each sync publishes `{lora_name}-v{N}` instead of overwriting the previous adapter in place, because
        `load_inplace=True` makes the server re-read the adapter from disk on every request
        (https://github.com/vllm-project/vllm/pull/41482) and vLLM keys its prefix cache on the adapter name alone
        (https://github.com/vllm-project/vllm/issues/42125), so an in-place swap would serve the previous policy's KV
        blocks. A new name is a fresh cache namespace, so no `reset_prefix_cache()` is needed.

        Args:
            t0 (`float`):
                When the sync started, for the phase log lines.

        Returns:
            `tuple[float, float, float]`:
                The pause, barrier and transfer phase marks, which [`_sync_weight`] turns into metrics.
        """
        version = self.model_version + 1
        adapter_dir = os.path.join(self._lora_dir, f"{self._lora_name}-v{version}")
        # Under DDP `self.model` is the wrapper, which does not expose `save_pretrained`.
        model = self.accelerator.unwrap_model(self.model)

        logger.info("Weight sync: pausing vLLM...")
        if self.accelerator.is_main_process:
            self.vllm_client.pause()
        t_pause = time.time()
        logger.info(f"Weight sync: pause took {t_pause - t0:.1f}s, waiting for all ranks...")

        self.accelerator.wait_for_everyone()
        t_barrier = time.time()

        logger.info(f"Weight sync: transferring adapter... (barrier took {t_barrier - t_pause:.1f}s)")
        # Every rank calls this, not just rank 0: materializing a sharded adapter parameter is a collective, even
        # though only the main process writes the files.
        save_lora_adapter(model, self.accelerator, self._adapter_name, adapter_dir)
        t_transfer = time.time()

        self.accelerator.wait_for_everyone()

        logger.info(f"Weight sync: resuming vLLM... (transfer took {t_transfer - t_barrier:.1f}s)")
        if self.accelerator.is_main_process:
            # The adapter has to exist on the server BEFORE the version moves below: the rollout worker derives the
            # adapter it requests from `model_version`, so bumping first would name one that is not loaded yet.
            self.vllm_client.load_lora_adapter(f"{self._lora_name}-v{version}", adapter_dir)
            self.vllm_client.resume()
            stale_version = version - (self.args.max_staleness + 1)
            if stale_version > 0:
                self.vllm_client.unload_lora_adapter(f"{self._lora_name}-v{stale_version}")
            # The files outlive the unload by one sync: vLLM resolves `lora_path` lazily inside the engine, where
            # a missing directory is not a 404 but a fatal `HFValidationError`.
            if stale_version > 1:
                stale_dir = f"{self._lora_name}-v{stale_version - 1}"
                shutil.rmtree(os.path.join(self._lora_dir, stale_dir), ignore_errors=True)
            self.model_version = version
            if self.rollout_worker:
                self.rollout_worker.update_model_version(self.model_version)
        return t_pause, t_barrier, t_transfer

    def _save_checkpoint(self, model, trial):
        # [FUNCTION DOC]
        # When main and the worker is the default AsyncRolloutWorker, writes rollout_state.json into the checkpoint folder
        # with prompt_index (dataset_start_index plus the smallest group id not yet trained) and model_version, then calls the
        # base _save_checkpoint.
        # Inputs: model, trial. Outputs: none. Exceptions: OSError. Side effects: files written.
        if self.accelerator.is_main_process and isinstance(self.rollout_worker, AsyncRolloutWorker):
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            checkpoint_dir = os.path.join(self._get_output_dir(trial=trial), checkpoint_folder)
            os.makedirs(checkpoint_dir, exist_ok=True)
            trained = self._trained_groups
            first_untrained = next(g for g in itertools.count() if g not in trained)
            prompt_index = self.rollout_worker._loop_kwargs["dataset_start_index"] + first_untrained
            # `model_version` rides along so adapter names keep counting across a resume: restarting at v1 would
            # republish a different adapter under a name a still-running server already holds.
            rollout_state = {"prompt_index": prompt_index, "model_version": self.model_version}
            with open(os.path.join(checkpoint_dir, "rollout_state.json"), "w") as f:
                json.dump(rollout_state, f)
        super()._save_checkpoint(model, trial)

    def _inner_training_loop(self, *args, **kwargs):
        # [FUNCTION DOC]
        # With the default worker: resets dataset_start_index and _groups_before_resume to 0; when resume_from_checkpoint is
        # in kwargs, restores prompt_index (as start index and groups-before-resume) and model_version from rollout_state.json
        # if present and the dataset is a map-style Dataset (warning otherwise). Runs the base loop; finally on main stops the
        # worker and destroys the weight transfer.
        # KNOWN ISSUE: the leading comment refers to _StartRolloutWorkerCallback, which does not exist; the worker is started
        # by _TrainBeginCallback.
        # Inputs: base loop args/kwargs. Outputs: base result. Exceptions: propagates. Side effects: worker lifecycle.
        # When resuming, pass the saved prompt position to the worker before _StartRolloutWorkerCallback fires.
        # Skipped for IterableDataset since len() isn't available on streaming datasets.
        # Always reset first so a stale value from a prior train() call is never carried over.
        if isinstance(self.rollout_worker, AsyncRolloutWorker):
            self.rollout_worker._loop_kwargs["dataset_start_index"] = 0
            self._groups_before_resume = 0
            resume_from_checkpoint = kwargs.get("resume_from_checkpoint")
            if resume_from_checkpoint is not None:
                rollout_state_file = os.path.join(resume_from_checkpoint, "rollout_state.json")
                # IterableDataset is skipped deliberately: streaming datasets have no len() and can't be repositioned.
                if not os.path.isfile(rollout_state_file):
                    logger.warning(
                        "rollout_state.json not found in the checkpoint; "
                        "the rollout worker will restart from prompt 0."
                    )
                elif not isinstance(self.train_dataset, Dataset):
                    logger.warning("Resuming with an IterableDataset; the rollout worker will restart from prompt 0.")
                else:
                    with open(rollout_state_file) as f:
                        rollout_state = json.load(f)
                    self.rollout_worker._loop_kwargs["dataset_start_index"] = rollout_state["prompt_index"]
                    self._groups_before_resume = rollout_state["prompt_index"]
                    # Older checkpoints predate this field; resuming from one restarts numbering at v1, which is
                    # only safe against a freshly started server.
                    self.model_version = rollout_state.get("model_version", 0)
                    self.rollout_worker.update_model_version(self.model_version)
        try:
            return super()._inner_training_loop(*args, **kwargs)
        finally:
            if self.accelerator.is_main_process:
                if self.rollout_worker:
                    self.rollout_worker.stop()
                if self.weight_transfer:
                    self.weight_transfer.destroy()
