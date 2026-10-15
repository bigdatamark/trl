# [MODULE DOC]
# Module trl.extras.profiling: lightweight wall-clock profiling for trainer code. Timings are logged as
# metrics named "<metric_prefix>: <name>" to Weights & Biases, MLflow and/or Trackio.
# Main contents: ProfilingContext (timing context manager), profiling_context (factory from a Trainer) and
# profiling_decorator (method decorator used throughout TRL trainers and VLLMGeneration).
# Import-time side effects: imports mlflow, trackio and wandb when each is installed.

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

import functools
import time
from collections.abc import Callable

from transformers import Trainer
from transformers.integrations import is_mlflow_available, is_trackio_available, is_wandb_available


if is_mlflow_available():
    import mlflow


if is_trackio_available():
    import trackio


if is_wandb_available():
    import wandb


class ProfilingContext:
    # [CLASS DOC]
    # Context manager that measures elapsed time of a block with time.perf_counter and logs it on exit.
    # State: name, report_to (list of integration names), is_main_process, step (optional), metric_prefix
    # (default "profiling/Time taken"), _start_time. Independent of Trainer.
    """
    Context manager for profiling code blocks with configurable logging.

    This class handles timing of code execution and logging metrics to various backends (Weights & Biases, MLflow,
    Trackio) without being coupled to the Trainer class.

    Args:
        name (`str`):
            Name of the profiling context. Used in the metric name.
        report_to (`list` of `str`):
            List of integrations to report metrics to (e.g., ["wandb", "mlflow", "trackio"]).
        is_main_process (`bool`, *optional*, defaults to `True`):
            Whether this is the main process in distributed training. Metrics are only logged from the main process.
        step (`int`, *optional*):
            Training step to associate with the logged metrics.
        metric_prefix (`str`, *optional*, defaults to `"profiling/Time taken"`):
            Prefix for the metric name in logs.

    Example:
    ```python
    # Direct usage
    from trl.extras.profiling import ProfilingContext

    with ProfilingContext(
        name="MyClass.expensive_operation",
        report_to=["wandb"],
        is_main_process=True,
        step=100,
    ):
        # Code to profile
        result = expensive_computation()

    # With Trainer (backwards compatible via profiling_context function)
    from transformers import Trainer
    from trl.extras.profiling import profiling_context


    class MyTrainer(Trainer):
        def some_method(self):
            with profiling_context(self, "matrix_multiplication"):
                result = matrix_multiply()
    ```
    """

    def __init__(
        self,
        name: str,
        report_to: list[str],
        is_main_process: bool = True,
        step: int | None = None,
        metric_prefix: str = "profiling/Time taken",
    ):
        # [FUNCTION DOC]
        # Behavior: store the arguments and set _start_time to None.
        # Inputs: name, report_to, is_main_process (default True), step (optional), metric_prefix.
        # Outputs: none. Exceptions: none. Side effects: none.
        self.name = name
        self.report_to = report_to
        self.is_main_process = is_main_process
        self.step = step
        self.metric_prefix = metric_prefix
        self._start_time = None

    def __enter__(self):
        # [FUNCTION DOC]
        # Behavior: record the start time with perf_counter and return self.
        # Inputs: none. Outputs: self. Exceptions: none. Side effects: sets _start_time.
        """Start timing when entering the context."""
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # [FUNCTION DOC]
        # Behavior: if timing started, compute the elapsed seconds and log them via _log_metrics; always return
        # False so exceptions propagate (the duration is still logged when the block raised).
        # Inputs: exc_type, exc_val, exc_tb. Outputs: False. Exceptions: errors from logging backends.
        # Side effects: metric logging.
        """Stop timing and log metrics when exiting the context."""
        if self._start_time is not None:
            duration = time.perf_counter() - self._start_time
            self._log_metrics(duration)
        return False

    def _log_metrics(self, duration: float) -> None:
        # [FUNCTION DOC]
        # HELPER: Send one duration metric to the configured backends, only on the main process.
        # Behavior: wandb.log when "wandb" in report_to, wandb is installed and a run is active (no step passed);
        # mlflow.log_metrics with step when "mlflow" is configured and an active run exists; trackio.log with step
        # when "trackio" is configured and installed.
        # Inputs: duration (float seconds). Outputs: none. Exceptions: backend errors.
        # Side effects: network/file writes by the logging backends.
        """
        Log profiling metrics to configured backends.

        Args:
            duration (`float`):
                Execution time in seconds.
        """
        if not self.is_main_process:
            return

        metric_name = f"{self.metric_prefix}: {self.name}"
        metrics = {metric_name: duration}

        # Log to Weights & Biases if configured
        if "wandb" in self.report_to and is_wandb_available() and wandb.run is not None:
            wandb.log(metrics)

        # Log to MLflow if configured
        if "mlflow" in self.report_to and is_mlflow_available() and mlflow.active_run() is not None:
            mlflow.log_metrics(metrics, step=self.step)

        # Log to Trackio if configured
        if "trackio" in self.report_to and is_trackio_available():
            trackio.log(metrics, step=self.step)


def profiling_context(trainer: Trainer, name: str) -> ProfilingContext:
    # [FUNCTION DOC]
    # Build a ProfilingContext for a Trainer: name is "<TrainerClass>.<name>", report_to from trainer.args,
    # is_main_process from trainer.accelerator, step from trainer.state.global_step.
    # Inputs: trainer (Trainer-like with state, args, accelerator), name (str). Outputs: ProfilingContext.
    # Exceptions: AttributeError when the trainer lacks those attributes. Side effects: none.
    """
    Factory function to create a ProfilingContext from a Trainer instance.

    This function maintains backwards compatibility with existing code while using the decoupled ProfilingContext class
    internally.

    Args:
        trainer (`~transformers.Trainer`):
            Trainer object containing configuration for logging.
        name (`str`):
            Name of the block to be profiled. Will be prefixed with the trainer class name.

    Returns:
        `ProfilingContext`: A configured profiling context manager.

    Example:
    ```python
    >>> from transformers import Trainer
    >>> from trl.extras.profiling import profiling_context


    >>> class MyTrainer(Trainer):
    ...     def some_method(self):
    ...         A = np.random.rand(1000, 1000)
    ...         B = np.random.rand(1000, 1000)
    ...         with profiling_context(self, "matrix_multiplication"):
    ...             # Code to profile: simulate a computationally expensive operation
    ...             result = A @ B  # Matrix multiplication
    ```
    """
    context_name = f"{trainer.__class__.__name__}.{name}"
    step = trainer.state.global_step

    return ProfilingContext(
        name=context_name,
        report_to=trainer.args.report_to,
        is_main_process=trainer.accelerator.is_main_process,
        step=step,
    )


def profiling_decorator(func: Callable) -> Callable:
    # [FUNCTION DOC]
    # Decorator timing a method call on self.
    # Behavior: returns a functools.wraps wrapper. If self has state and args, wrap the call in
    # profiling_context(self, function name). Else if self has accelerator, wrap in a ProfilingContext with
    # report_to empty (so nothing is actually logged). Otherwise just call the function.
    # KNOWN ISSUE: the docstring says non-Trainer objects with an accelerator use it "for logging
    # configuration", but report_to is empty for them, so their timings are measured and never reported.
    # Inputs: func (callable taking self). Outputs: wrapped callable. Exceptions: none.
    # Side effects: none at decoration time.
    """
    Decorator to profile a function and log execution time using [`extras.profiling.profiling_context`].

    This decorator works with methods that have access to a trainer instance (typically as `self`). For non-Trainer
    objects that have an `accelerator` attribute, it will use that for logging configuration.

    Args:
        func (`Callable`):
            Function to be profiled.

    Returns:
        `Callable`: Wrapped function that profiles execution time.

    Example:
    ```python
    >>> from transformers import Trainer
    >>> from trl.extras.profiling import profiling_decorator


    >>> class MyTrainer(Trainer):
    ...     @profiling_decorator
    ...     def some_method(self):
    ...         A = np.random.rand(1000, 1000)
    ...         B = np.random.rand(1000, 1000)
    ...         # Code to profile: simulate a computationally expensive operation
    ...         result = A @ B
    ```
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: The wrapper described in profiling_decorator; forwards args/kwargs and returns func's result.
        # Inputs: self, args, kwargs. Outputs: func's return value. Exceptions: propagates func's exceptions.
        # Side effects: profiling metric logging for Trainer-like objects.
        # Check if self is a Trainer-like object with required attributes
        if hasattr(self, "state") and hasattr(self, "args"):
            with profiling_context(self, func.__name__):
                return func(self, *args, **kwargs)
        # For non-Trainer objects (e.g., VLLMGeneration), use ProfilingContext directly
        elif hasattr(self, "accelerator"):
            context_name = f"{self.__class__.__name__}.{func.__name__}"
            with ProfilingContext(
                name=context_name,
                report_to=[],  # No reporting for non-Trainer objects without args
                is_main_process=self.accelerator.is_main_process,
                step=None,
            ):
                return func(self, *args, **kwargs)
        else:
            # No profiling available, just run the function
            return func(self, *args, **kwargs)

    return wrapper
