# [MODULE DOC]
# TrainerCallback implementations shipped with TRL: SyncRefModelCallback (periodic soft/hard sync of a
# reference model, TR-DPO style), RichProgressCallback (Rich live progress and metric tables),
# LogCompletionsCallback (periodic generation logged to W&B / Comet), WeaveCallback (generation traces and
# optional scores to W&B Weave at evaluation), BEMACallback (bias-corrected EMA of weights saved at the end),
# plus the _generate_completions helper.
# Import-time side effects: conditionally imports rich, wandb and weave when available; creates a module
# logger.

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

import logging

import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import gather_object, is_wandb_available
from transformers import (
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import has_length
from transformers.utils import is_rich_available

from ..data_utils import maybe_apply_chat_template
from ..import_utils import is_weave_available
from ..models.utils import unwrap_model_for_generation
from .utils import log_table_to_comet_experiment


if is_rich_available():
    from rich.columns import Columns
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress
    from rich.table import Table


if is_wandb_available():
    import wandb


if is_weave_available():
    import weave
    from weave import EvaluationLogger
    from weave.trace.context import weave_client_context


# Logger for module-level logging
logger = logging.getLogger(__name__)


def _generate_completions(
    prompts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    accelerator: Accelerator,
    generation_config: GenerationConfig | None,
    batch_size: int = 1,
    gradient_checkpointing_kwargs: dict | None = None,
) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Generates text completions for already-formatted prompts.
    # Behavior: inside unwrap_model_for_generation (handles DeepSpeed/FSDP gathering and gradient checkpointing),
    # processes prompts in chunks of batch_size: tokenizes with padding and truncation to model.device, calls
    # generate with generation_config, strips the prompt-length prefix from each output (relies on left padding so
    # all prompts in the batch share the padded length) and decodes skipping special tokens.
    # Inputs: prompts (list of str), model, tokenizer, accelerator, generation_config (or None for the model
    # default), batch_size (default 1), gradient_checkpointing_kwargs (optional).
    # Outputs: list of str, one per prompt, in order.
    # Exceptions: generation/tokenizer errors propagate.
    # Side effects: temporary model unwrapping (may gather sharded params).
    """
    Generates completions for a list of pre-formatted prompts from the given model.

    Args:
        prompts (`list[str]`):
            A list of input prompts for which completions are to be generated.
        model ([`~transformers.PreTrainedModel`]):
            The pre-trained model to be used for generation.
        tokenizer ([`~transformers.PreTrainedTokenizerBase`]):
            The tokenizer to be used for encoding and decoding.
        accelerator ([`~accelerate.Accelerator`]):
            The accelerator to be used for model execution.
        generation_config ([`~transformers.GenerationConfig`]):
            Configuration for text generation.
        batch_size (`int`, *optional*, defaults to `1`):
            The number of prompts to process in each batch.
        gradient_checkpointing_kwargs (`dict`, *optional*):
            Additional kwargs for gradient checkpointing enabling.

    Returns:
        `list[str]`:
            A list of generated text completions corresponding to the input prompts.
    """
    completions = []
    # TODO: Override model.generation_config with generation_kwargs
    with unwrap_model_for_generation(
        model, accelerator, gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
    ) as unwrapped_model:
        for idx in range(0, len(prompts), batch_size):
            batch = prompts[idx : idx + batch_size]
            tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
            generations = unwrapped_model.generate(
                **tokenized_batch,
                generation_config=generation_config,
            )
            for prompt, generation in zip(tokenized_batch.input_ids, generations, strict=True):
                # Remove prompt from generation
                generation = generation[len(prompt) :]
                completion = tokenizer.decode(generation, skip_special_tokens=True)
                completions.append(completion)
    return completions


class SyncRefModelCallback(TrainerCallback):
    # [CLASS DOC]
    # Callback that blends the policy into a reference model every args.ref_model_sync_steps steps using
    # args.ref_model_mixup_alpha (ref = (1 - alpha) * ref + alpha * policy, parameter-wise). State: ref_model,
    # accelerator (used to unwrap the policy). Supports DeepSpeed ZeRO-3 by gathering parameters first.
    """
    Callback to synchronize the model with a reference model.
    """

    def __init__(
        self,
        ref_model: PreTrainedModel | torch.nn.Module,
        accelerator: Accelerator | None,
    ):
        # [FUNCTION DOC]
        # Stores ref_model and accelerator (may be None). Inputs: ref_model, accelerator. Outputs: none.
        # Exceptions: none. Side effects: sets attributes.
        self.accelerator = accelerator
        self.ref_model = ref_model

    @staticmethod
    def _sync_target_model(model, target_model, alpha):
        # [FUNCTION DOC]
        # HELPER: In place, for each pair of parameters (zipped strictly in order), scales the target by 1 - alpha
        # and adds alpha times the source.
        # Inputs: model (source), target_model, alpha (float). Outputs: none.
        # Exceptions: ValueError if parameter counts differ. Side effects: mutates target_model parameters.
        for target_param, copy_param in zip(target_model.parameters(), model.parameters(), strict=True):
            target_param.data.mul_(1.0 - alpha).add_(copy_param.data, alpha=alpha)

    @staticmethod
    def sync_target_model(model, target_model, alpha):
        # [FUNCTION DOC]
        # Behavior: if the DeepSpeed plugin uses ZeRO stage 3, gathers both models' parameters (modifier rank 0) and
        # performs the blend on rank 0 only (changes are re-partitioned on exit); otherwise blends directly.
        # Inputs: model, target_model, alpha. Outputs: none.
        # Exceptions: as _sync_target_model. Side effects: mutates target parameters; collective communication under
        # ZeRO-3.
        deepspeed_plugin = AcceleratorState().deepspeed_plugin
        if deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3:
            import deepspeed

            with deepspeed.zero.GatheredParameters(
                list(model.parameters()) + list(target_model.parameters()), modifier_rank=0
            ):
                if deepspeed.comm.get_rank() == 0:
                    SyncRefModelCallback._sync_target_model(model, target_model, alpha)
        else:
            SyncRefModelCallback._sync_target_model(model, target_model, alpha)

    def on_step_end(self, args, state, control, **kwargs):
        # [FUNCTION DOC]
        # Behavior: when ref_model exists and global_step is a multiple of args.ref_model_sync_steps, unwraps the
        # model (if accelerator given) and syncs it into ref_model with args.ref_model_mixup_alpha.
        # Inputs: args (must have ref_model_sync_steps, ref_model_mixup_alpha), state, control, kwargs["model"].
        # Outputs: none. Exceptions: KeyError if model missing. Side effects: mutates ref_model.
        model: PreTrainedModel = kwargs["model"]

        if self.ref_model is not None and state.global_step % args.ref_model_sync_steps == 0:
            if self.accelerator:
                model = self.accelerator.unwrap_model(model)
            self.sync_target_model(model, self.ref_model, args.ref_model_mixup_alpha)


class RichProgressCallback(TrainerCallback):
    # [CLASS DOC]
    # Callback rendering a Rich live panel with a training progress bar, an evaluation progress bar and a status
    # area showing the latest logged metrics grouped by key prefix. All work happens on world process zero.
    # State: training_bar, evaluation_bar (Progress), training_task, evaluation_task, rich_group (Live),
    # rich_console, training_status, current_step; all None outside training.
    """
    A [`~transformers.TrainerCallback`] that displays the progress of training or evaluation using Rich.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Behavior: requires rich; initializes all state attributes to None.
        # Inputs: none. Outputs: none. Exceptions: ImportError if rich missing. Side effects: sets attributes.
        if not is_rich_available():
            raise ImportError("RichProgressCallback requires the `rich` extra. To install, run `pip install rich`.")

        self.training_bar = None
        self.evaluation_bar = None
        self.training_task = None
        self.evaluation_task = None
        self.rich_group = None
        self.rich_console = None
        self.training_status = None
        self.current_step = None

    def on_train_begin(self, args, state, control, **kwargs):
        # [FUNCTION DOC]
        # Behavior (main process only): creates the two Progress bars, a Console and a status line ("Nothing to log
        # yet ..."), wraps them in a Live panel and starts it, adds a training task with total max_steps and sets
        # current_step to 0.
        # Inputs: callback args. Outputs: none. Exceptions: none expected. Side effects: starts terminal rendering.
        if not state.is_world_process_zero:
            return

        self.training_bar = Progress()
        self.evaluation_bar = Progress()
        self.rich_console = Console()
        self.training_status = self.rich_console.status("Nothing to log yet ...")
        self.rich_group = Live(Panel(Group(self.training_bar, self.evaluation_bar, self.training_status)))
        self.rich_group.start()
        self.training_task = self.training_bar.add_task("[blue]Training  ", total=state.max_steps)
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        # [FUNCTION DOC]
        # Behavior (main process only): advances the training bar by the steps since current_step and records
        # global_step as current_step.
        # Inputs: callback args. Outputs: none. Exceptions: none. Side effects: updates display/state.
        if not state.is_world_process_zero:
            return

        self.training_bar.update(self.training_task, advance=state.global_step - self.current_step, update=True)
        self.current_step = state.global_step

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        # [FUNCTION DOC]
        # Behavior (main process only): if the eval dataloader has a length, lazily creates the evaluation task
        # (total = number of batches) and advances it by one.
        # Inputs: callback args, eval_dataloader. Outputs: none. Exceptions: none. Side effects: updates display.
        if not state.is_world_process_zero:
            return

        if has_length(eval_dataloader):
            if self.evaluation_task is None:
                self.evaluation_task = self.evaluation_bar.add_task("[blue]Evaluation", total=len(eval_dataloader))
            self.evaluation_bar.update(self.evaluation_task, advance=1, update=True)

    def on_evaluate(self, args, state, control, **kwargs):
        # [FUNCTION DOC]
        # Behavior (main process only): removes the evaluation task if any and resets it to None.
        # Inputs: callback args. Outputs: none. Exceptions: none. Side effects: updates display/state.
        if not state.is_world_process_zero:
            return

        if self.evaluation_task is not None:
            self.evaluation_bar.remove_task(self.evaluation_task)
            self.evaluation_task = None

    def on_predict(self, args, state, control, **kwargs):
        # [FUNCTION DOC]
        # Behavior (main process only): same as on_evaluate, removing the evaluation task after prediction.
        # Inputs: callback args. Outputs: none. Exceptions: none. Side effects: updates display/state.
        if not state.is_world_process_zero:
            return

        if self.evaluation_task is not None:
            self.evaluation_bar.remove_task(self.evaluation_task)
            self.evaluation_task = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        # [FUNCTION DOC]
        # Behavior (main process with an active training bar only): groups log keys by the part before the first "/"
        # (keys without "/" go in an untitled group, keyed by full name), builds one Rich table per group with
        # Metric/Value columns (numbers formatted to 3 decimals, others as str), lays the panels out in columns and
        # updates the status area titled with the current step.
        # Inputs: callback args, logs dict. Outputs: none.
        # Exceptions: AttributeError if logs is None. Side effects: updates terminal display.
        if not (state.is_world_process_zero and self.training_bar):
            return

        # Group keys by top-level prefix
        grouped_logs = {}
        for key, value in logs.items():
            parts = key.split("/")
            group = parts[0] if len(parts) > 1 else None
            subkey = "/".join(parts[1:]) if len(parts) > 1 else key
            grouped_logs.setdefault(group, {})[subkey] = value

        # Create a table per group
        tables = []
        for group_name, metrics in grouped_logs.items():
            table = Table(
                title=f"[bold blue]{group_name}[/]" if group_name else None, header_style="bold magenta", box=None
            )
            table.add_column("Metric", justify="left", no_wrap=True)
            table.add_column("Value", justify="right")

            for metric, val in metrics.items():
                formatted = f"{val:.3f}" if isinstance(val, (float, int)) else str(val)
                table.add_row(metric, formatted)

            tables.append(Panel(table, border_style="cyan", padding=(0, 1)))

        # Arrange tables in columns using Columns
        column_layout = Columns(tables, equal=False, expand=True)
        self.training_status.update(
            Panel(column_layout, title=f"[bold green]Step {state.global_step}[/bold green]", border_style="green")
        )

    def on_train_end(self, args, state, control, **kwargs):
        # [FUNCTION DOC]
        # Behavior (main process only): stops the Live display and resets all state attributes to None.
        # Inputs: callback args. Outputs: none. Exceptions: none. Side effects: stops rendering.
        if not state.is_world_process_zero:
            return

        self.rich_group.stop()
        self.training_bar = None
        self.evaluation_bar = None
        self.training_task = None
        self.evaluation_task = None
        self.rich_group = None
        self.rich_console = None
        self.training_status = None
        self.current_step = None


class LogCompletionsCallback(TrainerCallback):
    # [CLASS DOC]
    # Callback that, every freq steps (default state.eval_steps), generates completions for the eval dataset's
    # "prompt" column across processes and logs an accumulating table (step, prompt, completion) to W&B
    # ("completions") and/or Comet ("completions.csv"). State: trainer, generation_config, freq, eval_dataset
    # (optionally truncated to num_prompts), table (accumulated rows), _last_logged_step.
    r"""
    A [`~transformers.TrainerCallback`] that logs completions to Weights & Biases and/or Comet.

    Usage:
    ```python
    >>> trainer = DPOTrainer(...)
    >>> completions_callback = LogCompletionsCallback(trainer=trainer)
    >>> trainer.add_callback(completions_callback)
    ```

    Args:
        trainer (`Trainer`):
            Trainer to which the callback will be attached. The trainer's evaluation dataset must include a `"prompt"`
            column containing the prompts for generating completions.
        generation_config ([`~transformers.GenerationConfig`], *optional*):
            The generation config to use for generating completions.
        num_prompts (`int`, *optional*):
            The number of prompts to generate completions for. If not provided, defaults to the number of examples in
            the evaluation dataset.
        freq (`int`, *optional*):
            The frequency at which to log completions. If not provided, defaults to the trainer's `eval_steps`.
    """

    def __init__(
        self,
        trainer: Trainer,
        generation_config: GenerationConfig | None = None,
        num_prompts: int | None = None,
        freq: int | None = None,
    ):
        # [FUNCTION DOC]
        # Behavior: stores args, initializes an empty table and _last_logged_step -1, takes the trainer eval dataset
        # and selects the first num_prompts rows when given.
        # Inputs: trainer, generation_config, num_prompts, freq (all but trainer optional).
        # Outputs: none.
        # Exceptions: ValueError when the trainer has no eval dataset.
        # Side effects: sets attributes.
        self.trainer = trainer
        self.generation_config = generation_config
        self.freq = freq
        self.table = []
        self._last_logged_step = -1

        if self.trainer.eval_dataset is None:
            raise ValueError("Trainer must have an evaluation dataset to use the LogCompletionsCallback.")
        else:
            self.eval_dataset = self.trainer.eval_dataset

        if num_prompts is not None:
            self.eval_dataset = self.eval_dataset.select(range(num_prompts))

    def on_step_end(self, args, state, control, **kwargs):
        # [FUNCTION DOC]
        # Behavior: skips if already logged at this step or global_step is not a multiple of freq (or eval_steps).
        # Sets the processing class padding side to left, splits prompts across processes, applies the chat template
        # if conversational, generates with _generate_completions (per-device eval batch size, gradient checkpointing
        # kwargs), gathers prompts and completions. On the main process extends the table with rows for this step and
        # logs the full table as a DataFrame to wandb and/or comet depending on args.report_to. Records the step.
        # Inputs: args, state, control, kwargs["processing_class"].
        # Outputs: none.
        # Exceptions: KeyError if no processing_class; ZeroDivision/TypeError if both freq and eval_steps unset.
        # Side effects: mutates tokenizer.padding_side permanently; generation; external logging; mutates table.
        # Only log once per step (this method may be called multiple times)
        if state.global_step == self._last_logged_step:
            return

        # Only log every `freq` steps (if no `freq` is provided, log every `eval_steps` steps)
        freq = self.freq or state.eval_steps
        if state.global_step % freq != 0:
            return

        tokenizer = kwargs["processing_class"]
        tokenizer.padding_side = "left"
        accelerator = self.trainer.accelerator
        model = self.trainer.model_wrapped
        with accelerator.split_between_processes(self.eval_dataset["prompt"]) as prompts:
            prompts = [maybe_apply_chat_template({"prompt": prompt}, tokenizer)["prompt"] for prompt in prompts]
            completions = _generate_completions(
                prompts,
                model=model,
                tokenizer=tokenizer,
                accelerator=accelerator,
                generation_config=self.generation_config,
                batch_size=args.per_device_eval_batch_size,
                gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs,
            )
            completions = gather_object(completions)
            prompts = gather_object(prompts)

        # Build the data to log
        if self.trainer.accelerator.is_main_process:
            global_step = [str(state.global_step)] * len(prompts)
            data = list(zip(global_step, prompts, completions, strict=True))
            self.table.extend(data)
            table = pd.DataFrame(columns=["step", "prompt", "completion"], data=self.table)

            if "wandb" in args.report_to:
                wandb.log({"completions": table})

            if "comet_ml" in args.report_to:
                log_table_to_comet_experiment(
                    name="completions.csv",
                    table=table,
                )

        # Save the last logged step, so we don't log the same completions multiple times
        self._last_logged_step = state.global_step


class WeaveCallback(TrainerCallback):
    # [CLASS DOC]
    # Callback that, at each evaluation, generates completions for eval prompts and logs them via Weave
    # EvaluationLogger. Tracing mode when scorers is empty; evaluation mode when scorers (name to function of
    # prompt and completion) are given, which also logs per-prediction scores and a summary with averages.
    # State: trainer, project_name, scorers, generation_config, dataset_name, model_name, eval_dataset,
    # _last_logged_step, _weave_initialized, _eval_logger (unused), and after init _weave_client and
    # _EvaluationLogger. Weave initialization is lazy (_initialize_weave).
    # KNOWN ISSUE: the class docstring usage examples and an error message refer to "WeaveTraceCallback", which
    # does not exist; the class is WeaveCallback.
    r"""
    A [`~transformers.TrainerCallback`] that logs traces and evaluations to W&B Weave. The callback uses
    https://weave-docs.wandb.ai/guides/evaluation/evaluation_logger/ to log traces and evaluations at each evaluation
    step.

    Supports two modes based on the `scorers` parameter:
    - **Tracing Mode** (when scorers=None): Logs predictions for data exploration and analysis
    - **Evaluation Mode** (when scorers provided): Logs predictions with scoring and summary metrics

    Both modes use Weave's EvaluationLogger for structured, consistent data logging.

    The callback logs data during evaluation phases (`on_evaluate`) rather than training steps, making it more
    efficient and semantically correct. It gracefully handles missing weave installation by logging warnings and
    skipping weave-specific functionality. It also checks for existing weave clients before initializing new ones.

    Usage:
    ```python
    # Tracing mode (just log predictions)
    trainer = DPOTrainer(...)
    weave_callback = WeaveTraceCallback(trainer=trainer)  # project_name optional
    trainer.add_callback(weave_callback)

    # Or specify a project name
    weave_callback = WeaveTraceCallback(trainer=trainer, project_name="my-llm-training")
    trainer.add_callback(weave_callback)


    # Evaluation mode (log predictions + scores + summary)
    def accuracy_scorer(prompt: str, completion: str) -> float:
        # Your scoring logic here (metadata available via eval_attributes)
        return score


    weave_callback = WeaveTraceCallback(
        trainer=trainer,
        project_name="my-llm-training",  # optional and needed only if weave client is not initialized
        scorers={"accuracy": accuracy_scorer},
    )
    trainer.add_callback(weave_callback)
    ```

    Args:
        trainer (`Trainer`):
            Trainer to which the callback will be attached. The trainer's evaluation dataset must include a `"prompt"`
            column containing the prompts for generating completions.
        project_name (`str`, *optional*):
            Name of the Weave project where data will be logged. If not provided, will try to use existing weave client
            or fall back to the active wandb run's project name. Raises an error if none of these are available.
        scorers (`dict[str, Callable]`, *optional*):
            Dictionary mapping scorer names to scorer functions. If `None`, operates in tracing mode (predictions
            only). If provided, operates in evaluation mode (predictions + scores + summary). Scorer functions should
            have signature: `scorer(prompt: str, completion: str) -> float | int`
        generation_config ([`~transformers.GenerationConfig`], *optional*):
            Generation config to use for generating completions.
        num_prompts (`int`, *optional*):
            The number of prompts to generate completions for. If not provided, defaults to the number of examples in
            the evaluation dataset.
        dataset_name (`str`, *optional*, defaults to `"eval_dataset"`):
            Name for the dataset metadata in Weave.
        model_name (`str`, *optional*):
            Name for the model metadata in Weave. If not provided, attempts to extract from model config.
    """

    def __init__(
        self,
        trainer: Trainer,
        project_name: str | None = None,
        scorers: dict[str, callable] | None = None,
        generation_config: GenerationConfig | None = None,
        num_prompts: int | None = None,
        dataset_name: str = "eval_dataset",
        model_name: str | None = None,
    ):
        # [FUNCTION DOC]
        # Behavior: stores arguments (scorers defaults to an empty dict), initializes state, takes the eval dataset
        # and optionally truncates it to num_prompts.
        # Inputs: trainer, project_name, scorers, generation_config, num_prompts, dataset_name ("eval_dataset"),
        # model_name. Outputs: none.
        # Exceptions: ValueError when the trainer has no eval dataset. Side effects: sets attributes.
        self.trainer = trainer
        self.project_name = project_name
        self.scorers = scorers or {}
        self.generation_config = generation_config
        self.dataset_name = dataset_name
        self.model_name = model_name
        self._last_logged_step = -1
        self._weave_initialized = False
        self._eval_logger = None

        if self.trainer.eval_dataset is None:
            raise ValueError("Trainer must have an evaluation dataset to use the WeaveCallback.")
        else:
            self.eval_dataset = self.trainer.eval_dataset

        if num_prompts is not None:
            self.eval_dataset = self.eval_dataset.select(range(num_prompts))

    def _initialize_weave(self):
        # [FUNCTION DOC]
        # HELPER: One-time Weave setup.
        # Behavior: no-op if already initialized. If weave is not installed, logs a warning and returns (stays
        # uninitialized). Reuses the current Weave client if one exists; otherwise derives project_name from the
        # active wandb run ("entity/project") when not given, and calls weave.init. Sets model_name from the wrapped
        # model config's _name_or_path (default "unknown_model") if unset, stores the EvaluationLogger class and
        # marks initialized.
        # Inputs: none. Outputs: none.
        # Exceptions: ValueError when no client, no project_name and no wandb run.
        # Side effects: may initialize a Weave client (network); logging; sets attributes.
        """Initialize Weave and EvaluationLogger if not already initialized."""
        if not self._weave_initialized:
            if not is_weave_available():
                logger.warning("Weave is not available. Please install weave to enable logging: `pip install weave`")
                return

            if wc := weave_client_context.get_weave_client():
                self._weave_client = wc
            else:
                if self.project_name is None:
                    if is_wandb_available():
                        if wandb.run is not None:
                            self.project_name = wandb.run.entity + "/" + wandb.run.project
                            logger.info(f"Using project name from active wandb run: {self.project_name}")

                    if self.project_name is None:
                        raise ValueError(
                            "No existing Weave client found and no project_name provided. "
                            "Please either initialize weave with `weave.init('project-name')`, "
                            "provide a project_name to the `WeaveTraceCallback`, "
                            "or ensure an active wandb run exists."
                        )

                self._weave_client = weave.init(self.project_name)
                logger.info(f"Initialized Weave with project: {self.project_name}")

            if self.model_name is None:
                self.model_name = getattr(self.trainer.model_wrapped.config, "_name_or_path", "unknown_model")

            self._EvaluationLogger = EvaluationLogger

            self._weave_initialized = True

    @property
    def is_evaluation_mode(self) -> bool:
        # [FUNCTION DOC]
        # Property: True when any scorers were provided (evaluation mode), else False (tracing mode).
        # Inputs: none. Outputs: bool. Exceptions: none. Side effects: none.
        """True if scorers are provided (evaluation mode), False for tracing mode."""
        return bool(self.scorers)

    def on_train_begin(self, args, state, control, **kwargs):
        # [FUNCTION DOC]
        # Behavior: initializes Weave at the start of training. Inputs: callback args. Outputs: none.
        # Exceptions: as _initialize_weave. Side effects: as _initialize_weave.
        """Initialize Weave when training begins."""
        self._initialize_weave()

    def on_evaluate(self, args, state, control, **kwargs):
        # [FUNCTION DOC]
        # Behavior: skips if already logged at this step; initializes Weave and skips (debug log) if unavailable.
        # Sets padding side left, splits prompts across processes, applies the chat template, generates and gathers
        # prompts/completions. On the main process creates an EvaluationLogger with model, dataset and attributes
        # (training_step, model_name, generation config dict). Logs each prediction; in evaluation mode applies each
        # scorer, logs the score and collects values (scorer failures only warn). Prediction failures warn and are
        # skipped. In evaluation mode with any scores, logs a summary (total_predictions, successful_predictions and
        # avg_<scorer>); otherwise finishes the logger. Records the step.
        # Inputs: args, state, control, kwargs["processing_class"]. Outputs: none.
        # Exceptions: logging errors are caught and warned; generation errors propagate.
        # Side effects: mutates tokenizer.padding_side; generation; network logging to Weave.
        if state.global_step == self._last_logged_step:
            return

        self._initialize_weave()

        if not self._weave_initialized:
            logger.debug("Weave not initialized, skipping logging")
            return

        tokenizer = kwargs["processing_class"]
        tokenizer.padding_side = "left"
        accelerator = self.trainer.accelerator
        model = self.trainer.model_wrapped

        with accelerator.split_between_processes(self.eval_dataset["prompt"]) as prompts:
            prompts = [maybe_apply_chat_template({"prompt": prompt}, tokenizer)["prompt"] for prompt in prompts]

            completions = _generate_completions(
                prompts=prompts,
                model=model,
                tokenizer=tokenizer,
                accelerator=accelerator,
                generation_config=self.generation_config,
                batch_size=args.per_device_eval_batch_size,
                gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs,
            )

            all_prompts = gather_object(prompts)
            all_completions = gather_object(completions)

        if self.trainer.accelerator.is_main_process:
            eval_attributes = {
                "training_step": state.global_step,
                "model_name": self.model_name,
                "generation_config": (self.generation_config.to_dict() if self.generation_config else None),
            }

            eval_logger = self._EvaluationLogger(
                model=self.model_name,
                dataset=self.dataset_name,
                eval_attributes=eval_attributes,
            )

            successful_predictions = 0
            total_score_values = {}  # For summary statistics

            for prompt, completion in zip(all_prompts, all_completions, strict=True):
                try:
                    pred_logger = eval_logger.log_prediction(inputs={"prompt": prompt}, output=completion)

                    if self.is_evaluation_mode:
                        for scorer_name, scorer_func in self.scorers.items():
                            try:
                                score = scorer_func(prompt, completion)
                                pred_logger.log_score(scorer=scorer_name, score=score)

                                if scorer_name not in total_score_values:
                                    total_score_values[scorer_name] = []
                                total_score_values[scorer_name].append(score)

                            except Exception as scorer_e:
                                logger.warning(f"Failed to apply scorer '{scorer_name}': {scorer_e}")

                    pred_logger.finish()
                    successful_predictions += 1

                except Exception as pred_e:
                    logger.warning(f"Failed to log prediction for prompt: {pred_e}")
                    # Continue with other predictions even if one fails

            if self.is_evaluation_mode and total_score_values:
                try:
                    summary_stats = {
                        "total_predictions": len(all_prompts),
                        "successful_predictions": successful_predictions,
                    }

                    for scorer_name, scores in total_score_values.items():
                        if scores:  # Only if we have valid scores
                            summary_stats[f"avg_{scorer_name}"] = sum(scores) / len(scores)

                    eval_logger.log_summary(summary_stats)

                except Exception as summary_e:
                    logger.warning(f"Failed to log summary: {summary_e}")
            else:
                try:
                    eval_logger.finish()
                except Exception as finish_e:
                    logger.warning(f"Failed to finish evaluation logger: {finish_e}")

        self._last_logged_step = state.global_step


class BEMACallback(TrainerCallback):
    # [CLASS DOC]
    # Callback implementing BEMA (bias-corrected EMA of weights). State: hyperparameters (update_freq 400,
    # ema_power 0.5 as kappa, bias_power 0.2 as eta, lag 10 as rho, update_after 0 as tau, multiplier 1.0 as
    # gamma, min_ema_multiplier 0.0, device "cpu"); param_names and thetat_params (live trainable parameters),
    # theta0_params and ema_params (buffers on device), running_model (a separate model copy on device holding
    # BEMA weights). Flow: on_train_begin builds buffers; on_step_end snapshots at update_after and updates every
    # update_freq steps after it; on_train_end saves running_model to output_dir/bema.
    # docstyle-ignore
    r"""
    A [`~transformers.TrainerCallback`] that implements [BEMA](https://huggingface.co/papers/2508.00180)
    (Bias-Corrected Exponential Moving Average) by [Adam Block](https://huggingface.co/abblock) and [Cyril
    Zhang](https://huggingface.co/cyrilzhang). Code from https://github.com/abblock/bema under MIT license.

    BEMA computes model weights that scale like:

    $$
    \theta_t' = \alpha_t \cdot (\theta_t - \theta_0) + \text{EMA}_t
    $$

    where  \\( \theta_t \\) is the current model weights,  \\( \theta_0 \\) is a snapshot of the model weights at the
    first `update_after` step,  \\( \text{EMA}_t  \\) is the exponential moving average of the model weights, and
     \\( \alpha_t \\) is a scaling factor that decays with the number of steps  \\( t \\) as

    $$
    \alpha_t = (\rho + \gamma \cdot t)^{-\eta}.
    $$

    The EMA is computed as:

    $$
    \text{EMA}_t = (1 - \beta_t) \cdot \text{EMA}_{t-1} + \beta_t \cdot \theta_t
    $$

    where  \\( \beta_t \\) is a decay factor that decays with the number of steps  \\( t \\) as

    $$
    \beta_t = (\rho + \gamma \cdot t)^{-\kappa}.
    $$

    Args:
        update_freq (`int`, *optional*, defaults to `400`):
            Update the BEMA weights every X steps. Denoted this as  \\( \phi \\) in the paper.
        ema_power (`float`, *optional*, defaults to `0.5`):
            Power for the EMA decay factor. Denoted  \\( \kappa \\) in the paper. To disable EMA, set this to `0.0`.
        bias_power (`float`, *optional*, defaults to `0.2`):
            Power for the BEMA scaling factor. Denoted  \\( \eta \\) in the paper. A large value (e.g.
            `8.0`) makes  \\( \alpha_t \\) decay to `0`, approximating disabled bias-correction; `0.0`
            instead pins  \\( \alpha_t \\) at `1` for every step (maximum, undecayed correction).
        lag (`int`, *optional*, defaults to `10`):
            Initial offset in the weight decay schedule that controls early-stage smoothness by acting as a virtual
            starting age for the updates. Denoted as  \\( \rho \\) in the paper.
        update_after (`int`, *optional*, defaults to `0`):
            Burn-in time before starting to update the BEMA weights. Denoted  \\( \tau \\) in the paper.
        multiplier (`float`, *optional*, defaults to `1.0`):
            Initial value for the EMA decay factor. Denoted as  \\( \gamma \\) in the paper.
        min_ema_multiplier (`float`, *optional*, defaults to `0.0`):
            Minimum value for the EMA decay factor.
        device (`str`, *optional*, defaults to `"cpu"`):
            Device to use for the BEMA buffers, e.g. `"cpu"` or `"cuda"`. Note that in most cases, this device SHOULD
            BE DIFFERENT from the device used for training in order to avoid OOM.

    Example:

    ```python
    >>> from trl import BEMACallback

    >>> trainer = Trainer(..., callbacks=[BEMACallback()])
    ```
    """

    def __init__(
        self,
        update_freq: int = 400,
        ema_power: float = 0.5,
        bias_power: float = 0.2,
        lag: int = 10,
        update_after: int = 0,
        multiplier: float = 1.0,
        min_ema_multiplier: float = 0.0,
        device: str = "cpu",
    ):
        # [FUNCTION DOC]
        # Behavior: stores hyperparameters and initializes empty buffer lists and running_model None.
        # Inputs: the hyperparameters listed in the class block. Outputs: none. Exceptions: none.
        # Side effects: sets attributes.
        # User-provided hyperparams
        self.update_freq = update_freq
        self.ema_power = ema_power
        self.bias_power = bias_power
        self.lag = lag
        self.update_after = update_after
        self.multiplier = multiplier
        self.min_ema_multiplier = min_ema_multiplier
        self.device = device

        # Internal state
        self.param_names = []  # references to training model param names
        self.thetat_params = []  # references to training model params
        self.theta0_params = []  # θ₀ buffers (on self.device)
        self.ema_params = []  # EMA buffers (on self.device)
        self.running_model = None  # a copy of the model to run BEMA on

    @staticmethod
    def _unwrap_model(model):
        # [FUNCTION DOC]
        # HELPER: Strips wrappers: returns model.module for DeepSpeed engines (has module and engine), the
        # _fsdp_wrapped_module for FSDP, model.module for (D)DP, else the model itself.
        # Inputs: model. Outputs: inner module. Exceptions: none. Side effects: none.
        """
        Helper function to unwrap model from various wrappers including DataParallel, DistributedDataParallel,
        DeepSpeed, and FSDP.
        """
        # Handle DeepSpeed
        if hasattr(model, "module") and hasattr(model, "engine"):
            # DeepSpeed engine
            return model.module

        # Handle FSDP
        if hasattr(model, "_fsdp_wrapped_module"):
            # FSDP wrapped model
            return model._fsdp_wrapped_module

        # Handle DataParallel/DistributedDataParallel
        if hasattr(model, "module"):
            return model.module

        return model

    @torch.no_grad()
    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: PreTrainedModel, **kwargs
    ):
        # [FUNCTION DOC]
        # Behavior (no grad): unwraps the model, constructs a fresh instance of the same class from its config on the
        # BEMA device and loads the model state dict into it. For each trainable parameter (requires_grad) records
        # its name and reference, and clones it to the device as theta0 and as the initial EMA.
        # Inputs: args, state, control, model. Outputs: none.
        # Exceptions: errors if the model class cannot be built from its config (e.g. PEFT wrappers).
        # Side effects: allocates a full model copy and buffers on device.
        model = self._unwrap_model(model)

        # Create a new instance and load state_dict
        self.running_model = type(model)(model.config).to(self.device)
        self.running_model.load_state_dict(model.state_dict())

        # Cache trainable parameters once in a fixed order
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.param_names.append(name)
            self.thetat_params.append(param)

            # Clone θ₀ and EMA on the same device as model
            theta0 = param.detach().clone().to(self.device)
            self.theta0_params.append(theta0)
            self.ema_params.append(theta0.clone())  # initialize EMA with θ₀

    def _ema_beta(self, step: int) -> float:
        # [FUNCTION DOC]
        # HELPER: Returns beta = (lag + multiplier times step) to the power of minus ema_power, floored at
        # min_ema_multiplier. Inputs: step (int). Outputs: float. Exceptions: none. Side effects: none.
        """Compute the EMA decay factor βₜ = (ρ + γ·t)⁻ᵏᵃᵖᵖᵃ."""
        beta = (self.lag + self.multiplier * step) ** (-self.ema_power)
        return max(beta, self.min_ema_multiplier)

    def _bema_alpha(self, step: int) -> float:
        # [FUNCTION DOC]
        # HELPER: Returns alpha = (lag + multiplier times step) to the power of minus bias_power.
        # Inputs: step (int). Outputs: float. Exceptions: none. Side effects: none.
        """Compute the BEMA scaling factor αₜ = (ρ + γ·t)⁻ᵉᵗᵃ."""
        return (self.lag + self.multiplier * step) ** (-self.bias_power)

    def _update_bema_weights(self, step: int):
        # [FUNCTION DOC]
        # HELPER: For each tracked parameter: moves the live value to device, updates EMA in place as
        # (1 - beta) times EMA plus beta times theta_t, and writes EMA plus alpha times (theta_t minus theta0) into the
        # matching running_model parameter.
        # Inputs: step (int). Outputs: none.
        # Exceptions: ValueError from the strict zip when parameter counts differ.
        # Side effects: mutates EMA buffers and running_model parameters.
        # KNOWN ISSUE: running_model.parameters() includes all parameters while thetat_params only holds trainable
        # ones, so with any frozen parameter the strict zip raises (or, if counts matched by chance, would pair the
        # wrong tensors).
        beta = self._ema_beta(step)
        alpha = self._bema_alpha(step)

        # Compute EMA + BEMA in-place and write directly to running_model
        for thetat, theta0, ema, run_param in zip(
            self.thetat_params, self.theta0_params, self.ema_params, self.running_model.parameters(), strict=True
        ):
            thetat = thetat.detach().to(self.device)
            ema.mul_(1 - beta).add_(thetat, alpha=beta)  # EMA update: ema = (1 - beta) * ema + beta * θₜ
            run_param.copy_(ema + alpha * (thetat - theta0))  # BEMA update: run_param = ema + alpha * (θₜ - θ₀)

    @torch.no_grad()
    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: PreTrainedModel, **kwargs
    ):
        # [FUNCTION DOC]
        # Behavior (no grad): does nothing before update_after; at exactly update_after copies live parameters into
        # theta0 and EMA buffers; afterwards, when (step - update_after) is a multiple of update_freq, updates BEMA
        # weights (using the absolute global step in the schedules) and logs info.
        # Inputs: args, state, control, model. Outputs: none. Exceptions: as _update_bema_weights.
        # Side effects: mutates buffers / running_model.
        step = state.global_step

        # If we haven't reached the update_after step, skip the BEMA update
        if step < self.update_after:
            return

        # Snapshot θ₀ and EMA at first update
        if step == self.update_after:
            for thetat_param, theta0_param, ema_param in zip(
                self.thetat_params, self.theta0_params, self.ema_params, strict=True
            ):
                theta0_param.copy_(thetat_param)
                ema_param.copy_(thetat_param)

        # Update BEMA weights every `update_freq` steps
        elif (step - self.update_after) % self.update_freq == 0:
            self._update_bema_weights(step)
            logger.info(f"Updated BEMA weights at step {step}")

    @torch.no_grad()
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # [FUNCTION DOC]
        # Behavior: on world process zero saves running_model with save_pretrained to output_dir/bema and logs.
        # Inputs: args, state, control. Outputs: none. Exceptions: I/O errors.
        # Side effects: writes model files.
        if state.is_world_process_zero:
            save_directory = f"{args.output_dir}/bema"
            self.running_model.save_pretrained(save_directory)
            logger.info(f"Saved BEMA model to {save_directory}")
