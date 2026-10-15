# [MODULE DOC]
# Experimental trainer implementing A*-PO (Optimal Advantage Regression, https://huggingface.co/papers/2505.20686).
# Contents: the RewardFunc type alias (callable returning a list of floats) and A2POTrainer.
# Import-time side effects: creates a module logger.

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

import inspect
import math
import textwrap
from collections import defaultdict
from collections.abc import Callable

import torch
from accelerate.logging import get_logger
from accelerate.utils import gather_object
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from ...data_utils import maybe_apply_chat_template
from ...models import unwrap_model_for_generation
from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import selective_log_softmax
from ..utils import create_reference_model
from .a2po_config import A2POConfig


logger = get_logger(__name__)

# A reward function is a callable that returns a list of floats (the rewards). The callable receives prompts,
# completions, and additional columns from the dataset. To ensure forward compatibility, it should accept **kwargs.
RewardFunc = Callable[..., list[float]]


class A2POTrainer(_BaseTrainer):
    # [CLASS DOC]
    # Trainer (subclass of _BaseTrainer / transformers Trainer) running A*-PO in two stages.
    # Stage 1 (_estimate_optimal_values, run once before training or lazily on first input preparation): sample
    # num_value_samples completions per prompt from a frozen reference copy of the model, score them, and cache
    # V*(x) = beta1 times log of the mean of exp(r/beta1), keyed by the chat-templated prompt text; optionally drop
    # training prompts whose samples all scored zero.
    # Stage 2 (_prepare_inputs + compute_loss): generate one completion per prompt with the current policy, compute its
    # reward and the reference sequence log-prob, then regress beta2 times (policy minus reference sequence log-prob) onto
    # r minus V* with a mean squared error.
    # Key state: ref_model (frozen copy, prepared by accelerate in eval mode), reward_funcs and reward_weights (float32
    # tensor), eos_token_ids, value_generation_config / generation_config (sampling configs with N and 1 return
    # sequences), _optimal_values (dict prompt text -> float, None until Stage 1), _metrics (per-mode lists averaged in
    # log), model_kwarg_keys (forward signature parameter names). Class attributes _tag_names, _name, _paper feed model
    # card generation.
    """
    Trainer for the A*-PO (Optimal Advantage Regression) method, introduced in [Accelerating RL for LLM Reasoning with
    Optimal Advantage Regression](https://huggingface.co/papers/2505.20686).

    A*-PO runs in two stages:

    1. **Offline value estimation.** Before training, `num_value_samples` completions are sampled from the reference
       policy for every training prompt and scored with `reward_funcs`. The optimal value is estimated as `V*(x) =
       beta1 * log(mean_i exp(r(x, y_i) / beta1))` and cached per prompt.
    2. **On-policy regression.** During training, a single completion is generated per prompt from the current policy.
       The loss is the squared error between the implicit reward `beta2 * log(pi(y|x) / pi_ref(y|x))` and the optimal
       advantage estimate `r(x, y) - V*(x)`.

    Args:
        model (`PreTrainedModel` or `str`):
            Model to be trained, or a model identifier (string) passed to
            [`~transformers.AutoModelForCausalLM.from_pretrained`].
        reward_funcs (`Callable` or `list[Callable]`):
            Reward function(s). Each takes `prompts` and `completions` (plus dataset columns as keyword arguments) and
            returns a list of float rewards. When multiple are provided, their weighted sum (see
            [`A2POConfig.reward_weights`]) is the scalar reward `r`, which A*-PO assumes to be binary (in `{0, 1}`).
        args ([`experimental.a2po.A2POConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`], *optional*):
            Training dataset. Must contain a `"prompt"` column.
        eval_dataset ([`~datasets.Dataset`], *optional*):
            Evaluation dataset.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*):
            Processing class used to process the data. If `None`, it is loaded from the model's name with
            [`~transformers.AutoTokenizer.from_pretrained`].
        callbacks (`list[~transformers.TrainerCallback]`, *optional*):
            List of callbacks to customize the training loop.
        optimizers (`tuple[~torch.optim.Optimizer, ~torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            Tuple containing the optimizer and the learning rate scheduler.
    """

    _tag_names = ["trl", "a2po"]
    _name = "A2PO"
    _paper = {
        "title": "Accelerating RL for LLM Reasoning with Optimal Advantage Regression",
        "id": "2505.20686",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{brantley2025accelerating,
                title        = {{Accelerating RL for LLM Reasoning with Optimal Advantage Regression}},
                author       = {Kiant\'e Brantley and Mingyu Chen and Zhaolin Gao and Jason D. Lee and Wen Sun and Wenhao Zhan and Xuezhou Zhang},
                year         = 2025,
                eprint       = {arXiv:2505.20686},
            }"""),
    }

    def __init__(
        self,
        model: PreTrainedModel | str,
        reward_funcs: RewardFunc | list[RewardFunc],
        args: A2POConfig | None = None,
        train_dataset=None,
        eval_dataset=None,
        processing_class: PreTrainedTokenizerBase | None = None,
        callbacks=None,
        optimizers=(None, None),
    ):
        # [FUNCTION DOC]
        # Builds the trainer.
        # Behavior: default args to A2POConfig with output dir "<model name>-A2PO"; if model is a string, load it with
        # AutoModelForCausalLM using model_init_kwargs (setting trust_remote_code default, remembering revision); record the
        # forward signature keys; create the reference model; load a left-padding tokenizer when none is given; collect EOS ids
        # from the generation config (int or list) and prepend the tokenizer's eos id if missing; wrap a single reward function
        # in a list and build reward_weights (ones or args.reward_weights); build the two GenerationConfigs (do_sample, given
        # temperature/top_p/top_k, max_new_tokens = max_completion_length, pad id from tokenizer); use an identity data
        # collator (list of raw examples); call the parent constructor; then prepare ref_model with accelerate in eval mode.
        # Inputs: model (PreTrainedModel or id), reward_funcs (callable or list), args, train_dataset (needs "prompt"),
        # eval_dataset, processing_class, callbacks, optimizers.
        # Outputs: none. Exceptions: model/tokenizer loading errors; parent Trainer validation errors.
        # Side effects: downloads/loads model weights; mutates args.model_init_kwargs (setdefault) when it is a dict.
        # Args
        if args is None:
            args = A2POConfig(f"{model if isinstance(model, str) else model.config._name_or_path}-A2PO")

        # Models
        model_revision = None
        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            model_init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)
            model_revision = model_init_kwargs.get("revision")
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        model_id = model.config._name_or_path

        # Some models (e.g. SmolVLM/Idefics3) don't support the `logits_to_keep` argument and error out if we pass it.
        # Inspect the forward method so Stage 2 can pass the argument only when it is supported.
        self.model_kwarg_keys = inspect.signature(model.forward).parameters.keys()

        # Reference model: a frozen copy of the initial policy. Stage 1 samples from it and Stage 2 regularizes to it.
        self.ref_model = create_reference_model(model)

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(
                model_id, revision=model_revision, padding_side="left", trust_remote_code=args.trust_remote_code
            )

        # Every eos id the model declares (e.g. an end-of-turn token) ends a completion, not only the tokenizer's
        eos_token_ids = model.generation_config.eos_token_id
        if eos_token_ids is None:
            eos_token_ids = []
        elif isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        if processing_class.eos_token_id not in eos_token_ids:
            eos_token_ids = [processing_class.eos_token_id, *eos_token_ids]
        self.eos_token_ids = eos_token_ids

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs
        if args.reward_weights is None:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)
        else:
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)

        # Generation: N samples per prompt in Stage 1, a single sample in Stage 2
        self.value_generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_return_sequences=args.num_value_samples,
            pad_token_id=processing_class.pad_token_id,
        )
        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_return_sequences=1,
            pad_token_id=processing_class.pad_token_id,
        )

        # Optimal values V*(x), keyed by prompt text. Populated lazily by Stage 1 at the start of training.
        self._optimal_values: dict[str, float] | None = None

        # Metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

        # The data collator returns the list of features untouched; generation happens in `_prepare_inputs`.
        def data_collator(features):
            # [FUNCTION DOC]
            # HELPER: Identity collator - returns the list of dataset examples unchanged (generation happens in _prepare_inputs).
            # Inputs: features (list of dict). Outputs: the same list. Exceptions: none. Side effects: none.
            return features

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def _calculate_rewards(self, prompts, completions, **reward_kwargs):
        # [FUNCTION DOC]
        # HELPER: Calls each reward function with prompts, completions and extra keyword columns, stores outputs in a
        # (num_samples, num_funcs) float32 tensor on the accelerator device, and returns the reward_weights-weighted sum per
        # sample (shape num_samples). A*-PO assumes this total is binary.
        # Inputs: prompts (list), completions (list of str), reward_kwargs (column name -> list aligned with prompts).
        # Outputs: 1D float tensor. Exceptions: errors from reward functions or length mismatches. Side effects: none.
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, reward_func in enumerate(self.reward_funcs):
            output = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output, dtype=torch.float32, device=device)
        # A*-PO assumes the (weighted) total reward is binary in {0, 1}.
        return (rewards_per_func * self.reward_weights.to(device)).sum(dim=1)

    def _get_sequence_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # [FUNCTION DOC]
        # HELPER: Sums per-token log-probabilities of the last logits_to_keep tokens (the completion) under model.
        # Behavior: forward with input_ids/attention_mask (adding logits_to_keep + 1 when the model supports it); drop the
        # final logit position; keep the last logits_to_keep positions; gather log-softmax at the completion token ids;
        # multiply by the completion part of attention_mask and sum along the sequence.
        # Inputs: model, input_ids and attention_mask (batch, seq), logits_to_keep (int).
        # Outputs: tensor of shape (batch,). Exceptions: model forward errors. Side effects: none (grad tracked if enabled).
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        # Only pass `logits_to_keep` when the model supports it (some models and VLMs don't).
        if "logits_to_keep" in self.model_kwarg_keys:
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            model_inputs["logits_to_keep"] = logits_to_keep + 1
        logits = model(**model_inputs).logits
        logits = logits[:, :-1, :]  # the last logit predicts beyond the sequence
        # Only keep the last logits_to_keep. For models that support logits_to_keep, this is a no-op.
        logits = logits[:, -logits_to_keep:, :]
        completion_ids = input_ids[:, -logits_to_keep:]
        per_token_logps = selective_log_softmax(logits, completion_ids)
        completion_mask = attention_mask[:, -logits_to_keep:]
        return (per_token_logps * completion_mask).sum(dim=1)

    # Stage 1: offline optimal value estimation
    def _estimate_optimal_values(self):
        # [FUNCTION DOC]
        # Stage 1: offline estimation of V* for every train (and eval, if given) prompt.
        # Behavior: for each dataset, iterate an accelerate-prepared DataLoader (batch size per_device_train_batch_size,
        # collate to list); chat-template prompts; tokenize with padding/left truncation to max_prompt_length without special
        # tokens; generate num_value_samples completions per prompt from the unwrapped reference model; decode completion
        # tokens; repeat prompts and extra columns (except "prompt"/"completion") N times and compute rewards; reshape to
        # (batch, N); compute V* stably as beta1 times (logsumexp of rewards/beta1 minus log N); store per prompt text and
        # mark prompts whose rewards sum to zero. Then all-gather the (prompt, value) pairs and the all-incorrect set across
        # ranks into _optimal_values; if filter_all_incorrect, filter train_dataset to remove all-incorrect prompts (eval kept).
        # Inputs: none (uses args, datasets). Outputs: none.
        # Exceptions: generation/reward errors propagate.
        # Side effects: sets self._optimal_values, may replace self.train_dataset, logs an info message, collective comms.
        beta1 = self.args.beta1
        n = self.args.num_value_samples
        optimal_values = {}
        all_incorrect = set()

        # Stage 2 looks up V* for every prompt it scores, including eval prompts, so estimate over both datasets.
        datasets = [self.train_dataset] if self.eval_dataset is None else [self.train_dataset, self.eval_dataset]
        for dataset in datasets:
            dataloader = self.accelerator.prepare(
                DataLoader(dataset, batch_size=self.args.per_device_train_batch_size, collate_fn=list)
            )
            for batch in dataloader:
                prompts = [example["prompt"] for example in batch]
                prompts_text = [
                    maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in batch
                ]

                # Sample N completions per prompt from the reference policy
                inputs = self.processing_class(
                    prompts_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.args.max_prompt_length,
                    add_special_tokens=False,
                ).to(self.accelerator.device)
                with unwrap_model_for_generation(self.ref_model, self.accelerator) as ref_model:
                    completion_ids = ref_model.generate(**inputs, generation_config=self.value_generation_config)

                prompt_length = inputs["input_ids"].size(1)
                completions_text = self.processing_class.batch_decode(
                    completion_ids[:, prompt_length:], skip_special_tokens=True
                )

                # Each prompt is repeated N times by `num_return_sequences`. Forward any extra dataset columns (e.g.
                # "solution") to the reward functions, repeated to align with the N samples per prompt.
                keys = [key for key in batch[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in batch for _ in range(n)] for key in keys}
                repeated_prompts = [p for p in prompts for _ in range(n)]
                rewards = self._calculate_rewards(repeated_prompts, completions_text, **reward_kwargs)
                rewards = rewards.view(len(prompts), n)

                # V*(x) = beta1 * log(mean_i exp(r_i / beta1)), computed stably
                v_star = beta1 * (torch.logsumexp(rewards / beta1, dim=1) - math.log(n))

                for j, prompt_text in enumerate(prompts_text):
                    optimal_values[prompt_text] = v_star[j].item()
                    if rewards[j].sum() == 0:
                        all_incorrect.add(prompt_text)

        # Each rank estimates V* for its shard of prompts; share them so any rank can score any prompt.
        # `gather_object` concatenates the per-rank lists, so pass flat lists and rebuild on every rank.
        self._optimal_values = dict(gather_object(list(optimal_values.items())))
        all_incorrect = set(gather_object(list(all_incorrect)))

        # Drop training prompts whose reference samples all scored zero (no learning signal). Eval prompts are kept
        # so evaluation can still look up their V*.
        if self.args.filter_all_incorrect:
            self.train_dataset = self.train_dataset.filter(
                lambda example: (
                    maybe_apply_chat_template(example, self.processing_class)["prompt"] not in all_incorrect
                )
            )
        logger.info(f"Stage 1 complete: estimated V* for {len(self._optimal_values)} prompts.")

    # Stage 2: on-policy regression
    def _prepare_inputs(self, inputs):
        # [FUNCTION DOC]
        # Stage 2: turns a list of raw examples into a loss batch.
        # Behavior: runs Stage 1 first if V* is not yet available; chat-templates and tokenizes prompts as in Stage 1;
        # generates one completion per prompt with the unwrapped policy; decodes; computes rewards (forwarding extra columns);
        # looks up cached V* by prompt text; builds a completion mask that is 1 up to and including the first EOS (any id in
        # eos_token_ids; whole length if none) and concatenates it after the prompt attention mask; computes reference sequence
        # log-probs without grad. Appends mean reward and mean v_star to _metrics[mode] (mode from model.training).
        # Inputs: inputs (list of dict with "prompt" plus extra columns).
        # Outputs: dict with input_ids, attention_mask, logits_to_keep (completion length), ref_logps, rewards, v_star.
        # Exceptions: KeyError if a prompt's V* is missing (e.g. prompt not in the estimated datasets).
        # Side effects: may run Stage 1; appends metrics.
        # Estimate V* on first use, e.g. when `evaluate()` is called without a preceding `train()`.
        if self._optimal_values is None:
            self._estimate_optimal_values()
        mode = "train" if self.model.training else "eval"
        device = self.accelerator.device
        prompts = [example["prompt"] for example in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        # One on-policy completion per prompt
        prompt_inputs = self.processing_class(
            prompts_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.max_prompt_length,
            add_special_tokens=False,
        ).to(device)
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Scalar (binary) reward and cached optimal value. Forward extra dataset columns to the reward functions.
        keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        rewards = self._calculate_rewards(prompts, completions_text, **reward_kwargs)
        v_star = torch.tensor([self._optimal_values[p] for p in prompts_text], dtype=torch.float32, device=device)

        # Attention mask: the tokenizer's prompt mask followed by the completion mask. The completion mask is 1 up to
        # and including the first EOS and 0 afterwards, so the terminal EOS stays in the log-prob sum (a plain
        # `!= pad_token_id` mask would drop it when `pad_token == eos_token`).
        is_eos = torch.isin(completion_ids, torch.tensor(self.eos_token_ids, device=device))
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        attention_mask = torch.cat([prompt_inputs["attention_mask"], completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            ref_logps = self._get_sequence_logps(self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep)

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["v_star"].append(v_star.mean().item())

        return {
            "input_ids": prompt_completion_ids,
            "attention_mask": attention_mask,
            "logits_to_keep": logits_to_keep,
            "ref_logps": ref_logps,
            "rewards": rewards,
            "v_star": v_star,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # [FUNCTION DOC]
        # Computes the A*-PO regression loss: policy sequence log-probs via _get_sequence_logps; implicit reward = beta2 times
        # (policy minus ref log-probs); target = rewards minus v_star; loss = mean squared difference. Logs mean implicit
        # reward and mean advantage (target) to _metrics[mode].
        # Inputs: model, inputs (dict from _prepare_inputs); return_outputs and num_items_in_batch are ignored.
        # Outputs: scalar loss tensor. Exceptions: none of its own. Side effects: appends metrics.
        mode = "train" if self.model.training else "eval"

        policy_logps = self._get_sequence_logps(
            model, inputs["input_ids"], inputs["attention_mask"], inputs["logits_to_keep"]
        )

        # Implicit reward beta2 * log(pi / pi_ref), regressed onto the optimal advantage r - V*
        implicit_reward = self.args.beta2 * (policy_logps - inputs["ref_logps"])
        target = inputs["rewards"] - inputs["v_star"]
        loss = ((implicit_reward - target) ** 2).mean()

        self._metrics[mode]["implicit_reward"].append(implicit_reward.mean().item())
        self._metrics[mode]["advantage"].append(target.mean().item())
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: list[str] | None = None):
        # [FUNCTION DOC]
        # Evaluation step: prepares inputs (generation + rewards), computes the loss under no_grad within the compute-loss
        # context manager, and returns (detached mean loss, None, None).
        # Inputs: model, inputs (raw examples), prediction_loss_only and ignore_keys (ignored).
        # Outputs: tuple (loss, None, None). Exceptions: as _prepare_inputs. Side effects: appends eval metrics.
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def train(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Runs Stage 1 (with an info log) if V* has not been estimated yet, so prompt filtering happens before the train
        # dataloader is built, then delegates to the parent train.
        # Inputs: parent train args/kwargs. Outputs: parent TrainOutput. Exceptions: as parent / Stage 1.
        # Side effects: training, may filter train_dataset.
        if self._optimal_values is None:
            logger.info("Running Stage 1: offline optimal value estimation...")
            self._estimate_optimal_values()
        return super().train(*args, **kwargs)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        # [FUNCTION DOC]
        # Averages the accumulated metric lists for the current mode, prefixes keys with "eval_" in eval mode, merges them into
        # logs, forwards to the parent log, and clears that mode's metrics.
        # Inputs: logs (dict), start_time (optional float). Outputs: none.
        # Exceptions: none expected.
        # Side effects: emits logs to reporters; clears self._metrics[mode].
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()
