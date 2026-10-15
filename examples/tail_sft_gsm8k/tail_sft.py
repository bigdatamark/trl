# [MODULE DOC]
# Example implementing TailSFT (paper 2608.25756) on GSM8K: SFT where, in each selection batch (one forward pass across
# all ranks), the examples whose length-normalized loss has fallen furthest below the initial policy's loss are dropped
# from the gradient, with a static or linearly ramped drop fraction.
# Contents: argument dataclasses with paper defaults (TailSFTScriptArguments, TailSFTConfig, TailSFTModelConfig), a
# collator that also batches initial_loss, TailSFTTrainer (SFTTrainer subclass), main and make_parser.
# Under the main guard parses the three dataclasses with TrlParser and calls main. Importing has no side effects.

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

# /// script
# dependencies = [
#     "trl",
# ]
# ///

"""TailSFT on GSM8K, based on https://huggingface.co/papers/2608.25756."""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func

from trl import ModelConfig, ScriptArguments, SFTConfig, SFTTrainer, TrlParser
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


@dataclass
class TailSFTScriptArguments(ScriptArguments):
    # [CLASS DOC]
    # ScriptArguments subclass whose defaults point at openai/gsm8k with config main. No methods.
    r"""
    [`ScriptArguments`] with GSM8K as the default dataset.

    Parameters whose default values are overridden:

    > - `dataset_name`: Defaults to `"openai/gsm8k"`.
    > - `dataset_config`: Defaults to `"main"`.
    """

    dataset_name: str = "openai/gsm8k"
    dataset_config: str | None = "main"


@dataclass
class TailSFTConfig(SFTConfig):
    # [CLASS DOC]
    # SFTConfig subclass with the paper's training defaults (learning rate 3e-5, adam beta2 0.95, weight decay 0.1, 3%
    # warmup, constant-with-warmup schedule, 2 epochs, per-device batch 2, gradient accumulation 4, max length 4096) and two
    # new fields: filter_fraction (float, default 0.5, fraction of each selection batch to drop; 0 means standard SFT) and
    # filter_schedule (static or ramp, default ramp). No methods.
    r"""
    [`SFTConfig`] with the filtering schedule of TailSFT.

    Parameters whose default values are overridden:

    > - `learning_rate`: Defaults to `3e-5`.
    > - `adam_beta2`: Defaults to `0.95`.
    > - `weight_decay`: Defaults to `0.1`.
    > - `warmup_steps`: Defaults to `0.03` (3% of the training steps).
    > - `lr_scheduler_type`: Defaults to `"constant_with_warmup"`.
    > - `num_train_epochs`: Defaults to `2`.
    > - `per_device_train_batch_size`/`gradient_accumulation_steps`: Default to `2`/`4` (global batch size 64 and a
    >   16-example selection batch on 8 GPUs).
    > - `max_length`: Defaults to `4096`.

    Additional parameters:

    > - `filter_fraction`: Defaults to `0.5`.
    > - `filter_schedule`: Defaults to `"ramp"`.
    """

    learning_rate: float = 3e-5
    adam_beta2: float = 0.95
    weight_decay: float = 0.1
    warmup_steps: float = 0.03
    lr_scheduler_type: str = "constant_with_warmup"
    num_train_epochs: float = 2.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_length: int = 4096
    filter_fraction: float = field(
        default=0.5,
        metadata={"help": "Fraction of each selection batch to drop. `0` recovers standard SFT."},
    )
    filter_schedule: str = field(
        default="ramp",
        metadata={
            "help": "How the filter fraction moves over training. `static` holds it at `filter_fraction`; `ramp` "
            "raises it linearly from 0 at the first step to `filter_fraction` at the last.",
            "choices": ["static", "ramp"],
        },
    )


@dataclass
class TailSFTModelConfig(ModelConfig):
    # [CLASS DOC]
    # ModelConfig subclass defaulting model_name_or_path to allenai/Olmo-3-1025-7B and dtype to bfloat16. No methods.
    r"""
    [`ModelConfig`] with the paper's model as default.

    Parameters whose default values are overridden:

    > - `model_name_or_path`: Defaults to `"allenai/Olmo-3-1025-7B"`.
    > - `dtype`: Defaults to `"bfloat16"`.
    """

    model_name_or_path: str = "allenai/Olmo-3-1025-7B"
    dtype: str = "bfloat16"


@dataclass
class DataCollatorForTailSFT(DataCollatorForLanguageModeling):
    # [CLASS DOC]
    # HELPER: DataCollatorForLanguageModeling subclass that adds a 1-D initial_loss tensor (one float per example) to the
    # batch produced by the parent collator.
    """SFT collator that also batches the recorded initial-policy loss of each example."""

    def torch_call(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        # [FUNCTION DOC]
        # HELPER: Calls the parent collator then sets initial_loss to a tensor of each example's initial_loss value.
        # Inputs: examples (list of dict, each with initial_loss). Outputs: batch dict of tensors.
        # Exceptions: KeyError if an example lacks initial_loss. Side effects: none.
        output = super().torch_call(examples)
        output["initial_loss"] = torch.tensor([example["initial_loss"] for example in examples])
        return output


class TailSFTTrainer(SFTTrainer):
    # [CLASS DOC]
    # SFTTrainer subclass implementing TailSFT filtering. On construction it rejects packing, forces loss scaling for
    # gradient accumulation, records per-example initial-policy losses as an initial_loss dataset column (unless already
    # present), and swaps in DataCollatorForTailSFT. compute_loss performs the filtering during training and falls back to
    # the parent loss in eval. Logs filtered_fraction (and aux_loss for MoE) into self._metrics train.
    """
    [`SFTTrainer`] with the sequence filtering of TailSFT.

    Reads the `initial_loss` column of the training dataset, and records it with the initial policy when the column is
    absent. An evaluation dataset, if any, needs the column too. For Mixture-of-Experts models, the load-balancing loss
    is computed only over the sequences kept by the filter.
    """

    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Initializes the parent SFTTrainer, then adapts it for TailSFT.
        # Behavior: raises if args.packing; sets model_accepts_loss_kwargs False so the parent scales the loss under gradient
        # accumulation; if the tokenized train dataset lacks initial_loss, adds it from record_initial_losses; replaces the
        # data collator with DataCollatorForTailSFT reusing the pad token id and the pad_to_multiple_of arg.
        # Inputs: same as SFTTrainer. Outputs: none.
        # Exceptions: ValueError when packing is enabled. Side effects: a full forward pass over the training set; mutates
        # train_dataset and data_collator.
        super().__init__(*args, **kwargs)
        if self.args.packing:
            raise ValueError("Packing is not supported: filtering is per sequence, and packing concatenates them.")

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Algorithm 1, line 2. Recorded here, once the parent has tokenized the dataset, so the initial loss is measured
        # over exactly the target tokens the training loss will use.
        if "initial_loss" not in self.train_dataset.column_names:
            self.train_dataset = self.train_dataset.add_column("initial_loss", self.record_initial_losses())

        self.data_collator = DataCollatorForTailSFT(
            pad_token_id=self.data_collator.pad_token_id, pad_to_multiple_of=self.args.pad_to_multiple_of
        )

    def _set_signature_columns_if_needed(self):
        # [FUNCTION DOC]
        # HELPER: Extends the parent's signature columns with initial_loss so remove_unused_columns keeps it for the collator.
        # Inputs: none. Outputs: none. Exceptions: none. Side effects: mutates self._signature_columns.
        # Without this, `initial_loss` is dropped from the dataset before it reaches the collator.
        super()._set_signature_columns_if_needed()
        self._signature_columns = [*self._signature_columns, "initial_loss"]

    @torch.no_grad()
    def record_initial_losses(self) -> list[float]:
        # [FUNCTION DOC]
        # Scores every training example with the initial policy, without gradients.
        # Behavior: every rank iterates the whole train dataset (not sharded) in batches of per_device_train_batch_size using
        # the current collator; model in eval mode; for each batch moves tensors to the model device, runs a forward on
        # input_ids and attention_mask, gets per-token loss and mask from per_token_loss, and appends each sequence's summed
        # loss divided by its target-token count; restores train mode.
        # Inputs: none. Outputs: list of floats (one per training example, in dataset order).
        # Exceptions: propagated from the model; division by zero yields NaN for examples without target tokens.
        # Side effects: toggles model mode; GPU compute.
        """Length-normalized loss of the initial policy over every training example."""
        # Every rank scores the whole dataset. The pass is redundant across ranks, but it is forward-only and keeps the
        # recording free of cross-rank bookkeeping.
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
        )
        self.model.eval()
        initial_losses = []
        for inputs in dataloader:
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            logits = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
            per_token_loss, loss_mask = self.per_token_loss(logits, inputs["labels"])
            initial_losses += (per_token_loss.sum(-1) / loss_mask.sum(-1)).tolist()
        self.model.train()
        return initial_losses

    def per_token_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC]
        # HELPER: Next-token cross-entropy per position.
        # Behavior: drops the last logit position and the first label position, computes unreduced cross-entropy with
        # ignore index -100 (zero loss at ignored positions), and returns the mask of non-ignored shifted labels.
        # Inputs: logits (batch, seq, vocab), labels (batch, seq). Outputs: tuple of loss (batch, seq-1) and bool mask
        # (batch, seq-1). Exceptions: none. Side effects: none.
        """Cross-entropy of every next-token prediction, zero outside the target tokens, and the mask of those."""
        shift_logits, shift_labels = logits[..., :-1, :], labels[..., 1:]
        loss = F.cross_entropy(shift_logits.transpose(1, 2), shift_labels, ignore_index=-100, reduction="none")
        return loss, shift_labels != -100

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # [FUNCTION DOC]
        # TailSFT training loss with per-sequence filtering.
        # Behavior: pops initial_loss; if the model is not training, returns the parent compute_loss. Otherwise pops labels,
        # requests router logits when aux loss is enabled, runs the model with use_cache False and gets per-token loss/mask.
        # Fraction is filter_fraction, multiplied for the ramp schedule by global_step divided by max(max_steps minus 1, 1).
        # Margins are the detached length-normalized current loss minus initial_loss, gathered across ranks. The examples
        # with the smallest margins, count round(n times fraction) capped at n minus 1, are marked dropped; the dropped share
        # is recorded; the keep mask is reshaped to (num_processes, per-rank batch) and this rank's row is taken. Loss is the
        # sum of kept-token losses times num_processes divided by the total kept-token count gathered across ranks (so the
        # DDP average equals the global token mean). For MoE (aux_loss_enabled), recomputes the load-balancing loss with
        # dropped sequences masked out of the attention mask (zero-valued graph-connected loss if all are dropped), adds it
        # times router_aux_loss_coef, and logs its gathered mean as aux_loss. Logs filtered_fraction.
        # Inputs: model, inputs (batch dict), return_outputs (bool), num_items_in_batch (unused in training path).
        # Outputs: loss tensor, or (loss, outputs) when return_outputs.
        # Exceptions: propagated; reshape fails if ranks have different batch sizes.
        # Side effects: mutates inputs (pops keys, may add output_router_logits); collective gathers; appends to self._metrics.
        initial_loss = inputs.pop("initial_loss")
        if not self.model.training:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        labels = inputs.pop("labels")
        if self.aux_loss_enabled:
            inputs["output_router_logits"] = True
        outputs = model(**inputs, use_cache=False)
        per_token_loss, loss_mask = self.per_token_loss(outputs.logits, labels)

        # Algorithm 1, line 6: over the selection batch, the examples of a single forward pass across all ranks, drop
        # the fraction whose loss has dropped the furthest below where the initial policy left it. Both losses are
        # length-normalized, so the decision doesn't favor longer or shorter responses.
        fraction = self.args.filter_fraction
        if self.args.filter_schedule == "ramp":
            fraction *= self.state.global_step / max(self.state.max_steps - 1, 1)
        margins = (per_token_loss.sum(-1) / loss_mask.sum(-1)).detach() - initial_loss
        margins = self.accelerator.gather(margins)
        # The paper drops round(world_size * batch_size * f) examples; keep one so the token average stays defined.
        keep = torch.ones_like(margins, dtype=torch.bool)
        keep[margins.argsort()[: min(round(margins.numel() * fraction), margins.numel() - 1)]] = False
        filtered_fraction = 1 - keep.float().mean().item()
        keep = keep.view(self.accelerator.num_processes, -1)[self.accelerator.process_index]

        # Algorithm 1, line 8: token-averaged cross-entropy over the survivors. The average runs over the whole selection
        # batch, so the per-rank losses are scaled to sum to it once the backward pass averages them.
        loss_mask = loss_mask & keep.unsqueeze(-1)
        num_tokens = self.accelerator.gather(loss_mask.sum()).sum()
        loss = (per_token_loss * loss_mask).sum() * self.accelerator.num_processes / num_tokens

        if self.aux_loss_enabled:
            # The model's auxiliary loss includes every sequence from the forward pass. Recompute it with filtered
            # sequences masked out so they do not update the router.
            router_attention_mask = inputs["attention_mask"] * keep.unsqueeze(-1)
            if router_attention_mask.any():
                aux_loss = load_balancing_loss_func(
                    outputs.router_logits,
                    outputs.router_logits[0].shape[-1],
                    self.model.config.get_text_config().num_experts_per_tok,
                    router_attention_mask,
                )
            else:
                aux_loss = outputs.router_logits[0].sum() * 0.0
            loss = loss + self.args.router_aux_loss_coef * aux_loss.to(loss.device)
            aux_loss = self.accelerator.gather_for_metrics(aux_loss.detach()).mean().item()
            self._metrics["train"]["aux_loss"].append(aux_loss)

        self._metrics["train"]["filtered_fraction"].append(filtered_fraction)
        return (loss, outputs) if return_outputs else loss


def main(script_args, training_args, model_args):
    # [FUNCTION DOC]
    # Runs TailSFT on GSM8K.
    # Behavior: sets model_init_kwargs (revision, attention, dtype); loads the train split; maps each row to a
    # prompt-completion pair whose user message is the question followed by a blank line and an instruction to put the
    # answer in a LaTeX boxed expression, and whose assistant message is the reference answer (other columns removed);
    # builds TailSFTTrainer from the model name; trains, saves, optionally pushes, calls end.
    # Inputs: script_args, training_args (TailSFTConfig), model_args. Outputs: none.
    # Exceptions: propagated. Side effects: downloads, training, file writes, optional Hub upload.
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )

    dataset = load_dataset(script_args.dataset_name, script_args.dataset_config, split=script_args.dataset_train_split)
    instruction = "Present the answer in LaTex format: \\boxed{Your answer}"
    dataset = dataset.map(
        lambda example: {
            "prompt": [{"role": "user", "content": f"{example['question']}\n\n{instruction}"}],
            "completion": [{"role": "assistant", "content": example["answer"]}],
        },
        remove_columns=dataset.column_names,
    )

    trainer = TailSFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.end()


def make_parser():
    # [FUNCTION DOC]
    # HELPER: Returns a TrlParser over TailSFTScriptArguments, TailSFTConfig and TailSFTModelConfig.
    # Inputs: none. Outputs: TrlParser. Exceptions: none. Side effects: none.
    dataclass_types = (TailSFTScriptArguments, TailSFTConfig, TailSFTModelConfig)
    return TrlParser(dataclass_types)


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
