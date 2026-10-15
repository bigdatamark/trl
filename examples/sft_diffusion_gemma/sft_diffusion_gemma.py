# [MODULE DOC]
# Example implementing block-diffusion SFT of DiffusionGemma (encoder-decoder diffusion LM) on GSM8K, optionally with
# LoRA. The objective (one random response block per example, uniform random-token corruption, two-pass
# self-conditioning, flat cross-entropy over the canvas, plus an autoregressive encoder co-loss, with an optional
# leave-one-out logit conversion) is described in the module docstring and implemented in DiffusionGemmaSFTTrainer.
# Contents: argument dataclasses with reference-recipe defaults, DiffusionGemmaSFTTrainer, main, make_parser.
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
#     "peft",
#     "transformers @ git+https://github.com/huggingface/transformers.git",
# ]
# ///

"""
Supervised fine-tuning of DiffusionGemma, a block-diffusion language model, on GSM8K with LoRA.

DiffusionGemma ([`google/diffusiongemma-26B-A4B-it`](https://huggingface.co/google/diffusiongemma-26B-A4B-it)) couples
a causal encoder with a bidirectional decoder: the encoder reads clean context tokens into a KV cache, and the decoder
denoises a block of response tokens (the "canvas") with bidirectional attention, cross-attending to that cache.
Training therefore differs from autoregressive SFT:

1. Per example, one response block of `canvas_length` tokens is selected at random; the encoder reads the full clean
   sequence, and the decoder mask restricts the canvas to the prompt plus the clean response blocks before it.
2. The clean canvas (EOS-filled past the end of the response) is corrupted by independently replacing tokens with
   uniform-random vocabulary tokens with per-example probability `t ~ U(eps, 1 - eps)`. There is no mask token.
3. With probability 0.5 per example, the decoder is self-conditioned on its own logits from a first, no-grad pass.
4. The loss is plain mean cross-entropy between the decoder logits and the clean tokens over the whole canvas
   (corrupted and uncorrupted positions alike), plus an autoregressive co-loss on the encoder.

For uniform diffusion the plug-in softmax does not parameterize the plain denoiser but the leave-one-out (LOO)
posterior, which predicts each token without using its own corrupted value (https://huggingface.co/papers/2605.22765).
Passing `--model_prediction_type mean_loo` keeps the LOO target by converting the decoder logits to the denoiser
before the cross-entropy; the paper reports better generation this way, but the model then parameterizes the LOO
posterior, so sampling must apply the same conversion. The default `mean` trains the plain denoiser, matching the
released checkpoint.

Requires transformers from source: gradient checkpointing support
(https://github.com/huggingface/transformers/pull/46572) is merged but not released yet.

The script trains with `assistant_only_loss`, so [`SFTTrainer`] swaps in TRL's DiffusionGemma training chat template
(`trl/chat_templates/diffusion_gemma_training.jinja`), whose `{% generation %}` markers cover the assistant content
plus the closing `<turn|>` token.

LoRA follows the reference fine-tune of the released checkpoint: rank 16, alpha 32, adapting the attention and
dense-MLP linears of the encoder and decoder layers only; the MoE experts, the router, the self-conditioning block,
and the vision tower stay frozen.

The default hyperparameters follow the reference fine-tuning configs of the released checkpoint (learning rate
1.5e-4, Adam betas (0.95, 0.99), weight decay 1e-4, 25 warmup steps then cosine to 10% of the initial learning rate,
global batch size 8, sequence length 1024, 800 steps, LoRA rank 16 with alpha 32), so a run only needs:

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/sft_diffusion_gemma/sft_diffusion_gemma.py \
    --use_peft \
    --gradient_checkpointing \
    --output_dir diffusiongemma-26B-A4B-it-gsm8k-lora

Drop `--use_peft` for full fine-tuning, which keeps the MoE router frozen like the reference recipe.

Hardware: the 26B parameters do not fit on a single 80 GB GPU, so this needs at least 2 GPUs to shard them
across ranks with ZeRO-3. Gradient checkpointing (`--gradient_checkpointing`) is required to keep activations within
memory. Two 80 GB H100s are enough for the defaults above; more GPUs increase throughput.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, DiffusionGemmaForBlockDiffusion

from trl import ModelConfig, ScriptArguments, SFTConfig, SFTTrainer, TrlParser
from trl.trainer.utils import maybe_gather_lm_head_ctx


@dataclass
class DiffusionGemmaScriptArguments(ScriptArguments):
    # [CLASS DOC]
    # ScriptArguments subclass defaulting dataset_name to openai/gsm8k and dataset_config to main. No methods.
    r"""
    [`ScriptArguments`] with GSM8K as the default dataset.

    Parameters whose default values are overridden:

    > - `dataset_name`: Defaults to `"openai/gsm8k"`.
    > - `dataset_config`: Defaults to `"main"`.
    """

    dataset_name: str = "openai/gsm8k"
    dataset_config: str | None = "main"


@dataclass
class DiffusionGemmaSFTConfig(SFTConfig):
    # [CLASS DOC]
    # SFTConfig subclass with the released checkpoint's fine-tuning defaults (learning rate 1.5e-4, Adam betas 0.95/0.99,
    # weight decay 1e-4, 25 warmup steps, cosine schedule down to 10% of the peak, 800 max steps, per-device batch 1 with
    # gradient accumulation 8, max length 1024) and one new field model_prediction_type (mean or mean_loo, default mean).
    # No methods.
    r"""
    [`SFTConfig`] whose defaults follow the reference fine-tuning configs of the released checkpoint.

    Parameters whose default values are overridden:

    > - `learning_rate`: Defaults to `1.5e-4`.
    > - `adam_beta1`/`adam_beta2`: Default to `0.95`/`0.99`.
    > - `weight_decay`: Defaults to `1e-4`.
    > - `warmup_steps`: Defaults to `25`.
    > - `lr_scheduler_type`: Defaults to `"cosine_with_min_lr"` with `lr_scheduler_kwargs={"min_lr_rate": 0.1}`.
    > - `max_steps`: Defaults to `800`.
    > - `per_device_train_batch_size`/`gradient_accumulation_steps`: Default to `1`/`8` (global batch size 8).
    > - `max_length`: Defaults to `1024`.

    Additional parameters:

    > - `model_prediction_type`: Defaults to `"mean"`.
    """

    learning_rate: float = 1.5e-4
    adam_beta1: float = 0.95
    adam_beta2: float = 0.99
    weight_decay: float = 1e-4
    warmup_steps: int = 25
    lr_scheduler_type: str = "cosine_with_min_lr"
    lr_scheduler_kwargs: dict | None = field(default_factory=lambda: {"min_lr_rate": 0.1})
    max_steps: int = 800
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_length: int = 1024
    model_prediction_type: str = field(
        default="mean",
        metadata={
            "help": "What the decoder softmax parameterizes. `mean` is the plain denoiser, matching the released "
            "checkpoint. `mean_loo` is the leave-one-out (LOO) posterior, which predicts each token without its own "
            "corrupted value: the logits are converted to the denoiser before the cross-entropy "
            "(https://huggingface.co/papers/2605.22765). LOO improves generation for uniform diffusion but the model "
            "then parameterizes the LOO posterior, so sampling must apply the same conversion.",
            "choices": ["mean", "mean_loo"],
        },
    )


@dataclass
class DiffusionGemmaModelConfig(ModelConfig):
    # [CLASS DOC]
    # ModelConfig subclass defaulting to google/diffusiongemma-26B-A4B-it in bfloat16 with LoRA dropout 0. No methods.
    r"""
    [`ModelConfig`] with the released checkpoint and its reference LoRA settings as defaults.

    Parameters whose default values are overridden:

    > - `model_name_or_path`: Defaults to `"google/diffusiongemma-26B-A4B-it"`.
    > - `dtype`: Defaults to `"bfloat16"`.
    > - `lora_dropout`: Defaults to `0.0`.
    """

    model_name_or_path: str = "google/diffusiongemma-26B-A4B-it"
    dtype: str = "bfloat16"
    lora_dropout: float = 0.0


class DiffusionGemmaSFTTrainer(SFTTrainer):
    # [CLASS DOC]
    # SFTTrainer subclass that replaces the autoregressive loss with the DiffusionGemma block-diffusion loss.
    # Class attributes: eps (1e-4, margin on sampled time t) and self_conditioning_p (0.5). Instance state set in
    # __init__: canvas_length, vocab_size, final_logit_softcapping (from the model's text config), eos_token_id (from the
    # tokenizer) and model_prediction_type. Expects batches from the default SFT collator with labels of -100 outside the
    # assistant response; only compute_loss is overridden (used for both training and evaluation).
    """
    SFTTrainer with the block-diffusion objective of DiffusionGemma.

    Expects the batches produced by the default SFT collator (`input_ids`, `attention_mask`, and `labels` with `-100`
    outside the assistant response) and replaces the autoregressive loss with the denoising loss described in the
    module docstring.
    """

    eps = 1e-4  # safety margin on the sampled time, matching the reference `SafeSpan(safety_epsilon=1e-4)`
    self_conditioning_p = 0.5

    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Initializes the parent trainer, rejects packing, enables loss scaling for gradient accumulation (by setting
        # model_accepts_loss_kwargs False) and caches canvas/vocab/softcap/eos/prediction-type settings.
        # Inputs: same as SFTTrainer. Outputs: none. Exceptions: ValueError when packing is enabled.
        # Side effects: sets instance attributes.
        super().__init__(*args, **kwargs)
        if self.args.packing:
            raise ValueError("Packing is not supported: the diffusion loss needs per-example response spans.")

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        config = self.model.config
        self.canvas_length = config.canvas_length
        self.vocab_size = config.text_config.vocab_size
        self.final_logit_softcapping = config.text_config.final_logit_softcapping
        self.eos_token_id = self.processing_class.eos_token_id
        self.model_prediction_type = self.args.model_prediction_type

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # [FUNCTION DOC]
        # Computes diffusion loss plus encoder autoregressive loss for a batch.
        # Behavior: (1) Response span: supervised positions are labels not equal to -100; span_end is the last supervised
        # position (or -1); prefix_len is the start of the last supervised span (clamped at 0); response_len is span length
        # (0 when nothing is supervised). (2) Truncation flag: when max_length is set, an example is truncated if its real
        # length reaches max_length and its response ends at the last real token. (3) Block selection: number of blocks is
        # ceil of response_len over canvas_length (at least 1); a random block index per example; encoder_len is prefix_len
        # plus block index times block size. (4) Clean canvas: tokens at positions encoder_len plus 0..block_size-1 (indices
        # clamped to the sequence), replaced by eos past the end of the response. (5) Corruption: per-example t uniform in
        # [eps, 1-eps]; each canvas position is replaced by a uniform random vocab token with probability t. (6) Decoder mask:
        # cache positions before encoder_len plus all canvas positions; decoder position ids continue from encoder_len.
        # (7) Self-conditioning: a no-grad forward produces logits that are passed as self_conditioning_logits to the second
        # (grad) forward, with a per-example Bernoulli(0.5) self_conditioning_mask. (8) Diffusion loss: mean cross-entropy
        # between decoder logits and the clean canvas over all positions, excluding the eos fill of truncated examples; with
        # mean_loo, logits are upcast to float32 and the observed (corrupted) token's logit is increased by
        # log(1 + K times alpha over (1 - alpha)) with alpha = 1 - t and K the vocab size; if no target is valid, a zero loss
        # connected to the graph is used. (9) AR co-loss: encoder last hidden states are projected through lm_head (inside a
        # context that gathers sharded ZeRO-3 weights), upcast, soft-capped by tanh with final_logit_softcapping, and scored by
        # next-token cross-entropy over positions where both the current and next token are real.
        # Inputs: model, inputs (dict with input_ids, attention_mask, labels), return_outputs (bool), num_items_in_batch
        # (unused). Outputs: scalar loss (diffusion plus AR), or (loss, outputs) when return_outputs.
        # Exceptions: propagated from the model. Side effects: consumes RNG; two model forwards per call.
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        block_size = self.canvas_length

        # The canvas covers the final assistant turn only, like the reference recipe: multi-turn conversations have
        # several supervised spans, and the diffusion loss needs one contiguous response
        supervised = labels != -100
        positions = torch.arange(seq_len, device=device)
        span_starts = supervised & ~F.pad(supervised, (1, 0))[:, :-1]
        span_end = torch.where(supervised, positions, -1).amax(dim=1)
        prefix_len = torch.where(span_starts, positions, -1).amax(dim=1).clamp(min=0)
        response_len = (span_end - prefix_len + 1) * (span_end >= 0)

        # An example clipped to `max_length` can lose its assistant turn terminator, so its response has no real end
        # and the EOS fill below would train the model to stop at the truncation boundary. Flag those examples to keep
        # the fill out of their loss.
        if self.args.max_length is None:
            truncated = torch.zeros(batch_size, dtype=torch.bool, device=device)
        else:
            # Detect a real clip by the response reaching the last real token, not by length alone: an example that
            # fills `max_length` but keeps a terminator after its response is complete, not truncated.
            real_len = attention_mask.sum(dim=1)
            truncated = (real_len >= self.args.max_length) & (span_end == real_len - 1)

        # One canvas per step: select one response block per example. The encoder reads the full clean sequence (its
        # autoregressive co-loss covers it all), but the decoder may only see the prompt and the clean response
        # blocks *before* the selected one, so the decoder mask cuts the cache off there.
        num_blocks = (response_len - 1).clamp(min=0) // block_size + 1
        block_idx = (torch.rand(batch_size, device=device) * num_blocks).long()
        encoder_len = prefix_len + block_idx * block_size

        # Clean canvas: the selected block, EOS-filled up to block_size past the end of the response. The fill is
        # supervised, so every canvas position contributes to the loss.
        offsets = torch.arange(block_size, device=device)
        abs_idx = (encoder_len[:, None] + offsets).clamp(max=seq_len - 1)
        in_response = offsets < (response_len - block_idx * block_size)[:, None]
        canvas_target = torch.where(in_response, input_ids.gather(1, abs_idx), self.eos_token_id)

        # Uniform random-token corruption: per-example t ~ U(eps, 1 - eps), each canvas position is independently
        # replaced with a token drawn uniformly from the vocabulary (there is no mask token)
        t = self.eps + (1 - 2 * self.eps) * torch.rand(batch_size, 1, device=device)
        corrupt = torch.rand(batch_size, block_size, device=device) < t
        random_tokens = torch.randint(self.vocab_size, (batch_size, block_size), device=device)
        canvas_ids = torch.where(corrupt, random_tokens, canvas_target)

        cache_mask = torch.arange(seq_len, device=device) < encoder_len[:, None]
        canvas_mask = torch.ones(batch_size, block_size, dtype=torch.bool, device=device)
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": canvas_ids,
            "decoder_attention_mask": torch.cat([cache_mask, canvas_mask], dim=1),
            "decoder_position_ids": encoder_len[:, None] + offsets,
        }

        # Two-pass self-conditioning: a first no-grad pass produces the logits the second pass conditions on,
        # gated per example with probability p
        with torch.no_grad():
            model_kwargs["self_conditioning_logits"] = model(**model_kwargs).logits
        model_kwargs["self_conditioning_mask"] = torch.rand(batch_size, device=device) < self.self_conditioning_p

        outputs = model(**model_kwargs)

        # Flat cross-entropy over all canvas positions, corrupted and uncorrupted (no 1/t weighting: the uniform
        # corruption kernel has a flat ELBO). For truncated turns the EOS fill is not a real terminator, so it is
        # dropped from the loss while the surviving response tokens are still supervised.
        diffusion_target = canvas_target.masked_fill(truncated[:, None] & ~in_response, -100)
        diffusion_logits = outputs.logits
        if self.model_prediction_type == "mean_loo":
            # The plug-in softmax parameterizes the leave-one-out posterior, not the plain denoiser
            # (https://huggingface.co/papers/2605.22765). Convert it to the denoiser before the cross-entropy by
            # adding log(1 + K * alpha_t / (1 - alpha_t)) to the logit of the observed (corrupted) token, where the
            # uniform forward keeps the clean token with probability alpha_t = 1 - t. Upcast like the reference
            # implementation: the correction reaches ~log(K) nats and would lose precision in bf16.
            alpha = (1 - t).float()  # [batch_size, 1], keep probability of the uniform forward
            K = diffusion_logits.shape[-1]
            bump = torch.log1p(K * alpha / (1 - alpha))
            diffusion_logits = diffusion_logits.float().scatter_add(
                2, canvas_ids.unsqueeze(-1), bump[:, :, None].expand(-1, block_size, 1)
            )
        if (diffusion_target != -100).any():
            diffusion_loss = F.cross_entropy(
                diffusion_logits.flatten(0, 1), diffusion_target.flatten(), ignore_index=-100
            )
        else:
            diffusion_loss = outputs.logits.sum() * 0.0

        # Autoregressive co-loss on the encoder, over all valid next-token pairs
        lm_head = self.model.lm_head
        hidden_states = outputs.encoder_last_hidden_state.to(lm_head.weight.dtype)
        with maybe_gather_lm_head_ctx(lm_head.weight, lm_head.bias):
            encoder_logits = hidden_states @ lm_head.weight.t()
            if lm_head.bias is not None:
                encoder_logits = encoder_logits + lm_head.bias
        encoder_logits = encoder_logits.float()
        cap = self.final_logit_softcapping
        encoder_logits = torch.tanh(encoder_logits / cap) * cap
        ar_mask = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
        ar_targets = torch.where(ar_mask, input_ids[:, 1:], -100)
        ar_loss = F.cross_entropy(encoder_logits[:, :-1].flatten(0, 1), ar_targets.flatten(), ignore_index=-100)

        loss = diffusion_loss + ar_loss
        return (loss, outputs) if return_outputs else loss


def main(script_args, training_args, model_args):
    # [FUNCTION DOC]
    # Loads the model, configures trainability, prepares GSM8K and trains.
    # Behavior: loads DiffusionGemmaForBlockDiffusion (revision, dtype, attention implementation) and the tokenizer. With
    # use_peft, builds a LoraConfig (rank, alpha, dropout from model_args) targeting by regex the q/k/v/o attention
    # projections and gate/up/down MLP projections of encoder language-model and decoder layers only; otherwise freezes
    # every parameter whose name contains .router. for full fine-tuning. Loads the train split, converts each row to a
    # user question / assistant answer messages list (other columns removed), forces assistant_only_loss True (which also
    # makes SFTTrainer use TRL's DiffusionGemma training chat template), builds DiffusionGemmaSFTTrainer, trains, saves,
    # optionally pushes, calls end.
    # Inputs: script_args, training_args (DiffusionGemmaSFTConfig), model_args (DiffusionGemmaModelConfig).
    # Outputs: none. Exceptions: propagated. Side effects: downloads, multi-GPU training, file writes, optional upload.
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        dtype=model_args.dtype,
        attn_implementation=model_args.attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)

    if model_args.use_peft:
        # LoRA on the attention and dense-MLP linears of the encoder/decoder layers only; the MoE experts, the
        # router, the self-conditioning block, and the vision tower stay frozen. The encoder and decoder share
        # tied base weights but get independent adapters, so keep the adapters unmerged (`merge_and_unload` would
        # fold both deltas into the shared tensors).
        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=r"model\.(encoder\.language_model|decoder)\.layers\.\d+"
            r"\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)",
        )
    else:
        peft_config = None
        # The reference recipe keeps the MoE router frozen during full fine-tuning
        for name, param in model.named_parameters():
            if ".router." in name:
                param.requires_grad_(False)

    dataset = load_dataset(script_args.dataset_name, script_args.dataset_config, split=script_args.dataset_train_split)

    def to_messages(example):
        # [FUNCTION DOC]
        # HELPER: Map function turning a GSM8K row into a two-message conversation (user question, assistant answer).
        # Inputs: example (dict with question, answer). Outputs: dict with messages. Exceptions: KeyError if missing.
        # Side effects: none.
        return {
            "messages": [
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": example["answer"]},
            ]
        }

    dataset = dataset.map(to_messages, remove_columns=dataset.column_names)

    # The supervised canvas is derived from the labels, so the loss must be restricted to the assistant response.
    # This also makes SFTTrainer swap in TRL's DiffusionGemma training chat template.
    training_args.assistant_only_loss = True

    trainer = DiffusionGemmaSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.end()


def make_parser():
    # [FUNCTION DOC]
    # HELPER: Returns a TrlParser over DiffusionGemmaScriptArguments, DiffusionGemmaSFTConfig and
    # DiffusionGemmaModelConfig. Inputs: none. Outputs: TrlParser. Exceptions: none. Side effects: none.
    dataclass_types = (DiffusionGemmaScriptArguments, DiffusionGemmaSFTConfig, DiffusionGemmaModelConfig)
    return TrlParser(dataclass_types)


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
