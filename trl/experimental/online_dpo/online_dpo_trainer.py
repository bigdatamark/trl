# [MODULE DOC]
# Experimental Online DPO trainer (paper 2402.04792, "Direct Language Model Alignment from Online AI Feedback").
# Contents: the RewardFunc type alias (model id string, PreTrainedModel, or callable returning a list of floats/None)
# and OnlineDPOTrainer, which for each batch samples two completions per prompt from the current policy (transformers
# generate or vLLM in server/colocate mode), scores them with reward functions, picks the higher-scored one as chosen,
# and applies a sigmoid (DPO) or IPO loss against a reference model.
# Import-time side effects: conditionally imports nested_gather (transformers >= 5.2.0), bitsandbytes, peft, SageMaker MP
# version flag (IS_SAGEMAKER_MP_POST_1_10), and vLLM classes when available; creates an accelerate logger.

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

import os
import re
import textwrap
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import transformers
from accelerate.logging import get_logger
from accelerate.utils import broadcast_object_list, gather_object, is_peft_model, set_seed
from datasets import Dataset
from packaging.version import Version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollator,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    is_bitsandbytes_available,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import OptimizerNames
from transformers.utils import is_peft_available, is_sagemaker_mp_enabled

from ...data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
    prepare_multimodal_messages,
)
from ...extras.profiling import profiling_context
from ...generation.vllm_client import VLLMClient
from ...import_utils import is_vllm_available
from ...models.utils import prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation
from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import disable_dropout_in_model, ensure_master_addr_port, get_callable_name, get_config_model_id
from ..utils import DPODataCollatorWithPadding, create_reference_model, empty_cache, prepare_peft_model, truncate_right
from .online_dpo_config import OnlineDPOConfig


if Version(transformers.__version__) >= Version("5.2.0"):
    from transformers.trainer_pt_utils import nested_gather


if is_bitsandbytes_available():
    import bitsandbytes as bnb


if is_peft_available():
    from peft import PeftConfig


if is_sagemaker_mp_enabled():
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = Version(SMP_VERSION) >= Version("1.10")

else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams


logger = get_logger(__name__)


# A reward function can be a string, interpreted as a model ID and loaded as a pretrained model, a pretrained model, or
# a callable that returns a list of floats (the rewards). The callable receives prompts, completions, and additional
# arguments from the trainer (refer to the trainer's source for details). To ensure forward compatibility, it should
# accept **kwargs.
RewardFunc = str | PreTrainedModel | Callable[..., list[float | None]]


class OnlineDPOTrainer(_BaseTrainer):
    # [CLASS DOC]
    # Trainer subclass (via _BaseTrainer) implementing online DPO with on-the-fly generation and reward scoring.
    # Key state: ref_model (copy of the policy, or None with PEFT where the adapter is disabled instead), reward_funcs and
    # reward_processing_classes, reward_weights tensor, reward_func_names, stats (dict of metric lists accumulated per step
    # and averaged at log time), generation settings (num_generations fixed to 2, temperature, top_p/top_k/min_p,
    # repetition_penalty), eos_token_ids, vision token ids/strings for VLMs, generation_config (SamplingParams for vLLM or
    # GenerationConfig otherwise), vllm_client or llm, _last_loaded_step, and _beta (float or per-epoch list).
    # Flow: __init__ prepares models/tokenizers/vLLM; training_step generates (_generate or _generate_vllm), computes
    # policy/reference log-probs (_forward), rewards (_calculate_rewards_from_functions), the loss and stats, and backprops;
    # _move_model_to_vllm and helpers sync weights to vLLM; _maybe_log_save_evaluate logs averaged stats;
    # _save_checkpoint also writes a model card. Model card metadata in _tag_names, _name and _paper.
    r"""
    Initialize OnlineDPOTrainer.

    Args:
        model (`str | nn.Module | PreTrainedModel`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keyword arguments in
              `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        ref_model ([`~transformers.PreTrainedModel`] or `torch.nn.Module` or `None`):
            The reference model to use for training. If None is specified, the reference model will be created from the
            model.
        reward_funcs (`RewardFunc | list[RewardFunc]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function: Can be a string (path to model), a [`~transformers.PreTrainedModel`], or a
              custom callable function.
            - A list of reward functions: Must all be of compatible types.
        args ([`experimental.online_dpo.OnlineDPOConfig`]):
            The online DPO config arguments to use for training.
        data_collator ([`~transformers.DataCollator`]):
            The data collator to use for training. If None is specified, the default data collator
            ([`experimental.utils.DPODataCollatorWithPadding`]) will be used which will pad the sequences to the
            maximum length of the sequences in the batch, given a dataset of paired sequences.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            The dataset to use for training.
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Dataset | IterableDataset]`):
            The dataset to use for evaluation.
        processing_class ([`~transformers.PreTrainedTokenizerBase`] or [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. If provided, will be used to automatically process the inputs
            for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
            reuse the fine-tuned model.
        reward_processing_classes ([`~transformers.PreTrainedTokenizerBase`] or `list[PreTrainedTokenizerBase]`, *optional*):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.

            If set to `None`, the tokenizer for each model-based reward function is automatically loaded using
            [`~transformers.AutoTokenizer.from_pretrained`].
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return a dictionary string to
            metric values.
        callbacks (`list[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
    """

    _tag_names = ["trl", "online-dpo"]
    _name = "Online DPO"
    _paper = {
        "title": "Direct Language Model Alignment from Online AI Feedback",
        "id": "2402.04792",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{guo2024direct,
                title        = {{Direct Language Model Alignment from Online AI Feedback}},
                author       = {Shangmin Guo and Biao Zhang and Tianlin Liu and Tianqi Liu and Misha Khalman and Felipe Llinares and Alexandre Ram{\'{e}} and Thomas Mesnard and Yao Zhao and Bilal Piot and Johan Ferret and Mathieu Blondel},
                year         = 2024,
                eprint       = {arXiv:2402.04792}
            }"""),
    }

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str,
        ref_model: PreTrainedModel | nn.Module | None = None,
        reward_funcs: RewardFunc | list[RewardFunc] | None = None,
        args: OnlineDPOConfig | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        peft_config: "PeftConfig | None" = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Builds the trainer: validates inputs, loads models and reward functions, configures generation and vLLM.
        # Steps:
        # 1. Requires train_dataset, reward_funcs and processing_class; rejects ref_model identical to model.
        # 2. Wraps reward_funcs in a list; string entries are loaded as single-label AutoModelForSequenceClassification with
        # model_init_kwargs (trust_remote_code defaulted from args), remembering the revision; records names (model id tail for
        # modules, callable name otherwise).
        # 3. Normalizes reward_processing_classes to a list (length must match); for PreTrainedModel rewards, loads the
        # tokenizer from the model path if missing, sets pad to eos if absent, and mirrors the pad id onto the reward config.
        # 4. reward_weights from args (length must match) else all ones.
        # 5. Seeds, then loads the policy from a string with dtype resolution ("auto"/None/torch dtype/string name) and
        # device_map default "auto"; rejects model_init_kwargs for an already-instantiated model. Records is_encoder_decoder and
        # is_vision_model (model type in the image-text-to-text mapping).
        # 6. Validates peft_config type and wraps with prepare_peft_model when given or already PEFT; enables gradient
        # checkpointing; optionally disables dropout on model and ref model.
        # 7. Reference model: copied from the policy without PEFT, None with PEFT, or the provided one set to eval. Reward
        # models set to eval.
        # 8. Initializes stats keys (kl, entropy, non_score_reward, chosen/rejected rewards, accuracies, margins, logps,
        # contain_eos_token, beta, plus rlhf_reward, scores_margin, scores).
        # 9. Extracts the tokenizer from a processor, sets pad to eos if missing, mirrors pad id onto model config and
        # generation config, builds eos_token_ids (model's generation eos ids plus tokenizer eos first if absent), reads vision
        # token ids and decodes the image token string.
        # 10. Default collator DPODataCollatorWithPadding; for transformers < 5.0 sets use_reentrant False by default.
        # 11. Calls the parent constructor, adds model tags, stores _beta.
        # 12. vLLM: server mode creates a VLLMClient on the main process (base URL or host/port) and initializes the NCCL
        # communicator on the current accelerator device; colocate mode detects 4-bit bitsandbytes (8-bit raises), sets
        # RANK/LOCAL_RANK/WORLD_SIZE and master addr/port env vars, builds an LLM (external_launcher, max_model_len max_length
        # plus max_new_tokens, max_num_batched_tokens 4096, per-TP-group seed) and optionally puts it to sleep. Builds
        # SamplingParams (n 2, detokenize off, generation_kwargs merged, structured outputs regex taking precedence over a
        # structured_outputs entry, dict converted to StructuredOutputsParams); waits for all processes.
        # Without vLLM, builds a GenerationConfig (sampling, eos ids, cache unless gradient checkpointing, min_p if set,
        # generation_kwargs merged, None values dropped) and keeps generation_kwargs.
        # 13. Prepares ref model and reward models with DeepSpeed, FSDP or accelerate (eval mode).
        # Inputs: see the class docstring. Outputs: none.
        # Exceptions: ValueError (missing/invalid arguments, mismatched lengths, bad dtype, 8-bit with vLLM, bad vllm_mode),
        # ImportError (peft or vLLM missing), TypeError (bad peft_config or processing_class).
        # Side effects: model downloads, env var mutation, vLLM engine/server connection, mutation of args and model configs.
        # KNOWN ISSUE: args.model_init_kwargs and args.reward_weights are read before the "args must be provided" check, so a
        # None args raises AttributeError instead of the intended ValueError. The model_init_kwargs dict from args is mutated
        # in place (trust_remote_code, dtype, device_map).
        if train_dataset is None:
            raise ValueError("`train_dataset` is required")

        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, either omit the `ref_model` argument or pass `None`."
            )

        self.ref_model = ref_model

        if reward_funcs is None:
            raise ValueError("`reward_funcs` must be provided.")

        # Handle reward_funcs
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []

        # Process reward functions (convert strings to models, collect names)
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)
        reward_model_revisions = [None] * len(reward_funcs)
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                # Load model from string path
                reward_model_revisions[i] = model_init_kwargs.get("revision")
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            if isinstance(reward_funcs[i], nn.Module):
                self.reward_func_names.append(get_config_model_id(reward_funcs[i].config).split("/")[-1])
            else:
                self.reward_func_names.append(get_callable_name(reward_funcs[i]))
        self.reward_funcs = reward_funcs

        # Handle reward processing classes for reward_funcs
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        self.reward_processing_classes = []
        for i, (reward_processing_class_i, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs, strict=True)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class_i is None:
                    reward_processing_class_i = AutoTokenizer.from_pretrained(
                        reward_func.config._name_or_path,
                        revision=reward_model_revisions[i],
                        trust_remote_code=args.trust_remote_code,
                    )
                if reward_processing_class_i.pad_token_id is None:
                    reward_processing_class_i.pad_token = reward_processing_class_i.eos_token
                # Set pad token ID on reward model config
                reward_func.config.get_text_config().pad_token_id = reward_processing_class_i.pad_token_id
            self.reward_processing_classes.append(reward_processing_class_i)

        # Handle reward_weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(self.reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(self.reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(self.reward_funcs), dtype=torch.float32)

        if args is None:
            raise ValueError("`args` must be provided.")

        # Check that the processing_class is provided
        if processing_class is None:
            raise ValueError("`processing_class` must be provided.")

        # PEFT initializes the adapter weights randomly, so set_seed must be done before creating the model to ensure
        # reproducibility.
        set_seed(args.seed)
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model

            # Handle dtype in model_init_kwargs
            dtype = model_init_kwargs.get("dtype", "auto")
            if isinstance(dtype, torch.dtype) or dtype == "auto" or dtype is None:
                pass
            elif isinstance(dtype, str):
                dtype = getattr(torch, dtype)
                model_init_kwargs["dtype"] = dtype
            else:
                raise ValueError(
                    "Invalid `dtype` passed to `OnlineDPOConfig`. Expected either 'auto' or a string "
                    f"representing a `torch.dtype` (e.g., 'float32'), but got {dtype}."
                )
            model_init_kwargs["device_map"] = model_init_kwargs.get("device_map", "auto")
            model_init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)

            model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `OnlineDPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.is_vision_model = model.config.model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.keys()

        # PEFT
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
        if peft_config is not None or is_peft_model(model):
            model = prepare_peft_model(model, peft_config, args)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Disable dropout in the model and reference model
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Handle the ref_model
        # Usually, the user wants the ref model to be the initial version of the model. When using PEFT, it's easy to
        # get the ref model, as it's just the model with a disabled adapter. When not using PEFT, we need to create
        # the ref model from the model by copying it and disable the gradients and set it in evaluation mode.
        if ref_model is None:  # No ref model provided, the most common case
            if peft_config is None:
                self.ref_model = create_reference_model(model)  # copy, disable gradients, set eval mode
            else:
                self.ref_model = None  # we don't need a ref model here, we can just disable the adapter.
        else:  # rare case, the user provided a ref model
            self.ref_model = ref_model
            self.ref_model.eval()

        # Disable the gradient and set the reward model in eval mode
        if reward_funcs is not None:
            for reward_func in reward_funcs:
                if isinstance(reward_func, PreTrainedModel):
                    reward_func.eval()

        self.max_length = args.max_length

        self.stats = {
            "objective/kl": [],
            "objective/entropy": [],
            "objective/non_score_reward": [],
            "rewards/chosen": [],
            "rewards/rejected": [],
            "rewards/accuracies": [],
            "rewards/margins": [],
            "logps/chosen": [],
            "logps/rejected": [],
            "val/contain_eos_token": [],
            "beta": [],
        }
        if self.reward_funcs is not None:
            self.stats["objective/rlhf_reward"] = []
            self.stats["objective/scores_margin"] = []
            self.stats["objective/scores"] = []

        # Store generation parameters for later use
        self.use_vllm = args.use_vllm
        self.num_generations = 2  # Generate 2 completions per prompt for Online DPO
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.vllm_mode = args.vllm_mode if args.use_vllm else None
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
        self.vllm_model_impl = args.vllm_model_impl

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            self._tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            self._tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # The model must agree with the tokenizer on the pad token from construction, so mirror it onto the model
        # configs.
        model.config.get_text_config().pad_token_id = self._tokenizer.pad_token_id
        model.generation_config.pad_token_id = self._tokenizer.pad_token_id

        # Every eos id the model declares (e.g. an end-of-turn token) ends a completion, not only the tokenizer's
        eos_token_ids = model.generation_config.eos_token_id
        if eos_token_ids is None:
            eos_token_ids = []
        elif isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        if self._tokenizer.eos_token_id not in eos_token_ids:
            eos_token_ids = [self._tokenizer.eos_token_id, *eos_token_ids]
        self.eos_token_ids = eos_token_ids

        # Vision tokens for VLM support
        self.image_token_id = getattr(processing_class, "image_token_id", None)
        self.vision_start_token_id = getattr(processing_class, "vision_start_token_id", None)
        self.vision_end_token_id = getattr(processing_class, "vision_end_token_id", None)
        # Get the image token string for token collapsing
        self.image_token = None
        if self.image_token_id is not None:
            self.image_token = self._tokenizer.decode([self.image_token_id])

        # Define the collator if not provided
        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(pad_token_id=self._tokenizer.pad_token_id)

        # Transformers explicitly set use_reentrant=True in the past to silence a PyTorch warning, but the default was
        # never updated once PyTorch switched to recommending use_reentrant=False. Until that change lands upstream
        # (see https://github.com/huggingface/transformers/pull/43203) and is released (most likely in 5.0.0), we
        # default to the recommended non-reentrant behavior here, while preserving any user-provided value.
        if args.gradient_checkpointing and Version(transformers.__version__) < Version("5.0.0"):
            args.gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
            args.gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        self._beta = args.beta

        # Set up generation configuration and vLLM after super().__init__
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install trl[vllm]` to use it."
                )

            if self.vllm_mode == "server":
                if self.accelerator.is_main_process:
                    if args.vllm_server_base_url is not None:
                        base_url = args.vllm_server_base_url
                    else:
                        base_url = f"http://{args.vllm_server_host}:{args.vllm_server_port}"
                    self.vllm_client = VLLMClient(
                        base_url=base_url, group_port=args.vllm_group_port, connection_timeout=args.vllm_server_timeout
                    )

                    # Determine device type (supports cuda, xpu, etc.)
                    if Version(torch.__version__) >= Version("2.6.0"):
                        accelerator_type = torch.accelerator.current_accelerator().type
                    else:  # `torch.accelerator` was introduced in torch 2.6
                        accelerator_type = "cuda"
                    current_device = getattr(torch, accelerator_type).current_device()
                    self.vllm_client.init_communicator(device=current_device)
                else:
                    self.vllm_client = None
            elif self.vllm_mode == "colocate":
                # vLLM dynamically adjusts the size of the key-value cache based on available GPU memory at instantiation.
                # A larger cache size improves speed, so we would expect gpu_memory_utilization=1.
                # However, at this stage, the optimizer's weights are not yet loaded onto the GPU; they will be loaded
                # after the first optimizer step and remain in GPU memory throughout training. So we must reserve enough
                # space for them.
                # Configure vLLM parameters
                vllm_quantization = None
                if is_bitsandbytes_available():
                    for _, module in model.named_modules():
                        if isinstance(module, bnb.nn.Linear4bit):
                            vllm_quantization = "bitsandbytes"
                            break
                        elif isinstance(module, bnb.nn.Linear8bitLt):
                            raise ValueError("vLLM does not support in-flight 8-bit quantization.")
                vllm_kwargs = {
                    "model": model.name_or_path,
                    "tensor_parallel_size": self.vllm_tensor_parallel_size,
                    "gpu_memory_utilization": self.vllm_gpu_memory_utilization,
                    "model_impl": self.vllm_model_impl,
                    "max_num_seqs": self.args.per_device_train_batch_size * self.vllm_tensor_parallel_size,
                    "max_model_len": args.max_length + args.max_new_tokens,  # max_length includes prompt + completion
                    "distributed_executor_backend": "external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    "seed": self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    # Latest vLLM v1 memory profiler is misled by the high default value (i.e., 32768)
                    "max_num_batched_tokens": 4096,
                    "enable_sleep_mode": self.args.vllm_enable_sleep_mode,
                    "quantization": vllm_quantization,
                }

                # vLLM requires the environment variables to be set for distributed training.
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                # Ensure distributed rendezvous variables are set without colliding across concurrent runs
                ensure_master_addr_port()

                self.llm = LLM(**vllm_kwargs)
                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=2)
            else:
                raise ValueError(f"vllm_mode must be either 'server' or 'colocate', got '{self.vllm_mode}'.")
            # vLLM specific sampling arguments
            self.structured_outputs_regex = args.vllm_structured_outputs_regex
            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

            # Set up vLLM generation config
            generation_kwargs = {
                "n": 2,  # 2 generations per prompt for Online DPO
                "repetition_penalty": self.repetition_penalty,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": 0.0 if self.min_p is None else self.min_p,
                "max_tokens": args.max_new_tokens,
                "detokenize": False,  # to avoid vllm to decode (we don't need it)
            }
            if args.generation_kwargs is not None:
                generation_kwargs.update(args.generation_kwargs)
            if self.structured_outputs_regex is not None:
                if generation_kwargs.get("structured_outputs") is not None:
                    logger.warning(
                        "Both `vllm_structured_outputs_regex` and `generation_kwargs['structured_outputs']` are set; "
                        "`vllm_structured_outputs_regex` takes precedence."
                    )
                generation_kwargs["structured_outputs"] = StructuredOutputsParams(regex=self.structured_outputs_regex)
            elif isinstance(structured_outputs_kwargs := generation_kwargs.get("structured_outputs"), dict):
                generation_kwargs["structured_outputs"] = StructuredOutputsParams(**structured_outputs_kwargs)
            self.generation_config = SamplingParams(**generation_kwargs)

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            # Set up transformers generation config
            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": True,
                "pad_token_id": self._tokenizer.pad_token_id,
                "bos_token_id": self._tokenizer.bos_token_id,
                "eos_token_id": self.eos_token_ids,
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty,
                "use_cache": True if not self.args.gradient_checkpointing else False,
            }
            # Add min_p if supported
            if self.min_p is not None:
                generation_kwargs["min_p"] = self.min_p
            if args.generation_kwargs is not None:
                generation_kwargs.update(args.generation_kwargs)
            # Remove None values
            generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
            self.generation_config = GenerationConfig(**generation_kwargs)
            # Keep training-specific generation kwargs to overwrite model's original generation config
            self.generation_kwargs = generation_kwargs

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        if self.reward_funcs is not None:
            for i, reward_func in enumerate(self.reward_funcs):
                if isinstance(reward_func, PreTrainedModel):
                    if self.is_deepspeed_enabled:
                        self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                    else:
                        # set device placement to True to make `prepare_model` move `reward_func` to device when using fsdp
                        self.reward_funcs[i] = self.accelerator.prepare_model(
                            reward_func, evaluation_mode=True, device_placement=True
                        )

    @property
    def beta(self):
        # [FUNCTION DOC]
        # Current beta value. If _beta is a list, returns the entry indexed by the current epoch, or the last entry once the
        # epoch exceeds the list length; otherwise returns the float.
        # Inputs: none. Outputs: float. Exceptions: none expected. Side effects: none.
        # KNOWN ISSUE: state.epoch is a float (fractional epochs), so indexing the list with it raises TypeError; it should be
        # converted to an integer epoch.
        if isinstance(self._beta, list):
            epoch = self.state.epoch
            return self._beta[epoch] if epoch < len(self._beta) else self._beta[-1]
        else:
            return self._beta

    @staticmethod
    def tokenize_row(feature, is_encoder_decoder: bool, tokenizer: PreTrainedTokenizerBase) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Static helper tokenizing the prompt of one DPO-style row.
        # Behavior: decoder-only: tokenizes without special tokens and prepends BOS (to ids and mask) if the tokenizer has one
        # and the prompt does not already start with it; encoder-decoder: tokenizes with special tokens. Prefixes every key with
        # "prompt_".
        # Inputs: feature (dict with "prompt"), is_encoder_decoder, tokenizer.
        # Outputs: dict such as prompt_input_ids and prompt_attention_mask.
        # Exceptions: KeyError if "prompt" missing. Side effects: none.
        """Tokenize a single row from a DPO specific dataset."""
        if not is_encoder_decoder:
            batch = tokenizer(feature["prompt"], add_special_tokens=False)
            # Add BOS token to head of prompt. Avoid adding if it's already there
            if tokenizer.bos_token_id is not None:
                prompt_len_input_ids = len(batch["input_ids"])
                if prompt_len_input_ids == 0 or tokenizer.bos_token_id != batch["input_ids"][0]:
                    batch["input_ids"] = [tokenizer.bos_token_id] + batch["input_ids"]
                    batch["attention_mask"] = [1] + batch["attention_mask"]
        else:
            batch = tokenizer(feature["prompt"], add_special_tokens=True)
        batch = {f"prompt_{key}": value for key, value in batch.items()}
        return batch

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: OnlineDPOConfig) -> PreTrainedModel:
        # [FUNCTION DOC]
        # HELPER: Turns on gradient checkpointing.
        # Behavior: disables use_cache on the config, enables checkpointing on the base model for PEFT models or the model
        # itself otherwise, then enables input require-grads so checkpointing works with frozen embeddings.
        # Inputs: model, args (unused). Outputs: the same model. Exceptions: none. Side effects: mutates the model.
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        model.enable_input_require_grads()
        return model

    def _generate_vllm(self, prompts, images=None):
        # [FUNCTION DOC]
        # HELPER: Generates two completions per prompt with vLLM and returns padded tensors.
        # Behavior: dispatches to server or colocate helper (both return lists in block layout: all prompts' first completion,
        # then all second completions). Left-pads prompt ids to the longest prompt with pad id and builds a left-padded mask;
        # completion mask is ones over the generated length then zeros up to max_tokens; appends the first eos id to completions
        # that did not end with an eos and are shorter than max_tokens; right-pads completions to max_tokens.
        # Inputs: prompts (list of strings or conversations), images (optional list).
        # Outputs: prompt_ids, prompt_mask, completion_ids, completion_mask tensors on the accelerator device;
        # shapes (2B, P) and (2B, max_tokens).
        # Exceptions: IndexError if a completion is empty (its last token is read). Side effects: those of the helpers.
        # KNOWN ISSUE: the completion mask is computed before the eos is appended, so the appended eos token is masked out.
        pad_token_id = self._tokenizer.pad_token_id

        # Generate completion_ids and prompt_ids based on mode
        if self.vllm_mode == "server":
            completion_ids, prompt_ids = self._generate_vllm_server(prompts, images)
        elif self.vllm_mode == "colocate":
            completion_ids, prompt_ids = self._generate_vllm_colocate(prompts, images)

        # Shared padding, masking, and tensor conversion logic
        max_prompt_length = max(len(ids) for ids in prompt_ids)
        prompt_mask = [[0] * (max_prompt_length - len(ids)) + [1] * len(ids) for ids in prompt_ids]
        prompt_ids = [[pad_token_id] * (max_prompt_length - len(ids)) + ids for ids in prompt_ids]
        max_tokens = self.generation_config.max_tokens
        completion_mask = [[1] * len(ids) + [0] * (max_tokens - len(ids)) for ids in completion_ids]
        completion_ids = [
            ids + [self.eos_token_ids[0]] if ids[-1] not in self.eos_token_ids and len(ids) < max_tokens else ids
            for ids in completion_ids
        ]
        completion_ids = [ids + [pad_token_id] * (max_tokens - len(ids)) for ids in completion_ids]

        # Convert to tensors
        prompt_ids = torch.tensor(prompt_ids, device=self.accelerator.device)
        prompt_mask = torch.tensor(prompt_mask, device=self.accelerator.device)
        completion_ids = torch.tensor(completion_ids, device=self.accelerator.device)
        completion_mask = torch.tensor(completion_mask, device=self.accelerator.device)

        return prompt_ids, prompt_mask, completion_ids, completion_mask

    def _generate_vllm_server(self, prompts, images=None):
        # [FUNCTION DOC]
        # HELPER: Generates completions through a remote vLLM server.
        # Behavior: pushes weights to the server when global_step changed since the last load (or on first call); applies the
        # chat template for conversational prompts; gathers all prompts (and, with images, multimodal messages with the image
        # inlined) across processes; on the main process calls the client's chat or generate with n 2 and sampling settings
        # (top_k None mapped to -1, min_p None to 0.0, structured regex, generation_kwargs); others hold placeholders.
        # Broadcasts results, keeps this process's slice (2 per local prompt), reorders from interleaved to block layout, and
        # tokenizes prompts locally (left padded, no special tokens) duplicated twice.
        # Inputs: prompts, images (optional).
        # Outputs: (completion_ids list of lists, prompt_ids list of lists).
        # Exceptions: client/network errors. Side effects: weight sync, cross-process communication, HTTP requests.
        """Generate completions using vLLM server mode"""
        has_images = images is not None

        # Update vLLM server weights if needed
        if hasattr(self, "_last_loaded_step") and self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step
        elif not hasattr(self, "_last_loaded_step"):
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Apply chat template if conversational
        if is_conversational({"prompt": prompts[0]}):
            prompts_text = [apply_chat_template({"prompt": p}, self.processing_class)["prompt"] for p in prompts]
        else:
            prompts_text = prompts
        # Gather all prompts to main process
        all_prompts = gather_object(prompts_text)
        if has_images:
            # The server can't take images alongside text prompts, so multimodal prompts are sent as messages, with
            # the images inlined in place of their placeholders.
            messages = [
                prepare_multimodal_messages(prompt, images=[image] if image is not None else None)
                for prompt, image in zip(prompts, images, strict=True)
            ]
            all_messages = gather_object(messages)

        if self.accelerator.is_main_process:
            sampling_kwargs = {
                "n": self.num_generations,
                "repetition_penalty": self.repetition_penalty,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": -1 if self.top_k is None else self.top_k,
                "min_p": 0.0 if self.min_p is None else self.min_p,
                "max_tokens": self.generation_config.max_tokens,
                "structured_outputs_regex": self.structured_outputs_regex
                if hasattr(self, "structured_outputs_regex")
                else None,
                "generation_kwargs": self.args.generation_kwargs,
            }
            if has_images:
                completion_ids = self.vllm_client.chat(messages=all_messages, **sampling_kwargs)["completion_ids"]
            else:
                completion_ids = self.vllm_client.generate(prompts=all_prompts, **sampling_kwargs)["completion_ids"]
        else:
            completion_ids = [None] * (len(all_prompts) * 2)

        # Broadcast completions to all processes
        completion_ids = broadcast_object_list(completion_ids, from_process=0)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts) * 2,
            (self.accelerator.process_index + 1) * len(prompts) * 2,
        )
        completion_ids = completion_ids[process_slice]
        # Reorder to block layout ([p0c0, p1c0, ..., p0c1, p1c1, ...]) to match the colocate and transformers
        # generation paths, which is what the loss expects (`rewards.split(batch_size)`).
        completion_ids = completion_ids[0::2] + completion_ids[1::2]

        # Create prompt_ids by tokenizing locally, in the same block layout (2 copies per prompt)
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_ids = [prompt_tokens.tolist() for prompt_tokens in prompt_inputs["input_ids"]]
        prompt_ids = prompt_ids + prompt_ids  # 2 copies for 2 completions
        return completion_ids, prompt_ids

    def _generate_vllm_colocate(self, prompts, images=None):
        # [FUNCTION DOC]
        # HELPER: Generates completions with the in-process vLLM engine.
        # Behavior: with sleep mode, empties cache and wakes weights; syncs weights when global_step changed; applies the chat
        # template when conversational; builds vLLM inputs (dict with multi_modal_data image when an image is present); wakes
        # the KV cache if sleeping; generates with generation_config; collects completions in block layout (all first outputs,
        # then all second outputs) and prompt token ids duplicated in the same layout; sleeps the engine again if enabled.
        # Inputs: prompts, images (optional).
        # Outputs: (completion_ids, prompt_ids) as lists of lists of ints.
        # Exceptions: vLLM errors. Side effects: GPU memory changes, weight loading into vLLM.
        """Generate completions using vLLM colocate mode"""
        if self.args.vllm_enable_sleep_mode:
            # wake up colocated vLLM instances if needed
            empty_cache()  # required to avoid OOM in some cases
            self.llm.wake_up(tags=["weights"])

        # Update model weights if needed - only after gradient accumulation completes
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Apply chat template if conversational
        if is_conversational({"prompt": prompts[0]}):
            prompts_text = [apply_chat_template({"prompt": p}, self.processing_class)["prompt"] for p in prompts]
        else:
            prompts_text = prompts

        # Prepare vLLM inputs with images if available
        if images is not None:
            vllm_inputs = []
            for prompt, image in zip(prompts_text, images, strict=True):
                if image is not None:
                    vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
                else:
                    vllm_inputs.append(prompt)
        else:
            vllm_inputs = prompts_text

        if self.args.vllm_enable_sleep_mode:
            self.llm.wake_up(tags=["kv_cache"])

        outputs = self.llm.generate(vllm_inputs, self.generation_config, use_tqdm=False)

        completion_ids = [list(output.outputs[i].token_ids) for i in range(2) for output in outputs]
        prompt_ids = [list(output.prompt_token_ids) for _ in range(2) for output in outputs]
        if self.args.vllm_enable_sleep_mode:
            self.llm.sleep(level=2)

        return completion_ids, prompt_ids

    def _sync_fsdp2_params_to_vllm(self, module: nn.Module):
        # [FUNCTION DOC]
        # HELPER: Pushes FSDP2 parameters to vLLM.
        # Behavior: for each named parameter, strips PEFT prefixes ("base_model.model." and ".base_layer"), skips adapter
        # parameters (PEFT prefix in name) and "original_module" copies, cleans the name (also removing
        # "modules_to_save.default."), moves CPU params to the device, materializes the full tensor, then sends it to the server
        # (main process) or loads it into the colocated model.
        # Inputs: module. Outputs: none. Exceptions: communication errors. Side effects: updates vLLM weights.
        # For FSDP2, module.named_parameters() covers all parameters and yields tied weights once, as vLLM requires
        for name, param in module.named_parameters():
            # When using PEFT, we need to recover the original parameter name
            name = name.removeprefix("base_model.model.").replace(".base_layer", "")
            # Skip PEFT layers: they don’t exist in vLLM, and they are merged already.
            if is_peft_model(module) and module.prefix in name:
                continue
            # When module to save, remove its prefix and discard the original module
            if "original_module" in name:
                continue
            name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

            if param.is_cpu:
                param = param.to(self.accelerator.device)
            param = param.full_tensor()

            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                self.vllm_client.update_named_param(name, param)
            elif self.vllm_mode == "colocate":
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights([(name, param)])

    def _move_model_to_vllm(self):
        # [FUNCTION DOC]
        # HELPER: Syncs current policy weights to vLLM and resets its prefix cache.
        # Behavior: on the server main process wraps the inner sync in the client's weight_update context (one update per
        # model); otherwise calls the inner sync directly (other ranks participate in gathers). Then resets the prefix cache on
        # the server (main process) or colocated engine.
        # Inputs: none. Outputs: none. Side effects: network/GPU weight transfers.
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            # Announce one weight update for the whole model: the server prepares and finalizes it once, rather than
            # once per tensor pushed below. Only the main process holds a client; the other ranks only take part in
            # the parameter gathers below.
            with self.vllm_client.weight_update():
                self._move_model_to_vllm_inner()
        else:
            self._move_model_to_vllm_inner()

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()

    def _move_model_to_vllm_inner(self):
        # [FUNCTION DOC]
        # HELPER: Performs the parameter transfer for _move_model_to_vllm.
        # Behavior: uses DeepSpeed GatheredParameters under ZeRO-3, else a no-op context. PEFT models: gather all parameters,
        # merge adapters, then sync via FSDP1 traversal, FSDP2 helper, or a loop that cleans names, skips adapter and
        # original_module params and sends data; finally unmerge adapters. Non-PEFT: FSDP1/FSDP2 helpers when FSDP is enabled,
        # else per-parameter gather and send (server main process) or load into the colocated model.
        # Inputs: none. Outputs: none.
        # Side effects: temporary adapter merge, weight transfers.
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if is_peft_model(self.model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            # TODO: does this work with FSDP?
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                    fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                    if fsdp_version == 1:
                        self._sync_fsdp1_params_to_vllm(
                            self.model
                        )  # use memory-efficient post-order traversal for FSDP
                    elif fsdp_version == 2:
                        self._sync_fsdp2_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        # Skip PEFT layers: they don’t exist in vLLM, and they are merged already.
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                if fsdp_version == 1:
                    self._sync_fsdp1_params_to_vllm(self.model)  # use memory-efficient post-order traversal for FSDP
                elif fsdp_version == 2:
                    self._sync_fsdp2_params_to_vllm(self.model)
            else:
                for name, param in self.model.named_parameters():
                    name = self._fix_param_name_to_vllm(name)
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

    def _sync_fsdp1_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        # [FUNCTION DOC]
        # HELPER: Memory-efficient post-order traversal syncing FSDP1-wrapped parameters to vLLM.
        # Behavior: recurses into children first (building dotted prefixes), then for FSDP modules summons full params
        # (non-recursive, no writeback) and sends each parameter with cleaned name (removing "_fsdp_wrapped_module."), skipping
        # names already in visited.
        # Inputs: module, prefix (default ""), visited (set shared across recursion; created if None).
        # Outputs: none. Side effects: updates vLLM weights; mutates visited.
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM."""
        # For FSDP1, we need to recurse into children and also use summon_full_params
        if visited is None:
            visited = set()
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp1_params_to_vllm(
                child_module, prefix=child_prefix, visited=visited
            )  # recurse into the child

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    full_name = self._fix_param_name_to_vllm(full_name, extra_prefixes=["_fsdp_wrapped_module."])

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param.data)])

    def _fix_param_name_to_vllm(self, name, extra_prefixes: list[str] | None = None):
        # [FUNCTION DOC]
        # HELPER: Removes wrapper prefixes from a parameter name: always "_checkpoint_wrapped_module.", plus any extra_prefixes,
        # by substring removal anywhere in the name.
        # Inputs: name, extra_prefixes (optional list). Outputs: cleaned name. Exceptions: none. Side effects: none.
        """Clean parameter names for vLLM compatibility"""
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for prefix in prefixes:
            name = name.replace(prefix, "")
        return name

    def process_vision_row(
        self, features: dict[str, list | torch.Tensor], processing_class=None
    ) -> dict[str, list[int]]:
        # [FUNCTION DOC]
        # Processes one vision row with the processor (given or self.processing_class): single image plus prompt text, no
        # special tokens.
        # Outputs: dict with prompt_input_ids, prompt_attention_mask, and pixel_values, pixel_attention_mask, image_sizes
        # when the processor produced them (first element of each).
        # Inputs: features (needs "image" and "prompt"), processing_class (optional).
        # Exceptions: KeyError on missing fields. Side effects: none.
        """
        Process a vision row for VLM models (adapted from DPO trainer)
        """
        processor = processing_class or self.processing_class
        processed_features = processor(images=[features["image"]], text=features["prompt"], add_special_tokens=False)

        prompt_input_ids = processed_features["input_ids"][0]

        # Create the output dict with required fields
        output = {
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": processed_features["attention_mask"][0],
        }

        # Add vision-specific fields
        if "pixel_values" in processed_features:
            output["pixel_values"] = processed_features["pixel_values"][0]
        if "pixel_attention_mask" in processed_features:
            output["pixel_attention_mask"] = processed_features["pixel_attention_mask"][0]
        if "image_sizes" in processed_features:
            output["image_sizes"] = processed_features["image_sizes"][0]

        return output

    def _generate(self, model, prompts, images=None):
        # [FUNCTION DOC]
        # HELPER: Generates two completions per prompt with transformers generate.
        # Behavior: builds input dicts (with image when given) and applies the chat template if conversational. For VLMs with
        # an image token: if the chat template contains the image token, collapses runs of repeated image tokens to one;
        # otherwise removes image token runs (followed by the vision-end token if known). Tokenizes with the processing class
        # (left padding, no special tokens, images as single-image lists), moves to the model device, casts pixel_values to the
        # model dtype (also checking a wrapped module). Repeats prompt ids and mask twice along the batch (block layout) and
        # duplicates vision inputs. Generates under profiling, unwrap_model_for_generation (ZeRO-3 gather setting and
        # generation_kwargs override), no_grad and FSDP full params when FSDP is enabled; applies cache_implementation if set.
        # Slices off prompt tokens and truncates completions after the first eos with truncate_right (pad after).
        # Inputs: model, prompts, images (optional).
        # Outputs: prompt_ids, prompt_mask, completion_ids, completion_mask, vision kwargs dict or None.
        # Exceptions: generation errors. Side effects: none beyond compute.
        """Generate completions using the model"""
        device = next(model.parameters()).device
        pad_token_id = self._tokenizer.pad_token_id

        # Apply chat template and tokenize the input
        inputs = [{"prompt": prompt} for prompt in prompts]

        # Add images if provided (VLM support)
        if images is not None:
            for i, image in enumerate(images):
                inputs[i]["image"] = image

        # Apply chat template to get text prompts
        prompts_text = [maybe_apply_chat_template(x, self.processing_class)["prompt"] for x in inputs]

        # Handle image token collapsing/removal
        # The chat template sometimes inserts a single image token into the prompt text. However, when this text is
        # later tokenized, the single image token string is expanded into multiple image token IDs, depending on the
        # image size. We need to handle this properly.
        if self.image_token is not None and images is not None:
            escaped_img_token = re.escape(self.image_token)
            # Search for the image token in the chat template
            if hasattr(self.processing_class, "chat_template") and self.processing_class.chat_template:
                if re.search(escaped_img_token, self.processing_class.chat_template):
                    # Collapse repeated image tokens back into a single token
                    prompts_text = [
                        re.sub(rf"({escaped_img_token})+", self.image_token, text) for text in prompts_text
                    ]
                else:
                    # If the chat template doesn't use the image token, remove all instances
                    if self.vision_end_token_id is not None:
                        escaped_eoi_token = re.escape(self._tokenizer.decode([self.vision_end_token_id]))
                        prompts_text = [
                            re.sub(rf"({escaped_img_token})+{escaped_eoi_token}", "", text) for text in prompts_text
                        ]
                    else:
                        # If vision_end_token_id is None, just remove the image tokens
                        prompts_text = [re.sub(rf"({escaped_img_token})+", "", text) for text in prompts_text]

        # Prepare kwargs for processing class
        kwargs = {}
        if images is not None:
            kwargs = {"images": [[img] for img in images]}

        # Process inputs using the processing class (handles both VLM and LLM)
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            **kwargs,
        )

        prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
        # Convert vision inputs to model's dtype for proper computation
        if "pixel_values" in prompt_inputs:
            # Handle DataParallel wrapped models
            model_dtype = getattr(model, "dtype", None)
            if model_dtype is None and hasattr(model, "module"):
                model_dtype = model.module.dtype
            if model_dtype is not None:
                prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].to(model_dtype)

        # Sample 2 completions per prompt of size `max_new_tokens` from the model
        prompt_ids = prompt_inputs["input_ids"].repeat(2, 1)
        prompt_mask = prompt_inputs["attention_mask"].repeat(2, 1)

        # Prepare vision inputs if available, concatenated as their number of dimensions varies (5D for Idefics2)
        vision_generation_kwargs = {}
        if self.is_vision_model and images is not None:
            if "pixel_values" in prompt_inputs:
                vision_generation_kwargs["pixel_values"] = torch.cat([prompt_inputs["pixel_values"]] * 2)
            if "pixel_attention_mask" in prompt_inputs:
                vision_generation_kwargs["pixel_attention_mask"] = torch.cat(
                    [prompt_inputs["pixel_attention_mask"]] * 2
                )
            if "image_sizes" in prompt_inputs:
                vision_generation_kwargs["image_sizes"] = torch.cat([prompt_inputs["image_sizes"]] * 2)
            if "image_grid_thw" in prompt_inputs:
                vision_generation_kwargs["image_grid_thw"] = torch.cat([prompt_inputs["image_grid_thw"]] * 2)

        with (
            profiling_context(self, "transformers.generate"),
            unwrap_model_for_generation(
                model,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                generation_kwargs=self.generation_kwargs,  # Override model.generation_config with generation_kwargs to fix transformers#42762
            ) as unwrapped_model,
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
        ):
            # Setup cache implementation if specified
            if self.args.cache_implementation is not None:
                unwrapped_model.generation_config.cache_implementation = self.args.cache_implementation

            # Standard generation
            output = unwrapped_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                generation_config=self.generation_config,
                **vision_generation_kwargs,
            )

        completion_ids = output[:, prompt_ids.size(1) :]
        completion_ids, completion_mask = truncate_right(completion_ids, self.eos_token_ids, pad_token_id)

        return prompt_ids, prompt_mask, completion_ids, completion_mask, vision_generation_kwargs or None

    def _calculate_rewards_from_functions(self, prompts, completions, completion_ids_list, **reward_kwargs):
        # [FUNCTION DOC]
        # HELPER: Computes weighted total rewards per completion.
        # Behavior: adds trainer_state to reward_kwargs. For model rewards: conversational inputs are prompt plus completion
        # messages rendered with the reward tokenizer's chat template, otherwise string concatenation; tokenized right-padded,
        # and the first logit column is the reward (inference mode). For callables: called with prompts, completions,
        # completion_ids and the kwargs; None becomes NaN. Multiplies by reward_weights and sums ignoring NaN.
        # Inputs: prompts, completions, completion_ids_list, reward_kwargs (extra dataset columns).
        # Outputs: tensor of shape (number of completions,).
        # Exceptions: errors from reward functions. Side effects: none.
        """
        Calculate rewards using reward functions
        """
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Add trainer state to reward kwargs for dynamic reward shaping
        reward_kwargs["trainer_state"] = self.state

        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, strict=True)
        ):
            if isinstance(reward_func, nn.Module):  # Model-based reward function
                # Handle conversational vs text input
                if is_conversational({"prompt": prompts[0]}):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions, strict=True)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions, strict=True)]

                # Tokenize and get reward scores
                reward_inputs = reward_processing_class(
                    text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = {k: v.to(device) for k, v in reward_inputs.items()}

                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Custom reward function
                output_reward_func = reward_func(
                    prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                )
                # Convert None values to NaN
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Weight and sum across all reward functions
        if self.reward_weights is not None:
            total_rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        else:
            total_rewards = rewards_per_func.nansum(dim=1)

        return total_rewards

    def _forward(self, model, prompt_ids, prompt_mask, completion_ids, completion_mask, vision_inputs=None):
        # [FUNCTION DOC]
        # HELPER: Per-token log-probabilities of completion tokens under a model.
        # Behavior: without vision inputs, truncates prompt from the left so prompt plus completion fits max_length; concatenates
        # prompt and completion ids and masks; forwards with attention mask and any vision inputs; takes logits from the last
        # prompt position to the second-to-last position (all logits if prompt empty), applies log-softmax and gathers the
        # log-prob of each completion token.
        # Inputs: model, prompt_ids, prompt_mask, completion_ids, completion_mask, vision_inputs (optional dict).
        # Outputs: tensor (batch, completion_length) of log-probs.
        # Exceptions: none specific. Side effects: none.
        # Truncate left to avoid oom, except with images, as it would drop image tokens
        if vision_inputs is None:
            num_tokens_to_truncate = max(prompt_ids.size(1) + completion_ids.size(1) - self.max_length, 0)
            prompt_ids = prompt_ids[:, num_tokens_to_truncate:]
            prompt_mask = prompt_mask[:, num_tokens_to_truncate:]

        # Concat the prompt and completion
        prompt_completion_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        prompt_completion_mask = torch.cat((prompt_mask, completion_mask), dim=1)

        # Prepare model kwargs with vision inputs if available
        model_kwargs = {"attention_mask": prompt_completion_mask}
        if vision_inputs is not None:
            if "pixel_values" in vision_inputs:
                model_kwargs["pixel_values"] = vision_inputs["pixel_values"]
            if "pixel_attention_mask" in vision_inputs:
                model_kwargs["pixel_attention_mask"] = vision_inputs["pixel_attention_mask"]
            if "image_sizes" in vision_inputs:
                model_kwargs["image_sizes"] = vision_inputs["image_sizes"]
            if "image_grid_thw" in vision_inputs:
                model_kwargs["image_grid_thw"] = vision_inputs["image_grid_thw"]

        # Get the logprobs of the completions from the model
        output = model(prompt_completion_ids, **model_kwargs)

        # There is 1 offset, because the model predicts the next token
        prompt_len = prompt_ids.size(1)
        start_idx = prompt_len - 1 if prompt_len > 0 else 0
        # Only slice off the last logit when we have a prompt, otherwise we need all logits
        end_idx = -1 if prompt_len > 0 else None
        logits = output.logits[:, start_idx:end_idx]

        # Take the completion tokens logprob
        logprobs = torch.take_along_dim(logits.log_softmax(dim=-1), completion_ids.unsqueeze(-1), dim=2).squeeze(-1)
        return logprobs

    def training_step(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items_in_batch: int | None = None
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # One online DPO optimization step (overrides Trainer.training_step).
        # Behavior:
        # 1. Sets train mode; for image batches, rewrites string user message contents to an image placeholder plus text and
        # system contents to a text part (mutates prompts in place).
        # 2. Generates two completions per prompt (vLLM or transformers) and flags which contain an eos token.
        # 3. Computes policy log-probs and, without grad, reference log-probs (ref model, or the policy with adapter disabled).
        # 4. Decodes completions (wrapped as assistant messages when conversational), duplicates other list columns for reward
        # kwargs, computes rewards for prompts repeated twice; subtracts missing_eos_penalty for completions without eos.
        # 5. Splits rewards into first/second halves; the higher (ties to first) is chosen. Gathers chosen/rejected log-probs,
        # sums over non-padding tokens, computes logits as policy log-ratio minus reference log-ratio.
        # 6. Loss: sigmoid uses negative log-sigmoid of beta times logits; ipo uses squared (logits minus 1/(2 beta)); mean.
        # 7. Appends gathered metrics to stats: scores_margin, scores, contain_eos_token, chosen/rejected logps, kl (summed
        # per-sequence then averaged), non_score_reward (negative beta times kl), rlhf_reward, entropy (negative summed
        # log-probs), chosen/rejected implicit rewards, margins, accuracies, beta.
        # 8. Optionally empties cache every torch_empty_cache_steps; passes learning_rate for LOMO optimizers; backward.
        # Inputs: model, inputs (batch dict with "prompt" and optional "image"/other columns), num_items_in_batch (unused).
        # Outputs: detached loss divided by gradient_accumulation_steps.
        # Exceptions: NotImplementedError for an unknown loss_type.
        # Side effects: gradients, stats mutation, possible vLLM weight sync.
        # KNOWN ISSUE: the kl, non_score_reward and entropy metrics sum over all completion positions including padding, since
        # they are not multiplied by the completion mask.
        model.train()

        prompts = inputs["prompt"]
        batch_size = len(prompts)

        # Handle images for VLM support
        has_images = "image" in inputs
        images = None
        if has_images:
            images = inputs["image"]
            # Convert conversational prompts to include image tokens
            for prompt in prompts:
                if isinstance(prompt, list):
                    for message in prompt:
                        if not isinstance(message, dict):
                            continue
                        content = message.get("content")
                        role = message.get("role")
                        if isinstance(content, str):
                            if role == "user":
                                message["content"] = [{"type": "image"}, {"type": "text", "text": content}]
                            elif role == "system":
                                message["content"] = [{"type": "text", "text": content}]

        vision_inputs = None
        if self.args.use_vllm:
            prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate_vllm(prompts, images)
        else:
            prompt_ids, prompt_mask, completion_ids, completion_mask, vision_inputs = self._generate(
                model, prompts, images
            )

        contain_eos_token = torch.any(
            torch.isin(completion_ids, torch.tensor(self.eos_token_ids, device=completion_ids.device)), dim=-1
        )

        logprobs = self._forward(model, prompt_ids, prompt_mask, completion_ids, completion_mask, vision_inputs)
        with torch.no_grad():
            if self.ref_model is not None:
                ref_logprobs = self._forward(
                    self.ref_model, prompt_ids, prompt_mask, completion_ids, completion_mask, vision_inputs
                )
            else:  # peft case: we just need to disable the adapter
                with self.model.disable_adapter():
                    ref_logprobs = self._forward(
                        self.model, prompt_ids, prompt_mask, completion_ids, completion_mask, vision_inputs
                    )

        # Decode the completions, and format them if the input is conversational
        device = logprobs.device
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational({"prompt": prompts[0]}):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Get the reward from reward functions
        # First create completion_ids_list for custom reward functions
        completion_ids_list = [completion_ids[i].tolist() for i in range(completion_ids.shape[0])]

        # Extract additional fields from inputs for reward functions
        reward_kwargs = {}
        keys = [key for key in inputs if key not in ["prompt"]]
        for key in keys:
            if isinstance(inputs[key], (list, tuple)):
                # Repeat input fields to match number of completions (2 per prompt)
                reward_kwargs[key] = inputs[key] * 2
            else:
                reward_kwargs[key] = inputs[key]

        # Calculate rewards using reward functions
        rewards = self._calculate_rewards_from_functions(
            prompts=2 * prompts, completions=completions, completion_ids_list=completion_ids_list, **reward_kwargs
        )

        # Apply missing EOS penalty if configured
        if self.args.missing_eos_penalty is not None:
            rewards[~contain_eos_token] -= self.args.missing_eos_penalty

        # Split rewards into chosen/rejected pairs
        first_half, second_half = rewards.split(batch_size)
        mask = first_half >= second_half

        batch_range = torch.arange(batch_size, device=device)
        chosen_indices = batch_range + (~mask * batch_size)
        rejected_indices = batch_range + (mask * batch_size)

        # Build tensor so that the first half is the chosen examples and the second half the rejected examples
        cr_indices = torch.cat((chosen_indices, rejected_indices), dim=0)  # cr = chosen and rejected
        cr_logprobs = logprobs[cr_indices]
        cr_ref_logprobs = ref_logprobs[cr_indices]

        # mask out the padding tokens
        padding_mask = ~completion_mask.bool()
        cr_padding_mask = padding_mask[cr_indices]

        cr_logprobs_sum = (cr_logprobs * ~cr_padding_mask).sum(1)
        cr_ref_logprobs_sum = (cr_ref_logprobs * ~cr_padding_mask).sum(1)

        # Split the chosen and rejected examples
        chosen_logprobs_sum, rejected_logprobs_sum = torch.split(cr_logprobs_sum, batch_size)
        chosen_ref_logprobs_sum, rejected_ref_logprobs_sum = torch.split(cr_ref_logprobs_sum, batch_size)
        pi_logratios = chosen_logprobs_sum - rejected_logprobs_sum
        ref_logratios = chosen_ref_logprobs_sum - rejected_ref_logprobs_sum

        logits = pi_logratios - ref_logratios

        if self.args.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits)
        elif self.args.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.beta)) ** 2
        else:
            raise NotImplementedError(f"invalid loss type {self.args.loss_type}")

        loss = losses.mean()

        # Log everything
        if self.reward_funcs is not None:
            # When using reward_funcs, we have rewards instead of scores
            scores_margin = rewards[chosen_indices] - rewards[rejected_indices]
            self.stats["objective/scores_margin"].append(
                self.accelerator.gather_for_metrics(scores_margin.mean()).mean().item()
            )
            self.stats["objective/scores"].append(self.accelerator.gather_for_metrics(rewards.mean()).mean().item())
        self.stats["val/contain_eos_token"].append(contain_eos_token.float().mean().item())
        self.stats["logps/chosen"].append(self.accelerator.gather_for_metrics(chosen_logprobs_sum).mean().item())
        self.stats["logps/rejected"].append(self.accelerator.gather_for_metrics(rejected_logprobs_sum).mean().item())

        kl = logprobs - ref_logprobs
        mean_kl = kl.sum(1).mean()
        self.stats["objective/kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        non_score_reward = (-self.beta * kl).sum(1)
        mean_non_score_reward = non_score_reward.mean()
        self.stats["objective/non_score_reward"].append(
            self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
        )
        if self.reward_funcs is not None:
            # Calculate RLHF reward by combining rewards with non_score_reward
            rlhf_reward = rewards + non_score_reward
            self.stats["objective/rlhf_reward"].append(self.accelerator.gather_for_metrics(rlhf_reward).mean().item())

        mean_entropy = -logprobs.sum(1).mean()
        self.stats["objective/entropy"].append(self.accelerator.gather_for_metrics(mean_entropy).mean().item())
        chosen_rewards = self.beta * (chosen_logprobs_sum - chosen_ref_logprobs_sum)
        gathered_chosen_rewards = self.accelerator.gather_for_metrics(chosen_rewards)
        self.stats["rewards/chosen"].append(gathered_chosen_rewards.mean().item())
        rejected_rewards = self.beta * (rejected_logprobs_sum - rejected_ref_logprobs_sum)
        gathered_rejected_rewards = self.accelerator.gather_for_metrics(rejected_rewards)
        self.stats["rewards/rejected"].append(gathered_rejected_rewards.mean().item())
        margin = gathered_chosen_rewards - gathered_rejected_rewards
        self.stats["rewards/margins"].append(margin.mean().item())
        accuracy = margin > 0
        self.stats["rewards/accuracies"].append(accuracy.float().mean().item())
        self.stats["beta"].append(self.beta)

        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learning rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss, **kwargs)

        return loss.detach() / self.args.gradient_accumulation_steps

    # Same as Trainer._maybe_log_save_evaluate but log our metrics
    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        # [FUNCTION DOC]
        # Override of Trainer._maybe_log_save_evaluate that also logs averaged stats.
        # Behavior: when logging is due, gathers and averages tr_loss (nested_gather for transformers >= 5.2.0, otherwise the
        # trainer method), zeroes it in place, logs loss per step since last log, grad_norm, learning rate, and the mean of each
        # stats list, then resets stats, updates totals and stores flos. Evaluates when due (updating save flag under the
        # "best" strategy) and saves a checkpoint and fires on_save when due.
        # Inputs: tr_loss, grad_norm, model, trial, epoch (unused), ignore_keys_for_eval, start_time, learning_rate (optional).
        # Outputs: none. Exceptions: ZeroDivisionError if a stats list is empty at log time.
        # Side effects: logging, tr_loss mutation, evaluation, checkpoint writing.
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            if Version(transformers.__version__) >= Version("5.2.0"):
                tr_loss_scalar = nested_gather(tr_loss, self.args.parallel_mode).mean().item()
            else:
                tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            # Add our metrics
            for key, val in self.stats.items():
                logs[key] = sum(val) / len(val)
            self.stats = {key: [] for key in self.stats}  # reset stats

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == "best":
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        # [FUNCTION DOC]
        # Writes a model card (named after hub_model_id's last part or the output_dir name) before delegating to the parent
        # checkpoint save.
        # Inputs: model, trial. Outputs: none. Side effects: writes README/model card and checkpoint files.
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
