# [MODULE DOC]
# Experimental model-merging utilities built on the optional mergekit library.
# Contents: upload_model_to_hf (push a folder to the Hub), MergeConfig (builds mergekit configs for linear, ties,
# dare_ties and slerp merges between a policy checkpoint and a target model), merge_models (runs a merge), and
# MergeModelCallback (a TrainerCallback that merges the saved checkpoint with a target model on save or at train end,
# optionally pushing the result to the Hub).
# Import-time side effects: imports mergekit symbols only if mergekit is available; creates a module logger.

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
import os

import torch
from huggingface_hub import HfApi
from transformers import TrainerCallback

from ..import_utils import is_mergekit_available
from ..trainer.utils import get_config_model_id


if is_mergekit_available():
    from mergekit.config import MergeConfiguration
    from mergekit.merge import MergeOptions, run_merge


# Logger for module-level logging
logger = logging.getLogger(__name__)


def upload_model_to_hf(folder_path: str, repo_id: str):
    # [FUNCTION DOC]
    # HELPER: Uploads a local folder as a model repository on the Hugging Face Hub.
    # Behavior: creates the model repo repo_id via HfApi (not exist_ok, so an existing repo raises), then uploads the whole
    # folder to the returned repo id with the returned repo type.
    # Inputs: folder_path (local directory), repo_id (Hub repo name).
    # Outputs: none.
    # Exceptions: huggingface_hub HTTP errors (auth failure, repo already exists, network errors).
    # Side effects: network calls creating a repo and uploading files.
    api = HfApi()
    # Create the repository if it doesn't exist
    repo = api.create_repo(repo_id, repo_type="model")

    # Upload the folder to the specified repository
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo.repo_id,
        repo_type=repo.repo_type,
    )


class MergeConfig:
    # [CLASS DOC]
    # Holds parameters for merging a policy model with a target model and converts them into a mergekit MergeConfiguration.
    # State: method, policy_model_path and target_model_path (both None until set by the caller or MergeModelCallback),
    # plus method-specific weights/densities/normalize/t_values and dtype set in __init__.
    # Methods: one create_merge_config_* builder per method, and create which dispatches on method.
    # Not a dataclass; no base class. Requires mergekit at construction time.
    # KNOWN ISSUE: the docstring describes dare_ties as "a variant of TIES for domain adaptation"; DARE is actually
    # drop-and-rescale of delta weights, not domain adaptation.
    r"""
    Configuration class for merging two models using `mergekit`.

    This class provides a structured way to configure and generate merge configurations for various merge methods, such
    as `linear`, `ties`, `dare_ties`, and `slerp`.

    Args:
        method (`str`, *optional*, defaults to `"linear"`):
            Merge method to use. Supported methods include:

            - `"linear"`: Linearly combines two models with specified weights.
            - `"ties"`: Combines two models using the TIES method with density parameters.
            - `"dare_ties"`: A variant of TIES for domain adaptation.
            - `"slerp"`: Combines models using spherical linear interpolation.

    Note:

        For more details about the merge methods and how they are implemented, see the [MergeKit GitHub
        repository](https://github.com/arcee-ai/mergekit?tab=readme-ov-file#merge-methods).

    Attributes:
        method (`str`): The merge method to use.
        policy_model_path (`str` or `None`): Path to the policy model.
        target_model_path (`str` or `None`): Path to the target model.
        policy_model_weight (`float`): Weight for the policy model (for `linear` and `ties` methods).
        target_model_weight (`float`): Weight for the target model (for `linear` and `ties` methods).
        policy_model_density (`list[float]`): Density parameters for the policy model (for `ties` and `dare_ties`).
        target_model_density (`list[float]`): Density parameters for the target model (for `ties` and `dare_ties`).
        normalize (`float` or `None`): Normalization factor for the TIES method.
        t_values (`float` or `None`): Interpolation factor for the SLERP method.
        dtype (`str`): Data type to use for merging, e.g., `"float16"`.
    """

    def __init__(self, method: str = "linear"):
        # [FUNCTION DOC]
        # Initializes default merge parameters for the chosen method.
        # Behavior: linear sets both weights to 0.5; ties and dare_ties set policy weight 1.0 with densities [1.0, 0.7, 0.1],
        # target weight 1.0 with density [1.0], normalize 1.0; slerp sets t_values 0.5. All methods use dtype "float16".
        # Both model paths start as None.
        # Inputs: method (str, default "linear").
        # Outputs: none.
        # Exceptions: ImportError if mergekit is not installed; ValueError for an unsupported method.
        # Side effects: sets instance attributes.
        if not is_mergekit_available():
            raise ImportError("MergeConfig requires the `mergekit` extra. To install, run `pip install mergekit`.")
        self.method = method
        self.policy_model_path = None
        self.target_model_path = None

        # Initialize relevant parameters based on the method
        if method == "linear":
            self.policy_model_weight = 0.5
            self.target_model_weight = 0.5
            self.dtype = "float16"
        elif method == "ties":
            self.policy_model_weight = 1.0
            self.policy_model_density = [1.0, 0.7, 0.1]
            self.target_model_weight = 1.0
            self.target_model_density = [1.0]
            self.normalize = 1.0
            self.dtype = "float16"
        elif method == "dare_ties":
            self.policy_model_weight = 1.0
            self.policy_model_density = [1.0, 0.7, 0.1]
            self.target_model_weight = 1.0
            self.target_model_density = [1.0]
            self.normalize = 1.0
            self.dtype = "float16"
        elif method == "slerp":
            self.t_values = 0.5
            self.dtype = "float16"
        else:
            raise ValueError(f"Unsupported merge method: {method}")

    def create_merge_config_linear(self) -> "MergeConfiguration":
        # [FUNCTION DOC]
        # Builds a mergekit MergeConfiguration for the linear merge method from the current attributes.
        # Behavior: assembles a plain dict in mergekit's schema listing policy then target model each with its weight and validates it with MergeConfiguration.model_validate.
        # Inputs: self attributes dtype, policy/target model paths and weights.
        # Outputs: a validated MergeConfiguration.
        # Exceptions: pydantic validation errors from mergekit (e.g. when model paths are still None); NameError if mergekit is not installed.
        # Side effects: none.
        """
        Creates a merge configuration for a linear merge of two models with specified weights.
        """
        # Create the merge configuration dictionary
        merge_config_dict = {
            "dtype": self.dtype,
            "merge_method": "linear",
            "models": [
                {"model": self.policy_model_path, "parameters": {"weight": self.policy_model_weight}},
                {"model": self.target_model_path, "parameters": {"weight": self.target_model_weight}},
            ],
        }

        # Create the MergeConfiguration from the dictionary
        merge_config = MergeConfiguration.model_validate(merge_config_dict)

        return merge_config

    def create_merge_config_ties(self) -> "MergeConfiguration":
        # [FUNCTION DOC]
        # Builds a mergekit MergeConfiguration for the ties merge method from the current attributes.
        # Behavior: assembles a plain dict in mergekit's schema: target model listed first then policy, each with density and weight; normalize as global parameter; the policy model is also the base model; tokenizer/chat-template/out_dtype fields left None and validates it with MergeConfiguration.model_validate.
        # Inputs: self attributes model paths, weights, densities, normalize, dtype.
        # Outputs: a validated MergeConfiguration.
        # Exceptions: pydantic validation errors from mergekit (e.g. when model paths are still None); NameError if mergekit is not installed.
        # Side effects: none.
        """
        Creates a merge configuration for a TIES merge of two models, with specified weights and densities.
        """
        # Create the TIES merge configuration dictionary
        merge_config_dict = {
            "merge_method": "ties",
            "slices": None,  # Optional slices if needed
            "models": [
                {
                    "model": {
                        "model": {"path": self.target_model_path, "revision": None},
                        "lora": None,
                        "override_architecture": None,
                    },
                    "parameters": {"density": self.target_model_density, "weight": self.target_model_weight},
                },
                {
                    "model": {
                        "model": {"path": self.policy_model_path, "revision": None},
                        "lora": None,
                        "override_architecture": None,
                    },
                    "parameters": {"density": self.policy_model_density, "weight": self.policy_model_weight},
                },
            ],
            "parameters": {"normalize": self.normalize},
            "base_model": {
                "model": {"path": self.policy_model_path, "revision": None},
                "lora": None,
                "override_architecture": None,
            },
            "dtype": self.dtype,
            "tokenizer_source": None,
            "tokenizer": None,
            "chat_template": None,
            "out_dtype": None,
        }

        # Create the MergeConfiguration from the dictionary
        merge_config = MergeConfiguration.model_validate(merge_config_dict)

        return merge_config

    def create_merge_config_dare_ties(self) -> "MergeConfiguration":
        # [FUNCTION DOC]
        # Builds a mergekit MergeConfiguration for the dare_ties merge method from the current attributes.
        # Behavior: assembles a plain dict in mergekit's schema identical in structure to the ties builder (target then policy, policy as base model) with merge_method dare_ties and validates it with MergeConfiguration.model_validate.
        # Inputs: self attributes model paths, weights, densities, normalize, dtype.
        # Outputs: a validated MergeConfiguration.
        # Exceptions: pydantic validation errors from mergekit (e.g. when model paths are still None); NameError if mergekit is not installed.
        # Side effects: none.
        """
        Creates a merge configuration for a DARE TIES merge of two models, with specified weights and densities.
        """
        # Create the DARE TIES merge configuration dictionary
        merge_config_dict = {
            "merge_method": "dare_ties",
            "slices": None,  # Optional slices if needed
            "models": [
                {
                    "model": {
                        "model": {"path": self.target_model_path, "revision": None},
                        "lora": None,
                        "override_architecture": None,
                    },
                    "parameters": {"density": self.target_model_density, "weight": self.target_model_weight},
                },
                {
                    "model": {
                        "model": {"path": self.policy_model_path, "revision": None},
                        "lora": None,
                        "override_architecture": None,
                    },
                    "parameters": {"density": self.policy_model_density, "weight": self.policy_model_weight},
                },
            ],
            "parameters": {"normalize": self.normalize},
            "base_model": {
                "model": {"path": self.policy_model_path, "revision": None},
                "lora": None,
                "override_architecture": None,
            },
            "dtype": self.dtype,
            "tokenizer_source": None,
            "tokenizer": None,
            "chat_template": None,
            "out_dtype": None,
        }

        # Create the MergeConfiguration from the dictionary
        merge_config = MergeConfiguration.model_validate(merge_config_dict)

        return merge_config

    def create_merge_config_slerp(self) -> "MergeConfiguration":
        # [FUNCTION DOC]
        # Builds a mergekit MergeConfiguration for the slerp merge method from the current attributes.
        # Behavior: assembles a plain dict in mergekit's schema: the target model is the only listed model (no parameters), the policy model is the base model, and t_values is passed as global parameter t and validates it with MergeConfiguration.model_validate.
        # Inputs: self attributes model paths, t_values, dtype.
        # Outputs: a validated MergeConfiguration.
        # Exceptions: pydantic validation errors from mergekit (e.g. when model paths are still None); NameError if mergekit is not installed.
        # Side effects: none.
        """
        Creates a merge configuration for a SLERP merge of a model with a base model.
        """

        # Create the SLERP merge configuration dictionary
        merge_config_dict = {
            "merge_method": "slerp",
            "slices": None,  # Optional slices if needed
            "models": [
                {
                    "model": {
                        "model": {"path": self.target_model_path, "revision": None},
                        "lora": None,
                        "override_architecture": None,
                    },
                    "parameters": None,  # No specific parameters for SLERP model
                }
            ],
            "parameters": {
                "t": self.t_values  # Set the t values for SLERP
            },
            "base_model": {
                "model": {"path": self.policy_model_path, "revision": None},
                "lora": None,
                "override_architecture": None,
            },
            "dtype": self.dtype,
            "tokenizer_source": None,
            "tokenizer": None,
            "chat_template": None,
            "out_dtype": None,
        }

        # Create the MergeConfiguration from the dictionary
        merge_config = MergeConfiguration.model_validate(merge_config_dict)

        return merge_config

    def create(self) -> "MergeConfiguration":
        # [FUNCTION DOC]
        # Dispatches to the builder matching self.method (linear, ties, dare_ties, slerp).
        # Inputs: none. Outputs: the MergeConfiguration from that builder; implicitly None for any other method value
        # (unreachable through __init__, which rejects unknown methods, but possible if method is reassigned).
        # Exceptions: those of the builder. Side effects: none.
        if self.method == "linear":
            return self.create_merge_config_linear()
        elif self.method == "ties":
            return self.create_merge_config_ties()
        elif self.method == "dare_ties":
            return self.create_merge_config_dare_ties()
        elif self.method == "slerp":
            return self.create_merge_config_slerp()


def merge_models(config: "MergeConfiguration", out_path: str):
    # [FUNCTION DOC]
    # Runs a mergekit merge and writes the merged model to disk.
    # Behavior: calls run_merge with MergeOptions: device auto, CUDA enabled if available, tokenizer copied,
    # lazy_unpickle and low_cpu_memory disabled.
    # Inputs: config (MergeConfiguration), out_path (output directory).
    # Outputs: none.
    # Exceptions: ImportError if mergekit is missing; any mergekit/IO error during merging.
    # Side effects: loads models (possibly downloading), writes merged weights and tokenizer to out_path, may use GPU.
    """
    Merge two models using mergekit

    Args:
        config (`MergeConfiguration`): The merge configuration.
        out_path (`str`): The output path for the merged model.
    """
    if not is_mergekit_available():
        raise ImportError("merge_models requires the `mergekit` extra. To install, run `pip install mergekit`.")
    run_merge(
        config,
        out_path=out_path,
        options=MergeOptions(
            device="auto",
            cuda=torch.cuda.is_available(),
            copy_tokenizer=True,
            lazy_unpickle=False,
            low_cpu_memory=False,
        ),
    )


class MergeModelCallback(TrainerCallback):
    # [CLASS DOC]
    # TrainerCallback that merges the training checkpoint with a target model.
    # State: merge_config (MergeConfig, default linear), merge_at_every_checkpoint, push_to_hub.
    # Flow: on_save merges each saved checkpoint when merge_at_every_checkpoint is True; otherwise on_train_end merges
    # once using the checkpoint directory of the final global step. Both go through _merge_and_maybe_push.
    r"""
    A [`~transformers.TrainerCallback`] that merges the policy model (the model being trained) with another model based
    on a merge configuration.

    Args:
        merge_config ([`experimental.merge_model_callback.MergeConfig`], *optional*):
            Configuration used for the merging process. If not provided, the default
            [`~experimental.merge_model_callback.MergeConfig`] is used.
        merge_at_every_checkpoint (`bool`, *optional*, defaults to `False`):
            Whether to merge the model at every checkpoint.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the merged model to the Hub after merging.

    Example:

    ```python
    >>> from trl.experimental.merge_model_callback import MergeConfig, MergeModelCallback

    >>> config = MergeConfig()
    >>> merge_callback = MergeModelCallback(config)
    >>> trainer = DPOTrainer(..., callbacks=[merge_callback])
    ```
    """

    def __init__(
        self,
        merge_config: "MergeConfig | None" = None,
        merge_at_every_checkpoint: bool = False,
        push_to_hub: bool = False,
    ):
        # [FUNCTION DOC]
        # Stores the callback options; uses a default MergeConfig when merge_config is None (or falsy).
        # Inputs: merge_config, merge_at_every_checkpoint (default False), push_to_hub (default False).
        # Outputs: none. Exceptions: ImportError if mergekit is not installed.
        # Side effects: sets attributes.
        if not is_mergekit_available():
            raise ImportError(
                "MergeModelCallback requires the `mergekit` extra. To install, run `pip install mergekit`."
            )
        self.merge_config = merge_config or MergeConfig()
        self.merge_at_every_checkpoint = merge_at_every_checkpoint
        self.push_to_hub = push_to_hub

    def _merge_and_maybe_push(self, output_dir, global_step, model):
        # [FUNCTION DOC]
        # HELPER: Merges the checkpoint at output_dir/checkpoint-<global_step> and optionally uploads it.
        # Behavior: sets merge_config.policy_model_path to that checkpoint path; if target_model_path is unset, fills it with
        # the model id from the model's config (so the original base model is the merge target); writes the merge into a
        # "merged" subfolder of the checkpoint; if push_to_hub, uploads it to a repo named
        # "<output_dir>_checkpoint-<global_step>_merged".
        # Inputs: output_dir, global_step, model (needs a config attribute).
        # Outputs: none.
        # Exceptions: errors from merging or uploading.
        # KNOWN ISSUE: at train end the checkpoint for the final step only exists if a save occurred at that step; otherwise
        # the merge points at a missing directory. The repo name embeds output_dir verbatim, which may contain path separators.
        # Side effects: mutates merge_config paths, writes files, possible network upload.
        checkpoint_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        self.merge_config.policy_model_path = checkpoint_path
        if self.merge_config.target_model_path is None:
            self.merge_config.target_model_path = get_config_model_id(model.config)
        merge_path = os.path.join(checkpoint_path, "merged")

        merge_models(self.merge_config.create(), merge_path)

        if self.push_to_hub:
            repo_name = f"{output_dir}_checkpoint-{global_step}_merged"
            upload_model_to_hf(merge_path, repo_name)

    def on_save(self, args, state, control, model=None, **kwargs):
        # [FUNCTION DOC]
        # Trainer hook called after a checkpoint save; merges (and maybe pushes) that checkpoint when merge_at_every_checkpoint is True.
        # Inputs: args (uses output_dir), state (uses global_step), control, model, extra kwargs ignored.
        # Outputs: none. Exceptions: those of the merge. Side effects: see _merge_and_maybe_push.
        if self.merge_at_every_checkpoint:
            self._merge_and_maybe_push(args.output_dir, state.global_step, model)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        # [FUNCTION DOC]
        # Trainer hook at the end of training; merges the final-step checkpoint when merge_at_every_checkpoint is False.
        # Inputs: args, state, control, model, extra kwargs ignored.
        # Outputs: none. Exceptions: those of the merge. Side effects: see _merge_and_maybe_push.
        if not self.merge_at_every_checkpoint:
            self._merge_and_maybe_push(args.output_dir, state.global_step, model)
