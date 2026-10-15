# [MODULE DOC]
# Tests for trl RLOOTrainer and RLOOConfig (REINFORCE Leave-One-Out online RL trainer).
# Main contents: a large TestRLOOTrainer class covering construction, dataset handling, reward functions,
# PEFT/quantization, generation backends (transformers, vLLM), metrics and logging, plus VLM tests.
# Uses tiny models from trl-internal-testing and the zen prompt-only datasets. PEFT symbols are imported only
# if peft is installed; optional-dependency tests are gated by require_* decorators.
# Import-time side effects: conditional peft import.

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

from unittest.mock import patch

import pytest
import torch
import transformers
from datasets import DatasetDict, IterableDatasetDict, load_dataset
from packaging.version import Version
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.testing_utils import torch_device
from transformers.utils import is_peft_available

from trl import RLOOConfig, RLOOTrainer

from .testing_utils import (
    TrlTestCase,
    is_ampere_or_newer,
    require_bitsandbytes,
    require_liger_kernel,
    require_peft,
    require_peft_target_parameters,
    require_torch_accelerator,
    require_vision,
    require_vllm,
)


if is_peft_available():
    import peft
    from peft import LoraConfig, get_peft_model


class TestRLOOTrainer(TrlTestCase):
    # [CLASS DOC]
    # End-to-end RLOOTrainer tests. Most use self.tmp_dir as output_dir and report_to none, and a "small RL config":
    # per_device_train_batch_size 3, num_generations 3, max_completion_length 8, often learning_rate 0.1 so tiny
    # gradients still move weights. The default reward model is the tiny Qwen2 sequence-classification checkpoint.
    # Common "standard train check": snapshot named parameters, train, assert the last log entry has train_loss and
    # every parameter changed. Common "PEFT train check": base-model parameters (prefixed base_model.model.)
    # unchanged and other non-base_layer parameters changed. Inherits TrlTestCase.
    def test_init_minimal(self):
        # [FUNCTION DOC]
        # Asserts the trainer can be constructed from only a model id, a reward model id and a train dataset (default
        # args). Inputs/Outputs: none. Exceptions: any construction error. Side effects: may create the default output dir.
        # Test that RLOOTrainer can be instantiated with only model, reward_model and train_dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            train_dataset=dataset,
        )

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Cohere2ForCausalLM",
            pytest.param(
                "trl-internal-testing/tiny-Glm4MoeForCausalLM",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.0.0"),
                    reason="GLM4 tokenizer requires transformers>=5.0.0",
                ),
            ),
            "trl-internal-testing/tiny-GptOssForCausalLM",
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            "trl-internal-testing/tiny-Qwen3MoeForCausalLM",
            pytest.param(
                "trl-internal-testing/tiny-NemotronHForCausalLM-nano",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.7.0"),
                    reason="Nemotron 3 gradient checkpointing requires transformers>=5.7.0 (see transformers#45625)",
                ),
            ),
        ],
    )
    def test_train(self, model_id):
        # [FUNCTION DOC]
        # Runs the standard train check with the small RL config across causal LMs (Cohere2, GLM4 MoE, GPT-OSS, Qwen2,
        # Qwen3 MoE, Nemotron-H; some skipped on old transformers); when aux_loss_enabled, also asserts aux_loss is
        # logged. Inputs: model_id (str, parametrized). Outputs: none. Exceptions: AssertionError.
        # Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
            model=model_id,
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # MoE models log the load-balancing auxiliary loss (on by default)
        if trainer.aux_loss_enabled:
            assert trainer.state.log_history[-1]["aux_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_liger_kernel
    def test_train_with_liger(self):
        # [FUNCTION DOC]
        # Requires liger-kernel. Asserts use_liger_kernel True emits a deprecation FutureWarning and that
        # fused_linear_cross_entropy is disabled in liger_kernel_config, then runs the standard train check.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            use_liger_kernel=True,
            report_to="none",
        )
        with pytest.warns(FutureWarning, match="`use_liger_kernel=True` is deprecated"):
            trainer = RLOOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )
        # Liger's fused linear cross-entropy would replace the forward that carries the fused LM head
        assert trainer.args.liger_kernel_config["fused_linear_cross_entropy"] is False

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_logps_match_plain_forward(self):
        # [FUNCTION DOC]
        # Builds a trainer in full precision with temperature 0.7, creates random 2x16 input ids with left padding on
        # row 0 and trailing padding on row 1, and calls the private _get_per_token_logps_and_entropies with
        # logits_to_keep 10 and entropy. Compares against a plain forward: logits for the last 10 predicted positions
        # divided by 0.7, log-softmax, gathered at the next tokens, and entropy of that distribution. Asserts closeness
        # (1e-4) on unmasked positions and that masked positions have logps of exactly 0.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        # The fused LM head scores the completion tokens like a plain forward of the model, at the sampling temperature
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            bf16=False,  # compare in full precision
            per_device_train_batch_size=2,  # reduce the batch size to reduce memory usage
            num_generations=2,  # reduce the number of generations to reduce memory usage
            temperature=0.7,
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )
        input_ids = torch.randint(0, trainer.model.config.vocab_size, (2, 16), device=trainer.accelerator.device)
        attention_mask = torch.ones_like(input_ids)
        attention_mask[0, :3] = 0  # left padding
        attention_mask[1, -2:] = 0  # padding after the end of the completion

        logps, entropies, _ = trainer._get_per_token_logps_and_entropies(
            trainer.model, input_ids, attention_mask, logits_to_keep=10, compute_entropy=True
        )
        with torch.no_grad():
            logits = trainer.model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -11:-1] / 0.7
        ref_logps = logits.log_softmax(-1)
        completion_mask = attention_mask[:, -10:].bool()

        expected_logps = ref_logps.gather(-1, input_ids[:, -10:, None]).squeeze(-1)
        expected_entropies = -(ref_logps.exp() * ref_logps).sum(-1)
        torch.testing.assert_close(logps[completion_mask], expected_logps[completion_mask], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            entropies[completion_mask], expected_entropies[completion_mask], atol=1e-4, rtol=1e-4
        )
        assert (logps[~completion_mask] == 0).all()

    def test_reward_func_wrong_number_of_rewards(self):
        # [FUNCTION DOC]
        # Asserts training with a reward function returning a single reward per batch raises ValueError mentioning that
        # it returned 1 rewards. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        # A reward function that returns the wrong number of rewards should raise a clear error instead of silently
        # broadcasting (when it returns a single value) or failing later with an opaque shape error.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def wrong_length_reward(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Reward function that always returns a one-element list with 1.0, regardless of batch size.
            # Inputs: completions and keyword arguments. Outputs: list. Exceptions: none. Side effects: none.
            return [1.0]  # too few: one reward is expected per completion

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=wrong_length_reward,
            args=training_args,
            train_dataset=dataset,
        )
        with pytest.raises(ValueError, match="returned 1 rewards"):
            trainer.train()

    @pytest.mark.parametrize("config_name", ["standard_prompt_only", "conversational_prompt_only"])
    def test_train_dataset_format(self, config_name):
        # [FUNCTION DOC]
        # Runs the standard train check on standard and conversational prompt-only datasets.
        # Inputs: config_name (str, parametrized). Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", config_name, split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_trust_remote_code(self):
        # [FUNCTION DOC]
        # With a zero reward function, asserts a remote-code model id raises ValueError mentioning custom code without
        # trust_remote_code and loads as RemoteForCausalLM with it. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: may download remote code.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        model_id = "trl-internal-testing/tiny-RemoteForCausalLM"

        def reward_func(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns a list of 0.0 with one entry per completion. Exceptions: none. Side effects: none.
            return [0.0] * len(completions)

        with pytest.raises(ValueError, match="custom code"):
            RLOOTrainer(
                model=model_id,
                args=RLOOConfig(output_dir=self.tmp_dir, report_to="none"),
                reward_funcs=reward_func,
                train_dataset=dataset,
            )

        trainer = RLOOTrainer(
            model=model_id,
            args=RLOOConfig(output_dir=self.tmp_dir, report_to="none", trust_remote_code=True),
            reward_funcs=reward_func,
            train_dataset=dataset,
        )
        assert type(trainer.model).__name__ == "RemoteForCausalLM"

    def test_train_with_eval(self):
        # [FUNCTION DOC]
        # Smoke test: trains with an eval dataset evaluated every 2 steps (eval batch 3); passes if no exception.
        # Inputs/Outputs: none. Exceptions: any training error. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            per_device_eval_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            eval_strategy="steps",
            eval_steps=2,
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        trainer.train()

    def test_train_with_num_generations_eval(self):
        # [FUNCTION DOC]
        # Smoke test like test_train_with_eval but with num_generations_eval 1 for evaluation.
        # Inputs/Outputs: none. Exceptions: any training error. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            per_device_eval_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_generations_eval=1,
            eval_strategy="steps",
            eval_steps=2,
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        trainer.train()

    def test_train_with_iterable_dataset(self):
        # [FUNCTION DOC]
        # On a streaming dataset with max_steps 4, asserts accelerator_config.dispatch_batches was forced to False,
        # then runs the standard train check. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: writes tmp_dir.
        # Iterable (streaming) datasets have no length, so `max_steps` is required.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train", streaming=True)

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            max_steps=4,
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        # Iterable datasets force `dispatch_batches=False` so the stream can be sharded per process.
        assert trainer.args.accelerator_config.dispatch_batches is False

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize("train_dataset_type", ["dataset", "iterable_dataset", "none", "unsupported_dataset_dict"])
    def test_init_with_train_dataset(self, train_dataset_type):
        # [FUNCTION DOC]
        # Checks train_dataset validation: None raises ValueError (train_dataset is required), a DatasetDict raises
        # TypeError (must be Dataset or IterableDataset); Dataset and streaming dataset (max_steps 4) are stored as the
        # same object (no preprocessing). Inputs: train_dataset_type (str, parametrized). Outputs: none.
        # Exceptions: AssertionError. Side effects: none.
        streaming = "iterable" in train_dataset_type
        if train_dataset_type == "none":
            train_dataset = None
        else:
            train_dataset = load_dataset(
                "trl-internal-testing/zen", "standard_prompt_only", split="train", streaming=streaming
            )
            if train_dataset_type == "unsupported_dataset_dict":
                # `DatasetDict` is representative of any unsupported type here; not exhaustive
                train_dataset = DatasetDict({"train": train_dataset})

        # Iterable (streaming) datasets have no length, so `max_steps` is required.
        training_args = RLOOConfig(output_dir=self.tmp_dir, max_steps=4 if streaming else -1, report_to="none")

        if train_dataset_type == "none":
            with pytest.raises(ValueError, match="`train_dataset` is required"):
                RLOOTrainer(
                    model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                    reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                    args=training_args,
                    train_dataset=train_dataset,
                )
        elif train_dataset_type == "unsupported_dataset_dict":
            with pytest.raises(TypeError, match="`train_dataset` must be a `Dataset` or `IterableDataset`"):
                RLOOTrainer(
                    model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                    reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                    args=training_args,
                    train_dataset=train_dataset,
                )
        else:
            trainer = RLOOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=train_dataset,
            )
            assert trainer.train_dataset is train_dataset

    @pytest.mark.parametrize(
        "eval_dataset_type",
        [
            "dataset",
            "iterable_dataset",
            "dataset_dict",
            "iterable_dataset_dict",
            "dict_of_dataset",
            "dict_of_iterable_dataset",
            "none",
        ],
    )
    def test_init_with_eval_dataset(self, eval_dataset_type):
        # [FUNCTION DOC]
        # Checks eval_dataset handling: None stays None; dict-like inputs give a dict with keys data1 and data2; single
        # datasets are stored as the same object. Inputs: eval_dataset_type (str, parametrized). Outputs: none.
        # Exceptions: AssertionError. Side effects: none.
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        if eval_dataset_type == "none":
            eval_dataset = None
        else:
            streaming = "iterable" in eval_dataset_type
            eval_split = load_dataset(
                "trl-internal-testing/zen", "standard_prompt_only", split="test", streaming=streaming
            )
            if eval_dataset_type in ("dataset", "iterable_dataset"):
                eval_dataset = eval_split
            elif eval_dataset_type in ("dataset_dict", "iterable_dataset_dict"):
                dataset_dict_cls = IterableDatasetDict if streaming else DatasetDict
                eval_dataset = dataset_dict_cls({"data1": eval_split, "data2": eval_split})
            else:  # "dict_of_dataset" or "dict_of_iterable_dataset"
                eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = RLOOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        if eval_dataset_type == "none":
            assert trainer.eval_dataset is None
        elif isinstance(trainer.eval_dataset, dict):
            assert set(trainer.eval_dataset.keys()) == {"data1", "data2"}
        else:
            assert trainer.eval_dataset is eval_dataset

    def test_iterable_dataset_requires_dispatch_batches_false(self):
        # [FUNCTION DOC]
        # Asserts a streaming train dataset with accelerator_config dispatch_batches True raises ValueError (iterable
        # datasets require dispatch_batches False). Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        # `dispatch_batches=True` is incompatible with iterable datasets (see get_train_dataloader).
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train", streaming=True)

        training_args = RLOOConfig(
            output_dir=self.tmp_dir, accelerator_config={"dispatch_batches": True}, report_to="none"
        )
        with pytest.raises(ValueError, match="Iterable datasets require `dispatch_batches=False`"):
            RLOOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

    def test_iterable_dataset_forces_num_workers_zero(self):
        # [FUNCTION DOC]
        # Asserts a streaming train dataset with dataloader_num_workers 4 has it forced to 0 (to keep generation groups
        # together). Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        # Multiple workers would shard and interleave the repeated stream, splitting num_generations groups, so
        # `dataloader_num_workers` is forced to 0 for iterable datasets.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train", streaming=True)

        training_args = RLOOConfig(output_dir=self.tmp_dir, dataloader_num_workers=4, max_steps=1, report_to="none")
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        assert trainer.args.dataloader_num_workers == 0

    def test_iterable_eval_keeps_map_style_train_workers(self):
        # [FUNCTION DOC]
        # With a map-style train set, streaming eval set and 4 dataloader workers: asserts args keep 4 workers, the train
        # dataloader has 4 workers, the eval dataloader has 0, and args still show 4 afterwards (override not persisted).
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: builds dataloaders.
        # A map-style train set keeps its workers even when the eval set is iterable; the single-worker restriction is
        # scoped to the iterable eval loader and not persisted.
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        eval_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="test", streaming=True)

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            per_device_eval_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            dataloader_num_workers=4,
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        assert trainer.args.dataloader_num_workers == 4  # not disabled by the iterable eval set
        assert trainer.get_train_dataloader().num_workers == 4  # map-style train keeps its workers
        assert trainer.get_eval_dataloader().num_workers == 0  # iterable eval loader uses a single worker
        assert trainer.args.dataloader_num_workers == 4  # override is scoped, not persisted

    @pytest.mark.parametrize(
        "eval_dataset_type",
        [
            "dataset",
            "iterable_dataset",
            "dataset_dict",
            "iterable_dataset_dict",
            "dict_of_dataset",
            "dict_of_iterable_dataset",
        ],
    )
    def test_evaluate_with_eval_dataset(self, eval_dataset_type):
        # [FUNCTION DOC]
        # Asserts evaluate accepts an eval dataset passed directly in six container forms (map/iterable, single, dataset
        # dict, plain dict), yielding eval_loss or eval_data1_loss and eval_data2_loss.
        # Inputs: eval_dataset_type (str, parametrized). Outputs: none. Exceptions: AssertionError. Side effects: none.
        # `evaluate` accepts a dataset passed directly, not only an `eval_dataset` set at init. Iterable datasets passed
        # this way must still be configured for the iterable path, since they are absent at init.
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        streaming = "iterable" in eval_dataset_type
        eval_split = load_dataset(
            "trl-internal-testing/zen", "standard_prompt_only", split="test", streaming=streaming
        )
        if eval_dataset_type in ("dataset", "iterable_dataset"):
            eval_dataset = eval_split
        elif eval_dataset_type in ("dataset_dict", "iterable_dataset_dict"):
            dataset_dict_cls = IterableDatasetDict if streaming else DatasetDict
            eval_dataset = dataset_dict_cls({"data1": eval_split, "data2": eval_split})
        else:  # "dict_of_dataset" or "dict_of_iterable_dataset"
            eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            per_device_eval_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=train_dataset,
        )

        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        if eval_dataset_type in ("dataset", "iterable_dataset"):
            assert metrics["eval_loss"] is not None
        else:
            assert metrics["eval_data1_loss"] is not None
            assert metrics["eval_data2_loss"] is not None

    def test_train_multiple_iterations(self):
        # [FUNCTION DOC]
        # Runs the standard train check with num_iterations 2 (reusing each generation batch twice).
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_iterations=2,
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_peft
    def test_peft_init_is_seeded(self):
        # [FUNCTION DOC]
        # Builds two LoRA trainers after different global torch seeds and asserts identical non-empty lora_A weights.
        # Requires peft. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: changes the global seed.
        # Two trainers with the same seed start from the same adapter weights
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        adapters = []
        for global_seed in range(2):
            torch.manual_seed(global_seed)  # a different global RNG state, as in two separate runs
            trainer = RLOOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=RLOOConfig(output_dir=self.tmp_dir, report_to="none"),
                train_dataset=dataset,
                peft_config=LoraConfig(),
            )
            adapters.append({n: p.clone() for n, p in trainer.model.named_parameters() if "lora_A" in n})

        assert adapters[0]
        for n, param in adapters[0].items():
            assert torch.equal(param, adapters[1][n]), f"Parameter {n} differs between the two trainers."

    @require_peft
    def test_train_peft_config(self):
        # [FUNCTION DOC]
        # Runs the PEFT train check passing a float32 model instance plus a default LoraConfig. Requires peft.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
            model=model,
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(),
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "base_layer" not in n:  # We expect the peft params to be different (except for the base layer)
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_peft
    @require_bitsandbytes
    def test_train_peft_and_quantization(self):
        # [FUNCTION DOC]
        # With peft and bitsandbytes, loads the model id in 4-bit NF4 double-quant (bf16 compute) with LoRA; asserts
        # 4-bit loading, trains, and asserts LoRA parameters are bfloat16 and changed (base params unchecked).
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",  # identifier, so that the trainer quantizes it
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
            quantization_config=quantization_config,
            peft_config=LoraConfig(),
        )

        # Check that the trainer applied the quantization config when loading the model
        assert trainer.model.base_model.model.is_loaded_in_4bit

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed, and that they are cast to bfloat16, as recommended by the QLoRA
        # paper. The base model params are not checked: bitsandbytes casts the biases of a Linear4bit in-place during
        # the forward pass, so some of them change in a way that is unrelated to training.
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if "lora" in n:  # We expect the peft params to be different
                assert param.dtype == torch.bfloat16, f"Parameter {n} is not in bfloat16."
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_peft
    @require_bitsandbytes
    def test_train_peft_dora_and_quantization(self):
        # [FUNCTION DOC]
        # Same as the 4-bit LoRA test with DoRA: lora_magnitude_vector parameters must stay float32 and change, other
        # LoRA parameters must be bfloat16 and change. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",  # identifier, so that the trainer quantizes it
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
            quantization_config=quantization_config,
            peft_config=LoraConfig(use_dora=True),
        )

        # Check that the trainer applied the quantization config when loading the model
        assert trainer.model.base_model.model.is_loaded_in_4bit

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed, and that they are cast to bfloat16, as recommended by the QLoRA
        # paper. The base model params are not checked: bitsandbytes casts the biases of a Linear4bit in-place during
        # the forward pass, so some of them change in a way that is unrelated to training.
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            # The DoRA magnitude vector is the exception: it must stay in float32, since its optimizer updates can be
            # smaller than bfloat16 can represent, which would otherwise silently freeze it, see #7268
            if "lora_magnitude_vector" in n:
                assert param.dtype == torch.float32, f"Parameter {n} is not in float32."
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."
            elif "lora" in n:  # We expect the peft params to be different
                assert param.dtype == torch.bfloat16, f"Parameter {n} is not in bfloat16."
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_peft
    def test_peft_lm_head_raises(self):
        # [FUNCTION DOC]
        # Asserts a LoRA config targeting lm_head (plus q_proj, v_proj) raises ValueError mentioning lm_head, since the
        # chunked projection reads lm_head weights directly. Sets ensure_weight_tying only for peft versions from 0.19.0
        # up to before 0.19.2.dev0. Requires peft. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        # The chunked projection reads `lm_head.weight` directly, so a LoRA adapter on `lm_head` is silently
        # ignored and never trained. The trainer must fail fast instead of training a silently-frozen head (#4612).
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", dtype="float32")
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        training_args = RLOOConfig(output_dir=self.tmp_dir, report_to="none")
        # `ensure_weight_tying=True` silences PEFT's weight-tying warning fired when lm_head is in the adapter on a
        # model with untied embeddings; once tying respects `tie_word_embeddings`, the flag itself warns that no tied
        # modules were found, so it must only be set on the affected range.
        # - Introduced in PEFT 0.19.0 (peft#2879); fixed on main, unreleased as of 0.19.2.dev0 (peft#3171)
        needs_ensure_weight_tying = Version("0.19.0") <= Version(peft.__version__) < Version("0.19.2.dev0")
        lora_config = LoraConfig(
            target_modules=["q_proj", "v_proj", "lm_head"],
            **({"ensure_weight_tying": True} if needs_ensure_weight_tying else {}),
        )
        with pytest.raises(ValueError, match="lm_head"):
            RLOOTrainer(
                model=model,
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
                peft_config=lora_config,
            )

    @require_peft
    def test_train_peft_model(self):
        # [FUNCTION DOC]
        # Trains an already-wrapped PEFT model (default LoRA) with the small RL config; base parameters unchanged and
        # all others except base_layer and ref adapter parameters changed. Requires peft. Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: writes tmp_dir.
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]
        lora_config = LoraConfig()
        model = get_peft_model(model, lora_config)
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
            model=model,
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "base_layer" not in n and "ref" not in n:  # and the peft params to be different (except base and ref)
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_peft_target_parameters
    def test_train_moe_peft_model(self):
        # [FUNCTION DOC]
        # Regression for issue 5222 on tiny GPT-OSS with LoRA target_parameters on expert weights: asserts a ref adapter
        # exists in peft_config for peft 0.20.0 or newer and is absent before; then the same check as
        # test_train_peft_model. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        # Regression test for https://github.com/huggingface/trl/issues/5222. Before PEFT 0.20.0, only one adapter per
        # model was supported when the LoRA config uses `target_parameters` (see peft#3340, fixed in peft#3350), so no
        # "ref" adapter could be created and the reference log probs were computed with adapters disabled instead.
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-GptOssForCausalLM", dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]
        lora_config = LoraConfig(target_parameters=["mlp.experts.down_proj", "mlp.experts.gate_up_proj"])
        model = get_peft_model(model, lora_config)
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
            model=model,
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        if Version(peft.__version__) < Version("0.20.0"):
            assert "ref" not in trainer.model.peft_config
        else:
            assert "ref" in trainer.model.peft_config

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "base_layer" not in n and "ref" not in n:  # and the peft params to be different (except base and ref)
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    # In practice, this test is the same as `test_train_peft_config`, since gradient checkpointing is enabled by
    # default in `RLOOTrainer`. We keep it as a regression guard: if the default ever changes, we still explicitly test
    # PEFT + gradient checkpointing, which has caused issues in the past.
    @require_peft
    def test_train_peft_with_gradient_checkpointing(self):
        # [FUNCTION DOC]
        # Runs the PEFT train check with a model instance, LoraConfig and gradient_checkpointing explicitly True
        # (regression guard). Requires peft. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: writes tmp_dir.
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            gradient_checkpointing=True,  # enable gradient checkpointing
            report_to="none",
        )
        trainer = RLOOTrainer(
            model=model,
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(),
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "base_layer" not in n:  # We expect the peft params to be different (except for the base layer)
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_torch_accelerator
    def test_train_with_fp16(self):
        # [FUNCTION DOC]
        # Runs the standard train check with fp16; requires an accelerator. Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            fp16=True,
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_different_reward_model(self):
        # [FUNCTION DOC]
        # Runs the standard train check on conversational prompts with a tiny Llama sequence-classification reward model
        # and its own tokenizer passed via reward_processing_classes, using a dedicated pad token (not eos) so reward
        # scoring picks the correct last token. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: writes tmp_dir.
        # Use a reward model different from the model: different chat template, tokenization, etc.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        reward_model_id = "trl-internal-testing/tiny-LlamaForSequenceClassification-3.2"
        reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_id)
        reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_id)
        # By default, the trainer uses the eos token as the padding token. However, for Llama models, the eos token
        # appears in the chat template. Using it as a pad token disrupts the reward calculation, as the calculation
        # considers the score of the last token before the first pad token. To ensure correct reward calculations,
        # we use a separate pad token instead.
        reward_tokenizer.pad_token = "<|finetune_right_pad_id|>"

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_model,
            args=training_args,
            train_dataset=dataset,
            reward_processing_classes=reward_tokenizer,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_reward_func_standard(self):
        # [FUNCTION DOC]
        # Runs the standard train check with a Python reward function on standard-format completions.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        # Test if trainer can handle reward function with standard format
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the character length of each completion as a float. Exceptions: none. Side effects: none.
            """Reward function that rewards longer completions."""
            return [float(len(completion)) for completion in completions]

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_reward_func_conversational(self):
        # [FUNCTION DOC]
        # Runs the standard train check with a reward function on conversational completions.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        # Test if trainer can handle reward function with conversational format
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the length of the first message content of each completion as a float.
            # Exceptions: none. Side effects: none.
            """Reward function that gives higher scores to longer completion content."""
            completion_contents = [completion[0]["content"] for completion in completions]
            return [float(len(content)) for content in completion_contents]

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_multiple_reward_funcs(self):
        # [FUNCTION DOC]
        # Runs the standard train check with two reward functions (length and unique-character count).
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        # Test that RLOOTrainer can be instantiated with multiple reward functions
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def reward_func1(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns each completion length as a float. Exceptions: none. Side effects: none.
            """Reward function that rewards longer completions."""
            return [float(len(completion)) for completion in completions]

        def reward_func2(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the number of distinct characters per completion as a float. Exceptions: none.
            # Side effects: none.
            """Reward function that rewards completions with more unique letters."""
            return [float(len(set(completion))) for completion in completions]

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=[reward_func1, reward_func2],
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_sync_and_async_reward_funcs(self):
        # [FUNCTION DOC]
        # Runs the standard train check with two sync reward functions, an async function and an instance of an async
        # callable class, verifying mixed sync/async rewards are supported. Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: writes tmp_dir.
        # Test that RLOOTrainer can be instantiated with multiple reward functions, one of which is async, in both the
        # function and the callable class form.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def sync_reward_func1(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns each completion length as a float. Exceptions: none. Side effects: none.
            """Reward function that rewards longer completions."""
            return [float(len(completion)) for completion in completions]

        def sync_reward_func2(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the integer 1 for every completion. Exceptions: none. Side effects: none.
            return [1 for _ in completions]

        async def async_reward_func(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Coroutine returning the distinct-character count per completion as a float. Exceptions: none.
            # Side effects: none.
            """Async Reward function that rewards completions with more unique letters."""
            return [float(len(set(completion))) for completion in completions]

        class AsyncCallableReward:
            # [CLASS DOC]
            # HELPER: Reward written as a class whose instances are awaited when called; no state.
            """Async reward function written as a callable class."""

            async def __call__(self, completions, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Coroutine returning the number of spaces in each completion as a float.
                # Inputs: completions, keyword arguments. Outputs: list of floats. Exceptions: none. Side effects: none.
                return [float(completion.count(" ")) for completion in completions]

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=[sync_reward_func1, sync_reward_func2, async_reward_func, AsyncCallableReward()],
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_multiple_reward_funcs_with_None_output(self):
        # [FUNCTION DOC]
        # Runs the standard train check (on parameters requiring grad) with one applicable reward and one returning None
        # for every sample, verifying None rewards are tolerated. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: writes tmp_dir.
        """Test that a valid math reward function is processed correctly while the code reward function returns None."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def applicable_reward_func(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns each completion length as a float. Exceptions: none. Side effects: none.
            """A reward function that rewards longer completions."""
            return [float(len(completion)) for completion in completions]

        def non_applicable_reward_func(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns a list of None, one per completion. Exceptions: none. Side effects: none.
            """A reward function that returns None for all inputs, as it is not applicable to this sample."""
            return [None] * len(completions)

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )

        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=[
                applicable_reward_func,
                non_applicable_reward_func,
            ],  # One applicable, one non applicable
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {
            n: param.clone() for n, param in trainer.model.named_parameters() if param.requires_grad
        }

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_multiple_reward_funcs_with_weights(self):
        # [FUNCTION DOC]
        # Trains with two reward functions and reward_weights 0.7 and 0.3; asserts train_loss and per-function
        # rewards/NAME/mean and rewards/NAME/std metrics are in the last log entry, and all parameters changed.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        """Test that RLOOTrainer can handle multiple reward functions with weights."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def reward_func1(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns each completion length as a float. Exceptions: none. Side effects: none.
            """Reward function that rewards longer completions."""
            return [float(len(completion)) for completion in completions]

        def reward_func2(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the distinct-character count per completion as a float. Exceptions: none. Side effects: none.
            """Reward function that rewards completions with more unique letters."""
            return [float(len(set(completion))) for completion in completions]

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
            reward_weights=[0.7, 0.3],  # weight of reward_func1 and reward_func2 respectively
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=[reward_func1, reward_func2],
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        # Check that training logs contain both reward metrics
        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert "rewards/reward_func1/mean" in trainer.state.log_history[-1]
        assert "rewards/reward_func1/std" in trainer.state.log_history[-1]
        assert "rewards/reward_func2/mean" in trainer.state.log_history[-1]
        assert "rewards/reward_func2/std" in trainer.state.log_history[-1]

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_reward_metric_reflects_reward_weights(self):
        # [FUNCTION DOC]
        # With constant rewards 1.0 and 0.0 weighted 0.7 and 0.3, asserts the logged reward metric is 0.7 (within 1e-5),
        # i.e. weighted rather than an unweighted sum. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: writes tmp_dir.
        """Test that the logged 'reward' metric uses reward_weights, not an unweighted sum."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def constant_reward_1(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns 1.0 for every completion. Exceptions: none. Side effects: none.
            return [1.0] * len(completions)

        def constant_reward_0(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns 0.0 for every completion. Exceptions: none. Side effects: none.
            return [0.0] * len(completions)

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
            reward_weights=[0.7, 0.3],
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=[constant_reward_1, constant_reward_0],
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        log = trainer.state.log_history[-1]
        # With reward_weights=[0.7, 0.3] and rewards [1.0, 0.0]:
        # weighted reward = 0.7*1.0 + 0.3*0.0 = 0.7
        # unweighted reward = 1.0 + 0.0 = 1.0
        assert abs(log["reward"] - 0.7) < 1e-5, (
            f"Expected logged reward to be ~0.7 (weighted), got {log['reward']}. "
            "The reward metric should reflect reward_weights."
        )

    def test_train_multiple_mixed_reward_funcs(self):
        # [FUNCTION DOC]
        # Runs the standard train check with a mix of a Python reward function and a reward model id.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        # Test if the trainer can handle a mix of reward functions and reward models
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns each completion length as a float. Exceptions: none. Side effects: none.
            """Reward function that rewards longer completions."""
            return [float(len(completion)) for completion in completions]

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=[reward_func, "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"],
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_reward_func_additional_column(self):
        # [FUNCTION DOC]
        # Adds a some_values column (row index) and runs the standard train check with a reward that consumes it,
        # verifying extra dataset columns are forwarded to reward functions as keyword arguments.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        # Test if trainer can handle reward function that rely on additional columns in the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        # Add a column to the dataset (dummy example, the column could be anything)
        some_values = list(range(len(dataset)))
        dataset = dataset.add_column("some_values", some_values)

        def reward_func(completions, some_values, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the absolute difference between each completion length and its some_values entry, as floats.
            # Inputs: completions, some_values, keyword arguments. Exceptions: ValueError if lengths differ (strict zip).
            # Side effects: none.
            """Reward function that rewards completions with lengths closer to the values in some_values."""
            return [
                float(abs(len(completion) - value)) for completion, value in zip(completions, some_values, strict=True)
            ]

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_ref_model_like_model_instance(self):
        # [FUNCTION DOC]
        # With beta 0.1 and a bfloat16 eager-attention model instance, asserts the built ref_model has bfloat16 dtype and
        # eager attention. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        # A policy passed as an instance has no `model_init_kwargs`: the reference is built with its dtype and attention
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", dtype=torch.bfloat16, attn_implementation="eager"
        )
        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            beta=0.1,  # set beta to a non-zero value so that the trainer needs a reference model
            report_to="none",
        )
        trainer = RLOOTrainer(
            model=model,
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        assert trainer.ref_model.dtype == torch.bfloat16
        assert trainer.ref_model.config._attn_implementation == "eager"

    def test_train_with_sync_ref_model(self):
        # [FUNCTION DOC]
        # With beta 0.1 and sync_ref_model True, asserts a ref model exists and that after training both policy and
        # reference parameters changed (reference synced from the policy). Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            beta=0.1,  # ensure ref model is created so sync can update it
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            sync_ref_model=True,
            ref_model_sync_steps=2,  # reduce sync steps to ensure a sync happens
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}
        assert trainer.ref_model is not None
        previous_ref_params = {n: param.clone() for n, param in trainer.ref_model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."
            new_ref_param = trainer.ref_model.get_parameter(n)
            assert not torch.equal(previous_ref_params[n], new_ref_param), f"Ref Parameter {n} has not changed."

    def test_train_beta_zero(self):
        # [FUNCTION DOC]
        # Runs the standard train check with beta 0.0 (no reference model / KL term).
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            beta=0.0,  # set beta to zero value to test the case where the reference model is not used
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_with_pad_to_multiple_of(self):
        # [FUNCTION DOC]
        # Runs the standard train check with pad_to_multiple_of 8. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            pad_to_multiple_of=8,
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_vllm
    def test_train_vllm(self):
        # [FUNCTION DOC]
        # Requires vLLM. Runs the standard train check with use_vllm True on Qwen2.5-0.5B-Instruct (tiny models are too
        # small for vLLM). Inputs/Outputs: none. Exceptions: AssertionError. Side effects: downloads a model, uses GPU.
        """Test that training works with vLLM for generation."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
            use_vllm=True,
        )
        trainer = RLOOTrainer(
            model="Qwen/Qwen2.5-0.5B-Instruct",  # tiny model is too small for vLLM
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_peft
    @require_vllm
    @pytest.mark.skip(reason="We should add a mock for the vLLM server.")
    def test_train_vllm_and_peft(self):
        # [FUNCTION DOC]
        # Currently skipped (needs a vLLM server mock). Would train Qwen2.5-0.5B with vLLM and LoRA on all-linear
        # modules plus modules_to_save embed_tokens and lm_head; base parameters unchanged, others except base_layer and
        # original_module changed. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none while skipped.
        """Test that training works with vLLM for generation."""
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", dtype="float32"
        )  # tiny model is too small for vLLM
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
            use_vllm=True,
        )
        lora_config = LoraConfig(
            target_modules="all-linear",
            # test with non-default modules as it adds extra keys in state_dict that we need to handle
            modules_to_save=["embed_tokens", "lm_head"],
        )
        trainer = RLOOTrainer(
            model=model,
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
            peft_config=lora_config,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "base_layer" not in n and "original_module" not in n:
                # We expect the peft params to be different (except for the base layer)
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_vllm
    @pytest.mark.skip(reason="We should add a mock for the vLLM server.")
    def test_train_vllm_structured_outputs(self):
        # [FUNCTION DOC]
        # Currently skipped (needs a vLLM server mock). Would run the standard train check with vLLM constrained by a
        # reasoning/answer regex via vllm_structured_outputs_regex. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: none while skipped.
        """Test that training works with vLLM for generation with structured outputs."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
            use_vllm=True,
            vllm_structured_outputs_regex=r"<reasoning>\n.*\n</reasoning>\n<answer>\n.*\n</answer>",
        )
        trainer = RLOOTrainer(
            model="Qwen/Qwen2.5-0.5B-Instruct",  # tiny model is too small for vLLM
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_with_additional_generation_kwargs(self):
        # [FUNCTION DOC]
        # Runs the standard train check with top_p 0.9, top_k 10, min_p 0.01 and repetition_penalty 1.1.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        """Test that training works with additional generation kwargs."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
            top_p=0.9,
            top_k=10,
            min_p=0.01,
            repetition_penalty=1.1,
        )

        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_vllm
    @pytest.mark.skip(reason="We should add a mock for the vLLM server.")
    def test_train_vllm_with_additional_generation_kwargs(self):
        # [FUNCTION DOC]
        # Currently skipped (needs a vLLM server mock). Same sampling kwargs as the previous test but generating with
        # vLLM. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none while skipped.
        """Test that training works with vLLM and additional generation kwargs."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
            use_vllm=True,
            top_p=0.9,
            top_k=10,
            min_p=0.01,
            repetition_penalty=1.1,
        )

        trainer = RLOOTrainer(
            model="Qwen/Qwen2.5-0.5B-Instruct",  # tiny model is too small for vLLM
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    # Flash Attention needs a `head_size` multiple of 8, hence a `small-*` model (`tiny-*` ones have `head_size=2`)
    @pytest.mark.skipif(
        Version(transformers.__version__) < Version("5.8.0"),
        reason="transformers continuous batching requires transformers>=5.8.0",
    )
    @pytest.mark.skipif(
        not is_ampere_or_newer() and torch_device != "xpu",
        reason="transformers continuous batching switches attention to Flash Attention, which requires an Ampere or "
        "newer GPU, or XPU (see https://github.com/huggingface/transformers/issues/47926)",
    )
    def test_train_with_transformers_continuous_batching(self):
        # [FUNCTION DOC]
        # Runs the standard train check for 2 steps with use_transformers_continuous_batching on small Qwen3 (head size
        # compatible with Flash Attention). Skipped before transformers 5.8.0 or without an Ampere-or-newer GPU or XPU.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            max_steps=2,  # continuous batching adds about 1s per step
            report_to="none",
            use_transformers_continuous_batching=True,
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/small-Qwen3ForCausalLM",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_with_normalized_advantages(self):
        # [FUNCTION DOC]
        # Runs the standard train check with normalize_advantages True. Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            normalize_advantages=True,
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_with_clipped_rewards(self):
        # [FUNCTION DOC]
        # Runs the standard train check with reward_clip_range from -1 to 1. Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            reward_clip_range=(-1, 1),
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @patch("transformers.generation.utils.GenerationMixin.generate")
    def test_train_with_mask_truncated_completions(self, mock_generate):
        # [FUNCTION DOC]
        # Patches GenerationMixin.generate to return three fixed 8-token completions (one with no eos, one with eos then
        # padding, one with eos as last token) so some completions are not truncated; runs the standard train check with
        # mask_truncated_completions True. Inputs: mock_generate (patched mock). Outputs: none.
        # Exceptions: AssertionError. Side effects: generate patched during the test.
        """Test that training works with mask_truncated_completions=True parameter."""

        # We mock the generate method because the model's random weights make it extremely unlikely to produce a
        # sequence containing the EOS token within the allowed max_completion_length. As a result, all tokens are
        # masked in the loss, the model doesn't update, and the final check (which verifies the update) fails.
        def fake_generate(input_ids, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Appends a fixed 3x8 completion tensor (on the input device) to input_ids along the sequence dimension.
            # Assumes a generation batch of exactly 3 rows. Inputs: input_ids, other kwargs ignored.
            # Outputs: concatenated tensor. Exceptions: shape errors if batch is not 3. Side effects: none.
            # pad_token_id = 151643; eos_token_id = 151645
            completion_ids = torch.tensor(
                [
                    [1, 2, 3, 4, 5, 6, 7, 8],  # this one is truncated
                    [9, 10, 11, 151645, 151643, 151643, 151643, 151643],  # this one contains eos
                    [12, 13, 14, 15, 16, 17, 18, 151645],  # particular case, eos is generated just within the limit
                ],
                device=input_ids.device,
            )
            return torch.cat([input_ids, completion_ids], dim=1)

        mock_generate.side_effect = fake_generate

        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            mask_truncated_completions=True,  # Enable masking of truncated completions
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_with_mask_truncated_completions_all_masked(self):
        # [FUNCTION DOC]
        # With mask_truncated_completions True and an untrained model that almost never emits eos, asserts train_loss is
        # logged and every parameter is unchanged (no learning signal). Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: writes tmp_dir.
        """
        Test that when all generated completions are truncated (i.e., none contain an EOS token), and
        mask_truncated_completions=True, the model receives no effective learning signal and therefore does not update
        its parameters.

        Here, we don't mock the generate method, be we rely on the fact that the model the probability of generating
        the EOS token is extremely low, so all generated completions are truncated.
        """
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            mask_truncated_completions=True,  # Enable masking of truncated completions
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert torch.equal(param, new_param), f"Parameter {n} has changed."

    def test_warning_raised_all_rewards_none(self, caplog):
        # [FUNCTION DOC]
        # With a reward function returning None for everything, asserts training logs a WARNING from
        # trl.trainer.rloo_trainer containing "All reward functions returned None for the following kwargs:".
        # Inputs: caplog (pytest fixture). Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        """Test that a proper warning is raised when all rewards are None."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def always_none_reward_func(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns None for every completion. Exceptions: none. Side effects: none.
            """Reward function that always returns None."""
            return [None] * len(completions)

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=always_none_reward_func,
            args=training_args,
            train_dataset=dataset,
        )

        with caplog.at_level("WARNING", logger="trl.trainer.rloo_trainer"):
            trainer.train()

        expected_warning = "All reward functions returned None for the following kwargs:"
        assert expected_warning in caplog.text

    def test_train_num_generations_larger_than_batch_size(self):
        # [FUNCTION DOC]
        # Runs the standard train check with num_generations 6 larger than the per-device batch, made valid by
        # gradient_accumulation_steps 2. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_generations=6,  # the number of generations is larger than the batch size, but
            gradient_accumulation_steps=2,  # gradient accumulation should allow that
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_multiple_dataloader_workers(self):
        # [FUNCTION DOC]
        # Forces the torch multiprocessing spawn start method and runs the standard train check with 2 dataloader
        # workers. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: changes the global multiprocessing start method; writes tmp_dir.
        # Pytest/CI often starts background threads before tests run. With Python 3.12, using the default "fork" start
        # method in a multi-threaded process emits a DeprecationWarning and may deadlock.
        #
        # We force "spawn" here to make multiprocessing safe under pytest when DataLoader workers are enabled. This is
        # test-environment–specific and not required by the training logic itself.
        #
        # This means the test does not cover "fork". However, "spawn" is stricter (requires full picklability and clean
        # state) and avoids fork-after-threads issues that pytest cannot reliably test anyway. Fork-specific behavior,
        # if needed, should be tested in a clean process outside pytest.
        torch.multiprocessing.set_start_method("spawn", force=True)

        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            dataloader_num_workers=2,  # use multiple dataloader workers
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_with_generation_kwargs(self):
        # [FUNCTION DOC]
        # Runs the standard train check with generation_kwargs (sampling, top_k 50, 2 beams, length_penalty -0.1).
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            # Pass gen kwargs
            generation_kwargs={"do_sample": True, "top_k": 50, "num_beams": 2, "length_penalty": -0.1},
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_with_reward_func_accessing_trainer_state(self):
        # [FUNCTION DOC]
        # Trains with a reward function that asserts a trainer_state keyword is passed and has global_step; passes if no
        # exception. Inputs/Outputs: none. Exceptions: AssertionError from within training. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Asserts trainer_state kwarg is present with global_step, returns distinct-character counts as floats.
            # Exceptions: AssertionError. Side effects: none.
            trainer_state = kwargs.get("trainer_state")
            assert trainer_state is not None
            # transformers.TrainerState instance should have a `global_step` property.
            assert hasattr(trainer_state, "global_step")
            return [float(len(set(completion))) for completion in completions]

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()

    def test_train_reward_func_with_log_extra(self):
        # [FUNCTION DOC]
        # With log_completions True, trains with a reward function that calls the log_extra callback; asserts test_column
        # appears in the trainer's extra logs. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Asserts log_extra is provided, logs the first 5 characters of each completion under test_column, and
            # returns completion lengths as floats. Exceptions: AssertionError. Side effects: writes to trainer extra logs.
            log_extra = kwargs.get("log_extra")
            assert log_extra is not None
            log_extra("test_column", [completion[:5] for completion in completions])
            return [float(len(completion)) for completion in completions]

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
            log_completions=True,
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()
        assert "test_column" in trainer._logs["extra"]

    def test_train_reward_func_with_log_metric(self):
        # [FUNCTION DOC]
        # Trains with a reward function calling the log_metric callback; asserts custom_accuracy appears in some log
        # entry. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Asserts log_metric is provided, logs custom_accuracy 0.75, and returns completion lengths as floats.
            # Exceptions: AssertionError. Side effects: records a trainer metric.
            log_metric = kwargs.get("log_metric")
            assert log_metric is not None
            log_metric("custom_accuracy", 0.75)
            return [float(len(completion)) for completion in completions]

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()
        # log_metric appends to _metrics, which gets averaged and merged into log_history
        logged_keys = {k for entry in trainer.state.log_history for k in entry}
        assert "custom_accuracy" in logged_keys

    def test_prepare_input_called_with_correct_data(self):
        # [FUNCTION DOC]
        # With batch 6, steps_per_generation 4, num_generations 4, num_iterations 2, gradient accumulation 3 and no
        # shuffling, wraps training_step with a spy and trains. Asserts 48 calls (3 epochs x 2 iterations x 2
        # generation batches x 4 steps) and that the first 8 calls receive the first generation batch (6 prompts each
        # repeated 4 times, in dataset order) and calls 8 to 15 the second generation batch.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: training_step patched during training.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            gradient_accumulation_steps=3,  # can be anything in this test
            # steps_per_generation*per_device_train_batch_size=24 is divisible by num_generations=4
            steps_per_generation=4,
            num_generations=4,
            per_device_train_batch_size=6,  # reduce the batch size to reduce memory usage
            num_iterations=2,
            shuffle_dataset=False,
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )
        # steps_per_generation=4, per_device_train_batch_size=6 and num_generations=4, so we expect a
        # generation batch of 24 samples (steps_per_generation * per_device_train_batch_size), containing 6
        # different prompts (steps_per_generation * per_device_train_batch_size // num_generations), each repeated
        # 4 times (num_generations).
        expected_first_generation_batch = (
            [{"prompt": "Beautiful is better than"}] * 4
            + [{"prompt": "Explicit is"}] * 4
            + [{"prompt": "Simple is better"}] * 4
            + [{"prompt": "Complex"}] * 4
            + [{"prompt": "Flat is better than"}] * 4
            + [{"prompt": "Sparse is better"}] * 4
        )
        expected_second_generation_batch = (
            [{"prompt": "Readability"}] * 4
            + [{"prompt": "Special cases aren't special"}] * 4
            + [{"prompt": "Although practicality beats"}] * 4
            + [{"prompt": "Errors should never"}] * 4
            + [{"prompt": "Unless explicitly"}] * 4
            + [{"prompt": "In the face of ambiguity, refuse"}] * 4
        )

        with patch.object(RLOOTrainer, "training_step", wraps=trainer.training_step) as mock_prepare:
            trainer.train()
            # 3 epochs * 2 iterations * 2 generation batches to cover the dataset * 4 steps_per_generation
            assert mock_prepare.call_count == 48
            for i in range(0, 8):  # Generation batch repeated 8 times (steps_per_generation*num_iterations)
                assert mock_prepare.call_args_list[i].args[1] == expected_first_generation_batch
            for i in range(8, 16):
                assert mock_prepare.call_args_list[i].args[1] == expected_second_generation_batch

    def test_train_stops_on_every_eos_token_id(self):
        # [FUNCTION DOC]
        # With tiny Phi-3.5 (whose generation config, not tokenizer, lists the end-of-turn token 32007 as eos), asserts
        # 32007 is in generation_config.eos_token_id; patches model.generate to append a 4-token completion with 32007 in
        # second position; after training asserts completions/mean_length is 2.0 and completions/clipped_ratio is 0.0
        # (cut after the eos and counted as terminated). Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: generate patched during training.
        # Phi-3.5 ends a turn with `<|end|>`, which only its generation config declares as eos, not its tokenizer
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Phi3ForCausalLM-3.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )
        assert 32007 in trainer.generation_config.eos_token_id  # `<|end|>`

        def fake_generate(input_ids, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Appends the fixed token ids for Blue, end-of-turn, and, green to every row of input_ids along the last
            # dimension. Inputs: input_ids, kwargs ignored. Outputs: tensor. Exceptions: none. Side effects: none.
            # 'Blue<|end|>and green': the turn ends on `<|end|>`, and generation goes on after it
            completion_ids = torch.tensor([[10924, 32007, 322, 7933]] * input_ids.shape[0], device=input_ids.device)
            return torch.cat([input_ids, completion_ids], dim=-1)

        with patch.object(trainer.model, "generate", side_effect=fake_generate):
            trainer.train()

        # Completions are cut after `<|end|>` and count as terminated, not clipped
        assert trainer.state.log_history[-1]["completions/mean_length"] == 2.0
        assert trainer.state.log_history[-1]["completions/clipped_ratio"] == 0.0

    def test_train_with_chat_template_kwargs(self):
        # [FUNCTION DOC]
        # Runs the standard train check on tiny Qwen3 with conversational prompts and chat_template_kwargs
        # enable_thinking False. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
            chat_template_kwargs={"enable_thinking": False},
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_mismatched_reward_processing_classes_length(self):
        # [FUNCTION DOC]
        # Asserts two reward model ids with a single reward processing class raises ValueError containing must match.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Test that mismatched length between reward_funcs and reward_processing_classes raises error."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        # Use two reward models
        reward_models = [
            "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            "trl-internal-testing/tiny-Qwen3ForSequenceClassification",
        ]

        # Create a single processing class (tokenizer)
        single_processing_class = AutoTokenizer.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"
        )

        training_args = RLOOConfig(output_dir=self.tmp_dir, report_to="none")

        with pytest.raises(ValueError, match="must match"):
            RLOOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs=reward_models,
                reward_processing_classes=single_processing_class,  # only one, but need two
                args=training_args,
                train_dataset=dataset,
            )

    def test_correct_reward_processing_classes_list(self):
        # [FUNCTION DOC]
        # Asserts two reward models with a list of two tokenizers yields reward_processing_classes of length 2.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Test that correct list of reward_processing_classes works properly."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        # Use two reward models
        reward_models = [
            "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            "trl-internal-testing/tiny-Qwen3ForSequenceClassification",
        ]

        # Create processing classes
        processing_class1 = AutoTokenizer.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"
        )
        processing_class2 = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen3ForSequenceClassification")

        training_args = RLOOConfig(output_dir=self.tmp_dir, report_to="none")

        # Correct list length should work
        correct_processing_classes = [processing_class1, processing_class2]

        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_models,
            reward_processing_classes=correct_processing_classes,
            args=training_args,
            train_dataset=dataset,
        )

        assert len(trainer.reward_processing_classes) == len(reward_models)

    def test_single_reward_model_with_single_processing_class(self):
        # [FUNCTION DOC]
        # Asserts one reward model with a single tokenizer object yields a one-element list holding that tokenizer.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Test that single reward model with single processing class works."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        # Use single reward model
        reward_model = "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"

        # Create a single processing class (tokenizer)
        single_processing_class = AutoTokenizer.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"
        )

        training_args = RLOOConfig(output_dir=self.tmp_dir, report_to="none")

        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_model,
            reward_processing_classes=single_processing_class,  # single object for single reward model
            args=training_args,
            train_dataset=dataset,
        )

        assert len(trainer.reward_processing_classes) == 1
        assert trainer.reward_processing_classes[0] == single_processing_class

    def test_pad_token_id_synced_with_model_config(self):
        # [FUNCTION DOC]
        # With tiny Mistral (no pad token, eos used instead) and a zero reward, asserts the pad id is set and matches
        # model.config.pad_token_id and model.generation_config.pad_token_id. Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: none.
        # This model's tokenizer has no pad token, so the trainer falls back to the eos token. The model configs must
        # follow: otherwise `Trainer` realigns them at train time and reports it as a change the user did not make.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns 0.0 per completion. Exceptions: none. Side effects: none.
            return [0.0] * len(completions)

        training_args = RLOOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-MistralForCausalLM-0.2",
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
        )

        pad_token_id = trainer.processing_class.pad_token_id
        assert pad_token_id is not None
        assert trainer.model.config.pad_token_id == pad_token_id
        assert trainer.model.generation_config.pad_token_id == pad_token_id

    def test_reward_model_pad_token_id_synced_with_text_config(self):
        # [FUNCTION DOC]
        # With a composite Gemma3 sequence-classification reward model (num_labels 1) and its tokenizer with pad set to
        # eos, asserts the reward pad id is set and equals the reward model text config pad_token_id.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: mutates the reward model config.
        # A composite reward model resolves the pad token through its text config, not the top-level one.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            "trl-internal-testing/tiny-Gemma3ForConditionalGeneration", num_labels=1
        )
        reward_processing_class = AutoTokenizer.from_pretrained(
            "trl-internal-testing/tiny-Gemma3ForConditionalGeneration"
        )
        reward_processing_class.pad_token = reward_processing_class.eos_token

        training_args = RLOOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_model,
            reward_processing_classes=reward_processing_class,
            args=training_args,
            train_dataset=dataset,
        )

        pad_token_id = trainer.reward_processing_classes[0].pad_token_id
        assert pad_token_id is not None
        assert reward_model.config.get_text_config().pad_token_id == pad_token_id


@require_vision
class TestRLOOTrainerVLM(TrlTestCase):
    # [CLASS DOC]
    # RLOOTrainer tests on tiny vision-language models with zen-image / zen-multi-image conversational prompt-only
    # datasets (requires vision extras). Most use a length-based reward on the first message content, lr 0.1, batch
    # 2 and num_generations 2. Uses the standard and PEFT train checks described on TestRLOOTrainer.
    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
            pytest.param(
                "trl-internal-testing/tiny-Gemma4ForConditionalGeneration",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.5.0"),
                    reason="Gemma4 models were introduced in transformers-5.5.0",
                ),
            ),
            "trl-internal-testing/tiny-InternVLForConditionalGeneration",
            "trl-internal-testing/tiny-LlavaNextForConditionalGeneration",
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
            "trl-internal-testing/tiny-Qwen2VLForConditionalGeneration",
            pytest.param(
                "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration-NoThink",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.2.0"),
                    reason="Qwen3.5 models were introduced in transformers-5.2.0",
                ),
            ),
            pytest.param(
                "trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.2.0"),
                    reason="Qwen3.5 models were introduced in transformers-5.2.0",
                ),
            ),
            # "trl-internal-testing/tiny-SmolVLMForConditionalGeneration", seems not to support bf16 properly
        ],
    )
    def test_train_vlm(self, model_id):
        # [FUNCTION DOC]
        # Runs the standard train check for each listed VLM (Gemma3/4, InternVL, LLaVA-Next, Qwen2/2.5-VL, Qwen3.5 dense
        # and MoE; newer ones skipped on old transformers), except that for LLaVA models vision encoder layers.1 and
        # post_layernorm must stay unchanged. Inputs: model_id (str, parametrized). Outputs: none.
        # Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the length of the first message content of each conversational completion as a float.
            # Exceptions: none. Side effects: none.
            """Reward function that rewards longer completions."""
            return [float(len(completion[0]["content"])) for completion in completions]

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=2,  # VLM training is memory intensive, reduce batch size to avoid OOM
            num_generations=2,  # VLM training is memory intensive, reduce num_generations to avoid OOM
            # note: num_generations=2 is only suitable for CI testing; production training should use more generations
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
            model=model_id,
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            # LLaVA & LLaVA-Next: vision_feature_layer=-2 leaves the last encoder layer (layers.1) and
            # post_layernorm (pooler-only path) without gradient by design. Assert they stay frozen — if they
            # ever start training, the feature-selection plumbing has likely regressed.
            if model_id in (
                "trl-internal-testing/tiny-LlavaForConditionalGeneration",
                "trl-internal-testing/tiny-LlavaNextForConditionalGeneration",
            ) and ("encoder.layers.1" in n or "post_layernorm" in n):
                assert torch.equal(param, new_param), f"Param {n} expected frozen by LLaVA design, but changed"
            else:
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_generation_suppresses_image_token(self):
        # [FUNCTION DOC]
        # With Gemma3, asserts generation_config.suppress_tokens is exactly the config image_token_id (not the
        # processor's). Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        # Gemma3: its processor's `image_token_id` is not the placeholder token, unlike its config's
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")
        training_args = RLOOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        assert trainer.generation_config.suppress_tokens == [trainer.model.config.image_token_id]

    def test_train_vlm_with_pad_to_multiple_of(self):
        # [FUNCTION DOC]
        # Runs the standard train check on Gemma3 with pad_to_multiple_of 7, verifying extra inputs such as
        # token_type_ids are padded too. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        # Models like Gemma3 use other forward keyword arguments like token_type_ids that also need to be padded when
        # using pad_to_multiple_of, so we test that the trainer correctly pads all the necessary inputs in this case.
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the length of the first message content of each conversational completion as a float.
            # Exceptions: none. Side effects: none.
            """Reward function that rewards longer completions."""
            return [float(len(completion[0]["content"])) for completion in completions]

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=2,  # VLM training is memory intensive, reduce batch size to avoid OOM
            num_generations=2,  # VLM training is memory intensive, reduce num_generations to avoid OOM
            # note: num_generations=2 is only suitable for CI testing; production training should use more generations
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            pad_to_multiple_of=7,
            report_to="none",
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
        ],
    )
    def test_train_vlm_beta_non_zero(self, model_id):
        # [FUNCTION DOC]
        # Runs the standard train check on Qwen2.5-VL with beta 0.1 (reference model used).
        # Inputs: model_id (str, parametrized). Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the length of the first message content of each conversational completion as a float.
            # Exceptions: none. Side effects: none.
            """Reward function that rewards longer completions."""
            return [float(len(completion[0]["content"])) for completion in completions]

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            beta=0.1,  # set beta to non-zero value to test the case where the reference model is used
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=2,  # VLM training is memory intensive, reduce batch size to avoid OOM
            num_generations=2,  # VLM training is memory intensive, reduce num_generations to avoid OOM
            # note: num_generations=2 is only suitable for CI testing; production training should use more generations
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
            model=model_id,
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
        ],
    )
    @require_peft
    def test_train_vlm_peft(self, model_id):
        # [FUNCTION DOC]
        # Runs the PEFT train check on a float32 Qwen2.5-VL instance with LoRA on q_proj and v_proj. Requires peft.
        # Inputs: model_id (str, parametrized). Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the length of the first message content of each conversational completion as a float.
            # Exceptions: none. Side effects: none.
            """Reward function that rewards longer completions."""
            return [float(len(completion[0]["content"])) for completion in completions]

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=2,  # VLM training is memory intensive, reduce batch size to avoid OOM
            num_generations=2,  # VLM training is memory intensive, reduce num_generations to avoid OOM
            # note: num_generations=2 is only suitable for CI testing; production training should use more generations
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
            model=model,
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(target_modules=["q_proj", "v_proj"]),
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "base_layer" not in n:  # We expect the peft params to be different (except for the base layer)
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
        ],
    )
    @require_peft
    @require_bitsandbytes
    def test_train_vlm_peft_and_quantization(self, model_id):
        # [FUNCTION DOC]
        # Loads Qwen2.5-VL by id in 4-bit NF4 (bf16 compute) with LoRA on q_proj and v_proj; asserts 4-bit loading and
        # that LoRA parameters are bfloat16 and changed. Requires peft and bitsandbytes.
        # Inputs: model_id (str, parametrized). Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the length of the first message content of each conversational completion as a float.
            # Exceptions: none. Side effects: none.
            """Reward function that rewards longer completions."""
            return [float(len(completion[0]["content"])) for completion in completions]

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=2,  # VLM training is memory intensive, reduce batch size to avoid OOM
            num_generations=2,  # VLM training is memory intensive, reduce num_generations to avoid OOM
            # note: num_generations=2 is only suitable for CI testing; production training should use more generations
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        trainer = RLOOTrainer(
            model=model_id,  # pass the model as an identifier, so that the trainer applies the quantization config
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
            quantization_config=quantization_config,
            peft_config=LoraConfig(target_modules=["q_proj", "v_proj"]),
        )

        # Check that the trainer applied the quantization config when loading the model
        assert trainer.model.base_model.model.is_loaded_in_4bit

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed, and that they are cast to bfloat16, as recommended by the QLoRA
        # paper. The base model params are not checked: bitsandbytes casts the biases of a Linear4bit in-place during
        # the forward pass, so some of them change in a way that is unrelated to training.
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if "lora" in n:  # We expect the peft params to be different
                assert param.dtype == torch.bfloat16, f"Parameter {n} is not in bfloat16."
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
        ],
    )
    @require_peft
    @require_bitsandbytes
    def test_train_vlm_peft_dora_and_quantization(self, model_id):
        # [FUNCTION DOC]
        # Same as the 4-bit VLM LoRA test with DoRA: magnitude vectors stay float32 and change, other LoRA parameters are
        # bfloat16 and change. Inputs: model_id (str, parametrized). Outputs: none. Exceptions: AssertionError.
        # Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the length of the first message content of each conversational completion as a float.
            # Exceptions: none. Side effects: none.
            """Reward function that rewards longer completions."""
            return [float(len(completion[0]["content"])) for completion in completions]

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=2,  # VLM training is memory intensive, reduce batch size to avoid OOM
            num_generations=2,  # VLM training is memory intensive, reduce num_generations to avoid OOM
            # note: num_generations=2 is only suitable for CI testing; production training should use more generations
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        trainer = RLOOTrainer(
            model=model_id,  # pass the model as an identifier, so that the trainer applies the quantization config
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
            quantization_config=quantization_config,
            peft_config=LoraConfig(target_modules=["q_proj", "v_proj"], use_dora=True),
        )

        # Check that the trainer applied the quantization config when loading the model
        assert trainer.model.base_model.model.is_loaded_in_4bit

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed, and that they are cast to bfloat16, as recommended by the QLoRA
        # paper. The base model params are not checked: bitsandbytes casts the biases of a Linear4bit in-place during
        # the forward pass, so some of them change in a way that is unrelated to training.
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            # The DoRA magnitude vector is the exception: it must stay in float32, since its optimizer updates can be
            # smaller than bfloat16 can represent, which would otherwise silently freeze it, see #7268
            if "lora_magnitude_vector" in n:
                assert param.dtype == torch.float32, f"Parameter {n} is not in float32."
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."
            elif "lora" in n:  # We expect the peft params to be different
                assert param.dtype == torch.bfloat16, f"Parameter {n} is not in bfloat16."
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
            "trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
            pytest.param(
                "trl-internal-testing/tiny-Gemma4ForConditionalGeneration",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.5.0"),
                    reason="Gemma4 models were introduced in transformers-5.5.0",
                ),
            ),
        ],
    )
    @require_vllm
    @pytest.mark.skip(reason="We should add a mock for the vLLM server.")
    def test_train_vlm_and_vllm(self, model_id) -> None:
        # [FUNCTION DOC]
        # Currently skipped (needs a vLLM server mock). Would run the standard train check for Qwen2.5-VL, Gemma3 and
        # Gemma4 using vLLM in server mode. Inputs: model_id (str, parametrized). Outputs: none.
        # Exceptions: AssertionError. Side effects: none while skipped.
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the length of the first message content of each conversational completion as a float.
            # Exceptions: none. Side effects: none.
            """Reward function that rewards longer completions."""
            return [float(len(completion[0]["content"])) for completion in completions]

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=2,  # VLM training is memory intensive, reduce batch size to avoid OOM
            num_generations=2,  # VLM training is memory intensive, reduce num_generations to avoid OOM
            # note: num_generations=2 is only suitable for CI testing; production training should use more generations
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
            use_vllm=True,
            vllm_mode="server",
        )
        trainer = RLOOTrainer(
            model=model_id,
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Idefics3ForConditionalGeneration",
            "trl-internal-testing/tiny-LlavaForConditionalGeneration",
            "trl-internal-testing/tiny-LlavaNextForConditionalGeneration",
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
        ],
    )
    def test_train_vlm_multi_image(self, model_id):
        # [FUNCTION DOC]
        # On the multi-image dataset (some samples without images) for Idefics3, LLaVA, LLaVA-Next and Qwen2.5-VL:
        # asserts ddp_find_unused_parameters is enabled, then the standard train check with the LLaVA frozen-layer
        # exception. Inputs: model_id (str, parametrized). Outputs: none. Exceptions: AssertionError.
        # Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen-multi-image", "conversational_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the length of the first message content of each conversational completion as a float.
            # Exceptions: none. Side effects: none.
            """Reward function that rewards longer completions."""
            return [float(len(completion[0]["content"])) for completion in completions]

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=2,  # VLM training is memory intensive, reduce batch size to avoid OOM
            num_generations=2,  # VLM training is memory intensive, reduce num_generations to avoid OOM
            # note: num_generations=2 is only suitable for CI testing; production training should use more generations
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = RLOOTrainer(
            model=model_id,
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
        )
        # Some samples have no image: their batches leave the vision tower without gradients
        assert trainer.args.ddp_find_unused_parameters

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            # LLaVA & LLaVA-Next: vision_feature_layer=-2 leaves the last encoder layer (layers.1) and
            # post_layernorm (pooler-only path) without gradient by design. Assert they stay frozen — if they
            # ever start training, the feature-selection plumbing has likely regressed.
            if model_id in (
                "trl-internal-testing/tiny-LlavaForConditionalGeneration",
                "trl-internal-testing/tiny-LlavaNextForConditionalGeneration",
            ) and ("encoder.layers.1" in n or "post_layernorm" in n):
                assert torch.equal(param, new_param), f"Param {n} expected frozen by LLaVA design, but changed"
            else:
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_vlm_log_multimodal_false(self):
        # [FUNCTION DOC]
        # Trains Gemma3 with log_completions True and log_multimodal False and asserts no images were logged in the
        # trainer's internal logs. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: writes tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the length of the first message content of each conversational completion as a float.
            # Exceptions: none. Side effects: none.
            """Reward function that rewards longer completions."""
            return [float(len(completion[0]["content"])) for completion in completions]

        training_args = RLOOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,  # VLM training is memory intensive, reduce batch size to avoid OOM
            num_generations=2,  # VLM training is memory intensive, reduce num_generations to avoid OOM
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
            log_completions=True,
            log_multimodal=False,
        )
        trainer = RLOOTrainer(
            model="trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()
        assert len(trainer._logs["images"]) == 0
