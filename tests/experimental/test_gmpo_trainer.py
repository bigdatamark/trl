# [MODULE DOC]
# Tests for the experimental GMPO (Geometric-Mean Policy Optimization) config and trainer
# (trl.experimental.gmpo). Covers config defaults (log-space clipping epsilon), basic training,
# eval-dataset handling, conversational prompts, and the KL-regularized path. Uses tiny Qwen2 policy
# and sequence-classification reward models and the "zen" prompt-only datasets.
# Import-time side effects: none.

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

import pytest
import torch
from datasets import DatasetDict, load_dataset

from trl.experimental.gmpo import GMPOConfig, GMPOTrainer

from ..testing_utils import TrlTestCase


class TestGMPOConfig:
    # [CLASS DOC]
    # Unit tests for GMPOConfig defaults; no model is loaded. Plain pytest class (no TrlTestCase).
    def test_default_epsilon_is_log_space(self):
        # [FUNCTION DOC]
        # Tests that GMPOConfig defaults epsilon to 0.4 (log-space clip range) and leaves the inherited
        # epsilon_high as None (symmetric range).
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        # GMPO expresses the clip range in log space; default is the paper's (exp(-0.4), exp(0.4)).
        args = GMPOConfig("dummy")
        assert args.epsilon == 0.4
        # epsilon_high is inherited from GRPOConfig and defaults to None, so the range is symmetric.
        assert args.epsilon_high is None


class TestGMPOTrainer(TrlTestCase):
    # [CLASS DOC]
    # Integration tests for GMPOTrainer. Inherits TrlTestCase (self.tmp_dir for outputs). Each test
    # builds a GMPOConfig with tiny batch/generation/completion sizes and a GMPOTrainer from hub model ids.
    def test_train(self):
        # [FUNCTION DOC]
        # Tests that training on standard prompt-only data (lr 0.1, batch 3, 3 generations, 8 completion
        # tokens, num_iterations 2) logs a train_loss and changes every model parameter.
        # Setup: snapshot all named parameters before training. Assertions: final log has train_loss; each
        # parameter differs from its snapshot.
        # Exceptions: AssertionError. Side effects: writes to tmp_dir; downloads models.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GMPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_iterations=2,  # the importance sampling weights won't be 0 in this case
            report_to="none",
        )
        trainer = GMPOTrainer(
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

    @pytest.mark.parametrize("eval_dataset_type", ["dataset", "dataset_dict", "dict_of_dataset", "none"])
    def test_init_with_eval_dataset(self, eval_dataset_type):
        # [FUNCTION DOC]
        # Parametrized over eval_dataset_type in dataset, dataset_dict, dict_of_dataset, none. Tests that the
        # trainer stores eval_dataset correctly: None stays None; DatasetDict or dict becomes a dict with
        # keys data1 and data2; a single Dataset is stored as the identical object. No training is run.
        # Inputs: eval_dataset_type (str). Outputs: none. Exceptions: AssertionError. Side effects: downloads.
        # Streaming datasets are not yet supported in GMPO
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        if eval_dataset_type == "none":
            eval_dataset = None
        elif eval_dataset_type == "dataset":
            eval_dataset = dataset["test"]
        elif eval_dataset_type == "dataset_dict":
            eval_dataset = DatasetDict({"data1": dataset["test"], "data2": dataset["test"]})
        else:  # "dict_of_dataset"
            eval_dataset = {"data1": dataset["test"], "data2": dataset["test"]}

        training_args = GMPOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = GMPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=eval_dataset,
        )

        if eval_dataset_type == "none":
            assert trainer.eval_dataset is None
        elif isinstance(trainer.eval_dataset, dict):
            assert set(trainer.eval_dataset.keys()) == {"data1", "data2"}
        else:
            assert trainer.eval_dataset is eval_dataset

    @pytest.mark.parametrize("config_name", ["standard_prompt_only", "conversational_prompt_only"])
    def test_train_conversational(self, config_name):
        # [FUNCTION DOC]
        # Parametrized over standard and conversational prompt-only configs; tests that training completes
        # and logs a train_loss with the small GMPO settings (num_iterations 2).
        # Inputs: config_name (str). Outputs: none. Exceptions: AssertionError. Side effects: tmp_dir writes.
        dataset = load_dataset("trl-internal-testing/zen", config_name, split="train")

        training_args = GMPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_iterations=2,  # the importance sampling weights won't be 0 in this case
            report_to="none",
        )
        trainer = GMPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_train_with_kl(self):
        # [FUNCTION DOC]
        # Tests the KL-regularized path (beta=0.1): training logs train_loss, the final log entry contains
        # a "kl" key, and every parameter changes from its pre-training snapshot.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: tmp_dir writes; downloads.
        # GMPO sequence-averages the KL when beta > 0; exercise that path.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GMPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            beta=0.1,  # enable KL regularization toward the reference model
            report_to="none",
        )
        trainer = GMPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert "kl" in trainer.state.log_history[-1]  # KL is logged when beta > 0

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."
