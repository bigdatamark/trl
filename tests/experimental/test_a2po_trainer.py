# [MODULE DOC]
# Tests for the experimental A2PO trainer (two-stage: Stage 1 estimates an optimal value V* per prompt by
# sampling, Stage 2 regresses on one on-policy generation), using the tiny Qwen2.5 causal LM and tiny
# in-memory prompt datasets.
# Contents: completion_parity_reward helper and TestA2POTrainer.
# Side effects at import: none beyond imports; tests download tiny models from the Hub and write to temp dirs.

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
from datasets import Dataset, DatasetDict

from trl.experimental.a2po import A2POConfig, A2POTrainer

from ..testing_utils import TrlTestCase


def completion_parity_reward(completions, **kwargs):
    # [FUNCTION DOC]
    # HELPER: Binary reward that varies by completion: 1.0 when the completion length is even, else 0.0.
    # Inputs: completions (list), extra kwargs ignored. Outputs: list of float. Exceptions: none. Side effects: none.
    """Completion-dependent binary reward, so samples within a prompt vary (nonzero advantages)."""
    return [float(len(completion) % 2 == 0) for completion in completions]


class TestA2POTrainer(TrlTestCase):
    # [CLASS DOC]
    # Test class for A2POTrainer; inherits TrlTestCase, which provides a per-test temporary output dir (tmp_dir).
    # Each test builds a real A2POConfig and A2POTrainer with report_to disabled.
    def test_trust_remote_code(self):
        # [FUNCTION DOC]
        # Checks remote-code gating: building the trainer from tiny-RemoteForCausalLM without trust_remote_code raises
        # ValueError mentioning "custom code"; with trust_remote_code set it loads and the model class name is
        # RemoteForCausalLM.
        dataset = Dataset.from_dict({"prompt": ["The capital of France is", "Two plus two equals"]})
        model_id = "trl-internal-testing/tiny-RemoteForCausalLM"

        with pytest.raises(ValueError, match="custom code"):
            A2POTrainer(
                model=model_id,
                reward_funcs=completion_parity_reward,
                args=A2POConfig(output_dir=self.tmp_dir, report_to="none"),
                train_dataset=dataset,
            )

        trainer = A2POTrainer(
            model=model_id,
            reward_funcs=completion_parity_reward,
            args=A2POConfig(output_dir=self.tmp_dir, report_to="none", trust_remote_code=True),
            train_dataset=dataset,
        )
        assert type(trainer.model).__name__ == "RemoteForCausalLM"

    def test_train(self):
        # [FUNCTION DOC]
        # End-to-end smoke test: 4 prompts, lr 0.1, batch size 2, max completion length 8, 2 value samples, no
        # all-incorrect filtering. Asserts the last log has train_loss, the cached _optimal_values has one entry per
        # prompt, and every parameter changed compared with a pre-training clone.
        # Main two-stage smoke test: Stage 1 estimates V*, Stage 2 regresses on a single on-policy generation.
        dataset = Dataset.from_dict(
            {"prompt": ["The capital of France is", "Two plus two equals", "Water is made of", "The sky is"]}
        )
        training_args = A2POConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=2,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_value_samples=2,  # reduce Stage 1 sampling to reduce memory usage
            filter_all_incorrect=False,  # keep all training prompts (the dummy reward may score a prompt all-zero)
            report_to="none",
        )
        trainer = A2POTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=completion_parity_reward,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert trainer._optimal_values is not None and len(trainer._optimal_values) == len(dataset)

        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize("eval_dataset_type", ["dataset", "dataset_dict", "dict_of_dataset", "none"])
    def test_init_with_eval_dataset(self, eval_dataset_type):
        # [FUNCTION DOC]
        # Parametrized over eval dataset forms (single Dataset, DatasetDict, plain dict of Datasets, None). Asserts the
        # trainer keeps None as None, exposes dict forms with keys data1 and data2, and keeps a single Dataset by identity.
        train_dataset = Dataset.from_dict(
            {"prompt": ["The capital of France is", "Two plus two equals", "Water is made of", "The sky is"]}
        )
        eval_split = Dataset.from_dict({"prompt": ["The capital of Italy is", "Three plus three equals"]})

        if eval_dataset_type == "none":
            eval_dataset = None
        elif eval_dataset_type == "dataset":
            eval_dataset = eval_split
        elif eval_dataset_type == "dataset_dict":
            eval_dataset = DatasetDict({"data1": eval_split, "data2": eval_split})
        else:  # "dict_of_dataset"
            eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = A2POConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = A2POTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=completion_parity_reward,
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

    def test_reward_kwargs_are_forwarded(self):
        # [FUNCTION DOC]
        # Regression test: an extra dataset column (subject) must be forwarded to the reward function. Trains with a
        # reward that records the received subject values and returns all ones; asserts something was received and
        # every value belongs to the dataset's subject set.
        # Regression test: extra dataset columns must reach the reward function (e.g. a verifier needs `solution`).
        received_subjects = []

        def reward_using_extra_column(prompts, completions, subject, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Reward that appends the subject column values to the enclosing received_subjects list and returns 1.0
            # per prompt. Side effects: mutates received_subjects.
            received_subjects.extend(subject)
            return [1.0] * len(prompts)

        dataset = Dataset.from_dict({"prompt": ["q1", "q2", "q3", "q4"], "subject": ["math", "geo", "chem", "phys"]})
        training_args = A2POConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=2,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_value_samples=2,  # reduce Stage 1 sampling to reduce memory usage
            filter_all_incorrect=False,  # keep all training prompts (the dummy reward may score a prompt all-zero)
            report_to="none",
        )
        trainer = A2POTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_using_extra_column,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()

        # Stage 1 repeats each prompt's column value `num_value_samples` times; the column must have been forwarded.
        assert received_subjects, "reward function never received the `subject` column"
        assert set(received_subjects) <= {"math", "geo", "chem", "phys"}

    def test_filter_all_incorrect_drops_prompts(self):
        # [FUNCTION DOC]
        # Regression test for filter_all_incorrect: with a reward equal to a per-row target column (1, 0, 1), the
        # prompt whose Stage 1 samples all score zero must be removed rather than causing a KeyError in Stage 2.
        # Asserts training completes, the train dataset has 2 rows and "unsolvable" is absent.
        # Regression test: prompts whose reference samples all score zero must be dropped, not crash Stage 2.
        def reward_from_target(prompts, completions, target, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the target column values cast to float, ignoring prompts and completions. Side effects: none.
            return [float(t) for t in target]

        dataset = Dataset.from_dict({"prompt": ["solvable a", "unsolvable", "solvable b"], "target": [1, 0, 1]})
        training_args = A2POConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=2,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_value_samples=2,  # reduce Stage 1 sampling to reduce memory usage
            filter_all_incorrect=True,
            report_to="none",
        )
        trainer = A2POTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_from_target,
            args=training_args,
            train_dataset=dataset,
        )

        # Should run without raising a KeyError for the dropped prompt.
        trainer.train()

        assert len(trainer.train_dataset) == 2
        assert "unsolvable" not in list(trainer.train_dataset["prompt"])

    def test_evaluate(self):
        # [FUNCTION DOC]
        # Regression test: calling evaluate without prior training works and computes V* for unseen eval prompts.
        # Asserts eval_loss is in the metrics and both eval prompts (gamma, delta) are keys of _optimal_values.
        # Regression test: eval prompts (never seen in training) must also get a cached V*, and evaluating
        # without a preceding train() must not raise.
        train_dataset = Dataset.from_dict({"prompt": ["alpha", "beta"]})
        eval_dataset = Dataset.from_dict({"prompt": ["gamma", "delta"]})
        training_args = A2POConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=2,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_value_samples=2,  # reduce Stage 1 sampling to reduce memory usage
            filter_all_incorrect=False,  # keep all training prompts (the dummy reward may score a prompt all-zero)
            report_to="none",
        )
        trainer = A2POTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=completion_parity_reward,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        metrics = trainer.evaluate()

        assert "eval_loss" in metrics
        assert "gamma" in trainer._optimal_values and "delta" in trainer._optimal_values
