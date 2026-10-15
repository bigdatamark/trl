# [MODULE DOC]
# Tests for the experimental SDPOTrainer / SDPOConfig (trl.experimental.sdpo), self-distillation
# policy optimization: GRPO-style rollouts whose teacher is reprompted with a successful sibling
# solution and/or environment feedback (privileged_context). Covers remote-code gating, positional
# args, config defaults and CLI parsing, training, teacher construction, eval datasets, old-logprob
# capture, eval grouping, reprompt templating, warnings, conversational context, teacher masks, the
# teacher-server loss path, and pad tokens. Defines a capture callback and a fake teacher client.
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

import logging

import pytest
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoModelForCausalLM, HfArgumentParser, TrainerCallback

from trl.experimental.sdpo import SDPOConfig, SDPOTrainer

from ..testing_utils import TrlTestCase


class SelfDistillationCaptureCallback(TrainerCallback):
    # [CLASS DOC]
    # HELPER: TrainerCallback recording what SDPO emits via its custom events: decoded teacher inputs
    # (first row and all rows), first teacher attention mask, completion mask, self-distillation mask,
    # and first old_per_token_logps (all detached to CPU).
    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Initializes all captured_* fields to None, except captured_teacher_input_texts to an
        # empty list. Inputs: none. Outputs: none. Exceptions: none. Side effects: sets attributes.
        self.captured_teacher_input_text = None
        self.captured_teacher_input_texts = []
        self.captured_self_distillation_mask = None
        self.captured_teacher_attention_mask = None
        self.captured_completion_mask = None
        self.captured_old_per_token_logps = None

    def on_teacher_context_built(
        self,
        processing_class=None,
        teacher_input_ids=None,
        teacher_attention_mask=None,
        completion_mask=None,
        self_distillation_mask=None,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # HELPER: When teacher_input_ids is given: stores the decoded first row (skip special tokens) once,
        # and appends decoded text of every row to captured_teacher_input_texts on each call. Stores
        # teacher_attention_mask, completion_mask, and self_distillation_mask (detached CPU) on first
        # non-None value. Inputs: processing_class, teacher_input_ids, teacher_attention_mask,
        # completion_mask, self_distillation_mask (all optional), kwargs. Outputs: None.
        # Exceptions: AttributeError if ids are passed without processing_class. Side effects: mutates self.
        if self.captured_teacher_input_text is None and teacher_input_ids is not None:
            self.captured_teacher_input_text = processing_class.decode(teacher_input_ids[0], skip_special_tokens=True)
        if teacher_input_ids is not None:
            self.captured_teacher_input_texts.extend(
                processing_class.decode(ids, skip_special_tokens=True) for ids in teacher_input_ids
            )
        if self.captured_teacher_attention_mask is None and teacher_attention_mask is not None:
            self.captured_teacher_attention_mask = teacher_attention_mask.detach().cpu()
        if self.captured_completion_mask is None and completion_mask is not None:
            self.captured_completion_mask = completion_mask.detach().cpu()
        if self.captured_self_distillation_mask is None and self_distillation_mask is not None:
            self.captured_self_distillation_mask = self_distillation_mask.detach().cpu()

    def on_self_distillation_batch_prepared(self, old_per_token_logps=None, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Stores old_per_token_logps (detached CPU) on first non-None value.
        # Inputs: old_per_token_logps (optional), kwargs. Outputs: None. Exceptions: none.
        # Side effects: mutates self.
        if self.captured_old_per_token_logps is None and old_per_token_logps is not None:
            self.captured_old_per_token_logps = old_per_token_logps.detach().cpu()


class RecordingTeacherClient:
    # [CLASS DOC]
    # HELPER: Fake vLLM teacher-server client (remote server cannot run in tests). Returns a fixed
    # response from get_sequence_logprobs and records each call's kwargs in self.calls.
    """Stands in for the vLLM server client and records scoring requests."""

    def __init__(self, response):
        # [FUNCTION DOC]
        # HELPER: Stores response and initializes an empty calls list. Inputs: response (dict).
        # Outputs: none. Exceptions: none. Side effects: sets attributes.
        self.response = response
        self.calls = []

    def get_sequence_logprobs(self, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Appends kwargs to self.calls and returns self.response. Inputs: kwargs. Outputs: the
        # stored response. Exceptions: none. Side effects: mutates self.calls.
        self.calls.append(kwargs)
        return self.response


class TestSDPOTrainer(TrlTestCase):
    # [CLASS DOC]
    # Integration tests for SDPOTrainer; inherits TrlTestCase. Most tests use tiny Qwen2, tiny
    # hand-built datasets, 1 step, small generation groups, and inline reward functions.
    def test_trust_remote_code(self):
        # [FUNCTION DOC]
        # Tests a remote-code model raises ValueError matching "custom code" without trust_remote_code and
        # loads as RemoteForCausalLM with it (zero-reward lambda).
        # Inputs/Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models; writes to tmp_dir.
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2.", "Name the capital of France."],
                "privileged_context": ["Example answer: 4.", "Example answer: Paris."],
            }
        )
        model_id = "trl-internal-testing/tiny-RemoteForCausalLM"

        with pytest.raises(ValueError, match="custom code"):
            SDPOTrainer(
                model=model_id,
                reward_funcs=lambda **kwargs: [0.0] * len(kwargs["prompts"]),
                args=SDPOConfig(output_dir=self.tmp_dir, report_to="none"),
                train_dataset=dataset,
            )

        trainer = SDPOTrainer(
            model=model_id,
            reward_funcs=lambda **kwargs: [0.0] * len(kwargs["prompts"]),
            args=SDPOConfig(output_dir=self.tmp_dir, report_to="none", trust_remote_code=True),
            train_dataset=dataset,
        )
        assert type(trainer.model).__name__ == "RemoteForCausalLM"

    def test_train_with_positional_config_argument(self):
        # [FUNCTION DOC]
        # Tests passing model, reward_funcs, args, train_dataset positionally: 1-step training with 2
        # generations and include_environment_feedback True works, args are preserved, train_loss logged.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models; writes to tmp_dir.
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2."],
                "privileged_context": ["Your earlier answer used the wrong format."],
            }
        )

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            generation_batch_size=2,
            num_generations=2,
            max_completion_length=8,
            include_environment_feedback=True,
            max_steps=1,
            report_to="none",
        )

        trainer = SDPOTrainer(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            lambda **kwargs: [0.0] * len(kwargs["prompts"]),
            training_args,
            dataset,
        )

        trainer.train()

        assert trainer.args.output_dir == self.tmp_dir
        assert trainer.args.include_environment_feedback is True
        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_vllm_config_defaults_match_reference_trainers(self):
        # [FUNCTION DOC]
        # Tests SDPOConfig defaults vllm_mode colocate and vllm_model_impl vllm.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        config = SDPOConfig(output_dir=self.tmp_dir)

        assert config.vllm_mode == "colocate"
        assert config.vllm_model_impl == "vllm"

    def test_train(self):
        # [FUNCTION DOC]
        # Trains on zen prompt-only data with a tiny reward model, topk_logits distillation (topk 5, no IS
        # clip), batch 3, 3 generations; asserts train_loss logged and every nonzero-sum parameter changed.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models; writes to tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            distillation_mode="topk_logits",
            distillation_topk=5,
            distillation_is_clip=None,
            report_to="none",
        )
        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_teacher_model_like_model_instance(self):
        # [FUNCTION DOC]
        # Tests that a student instance in bfloat16 with eager attention yields a teacher with the same
        # dtype and attention implementation.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models; writes to tmp_dir.
        # A student passed as an instance has no `model_init_kwargs`: the teacher is built with its dtype and attention
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", dtype=torch.bfloat16, attn_implementation="eager"
        )
        training_args = SDPOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SDPOTrainer(
            model=model,
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        assert trainer.teacher_model.dtype == torch.bfloat16
        assert trainer.teacher_model.config._attn_implementation == "eager"

    @pytest.mark.parametrize("eval_dataset_type", ["dataset", "dataset_dict", "dict_of_dataset", "none"])
    def test_init_with_eval_dataset(self, eval_dataset_type):
        # [FUNCTION DOC]
        # Parametrized over dataset, dataset_dict, dict_of_dataset, none. Tests None stays None; dict-like
        # becomes a dict with keys data1/data2; a single Dataset is the same object.
        # Inputs: eval_dataset_type (str). Outputs: none. Exceptions: AssertionError.
        # Side effects: downloads; tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        if eval_dataset_type == "none":
            eval_dataset = None
        elif eval_dataset_type == "dataset":
            eval_dataset = dataset["test"]
        elif eval_dataset_type == "dataset_dict":
            eval_dataset = DatasetDict({"data1": dataset["test"], "data2": dataset["test"]})
        else:  # "dict_of_dataset"
            eval_dataset = {"data1": dataset["test"], "data2": dataset["test"]}

        training_args = SDPOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SDPOTrainer(
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

    def test_train_without_successful_rollouts(self):
        # [FUNCTION DOC]
        # Tests training with an always-zero reward (no successful rollouts) still completes and logs
        # train_loss.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models; writes to tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            distillation_is_clip=None,
            report_to="none",
        )

        def zero_reward(**kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns 0.0 for every prompt. Inputs: kwargs with prompts. Outputs: list[float].
            # Exceptions: KeyError if prompts missing. Side effects: none.
            prompts = kwargs["prompts"]
            return [0.0] * len(prompts)

        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=zero_reward,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_train_populates_old_log_probs_for_distillation_clipping_when_misaligned(self):
        # [FUNCTION DOC]
        # Tests gradient_accumulation_steps 3 vs steps_per_generation 2 leads to old_per_token_logps being
        # present in the self-distillation batch.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models; writes to tmp_dir.
        dataset = Dataset.from_dict({"prompt": ["Solve 2+2.", "Solve 3+3."]})

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=3,
            steps_per_generation=2,
            num_generations=2,
            max_completion_length=8,
            max_steps=1,
            report_to="none",
        )

        capture_callback = SelfDistillationCaptureCallback()
        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=lambda **kwargs: [0.0] * len(kwargs["prompts"]),
            args=training_args,
            train_dataset=dataset,
            callbacks=[capture_callback],
        )

        trainer.train()

        assert capture_callback.captured_old_per_token_logps is not None

    def test_evaluation_uses_num_generations_eval_for_teacher_grouping(self):
        # [FUNCTION DOC]
        # Tests evaluate() groups by num_generations_eval (2) rather than num_generations (3): eval batch of
        # 4 prompts yields Alpha x2 and Beta x2; the reward marks only the first Alpha sample successful.
        # Asserts teacher inputs were captured for Alpha and Beta, some Alpha teacher input contains
        # "Correct solution:", and no Beta teacher input does.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models; writes to tmp_dir.
        eval_dataset = Dataset.from_dict({"prompt": ["Alpha prompt", "Beta prompt", "Gamma prompt", "Delta prompt"]})

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=4,
            generation_batch_size=3,
            num_generations=3,
            num_generations_eval=2,
            max_completion_length=8,
            success_reward_threshold=0.5,
            dont_reprompt_on_self_success=False,
            distillation_is_clip=None,
            max_steps=1,
            report_to="none",
        )

        def eval_rewards(**kwargs):
            # [FUNCTION DOC]
            # HELPER: If prompts has length 4 with exactly two Alpha and two Beta prompts, returns
            # [1.0, 0.0, 0.0, 0.0]; otherwise all zeros. Inputs: kwargs with prompts. Outputs: list[float].
            # Exceptions: KeyError if prompts missing. Side effects: none.
            prompts = kwargs["prompts"]
            if len(prompts) == 4 and prompts.count("Alpha prompt") == 2 and prompts.count("Beta prompt") == 2:
                return [1.0, 0.0, 0.0, 0.0]
            return [0.0] * len(prompts)

        capture_callback = SelfDistillationCaptureCallback()
        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=eval_rewards,
            args=training_args,
            train_dataset=eval_dataset.select(range(1)),
            eval_dataset=eval_dataset,
            callbacks=[capture_callback],
        )

        trainer.evaluate()

        assert capture_callback.captured_teacher_input_texts
        alpha_teachers = [text for text in capture_callback.captured_teacher_input_texts if "Alpha prompt" in text]
        beta_teachers = [text for text in capture_callback.captured_teacher_input_texts if "Beta prompt" in text]
        assert alpha_teachers
        assert beta_teachers
        assert any("Correct solution:" in text for text in alpha_teachers)
        assert all("Correct solution:" not in text for text in beta_teachers)

    def test_teacher_reprompt_preserves_curly_braces_in_solution_and_feedback(self):
        # [FUNCTION DOC]
        # Tests reprompt templating keeps literal curly braces: the teacher input contains the prompt with
        # braces and the feedback with a JSON-like brace snippet, and no doubled braces appear.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models; writes to tmp_dir.
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve f(x) = {x^2}."],
                "privileged_context": ['Feedback: use {"x": 2} as a check.'],
            }
        )

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            generation_batch_size=2,
            num_generations=2,
            max_completion_length=8,
            include_environment_feedback=True,
            success_reward_threshold=0.5,
            dont_reprompt_on_self_success=False,
            max_steps=1,
            report_to="none",
        )

        def reward_with_one_success(**kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the first len(prompts) items of [1.0, 0.0], so only the first sample of a
            # 2-sample group succeeds. Inputs: kwargs with prompts. Outputs: list[float]. Exceptions: KeyError if
            # prompts missing. Side effects: none.
            prompts = kwargs["prompts"]
            return [1.0, 0.0][: len(prompts)]

        capture_callback = SelfDistillationCaptureCallback()
        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_with_one_success,
            args=training_args,
            train_dataset=dataset,
            callbacks=[capture_callback],
        )

        trainer.train()

        assert capture_callback.captured_teacher_input_text is not None
        assert "Solve f(x) = {x^2}." in capture_callback.captured_teacher_input_text
        assert 'Feedback: use {"x": 2} as a check.' in capture_callback.captured_teacher_input_text
        assert "{{" not in capture_callback.captured_teacher_input_text
        assert "}}" not in capture_callback.captured_teacher_input_text

    def test_scale_rewards_accepts_string_via_cli(self):
        # [FUNCTION DOC]
        # Tests HfArgumentParser parses --scale_rewards batch and none as the strings batch and none.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        parser = HfArgumentParser((SDPOConfig,))
        (args,) = parser.parse_args_into_dataclasses(["--output_dir", self.tmp_dir, "--scale_rewards", "batch"])
        assert args.scale_rewards == "batch"
        (args,) = parser.parse_args_into_dataclasses(["--output_dir", self.tmp_dir, "--scale_rewards", "none"])
        assert args.scale_rewards == "none"

    def test_warns_when_feedback_available_but_disabled(self, caplog):
        # [FUNCTION DOC]
        # Tests that training on data with privileged_context while include_environment_feedback is False
        # logs a warning (logger trl.experimental.sdpo.sdpo_trainer) mentioning include_environment_feedback.
        # Inputs: caplog (pytest fixture). Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models; writes to tmp_dir.
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2."],
                "privileged_context": ["The correct answer is 4."],
            }
        )
        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            generation_batch_size=2,
            num_generations=2,
            max_completion_length=8,
            include_environment_feedback=False,
            max_steps=1,
            report_to="none",
        )

        def reward(**kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns 0.0 for every prompt. Inputs: kwargs with prompts. Outputs: list[float].
            # Exceptions: KeyError if prompts missing. Side effects: none.
            return [0.0] * len(kwargs["prompts"])

        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward,
            args=training_args,
            train_dataset=dataset,
        )
        with caplog.at_level(logging.WARNING, logger="trl.experimental.sdpo.sdpo_trainer"):
            trainer.train()

        assert any("include_environment_feedback" in record.message for record in caplog.records)

    def test_train_with_conversational_prompts_preserves_context(self):
        # [FUNCTION DOC]
        # Tests conversational prompts with a system message: with only sample 0 succeeding and the default
        # dont_reprompt_on_self_success, sample 1 is reprompted; the teacher input keeps the system text
        # and user question, and a self-distillation mask is produced.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models; writes to tmp_dir.
        dataset = Dataset.from_dict(
            {
                "prompt": [
                    [
                        {"role": "system", "content": "You are a careful assistant."},
                        {"role": "user", "content": "Solve 2+2."},
                    ]
                ]
            }
        )

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            generation_batch_size=2,
            num_generations=2,
            max_completion_length=8,
            distillation_is_clip=None,
            success_reward_threshold=0.5,
            max_steps=1,
            report_to="none",
        )

        def first_only_reward(**kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the first len(prompts) items of [1.0, 0.0], so only the first sample of a
            # 2-sample group succeeds. Inputs: kwargs with prompts. Outputs: list[float]. Exceptions: KeyError if
            # prompts missing. Side effects: none.
            """Only the first sample in each group succeeds — exercises dont_reprompt_on_self_success default."""
            return [1.0, 0.0][: len(kwargs["prompts"])]

        capture_callback = SelfDistillationCaptureCallback()
        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=first_only_reward,
            args=training_args,
            train_dataset=dataset,
            callbacks=[capture_callback],
        )

        trainer.train()

        # With dont_reprompt_on_self_success=True (default), sample 0 skips itself,
        # but sample 1 finds sample 0's success and gets a teacher reprompt.
        assert capture_callback.captured_teacher_input_text is not None
        assert "careful assistant" in capture_callback.captured_teacher_input_text
        assert "Solve 2+2" in capture_callback.captured_teacher_input_text
        assert capture_callback.captured_self_distillation_mask is not None

    def test_train_with_feedback_only_reprompts_teacher(self):
        # [FUNCTION DOC]
        # Tests that with zero rewards but include_environment_feedback True, the teacher is reprompted
        # using the feedback text (format requirements appears) and row 0's self-distillation mask is 1.0.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models; writes to tmp_dir.
        dataset = Dataset.from_dict(
            {
                "prompt": [
                    [
                        {"role": "system", "content": "You are a careful assistant."},
                        {"role": "user", "content": "Try the puzzle again."},
                    ]
                ],
                "privileged_context": ["Your earlier answer violated the format requirements."],
            }
        )

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            generation_batch_size=2,
            num_generations=2,
            max_completion_length=8,
            distillation_is_clip=None,
            include_environment_feedback=True,
            max_steps=1,
            report_to="none",
        )

        def zero_reward(**kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns 0.0 for every prompt. Inputs: kwargs with prompts. Outputs: list[float].
            # Exceptions: KeyError if prompts missing. Side effects: none.
            prompts = kwargs["prompts"]
            return [0.0] * len(prompts)

        capture_callback = SelfDistillationCaptureCallback()
        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=zero_reward,
            args=training_args,
            train_dataset=dataset,
            callbacks=[capture_callback],
        )

        trainer.train()

        assert capture_callback.captured_teacher_input_text is not None
        assert "format requirements" in capture_callback.captured_teacher_input_text
        assert capture_callback.captured_self_distillation_mask is not None
        assert capture_callback.captured_self_distillation_mask[0].item() == 1.0

    def test_train_warns_when_sdpo_rewards_are_flat(self, caplog):
        # [FUNCTION DOC]
        # Tests that with zero rewards over 2 steps and diagnostics_warning_interval 2, warnings about flat
        # SDPO rewards and inactive self-distillation (no reprompted samples) are logged.
        # Inputs: caplog (pytest fixture). Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models; writes to tmp_dir.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            diagnostics_warning_interval=2,
            max_steps=2,
            report_to="none",
        )

        def zero_reward(**kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns 0.0 for every prompt. Inputs: kwargs with prompts. Outputs: list[float].
            # Exceptions: KeyError if prompts missing. Side effects: none.
            return [0.0] * len(kwargs["prompts"])

        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=zero_reward,
            args=training_args,
            train_dataset=dataset,
        )

        with caplog.at_level(logging.WARNING):
            trainer.train()

        assert "Observed flat SDPO rewards across all sampled generations" in caplog.text
        assert "SDPO self-distillation is inactive because no reprompted samples were constructed" in caplog.text

    def test_train_preserves_teacher_completion_attention_mask(self):
        # [FUNCTION DOC]
        # Tests that the last completion-length columns of the teacher attention mask for row 0 equal row 0
        # of the completion mask (teacher completion masking matches the student's).
        # Inputs/Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models; writes to tmp_dir.
        dataset = Dataset.from_dict({"prompt": ["Solve 2+2."]})

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            generation_batch_size=2,
            num_generations=2,
            max_completion_length=8,
            success_reward_threshold=0.5,
            max_steps=1,
            report_to="none",
        )

        def first_only_reward(**kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the first len(prompts) items of [1.0, 0.0], so only the first sample of a
            # 2-sample group succeeds. Inputs: kwargs with prompts. Outputs: list[float]. Exceptions: KeyError if
            # prompts missing. Side effects: none.
            return [1.0, 0.0][: len(kwargs["prompts"])]

        capture_callback = SelfDistillationCaptureCallback()
        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=first_only_reward,
            args=training_args,
            train_dataset=dataset,
            callbacks=[capture_callback],
        )

        trainer.train()

        assert capture_callback.captured_teacher_attention_mask is not None
        assert capture_callback.captured_completion_mask is not None

        completion_length = capture_callback.captured_completion_mask.shape[1]
        teacher_completion_attention = capture_callback.captured_teacher_attention_mask[0, -completion_length:]
        assert torch.equal(teacher_completion_attention, capture_callback.captured_completion_mask[0])

    def test_server_loss_finite_with_masked_and_padded_rows(self):
        # [FUNCTION DOC]
        # Drives compute_loss through the teacher-server path: sets use_teacher_server and installs a
        # RecordingTeacherClient returning ragged top-k logprobs (row 0 empty, row 1 two tokens). Config:
        # topk_logits distillation, topk 2, alpha 0.5, add_tail, no IS clipping, 3 completion tokens. Builds a
        # hand-made 2-row batch where row 0's completion is fully masked and row 1 is padded. Asserts the
        # loss is finite, backward yields finite grads, and the client was called with top_logprobs 2.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models; writes to tmp_dir.
        # Drives the teacher-server path through `compute_loss` with a fake server client: row 0 is fully masked
        # (zero-length scored completion) and row 1 has a shorter completion than the padded batch, so the client
        # response is ragged and the padded tail comes back as -inf. Neither may leak NaN or inf.
        dataset = Dataset.from_dict({"prompt": ["Solve 2+2."], "privileged_context": ["Example answer: 4."]})
        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,
            max_completion_length=3,
            num_generations=1,
            distillation_mode="topk_logits",
            distillation_topk=2,
            distillation_alpha=0.5,
            distillation_add_tail=True,
            distillation_is_clip=None,
            report_to="none",
        )
        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=lambda **kwargs: [0.0] * len(kwargs["prompts"]),
            args=training_args,
            train_dataset=dataset,
        )
        trainer.use_teacher_server = True
        trainer.teacher_client = RecordingTeacherClient(
            {
                "actual_logprobs": [[], [[-1.1], [-0.4]]],
                "logprobs": [[], [[-1.1, -1.5], [-0.4, -0.9]]],
                "logprob_token_ids": [[], [[14, 15], [16, 17]]],
            }
        )

        device = next(trainer.model.parameters()).device
        batch = {
            "prompt_ids": torch.tensor([[10, 11], [12, 13]], device=device),
            "prompt_mask": torch.tensor([[1, 1], [1, 1]], device=device),
            "completion_ids": torch.tensor([[14, 15, 16], [17, 18, 19]], device=device),
            "completion_mask": torch.tensor([[0, 0, 0], [1, 1, 0]], device=device),
            "teacher_input_ids": torch.tensor([[20, 21, 22, 14, 15, 16], [23, 24, 25, 17, 18, 19]], device=device),
            "teacher_attention_mask": torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]], device=device),
        }

        loss = trainer.compute_loss(trainer.model, batch)

        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None)
        assert trainer.teacher_client.calls[0]["top_logprobs"] == 2

    def test_pad_token_id_synced_with_model_config(self):
        # [FUNCTION DOC]
        # Tests with tiny Mistral (no pad token) that the trainer sets a pad token and model config and
        # generation config pad_token_id equal it.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models; writes to tmp_dir.
        # This model's tokenizer has no pad token, so the trainer falls back to the eos token. The model configs must
        # follow: otherwise `Trainer` realigns them at train time and reports it as a change the user did not make.
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2.", "Name the capital of France."],
                "privileged_context": ["Example answer: 4.", "Example answer: Paris."],
            }
        )

        training_args = SDPOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-MistralForCausalLM-0.2",
            reward_funcs=lambda **kwargs: [0.0] * len(kwargs["prompts"]),
            args=training_args,
            train_dataset=dataset,
        )

        pad_token_id = trainer.processing_class.pad_token_id
        assert pad_token_id is not None
        assert trainer.model.config.pad_token_id == pad_token_id
        assert trainer.model.generation_config.pad_token_id == pad_token_id
