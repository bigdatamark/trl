# [MODULE DOC]
# Tests for the experimental SDFTTrainer / SDFTConfig (trl.experimental.sdft), self-distillation
# fine-tuning where a teacher sees the prompt plus a privileged_context column. Covers remote-code
# gating, training, teacher construction, eval datasets, privileged_context validation,
# generate_from_teacher, chat_template_kwargs, PEFT (incl. EMA teacher sync), old-logprob capture for
# distillation clipping, generation-batch buffering, the teacher-server loss path, and pad tokens.
# Defines a capture callback and a fake teacher server client.
# Import-time side effects: imports peft LoraConfig when peft is available.

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
from transformers import AutoModelForCausalLM, TrainerCallback
from transformers.utils import is_peft_available

from trl.experimental.sdft import SDFTConfig, SDFTTrainer

from ..testing_utils import TrlTestCase, require_peft


if is_peft_available():
    from peft import LoraConfig


class SelfDistillationCaptureCallback(TrainerCallback):
    # [CLASS DOC]
    # HELPER: TrainerCallback that records data emitted by SDFT's custom callback events: the first
    # non-None generation_prompts, the first old_per_token_logps and prompt_ids (detached, on CPU), and a
    # count of generation batches built.
    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Initializes captured_generation_prompts, captured_old_per_token_logps, captured_prompt_ids
        # to None and generation_batch_build_count to 0. Inputs: none. Outputs: none. Exceptions: none.
        # Side effects: sets attributes.
        self.captured_generation_prompts = None
        self.captured_old_per_token_logps = None
        self.captured_prompt_ids = None
        self.generation_batch_build_count = 0

    def on_generation_prompts_selected(self, generation_prompts=None, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Stores generation_prompts on first non-None call; later calls are ignored.
        # Inputs: generation_prompts (optional), kwargs. Outputs: None. Exceptions: none.
        # Side effects: mutates self.
        if self.captured_generation_prompts is None and generation_prompts is not None:
            self.captured_generation_prompts = generation_prompts

    def on_self_distillation_batch_prepared(self, old_per_token_logps=None, prompt_ids=None, **kwargs):
        # [FUNCTION DOC]
        # HELPER: On first non-None value each, stores old_per_token_logps and prompt_ids as detached CPU
        # tensors. Inputs: old_per_token_logps, prompt_ids (tensors, optional), kwargs. Outputs: None.
        # Exceptions: none. Side effects: mutates self.
        if self.captured_old_per_token_logps is None and old_per_token_logps is not None:
            self.captured_old_per_token_logps = old_per_token_logps.detach().cpu()
        if self.captured_prompt_ids is None and prompt_ids is not None:
            self.captured_prompt_ids = prompt_ids.detach().cpu()

    def on_generation_batch_built(self, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Increments generation_batch_build_count. Inputs: kwargs. Outputs: None. Exceptions: none.
        # Side effects: mutates self.
        self.generation_batch_build_count += 1


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


class TestSDFTTrainer(TrlTestCase):
    # [CLASS DOC]
    # Integration tests for SDFTTrainer; inherits TrlTestCase. Provides two static helpers to snapshot
    # trainable params and assert at least one changed.
    @staticmethod
    def _trainable_param_snapshot(model):
        # [FUNCTION DOC]
        # HELPER: Returns a dict of name to detached clone for each parameter with requires_grad.
        # Inputs: model. Outputs: dict[str, Tensor]. Exceptions: none. Side effects: none.
        return {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}

    @staticmethod
    def _assert_any_trainable_param_changed(model, previous_trainable_params):
        # [FUNCTION DOC]
        # HELPER: Asserts at least one snapshot parameter is not allclose (rtol=atol=1e-12) to the model's
        # current parameter of the same name. Inputs: model, previous_trainable_params (dict).
        # Outputs: None. Exceptions: AssertionError if none changed. Side effects: none.
        assert any(
            not torch.allclose(previous_param, model.get_parameter(name), rtol=1e-12, atol=1e-12)
            for name, previous_param in previous_trainable_params.items()
        )

    def test_trust_remote_code(self):
        # [FUNCTION DOC]
        # Tests a remote-code model raises ValueError matching "custom code" without trust_remote_code and
        # loads as RemoteForCausalLM with it.
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
            SDFTTrainer(
                model=model_id,
                args=SDFTConfig(output_dir=self.tmp_dir, report_to="none"),
                train_dataset=dataset,
            )

        trainer = SDFTTrainer(
            model=model_id,
            args=SDFTConfig(output_dir=self.tmp_dir, report_to="none", trust_remote_code=True),
            train_dataset=dataset,
        )
        assert type(trainer.model).__name__ == "RemoteForCausalLM"

    def test_train(self):
        # [FUNCTION DOC]
        # Uses a tiny 2-row prompt/privileged_context dataset and 1-step training (batch 1, 1 generation, 8 completion tokens, lr 0.1); asserts train_loss logged and some trainable parameter changed.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models; writes to tmp_dir.
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2.", "Name the capital of France."],
                "privileged_context": [
                    "Example answer: 4.",
                    "Example answer: Paris.",
                ],
            }
        )

        training_args = SDFTConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            report_to="none",
        )

        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )
        previous_trainable_params = self._trainable_param_snapshot(trainer.model)

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        self._assert_any_trainable_param_changed(trainer.model, previous_trainable_params)

    def test_teacher_model_like_model_instance(self):
        # [FUNCTION DOC]
        # Tests that when the student is passed as an instance (bfloat16, eager attention), the teacher is
        # built with the same dtype and attention implementation.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models; writes to tmp_dir.
        # A student passed as an instance has no `model_init_kwargs`: the teacher is built with its dtype and attention
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2.", "Name the capital of France."],
                "privileged_context": ["Example answer: 4.", "Example answer: Paris."],
            }
        )
        model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", dtype=torch.bfloat16, attn_implementation="eager"
        )
        training_args = SDFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SDFTTrainer(model=model, args=training_args, train_dataset=dataset)

        assert trainer.teacher_model.dtype == torch.bfloat16
        assert trainer.teacher_model.config._attn_implementation == "eager"

    @pytest.mark.parametrize("eval_dataset_type", ["dataset", "dataset_dict", "dict_of_dataset", "none"])
    def test_init_with_eval_dataset(self, eval_dataset_type):
        # [FUNCTION DOC]
        # Parametrized over dataset, dataset_dict, dict_of_dataset, none. Tests eval datasets are stored as
        # is: None stays None; dict-like becomes a dict with keys data1/data2; a single Dataset is the same
        # object.
        # Inputs: eval_dataset_type (str). Outputs: none. Exceptions: AssertionError.
        # Side effects: downloads; tmp_dir.
        # Streaming datasets are not yet supported in SDFT
        train_dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2.", "Name the capital of France."],
                "privileged_context": ["Example answer: 4.", "Example answer: Paris."],
            }
        )
        eval_split = Dataset.from_dict({"prompt": ["Solve 3+3."], "privileged_context": ["Example answer: 6."]})

        if eval_dataset_type == "none":
            eval_dataset = None
        elif eval_dataset_type == "dataset":
            eval_dataset = eval_split
        elif eval_dataset_type == "dataset_dict":
            eval_dataset = DatasetDict({"data1": eval_split, "data2": eval_split})
        else:  # "dict_of_dataset"
            eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = SDFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # SDFT tokenizes prompts/completions on the fly during generation, so eval datasets are stored as-is.
        if eval_dataset_type == "none":
            assert trainer.eval_dataset is None
        elif isinstance(trainer.eval_dataset, dict):
            assert set(trainer.eval_dataset.keys()) == {"data1", "data2"}
        else:
            assert trainer.eval_dataset is eval_dataset

    def test_train_rejects_none_privileged_context(self):
        # [FUNCTION DOC]
        # Tests training on a row whose privileged_context is None raises ValueError matching
        # "privileged_context must not be None".
        # Inputs/Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models; writes to tmp_dir.
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2."],
                "privileged_context": [None],
            }
        )

        training_args = SDFTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            report_to="none",
        )

        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        with pytest.raises(ValueError, match="`privileged_context` must not be None"):
            trainer.train()

    def test_train_with_generate_from_teacher(self):
        # [FUNCTION DOC]
        # Tests generate_from_teacher=True: the captured generation prompt is the prompt joined to the
        # privileged context with a blank line, while the decoded student prompt_ids contain the prompt but
        # not the teacher hint.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models; writes to tmp_dir.
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2.", "Solve 3+3."],
                "privileged_context": [
                    "Teacher hint: answer with 4 and explain briefly.",
                    "Teacher hint: answer with 6 and explain briefly.",
                ],
            }
        )

        training_args = SDFTConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            generate_from_teacher=True,
            report_to="none",
        )

        capture_callback = SelfDistillationCaptureCallback()
        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            callbacks=[capture_callback],
        )

        trainer.train()

        assert capture_callback.captured_generation_prompts == [
            "Solve 2+2.\n\nTeacher hint: answer with 4 and explain briefly."
        ]
        student_prompt_text = trainer.processing_class.decode(
            capture_callback.captured_prompt_ids[0],
            skip_special_tokens=True,
        )
        assert "Teacher hint" not in student_prompt_text
        assert "Solve 2+2." in student_prompt_text

    def test_train_with_chat_template_kwargs(self):
        # [FUNCTION DOC]
        # Tests training a tiny Qwen3 on conversational prompts with chat_template_kwargs enable_thinking
        # False: train_loss logged and some trainable parameter changed.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models; writes to tmp_dir.
        dataset = Dataset.from_dict(
            {
                "prompt": [
                    [{"role": "user", "content": "Solve 2+2."}],
                    [{"role": "user", "content": "Solve 3+3."}],
                ],
                "privileged_context": [
                    "Teacher hint: answer with 4.",
                    "Teacher hint: answer with 6.",
                ],
            }
        )

        training_args = SDFTConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            chat_template_kwargs={"enable_thinking": False},
            report_to="none",
        )

        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = self._trainable_param_snapshot(trainer.model)

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        self._assert_any_trainable_param_changed(trainer.model, previous_trainable_params)

    @require_peft
    def test_peft_init_is_seeded(self):
        # [FUNCTION DOC]
        # Requires peft. Two trainers with default LoraConfig built after different global seeds must have
        # non-empty, identical lora_A parameters.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models; writes to tmp_dir.
        # Two trainers with the same seed start from the same adapter weights
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2.", "Name the capital of France."],
                "privileged_context": ["Example answer: 4.", "Example answer: Paris."],
            }
        )
        adapters = []
        for global_seed in range(2):
            torch.manual_seed(global_seed)  # a different global RNG state, as in two separate runs
            trainer = SDFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=SDFTConfig(output_dir=self.tmp_dir, report_to="none"),
                train_dataset=dataset,
                peft_config=LoraConfig(),
            )
            adapters.append({n: p.clone() for n, p in trainer.model.named_parameters() if "lora_A" in n})

        assert adapters[0]
        for n, param in adapters[0].items():
            assert torch.equal(param, adapters[1][n]), f"Parameter {n} differs between the two trainers."

    @require_peft
    def test_train_with_peft_model(self):
        # [FUNCTION DOC]
        # Requires peft. Uses a tiny 2-row prompt/privileged_context dataset and 1-step training (batch 1, 1 generation, 8 completion tokens, lr 0.1) with LoRA on q_proj/v_proj; asserts train_loss and some
        # trainable parameter changed.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models; writes to tmp_dir.
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2.", "Name the capital of France."],
                "privileged_context": [
                    "Example answer: 4.",
                    "Example answer: Paris.",
                ],
            }
        )

        training_args = SDFTConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            report_to="none",
        )

        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj"],
            ),
        )

        previous_trainable_params = self._trainable_param_snapshot(trainer.model)

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        self._assert_any_trainable_param_changed(trainer.model, previous_trainable_params)

    @require_peft
    def test_train_with_peft_model_and_ema_teacher_sync(self):
        # [FUNCTION DOC]
        # Requires peft. Like the LoRA test but 2 steps with teacher_model_kind ema, teacher_update_rate
        # 0.05, teacher_sync_steps 1; asserts train_loss and some trainable parameter changed.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models; writes to tmp_dir.
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2.", "Name the capital of France."],
                "privileged_context": [
                    "Example answer: 4.",
                    "Example answer: Paris.",
                ],
            }
        )

        training_args = SDFTConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=2,
            num_generations=1,
            teacher_model_kind="ema",
            teacher_update_rate=0.05,
            teacher_sync_steps=1,
            report_to="none",
        )

        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj"],
            ),
        )
        previous_trainable_params = self._trainable_param_snapshot(trainer.model)

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        self._assert_any_trainable_param_changed(trainer.model, previous_trainable_params)

    def test_train_populates_old_log_probs_for_distillation_clipping_when_misaligned(self):
        # [FUNCTION DOC]
        # Tests that with gradient_accumulation_steps 3 differing from steps_per_generation 2, the batch
        # prepared for self-distillation carries old_per_token_logps (captured is not None).
        # Inputs/Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models; writes to tmp_dir.
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2.", "Solve 3+3."],
                "privileged_context": [
                    "Example answer: 4.",
                    "Example answer: 6.",
                ],
            }
        )

        training_args = SDFTConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=3,
            steps_per_generation=2,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            report_to="none",
        )

        capture_callback = SelfDistillationCaptureCallback()
        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            callbacks=[capture_callback],
        )

        trainer.train()

        assert capture_callback.captured_old_per_token_logps is not None

    def test_train_with_generate_from_teacher_skips_old_log_probs_for_distillation_clipping(self):
        # [FUNCTION DOC]
        # Tests the same misaligned setup with generate_from_teacher=True produces no old_per_token_logps.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models; writes to tmp_dir.
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2.", "Solve 3+3."],
                "privileged_context": [
                    "Teacher hint: answer with 4.",
                    "Teacher hint: answer with 6.",
                ],
            }
        )

        training_args = SDFTConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=3,
            steps_per_generation=2,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            generate_from_teacher=True,
            report_to="none",
        )

        capture_callback = SelfDistillationCaptureCallback()
        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            callbacks=[capture_callback],
        )

        trainer.train()

        assert capture_callback.captured_old_per_token_logps is None

    def test_train_reuses_buffered_generation_batches(self):
        # [FUNCTION DOC]
        # Tests steps_per_generation 2 over 2 max_steps builds the generation batch exactly once.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models; writes to tmp_dir.
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2.", "Solve 3+3."],
                "privileged_context": [
                    "Example answer: 4.",
                    "Example answer: 6.",
                ],
            }
        )

        training_args = SDFTConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            steps_per_generation=2,
            max_completion_length=8,
            max_steps=2,
            num_generations=1,
            report_to="none",
        )

        capture_callback = SelfDistillationCaptureCallback()
        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            callbacks=[capture_callback],
        )

        trainer.train()

        assert capture_callback.generation_batch_build_count == 1

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
        training_args = SDFTConfig(
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
        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
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

        training_args = SDFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-MistralForCausalLM-0.2", args=training_args, train_dataset=dataset
        )

        pad_token_id = trainer.processing_class.pad_token_id
        assert pad_token_id is not None
        assert trainer.model.config.pad_token_id == pad_token_id
        assert trainer.model.generation_config.pad_token_id == pad_token_id
