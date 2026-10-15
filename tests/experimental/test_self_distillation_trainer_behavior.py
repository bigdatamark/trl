# [MODULE DOC]
# Unit/behavior tests for the experimental self-distillation (SDFT) trainer in trl.experimental.sdft.
# Main contents: one test class covering the standalone loss helpers in trl.experimental.sdft.loss_utils
# (full-logit, sampled-token and top-k self-distillation losses, importance-sampling clipping), the teacher
# selection logic of SDFTTrainer (teacher_model_kind live/base/ema), the warning for pre-wrapped PEFT students,
# the PEFTAdapterEMACallback teacher-adapter EMA update, and SDFTTrainer._compute_self_distillation_loss masking
# and clipping plus its metric logging. Uses the tiny Qwen2 test model and a one-prompt dataset.
# Import-time side effects: imports peft (LoraConfig, get_peft_model, get_peft_model_state_dict) only if available.

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
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, TrainerControl, TrainerState, TrainingArguments
from transformers.utils import is_peft_available

from trl.experimental.sdft import SDFTConfig, SDFTTrainer
from trl.experimental.sdft.loss_utils import (
    apply_importance_sampling_clipping,
    compute_full_logit_self_distillation_loss,
    compute_sampled_token_self_distillation_loss,
    compute_topk_self_distillation_loss,
)
from trl.experimental.sdft.teacher_sync import PEFTAdapterEMACallback

from ..testing_utils import TrlTestCase


if is_peft_available():
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict


class TestSelfDistillationTrainerBehavior(TrlTestCase):
    # [CLASS DOC]
    # Test case grouping SDFT loss-function and trainer-construction tests.
    # Inherits TrlTestCase (provides self.tmp_dir as a per-test output directory).
    # State: none beyond the inherited temp dir; each test builds its own tensors/trainer.
    def _make_loss_test_trainer(self, **args_overrides):
        # [FUNCTION DOC]
        # HELPER: Builds a minimal SDFTTrainer for exercising _compute_self_distillation_loss.
        # Behavior: starts from default loss args (distillation_mode sampled_token, distillation_topk None,
        # distillation_alpha 1.0, distillation_add_tail False, distillation_is_clip None), overrides them with the
        # keyword args, builds an SDFTConfig (output_dir self.tmp_dir, report_to none) and an SDFTTrainer on the tiny
        # Qwen2 model with a single-prompt dataset, then puts the model in train mode so metrics land under "train".
        # Inputs: args_overrides - any SDFTConfig distillation fields to override.
        # Outputs: the constructed SDFTTrainer.
        # Exceptions: none expected (model download/config errors propagate).
        # Side effects: loads the tiny model from the Hub/cache; sets the model to training mode.
        args = {
            "distillation_mode": "sampled_token",
            "distillation_topk": None,
            "distillation_alpha": 1.0,
            "distillation_add_tail": False,
            "distillation_is_clip": None,
        }
        args.update(args_overrides)
        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=SDFTConfig(output_dir=self.tmp_dir, report_to="none", **args),
            train_dataset=Dataset.from_dict({"prompt": ["Solve 2+2."]}),
        )
        trainer.model.train()  # the loss is logged under the mode of the model it is computed for
        return trainer

    def test_full_logit_loss_matches_forward_kl(self):
        # [FUNCTION DOC]
        # Checks that compute_full_logit_self_distillation_loss with distillation_alpha 0.0 equals the forward KL
        # (teacher || student) summed over the vocabulary.
        # Setup: one-token, two-class student probs (0.8, 0.2) and teacher probs (0.5, 0.5), passed as log probs.
        # Assertion: loss equals sum of teacher_p times (log teacher_p minus log student_p), shaped (1, 1).
        # Inputs/Outputs: none. Exceptions: assertion failure on mismatch. Side effects: none.
        student_probs = torch.tensor([[[0.8, 0.2]]], dtype=torch.float32)
        teacher_probs = torch.tensor([[[0.5, 0.5]]], dtype=torch.float32)

        loss = compute_full_logit_self_distillation_loss(
            student_probs.log(),
            teacher_probs.log(),
            distillation_alpha=0.0,
        )

        expected_loss = teacher_probs[0, 0, 0] * (
            teacher_probs[0, 0, 0].log() - student_probs[0, 0, 0].log()
        ) + teacher_probs[0, 0, 1] * (teacher_probs[0, 0, 1].log() - student_probs[0, 0, 1].log())
        torch.testing.assert_close(loss, expected_loss.reshape(1, 1))

    def test_sampled_token_loss_uses_selected_completion_ids(self):
        # [FUNCTION DOC]
        # Checks that compute_sampled_token_self_distillation_loss gathers log probs at the sampled completion ids.
        # Setup: two-token student/teacher distributions over two classes, completion ids (1, 0), distillation_alpha 1.0.
        # Assertion: per-token loss equals (student_logp minus teacher_logp) multiplied by student_logp at the chosen
        # tokens (student probs 0.9/0.7, teacher 0.6/0.2), i.e. the reverse-KL style sampled estimator.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        student_probs = torch.tensor([[[0.1, 0.9], [0.7, 0.3]]], dtype=torch.float32)
        teacher_probs = torch.tensor([[[0.4, 0.6], [0.2, 0.8]]], dtype=torch.float32)
        completion_ids = torch.tensor([[1, 0]])

        loss = compute_sampled_token_self_distillation_loss(
            student_probs.log(),
            teacher_probs.log(),
            completion_ids,
            distillation_alpha=1.0,
        )

        expected_student_logps = torch.tensor([[0.9, 0.7]], dtype=torch.float32).log()
        expected_teacher_logps = torch.tensor([[0.6, 0.2]], dtype=torch.float32).log()
        expected_loss = (expected_student_logps - expected_teacher_logps) * expected_student_logps
        torch.testing.assert_close(loss, expected_loss)

    def test_topk_loss_renormalizes_selected_student_support(self):
        # [FUNCTION DOC]
        # Checks compute_topk_self_distillation_loss without a tail bucket.
        # Setup: student (0.5, 0.3, 0.2), teacher (0.2, 0.6, 0.2), distillation_topk 2, alpha 0.0, add_tail False.
        # Assertion: loss equals forward KL between the teacher and student restricted to the student's top-2 indices
        # (0 and 1), each renormalized to sum to one; result shape (1, 1).
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        student_probs = torch.tensor([[[0.5, 0.3, 0.2]]], dtype=torch.float32)
        teacher_probs = torch.tensor([[[0.2, 0.6, 0.2]]], dtype=torch.float32)

        loss = compute_topk_self_distillation_loss(
            student_probs.log(),
            teacher_probs.log(),
            distillation_topk=2,
            distillation_alpha=0.0,
            distillation_add_tail=False,
        )

        student_topk = torch.tensor([0.5, 0.3], dtype=torch.float32)
        student_topk = student_topk / student_topk.sum()
        teacher_topk = torch.tensor([0.2, 0.6], dtype=torch.float32)
        teacher_topk = teacher_topk / teacher_topk.sum()
        expected_loss = (teacher_topk * (teacher_topk.log() - student_topk.log())).sum()
        torch.testing.assert_close(loss, expected_loss.reshape(1, 1))

    def test_topk_loss_can_include_tail_bucket(self):
        # [FUNCTION DOC]
        # Checks compute_topk_self_distillation_loss with distillation_add_tail True.
        # Setup: same distributions as the renormalization test, topk 2, alpha 0.0.
        # Assertion: with the remaining mass collapsed into a tail bucket, the loss equals the full three-way forward KL
        # (the tail here is exactly the third class), shape (1, 1).
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        student_probs = torch.tensor([[[0.5, 0.3, 0.2]]], dtype=torch.float32)
        teacher_probs = torch.tensor([[[0.2, 0.6, 0.2]]], dtype=torch.float32)

        loss = compute_topk_self_distillation_loss(
            student_probs.log(),
            teacher_probs.log(),
            distillation_topk=2,
            distillation_alpha=0.0,
            distillation_add_tail=True,
        )

        student_with_tail = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32)
        teacher_with_tail = torch.tensor([0.2, 0.6, 0.2], dtype=torch.float32)
        expected_loss = (teacher_with_tail * (teacher_with_tail.log() - student_with_tail.log())).sum()
        torch.testing.assert_close(loss, expected_loss.reshape(1, 1))

    def test_importance_sampling_clipping_caps_token_ratio(self):
        # [FUNCTION DOC]
        # Checks apply_importance_sampling_clipping multiplies per-token loss by min(student/old ratio, clip_coeff).
        # Setup: per-token loss (1, 2), student probs (0.4, 0.3), old probs (0.1, 0.2), clip_coeff 2.0.
        # Assertion: ratios 4.0 and 1.5 become 2.0 and 1.5, giving (2.0, 3.0).
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        per_token_loss = torch.tensor([[1.0, 2.0]])
        student_log_probs = torch.tensor([[0.4, 0.3]], dtype=torch.float32).log()
        old_log_probs = torch.tensor([[0.1, 0.2]], dtype=torch.float32).log()

        loss = apply_importance_sampling_clipping(
            per_token_loss,
            student_log_probs,
            old_log_probs,
            clip_coeff=2.0,
        )

        torch.testing.assert_close(loss, torch.tensor([[2.0, 3.0]]))

    def test_teacher_model_kind_live_uses_student_model(self):
        # [FUNCTION DOC]
        # Checks that teacher_model_kind "live" makes the trainer reuse the student as teacher.
        # Setup: SDFTConfig with batch size 1, max_completion_length 8, max_steps 1, num_generations 1; tiny Qwen2 model.
        # Assertion: trainer.teacher_model is the same object as trainer.model.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: loads the tiny model.
        dataset = Dataset.from_dict({"prompt": ["Solve 2+2."]})
        training_args = SDFTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            teacher_model_kind="live",
            report_to="none",
        )

        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        assert trainer.teacher_model is trainer.model

    @pytest.mark.skipif(not is_peft_available(), reason="PEFT is required for this test")
    def test_warns_when_initial_student_already_has_a_peft_adapter(self, caplog):
        # [FUNCTION DOC]
        # Checks that passing an already PEFT-wrapped student with teacher_model_kind "base" logs a warning.
        # Setup: tiny Qwen2 wrapped with a LoRA adapter (r 4, alpha 8, q_proj/v_proj); trainer constructed while caplog
        # captures WARNING on logger trl.experimental.sdft.sdft_trainer. Skipped when PEFT is unavailable.
        # Assertions: captured text mentions "already contains a PEFT adapter" and that base may refer to underlying
        # base weights.
        # Inputs: caplog - pytest log capture fixture. Outputs: none. Exceptions: assertion failure.
        # Side effects: loads the model.
        dataset = Dataset.from_dict({"prompt": ["Solve 2+2."]})
        training_args = SDFTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            teacher_model_kind="base",
            report_to="none",
        )
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        model = get_peft_model(
            model,
            LoraConfig(
                r=4,
                lora_alpha=8,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )

        with caplog.at_level(logging.WARNING, logger="trl.experimental.sdft.sdft_trainer"):
            SDFTTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
            )

        assert "already contains a PEFT adapter" in caplog.text
        assert "`teacher_model_kind='base'` may refer to the underlying base weights" in caplog.text

    @pytest.mark.skipif(not is_peft_available(), reason="PEFT is required for this test")
    def test_peft_adapter_ema_callback_updates_teacher_adapter(self):
        # [FUNCTION DOC]
        # Checks PEFTAdapterEMACallback creates a zero-initialized "teacher" adapter and EMA-updates it.
        # Setup: tiny Qwen2 on CPU with LoRA adapter "default" (r 8, q_proj/v_proj); callback with update_rate 0.5,
        # sync_steps 1; real TrainingArguments, TrainerState (global_step 0) and TrainerControl.
        # Steps/assertions: after on_train_begin, "teacher" exists in peft_config, shadow_weights is set, every teacher
        # adapter tensor is zero, and shadow_weights keys match the student adapter state keys. After setting global_step
        # to 1 and calling on_step_end, each shadow weight equals update_rate times the student weight (EMA from zero),
        # and the teacher adapter state (cast to float) equals the shadow weights.
        # Inputs/Outputs: none. Exceptions: assertion failure. Skipped without PEFT. Side effects: loads the model.
        model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            device_map="cpu",
        )
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj"],
                r=8,
            ),
            adapter_name="default",
        )

        update_rate = 0.5
        callback = PEFTAdapterEMACallback(
            model=model,
            teacher_adapter_name="teacher",
            update_rate=update_rate,
            sync_steps=1,
        )
        args = TrainingArguments(output_dir=self.tmp_dir, report_to="none")
        state = TrainerState(global_step=0)
        control = TrainerControl()

        callback.on_train_begin(args, state, control)

        assert "teacher" in model.peft_config
        assert callback.shadow_weights is not None
        teacher_state = get_peft_model_state_dict(model, adapter_name="teacher")
        for key, param in teacher_state.items():
            assert torch.all(param == 0), f"Teacher param {key} should be zero-initialized"

        student_state = {
            key: value.clone() for key, value in get_peft_model_state_dict(model, adapter_name="default").items()
        }
        assert set(callback.shadow_weights.keys()) == set(student_state.keys())

        state.global_step = 1
        callback.on_step_end(args, state, control)

        for key in callback.shadow_weights:
            expected = update_rate * student_state[key]
            torch.testing.assert_close(callback.shadow_weights[key], expected)

        teacher_state = get_peft_model_state_dict(model, adapter_name="teacher")
        for key in teacher_state:
            torch.testing.assert_close(teacher_state[key].float(), callback.shadow_weights[key])

    @pytest.mark.parametrize("teacher_model_kind", ["base", "ema"])
    def test_teacher_model_kind_base_and_ema_use_frozen_teacher_copy(self, teacher_model_kind):
        # [FUNCTION DOC]
        # Parametrized over teacher_model_kind "base" and "ema": checks the teacher is a separate frozen copy.
        # Setup: same small SDFTConfig as the live test with the given teacher kind.
        # Assertions: teacher_model is not the student, is in eval mode, its first parameter does not require grad and
        # does not share storage with the student's first parameter.
        # Inputs: teacher_model_kind - parametrized string. Outputs: none. Exceptions: assertion failure.
        # Side effects: loads the model.
        dataset = Dataset.from_dict({"prompt": ["Solve 2+2."]})
        training_args = SDFTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            teacher_model_kind=teacher_model_kind,
            report_to="none",
        )

        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        assert trainer.teacher_model is not trainer.model
        assert trainer.teacher_model.training is False

        student_param = next(trainer.model.parameters())
        teacher_param = next(trainer.teacher_model.parameters())
        assert teacher_param.requires_grad is False
        assert teacher_param.data_ptr() != student_param.data_ptr()

    def test_compute_self_distillation_loss_ignores_masked_completion_tokens(self):
        # [FUNCTION DOC]
        # Checks SDFTTrainer._compute_self_distillation_loss averages only over unmasked tokens and logs the metric.
        # Setup: trainer via the helper with full_logits mode and alpha 0.0; a SimpleNamespace stand-in for the
        # distillation outputs with two completion tokens, loss_mask (1, 0), and student/teacher log probs.
        # Assertions: returned loss equals the forward KL of the first (unmasked) token only; the
        # "self_distillation/distillation_loss" list in trainer._metrics["train"] holds exactly that value.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: loads the model; writes trainer metrics.
        trainer = self._make_loss_test_trainer(
            distillation_mode="full_logits",
            distillation_alpha=0.0,
        )
        model = trainer.model

        student_probs = torch.tensor([[[0.8, 0.2], [0.01, 0.99]]], dtype=torch.float32)
        teacher_probs = torch.tensor([[[0.5, 0.5], [0.99, 0.01]]], dtype=torch.float32)
        distillation_logits = SimpleNamespace(
            completion_ids=torch.tensor([[0, 1]], dtype=torch.long),
            loss_mask=torch.tensor([[1, 0]], dtype=torch.long),
            student_logits=student_probs.log(),
            teacher_logits=teacher_probs.log(),
        )

        loss = trainer._compute_self_distillation_loss(model, {}, distillation_logits)

        expected_active_token_loss = teacher_probs[0, 0, 0] * (
            teacher_probs[0, 0, 0].log() - student_probs[0, 0, 0].log()
        ) + teacher_probs[0, 0, 1] * (teacher_probs[0, 0, 1].log() - student_probs[0, 0, 1].log())
        torch.testing.assert_close(loss, expected_active_token_loss)
        torch.testing.assert_close(
            torch.tensor(trainer._metrics["train"]["self_distillation/distillation_loss"]),
            expected_active_token_loss.unsqueeze(0),
        )

    def test_compute_self_distillation_loss_applies_importance_sampling_clip(self):
        # [FUNCTION DOC]
        # Checks _compute_self_distillation_loss applies importance-sampling clipping when distillation_is_clip is set.
        # Setup: helper trainer (sampled_token mode, alpha 1.0) with distillation_is_clip 2.0; stand-in outputs with
        # completion ids (0, 1), full mask, student token probs 0.2/0.4, teacher 0.5/0.5; inputs carry
        # old_per_token_logps from probs 0.05/0.4.
        # Assertions: loss equals the mean over tokens of the sampled-token loss times min(student/old, 2.0), and the
        # logged train metric "self_distillation/distillation_loss" equals that value.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: loads the model; writes trainer metrics.
        trainer = self._make_loss_test_trainer(distillation_is_clip=2.0)
        model = trainer.model

        student_token_probs = torch.tensor([[0.2, 0.4]], dtype=torch.float32)
        teacher_token_probs = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        old_token_probs = torch.tensor([[0.05, 0.4]], dtype=torch.float32)
        clip_coeff = trainer.args.distillation_is_clip

        distillation_logits = SimpleNamespace(
            completion_ids=torch.tensor([[0, 1]], dtype=torch.long),
            loss_mask=torch.tensor([[1, 1]], dtype=torch.long),
            student_logits=torch.log(torch.tensor([[[0.2, 0.8], [0.6, 0.4]]], dtype=torch.float32)),
            teacher_logits=torch.log(torch.tensor([[[0.5, 0.5], [0.5, 0.5]]], dtype=torch.float32)),
        )

        loss = trainer._compute_self_distillation_loss(
            model,
            {"old_per_token_logps": old_token_probs.log()},
            distillation_logits,
        )

        raw_per_token_loss = (student_token_probs.log() - teacher_token_probs.log()) * student_token_probs.log()
        clipped_ratio = torch.minimum(
            student_token_probs / old_token_probs, torch.full_like(student_token_probs, clip_coeff)
        )
        expected_loss = (raw_per_token_loss * clipped_ratio).mean()

        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(
            torch.tensor(trainer._metrics["train"]["self_distillation/distillation_loss"]),
            expected_loss.unsqueeze(0),
        )
