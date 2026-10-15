# [MODULE DOC]
# Tests for the experimental ServerDistillationTrainer (trl.experimental.server_distillation), which distills a
# student from teacher log probs fetched from a remote vLLM server.
# Main contents: module-level helpers producing config kwargs, canned/ragged teacher server responses and a small
# chat-prompt dataset; config validation tests; a parametrized test of build_teacher_request_inputs; tests that the
# server response is padded with a -inf sentinel; a numerics test of _add_tail_bucket plus _jsd_divergence with
# padding masked; and an end-to-end one-step training test checking finite grad norms with ragged completions.
# The remote teacher is mocked by monkeypatching trl.generation.vllm_client.VLLMClient with a MagicMock.
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

import math
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.experimental.server_distillation import (
    ServerDistillationConfig,
    ServerDistillationTrainer,
    build_teacher_request_inputs,
)
from trl.experimental.server_distillation.server_distillation_trainer import _add_tail_bucket, _jsd_divergence

from ..testing_utils import TrlTestCase


def _make_server_config_kwargs(tmp_path):
    # [FUNCTION DOC]
    # HELPER: Returns the minimal ServerDistillationConfig kwargs for tests.
    # Inputs: tmp_path - output directory (path-like, converted to str).
    # Outputs: dict with output_dir, report_to "none" and teacher_model_server_url "http://localhost:8000".
    # Exceptions: none. Side effects: none.
    return {
        "output_dir": str(tmp_path),
        "report_to": "none",
        "teacher_model_server_url": "http://localhost:8000",
    }


def _ragged_server_response():
    # [FUNCTION DOC]
    # HELPER: Returns a fixed teacher-server response for two sequences with ragged completion lengths (1 and 3).
    # Outputs: dict with logprobs and actual_logprobs (per sequence, per completion token, a one-element top-k list:
    # -2.3 for the first; -1.1, -0.4, -3.0 for the second) and logprob_token_ids (90; 90, 9217, 100).
    # Inputs: none. Exceptions: none. Side effects: none.
    # Two samples with completion lengths 1 and 3 respectively; matches the wire format
    # of VLLMClient.get_sequence_logprobs (per-sample shape (comp_len, top_k=1)).
    return {
        "logprobs": [[[-2.3]], [[-1.1], [-0.4], [-3.0]]],
        "logprob_token_ids": [[[90]], [[90], [9217], [100]]],
        "actual_logprobs": [[[-2.3]], [[-1.1], [-0.4], [-3.0]]],
    }


def _canned_teacher_logprobs(**kwargs):
    # [FUNCTION DOC]
    # HELPER: Fake implementation of VLLMClient.get_sequence_logprobs used as a MagicMock side effect.
    # Behavior: for each sequence with its prompt length, completion length is sequence length minus prompt length;
    # for completion position i it emits top-k log probs all equal to -1.0 minus 0.05 times i (top_k from the
    # top_logprobs kwarg, default 1), the token id at that position repeated top_k times, and a single actual logprob
    # with the same value.
    # Inputs: kwargs - must contain sequences and prompt_lengths; optional top_logprobs.
    # Outputs: dict with logprobs, logprob_token_ids, actual_logprobs (nested lists).
    # Exceptions: KeyError if sequences/prompt_lengths missing; ValueError from strict zip if lengths differ.
    # Side effects: none.
    # Fabricate ragged per-sample logprobs matching the requested sequence shapes.
    sequences = kwargs["sequences"]
    prompt_lengths = kwargs["prompt_lengths"]
    top_k = kwargs.get("top_logprobs", 1)
    logprobs, token_ids, actual = [], [], []
    for seq, plen in zip(sequences, prompt_lengths, strict=True):
        comp_len = len(seq) - plen
        logprobs.append([[-1.0 - 0.05 * i] * top_k for i in range(comp_len)])
        token_ids.append([[int(seq[plen + i])] * top_k for i in range(comp_len)])
        actual.append([[-1.0 - 0.05 * i] for i in range(comp_len)])
    return {"logprobs": logprobs, "logprob_token_ids": token_ids, "actual_logprobs": actual}


def _variable_length_dataset():
    # [FUNCTION DOC]
    # HELPER: Builds a two-row conversational prompt-only Dataset with prompts of different lengths
    # ("What's 2+2?" and "Name three primary colors.").
    # Inputs: none. Outputs: datasets.Dataset with a "prompt" column of message lists. Exceptions/Side effects: none.
    return Dataset.from_list(
        [
            {"prompt": [{"role": "user", "content": "What's 2+2?"}]},
            {"prompt": [{"role": "user", "content": "Name three primary colors."}]},
        ]
    )


def test_config_rejects_reverse_kl_argmax(tmp_path):
    # [FUNCTION DOC]
    # Checks ServerDistillationConfig raises ValueError mentioning that reverse_kl_top_1_mode "argmax" is not
    # supported. Inputs: tmp_path fixture. Outputs: none. Exceptions: test fails if no error. Side effects: none.
    with pytest.raises(ValueError, match="reverse_kl_top_1_mode='argmax' is not supported"):
        ServerDistillationConfig(**_make_server_config_kwargs(tmp_path), reverse_kl_top_1_mode="argmax")


def test_config_rejects_invalid_reverse_kl_top_1_mode(tmp_path):
    # [FUNCTION DOC]
    # Checks ServerDistillationConfig raises ValueError ("reverse_kl_top_1_mode must be one of") for an unknown mode.
    # Inputs: tmp_path fixture. Outputs: none. Exceptions: test fails if no error. Side effects: none.
    with pytest.raises(ValueError, match="reverse_kl_top_1_mode must be one of"):
        ServerDistillationConfig(**_make_server_config_kwargs(tmp_path), reverse_kl_top_1_mode="invalid")


def test_config_rejects_mixed_loss_without_top_1(tmp_path):
    # [FUNCTION DOC]
    # Checks that beta 0.5 combined with loss_top_k 2 raises ValueError ("loss_top_k must be 1 with beta>0").
    # Inputs: tmp_path fixture. Outputs: none. Exceptions: test fails if no error. Side effects: none.
    with pytest.raises(ValueError, match="loss_top_k must be 1 with beta>0"):
        ServerDistillationConfig(**_make_server_config_kwargs(tmp_path), beta=0.5, loss_top_k=2)


def test_config_requires_server_url(tmp_path):
    # [FUNCTION DOC]
    # Checks that omitting teacher_model_server_url raises ValueError naming the missing server URL, even with an
    # otherwise valid beta 0.5 / loss_top_k 1. Inputs: tmp_path fixture. Outputs: none.
    # Exceptions: test fails if no error. Side effects: none.
    kwargs = _make_server_config_kwargs(tmp_path)
    kwargs.pop("teacher_model_server_url")
    with pytest.raises(ValueError, match="teacher_model_server_url must be set for ServerDistillationTrainer"):
        ServerDistillationConfig(**kwargs, beta=0.5, loss_top_k=1)


@pytest.mark.parametrize(
    (
        "input_ids",
        "attention_mask",
        "prompt_attention_mask",
        "labels",
        "expected_sequences",
        "expected_prompt_lengths",
        "expected_completion_lengths",
    ),
    [
        pytest.param(
            torch.tensor([[0, 0, 10, 11, 12, 20, 21], [30, 31, 32, 33, 34, 40, 41]], dtype=torch.long),
            torch.tensor([[0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]], dtype=torch.long),
            None,
            torch.tensor(
                [[-100, -100, -100, -100, -100, 20, 21], [-100, -100, -100, -100, -100, 40, 41]],
                dtype=torch.long,
            ),
            [[10, 11, 12, 20, 21], [30, 31, 32, 33, 34, 40, 41]],
            [3, 5],
            [2, 2],
            id="variable_prompt_lengths",
        ),
        pytest.param(
            torch.tensor(
                [[0, 0, 10, 11, 12, 20, 21, 0, 0], [30, 31, 32, 33, 34, 40, 41, 0, 0]],
                dtype=torch.long,
            ),
            torch.tensor(
                [[0, 0, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0]],
                dtype=torch.long,
            ),
            torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.long),
            torch.tensor(
                [
                    [-100, -100, -100, -100, -100, 20, 21, -100, -100],
                    [-100, -100, -100, -100, -100, 40, 41, -100, -100],
                ],
                dtype=torch.long,
            ),
            [[10, 11, 12, 20, 21], [30, 31, 32, 33, 34, 40, 41]],
            [3, 5],
            [2, 2],
            id="padded_on_policy_completions",
        ),
        pytest.param(
            torch.tensor([[10, 11, 12]], dtype=torch.long),
            torch.tensor([[1, 1, 1]], dtype=torch.long),
            None,
            torch.tensor([[-100, -100, -100]], dtype=torch.long),
            [[10, 11, 12]],
            [3],
            [0],
            id="empty_completion",
        ),
    ],
)
def test_build_teacher_request_inputs(
    input_ids,
    attention_mask,
    prompt_attention_mask,
    labels,
    expected_sequences,
    expected_prompt_lengths,
    expected_completion_lengths,
):
    # [FUNCTION DOC]
    # Parametrized check of build_teacher_request_inputs, which converts a padded batch into per-sample unpadded
    # token lists plus prompt and completion lengths for the teacher server.
    # Cases: left-padded prompts with variable lengths (prompt length from the attention mask and labels);
    # the same with extra right padding and an explicit prompt_attention_mask; and a single sample whose labels are
    # all -100 (empty completion, length 0).
    # Assertions: returned sequences, prompt_lengths and completion_lengths equal the expected lists.
    # Inputs: parametrized tensors and expected lists. Outputs: none. Exceptions: assertion failure. Side effects: none.
    sequences, prompt_lengths, completion_lengths = build_teacher_request_inputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_attention_mask=prompt_attention_mask,
        labels=labels,
    )

    assert sequences == expected_sequences
    assert prompt_lengths == expected_prompt_lengths
    assert completion_lengths == expected_completion_lengths


class TestGetTeacherTokenLogprobsFromServer(TrlTestCase):
    # [CLASS DOC]
    # Tests ServerDistillationTrainer._get_teacher_token_logprobs_from_server padding behavior.
    # Inherits TrlTestCase (self.tmp_dir). No extra state.
    def test_variable_lengths_use_neg_inf_sentinel_at_padding(self, monkeypatch):
        # [FUNCTION DOC]
        # Checks ragged server responses are packed into dense tensors with -inf at padded completion positions.
        # Setup: VLLMClient monkeypatched to return a MagicMock whose get_sequence_logprobs yields the ragged response;
        # trainer on the tiny Qwen2 model; a two-row batch whose completions (after prompt length 2) have lengths 1 and 3.
        # Assertions: actual_logprobs shape (2, 3) and topk_logprobs shape (2, 3, 1); real values copied at the right
        # positions; padded positions of the short row are -inf in both tensors; the full-length row is all finite.
        # Inputs: monkeypatch fixture. Outputs: none. Exceptions: assertion failure. Side effects: loads the tiny model.
        from trl.generation import vllm_client as vllm_client_module

        # Stand in for the teacher server, which cannot run here
        teacher_client = MagicMock()
        teacher_client.get_sequence_logprobs.return_value = _ragged_server_response()
        monkeypatch.setattr(vllm_client_module, "VLLMClient", lambda *args, **kwargs: teacher_client)
        trainer = ServerDistillationTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=ServerDistillationConfig(**_make_server_config_kwargs(self.tmp_dir)),
            train_dataset=_variable_length_dataset(),
        )

        inputs = {
            "input_ids": torch.tensor([[10, 11, 90, 0, 0], [10, 11, 90, 9217, 100]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]),
            "labels": torch.tensor([[-100, -100, 90, -100, -100], [-100, -100, 90, 9217, 100]]),
        }

        out = trainer._get_teacher_token_logprobs_from_server(inputs, aligned_prompt_length=2)

        assert out["actual_logprobs"].shape == (2, 3)
        assert out["topk_logprobs"].shape == (2, 3, 1)

        # Real completion positions preserved.
        assert out["actual_logprobs"][0, 0].item() == pytest.approx(-2.3, rel=1e-5)
        assert out["actual_logprobs"][1, 0].item() == pytest.approx(-1.1, rel=1e-5)
        assert out["actual_logprobs"][1, 2].item() == pytest.approx(-3.0, rel=1e-5)

        # Sample 0 is 1 token long; positions 1 and 2 are padded with the -inf sentinel.
        assert out["actual_logprobs"][0, 1].item() == float("-inf")
        assert out["actual_logprobs"][0, 2].item() == float("-inf")
        assert out["topk_logprobs"][0, 1, 0].item() == float("-inf")

        # Sample 1 is full-length and fully finite.
        assert torch.isfinite(out["actual_logprobs"][1, :]).all()


class TestServerReverseKLPaddingMask(TrlTestCase):
    # [CLASS DOC]
    # Numerical test that masked padding keeps the tail-bucket JSD loss and its gradient finite.
    # Inherits TrlTestCase. No state.
    def test_mask_keeps_forward_and_backward_finite(self):
        # [FUNCTION DOC]
        # Checks that replacing -inf teacher log probs at label -100 positions with zero, then calling _add_tail_bucket
        # and _jsd_divergence (beta 1.0, reverse KL) on random student log probs, gives finite tail values, finite loss
        # and finite gradients after summing and backpropagating.
        # Setup: teacher top-1 tensor of shape (2, 3, 1) with -inf padding in row 0; student random logits (2, 3, 2).
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none (random tensors are unseeded).
        # Simulates the getter's output: sample 0 has completion length 1 (positions 1-2
        # padded with -inf), sample 1 is full-length.
        teacher_topk = torch.tensor(
            [[[-2.3], [float("-inf")], [float("-inf")]], [[-1.1], [-0.4], [-3.0]]],
            dtype=torch.float32,
        )
        labels = torch.tensor([[90, -100, -100], [90, 9217, 100]])

        # Strategy B: neutralise -inf at labels == -100 before the divergence math.
        pad_mask = (labels == -100).unsqueeze(-1)
        zero = torch.zeros((), dtype=teacher_topk.dtype)
        teacher_topk = torch.where(pad_mask, zero, teacher_topk)

        valid_mask = torch.ones_like(teacher_topk, dtype=torch.bool)
        teacher_with_tail, support_mask = _add_tail_bucket(teacher_topk, valid_mask)
        assert torch.isfinite(teacher_with_tail).all()

        raw_student = torch.randn(2, 3, 2, requires_grad=True)
        student_log_probs = F.log_softmax(raw_student, dim=-1)
        loss = _jsd_divergence(student_log_probs, teacher_with_tail, beta=1.0, support_mask=support_mask)
        assert torch.isfinite(loss).all()

        loss.sum().backward()
        assert torch.isfinite(raw_student.grad).all()


class TestServerDistillationTrainerRaggedGrad(TrlTestCase):
    # [CLASS DOC]
    # End-to-end training test that ragged completion batches do not leak -inf into backward.
    # Class state (set in setup_class): model_id, device (cuda if available else cpu), tokenizer with pad token set to
    # eos. Inherits TrlTestCase. _run_one_step does the training; the test asserts on its log records.
    @classmethod
    def setup_class(cls):
        # [FUNCTION DOC]
        # HELPER: Class-level fixture: stores the tiny Qwen2 model id, picks the device and loads the tokenizer, setting
        # pad_token to eos_token.
        # Inputs: cls. Outputs: none. Exceptions: tokenizer load errors propagate. Side effects: sets class attributes.
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.tokenizer = AutoTokenizer.from_pretrained(model_id)
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.model_id = model_id

    def _run_one_step(self, bs, ga, monkeypatch):
        # [FUNCTION DOC]
        # HELPER: Runs a short ServerDistillationTrainer training with a fake teacher and returns logged records.
        # Behavior: monkeypatches VLLMClient to return a MagicMock whose get_sequence_logprobs uses _canned_teacher_logprobs;
        # builds a config (given batch size and gradient accumulation, lr 1e-4, max_completion_length 32, loss_top_k 1,
        # beta 1.0, loss_add_tail True, no saving, logging every step, fp32); loads the model in float32 on the device,
        # builds the trainer on the two-prompt dataset, also assigns the fake client to trainer.teacher_client, trains.
        # Inputs: bs - per-device batch size; ga - gradient accumulation steps; monkeypatch - pytest fixture.
        # Outputs: list of log_history entries containing "grad_norm".
        # Exceptions: training errors propagate. Side effects: loads the model, runs training in self.tmp_dir.
        from trl.generation import vllm_client as vllm_client_module

        fake_client = MagicMock()
        fake_client.get_sequence_logprobs.side_effect = _canned_teacher_logprobs
        monkeypatch.setattr(vllm_client_module, "VLLMClient", lambda *a, **kw: fake_client)

        config = ServerDistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=ga,
            learning_rate=1e-4,
            max_completion_length=32,
            teacher_model_server_url="http://fake-teacher.invalid:8000",
            loss_top_k=1,
            beta=1.0,
            loss_add_tail=True,
            save_strategy="no",
            report_to="none",
            logging_steps=1,
            bf16=False,
        )
        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.float32).to(self.device)
        trainer = ServerDistillationTrainer(
            model=model,
            args=config,
            train_dataset=_variable_length_dataset(),
            processing_class=self.tokenizer,
        )
        trainer.teacher_client = fake_client
        trainer.train()
        return [rec for rec in trainer.state.log_history if "grad_norm" in rec]

    @pytest.mark.parametrize(("bs", "ga"), [(1, 2), (2, 1)])
    def test_reverse_kl_finite_grad_with_ragged_batch(self, bs, ga, monkeypatch):
        # [FUNCTION DOC]
        # Parametrized over (bs, ga) in (1, 2) and (2, 1): checks that training logs at least one grad_norm entry and that
        # every logged grad_norm and loss is finite.
        # Inputs: bs, ga, monkeypatch. Outputs: none. Exceptions: assertion failure. Side effects: runs training.
        records = self._run_one_step(bs=bs, ga=ga, monkeypatch=monkeypatch)
        assert records, "Expected at least one grad_norm log entry during training"
        for record in records:
            assert math.isfinite(record["grad_norm"]), f"grad_norm={record['grad_norm']} leaked -inf into backward"
            assert math.isfinite(record["loss"])
