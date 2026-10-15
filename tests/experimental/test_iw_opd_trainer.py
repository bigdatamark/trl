# [MODULE DOC]
# Tests for the experimental IWOPDTrainer / IWOPDConfig (trl.experimental.iw_opd), importance-
# weighted on-policy distillation built on a generalized distillation trainer that supports JSD and
# iw_opd objectives with a local teacher or a vLLM teacher server. Covers config validation,
# build_teacher_request_inputs, server logprob padding with -inf sentinels, tail-bucket JSD finiteness,
# generalized_jsd_loss, trainer training/eval/loss normalization, local vs server parity, the IW-OPD
# loss and rollout logprob buffering, ragged server batches, and _RepeatBatchDataLoader delegation.
# Contains helpers that fabricate teacher-server responses. Import-time side effects: none.
# KNOWN ISSUE: several tests build trainers with object.__new__ and SimpleNamespace / MagicMock
# stand-ins, contrary to the repository test guideline of exercising real objects.

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
import os
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, IterableDatasetDict, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.experimental.iw_opd import IWOPDConfig, IWOPDTrainer
from trl.experimental.iw_opd.iw_opd_trainer import (
    _add_tail_bucket,
    _jsd_divergence,
    _RepeatBatchDataLoader,
    build_teacher_request_inputs,
)

from ..testing_utils import TrlTestCase


def _make_distillation_config_kwargs(tmp_path):
    # [FUNCTION DOC]
    # HELPER: Returns minimal IWOPDConfig kwargs: output_dir as str(tmp_path) and report_to none.
    # Inputs: tmp_path. Outputs: dict. Exceptions: none. Side effects: none.
    return {"output_dir": str(tmp_path), "report_to": "none"}


def _build_server_result(teacher_logits, inputs, temperature=1.0):
    # [FUNCTION DOC]
    # HELPER: Simulates a vLLM teacher-server response from local teacher logits. Uses
    # build_teacher_request_inputs to get per-sample completion lengths; for each sample with a
    # completion, finds the first supervised label position, takes the teacher logits from one position
    # earlier for comp_len steps, applies temperature-scaled log-softmax, and records the actual
    # completion-token logprobs plus top-1 logprobs and ids (each as nested lists of shape comp_len by 1).
    # Empty completions get empty lists.
    # Inputs: teacher_logits (batch, seq, vocab), inputs dict (input_ids, attention_mask, labels,
    # optional prompt_attention_mask), temperature (float, default 1.0).
    # Outputs: dict with actual_logprobs, logprobs, logprob_token_ids. Exceptions: none expected.
    # Side effects: none.
    """Simulate a vLLM server response with variable-length per-sample completions."""
    _, _, completion_lengths = build_teacher_request_inputs(
        inputs["input_ids"],
        inputs["attention_mask"],
        prompt_attention_mask=inputs.get("prompt_attention_mask"),
        labels=inputs.get("labels"),
    )

    label_mask = inputs["labels"] != -100
    actual_logprobs = []
    logprobs = []
    logprob_token_ids = []

    for i, comp_len in enumerate(completion_lengths):
        if comp_len == 0:
            actual_logprobs.append([])
            logprobs.append([])
            logprob_token_ids.append([])
            continue

        comp_start = int(torch.nonzero(label_mask[i], as_tuple=False)[0].item())
        sample_logits = teacher_logits[i, comp_start - 1 : comp_start - 1 + comp_len, :]
        sample_log_probs = F.log_softmax(sample_logits / temperature, dim=-1)
        comp_tokens = inputs["input_ids"][i, comp_start : comp_start + comp_len]

        top1_ids = sample_logits.argmax(dim=-1, keepdim=True)
        top1_lps = sample_log_probs.gather(dim=-1, index=top1_ids)
        actual_lps = sample_log_probs.gather(dim=-1, index=comp_tokens.unsqueeze(-1))

        actual_logprobs.append(actual_lps.tolist())
        logprobs.append(top1_lps.tolist())
        logprob_token_ids.append(top1_ids.tolist())

    return {
        "actual_logprobs": actual_logprobs,
        "logprobs": logprobs,
        "logprob_token_ids": logprob_token_ids,
    }


class RecordingTeacherClient:
    # [CLASS DOC]
    # HELPER: Fake teacher server client returning a preset result and recording call arguments.
    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Initializes calls to an empty list and result to None. Inputs: none. Outputs: none.
        # Exceptions: none. Side effects: sets attributes.
        self.calls = []
        self.result = None

    def get_sequence_logprobs(self, sequences, prompt_lengths, top_logprobs, temperature):
        # [FUNCTION DOC]
        # HELPER: Records sequences, prompt_lengths, top_logprobs, temperature in a dict appended to calls,
        # and returns self.result. Inputs: those four args. Outputs: the preset result. Exceptions: none.
        # Side effects: mutates calls.
        self.calls.append(
            {
                "sequences": sequences,
                "prompt_lengths": prompt_lengths,
                "top_logprobs": top_logprobs,
                "temperature": temperature,
            }
        )
        return self.result


def _ragged_server_response():
    # [FUNCTION DOC]
    # HELPER: Returns a fixed server response for two samples with completion lengths 1 and 3 (top-k 1
    # logprobs, token ids, and actual logprobs). Inputs: none. Outputs: dict. Exceptions: none.
    # Side effects: none.
    # Two samples with completion lengths 1 and 3 respectively; matches the wire format
    # of VLLMClient.get_sequence_logprobs (per-sample shape (comp_len, top_k=1)).
    return {
        "logprobs": [[[-2.3]], [[-1.1], [-0.4], [-3.0]]],
        "logprob_token_ids": [[[90]], [[90], [9217], [100]]],
        "actual_logprobs": [[[-2.3]], [[-1.1], [-0.4], [-3.0]]],
    }


def _canned_teacher_logprobs(**kwargs):
    # [FUNCTION DOC]
    # HELPER: Fabricates a ragged server response for requested sequences: for each sequence, comp_len
    # is its length minus its prompt length; position i gets logprob -1.0 - 0.05 i repeated top_k times
    # (top_logprobs, default 1), token id equal to the sequence's own completion token, and the same
    # value as actual logprob. Inputs: kwargs sequences, prompt_lengths, optional top_logprobs.
    # Outputs: dict with logprobs, logprob_token_ids, actual_logprobs. Exceptions: KeyError if required
    # keys are missing; ValueError on length mismatch. Side effects: none.
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
    # HELPER: Returns a 2-row conversational messages Dataset with a very short and a long assistant
    # reply (variable completion lengths). Inputs: none. Outputs: Dataset. Exceptions: none.
    # Side effects: none.
    return Dataset.from_list(
        [
            {"messages": [{"role": "user", "content": "What's 2+2?"}, {"role": "assistant", "content": "4."}]},
            {
                "messages": [
                    {"role": "user", "content": "Name three primary colors."},
                    {
                        "role": "assistant",
                        "content": "Red, green, and blue are the three primary colors commonly used in additive color mixing.",
                    },
                ]
            },
        ]
    )


def test_distillation_config_rejects_invalid_reverse_kl_top_1_mode(tmp_path):
    # [FUNCTION DOC]
    # Tests an unknown reverse_kl_top_1_mode raises ValueError ("must be one of").
    # Inputs: tmp_path (pytest fixture). Outputs: none. Exceptions: AssertionError if no matching ValueError.
    # Side effects: none.
    with pytest.raises(ValueError, match="reverse_kl_top_1_mode must be one of"):
        IWOPDConfig(**_make_distillation_config_kwargs(tmp_path), reverse_kl_top_1_mode="invalid")


def test_distillation_config_rejects_invalid_distillation_objective(tmp_path):
    # [FUNCTION DOC]
    # Tests an unknown distillation_objective raises ValueError ("must be one of").
    # Inputs: tmp_path (pytest fixture). Outputs: none. Exceptions: AssertionError if no matching ValueError.
    # Side effects: none.
    with pytest.raises(ValueError, match="distillation_objective must be one of"):
        IWOPDConfig(**_make_distillation_config_kwargs(tmp_path), distillation_objective="invalid")


def test_distillation_config_rejects_invalid_iw_opd_gamma(tmp_path):
    # [FUNCTION DOC]
    # Tests a negative iw_opd_gamma raises ValueError ("must be non-negative").
    # Inputs: tmp_path (pytest fixture). Outputs: none. Exceptions: AssertionError if no matching ValueError.
    # Side effects: none.
    with pytest.raises(ValueError, match="iw_opd_gamma must be non-negative"):
        IWOPDConfig(**_make_distillation_config_kwargs(tmp_path), iw_opd_gamma=-0.1)


def test_distillation_config_rejects_invalid_iw_opd_epsilon(tmp_path):
    # [FUNCTION DOC]
    # Tests iw_opd_epsilon 0.0 raises ValueError ("must be positive").
    # Inputs: tmp_path (pytest fixture). Outputs: none. Exceptions: AssertionError if no matching ValueError.
    # Side effects: none.
    with pytest.raises(ValueError, match="iw_opd_epsilon must be positive"):
        IWOPDConfig(**_make_distillation_config_kwargs(tmp_path), iw_opd_epsilon=0.0)


def test_distillation_config_rejects_iw_opd_when_not_fully_on_policy(tmp_path):
    # [FUNCTION DOC]
    # Tests the iw_opd objective with lmbda 0.5 raises ValueError (requires lmbda 1.0).
    # Inputs: tmp_path (pytest fixture). Outputs: none. Exceptions: AssertionError if no matching ValueError.
    # Side effects: none.
    with pytest.raises(ValueError, match="distillation_objective='iw_opd' requires lmbda=1.0"):
        IWOPDConfig(
            **_make_distillation_config_kwargs(tmp_path),
            distillation_objective="iw_opd",
            lmbda=0.5,
        )


def test_distillation_config_rejects_iw_opd_with_argmax_top_1_mode(tmp_path):
    # [FUNCTION DOC]
    # Tests the iw_opd objective with reverse_kl_top_1_mode argmax raises ValueError (requires sampled).
    # Inputs: tmp_path (pytest fixture). Outputs: none. Exceptions: AssertionError if no matching ValueError.
    # Side effects: none.
    with pytest.raises(ValueError, match="distillation_objective='iw_opd' requires reverse_kl_top_1_mode='sampled'"):
        IWOPDConfig(
            **_make_distillation_config_kwargs(tmp_path),
            distillation_objective="iw_opd",
            reverse_kl_top_1_mode="argmax",
        )


def test_distillation_config_rejects_teacher_server_with_reverse_kl_argmax(tmp_path):
    # [FUNCTION DOC]
    # Tests the jsd objective with a teacher server and argmax top-1 mode raises ValueError (not supported).
    # Inputs: tmp_path (pytest fixture). Outputs: none. Exceptions: AssertionError if no matching ValueError.
    # Side effects: none.
    with pytest.raises(ValueError, match="reverse_kl_top_1_mode='argmax' is not supported"):
        IWOPDConfig(
            **_make_distillation_config_kwargs(tmp_path),
            distillation_objective="jsd",
            use_teacher_server=True,
            teacher_model_server_url="http://localhost:8000",
            reverse_kl_top_1_mode="argmax",
        )


def test_distillation_config_rejects_teacher_server_mixed_loss_without_top_1(tmp_path):
    # [FUNCTION DOC]
    # Tests a teacher server with beta 0.5 and loss_top_k 2 raises ValueError (loss_top_k must be 1).
    # Inputs: tmp_path (pytest fixture). Outputs: none. Exceptions: AssertionError if no matching ValueError.
    # Side effects: none.
    with pytest.raises(ValueError, match="loss_top_k must be 1 when using use_teacher_server=True with beta>0"):
        IWOPDConfig(
            **_make_distillation_config_kwargs(tmp_path),
            use_teacher_server=True,
            teacher_model_server_url="http://localhost:8000",
            beta=0.5,
            loss_top_k=2,
        )


def test_distillation_config_requires_teacher_server_url(tmp_path):
    # [FUNCTION DOC]
    # Tests use_teacher_server without teacher_model_server_url raises ValueError.
    # Inputs: tmp_path (pytest fixture). Outputs: none. Exceptions: AssertionError if no matching ValueError.
    # Side effects: none.
    with pytest.raises(ValueError, match="teacher_model_server_url must be set when use_teacher_server=True"):
        IWOPDConfig(
            **_make_distillation_config_kwargs(tmp_path),
            use_teacher_server=True,
            beta=0.5,
            loss_top_k=1,
        )


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
    # Parametrized cases: variable (left-padded) prompt lengths, padded on-policy completions with a
    # prompt_attention_mask and right padding, and an empty completion. Tests that
    # build_teacher_request_inputs strips padding to return unpadded sequences, real prompt lengths, and
    # completion lengths (count of supervised labels) as expected.
    # Inputs: input_ids, attention_mask, prompt_attention_mask (optional), labels tensors and expected
    # lists. Outputs: none. Exceptions: AssertionError on failure. Side effects: none.
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
    # Tests _get_teacher_token_logprobs_from_server densification of ragged server output.
    def test_variable_lengths_use_neg_inf_sentinel_at_padding(self, monkeypatch):
        # [FUNCTION DOC]
        # Patches VLLMClient with a MagicMock returning the ragged response, builds a server-mode trainer, and
        # calls the getter for a 2-row batch with aligned prompt length 2. Asserts actual_logprobs shape
        # (2, 3) and topk_logprobs shape (2, 3, 1), real values preserved, sample 0's padded positions set
        # to negative infinity, and sample 1 fully finite.
        # Inputs: monkeypatch. Outputs: none. Exceptions: AssertionError. Side effects: downloads; tmp_dir.
        from trl.generation import vllm_client as vllm_client_module

        # Stand in for the teacher server, which cannot run here
        teacher_client = MagicMock()
        teacher_client.get_sequence_logprobs.return_value = _ragged_server_response()
        monkeypatch.setattr(vllm_client_module, "VLLMClient", lambda *args, **kwargs: teacher_client)
        trainer = IWOPDTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=IWOPDConfig(
                output_dir=self.tmp_dir,
                report_to="none",
                use_teacher_server=True,
                teacher_model_server_url="http://localhost:8000",
            ),
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
    # Tests that masking -inf padded teacher logprobs before divergence math keeps loss and gradients
    # finite.
    def test_mask_keeps_forward_and_backward_finite(self):
        # [FUNCTION DOC]
        # Replaces -inf teacher top-k values at label -100 positions with 0, adds a tail bucket via
        # _add_tail_bucket (all-valid mask), asserts it is finite, computes _jsd_divergence (beta 1.0) with
        # random student log-probs over 2 buckets, and asserts loss and student gradients are finite.
        # Inputs: none. Outputs: none. Exceptions: AssertionError on failure. Side effects: none.
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


class TestGeneralizedJSDLoss(TrlTestCase):
    # [CLASS DOC]
    # Tests the static IWOPDTrainer.generalized_jsd_loss on random logits (batch 2, seq 3, vocab 5) built
    # in setup_method.
    def setup_method(self):
        # [FUNCTION DOC]
        # HELPER: Sets batch_size 2, seq_length 3, vocab_size 5 and random student_logits / teacher_logits
        # of that shape. Inputs/Outputs: none. Exceptions: none. Side effects: consumes global torch RNG.
        self.batch_size = 2
        self.seq_length = 3
        self.vocab_size = 5
        self.student_logits = torch.randn(self.batch_size, self.seq_length, self.vocab_size)
        self.teacher_logits = torch.randn(self.batch_size, self.seq_length, self.vocab_size)

    def test_uniform_distribution(self):
        # [FUNCTION DOC]
        # Tests identical uniform logits give a loss of 0 to 5 decimals.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure. Side effects: none beyond model forward passes.
        logits = torch.ones(1, 1, self.vocab_size)
        loss = IWOPDTrainer.generalized_jsd_loss(logits, logits)
        assert round(abs(loss.item() - 0), 5) == 0

    def test_generalized_jsd_loss_edge_cases(self):
        # [FUNCTION DOC]
        # Tests on two-token distributions (0.1/0.9 vs 0.9/0.1) that beta=1 equals the batchmean KL
        # divergence computed with teacher log-probs as input and student probs as target, and beta=0 equals
        # the one with student log-probs as input and teacher probs as target (5-decimal tolerance).
        # Inputs/Outputs: none. Exceptions: AssertionError on failure. Side effects: none beyond model forward passes.
        # Setup
        student_logits = torch.log(torch.tensor([[0.1, 0.9]])).unsqueeze(0)
        teacher_logits = torch.log(torch.tensor([[0.9, 0.1]])).unsqueeze(0)

        # Case 1: beta = 1 (should be equivalent to KL(student || teacher))
        loss_beta_1 = IWOPDTrainer.generalized_jsd_loss(student_logits, teacher_logits, beta=1)
        expected_loss_beta_1 = F.kl_div(
            F.log_softmax(teacher_logits, dim=-1), F.softmax(student_logits, dim=-1), reduction="batchmean"
        )
        assert round(abs(loss_beta_1.item() - expected_loss_beta_1.item()), 5) == 0

        # Case 2: beta = 0 (should be equivalent to KL(teacher || student))
        loss_beta_0 = IWOPDTrainer.generalized_jsd_loss(student_logits, teacher_logits, beta=0)
        expected_loss_beta_0 = F.kl_div(
            F.log_softmax(student_logits, dim=-1), F.softmax(teacher_logits, dim=-1), reduction="batchmean"
        )
        assert round(abs(loss_beta_0.item() - expected_loss_beta_0.item()), 5) == 0

    def test_output_shape(self):
        # [FUNCTION DOC]
        # Tests the default reduction returns a scalar tensor.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure. Side effects: none beyond model forward passes.
        loss = IWOPDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits)
        assert torch.is_tensor(loss)
        assert loss.shape == torch.Size([])

    def test_beta_values(self):
        # [FUNCTION DOC]
        # Tests beta 0 and beta 1 give different losses.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure. Side effects: none beyond model forward passes.
        loss_beta_0 = IWOPDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, beta=0)
        loss_beta_1 = IWOPDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, beta=1)
        assert loss_beta_0 != loss_beta_1

    def test_temperature_scaling(self):
        # [FUNCTION DOC]
        # Tests temperature 1 and 2 give different losses.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure. Side effects: none beyond model forward passes.
        loss_temp_1 = IWOPDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, temperature=1)
        loss_temp_2 = IWOPDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, temperature=2)
        assert loss_temp_1 != loss_temp_2

    def test_reduction_methods(self):
        # [FUNCTION DOC]
        # Tests batchmean, sum, and mean reductions return scalars and none returns the logits' shape.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure. Side effects: none beyond model forward passes.
        loss_batchmean = IWOPDTrainer.generalized_jsd_loss(
            self.student_logits, self.teacher_logits, reduction="batchmean"
        )
        loss_sum = IWOPDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, reduction="sum")
        loss_mean = IWOPDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, reduction="mean")
        loss_none = IWOPDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, reduction="none")

        assert loss_batchmean.shape == torch.Size([])
        assert loss_sum.shape == torch.Size([])
        assert loss_mean.shape == torch.Size([])
        assert loss_none.shape == self.student_logits.shape

    def test_symmetry(self):
        # [FUNCTION DOC]
        # Tests swapping student and teacher changes the loss at beta 0.1 but not at beta 0.5 (symmetric JSD).
        # Inputs/Outputs: none. Exceptions: AssertionError on failure. Side effects: none beyond model forward passes.
        student_teacher = IWOPDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, beta=0.1)
        teacher_student = IWOPDTrainer.generalized_jsd_loss(self.teacher_logits, self.student_logits, beta=0.1)
        assert student_teacher != teacher_student

        student_teacher = IWOPDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, beta=0.5)
        teacher_student = IWOPDTrainer.generalized_jsd_loss(self.teacher_logits, self.student_logits, beta=0.5)
        assert student_teacher == teacher_student

    def test_zero_loss_for_identical_inputs(self):
        # [FUNCTION DOC]
        # Tests identical random logits give a loss of 0 to 6 decimals.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure. Side effects: none beyond model forward passes.
        identical_logits = torch.randn(self.batch_size, self.seq_length, self.vocab_size)
        loss = IWOPDTrainer.generalized_jsd_loss(identical_logits, identical_logits)
        assert round(abs(loss.item() - 0), 6) == 0


class TestIWOPDTrainer(TrlTestCase):
    # [CLASS DOC]
    # Integration tests for IWOPDTrainer on CPU with tiny Qwen2 as both student and teacher. Helpers
    # build configs (JSD objective pinned by default), local/server trainers, and collated batches.
    def setup_method(self):
        # [FUNCTION DOC]
        # HELPER: Sets self.model_id and self.tokenizer (pad = eos). Inputs/Outputs: none.
        # Exceptions: load errors propagate. Side effects: network/cache.
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _make_args(self, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Builds IWOPDConfig with test defaults (batch 2, grad accum 1, 1 step, no saving, CPU, no
        # bf16, lmbda 0.0, distillation_objective jsd, max_length 128, max_completion_length 32, float32
        # student and teacher init kwargs with no device_map), overridden by kwargs.
        # Inputs: kwargs. Outputs: IWOPDConfig. Exceptions: config validation ValueError. Side effects: none.
        args = {
            "output_dir": self.tmp_dir,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "max_steps": 1,
            "save_strategy": "no",
            "report_to": "none",
            "disable_tqdm": True,
            "use_cpu": True,
            "bf16": False,
            "lmbda": 0.0,
            # These are the JSD-path tests inherited from the DistillationTrainer suite; IWOPDConfig now defaults
            # distillation_objective to "iw_opd", so pin "jsd" here. IW-OPD tests override it via kwargs.
            "distillation_objective": "jsd",
            "max_length": 128,
            "max_completion_length": 32,
            "model_init_kwargs": {"dtype": "float32", "device_map": None},
            "teacher_model_init_kwargs": {"dtype": "float32", "device_map": None},
        }
        args.update(kwargs)
        return IWOPDConfig(**args)

    def _make_local_trainer(self, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Returns an IWOPDTrainer with the model id as student and local teacher on zen
        # conversational LM train split. Inputs: kwargs for _make_args. Outputs: IWOPDTrainer.
        # Exceptions: propagates. Side effects: downloads.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")
        return IWOPDTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=self._make_args(**kwargs),
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )

    def _make_server_trainer(self, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Like the local trainer but with no teacher model and use_teacher_server True at
        # localhost:8000. Inputs: kwargs. Outputs: IWOPDTrainer. Exceptions: propagates.
        # Side effects: constructs whatever VLLMClient is currently bound (patched in tests).
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")
        return IWOPDTrainer(
            model=self.model_id,
            teacher_model=None,
            args=self._make_args(use_teacher_server=True, teacher_model_server_url="http://localhost:8000", **kwargs),
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )

    def _make_batch(self, trainer):
        # [FUNCTION DOC]
        # HELPER: Collates the first two train examples with the trainer's data collator.
        # Inputs: trainer. Outputs: dict batch. Exceptions: none expected. Side effects: none.
        examples = [trainer.train_dataset[i] for i in range(2)]
        return trainer.data_collator(examples)

    @staticmethod
    def _move_batch_to_device(batch, device):
        # [FUNCTION DOC]
        # HELPER: Returns a new dict with every value moved to device. Inputs: batch (dict of tensors),
        # device. Outputs: dict. Exceptions: AttributeError for non-tensor values. Side effects: none.
        return {key: value.to(device) for key, value in batch.items()}

    def test_distillation_trainer_train_runs_with_local_teacher(self):
        # [FUNCTION DOC]
        # Trains 4 steps with eval and saves every 2 steps; asserts train_loss and eval_loss logged, overall
        # train_loss approximately 0 (student equals teacher), and model.safetensors in checkpoint-2.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models/datasets; writes to tmp_dir.
        training_args = self._make_args(
            dataloader_drop_last=True,
            eval_strategy="steps",
            max_steps=4,
            eval_steps=2,
            save_strategy="steps",
            save_steps=2,
            per_device_eval_batch_size=2,
        )
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling")
        trainer = IWOPDTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            processing_class=self.tokenizer,
        )

        train_result = trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert trainer.state.log_history[0]["eval_loss"] is not None
        # Student and teacher are the same model, so the JSD is exactly zero up to float32 rounding.
        assert train_result.metrics["train_loss"] == pytest.approx(0.0, abs=1e-6)
        assert "model.safetensors" in os.listdir(self.tmp_dir + "/checkpoint-2")

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
        # Parametrized over regular and streaming variants of dataset, dataset dict, and dict of datasets,
        # plus none. Tests eval datasets are stored as is (no preparation): None stays None; dicts keep keys
        # data1/data2; single datasets are the same object.
        # Inputs: eval_dataset_type (str). Outputs: none. Exceptions: AssertionError.
        # Side effects: downloads; tmp_dir.
        train_dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

        if eval_dataset_type == "none":
            eval_dataset = None
        else:
            streaming = "iterable" in eval_dataset_type
            eval_split = load_dataset(
                "trl-internal-testing/zen", "conversational_language_modeling", split="test", streaming=streaming
            )
            if eval_dataset_type in ("dataset", "iterable_dataset"):
                eval_dataset = eval_split
            elif eval_dataset_type in ("dataset_dict", "iterable_dataset_dict"):
                dataset_dict_cls = IterableDatasetDict if streaming else DatasetDict
                eval_dataset = dataset_dict_cls({"data1": eval_split, "data2": eval_split})
            else:  # "dict_of_dataset" or "dict_of_iterable_dataset"
                eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = IWOPDConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = IWOPDTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )

        # The distillation collator consumes raw examples, so eval datasets are stored as-is (not tokenized).
        if eval_dataset_type == "none":
            assert trainer.eval_dataset is None
        elif isinstance(trainer.eval_dataset, dict):
            assert set(trainer.eval_dataset.keys()) == {"data1", "data2"}
        else:
            assert trainer.eval_dataset is eval_dataset

    @pytest.mark.parametrize("loss_top_k", [0, 1])
    def test_loss_normalizes_by_num_items_in_batch(self, loss_top_k):
        # [FUNCTION DOC]
        # Parametrized over loss_top_k 0 (full-vocab JSD) and 1 (mixed top-1). Perturbs teacher weights,
        # counts valid labels after _compute_prompt_length, and asserts compute_loss with that count equals
        # the local mean while double the count halves it.
        # Inputs: loss_top_k (int). Outputs: none. Exceptions: AssertionError. Side effects: downloads.
        # When `num_items_in_batch` is passed (as under gradient accumulation), the divergence loss must be reduced as
        # sum / num_items_in_batch rather than the local per-microbatch mean. See issue #4719. Both the full-vocabulary
        # JSD path (loss_top_k=0) and the default mixed top-1 path (loss_top_k=1) route through
        # `_reduce_divergence_loss`, so both must honor `num_items_in_batch`.
        trainer = self._make_local_trainer(beta=0.5, loss_top_k=loss_top_k)

        # Diverge the teacher from the student so the divergence is well above fp noise (else the loss is ~0).
        torch.manual_seed(0)
        with torch.no_grad():
            for p in trainer.teacher_model.parameters():
                p.add_(0.5 * torch.randn_like(p))

        batch = self._move_batch_to_device(self._make_batch(trainer), trainer.accelerator.device)

        # Number of valid (non-ignored) tokens in the local batch, sliced the same way `compute_loss` does.
        prompt_length = trainer._compute_prompt_length(batch)
        num_valid = (batch["labels"][:, prompt_length:] != -100).sum()

        trainer.model.eval()
        with torch.no_grad():
            loss_mean = trainer.compute_loss(trainer.model, batch)  # num_items_in_batch=None -> local mean
            loss_global = trainer.compute_loss(trainer.model, batch, num_items_in_batch=num_valid)
            loss_double = trainer.compute_loss(trainer.model, batch, num_items_in_batch=num_valid * 2)

        # With num_items_in_batch equal to the local valid-token count, sum/N equals the local mean.
        torch.testing.assert_close(loss_global, loss_mean, rtol=1e-4, atol=1e-6)
        # Doubling the global count exactly halves the loss (sum / num_items is linear in 1/num_items).
        torch.testing.assert_close(loss_double, loss_mean / 2, rtol=1e-4, atol=1e-6)

    def test_sampled_mode_matches_between_local_and_external_teachers(self, monkeypatch):
        # [FUNCTION DOC]
        # Patches VLLMClient with a RecordingTeacherClient, builds local (sampled top-1 mode) and server
        # trainers, computes the local teacher logits, loads the fake server result from them, and asserts
        # the server request used the expected unpadded sequences, prompt lengths and top_logprobs 1, and the
        # two losses match.
        # Inputs: monkeypatch. Outputs: none. Exceptions: AssertionError. Side effects: downloads.
        import trl.generation.vllm_client as vllm_client_module

        teacher_client = RecordingTeacherClient()
        monkeypatch.setattr(vllm_client_module, "VLLMClient", lambda *args, **kwargs: teacher_client)

        local_trainer = self._make_local_trainer(beta=0.5, loss_top_k=1, reverse_kl_top_1_mode="sampled")
        server_trainer = self._make_server_trainer(beta=0.5, loss_top_k=1)

        cpu_inputs = self._make_batch(local_trainer)
        expected_sequences, expected_prompt_lengths, _ = build_teacher_request_inputs(
            cpu_inputs["input_ids"],
            cpu_inputs["attention_mask"],
            prompt_attention_mask=cpu_inputs["prompt_attention_mask"],
            labels=cpu_inputs["labels"],
        )

        local_inputs = self._move_batch_to_device(cpu_inputs, local_trainer.accelerator.device)
        server_inputs = self._move_batch_to_device(cpu_inputs, server_trainer.accelerator.device)

        local_trainer.teacher_model.eval()
        with torch.no_grad():
            teacher_logits = local_trainer.teacher_model(
                input_ids=local_inputs["input_ids"],
                attention_mask=local_inputs["attention_mask"],
            ).logits
            teacher_client.result = _build_server_result(
                teacher_logits,
                local_inputs,
                temperature=local_trainer.temperature,
            )
            local_loss = local_trainer.compute_loss(local_trainer.model, local_inputs)
            server_loss = server_trainer.compute_loss(server_trainer.model, server_inputs)

        assert teacher_client.calls[0]["sequences"] == expected_sequences
        assert teacher_client.calls[0]["prompt_lengths"] == expected_prompt_lengths
        assert teacher_client.calls[0]["top_logprobs"] == 1
        torch.testing.assert_close(local_loss, server_loss)

    def test_iw_opd_prefix_weights_and_loss(self):
        # [FUNCTION DOC]
        # Builds a bare trainer via object.__new__ (temperature 1, gamma 0.5, epsilon 1e-8, metrics dict,
        # training model stand-in) and calls _compute_iw_opd_loss on uniform zero student logits with teacher
        # logprobs offset from the student's by known advantages (row 1 has only one valid token). Asserts
        # the loss equals minus student logprob times the sum of weight times advantage over 4 valid
        # tokens, with prefix weights 1.5, 1.4, 1.25 and 1.5; that max/min weight metrics are 1.5 and 1.25;
        # that num_items_in_batch 8 rescales the loss by 4/8; and that gradients are finite.
        # Inputs: none. Outputs: none. Exceptions: AssertionError on failure. Side effects: none.
        trainer = IWOPDTrainer.__new__(IWOPDTrainer)
        trainer.temperature = 1.0
        trainer.iw_opd_gamma = 0.5
        trainer.iw_opd_epsilon = 1e-8
        trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        trainer.model = SimpleNamespace(training=True)

        student_logits = torch.zeros(2, 3, 4, requires_grad=True)
        completion_tokens = torch.tensor([[0, 1, 2], [1, 2, 3]])
        labels = torch.tensor([[0, 1, 2], [1, -100, -100]])
        student_logprob = -math.log(4)
        teacher_actual_logprobs = torch.tensor(
            [
                [student_logprob + 0.2, student_logprob - 0.3, student_logprob + 0.5],
                [student_logprob + 0.4, float("-inf"), float("-inf")],
            ]
        )

        loss = trainer._compute_iw_opd_loss(
            student_logits=student_logits,
            completion_tokens=completion_tokens,
            labels=labels,
            teacher_actual_logprobs=teacher_actual_logprobs,
        )

        advantages = torch.tensor([[0.2, -0.3, 0.5], [0.4, 0.0, 0.0]])
        weights = torch.tensor([[1.5, 1.4, 1.25], [1.5, 1.0, 1.0]])
        expected = -student_logprob * (weights * advantages).sum() / 4

        torch.testing.assert_close(loss, expected)
        assert trainer._metrics["train"]["iw_opd/max_weight"] == [1.5]
        assert trainer._metrics["train"]["iw_opd/min_weight"] == [1.25]

        # Under gradient accumulation the sum is divided by the global token count instead of the local one
        loss_ga = trainer._compute_iw_opd_loss(
            student_logits=student_logits,
            completion_tokens=completion_tokens,
            labels=labels,
            teacher_actual_logprobs=teacher_actual_logprobs,
            num_items_in_batch=8,
        )
        torch.testing.assert_close(loss_ga, expected * 4 / 8)

        loss.backward()
        assert student_logits.grad is not None
        assert torch.isfinite(student_logits.grad).all()

    def test_iw_opd_uses_cached_rollout_logprobs(self):
        # [FUNCTION DOC]
        # Part 1: with gamma 0, asserts _compute_iw_opd_loss uses rollout_logprobs (not current student
        # logprobs) for the advantage, giving minus the current logprob times the summed teacher minus
        # rollout difference over 2 tokens. Part 2: on another bare trainer with a fake tokenizer, stores two
        # completions of different prompt lengths via _store_completions_in_buffer and asserts the buffered
        # rollout_logprobs have the same full-sequence shape as labels, prompt length is 1, and after slicing
        # every supervised token maps to its recorded logprob.
        # Inputs: none. Outputs: none. Exceptions: AssertionError on failure. Side effects: none.
        trainer = IWOPDTrainer.__new__(IWOPDTrainer)
        trainer.temperature = 1.0
        trainer.iw_opd_gamma = 0.0
        trainer.iw_opd_epsilon = 1e-8
        trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        trainer.model = SimpleNamespace(training=True)

        student_logits = torch.zeros(1, 2, 4, requires_grad=True)
        completion_tokens = torch.tensor([[0, 1]])
        labels = torch.tensor([[0, 1]])
        current_student_logprob = -math.log(4)
        rollout_logprobs = torch.tensor([[-0.5, -0.25]])
        teacher_actual_logprobs = torch.tensor([[-0.2, -0.05]])

        loss = trainer._compute_iw_opd_loss(
            student_logits=student_logits,
            completion_tokens=completion_tokens,
            labels=labels,
            teacher_actual_logprobs=teacher_actual_logprobs,
            rollout_logprobs=rollout_logprobs,
        )

        expected_advantage_sum = (teacher_actual_logprobs - rollout_logprobs).sum()
        expected = -current_student_logprob * expected_advantage_sum / 2
        torch.testing.assert_close(loss, expected)

        # Buffer layout: with mixed prompt lengths, prompt_length (shortest real prompt) is smaller than the
        # padded prompt width, so rollout logprobs must be stored at full sequence width to survive the
        # labels-style slice in compute_loss.
        class _Tok:
            # [CLASS DOC]
            # HELPER: Minimal tokenizer stand-in with pad_token_id 0 and a batch_decode joining token ids.
            pad_token_id = 0

            @staticmethod
            def batch_decode(sequences, skip_special_tokens=False, clean_up_tokenization_spaces=False):
                # [FUNCTION DOC]
                # HELPER: Returns each sequence's integer ids joined by spaces. Inputs: sequences, ignored flags.
                # Outputs: list[str]. Exceptions: none. Side effects: none.
                return [" ".join(str(int(t)) for t in seq) for seq in sequences]

        trainer = IWOPDTrainer.__new__(IWOPDTrainer)
        trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
        trainer.processing_class = _Tok()
        trainer.generation_config = SimpleNamespace(max_new_tokens=3)
        trainer._buffered_inputs = [None]
        trainer._buffered_text_logs = [None]

        trainer._store_completions_in_buffer(
            slices=[{}],
            on_policy_indices=[0],
            local_slice_indices=[0, 0],
            local_prompts=[torch.tensor([5]), torch.tensor([6, 7])],
            completion_ids=[[11, 12, 13], [14]],
            logprobs=[[[-0.1], [-0.2], [-0.3]], [[-0.4]]],
            logprob_token_ids=[[[11], [12], [13]], [[14]]],
        )

        buffered = trainer._buffered_inputs[0]
        labels = buffered["labels"]
        rollout_logprobs = buffered["rollout_logprobs"]
        assert rollout_logprobs.shape == labels.shape

        prompt_length = trainer._compute_prompt_length(buffered)
        assert prompt_length == 1
        sliced_labels = labels[:, prompt_length:]
        sliced_rollout = rollout_logprobs[:, prompt_length:]
        expected = {(0, 11): -0.1, (0, 12): -0.2, (0, 13): -0.3, (1, 14): -0.4}
        valid = sliced_labels != -100
        assert valid.sum() == 4
        for row, col in valid.nonzero().tolist():
            token = int(sliced_labels[row, col])
            torch.testing.assert_close(sliced_rollout[row, col].item(), expected[(row, token)])

    def test_iw_opd_train_runs_with_local_teacher(self):
        # [FUNCTION DOC]
        # Trains 1 step with iw_opd objective (lmbda 1.0, 8 completion tokens); asserts a finite train_loss
        # and an iw_opd/mean_weight metric logged.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models/datasets; writes to tmp_dir.
        trainer = self._make_local_trainer(
            distillation_objective="iw_opd",
            lmbda=1.0,
            max_completion_length=8,
        )

        train_result = trainer.train()

        assert train_result.metrics["train_loss"] is not None
        assert math.isfinite(train_result.metrics["train_loss"])
        assert any("iw_opd/mean_weight" in record for record in trainer.state.log_history)

    def test_iw_opd_server_path_uses_actual_logprobs(self, monkeypatch):
        # [FUNCTION DOC]
        # Patches VLLMClient with a RecordingTeacherClient preloaded with canned logprobs for the batch's
        # request; asserts compute_loss under iw_opd server mode is finite and requested top_logprobs 1.
        # Inputs: monkeypatch. Outputs: none. Exceptions: AssertionError. Side effects: downloads.
        import trl.generation.vllm_client as vllm_client_module

        teacher_client = RecordingTeacherClient()
        monkeypatch.setattr(vllm_client_module, "VLLMClient", lambda *args, **kwargs: teacher_client)

        trainer = self._make_server_trainer(distillation_objective="iw_opd", lmbda=1.0)
        inputs = self._move_batch_to_device(self._make_batch(trainer), trainer.accelerator.device)
        sequences, prompt_lengths, _ = build_teacher_request_inputs(
            inputs["input_ids"],
            inputs["attention_mask"],
            prompt_attention_mask=inputs["prompt_attention_mask"],
            labels=inputs["labels"],
        )
        teacher_client.result = _canned_teacher_logprobs(
            sequences=sequences,
            prompt_lengths=prompt_lengths,
            top_logprobs=1,
        )

        loss = trainer.compute_loss(trainer.model, inputs)

        assert torch.isfinite(loss)
        assert teacher_client.calls[0]["top_logprobs"] == 1

    def test_pad_token_id_synced_with_model_config(self):
        # [FUNCTION DOC]
        # Tests with tiny Mistral (no pad token) that the trainer sets a pad token and the student config
        # and generation config pad_token_id equal it.
        # Inputs/Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: downloads tiny models/datasets; writes to tmp_dir.
        # This model's tokenizer has no pad token, so the trainer falls back to the eos token. The model configs must
        # follow: otherwise `Trainer` realigns them at train time and reports it as a change the user did not make.
        model_id = "trl-internal-testing/tiny-MistralForCausalLM-0.2"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

        trainer = IWOPDTrainer(
            model=model_id,
            teacher_model=model_id,
            args=IWOPDConfig(output_dir=self.tmp_dir, report_to="none"),
            train_dataset=dataset,
        )

        pad_token_id = trainer.processing_class.pad_token_id
        assert pad_token_id is not None
        assert trainer.model.config.pad_token_id == pad_token_id
        assert trainer.model.generation_config.pad_token_id == pad_token_id


class TestIWOPDTrainerServerPath(TrlTestCase):
    # [CLASS DOC]
    # End-to-end server-path training with a fake teacher client on ragged batches; setup_class loads the
    # tokenizer (pad = eos) and picks cuda or cpu.
    @classmethod
    def setup_class(cls):
        # [FUNCTION DOC]
        # HELPER: Sets cls.device, cls.tokenizer (pad = eos), cls.model_id. Inputs: cls. Outputs: none.
        # Exceptions: load errors. Side effects: network/cache.
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.tokenizer = AutoTokenizer.from_pretrained(model_id)
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.model_id = model_id

    def _run_one_step(self, bs, ga, monkeypatch):
        # [FUNCTION DOC]
        # HELPER: Patches VLLMClient with a MagicMock whose get_sequence_logprobs uses
        # _canned_teacher_logprobs, trains a server-mode JSD trainer (beta 1.0, top-1 with tail, lmbda 0,
        # lr 1e-4, logging every step) on the variable-length dataset with the given batch size and grad
        # accumulation, and returns log records containing grad_norm.
        # Inputs: bs, ga (int), monkeypatch. Outputs: list[dict]. Exceptions: propagates training errors.
        # Side effects: monkeypatch; tmp_dir.
        from trl.generation import vllm_client as vllm_client_module

        fake_client = MagicMock()
        fake_client.get_sequence_logprobs.side_effect = _canned_teacher_logprobs
        monkeypatch.setattr(vllm_client_module, "VLLMClient", lambda *a, **kw: fake_client)

        config = IWOPDConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=ga,
            learning_rate=1e-4,
            max_length=64,
            max_prompt_length=32,
            max_completion_length=32,
            use_teacher_server=True,
            teacher_model_server_url="http://fake-teacher.invalid:8000",
            loss_top_k=1,
            beta=1.0,
            lmbda=0.0,
            distillation_objective="jsd",
            loss_add_tail=True,
            save_strategy="no",
            report_to="none",
            logging_steps=1,
            bf16=False,
        )
        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.float32).to(self.device)
        trainer = IWOPDTrainer(
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
        # Parametrized over (bs, ga) of (1, 2) and (2, 1): asserts at least one grad_norm record and that all
        # grad_norm and loss values are finite (no -inf padding leaks into backward).
        # Inputs: bs, ga, monkeypatch. Outputs: none. Exceptions: AssertionError. Side effects: as helper.
        records = self._run_one_step(bs=bs, ga=ga, monkeypatch=monkeypatch)
        assert records, "Expected at least one grad_norm log entry during training"
        for record in records:
            assert math.isfinite(record["grad_norm"]), f"grad_norm={record['grad_norm']} leaked -inf into backward"
            assert math.isfinite(record["loss"])


def test_repeat_batch_dataloader_delegates_set_epoch_via_getattr():
    # [FUNCTION DOC]
    # Tests _RepeatBatchDataLoader forwards set_epoch(7) to the wrapped dataloader (epoch becomes 7).
    # Inputs: none. Outputs: none. Exceptions: AssertionError on failure. Side effects: none.
    class DummyDataLoader:
        # [CLASS DOC]
        # HELPER: Minimal dataloader yielding one batch, length 1, recording set_epoch calls.
        def __init__(self):
            # [FUNCTION DOC]
            # HELPER: Sets epoch to None. Inputs/Outputs: none. Exceptions: none. Side effects: sets attribute.
            self.epoch = None

        def __iter__(self):
            # [FUNCTION DOC]
            # HELPER: Generator yielding a single batch with x 1. Inputs: none. Outputs: iterator of dict.
            # Exceptions: none. Side effects: none.
            yield {"x": 1}

        def __len__(self):
            # [FUNCTION DOC]
            # HELPER: Returns 1. Inputs: none. Outputs: int. Exceptions: none. Side effects: none.
            return 1

        def set_epoch(self, epoch):
            # [FUNCTION DOC]
            # HELPER: Stores epoch. Inputs: epoch (int). Outputs: None. Exceptions: none.
            # Side effects: sets self.epoch.
            self.epoch = epoch

    dataloader = DummyDataLoader()
    wrapper = _RepeatBatchDataLoader(dataloader, repeat_count=2)

    wrapper.set_epoch(7)

    assert dataloader.epoch == 7
