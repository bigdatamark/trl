# [MODULE DOC]
# Multi-accelerator tests (skipped unless at least two accelerators are present).
# Contents: run_command and get_config_path helpers; TestDistributed, which launches the TRL CLI scripts under
# accelerate with DDP/DeepSpeed ZeRO-2/ZeRO-3/FSDP2 configs; TestModelParallel, which trains in-process models
# split across two devices with a device_map (naive pipeline parallelism).
# Side effects at import: computes ROOT (repository root, two levels above tests/distributed). Tests spawn
# subprocesses, download tiny models/datasets from the Hub and write to temp dirs.

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
import subprocess
from pathlib import Path

import pytest
import torch
import transformers
from datasets import load_dataset
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

from trl import (
    DistillationConfig,
    DistillationTrainer,
    DPOConfig,
    DPOTrainer,
    GRPOConfig,
    GRPOTrainer,
    KTOConfig,
    KTOTrainer,
    RewardConfig,
    RewardTrainer,
    RLOOConfig,
    RLOOTrainer,
    SFTConfig,
    SFTTrainer,
)
from trl.trainer.utils import add_fused_lm_head

from ..testing_utils import TrlTestCase, require_liger_kernel, require_torch_multi_accelerator


ROOT = Path(__file__).resolve().parents[2]


def run_command(command: list[str], env: dict[str, str]) -> None:
    # [FUNCTION DOC]
    # HELPER: Run a command as a subprocess from the repository root with the given environment and assert it
    # succeeded. Inputs: command (list of str), env (dict). Outputs: None.
    # Exceptions: AssertionError on non-zero return code. Side effects: spawns a blocking subprocess (output not captured).
    result = subprocess.run(command, env=env, cwd=ROOT)
    assert result.returncode == 0


@pytest.fixture
def get_config_path(lazy_shared_datadir):
    # [FUNCTION DOC]
    # HELPER: Pytest fixture returning a function that maps an accelerate config name to the path
    # accelerate_configs/<name>.yaml inside the lazily copied shared test data dir.
    # Inputs: lazy_shared_datadir fixture. Outputs: callable. Exceptions: none. Side effects: none.
    def _get_config_path(config_name):
        # [FUNCTION DOC]
        # HELPER: Return the Path of the YAML accelerate config with the given name (not checked for existence).
        # Inputs: config_name (str). Outputs: pathlib Path. Exceptions: none. Side effects: none.
        return lazy_shared_datadir / "accelerate_configs" / f"{config_name}.yaml"

    return _get_config_path


@require_torch_multi_accelerator
class TestDistributed(TrlTestCase):
    # [CLASS DOC]
    # Test class (inherits TrlTestCase for tmp_dir) running each TRL CLI training script end-to-end under
    # accelerate launch with several distributed backends. Every test only checks the launched process exits cleanly.
    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            pytest.param(
                "zero2",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            pytest.param(
                "zero3",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            "fsdp2",
        ],
    )
    def test_sft(self, config, get_config_path):
        # [FUNCTION DOC]
        # SFT smoke test.
        # Parametrized over accelerate configs ddp, zero2, zero3 and fsdp2 (zero2/zero3 marked xfail on transformers 5.1.0).
        # Launches trl/scripts/sft.py through accelerate launch with the tiny Qwen2.5 causal LM on the zen dataset config standard_language_modeling,
        # writing to the test tmp_dir. Asserts only that the subprocess exits with code 0.
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/sft.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "standard_language_modeling",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            pytest.param(
                "zero2",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            pytest.param(
                "zero3",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            "fsdp2",
        ],
    )
    def test_sft_nll_loss(self, config, get_config_path):
        # [FUNCTION DOC]
        # SFT with the nll loss type.
        # Parametrized over accelerate configs ddp, zero2, zero3 and fsdp2 (zero2/zero3 marked xfail on transformers 5.1.0).
        # Launches trl/scripts/sft.py through accelerate launch with the tiny Qwen2.5 causal LM on the zen dataset config standard_language_modeling with loss_type nll,
        # writing to the test tmp_dir. Asserts only that the subprocess exits with code 0.
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/sft.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "standard_language_modeling",
                "--loss_type", "nll",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            pytest.param(
                "zero2",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            pytest.param(
                "zero3",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            "fsdp2",
        ],
    )
    def test_dpo(self, config, get_config_path):
        # [FUNCTION DOC]
        # DPO smoke test.
        # Parametrized over accelerate configs ddp, zero2, zero3 and fsdp2 (zero2/zero3 marked xfail on transformers 5.1.0).
        # Launches trl/scripts/dpo.py through accelerate launch with the tiny Qwen2.5 causal LM on the zen dataset config standard_preference,
        # writing to the test tmp_dir. Asserts only that the subprocess exits with code 0.
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/dpo.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "standard_preference",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            pytest.param(
                "zero2",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            pytest.param(
                "zero3",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            "fsdp2",
        ],
    )
    def test_dpo_precompute_ref_log_probs(self, config, get_config_path):
        # [FUNCTION DOC]
        # DPO with precomputed reference log-probs for both train and eval splits (eval_strategy epoch),
        # regression for multi-GPU precompute fingerprint and ZeRO-3 re-initialization bugs.
        # Parametrized over accelerate configs ddp, zero2, zero3 and fsdp2 (zero2/zero3 marked xfail on transformers 5.1.0).
        # Launches trl/scripts/dpo.py through accelerate launch with the tiny Qwen2.5 causal LM on the zen dataset config standard_preference with precompute_ref_log_probs and eval_strategy epoch,
        # writing to the test tmp_dir. Asserts only that the subprocess exits with code 0.
        # `--eval_strategy epoch` passes an eval dataset, so reference log-probs are precomputed for both the train and
        # eval splits (two passes), which is what previously broke multi-GPU precompute (fingerprint cache mismatch, and
        # a corrupted ZeRO-3 parameter coordinator from re-initializing DeepSpeed on the policy model per pass).
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/dpo.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "standard_preference",
                "--precompute_ref_log_probs",
                "--eval_strategy", "epoch",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            pytest.param(
                "zero2",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            pytest.param(
                "zero3",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            "fsdp2",
        ],
    )
    def test_kto_precompute_ref_log_probs(self, config, get_config_path):
        # [FUNCTION DOC]
        # KTO with precomputed reference log-probs and per-epoch evaluation.
        # Parametrized over accelerate configs ddp, zero2, zero3 and fsdp2 (zero2/zero3 marked xfail on transformers 5.1.0).
        # Launches trl/scripts/kto.py through accelerate launch with the tiny Qwen2.5 causal LM on the zen dataset config standard_unpaired_preference with precompute_ref_log_probs and eval_strategy epoch,
        # writing to the test tmp_dir. Asserts only that the subprocess exits with code 0.
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/kto.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "standard_unpaired_preference",
                "--precompute_ref_log_probs",
                "--eval_strategy", "epoch",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @require_liger_kernel
    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            pytest.param(
                "zero2",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            pytest.param(
                "zero3",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            "fsdp2",
        ],
    )
    def test_dpo_liger(self, config, get_config_path):
        # [FUNCTION DOC]
        # DPO with Liger kernels (requires liger-kernel).
        # Parametrized over accelerate configs ddp, zero2, zero3 and fsdp2 (zero2/zero3 marked xfail on transformers 5.1.0).
        # Launches trl/scripts/dpo.py through accelerate launch with the tiny Qwen2.5 causal LM on the zen dataset config standard_preference with use_liger_kernel,
        # writing to the test tmp_dir. Asserts only that the subprocess exits with code 0.
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/dpo.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "standard_preference",
                "--use_liger_kernel",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @require_liger_kernel
    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            pytest.param(
                "zero2",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            pytest.param(
                "zero3",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            "fsdp2",
        ],
    )
    def test_kto_liger(self, config, get_config_path):
        # [FUNCTION DOC]
        # KTO with Liger kernels (requires liger-kernel).
        # Parametrized over accelerate configs ddp, zero2, zero3 and fsdp2 (zero2/zero3 marked xfail on transformers 5.1.0).
        # Launches trl/scripts/kto.py through accelerate launch with the tiny Qwen2.5 causal LM on the zen dataset config standard_unpaired_preference with use_liger_kernel,
        # writing to the test tmp_dir. Asserts only that the subprocess exits with code 0.
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/kto.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "standard_unpaired_preference",
                "--use_liger_kernel",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            pytest.param(
                "zero2",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            pytest.param(
                "zero3",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            "fsdp2",
        ],
    )
    def test_sft_dataset_streaming(self, config, get_config_path):
        # [FUNCTION DOC]
        # SFT with a streamed dataset.
        # Parametrized over accelerate configs ddp, zero2, zero3 and fsdp2 (zero2/zero3 marked xfail on transformers 5.1.0).
        # Launches trl/scripts/sft.py through accelerate launch with the tiny Qwen2.5 causal LM on the zen dataset config standard_language_modeling with dataset_streaming and max_steps 3,
        # writing to the test tmp_dir. Asserts only that the subprocess exits with code 0.
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/sft.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "standard_language_modeling",
                "--dataset_streaming",
                "--max_steps", "3",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            pytest.param(
                "zero2",
                marks=pytest.mark.xfail(
                    condition=Version("2.10") <= Version(torch.__version__)
                    and Version(transformers.__version__) < Version("5.1.0"),
                    reason="ZeRO 2 + PEFT was failing before transformers 5.1.0 on torch 2.10; see #4884",
                ),
            ),
            pytest.param(
                "zero3",
                marks=pytest.mark.xfail(
                    condition=Version("2.10") <= Version(torch.__version__)
                    and Version(transformers.__version__) < Version("5.1.0"),
                    reason="ZeRO 3 + PEFT was failing before transformers 5.1.0 on torch 2.10; see #4884",
                ),
            ),
            "fsdp2",
        ],
    )
    def test_sft_peft(self, config, get_config_path):
        # [FUNCTION DOC]
        # SFT with PEFT (LoRA).
        # Parametrized over ddp, zero2, zero3, fsdp2; zero2/zero3 marked xfail on torch >= 2.10 with transformers < 5.1.0.
        # Launches trl/scripts/sft.py through accelerate launch with the tiny Qwen2.5 causal LM on the zen dataset config standard_language_modeling with use_peft,
        # writing to the test tmp_dir. Asserts only that the subprocess exits with code 0.
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/sft.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "standard_language_modeling",
                "--use_peft",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            pytest.param(
                "zero2",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            pytest.param(
                "zero3",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            "fsdp2",
        ],
    )
    def test_reward(self, config, get_config_path):
        # [FUNCTION DOC]
        # Reward model training smoke test.
        # Parametrized over accelerate configs ddp, zero2, zero3 and fsdp2 (zero2/zero3 marked xfail on transformers 5.1.0).
        # Launches trl/scripts/reward.py through accelerate launch with the tiny Qwen2.5 causal LM on the zen dataset config conversational_implicit_prompt_preference,
        # writing to the test tmp_dir. Asserts only that the subprocess exits with code 0.
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/reward.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "conversational_implicit_prompt_preference",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            pytest.param(
                "zero2",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            pytest.param(
                "zero3",
                marks=pytest.mark.xfail(
                    Version("5.0.0") <= Version(transformers.__version__) < Version("5.5.4"),
                    reason="ZeRO-3 fails with transformers >= 5.0.0 and < 5.5.4 (fixed in transformers#45414), see #4899",
                    strict=True,
                ),
            ),
            pytest.param(
                "fsdp2",
                marks=pytest.mark.skipif(
                    Version("5.4.0") <= Version(transformers.__version__) < Version("5.6.0"),
                    reason="Upstream issue: NaN weights on non-rank-0 FSDP processes (see #5386 and transformers#45050)",
                ),
            ),
        ],
    )
    def test_rloo(self, config, get_config_path):
        # [FUNCTION DOC]
        # RLOO smoke test.
        # Parametrized over ddp, zero2 (xfail on transformers 5.1.0), zero3 (strict xfail for transformers in [5.0.0, 5.5.4))
        # and fsdp2 (skipped for transformers in [5.4.0, 5.6.0)).
        # Launches trl/scripts/rloo.py through accelerate launch with the tiny Qwen2.5 causal LM on the zen dataset config conversational_prompt_only using trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5 as reward model,
        # writing to the test tmp_dir. Asserts only that the subprocess exits with code 0.
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/rloo.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "conversational_prompt_only",
                "--reward_model_name_or_path", "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            pytest.param(
                "zero2",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            pytest.param(
                "zero3",
                marks=pytest.mark.xfail(
                    Version("5.0.0") <= Version(transformers.__version__) < Version("5.5.4"),
                    reason="ZeRO-3 fails with transformers >= 5.0.0 and < 5.5.4 (fixed in transformers#45414), see #4899",
                    strict=True,
                ),
            ),
            "fsdp2",
        ],
    )
    def test_grpo(self, config, get_config_path):
        # [FUNCTION DOC]
        # GRPO smoke test.
        # Parametrized over ddp, zero2 (xfail on transformers 5.1.0), zero3 (strict xfail for transformers in [5.0.0, 5.5.4))
        # and fsdp2.
        # Launches trl/scripts/grpo.py through accelerate launch with the tiny Qwen2.5 causal LM on the zen dataset config conversational_prompt_only using trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5 as reward model,
        # writing to the test tmp_dir. Asserts only that the subprocess exits with code 0.
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/grpo.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "conversational_prompt_only",
                "--reward_model_name_or_path", "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @require_liger_kernel
    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            pytest.param(
                "zero2",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            pytest.param(
                "zero3",
                marks=pytest.mark.xfail(
                    Version("5.0.0") <= Version(transformers.__version__) < Version("5.5.4"),
                    reason="ZeRO-3 fails with transformers >= 5.0.0 and < 5.5.4 (fixed in transformers#45414), see #4899",
                    strict=True,
                ),
            ),
            "fsdp2",
        ],
    )
    def test_grpo_liger(self, config, get_config_path):
        # [FUNCTION DOC]
        # GRPO with Liger kernels (requires liger-kernel).
        # Parametrized like test_grpo (ddp, zero2 xfail on 5.1.0, zero3 strict xfail in [5.0.0, 5.5.4), fsdp2).
        # Launches trl/scripts/grpo.py through accelerate launch with the tiny Qwen2.5 causal LM on the zen dataset config conversational_prompt_only using trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5 as reward model and use_liger_kernel,
        # writing to the test tmp_dir. Asserts only that the subprocess exits with code 0.
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/grpo.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "conversational_prompt_only",
                "--reward_model_name_or_path", "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                "--use_liger_kernel",
            ],
            os.environ.copy(),
        )
        # fmt: on


@require_torch_multi_accelerator
class TestModelParallel:
    # [CLASS DOC]
    # Tests for single-process naive model parallelism: models loaded with a device_map that puts embeddings
    # and layer 0 on device 0, layer 1, norm and rotary embedding on device 1, and the head on device 0.
    # Plain class (not TrlTestCase); uses pytest tmp_path for output dirs.
    def test_fused_lm_head(self):
        # [FUNCTION DOC]
        # Checks the fused LM head on a split model matches a single-device model.
        # Setup: tiny Qwen3 loaded on device 0 and loaded split; add_fused_lm_head on both with log_probs and entropy
        # outputs; random input ids of shape 2 by 16 with the first 4 label positions masked to -100.
        # Asserts log_probs and entropy are close between the two, and that backward on the split model's loss gives
        # every parameter a gradient.
        """With the model split across two devices, the fused LM head matches the model on a single device."""
        model_id = "trl-internal-testing/tiny-Qwen3ForCausalLM"
        # The layout documented for naive pipeline parallelism: layers split across devices, `lm_head` on device 0
        device_map = {
            "model.embed_tokens": 0,
            "model.layers.0": 0,
            "model.layers.1": 1,
            "model.norm": 1,
            "model.rotary_emb": 1,
            "lm_head": 0,
        }
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0})
        model_split = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map)
        add_fused_lm_head(model, outputs=("log_probs", "entropy"))
        add_fused_lm_head(model_split, outputs=("log_probs", "entropy"))
        input_ids = torch.randint(0, model.config.vocab_size, (2, 16), device=model.device)
        labels = input_ids.masked_fill(torch.arange(16, device=model.device) < 4, -100)

        out = model(input_ids=input_ids, labels=labels, fused_lm_head=True)
        out_split = model_split(input_ids=input_ids, labels=labels, fused_lm_head=True)

        torch.testing.assert_close(out_split["log_probs"], out["log_probs"])
        torch.testing.assert_close(out_split["entropy"], out["entropy"])
        out_split["loss"].backward()
        assert all(param.grad is not None for param in model_split.parameters())

    def test_train_sft(self, tmp_path):
        # [FUNCTION DOC]
        # Trains SFTTrainer on a split tiny-Qwen3 model (zen standard_language_modeling, default SFTConfig) and asserts train_loss is logged and every
        # parameter changed vs a pre-training clone. Shows Trainer does not wrap split models in DataParallel.
        """A model split across two devices trains, since `Trainer` doesn't wrap it in `nn.DataParallel`."""
        model_id = "trl-internal-testing/tiny-Qwen3ForCausalLM"
        # The layout documented for naive pipeline parallelism: layers split across devices, `lm_head` on device 0
        device_map = {
            "model.embed_tokens": 0,
            "model.layers.0": 0,
            "model.layers.1": 1,
            "model.norm": 1,
            "model.rotary_emb": 1,
            "lm_head": 0,
        }
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32", device_map=device_map)
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=str(tmp_path), report_to="none")
        trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset)

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_distillation(self, tmp_path):
        # [FUNCTION DOC]
        # Trains DistillationTrainer with a split tiny-Qwen3 student and a split small-Qwen3 teacher (same
        # device map) on zen standard_prompt_only (lr 0.1, batch 3, completion length 8). Asserts train_loss is logged
        # and all student parameters changed.
        """A student and a teacher split across two devices train with the chunked divergence loss."""
        # The layout documented for naive pipeline parallelism: layers split across devices, `lm_head` on device 0
        device_map = {
            "model.embed_tokens": 0,
            "model.layers.0": 0,
            "model.layers.1": 1,
            "model.norm": 1,
            "model.rotary_emb": 1,
            "lm_head": 0,
        }
        model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen3ForCausalLM", dtype="float32", device_map=device_map
        )
        teacher_model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/small-Qwen3ForCausalLM", dtype="float32", device_map=device_map
        )
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = DistillationConfig(
            output_dir=str(tmp_path),
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = DistillationTrainer(
            model=model, teacher_model=teacher_model, args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_distillation_heads_on_different_devices(self, tmp_path):
        # [FUNCTION DOC]
        # Checks DistillationTrainer refuses when the student head (device 0) and teacher head (device 1, teacher
        # loaded with model on 0 and lm_head on 1) are on different devices: train raises ValueError matching
        # "Place both on the same device".
        """The chunked divergence loss needs the student and teacher heads on the same device, so training refuses."""
        # The layout documented for naive pipeline parallelism: layers split across devices, `lm_head` on device 0
        device_map = {
            "model.embed_tokens": 0,
            "model.layers.0": 0,
            "model.layers.1": 1,
            "model.norm": 1,
            "model.rotary_emb": 1,
            "lm_head": 0,
        }
        model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen3ForCausalLM", dtype="float32", device_map=device_map
        )
        # An untied teacher, so its head can sit on another device than its embeddings
        teacher_model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen3ForCausalLM", dtype="float32", device_map={"model": 0, "lm_head": 1}
        )
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = DistillationConfig(
            output_dir=str(tmp_path),
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = DistillationTrainer(
            model=model, teacher_model=teacher_model, args=training_args, train_dataset=dataset
        )

        with pytest.raises(ValueError, match="Place both on the same device"):
            trainer.train()

    def test_train_dpo(self, tmp_path):
        # [FUNCTION DOC]
        # Trains DPOTrainer on a split tiny-Qwen3 model (zen standard_preference, lr 0.1) and asserts train_loss is logged and every
        # parameter changed vs a pre-training clone. Shows Trainer does not wrap split models in DataParallel.
        """A model split across two devices trains, since `Trainer` doesn't wrap it in `nn.DataParallel`."""
        model_id = "trl-internal-testing/tiny-Qwen3ForCausalLM"
        # The layout documented for naive pipeline parallelism: layers split across devices, `lm_head` on device 0
        device_map = {
            "model.embed_tokens": 0,
            "model.layers.0": 0,
            "model.layers.1": 1,
            "model.norm": 1,
            "model.rotary_emb": 1,
            "lm_head": 0,
        }
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32", device_map=device_map)
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(
            output_dir=str(tmp_path),
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = DPOTrainer(model=model, args=training_args, train_dataset=dataset)

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_kto(self, tmp_path):
        # [FUNCTION DOC]
        # Trains KTOTrainer on a split tiny-Qwen3 model (zen standard_unpaired_preference, lr 0.1) and asserts train_loss is logged and every
        # parameter changed vs a pre-training clone. Shows Trainer does not wrap split models in DataParallel.
        """A model split across two devices trains, since `Trainer` doesn't wrap it in `nn.DataParallel`."""
        model_id = "trl-internal-testing/tiny-Qwen3ForCausalLM"
        # The layout documented for naive pipeline parallelism: layers split across devices, `lm_head` on device 0
        device_map = {
            "model.embed_tokens": 0,
            "model.layers.0": 0,
            "model.layers.1": 1,
            "model.norm": 1,
            "model.rotary_emb": 1,
            "lm_head": 0,
        }
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32", device_map=device_map)
        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        training_args = KTOConfig(
            output_dir=str(tmp_path),
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = KTOTrainer(model=model, args=training_args, train_dataset=dataset)

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_reward(self, tmp_path):
        # [FUNCTION DOC]
        # Trains RewardTrainer on a split tiny-Qwen3 model (AutoModelForSequenceClassification with num_labels 1 and the score head on device 0,
        # zen standard_implicit_prompt_preference, default RewardConfig) and asserts train_loss is logged and every
        # parameter changed vs a pre-training clone. Shows Trainer does not wrap split models in DataParallel.
        """A model split across two devices trains, since `Trainer` doesn't wrap it in `nn.DataParallel`."""
        model_id = "trl-internal-testing/tiny-Qwen3ForCausalLM"
        # The layout documented for naive pipeline parallelism: layers split across devices, `score` on device 0
        device_map = {
            "model.embed_tokens": 0,
            "model.layers.0": 0,
            "model.layers.1": 1,
            "model.norm": 1,
            "model.rotary_emb": 1,
            "score": 0,
        }
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=1, dtype="float32", device_map=device_map
        )
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        training_args = RewardConfig(output_dir=str(tmp_path), report_to="none")
        trainer = RewardTrainer(model=model, args=training_args, train_dataset=dataset)

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_grpo(self, tmp_path):
        # [FUNCTION DOC]
        # Trains GRPOTrainer on a split tiny-Qwen3 model (zen standard_prompt_only, lr 0.1, batch 3, 3 generations, completion length 8,
        # tiny Qwen2.5 sequence classifier as reward) and asserts train_loss is logged and every
        # parameter changed vs a pre-training clone. Shows Trainer does not wrap split models in DataParallel.
        """A model split across two devices trains, since `Trainer` doesn't wrap it in `nn.DataParallel`."""
        model_id = "trl-internal-testing/tiny-Qwen3ForCausalLM"
        # The layout documented for naive pipeline parallelism: layers split across devices, `lm_head` on device 0
        device_map = {
            "model.embed_tokens": 0,
            "model.layers.0": 0,
            "model.layers.1": 1,
            "model.norm": 1,
            "model.rotary_emb": 1,
            "lm_head": 0,
        }
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32", device_map=device_map)
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GRPOConfig(
            output_dir=str(tmp_path),
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = GRPOTrainer(
            model=model,
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

    def test_train_rloo(self, tmp_path):
        # [FUNCTION DOC]
        # Trains RLOOTrainer on a split tiny-Qwen3 model (zen standard_prompt_only, lr 0.1, batch 3, 3 generations, completion length 8,
        # tiny Qwen2.5 sequence classifier as reward) and asserts train_loss is logged and every
        # parameter changed vs a pre-training clone. Shows Trainer does not wrap split models in DataParallel.
        """A model split across two devices trains, since `Trainer` doesn't wrap it in `nn.DataParallel`."""
        model_id = "trl-internal-testing/tiny-Qwen3ForCausalLM"
        # The layout documented for naive pipeline parallelism: layers split across devices, `lm_head` on device 0
        device_map = {
            "model.embed_tokens": 0,
            "model.layers.0": 0,
            "model.layers.1": 1,
            "model.norm": 1,
            "model.rotary_emb": 1,
            "lm_head": 0,
        }
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32", device_map=device_map)
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = RLOOConfig(
            output_dir=str(tmp_path),
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

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."
