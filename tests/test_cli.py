# [MODULE DOC]
# Tests for the trl command-line interface (trl.cli.main and the per-trainer scripts in trl.scripts).
# Main contents: a --help regression test for every training subcommand; TestCLI with resume-argument wiring
# tests (trainer class mocked), a real SFT resume-from-checkpoint test, end-to-end smoke runs for each subcommand
# (distillation, dpo, grpo, kto, reward, rloo, sft), the env subcommand, YAML --config files for sft and
# vllm-serve; and TestCLIModelInitKwargs checking how model_init_kwargs are assembled from script/model args.
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

import importlib
import os
from io import StringIO
from unittest.mock import patch

import pytest
import yaml

from .testing_utils import TrlTestCase


@pytest.mark.parametrize("command", ["distillation", "dpo", "grpo", "kto", "reward", "rloo", "sft"])
def test_help_no_type_error(command):
    # [FUNCTION DOC]
    # Parametrized over training subcommands: regression for issue 5099 (unescaped percent signs in help strings
    # raising TypeError in argparse). Runs main with "trl <command> --help" and stdout captured.
    # Assertion: SystemExit is raised with exit code 0.
    # Inputs: command - subcommand name. Outputs: none. Exceptions: assertion failure. Side effects: none (stdout captured).
    # Regression test for https://github.com/huggingface/trl/issues/5099:
    # TrainingArguments help strings with unescaped "%" caused TypeError in argparse.
    from trl.cli import main

    with pytest.raises(SystemExit) as exc_info:
        with patch("sys.argv", ["trl", command, "--help"]), patch("sys.stdout", new_callable=StringIO):
            main()
    assert exc_info.value.code == 0


class TestCLI(TrlTestCase):
    # [CLASS DOC]
    # CLI integration tests. Inherits TrlTestCase (self.tmp_dir).
    @pytest.mark.parametrize(
        "command,trainer_name",
        [
            ("distillation", "DistillationTrainer"),
            ("dpo", "DPOTrainer"),
            ("grpo", "GRPOTrainer"),
            ("kto", "KTOTrainer"),
            ("reward", "RewardTrainer"),
            ("rloo", "RLOOTrainer"),
            ("sft", "SFTTrainer"),
        ],
    )
    @pytest.mark.parametrize("checkpoint", [None, "checkpoint-2"])
    def test_resume_argument(self, command, trainer_name, checkpoint):
        # [FUNCTION DOC]
        # Parametrized over (command, trainer class name) and checkpoint (None or "checkpoint-2"): checks the script
        # forwards resume_from_checkpoint to trainer.train.
        # Setup: imports trl.scripts.<command>, parses minimal args (output dir, dummy dataset name, no reporting; plus a
        # teacher model for distillation; plus --resume_from_checkpoint when given) via make_parser().parse_args_and_config;
        # patches datasets.load_dataset to return an empty train split and patches trl.<TrainerName> with a mock; runs
        # script.main with the parsed tuple.
        # Assertion: the mock trainer's train was called once with resume_from_checkpoint equal to the checkpoint.
        # Inputs: command, trainer_name, checkpoint. Outputs: none. Exceptions: assertion failure. Side effects: none.
        script = importlib.import_module(f"trl.scripts.{command}")
        argv = ["--output_dir", self.tmp_dir, "--dataset_name", "unused", "--report_to", "none"]
        if command == "distillation":
            argv += ["--teacher_model_name_or_path", "unused"]
        if checkpoint is not None:
            argv += ["--resume_from_checkpoint", checkpoint]
        parsed = script.make_parser().parse_args_and_config(argv)

        with patch("datasets.load_dataset", return_value={"train": []}), patch(f"trl.{trainer_name}") as trainer:
            script.main(*parsed)

        trainer.return_value.train.assert_called_once_with(resume_from_checkpoint=checkpoint)

    def test_sft_resume(self):
        # [FUNCTION DOC]
        # Checks real resume with the sft script: trains 2 steps saving at step 2 (batch 2, fp32, nll loss), then
        # raises max_steps to 3 and resumes from checkpoint-2 while wrapping SFTTrainer.training_step in a spy that still
        # runs the real implementation.
        # Assertion: training_step is called exactly once (only step 3 remains).
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: trains and writes checkpoints in self.tmp_dir.
        from trl import SFTTrainer
        from trl.scripts import sft

        argv = [
            "--output_dir",
            self.tmp_dir,
            "--model_name_or_path",
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            "--dataset_name",
            "trl-internal-testing/zen",
            "--dataset_config",
            "standard_language_modeling",
            "--report_to",
            "none",
            "--max_steps",
            "2",
            "--save_steps",
            "2",
            "--per_device_train_batch_size",
            "2",
            "--gradient_accumulation_steps",
            "1",
            "--bf16",
            "false",
            "--loss_type",
            "nll",
        ]
        sft.main(*sft.make_parser().parse_args_and_config(argv))
        checkpoint = os.path.join(self.tmp_dir, "checkpoint-2")
        argv[argv.index("--max_steps") + 1] = "3"
        argv += ["--resume_from_checkpoint", checkpoint]

        # Execute the real training step: only step 3 remains after restoring checkpoint-2.
        with patch.object(SFTTrainer, "training_step", autospec=True, side_effect=SFTTrainer.training_step) as step:
            sft.main(*sft.make_parser().parse_args_and_config(argv))
        assert step.call_count == 1

    def test_distillation(self):
        # [FUNCTION DOC]
        # Smoke test: runs the trl CLI entry point with sys.argv patched to "trl distillation" on tiny Qwen2 models and a zen
        # dataset config, output to self.tmp_dir, reporting disabled (student and teacher both tiny Qwen2, prompt-only data, max_completion_length 32). Passes if main returns without error.
        # Inputs/Outputs: none. Exceptions: any CLI/training error fails the test.
        # Side effects: downloads models/datasets, trains and writes to self.tmp_dir.
        from trl.cli import main

        command = f"trl distillation --output_dir {self.tmp_dir} --model_name_or_path trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --teacher_model_name_or_path trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --dataset_name trl-internal-testing/zen --dataset_config standard_prompt_only --max_completion_length 32 --report_to none"
        with patch("sys.argv", command.split(" ")):
            main()

    def test_dpo(self):
        # [FUNCTION DOC]
        # Smoke test: runs the trl CLI entry point with sys.argv patched to "trl dpo" on tiny Qwen2 models and a zen
        # dataset config, output to self.tmp_dir, reporting disabled (standard_preference). Passes if main returns without error.
        # Inputs/Outputs: none. Exceptions: any CLI/training error fails the test.
        # Side effects: downloads models/datasets, trains and writes to self.tmp_dir.
        from trl.cli import main

        command = f"trl dpo --output_dir {self.tmp_dir} --model_name_or_path trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --dataset_name trl-internal-testing/zen --dataset_config standard_preference --report_to none"
        with patch("sys.argv", command.split(" ")):
            main()

    def test_dpo_multiple_loss_types(self):
        # [FUNCTION DOC]
        # Smoke test: runs the trl CLI entry point with sys.argv patched to "trl dpo" on tiny Qwen2 models and a zen
        # dataset config, output to self.tmp_dir, reporting disabled with two loss types (sigmoid and bco_pair) weighted 1.0 and 0.5. Passes if main returns without error.
        # Inputs/Outputs: none. Exceptions: any CLI/training error fails the test.
        # Side effects: downloads models/datasets, trains and writes to self.tmp_dir.
        from trl.cli import main

        command = f"trl dpo --output_dir {self.tmp_dir} --model_name_or_path trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --dataset_name trl-internal-testing/zen --dataset_config standard_preference --report_to none --loss_type sigmoid bco_pair --loss_weights 1.0 0.5"
        with patch("sys.argv", command.split(" ")):
            main()

    @patch("sys.stdout", new_callable=StringIO)
    def test_env(self, mock_stdout):
        # [FUNCTION DOC]
        # Runs "trl env" with stdout patched and asserts the output contains "TRL version: ".
        # Inputs: mock_stdout - StringIO injected by the patch decorator. Outputs: none. Exceptions: assertion failure.
        # Side effects: none.
        from trl.cli import main

        command = "trl env"
        with patch("sys.argv", command.split(" ")):
            main()
        assert "TRL version: " in mock_stdout.getvalue().strip()

    def test_grpo(self):
        # [FUNCTION DOC]
        # Smoke test: runs the trl CLI entry point with sys.argv patched to "trl grpo" on tiny Qwen2 models and a zen
        # dataset config, output to self.tmp_dir, reporting disabled with a tiny sequence-classification reward model, 4 generations and 32-token completions. Passes if main returns without error.
        # Inputs/Outputs: none. Exceptions: any CLI/training error fails the test.
        # Side effects: downloads models/datasets, trains and writes to self.tmp_dir.
        from trl.cli import main

        command = f"trl grpo --output_dir {self.tmp_dir} --model_name_or_path trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --reward_model_name_or_path trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5 --dataset_name trl-internal-testing/zen --dataset_config standard_prompt_only --num_generations 4 --max_completion_length 32 --report_to none"
        with patch("sys.argv", command.split(" ")):
            main()

    def test_kto(self):
        # [FUNCTION DOC]
        # Smoke test: runs the trl CLI entry point with sys.argv patched to "trl kto" on tiny Qwen2 models and a zen
        # dataset config, output to self.tmp_dir, reporting disabled (standard_unpaired_preference). Passes if main returns without error.
        # Inputs/Outputs: none. Exceptions: any CLI/training error fails the test.
        # Side effects: downloads models/datasets, trains and writes to self.tmp_dir.
        from trl.cli import main

        command = f"trl kto --output_dir {self.tmp_dir} --model_name_or_path trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --dataset_name trl-internal-testing/zen --dataset_config standard_unpaired_preference --report_to none"
        with patch("sys.argv", command.split(" ")):
            main()

    def test_reward(self):
        # [FUNCTION DOC]
        # Smoke test: runs the trl CLI entry point with sys.argv patched to "trl reward" on tiny Qwen2 models and a zen
        # dataset config, output to self.tmp_dir, reporting disabled (standard_implicit_prompt_preference). Passes if main returns without error.
        # Inputs/Outputs: none. Exceptions: any CLI/training error fails the test.
        # Side effects: downloads models/datasets, trains and writes to self.tmp_dir.
        from trl.cli import main

        command = f"trl reward --output_dir {self.tmp_dir} --model_name_or_path trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --dataset_name trl-internal-testing/zen --dataset_config standard_implicit_prompt_preference --report_to none"
        with patch("sys.argv", command.split(" ")):
            main()

    def test_rloo(self):
        # [FUNCTION DOC]
        # Smoke test: runs the trl CLI entry point with sys.argv patched to "trl rloo" on tiny Qwen2 models and a zen
        # dataset config, output to self.tmp_dir, reporting disabled with a tiny sequence-classification reward model, 2 generations and 32-token completions. Passes if main returns without error.
        # Inputs/Outputs: none. Exceptions: any CLI/training error fails the test.
        # Side effects: downloads models/datasets, trains and writes to self.tmp_dir.
        from trl.cli import main

        command = f"trl rloo --output_dir {self.tmp_dir} --model_name_or_path trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --reward_model_name_or_path trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5 --dataset_name trl-internal-testing/zen --dataset_config standard_prompt_only --num_generations 2 --max_completion_length 32 --report_to none"
        with patch("sys.argv", command.split(" ")):
            main()

    def test_sft(self):
        # [FUNCTION DOC]
        # Smoke test: runs the trl CLI entry point with sys.argv patched to "trl sft" on tiny Qwen2 models and a zen
        # dataset config, output to self.tmp_dir, reporting disabled (standard_language_modeling). Passes if main returns without error.
        # Inputs/Outputs: none. Exceptions: any CLI/training error fails the test.
        # Side effects: downloads models/datasets, trains and writes to self.tmp_dir.
        from trl.cli import main

        command = f"trl sft --output_dir {self.tmp_dir} --model_name_or_path trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --dataset_name trl-internal-testing/zen --dataset_config standard_language_modeling --report_to none"
        with patch("sys.argv", command.split(" ")):
            main()

    def test_sft_config_file(self):
        # [FUNCTION DOC]
        # Writes a YAML config (model, zen language-modeling dataset, no reporting, output_dir under tmp, cosine with
        # restarts scheduler) and runs "trl sft --config <path>".
        # Assertion: the configured output directory exists afterwards.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: writes config, trains, writes outputs.
        from trl.cli import main

        output_dir = os.path.join(self.tmp_dir, "output")

        # Create a temporary config file
        config_path = os.path.join(self.tmp_dir, "config.yaml")
        config_content = {
            "model_name_or_path": "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            "dataset_name": "trl-internal-testing/zen",
            "dataset_config": "standard_language_modeling",
            "report_to": "none",
            "output_dir": output_dir,
            "lr_scheduler_type": "cosine_with_restarts",
        }
        with open(config_path, "w") as config_file:
            yaml.dump(config_content, config_file)

        # Test the CLI with config file
        command = f"trl sft --config {config_path}"
        with patch("sys.argv", command.split(" ")):
            main()

        # Verify that output directory was created
        assert os.path.exists(output_dir)

    def test_vllm_serve_config_file(self):
        # [FUNCTION DOC]
        # Checks "trl vllm-serve --config <yaml>" accepts the required model field from the config file rather than
        # failing argument validation. trl.scripts.vllm_serve.main is patched with a mock.
        # Assertions: the mock is called once and its first positional argument has model equal to the tiny Qwen2 id.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: writes a YAML file in self.tmp_dir.
        """
        Test `trl vllm-serve --config config.yaml` must not raise "the following arguments are required: --model" when
        the required field is satisfied by the config file rather than the command line.
        """
        from trl.cli import main

        config_path = os.path.join(self.tmp_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump({"model": "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"}, f)

        # Patch the actual function that `VllmServeCommand.run` imports as `vllm_serve_main`
        with patch("trl.scripts.vllm_serve.main") as mock_serve:
            with patch("sys.argv", ["trl", "vllm-serve", "--config", config_path]):
                main()

        mock_serve.assert_called_once()
        script_args = mock_serve.call_args.args[0]
        assert script_args.model == "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


class TestCLIModelInitKwargs(TrlTestCase):
    # [CLASS DOC]
    # Tests that training scripts build model_init_kwargs for the trainer config. Inherits TrlTestCase.
    @pytest.mark.parametrize(
        "command,trainer_name",
        [
            ("distillation", "DistillationTrainer"),
            ("dpo", "DPOTrainer"),
            ("grpo", "GRPOTrainer"),
            ("kto", "KTOTrainer"),
            ("reward", "RewardTrainer"),
            ("rloo", "RLOOTrainer"),
            ("sft", "SFTTrainer"),
        ],
    )
    @pytest.mark.parametrize(
        "model_init_kwargs",
        [None, {}, {"revision": "custom", "attn_implementation": "eager", "attention_dropout": 0.0}],
    )
    def test_model_init_kwargs(self, command, trainer_name, model_init_kwargs):
        # [FUNCTION DOC]
        # Parametrized over (command, trainer name) and user model_init_kwargs (None, empty, or a dict with revision,
        # attn_implementation, attention_dropout): parses minimal args, sets the training config's model_init_kwargs, and runs
        # the script with load_dataset and the trainer class mocked.
        # Assertion: the args passed to the trainer have model_init_kwargs equal to revision (from model_revision),
        # trust_remote_code, attn_implementation and dtype from the model/training args, overridden by the user dict.
        # Inputs: command, trainer_name, model_init_kwargs. Outputs: none. Exceptions: assertion failure. Side effects: none.
        script = importlib.import_module(f"trl.scripts.{command}")
        argv = ["--output_dir", self.tmp_dir, "--dataset_name", "unused", "--report_to", "none"]
        if command == "distillation":
            argv += ["--teacher_model_name_or_path", "unused"]
        parsed = script.make_parser().parse_args_and_config(argv)
        parsed[1].model_init_kwargs = model_init_kwargs
        expected = {
            "revision": parsed[2].model_revision,
            "trust_remote_code": parsed[1].trust_remote_code,
            "attn_implementation": parsed[2].attn_implementation,
            "dtype": parsed[2].dtype,
            **(model_init_kwargs or {}),
        }

        with patch("datasets.load_dataset", return_value={"train": []}), patch(f"trl.{trainer_name}") as trainer:
            script.main(*parsed)

        assert trainer.call_args.kwargs["args"].model_init_kwargs == expected

    def test_sft_model_init_kwargs(self):
        # [FUNCTION DOC]
        # Runs the real sft script for 1 step with --model_init_kwargs as a JSON string (eager attention, attention_dropout
        # 0.25), spying on SFTTrainer.train to capture the trainer.
        # Assertions: global_step is 1, the loaded model uses eager attention and attention_dropout 0.25.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: trains in self.tmp_dir.
        from trl import SFTTrainer
        from trl.scripts import sft

        argv = [
            "--output_dir",
            self.tmp_dir,
            "--model_name_or_path",
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            "--dataset_name",
            "trl-internal-testing/zen",
            "--dataset_config",
            "standard_language_modeling",
            "--report_to",
            "none",
            "--max_steps",
            "1",
            "--per_device_train_batch_size",
            "2",
            "--gradient_accumulation_steps",
            "1",
            "--bf16",
            "false",
            "--loss_type",
            "nll",
            "--model_init_kwargs",
            '{"attn_implementation": "eager", "attention_dropout": 0.25}',
        ]
        # Execute real training and inspect the model that the script loaded.
        with patch.object(SFTTrainer, "train", autospec=True, side_effect=SFTTrainer.train) as train:
            sft.main(*sft.make_parser().parse_args_and_config(argv))
        trainer = train.call_args.args[0]
        assert trainer.state.global_step == 1
        assert trainer.model.config._attn_implementation == "eager"
        assert trainer.model.config.attention_dropout == 0.25
