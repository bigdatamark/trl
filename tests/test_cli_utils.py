# [MODULE DOC]
# Tests for TRL's CLI utilities: TrlParser (dataclass argument parser with YAML --config support, env section,
# remaining strings and subparsers) and get_dataset with DatasetMixtureConfig/DatasetConfig (loading, streaming,
# mixing, column selection, fraction weighting, test split creation, YAML parsing).
# Main contents: two small dataclasses used as parser targets, TestTrlParser and TestGetDataset.
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

import tempfile
from dataclasses import dataclass
from unittest.mock import mock_open, patch

import pytest
from datasets import DatasetDict, load_dataset

from trl import (
    DatasetMixtureConfig,
    DistillationConfig,
    DPOConfig,
    GRPOConfig,
    KTOConfig,
    RewardConfig,
    RLOOConfig,
    SFTConfig,
    TrlParser,
    get_dataset,
)
from trl.scripts.utils import DatasetConfig

from .testing_utils import TrlTestCase


@dataclass
class MyDataclass:
    # [CLASS DOC]
    # HELPER: Minimal parser target dataclass with arg1 (int, required) and arg2 (str, default "default").
    arg1: int
    arg2: str = "default"


@dataclass
class InvalidDataclass:
    # [CLASS DOC]
    # HELPER: Dataclass with a field named "config", which TrlParser must reject because it clashes with --config.
    config: str  # This should raise an error in the TrlParser


class TestTrlParser(TrlTestCase):
    # [CLASS DOC]
    # Tests TrlParser construction and parse_args_and_config behavior. Inherits TrlTestCase.
    @pytest.mark.parametrize(
        ("config_cls", "field_name", "value", "expected"),
        [
            (SFTConfig, "dataset_kwargs", '{"skip_prepare_dataset": true}', {"skip_prepare_dataset": True}),
            (GRPOConfig, "chat_template_kwargs", '{"enable_thinking": false}', {"enable_thinking": False}),
            (GRPOConfig, "generation_kwargs", '{"suppress_tokens": [1, 2]}', {"suppress_tokens": [1, 2]}),
            (RLOOConfig, "chat_template_kwargs", '{"enable_thinking": false}', {"enable_thinking": False}),
            (DistillationConfig, "generation_kwargs", '{"suppress_tokens": [1]}', {"suppress_tokens": [1]}),
        ],
    )
    def test_dict_field_from_command_line(self, config_cls, field_name, value, expected):
        # [FUNCTION DOC]
        # Parametrized over config classes and dict-typed fields (SFT dataset_kwargs, GRPO/RLOO chat_template_kwargs,
        # GRPO/Distillation generation_kwargs): parses "--output_dir dummy --<field> <json>" and asserts the field equals
        # the decoded dict. Inputs: config_cls, field_name, value, expected. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        """Dict-typed fields are passed as JSON on the command line and decoded into dicts."""
        parser = TrlParser(dataclass_types=[config_cls])
        (config,) = parser.parse_args_into_dataclasses(["--output_dir", "dummy", f"--{field_name}", value])
        assert getattr(config, field_name) == expected

    @pytest.mark.parametrize(
        "config_cls", [DistillationConfig, DPOConfig, GRPOConfig, KTOConfig, RewardConfig, RLOOConfig, SFTConfig]
    )
    @pytest.mark.parametrize(
        ("value", "expected"), [("128", 128), ("None", None), ("none", None), ("null", None), ("auto", None)]
    )
    def test_pad_to_multiple_of_from_command_line(self, config_cls, value, expected):
        # [FUNCTION DOC]
        # Parametrized over all trainer config classes and string values "128", "None", "none", "null", "auto": asserts
        # pad_to_multiple_of parses to 128 or None ("auto" resolves to None when not on Neuron).
        # Inputs: config_cls, value, expected. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        """`pad_to_multiple_of` arrives as a string on the command line; `"auto"` resolves to `None` off Neuron."""
        parser = TrlParser(dataclass_types=[config_cls])
        (config,) = parser.parse_args_into_dataclasses(["--output_dir", "dummy", "--pad_to_multiple_of", value])
        assert config.pad_to_multiple_of == expected

    def test_init_without_config_field(self):
        # [FUNCTION DOC]
        # Checks TrlParser accepts MyDataclass and returns a TrlParser instance. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        """Test initialization without 'config' field in the dataclasses."""
        parser = TrlParser(dataclass_types=[MyDataclass])
        assert isinstance(parser, TrlParser)

    def test_init_with_config_field(self):
        # [FUNCTION DOC]
        # Checks TrlParser raises ValueError ("has a field named 'config'") for InvalidDataclass. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        """Test initialization with a 'config' field in the dataclass (should raise ValueError)."""
        with pytest.raises(ValueError, match="has a field named 'config'"):
            TrlParser(dataclass_types=[InvalidDataclass])

    @patch("builtins.open", mock_open(read_data="env:\n VAR1: value1\n VAR2: value2\narg1: 2"))
    @patch("yaml.safe_load")
    @patch("os.environ", new_callable=dict)  # Mock os.environ as a dictionary
    def test_parse_args_and_config_with_valid_config(self, mock_environ, mock_yaml_load):
        # [FUNCTION DOC]
        # Checks a YAML config with an env section and arg1 2 combined with CLI --arg2 value yields MyDataclass(arg1 2,
        # arg2 "value"). Setup: builtins.open is patched with mock_open and yaml.safe_load is patched to return the config dict, so
        # "config.yaml" is never read from disk. os.environ is replaced with a plain dict.
        # KNOWN ISSUE: the test writes VAR1/VAR2 into the mocked environ itself before asserting them, so it does not verify
        # that the parser exported the env section.
        # Inputs: mock_environ, mock_yaml_load (patch-injected). Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        """Test parse_args_and_config method with valid arguments and config."""
        mock_yaml_load.return_value = {"env": {"VAR1": "value1", "VAR2": "value2"}, "arg1": 2}

        parser = TrlParser(dataclass_types=[MyDataclass])

        args = ["--arg2", "value", "--config", "config.yaml"]  # don't set arg1 to test default value

        # Simulate the config being loaded and environment variables being set
        result_args = parser.parse_args_and_config(args)

        # Set the environment variables using the mock
        mock_environ["VAR1"] = "value1"
        mock_environ["VAR2"] = "value2"

        # Ensure that the environment variables were set correctly
        assert mock_environ.get("VAR1") == "value1"
        assert mock_environ.get("VAR2") == "value2"

        # Check the parsed arguments
        assert len(result_args) == 1
        assert isinstance(result_args[0], MyDataclass)
        assert result_args[0].arg1 == 2
        assert result_args[0].arg2 == "value"

    @patch("builtins.open", mock_open(read_data="arg1: 2"))
    @patch("yaml.safe_load")
    def test_parse_args_and_arg_override_config(self, mock_yaml_load):
        # [FUNCTION DOC]
        # Checks a CLI value (--arg1 3) overrides the config value (arg1 2). Setup: builtins.open is patched with mock_open and yaml.safe_load is patched to return the config dict, so
        # "config.yaml" is never read from disk.
        # Inputs: mock_yaml_load. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        """Test parse_args_and_config method and check that arguments override the config."""
        mock_yaml_load.return_value = {"arg1": 2}  # this arg is meant to be overridden

        parser = TrlParser(dataclass_types=[MyDataclass])

        args = ["--arg1", "3", "--config", "config.yaml"]  # override arg1 default with 3

        # Simulate the config being loaded and arguments being passed
        result_args = parser.parse_args_and_config(args)

        # Check the parsed arguments
        assert len(result_args) == 1
        assert isinstance(result_args[0], MyDataclass)
        assert result_args[0].arg1 == 3

    @patch("builtins.open", mock_open(read_data="env: not_a_dict"))
    @patch("yaml.safe_load")
    def test_parse_args_and_config_with_invalid_env(self, mock_yaml_load):
        # [FUNCTION DOC]
        # Checks a config whose env field is a string raises ValueError ("env field should be a dict").
        # Setup: builtins.open is patched with mock_open and yaml.safe_load is patched to return the config dict, so
        # "config.yaml" is never read from disk. Inputs: mock_yaml_load. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        """Test parse_args_and_config method when the 'env' field is not a dictionary."""
        mock_yaml_load.return_value = {"env": "not_a_dict"}

        parser = TrlParser(dataclass_types=[MyDataclass])

        args = ["--arg1", "2", "--arg2", "value", "--config", "config.yaml"]

        with pytest.raises(ValueError, match="`env` field should be a dict in the YAML file."):
            parser.parse_args_and_config(args)

    @pytest.mark.parametrize("config", [None, [], "value", 42])
    @patch("builtins.open", mock_open())
    @patch("yaml.safe_load")
    def test_parse_args_and_config_with_non_mapping_config(self, mock_yaml_load, config):
        # [FUNCTION DOC]
        # Parametrized over YAML contents None, empty list, string and int: checks parse_args_and_config raises ValueError
        # ("must contain a YAML mapping"). Setup: builtins.open is patched with mock_open and yaml.safe_load is patched to return the config dict, so
        # "config.yaml" is never read from disk. Inputs: mock_yaml_load, config. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        mock_yaml_load.return_value = config
        parser = TrlParser(dataclass_types=[MyDataclass])

        with pytest.raises(ValueError, match="must contain a YAML mapping"):
            parser.parse_args_and_config(["--config", "config.yaml"])

    def test_parse_args_and_config_without_config(self):
        # [FUNCTION DOC]
        # Checks plain CLI parsing without --config returns one MyDataclass with arg1 2 and arg2 "value". Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        """Test parse_args_and_config without the `--config` argument."""
        parser = TrlParser(dataclass_types=[MyDataclass])

        args = ["--arg1", "2", "--arg2", "value"]

        # Simulate no config, just parse args normally
        result_args = parser.parse_args_and_config(args)

        # Check that the arguments are parsed as is
        assert len(result_args) == 1
        assert isinstance(result_args[0], MyDataclass)
        assert result_args[0].arg1 == 2
        assert result_args[0].arg2 == "value"

    def test_set_defaults_with_config(self):
        # [FUNCTION DOC]
        # Checks set_defaults_with_config(arg1 42) makes parsing an empty argv yield arg1 42. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        """Test set_defaults_with_config updates the defaults."""
        parser = TrlParser(dataclass_types=[MyDataclass])

        # Update defaults
        parser.set_defaults_with_config(arg1=42)

        # Ensure the default value is updated
        result_args = parser.parse_args_and_config([])
        assert len(result_args) == 1
        assert isinstance(result_args[0], MyDataclass)
        assert result_args[0].arg1 == 42

    def test_parse_args_and_config_with_remaining_strings(self):
        # [FUNCTION DOC]
        # Checks return_remaining_strings True returns the dataclass plus the unknown positional list ["remaining"]. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        parser = TrlParser(dataclass_types=[MyDataclass])

        args = ["--arg1", "2", "--arg2", "value", "remaining"]

        # Simulate no config, just parse args normally
        result_args = parser.parse_args_and_config(args, return_remaining_strings=True)

        # Check that the arguments are parsed as is
        assert len(result_args) == 2
        assert isinstance(result_args[0], MyDataclass)
        assert result_args[0].arg1 == 2
        assert result_args[0].arg2 == "value"
        assert result_args[1] == ["remaining"]

    @patch("builtins.open", mock_open(read_data="remaining_string_in_config: abc"))
    @patch("yaml.safe_load")
    def test_parse_args_and_config_with_remaining_strings_in_config_and_args(self, mock_yaml_load):
        # [FUNCTION DOC]
        # Checks unknown keys from the config and unknown CLI flags are both returned as remaining strings, config-derived
        # flags first ("--remaining_string_in_config abc" then "--remaining_string_in_args def"). Setup: builtins.open is patched with mock_open and yaml.safe_load is patched to return the config dict, so
        # "config.yaml" is never read from disk.
        # Inputs: mock_yaml_load. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        mock_yaml_load.return_value = {"remaining_string_in_config": "abc"}

        parser = TrlParser(dataclass_types=[MyDataclass])

        args = ["--arg1", "2", "--remaining_string_in_args", "def", "--config", "config.yaml"]

        # Simulate the config being loaded and arguments being passed
        result_args = parser.parse_args_and_config(args, return_remaining_strings=True)

        # Check that the arguments are parsed as is
        assert len(result_args) == 2
        assert isinstance(result_args[0], MyDataclass)
        assert result_args[0].arg1 == 2
        assert result_args[1] == ["--remaining_string_in_config", "abc", "--remaining_string_in_args", "def"]

    @patch("builtins.open", mock_open(read_data="arg1: 2\narg2: config_value"))
    @patch("yaml.safe_load")
    def test_subparsers_with_config_defaults(self, mock_yaml_load):
        # [FUNCTION DOC]
        # Checks config values become defaults for a subparser: parsing "subcommand --config" yields arg1 2 and arg2
        # "config_value". Setup: a bare TrlParser with subparsers (dest "command", parser_class TrlParser) and a "subcommand" parser
        # for MyDataclass; YAML loading mocked to return arg1 2 and a second key. Inputs: mock_yaml_load. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        """Test that config defaults are applied to all subparsers."""
        mock_yaml_load.return_value = {"arg1": 2, "arg2": "config_value"}

        # Create the main parser
        parser = TrlParser()

        # Add subparsers
        subparsers = parser.add_subparsers(dest="command", parser_class=TrlParser)

        # Create a subparser for a specific command
        subparsers.add_parser("subcommand", dataclass_types=[MyDataclass])

        # Parse with config file
        args = ["subcommand", "--config", "config.yaml"]
        result_args = parser.parse_args_and_config(args)

        # Check main parser arguments
        assert len(result_args) == 1

        # Check that config values were applied to the subparser
        assert result_args[0].arg1 == 2  # Default from config
        assert result_args[0].arg2 == "config_value"  # Default from config

    @patch("builtins.open", mock_open(read_data="arg1: 2\narg2: config_value"))
    @patch("yaml.safe_load")
    def test_subparsers_with_config_defaults_and_arg_override(self, mock_yaml_load):
        # [FUNCTION DOC]
        # Checks a CLI flag to a subcommand (--arg1 3) overrides the config while arg2 still comes from the config.
        # Setup: a bare TrlParser with subparsers (dest "command", parser_class TrlParser) and a "subcommand" parser
        # for MyDataclass; YAML loading mocked to return arg1 2 and a second key. Inputs: mock_yaml_load. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        """Test that config defaults are applied to all subparsers."""
        mock_yaml_load.return_value = {"arg1": 2, "arg2": "config_value"}

        # Create the main parser
        parser = TrlParser()

        # Add subparsers
        subparsers = parser.add_subparsers(dest="command", parser_class=TrlParser)

        # Create a subparser for a specific command
        subparsers.add_parser("subcommand", dataclass_types=[MyDataclass])

        # Test with command line arguments overriding config
        args = ["subcommand", "--arg1", "3", "--config", "config.yaml"]
        result_args = parser.parse_args_and_config(args)

        # Command line arguments should override config
        assert result_args[0].arg1 == 3
        assert result_args[0].arg2 == "config_value"  # Still from config

    @patch("builtins.open", mock_open(read_data="arg1: 2\nthis_arg_does_not_exist: config_value"))
    @patch("yaml.safe_load")
    def test_subparsers_with_config_defaults_and_arg_override_wrong_name(self, mock_yaml_load):
        # [FUNCTION DOC]
        # Checks an unknown config key raises ValueError by default and is accepted when fail_with_unknown_args is False.
        # Setup: a bare TrlParser with subparsers (dest "command", parser_class TrlParser) and a "subcommand" parser
        # for MyDataclass; YAML loading mocked to return arg1 2 and a second key.
        # KNOWN ISSUE: its docstring is copied from the defaults test and does not describe the unknown-key behavior tested.
        # Inputs: mock_yaml_load. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        """Test that config defaults are applied to all subparsers."""
        mock_yaml_load.return_value = {"arg1": 2, "this_arg_does_not_exist": "config_value"}

        # Create the main parser
        parser = TrlParser()

        # Add subparsers
        subparsers = parser.add_subparsers(dest="command", parser_class=TrlParser)

        # Create a subparser for a specific command
        subparsers.add_parser("subcommand", dataclass_types=[MyDataclass])

        # Test with command line arguments overriding config
        args = ["subcommand", "--arg1", "3", "--config", "config.yaml"]
        with pytest.raises(ValueError):
            parser.parse_args_and_config(args)

        parser.parse_args_and_config(args, fail_with_unknown_args=False)

    @patch("builtins.open", mock_open(read_data="arg1: 2\narg2: config_value"))
    @patch("yaml.safe_load")
    def test_subparsers_multiple_with_config_defaults(self, mock_yaml_load):
        # [FUNCTION DOC]
        # Checks config defaults apply to each of two subparsers (subcommand0, subcommand1): both parse to arg1 2 and
        # arg2 "config_value". Setup: a bare TrlParser with subparsers (dest "command", parser_class TrlParser) and a "subcommand" parser
        # for MyDataclass; YAML loading mocked to return arg1 2 and a second key. Inputs: mock_yaml_load. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        """Test that config defaults are applied to all subparsers."""
        mock_yaml_load.return_value = {"arg1": 2, "arg2": "config_value"}

        # Create the main parser
        parser = TrlParser()

        # Add subparsers
        subparsers = parser.add_subparsers(dest="command", parser_class=TrlParser)

        # Create a subparser for a specific command
        subparsers.add_parser("subcommand0", dataclass_types=[MyDataclass])
        subparsers.add_parser("subcommand1", dataclass_types=[MyDataclass])

        for idx in range(2):
            # Parse with config file
            args = [f"subcommand{idx}", "--config", "config.yaml"]
            result_args = parser.parse_args_and_config(args)

            # Check main parser arguments
            assert len(result_args) == 1

            # Check that config values were applied to the subparser
            assert result_args[0].arg1 == 2  # Default from config
            assert result_args[0].arg2 == "config_value"  # Default from config


class TestGetDataset:
    # [CLASS DOC]
    # Tests get_dataset on real zen dataset configs. Plain pytest class (no TrlTestCase), no state.
    def test_single_dataset_with_config(self):
        # [FUNCTION DOC]
        # Checks a single language-modeling dataset entry yields the same train split as load_dataset. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: downloads/loads zen dataset configs.
        mixture_config = DatasetMixtureConfig(
            datasets=[DatasetConfig(path="trl-internal-testing/zen", name="standard_language_modeling")]
        )
        result = get_dataset(mixture_config)
        expected = load_dataset("trl-internal-testing/zen", "standard_language_modeling")
        assert expected["train"][:] == result["train"][:]

    def test_single_dataset_preference_config(self):
        # [FUNCTION DOC]
        # Same as the single-dataset test for the standard_preference config. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: downloads/loads zen dataset configs.
        mixture_config = DatasetMixtureConfig(
            datasets=[DatasetConfig(path="trl-internal-testing/zen", name="standard_preference")]
        )
        result = get_dataset(mixture_config)
        expected = load_dataset("trl-internal-testing/zen", "standard_preference")
        assert expected["train"][:] == result["train"][:]

    def test_single_dataset_streaming(self):
        # [FUNCTION DOC]
        # Checks streaming True yields an iterable train split whose rows equal the non-streaming train rows. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: downloads/loads zen dataset configs.
        mixture_config = DatasetMixtureConfig(
            datasets=[DatasetConfig(path="trl-internal-testing/zen", name="standard_language_modeling")],
            streaming=True,
        )
        result = get_dataset(mixture_config)
        expected = load_dataset("trl-internal-testing/zen", "standard_language_modeling")
        assert expected["train"].to_list() == list(result["train"])

    def test_dataset_mixture_basic(self):
        # [FUNCTION DOC]
        # Checks mixing the prompt_completion and preference train splits restricted to the prompt column gives a
        # DatasetDict whose train split has only "prompt", with the first half equal to the preference prompts and the
        # second half equal to the prompt_completion prompts. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: downloads/loads zen dataset configs.
        dataset_config1 = DatasetConfig(
            path="trl-internal-testing/zen", name="standard_prompt_completion", split="train", columns=["prompt"]
        )
        dataset_config2 = DatasetConfig(
            path="trl-internal-testing/zen", name="standard_preference", split="train", columns=["prompt"]
        )
        mixture_config = DatasetMixtureConfig(datasets=[dataset_config1, dataset_config2])
        result = get_dataset(mixture_config)
        assert isinstance(result, DatasetDict)
        assert "train" in result
        train_dataset = result["train"]
        assert train_dataset.column_names == ["prompt"]
        prompts = train_dataset["prompt"]
        expected_first_half = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        assert prompts[: len(prompts) // 2] == expected_first_half["prompt"]
        expected_second_half = load_dataset("trl-internal-testing/zen", "standard_prompt_completion", split="train")
        assert prompts[len(prompts) // 2 :] == expected_second_half["prompt"]

    def test_dataset_mixture_with_weights(self):
        # [FUNCTION DOC]
        # Same as the basic mixture but each source uses split "train[:50%]"; asserts the same half/half ordering.
        # KNOWN ISSUE: despite its name no weights or fraction are set; it only tests percentage split slicing.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: downloads/loads zen dataset configs.
        dataset_config1 = DatasetConfig(
            path="trl-internal-testing/zen", name="standard_prompt_completion", split="train[:50%]", columns=["prompt"]
        )
        dataset_config2 = DatasetConfig(
            path="trl-internal-testing/zen", name="standard_preference", split="train[:50%]", columns=["prompt"]
        )
        mixture_config = DatasetMixtureConfig(datasets=[dataset_config1, dataset_config2])
        result = get_dataset(mixture_config)
        assert isinstance(result, DatasetDict)
        assert "train" in result
        train_dataset = result["train"]
        assert train_dataset.column_names == ["prompt"]
        prompts = train_dataset["prompt"]
        expected_first_half = load_dataset("trl-internal-testing/zen", "standard_preference", split="train[:50%]")
        assert prompts[: len(prompts) // 2] == expected_first_half["prompt"]
        expected_second_half = load_dataset(
            "trl-internal-testing/zen", "standard_prompt_completion", split="train[:50%]"
        )
        assert prompts[len(prompts) // 2 :] == expected_second_half["prompt"]

    def test_dataset_fraction(self):
        # [FUNCTION DOC]
        # Checks fraction 0.75/0.25 on two copies of the same source: the result is the full first source followed by
        # the first round(len / 3) rows of the second (fractions normalized and capped so no source is oversampled). Inputs/Outputs: none. Exceptions: assertion failure. Side effects: downloads/loads zen dataset configs.
        # `fraction` sets each source's target share of the final mixture: normalized to sum to one and capped so no
        # source is oversampled. With weights 0.75/0.25 on two equally-sized sources, the 0.75 source is the binding
        # constraint (kept in full) and the 0.25 source contributes a third of its rows.
        mixture_config = DatasetMixtureConfig(
            datasets=[
                DatasetConfig(path="trl-internal-testing/zen", name="standard_language_modeling", fraction=0.75),
                DatasetConfig(path="trl-internal-testing/zen", name="standard_language_modeling", fraction=0.25),
            ]
        )
        result = get_dataset(mixture_config)
        full = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")
        n_second = round(len(full) / 3)  # 0.25 * (len(full) / 0.75)
        expected = list(full) + list(full.select(range(n_second)))
        assert list(result["train"]) == expected

    def test_dataset_fraction_partial_raises_error(self):
        # [FUNCTION DOC]
        # Checks setting fraction on only one of two datasets raises ValueError ("must be set for either all datasets"). Inputs/Outputs: none. Exceptions: assertion failure. Side effects: downloads/loads zen dataset configs.
        mixture_config = DatasetMixtureConfig(
            datasets=[
                DatasetConfig(path="trl-internal-testing/zen", name="standard_language_modeling", fraction=0.5),
                DatasetConfig(path="trl-internal-testing/zen", name="standard_language_modeling"),
            ]
        )
        with pytest.raises(ValueError, match="must be set for either all datasets"):
            get_dataset(mixture_config)

    def test_dataset_fraction_streaming_raises_error(self):
        # [FUNCTION DOC]
        # Checks fraction combined with streaming raises ValueError ("not supported with streaming datasets"). Inputs/Outputs: none. Exceptions: assertion failure. Side effects: downloads/loads zen dataset configs.
        mixture_config = DatasetMixtureConfig(
            datasets=[DatasetConfig(path="trl-internal-testing/zen", name="standard_language_modeling", fraction=0.5)],
            streaming=True,
        )
        with pytest.raises(ValueError, match="not supported with streaming datasets"):
            get_dataset(mixture_config)

    def test_dataset_mixture_with_test_split(self):
        # [FUNCTION DOC]
        # Checks test_split_size 2 splits the 17-row language-modeling train set into 15 train and 2 test rows. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: downloads/loads zen dataset configs.
        mixture_config = DatasetMixtureConfig(
            datasets=[DatasetConfig(path="trl-internal-testing/zen", name="standard_language_modeling")],
            test_split_size=2,
        )
        result = get_dataset(mixture_config)
        assert isinstance(result, DatasetDict)
        assert "train" in result
        assert "test" in result
        assert len(result["train"]) == 15
        assert len(result["test"]) == 2

    def test_empty_dataset_mixture_raises_error(self):
        # [FUNCTION DOC]
        # Checks an empty datasets list raises ValueError ("No datasets were loaded"). Inputs/Outputs: none. Exceptions: assertion failure. Side effects: none.
        mixture_config = DatasetMixtureConfig(datasets=[])

        with pytest.raises(ValueError, match="No datasets were loaded"):
            get_dataset(mixture_config)

    def test_mixture_multiple_different_configs(self):
        # [FUNCTION DOC]
        # Checks mixing the conversational_preference train split (prompt column) with conversational_prompt_only test
        # split produces a DatasetDict with a non-empty train split. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: downloads/loads zen dataset configs.
        dataset_config1 = DatasetConfig(
            path="trl-internal-testing/zen", name="conversational_preference", split="train", columns=["prompt"]
        )
        dataset_config2 = DatasetConfig(
            path="trl-internal-testing/zen", name="conversational_prompt_only", split="test"
        )
        mixture_config = DatasetMixtureConfig(datasets=[dataset_config1, dataset_config2])
        result = get_dataset(mixture_config)
        assert isinstance(result, DatasetDict)
        assert "train" in result
        assert len(result["train"]) > 0

    def test_trlparser_parses_yaml_config_correctly(self):
        # [FUNCTION DOC]
        # Checks TrlParser parses a YAML file (written to a NamedTemporaryFile) with a datasets list into a
        # DatasetMixtureConfig with two DatasetConfig entries with the right path/name, columns None for the first and
        # ["prompt"] for the second. Inputs/Outputs: none. Exceptions: assertion failure. Side effects: temp file.
        # Prepare YAML content exactly like your example
        # docstyle-ignore
        yaml_content = """
        datasets:
        - path: trl-internal-testing/zen
          name: standard_prompt_only
        - path: trl-internal-testing/zen
          name: standard_preference
          columns:
          - prompt
        """

        # Write YAML to a temporary file
        with tempfile.NamedTemporaryFile("w+", suffix=".yaml") as tmpfile:
            tmpfile.write(yaml_content)
            tmpfile.flush()
            parser = TrlParser((DatasetMixtureConfig,))
            args = parser.parse_args_and_config(args=["--config", tmpfile.name])[0]

        # Assert that we got DatasetMixtureConfig instance
        assert isinstance(args, DatasetMixtureConfig)

        # Assert datasets list length
        assert len(args.datasets) == 2

        # Check first dataset
        dataset_config1 = args.datasets[0]
        assert isinstance(dataset_config1, DatasetConfig)
        assert dataset_config1.path == "trl-internal-testing/zen"
        assert dataset_config1.name == "standard_prompt_only"
        assert dataset_config1.columns is None  # No columns specified

        # Check second dataset
        dataset_config2 = args.datasets[1]
        assert isinstance(dataset_config2, DatasetConfig)
        assert dataset_config2.path == "trl-internal-testing/zen"
        assert dataset_config2.name == "standard_preference"
        assert dataset_config2.columns == ["prompt"]  # Columns specified

    def test_trlparser_parses_yaml_and_loads_dataset(self):
        # [FUNCTION DOC]
        # Checks a YAML-parsed DatasetMixtureConfig passed to get_dataset yields the same train split as load_dataset.
        # Inputs/Outputs: none. Exceptions: assertion failure. Side effects: temp file; loads the dataset.
        # Prepare YAML content exactly like your example
        # docstyle-ignore
        yaml_content = """
        datasets:
        - path: trl-internal-testing/zen
          name: standard_language_modeling
        """

        # Write YAML to a temporary file
        with tempfile.NamedTemporaryFile("w+", suffix=".yaml") as tmpfile:
            tmpfile.write(yaml_content)
            tmpfile.flush()
            parser = TrlParser((DatasetMixtureConfig,))
            args = parser.parse_args_and_config(args=["--config", tmpfile.name])[0]

        # Load the dataset using get_dataset
        result = get_dataset(args)
        expected = load_dataset("trl-internal-testing/zen", "standard_language_modeling")
        assert expected["train"][:] == result["train"][:]
