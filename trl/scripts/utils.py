# [MODULE DOC]
# Shared utilities for TRL CLI scripts: dataset mixture configs (DatasetConfig, DatasetMixtureConfig) and
# get_dataset, common ScriptArguments, TrlParser (YAML-config-aware argument parser built on the vendored
# HfArgumentParser), init_zero_verbose (quiet logging setup) and get_git_commit_hash.
# Import-time side effects: creates a module logger; imports the local _hf_argparser (not transformers') to
# avoid an upstream import-latency issue.

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

import argparse
import importlib
import inspect
import logging
import os
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# Temporarily import from the local module instead of transformers to avoid an upstream latency issue
# See https://github.com/huggingface/transformers/issues/44273
# This workaround can be reverted once the fix is included in the minimum required transformers version
from trl.scripts._hf_argparser import DataClass, DataClassType, HfArgumentParser


if TYPE_CHECKING:
    from datasets import DatasetDict


logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    # [CLASS DOC]
    # Dataclass describing one dataset to load with datasets.load_dataset: path (required), name, data_dir,
    # data_files, split (default "train"), columns (optional column subset) and fraction (optional target share of
    # the mixture; must be set for all datasets or none; not for streaming). Consumed by get_dataset.
    """
    Configuration for a dataset.

    This class matches the signature of [`~datasets.load_dataset`] and the arguments are used directly in the
    [`~datasets.load_dataset`] function. You can refer to the [`~datasets.load_dataset`] documentation for more
    details.

    Parameters:
        path (`str`):
            Path or name of the dataset.
        name (`str`, *optional*):
            Defining the name of the dataset configuration.
        data_dir (`str`, *optional*):
            Defining the `data_dir` of the dataset configuration. If specified for the generic builders(csv, text etc.)
            or the Hub datasets and `data_files` is `None`, the behavior is equal to passing `os.path.join(data_dir,
            **)` as `data_files` to reference all the files in a directory.
        data_files (`str` or `Sequence` or `Mapping`, *optional*):
            Path(s) to source data file(s).
        split (`str`, *optional*, defaults to `"train"`):
            Which split of the data to load.
        columns (`list[str]`, *optional*):
            List of column names to select from the dataset. If `None`, all columns are selected.
        fraction (`float`, *optional*):
            Target share of this dataset in the final mixture. Fractions are normalized to sum to one across all
            datasets, and the mixture size is capped so that no dataset is oversampled: the first `round(fraction * N)`
            rows of each dataset are kept, where `N` is the largest mixture size that avoids oversampling. Must be set
            for either all datasets in the mixture or none of them. Not supported for streaming datasets. When unset,
            the full datasets are concatenated.
    """

    path: str
    name: str | None = None
    data_dir: str | None = None
    data_files: str | list[str] | dict[str, str] | None = None
    split: str = "train"
    columns: list[str] | None = None
    fraction: float | None = None


@dataclass
class DatasetMixtureConfig:
    # [CLASS DOC]
    # Dataclass for a mixture of datasets parsed from CLI/YAML: datasets (list of DatasetConfig, default empty),
    # streaming (default False), test_split_size (optional float/int passed as test_size to train_test_split).
    # __post_init__ converts dict entries (as produced by YAML parsing) into DatasetConfig objects.
    """
    Configuration class for a mixture of datasets.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        datasets (`list[DatasetConfig]`):
            List of dataset configurations to include in the mixture.
        streaming (`bool`, *optional*, defaults to `False`):
            Whether to stream the datasets. If `True`, the datasets will be loaded in streaming mode.
        test_split_size (`float`, *optional*):
            Size of the test split. Refer to the `test_size` parameter in the [`~datasets.train_test_split`] function
            for more details. If `None`, the dataset will not be split into train and test sets.

    Usage:
        When using the CLI, you can add the following section to your YAML config file:

        ```yaml
        datasets:
          - path: ...
            name: ...
            data_dir: ...
            data_files: ...
            split: ...
            columns: ...
            fraction: ...
          - path: ...
            name: ...
            data_dir: ...
            data_files: ...
            split: ...
            columns: ...
        streaming: ...
        test_split_size: ...
        ```
    """

    datasets: list[DatasetConfig] = field(
        default_factory=list,
        metadata={"help": "List of dataset configurations to include in the mixture."},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Whether to stream the datasets. If True, the datasets will be loaded in streaming mode."},
    )
    test_split_size: float | None = field(
        default=None,
        metadata={
            "help": "Size of the test split. Refer to the `test_size` parameter in the `datasets.train_test_split` "
            "function for more details. If None, the dataset will not be split into train and test sets."
        },
    )

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: replaces, in place, every dict in self.datasets with DatasetConfig built from its keys.
        # Inputs: none. Outputs: none.
        # Exceptions: TypeError for unknown or missing DatasetConfig keys.
        # Side effects: mutates self.datasets.
        # Convert any dataset dicts (from CLI/config parsing) into DatasetConfig objects
        for idx, dataset in enumerate(self.datasets):
            if isinstance(dataset, dict):
                # If it's a dict, convert it to DatasetConfig
                self.datasets[idx] = DatasetConfig(**dataset)


@dataclass
class ScriptArguments:
    # [CLASS DOC]
    # Dataclass of arguments common to all TRL scripts: dataset_name, dataset_config (both optional),
    # dataset_train_split ("train"), dataset_test_split ("test"), dataset_streaming (False), and
    # ignore_bias_buffers (False; DPO script DDP workaround). The dataset fields are ignored when a
    # DatasetMixtureConfig.datasets list is provided. No methods.
    """
    Arguments common to all scripts.

    Args:
        dataset_name (`str`,, *optional*):
            Path or name of the dataset to load. If `datasets` is provided, this will be ignored.
        dataset_config (`str`, *optional*):
            Dataset configuration name. Corresponds to the `name` argument of the [`~datasets.load_dataset`] function.
            If `datasets` is provided, this will be ignored.
        dataset_train_split (`str`, *optional*, defaults to `"train"`):
            Dataset split to use for training. If `datasets` is provided, this will be ignored.
        dataset_test_split (`str`, *optional*, defaults to `"test"`):
            Dataset split to use for evaluation. If `datasets` is provided, this will be ignored.
        dataset_streaming (`bool`, *optional*, defaults to `False`):
            Whether to stream the dataset. If True, the dataset will be loaded in streaming mode. If `datasets` is
            provided, this will be ignored.
        ignore_bias_buffers (`bool`, *optional*, defaults to `False`):
            Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid scalar
            type, inplace operation. See
            https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992.
    """

    dataset_name: str | None = field(
        default=None,
        metadata={"help": "Path or name of the dataset to load. If `datasets` is provided, this will be ignored."},
    )
    dataset_config: str | None = field(
        default=None,
        metadata={
            "help": "Dataset configuration name. Corresponds to the `name` argument of the `datasets.load_dataset` "
            "function. If `datasets` is provided, this will be ignored."
        },
    )
    dataset_train_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use for training. If `datasets` is provided, this will be ignored."},
    )
    dataset_test_split: str = field(
        default="test",
        metadata={"help": "Dataset split to use for evaluation. If `datasets` is provided, this will be ignored."},
    )
    dataset_streaming: bool = field(
        default=False,
        metadata={
            "help": "Whether to stream the dataset. If True, the dataset will be loaded in streaming mode. If "
            "`datasets` is provided, this will be ignored."
        },
    )
    ignore_bias_buffers: bool = field(
        default=False,
        metadata={
            "help": "Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid "
            "scalar type, inplace operation. See "
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992."
        },
    )


def init_zero_verbose():
    # [FUNCTION DOC]
    # Behavior: configures root logging at ERROR level with "%(message)s" format and "[%X]" date format, using a
    # Rich handler when rich is installed else a StreamHandler, and replaces warnings.showwarning so warnings are
    # emitted through logging.warning as "file:line: Category: message".
    # Inputs: none. Outputs: none. Exceptions: none expected.
    # Side effects: global logging configuration and global warnings hook.
    """
    Perform zero verbose init - use this method on top of the CLI modules to make logging and warning output cleaner.
    Uses Rich if available, falls back otherwise.
    """
    import logging
    import warnings

    from transformers.utils import is_rich_available

    FORMAT = "%(message)s"

    if is_rich_available():
        from rich.logging import RichHandler

        handler = RichHandler()
    else:
        handler = logging.StreamHandler()

    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[handler], level=logging.ERROR)

    # Custom warning handler to redirect warnings to the logging system
    def warning_handler(message, category, filename, lineno, file=None, line=None):
        # [FUNCTION DOC]
        # HELPER: Replacement for warnings.showwarning that logs the warning at WARNING level (file and line args are
        # accepted but unused). Inputs: standard showwarning args. Outputs: none. Exceptions: none.
        # Side effects: writes a log record.
        logging.warning(f"{filename}:{lineno}: {category.__name__}: {message}")

    # Add the custom warning handler - we need to do that before importing anything to make sure the loggers work well
    warnings.showwarning = warning_handler


class TrlParser(HfArgumentParser):
    # [CLASS DOC]
    # HfArgumentParser subclass adding YAML config file support (via "--config") with an optional "env" section
    # that sets environment variables, and the ability to set argument defaults (including in subparsers) from a
    # config. Construction forbids dataclasses with a field named "config". Main entry: parse_args_and_config;
    # set_defaults_with_config is its helper (also used by the trl CLI).
    """
    A subclass of [`transformers.HfArgumentParser`] designed for parsing command-line arguments with dataclass-backed
    configurations, while also supporting configuration file loading and environment variable management.

    Args:
        dataclass_types (`DataClassType | Iterable[DataClassType]`, *optional*):
            Dataclass types to use for argument parsing.
        **kwargs:
            Additional keyword arguments passed to the [`transformers.HfArgumentParser`] constructor.

    Examples:

    ```yaml
    # config.yaml
    env:
        VAR1: value1
    arg1: 23
    ```

    ```python
    # main.py
    import os
    from dataclasses import dataclass
    from trl import TrlParser


    @dataclass
    class MyArguments:
        arg1: int
        arg2: str = "alpha"


    parser = TrlParser(dataclass_types=[MyArguments])
    training_args = parser.parse_args_and_config()

    print(training_args, os.environ.get("VAR1"))
    ```

    ```bash
    $ python main.py --config config.yaml
    (MyArguments(arg1=23, arg2='alpha'),) value1

    $ python main.py --arg1 5 --arg2 beta
    (MyArguments(arg1=5, arg2='beta'),) None
    ```
    """

    def __init__(
        self,
        dataclass_types: DataClassType | Iterable[DataClassType] | None = None,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Behavior: normalizes dataclass_types to a list, rejects any dataclass that declares a "config" field, then
        # initializes HfArgumentParser.
        # Inputs: dataclass_types (type, iterable or None); kwargs forwarded.
        # Outputs: none.
        # Exceptions: ValueError for a reserved "config" field.
        # Side effects: registers arguments.
        # Make sure dataclass_types is an iterable
        if dataclass_types is None:
            dataclass_types = []
        elif not isinstance(dataclass_types, Iterable):
            dataclass_types = [dataclass_types]

        # Check that none of the dataclasses have the "config" field
        for dataclass_type in dataclass_types:
            if "config" in dataclass_type.__dataclass_fields__:
                raise ValueError(
                    f"Dataclass {dataclass_type.__name__} has a field named 'config'. This field is reserved for the "
                    f"config file path and should not be used in the dataclass."
                )

        super().__init__(dataclass_types=dataclass_types, **kwargs)

    def parse_args_and_config(
        self,
        args: Iterable[str] | None = None,
        return_remaining_strings: bool = False,
        fail_with_unknown_args: bool = True,
        separate_remaining_strings: bool = False,
    ) -> tuple[DataClass, ...]:
        # [FUNCTION DOC]
        # Behavior: copies args (default sys.argv without program name). If "--config" is present, removes the flag
        # and its following path, safe-loads the YAML (must be a mapping), pops "env" (must be a dict) and sets each
        # entry in os.environ as a string, then applies remaining keys as parser defaults via
        # set_defaults_with_config, collecting unconsumed keys as "--key value" strings. Then parses the CLI with
        # parse_args_into_dataclasses (CLI overrides config). When return_remaining_strings, returns the dataclasses
        # plus either one merged list (config leftovers then CLI leftovers) or, with separate_remaining_strings, two
        # lists. Otherwise raises if config leftovers exist and fail_with_unknown_args, else returns the dataclasses.
        # Inputs: args, return_remaining_strings (False), fail_with_unknown_args (True), separate_remaining_strings
        # (False).
        # Outputs: tuple of dataclass instances (plus remaining-string lists as described).
        # Exceptions: ValueError for non-mapping config, non-dict env, or unknown config keys; ImportError without
        # PyYAML; ValueError from the base parser for unknown CLI args.
        # Side effects: reads the config file, sets environment variables, mutates parser action defaults.
        # KNOWN ISSUE: "--config" given as the last argument with no path raises an uncaught IndexError; the
        # "--config=path" form is not recognized.
        """
        Parse command-line args and config file into instances of the specified dataclass types.

        This method wraps [`transformers.HfArgumentParser.parse_args_into_dataclasses`] and also parses the config file
        specified with the `--config` flag. The config file (in YAML format) provides argument values that replace the
        default values in the dataclasses. Command line arguments can override values set by the config file. The
        method also sets any environment variables specified in the `env` field of the config file.
        """
        import yaml

        args = list(args) if args is not None else sys.argv[1:]
        if "--config" in args:
            # Get the config file path from
            config_index = args.index("--config")
            args.pop(config_index)  # remove the --config flag
            config_path = args.pop(config_index)  # get the path to the config file
            with open(config_path) as yaml_file:
                config = yaml.safe_load(yaml_file)
            if not isinstance(config, dict):
                raise ValueError(f"Config file {config_path} must contain a YAML mapping.")

            # Set the environment variables specified in the config file
            if "env" in config:
                env_vars = config.pop("env", {})
                if not isinstance(env_vars, dict):
                    raise ValueError("`env` field should be a dict in the YAML file.")
                for key, value in env_vars.items():
                    os.environ[key] = str(value)

            # Set the defaults from the config values
            config_remaining_strings = self.set_defaults_with_config(**config)
        else:
            config_remaining_strings = []

        # Parse the arguments from the command line
        output = self.parse_args_into_dataclasses(args=args, return_remaining_strings=return_remaining_strings)

        # Merge remaining strings from the config file with the remaining strings from the command line
        if return_remaining_strings:
            args_remaining_strings = output[-1]
            if separate_remaining_strings:
                return output[:-1] + (config_remaining_strings, args_remaining_strings)
            return output[:-1] + (config_remaining_strings + args_remaining_strings,)
        elif fail_with_unknown_args and config_remaining_strings:
            raise ValueError(
                f"Unknown arguments from config file: {config_remaining_strings}. Please remove them, add them to the "
                "dataclass, or set `fail_with_unknown_args=False`."
            )
        else:
            return output

    def set_defaults_with_config(self, **kwargs) -> list[str]:
        # [FUNCTION DOC]
        # Behavior: walks the parser actions recursively (descending into every subparser); for each action whose
        # dest is a key in kwargs, sets its default to that value and marks it not required. Keys not consumed by any
        # action are returned as flattened "--key", "str(value)" pairs.
        # Inputs: kwargs (config values). Outputs: list of str leftovers.
        # Exceptions: none expected. Side effects: mutates argparse actions of this parser and its subparsers.
        """
        Overrides the parser's default values with those provided via keyword arguments, including for subparsers.

        Any argument with an updated default will also be marked as not required if it was previously required.

        Returns a list of strings that were not consumed by the parser.
        """

        def apply_defaults(parser, kw):
            # [FUNCTION DOC]
            # HELPER: Recursive worker: applies defaults on one parser and its subparsers; returns the set of used keys.
            # Inputs: parser, kw dict. Outputs: set of str. Exceptions: none. Side effects: mutates actions.
            used_keys = set()
            for action in parser._actions:
                # Handle subparsers recursively
                if isinstance(action, argparse._SubParsersAction):
                    for subparser in action.choices.values():
                        used_keys.update(apply_defaults(subparser, kw))
                elif action.dest in kw:
                    action.default = kw[action.dest]
                    action.required = False
                    used_keys.add(action.dest)
            return used_keys

        used_keys = apply_defaults(self, kwargs)
        # Remaining args not consumed by the parser
        remaining = [
            item for key, value in kwargs.items() if key not in used_keys for item in (f"--{key}", str(value))
        ]
        return remaining


def get_git_commit_hash(package_name):
    # [FUNCTION DOC]
    # HELPER: Returns the git HEAD hash of the repository containing an installed package.
    # Behavior: imports the package, takes the parent of its directory as the repo root; if it contains a ".git"
    # directory, runs git rev-parse HEAD there and returns the decoded hash, else None.
    # Inputs: package_name (str). Outputs: hash str, None, or an "Error: ..." string on any exception.
    # Exceptions: none raised (all caught and returned as a string).
    # Side effects: imports the package; spawns a git subprocess.
    # KNOWN ISSUE: on failure it returns an "Error: ..." string, which callers such as print_env treat as a commit
    # hash (slicing its first 7 characters into the version string).
    try:
        # Import the package to locate its path
        package = importlib.import_module(package_name)
        # Get the path to the package using inspect
        package_path = os.path.dirname(inspect.getfile(package))

        # Navigate up to the Git repository root if the package is inside a subdirectory
        git_repo_path = os.path.abspath(os.path.join(package_path, ".."))
        git_dir = os.path.join(git_repo_path, ".git")

        if os.path.isdir(git_dir):
            # Run the git command to get the current commit hash
            commit_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=git_repo_path).strip().decode("utf-8")
            )
            return commit_hash
        else:
            return None
    except Exception as e:
        return f"Error: {str(e)}"


def get_dataset(mixture_config: DatasetMixtureConfig) -> "DatasetDict":
    # [FUNCTION DOC]
    # Behavior: loads each DatasetConfig with datasets.load_dataset (path, name, data_dir, data_files, split,
    # mixture streaming flag), selecting columns when given. If fractions are used: requires all set and
    # non-streaming, normalizes them to weights, computes total as the minimum over positive-weight datasets of
    # length / weight, and keeps the first round(weight times total) rows of each. Concatenates all datasets
    # (logging the size for non-iterable datasets); with test_split_size returns train_test_split's DatasetDict,
    # else a DatasetDict with only "train".
    # Inputs: mixture_config (DatasetMixtureConfig).
    # Outputs: datasets.DatasetDict.
    # Exceptions: ValueError for partial fractions, fractions with streaming, or an empty mixture; dataset loading
    # errors; ZeroDivisionError if all fractions sum to zero.
    # Side effects: downloads/caches datasets; logs info messages.
    """
    Load a mixture of datasets based on the configuration.

    Args:
        mixture_config ([`DatasetMixtureConfig`]):
            Script arguments containing dataset configuration.

    Returns:
        [`~datasets.DatasetDict`]:
            Combined dataset(s) from the mixture configuration, with optional train/test split if `test_split_size` is
            set.

    Example:
    ```python
    >>> from trl import DatasetMixtureConfig, get_dataset
    >>> from trl.scripts.utils import DatasetConfig

    >>> mixture_config = DatasetMixtureConfig(datasets=[DatasetConfig(path="trl-lib/tldr")])
    >>> get_dataset(mixture_config)
    DatasetDict({
        train: Dataset({
            features: ['prompt', 'completion'],
            num_rows: 116722
        })
    })
    ```
    """
    import datasets

    logger.info(f"Creating dataset mixture with {len(mixture_config.datasets)} datasets")
    datasets_list = []
    for dataset_config in mixture_config.datasets:
        logger.info(f"Loading dataset for mixture: {dataset_config.path} (config name: {dataset_config.name})")
        dataset = datasets.load_dataset(
            path=dataset_config.path,
            name=dataset_config.name,
            data_dir=dataset_config.data_dir,
            data_files=dataset_config.data_files,
            split=dataset_config.split,
            streaming=mixture_config.streaming,
        )
        if dataset_config.columns is not None:
            dataset = dataset.select_columns(dataset_config.columns)
        datasets_list.append(dataset)

    # If `fraction` is set, treat the values as target shares of the final mixture. They are normalized to sum to one,
    # and the mixture size is capped so that no dataset is oversampled: we keep the first `round(weight * total)` rows of
    # each dataset, where `total` is the largest mixture size such that no dataset contributes more rows than it has.
    fractions = [dataset_config.fraction for dataset_config in mixture_config.datasets]
    if any(fraction is not None for fraction in fractions):
        if any(fraction is None for fraction in fractions):
            raise ValueError("`fraction` must be set for either all datasets in the mixture or none of them.")
        if mixture_config.streaming:
            raise ValueError("Using a dataset `fraction` is not supported with streaming datasets.")
        weights = [fraction / sum(fractions) for fraction in fractions]
        total = min(
            len(dataset) / weight for dataset, weight in zip(datasets_list, weights, strict=False) if weight > 0
        )
        datasets_list = [
            dataset.select(range(round(weight * total)))
            for dataset, weight in zip(datasets_list, weights, strict=False)
        ]

    if datasets_list:
        combined_dataset = datasets.concatenate_datasets(datasets_list)
        if isinstance(combined_dataset, datasets.Dataset):  # IterableDataset does not have a length
            logger.info(f"Created dataset mixture with {len(combined_dataset)} examples")

        if mixture_config.test_split_size is not None:
            logger.info(f"Splitting dataset into train and test sets with test size: {mixture_config.test_split_size}")
            combined_dataset = combined_dataset.train_test_split(test_size=mixture_config.test_split_size)
            return combined_dataset
        else:
            return datasets.DatasetDict({"train": combined_dataset})
    else:
        raise ValueError("No datasets were loaded from the mixture configuration")
