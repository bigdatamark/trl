# [MODULE DOC]
# Module trl.import_utils: optional-dependency detection and warning helpers for TRL.
# Main contents: _is_package_available (find_spec plus optional version lookup), one is_<pkg>_available
# function per optional dependency (aiohttp, deepspeed, harbor, jmespath, liger_kernel with a minimum version,
# math_verify, mergekit, openreward, requests, unsloth, vllm with supported-range warning, weave),
# TRLExperimentalWarning and the suppress_warning / suppress_experimental_warning context managers.
# Import-time side effects: computes PACKAGE_DISTRIBUTION_MAPPING (import name to distributions) via
# importlib.metadata.packages_distributions; defines LIGER_KERNEL_MIN_VERSION "0.8.2".

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
import importlib.metadata
import importlib.util
import warnings
from contextlib import contextmanager

from packaging.version import Version


LIGER_KERNEL_MIN_VERSION = "0.8.2"
PACKAGE_DISTRIBUTION_MAPPING = importlib.metadata.packages_distributions()


# From transformers: https://github.com/huggingface/transformers/blob/556312cd45a5e619c41b0f8adf680eab0d334324/src/transformers/utils/import_utils.py#L48-L77
def _is_package_available(pkg_name: str, return_version: bool = False) -> tuple[bool, str] | bool:
    # [FUNCTION DOC]
    # HELPER: Check whether a package is importable and optionally get its version.
    # Behavior: existence via importlib.util.find_spec. When return_version and the package exists, find its
    # distribution: prefer the name with underscores turned into hyphens, then the exact name, else the first
    # distribution listed; read its metadata version. If metadata or mapping lookup fails, import the package and
    # use its __version__ (or "N/A").
    # Inputs: pkg_name (str), return_version (default False).
    # Outputs: bool, or (bool, version string) when return_version (version "N/A" if absent).
    # Exceptions: import errors from the fallback import. Side effects: the fallback may import the package.
    """Check if `pkg_name` exist, and optionally try to get its version"""
    spec = importlib.util.find_spec(pkg_name)
    package_exists = spec is not None
    package_version = "N/A"
    if package_exists and return_version:
        try:
            # importlib.metadata works with the distribution package, which may be different from the import
            # name (e.g. `PIL` is the import name, but `pillow` is the distribution name)
            distributions = PACKAGE_DISTRIBUTION_MAPPING[pkg_name]
            # Per PEP 503, underscores and hyphens are equivalent in package names.
            # Prefer the distribution that matches the (normalized) package name.
            normalized_pkg_name = pkg_name.replace("_", "-")
            if normalized_pkg_name in distributions:
                distribution_name = normalized_pkg_name
            elif pkg_name in distributions:
                distribution_name = pkg_name
            else:
                distribution_name = distributions[0]
            package_version = importlib.metadata.version(distribution_name)
        except (importlib.metadata.PackageNotFoundError, KeyError):
            # If we cannot find the metadata (because of editable install for example), try to import directly.
            # Note that this branch will almost never be run, so we do not import packages for nothing here
            package = importlib.import_module(pkg_name)
            package_version = getattr(package, "__version__", "N/A")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


def is_aiohttp_available() -> bool:
    # [FUNCTION DOC]
    # HELPER: True if aiohttp is importable. Inputs: none. Outputs: bool. Exceptions: none. Side effects: none.
    return _is_package_available("aiohttp")


def is_deepspeed_available() -> bool:
    # [FUNCTION DOC]
    # HELPER: True if deepspeed is importable. Inputs: none. Outputs: bool. Exceptions: none. Side effects: none.
    return _is_package_available("deepspeed")


def is_harbor_available() -> bool:
    # [FUNCTION DOC]
    # HELPER: True if harbor is importable. Inputs: none. Outputs: bool. Exceptions: none. Side effects: none.
    return _is_package_available("harbor")


def is_jmespath_available() -> bool:
    # [FUNCTION DOC]
    # HELPER: True if jmespath is importable. Inputs: none. Outputs: bool. Exceptions: none. Side effects: none.
    return _is_package_available("jmespath")


def is_liger_kernel_available(min_version: str = LIGER_KERNEL_MIN_VERSION) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if liger_kernel is installed at version min_version (default LIGER_KERNEL_MIN_VERSION) or later.
    # Inputs: min_version (str). Outputs: bool.
    # Exceptions: InvalidVersion if the reported version is unparsable (e.g. "N/A"). Side effects: none.
    _liger_kernel_available, _liger_kernel_version = _is_package_available("liger_kernel", return_version=True)
    return _liger_kernel_available and Version(_liger_kernel_version) >= Version(min_version)


def is_math_verify_available() -> bool:
    # [FUNCTION DOC]
    # HELPER: True if math_verify is importable. Inputs: none. Outputs: bool. Exceptions: none. Side effects: none.
    return _is_package_available("math_verify")


def is_mergekit_available() -> bool:
    # [FUNCTION DOC]
    # HELPER: True if mergekit is importable. Inputs: none. Outputs: bool. Exceptions: none. Side effects: none.
    return _is_package_available("mergekit")


def is_openreward_available() -> bool:
    # [FUNCTION DOC]
    # HELPER: True if openreward is importable. Inputs: none. Outputs: bool. Exceptions: none. Side effects: none.
    return _is_package_available("openreward")


def is_requests_available() -> bool:
    # [FUNCTION DOC]
    # HELPER: True if requests is importable. Inputs: none. Outputs: bool. Exceptions: none. Side effects: none.
    return _is_package_available("requests")


def is_unsloth_available() -> bool:
    # [FUNCTION DOC]
    # HELPER: True if unsloth is importable. Inputs: none. Outputs: bool. Exceptions: none. Side effects: none.
    return _is_package_available("unsloth")


def is_vllm_available(min_version: str | None = None) -> bool:
    # [FUNCTION DOC]
    # HELPER: Check vLLM availability, optionally with a minimum version.
    # Behavior: when installed, warn (every call) if the base version (local segment dropped) is outside the
    # supported range 0.22.0 to 0.31.0 inclusive; return False if min_version is given and the full installed
    # version is lower; otherwise return whether vLLM is installed.
    # Inputs: min_version (optional str). Outputs: bool.
    # Exceptions: InvalidVersion for unparsable versions. Side effects: UserWarning emission.
    _vllm_available, _vllm_version = _is_package_available("vllm", return_version=True)
    if _vllm_available:
        # Use base_version to drop any local segment (e.g. the "+cu129" in "0.24.0+cu129"), which PEP 440 orders
        # above the plain release and would otherwise fail the upper-bound check.
        _vllm_base_version = Version(Version(_vllm_version).base_version)
        if not (Version("0.22.0") <= _vllm_base_version <= Version("0.31.0")):
            warnings.warn(
                f"TRL currently supports vLLM versions from 0.22.0 to 0.31.0. You have version {_vllm_version} "
                "installed. We recommend installing a supported version to avoid compatibility issues.",
                stacklevel=2,
            )
        if min_version is not None and Version(_vllm_version) < Version(min_version):
            return False
    return _vllm_available


def is_weave_available() -> bool:
    # [FUNCTION DOC]
    # HELPER: True if weave is importable. Inputs: none. Outputs: bool. Exceptions: none. Side effects: none.
    return _is_package_available("weave")


class TRLExperimentalWarning(UserWarning):
    # [CLASS DOC]
    # UserWarning subclass emitted when trl.experimental is used, so it can be filtered by category. No state.
    """Warning for using the 'trl.experimental' submodule."""

    pass


@contextmanager
def suppress_warning(category):
    # [FUNCTION DOC]
    # HELPER: Context manager that ignores warnings of the given category inside the block, restoring the filters
    # afterwards (warnings.catch_warnings).
    # Inputs: category (Warning subclass). Outputs: yields None. Exceptions: none.
    # Side effects: temporarily alters the global warning filters.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=category)
        yield


def suppress_experimental_warning():
    # [FUNCTION DOC]
    # HELPER: Return suppress_warning(TRLExperimentalWarning), a context manager silencing experimental warnings.
    # Inputs: none. Outputs: context manager. Exceptions: none. Side effects: none until entered.
    return suppress_warning(TRLExperimentalWarning)
