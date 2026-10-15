# [MODULE DOC]
# HELPER: Compatibility shims for third-party dependencies, applied automatically when trl is imported.
# Contents: version-comparison helpers (_is_package_version_below, _is_package_version_at_least) and patch functions
# (_patch_vllm_logging, _patch_transformers_hybrid_cache).
# Import-time side effects: runs both patches at import - may set the VLLM_LOGGING_LEVEL environment variable and may
# monkeypatch transformers.cache_utils and transformers' _LazyModule.__init__.

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

"""
Compatibility shims for third-party dependencies.

This module contains temporary patches to handle version incompatibilities between TRL's dependencies.

Each patch should be removed when minimum version requirements eliminate the need.
"""

import warnings

from packaging.version import Version

from .import_utils import _is_package_available


def _is_package_version_below(package_name: str, version_threshold: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: Returns True when package_name is installed and its version is strictly below version_threshold.
    # Inputs: package_name (str), version_threshold (str, PEP 440 version).
    # Outputs: bool; False when the package is missing or the check fails.
    # Exceptions: none - any exception during lookup/parsing is caught, a warning is emitted, and False is returned.
    # Side effects: may emit a UserWarning.
    """
    Check if installed package version is below the given threshold.

    Args:
        package_name (`str`):
            Package name.
        version_threshold (`str`):
            Maximum version threshold.

    Returns:
        - True if package is installed and version < version_threshold.
        - False if package is not installed or version >= version_threshold.
    """
    try:
        is_available, version = _is_package_available(package_name, return_version=True)
        return is_available and Version(version) < Version(version_threshold)
    except Exception as e:
        warnings.warn(
            f"Failed to check {package_name} version against {version_threshold}: {e}. "
            f"Compatibility patch may not be applied.",
            stacklevel=2,
        )
        return False


def _is_package_version_at_least(package_name: str, version_threshold: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: Returns True when package_name is installed and its version is greater than or equal to version_threshold.
    # Inputs/outputs/exceptions/side effects: same as _is_package_version_below, with the comparison reversed.
    """
    Check if installed package version is at least the given threshold.

    Args:
        package_name (`str`):
            Package name.
        version_threshold (`str`):
            Minimum version threshold.

    Returns:
        - True if package is installed and version >= version_threshold.
        - False if package is not installed or version < version_threshold.
    """
    try:
        is_available, version = _is_package_available(package_name, return_version=True)
        return is_available and Version(version) >= Version(version_threshold)
    except Exception as e:
        warnings.warn(
            f"Failed to check {package_name} version against {version_threshold}: {e}. "
            f"Compatibility patch may not be applied.",
            stacklevel=2,
        )
        return False


def _patch_vllm_logging() -> None:
    # [FUNCTION DOC]
    # HELPER: When vLLM is installed, sets VLLM_LOGGING_LEVEL to ERROR unless the user already set it.
    # Inputs: none. Outputs: none. Exceptions: none.
    # Side effects: writes os.environ["VLLM_LOGGING_LEVEL"].
    """Set vLLM logging level to ERROR by default to reduce noise."""
    if _is_package_available("vllm"):
        import os

        os.environ["VLLM_LOGGING_LEVEL"] = os.getenv("VLLM_LOGGING_LEVEL", "ERROR")


def _patch_transformers_hybrid_cache() -> None:
    # [FUNCTION DOC]
    # HELPER: Restores the removed transformers HybridCache name for peft<0.18 on transformers>=5.
    # Behavior: only when transformers>=5.0.0 and peft<0.18.0: alias transformers.cache_utils.HybridCache to Cache (for
    # liger_kernel), then wrap transformers' _LazyModule.__init__ so that, when the "transformers" lazy module is built,
    # "HybridCache" is registered in its import structure under cache_utils, in its class-to-module map, in __all__, and
    # set as an attribute equal to Cache.
    # Inputs: none. Outputs: none.
    # Exceptions: none - failures are caught and reported as a warning.
    # Side effects: monkeypatches transformers.cache_utils and _LazyModule.__init__ globally.
    """
    Fix HybridCache import for transformers v5 compatibility.

    - Issue: peft import HybridCache from transformers.cache_utils
    - HybridCache removed in https://github.com/huggingface/transformers/pull/43168 (transformers>=5.0.0)
    - Fixed in peft: https://github.com/huggingface/peft/pull/2735 (released in v0.18.0)
    - This can be removed when TRL requires peft>=0.18.0
    """
    if _is_package_version_at_least("transformers", "5.0.0") and _is_package_version_below("peft", "0.18.0"):
        try:
            import transformers.cache_utils
            from transformers.utils.import_utils import _LazyModule

            Cache = transformers.cache_utils.Cache

            # Patch for liger_kernel: Add HybridCache as an alias for Cache in the cache_utils module
            transformers.cache_utils.HybridCache = Cache

            # Patch for peft: Patch _LazyModule.__init__ to add HybridCache to transformers' lazy loading structures
            _original_lazy_module_init = _LazyModule.__init__

            def _patched_lazy_module_init(self, name, *args, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Replacement for _LazyModule.__init__; calls the original and, if the module name is "transformers", injects
                # HybridCache into _import_structure["cache_utils"], _class_to_module, __all__ (each only if present / not already
                # there) and as an attribute aliasing Cache.
                # Inputs: self, name, and the original init's args/kwargs. Outputs: none.
                # Exceptions: whatever the original init raises. Side effects: mutates the lazy module instance.
                _original_lazy_module_init(self, name, *args, **kwargs)
                if name == "transformers":
                    # Update _LazyModule's internal structures
                    if hasattr(self, "_import_structure") and "cache_utils" in self._import_structure:
                        if "HybridCache" not in self._import_structure["cache_utils"]:
                            self._import_structure["cache_utils"].append("HybridCache")

                    if hasattr(self, "_class_to_module"):
                        self._class_to_module["HybridCache"] = "cache_utils"

                    if hasattr(self, "__all__") and "HybridCache" not in self.__all__:
                        self.__all__.append("HybridCache")

                    self.HybridCache = Cache

            _LazyModule.__init__ = _patched_lazy_module_init

        except Exception as e:
            warnings.warn(f"Failed to patch transformers HybridCache compatibility: {e}", stacklevel=2)


# Apply vLLM patches
_patch_vllm_logging()

# Apply transformers patches
_patch_transformers_hybrid_cache()
