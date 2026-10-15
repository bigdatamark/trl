# [MODULE DOC]
# HELPER: Defines _LazyModule, the module subclass used by trl package __init__ files to defer importing submodules
# until one of their exported names is accessed (adapted from transformers / optuna).
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
from itertools import chain
from types import ModuleType
from typing import Any


class _LazyModule(ModuleType):
    # [CLASS DOC]
    # HELPER: A ModuleType replacement installed into sys.modules that resolves exported names on demand.
    # Construction: name (package name), module_file (path of the package __init__.py), import_structure (dict mapping
    # submodule name -> list of exported names), module_spec (optional), extra_objects (optional dict of eagerly-available
    # attributes such as __version__).
    # State: _modules (set of submodule names), _class_to_module (exported name -> submodule), __all__ (submodules plus all
    # exported names), __file__, __spec__, __path__ (directory of module_file so subpackages still import), _objects,
    # _name, _import_structure.
    # Flow: __getattr__ is only hit for missing attributes; it imports the owning submodule via _get_module, caches the
    # value with setattr, and returns it. __dir__ advertises everything in __all__; __reduce__ makes it picklable.
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(self, name, module_file, import_structure, module_spec=None, extra_objects=None):
        # [FUNCTION DOC]
        # Builds the lazy module state described in the class block from the constructor arguments.
        # Inputs: name, module_file, import_structure, module_spec (default None), extra_objects (default None -> empty dict).
        # Outputs: none. Exceptions: none. Side effects: sets instance attributes.
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + list(chain(*import_structure.values()))
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

    # Needed for autocompletion in an IDE
    def __dir__(self):
        # [FUNCTION DOC]
        # Returns the default dir() listing extended with every entry of __all__ not already present (for IDE completion).
        # Inputs: none. Outputs: list of str. Exceptions: none. Side effects: none.
        result = super().__dir__()
        # The elements of self.__all__ that are submodules may or may not be in the dir already, depending on whether
        # they have been accessed or not. So we only add the elements of self.__all__ that are not already in the dir.
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str) -> Any:
        # [FUNCTION DOC]
        # Resolves a missing attribute lazily.
        # Behavior: return extra_objects entry if present; else if name is a submodule, import it; else if name is an exported
        # object, import its submodule and fetch the object from it; otherwise raise. The resolved value is cached on the
        # module with setattr so later lookups bypass this method.
        # Inputs: name (str). Outputs: the resolved object or submodule.
        # Exceptions: AttributeError for unknown names; RuntimeError (from _get_module) if the submodule import fails.
        # Side effects: imports submodules; sets an attribute on the module.
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        # [FUNCTION DOC]
        # HELPER: Imports the submodule relative to this package (".<module_name>").
        # Inputs: module_name (str). Outputs: the imported module.
        # Exceptions: RuntimeError wrapping (chained) any exception raised during import.
        # Side effects: triggers the submodule's import.
        try:
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {self.__name__}.{module_name} because of the following error (look up to see its"
                f" traceback):\n{e}"
            ) from e

    def __reduce__(self):
        # [FUNCTION DOC]
        # Pickle support: reconstructs the module by calling the class with (name, __file__, import_structure).
        # Inputs: none. Outputs: tuple (class, args). Exceptions: none. Side effects: none.
        # Note: module_spec and extra_objects are not preserved across pickling.
        return (self.__class__, (self._name, self.__file__, self._import_structure))
