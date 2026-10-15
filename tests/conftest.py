# [MODULE DOC]
# Pytest conftest shared by the whole test suite: process-wide hooks and autouse fixtures that keep tests
# isolated (memory cleanup, environment restoration, model-revision overrides, CPU fallback).
# Contents: _DropLoadReport log filter, pytest_runtest_makereport hook, MODEL_REVISIONS dict and autouse fixtures
# apply_model_revisions, force_use_cpu_without_accelerator, restore_mixed_precision_env, cleanup_gpu,
# cleanup_vllm_dist_env, undo_liger_kernel_patching.
# Inputs: none. Outputs: none.
# Exceptions: none at import.
# Side effects at import: attaches a _DropLoadReport filter to the transformers.modeling_utils,
# transformers.modeling_layers and transformers.integrations.peft loggers; imports torch and transformers.

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

import gc
import logging
import os
import sys
import traceback
from functools import wraps

import pytest
import torch
from transformers.utils import is_liger_kernel_available, is_peft_available, is_torch_xpu_available


# ============================================================================
# Silence transformers "LOAD REPORT" tables
# ============================================================================
# transformers >= 5 prints a "LOAD REPORT" table whenever a checkpoint has missing/mismatched/unexpected keys
# (transformers/utils/loading_report.py). TRL's tiny test models don't serialize the tied `lm_head.weight`, so
# this fires on nearly every model load and floods the test output. It is emitted through `logging` at WARNING
# level from three loggers; we drop only these records, keeping all other warnings visible.
class _DropLoadReport(logging.Filter):
    # [CLASS DOC]
    # HELPER: logging.Filter subclass that suppresses transformers "LOAD REPORT" warning records.
    # State: none. Construction: no arguments. Used by attaching an instance to specific transformers loggers.
    def filter(self, record):
        # [FUNCTION DOC]
        # HELPER: Decide whether a log record is emitted.
        # Inputs: record (logging.LogRecord). Outputs: False if the formatted message contains "LOAD REPORT", else True.
        # Exceptions: none. Side effects: none.
        return "LOAD REPORT" not in record.getMessage()


for _logger_name in ("transformers.modeling_utils", "transformers.modeling_layers", "transformers.integrations.peft"):
    logging.getLogger(_logger_name).addFilter(_DropLoadReport())


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # [FUNCTION DOC]
    # Hookwrapper around report creation that frees memory held by failed-test tracebacks.
    # Behavior: yields to let the report be built; afterwards, only for the "call" phase with an exception, clears
    # frame locals of the main traceback, then walks the exception graph (depth-first over __context__ and
    # __cause__, skipping None and already-seen ids) clearing each traceback's frames and setting __traceback__ to None.
    # Inputs: item (pytest Item), call (CallInfo).
    # Outputs: none. Exceptions: none raised.
    # Side effects: mutates exception objects (drops tracebacks and frame locals) so later gc can release CUDA memory.
    """Clear traceback frame locals after a failed test to release CUDA tensor references.

    When a test fails (especially with OOM), the exception traceback holds references to every local variable in every
    frame on the call stack at the time of failure — including the model, trainer, and all intermediate tensors.
    gc.collect() cannot free objects that are still reachable through a live traceback, so memory accumulates across
    reruns (~2 GiB per rerun for Gemma4, reaching 5 × 2.38 GiB = 11.89 GiB after 5 reruns). Clearing the frame locals
    breaks those reference chains so that the subsequent gc.collect() + empty_cache() in cleanup_gpu can actually
    reclaim the CUDA memory before the next attempt.
    """
    yield
    if call.when == "call" and call.excinfo is not None:
        traceback.clear_frames(call.excinfo.tb)
        # Also clear all reachable chained exception tracebacks (both __context__ and __cause__ at
        # every node): when OOM fires inside a try/except in the trainer, the OOM becomes __context__
        # of the outer exception and its traceback holds frame locals (model, tensors) that prevent gc
        # from releasing CUDA memory even after clear_frames above.
        stack, seen = [call.excinfo.value], set()
        while stack:
            exc = stack.pop()
            if exc is None or id(exc) in seen:
                continue
            seen.add(id(exc))
            if exc.__traceback__ is not None:
                traceback.clear_frames(exc.__traceback__)
                exc.__traceback__ = None
            stack.append(exc.__context__)
            stack.append(exc.__cause__)


# ============================================================================
# Model Revision Override
# ============================================================================
# To test a tiny model PR before merging to main:
# 1. Add the full model_id and PR revision to this dict
# 2. Commit and push to trigger CI
# 3. Once CI is green, merge the tiny model PR on HF Hub
# 4. Remove the entry from this dict and commit
#
# Example:
#   MODEL_REVISIONS = {
#       "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5": "refs/pr/3",
#       "trl-internal-testing/tiny-LlavaForConditionalGeneration": "refs/pr/5",
#   }
# ============================================================================

MODEL_REVISIONS = {
    # Add model_id: revision mappings here to test PRs
}


@pytest.fixture(autouse=True)
def apply_model_revisions(monkeypatch):
    # [FUNCTION DOC]
    # HELPER: Autouse fixture that injects a Hub revision for model ids listed in MODEL_REVISIONS.
    # Behavior: no-op when MODEL_REVISIONS is empty (the default). Otherwise monkeypatches from_pretrained on
    # AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoProcessor, AutoTokenizer and the
    # base config/model/tokenizer/processor classes with wrappers that add the revision kwarg; when PEFT is
    # installed also patches AutoPeftModelForCausalLM.from_pretrained, PeftModel.from_pretrained (adapter id is the
    # second argument) and PeftModel.load_adapter (instance method).
    # Inputs: monkeypatch fixture. Outputs: none (fixture value None).
    # Exceptions: none. Side effects: patches are undone by monkeypatch at test teardown.
    """Auto-inject revision parameter for models defined in MODEL_REVISIONS."""
    if not MODEL_REVISIONS:
        return

    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoProcessor,
        AutoTokenizer,
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizerBase,
        ProcessorMixin,
    )

    def create_classmethod_wrapper(original_classmethod):
        # [FUNCTION DOC]
        # HELPER: Build a classmethod replacement for a from_pretrained classmethod.
        # Inputs: original_classmethod (bound classmethod). Outputs: a new classmethod whose wrapper, when the model id
        # is in MODEL_REVISIONS and no revision kwarg is given, sets revision and removes any _commit_hash kwarg before
        # calling the original function. Exceptions: none. Side effects: none.
        # Extract the underlying function from the classmethod
        original_func = original_classmethod.__func__

        @wraps(original_func)
        def wrapper(cls, pretrained_model_name_or_path, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Wrapped from_pretrained; injects the overridden revision (and drops _commit_hash) for listed ids, then
            # delegates. Inputs: cls, pretrained_model_name_or_path, extra args/kwargs. Outputs: the original's return
            # value. Exceptions: whatever the original raises. Side effects: mutates its kwargs dict only.
            # Direct lookup: only inject if model_id is in the override dict
            if pretrained_model_name_or_path in MODEL_REVISIONS:
                if "revision" not in kwargs:
                    kwargs["revision"] = MODEL_REVISIONS[pretrained_model_name_or_path]
                    # Clear _commit_hash: Auto classes resolve it from the default branch before calling
                    # sub-loaders, so the cached hash points to main. If we don't clear it, it silently
                    # overrides the injected revision for the config load while the weight loader uses the
                    # revision, producing a config/weights shape mismatch.
                    kwargs.pop("_commit_hash", None)

            return original_func(cls, pretrained_model_name_or_path, *args, **kwargs)

        # Re-wrap as classmethod
        return classmethod(wrapper)

    def create_method_wrapper(original_method):
        # [FUNCTION DOC]
        # HELPER: Build a wrapper for an instance method taking a model id (used for PeftModel.load_adapter).
        # Inputs: original_method. Outputs: wrapper function that injects revision for listed ids when absent.
        # Exceptions: none. Side effects: none.
        @wraps(original_method)
        def wrapper(self, model_id, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Injects revision from MODEL_REVISIONS for listed model_id when not given, then calls the original.
            # Inputs: self, model_id, args/kwargs. Outputs: original return value. Exceptions: propagated. Side effects: none.
            if model_id in MODEL_REVISIONS and "revision" not in kwargs:
                kwargs["revision"] = MODEL_REVISIONS[model_id]
            return original_method(self, model_id, *args, **kwargs)

        return wrapper

    # Patch the transformers Auto* classes and the base classes they dispatch to
    for cls in [
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        # `AutoTokenizer`/`AutoProcessor` resolve the chat template before dispatching, so patching only the base
        # classes loads it from the default branch.
        AutoProcessor,
        AutoTokenizer,
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizerBase,
        ProcessorMixin,
    ]:
        monkeypatch.setattr(cls, "from_pretrained", create_classmethod_wrapper(cls.from_pretrained))

    def create_peft_classmethod_wrapper(original_classmethod):
        # [FUNCTION DOC]
        # HELPER: Build a classmethod wrapper for PeftModel.from_pretrained, whose adapter id is the second positional
        # argument after the base model. Inputs: original_classmethod. Outputs: new classmethod.
        # Exceptions: none. Side effects: none.
        original_func = original_classmethod.__func__

        @wraps(original_func)
        def wrapper(cls, model, model_id, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Injects revision for listed model_id when absent and delegates to the original function.
            # Inputs: cls, model (base model), model_id, args/kwargs. Outputs: original return value.
            # Exceptions: propagated. Side effects: none.
            if model_id in MODEL_REVISIONS and "revision" not in kwargs:
                kwargs["revision"] = MODEL_REVISIONS[model_id]
            return original_func(cls, model, model_id, *args, **kwargs)

        return classmethod(wrapper)

    # PEFT adapters never reach the loaders above.
    if is_peft_available():
        from peft import AutoPeftModelForCausalLM, PeftModel

        monkeypatch.setattr(
            AutoPeftModelForCausalLM,
            "from_pretrained",
            create_classmethod_wrapper(AutoPeftModelForCausalLM.from_pretrained),
        )
        # `PeftModel.from_pretrained` takes the base model first and the adapter id second.
        monkeypatch.setattr(PeftModel, "from_pretrained", create_peft_classmethod_wrapper(PeftModel.from_pretrained))
        # `load_adapter` is an instance method, not a classmethod.
        monkeypatch.setattr(PeftModel, "load_adapter", create_method_wrapper(PeftModel.load_adapter))


@pytest.fixture(autouse=True)
def force_use_cpu_without_accelerator(monkeypatch):
    # [FUNCTION DOC]
    # HELPER: Autouse fixture forcing use_cpu on every TRL config when no accelerator exists.
    # Behavior: returns immediately if transformers' torch_device is set and not "cpu". Otherwise monkeypatches
    # _BaseConfig.__post_init__ with a version that sets use_cpu to True before calling the original, avoiding the
    # bf16-unsupported error raised by TrainingArguments validation.
    # Inputs: monkeypatch. Outputs: none. Exceptions: none.
    # Side effects: class-level patch, reverted at teardown.
    """Force `use_cpu=True` on all TRL configs when no accelerator is available.

    TRL configs default `bf16` to `True` (see `_BaseConfig.__post_init__`), which makes
    `transformers.TrainingArguments.__post_init__` raise `ValueError: Your setup doesn't support bf16/gpu. You need to
    assign use_cpu ...` on machines without a GPU. Setting `use_cpu=True` before that validation runs lets the whole
    suite run on CPU without editing every individual config in the tests. On a machine with an accelerator (CUDA, XPU,
    NPU, ...) this fixture is a no-op, so on-device tests and explicit `use_cpu=False` are left untouched.
    """
    from transformers.testing_utils import torch_device

    if torch_device is not None and torch_device != "cpu":
        return

    from trl.trainer.base_config import _BaseConfig

    original_post_init = _BaseConfig.__post_init__

    def patched_post_init(self):
        # [FUNCTION DOC]
        # HELPER: Sets use_cpu to True on the config instance and then runs the original _BaseConfig.__post_init__.
        # Inputs: self (config). Outputs: None. Exceptions: those of the original. Side effects: mutates self.
        self.use_cpu = True
        original_post_init(self)

    monkeypatch.setattr(_BaseConfig, "__post_init__", patched_post_init)


@pytest.fixture(autouse=True)
def restore_mixed_precision_env(monkeypatch):
    # [FUNCTION DOC]
    # HELPER: Autouse fixture that sets ACCELERATE_MIXED_PRECISION to its current value (or "no" when unset) via
    # monkeypatch so that any value leaked by TrainingArguments during the test is restored at teardown.
    # Inputs: monkeypatch. Outputs: none. Exceptions: none. Side effects: env var set for the test, restored after.
    """
    Restore the mixed precision that `transformers.TrainingArguments` publishes process-wide.

    On transformers < 5, `TrainingArguments.__post_init__` writes the mixed precision to `ACCELERATE_MIXED_PRECISION`
    and reads that same variable back as the default for the next instantiation, where a `bf16=False` flag is
    indistinguishable from an unset one. TRL configs default `bf16` to `True`, so the first trainer built in a worker
    sets the variable to `bf16` for the whole process, and no config built later can clear it. transformers < 5 also
    builds the `Accelerator` without passing the mixed precision, so accelerate falls back to the leaked variable and a
    test that asks for `bf16=False` silently runs in bf16: with `test_chunked_logps_match_full_logits`, the streamed
    projection enters autocast and the full-logits path does not, and the two no longer match. Setting the variable to
    its current value arms monkeypatch's restore, so each test sees the value the session started with.

    Only the minimum-versions CI job is affected: transformers >= 5 passes the mixed precision to the `Accelerator`
    explicitly and never writes the variable.
    """
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", os.environ.get("ACCELERATE_MIXED_PRECISION", "no"))


@pytest.fixture(autouse=True)
def cleanup_gpu():
    # [FUNCTION DOC]
    # HELPER: Autouse generator fixture that, after each test, runs gc.collect and, when CUDA or XPU is available,
    # synchronizes the device and empties its memory cache.
    # Inputs: none. Outputs: yields None. Exceptions: none expected. Side effects: frees cached accelerator memory.
    """
    Automatically cleanup accelerator memory after each test.

    This fixture helps prevent accelerator out of memory errors when running tests in parallel with pytest-xdist by
    ensuring models and tensors are properly garbage collected and accelerator memory caches are cleared between tests.
    """
    yield
    # Cleanup after test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    if is_torch_xpu_available():
        torch.xpu.synchronize()
        torch.xpu.empty_cache()


@pytest.fixture(autouse=True)
def cleanup_vllm_dist_env():
    # [FUNCTION DOC]
    # HELPER: Autouse generator fixture that tears down distributed state left by vLLM colocate mode.
    # Behavior: before the test snapshots RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT (None if unset);
    # after the test, if torch.distributed is initialized, calls vLLM's cleanup_dist_env_and_memory (imported
    # lazily, so vLLM must be installed in that case); then restores each env var, removing it if it was unset.
    # Inputs: none. Outputs: yields None.
    # Exceptions: ImportError if a process group exists but vLLM is not installed.
    # Side effects: destroys process groups and mutates os.environ.
    """
    Destroy the torch.distributed process group left behind by vLLM colocate mode after each test.

    vLLM colocate mode initializes `torch.distributed` internally (`distributed_executor_backend="external_launcher"`)
    but the offline `LLM` class never tears it down: neither `LLM` nor `LLMEngine` calls vLLM's own
    `cleanup_dist_env_and_memory()` on exit. Left undestroyed, PyTorch warns at interpreter shutdown (e.g.
    `TestGRPOTrainerSlow::test_vlm_processor_vllm_colocate_mode`). We call vLLM's own cleanup helper rather than a bare
    `torch.distributed.destroy_process_group()` so that vLLM's internal parallel-state globals (`_TP`, `_WORLD`, ...)
    are reset too, letting the next colocate test reinitialize cleanly.

    It also restores `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR` and `MASTER_PORT`, which TRL sets before
    building the vLLM `LLM` in colocate mode. Left set, they make the next test's `Accelerator` initialize
    `torch.distributed` on the leaked port.
    """
    dist_env = {
        name: os.environ.get(name) for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    }
    yield
    if torch.distributed.is_initialized():
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory()
    for name, value in dist_env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@pytest.fixture(autouse=True)
def undo_liger_kernel_patching(monkeypatch):
    # [FUNCTION DOC]
    # HELPER: Autouse generator fixture that reverts module-level rebinding performed by Liger Kernel.
    # Behavior: if liger_kernel is not available just yields. Otherwise wraps
    # liger_kernel.transformers._apply_liger_kernel_to_instance (via monkeypatch) so that, on each call, every
    # already-imported transformers.models.*.modeling_* module has its namespace dict snapshotted once (first
    # snapshot wins) before delegating. After the test, updates each snapshotted module's namespace with the saved
    # values, restoring the original names.
    # Inputs: monkeypatch. Outputs: yields None. Exceptions: none.
    # Side effects: mutates transformers modeling module globals at teardown. Names newly added by Liger are not
    # removed, only overwritten ones are restored.
    """
    Restore the transformers modeling modules that Liger Kernel patches process-wide.

    `use_liger_kernel=True` makes transformers call `liger_kernel.transformers._apply_liger_kernel_to_instance`, which
    patches the model instance but also rebinds module-level names in `transformers.models.<arch>.modeling_<arch>`
    (`Qwen3RMSNorm` -> `LigerRMSNorm`, `Qwen3MLP` -> `LigerSwiGLUMLP`, `apply_rotary_pos_emb` ->
    `liger_rotary_pos_emb`, ...). Liger never restores them, so every model of that architecture instantiated later in
    the same process silently gets Triton kernels and fails on CPU tensors with `ValueError: Pointer argument cannot be
    accessed from Triton (cpu tensor?)`. Which test trips over it depends on how pytest-xdist spreads tests across
    workers, so the failure surfaces in an unrelated test (e.g. `TestUseAdapter`, whose PEFT model has a Qwen3 base)
    and only in some runs. Snapshot the modeling modules when Liger patches, and restore them once the test is over.

    Only the trigger is transformers-version dependent, not the leak itself: transformers < 5.3 applies the kernels in
    `Trainer.__init__`, so a test that merely builds a trainer leaks, while transformers >= 5.3 applies them in
    `Trainer.train()`. Either way they are never reverted, which is why only the minimum-versions CI job fails: the
    Liger tests that use a Qwen3 model build a trainer but never train.
    """
    if not is_liger_kernel_available():
        yield
        return

    import liger_kernel.transformers as liger_transformers

    apply_to_instance = liger_transformers._apply_liger_kernel_to_instance
    snapshots = {}

    def apply_to_instance_and_snapshot(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Snapshot transformers modeling modules then call the original Liger apply function.
        # Inputs: args/kwargs forwarded unchanged. Outputs: original return value.
        # Exceptions: propagated. Side effects: adds entries to the enclosing snapshots dict.
        # Liger imports the modeling modules it patches inside the patch function, but a module can only be patched
        # if the model being patched already uses it, so it is imported by now.
        for name, module in list(sys.modules.items()):
            if name.startswith("transformers.models.") and ".modeling_" in name:
                snapshots.setdefault(name, (module, vars(module).copy()))
        return apply_to_instance(*args, **kwargs)

    monkeypatch.setattr(liger_transformers, "_apply_liger_kernel_to_instance", apply_to_instance_and_snapshot)
    yield
    for module, snapshot in snapshots.values():
        vars(module).update(snapshot)
