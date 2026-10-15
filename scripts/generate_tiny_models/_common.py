# [MODULE DOC]
# HELPER: Shared utilities for the tiny-model generation scripts: version pinning, seeding, a forward-pass smoke test,
# dtype and config comparisons against the reference Hub checkpoint, CLI flag parsing, single-commit Hub upload and
# safe weight initialization. Constants: ORGANIZATION ("trl-internal-testing"), SEED (42), MODEL_CARD template,
# _DIFF_IGNORE keys and a torch-dtype to safetensors-dtype-string table.
# Import-time side effects: creates a module-level HfApi client (no network call).

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

# Shared utilities for the tiny-model generation scripts in this directory.
# Each sibling script builds a single tiny model and pushes it to the Hub under
# the `trl-internal-testing` organization.

import argparse
import os
import re
import tempfile
from pathlib import Path

import torch
from huggingface_hub import CommitOperationAdd, HfApi, ModelCard
from huggingface_hub.errors import NotASafetensorsRepoError
from packaging.version import Version
from torch import nn
from transformers import AutoConfig, ProcessorMixin
from transformers import set_seed as _set_seed


ORGANIZATION = "trl-internal-testing"

SEED = 42

MODEL_CARD = """
---
library_name: transformers
tags: [trl]
---

# Tiny {model_class_name}

This is a minimal model built for unit tests in the [TRL](https://github.com/huggingface/trl) library.
"""


api = HfApi()


def _transformers_floor_from_pyproject():
    # [FUNCTION DOC]
    # HELPER: Returns the minimum transformers version declared in the repository's pyproject.toml.
    # Behavior: reads pyproject.toml two directories above this file and regex-searches for a quoted transformers>= bound.
    # Inputs: none. Outputs: version string after the >= sign.
    # Exceptions: RuntimeError if no bound is found; OSError if the file is missing.
    # Side effects: reads a file.
    """Read the `transformers>=X.Y.Z` floor from the repo's pyproject.toml."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    match = re.search(r'"transformers>=([^"]+)"', pyproject.read_text())
    if match is None:
        raise RuntimeError(f"Could not find a `transformers>=` bound in {pyproject}")
    return match.group(1)


def check_transformers_version(expected_version=None):
    # [FUNCTION DOC]
    # HELPER: Ensures the installed transformers version equals an expected version exactly (parsed with packaging Version).
    # Inputs: expected_version (str or None; None means the pyproject floor). Outputs: none.
    # Exceptions: RuntimeError naming both versions on mismatch. Side effects: imports transformers.
    """Raise unless the installed transformers matches `expected_version` exactly.

    If `expected_version` is None, defaults to the `transformers>=` floor from pyproject.toml.
    """
    import transformers

    if expected_version is None:
        expected_version = _transformers_floor_from_pyproject()

    if Version(transformers.__version__) != Version(expected_version):
        raise RuntimeError(
            f"This script requires transformers=={expected_version}, but {transformers.__version__} is installed."
        )


def set_seed(seed=SEED):
    # [FUNCTION DOC]
    # HELPER: Seeds all RNGs through transformers.set_seed so regenerated weights are reproducible.
    # Inputs: seed (int, default SEED = 42). Outputs: none. Exceptions: none. Side effects: global RNG state.
    """Seed the RNGs that model construction draws from.

    Without this every run produces different weights, so re-pushing a tiny model always creates a new
    `model.safetensors` even when nothing else changed. Call it before building the model: most scripts never call
    `init_weights_tiny_model`, and the randomness has already happened by then.

    Pass a different `seed` when a script must produce weights that differ from another script's.
    """
    _set_seed(seed)


def smoke_test(model, tokenizer_or_processor=None):
    # [FUNCTION DOC]
    # HELPER: Runs one forward pass and fails on missing outputs or NaNs.
    # Behavior: puts the model in eval mode and uses the device of its first parameter. If tokenizer_or_processor is a
    # ProcessorMixin, builds a batch of two chat conversations each with one 24x24 solid-color image (red / blue) and a
    # text question, applies the chat template with generation prompt, tokenization and padding to tensors. Otherwise
    # uses a single input_ids sequence 1,2,3,4. Runs the model under no_grad, takes logits or else last_hidden_state.
    # Inputs: model, tokenizer_or_processor (optional; tokenizers are ignored). Outputs: none.
    # Exceptions: RuntimeError if neither output exists or the tensor contains NaN.
    # Side effects: switches model to eval mode; prints an OK line with the output shape.
    """Run a minimal forward pass to sanity-check the tiny model doesn't crash or produce NaNs."""
    model.eval()
    device = next(model.parameters()).device

    if isinstance(tokenizer_or_processor, ProcessorMixin):
        # VLM path: build a dummy (image, text) input via the processor.
        from PIL import Image

        processor = tokenizer_or_processor
        red = Image.new("RGB", (24, 24), color="red")
        blue = Image.new("RGB", (24, 24), color="blue")
        messages = [
            [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": red}, {"type": "text", "text": "What is this?"}],
                }
            ],
            [{"role": "user", "content": [{"type": "text", "text": "Is it blue?"}, {"type": "image", "image": blue}]}],
        ]
        inputs = processor.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(device)
    else:
        inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]], device=device)}

    with torch.no_grad():
        out = model(**inputs)

    if "logits" in out:
        output_tensor = out["logits"]
    elif "last_hidden_state" in out:
        output_tensor = out["last_hidden_state"]
    else:
        raise RuntimeError(f"[smoke_test] {model.__class__.__name__}: no logits or last_hidden_state on output")
    if torch.isnan(output_tensor).any():
        raise RuntimeError(f"[smoke_test] {model.__class__.__name__}: NaN in forward output")
    print(f"[smoke_test] {model.__class__.__name__}: OK (output shape {tuple(output_tensor.shape)})")


def _flatten(d, prefix=""):
    # [FUNCTION DOC]
    # HELPER: Flattens a nested dict into one level with dot-joined keys (recursing into dict values only).
    # Inputs: d (dict), prefix (str, default empty). Outputs: new flat dict. Exceptions: none. Side effects: none.
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


_DIFF_IGNORE = {"_name_or_path", "transformers_version", "architectures", "model_type", "torch_dtype", "dtype"}


_TORCH_TO_SAFETENSORS_DTYPE = {
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float64: "F64",
    torch.float8_e4m3fn: "F8_E4M3",
    torch.float8_e5m2: "F8_E5M2",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}


def check_dtype_pattern(reference_id, model):
    # [FUNCTION DOC]
    # HELPER: Reports tensors whose dtype differs from the reference checkpoint, without downloading weights.
    # Behavior: fetches safetensors metadata for reference_id via the Hub API (prints and returns if the repo is not a
    # safetensors repo), builds a name to dtype map over all shards, then for every tensor in model.state_dict that has a
    # same-named reference tensor, compares the safetensors dtype string; prints either an all-matched line or each
    # mismatch with reference and tiny dtypes.
    # Inputs: reference_id (str), model. Outputs: none (report only, never raises on mismatch).
    # Exceptions: Hub errors other than NotASafetensorsRepoError propagate. Side effects: network call, printing.
    """Flag tensors whose dtype diverges from the reference checkpoint.

    Reads the reference safetensors header via the Hub API (no weight download). Useful to catch cases
    like Qwen3.5 where specific params (e.g. linear_attn.A_log) are kept in fp32 while the rest is bf16.
    """
    try:
        metadata = api.get_safetensors_metadata(reference_id)
    except NotASafetensorsRepoError:
        print(f"[dtype_check] {reference_id}: not a safetensors repo, skipping")
        return
    ref_dtypes = {name: info.dtype for fm in metadata.files_metadata.values() for name, info in fm.tensors.items()}

    mismatches = []
    for name, tensor in model.state_dict().items():
        ref_dtype = ref_dtypes.get(name)
        if ref_dtype is None:
            continue  # tensor has no counterpart in the reference (e.g. scale-down, PEFT wrapper, tying)
        tiny_dtype = _TORCH_TO_SAFETENSORS_DTYPE.get(tensor.dtype)
        if tiny_dtype != ref_dtype:
            mismatches.append((name, ref_dtype, tiny_dtype))

    if not mismatches:
        print(f"[dtype_check] {reference_id}: all matched tensors have the reference dtype")
        return

    print(f"[dtype_check] {reference_id}: {len(mismatches)} tensors differ from reference:")
    for name, ref, tiny in mismatches:
        print(f"  {name}: reference={ref}, tiny={tiny}")


def _format_diff_value(value, width):
    # [FUNCTION DOC]
    # HELPER: Renders a config value for the diff table.
    # Behavior: lists are prefixed with their item count; the string is cut to width minus one characters plus an
    # ellipsis when longer than width; width None disables truncation.
    # Inputs: value, width (int or None). Outputs: str. Exceptions: none. Side effects: none.
    """Render a config value for the diff table, tagging lists with their length and marking truncation."""
    text = f"[{len(value)} items] {value}" if isinstance(value, list) else str(value)
    if width is None or len(text) <= width:
        return text
    return text[: width - 1] + "\u2026"


def print_config_diff(reference_id, model, full=None):
    # [FUNCTION DOC]
    # HELPER: Prints every differing flattened key between the reference Hub config and model.config.
    # Behavior: full None reads the --full-diff CLI flag; column width 34 unless full. Loads AutoConfig for reference_id,
    # flattens both config dicts, iterates sorted union of keys, skips keys equal to or ending with an ignored name
    # (name/path, versions, architectures, model_type, dtype), uses the placeholder missing marker for absent keys, and
    # prints a header with the difference count and one aligned row per difference.
    # Inputs: reference_id (str), model, full (bool or None). Outputs: none.
    # Exceptions: Hub/config loading errors propagate. Side effects: network call, printing.
    """Print the flat, recursive diff between the reference Hub config and the tiny-model config.

    Values are truncated to keep the table aligned; pass `--full-diff` (or `full=True`) to print them in full.
    """
    if full is None:
        full = _parse_args().full_diff
    width = None if full else 34
    reference_config = AutoConfig.from_pretrained(reference_id)
    ref_flat = _flatten(reference_config.to_dict())
    tiny_flat = _flatten(model.config.to_dict())

    keys = sorted(set(ref_flat) | set(tiny_flat))
    rows = []
    for k in keys:
        if any(k == ig or k.endswith(f".{ig}") for ig in _DIFF_IGNORE):
            continue
        rv, tv = ref_flat.get(k, "<missing>"), tiny_flat.get(k, "<missing>")
        if rv != tv:
            rows.append((k, rv, tv))

    print(f"[config_diff] {reference_id} vs tiny ({len(rows)} differences)")
    for k, r, t in rows:
        print(f"  {k:48s} {_format_diff_value(r, width):34s} → {_format_diff_value(t, width)}")


def _parse_args():
    # [FUNCTION DOC]
    # HELPER: Parses the shared optional CLI flags --create-pr and --full-diff (both store_true) with parse_known_args,
    # ignoring unknown args and without adding a help flag. Inputs: sys.argv. Outputs: argparse Namespace.
    # Exceptions: none in practice. Side effects: none.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="If the repo already exists, open a PR instead of skipping.",
    )
    parser.add_argument(
        "--full-diff",
        action="store_true",
        help="Print untruncated values in the config diff.",
    )
    args, _ = parser.parse_known_args()
    return args


def push_to_hub(model, tokenizer, generation_config, prefix=None, suffix=None, create_pr=None):
    # [FUNCTION DOC]
    # HELPER: Uploads the model, tokenizer/processor, generation config and a model card to the Hub in one commit.
    # Behavior: create_pr None reads --create-pr. Model card is MODEL_CARD filled with the model class name. Repo id is
    # ORGANIZATION / [prefix-]ClassName[-suffix]. If the repo exists and create_pr is false, prints a skip message and
    # returns. Creates the repo when missing. Saves everything (README.md included) into a temp directory, walks it to
    # build one add-operation per file, and creates a single commit (as a PR only when the repo already existed and
    # create_pr is set); prints the PR URL when one is opened.
    # Inputs: model, tokenizer (or None), generation_config (or None), prefix (str, optional), suffix (str, optional),
    # create_pr (bool or None). Outputs: none.
    # Exceptions: Hub API errors propagate. Side effects: network calls, temp files, repo creation/commit, printing.
    if create_pr is None:
        create_pr = _parse_args().create_pr

    model_class_name = model.__class__.__name__
    content = MODEL_CARD.format(model_class_name=model_class_name)
    model_card = ModelCard(content)
    if prefix is not None:
        model_class_name = f"{prefix}-{model_class_name}"
    repo_id = f"{ORGANIZATION}/{model_class_name}"
    if suffix is not None:
        repo_id += f"-{suffix}"

    exists = api.repo_exists(repo_id)
    if exists and not create_pr:
        print(f"Model {repo_id} already exists, skipping (pass --create-pr to open a PR)")
        return

    if not exists:
        api.create_repo(repo_id, exist_ok=True)

    # Save all artifacts to a temp dir and upload them in a single commit, so --create-pr opens one PR.
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        if tokenizer is not None:
            tokenizer.save_pretrained(tmpdir)
        if generation_config is not None:
            generation_config.save_pretrained(tmpdir)
        model_card.save(os.path.join(tmpdir, "README.md"))

        operations = [
            CommitOperationAdd(
                path_in_repo=os.path.relpath(os.path.join(root, name), tmpdir),
                path_or_fileobj=os.path.join(root, name),
            )
            for root, _, files in os.walk(tmpdir)
            for name in files
        ]
        commit_info = api.create_commit(
            repo_id=repo_id,
            operations=operations,
            commit_message=f"Upload {model.__class__.__name__}",
            create_pr=exists and create_pr,
        )
        if commit_info.pr_url:
            print(f"[push_to_hub] PR opened: {commit_info.pr_url}")


def init_weights_tiny_model(model):
    # [FUNCTION DOC]
    # HELPER: Re-initializes common module types in place to avoid NaNs from uninitialized memory.
    # Behavior: for every submodule: Linear and Conv1d get Xavier-uniform weights and zero biases; Embedding gets
    # normal(0, 0.02); LayerNorm gets weight ones and zero bias. Other module types (e.g. RMSNorm) are untouched.
    # Inputs: model (nn.Module). Outputs: none. Exceptions: none. Side effects: mutates model parameters.
    """
    Initialize tiny test models to avoid NaNs from uninitialized weights.

    Uses safe defaults:
      - Linear/Conv1d: Xavier uniform (weights), zero (biases)
      - Embedding: Normal(0, 0.02)
      - LayerNorm: Ones (weights), zero (biases)
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            nn.init.xavier_uniform_(module.weight)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Conv1d):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            nn.init.xavier_uniform_(module.weight)
