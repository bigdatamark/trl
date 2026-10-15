# [MODULE DOC]
# Maintenance script that verifies each reference Hub repo in REPOS still ships a chat template identical to one of the
# stored copies in trl/chat_templates (TRL recognizes models by exact template match). Run from the repo root; gated
# repos require an authorized HF token. Contents: REPOS list, get_chat_template, main. Import-time side effects: none.

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
Check that the chat template each reference Hub repo currently ships is one of the templates stored in
`trl/chat_templates/`. TRL recognizes a model by exact match against these copies, so a repo that changes its chat
template stops being recognized until the new revision is stored.

Some reference repos are gated, so the Hugging Face token in use needs access to them.

Usage, from the repository root:

```sh
python scripts/check_chat_templates.py
```
"""

import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPOS = [
    "CohereLabs/aya-expanse-8b",
    "CohereLabs/tiny-aya-earth",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "google/diffusiongemma-26B-A4B-it",
    "google/gemma-2-2b-it",
    "google/gemma-3-4b-it",
    "google/gemma-4-E2B-it",
    "google/gemma-7b-it",
    "HuggingFaceM4/Idefics3-8B-Llama3",
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "LiquidAI/LFM2-1.2B",
    "LiquidAI/LFM2.5-230M",
    "LiquidAI/LFM2.5-VL-3B",
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-models/Muse-Glimmer-30B",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    "openai/gpt-oss-20b",
    "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.6-35B-A3B",
    "Qwen/Qwen3.8-27B",
    "zai-org/GLM-4.5",
]


def get_chat_template(api, repo):
    # [FUNCTION DOC]
    # HELPER: Fetches the chat template a Hub repo ships, using transformers' precedence.
    # Behavior: lists repo files; if chat_template.jinja exists, downloads and returns its text. Otherwise checks
    # chat_template.json then tokenizer_config.json, reading the chat_template field; if that field is a list of named
    # variants, picks the one named default; returns the first non-None value.
    # Inputs: api (HfApi), repo (str). Outputs: template string, or None if no source has one.
    # Exceptions: Hub errors propagate; KeyError if a variant list has no default entry.
    # Side effects: network calls and Hub cache downloads.
    # Same precedence as transformers: `chat_template.jinja`, then `chat_template.json` (processors), then the
    # `chat_template` field of `tokenizer_config.json`
    files = api.list_repo_files(repo)
    if "chat_template.jinja" in files:
        return Path(hf_hub_download(repo, "chat_template.jinja")).read_text(encoding="utf-8")
    for filename in ("chat_template.json", "tokenizer_config.json"):
        if filename in files:
            chat_template = json.loads(Path(hf_hub_download(repo, filename)).read_text(encoding="utf-8")).get(
                "chat_template"
            )
            if isinstance(chat_template, list):  # named variants: the stored copy is the `default` one
                chat_template = {variant["name"]: variant["template"] for variant in chat_template}["default"]
            if chat_template is not None:
                return chat_template


def main():
    # [FUNCTION DOC]
    # Compares every reference repo's template against the stored set and sets the exit status.
    # Behavior: reads all .jinja files under trl/chat_templates (relative to cwd) into a set, fetches each repo's template,
    # collects repos whose template is not in the set, prints a line per drifted repo and a summary count, exits 1 if any
    # drifted else 0.
    # Inputs: none. Outputs: none. Exceptions: SystemExit; Hub errors propagate.
    # Side effects: network calls, printing, process exit.
    stored = {path.read_text(encoding="utf-8") for path in Path("trl/chat_templates").glob("*.jinja")}
    api = HfApi()
    drifted = [repo for repo in REPOS if get_chat_template(api, repo) not in stored]
    for repo in drifted:
        print(f"{repo}: the chat template it ships is not stored in trl/chat_templates/")
    print(f"{len(REPOS) - len(drifted)}/{len(REPOS)} reference repos ship a stored chat template")
    sys.exit(1 if drifted else 0)


if __name__ == "__main__":
    main()
