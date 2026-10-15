# [MODULE DOC]
# Example script: GRPO against a Harbor task suite using trl.experimental.harbor.HarborSpec, which supplies the
# train dataset, environment factory and reward functions from one object. The agent harness is chosen with --agent
# (built-in "bash" or an import/file path to a HarborEnv subclass) and the sandbox backend with --env.
# Main contents: parse_args, main. Import-time side effects: none beyond imports.

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

# /// script
# dependencies = ["trl[vllm,harbor]", "harbor[e2b]"]  # harbor[e2b]: the --env e2b backend (swap per --env)
# ///

"""GRPO training against a Harbor task suite.

One ``HarborSpec`` fans out into TRL's three slots — ``.train_dataset`` / ``.environment_factory`` /
``.reward_funcs``. The base agent (harness) is selected with ``--agent``: the built-in ``"bash"``, or
an import/file path to a custom ``HarborEnv`` subclass. Harbor runs in-process, so install with
``trl[harbor]`` (Python >= 3.12) plus the chosen sandbox backend's extra (``pip install "harbor[e2b]"``
for the recommended ``--env e2b``; the ``docker`` backend just needs a reachable Docker daemon).

Usage (server vLLM, single-node 2+2 GPU split):

```sh
# Terminal 1 — vLLM
CUDA_VISIBLE_DEVICES=2,3 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-4B --tensor-parallel-size 2 --port 8000 \
    --weight-transfer-config '{"backend": "nccl"}' \
    --logprobs-mode processed_logprobs \
    --max-logprobs -1

# Terminal 2 — training
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero2.yaml --num_processes 2 \
    examples/grpo_harbor/grpo_harbor.py \
    --vllm-mode server --vllm-server-base-url http://localhost:8000 --env e2b
```
"""

import argparse

from trl import GRPOConfig, GRPOTrainer
from trl.experimental.harbor import HarborSpec


def parse_args() -> argparse.Namespace:
    # [FUNCTION DOC]
    # Behavior: argparse for model (Qwen/Qwen3-4B), dataset (HF repo or local path with tasks/), agent, sandbox backend
    # (stored as environment_type, default docker), number of tasks (64), learning rate, batch size, generations,
    # completion length, max steps, max tool-calling iterations (25), vLLM mode/server URL/GPU memory fraction (0.3),
    # output dir and comma-separated report_to (default "none").
    # Inputs: command line. Outputs: Namespace. Exceptions: SystemExit on bad args. Side effects: may print usage.
    parser = argparse.ArgumentParser(description="GRPO training against a Harbor task suite.")

    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--dataset",
        type=str,
        default="AdithyaSK/data_agent_rl_environment_train",
        help="HF dataset repo id with a Harbor task tree, or a local path containing a tasks/ subtree.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="bash",
        help="Base agent/harness: built-in name ('bash'), import path 'pkg.mod:Class', or file 'path.py:Class'.",
    )
    parser.add_argument(
        "--env",
        dest="environment_type",
        type=str,
        default="docker",
        help="Harbor sandbox backend (docker, e2b, daytona, gke, modal, runloop, ...).",
    )
    parser.add_argument("--num-tasks", type=int, default=64)

    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-tool-calling-iterations", type=int, default=25)

    parser.add_argument("--vllm-mode", choices=("colocate", "server"), default="colocate")
    parser.add_argument("--vllm-server-base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3)

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--report-to", type=str, default="none")

    return parser.parse_args()


def main() -> None:
    # [FUNCTION DOC]
    # Behavior: builds a HarborSpec from dataset, agent, backend and task count; assembles GRPOConfig kwargs (thinking
    # disabled, completion logging, vLLM always on, report_to parsed into a list excluding blanks and "none", falling
    # back to "none"); adds output_dir if given, GPU memory fraction for colocate mode or server base URL for server mode;
    # trains GRPOTrainer with the spec's dataset, environment factory and reward functions; calls trainer.end().
    # Inputs: command line. Outputs: none. Exceptions: propagates Harbor/sandbox/training errors.
    # Side effects: dataset download, sandbox creation, GPU training, logging, output files.
    args = parse_args()

    # One spec object — fans out into TRL's three slots.
    spec = HarborSpec(
        args.dataset,
        agent=args.agent,
        environment_type=args.environment_type,
        num_tasks=args.num_tasks,
    )

    config_kwargs: dict = dict(
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        max_tool_calling_iterations=args.max_tool_calling_iterations,
        chat_template_kwargs={"enable_thinking": False},
        log_completions=True,
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        report_to=[s.strip() for s in args.report_to.split(",") if s.strip() and s.strip() != "none"] or "none",
    )
    if args.output_dir:
        config_kwargs["output_dir"] = args.output_dir
    if args.vllm_mode == "colocate":
        config_kwargs["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
    else:
        config_kwargs["vllm_server_base_url"] = args.vllm_server_base_url

    trainer = GRPOTrainer(
        model=args.model,
        args=GRPOConfig(**config_kwargs),
        train_dataset=spec.train_dataset,
        environment_factory=spec.environment_factory,
        reward_funcs=spec.reward_funcs,
    )
    trainer.train()

    trainer.end()


if __name__ == "__main__":
    main()
