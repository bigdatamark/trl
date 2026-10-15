# [MODULE DOC]
# Example script: AsyncGRPO training on Harbor coding tasks where an off-the-shelf agent (default mini-swe-agent)
# runs inside sandboxes managed by an OpenEnv Harbor server; the server's capture proxy records token ids/logprobs of
# every model call to vLLM, and AsyncGRPOTrainer trains on them via a HarnessRolloutWorker.
# Main contents: reward shaping constants (W_TOOL_EFFICIENCY 0.3, TOOL_BUDGET 15.0), tool_efficiency, harbor_reward,
# task_indices, parse_args, main. Not part of the trl package.
# Import-time side effects: configures root logging at WARNING with a timestamped format; requires the harbor_env
# package (from the pinned OpenEnv checkout) to be importable.

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
# dependencies = [
#     "trl",
#     "trackio",
#     "datasets",
#     "openenv @ git+https://github.com/huggingface/OpenEnv.git@86a180ede21e044f7929b9a7783ad83aa67d83a3",
# ]
# ///

"""AsyncGRPO on Harbor tasks, with an off-the-shelf coding agent, served through OpenEnv.

A Harbor task contains a container image, instruction, and held-out verifier. This example uses
`mini-swe-agent` in an E2B sandbox:

    mini-swe-agent solves a Harbor task in a sandbox
      -> every model call it makes goes through the OpenEnv server's capture proxy to your vLLM
      -> the proxy records exact token ids and the sampling distribution's logprobs
      -> AsyncGRPO trains on them and syncs new weights back into that same vLLM

The agent owns the tool loop; TRL consumes OpenEnv's captured token IDs, logprobs, and masks.
Rewritten histories fork into separate training rows to preserve eligible tokens. See README.md
for the contract, dependency pin, and memory/weighting limitations.

Select another qualified agent with `--harness`. Rollouts and weight updates must use the same vLLM
instance and full-vocabulary sampling policy. Weight updates pause active inference requests without waiting for them to finish.

The reward adds `0.3 * tool_efficiency` only when correctness is at least 1.0. Use `--reward-key` to
select a component when the verifier returns a reward dictionary.

Requirements:
  - A running OpenEnv Harbor server, which owns the dataset and the sandbox templates:
        MAX_CONCURRENT_ENVS=9 openenv harbor serve --dataset <hf-dataset> --port 8200 --capture-port 8300 --expose gradio
  - A sandbox backend credential for the server's environment, e.g. `E2B_API_KEY`.
  - An OpenAI-compatible vLLM server (below) reachable at `--vllm-url`.
  - The pinned OpenEnv checkout and `PYTHONPATH` setup in this example's README.md.

Run (2 GPUs: vLLM on one, trainer on the other):

```sh
# Terminal 1: serve the policy with token capture and weight updates.
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3.5-2B \
    --host 0.0.0.0 --port 8000 \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --logprobs-mode processed_logprobs \
    --return-tokens-as-token-ids \
    --weight-transfer-config '{"backend":"nccl"}'

# Terminal 2 - train.
CUDA_VISIBLE_DEVICES=1 python examples/async_grpo_harbor/async_grpo_harbor.py \
    --server http://localhost:8200 \
    --vllm-url http://localhost:8000 \
    --model Qwen/Qwen3.5-2B \
    --split <hf-dataset> \
    --max-steps 20
```
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import uuid
from functools import partial

from datasets import Dataset
from harbor_env.harness import HarborSessionFactory
from transformers import AutoTokenizer

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo.openenv_harness import HarnessRolloutOutcome, HarnessRolloutWorker


logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

W_TOOL_EFFICIENCY = 0.3
TOOL_BUDGET = 15.0


def tool_efficiency(n_tool_calls: int | None) -> float | None:
    # [FUNCTION DOC]
    # HELPER: Scores how frugally the agent used tools.
    # Behavior: returns None if the count is None or TOOL_BUDGET is not positive; otherwise one minus count divided by
    # TOOL_BUDGET, clamped to the range 0 to 1.
    # Inputs: n_tool_calls (int or None). Outputs: float in [0, 1] or None.
    # Exceptions: none. Side effects: none.
    """`clip(1 - n/TOOL_BUDGET, 0, 1)`, or `None` when the tool count is unknown."""
    if n_tool_calls is None or TOOL_BUDGET <= 0:
        return None
    return max(0.0, min(1.0, 1.0 - n_tool_calls / TOOL_BUDGET))


def harbor_reward(outcome: HarnessRolloutOutcome) -> float | None:
    # [FUNCTION DOC]
    # Behavior: rollout reward function. If outcome.env_reward (verifier score) is None, logs a warning and returns None
    # (rollout excluded from the group baseline). If outcome.timed_out, logs a warning but keeps the score. Reward is the
    # float correctness, plus W_TOOL_EFFICIENCY times tool_efficiency(outcome.tool_call_count) only when correctness is at
    # least 1.0 and efficiency is known.
    # Inputs: outcome, a HarnessRolloutOutcome. Outputs: float reward or None.
    # Exceptions: ValueError/TypeError if env_reward cannot be cast to float.
    # Side effects: warning log lines.
    """Reward for one rollout, or `None` when it is unscorable.

    Args:
        outcome (`HarnessRolloutOutcome`):
            The verifier's reward, transcript, tool-call count, and
            whether the agent ran out of wall clock.

    Returns:
        `float` or `None`: Unscored rollouts are excluded from the group baseline.
    """
    correctness = outcome.env_reward
    if correctness is None:
        logger.warning("verifier did not run (tool_calls=%d); rollout unscorable", outcome.tool_call_count)
        return None

    if outcome.timed_out:
        logger.warning("agent timed out; keeping the verifier's score of %.3f on the partial work", correctness)

    correctness = float(correctness)
    reward = correctness

    eff = tool_efficiency(outcome.tool_call_count)
    if eff is not None and correctness >= 1.0:
        reward += W_TOOL_EFFICIENCY * eff

    return reward


def task_indices(spec: str) -> list[int] | None:
    # [FUNCTION DOC]
    # HELPER: Parses the --task-indices option.
    # Behavior: strips the spec (None treated as empty); empty returns None. A leading "@" means read the rest as a file
    # path and use its text. Newlines are treated as commas; non-blank comma-separated items are converted to int.
    # Inputs: spec string. Outputs: list of ints or None.
    # Exceptions: ValueError on non-integer items; OSError if the file cannot be read.
    # Side effects: file read when "@path" is used.
    """Read comma-separated task indices, directly or from `@path` (safe for Slurm exports)."""
    spec = (spec or "").strip()
    if not spec:
        return None
    if spec.startswith("@"):
        spec = pathlib.Path(spec[1:]).read_text()
    return [int(x) for x in spec.replace("\n", ",").split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    # [FUNCTION DOC]
    # Behavior: builds an argparse parser (description is the module docstring) with options for the OpenEnv server URL,
    # vLLM URL (required), model, task split (required), harness, sandbox backend, reward key, task count/indices,
    # generation count, in-flight task limit, completion length, steps, learning rate, temperature, max staleness,
    # agent timeout/step limit, per-device batch size, output dir, trackio project/space and run name; parses sys.argv.
    # Inputs: command line. Outputs: argparse.Namespace.
    # Exceptions: SystemExit on invalid arguments or --help. Side effects: may print usage.
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default="http://127.0.0.1:8200", help="a running `openenv harbor serve`")
    p.add_argument("--vllm-url", required=True, help="the engine AsyncGRPO generates from and syncs weights into")
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--split", required=True, help="the Harbor task dataset the server was started with")
    p.add_argument("--harness", default="mini-swe-agent", help="any harness the server reports; see the docstring")
    p.add_argument("--sandbox", default="e2b")
    p.add_argument("--reward-key", default="")
    p.add_argument("--n-tasks", type=int, default=32)
    p.add_argument("--task-indices", default="", help="comma-separated indices, or @path to a file of them")
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-inflight", type=int, default=8)
    p.add_argument("--max-completion-length", type=int, default=1024)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-staleness", type=int, default=4)
    # Sandbox setup has its own timeouts; this bounds the agent run.
    p.add_argument("--agent-timeout", type=float, default=300.0)
    # Bound model calls; rewritten histories can still produce multiple training rows.
    p.add_argument("--agent-step-limit", type=int, default=12)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--project", default="async-grpo-harbor")
    p.add_argument("--trackio-space-id", default=None, help="host the trackio dashboard on a HF Space")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def main() -> None:
    # [FUNCTION DOC]
    # Behavior: parses args; builds a run stamp from SLURM_JOB_ID, HF_JOB_ID, JOB_ID or "local" plus 8 random hex chars;
    # derives run_name and output_dir (runs/async_grpo_harbor/<run_name>) unless given. Creates a partial
    # HarborSessionFactory with all harness/sandbox/agent settings, instantiates one to fetch prompt rows as a Dataset,
    # and closes its private _env in a finally block. Prints a summary, loads the tokenizer, builds AsyncGRPOConfig
    # (checkpoint every 50 steps keeping 3, full-vocab sampling top_p 1 / top_k -1, bf16, non-reentrant gradient
    # checkpointing, trackio, log every step, seed 0), builds a HarnessRolloutWorker using harbor_reward as rollout reward
    # (no reward_funcs, thinking disabled), then trains an AsyncGRPOTrainer with that worker and saves the model.
    # Inputs: command line. Outputs: none.
    # Exceptions: propagates server/network/training errors.
    # Side effects: network calls to OpenEnv and vLLM servers, sandbox creation, stdout prints, trackio logging,
    # checkpoints and final model written to output_dir.
    args = parse_args()
    stamp = os.environ.get("SLURM_JOB_ID") or os.environ.get("HF_JOB_ID") or os.environ.get("JOB_ID") or "local"
    stamp = f"{stamp}-{uuid.uuid4().hex[:8]}"
    run_name = args.run_name or f"{args.model.split('/')[-1]}-{args.harness}-{args.max_steps}steps-{stamp}"
    output_dir = args.output_dir or f"runs/async_grpo_harbor/{run_name}"

    factory = partial(
        HarborSessionFactory,
        args.server,
        split=args.split,
        harness=args.harness,
        sandbox=args.sandbox,
        llm_url=args.vllm_url,
        model=args.model,
        agent_timeout_sec=args.agent_timeout,
        agent_step_limit=args.agent_step_limit,
        reward_key=args.reward_key,
        num_tasks=args.n_tasks,
        indices=task_indices(args.task_indices),
    )
    metadata_factory = factory()
    try:
        dataset = Dataset.from_list(metadata_factory.prompt_rows())
    finally:
        # The pinned OpenEnv factory has no public close method.
        if metadata_factory._env is not None:
            metadata_factory._env.close()

    print(f"server    {args.server}")
    print(f"vllm      {args.vllm_url}   model {args.model}")
    print(f"rollouts  {args.harness} on {args.sandbox}, {args.num_generations}x{args.max_inflight}")
    print(f"tasks     {len(dataset)} from {args.split}")
    print(f"output    {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    config = AsyncGRPOConfig(
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        top_p=1.0,
        top_k=-1,
        max_staleness=args.max_staleness,
        vllm_server_base_url=args.vllm_url,
        optim="adamw_torch",
        dtype="bfloat16",
        bf16=True,
        gradient_checkpointing=True,
        # Support layers whose inputs are passed as keyword arguments.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="trackio",
        project=args.project,
        trackio_space_id=args.trackio_space_id,
        run_name=run_name,
        log_completions=True,
        logging_steps=1,
        seed=0,
    )

    worker = HarnessRolloutWorker(
        harness_session_factory=factory,
        harness_adapter=None,
        rollout_reward_fn=harbor_reward,
        model_name=args.model,
        dataset=dataset,
        reward_funcs=[],  # the reward is the task's own verifier, via `rollout_reward_fn`
        processing_class=tokenizer,
        chat_template_kwargs={"enable_thinking": False},
        num_generations=args.num_generations,
        max_inflight_tasks=args.max_inflight,
        vllm_server_url=args.vllm_url,
        max_tokens=args.max_completion_length,
        temperature=args.temperature,
        top_p=1.0,
        top_k=-1,
        log_completions=True,
        num_completions_to_print=2,
    )

    trainer = AsyncGRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        rollout_worker=worker,
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
