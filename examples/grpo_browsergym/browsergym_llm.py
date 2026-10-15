# [MODULE DOC]
# Example script: GRPO training of a text-only LLM (default Qwen3-0.6B) on OpenEnv BrowserGym web tasks (default
# MiniWoB click-test) hosted on a Hugging Face Space, using the accessibility tree as observation and vLLM generation
# (colocate or server). Main contents: parse_args, sanitize_name, SYSTEM_PROMPT (describes the bid notation and the
# tools), reward_completion, main (which defines the BrowserGymLLMEnv environment class inline).
# Import-time side effects: none beyond imports (requires the browsergym_env package).

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
#     "trl[vllm,peft]",
#     "trackio",
#     "kernels",
#     "openenv-browsergym-env @ git+https://huggingface.co/spaces/openenv/browsergym_env",
# ]
# ///

"""
Simple script to run GRPO training with OpenEnv's BrowserGym environment and vLLM for LLMs.

This script is optimized for text-only Language Models (LLMs). It uses the accessibility
tree text from BrowserGym, making it memory-efficient.

The environment runs on a Hugging Face Space by default.

Setup (Option A - Install from HF Space, recommended):

```sh
uv pip install git+https://huggingface.co/spaces/openenv/browsergym_env
```

Setup (Option B - Clone OpenEnv repo, for development):

```sh
git clone https://github.com/huggingface/OpenEnv.git
cd OpenEnv/envs/browsergym_env
uv pip install -e .
```

# Option 1: HF Spaces + Colocated vLLM (1 GPU required)
```sh
python examples/grpo_browsergym/browsergym_llm.py --vllm-mode colocate
```

# Option 2: HF Spaces + Separate vLLM server (2 GPUs required)

# Spin up vLLM server (Terminal 1)
```sh
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8001 \
    --weight-transfer-config '{"backend": "nccl"}' \
    --logprobs-mode processed_logprobs \
    --max-logprobs -1
```

# Run training (Terminal 2)
```sh
CUDA_VISIBLE_DEVICES=1 python examples/grpo_browsergym/browsergym_llm.py --vllm-mode server --vllm-server-url http://localhost:8001
```
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from browsergym_env import BrowserGymAction, BrowserGymEnv
from datasets import Dataset

from trl import GRPOConfig, GRPOTrainer


def parse_args() -> argparse.Namespace:
    # [FUNCTION DOC]
    # Behavior: argparse parser for model id, Space URL, benchmark, task name, dataset prompt/size, max steps per episode,
    # completion length, sampling (temperature 0.7, top-k 50, optional top-p), optimizer settings (lr 5e-6, weight decay,
    # grad accumulation 32, warmup 10), batch size, generations per prompt (4), epochs, checkpoint interval/limit, output
    # dir, run name, project, vLLM mode/server URL and logging steps.
    # Inputs: command line. Outputs: argparse.Namespace. Exceptions: SystemExit on bad args. Side effects: may print usage.
    parser = argparse.ArgumentParser(description="Run GRPO training for BrowserGym MiniWoB using OpenEnv environment.")
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-0.6B",
        help="Model identifier passed to GRPOTrainer for fine-tuning.",
    )
    parser.add_argument(
        "--space-url",
        type=str,
        default="https://openenv-browsergym-env.hf.space",
        help="URL for the Hugging Face Space running the BrowserGym environment.",
    )
    parser.add_argument(
        "--benchmark",
        default="miniwob",
        help="BrowserGym benchmark to use (miniwob, webarena, etc.).",
    )
    parser.add_argument(
        "--task-name",
        default="click-test",
        help="Specific task within the benchmark (e.g., click-test, click-button).",
    )
    parser.add_argument(
        "--dataset-prompt",
        default="Complete the web task successfully.",
        help="Prompt text used to seed the training dataset.",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=1000,
        help="Number of entries to include in the synthetic training dataset.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum number of steps per episode.",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=1024,
        help="Maximum completion length in tokens for tool-calling generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature used during rollout generation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter forwarded to vLLM.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Optional top-p sampling parameter forwarded to vLLM.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate for GRPO training.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay applied during optimization.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=32,
        help="Gradient accumulation steps for GRPO training.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Warmup steps for the scheduler.",
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=1,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="Number of rollout generations per dataset prompt.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=50,
        help="Interval (in steps) between checkpoint saves.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=None,
        help="Maximum number of checkpoints to keep.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where training outputs and checkpoints are stored.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name for logging systems.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Optional project identifier for logging systems.",
    )
    parser.add_argument(
        "--vllm-mode",
        choices=("colocate", "server"),
        default="colocate",
        help="vLLM execution mode: 'colocate' or 'server'.",
    )
    parser.add_argument(
        "--vllm-server-url",
        type=str,
        default="http://localhost:8001",
        help="URL for the vLLM server (only used when --vllm-mode=server).",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=1,
        help="Frequency of logging steps for GRPO training.",
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Replaces every "/" with "-" so model ids can be used in paths and run names.
    # Inputs: name. Outputs: string. Exceptions: none. Side effects: none.
    return name.replace("/", "-")


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You control a web browser to complete tasks.

The page structure shows elements as: [bid] element_type 'element_text'
For example: [13] button 'Click Me!' means the element has bid='13'.

Use the available tools to interact with the page:
- click: Click an element by its bid
- fill: Fill an input field with text
- send_keys: Send keyboard input
- scroll: Scroll the page
- noop: Do nothing

Complete the given task as efficiently as possible."""


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


def reward_completion(environments, **kwargs) -> list[float]:
    # [FUNCTION DOC]
    # Behavior: reward function returning each environment's reward attribute (shaped success reward).
    # Inputs: environments list, other kwargs ignored. Outputs: list of floats. Exceptions: none. Side effects: none.
    """Reward for task completion."""
    return [env.reward for env in environments]


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    # [FUNCTION DOC]
    # Behavior: parses args; builds a dataset of dataset_size identical prompts (system SYSTEM_PROMPT plus user
    # dataset_prompt); defines BrowserGymLLMEnv capturing space_url, max_steps and args by closure; derives output_dir
    # (outputs/browsergym-grpo-<model>-<timestamp> by default); builds GRPOConfig (vLLM on, 0.4 GPU memory, server URL
    # only in server mode, generation batch equal to num_generations, trackio space named after model and timestamp,
    # step checkpoints, thinking disabled), sets run_name and project defaults, creates GRPOTrainer with
    # reward_completion and environment_factory BrowserGymLLMEnv, prints a summary banner, trains and calls trainer.end().
    # Inputs: command line. Outputs: none.
    # KNOWN ISSUE: the --benchmark option is only printed; it is never passed to the environment.
    # Exceptions: propagates network/training errors.
    # Side effects: HTTP/WebSocket traffic to the Space, GPU training, trackio logging, checkpoints, stdout prints.
    args = parse_args()

    space_url = args.space_url
    max_steps = args.max_steps

    dataset = Dataset.from_dict(
        {
            "prompt": [
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": args.dataset_prompt},
                ]
            ]
            * args.dataset_size
        }
    )

    class BrowserGymLLMEnv:
        # [CLASS DOC]
        # Per-rollout environment wrapping a BrowserGymEnv client connected to the Space. State: client, reward (shaped
        # episode reward read by reward_completion), _done flag and _step_count. reset starts an episode; click, fill,
        # send_keys, scroll and noop are tools exposed to the model (via docstrings) that all delegate to _do_action, which
        # sends a BrowserGym action string and formats the observation. No base class.
        def __init__(self):
            # [FUNCTION DOC]
            # Behavior: creates the BrowserGymEnv client for the closed-over space_url and zeroes reward, done and step count.
            # Inputs: none. Outputs: none. Exceptions: client construction errors. Side effects: creates a client object.
            self.client = BrowserGymEnv(base_url=space_url)
            self.reward = 0.0
            self._done = False
            self._step_count = 0

        def _ensure_large_max_size(self):
            # [FUNCTION DOC]
            # HELPER: Works around the 1 MB default WebSocket message limit for large accessibility trees.
            # Behavior: connects the client, takes its private _ws; if it has a protocol object, picks the attribute name
            # max_size (older websockets) or max_message_size (newer), and if its value is exactly 2**20 bytes raises it to 100 MB.
            # Inputs: none. Outputs: none. Exceptions: connection errors.
            # Side effects: opens the WebSocket connection; mutates protocol limits.
            """Raise WebSocket max message size for large observations (e.g. accessibility trees).

            openenv-core<=0.2.1 does not pass max_size to ws_connect, so the websockets library
            defaults to 1MB. We force a connection and patch it to 100MB before any messages are sent.
            """
            self.client.connect()
            ws = self.client._ws
            if ws is not None and hasattr(ws, "protocol"):
                proto = ws.protocol
                # websockets <16: max_size; websockets >=16: max_message_size
                attr = "max_size" if hasattr(proto, "max_size") else "max_message_size"
                if getattr(proto, attr) == 2**20:
                    setattr(proto, attr, 100 * 1024 * 1024)

        def reset(self, **kwargs) -> str:
            # [FUNCTION DOC]
            # Behavior: zeroes reward, done and step count, ensures the large message limit, resets the remote env with the
            # closed-over task name, stores result.done and returns the formatted observation.
            # Inputs: kwargs ignored. Outputs: observation string. Exceptions: network errors. Side effects: remote reset.
            self.reward = 0.0
            self._done = False
            self._step_count = 0
            self._ensure_large_max_size()
            result = self.client.reset(task_name=args.task_name)
            self._done = result.done
            return self._format_observation(result.observation)

        def click(self, bid: str) -> str:
            # [FUNCTION DOC]
            # Tool: clicks the element with the given bid by sending a click action with the repr-quoted bid.
            # Inputs: bid string. Outputs: observation string. Exceptions: ValueError if the episode is done (from _do_action).
            # Side effects: remote step.
            """Click an element on the page.

            Args:
                bid: The BrowserGym ID of the element to click.

            Returns:
                The updated page observation.
            """
            return self._do_action(f"click({bid!r})")

        def fill(self, bid: str, text: str) -> str:
            # [FUNCTION DOC]
            # Tool: fills the input with given bid with text via a fill action with both arguments repr-quoted.
            # Inputs: bid, text. Outputs: observation string. Exceptions: as _do_action. Side effects: remote step.
            """Fill an input field with text.

            Args:
                bid: The BrowserGym ID of the input field.
                text: The text to type into the field.

            Returns:
                The updated page observation.
            """
            return self._do_action(f"fill({bid!r}, {text!r})")

        def send_keys(self, text: str) -> str:
            # [FUNCTION DOC]
            # Tool: sends keyboard input via a send_keys action with the repr-quoted text.
            # Inputs: text. Outputs: observation string. Exceptions: as _do_action. Side effects: remote step.
            """Send keyboard input to the page.

            Args:
                text: The keyboard input to send.

            Returns:
                The updated page observation.
            """
            return self._do_action(f"send_keys({text!r})")

        def scroll(self, direction: str) -> str:
            # [FUNCTION DOC]
            # Tool: scrolls via a scroll action with the repr-quoted direction (expected "up" or "down", not validated).
            # Inputs: direction. Outputs: observation string. Exceptions: as _do_action. Side effects: remote step.
            """Scroll the page.

            Args:
                direction: Direction to scroll, either 'up' or 'down'.

            Returns:
                The updated page observation.
            """
            return self._do_action(f"scroll({direction!r})")

        def noop(self) -> str:
            # [FUNCTION DOC]
            # Tool: sends a no-op action and returns the current observation.
            # Inputs: none. Outputs: observation string. Exceptions: as _do_action. Side effects: remote step.
            """Do nothing and observe the current page state.

            Returns:
                The current page observation.
            """
            return self._do_action("noop()")

        def _do_action(self, action_str: str) -> str:
            # [FUNCTION DOC]
            # HELPER: Executes one action string.
            # Behavior: raises if done; increments step count; steps the client; reward shaping: if the episode is done, reward
            # is 1.0 when the step reward is positive else 0.0; otherwise reward equals the step reward (None treated as 0).
            # Marks done once step count reaches the closed-over max_steps. Returns the formatted observation.
            # Inputs: action_str. Outputs: observation string.
            # Exceptions: ValueError("Episode is done.") when already done; network errors.
            # Side effects: remote step; mutates reward, _done, _step_count.
            if self._done:
                raise ValueError("Episode is done.")

            self._step_count += 1
            result = self.client.step(BrowserGymAction(action_str=action_str))
            observation = result.observation
            step_reward = float(result.reward or 0.0)
            self._done = result.done

            # Reward shaping: binary success/failure on completion
            if self._done and step_reward > 0:
                self.reward = 1.0
            elif self._done:
                self.reward = 0.0
            else:
                self.reward = step_reward

            # Enforce max steps
            if self._step_count >= max_steps:
                self._done = True

            return self._format_observation(observation)

        def _format_observation(self, observation) -> str:
            # [FUNCTION DOC]
            # HELPER: Builds the text observation: "Goal: ..." if a goal exists, "Error: ..." if last_action_error and error are
            # set, and "Page structure:" followed by the accessibility tree truncated to 2000 chars plus "..." if longer; parts
            # are joined with blank lines, or "No observation available." if empty.
            # Inputs: BrowserGym observation. Outputs: string. Exceptions: none. Side effects: none.
            parts = []
            if observation.goal:
                parts.append(f"Goal: {observation.goal}")
            if observation.last_action_error and observation.error:
                parts.append(f"Error: {observation.error}")
            if observation.axtree_txt:
                axtree = observation.axtree_txt
                if len(axtree) > 2000:
                    axtree = axtree[:2000] + "..."
                parts.append(f"Page structure:\n{axtree}")
            return "\n\n".join(parts) if parts else "No observation available."

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_output_dir = Path("outputs") / f"browsergym-grpo-{sanitize_name(args.model_id)}-{timestamp}"
    output_dir = Path(args.output_dir or default_output_dir)

    grpo_config = GRPOConfig(
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_url if args.vllm_mode == "server" else None,
        vllm_gpu_memory_utilization=0.4,
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        warmup_steps=args.warmup_steps,
        num_generations=args.num_generations,
        generation_batch_size=args.num_generations,
        max_completion_length=args.max_completion_length,
        logging_steps=args.logging_steps,
        report_to="trackio",
        trackio_space_id=f"browsergym-grpo-{sanitize_name(args.model_id)}-{timestamp}",
        save_strategy="steps",
        save_steps=args.save_interval,
        save_total_limit=args.save_total_limit,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        chat_template_kwargs={"enable_thinking": False},
    )

    grpo_config.run_name = args.run_name or f"run-{timestamp}"
    grpo_config.project = args.project or f"group-{sanitize_name(args.model_id)}"

    trainer = GRPOTrainer(
        model=args.model_id,
        reward_funcs=[reward_completion],
        train_dataset=dataset,
        args=grpo_config,
        environment_factory=BrowserGymLLMEnv,
    )

    print("=" * 80)
    print("Starting GRPO training with BrowserGym environment (LLM mode)")
    print(f"Benchmark: {args.benchmark}")
    print(f"Task: {args.task_name}")
    print(f"Model: {args.model_id}")
    print("Mode: LLM (text-only, using accessibility tree)")
    print(f"Using {args.num_generations} rollouts per dataset prompt")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    trainer.train()

    trainer.end()


if __name__ == "__main__":
    main()
