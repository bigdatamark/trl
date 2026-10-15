# [MODULE DOC]
# Example script: GRPO with vLLM on OpenEnv's OpenSpiel Catch game (default Qwen2.5-0.5B-Instruct). The model
# controls a paddle through move/stay tools; reward is the clamped game outcome. The environment can be launched
# locally with uvicorn, used from an already running Docker container, started from a Docker image, or reached on a
# HF Space. Main contents: parse_args, start_env_server, BASE_PROMPT, reward_from_env, main (defines CatchEnv inline).
# Import-time side effects: none beyond imports (requires openenv and openspiel_env).

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
#     "openenv-openspiel-env @ git+https://huggingface.co/spaces/openenv/openspiel_env",
# ]
# ///


"""
Simple script to run GRPO training with OpenEnv's Catch environment (OpenSpiel) and vLLM. The reward function
is based on the catch game where the agent tries to catch falling balls.

Setup (Option A - Install from HF Space, recommended):

```sh
uv pip install git+https://huggingface.co/spaces/openenv/openspiel_env
```

Setup (Option B - Clone OpenEnv repo, for development):

```sh
git clone https://github.com/huggingface/OpenEnv.git
cd OpenEnv/envs/openspiel_env
uv pip install -e .
```

# Option 1: HF Spaces + Colocated vLLM (1 GPU required)
```sh
python examples/grpo_catch/grpo_catch.py --env-mode space --env-host https://openenv-openspiel-env.hf.space --vllm-mode colocate
```

# Option 2: HF Spaces + Separate vLLM server (2 GPUs required)

# Spin up vLLM server (Terminal 1)
```sh
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen2.5-0.5B-Instruct --host 0.0.0.0 --port 8000 \
    --weight-transfer-config '{"backend": "nccl"}' \
    --logprobs-mode processed_logprobs \
    --max-logprobs -1
```

# Run training (Terminal 2)
```sh
CUDA_VISIBLE_DEVICES=1 python examples/grpo_catch/grpo_catch.py --env-mode space --env-host https://openenv-openspiel-env.hf.space --vllm-mode server --vllm-server-url http://localhost:8000
```

# Option 3: Local + Colocated vLLM (1 GPU required)

# Start the environment only if using --env-mode docker-local
```sh
docker run -d -p 8001:8001 registry.hf.space/openenv-openspiel-env:latest
```

```sh
python examples/grpo_catch/grpo_catch.py --env-mode docker-local --vllm-mode colocate
```
"""

# ruff: noqa: T201
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from datasets import Dataset
from openenv.core.containers.runtime import LocalDockerProvider
from openspiel_env import OpenSpielEnv
from openspiel_env.models import OpenSpielAction

from trl import GRPOConfig, GRPOTrainer, RichProgressCallback


def parse_args():
    # [FUNCTION DOC]
    # Behavior: argparse for env host (0.0.0.0) and port (8001), env mode (local, docker-local, docker-image default,
    # space), model, dataset size (1000), Docker image name, vLLM mode and server URL.
    # Inputs: command line. Outputs: Namespace. Exceptions: SystemExit on bad args. Side effects: may print usage.
    parser = argparse.ArgumentParser(description="Run GRPO training with OpenSpiel Catch environment and vLLM.")

    # --- Environment settings ---
    parser.add_argument("--env-host", type=str, default="0.0.0.0", help="Host for the environment server.")
    parser.add_argument("--env-port", type=int, default=8001, help="Port for the environment server.")
    parser.add_argument(
        "--env-mode",
        choices=["local", "docker-local", "docker-image", "space"],
        default="docker-image",
        help="Where to run the environment: 'local' to launch it, 'docker-local' if already running locally, 'docker-image' to run from a Docker image, or 'space' to use a remote Space URL.",
    )
    # --- Generation and model config ---
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model name or path.",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=1000,
        help="Number of prompts to use for training dataset.",
    )
    parser.add_argument(
        "--env-image", type=str, default="openspiel-env:latest", help="Docker image for the OpenSpiel environment."
    )
    parser.add_argument(
        "--vllm-mode",
        choices=["colocate", "server"],
        default="colocate",
        help="vLLM execution mode: 'colocate' or 'server'.",
    )
    parser.add_argument(
        "--vllm-server-url",
        type=str,
        default="http://localhost:8000",
        help="URL for the vLLM server (only used when --vllm-mode=server).",
    )

    return parser.parse_args()


def start_env_server(env_host: str, env_port: int):
    # [FUNCTION DOC]
    # Behavior: launches "python -m uvicorn envs.openspiel_env.server.app:app" on host/port with cwd set to the parent of
    # the current directory and PYTHONPATH pointing to its src folder, capturing stdout/stderr as text pipes; sleeps 5 s,
    # then probes /health with a 2 s timeout. On failure prints the error and the server's stderr and re-raises.
    # Inputs: env_host, env_port. Outputs: the Popen handle.
    # Exceptions: any requests exception from the health probe.
    # KNOWN ISSUE: stdout/stderr are piped but never drained after startup, so a chatty server can block once the pipe
    # buffer fills.
    # Side effects: spawns a process, sleeps, HTTP request, prints.
    """Launch the OpenSpiel Catch environment locally via uvicorn."""
    env_url = f"http://{env_host}:{env_port}"
    print(f"⚡ Starting FastAPI server for OpenSpiel Catch Environment on {env_url}...")

    work_dir = str(Path.cwd().parent.absolute())
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "envs.openspiel_env.server.app:app",
            "--host",
            env_host,
            "--port",
            str(env_port),
        ],
        env={**os.environ, "PYTHONPATH": f"{work_dir}/src"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=work_dir,
    )

    print("⏳ Waiting for server to start...")
    time.sleep(5)

    try:
        requests.get(f"{env_url}/health", timeout=2)
        print("\n✅ OpenSpiel Catch Environment server is running!")
    except Exception as e:
        print(f"\n❌ Server failed to start: {e}")
        if process.stderr:
            print(process.stderr.read())
        raise

    return process


BASE_PROMPT = """You are an AI agent playing the game **Catch**.

### Game Description
- The game is played on a **10×5 grid**.
- There is one **falling ball** and one **paddle** that you control at the bottom.
- The objective is to **move the paddle left or right to catch the ball** as it falls.
- The episode ends when the ball reaches the bottom row:
  - You get **+1 reward** if you catch it.
  - You get **–1 reward** if you miss it.

### Observation Format
Each observation is a flattened 10x5 grid (list of 50 floats).
- 1.0 → occupied (ball or paddle)
- 0.0 → empty cell

You have the following tools available:
- `move(direction)`: Move the paddle left or right. Direction must be "left" or "right".
- `stay`: Do nothing and let the ball fall one step.

Observe the grid, determine where the ball is relative to the paddle, then move accordingly.
"""


def reward_from_env(environments, **kwargs):
    # [FUNCTION DOC]
    # Behavior: for each environment, returns max(reward, 0) if the episode is done (catch +1 stays 1, miss -1 becomes
    # 0), else 0.0 for incomplete episodes.
    # Inputs: environments, kwargs ignored. Outputs: list of floats. Exceptions: none. Side effects: none.
    rewards = []
    for env in environments:
        if env.done:
            # Catch gives +1 for catching, -1 for missing. Clamp to [0, 1] for GRPO advantage estimation.
            rewards.append(max(env.reward, 0.0))
        else:
            rewards.append(0.0)  # Incomplete episode
    return rewards


def main():
    # [FUNCTION DOC]
    # Behavior: parses args and resolves env_url by mode: local (start_env_server, keep process), docker-local
    # (host:port), docker-image (LocalDockerProvider starts the image and waits for readiness), space (env_host used as
    # URL); unknown modes raise. Builds dataset_size identical BASE_PROMPT rows, defines CatchEnv, builds GRPOConfig
    # (output_dir and trackio space "<model>-GRPO-Catch", vLLM with 0.2 GPU memory, server URL only in server mode, log every
    # step with completions, batch 2, 8 generations, completion length 4096, grad accumulation 16, thinking disabled),
    # trains GRPOTrainer with reward_from_env, environment_factory CatchEnv and RichProgressCallback; in a finally block
    # terminates and waits for a locally launched server; then calls trainer.end().
    # Inputs: command line. Outputs: none.
    # Exceptions: ValueError for unknown env mode; propagates server/training errors.
    # Side effects: may start processes/containers, network traffic, GPU training, trackio logging, files, prints.
    args = parse_args()

    # Select environment mode — all modes resolve to env_url
    if args.env_mode == "local":
        env_url = f"http://{args.env_host}:{args.env_port}"
        server_process = start_env_server(args.env_host, args.env_port)
    elif args.env_mode == "docker-local":
        env_url = f"http://{args.env_host}:{args.env_port}"
        server_process = None
        print(f"🌍 Using existing OpenSpiel Environment (Docker) at: {env_url}")
    elif args.env_mode == "docker-image":
        provider = LocalDockerProvider()
        env_url = provider.start_container(args.env_image)
        provider.wait_for_ready(env_url)
        server_process = None
        print("🌍 Using OpenSpiel Environment (Docker) from local Image")
    elif args.env_mode == "space":
        env_url = args.env_host
        server_process = None
        print(f"🌍 Using Hugging Face Space environment at: {env_url}")
    else:
        raise ValueError(f"Unknown environment mode: {args.env_mode}")

    dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": BASE_PROMPT}]] * args.dataset_size})

    class CatchEnv:
        # [CLASS DOC]
        # Per-rollout Catch environment wrapping an OpenSpielEnv client at the closed-over env_url. Class constants ROWS 10 and
        # COLS 5 describe the grid. State: client, reward (last step reward), done. reset starts a game; move and stay are
        # tools that delegate to _do_action; _format_obs turns the flat grid into text. No base class.
        ROWS = 10
        COLS = 5

        def __init__(self):
            # [FUNCTION DOC]
            # Behavior: creates the OpenSpielEnv client and sets reward 0 and done False.
            # Inputs: none. Outputs: none. Exceptions: client errors. Side effects: creates a client.
            self.client = OpenSpielEnv(base_url=env_url)
            self.reward = 0.0
            self.done = False

        @staticmethod
        def _format_obs(info_state: list[float]) -> str:
            # [FUNCTION DOC]
            # HELPER: Static method describing the 50-cell flattened grid (row-major, 5 columns).
            # Behavior: for each cell equal to 1.0, computes row/column; cells above the last row are the ball (1-based row and
            # column stored), cells in the last row are the paddle (1-based column). Emits lines for ball position, paddle
            # column, and the ball's relative horizontal position (columns LEFT/RIGHT of the paddle or directly above).
            # Inputs: info_state list of floats. Outputs: newline-joined description (may be empty).
            # KNOWN ISSUE: when the ball reaches the last row it is indistinguishable from the paddle; the later cell overwrites
            # paddle_col and the ball line is omitted.
            # Exceptions: none. Side effects: none.
            """Convert the flat 50-float observation into a readable text description."""
            rows, cols = CatchEnv.ROWS, CatchEnv.COLS
            ball_row = ball_col = paddle_col = None
            for idx, val in enumerate(info_state):
                if val == 1.0:
                    r, c = divmod(idx, cols)
                    if r < rows - 1:
                        ball_row, ball_col = r + 1, c + 1
                    else:
                        paddle_col = c + 1
            parts = []
            if ball_row is not None and ball_col is not None:
                parts.append(f"Ball: row {ball_row}/{rows}, column {ball_col}/{cols}")
            if paddle_col is not None:
                parts.append(f"Paddle: column {paddle_col}/{cols}")
            if ball_col is not None and paddle_col is not None:
                diff = ball_col - paddle_col
                if diff < 0:
                    parts.append(f"The ball is {abs(diff)} column(s) to the LEFT of the paddle.")
                elif diff > 0:
                    parts.append(f"The ball is {diff} column(s) to the RIGHT of the paddle.")
                else:
                    parts.append("The ball is directly above the paddle.")
            return "\n".join(parts)

        def reset(self, **kwargs) -> str:
            # [FUNCTION DOC]
            # Behavior: resets the remote game, sets reward 0, stores done from the observation and returns the formatted grid.
            # Inputs: kwargs ignored. Outputs: string. Exceptions: network errors. Side effects: remote reset, mutates state.
            env_result = self.client.reset()
            self.reward = 0.0
            self.done = env_result.observation.done
            return self._format_obs(env_result.observation.info_state)

        def _do_action(self, action_id: int) -> str:
            # [FUNCTION DOC]
            # HELPER: Raises if done; steps the client with an OpenSpielAction for game "catch" and the given id; stores reward
            # (None treated as 0) and done; returns the formatted grid.
            # Inputs: action_id (0 left, 1 stay, 2 right). Outputs: string.
            # Exceptions: ValueError("Episode is done."); network errors. Side effects: remote step, mutates reward/done.
            if self.done:
                raise ValueError("Episode is done.")
            env_result = self.client.step(OpenSpielAction(action_id=action_id, game_name="catch"))
            self.reward = env_result.reward or 0.0
            self.done = env_result.observation.done
            return self._format_obs(env_result.observation.info_state)

        def move(self, direction: str) -> str:
            # [FUNCTION DOC]
            # Tool: maps "left" to action 0 and "right" to action 2 and performs it.
            # Inputs: direction. Outputs: observation string.
            # Exceptions: ValueError for any other direction or when the episode is done. Side effects: remote step.
            """Move the paddle left or right.

            Args:
                direction: Direction to move, either "left" or "right".

            Returns:
                The observation after moving.
            """
            if direction == "left":
                action_id = 0
            elif direction == "right":
                action_id = 2
            else:
                raise ValueError(f"Invalid direction {direction!r}: must be 'left' or 'right'.")
            return self._do_action(action_id)

        def stay(self) -> str:
            # [FUNCTION DOC]
            # Tool: performs action 1 (no paddle movement; the ball falls one step).
            # Inputs: none. Outputs: observation string. Exceptions: as _do_action. Side effects: remote step.
            """Do nothing and let the ball fall one step.

            Returns:
                The observation after staying.
            """
            return self._do_action(1)

    training_args = GRPOConfig(
        output_dir=f"{args.model.split('/')[-1]}-GRPO-Catch",
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_url if args.vllm_mode == "server" else None,
        vllm_gpu_memory_utilization=0.2,
        logging_steps=1,
        log_completions=True,
        report_to="trackio",
        trackio_space_id=f"{args.model.split('/')[-1]}-GRPO-Catch",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        num_generations=8,
        max_completion_length=4096,
        gradient_accumulation_steps=16,
        chat_template_kwargs={"enable_thinking": False},
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_from_env,
        args=training_args,
        train_dataset=dataset,
        environment_factory=CatchEnv,
        callbacks=[RichProgressCallback()],
    )

    try:
        trainer.train()
    finally:
        if server_process:
            print("🛑 Terminating environment server...")
            server_process.terminate()
            server_process.wait()

    trainer.end()


if __name__ == "__main__":
    main()
