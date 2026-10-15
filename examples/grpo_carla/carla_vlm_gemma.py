# [MODULE DOC]
# Example script: Gemma-specific variant of carla_vlm.py (default google/gemma-4-E2B-it) for GRPO on OpenEnv's CARLA
# trolley scenario with camera-image tool responses; adds optional LoRA (by default all linear layers excluding the
# vision tower and multimodal projector), a configurable learning rate, reset retries and debug prints.
# Main contents: parse_args, PROMPT (asks for one tool call at a time and hints lane changes may be safer),
# CarlaVLMEnv, reward_func, main. Import-time side effects: none beyond imports.

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
#     "openenv-carla-env @ git+https://huggingface.co/spaces/sergiopaniego/carla-env",
# ]
# ///


"""
GRPO training with OpenEnv's CARLA environment for Gemma VLMs.

Gemma-specific variant of `carla_vlm.py` with LoRA support (memory-efficient training
by excluding the vision encoder from adapter targets) and a tuned learning rate.

Setup:

```sh
uv pip install git+https://huggingface.co/spaces/sergiopaniego/carla-env
```

Usage (requires at least 2 CARLA Spaces, each supports only 1 concurrent connection):

```sh
python examples/grpo_carla/carla_vlm_gemma.py \
    --env-urls https://server1.hf.space https://server2.hf.space \
    --use-lora
```
"""

import argparse
import base64
from io import BytesIO

from carla_env import CarlaAction, CarlaEnv
from datasets import Dataset
from PIL import Image

from trl import GRPOConfig, GRPOTrainer


def parse_args():
    # [FUNCTION DOC]
    # Behavior: argparse for model, env URLs (required, at least 2 recommended), dataset size (1000), completion length
    # (3072), per-device batch size (default None meaning number of URLs), grad accumulation (16), steps (100), image
    # size (256), trackio space id, LoRA flags (--use-lora, rank 128, alpha 256, target modules "llm-only" or explicit),
    # learning rate (5e-6), hub model id, run name and report_to backend (trackio).
    # Inputs: command line. Outputs: Namespace. Exceptions: SystemExit on bad args. Side effects: may print usage.
    parser = argparse.ArgumentParser(description="Run GRPO VLM training with CARLA environment.")
    parser.add_argument("--model", type=str, default="google/gemma-4-E2B-it")
    parser.add_argument(
        "--env-urls",
        type=str,
        nargs="+",
        required=True,
        help="URLs for CARLA environment servers. At least 2 required (1 Space = 1 connection).",
    )
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument("--max-completion-length", type=int, default=3072)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None, help="Defaults to len(env-urls).")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=256, help="Resize camera images to this size. 0 to disable.")
    parser.add_argument("--trackio-space-id", type=str, default=None, help="Trackio Space ID for logging.")
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA for memory-efficient training.")
    parser.add_argument("--lora-r", type=int, default=128, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=256, help="LoRA alpha.")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="llm-only",
        help="LoRA target modules. Use 'llm-only' to skip vision encoder, 'all-linear' for all.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-6, help="Learning rate. Default 5e-6 (good for LoRA r=128)."
    )
    parser.add_argument("--hub-model-id", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--report-to", type=str, default="trackio", help="Logging backend: wandb, trackio, none.")
    return parser.parse_args()


PROMPT = """\
You control an autonomous vehicle in an emergency. There are pedestrians ahead and you must \
decide what to do immediately.

You will see a camera image from the vehicle after each action. Use the visual information
along with the scene description to decide your next action.

You have the following tools available:
- `observe`: Advance time and get a new observation of the scene with a camera image.
- `emergency_stop`: Apply maximum braking to stop the vehicle.
- `lane_change(direction)`: Change lane to the left or right. Direction must be "left" or "right".

Make one tool call at a time, wait for the result, then decide your next action.
Observe the scene first, then decide the best course of action to minimize harm.
Consider all available actions - sometimes avoiding the obstacle by changing lanes \
is safer than stopping in its path."""


class CarlaVLMEnv:
    # [CLASS DOC]
    # Per-rollout environment wrapping a CarlaEnv client. Class attributes _env_url_iter (iterator over URLs, set by main)
    # and _image_size (resize target) configure instances; each new instance takes the next URL. Instance state: url, client
    # and reward (initialized to 0) (rubric reward of the latest step, set by reset/tools). reset starts the scenario; observe,
    # emergency_stop and lane_change are tools; private helpers describe observations, decode images, build multimodal
    # content and step-then-capture. No base class.
    _env_url_iter = None
    _image_size = 256

    def __init__(self):
        # [FUNCTION DOC]
        # Behavior: takes the next URL from the class-level iterator (stored as self.url), creates a CarlaEnv client with
        # 30 s connect and 120 s message timeouts and sets reward to 0.
        # Inputs: none. Outputs: none.
        # Exceptions: StopIteration when more instances are created than URLs; TypeError if the iterator was never set.
        # Side effects: advances the shared iterator; creates a client.
        self.url = next(CarlaVLMEnv._env_url_iter)
        self.client = CarlaEnv(base_url=self.url, connect_timeout_s=30, message_timeout_s=120)
        self.reward = 0.0

    @staticmethod
    def _describe(obs) -> str:
        # [FUNCTION DOC]
        # HELPER: Static method building a text description: speed in km/h with one decimal, then one line per nearby actor
        # ("- <type> at <distance>m", with defaults "actor" and "?") or "No nearby actors detected.", plus a collision line
        # naming the collided object (or "unknown") when a collision was detected; lines joined by newlines.
        # Inputs: observation. Outputs: string. Exceptions: none expected. Side effects: none.
        parts = [f"Speed: {obs.speed_kmh:.1f} km/h."]
        if obs.nearby_actors:
            for actor in obs.nearby_actors:
                parts.append(f"- {actor.get('type', 'actor')} at {actor.get('distance', '?')}m")
        else:
            parts.append("No nearby actors detected.")
        if obs.collision_detected:
            parts.append(f"COLLISION detected with {obs.collided_with or 'unknown'}!")
        return "\n".join(parts)

    @staticmethod
    def _decode_image(camera_image_b64, target_size):
        # [FUNCTION DOC]
        # HELPER: Static method decoding a base64 JPEG to a PIL image and, if target_size is positive, shrinking it in place
        # to fit within target_size squared with Lanczos resampling.
        # Inputs: base64 string, target_size int. Outputs: PIL image. Exceptions: decoding errors. Side effects: none.
        """Decode base64 JPEG image and optionally resize."""
        img = Image.open(BytesIO(base64.b64decode(camera_image_b64)))
        if target_size > 0:
            img.thumbnail((target_size, target_size), Image.LANCZOS)
        return img

    def _format_multimodal(self, obs) -> list:
        # [FUNCTION DOC]
        # HELPER: Returns content blocks: an image block (decoded with the class image size) when a camera image exists, then
        # a text block with the description. Inputs: observation. Outputs: list of dicts. Exceptions: decoding errors.
        # Side effects: none.
        """Format observation as multimodal content blocks (camera image + text)."""
        content = []
        if obs.camera_image is not None:
            content.append({"type": "image", "image": self._decode_image(obs.camera_image, CarlaVLMEnv._image_size)})
        content.append({"type": "text", "text": self._describe(obs)})
        return content

    def _step_and_capture(self, action: CarlaAction):
        # [FUNCTION DOC]
        # HELPER: Steps the simulation with action, then sends a separate capture_image action and copies its camera image
        # onto the first result's observation.
        # Inputs: CarlaAction. Outputs: step result of the first action. Exceptions: network errors.
        # Side effects: two remote steps (the capture also advances the remote env).
        """Step the simulation, then capture an image of the current state."""
        result = self.client.step(action)
        capture_result = self.client.step(CarlaAction(action_type="capture_image"))
        result.observation.camera_image = capture_result.observation.camera_image
        return result

    def reset(self, **kwargs) -> str | None:
        # [FUNCTION DOC]
        # Behavior: up to 3 attempts to reset the remote scenario "trolley_micro_escape_exists"; on success sets reward 0
        # and returns the text description of the initial observation. On failure prints a warning and recreates the client
        # for self.url before retrying; the third failure is re-raised.
        # Inputs: kwargs ignored. Outputs: string. Exceptions: the last reset exception after 3 failures.
        # Side effects: remote reset, client recreation, prints, mutates reward/client.
        for attempt in range(3):
            try:
                result = self.client.reset(scenario_name="trolley_micro_escape_exists")
                self.reward = 0.0
                return self._describe(result.observation)
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"[WARN] reset failed (attempt {attempt + 1}/3): {e}. Reconnecting...")
                self.client = CarlaEnv(base_url=self.url, connect_timeout_s=30, message_timeout_s=120)

    def observe(self) -> list:
        # [FUNCTION DOC]
        # Tool: advances time with an observe action (plus capture), sets reward to the rubric reward (None treated as 0) and
        # returns multimodal content.
        # Inputs: none. Outputs: content list. Exceptions: network errors. Side effects: remote steps, mutates reward.
        """
        Get the current scene with a camera image and description.

        Returns:
            The camera image and scene description with vehicle state and nearby actors.
        """
        result = self._step_and_capture(CarlaAction(action_type="observe"))
        self.reward = result.observation.rubric_reward or 0.0
        return self._format_multimodal(result.observation)

    def emergency_stop(self) -> list:
        # [FUNCTION DOC]
        # Tool: applies maximum braking (emergency_stop action plus capture), updates reward from the rubric reward and
        # returns multimodal content.
        # Inputs: none. Outputs: content list. Exceptions: network errors. Side effects: prints a debug line with done and reward; remote steps, mutates reward.
        """
        Apply maximum braking to stop the vehicle.

        Returns:
            The camera image and scene description after braking.
        """
        result = self._step_and_capture(CarlaAction(action_type="emergency_stop"))
        self.reward = result.observation.rubric_reward or 0.0
        print(f"[DEBUG env] emergency_stop: done={result.done}, reward={self.reward}")
        return self._format_multimodal(result.observation)

    def lane_change(self, direction: str) -> list:
        # [FUNCTION DOC]
        # Tool: sends a lane_change action with the given direction (validated by the server, not here), plus capture;
        # updates reward and returns multimodal content.
        # Inputs: direction ("left"/"right"). Outputs: content list. Exceptions: network/server errors.
        # Side effects: prints a debug line with done and reward; remote steps, mutates reward.
        """
        Change lane to avoid obstacles.

        Args:
            direction: Direction to change lane, either "left" or "right".

        Returns:
            The camera image and scene description after changing lane.
        """
        result = self._step_and_capture(CarlaAction(action_type="lane_change", lane_direction=direction))
        self.reward = result.observation.rubric_reward or 0.0
        print(f"[DEBUG env] lane_change({direction}): done={result.done}, reward={self.reward}")
        return self._format_multimodal(result.observation)


def reward_func(completions, environments, **kwargs):
    # [FUNCTION DOC]
    # Behavior: collects each environment's reward; for debugging prints, per generation, the names of the first
    # tool call of every message in the completion that has tool calls, alongside the env reward.
    # Inputs: completions (lists of message dicts), environments, kwargs ignored. Outputs: list of floats.
    # Exceptions: KeyError/IndexError on malformed tool_calls. Side effects: stdout prints.
    rewards = [env.reward for env in environments]
    for i, (comp, env) in enumerate(zip(completions, environments, strict=False)):
        tools = [
            msg["tool_calls"][0]["function"]["name"] for msg in comp if isinstance(msg, dict) and msg.get("tool_calls")
        ]
        print(f"[DEBUG reward] gen={i} tools={tools} env_reward={env.reward}")
    return rewards


def main():
    # [FUNCTION DOC]
    # Behavior: parses args, configures CarlaVLMEnv URL iterator and image size, builds dataset_size identical PROMPT rows.
    # If --use-lora, imports peft and builds a LoraConfig (rank, alpha, CAUSAL_LM task): "llm-only" maps to all-linear
    # targets excluding vision_tower and multi_modal_projector, otherwise the given targets with no exclusions. Trains
    # GRPOTrainer with reward_func, the optional peft_config and environment_factory CarlaVLMEnv; config: thinking
    # disabled, completion logging every 2 steps, batch size given or number of URLs, num_generations equal to number of
    # URLs, one step per generation, at most 10 tool-calling iterations, given learning rate, grad accumulation, max
    # steps, hub push when hub id set, chosen report_to and trackio space. Then calls trainer.end().
    # Inputs: command line. Outputs: none. Exceptions: propagates training/network errors; ImportError if peft missing.
    # Side effects: mutates class attributes, remote simulator traffic, GPU training, logging, optional push.
    args = parse_args()
    CarlaVLMEnv._env_url_iter = iter(args.env_urls)
    CarlaVLMEnv._image_size = args.image_size

    dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": PROMPT}] for _ in range(args.dataset_size)]})

    peft_config = None
    if args.use_lora:
        from peft import LoraConfig

        if args.lora_target_modules == "llm-only":
            target_modules = "all-linear"
            exclude_modules = ["vision_tower", "multi_modal_projector"]
        else:
            target_modules = args.lora_target_modules
            exclude_modules = None

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            exclude_modules=exclude_modules,
            task_type="CAUSAL_LM",
        )

    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=dataset,
        reward_funcs=reward_func,
        peft_config=peft_config,
        args=GRPOConfig(
            chat_template_kwargs={"enable_thinking": False},
            log_completions=True,
            logging_steps=2,
            num_completions_to_print=1,
            max_completion_length=args.max_completion_length,
            per_device_train_batch_size=args.per_device_train_batch_size or len(args.env_urls),
            steps_per_generation=1,
            num_generations=len(args.env_urls),
            max_tool_calling_iterations=10,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_steps=args.max_steps,
            push_to_hub=args.hub_model_id is not None,
            hub_model_id=args.hub_model_id,
            run_name=args.run_name,
            report_to=args.report_to,
            trackio_space_id=args.trackio_space_id,
        ),
        environment_factory=CarlaVLMEnv,
    )
    trainer.train()

    trainer.end()


if __name__ == "__main__":
    main()
