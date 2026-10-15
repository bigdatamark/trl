# [MODULE DOC]
# Example script: GRPO training of a VLM (default Qwen3.5-0.8B) on OpenEnv's CARLA driving simulator (trolley-problem
# style scenario "trolley_micro_escape_exists"); each tool call returns a camera image plus a text scene description.
# Each CARLA Space supports one connection, so one environment URL is needed per concurrent rollout.
# Main contents: parse_args, PROMPT, CarlaGRPOEnv, reward_func, main. Import-time side effects: none beyond imports.

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
GRPO training with OpenEnv's CARLA environment for VLMs (Vision Language Models).

VLM adaptation of `grpo_carla.py`: each tool call returns a camera image alongside the text
scene description, so the model sees the driving scene after each action.

Setup:

```sh
uv pip install git+https://huggingface.co/spaces/sergiopaniego/carla-env
```

Usage (requires at least 2 CARLA Spaces, each supports only 1 concurrent connection):

```sh
python examples/grpo_carla/carla_vlm.py \
    --model Qwen/Qwen3.5-0.8B \
    --env-urls https://server1.hf.space https://server2.hf.space
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
    # Behavior: argparse for model, one or more env URLs, image resize size (256, 0 disables), completion length (1024),
    # grad accumulation (16), training steps (50), trackio space id, optional hub model id and run name.
    # Inputs: command line. Outputs: Namespace. Exceptions: SystemExit on bad args. Side effects: may print usage.
    parser = argparse.ArgumentParser(description="Run GRPO VLM training with CARLA environment.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-0.8B",
        help="Model to use for training.",
    )
    parser.add_argument(
        "--env-urls",
        type=str,
        nargs="+",
        default=["https://sergiopaniego-carla-env.hf.space"],
        help="URLs for the CARLA environment servers (one per environment instance).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Resize camera images to this size. 0 to disable.",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=1024,
        help="Maximum number of tokens in the generated completion.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
        help="Number of steps to accumulate gradients over before updating.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Number of training steps to run.",
    )
    parser.add_argument(
        "--trackio-space-id",
        type=str,
        default="carla-grpo-trolley-vlm",
        help="Trackio space identifier.",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="Hub model ID to push the trained model to.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for tracking.",
    )
    return parser.parse_args()


PROMPT = """You control an autonomous vehicle in an emergency. There are pedestrians ahead and you must \
decide what to do immediately.

You will see a camera image from the vehicle after each action. Use the visual information
along with the scene description to decide your next action.

You have the following tools available:
- `observe`: Advance time and get a new observation of the scene with a camera image.
- `emergency_stop`: Apply maximum braking to stop the vehicle.
- `lane_change(direction)`: Change lane to the left or right. Direction must be "left" or "right".

Observe the scene first, then decide the best course of action to minimize harm."""


class CarlaGRPOEnv:
    # [CLASS DOC]
    # Per-rollout environment wrapping a CarlaEnv client. Class attributes _env_url_iter (iterator over URLs, set by main)
    # and _image_size (resize target) configure instances; each new instance takes the next URL. Instance state: client
    # and reward (rubric reward of the latest step, set by reset/tools). reset starts the scenario; observe,
    # emergency_stop and lane_change are tools; private helpers describe observations, decode images, build multimodal
    # content and step-then-capture. No base class.
    _env_url_iter = None
    _image_size = 256

    def __init__(self):
        # [FUNCTION DOC]
        # Behavior: takes the next URL from the class-level iterator and creates a CarlaEnv client with 30 s connect and
        # 120 s message timeouts.
        # Inputs: none. Outputs: none.
        # Exceptions: StopIteration when more instances are created than URLs; TypeError if the iterator was never set.
        # Side effects: advances the shared iterator; creates a client.
        url = next(CarlaGRPOEnv._env_url_iter)
        self.client = CarlaEnv(base_url=url, connect_timeout_s=30, message_timeout_s=120)

    @staticmethod
    def _describe(obs) -> str:
        # [FUNCTION DOC]
        # HELPER: Static method building a text description: speed in km/h with one decimal, then one line per nearby actor
        # ("- <type> at <distance>m", with defaults "actor" and "?") or "No nearby actors detected.", plus a collision line
        # naming the collided object (or "unknown") when a collision was detected; lines joined by newlines.
        # Inputs: observation. Outputs: string. Exceptions: none expected. Side effects: none.
        """Build a text description from the observation fields."""
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
            content.append({"type": "image", "image": self._decode_image(obs.camera_image, CarlaGRPOEnv._image_size)})
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
        # Behavior: resets the remote scenario "trolley_micro_escape_exists", sets reward to 0 and returns the text
        # description of the initial observation (no image).
        # Inputs: kwargs ignored. Outputs: string. Exceptions: network errors. Side effects: remote reset, mutates reward.
        result = self.client.reset(scenario_name="trolley_micro_escape_exists")
        self.reward = 0.0
        return self._describe(result.observation)

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
        # Inputs: none. Outputs: content list. Exceptions: network errors. Side effects: remote steps, mutates reward.
        """
        Apply maximum braking to stop the vehicle.

        Returns:
            The camera image and scene description after braking.
        """
        result = self._step_and_capture(CarlaAction(action_type="emergency_stop"))
        self.reward = result.observation.rubric_reward or 0.0
        return self._format_multimodal(result.observation)

    def lane_change(self, direction: str) -> list:
        # [FUNCTION DOC]
        # Tool: sends a lane_change action with the given direction (validated by the server, not here), plus capture;
        # updates reward and returns multimodal content.
        # Inputs: direction ("left"/"right"). Outputs: content list. Exceptions: network/server errors.
        # Side effects: remote steps, mutates reward.
        """
        Change lane to avoid obstacles.

        Args:
            direction: Direction to change lane, either "left" or "right".

        Returns:
            The camera image and scene description after changing lane.
        """
        result = self._step_and_capture(CarlaAction(action_type="lane_change", lane_direction=direction))
        self.reward = result.observation.rubric_reward or 0.0
        return self._format_multimodal(result.observation)


def reward_func(environments, **kwargs):
    # [FUNCTION DOC]
    # Behavior: returns each environment's latest reward. Inputs: environments, kwargs ignored. Outputs: list of floats.
    # Exceptions: AttributeError if reset was never called. Side effects: none.
    return [environment.reward for environment in environments]


def main():
    # [FUNCTION DOC]
    # Behavior: parses args, configures CarlaGRPOEnv's URL iterator and image size, builds 1000 identical PROMPT rows and
    # trains GRPOTrainer with reward_func and environment_factory CarlaGRPOEnv; per-device batch and num_generations both
    # equal the number of env URLs, one step per generation, thinking disabled, completion logging, logging every 2 steps,
    # hub push when a hub model id is given, trackio reporting; then calls trainer.end().
    # Inputs: command line. Outputs: none. Exceptions: propagates training/network errors.
    # Side effects: mutates CarlaGRPOEnv class attributes, remote simulator traffic, GPU training, logging, optional push.
    args = parse_args()
    CarlaGRPOEnv._env_url_iter = iter(args.env_urls)
    CarlaGRPOEnv._image_size = args.image_size

    dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": PROMPT}] for _ in range(1000)]})

    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=dataset,
        reward_funcs=reward_func,
        args=GRPOConfig(
            chat_template_kwargs={"enable_thinking": False},
            log_completions=True,
            logging_steps=2,
            num_completions_to_print=1,
            max_completion_length=args.max_completion_length,
            per_device_train_batch_size=len(args.env_urls),
            steps_per_generation=1,
            num_generations=len(args.env_urls),
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_steps=args.max_steps,
            push_to_hub=args.hub_model_id is not None,
            hub_model_id=args.hub_model_id,
            run_name=args.run_name,
            report_to="trackio",
            trackio_space_id=args.trackio_space_id,
        ),
        environment_factory=CarlaGRPOEnv,
    )
    trainer.train()

    trainer.end()


if __name__ == "__main__":
    main()
