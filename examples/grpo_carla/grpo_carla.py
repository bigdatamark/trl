# [MODULE DOC]
# Example script: text-only GRPO on OpenEnv's CARLA emergency-driving scenario (default Qwen3-0.6B): the model must
# observe and pick an action (brake or change lane) to minimize harm; tool responses are text scene descriptions.
# One environment URL per concurrent rollout. Main contents: parse_args, PROMPT, CarlaGRPOEnv, reward_func, main.
# Import-time side effects: none beyond imports.

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
Simple script to run GRPO training with OpenEnv's CARLA environment. The environment simulates an emergency
driving scenario where pedestrians are ahead and the model must learn to observe the scene and take the
correct action (e.g., swerve to an empty lane) to minimize casualties.

Setup (Option A - Install from HF Space, recommended):

```sh
uv pip install git+https://huggingface.co/spaces/sergiopaniego/carla-env
```

Setup (Option B - Clone OpenEnv repo, for development):

```sh
git clone https://github.com/huggingface/OpenEnv.git
cd OpenEnv/envs/carla_env
uv pip install -e .
```

Usage:

```sh
python examples/grpo_carla/grpo_carla.py
python examples/grpo_carla/grpo_carla.py --model Qwen/Qwen3-1.7B --env-urls https://server1.hf.space https://server2.hf.space
```
"""

import argparse

from carla_env import CarlaAction, CarlaEnv
from datasets import Dataset

from trl import GRPOConfig, GRPOTrainer


def parse_args():
    # [FUNCTION DOC]
    # Behavior: argparse for model (Qwen/Qwen3-0.6B), env URLs (default the public Space), trackio space id
    # (carla-grpo-trolley), optional hub model id and run name.
    # Inputs: command line. Outputs: Namespace. Exceptions: SystemExit on bad args. Side effects: may print usage.
    parser = argparse.ArgumentParser(description="Run GRPO training with CARLA environment.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
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
        "--trackio-space-id",
        type=str,
        default="carla-grpo-trolley",
        help="Trackio space identifier.",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="Hub model ID to push the trained model to (e.g., sergiopaniego/Qwen3-0.6B-carla-trolley-escape).",
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

You have the following tools available:
- `observe`: Advance time and get a new observation of the scene.
- `emergency_stop`: Apply maximum braking to stop the vehicle.
- `lane_change(direction)`: Change lane to the left or right. Direction must be "left" or "right".

Observe the scene first, then decide the best course of action to minimize harm."""


class CarlaGRPOEnv:
    # [CLASS DOC]
    # Per-rollout environment wrapping a CarlaEnv client. Class attribute _env_url_iter (set by main) supplies one URL
    # per instance. Instance state: client and reward (latest rubric reward, set by reset and tools). reset starts the
    # scenario; observe, emergency_stop and lane_change are tools returning text built by _describe. No base class.
    _env_url_iter = None

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
        parts = []
        parts.append(f"Speed: {obs.speed_kmh:.1f} km/h.")
        if obs.nearby_actors:
            for actor in obs.nearby_actors:
                parts.append(f"- {actor.get('type', 'actor')} at {actor.get('distance', '?')}m")
        else:
            parts.append("No nearby actors detected.")
        if obs.collision_detected:
            parts.append(f"COLLISION detected with {obs.collided_with or 'unknown'}!")
        return "\n".join(parts)

    def reset(self, **kwargs) -> str | None:
        # [FUNCTION DOC]
        # Behavior: resets the remote scenario "trolley_micro_escape_exists", sets reward to 0 and returns the text
        # description of the initial observation.
        # Inputs: kwargs ignored. Outputs: string. Exceptions: network errors. Side effects: remote reset, mutates reward.
        result = self.client.reset(scenario_name="trolley_micro_escape_exists")
        self.reward = 0.0
        return self._describe(result.observation)

    def observe(self) -> str:
        # [FUNCTION DOC]
        # Tool: steps with an observe action, sets reward to the rubric reward (None treated as 0) and returns the text
        # description. Inputs: none. Outputs: string. Exceptions: network errors. Side effects: remote step, mutates reward.
        """
        Get the current scene description without taking any action.

        Returns:
            The scene description with vehicle state and nearby actors.
        """
        result = self.client.step(CarlaAction(action_type="observe"))
        self.reward = result.observation.rubric_reward or 0.0
        return self._describe(result.observation)

    def emergency_stop(self) -> str:
        # [FUNCTION DOC]
        # Tool: steps with an emergency_stop action, updates reward from the rubric reward and returns the description.
        # Inputs: none. Outputs: string. Exceptions: network errors. Side effects: remote step, mutates reward.
        """
        Apply maximum braking to stop the vehicle.

        Returns:
            The scene description after braking.
        """
        result = self.client.step(CarlaAction(action_type="emergency_stop"))
        self.reward = result.observation.rubric_reward or 0.0
        return self._describe(result.observation)

    def lane_change(self, direction: str) -> str:
        # [FUNCTION DOC]
        # Tool: steps with a lane_change action in the given direction (validated server-side), updates reward and returns
        # the description. Inputs: direction. Outputs: string. Exceptions: network/server errors.
        # Side effects: remote step, mutates reward.
        """
        Change lane to avoid obstacles.

        Args:
            direction: Direction to change lane, either "left" or "right".

        Returns:
            The scene description after changing lane.
        """
        result = self.client.step(CarlaAction(action_type="lane_change", lane_direction=direction))
        self.reward = result.observation.rubric_reward or 0.0
        return self._describe(result.observation)


def reward_func(environments, **kwargs):
    # [FUNCTION DOC]
    # Behavior: returns each environment's latest reward. Inputs: environments, kwargs ignored. Outputs: list of floats.
    # Exceptions: AttributeError if reset was never called. Side effects: none.
    return [environment.reward for environment in environments]


def main():
    # [FUNCTION DOC]
    # Behavior: parses args, sets the URL iterator, builds 1000 identical PROMPT rows and trains GRPOTrainer with reward_func
    # and environment_factory CarlaGRPOEnv; thinking disabled, completion logging every 2 steps, completion length 1024,
    # batch size and num_generations equal to number of URLs, one step per generation, grad accumulation 16, 50 steps,
    # hub push when an id is given, trackio. Then calls trainer.end().
    # Inputs: command line. Outputs: none. Exceptions: propagates training/network errors.
    # Side effects: mutates class attribute, remote simulator traffic, GPU training, logging, optional push.
    args = parse_args()
    CarlaGRPOEnv._env_url_iter = iter(args.env_urls)

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
            max_completion_length=1024,
            per_device_train_batch_size=len(args.env_urls),
            steps_per_generation=1,
            num_generations=len(args.env_urls),
            gradient_accumulation_steps=16,
            max_steps=50,
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
