# [MODULE DOC]
# Example script: multi-environment GRPO in a single run (Qwen3-1.7B): Wordle (TextArena) and Catch (OpenSpiel),
# both on HF Spaces, wrapped in one environment_factory class. A dataset "env" column routes each sample to its game
# at reset time; per-game reward functions return None for samples of the other game.
# Main contents: wordle_prompt, catch_prompt, default URLs, grid constants (10 rows, 5 columns), _format_catch_obs,
# MultiEnv, wordle_reward, catch_reward, main. Import-time side effects: none beyond imports.

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
#     "openenv-textarena @ git+https://huggingface.co/spaces/openenv/wordle",
#     "openenv-openspiel-env @ git+https://huggingface.co/spaces/openenv/openspiel_env",
# ]
# ///

"""
Multi-environment GRPO training with OpenEnv: Wordle + Catch in the same training run.

Demonstrates how to wrap multiple environments in a single `environment_factory` class. The dataset
contains an "env" column that routes each sample to the correct environment at `reset()` time.

Usage:
    python examples/grpo_multi_env/grpo_multi_env.py \\
        --wordle-url https://openenv-wordle.hf.space \\
        --catch-url https://openenv-openspiel-env.hf.space
"""

import argparse

from datasets import Dataset
from openspiel_env import OpenSpielEnv
from openspiel_env.models import OpenSpielAction
from textarena_env import TextArenaAction, TextArenaEnv

from trl import GRPOConfig, GRPOTrainer


wordle_prompt = """You are an expert Wordle solver with deep knowledge of English vocabulary, letter frequency patterns, and optimal guessing strategies.

Follow these rules to play Wordle:

1. The target is a 5-letter English word
2. You have 6 attempts to guess the correct word
3. After each guess, you receive color-coded feedback:
   - GREEN (G): Letter is correct and in the correct position
   - YELLOW (Y): Letter is in the word but in the wrong position
   - GRAY (X): Letter is not in the word at all
4. All guesses must be valid 5-letter English words
5. You cannot reuse a word you've already guessed
6. Use the tool `guess` to make a guess.
"""

catch_prompt = """You are an AI agent playing the game **Catch**.

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

DEFAULT_WORDLE_URL = "https://openenv-wordle.hf.space"
DEFAULT_CATCH_URL = "https://openenv-openspiel-env.hf.space"

CATCH_ROWS = 10
CATCH_COLS = 5


def _format_catch_obs(info_state: list[float]) -> str:
    # [FUNCTION DOC]
    # HELPER: Describes the flattened 10x5 Catch grid (row-major): cells equal to 1.0 above the last row are the ball
    # (1-based row/column), cells in the last row are the paddle (1-based column). Emits ball position, paddle column
    # and the ball's relative horizontal offset (LEFT/RIGHT columns or directly above), newline-joined.
    # Inputs: info_state floats. Outputs: string (possibly empty).
    # KNOWN ISSUE: when the ball reaches the last row it is treated as the paddle and the ball line is omitted.
    # Exceptions: none. Side effects: none.
    """Convert the flat 50-float observation into a readable text description."""
    ball_row = ball_col = paddle_col = None
    for idx, val in enumerate(info_state):
        if val == 1.0:
            r, c = divmod(idx, CATCH_COLS)
            if r < CATCH_ROWS - 1:
                ball_row, ball_col = r + 1, c + 1
            else:
                paddle_col = c + 1
    parts = []
    if ball_row is not None and ball_col is not None:
        parts.append(f"Ball: row {ball_row}/{CATCH_ROWS}, column {ball_col}/{CATCH_COLS}")
    if paddle_col is not None:
        parts.append(f"Paddle: column {paddle_col}/{CATCH_COLS}")
    if ball_col is not None and paddle_col is not None:
        diff = ball_col - paddle_col
        if diff < 0:
            parts.append(f"The ball is {abs(diff)} column(s) to the LEFT of the paddle.")
        elif diff > 0:
            parts.append(f"The ball is {diff} column(s) to the RIGHT of the paddle.")
        else:
            parts.append("The ball is directly above the paddle.")
    return "\n".join(parts)


class MultiEnv:
    # [CLASS DOC]
    # Per-rollout environment that hosts either a Wordle or a Catch session. Class attributes wordle_url and catch_url
    # (set by main) choose servers. State: lazily created _wordle_client/_catch_client, active game name, reward, done and
    # (Wordle) _last_full_feedback for diffing the cumulative transcript. Tools: guess (Wordle only), move and stay
    # (Catch only), with _catch_action shared by the Catch tools. No base class.
    wordle_url = DEFAULT_WORDLE_URL
    catch_url = DEFAULT_CATCH_URL

    def __init__(self):
        # [FUNCTION DOC]
        # Behavior: sets both clients and active to None, reward 0 and done False (no network).
        # Inputs: none. Outputs: none. Exceptions: none. Side effects: none.
        self._wordle_client = None
        self._catch_client = None
        self.active = None
        self.reward = 0.0
        self.done = False

    def reset(self, **kwargs) -> str | None:
        # [FUNCTION DOC]
        # Behavior: reads the "env" kwarg (default "wordle") into active and clears reward/done. For wordle: closes any
        # previous client (ignoring errors), creates a new TextArenaEnv, resets it, stores and returns the first message
        # content as the initial feedback. For catch: closes any previous OpenSpiel client similarly, creates a new one,
        # resets it, stores done and returns the formatted grid.
        # Inputs: dataset row kwargs including "env". Outputs: observation string.
        # Exceptions: ValueError for unknown env names; network errors. Side effects: creates/closes clients, remote resets.
        self.active = kwargs.get("env", "wordle")
        self.reward = 0.0
        self.done = False

        if self.active == "wordle":
            if self._wordle_client is not None:
                try:
                    self._wordle_client.close()
                except Exception:
                    pass
            self._wordle_client = TextArenaEnv(base_url=MultiEnv.wordle_url)
            result = self._wordle_client.reset()
            self._last_full_feedback = result.observation.messages[0].content
            self.reward = 0.0
            return self._last_full_feedback
        elif self.active == "catch":
            if self._catch_client is not None:
                try:
                    self._catch_client.close()
                except Exception:
                    pass
            self._catch_client = OpenSpielEnv(base_url=MultiEnv.catch_url)
            result = self._catch_client.reset()
            self.done = result.observation.done
            return _format_catch_obs(result.observation.info_state)
        else:
            raise ValueError(f"Unknown environment: {self.active}")

    def guess(self, guess: str) -> str:
        # [FUNCTION DOC]
        # Tool: Wordle guess. Raises unless active is wordle and the game is not done. Steps with the guess, takes the new
        # cumulative first-message content, returns only the suffix beyond the previous transcript and updates it. Reward is
        # 0 if the new feedback reports an invalid move, else the step reward; done is updated.
        # Inputs: guess string (expected bracketed, e.g. "[crane]"). Outputs: new feedback text.
        # Exceptions: ValueError for wrong game or finished game; network errors. Side effects: remote step, mutates state.
        """
        Make a guess in the Wordle environment.

        Args:
            guess: The guessed word, formatted as '[abcde]'

        Returns:
            The feedback message from the environment.
        """
        if self.active != "wordle":
            raise ValueError("guess is only available in Wordle")
        if self.done:
            raise ValueError("Game over.")
        result = self._wordle_client.step(TextArenaAction(message=guess))
        _full_feedback = result.observation.messages[0].content
        feedback = _full_feedback[len(self._last_full_feedback) :]
        self._last_full_feedback = _full_feedback
        if "You attempted an invalid move" in feedback:
            self.reward = 0.0
        else:
            self.reward = result.reward
        self.done = result.done
        return feedback

    def _catch_action(self, action_id: int) -> str:
        # [FUNCTION DOC]
        # HELPER: Raises if done; steps the Catch client with the action id for game "catch"; stores reward (None as 0) and
        # done; returns the formatted grid. Inputs: action_id (0 left, 1 stay, 2 right). Outputs: string.
        # Exceptions: ValueError("Episode is done."); network errors. Side effects: remote step, mutates reward/done.
        if self.done:
            raise ValueError("Episode is done.")
        result = self._catch_client.step(OpenSpielAction(action_id=action_id, game_name="catch"))
        self.reward = result.reward or 0.0
        self.done = result.observation.done
        return _format_catch_obs(result.observation.info_state)

    def move(self, direction: str) -> str:
        # [FUNCTION DOC]
        # Tool: Catch only; maps "left" to 0 and "right" to 2 and performs it.
        # Inputs: direction. Outputs: observation string.
        # Exceptions: ValueError when not in Catch, for invalid direction, or when done. Side effects: remote step.
        """Move the paddle left or right.

        Args:
            direction: Direction to move, either "left" or "right".

        Returns:
            The observation after moving.
        """
        if self.active != "catch":
            raise ValueError("move is only available in Catch")
        if direction == "left":
            action_id = 0
        elif direction == "right":
            action_id = 2
        else:
            raise ValueError(f"Invalid direction {direction!r}: must be 'left' or 'right'.")
        return self._catch_action(action_id)

    def stay(self) -> str:
        # [FUNCTION DOC]
        # Tool: Catch only; performs action 1. Inputs: none. Outputs: observation string.
        # Exceptions: ValueError when not in Catch or when done. Side effects: remote step.
        """Do nothing and let the ball fall one step.

        Returns:
            The observation after staying.
        """
        if self.active != "catch":
            raise ValueError("stay is only available in Catch")
        return self._catch_action(1)


def wordle_reward(environments, **kwargs) -> list[float | None]:
    # [FUNCTION DOC]
    # Behavior: per environment, its reward when active is wordle, else None (excluded from this reward).
    # Inputs: environments, kwargs ignored. Outputs: list of float or None. Exceptions: none. Side effects: none.
    return [env.reward if env.active == "wordle" else None for env in environments]


def catch_reward(environments, **kwargs) -> list[float | None]:
    # [FUNCTION DOC]
    # Behavior: per environment, None if not Catch; if done, max(reward, 0) (catch 1, miss 0); otherwise 0.0.
    # Inputs: environments, kwargs ignored. Outputs: list of float or None. Exceptions: none. Side effects: none.
    rewards = []
    for env in environments:
        if env.active != "catch":
            rewards.append(None)
        elif env.done:
            # Catch gives +1 for catching, -1 for missing. Clamp to [0, 1] for GRPO advantage estimation.
            rewards.append(max(env.reward, 0.0))
        else:
            rewards.append(0.0)  # Incomplete episode
    return rewards


def main() -> None:
    # [FUNCTION DOC]
    # Behavior: parses known args --wordle-url and --catch-url (unknown args ignored), sets MultiEnv class URLs, builds a
    # dataset with 500 Wordle and 500 Catch prompts plus the "env" routing column, and trains GRPOTrainer on
    # Qwen/Qwen3-1.7B with both reward functions and environment_factory MultiEnv (wandb reporting, completion logging,
    # log every step, thinking disabled, completion length 1024); calls trainer.end().
    # Inputs: command line. Outputs: none. Exceptions: propagates network/training errors.
    # Side effects: mutates class attributes, remote environment traffic, GPU training, wandb logging, output files.
    parser = argparse.ArgumentParser(description="Multi-environment GRPO training")
    parser.add_argument("--wordle-url", default=DEFAULT_WORDLE_URL, help="Wordle environment URL")
    parser.add_argument("--catch-url", default=DEFAULT_CATCH_URL, help="Catch environment URL")
    args, remaining = parser.parse_known_args()

    MultiEnv.wordle_url = args.wordle_url
    MultiEnv.catch_url = args.catch_url

    n = 500  # samples per environment
    dataset = Dataset.from_dict(
        {
            "prompt": (
                [[{"role": "user", "content": wordle_prompt}]] * n + [[{"role": "user", "content": catch_prompt}]] * n
            ),
            "env": ["wordle"] * n + ["catch"] * n,
        }
    )

    trainer = GRPOTrainer(
        model="Qwen/Qwen3-1.7B",
        reward_funcs=[wordle_reward, catch_reward],
        train_dataset=dataset,
        args=GRPOConfig(
            report_to="wandb",
            log_completions=True,
            num_completions_to_print=2,
            logging_steps=1,
            chat_template_kwargs={"enable_thinking": False},
            max_completion_length=1024,
        ),
        environment_factory=MultiEnv,
    )
    trainer.train()

    trainer.end()


if __name__ == "__main__":
    main()
