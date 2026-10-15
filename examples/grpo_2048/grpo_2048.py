# [MODULE DOC]
# Example script: GRPO with tool-calling environments where Qwen3-4B (LoRA) plays 2048 through a "move" tool; reward
# is the final game score. Main contents: PROMPT, Game2048Env (environment class passed as environment_factory),
# reward_score, main. Not part of the trl package. Import-time side effects: none.

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
#     "trl[peft]",
# ]
# ///

"""
Run:
python examples/grpo_2048/grpo_2048.py
"""

import random

from datasets import Dataset
from peft import LoraConfig

from trl import GRPOConfig, GRPOTrainer


PROMPT = "Play 2048 on a 4x4 board. Use the tool `move` with one of: up, down, left, right. Maximize the score."


class Game2048Env:
    # [CLASS DOC]
    # Pure-Python 4x4 2048 game used as a GRPOTrainer environment: one instance per rollout. Public methods: reset
    # (called by the trainer to start an episode and give the initial observation) and move (exposed to the model as a
    # tool via its docstring). State: board (4x4 list of ints, 0 empty), score (float, sum of merged tile values), done.
    # Private helpers handle tile spawning, line merging, applying a move, game-over detection and rendering.
    # No base class; constructed with no arguments.
    def reset(self, **kwargs) -> str:
        # [FUNCTION DOC]
        # Behavior: clears the board, zeroes score, sets done False, spawns two tiles and returns the observation text
        # "score=<score>", the rendered board, and "done=<done>" on separate lines.
        # Inputs: arbitrary keyword args (ignored, the trainer passes dataset columns). Outputs: observation string.
        # Exceptions: none. Side effects: resets instance state; uses the global random module.
        self.board = [[0] * 4 for _ in range(4)]
        self.score = 0.0
        self.done = False
        self._spawn()
        self._spawn()
        return f"score={self.score}\n{self._render()}\ndone={self.done}"

    def move(self, direction: str) -> str:
        # [FUNCTION DOC]
        # Behavior: tool call. Raises if the game is over; otherwise normalizes the direction (strip, lowercase) and applies
        # it; if any tile moved, adds gained points to score and spawns a tile. Then sets done when no move is possible and
        # returns the same observation format as reset. Invalid directions are a silent no-op.
        # Inputs: direction string. Outputs: observation string.
        # Exceptions: ValueError("Game over.") when done. Side effects: mutates board/score/done; random tile spawn.
        """
        Play one move in 2048.

        Args:
            direction: One of "up", "down", "left", "right".

        Returns:
            Environment feedback after the move.
        """
        if self.done:
            raise ValueError("Game over.")
        moved, gained = self._apply_move(direction.strip().lower())
        if moved:
            self.score += gained
            self._spawn()
        self.done = not self._can_move()
        return f"score={self.score}\n{self._render()}\ndone={self.done}"

    def _spawn(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Places a new tile in a uniformly random empty cell: 4 with probability 0.1, else 2; no-op if the board is
        # full. Inputs: none. Outputs: none. Exceptions: none. Side effects: mutates board; uses global random.
        empty = [(r, c) for r in range(4) for c in range(4) if self.board[r][c] == 0]
        if not empty:
            return
        r, c = random.choice(empty)
        self.board[r][c] = 4 if random.random() < 0.1 else 2

    @staticmethod
    def _merge_line(line: list[int]) -> tuple[list[int], int]:
        # [FUNCTION DOC]
        # HELPER: Static method sliding and merging one line toward index 0 using 2048 rules.
        # Behavior: drops zeros; walks values, merging each equal adjacent pair once into double its value (adding it to the
        # gain) and skipping both; pads with zeros to length 4.
        # Inputs: line of 4 ints. Outputs: (new line, points gained). Exceptions: none. Side effects: none.
        vals = [x for x in line if x]
        out = []
        gained = 0
        i = 0
        while i < len(vals):
            if i + 1 < len(vals) and vals[i] == vals[i + 1]:
                v = vals[i] * 2
                out.append(v)
                gained += v
                i += 2
            else:
                out.append(vals[i])
                i += 1
        out += [0] * (4 - len(out))
        return out, gained

    def _apply_move(self, direction: str) -> tuple[bool, int]:
        # [FUNCTION DOC]
        # HELPER: Applies a direction to the whole board.
        # Behavior: returns (False, 0) for an unknown direction. For left/right processes each row (reversing before and after
        # merging for right); for up/down processes each column likewise (reversing for down). Compares with a copy of the
        # previous board to determine whether anything moved.
        # Inputs: normalized direction. Outputs: (moved bool, total gained points).
        # Exceptions: none. Side effects: mutates board.
        if direction not in {"up", "down", "left", "right"}:
            return False, 0

        before = [row[:] for row in self.board]
        gained_total = 0

        if direction in {"left", "right"}:
            for r in range(4):
                row = self.board[r][:]
                if direction == "right":
                    row.reverse()
                merged, gained = self._merge_line(row)
                if direction == "right":
                    merged.reverse()
                self.board[r] = merged
                gained_total += gained
        else:
            for c in range(4):
                col = [self.board[r][c] for r in range(4)]
                if direction == "down":
                    col.reverse()
                merged, gained = self._merge_line(col)
                if direction == "down":
                    merged.reverse()
                for r in range(4):
                    self.board[r][c] = merged[r]
                gained_total += gained

        moved = self.board != before
        return moved, gained_total

    def _can_move(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: Returns True if any cell is empty or any horizontally/vertically adjacent pair is equal, else False.
        # Inputs: none. Outputs: bool. Exceptions: none. Side effects: none.
        if any(0 in row for row in self.board):
            return True
        for r in range(4):
            for c in range(4):
                if r + 1 < 4 and self.board[r][c] == self.board[r + 1][c]:
                    return True
                if c + 1 < 4 and self.board[r][c] == self.board[r][c + 1]:
                    return True
        return False

    def _render(self) -> str:
        # [FUNCTION DOC]
        # HELPER: Renders the board as 4 lines of space-separated values each right-aligned to width 3.
        # Inputs: none. Outputs: string. Exceptions: none. Side effects: none.
        return "\n".join(" ".join(f"{v:3d}" for v in row) for row in self.board)


def reward_score(environments, **kwargs):
    # [FUNCTION DOC]
    # Behavior: reward function returning each environment's current score.
    # Inputs: environments (list of Game2048Env), other kwargs ignored. Outputs: list of floats.
    # Exceptions: none. Side effects: none.
    return [env.score for env in environments]


def main() -> None:
    # [FUNCTION DOC]
    # Behavior: builds a 1000-row dataset with the same single-user-message PROMPT; trains GRPOTrainer on Qwen/Qwen3-4B
    # with reward_score, thinking disabled, logging every step with 2 printed completions, trackio (space trl-2048),
    # completion length 2048, batch 4, grad accumulation 2, environment_factory Game2048Env and default LoraConfig;
    # then calls trainer.end().
    # Inputs: none. Outputs: none. Exceptions: propagates training errors.
    # Side effects: model download, GPU training, trackio logging, output files.
    dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": PROMPT}] for _ in range(1000)]})

    trainer = GRPOTrainer(
        model="Qwen/Qwen3-4B",
        train_dataset=dataset,
        reward_funcs=reward_score,
        args=GRPOConfig(
            chat_template_kwargs={"enable_thinking": False},
            logging_steps=1,
            log_completions=True,
            num_completions_to_print=2,
            report_to="trackio",
            trackio_space_id="trl-2048",
            max_completion_length=2048,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
        ),
        environment_factory=Game2048Env,
        peft_config=LoraConfig(),
    )
    trainer.train()

    trainer.end()


if __name__ == "__main__":
    main()
