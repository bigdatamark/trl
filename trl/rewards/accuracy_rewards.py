# [MODULE DOC]
# Correctness-based math reward functions built on math_verify: accuracy_reward (whole completion),
# reasoning_accuracy_reward (only the text after a reasoning delimiter), and get_cosine_scaled_reward
# (correctness combined with a cosine length schedule). Part of trl.rewards.
# Import-time side effects: imports latex2sympy2_extended and math_verify only if math_verify is installed;
# the reward functions raise ImportError at call time otherwise.

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

import logging
import math
import threading
from collections.abc import Callable

from ..import_utils import is_math_verify_available


if is_math_verify_available():
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify


def accuracy_reward(
    completions: list[list[dict[str, str]]],
    solution: list[str],
    log_extra: Callable[[str, list], None] | None = None,
    **kwargs,
) -> list[float | None]:
    # [FUNCTION DOC]
    # Behavior: requires math_verify. Takes the "content" of the first message of each completion. Timeouts:
    # parsing 10s and verification 5s in the main thread; disabled (None) in other threads because math_verify
    # relies on SIGALRM, in which case the math_verify.parser/grader loggers are raised to ERROR. For each
    # (content, solution) pair: parse the gold solution; if parsing yields nothing, reward None (skip example).
    # Otherwise parse the completion with a LaTeX extraction config (units normalization, boxed answers tried
    # first, no anchor-free extraction, first match) and reward 1.0/0.0 from verify(gold, answer).
    # Records string forms of gold and parsed answer ("[unparseable]" / "[skipped]" placeholders).
    # Inputs: completions; solution (list of str, same length - strict zip); log_extra optional callback
    # (column name, values); kwargs ignored.
    # Outputs: list of float or None per example.
    # Exceptions: ImportError without math_verify; ValueError if lengths differ.
    # Side effects: when log_extra is given, logs columns "solution", "gold_parsed", "answer_parsed"; may change
    # math_verify logger levels globally (non-main thread).
    r"""
    Reward function that checks if the completion matches the ground truth.
        - If both gold and prediction are parseable → use math verification.
        - If gold is not parseable → return `None` to skip the example.

    Args:
        completions (`list[list[dict[str, str]]]`):
            List of completions to be evaluated. Each completion must be a list of one message, i.e. a dictionary
            containing the key `"content"` with the value being the text of the completion.
        solution (`list[str]`):
            List of the raw-text solutions to the questions/problems/prompts.
        log_extra (`callable`, *optional*):
            Callable to log extra columns to the completions table, provided automatically by the trainer. Defaults to
            `None` to allow calling the function directly outside of a trainer (e.g., for testing).
        **kwargs:
            Additional keyword arguments. This function does not use them, but they are required in the function
            signature to ensure compatibility with trainers like [`GRPOTrainer`].
    Example:
    ```python
    >>> from trl.rewards import accuracy_reward

    >>> solutions = [r"\frac{1}{3}", r"\frac{1}{3}"]
    >>> completions = [
    ...     [{"role": "assistant", "content": r"My answer is \boxed{\frac{1}{3}}"}],
    ...     [{"role": "assistant", "content": r"My answer is \boxed{\frac{1}{2}}"}],
    ... ]
    >>> accuracy_reward(completions, solutions)
    [1.0, 0.0]
    ```
    """
    if not is_math_verify_available():
        raise ImportError("Please install the `math_verify` package to use accuracy_reward")

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    gold_parsed_strs = []
    answer_parsed_strs = []

    # math_verify uses signal.alarm() for timeouts, which only works in the main thread.
    # Disable timeouts when running in a non-main thread to avoid ValueError.
    is_main_thread = threading.current_thread() is threading.main_thread()
    parsing_timeout = None if not is_main_thread else 10
    verify_timeout = None if not is_main_thread else 5

    # Suppress the "Timeout is disabled" warnings from math_verify when we intentionally disable timeouts
    if not is_main_thread:
        logging.getLogger("math_verify.parser").setLevel(logging.ERROR)
        logging.getLogger("math_verify.grader").setLevel(logging.ERROR)

    for content, sol in zip(contents, solution, strict=True):
        gold_parsed = parse(sol, parsing_timeout=parsing_timeout)
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(units=True),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
                parsing_timeout=parsing_timeout,
            )
            reward = float(verify(gold_parsed, answer_parsed, timeout_seconds=verify_timeout))
            gold_parsed_strs.append(str(gold_parsed))
            answer_parsed_strs.append(str(answer_parsed) if answer_parsed else "[unparseable]")
        else:
            # If the gold solution cannot be parsed, we assign `None` to skip this example
            reward = None
            gold_parsed_strs.append("[unparseable]")
            answer_parsed_strs.append("[skipped]")
        rewards.append(reward)

    if log_extra is not None:
        log_extra("solution", list(solution))
        log_extra("gold_parsed", gold_parsed_strs)
        log_extra("answer_parsed", answer_parsed_strs)

    return rewards


def get_cosine_scaled_reward(
    max_len: int,
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
) -> Callable:
    # [FUNCTION DOC]
    # Behavior: returns a _CosineScaledReward callable with the given bounds; no validation.
    # Inputs: max_len (tokens, normalizes the schedule); min_value_wrong (-1.0), max_value_wrong (-0.5),
    # min_value_correct (0.5), max_value_correct (1.0).
    # Outputs: _CosineScaledReward instance. Exceptions: none. Side effects: none.
    # docstyle-ignore
    r"""
    Reward function that scales a correctness reward by the completion length following a cosine schedule, to favor
    concise reasoning. Reference: Appendix C.1 of the "Demystifying Long Chain-of-Thought Reasoning" paper
    (https://huggingface.co/papers/2502.03373).

    Correctness is determined by math verification (as in [`~rewards.accuracy_reward`]), and the length is the number
    of completion tokens. The reward interpolates along a cosine schedule between a short-completion and a
    long-completion bound:

    $$
    R_{\text{cosine}}(y) = v_{\min} + \frac{1}{2}(v_{\max} - v_{\min})\left(1 + \cos\left(\frac{|y|}{L_{\max}}\pi\right)\right)
    $$

    For a **correct** completion, $(v_{\min}, v_{\max}) = (\texttt{min\_value\_correct}, \texttt{max\_value\_correct})$,
    so a shorter completion is rewarded more. For a **wrong** completion, the bounds are swapped to
    $(v_{\min}, v_{\max}) = (\texttt{max\_value\_wrong}, \texttt{min\_value\_wrong})$, so a shorter completion is
    penalized more (a longer wrong completion is penalized less, preserving exploration). When the gold solution is not
    parseable, the example is skipped (reward `None`), as in [`~rewards.accuracy_reward`].

    Args:
        max_len (`int`):
            Maximum completion length (in tokens) used to normalize the cosine schedule, $L_{\max}$.
        min_value_wrong (`float`, *optional*, defaults to `-1.0`):
            Reward of a wrong completion at the shortest length.
        max_value_wrong (`float`, *optional*, defaults to `-0.5`):
            Reward of a wrong completion at the longest length.
        min_value_correct (`float`, *optional*, defaults to `0.5`):
            Reward of a correct completion at the longest length.
        max_value_correct (`float`, *optional*, defaults to `1.0`):
            Reward of a correct completion at the shortest length.

    Returns:
        `Callable`:
            A reward function that takes completions, their solutions and token ids, and returns a list of rewards
            (`None` for examples with an unparseable gold solution).

    Example:
    ```python
    >>> from trl.rewards import get_cosine_scaled_reward

    >>> cosine_scaled_reward = get_cosine_scaled_reward(max_len=100)
    >>> completions = [[{"content": r"\boxed{\frac{1}{3}}"}], [{"content": r"\boxed{\frac{1}{2}}"}]]
    >>> solution = [r"\frac{1}{3}", r"\frac{1}{3}"]
    >>> completion_ids = [[1] * 50, [1] * 50]  # both completions are 50 tokens, half of max_len
    >>> cosine_scaled_reward(completions, solution, completion_ids)
    [0.75, -0.75]
    ```
    """
    return _CosineScaledReward(max_len, min_value_wrong, max_value_wrong, min_value_correct, max_value_correct)


class _CosineScaledReward:
    # [CLASS DOC]
    # HELPER: Picklable callable implementing the cosine length-scaled correctness reward (closure avoided so the
    # async GRPO rollout worker can pickle it). Class attribute __name__ "cosine_scaled_reward" names logged
    # metrics. State: max_len and the four bound values. Built by get_cosine_scaled_reward.
    # Callable class rather than a closure so the reward stays picklable: the async GRPO rollout
    # worker forwards reward funcs to a spawned child process, and closures can't be pickled.
    __name__ = "cosine_scaled_reward"

    def __init__(
        self,
        max_len: int,
        min_value_wrong: float,
        max_value_wrong: float,
        min_value_correct: float,
        max_value_correct: float,
    ):
        # [FUNCTION DOC]
        # HELPER: Stores max_len, min_value_wrong, max_value_wrong, min_value_correct, max_value_correct.
        # Inputs/Outputs: as named; returns none. Exceptions: none. Side effects: sets attributes.
        self.max_len = max_len
        self.min_value_wrong = min_value_wrong
        self.max_value_wrong = max_value_wrong
        self.min_value_correct = min_value_correct
        self.max_value_correct = max_value_correct

    def __call__(
        self,
        completions: list[list[dict[str, str]]],
        solution: list[str],
        completion_ids: list[list[int]],
        **kwargs,
    ) -> list[float | None]:
        # [FUNCTION DOC]
        # HELPER: Scores each completion.
        # Behavior: computes correctness with accuracy_reward (no log_extra). For each example: None correctness gives
        # None. Otherwise progress = number of completion tokens / max_len clamped to at most 1.0, cosine of
        # progress times pi; bounds are (min_value_correct, max_value_correct) when correct, else swapped wrong
        # bounds (max_value_wrong as min, min_value_wrong as max); reward = min + half of (max - min) times
        # (1 + cosine). So short correct answers approach max_value_correct and short wrong answers approach
        # min_value_wrong.
        # Inputs: completions, solution, completion_ids (token-id lists, same length); kwargs ignored.
        # Outputs: list of float or None.
        # Exceptions: ImportError (via accuracy_reward); ValueError on length mismatch; ZeroDivisionError if max_len 0.
        # Side effects: same logger-level side effect as accuracy_reward in non-main threads.
        is_correct = accuracy_reward(completions, solution)
        rewards = []
        for correct, ids in zip(is_correct, completion_ids, strict=True):
            if correct is None:
                # Gold solution was not parseable; skip the example, as in accuracy_reward.
                rewards.append(None)
                continue
            # Clamp to 1.0 so completions longer than max_len stay at the long-length bound: cos is periodic, so
            # without clamping the schedule would climb back up past max_len and reward very long completions.
            progress = min(len(ids) / self.max_len, 1.0)
            cosine = math.cos(progress * math.pi)
            if correct:
                min_value, max_value = self.min_value_correct, self.max_value_correct
            else:
                # Swap the bounds so that a shorter wrong completion is penalized more than a longer one.
                min_value, max_value = self.max_value_wrong, self.min_value_wrong
            rewards.append(float(min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)))
        return rewards


def reasoning_accuracy_reward(
    completions: list[list[dict[str, str]]],
    solution: list[str],
    reasoning_delimiters: list[str] | None = None,
    log_extra: Callable[[str, list], None] | None = None,
    **kwargs,
) -> list[float | None]:
    # [FUNCTION DOC]
    # Behavior: same as accuracy_reward but first strips reasoning: delimiters default to the single "</think>".
    # Delimiters are tried in list order; the first one present in the content is used and the content becomes the
    # text after its last occurrence. If no delimiter is present the reward is 0.0 (not None, to penalize
    # incomplete reasoning) and both logged strings are "[incomplete reasoning]". Then gold parsing / answer
    # parsing / verify exactly as in accuracy_reward (same timeouts, thread handling, None for unparseable gold).
    # Inputs: completions, solution, reasoning_delimiters (optional list of str), log_extra (optional), kwargs ignored.
    # Outputs: list of float or None.
    # Exceptions: ImportError without math_verify; ValueError on length mismatch.
    # Side effects: log_extra columns "solution", "gold_parsed", "answer_parsed"; logger levels in non-main threads.
    # KNOWN ISSUE: the docstring says the answer follows the last occurrence of any delimiter, but the code uses
    # the first delimiter in list order that appears, not the delimiter occurring last in the text.
    r"""
    Reward function that removes the reasoning content and checks if the final answer matches the ground truth.
        - If both gold and prediction are parseable → use math verification.
        - If gold is not parseable → return `None` to skip the example.

    Args:
        completions (`list[list[dict[str, str]]]`):
            List of completions to be evaluated. Each completion must be a list of one message, i.e. a dictionary
            containing the key `"content"` with the value being the text of the completion.
        solution (`list[str]`):
            List of the raw-text solutions to the questions/problems/prompts.
        reasoning_delimiters (`list[str]]`, *optional*):
            List of strings indicating where the reasoning content ends. The final answer is assumed to be after the
            last occurrence of any of these delimiters. If `None`, defaults to `["</think>"]`.
        log_extra (`callable`, *optional*):
            Callable to log extra columns to the completions table, provided automatically by the trainer. Defaults to
            `None` to allow calling the function directly outside of a trainer (e.g., for testing).
        **kwargs:
            Additional keyword arguments. This function does not use them, but they are required in the function
            signature to ensure compatibility with trainers like [`GRPOTrainer`].
    Example:
        ```python
        >>> from trl.rewards import reasoning_accuracy_reward

        >>> reasoning_delimiters = ["</think>"]
        >>> solutions = [r"\frac{1}{3}", r"\frac{1}{3}", r"\frac{1}{3}"]
        >>> completions = [
        ...     [
        ...         {
        ...             "role": "assistant",
        ...             "content": r"<think> Reasoning content </think> The final answer is \boxed{\frac{1}{3}}",
        ...         }
        ...     ],
        ...     [
        ...         {
        ...             "role": "assistant",
        ...             "content": r"<think> Reasoning content </think> The final answer is \boxed{\frac{1}{2}}",
        ...         }
        ...     ],
        ...     [
        ...         {
        ...             "role": "assistant",
        ...             "content": r"<think> Reasoning content with partial answers \boxed{\frac{1}{3}} but no final answer",
        ...         }
        ...     ],
        ... ]
        >>> reasoning_accuracy_reward(completions, solutions, reasoning_delimiters=reasoning_delimiters)
        [1.0, 0.0, 0.0]
        ```
    """
    if not is_math_verify_available():
        raise ImportError("Please install the `math_verify` package to use reasoning_accuracy_reward")

    if reasoning_delimiters is None:
        # Use sensible defaults for majority of reasoning models
        reasoning_delimiters = ["</think>"]

    rewards = []
    contents = [completion[0]["content"] for completion in completions]
    gold_parsed_strs = []
    answer_parsed_strs = []

    # math_verify uses signal.alarm() for timeouts, which only works in the main thread.
    # Disable timeouts when running in a non-main thread to avoid ValueError.
    is_main_thread = threading.current_thread() is threading.main_thread()
    parsing_timeout = None if not is_main_thread else 10
    verify_timeout = None if not is_main_thread else 5

    # Suppress the "Timeout is disabled" warnings from math_verify when we intentionally disable timeouts
    if not is_main_thread:
        logging.getLogger("math_verify.parser").setLevel(logging.ERROR)
        logging.getLogger("math_verify.grader").setLevel(logging.ERROR)

    for content, sol in zip(contents, solution, strict=True):
        # Split final answer from reasoning content
        is_reasoning_complete = False
        for delim in reasoning_delimiters:
            if delim in content:
                content = content.split(delim)[-1]
                is_reasoning_complete = True
                break
        if not is_reasoning_complete:
            # We assign zero reward instead of `None` to penalize incomplete reasoning
            rewards.append(0.0)
            gold_parsed_strs.append("[incomplete reasoning]")
            answer_parsed_strs.append("[incomplete reasoning]")
            continue

        gold_parsed = parse(sol, parsing_timeout=parsing_timeout)
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        boxed_match_priority=0,
                        normalization_config=NormalizationConfig(
                            units=True,
                        ),
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
                parsing_timeout=parsing_timeout,
            )
            reward = float(verify(gold_parsed, answer_parsed, timeout_seconds=verify_timeout))
            gold_parsed_strs.append(str(gold_parsed))
            answer_parsed_strs.append(str(answer_parsed) if answer_parsed else "[unparseable]")
        else:
            # If the gold solution cannot be parsed, we assign `None` to skip this example
            reward = None
            gold_parsed_strs.append("[unparseable]")
            answer_parsed_strs.append("[skipped]")
        rewards.append(reward)

    if log_extra is not None:
        log_extra("solution", list(solution))
        log_extra("gold_parsed", gold_parsed_strs)
        log_extra("answer_parsed", answer_parsed_strs)

    return rewards
