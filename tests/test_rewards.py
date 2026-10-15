# [MODULE DOC]
# Tests for the reward functions in trl.rewards: think_format_reward, get_soft_overlong_punishment,
# get_repetition_penalty_reward, accuracy_reward, reasoning_accuracy_reward and get_cosine_scaled_reward.
# Completions are given in conversational form (a list holding one dict with a content key per sample) or as
# token id lists. Math-verifying tests are gated by require_math_latex.
# Import-time side effects: none.

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

import pickle
import threading

import pytest

from trl.rewards import (
    accuracy_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    get_soft_overlong_punishment,
    reasoning_accuracy_reward,
    think_format_reward,
)

from .testing_utils import TrlTestCase, require_math_latex


class TestThinkFormatReward(TrlTestCase):
    # [CLASS DOC]
    # Tests think_format_reward, which returns 1.0 when a completion starts with a single think block followed by
    # the answer and 0.0 otherwise. No setup; inherits TrlTestCase.
    def test_valid_format(self):
        # [FUNCTION DOC]
        # Asserts 1.0 for five valid layouts: one-line, multiline, inner other tags, and empty reasoning.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        completions = [
            "<think>This is my reasoning.</think>This is my answer.",  # Simple, one-line reasoning
            "<think>\nThis is my reasoning.\n</think>\nThis is my answer.",  # Multiline reasoning
            "<think>\nThis is\nmy reasoning.\n</think>\nThis is my answer.",  # Multiline reasoning
            "<think>\nThis is <some tag> my reasoning.</think>\nThis is my answer.",  # Reasoning including other tags
            "<think></think>\nThis is my answer.",  # Empty reasoning
        ]
        completions = [[{"content": completion}] for completion in completions]
        expected_rewards = [1.0, 1.0, 1.0, 1.0, 1.0]  # All should be valid
        rewards = think_format_reward(completions)
        assert rewards == expected_rewards

    def test_invalid_format(self):
        # [FUNCTION DOC]
        # Asserts 0.0 for nine invalid layouts: missing closing or opening tag, no tags, tag not at start, nested think
        # tags, and two separate think blocks. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        completions = [
            "<think>\nThis is my reasoning.\nThis is my answer.",  # No closing </think>
            "<think>This is my reasoning.\nThis is my answer.",  # No closing </think>
            "This is my reasoning. This is my answer.",  # No <think> tags
            "This is my reasoning.\nThis is my answer.",  # No <think> tags
            "This is my reasoning.</think>\nThis is my answer.",  # No opening <think>
            "This is my reasoning.</think>This is my answer.",  # No opening <think>
            "This<think>is my reasoning.</think>\nThis is my answer.",  # <think> tag in the middle
            "<think>This is<think>my reasoning.</think></think>This is my answer.",  # Nested <think> tags
            "<think>This is</think>\nmy\n<think>reasoning.</think>\nThis is my answer.",  # Multiline <think>
        ]
        completions = [[{"content": completion}] for completion in completions]
        expected_rewards = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # All should be invalid
        rewards = think_format_reward(completions)
        assert rewards == expected_rewards

    def test_mixed_format(self):
        # [FUNCTION DOC]
        # Asserts a mixed batch of two valid and two invalid completions yields 1, 1, 0, 0 in order.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        completions = [
            "<think>This is my reasoning.</think>This is my answer.",  # Valid
            "<think>\nThis is my reasoning.\n</think>\nThis is my answer.",  # Valid
            "<think>This is my reasoning.\nThis is my answer.",  # Invalid
            "This is my reasoning. This is my answer.",  # Invalid
        ]
        completions = [[{"content": completion}] for completion in completions]
        expected_rewards = [1.0, 1.0, 0.0, 0.0]
        rewards = think_format_reward(completions)
        assert rewards == expected_rewards


class TestSoftOverlongPunishmentReward:
    # [CLASS DOC]
    # Tests the soft overlong punishment reward with max_completion_len 100 and soft_punish_cache 20: 0 up to
    # length 80, linear ramp to -1 between 80 and 100, -1 beyond 100.
    def test_soft_overlong_punishment_short_completion(self):
        # [FUNCTION DOC]
        # Asserts a 50-token completion (passed as keyword completion_ids) gets 0.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Test soft overlong punishment reward function with a short completion."""
        # length 50, with max=100 and soft cache=20, reward should be 0.
        reward_fn = get_soft_overlong_punishment(max_completion_len=100, soft_punish_cache=20)
        completion_ids = [[1] * 50]  # 50 <= 80
        rewards = reward_fn(completion_ids=completion_ids)
        assert rewards == [0]

    def test_soft_overlong_punishment_long_completion(self):
        # [FUNCTION DOC]
        # Asserts a 110-token completion (passed positionally) gets -1.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Test soft overlong punishment reward function with a longer than max completion."""
        # 110 > 100, reward should be -1.
        reward_fn = get_soft_overlong_punishment(max_completion_len=100, soft_punish_cache=20)
        completion_ids = [[1] * 110]
        rewards = reward_fn(completion_ids)
        assert rewards == [-1]

    def test_soft_overlong_punishment_intermediate_completion(self):
        # [FUNCTION DOC]
        # Asserts a 90-token completion gets -0.5 (to 4 decimals).
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Test soft overlong punishment reward function for intermediate length completion."""
        reward_fn = get_soft_overlong_punishment(max_completion_len=100, soft_punish_cache=20)
        completion_ids = [[1] * 90]  # 90 is between 80 and 100
        rewards = reward_fn(completion_ids)
        assert round(abs(rewards[0] - -0.5), 4) == 0


class TestRepetitionPenaltyReward:
    # [CLASS DOC]
    # Tests get_repetition_penalty_reward: reward is max_penalty times (1 minus unique n-grams over total n-grams),
    # 0 when there are no n-grams; validates arguments, ignores extra kwargs and must be picklable.
    def test_no_repetition_yields_zero(self):
        # [FUNCTION DOC]
        # Asserts four distinct tokens with bigrams yield 0.0. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: none.
        """A completion with only unique n-grams gets no penalty."""
        reward_fn = get_repetition_penalty_reward(ngram_size=2, max_penalty=-1.0)
        completion_ids = [[1, 2, 3, 4]]
        assert reward_fn(completion_ids) == [0.0]

    def test_full_repetition_approaches_max_penalty(self):
        # [FUNCTION DOC]
        # Asserts five identical tokens (4 bigrams, 1 unique) yield -0.75. Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: none.
        """A fully repetitive completion approaches max_penalty."""
        reward_fn = get_repetition_penalty_reward(ngram_size=2, max_penalty=-1.0)
        # [5, 5, 5, 5, 5] -> 4 bigrams, 1 unique -> scaling = 1 - 1/4 = 0.75
        completion_ids = [[5, 5, 5, 5, 5]]
        assert reward_fn(completion_ids) == [pytest.approx(-0.75)]

    def test_partial_repetition(self):
        # [FUNCTION DOC]
        # Asserts alternating 1,2 of length 6 (5 bigrams, 2 unique) yields -0.6. Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: none.
        reward_fn = get_repetition_penalty_reward(ngram_size=2, max_penalty=-1.0)
        # [1, 2, 1, 2, 1, 2] -> 5 bigrams, 2 unique -> scaling = 1 - 2/5 = 0.6
        completion_ids = [[1, 2, 1, 2, 1, 2]]
        assert reward_fn(completion_ids) == [pytest.approx(-0.6)]

    def test_completion_shorter_than_ngram_size_yields_zero(self):
        # [FUNCTION DOC]
        # Asserts a 2-token completion with ngram_size 3 yields 0.0. Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: none.
        reward_fn = get_repetition_penalty_reward(ngram_size=3, max_penalty=-1.0)
        completion_ids = [[1, 2]]  # 2 tokens < ngram_size
        assert reward_fn(completion_ids) == [0.0]

    def test_completion_exactly_ngram_size_yields_zero(self):
        # [FUNCTION DOC]
        # Asserts a 3-token completion with ngram_size 3 (one unique n-gram) yields 0.0. Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: none.
        reward_fn = get_repetition_penalty_reward(ngram_size=3, max_penalty=-1.0)
        completion_ids = [[1, 2, 3]]  # a single, unique n-gram
        assert reward_fn(completion_ids) == [0.0]

    def test_empty_completion_yields_zero(self):
        # [FUNCTION DOC]
        # Asserts an empty completion yields 0.0. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        reward_fn = get_repetition_penalty_reward(ngram_size=3, max_penalty=-1.0)
        completion_ids = [[]]
        assert reward_fn(completion_ids) == [0.0]

    def test_max_penalty_scales_reward(self):
        # [FUNCTION DOC]
        # Asserts max_penalty -0.5 scales the 0.75 repetition ratio to -0.375. Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: none.
        reward_fn = get_repetition_penalty_reward(ngram_size=2, max_penalty=-0.5)
        # scaling 0.75 * max_penalty -0.5 = -0.375
        completion_ids = [[5, 5, 5, 5, 5]]
        assert reward_fn(completion_ids) == [pytest.approx(-0.375)]

    def test_ngram_size_changes_reward(self):
        # [FUNCTION DOC]
        # Asserts the sequence 1,2,3,1,2,3 yields -0.4 with bigrams and -0.25 with trigrams.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        completion_ids = [[1, 2, 3, 1, 2, 3]]
        # bigrams: 5 total, 3 unique -> 1 - 3/5 = 0.4
        reward_bigram = get_repetition_penalty_reward(ngram_size=2, max_penalty=-1.0)
        assert reward_bigram(completion_ids) == [pytest.approx(-0.4)]
        # trigrams: 4 total, 3 unique -> 1 - 3/4 = 0.25
        reward_trigram = get_repetition_penalty_reward(ngram_size=3, max_penalty=-1.0)
        assert reward_trigram(completion_ids) == [pytest.approx(-0.25)]

    def test_batch_of_completions(self):
        # [FUNCTION DOC]
        # Asserts a batch of no-repetition, full-repetition and too-short completions yields 0.0, -0.75, 0.0.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        reward_fn = get_repetition_penalty_reward(ngram_size=2, max_penalty=-1.0)
        completion_ids = [
            [1, 2, 3, 4],  # no repetition
            [5, 5, 5, 5, 5],  # full repetition
            [9],  # shorter than ngram_size
        ]
        assert reward_fn(completion_ids) == [pytest.approx(0.0), pytest.approx(-0.75), pytest.approx(0.0)]

    def test_positive_max_penalty_raises(self):
        # [FUNCTION DOC]
        # Asserts building the reward with positive max_penalty 0.5 raises ValueError.
        # Inputs/Outputs: none. Exceptions: AssertionError if not raised. Side effects: none.
        with pytest.raises(ValueError):
            get_repetition_penalty_reward(ngram_size=2, max_penalty=0.5)

    def test_extra_kwargs_are_ignored(self):
        # [FUNCTION DOC]
        # Asserts passing prompts and completions keyword arguments does not change the -0.75 result.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Trainers pass prompts/completions/etc. as kwargs; the reward must accept and ignore them."""
        reward_fn = get_repetition_penalty_reward(ngram_size=2, max_penalty=-1.0)
        completion_ids = [[5, 5, 5, 5, 5]]
        rewards = reward_fn(completion_ids, prompts=["x"], completions=[[{"content": "5 5 5 5 5"}]])
        assert rewards == [pytest.approx(-0.75)]

    def test_reward_is_picklable(self):
        # [FUNCTION DOC]
        # Asserts the reward survives a pickle round trip, still computes -0.75 and keeps the name
        # repetition_penalty_reward. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """The reward must survive pickling for the async GRPO rollout worker."""
        reward_fn = get_repetition_penalty_reward(ngram_size=2, max_penalty=-1.0)
        unpickled = pickle.loads(pickle.dumps(reward_fn))
        completion_ids = [[5, 5, 5, 5, 5]]
        assert unpickled(completion_ids) == [pytest.approx(-0.75)]
        assert unpickled.__name__ == "repetition_penalty_reward"


class TestAccuracyReward:
    # [CLASS DOC]
    # Tests accuracy_reward (math-verify based): 1.0 when the boxed answer matches the gold solution, 0.0 when not,
    # None when the gold solution cannot be parsed; must work off the main thread. All gated by require_math_latex.
    @require_math_latex
    def test_accuracy_reward_correct_answer(self):
        # [FUNCTION DOC]
        # Asserts a boxed fraction 63/400 matches both a LaTeX and a plain-text gold, yielding 1.0 twice.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """Test accuracy_reward with a correct answer."""
        completion = [[{"content": r"\boxed{\frac{63}{400}}"}], [{"content": r"\boxed{\frac{63}{400}}"}]]
        solution = [r"\frac{63}{400}", "63/400"]
        rewards = accuracy_reward(completion, solution)
        assert rewards[0] == 1.0
        assert rewards[1] == 1.0

    @require_math_latex
    def test_accuracy_reward_wrong_answer(self):
        # [FUNCTION DOC]
        # Asserts a boxed 64/400 against gold 63/400 yields 0.0. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: none.
        """Test accuracy_reward with an incorrect answer."""
        completion = [[{"content": r"\boxed{\frac{64}{400}}"}]]
        solution = [r"\frac{63}{400}"]
        rewards = accuracy_reward(completion, solution)
        assert rewards[0] == 0.0

    @require_math_latex
    def test_accuracy_reward_wrong_answer_no_latex(self):
        # [FUNCTION DOC]
        # Asserts boxed 3 against plain gold 6 yields 0.0. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: none.
        """Test accuracy_reward with an incorrect answer and gold solution with no latex."""
        completion = [[{"content": r"\boxed{3}"}]]
        solution = ["6"]
        rewards = accuracy_reward(completion, solution)
        assert rewards[0] == 0.0

    @require_math_latex
    def test_accuracy_reward_unparsable_gold(self):
        # [FUNCTION DOC]
        # Asserts gold solutions written in words yield None for both samples. Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: none.
        """Test accuracy_reward with an unparsable gold solution."""
        completion = [
            [{"content": "Answer is forty two."}],
            [{"content": r"Some other content. \boxed{43}."}],
        ]
        solution = [
            "Answer is forty two.",
            "Answer is forty three.",
        ]
        rewards = accuracy_reward(completion, solution)
        assert rewards[0] is None
        assert rewards[1] is None

    @require_math_latex
    def test_accuracy_reward_in_worker_thread(self):
        # [FUNCTION DOC]
        # Runs accuracy_reward inside a separate thread (signal-based timeouts can fail off the main thread), collects
        # results and exceptions, joins, and asserts no exception and result 1.0.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: starts and joins a thread.
        """Test that accuracy_reward works when called from a non-main thread."""
        completions = [[{"content": r"\boxed{\frac{1}{3}}"}]]
        solutions = [r"\frac{1}{3}"]
        results = []
        exceptions = []

        def target():
            # [FUNCTION DOC]
            # HELPER: Thread body: extends results with accuracy_reward output, or appends any raised exception to
            # exceptions. Inputs: none (closure). Outputs: none. Exceptions: caught. Side effects: mutates the two lists.
            try:
                results.extend(accuracy_reward(completions, solutions))
            except Exception as e:
                exceptions.append(e)

        t = threading.Thread(target=target)
        t.start()
        t.join()

        assert not exceptions, f"accuracy_reward raised in worker thread: {exceptions[0]}"
        assert results == [1.0]


class TestReasoningAccuracyReward:
    # [CLASS DOC]
    # Tests reasoning_accuracy_reward: only the text after the reasoning delimiter (default closing think tag) is
    # verified; missing delimiter or answer only inside reasoning yields 0.0; unparsable gold yields None.
    @require_math_latex
    def test_correct_answer_yields_unit_reward(self):
        # [FUNCTION DOC]
        # Asserts a correct boxed answer after the closing think tag yields 1.0, with or without an opening tag.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        completions = [
            [{"content": r"<think> Reasoning content </think> \boxed{\frac{63}{400}}"}],
            [{"content": r"Reasoning content </think> \boxed{\frac{63}{400}}"}],
        ]
        solutions = [r"\frac{63}{400}", r"\frac{63}{400}"]
        rewards = reasoning_accuracy_reward(completions, solutions)
        assert rewards[0] == 1.0
        assert rewards[1] == 1.0

    @require_math_latex
    def test_correct_answer_with_custom_tags_yields_unit_reward(self):
        # [FUNCTION DOC]
        # Asserts custom reasoning_delimiters (closing REASONING_END tag) are honored, yielding 1.0.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        completions = [
            [{"content": r"<REASONING_START> Reasoning content </REASONING_END> \boxed{\frac{63}{400}}"}],
        ]
        solutions = [
            r"\frac{63}{400}",
        ]
        rewards = reasoning_accuracy_reward(completions, solutions, reasoning_delimiters=["</REASONING_END>"])
        assert rewards[0] == 1.0

    @require_math_latex
    def test_incorrect_answer_yields_zero_reward(self):
        # [FUNCTION DOC]
        # Asserts a wrong boxed answer after reasoning yields 0.0. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: none.
        completion = [[{"content": r"<think> Reasoning content </think> \boxed{\frac{64}{400}}"}]]
        solution = [r"\frac{63}{400}"]
        rewards = reasoning_accuracy_reward(completion, solution)
        assert rewards[0] == 0.0

    @require_math_latex
    def test_correct_answer_in_reasoning_yields_zero_reward(self):
        # [FUNCTION DOC]
        # Asserts a correct answer that appears only inside the reasoning yields 0.0 (also when a wrong answer follows).
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        completions = [
            [{"content": r"<think> My answer is \boxed{42} </think> Some other text."}],
            [{"content": r"<think> The answer is \boxed{42} </think> Here's a wrong answer: \boxed{43}."}],
        ]
        solutions = [r"\boxed{42}", r"\boxed{42}"]
        rewards = reasoning_accuracy_reward(completions, solutions)
        assert rewards[0] == 0.0
        assert rewards[1] == 0.0

    @require_math_latex
    def test_incomplete_reasoning_yields_zero_reward(self):
        # [FUNCTION DOC]
        # Asserts completions lacking the closing delimiter yield 0.0 even if the answer appears.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        completions = [
            [{"content": r"<think> Incomplete reasoning without closing tag"}],
            [{"content": r"Correct answer \frac{63}{400} but completely missing reasoning content"}],
        ]
        solutions = [r"\frac{63}{400}", r"\frac{63}{400}"]
        rewards = reasoning_accuracy_reward(completions, solutions)
        assert rewards[0] == 0.0
        assert rewards[1] == 0.0

    @require_math_latex
    def test_unparsable_gold_solution_yields_none_reward(self):
        # [FUNCTION DOC]
        # Asserts a gold solution in words yields None. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: none.
        completions = [
            [{"content": r"<think> Reasoning content </think> \boxed{42}"}],
        ]
        solutions = [
            "forty two",
        ]
        rewards = reasoning_accuracy_reward(completions, solutions)
        assert rewards[0] is None


class TestCosineScaledReward:
    # [CLASS DOC]
    # Tests get_cosine_scaled_reward with max_len 100 and default bounds (correct 0.5 to 1.0, wrong -1.0 to -0.5):
    # the reward interpolates along a cosine of length progress clamped at max_len; shorter correct answers score
    # higher, longer wrong answers are penalized less; unparsable gold yields None; must be picklable.
    @require_math_latex
    def test_correct_shorter_rewarded_more(self):
        # [FUNCTION DOC]
        # Asserts correct answers of length 25 and 75 yield about 0.92678 and 0.57322.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """For correct completions, a shorter one gets a higher reward."""
        reward_fn = get_cosine_scaled_reward(max_len=100)
        completions = [[{"content": r"\boxed{\frac{1}{3}}"}], [{"content": r"\boxed{\frac{1}{3}}"}]]
        solution = [r"\frac{1}{3}", r"\frac{1}{3}"]
        completion_ids = [[1] * 25, [1] * 75]
        rewards = reward_fn(completions, solution, completion_ids)
        assert rewards[0] > rewards[1]
        assert rewards == [pytest.approx(0.92678, abs=1e-4), pytest.approx(0.57322, abs=1e-4)]

    @require_math_latex
    def test_wrong_longer_penalized_less(self):
        # [FUNCTION DOC]
        # Asserts wrong answers of length 25 and 75 yield about -0.92678 and -0.57322.
        # Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """For wrong completions, a longer one is penalized less (closer to zero)."""
        reward_fn = get_cosine_scaled_reward(max_len=100)
        completions = [[{"content": r"\boxed{\frac{1}{2}}"}], [{"content": r"\boxed{\frac{1}{2}}"}]]
        solution = [r"\frac{1}{3}", r"\frac{1}{3}"]
        completion_ids = [[1] * 25, [1] * 75]
        rewards = reward_fn(completions, solution, completion_ids)
        assert rewards[1] > rewards[0]
        assert rewards == [pytest.approx(-0.92678, abs=1e-4), pytest.approx(-0.57322, abs=1e-4)]

    @require_math_latex
    def test_midpoint_values(self):
        # [FUNCTION DOC]
        # Asserts at length 50 correct yields 0.75 and wrong yields -0.75. Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: none.
        """At half of max_len (cosine = 0), correct -> 0.75 and wrong -> -0.75 with default bounds."""
        reward_fn = get_cosine_scaled_reward(max_len=100)
        completions = [[{"content": r"\boxed{\frac{1}{3}}"}], [{"content": r"\boxed{\frac{1}{2}}"}]]
        solution = [r"\frac{1}{3}", r"\frac{1}{3}"]
        completion_ids = [[1] * 50, [1] * 50]
        rewards = reward_fn(completions, solution, completion_ids)
        assert rewards == [pytest.approx(0.75), pytest.approx(-0.75)]

    @require_math_latex
    def test_correct_boundary_values(self):
        # [FUNCTION DOC]
        # Asserts correct answers at length 0 and 100 yield 1.0 and 0.5. Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: none.
        """Correct: shortest -> max_value_correct (1.0), longest -> min_value_correct (0.5)."""
        reward_fn = get_cosine_scaled_reward(max_len=100)
        completions = [[{"content": r"\boxed{\frac{1}{3}}"}], [{"content": r"\boxed{\frac{1}{3}}"}]]
        solution = [r"\frac{1}{3}", r"\frac{1}{3}"]
        completion_ids = [[], [1] * 100]
        rewards = reward_fn(completions, solution, completion_ids)
        assert rewards == [pytest.approx(1.0), pytest.approx(0.5)]

    @require_math_latex
    def test_wrong_boundary_values(self):
        # [FUNCTION DOC]
        # Asserts wrong answers at length 0 and 100 yield -1.0 and -0.5. Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: none.
        """Wrong: shortest -> min_value_wrong (-1.0), longest -> max_value_wrong (-0.5)."""
        reward_fn = get_cosine_scaled_reward(max_len=100)
        completions = [[{"content": r"\boxed{\frac{1}{2}}"}], [{"content": r"\boxed{\frac{1}{2}}"}]]
        solution = [r"\frac{1}{3}", r"\frac{1}{3}"]
        completion_ids = [[], [1] * 100]
        rewards = reward_fn(completions, solution, completion_ids)
        assert rewards == [pytest.approx(-1.0), pytest.approx(-0.5)]

    @require_math_latex
    def test_length_exceeding_max_len_is_clamped(self):
        # [FUNCTION DOC]
        # Asserts length 200 behaves like length 100: correct 0.5, wrong -0.5. Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: none.
        """Completions longer than max_len stay at the long-length bound (no climb back up past max_len)."""
        reward_fn = get_cosine_scaled_reward(max_len=100)
        completions = [[{"content": r"\boxed{\frac{1}{3}}"}], [{"content": r"\boxed{\frac{1}{2}}"}]]
        solution = [r"\frac{1}{3}", r"\frac{1}{3}"]
        completion_ids = [[1] * 200, [1] * 200]  # both 2x max_len
        rewards = reward_fn(completions, solution, completion_ids)
        # correct -> min_value_correct (0.5), wrong -> max_value_wrong (-0.5); same as at exactly max_len
        assert rewards == [pytest.approx(0.5), pytest.approx(-0.5)]

    @require_math_latex
    def test_unparsable_gold_yields_none(self):
        # [FUNCTION DOC]
        # Asserts a gold solution in words yields None. Inputs/Outputs: none. Exceptions: AssertionError.
        # Side effects: none.
        """An unparseable gold solution is skipped, as in accuracy_reward."""
        reward_fn = get_cosine_scaled_reward(max_len=100)
        completions = [[{"content": r"\boxed{42}"}]]
        solution = ["forty two"]
        completion_ids = [[1] * 50]
        rewards = reward_fn(completions, solution, completion_ids)
        assert rewards == [None]

    @require_math_latex
    def test_custom_value_bounds(self):
        # [FUNCTION DOC]
        # Asserts custom correct bounds 0.0 to 2.0 at length 50 give 1.0. Inputs/Outputs: none.
        # Exceptions: AssertionError. Side effects: none.
        reward_fn = get_cosine_scaled_reward(max_len=100, min_value_correct=0.0, max_value_correct=2.0)
        completions = [[{"content": r"\boxed{\frac{1}{3}}"}]]
        solution = [r"\frac{1}{3}"]
        completion_ids = [[1] * 50]  # progress 0.5, cosine 0 -> 0.0 + 0.5 * (2.0 - 0.0) * 1 = 1.0
        rewards = reward_fn(completions, solution, completion_ids)
        assert rewards == [pytest.approx(1.0)]

    @require_math_latex
    def test_reward_is_picklable(self):
        # [FUNCTION DOC]
        # Asserts the reward survives a pickle round trip, yields 0.75 at length 50 for a correct answer and keeps the
        # name cosine_scaled_reward. Inputs/Outputs: none. Exceptions: AssertionError. Side effects: none.
        """The reward must survive pickling for the async GRPO rollout worker."""
        reward_fn = get_cosine_scaled_reward(max_len=100)
        unpickled = pickle.loads(pickle.dumps(reward_fn))
        completions = [[{"content": r"\boxed{\frac{1}{3}}"}]]
        solution = [r"\frac{1}{3}"]
        completion_ids = [[1] * 50]
        assert unpickled(completions, solution, completion_ids) == [pytest.approx(0.75)]
        assert unpickled.__name__ == "cosine_scaled_reward"
