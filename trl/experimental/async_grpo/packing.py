# [MODULE DOC]
# Prefix-trie ("tree") packing of multi-turn rollouts for AsyncGRPOTrainer, plus the custom attention kernel it needs.
# Contents: TreeRow (folds samples of one DP rank into a token trie and lays it out as one packed row with loss
# targets), TREE_ATTENTION (attention implementation name "tree_fa3"), _flash_attn (cached hub kernel loader),
# tree_attention (attention function gathering each segment's root-to-segment K/V path and calling varlen flash
# attention), register_tree_attention (registers the function with transformers).
# Import-time side effects: none (registration only happens when register_tree_attention is called).

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

from functools import cache
from typing import Any

import torch
from transformers import AttentionInterface
from transformers.integrations.flash_attention import get_target_dtype
from transformers.masking_utils import AttentionMaskInterface


class TreeRow:
    # [CLASS DOC]
    # One packed training row built from a prefix trie over samples' input_ids (see docstring for the design rationale).
    # State: samples (added sample dicts), sample_nodes (per sample, the trie node id of each of its tokens), tokens
    # (node id -> token id), children (node id -> dict keyed by (token, next token or None) -> child node id), roots
    # (same keying for first tokens), groups (set of group_id values present), trained (count of completion tokens over
    # all samples, counted with duplicates), load (sum over newly created nodes of their depth+1, a proxy of attention
    # cost; not used inside this class).
    # Construction: no inputs. Usage: the planner calls fits then add for each sample, then pack once to obtain tensors.
    # Sample dicts must have input_ids, completion_mask, old_log_probs (1D tensors of equal length), advantage (scalar)
    # and group_id.
    """
    One DP rank's forward: its samples folded into a prefix trie, so a token shared by several samples is forwarded
    once.

    Multi-turn rollouts overlap heavily — a shared prompt, and after a fork a shared conversation — so the decoder's
    per-token work drops by the packing ratio (raw tokens ÷ unique tokens). Attention still shows each sample exactly
    what it saw: each segment of the trie attends to the K/V of its root-to-segment path (see [`tree_attention`]). With
    nothing shared, the trie is one chain per sample, which is plain padding-free packing.

    Any common prefix is shared: the rollouts of a prompt, and across prompts a common system prompt. The planner
    checks a sample against the row ([`~TreeRow.fits`]) before placing it, so the token budget bounds unique tokens,
    not raw ones, and it sends a sample to a row already holding its `group_id` ([`~TreeRow.groups`]), where its prompt
    is.

    A node is keyed on its token *and* the next one, so samples share a token only when they also agree on what follows
    it. The trie therefore forks one token before the samples diverge, and every node has a single next-token label: a
    branch costs one extra token per extra branch, and the loss can go through plain per-position labels. A sample's
    last token is keyed on `None`, which no other sample extends, so every sample ends on a leaf.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes an empty trie: empty lists/dicts for samples, sample_nodes, tokens, children, roots, empty groups set,
        # trained and load set to 0.
        # Inputs: none. Outputs: none. Exceptions: none. Side effects: sets instance attributes.
        self.samples: list[dict[str, Any]] = []
        self.sample_nodes: list[list[int]] = []  # sample -> trie node of each of its tokens
        self.tokens: list[int] = []  # node -> its token
        self.children: list[dict[tuple[int, int | None], int]] = []  # node -> {(token, next token): child node}
        self.roots: dict[tuple[int, int | None], int] = {}  # {(token, next token): root node}
        self.groups: set[Any] = set()
        self.trained = 0
        self.load = 0

    def fits(self, sample: dict[str, Any], budget: int) -> bool:
        # [FUNCTION DOC]
        # Checks whether adding sample keeps the row within budget.
        # Behavior: walks the trie from roots along keys (token, next token; None for the last token) of sample input_ids,
        # counting how many leading tokens already exist (shared). Novel tokens = length minus shared. Returns True iff
        # current unique token count plus novel is at most budget AND trained plus the sample's completion_mask sum is at
        # most budget.
        # Inputs: sample (dict with input_ids and completion_mask tensors), budget (int token budget).
        # Outputs: bool. Exceptions: KeyError if keys missing. Side effects: none (read-only).
        """Whether the row stays within `budget` unique tokens and `budget` trained tokens with `sample` added."""
        input_ids = sample["input_ids"].tolist()
        children, shared = self.roots, 0
        for key in zip(input_ids, [*input_ids[1:], None], strict=True):
            if key not in children:
                break
            children = self.children[children[key]]
            shared += 1
        novel = len(input_ids) - shared
        return len(self.tokens) + novel <= budget and self.trained + int(sample["completion_mask"].sum()) <= budget

    def add(self, sample: dict[str, Any]) -> None:
        # [FUNCTION DOC]
        # Inserts sample into the trie.
        # Behavior: walks keys (token, next token or None) from roots; for each missing key creates a new node (id = current
        # number of tokens), appends its token and an empty children dict, and increases load by depth+1. Records the node of
        # every token of the sample in order.
        # Inputs: sample dict (input_ids, completion_mask, group_id).
        # Outputs: none. Exceptions: KeyError on missing keys.
        # Side effects: mutates tokens, children, roots, load; appends to samples and sample_nodes; adds group_id to groups;
        # adds completion_mask sum to trained.
        input_ids = sample["input_ids"].tolist()
        children = self.roots
        nodes = []
        for depth, key in enumerate(zip(input_ids, [*input_ids[1:], None], strict=True)):
            if key not in children:
                children[key] = len(self.tokens)
                self.tokens.append(key[0])
                self.children.append({})
                self.load += depth + 1
            nodes.append(children[key])
            children = self.children[nodes[-1]]
        self.samples.append(sample)
        self.sample_nodes.append(nodes)
        self.groups.add(sample["group_id"])
        self.trained += int(sample["completion_mask"].sum())

    def pack(self) -> dict[str, torch.Tensor]:
        # [FUNCTION DOC]
        # Serializes the trie into one packed row and builds the per-target loss tensors (keys documented in docstring).
        # Behavior: iterative DFS using a stack seeded with roots in insertion order. Each popped entry starts a new segment:
        # follow the node and, while it has exactly one child, continue into it, appending each token to input_ids with its
        # depth as position id and recording its packed position. A segment ends at a node with 0 or 2+ children. The
        # segment's path is the parent segment's path plus its own positions; the path is appended to tree_kv_index and its
        # cumulative length recorded (cu_seqlens_k); segment end offsets give cu_seqlens_q. Children are pushed in reverse
        # so they pop in insertion order.
        # Then for each sample and each token position p at least 1 whose completion_mask is set: the label is placed at the
        # packed position of node p-1 in shift_labels (others stay -100), and token_index, old_log_probs (sample value at
        # p), advantages (sample advantage) and segment_id (sample index, despite the name) are appended.
        # Inputs: none (uses state). Must have at least one sample.
        # Outputs: dict of tensors (input_ids, position_ids, shift_labels default int64; tree_kv_index, token_index,
        # segment_id long; cu_seqlens int32; old_log_probs/advantages float32; tree_max_seqlens with max segment length and
        # max path length).
        # Exceptions: ValueError from max on an empty row. Side effects: none.
        """
        Lay the trie out in DFS pre-order, and select the loss terms taken over it.

        Unary chains collapse into contiguous segments. A segment ends at a branch or at a sample end, so every sample
        is exactly the concatenation of whole segments along one root-to-node path. A trained token is scored off its
        predecessor's hidden state, so each loss term points at the packed position of that predecessor's node.

        Returns:
            `dict[str, torch.Tensor]`, of `N` forwarded tokens, `S` segments and `M` trained tokens:
                - `input_ids` (`N`): the unique tokens.
                - `position_ids` (`N`): each token's position in its samples, which is its depth in the trie.
                - `shift_labels` (`N`): next token each position predicts, `-100` where it is not trained.
                - `tree_kv_index` (`K`): packed position of every K/V row [`tree_attention`] gathers, the
                  root-to-segment paths of all segments laid end to end.
                - `tree_cu_seqlens_q` (`S + 1`), `tree_cu_seqlens_k` (`S + 1`): segment boundaries in the row, path
                  boundaries in `tree_kv_index`.
                - `tree_max_seqlens` (`2`): longest segment and longest path.
                - `token_index` (`M`): position whose prediction is each trained token. A shared position serves every
                  sample that shares it, so there are more loss terms than labelled positions.
                - `old_log_probs` (`M`), `advantages` (`M`): generator log-probability and advantage of each target.
                - `segment_id` (`M`): index of the sample each target came from.
        """
        input_ids, position_ids, offsets, paths, kv_index, cu_seqlens_k = [], [], [0], [], [], [0]
        packed = [0] * len(self.tokens)
        stack = [(node, None, 0) for node in reversed(self.roots.values())]
        while stack:
            node, parent, depth = stack.pop()
            segment = len(paths)
            start = len(input_ids)
            while True:
                packed[node] = len(input_ids)
                input_ids.append(self.tokens[node])
                position_ids.append(depth)
                depth += 1
                if len(self.children[node]) != 1:
                    break
                node = next(iter(self.children[node].values()))
            offsets.append(len(input_ids))
            paths.append((paths[parent] if parent is not None else []) + list(range(start, len(input_ids))))
            kv_index.extend(paths[-1])
            cu_seqlens_k.append(len(kv_index))
            stack.extend((child, segment, depth) for child in reversed(self.children[node].values()))

        shift_labels = [-100] * len(input_ids)
        token_index, old_log_probs, advantages, segment_id = [], [], [], []
        for i, (sample, nodes) in enumerate(zip(self.samples, self.sample_nodes, strict=True)):
            ids, mask, logps = (sample[key].tolist() for key in ("input_ids", "completion_mask", "old_log_probs"))
            for p in range(1, len(nodes)):
                if mask[p]:
                    shift_labels[packed[nodes[p - 1]]] = ids[p]
                    token_index.append(packed[nodes[p - 1]])
                    old_log_probs.append(logps[p])
                    advantages.append(sample["advantage"])
                    segment_id.append(i)
        return {
            "input_ids": torch.tensor(input_ids),
            "position_ids": torch.tensor(position_ids),
            "shift_labels": torch.tensor(shift_labels),
            "tree_kv_index": torch.tensor(kv_index, dtype=torch.long),
            "tree_cu_seqlens_q": torch.tensor(offsets, dtype=torch.int32),
            "tree_cu_seqlens_k": torch.tensor(cu_seqlens_k, dtype=torch.int32),
            "tree_max_seqlens": torch.tensor(
                [max(b - a for a, b in zip(offsets[:-1], offsets[1:], strict=True)), max(map(len, paths))]
            ),
            "token_index": torch.tensor(token_index, dtype=torch.long),
            "old_log_probs": torch.tensor(old_log_probs, dtype=torch.float32),
            "advantages": torch.tensor(advantages, dtype=torch.float32),
            "segment_id": torch.tensor(segment_id, dtype=torch.long),
        }


TREE_ATTENTION = "tree_fa3"


@cache
def _flash_attn():
    # [FUNCTION DOC]
    # HELPER: Lazily loads and caches (functools.cache) the flash attention kernel from the Hugging Face kernels hub.
    # Behavior: imports kernels.get_kernel; on ROCm (torch.version.hip set) returns "kernels-community/aiter-flash-attn"
    # version 2, otherwise "kernels-community/flash-attn3" version 1.
    # Inputs: none. Outputs: kernel module exposing flash_attn_varlen_func.
    # Exceptions: ImportError if kernels is not installed; hub/download errors. Side effects: may download a kernel
    # (network, disk cache) on first call.
    from kernels import get_kernel

    if torch.version.hip:
        return get_kernel("kernels-community/aiter-flash-attn", version=2)
    return get_kernel("kernels-community/flash-attn3", version=1)


def tree_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    *,
    tree_kv_index,
    tree_cu_seqlens_q,
    tree_cu_seqlens_k,
    tree_max_seqlen_q,
    tree_max_seqlen_k,
    scaling=None,
    sliding_window=None,
    softcap=None,
    **kwargs,
):
    # [FUNCTION DOC]
    # Attention function (transformers AttentionInterface signature) for tree-packed rows.
    # Behavior: casts query/key/value to the target dtype (from get_target_dtype or query dtype), transposes from
    # (batch, heads, seq, dim) to (seq, heads, dim) taking batch element 0 (batch must be 1). Gathers K and V rows by
    # tree_kv_index so each segment's keys are its full root-to-segment path, then calls flash_attn_varlen_func with
    # cu_seqlens for queries (segments) and keys (paths), max lengths, softmax_scale=scaling, causal=True (bottom-right
    # aligned so a segment's queries see the whole prefix path plus causal within the segment), sliding window
    # (window-1, 0) if sliding_window else unlimited, and softcap only when set. attention_mask and extra kwargs are
    # ignored beyond being forwarded.
    # Inputs: module (attention module), query/key/value tensors, attention_mask (unused), keyword tree_* tensors and
    # max seqlens, optional scaling, sliding_window, softcap.
    # Outputs: tuple of (attention output with a leading batch dim of 1, shape (1, seq, heads, dim); None for weights).
    # Exceptions: kernel errors on shape/dtype mismatch. Side effects: may load kernel via _flash_attn.
    dtype = get_target_dtype(query, module) or query.dtype
    query, key, value = (x.transpose(1, 2)[0].to(dtype) for x in (query, key, value))
    out = _flash_attn().flash_attn_varlen_func(
        query,
        key.index_select(0, tree_kv_index),
        value.index_select(0, tree_kv_index),
        tree_cu_seqlens_q,
        tree_cu_seqlens_k,
        tree_max_seqlen_q,
        tree_max_seqlen_k,
        softmax_scale=scaling,
        causal=True,
        window_size=(sliding_window - 1, 0) if sliding_window else (-1, -1),
        **({"softcap": softcap} if softcap else {}),
    )
    out = out[0] if isinstance(out, tuple) else out
    return out.unsqueeze(0), None


def register_tree_attention() -> None:
    # [FUNCTION DOC]
    # Registers tree_attention under the name TREE_ATTENTION ("tree_fa3") in transformers AttentionInterface, and a mask
    # function returning None (no mask built) in AttentionMaskInterface.
    # Inputs: none. Outputs: none. Exceptions: none expected.
    # Side effects: mutates global transformers registries.
    AttentionInterface.register(TREE_ATTENTION, tree_attention)
    AttentionMaskInterface.register(TREE_ATTENTION, lambda **kwargs: None)
