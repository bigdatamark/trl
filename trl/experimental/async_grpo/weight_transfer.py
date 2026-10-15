# [MODULE DOC]
# NCCL weight streaming from the async GRPO trainer process into an external vLLM server.
# Main contents: WeightTransferClient, which pairs HTTP control calls (through VLLMClient) with NCCL broadcasts run
# on daemon threads.
# Supports two vLLM APIs: from vLLM 0.28.0 the module-level trainer_init and packed_nccl_broadcast_producer helpers
# (with packed set in the init info); before that, the static NCCLWeightTransferEngine methods (packed set in the
# per-update info).
# Import-time side effects: conditionally imports vLLM networking and weight-transfer helpers; computes the module
# flag _HAS_STATEFUL_TRAINER_ENGINE; creates a logger.

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

import contextlib
import threading
import time
from collections import Counter

from accelerate.logging import get_logger

from ...import_utils import is_vllm_available
from .vllm_client import VLLMClient


if is_vllm_available():
    from vllm.utils.network_utils import get_ip, get_open_port

# vLLM 0.28.0 (vllm-project/vllm#50902) replaced the static `NCCLWeightTransferEngine.trainer_init` /
# `trainer_send_weights` with a stateful trainer engine and moved `packed` from the per-update info to the init info.
# The rendezvous and packed-broadcast helpers the statics were built on are still exposed, so they are called directly.
_HAS_STATEFUL_TRAINER_ENGINE = is_vllm_available(min_version="0.28.0")
if _HAS_STATEFUL_TRAINER_ENGINE:
    from vllm.distributed.weight_transfer.nccl_common import trainer_init as open_trainer_endpoint
    from vllm.distributed.weight_transfer.packed_tensor import packed_nccl_broadcast_producer
elif is_vllm_available():
    from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerSendWeightsArgs, NCCLWeightTransferEngine


logger = get_logger(__name__)


class WeightTransferClient:
    # [CLASS DOC]
    # Streams trainer weights to the vLLM server over an NCCL group in which the trainer is rank 0 and server workers
    # take ranks starting at 1.
    # State: vllm (VLLMClient), weight_sync_timeout (seconds for handshake, transfer and finalisation),
    # _weight_update_info (names/dtypes/shapes; gets packed True added on old vLLM), model_update_group (NCCL group handle
    # or None), class constant _CONTROL_TIMEOUT 300 s for pause/resume/start-update.
    # Lifecycle: init_weight_transfer once, then for each sync pause, send_weights, resume; destroy at shutdown.
    # Each NCCL operation runs on a daemon thread while the HTTP request is issued on the caller thread, so HTTP errors
    # surface instead of NCCL hanging forever.
    """Streams the trainer's weights into the vLLM server over NCCL.

    Each transfer runs its NCCL side on a daemon thread while the HTTP request that drives the server stays on the main
    thread: NCCL has no timeout and cannot be interrupted, so a failure surfaces as the HTTP error rather than a hang,
    and the abandoned thread dies with the process.

    Args:
        vllm_client ([`VLLMClient`]):
            Client for the vLLM server that receives the weights.
        weight_update_info (`dict`):
            Names, dtypes and shapes of the parameters to send. The server sizes its receive buffers from it, so it
            must describe exactly what [`send_weights`] streams.
        weight_sync_timeout (`int`, *optional*, defaults to `1800`):
            Seconds allowed for the steps that scale with model size: the NCCL handshake, the transfer itself, and the
            finalisation that follows it. Pause, resume and the reload setup are bounded by `_CONTROL_TIMEOUT` instead.
    """

    _CONTROL_TIMEOUT = 300

    def __init__(
        self,
        vllm_client: VLLMClient,
        weight_update_info: dict,
        weight_sync_timeout: int = 1800,
    ):
        # [FUNCTION DOC]
        # Behavior: checks vLLM is installed and stores configuration. On vLLM older than 0.28.0, copies weight_update_info
        # with packed True added.
        # Inputs: vllm_client (VLLMClient), weight_update_info (dict), weight_sync_timeout (int, default 1800).
        # Outputs: none.
        # Exceptions: ImportError if vLLM is unavailable.
        # Side effects: sets vllm, weight_sync_timeout, _weight_update_info, model_update_group None.
        if not is_vllm_available():
            raise ImportError("vLLM is required to use WeightTransferClient. Install it with: pip install vllm")
        self.vllm = vllm_client
        self.weight_sync_timeout = weight_sync_timeout
        self._weight_update_info = weight_update_info
        if not _HAS_STATEFUL_TRAINER_ENGINE:
            self._weight_update_info = {**weight_update_info, "packed": True}
        self.model_update_group = None

    def init_weight_transfer(self) -> None:
        # [FUNCTION DOC]
        # Behavior: waits for the server to be healthy; compares the most common dtype in weight_update_info dtype_names
        # with the server dtype (with "torch." removed) and logs a warning on mismatch; queries the inference world size and
        # builds init_info with this host's IP, a free port, rank_offset 1 and world_size equal to inference world size plus
        # one (plus packed True on vLLM 0.28.0 or newer). Starts a daemon thread that opens the trainer NCCL endpoint (storing
        # the group in model_update_group, capturing any exception), then issues the server-side init HTTP call, joins the
        # thread and re-raises a captured error.
        # Inputs: none. Outputs: none.
        # Exceptions: TimeoutError from waiting for the server; RuntimeError wrapping any failure of the HTTP call or the
        # NCCL thread, with setup hints.
        # Side effects: network requests, NCCL group creation, sets model_update_group, warning/info logs.
        self.vllm.wait_for_server_ready()
        # Trainer and server precisions should agree: the precision gap between the two biases the importance ratio
        # https://huggingface.co/papers/2510.26788
        # https://huggingface.co/spaces/aminediroHF/trainer-generator-bf16-mismatch
        train_dtype = Counter(self._weight_update_info["dtype_names"]).most_common(1)[0][0]
        vllm_dtype = self.vllm.get_dtype().removeprefix("torch.")
        if vllm_dtype != train_dtype:
            logger.warning(
                f"The vLLM server serves in {vllm_dtype} but the weights sent to it are {train_dtype}. Set `dtype` in "
                f"`AsyncGRPOConfig` to '{vllm_dtype}', or start the server with `--dtype {train_dtype}`."
            )
        inference_world_size = self.vllm.get_world_size()
        world_size = inference_world_size + 1
        master_address = get_ip()
        master_port = get_open_port()
        init_info = {
            "master_address": master_address,
            "master_port": master_port,
            "rank_offset": 1,
            "world_size": world_size,
        }
        if _HAS_STATEFUL_TRAINER_ENGINE:
            init_info["packed"] = True

        error: list[BaseException] = []

        def trainer_init():
            # [FUNCTION DOC]
            # HELPER: Thread target that creates the trainer side of the NCCL group.
            # Behavior: uses open_trainer_endpoint on new vLLM or NCCLWeightTransferEngine.trainer_init on old vLLM, assigning
            # the result to the enclosing self.model_update_group.
            # Inputs: init_info from the enclosing scope. Outputs: none.
            # Exceptions: none raised; any BaseException is appended to the enclosing error list.
            # Side effects: sets model_update_group; mutates error list.
            try:
                if _HAS_STATEFUL_TRAINER_ENGINE:
                    self.model_update_group = open_trainer_endpoint(init_info)
                else:
                    self.model_update_group = NCCLWeightTransferEngine.trainer_init(init_info)
            except BaseException as exc:  # noqa: BLE001
                error.append(exc)

        thread = threading.Thread(target=trainer_init, daemon=True)
        thread.start()
        try:
            self.vllm.init_weight_transfer_engine(init_info, timeout=self.weight_sync_timeout)
            thread.join()
            if error:
                raise error[0]
        except Exception as exc:
            raise RuntimeError(
                f"Failed to set up the NCCL weight-transfer group with the vLLM server at {self.vllm.server_url}. "
                "Check that the server was started with `VLLM_SERVER_DEV_MODE=1` and "
                '`--weight-transfer-config \'{"backend":"nccl"}\'`, and that the trainer can reach it on the '
                "port it advertises."
            ) from exc
        logger.info("Initialised weight-transfer NCCL group with vLLM")

    def send_weights(self, iterator) -> None:
        # [FUNCTION DOC]
        # Behavior: no-op if model_update_group is None. Otherwise calls start_weight_update (control timeout), starts a
        # daemon thread that broadcasts the (name, tensor) iterator over NCCL, then issues the blocking update_weights HTTP
        # call with _weight_update_info, joins the thread and re-raises a captured thread error. On any failure, attempts
        # finish_weight_update while suppressing its errors, then raises. On success calls finish_weight_update and logs the
        # elapsed time at debug level.
        # Inputs: iterator yielding (name, tensor) pairs that must match weight_update_info exactly.
        # Outputs: none.
        # Exceptions: RuntimeError wrapping any HTTP or NCCL failure, with OOM guidance; exceptions from start_weight_update
        # or the final finish_weight_update propagate unwrapped.
        # Side effects: network calls, NCCL broadcast, server weights replaced, debug log.
        if self.model_update_group is None:
            return
        t0 = time.time()
        # Prepare the workers for the reload; must complete before any weights are sent.
        self.vllm.start_weight_update(timeout=self._CONTROL_TIMEOUT)

        error: list[BaseException] = []

        def trainer_send_weights():
            # [FUNCTION DOC]
            # HELPER: Thread target that performs the NCCL broadcast from rank 0.
            # Behavior: on new vLLM calls packed_nccl_broadcast_producer with the iterator, the group, source 0 and a post-iter
            # function selecting the tensor from each pair; on old vLLM calls NCCLWeightTransferEngine.trainer_send_weights with
            # packed True.
            # Inputs: iterator and group from the enclosing scope. Outputs: none.
            # Exceptions: none raised; any BaseException is appended to the enclosing error list.
            # Side effects: consumes the iterator; NCCL traffic.
            try:
                if _HAS_STATEFUL_TRAINER_ENGINE:
                    packed_nccl_broadcast_producer(
                        iterator=iterator, group=self.model_update_group, src=0, post_iter_func=lambda item: item[1]
                    )
                else:
                    NCCLWeightTransferEngine.trainer_send_weights(
                        iterator=iterator,
                        trainer_args=NCCLTrainerSendWeightsArgs(group=self.model_update_group, packed=True),
                    )
            except BaseException as exc:  # noqa: BLE001
                error.append(exc)

        thread = threading.Thread(target=trainer_send_weights, daemon=True)
        thread.start()
        try:
            self.vllm.update_weights(self._weight_update_info, timeout=self.weight_sync_timeout)
            thread.join()
            if error:
                raise error[0]
        except Exception as exc:
            # Best-effort: a failure to clean up must not mask the error below.
            with contextlib.suppress(Exception):
                self.vllm.finish_weight_update(timeout=self.weight_sync_timeout)
            raise RuntimeError(
                f"Weight sync to the vLLM server at {self.vllm.server_url} failed. The server keeps the partially "
                "updated weights but stays paused, and a new run re-sends every parameter at start-up. If the "
                "underlying error is a CUDA OOM on the server, restart it with a lower `--gpu-memory-utilization`: "
                "receiving a transfer needs a few GB of headroom on top of the weights and the KV cache."
            ) from exc
        self.vllm.finish_weight_update(timeout=self.weight_sync_timeout)
        logger.debug(f"[weight_sync] send_weights took {time.time() - t0:.1f}s")

    def pause(self) -> None:
        # [FUNCTION DOC]
        # Behavior: calls the server pause endpoint with _CONTROL_TIMEOUT and logs the duration at debug level.
        # Inputs: none. Outputs: none.
        # Exceptions: propagates VLLMClient.pause errors.
        # Side effects: network request; server paused; debug log.
        t0 = time.time()
        self.vllm.pause(timeout=self._CONTROL_TIMEOUT)
        logger.debug(f"[weight_sync] pause HTTP took {time.time() - t0:.1f}s")

    def resume(self) -> None:
        # [FUNCTION DOC]
        # Behavior: calls the server resume endpoint with _CONTROL_TIMEOUT and logs the duration at debug level.
        # Inputs: none. Outputs: none.
        # Exceptions: propagates VLLMClient.resume errors.
        # Side effects: network request; server resumed; debug log.
        t0 = time.time()
        self.vllm.resume(timeout=self._CONTROL_TIMEOUT)
        logger.debug(f"[weight_sync] resume HTTP took {time.time() - t0:.1f}s")

    def destroy(self) -> None:
        # [FUNCTION DOC]
        # Behavior: no-op if no group. Otherwise clears the group's store and socket references and sets model_update_group
        # to None so the NCCL resources can be released.
        # Inputs: none. Outputs: none.
        # Exceptions: AttributeError if the group object lacks a group attribute (not expected).
        # Side effects: mutates the group object and clears model_update_group.
        if self.model_update_group is None:
            return
        self.model_update_group.group.store = None
        self.model_update_group.group.socket = None
        self.model_update_group = None
