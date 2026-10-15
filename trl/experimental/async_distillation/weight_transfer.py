# [MODULE DOC]
# NCCL weight streaming from the async-distillation trainer (student) to its vLLM server.
# Contents: WeightTransferClient. At import, selects the vLLM API by version: for vLLM 0.28.0 or newer it imports the
# rendezvous helper (aliased open_trainer_endpoint) and packed_nccl_broadcast_producer; for older vLLM it imports the
# static NCCLWeightTransferEngine and NCCLTrainerSendWeightsArgs; also imports get_ip/get_open_port when vLLM is present.
# Import-time side effects: these conditional imports and the module logger; _HAS_STATEFUL_TRAINER_ENGINE flag.

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
    # Drives the trainer side (rank 0) of an NCCL group with the vLLM server workers (ranks 1..N) and pushes weights.
    # State: vllm (VLLMClient), weight_sync_timeout, _weight_update_info (with packed=True added for old vLLM),
    # model_update_group (None until init_weight_transfer succeeds), class constant _CONTROL_TIMEOUT = 300 s.
    # Flow: init_weight_transfer once; per sync: pause, send_weights, resume; destroy at shutdown. Every NCCL call runs on
    # a daemon thread while the matching HTTP call runs on the caller's thread.
    """Streams the student's weights to its own vLLM server over NCCL.

    Ported verbatim from [`~trl.experimental.async_grpo.weight_transfer.WeightTransferClient`]. Only the student is
    ever targeted: the teacher is static and never receives a weight update, so no analogous client exists for it.

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
        # Stores the client, timeout and update info; on vLLM older than 0.28.0 copies the info with packed set to True.
        # Inputs: vllm_client, weight_update_info (dict), weight_sync_timeout (int, default 1800).
        # Outputs: none. Exceptions: ImportError if vLLM is not installed. Side effects: sets attributes, model_update_group None.
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
        # Sets up the NCCL group with the server.
        # Behavior: waits for server readiness, queries its world size; total world size = server world size + 1; picks this
        # host's IP and a free port; builds init_info (master_address, master_port, rank_offset 1, world_size, plus packed
        # True on new vLLM). Starts a daemon thread running the trainer-side init (new or legacy API) storing the group in
        # model_update_group and capturing any exception; meanwhile POSTs init_weight_transfer_engine with
        # weight_sync_timeout, then joins the thread and re-raises a captured thread error.
        # Inputs: none. Outputs: none.
        # Exceptions: any failure is wrapped in RuntimeError with setup hints (VLLM_SERVER_DEV_MODE, nccl backend config);
        # TimeoutError from wait_for_server_ready propagates unwrapped.
        # Side effects: network calls, NCCL group creation, sets model_update_group, logs info.
        self.vllm.wait_for_server_ready()
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
            # HELPER: Thread target creating the trainer NCCL endpoint (open_trainer_endpoint on new vLLM, else
            # NCCLWeightTransferEngine.trainer_init) from the enclosing init_info.
            # Inputs/Outputs: none. Exceptions: none raised; any BaseException is appended to the enclosing error list.
            # Side effects: sets self.model_update_group.
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
        # Streams one full set of weights to the server.
        # Behavior: no-op if model_update_group is None. Calls start_weight_update (control timeout), then starts a daemon thread
        # broadcasting the iterator from rank 0 (new vLLM: packed producer with post_iter_func taking the tensor as the second
        # item of each pair; old vLLM: trainer_send_weights with packed=True), while POSTing update_weights with
        # _weight_update_info; joins and re-raises thread errors. On success calls finish_weight_update and logs the duration
        # at debug level.
        # Inputs: iterator yielding (name, tensor) pairs matching weight_update_info.
        # Outputs: none.
        # Exceptions: on failure (start_weight_update errors are not wrapped) attempts finish_weight_update (errors suppressed)
        # and raises RuntimeError with recovery hints chaining the cause.
        # Side effects: HTTP calls, NCCL broadcast, logging.
        if self.model_update_group is None:
            return
        t0 = time.time()
        # Prepare the workers for the reload; must complete before any weights are sent.
        self.vllm.start_weight_update(timeout=self._CONTROL_TIMEOUT)

        error: list[BaseException] = []

        def trainer_send_weights():
            # [FUNCTION DOC]
            # HELPER: Thread target performing the NCCL broadcast of the enclosing iterator via the version-appropriate vLLM API.
            # Inputs/Outputs: none. Exceptions: none raised; BaseException captured into the enclosing error list.
            # Side effects: NCCL traffic.
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
        # Pauses server generation via VLLMClient.pause with _CONTROL_TIMEOUT; logs the HTTP duration at debug.
        # Inputs/Outputs: none. Exceptions: propagates the client's request errors. Side effects: HTTP call, logging.
        t0 = time.time()
        self.vllm.pause(timeout=self._CONTROL_TIMEOUT)
        logger.debug(f"[weight_sync] pause HTTP took {time.time() - t0:.1f}s")

    def resume(self) -> None:
        # [FUNCTION DOC]
        # Resumes server generation via VLLMClient.resume with _CONTROL_TIMEOUT; logs the HTTP duration at debug.
        # Inputs/Outputs: none. Exceptions: propagates the client's request errors. Side effects: HTTP call, logging.
        t0 = time.time()
        self.vllm.resume(timeout=self._CONTROL_TIMEOUT)
        logger.debug(f"[weight_sync] resume HTTP took {time.time() - t0:.1f}s")

    def destroy(self) -> None:
        # [FUNCTION DOC]
        # Drops the NCCL group: no-op if none; otherwise clears the group's store and socket references and sets
        # model_update_group to None (no explicit communicator teardown).
        # Inputs/Outputs: none. Exceptions: AttributeError if the group object lacks a group attribute.
        # Side effects: mutates the group and self.
        if self.model_update_group is None:
            return
        self.model_update_group.group.store = None
        self.model_update_group.group.socket = None
        self.model_update_group = None
