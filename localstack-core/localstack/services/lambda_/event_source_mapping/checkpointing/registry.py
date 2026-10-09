"""Resumable consumption positions (checkpoints) and declarative shard ownership for streaming
event source mappings (Kinesis, DynamoDB Streams).

Design overview
---------------

For every event source stream the registry keeps, per shard, a *lease* describing which event
source mapping (ESM) currently owns the shard and a *checkpoint* describing the last sequence
number whose batch has been fully delivered to the target.

Ownership is declared deterministically: the set of active (enabled) ESMs consuming a stream is
sorted by UUID and the shard id is hashed onto that set, so every process and every poller
computes the exact same owner without a coordinator.

Ownership changes go through an explicit boundary:

    LEASED -> HANDOFF -> RELEASED -> LEASED (new owner, new epoch)

* The old owner stops fetching as soon as it observes ``HANDOFF``. Any in-flight batch is
  finished and checkpointed first (delivery and checkpoint updates share the same per-shard
  lock), then the lease is marked ``RELEASED``. Records merely buffered in the poller's batcher
  but never delivered are dropped; they lie after the checkpoint and are re-read by the new
  owner.
* The new owner may only claim a ``RELEASED`` shard and resumes at ``AFTER_SEQUENCE_NUMBER`` of
  the shared checkpoint. Therefore a handoff neither redelivers a delivered segment nor leaves
  a gap.
* If the old owner no longer exists (mapping deleted/disabled), no in-flight work can exist and
  the shard is force-released immediately.

All state is stored as plain nested dictionaries on the :class:`LambdaStore`, so it is included
in state snapshots and recovered after worker rebuilds and process restarts. The registry object
itself only holds locks and a back-reference to the store.
"""

import copy
import hashlib
import logging
import threading
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

LOG = logging.getLogger(__name__)

# Lease phases
PHASE_LEASED = "leased"
PHASE_HANDOFF = "handoff"
PHASE_RELEASED = "released"

# Queryable delivery results
DELIVERY_STATUS_OK = "OK"
DELIVERY_STATUS_RETRYING = "RETRYING"
DELIVERY_STATUS_ABANDONED = "ABANDONED"

# ESM states (see EsmState) that represent an active consumer of a stream.
_ACTIVE_STATES = ("Creating", "Enabling", "Enabled", "Updating")

_EXPORT_VERSION = 1
_MAX_ERROR_CHARS = 512


class CheckpointError(Exception):
    """Raised on invalid checkpoint transitions, e.g. non-monotonic commits."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _new_stream_state() -> dict:
    return {"shards": {}, "results": {}}


def _new_lease(owner: str, epoch: int = 1) -> dict:
    return {
        "owner": owner,
        "next_owner": None,
        "epoch": epoch,
        "phase": PHASE_LEASED,
        # Sequence number of the last record of the last fully delivered batch.
        "sequence_number": None,
        # Rollback marker: when set, the owner builds an AT_SEQUENCE_NUMBER iterator at it.
        "resume_at": None,
        "updated_at": _utc_now(),
    }


def _new_result(
    status: str,
    attempts: int,
    last_sequence_number: str | None,
    reason: str | None = None,
    error: str | None = None,
) -> dict:
    return {
        "status": status,
        "attempts": attempts,
        "last_sequence_number": last_sequence_number,
        "reason": reason,
        "error": error,
        "updated_at": _utc_now(),
    }


def _sequence_number_as_int(sequence_number: str) -> int:
    # Both Kinesis and DynamoDB Streams sequence numbers are big numeric strings.
    return int(sequence_number)


class StreamCheckpointRegistry:
    """Holds shard leases, checkpoints and delivery results for all streams of one
    account/region, backed by the corresponding :class:`LambdaStore`."""

    def __init__(self, store):
        self._store = store
        self._locks_guard = threading.RLock()
        self._stream_locks: dict[str, threading.RLock] = {}
        self._shard_locks: dict[tuple[str, str], threading.RLock] = {}

    # ----------------------------------------------------------------------------------
    # Locking
    # ----------------------------------------------------------------------------------

    def _stream_lock_for(self, stream_arn: str) -> threading.RLock:
        with self._locks_guard:
            lock = self._stream_locks.get(stream_arn)
            if lock is None:
                lock = threading.RLock()
                self._stream_locks[stream_arn] = lock
            return lock

    def _shard_lock_for(self, stream_arn: str, shard_id: str) -> threading.RLock:
        key = (stream_arn, shard_id)
        with self._locks_guard:
            lock = self._shard_locks.get(key)
            if lock is None:
                lock = threading.RLock()
                self._shard_locks[key] = lock
            return lock

    @contextmanager
    def shard_lock(self, stream_arn: str, shard_id: str) -> Iterator[None]:
        """Serializes batch delivery against checkpoint updates, failure rollback and ownership
        handoff for one shard. A position can never advance past a batch whose delivery has not
        completed: both happen under this lock."""
        lock = self._shard_lock_for(stream_arn, shard_id)
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

    # ----------------------------------------------------------------------------------
    # Persisted state access
    # ----------------------------------------------------------------------------------

    @property
    def _states(self) -> dict[str, dict]:
        return self._store.event_source_stream_state

    def _get_stream_state(self, stream_arn: str, create: bool = False) -> dict | None:
        state = self._states.get(stream_arn)
        if state is None and create:
            state = _new_stream_state()
            self._states[stream_arn] = state
        return state

    def _active_consumers(self, stream_arn: str) -> list[str]:
        """Sorted UUIDs of enabled ESMs consuming ``stream_arn``. This is the declarative
        membership of the consumer group; it is derived from persisted configuration rather
        than runtime worker registration, so it stays stable across worker rebuilds and
        process restarts."""
        consumers = {
            esm["UUID"]
            for esm in self._store.event_source_mappings.values()
            if esm.get("EventSourceArn") == stream_arn and esm.get("State") in _ACTIVE_STATES
        }
        return sorted(consumers)

    @staticmethod
    def declared_owner(shard_id: str, consumers: list[str]) -> str | None:
        """Deterministically maps a shard to one of the active consumers. Uses sha256 to avoid
        Python's per-process salted ``hash()``."""
        if not consumers:
            return None
        digest = hashlib.sha256(shard_id.encode("utf-8")).hexdigest()
        return consumers[int(digest, 16) % len(consumers)]

    # ----------------------------------------------------------------------------------
    # Ownership declaration / handoff
    # ----------------------------------------------------------------------------------

    def reconcile(
        self, stream_arn: str, shard_ids: list[str], consumer_uuid: str | None = None
    ) -> set[str]:
        """Reconciles declarative ownership for the given shards.

        Drives lease transitions (claim, handoff, release) and returns the set of shards
        currently leased to ``consumer_uuid``. Safe and idempotent to call on every poll
        cycle from every poller.
        """
        with self._stream_lock_for(stream_arn):
            state = self._get_stream_state(stream_arn, create=True)
            leases = state["shards"]
            consumers = self._active_consumers(stream_arn)

            if not consumers:
                # Do not reshuffle or force-release while no enabled consumer is configured
                # (e.g. during a process restart before workers are re-created).
                return set()

            now = _utc_now()
            for shard_id in shard_ids:
                target = self.declared_owner(shard_id, consumers)
                lease = leases.get(shard_id)
                if lease is None:
                    leases[shard_id] = _new_lease(target)
                    continue

                owner = lease["owner"]
                phase = lease["phase"]

                if phase == PHASE_RELEASED:
                    # The previous owner finished all in-flight work at the checkpoint.
                    # The declared target may claim the shard.
                    if target in consumers:
                        lease["owner"] = target
                        lease["next_owner"] = None
                        lease["phase"] = PHASE_LEASED
                        lease["epoch"] += 1
                        lease["updated_at"] = now
                elif owner == target:
                    if phase == PHASE_HANDOFF:
                        # Declaration flipped back to the current owner before it released:
                        # cancel the handoff, no boundary was crossed.
                        lease["phase"] = PHASE_LEASED
                        lease["next_owner"] = None
                        lease["updated_at"] = now
                elif owner not in consumers:
                    # The recorded owner is gone (mapping deleted/disabled) while the shard was
                    # leased or in handoff. Its poller can no longer deliver, so hand the shard
                    # straight to the declared owner without leaving an empty gap.
                    lease["owner"] = target
                    lease["next_owner"] = None
                    lease["phase"] = PHASE_LEASED
                    lease["epoch"] += 1
                    lease["updated_at"] = now
                elif phase == PHASE_LEASED:
                    # Old owner is still active. The owner poller runs reconcile at a poll-cycle
                    # boundary (single-threaded, so no batch of it is in flight at that point)
                    # and releases immediately; other pollers only flag the pending handoff.
                    if consumer_uuid == owner:
                        lease["phase"] = PHASE_RELEASED
                        lease["next_owner"] = target
                    else:
                        lease["phase"] = PHASE_HANDOFF
                        lease["next_owner"] = target
                        lease["epoch"] += 1
                    lease["updated_at"] = now
                elif phase == PHASE_HANDOFF:
                    if consumer_uuid == owner:
                        # Owner observes the pending handoff at a clean boundary: release.
                        lease["phase"] = PHASE_RELEASED
                        lease["next_owner"] = target
                    else:
                        # Keep the declaration up to date while waiting for the owner to release.
                        lease["next_owner"] = target
                    lease["updated_at"] = now

            if consumer_uuid is None:
                return set()
            return {
                shard_id
                for shard_id in shard_ids
                if (lease := leases.get(shard_id))
                and lease["phase"] == PHASE_LEASED
                and lease["owner"] == consumer_uuid
            }

    def begin_processing(self, stream_arn: str, consumer_uuid: str, shard_id: str) -> bool:
        """Called by a poller under :meth:`shard_lock` before fetching/delivering a shard.

        Returns ``True`` when the caller owns a ``LEASED`` shard and may proceed. When the shard
        is in ``HANDOFF`` and the caller is still the recorded owner, the current (about to be
        discarded) poll cycle is the clean boundary: no new delivery starts and the lease is
        marked ``RELEASED``. Everything else (foreign owner, already released) returns
        ``False``.
        """
        with self._stream_lock_for(stream_arn):
            state = self._get_stream_state(stream_arn)
            if state is None:
                return False
            lease = state["shards"].get(shard_id)
            if lease is None or lease["owner"] != consumer_uuid:
                return False
            if lease["phase"] == PHASE_LEASED:
                return True
            if lease["phase"] == PHASE_HANDOFF:
                lease["phase"] = PHASE_RELEASED
                lease["updated_at"] = _utc_now()
                LOG.info(
                    "Shard %s of stream %s released by mapping %s for handoff at sequence %s "
                    "(next owner: %s).",
                    shard_id,
                    stream_arn,
                    consumer_uuid,
                    lease["sequence_number"],
                    lease.get("next_owner"),
                )
            return False

    def release_consumer(self, consumer_uuid: str) -> None:
        """Releases all shards owned by ``consumer_uuid`` (e.g. when the ESM is deleted).

        Configuration already excludes deleting/disabled mappings from the active consumers, so
        their shards are force-released and can be claimed by the remaining mappings.
        """
        for stream_arn, state in list(self._states.items()):
            shard_ids = [
                shard_id
                for shard_id, lease in state.get("shards", {}).items()
                if lease.get("owner") == consumer_uuid
            ]
            if shard_ids:
                self.reconcile(stream_arn, shard_ids)

    # ----------------------------------------------------------------------------------
    # Iterator resolution / checkpoint commits
    # ----------------------------------------------------------------------------------

    def iterator_spec(self, stream_arn: str, shard_id: str) -> tuple[str | None, str | None]:
        """Resolves how the owning poller should create its shard iterator when (re)building
        workers.

        :return: ``(iterator_type, sequence_number)`` where iterator_type is either
                 ``AT_SEQUENCE_NUMBER`` or ``AFTER_SEQUENCE_NUMBER``, or ``(None, None)`` when
                 the caller should fall back to the configured StartingPosition.
        """
        with self.shard_lock(stream_arn, shard_id):
            state = self._get_stream_state(stream_arn)
            if state is None:
                return None, None
            lease = state["shards"].get(shard_id)
            if lease is None:
                return None, None
            now = _utc_now()
            if resume_at := lease.get("resume_at"):
                # Consume the rollback marker exactly once.
                lease["resume_at"] = None
                lease["updated_at"] = now
                return "AT_SEQUENCE_NUMBER", resume_at
            if sequence_number := lease.get("sequence_number"):
                return "AFTER_SEQUENCE_NUMBER", sequence_number
            return None, None

    def commit(
        self,
        stream_arn: str,
        consumer_uuid: str,
        shard_id: str,
        sequence_number: str,
    ) -> None:
        """Advances the checkpoint to ``sequence_number`` after a batch has been fully
        delivered. Only the current owner may commit and the position may never move
        backwards (except via an explicit :meth:`rollback`)."""
        with self.shard_lock(stream_arn, shard_id):
            lease = self._require_lease(stream_arn, consumer_uuid, shard_id)
            current = lease.get("sequence_number")
            if current is not None and _sequence_number_as_int(
                sequence_number
            ) < _sequence_number_as_int(current):
                raise CheckpointError(
                    f"Refusing to move checkpoint backwards on shard {shard_id}: "
                    f"{current} -> {sequence_number}"
                )
            lease["sequence_number"] = sequence_number
            lease["resume_at"] = None
            lease["updated_at"] = _utc_now()
            self._record_result(
                stream_arn,
                consumer_uuid,
                shard_id,
                status=DELIVERY_STATUS_OK,
                attempts=0,
                last_sequence_number=sequence_number,
            )

    def rollback(
        self,
        stream_arn: str,
        consumer_uuid: str,
        shard_id: str,
        failed_sequence_number: str,
        attempts: int,
        error: str | None = None,
        reason: str = "BatchItemFailure",
    ) -> None:
        """Declares that delivery failed at ``failed_sequence_number``: the next iterator (in a
        retry or after a worker rebuild/restart) starts AT that record and redelivers it."""
        with self.shard_lock(stream_arn, shard_id):
            lease = self._require_lease(stream_arn, consumer_uuid, shard_id)
            current = lease.get("sequence_number")
            if current is not None and _sequence_number_as_int(
                failed_sequence_number
            ) <= _sequence_number_as_int(current):
                # The failed record is already within the committed prefix; nothing to roll back.
                LOG.debug(
                    "Ignoring rollback to sequence %s on shard %s already checkpointed at %s.",
                    failed_sequence_number,
                    shard_id,
                    current,
                )
                return
            lease["resume_at"] = failed_sequence_number
            lease["updated_at"] = _utc_now()
            self._record_result(
                stream_arn,
                consumer_uuid,
                shard_id,
                status=DELIVERY_STATUS_RETRYING,
                attempts=attempts,
                last_sequence_number=failed_sequence_number,
                reason=reason,
                error=error,
            )

    def abandon(
        self,
        stream_arn: str,
        consumer_uuid: str,
        shard_id: str,
        last_sequence_number: str,
        reason: str,
        attempts: int,
        error: str | None = None,
    ) -> None:
        """Records final abandonment of failed records (already sent to the DLQ by the poller)
        and advances the checkpoint past them so consumption continues without a gap."""
        with self.shard_lock(stream_arn, shard_id):
            lease = self._require_lease(stream_arn, consumer_uuid, shard_id)
            current = lease.get("sequence_number")
            if current is None or _sequence_number_as_int(
                last_sequence_number
            ) > _sequence_number_as_int(current):
                lease["sequence_number"] = last_sequence_number
            lease["resume_at"] = None
            lease["updated_at"] = _utc_now()
            self._record_result(
                stream_arn,
                consumer_uuid,
                shard_id,
                status=DELIVERY_STATUS_ABANDONED,
                attempts=attempts,
                last_sequence_number=last_sequence_number,
                reason=reason,
                error=error,
            )

    def _require_lease(self, stream_arn: str, consumer_uuid: str, shard_id: str) -> dict:
        state = self._get_stream_state(stream_arn)
        if state is None:
            raise CheckpointError(f"Unknown stream state for {stream_arn}")
        lease = state["shards"].get(shard_id)
        if lease is None:
            raise CheckpointError(f"Unknown shard lease for {stream_arn}/{shard_id}")
        if lease["owner"] != consumer_uuid:
            raise CheckpointError(
                f"Mapping {consumer_uuid} does not own shard {shard_id} of {stream_arn}"
            )
        return lease

    # ----------------------------------------------------------------------------------
    # Delivery results
    # ----------------------------------------------------------------------------------

    def _record_result(
        self,
        stream_arn: str,
        consumer_uuid: str,
        shard_id: str,
        status: str,
        attempts: int,
        last_sequence_number: str | None,
        reason: str | None = None,
        error: str | None = None,
    ) -> None:
        state = self._get_stream_state(stream_arn, create=True)
        results = state["results"].setdefault(consumer_uuid, {})
        if error is not None:
            error = str(error)[:_MAX_ERROR_CHARS]
        results[shard_id] = _new_result(
            status=status,
            attempts=attempts,
            last_sequence_number=last_sequence_number,
            reason=reason,
            error=error,
        )

    # ----------------------------------------------------------------------------------
    # Query / export / import
    # ----------------------------------------------------------------------------------

    def list_checkpoints(
        self, stream_arn: str | None = None, consumer_uuid: str | None = None
    ) -> list[dict]:
        """Returns queryable per-shard positions and ownership."""
        rows = []
        for arn, state in self._iter_states(stream_arn):
            for shard_id, lease in state.get("shards", {}).items():
                if consumer_uuid and lease.get("owner") != consumer_uuid:
                    continue
                rows.append(
                    {
                        "stream_arn": arn,
                        "shard_id": shard_id,
                        "owner": lease.get("owner"),
                        "phase": lease.get("phase"),
                        "epoch": lease.get("epoch"),
                        "sequence_number": lease.get("sequence_number"),
                        "resume_at": lease.get("resume_at"),
                        "updated_at": lease.get("updated_at"),
                    }
                )
        return rows

    def list_delivery_results(
        self, stream_arn: str | None = None, consumer_uuid: str | None = None
    ) -> list[dict]:
        """Returns queryable retry/abandonment results."""
        rows = []
        for arn, state in self._iter_states(stream_arn):
            for esm_uuid, shard_results in state.get("results", {}).items():
                if consumer_uuid and esm_uuid != consumer_uuid:
                    continue
                for shard_id, result in shard_results.items():
                    rows.append(
                        {
                            "stream_arn": arn,
                            "esm_uuid": esm_uuid,
                            "shard_id": shard_id,
                            **copy.deepcopy(result),
                        }
                    )
        return rows

    def export_state(self, stream_arn: str | None = None) -> dict:
        """Exports a JSON-serializable snapshot of all positions, ownership leases and delivery
        results (optionally restricted to one stream)."""
        streams = {}
        for arn, state in self._iter_states(stream_arn):
            streams[arn] = copy.deepcopy(state)
        return {"version": _EXPORT_VERSION, "streams": streams}

    def import_state(self, data: dict) -> None:
        """Imports previously exported state, replacing current state for the included streams.
        Used to recover consumption positions after a reset or on a different instance."""
        if not isinstance(data, dict) or data.get("version") != _EXPORT_VERSION:
            raise CheckpointError(f"Unsupported checkpoint export format: {data!r:.200}")
        exported_streams = data.get("streams")
        if not isinstance(exported_streams, dict):
            raise CheckpointError("Checkpoint export must contain a 'streams' dict")
        for arn, exported_state in exported_streams.items():
            self._validate_stream_state(exported_state)
            with self._stream_lock_for(arn):
                self._states[arn] = copy.deepcopy(exported_state)

    @staticmethod
    def _validate_stream_state(state: dict) -> None:
        if not isinstance(state, dict):
            raise CheckpointError("stream state must be a dict")
        shards = state.get("shards")
        results = state.get("results")
        if not isinstance(shards, dict) or not isinstance(results, dict):
            raise CheckpointError("stream state must contain 'shards' and 'results' dicts")
        for shard_id, lease in shards.items():
            if not isinstance(shard_id, str) or not isinstance(lease, dict):
                raise CheckpointError(f"invalid lease entry: {shard_id!r}")
            if lease.get("phase") not in (
                PHASE_LEASED,
                PHASE_HANDOFF,
                PHASE_RELEASED,
            ):
                raise CheckpointError(f"invalid lease phase for shard {shard_id!r}")

    def _iter_states(self, stream_arn: str | None = None) -> Iterator[tuple[str, dict]]:
        if stream_arn is not None:
            state = self._states.get(stream_arn)
            if state is not None:
                yield stream_arn, state
            return
        for arn, state in list(self._states.items()):
            yield arn, state


# --------------------------------------------------------------------------------------
# Process-wide registry access (one registry per account/region store)
# --------------------------------------------------------------------------------------

_registries: "weakref.WeakKeyDictionary[object, StreamCheckpointRegistry]" = (
    weakref.WeakKeyDictionary()
)
_registries_guard = threading.Lock()


def get_stream_checkpoint_registry(store) -> StreamCheckpointRegistry:
    with _registries_guard:
        registry = _registries.get(store)
        if registry is None:
            registry = StreamCheckpointRegistry(store)
            _registries[store] = registry
        return registry


def reset_stream_checkpoint_registries() -> None:
    """Clears runtime registry state (locks). Called on state reset."""
    with _registries_guard:
        _registries.clear()
