import threading

import pytest

from localstack.services.lambda_.event_source_mapping.checkpointing.registry import (
    DELIVERY_STATUS_ABANDONED,
    DELIVERY_STATUS_OK,
    DELIVERY_STATUS_RETRYING,
    PHASE_HANDOFF,
    PHASE_LEASED,
    PHASE_RELEASED,
    CheckpointError,
    StreamCheckpointRegistry,
)

STREAM_ARN = "arn:aws:kinesis:us-east-1:000000000000:stream/events"
SHARD_IDS = [f"shardId-{i}" for i in range(4)]
MAPPING_A = "11111111-1111-1111-1111-111111111111"
MAPPING_B = "22222222-2222-2222-2222-222222222222"


class FakeStore:
    def __init__(self):
        self.event_source_mappings = {}
        self.event_source_stream_state = {}


def _enable_mapping(store: FakeStore, uuid: str, state: str = "Enabled"):
    store.event_source_mappings[uuid] = {
        "UUID": uuid,
        "EventSourceArn": STREAM_ARN,
        "State": state,
    }


@pytest.fixture
def store():
    return FakeStore()


@pytest.fixture
def registry(store):
    return StreamCheckpointRegistry(store)


class TestDeclarativeOwnership:
    def test_deterministic_and_stable(self, registry):
        owners_first = [
            registry.declared_owner(shard_id, [MAPPING_A, MAPPING_B]) for shard_id in SHARD_IDS
        ]
        # A second registry with no shared in-memory state computes the exact same assignment
        registry_other = StreamCheckpointRegistry(FakeStore())
        owners_second = [
            registry_other.declared_owner(shard_id, [MAPPING_A, MAPPING_B])
            for shard_id in SHARD_IDS
        ]
        assert owners_first == owners_second
        assert set(owners_first) <= {MAPPING_A, MAPPING_B}

    def test_single_consumer_owns_everything(self, store, registry):
        _enable_mapping(store, MAPPING_A)
        owned = registry.reconcile(STREAM_ARN, SHARD_IDS, MAPPING_A)
        assert owned == set(SHARD_IDS)
        for shard_id in SHARD_IDS:
            lease = store.event_source_stream_state[STREAM_ARN]["shards"][shard_id]
            assert lease["owner"] == MAPPING_A
            assert lease["phase"] == PHASE_LEASED
            assert lease["epoch"] == 1

    def test_no_consumers_does_not_reshuffle(self, registry):
        # Existing leases stay untouched when no enabled mapping exists (e.g. during restart)
        registry.reconcile(STREAM_ARN, SHARD_IDS, None)
        state = registry._states[STREAM_ARN]
        assert state["shards"] == {}


class TestHandoffBoundary:
    def test_handoff_release_claim_boundary(self, store, registry):
        _enable_mapping(store, MAPPING_A)
        registry.reconcile(STREAM_ARN, SHARD_IDS, MAPPING_A)
        # A has consumed up to sequence 10 on every shard
        for shard_id in SHARD_IDS:
            registry.commit(STREAM_ARN, MAPPING_A, shard_id, "10")

        # A second mapping appears: ownership is repartitioned deterministically
        _enable_mapping(store, MAPPING_B)
        consumers = sorted([MAPPING_A, MAPPING_B])
        b_targets = {
            shard_id
            for shard_id in SHARD_IDS
            if registry.declared_owner(shard_id, consumers) == MAPPING_B
        }
        assert b_targets  # B is declared owner of at least one shard
        # B cannot claim yet: it can only flag the pending handoff
        assert registry.reconcile(STREAM_ARN, SHARD_IDS, MAPPING_B) == set()
        for shard_id in SHARD_IDS:
            lease = store.event_source_stream_state[STREAM_ARN]["shards"][shard_id]
            if shard_id in b_targets:
                assert lease["phase"] == PHASE_HANDOFF
                assert lease["owner"] == MAPPING_A
                assert lease["next_owner"] == MAPPING_B
            else:
                assert lease["owner"] == MAPPING_A
                assert lease["phase"] == PHASE_LEASED

        # A reconciles at its poll-cycle boundary: no in-flight batch exists, so it releases
        a_owned = registry.reconcile(STREAM_ARN, SHARD_IDS, MAPPING_A)
        assert a_owned == set(SHARD_IDS) - b_targets
        for shard_id in b_targets:
            lease = store.event_source_stream_state[STREAM_ARN]["shards"][shard_id]
            assert lease["phase"] == PHASE_RELEASED
            # The handoff point is A's last delivered record
            assert lease["sequence_number"] == "10"

        # B claims the released shards and resumes at the checkpoint
        b_owned_after = registry.reconcile(STREAM_ARN, SHARD_IDS, MAPPING_B)
        assert b_owned_after == b_targets
        for shard_id in b_targets:
            lease = store.event_source_stream_state[STREAM_ARN]["shards"][shard_id]
            assert lease["phase"] == PHASE_LEASED
            assert lease["owner"] == MAPPING_B
            assert lease["sequence_number"] == "10"
            # epoch 1 (claim) -> 2 (handoff) -> 3 (re-claim by B)
            assert lease["epoch"] == 3

        # No gap, no duplicate boundary: every shard ends leased to the declared owner
        for shard_id in SHARD_IDS:
            owner = registry.declared_owner(shard_id, consumers)
            lease = store.event_source_stream_state[STREAM_ARN]["shards"][shard_id]
            assert lease["phase"] == PHASE_LEASED
            assert lease["owner"] == owner

    def test_removed_owner_is_force_released(self, store, registry):
        # A starts alone and owns everything, then B joins
        _enable_mapping(store, MAPPING_A)
        registry.reconcile(STREAM_ARN, SHARD_IDS, MAPPING_A)
        _enable_mapping(store, MAPPING_B)
        registry.reconcile(STREAM_ARN, SHARD_IDS, MAPPING_B)
        registry.reconcile(STREAM_ARN, SHARD_IDS, MAPPING_A)
        registry.reconcile(STREAM_ARN, SHARD_IDS, MAPPING_B)

        # Mapping A is deleted (state removed from the store): B takes over immediately,
        # including a shard still recorded mid-handoff as owned by A.
        a_shard = SHARD_IDS[0]
        state = store.event_source_stream_state[STREAM_ARN]
        state["shards"][a_shard] = {
            "owner": MAPPING_A,
            "next_owner": MAPPING_B,
            "epoch": 5,
            "phase": PHASE_HANDOFF,
            "sequence_number": "99",
            "resume_at": None,
            "updated_at": "now",
        }
        del store.event_source_mappings[MAPPING_A]
        registry.release_consumer(MAPPING_A)
        b_owned = registry.reconcile(STREAM_ARN, SHARD_IDS, MAPPING_B)
        assert b_owned == set(SHARD_IDS)
        for shard_id in SHARD_IDS:
            lease = state["shards"][shard_id]
            assert lease["phase"] == PHASE_LEASED
            assert lease["owner"] == MAPPING_B
        # The handover resumes at the last checkpoint, leaving no gap
        assert state["shards"][a_shard]["sequence_number"] == "99"

    def test_foreign_poller_cannot_release_shard(self, store, registry):
        _enable_mapping(store, MAPPING_A)
        registry.reconcile(STREAM_ARN, SHARD_IDS, MAPPING_A)
        # B appears afterwards: initially every shard is leased to A
        _enable_mapping(store, MAPPING_B)
        consumers = sorted([MAPPING_A, MAPPING_B])
        b_shard = next(
            sid for sid in SHARD_IDS if registry.declared_owner(sid, consumers) == MAPPING_B
        )
        # B reconciling flags the handoff but cannot release A's shard itself
        registry.reconcile(STREAM_ARN, [b_shard], MAPPING_B)
        lease = store.event_source_stream_state[STREAM_ARN]["shards"][b_shard]
        assert lease["phase"] == PHASE_HANDOFF
        assert lease["owner"] == MAPPING_A
        # B still cannot process the shard
        assert registry.begin_processing(STREAM_ARN, MAPPING_B, b_shard) is False
        # A releases at its boundary, then B may process
        registry.reconcile(STREAM_ARN, [b_shard], MAPPING_A)
        registry.reconcile(STREAM_ARN, [b_shard], MAPPING_B)
        assert registry.begin_processing(STREAM_ARN, MAPPING_B, b_shard) is True


class TestCheckpointLifecycle:
    def test_resume_after_and_at_sequence_number(self, store, registry):
        _enable_mapping(store, MAPPING_A)
        registry.reconcile(STREAM_ARN, SHARD_IDS, MAPPING_A)

        # No checkpoint yet: fall back to configured StartingPosition
        assert registry.iterator_spec(STREAM_ARN, SHARD_IDS[0]) == (None, None)

        registry.commit(STREAM_ARN, MAPPING_A, SHARD_IDS[0], "10")
        assert registry.iterator_spec(STREAM_ARN, SHARD_IDS[0]) == (
            "AFTER_SEQUENCE_NUMBER",
            "10",
        )

        # A failure rolls the position back to the failed record, exactly once
        registry.rollback(STREAM_ARN, MAPPING_A, SHARD_IDS[0], "11", attempts=1)
        assert registry.iterator_spec(STREAM_ARN, SHARD_IDS[0]) == (
            "AT_SEQUENCE_NUMBER",
            "11",
        )
        # marker consumed: subsequent builds resume after the last commit again
        assert registry.iterator_spec(STREAM_ARN, SHARD_IDS[0]) == (
            "AFTER_SEQUENCE_NUMBER",
            "10",
        )

    def test_commit_must_be_monotonic(self, store, registry):
        _enable_mapping(store, MAPPING_A)
        registry.reconcile(STREAM_ARN, SHARD_IDS, MAPPING_A)
        registry.commit(STREAM_ARN, MAPPING_A, SHARD_IDS[0], "10")
        with pytest.raises(CheckpointError):
            registry.commit(STREAM_ARN, MAPPING_A, SHARD_IDS[0], "9")

    def test_rollback_before_checkpoint_is_ignored(self, store, registry):
        _enable_mapping(store, MAPPING_A)
        registry.reconcile(STREAM_ARN, SHARD_IDS, MAPPING_A)
        registry.commit(STREAM_ARN, MAPPING_A, SHARD_IDS[0], "10")
        registry.rollback(STREAM_ARN, MAPPING_A, SHARD_IDS[0], "5", attempts=1)
        assert registry.iterator_spec(STREAM_ARN, SHARD_IDS[0]) == (
            "AFTER_SEQUENCE_NUMBER",
            "10",
        )

    def test_foreign_owner_cannot_commit(self, store, registry):
        _enable_mapping(store, MAPPING_A)
        registry.reconcile(STREAM_ARN, SHARD_IDS, MAPPING_A)
        with pytest.raises(CheckpointError):
            registry.commit(STREAM_ARN, MAPPING_B, SHARD_IDS[0], "1")

    def test_abandon_advances_checkpoint_and_records_result(self, store, registry):
        _enable_mapping(store, MAPPING_A)
        registry.reconcile(STREAM_ARN, SHARD_IDS, MAPPING_A)
        registry.commit(STREAM_ARN, MAPPING_A, SHARD_IDS[0], "10")
        registry.abandon(
            STREAM_ARN,
            MAPPING_A,
            SHARD_IDS[0],
            last_sequence_number="12",
            reason="RetryAttemptsExhausted",
            attempts=3,
            error={"httpStatusCode": 500},
        )
        lease = store.event_source_stream_state[STREAM_ARN]["shards"][SHARD_IDS[0]]
        assert lease["sequence_number"] == "12"
        assert lease["resume_at"] is None
        results = registry.list_delivery_results(consumer_uuid=MAPPING_A)
        result = next(r for r in results if r["shard_id"] == SHARD_IDS[0])
        assert result["status"] == DELIVERY_STATUS_ABANDONED
        assert result["attempts"] == 3
        assert result["reason"] == "RetryAttemptsExhausted"


class TestDeliveryResultsAndExport:
    def test_results_queryable_ok_retrying_abandoned(self, store, registry):
        _enable_mapping(store, MAPPING_A)
        registry.reconcile(STREAM_ARN, SHARD_IDS, MAPPING_A)
        registry.commit(STREAM_ARN, MAPPING_A, SHARD_IDS[0], "1")
        result = registry.list_delivery_results(consumer_uuid=MAPPING_A)[0]
        assert result["status"] == DELIVERY_STATUS_OK
        assert result["attempts"] == 0

        registry.rollback(
            STREAM_ARN, MAPPING_A, SHARD_IDS[1], "2", attempts=2, reason="FunctionError"
        )
        rows = {r["shard_id"]: r for r in registry.list_delivery_results(consumer_uuid=MAPPING_A)}
        assert rows[SHARD_IDS[1]]["status"] == DELIVERY_STATUS_RETRYING
        assert rows[SHARD_IDS[1]]["attempts"] == 2

    def test_export_import_roundtrip(self, store, registry):
        _enable_mapping(store, MAPPING_A)
        registry.reconcile(STREAM_ARN, SHARD_IDS, MAPPING_A)
        registry.commit(STREAM_ARN, MAPPING_A, SHARD_IDS[0], "42")

        export = registry.export_state()
        assert export["version"] == 1
        assert STREAM_ARN in export["streams"]

        new_store = FakeStore()
        new_registry = StreamCheckpointRegistry(new_store)
        new_registry.import_state(export)
        imported = new_registry.list_checkpoints(STREAM_ARN)
        by_shard = {row["shard_id"]: row for row in imported}
        assert by_shard[SHARD_IDS[0]]["sequence_number"] == "42"
        assert by_shard[SHARD_IDS[0]]["owner"] == MAPPING_A
        assert new_registry.iterator_spec(STREAM_ARN, SHARD_IDS[0]) == (
            "AFTER_SEQUENCE_NUMBER",
            "42",
        )

    def test_import_rejects_invalid_payload(self, registry):
        with pytest.raises(CheckpointError):
            registry.import_state({"version": 99, "streams": {}})
        with pytest.raises(CheckpointError):
            registry.import_state(
                {
                    "version": 1,
                    "streams": {STREAM_ARN: {"shards": {"s": {"phase": "bogus"}}, "results": {}}},
                }
            )


class TestShardLock:
    def test_commit_is_mutually_exclusive_with_delivery_lock(self, store, registry):
        _enable_mapping(store, MAPPING_A)
        registry.reconcile(STREAM_ARN, SHARD_IDS, MAPPING_A)

        committed = threading.Event()

        def commit_worker():
            # Must block until the in-flight "delivery" (the holder of shard_lock) completes.
            with registry.shard_lock(STREAM_ARN, SHARD_IDS[0]):
                registry.commit(STREAM_ARN, MAPPING_A, SHARD_IDS[0], "7")
                committed.set()

        with registry.shard_lock(STREAM_ARN, SHARD_IDS[0]):
            thread = threading.Thread(target=commit_worker)
            thread.start()
            assert not committed.wait(timeout=0.3)
        assert committed.wait(timeout=1)
        thread.join()
        assert (
            store.event_source_stream_state[STREAM_ARN]["shards"][SHARD_IDS[0]]["sequence_number"]
            == "7"
        )
