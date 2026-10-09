"""Unit tests for resumable checkpoints and declarative shard ownership in StreamPoller.

These tests drive real KinesisPoller instances against fakes for the Kinesis data plane and the
event processor, so the poll/filter/deliver/retry code paths are exercised end to end.
"""

from datetime import UTC, datetime

import pytest

from localstack.services.lambda_.event_source_mapping.checkpointing.registry import (
    DELIVERY_STATUS_ABANDONED,
    PHASE_LEASED,
    StreamCheckpointRegistry,
    get_stream_checkpoint_registry,
)
from localstack.services.lambda_.event_source_mapping.event_processor import BatchFailureError
from localstack.services.lambda_.event_source_mapping.pollers.kinesis_poller import KinesisPoller
from localstack.services.lambda_.event_source_mapping.pollers.poller import (
    EmptyPollResultsException,
)

STREAM_ARN = "arn:aws:kinesis:us-east-1:000000000000:stream/events"
MAPPING_A = "11111111-1111-1111-1111-111111111111"
MAPPING_B = "22222222-2222-2222-2222-222222222222"
SHARD_IDS = [f"shardId-{i}" for i in range(4)]


class FakeStore:
    def __init__(self):
        self.event_source_mappings = {}
        self.event_source_stream_state = {}


def enable_mapping(store, uuid, state="Enabled"):
    store.event_source_mappings[uuid] = {
        "UUID": uuid,
        "EventSourceArn": STREAM_ARN,
        "State": state,
    }


def make_records(shard_ids, per_shard):
    now = datetime(2024, 1, 1, tzinfo=UTC)
    return {
        shard_id: [
            {
                "SequenceNumber": str(seq),
                "PartitionKey": "p",
                "Data": f"{shard_id}-{seq}".encode(),
                "ApproximateArrivalTimestamp": now,
            }
            for seq in range(1, per_shard + 1)
        ]
        for shard_id in shard_ids
    }


class FakeKinesisExceptions:
    class ExpiredIteratorException(Exception):
        pass


class FakeKinesisClient:
    """Minimal Kinesis data plane: iterator tokens encode ``shardId:position``."""

    def __init__(self, shard_records):
        self.shard_records = shard_records
        self.exceptions = FakeKinesisExceptions()
        self.iterator_calls = []
        self.get_records_calls = 0

    def describe_stream(self, StreamARN):
        return {
            "StreamDescription": {
                "StreamStatus": "ACTIVE",
                "Shards": [{"ShardId": shard_id} for shard_id in self.shard_records],
            }
        }

    def get_shard_iterator(
        self, StreamARN, ShardId, ShardIteratorType, StartingSequenceNumber=None, **kwargs
    ):
        self.iterator_calls.append(
            {"shard": ShardId, "type": ShardIteratorType, "seq": StartingSequenceNumber}
        )
        records = self.shard_records[ShardId]
        if ShardIteratorType == "TRIM_HORIZON":
            position = 0
        elif ShardIteratorType == "LATEST":
            position = len(records)
        elif ShardIteratorType == "AT_SEQUENCE_NUMBER":
            position = self._index_of(ShardId, StartingSequenceNumber)
        elif ShardIteratorType == "AFTER_SEQUENCE_NUMBER":
            position = self._index_of(ShardId, StartingSequenceNumber) + 1
        else:
            raise AssertionError(f"unexpected iterator type {ShardIteratorType}")
        return {"ShardIterator": f"{ShardId}:{position}"}

    def _index_of(self, shard_id, sequence_number):
        for index, record in enumerate(self.shard_records[shard_id]):
            if record["SequenceNumber"] == sequence_number:
                return index
        raise AssertionError(f"unknown sequence number {sequence_number} on {shard_id}")

    def get_records(self, ShardIterator, Limit, **kwargs):
        self.get_records_calls += 1
        shard_id, position_s = ShardIterator.rsplit(":", 1)
        position = int(position_s)
        records = self.shard_records[shard_id][position : position + Limit]
        return {
            "Records": records,
            "NextShardIterator": f"{shard_id}:{position + len(records)}",
        }


class FakeProcessor:
    def __init__(self, fail_first=0):
        self.delivered = []
        self.calls = 0
        self.fail_first = fail_first

    def process_events_batch(self, events):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise BatchFailureError(error={"httpStatusCode": 500, "requestId": "r"})
        self.delivered.extend(event["eventID"] for event in events)

    def generate_event_failure_context(self, abort_condition, **kwargs):
        return {}


def source_parameters(batch_size=2, max_retries=-1, max_record_age=-1, starting="TRIM_HORIZON"):
    return {
        "KinesisStreamParameters": {
            "StartingPosition": starting,
            "BatchSize": batch_size,
            "MaximumBatchingWindowInSeconds": 0,
            "MaximumRetryAttempts": max_retries,
            "MaximumRecordAgeInSeconds": max_record_age,
        }
    }


def make_poller(store, client, uuid, processor, checkpointer=None, **params):
    registry = checkpointer
    if checkpointer is None:
        registry = get_stream_checkpoint_registry(store)
    return KinesisPoller(
        source_arn=STREAM_ARN,
        source_parameters=source_parameters(**params),
        source_client=client,
        processor=processor,
        esm_uuid=uuid,
        kinesis_namespace=False,
        checkpointer=registry,
    )


def poll_once(poller):
    try:
        poller.poll_events()
    except EmptyPollResultsException:
        pass


@pytest.fixture(autouse=True)
def reset_registries():
    yield
    from localstack.services.lambda_.event_source_mapping.checkpointing.registry import (
        reset_stream_checkpoint_registries,
    )

    reset_stream_checkpoint_registries()


class TestCheckpointResume:
    def test_worker_rebuild_resumes_without_duplicates(self):
        store = FakeStore()
        enable_mapping(store, MAPPING_A)
        client = FakeKinesisClient(make_records(["shardId-0"], 2))
        processor = FakeProcessor()
        poller = make_poller(store, client, MAPPING_A, processor)

        poll_once(poller)
        assert processor.delivered == ["shardId-0:1", "shardId-0:2"]

        checkpoints = get_stream_checkpoint_registry(store).list_checkpoints(STREAM_ARN)
        assert checkpoints[0]["sequence_number"] == "2"

        # Simulate a worker rebuild (update / function version switch / process restart): a
        # brand new poller with a fresh data-plane client resumes at the persisted position.
        rebuilt_client = FakeKinesisClient(make_records(["shardId-0"], 2))
        rebuilt_processor = FakeProcessor()
        rebuilt = make_poller(store, rebuilt_client, MAPPING_A, rebuilt_processor)
        poll_once(rebuilt)

        assert rebuilt_client.iterator_calls == [
            {"shard": "shardId-0", "type": "AFTER_SEQUENCE_NUMBER", "seq": "2"}
        ]
        assert rebuilt_processor.delivered == []
        # The segment was delivered exactly once overall
        assert processor.delivered + rebuilt_processor.delivered == [
            "shardId-0:1",
            "shardId-0:2",
        ]

    def test_retry_failure_then_success(self):
        store = FakeStore()
        enable_mapping(store, MAPPING_A)
        client = FakeKinesisClient(make_records(["shardId-0"], 2))
        processor = FakeProcessor(fail_first=1)
        poller = make_poller(store, client, MAPPING_A, processor, max_retries=10)

        poll_once(poller)

        assert processor.calls == 2
        assert processor.delivered == ["shardId-0:1", "shardId-0:2"]
        result = get_stream_checkpoint_registry(store).list_delivery_results(
            consumer_uuid=MAPPING_A
        )[0]
        assert result["status"] == "OK"

    def test_abandonment_after_retry_exhausted_is_queryable_and_skips_segment(self):
        store = FakeStore()
        enable_mapping(store, MAPPING_A)
        client = FakeKinesisClient(make_records(["shardId-0"], 4))
        processor = FakeProcessor(fail_first=10_000)  # never succeeds
        poller = make_poller(store, client, MAPPING_A, processor, max_retries=0)

        poll_once(poller)
        assert processor.delivered == []

        registry = get_stream_checkpoint_registry(store)
        result = registry.list_delivery_results(consumer_uuid=MAPPING_A)[0]
        assert result["status"] == DELIVERY_STATUS_ABANDONED
        assert result["reason"] == "RetryAttemptsExhausted"
        assert result["attempts"] >= 1
        checkpoint = registry.list_checkpoints(STREAM_ARN)[0]
        assert checkpoint["sequence_number"] == "2"

        # A rebuilt poller skips the abandoned segment and continues after it (no stuck shard)
        rebuilt_client = FakeKinesisClient(make_records(["shardId-0"], 4))
        rebuilt_processor = FakeProcessor()
        rebuilt = make_poller(store, rebuilt_client, MAPPING_A, rebuilt_processor)
        poll_once(rebuilt)
        assert rebuilt_client.iterator_calls[0]["type"] == "AFTER_SEQUENCE_NUMBER"
        assert rebuilt_client.iterator_calls[0]["seq"] == "2"
        assert rebuilt_processor.delivered == ["shardId-0:3", "shardId-0:4"]


class TestShardOwnership:
    def test_two_mappings_handoff_without_duplicates_or_gaps(self):
        store = FakeStore()
        enable_mapping(store, MAPPING_A)
        records = make_records(SHARD_IDS, 4)
        client_a = FakeKinesisClient(records)
        processor_a = FakeProcessor()
        poller_a = make_poller(store, client_a, MAPPING_A, processor_a)

        # A consumes two shard-rounds alone: shard0 seq1-2, shard1 seq1-2
        poll_once(poller_a)
        poll_once(poller_a)

        # A second mapping on the same stream appears
        enable_mapping(store, MAPPING_B)
        client_b = FakeKinesisClient(records)
        processor_b = FakeProcessor()
        poller_b = make_poller(store, client_b, MAPPING_B, processor_b)

        delivered_before = set(processor_a.delivered)
        # Drive both poll cycles until nothing new is delivered for several rounds
        idle_rounds = 0
        for _ in range(200):
            poll_once(poller_a)
            poll_once(poller_b)
            delivered_now = set(processor_a.delivered) | set(processor_b.delivered)
            if delivered_now == delivered_before:
                idle_rounds += 1
            else:
                idle_rounds = 0
            delivered_before = delivered_now
            if idle_rounds >= 8:
                break

        registry = get_stream_checkpoint_registry(store)
        delivered_a = processor_a.delivered
        delivered_b = processor_b.delivered

        # Every record of every shard delivered exactly once (no duplicate, no gap)
        all_delivered = delivered_a + delivered_b
        expected = {f"{shard_id}:{seq}" for shard_id in SHARD_IDS for seq in range(1, 5)}
        assert set(all_delivered) == expected
        assert len(all_delivered) == len(expected)
        # No record was delivered by both mappings
        assert not (set(delivered_a) & set(delivered_b))

        # All leases converge: leased, owned by the declared owner, checkpointed at the end
        consumers = sorted([MAPPING_A, MAPPING_B])
        for row in registry.list_checkpoints(STREAM_ARN):
            assert row["phase"] == PHASE_LEASED
            assert row["owner"] == registry.declared_owner(row["shard_id"], consumers)
            assert row["sequence_number"] == "4"

        # Export/re-import keeps the resumable state intact
        export = registry.export_state()
        new_store = FakeStore()
        StreamCheckpointRegistry(new_store).import_state(export)
        imported = StreamCheckpointRegistry(new_store).list_checkpoints(STREAM_ARN)
        assert len(imported) == 4
        assert {row["sequence_number"] for row in imported} == {"4"}
        for row in imported:
            assert row["owner"] == registry.declared_owner(row["shard_id"], consumers)

    def test_single_mapping_owns_all_shards_after_restart(self):
        store = FakeStore()
        enable_mapping(store, MAPPING_A)
        records = make_records(SHARD_IDS, 2)
        client = FakeKinesisClient(records)
        processor = FakeProcessor()
        poller = make_poller(store, client, MAPPING_A, processor)
        for _ in range(6):
            poll_once(poller)

        registry = get_stream_checkpoint_registry(store)
        rows = registry.list_checkpoints(STREAM_ARN)
        assert len(rows) == 4
        assert {row["owner"] for row in rows} == {MAPPING_A}
        # Process restart: a fresh registry over the same persisted store keeps ownership/positions
        restarted_registry = StreamCheckpointRegistry(store)
        for shard_id in SHARD_IDS:
            assert restarted_registry.iterator_spec(STREAM_ARN, shard_id) == (
                "AFTER_SEQUENCE_NUMBER",
                "2",
            )


class TestLegacySemantics:
    def test_without_checkpointer_positions_are_not_persisted(self):
        records = make_records(["shardId-0"], 2)
        client = FakeKinesisClient(records)
        processor = FakeProcessor()
        poller = KinesisPoller(
            source_arn=STREAM_ARN,
            source_parameters=source_parameters(),
            source_client=client,
            processor=processor,
            esm_uuid=MAPPING_A,
            kinesis_namespace=False,
        )
        poll_once(poller)
        assert processor.delivered == ["shardId-0:1", "shardId-0:2"]

        # Legacy behavior on rebuild: restart from the configured StartingPosition and re-read
        rebuilt_client = FakeKinesisClient(records)
        rebuilt_processor = FakeProcessor()
        rebuilt = KinesisPoller(
            source_arn=STREAM_ARN,
            source_parameters=source_parameters(),
            source_client=rebuilt_client,
            processor=rebuilt_processor,
            esm_uuid=MAPPING_A,
            kinesis_namespace=False,
        )
        poll_once(rebuilt)
        assert rebuilt_client.iterator_calls[0]["type"] == "TRIM_HORIZON"
        assert rebuilt_processor.delivered == ["shardId-0:1", "shardId-0:2"]
