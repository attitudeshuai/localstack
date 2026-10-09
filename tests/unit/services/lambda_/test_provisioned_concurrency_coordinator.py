"""Unit tests for the incremental provisioned concurrency coordinator:

* delta-only scale up/down with environment reuse
* MERGE / QUEUE policies for declarations arriving mid-adjustment
* queryable progress and terminal futures
* failure rollback to the pre-adjustment state
* FIFO reclaim that never stops an in-flight environment and closes the account
"""

import threading
import time
from types import SimpleNamespace

import pytest

from localstack.aws.api.lambda_ import (
    ProvisionedConcurrencyStatusEnum,
    ServiceException,
)
from localstack.services.lambda_.invocation import assignment as assignment_module
from localstack.services.lambda_.invocation.assignment import AssignmentService
from localstack.services.lambda_.invocation.counting_service import CountingService
from localstack.services.lambda_.invocation.execution_environment import RuntimeStatus
from localstack.services.lambda_.invocation.lambda_models import (
    Function,
    InitializationType,
    ProvisionedConcurrencyConfiguration,
    VersionIdentifier,
)
from localstack.services.lambda_.invocation.provisioned_concurrency import (
    ProvisionedConcurrencyCoordinator,
    ProvisionedConcurrencyLedger,
    ProvisionedConcurrencyUpdatePolicy,
)

# Reuse the fakes from the assignment primitives tests.
from tests.unit.services.lambda_.test_provisioned_concurrency_scaling import FakeEnvFactory

ACCOUNT = "123456789012"
REGION = "us-east-1"
FUNCTION = "fn-coord"
VM_ID = "vm-coord"
QUALIFIER = "1"


def _wait(predicate, timeout=10.0, interval=0.01, message=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError(message or f"condition not met within {timeout}s")


@pytest.fixture
def coordinator(monkeypatch):
    factory = FakeEnvFactory()
    monkeypatch.setattr(assignment_module, "ExecutionEnvironment", factory)
    assignment = AssignmentService()
    ledger = ProvisionedConcurrencyLedger()
    version_id = VersionIdentifier(
        function_name=FUNCTION, qualifier=QUALIFIER, region=REGION, account=ACCOUNT
    )
    function_version = SimpleNamespace(id=version_id)
    coord = ProvisionedConcurrencyCoordinator(
        qualified_arn=version_id.qualified_arn(),
        version_manager_id=VM_ID,
        function_version=function_version,
        assignment_service=assignment,
        ledger=ledger,
    )
    yield coord, assignment, factory, ledger, version_id
    coord.shutdown()
    coord.join(2)
    assignment.stop()


def _env_ids(assignment):
    return [env.id for env in assignment.provisioned_environments(VM_ID)]


def _wait_ready(coord, target):
    def ready():
        snap = coord.snapshot()
        return snap is not None and snap.status == ProvisionedConcurrencyStatusEnum.READY

    _wait(ready, message=f"did not reach READY: {coord.snapshot()}")
    snap = coord.snapshot()
    assert snap.allocated == target
    assert snap.ready == target
    return snap


class TestIncrementalScaling:
    def test_scale_up_and_down_touches_only_the_delta(self, coordinator):
        coord, assignment, factory, ledger, version_id = coordinator

        coord.declare(3, qualifier=QUALIFIER).result(timeout=10)
        snap = _wait_ready(coord, 3)
        assert snap.available == 3
        first_ids = set(_env_ids(assignment))
        assert first_ids == {"e0", "e1", "e2"}
        assert len(factory.created) == 3
        assert factory.stopped_ids == []

        # 3 -> 5: only two new environments, existing ones untouched
        coord.declare(5, qualifier=QUALIFIER).result(timeout=10)
        _wait_ready(coord, 5)
        second_ids = set(_env_ids(assignment))
        assert first_ids.issubset(second_ids)
        assert len(second_ids) == 5
        assert len(factory.created) == 5
        assert factory.stopped_ids == []

        # 5 -> 2: FIFO reclaim removes the three oldest (e0,e1,e2), keeps e3,e4
        coord.declare(2, qualifier=QUALIFIER).result(timeout=10)
        _wait_ready(coord, 2)
        assert set(_env_ids(assignment)) == {"e3", "e4"}
        assert factory.stopped_ids == ["e0", "e1", "e2"]
        account = ledger.get(version_id.qualified_arn())
        assert (account.allocated, account.ready, account.in_flight) == (2, 2, 0)


class TestMergePolicy:
    def test_declarations_during_progress_merge_to_last_value(self, coordinator):
        coord, assignment, factory, ledger, version_id = coordinator
        gate = threading.Event()
        factory.gate = gate

        f10 = coord.declare(10, qualifier=QUALIFIER)
        # progress is queryable while environments are starting
        _wait(lambda: coord.snapshot().allocated == 10)
        snap = coord.snapshot()
        assert snap.status == ProvisionedConcurrencyStatusEnum.IN_PROGRESS
        assert snap.requested == 10
        assert snap.ready == 0
        assert snap.available == 0

        # declarations while IN_PROGRESS are not rejected
        f12 = coord.declare(12, qualifier=QUALIFIER)
        f6 = coord.declare(6, qualifier=QUALIFIER)

        gate.set()
        for future in (f10, f12, f6):
            future.result(timeout=10)

        snap = _wait_ready(coord, 6)
        assert snap.available == 6
        # no full rebuild: every surviving id belongs to the initial batch
        assert len(_env_ids(assignment)) == 6
        assert set(_env_ids(assignment)).issubset({f"e{i}" for i in range(12)})
        assert len(factory.created) <= 12


class TestQueuePolicy:
    def test_queued_declarations_apply_fifo_with_visible_intermediate_state(self, coordinator):
        coord, assignment, factory, ledger, version_id = coordinator
        # Deterministic start gates: first batch (e0,e1) and second batch (e2,e3)
        first_batch = threading.Event()
        second_batch = threading.Event()
        factory.gates = {
            "e0": first_batch,
            "e1": first_batch,
            "e2": second_batch,
            "e3": second_batch,
        }

        f2 = coord.declare(2, qualifier=QUALIFIER, policy=ProvisionedConcurrencyUpdatePolicy.queue)
        _wait(lambda: coord.snapshot().allocated == 2)
        first_batch.set()
        f2.result(timeout=10)
        _wait_ready(coord, 2)

        f4 = coord.declare(4, qualifier=QUALIFIER, policy=ProvisionedConcurrencyUpdatePolicy.queue)
        f3 = coord.declare(3, qualifier=QUALIFIER, policy=ProvisionedConcurrencyUpdatePolicy.queue)

        # the intermediate 2->4 op creates e2/e3, which block at the gate:
        # the intermediate target/allocation is observable deterministically
        _wait(lambda: coord.snapshot().allocated == 4)
        snap4 = coord.snapshot()
        assert snap4.requested == 4
        assert snap4.ready == 2  # e2/e3 still starting
        assert snap4.queue_depth == 1
        second_batch.set()

        f4.result(timeout=10)
        f3.result(timeout=10)
        snap = _wait_ready(coord, 3)
        assert snap.requested == 3
        assert snap.queue_depth == 0
        assert len(_env_ids(assignment)) == 3


class TestLeaseReclaimAtomicity:
    """End-to-end across CountingService + coordinator + assignment: the
    reclaim gate must protect outstanding provisioned leases and the account
    must not close before the lease is fully released."""

    @staticmethod
    def _provisioned_function():
        function = Function(function_name=FUNCTION)
        function.provisioned_concurrency_configs[QUALIFIER] = ProvisionedConcurrencyConfiguration(
            2, "now"
        )
        return function

    def test_scale_to_zero_waits_for_full_lease_release(self, coordinator):
        coord, assignment, factory, ledger, version_id = coordinator
        counting = CountingService(provisioned_ledger=ledger)
        function = self._provisioned_function()
        function_version = SimpleNamespace(id=version_id)

        coord.declare(2, qualifier=QUALIFIER).result(timeout=10)
        _wait_ready(coord, 2)
        account = ledger.get(version_id.qualified_arn())

        scale_to_zero = None
        lease_context = counting.get_invocation_lease(
            function, function_version, version_id.qualified_arn()
        )
        try:
            lease_type = lease_context.__enter__()
            assert lease_type == InitializationType.provisioned_concurrency
            assert account.snapshot().in_flight == 1

            # same flow as LambdaVersionManager.invoke: reserve then invoke
            environment_context = assignment.get_environment(
                VM_ID, function_version, InitializationType.provisioned_concurrency
            )
            env = environment_context.__enter__()
            assert env.status == RuntimeStatus.INVOKING
            try:
                # scale to zero while the invocation is in flight
                scale_to_zero = coord.declare(0, qualifier=QUALIFIER)
                # the reclaim FIFO head is the busy environment; nothing stops
                time.sleep(0.3)
                assert factory.stopped_ids == []
                assert len(_env_ids(assignment)) == 2
                counters = account.snapshot()
                assert counters.allocated == 2
                assert counters.ready == 2
                assert counters.in_flight == 1
                assert ledger.get(version_id.qualified_arn()) is account
            finally:
                # invocation finished: physical release happens before the
                # lease's in-flight decrement, mirroring production ordering
                environment_context.__exit__(None, None, None)
        finally:
            lease_context.__exit__(None, None, None)

        # only after the lease is fully released can the last env be reclaimed
        assert scale_to_zero is not None
        scale_to_zero.result(timeout=10)
        assert factory.stopped_ids == ["e0", "e1"]
        assert _env_ids(assignment) == []
        assert ledger.get(version_id.qualified_arn()) is None
        assert coord.snapshot() is None
        assert account.snapshot().in_flight == 0

    def test_outstanding_lease_keeps_a_reservable_environment(self, coordinator):
        coord, assignment, factory, ledger, version_id = coordinator
        counting = CountingService(provisioned_ledger=ledger)
        function = self._provisioned_function()
        function_version = SimpleNamespace(id=version_id)

        coord.declare(2, qualifier=QUALIFIER).result(timeout=10)
        _wait_ready(coord, 2)

        with counting.get_invocation_lease(
            function, function_version, version_id.qualified_arn()
        ) as lease_type:
            assert lease_type == InitializationType.provisioned_concurrency
            account = ledger.get(version_id.qualified_arn())
            # scale 2 -> 1 while the lease is granted but not reserved yet;
            # the gate must leave at least one serviceable environment for it
            scale_down = coord.declare(1, qualifier=QUALIFIER)

            def one_serviceable_left():
                envs = assignment.provisioned_environments(VM_ID)
                return len(envs) == 1 and envs[0].status == RuntimeStatus.READY

            _wait(one_serviceable_left)
            counters = account.snapshot()
            assert counters.ready - counters.in_flight == 0
            # the granted lease still reserves an environment successfully
            with assignment.get_environment(
                VM_ID, function_version, InitializationType.provisioned_concurrency
            ) as env:
                assert env.status == RuntimeStatus.INVOKING
            scale_down.result(timeout=10)

        snap = _wait_ready(coord, 1)
        assert snap.in_flight == 0
        assert len(_env_ids(assignment)) == 1


class TestFailureRollback:
    def test_failed_scale_up_rolls_environments_and_account_back(self, coordinator):
        coord, assignment, factory, ledger, version_id = coordinator

        coord.declare(2, qualifier=QUALIFIER).result(timeout=10)
        _wait_ready(coord, 2)
        first_ids = set(_env_ids(assignment))

        # an in-flight call on a pre-existing environment is preserved through rollback
        account = ledger.get(version_id.qualified_arn())
        assert account.try_acquire_in_flight()
        assignment.provisioned_environments(VM_ID)[0].mark(RuntimeStatus.INVOKING)

        # the second *new* environment (e3) fails to start; e2 starts fine
        factory.fail_ids = {"e3"}
        future = coord.declare(4, qualifier=QUALIFIER)
        future.result(timeout=10)

        snap = coord.snapshot()
        assert snap.status == ProvisionedConcurrencyStatusEnum.FAILED
        assert snap.status_reason == "FUNCTION_ERROR_INIT_FAILURE"
        assert set(_env_ids(assignment)) == first_ids
        counters = account.snapshot()
        # pre-adjustment counters restored; the in-flight lease is untouched
        assert (counters.allocated, counters.ready, counters.in_flight) == (2, 2, 1)

        # releasing the in-flight call keeps the account consistent
        assignment.provisioned_environments(VM_ID)[0].mark(RuntimeStatus.READY)
        account.decrement_in_flight()

        # a new valid declaration can be applied after FAILED
        factory.fail_ids = set()
        coord.declare(3, qualifier=QUALIFIER).result(timeout=10)
        _wait_ready(coord, 3)
        assert account.snapshot().in_flight == 0


class TestReclaimOrderAndShutdown:
    def test_busy_head_blocks_fifo_reclaim_until_release_and_closes_account(self, coordinator):
        coord, assignment, factory, ledger, version_id = coordinator

        coord.declare(3, qualifier=QUALIFIER).result(timeout=10)
        _wait_ready(coord, 3)
        envs = assignment.provisioned_environments(VM_ID)
        envs[0].mark(RuntimeStatus.INVOKING)
        assert ledger.get(version_id.qualified_arn()).try_acquire_in_flight()

        future = coord.declare(0, qualifier=QUALIFIER)
        # strict FIFO: the busy head blocks all reclamation, even though b/c idle
        time.sleep(0.3)
        assert factory.stopped_ids == []
        assert len(_env_ids(assignment)) == 3

        envs[0].mark(RuntimeStatus.READY)
        ledger.get(version_id.qualified_arn()).decrement_in_flight()
        future.result(timeout=10)

        assert factory.stopped_ids == ["e0", "e1", "e2"]
        assert _env_ids(assignment) == []
        # account is closed once target 0 settles
        assert ledger.get(version_id.qualified_arn()) is None
        assert coord.snapshot() is None

    def test_shutdown_releases_worker_and_rejects_new_declarations(self, coordinator):
        coord, assignment, factory, ledger, version_id = coordinator
        coord.declare(1, qualifier=QUALIFIER).result(timeout=10)
        _wait_ready(coord, 1)
        coord.shutdown()
        coord.join(timeout=2)
        assert coord._worker is not None and not coord._worker.is_alive()
        with pytest.raises(ServiceException):
            coord.declare(2, qualifier=QUALIFIER)
