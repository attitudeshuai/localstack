"""Unit tests for the incremental AssignmentService primitives: delta creation,
FIFO counting and busy-safe reclamation. Uses a fake execution environment so
that no Docker runtime is required."""

import threading

import pytest

from localstack.services.lambda_.invocation import assignment as assignment_module
from localstack.services.lambda_.invocation.assignment import AssignmentService
from localstack.services.lambda_.invocation.execution_environment import (
    InvalidStatusException,
    RuntimeStatus,
)

VM_ID = "vm-test"


class _FakeTimer:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class _FakeRuntimeExecutor:
    def __init__(self, factory, env):
        self.factory = factory
        self.env = env
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1
        self.factory.stopped_ids.append(self.env.id)


class FakeExecutionEnvironment:
    """Drop-in replacement for ExecutionEnvironment in the assignment module."""

    def __init__(
        self,
        function_version,
        initialization_type,
        on_timeout,
        version_manager_id,
    ):
        self.factory = FakeExecutionEnvironment._last_factory
        self.id = f"e{self.factory.next_id()}"
        self.status = RuntimeStatus.INACTIVE
        self.status_lock = threading.RLock()
        self.function_version = function_version
        self.initialization_type = initialization_type
        self.on_timeout = on_timeout
        self.version_manager_id = version_manager_id
        self.keepalive_timer = _FakeTimer()
        self.factory.created.append(self)
        self.runtime_executor = _FakeRuntimeExecutor(self.factory, self)

    def start(self):
        with self.status_lock:
            if self.status != RuntimeStatus.INACTIVE:
                raise InvalidStatusException(f"unexpected start from {self.status}")
            self.status = RuntimeStatus.STARTING
        if self.factory.gate is not None:
            self.factory.gate.wait()
        per_id_gate = self.factory.gates.get(self.id)
        if per_id_gate is not None:
            per_id_gate.wait()
        if self.factory.delay:
            self.factory.delay_by(self.factory.delay)
        if self.id in self.factory.fail_ids:
            with self.status_lock:
                self.status = RuntimeStatus.STARTUP_FAILED
            raise RuntimeError(f"intentional startup failure of {self.id}")
        with self.status_lock:
            self.status = RuntimeStatus.READY

    def mark(self, status: RuntimeStatus):
        with self.status_lock:
            self.status = status

    def reserve(self):
        # Mirrors ExecutionEnvironment.reserve for the provisioned path.
        with self.status_lock:
            if self.status != RuntimeStatus.READY:
                raise InvalidStatusException(f"cannot reserve env in {self.status}")
            self.status = RuntimeStatus.INVOKING

    def release(self):
        # Mirrors ExecutionEnvironment.release for the provisioned path: an
        # invocation finishes, the environment becomes serviceable again.
        with self.status_lock:
            if self.status == RuntimeStatus.INVOKING:
                self.status = RuntimeStatus.READY


class FakeEnvFactory:
    def __init__(self):
        self._id_counter = 0
        self.created: list[FakeExecutionEnvironment] = []
        self.stopped_ids: list[str] = []
        self.gate: threading.Event | None = None
        # Optional per-environment start gates, keyed by fake env id (e0, e1, ...)
        self.gates: dict[str, threading.Event] = {}
        self.fail_ids: set[str] = set()
        self.delay: float = 0.0
        FakeExecutionEnvironment._last_factory = self

    def __call__(self, *args, **kwargs):
        return FakeExecutionEnvironment(*args, **kwargs)

    def next_id(self) -> int:
        value = self._id_counter
        self._id_counter += 1
        return value

    def by_id(self, env_id: str) -> FakeExecutionEnvironment:
        return next(env for env in self.created if env.id == env_id)

    @staticmethod
    def delay_by(seconds: float):
        # Local indirection so tests can avoid sleeping in the hot path.
        import time

        time.sleep(seconds)


@pytest.fixture
def assignment(monkeypatch):
    factory = FakeEnvFactory()
    monkeypatch.setattr(assignment_module, "ExecutionEnvironment", factory)
    service = AssignmentService()
    yield service, factory
    service.stop()


def _wait_all(futures, timeout=10):
    for _env, future in futures:
        future.result(timeout=timeout)


class TestIncrementalCreate:
    def test_creates_only_delta_and_keeps_existing(self, assignment):
        service, factory = assignment
        first = service.create_provisioned_environments(VM_ID, None, 3)
        _wait_all(first)
        first_ids = [env.id for env, _ in first]

        second = service.create_provisioned_environments(VM_ID, None, 2)
        _wait_all(second)
        second_ids = [env.id for env, _ in second]

        all_envs = service.provisioned_environments(VM_ID)
        assert len(all_envs) == 5
        # FIFO order is creation order
        assert [env.id for env in all_envs] == first_ids + second_ids
        assert set(first_ids).issubset({env.id for env in all_envs})
        total, serviceable, in_flight = service.count_provisioned_environments(VM_ID)
        assert (total, serviceable, in_flight) == (5, 5, 0)


class TestFifoReclaim:
    def test_reclaim_oldest_first_and_keep_remaining(self, assignment):
        service, factory = assignment
        created = service.create_provisioned_environments(VM_ID, None, 7)
        _wait_all(created)

        stopped_ids = []
        for _ in range(4):
            head = service.provisioned_environments(VM_ID)[0]
            result = service.reclaim_provisioned_environment(head)
            assert result == "stopped"
            stopped_ids.append(head.id)

        assert stopped_ids == ["e0", "e1", "e2", "e3"]
        remaining = service.provisioned_environments(VM_ID)
        assert [env.id for env in remaining] == ["e4", "e5", "e6"]
        total, serviceable, in_flight = service.count_provisioned_environments(VM_ID)
        assert (total, serviceable, in_flight) == (3, 3, 0)

    def test_busy_environment_cannot_be_reclaimed(self, assignment):
        service, factory = assignment
        created = service.create_provisioned_environments(VM_ID, None, 1)
        _wait_all(created)
        env = created[0][0]

        env.mark(RuntimeStatus.INVOKING)
        with pytest.raises(InvalidStatusException):
            service.reclaim_provisioned_environment(env)
        # still in the pool and not stopped
        assert service.provisioned_environments(VM_ID) == [env]
        assert env.runtime_executor.stop_calls == 0

        env.mark(RuntimeStatus.READY)
        assert service.reclaim_provisioned_environment(env) == "stopped"
        assert env.runtime_executor.stop_calls == 1
        assert service.provisioned_environments(VM_ID) == []

    def test_dead_slot_is_discarded_without_stop(self, assignment):
        service, factory = assignment
        created = service.create_provisioned_environments(VM_ID, None, 1)
        _wait_all(created)
        env = created[0][0]
        env.mark(RuntimeStatus.STARTUP_TIMED_OUT)

        assert service.reclaim_provisioned_environment(env) == "discarded"
        assert env.runtime_executor.stop_calls == 0
        assert service.provisioned_environments(VM_ID) == []

    def test_count_mixed_statuses(self, assignment):
        service, factory = assignment
        created = service.create_provisioned_environments(VM_ID, None, 4)
        _wait_all(created)
        envs = [env for env, _ in created]
        envs[0].mark(RuntimeStatus.INVOKING)
        envs[1].mark(RuntimeStatus.READY)  # unchanged
        envs[2].mark(RuntimeStatus.TIMING_OUT)
        envs[3].mark(RuntimeStatus.STARTUP_FAILED)

        total, serviceable, in_flight = service.count_provisioned_environments(VM_ID)
        assert total == 4
        assert serviceable == 2  # INVOKING + READY
        assert in_flight == 1
