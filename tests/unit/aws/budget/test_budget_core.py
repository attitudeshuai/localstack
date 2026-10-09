import threading
import time

import pytest

from localstack.aws.budget.core import (
    BudgetExhaustedError,
    ExecutionBudget,
    _current_budget,
    _DaemonThreadPool,
    current_budget,
)
from localstack.utils.threads import FuncThread


class TestExecutionBudget:
    def test_remaining_decreases(self):
        budget = ExecutionBudget(timeout=10)
        assert budget.remaining() == pytest.approx(10, abs=0.1)
        assert budget.is_live
        assert not budget.is_exhausted

    def test_check_raises_after_deadline(self):
        budget = ExecutionBudget(timeout=0.05)
        time.sleep(0.08)
        with pytest.raises(BudgetExhaustedError):
            budget.check()
        assert budget.is_exhausted
        assert budget.exhaustion_reason == "deadline-exceeded"

    def test_unlimited_budget(self):
        budget = ExecutionBudget(timeout=None)
        assert budget.remaining() is None
        assert not budget.is_live
        budget.check()  # must never raise

    def test_child_shares_deadline_and_cannot_extend(self):
        root = ExecutionBudget(timeout=10)
        time.sleep(0.01)
        child = root.child_for_internal_call()

        assert child.propagated is True
        assert child.budget_id == root.budget_id
        # the child deadline is the root deadline (same absolute point in time)
        assert child.deadline == root.deadline
        assert child.remaining() == pytest.approx(root.remaining(), abs=0.001)

    def test_propagated_child_cannot_exempt(self):
        root = ExecutionBudget(timeout=10)
        child = root.child_for_internal_call()
        with pytest.raises(RuntimeError):
            child.exempt("nope")
        assert child.is_live

    def test_root_exempt_disarms_deadline(self):
        budget = ExecutionBudget(timeout=0.02)
        budget.exempt("long-polling:sqs.ReceiveMessage.WaitTimeSeconds")
        assert budget.is_exempt
        assert budget.remaining() is None
        time.sleep(0.05)
        budget.check()  # exempt budgets never exhaust
        assert not budget.is_exhausted

    def test_exempt_after_exhaustion_noop(self):
        budget = ExecutionBudget(timeout=0.01)
        time.sleep(0.03)
        with pytest.raises(BudgetExhaustedError):
            budget.check()
        budget.exempt("too-late")
        assert budget.is_exhausted
        assert not budget.is_exempt

    def test_watchdog_invokes_cancel_callbacks(self):
        cancelled = threading.Event()
        budget = ExecutionBudget(timeout=0.05)
        budget.arm()
        budget.on_cancel(cancelled.set)
        assert cancelled.wait(timeout=2)
        assert budget.is_exhausted

    def test_on_cancel_registration_after_exhaustion_fires_immediately(self):
        budget = ExecutionBudget(timeout=0.01)
        budget.exhaust("manual")
        called = threading.Event()
        budget.on_cancel(called.set)
        assert called.is_set()

    def test_exhaust_cancels_tracked_thread_cooperatively(self):
        stop_event = threading.Event()

        class FakeThread:
            def __init__(self):
                self.stopped = False

            def stop(self):
                self.stopped = True
                stop_event.set()

        thread = FakeThread()
        budget = ExecutionBudget(timeout=10)
        budget.track_thread(thread)
        budget.exhaust("manual")
        assert stop_event.is_set()
        assert thread.stopped

    def test_spawn_propagates_budget_context(self):
        budget = ExecutionBudget(timeout=10)
        seen = {}

        def worker():
            seen["budget"] = current_budget()
            return 42

        token = _current_budget.set(budget)
        try:
            future = budget.spawn(worker)
        finally:
            _current_budget.reset(token)
        assert future.result(timeout=2) == 42
        assert seen["budget"] is budget
        budget.close()

    def test_spawned_pending_task_cancelled_on_exhaustion(self):
        budget = ExecutionBudget(timeout=10)
        started = threading.Event()
        release = threading.Event()

        def blocker():
            started.set()
            release.wait(timeout=5)

        # single-worker pool: the second task stays pending and can be cancelled
        budget._state.executor = _DaemonThreadPool(max_workers=1)
        budget.spawn(blocker)
        assert started.wait(timeout=2)
        pending = budget.spawn(lambda: None)
        budget.exhaust("manual")
        assert pending.cancelled()
        release.set()
        budget.close()

    def test_thread_pool_submit_propagates_budget(self):
        from localstack.utils.asyncio import AdaptiveThreadPool

        pool = AdaptiveThreadPool()
        budget = ExecutionBudget(timeout=10)
        cancelled = threading.Event()

        def worker():
            observed = current_budget()
            # register a resource from inside the worker: cancelled when the root budget dies
            observed.on_cancel(cancelled.set)
            return observed is budget

        token = _current_budget.set(budget)
        try:
            future = pool.submit(worker)
        finally:
            _current_budget.reset(token)

        assert future.result(timeout=2) is True
        budget.exhaust("manual")
        assert cancelled.wait(timeout=2)

    def test_func_thread_bound_to_budget_is_stopped_on_exhaustion(self):
        budget = ExecutionBudget(timeout=10)
        seen = {}
        running = threading.Event()

        def loop(_params):
            seen["budget"] = current_budget()
            running.set()
            while current_budget() and not current_budget().is_exhausted:
                time.sleep(0.005)

        token = _current_budget.set(budget)
        try:
            thread = FuncThread(loop)
            budget.bind_worker_thread(thread)
            thread.start()
        finally:
            _current_budget.reset(token)

        assert running.wait(timeout=2)
        assert seen["budget"] is budget
        budget.exhaust("manual")
        thread.join(timeout=2)
        assert not thread.is_alive()

    def test_concurrent_budgets_are_isolated(self):
        fast = ExecutionBudget(timeout=0.05)
        slow = ExecutionBudget(timeout=30)

        time.sleep(0.08)
        with pytest.raises(BudgetExhaustedError):
            fast.check()

        # the other concurrent request is completely unaffected
        assert slow.is_live
        assert not slow.is_exhausted
        assert slow.remaining() > 29

    def test_child_exhaustion_propagates_to_root(self):
        root = ExecutionBudget(timeout=0.05)
        child = root.child_for_internal_call()
        child.exhaust("nested-failure")
        assert root.is_exhausted
        assert root.exhaustion_reason == "nested-failure"
