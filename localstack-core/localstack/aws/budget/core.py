"""
Core primitives of the request-level execution budget.

An :class:`ExecutionBudget` is created when a request enters the gateway and follows the request
through the handler chain, the provider execution and internal cross-service calls. Nested
internal calls share the budget state of their root request, which means they can only consume
the remaining quota but never extend the deadline.

The module is deliberately kept free of other ``localstack.aws`` imports so that it can be
imported from low-level utilities (like the thread helpers) without creating import cycles.
"""

from __future__ import annotations

import concurrent.futures
import concurrent.futures.thread
import logging
import threading
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

LOG = logging.getLogger(__name__)

_current_budget: ContextVar[ExecutionBudget | None] = ContextVar(
    "localstack_request_budget", default=None
)


def current_budget() -> ExecutionBudget | None:
    """
    Return the budget bound to the current execution context, or ``None`` if the current thread
    does not process a budgeted request.
    """
    return _current_budget.get()


class BudgetExhaustedError(Exception):
    """
    Raised when an execution budget has been exhausted. The exception is serialized into a
    distinguishable failure response by the gateway exception handlers.
    """

    def __init__(self, budget: ExecutionBudget, reason: str):
        self.budget = budget
        self.reason = reason
        super().__init__(
            f"request execution budget exhausted after {budget.elapsed():.3f}s "
            f"(limit={budget.timeout}s, budget={budget.budget_id}, reason={reason})"
        )


class _DaemonThreadPool(concurrent.futures.thread.ThreadPoolExecutor):
    """
    Thread pool whose workers are removed from the global ``_threads_queues`` registry, so the
    interpreter does not join them at shutdown (mirrors the approach used by rolo's ASGI
    gateway). Pending tasks are additionally cancelled when the budget is exhausted or closed.
    """

    def _adjust_thread_count(self) -> None:
        super()._adjust_thread_count()
        for thread in list(self._threads):
            try:
                del concurrent.futures.thread._threads_queues[thread]
            except (KeyError, AttributeError):
                pass


class _BudgetState:
    """Mutable state shared by a root budget and all of its nested (internal call) budgets."""

    def __init__(self, deadline: float | None):
        self.deadline = deadline
        self.exempt: bool = False
        self.exemption_reason: str | None = None
        self.exhausted = threading.Event()
        self.exhaustion_reason: str | None = None
        self.lock = threading.RLock()
        self.cancel_callbacks: list[Callable[[], Any]] = []
        self.timer: threading.Timer | None = None
        self.armed = False
        self.executor: _DaemonThreadPool | None = None


class ExecutionBudget:
    """
    The execution budget of a single request.

    Budgets form a tree: the root budget is created at gateway entry, internal calls create child
    budgets that share the deadline, cancellation event and task registry of the root. A child
    budget therefore observes exactly the remaining quota of its parent and cannot extend it.
    """

    def __init__(
        self,
        timeout: float | None,
        *,
        propagated: bool = False,
        budget_id: str | None = None,
        start: float | None = None,
        _state: _BudgetState | None = None,
    ):
        """
        :param timeout: the budget in seconds, or ``None`` for an unlimited/exempt budget
        :param propagated: whether this budget was inherited from an upstream internal call
        :param budget_id: identifier shared across the entire budget tree
        :param start: monotonic start time (used for testing)
        :param _state: shared state of the parent budget (used for child budgets)
        """
        self.timeout = timeout
        self.propagated = propagated
        self.budget_id = budget_id or uuid.uuid4().hex[:12]
        self.started_at = start if start is not None else time.monotonic()
        if _state is not None:
            self._state = _state
        else:
            deadline = self.started_at + timeout if timeout is not None else None
            self._state = _BudgetState(deadline=deadline)
        # only the root budget owns the lifecycle (watchdog, task executor) of the shared state
        self._owns_state = _state is None

    #
    # introspection
    #

    @property
    def deadline(self) -> float | None:
        return None if self._state.exempt else self._state.deadline

    @property
    def is_exempt(self) -> bool:
        return self._state.exempt

    @property
    def exemption_reason(self) -> str | None:
        return self._state.exemption_reason

    @property
    def is_exhausted(self) -> bool:
        return self._state.exhausted.is_set()

    @property
    def exhaustion_reason(self) -> str | None:
        return self._state.exhaustion_reason

    @property
    def is_live(self) -> bool:
        """A budget is live when it enforces a deadline and has not been exhausted or exempted."""
        return not self.is_exhausted and not self.is_exempt and self._state.deadline is not None

    def remaining(self) -> float | None:
        """Remaining budget in seconds, or ``None`` if no deadline applies."""
        if self._state.exempt or self._state.deadline is None:
            return None
        return max(0.0, self._state.deadline - time.monotonic())

    def remaining_ms(self) -> int | None:
        """Remaining budget in milliseconds, or ``None`` if no deadline applies."""
        remaining = self.remaining()
        if remaining is None:
            return None
        return max(0, int(remaining * 1000))

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    #
    # enforcement
    #

    def check(self) -> None:
        """Raise :class:`BudgetExhaustedError` if the budget is no longer live."""
        if self._state.exhausted.is_set():
            raise BudgetExhaustedError(self, self._state.exhaustion_reason or "cancelled")
        deadline = self._state.deadline
        if deadline is not None and not self._state.exempt and time.monotonic() >= deadline:
            self.exhaust("deadline-exceeded")
            raise BudgetExhaustedError(self, self._state.exhaustion_reason)

    def arm(self) -> None:
        """Start the deadline watchdog. Idempotent and shared across the whole budget tree."""
        state = self._state
        if state.exempt or state.deadline is None:
            return
        with state.lock:
            if state.armed or state.exhausted.is_set():
                return
            delay = max(0.0, state.deadline - time.monotonic())
            state.timer = threading.Timer(delay, self.exhaust, args=("deadline-exceeded",))
            state.timer.daemon = True
            state.timer.name = f"budget-watchdog-{self.budget_id}"
            state.timer.start()
            state.armed = True

    def exhaust(self, reason: str = "exhausted") -> None:
        """Mark the entire budget tree as exhausted and cancel all registered downstream work."""
        state = self._state
        with state.lock:
            if not state.exhausted.is_set():
                state.exhaustion_reason = reason
                state.exhausted.set()
                if state.timer is not None:
                    state.timer.cancel()
                callbacks = list(state.cancel_callbacks)
                state.cancel_callbacks.clear()
            else:
                callbacks = []

        for callback in callbacks:
            try:
                callback()
            except Exception:
                LOG.debug("error while cancelling budget task", exc_info=True)

    # alias matching the requirement wording
    cancel_all = exhaust

    def exempt(self, reason: str) -> None:
        """
        Declare this budget exempt from enforcement (e.g. long polling or streaming responses).

        Only a root budget can be exempted: nested internal calls inherit a deadline and are not
        allowed to remove it. Exempting after the budget was exhausted has no effect.
        """
        if self.propagated:
            raise RuntimeError("cannot exempt a propagated (nested internal call) budget")
        state = self._state
        with state.lock:
            if state.exhausted.is_set():
                return
            state.exempt = True
            state.exemption_reason = reason
            if state.timer is not None:
                state.timer.cancel()
                state.timer = None
            state.armed = False

    def child_for_internal_call(self) -> ExecutionBudget:
        """
        Create the budget of a nested internal call. The child shares the deadline and all
        cancellation state, so it can only inherit the remaining quota and never extend it.
        """
        remaining = self.remaining()
        if remaining is None:
            timeout = None
        else:
            timeout = remaining
        return ExecutionBudget(
            timeout=timeout,
            propagated=True,
            budget_id=self.budget_id,
            start=self.started_at,
            _state=self._state,
        )

    #
    # downstream task tracking / cancellation
    #

    def on_cancel(self, callback: Callable[[], Any]) -> None:
        """Register a callback that interrupts downstream work (closes sockets, cancels tasks)."""
        state = self._state
        with state.lock:
            if state.exhausted.is_set():
                fire_now = True
            else:
                state.cancel_callbacks.append(callback)
                fire_now = False
        if fire_now:
            try:
                callback()
            except Exception:
                LOG.debug("error while cancelling budget resource", exc_info=True)

    def track_future(self, future: concurrent.futures.Future) -> concurrent.futures.Future:
        """Track a concurrent future so pending executions are cancelled on exhaustion."""
        self.on_cancel(lambda: future.cancel())
        return future

    def track_thread(self, thread: Any) -> Any:
        """Track an object exposing a ``stop()`` method (e.g. ``FuncThread``)."""
        self.on_cancel(thread.stop)
        return thread

    def track_asyncio_task(self, loop: Any, task: Any) -> Any:
        """Track an asyncio task running on the given event loop."""
        self.on_cancel(lambda: loop.call_soon_threadsafe(task.cancel))
        return task

    def bind_worker_thread(self, thread: Any) -> Any:
        """
        Bind a worker thread (e.g. ``FuncThread``) spawned downstream to this budget: the budget
        context is propagated into the thread, and the thread is stopped when the budget is
        exhausted.
        """
        original_func = thread.func
        budget = self

        def _propagate(params: Any, **kwargs: Any):
            token = _current_budget.set(budget)
            try:
                return original_func(params, **kwargs)
            finally:
                _current_budget.reset(token)

        thread.func = _propagate
        self.track_thread(thread)
        return thread

    def spawn(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> concurrent.futures.Future:
        """
        Run ``fn`` on the budget-owned thread pool. The budget context is propagated into the
        worker, and pending executions are cancelled when the budget is exhausted.
        """
        state = self._state
        with state.lock:
            if state.executor is None:
                state.executor = _DaemonThreadPool(
                    max_workers=None, thread_name_prefix=f"budget-{self.budget_id}"
                )
            executor = state.executor

        budget = self

        def _run():
            token = _current_budget.set(budget)
            try:
                return fn(*args, **kwargs)
            finally:
                _current_budget.reset(token)

        future = executor.submit(_run)
        self.on_cancel(lambda: future.cancel())
        return future

    #
    # lifecycle
    #

    def close(self) -> None:
        """Release the lifecycle resources of the budget. Only the root budget does any work."""
        if not self._owns_state:
            return
        state = self._state
        with state.lock:
            if state.timer is not None:
                state.timer.cancel()
                state.timer = None
            executor = state.executor
            state.executor = None
        if executor is not None:
            # do not wait for cooperative tasks, but make sure pending ones do not start
            executor.shutdown(wait=False, cancel_futures=True)

    def __repr__(self) -> str:
        return (
            f"<ExecutionBudget id={self.budget_id} timeout={self.timeout} "
            f"remaining={self.remaining()} propagated={self.propagated} "
            f"exempt={self.is_exempt} exhausted={self.is_exhausted}>"
        )
