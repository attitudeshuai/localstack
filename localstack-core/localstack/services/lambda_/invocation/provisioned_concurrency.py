"""Unified accounting and incremental scaling for Lambda provisioned concurrency.

This module provides:

* ``ProvisionedConcurrencyLedger`` — the single source of truth for provisioned
  concurrency quota, keyed by the qualified ARN of the *physical* function
  version. Every account tracks three quantities independently:

    - ``allocated``: provisioned slots currently accounted (starting, ready or
      invoking; including dead slots not yet pruned);
    - ``ready``: serviceable slots (READY or currently INVOKING);
    - ``in_flight``: slots currently processing an invocation.

  Differences are therefore always readable: ``allocated - ready`` is still
  provisioning (or failed), ``ready - in_flight`` is idle capacity and
  ``in_flight`` is the number of in-flight calls.

* ``ProvisionedConcurrencyCoordinator`` — a per-version-manager state machine
  that reconciles the physical execution environment pool towards declared
  targets using *incremental* delta operations, supports MERGE/QUEUE policies
  for declarations arriving while an adjustment is running, exposes queryable
  progress, and rolls environments and accounts back to the pre-adjustment
  state on failure.

The ledger is transient, like the previous ``ProvisionedConcurrencyState``;
the desired target remains persisted in
``Function.provisioned_concurrency_configs`` and is re-declared by the
provider during state restoration.
"""

import concurrent.futures
import dataclasses
import logging
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import Future
from enum import StrEnum
from threading import RLock
from typing import TYPE_CHECKING, cast

from localstack.aws.api.lambda_ import (
    ProvisionedConcurrencyStatusEnum,
    ServiceException,
)
from localstack.services.lambda_.api_utils import generate_lambda_date
from localstack.services.lambda_.invocation.execution_environment import (
    ExecutionEnvironment,
    InvalidStatusException,
    RuntimeStatus,
)

if TYPE_CHECKING:
    from localstack.services.lambda_.invocation.assignment import AssignmentService
    from localstack.services.lambda_.invocation.lambda_models import FunctionVersion

LOG = logging.getLogger(__name__)

# Maximum time a single scale-down adjustment waits for in-flight invocations
# to finish before giving up and rolling back. Mirrors the previous 20 minutes
# scaling wait in LambdaVersionManager.
SCALING_WAIT_TIMEOUT_SECONDS = 20 * 60

# Poll interval while waiting (on the account condition) for a busy
# environment to become reclaimable during scale-down.
RECLAIM_WAIT_POLL_SECONDS = 1.0


class ProvisionedConcurrencyUpdatePolicy(StrEnum):
    """Policy for declarations arriving while an adjustment is in progress.

    * ``merge`` (default, AWS-compatible): only the latest declared target
      matters; intermediate declarations may collapse.
    * ``queue``: every declaration is applied FIFO, in order.
    """

    merge = "merge"
    queue = "queue"


@dataclasses.dataclass(frozen=True)
class ProvisionedConcurrencyCounters:
    allocated: int = 0
    ready: int = 0
    in_flight: int = 0

    @property
    def available(self) -> int:
        return self.ready - self.in_flight

    @property
    def provisioning(self) -> int:
        return self.allocated - self.ready


@dataclasses.dataclass(frozen=True)
class ProvisionedConcurrencySnapshot:
    """Queryable progress/result of provisioned concurrency for one qualifier."""

    requested: int
    allocated: int
    ready: int
    in_flight: int
    available: int
    status: ProvisionedConcurrencyStatusEnum
    status_reason: str | None
    last_modified: str | None
    queue_depth: int


class ProvisionedConcurrencyAccount:
    """Quota account of one physical function version used by a declared
    qualifier (a version number or an alias name)."""

    account_id: str
    region: str
    function_name: str
    qualifier: str
    qualified_arn: str

    def __init__(
        self,
        *,
        account_id: str,
        region: str,
        function_name: str,
        qualifier: str,
        qualified_arn: str,
    ):
        self.account_id = account_id
        self.region = region
        self.function_name = function_name
        self.qualifier = qualifier
        self.qualified_arn = qualified_arn
        self.allocated = 0
        self.ready = 0
        self.in_flight = 0
        self.status: ProvisionedConcurrencyStatusEnum = ProvisionedConcurrencyStatusEnum.IN_PROGRESS
        self.status_reason: str | None = None
        # The condition protects the counters and is signalled whenever they
        # change (in particular on in_flight decrement and status transitions)
        # so that scale-down workers can wait for busy environments.
        self.condition = threading.Condition(RLock())

    # --- snapshots ----------------------------------------------------------

    def snapshot(self) -> ProvisionedConcurrencyCounters:
        with self.condition:
            return ProvisionedConcurrencyCounters(
                allocated=self.allocated, ready=self.ready, in_flight=self.in_flight
            )

    # --- status -------------------------------------------------------------

    def set_status(
        self,
        status: ProvisionedConcurrencyStatusEnum,
        status_reason: str | None = None,
    ) -> None:
        with self.condition:
            self.status = status
            self.status_reason = status_reason
            self.condition.notify_all()

    # --- counter mutations --------------------------------------------------

    def adjust_allocated(self, delta: int) -> None:
        with self.condition:
            self.allocated += delta
            self._validate_locked()
            # Notify reclaim/start waiters and progress pollers.
            self.condition.notify_all()

    def adjust_ready(self, delta: int) -> None:
        with self.condition:
            self.ready += delta
            self._validate_locked()
            self.condition.notify_all()

    def decrement_slot(self, *, serviceable: bool) -> None:
        """Atomically account one reclaimed slot: allocated always drops, and
        ready drops too iff the reclaimed environment was serviceable
        (READY/INVOKING) rather than a dead startup slot."""
        with self.condition:
            self.allocated -= 1
            if serviceable:
                self.ready -= 1
            self._validate_locked()
            self.condition.notify_all()

    def can_reclaim_serviceable(self) -> bool:
        """Whether a serviceable environment exists beyond the slots committed
        to outstanding in-flight leases (``ready - in_flight > 0``).

        Scale-down consults this under the account lock before reclaiming a
        READY environment, guaranteeing that every granted lease keeps an
        environment it can reserve even before it physically transitions to
        INVOKING. Evaluated under the same condition that is signalled on
        in-flight decrement.
        """
        with self.condition:
            return self.ready - self.in_flight > 0

    def wait_for_reclaim_capacity(self, timeout: float | None) -> bool:
        """Wait until an in-flight lease is released (or shutdown/status
        changes notify). Returns True unless the wait timed out."""
        with self.condition:
            return self.condition.wait(timeout)

    def try_acquire_in_flight(self) -> bool:
        """Atomically grant one provisioned invocation lease.

        Succeeds only while the adjustment is READY and an idle serviceable
        slot exists (``ready - in_flight > 0``). The check and the increment
        share one critical section with status flips, so a scale-down reclaim
        cannot give away a slot that was just accounted for reclamation and a
        reconciliation cannot reset a lease granted at READY.
        """
        with self.condition:
            if (
                self.status == ProvisionedConcurrencyStatusEnum.READY
                and self.ready - self.in_flight > 0
            ):
                self.in_flight += 1
                self._validate_locked()
                return True
            return False

    def increment_in_flight(self) -> None:
        with self.condition:
            self.in_flight += 1
            self._validate_locked()

    def decrement_in_flight(self) -> None:
        with self.condition:
            self.in_flight -= 1
            self._validate_locked()
            # Wake up scale-down waiting on the head environment to be released.
            self.condition.notify_all()

    def reset(
        self,
        *,
        allocated: int = 0,
        ready: int = 0,
        in_flight: int | None = None,
    ) -> None:
        """Hard-set counters while reconciling against physical truth.

        ``in_flight`` is owned by invocation leases and preserved unless an
        explicit value is provided, so reconciliation can never lose a lease
        that was granted concurrently with the status transition.
        """
        with self.condition:
            self.allocated = allocated
            self.ready = ready
            if in_flight is not None:
                self.in_flight = in_flight
            self._validate_locked()
            self.condition.notify_all()

    def _validate_locked(self) -> None:
        if self.ready < 0 or self.allocated < 0 or self.in_flight < 0:
            LOG.error(
                "Negative provisioned concurrency counters detected for %s: "
                "allocated=%d ready=%d in_flight=%d",
                self.qualified_arn,
                self.allocated,
                self.ready,
                self.in_flight,
            )
        if self.ready > self.allocated or self.in_flight > self.ready:
            LOG.error(
                "Inconsistent provisioned concurrency counters detected for %s: "
                "allocated=%d ready=%d in_flight=%d",
                self.qualified_arn,
                self.allocated,
                self.ready,
                self.in_flight,
            )


FunctionKey = tuple[str, str, str]
QualifierKey = tuple[str, str, str, str]


class ProvisionedConcurrencyLedger:
    """Global registry of provisioned concurrency accounts.

    Accounts are keyed by the qualified ARN of the physical function version.
    Two auxiliary indexes allow lookups by declared qualifier (version or
    alias) and aggregation by function."""

    def __init__(self) -> None:
        self._accounts: dict[str, ProvisionedConcurrencyAccount] = {}
        self._function_index: dict[FunctionKey, set[str]] = defaultdict(set)
        self._qualifier_bindings: dict[QualifierKey, str] = {}
        self._lock = RLock()

    def open(
        self,
        *,
        account_id: str,
        region: str,
        function_name: str,
        qualifier: str,
        qualified_arn: str,
    ) -> ProvisionedConcurrencyAccount:
        """Open (or rebind) the account for a declared provisioned concurrency
        configuration. ``qualifier`` is either the version number or the alias
        name; ``qualified_arn`` always identifies the physical version."""
        fkey = (account_id, region, function_name)
        qkey = (account_id, region, function_name, qualifier)
        with self._lock:
            account = self._accounts.get(qualified_arn)
            if account is None:
                account = ProvisionedConcurrencyAccount(
                    account_id=account_id,
                    region=region,
                    function_name=function_name,
                    qualifier=qualifier,
                    qualified_arn=qualified_arn,
                )
                self._accounts[qualified_arn] = account
            else:
                account.qualifier = qualifier
            self._function_index[fkey].add(qualified_arn)
            self._qualifier_bindings[qkey] = qualified_arn
            return account

    def close(
        self,
        *,
        account_id: str,
        region: str,
        function_name: str,
        qualifier: str,
        qualified_arn: str | None = None,
    ) -> None:
        """Close an account after its target reached zero or the version is
        shut down."""
        fkey = (account_id, region, function_name)
        qkey = (account_id, region, function_name, qualifier)
        with self._lock:
            resolved_arn = qualified_arn or self._qualifier_bindings.pop(qkey, None)
            self._qualifier_bindings.pop(qkey, None)
            if resolved_arn:
                self._accounts.pop(resolved_arn, None)
                arns = self._function_index.get(fkey)
                if arns is not None:
                    arns.discard(resolved_arn)
                    if not arns:
                        self._function_index.pop(fkey, None)

    def get(self, qualified_arn: str) -> ProvisionedConcurrencyAccount | None:
        with self._lock:
            return self._accounts.get(qualified_arn)

    def get_by_qualifier(
        self, *, account_id: str, region: str, function_name: str, qualifier: str
    ) -> ProvisionedConcurrencyAccount | None:
        qkey = (account_id, region, function_name, qualifier)
        with self._lock:
            qualified_arn = self._qualifier_bindings.get(qkey)
            if qualified_arn is None:
                return None
            return self._accounts.get(qualified_arn)

    def list_for_function(
        self, *, account_id: str, region: str, function_name: str
    ) -> list[ProvisionedConcurrencyAccount]:
        fkey = (account_id, region, function_name)
        with self._lock:
            arns = list(self._function_index.get(fkey, set()))
            return [self._accounts[arn] for arn in arns if arn in self._accounts]

    def function_totals(
        self, *, account_id: str, region: str, function_name: str
    ) -> ProvisionedConcurrencyCounters:
        total = ProvisionedConcurrencyCounters()
        for account in self.list_for_function(
            account_id=account_id, region=region, function_name=function_name
        ):
            snapshot = account.snapshot()
            total = ProvisionedConcurrencyCounters(
                allocated=total.allocated + snapshot.allocated,
                ready=total.ready + snapshot.ready,
                in_flight=total.in_flight + snapshot.in_flight,
            )
        return total


# Terminal statuses set on the account after an adjustment finishes.
_DEAD_ENV_STATUSES = (
    RuntimeStatus.STARTUP_FAILED,
    RuntimeStatus.STARTUP_TIMED_OUT,
    RuntimeStatus.STOPPED,
)


class ProvisionedConcurrencyCoordinator:
    """Incrementally reconciles provisioned environments for one version.

    A single lazy worker thread drains declarations:

    * MERGE policy: only the latest target matters; declarations while a
      reconciliation is running are picked up in the next loop iteration.
    * QUEUE policy: declarations are applied FIFO.

    Each reconciliation is an adjustment from the last applied target to the
    selected target and only creates or reclaims the delta. On failure, the
    newly created environments are removed (scale-up) or the previous target is
    restored (scale-down), so that environments and ledger end up in the
    pre-adjustment state with status FAILED.
    """

    def __init__(
        self,
        *,
        qualified_arn: str,
        version_manager_id: str,
        function_version: "FunctionVersion",
        assignment_service: "AssignmentService",
        ledger: ProvisionedConcurrencyLedger,
    ):
        self.qualified_arn = qualified_arn
        self.version_manager_id = version_manager_id
        self.function_version = function_version
        self._assignment = assignment_service
        self._ledger = ledger

        self._lock = RLock()
        self._condition = threading.Condition(self._lock)
        self._worker: threading.Thread | None = None
        self._active = False
        self._closing = False

        # MERGE policy state
        self._desired: int | None = None
        self._desired_seq = 0
        # Target most recently declared or currently being applied; backs the
        # queryable Requested field for both MERGE and QUEUE policies.
        self._requested = 0
        self._merge_waiters: list[tuple[int, Future[None]]] = []
        # Last MERGE target/seq that ended in FAILED; prevents an immediate
        # retry of the same declaration until a new one arrives.
        self._failed_attempt: tuple[int, int] | None = None

        # QUEUE policy state: (target, completion future)
        self._queue: deque[tuple[int, Future[None]]] = deque()

        # Last settled physical target; snapshot baseline for rollback.
        self._applied = 0

        self._status: ProvisionedConcurrencyStatusEnum = (
            ProvisionedConcurrencyStatusEnum.IN_PROGRESS
        )
        self._status_reason: str | None = None
        self._last_modified: str | None = None
        self._account: ProvisionedConcurrencyAccount | None = None
        # Declared qualifier (version number or alias name) of the active config
        self._qualifier: str | None = None

    # == Public API used by LambdaVersionManager / provider ==================

    def declare(
        self,
        target: int,
        *,
        qualifier: str,
        policy: ProvisionedConcurrencyUpdatePolicy = ProvisionedConcurrencyUpdatePolicy.merge,
    ) -> Future[None]:
        """Declare a new desired target while an adjustment may be running.

        Never rejected due to an in-progress adjustment. Returns a Future
        completing (without exception value) when the declaration settles
        (READY or FAILED); MERGE futures complete when the worker drains at the
        latest declaration, QUEUE futures complete when their item settles.
        """
        future: Future[None] = Future()
        with self._condition:
            if self._closing:
                raise ServiceException(
                    "Cannot update provisioned concurrency: the function version is shutting down."
                )
            self._qualifier = qualifier
            self._requested = target
            self._account = self._ledger.open(
                account_id=self.function_version.id.account,
                region=self.function_version.id.region,
                function_name=self.function_version.id.function_name,
                qualifier=qualifier,
                qualified_arn=self.qualified_arn,
            )
            self._last_modified = generate_lambda_date()
            self._status = ProvisionedConcurrencyStatusEnum.IN_PROGRESS
            self._status_reason = None
            self._account.set_status(self._status)

            if policy == ProvisionedConcurrencyUpdatePolicy.queue:
                self._queue.append((target, future))
            else:
                self._desired = target
                self._desired_seq += 1
                self._merge_waiters.append((self._desired_seq, future))

            self._ensure_worker_locked()
            # Wake a worker sleeping after a previous drain (a newly started
            # worker enters the lock directly, but an existing one waits here).
            self._condition.notify_all()
            return future

    def snapshot(self) -> ProvisionedConcurrencySnapshot | None:
        with self._condition:
            if self._account is None and self._desired is None and not self._queue:
                return None
            if self._account is not None:
                counters = self._account.snapshot()
                status = self._account.status
                status_reason = self._account.status_reason
            else:
                counters = ProvisionedConcurrencyCounters()
                status = self._status
                status_reason = self._status_reason
            return ProvisionedConcurrencySnapshot(
                requested=self._requested,
                allocated=counters.allocated,
                ready=counters.ready,
                in_flight=counters.in_flight,
                available=counters.available,
                status=status,
                status_reason=status_reason,
                last_modified=self._last_modified,
                queue_depth=len(self._queue),
            )

    def shutdown(self) -> None:
        """Stop accepting declarations and release the worker after the
        current reconciliation check. The ledger account is unregistered;
        outstanding leases still hold a reference to it and finish their
        accounting on the detached account."""
        with self._condition:
            self._closing = True
            self._condition.notify_all()
            account = self._account
        # Wake a scale-down worker potentially waiting on a busy environment.
        if account is not None:
            with account.condition:
                account.condition.notify_all()
            self._ledger.close(
                account_id=account.account_id,
                region=account.region,
                function_name=account.function_name,
                qualifier=account.qualifier,
                qualified_arn=account.qualified_arn,
            )

    def join(self, timeout: float | None = None) -> None:
        with self._condition:
            worker = self._worker
        if worker is not None:
            worker.join(timeout)

    # == Worker wiring ========================================================

    def _ensure_worker_locked(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(
                target=self._worker_loop,
                name=f"lambda-provisioned-scaling-{self.qualified_arn}",
                daemon=True,
            )
            self._worker.start()

    def _take_work_locked(
        self,
    ) -> tuple[str, int, object] | None:
        """Return ('queue', target, future) or ('merge', target, seq) under lock."""
        if self._queue:
            target, future = self._queue.popleft()
            self._requested = target
            return "queue", target, future
        if self._desired is not None and self._desired != self._applied:
            if self._failed_attempt != (self._desired, self._desired_seq):
                self._requested = self._desired
                return "merge", self._desired, self._desired_seq
        return None

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                work: tuple[str, int, object] | None = None
                while not self._closing:
                    work = self._take_work_locked()
                    if work is not None:
                        break
                    # Drained: settle all MERGE waiters at the terminal state.
                    self._active = False
                    self._complete_merge_waiters_locked()
                    self._condition.notify_all()
                    self._condition.wait()
                else:
                    # closing
                    self._active = False
                    self._complete_merge_waiters_locked()
                    self._complete_queue_waiters_locked()
                    self._condition.notify_all()
                    return
                self._active = True

            kind, target, payload = work
            baseline = self._applied
            succeeded = self._reconcile(
                kind=kind, target=target, payload=payload, baseline=baseline
            )

            with self._condition:
                if succeeded:
                    self._applied = target
                    self._failed_attempt = None
                else:
                    # Physical state was rolled back to the baseline (or matched
                    # against truth when rollback itself failed).
                    if kind == "merge":
                        merge_seq = cast(int, payload)
                        self._failed_attempt = (target, merge_seq)
                        newer_declaration = self._desired_seq != merge_seq
                    else:
                        newer_declaration = False
                    if not newer_declaration and self._account is not None:
                        self._status = ProvisionedConcurrencyStatusEnum.FAILED
                        self._status_reason = "FUNCTION_ERROR_INIT_FAILURE"
                        self._account.set_status(
                            ProvisionedConcurrencyStatusEnum.FAILED,
                            "FUNCTION_ERROR_INIT_FAILURE",
                        )
                if kind == "queue":
                    queue_future = cast(Future[None], payload)
                    if not queue_future.done():
                        queue_future.set_result(None)
                self._condition.notify_all()

    def _complete_merge_waiters_locked(self) -> None:
        waiters = self._merge_waiters
        self._merge_waiters = []
        for _seq, future in waiters:
            if not future.done():
                future.set_result(None)

    def _complete_queue_waiters_locked(self) -> None:
        while self._queue:
            _target, future = self._queue.popleft()
            if not future.done():
                future.set_result(None)

    # == Reconciliation =======================================================

    def _reconcile(self, *, kind: str, target: int, payload: object, baseline: int) -> bool:
        """One adjustment from the current physical pool to ``target``.

        Only the delta is created or reclaimed. Returns True on success and
        False after a rollback to ``baseline``.
        """
        account = self._account
        if account is None:
            LOG.error("Provisioned concurrency account missing for %s", self.qualified_arn)
            return False
        with self._condition:
            if not self._closing:
                self._status = ProvisionedConcurrencyStatusEnum.IN_PROGRESS
                self._status_reason = None
        account.set_status(ProvisionedConcurrencyStatusEnum.IN_PROGRESS)

        # 1) Prune dead slots and reconcile counters against physical truth.
        self._prune_dead_environments()
        total, serviceable, in_flight = self._assignment.count_provisioned_environments(
            self.version_manager_id
        )
        current = account.snapshot()
        if (
            current.allocated != total
            or current.ready != serviceable
            or current.in_flight != in_flight
        ):
            LOG.warning(
                "Provisioned concurrency account for %s drifted from the environment pool "
                "(ledger allocated=%d ready=%d in_flight=%d; actual total=%d serviceable=%d "
                "in_flight=%d); reconciling against physical truth",
                self.qualified_arn,
                current.allocated,
                current.ready,
                current.in_flight,
                total,
                serviceable,
                in_flight,
            )
            # in_flight is lease-owned and preserved: a lease granted at READY
            # immediately before the IN_PROGRESS flip may not have reserved its
            # environment yet, so the physical count can lag the ledger.
            account.reset(allocated=total, ready=serviceable)

        # 2) Apply the delta.
        if target > total:
            return self._scale_up(
                account=account,
                existing_total=total,
                target=target,
                baseline=baseline,
                kind=kind,
                payload=payload,
            )
        if target < total:
            return self._scale_down(
                account=account,
                existing_total=total,
                target=target,
                baseline=baseline,
                kind=kind,
                payload=payload,
            )

        # Already at target: just settle (covers in-flight truth above).
        self._settle_ready(account=account, target=target, kind=kind, payload=payload)
        return True

    def _newer_declaration_pending_locked(self, *, kind: str, target: int, payload: object) -> bool:
        """Whether a declaration beyond the one currently reconciling is queued.

        Used to suppress a transient READY state (and account closure) when the
        worker will immediately continue towards a newer target.
        """
        if self._queue:
            return True
        if kind == "merge":
            return self._desired != target or self._desired_seq != payload
        return False

    def _prune_dead_environments(self) -> None:
        for environment in self._assignment.provisioned_environments(self.version_manager_id):
            if environment.status in _DEAD_ENV_STATUSES:
                LOG.debug(
                    "Pruning dead provisioned environment %s (status=%s) for %s",
                    environment.id,
                    environment.status,
                    self.qualified_arn,
                )
                self._assignment.discard_provisioned_environment(environment)

    def _scale_up(
        self,
        *,
        account: ProvisionedConcurrencyAccount,
        existing_total: int,
        target: int,
        baseline: int,
        kind: str,
        payload: object,
    ) -> bool:
        count = target - existing_total
        LOG.debug(
            "Scaling provisioned concurrency up for %s: +%d (%d -> %d)",
            self.qualified_arn,
            count,
            existing_total,
            target,
        )
        created = self._assignment.create_provisioned_environments(
            self.version_manager_id, self.function_version, count
        )
        account.adjust_allocated(count)

        failure: Exception | None = None
        started: list[ExecutionEnvironment] = []
        deadline = time.monotonic() + SCALING_WAIT_TIMEOUT_SECONDS
        for environment, future in created:
            if not self._wait_for_future(future, deadline):
                failure = EnvironmentScalingTimeout(
                    f"Timed out starting provisioned environment {environment.id}"
                )
                break
            try:
                future.result()
                account.adjust_ready(1)
                started.append(environment)
            except Exception as e:
                failure = e
                break

        if failure is None and not self._closing:
            self._settle_ready(account=account, target=target, kind=kind, payload=payload)
            return True

        if self._closing:
            LOG.debug(
                "Provisioned concurrency scale-up of %s interrupted by shutdown",
                self.qualified_arn,
            )
            return False

        LOG.warning(
            "Failed to scale provisioned concurrency of %s to %d: %s; rolling back %d new "
            "environment(s) to the pre-adjustment target %d",
            self.qualified_arn,
            target,
            failure,
            len(created),
            baseline,
        )
        # Rollback: every environment created by this adjustment must leave the
        # pool (stopped if it started, discarded if it failed), counters first.
        self._rollback_created_environments(created=created, started=started, account=account)
        return False

    def _rollback_created_environments(
        self,
        *,
        created: list[tuple[ExecutionEnvironment, Future[None]]],
        started: list[ExecutionEnvironment],
        account: ProvisionedConcurrencyAccount,
    ) -> None:
        ready_to_reverse = len(started)
        deadline = time.monotonic() + SCALING_WAIT_TIMEOUT_SECONDS
        for environment, future in created:
            if not future.done():
                self._wait_for_future(future, deadline)
            removed = False
            if future.done() and future.exception() is None:
                # The status gate prevents leases while IN_PROGRESS, hence new
                # environments cannot be INVOKING here; wait defensively anyway.
                # Loop on the predicate: unrelated notifications and spurious
                # wakeups must not make us discard a still-running environment.
                while not self._closing:
                    try:
                        result = self._assignment.reclaim_provisioned_environment(environment)
                        removed = result == "stopped"
                        break
                    except InvalidStatusException:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            LOG.warning(
                                "Rollback stop of provisioned environment %s timed out; "
                                "discarding it from the pool without a confirmed stop",
                                environment.id,
                            )
                            break
                        account.wait_for_reclaim_capacity(min(RECLAIM_WAIT_POLL_SECONDS, remaining))
            if not removed:
                self._assignment.discard_provisioned_environment(environment)
            account.decrement_slot(serviceable=ready_to_reverse > 0)
            if ready_to_reverse > 0:
                ready_to_reverse -= 1

    def _scale_down(
        self,
        *,
        account: ProvisionedConcurrencyAccount,
        existing_total: int,
        target: int,
        baseline: int,
        kind: str,
        payload: object,
    ) -> bool:
        to_reclaim = existing_total - target
        LOG.debug(
            "Scaling provisioned concurrency down for %s: -%d (%d -> %d)",
            self.qualified_arn,
            to_reclaim,
            existing_total,
            target,
        )
        deadline = time.monotonic() + SCALING_WAIT_TIMEOUT_SECONDS
        reclaimed = 0
        while reclaimed < to_reclaim:
            if self._closing:
                LOG.debug(
                    "Provisioned concurrency scale-down of %s interrupted by shutdown",
                    self.qualified_arn,
                )
                return False
            environments = self._assignment.provisioned_environments(self.version_manager_id)
            if not environments:
                LOG.warning(
                    "Provisioned environment pool of %s emptied during scale-down",
                    self.qualified_arn,
                )
                break
            # Strict FIFO: always attempt the oldest declared environment.
            head = environments[0]

            # Dead slots (failed/timed-out starts) are not protected by leases
            # and can be pruned regardless of outstanding in-flight capacity.
            if head.status in _DEAD_ENV_STATUSES:
                result = self._assignment.reclaim_provisioned_environment(head)
                reclaimed += 1
                account.decrement_slot(serviceable=result == "stopped")
                continue

            # Lease-protection gate: never take away the last serviceable slot
            # of an outstanding lease -- even before it physically reserves an
            # environment (lease granted but reserve pending). Wait for the
            # in-flight decrement that signals the slot is fully released.
            if not account.can_reclaim_serviceable():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    LOG.warning(
                        "Timed out waiting for in-flight provisioned invocations of %s to "
                        "finish before reclaiming",
                        self.qualified_arn,
                    )
                    break
                account.wait_for_reclaim_capacity(min(RECLAIM_WAIT_POLL_SECONDS, remaining))
                continue

            try:
                result = self._assignment.reclaim_provisioned_environment(head)
            except InvalidStatusException:
                # Busy (or transiently starting): do not reclaim. Wait in order
                # for the in-flight call to finish; account condition is signaled
                # on in_flight decrement.
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    LOG.warning(
                        "Timed out waiting for busy provisioned environment %s of %s to "
                        "become reclaimable",
                        head.id,
                        self.qualified_arn,
                    )
                    break
                account.wait_for_reclaim_capacity(min(RECLAIM_WAIT_POLL_SECONDS, remaining))
                continue
            reclaimed += 1
            account.decrement_slot(serviceable=result == "stopped")

        if reclaimed == to_reclaim:
            self._settle_ready(account=account, target=target, kind=kind, payload=payload)
            return True

        # Rollback a partial scale-down by restoring the baseline target.
        LOG.warning(
            "Scale-down of %s to %s only reclaimed %d/%d environments; attempting rollback to "
            "the pre-adjustment target %d",
            self.qualified_arn,
            target,
            reclaimed,
            to_reclaim,
            baseline,
        )
        total, serviceable, _in_flight = self._assignment.count_provisioned_environments(
            self.version_manager_id
        )
        if total >= baseline:
            # The pool already covers the baseline (truth differs from the
            # attempted target); nothing to restart.
            account.reset(allocated=total, ready=serviceable)
            return False
        restored = self._scale_up(
            account=account,
            existing_total=total,
            target=baseline,
            baseline=baseline,
            kind=kind,
            payload=payload,
        )
        if restored:
            with self._condition:
                self._applied = baseline
        return False

    def _settle_ready(
        self,
        *,
        account: ProvisionedConcurrencyAccount,
        target: int,
        kind: str,
        payload: object,
    ) -> None:
        # Final truth check; in_flight is owned by invocations and preserved.
        total, serviceable, _in_flight = self._assignment.count_provisioned_environments(
            self.version_manager_id
        )
        if total != target or serviceable != target:
            LOG.error(
                "Provisioned concurrency settle mismatch for %s: target=%d actual total=%d "
                "serviceable=%d",
                self.qualified_arn,
                target,
                total,
                serviceable,
            )
        # Pending check under the coordinator lock only, then account mutations.
        # Lock order is always coordinator -> account; never nest the reverse.
        with self._condition:
            pending = self._newer_declaration_pending_locked(
                kind=kind, target=target, payload=payload
            )
        # in_flight is owned by invocation leases and preserved.
        account.reset(allocated=target, ready=target)
        if pending:
            # The worker continues immediately towards a newer declaration;
            # keep IN_PROGRESS and, for a zero target, keep the account open.
            account.set_status(ProvisionedConcurrencyStatusEnum.IN_PROGRESS)
        elif target == 0:
            # The scale-down gate should have drained all leases before this
            # point; wait defensively so the account is never closed while an
            # in-flight decrement is still outstanding.
            deadline = time.monotonic() + SCALING_WAIT_TIMEOUT_SECONDS
            while account.snapshot().in_flight > 0 and not self._closing:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    LOG.error(
                        "Provisioned concurrency account for %s still has %d in-flight lease(s) "
                        "at target zero; keeping the account open",
                        self.qualified_arn,
                        account.snapshot().in_flight,
                    )
                    return
                account.wait_for_reclaim_capacity(min(RECLAIM_WAIT_POLL_SECONDS, remaining))
            if not self._closing:
                self._close_account(account)
        else:
            account.set_status(ProvisionedConcurrencyStatusEnum.READY)

    def _close_account(self, account: ProvisionedConcurrencyAccount) -> None:
        with self._condition:
            qualifier = self._qualifier
            self._account = None
            self._desired = None
            self._applied = 0
            self._status = ProvisionedConcurrencyStatusEnum.IN_PROGRESS
            self._status_reason = None
        self._ledger.close(
            account_id=account.account_id,
            region=account.region,
            function_name=account.function_name,
            qualifier=qualifier or account.qualifier,
            qualified_arn=account.qualified_arn,
        )

    def _wait_for_future(self, future: Future[None], deadline: float) -> bool:
        """Wait for an environment start future without blocking shutdown."""
        while not self._closing:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            done, _not_done = concurrent.futures.wait([future], timeout=min(1.0, remaining))
            if done:
                return True
        return future.done()


class EnvironmentScalingTimeout(Exception):
    """Raised when an execution environment does not start within the scaling
    wait budget."""
