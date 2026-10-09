import abc
import functools
import logging
import threading
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from plux import Plugin, PluginLifecycleListener, PluginManager, PluginSpec

from localstack import config
from localstack.aws.skeleton import DispatchTable, Skeleton
from localstack.aws.spec import load_service
from localstack.config import ServiceProviderConfig
from localstack.runtime import hooks
from localstack.state import StateLifecycleHook, StateVisitable, StateVisitor
from localstack.utils.bootstrap import get_enabled_apis, is_api_enabled, log_duration
from localstack.utils.functions import call_safe
from localstack.utils.sync import SynchronizedDefaultDict

# set up logger
LOG = logging.getLogger(__name__)

# namespace for AWS provider plugins
PLUGIN_NAMESPACE = "localstack.aws.provider"

_default = object()  # sentinel object indicating a default value


# -----------------
# PLUGIN UTILITIES
# -----------------


class ServiceException(Exception):
    pass


class ServiceDisabled(ServiceException):
    pass


class ServiceStateException(ServiceException):
    pass


class IllegalServiceStateTransition(ServiceStateException):
    """Raised when a requested lifecycle state transition is not allowed by the state machine. The
    entire operation is rejected: the service remains in its current state, and no side effects occur."""

    def __init__(self, service_name: str, current: "ServiceState", target: "ServiceState") -> None:
        super().__init__(
            f"illegal state transition for service {service_name}: {current.value} -> {target.value}"
        )
        self.service_name = service_name
        self.current_state = current
        self.target_state = target


class ServiceLifecycleHook(StateLifecycleHook):
    def on_after_init(self):
        pass

    def on_before_start(self):
        pass

    def on_before_stop(self):
        pass

    def on_exception(self):
        pass


class ServiceProvider(Protocol):
    service: str


class Service:
    """
    FIXME: this has become frankenstein's monster, and it has to go. once we've rid ourselves of the legacy edge
     proxy, we can get rid of the ``listener`` concept. we should then do one iteration over all the
     ``start_dynamodb``, ``start_<whatever>``, ``check_<whatever>``, etc. methods, to make all of those integral part
     of the service provider. the assumption that every service provider starts a backend server is outdated, and then
     we can get rid of ``start``, and ``check``.
    """

    def __init__(
        self,
        name,
        start=_default,
        check=None,
        skeleton=None,
        active=False,
        stop=None,
        lifecycle_hook: ServiceLifecycleHook = None,
    ):
        self.plugin_name = name
        self.start_function = start
        self.skeleton = skeleton
        self.check_function = check
        self.default_active = active
        self.stop_function = stop
        self.lifecycle_hook = lifecycle_hook or ServiceLifecycleHook()
        self._provider = None
        call_safe(self.lifecycle_hook.on_after_init)

    def start(self, asynchronous):
        call_safe(self.lifecycle_hook.on_before_start)

        if not self.start_function:
            return

        if self.start_function is _default:
            return

        kwargs = {"asynchronous": asynchronous}
        if self.skeleton:
            kwargs["update_listener"] = self.skeleton
        return self.start_function(**kwargs)

    def stop(self):
        call_safe(self.lifecycle_hook.on_before_stop)
        if not self.stop_function:
            return
        return self.stop_function()

    def check(self, expect_shutdown=False, print_error=False):
        if not self.check_function:
            return
        return self.check_function(expect_shutdown=expect_shutdown, print_error=print_error)

    def name(self):
        return self.plugin_name

    def is_enabled(self):
        return is_api_enabled(self.plugin_name)

    def accept_state_visitor(self, visitor: StateVisitor):
        """
        Passes the StateVisitor to the ASF provider if it is set and implements the StateVisitable. Otherwise, it uses
        the ReflectionStateLocator to visit the service state.

        :param visitor: the visitor
        """
        if self._provider and isinstance(self._provider, StateVisitable):
            self._provider.accept_state_visitor(visitor)
            return

        from localstack.state.inspect import ReflectionStateLocator

        ReflectionStateLocator(service=self.name()).accept_state_visitor(visitor)

    @staticmethod
    def for_provider(
        provider: ServiceProvider,
        dispatch_table_factory: Callable[[ServiceProvider], DispatchTable] = None,
        service_lifecycle_hook: ServiceLifecycleHook = None,
    ) -> "Service":
        """
        Factory method for creating services for providers. This method hides a bunch of legacy code and
        band-aids/adapters to make persistence visitors work, while providing compatibility with the legacy edge proxy.

        :param provider: the service provider, i.e., the implementation of the generated ASF service API.
        :param dispatch_table_factory: a `MotoFallbackDispatcher` or something similar that uses the provider to
            create a dispatch table. this one's a bit clumsy.
        :param service_lifecycle_hook: if left empty, the factory checks whether the provider is a ServiceLifecycleHook.
        :return: a service instance
        """
        # determine the service_lifecycle_hook
        if service_lifecycle_hook is None:
            if isinstance(provider, ServiceLifecycleHook):
                service_lifecycle_hook = provider

        # determine the delegate for injecting into the skeleton
        delegate = dispatch_table_factory(provider) if dispatch_table_factory else provider
        service = Service(
            name=provider.service,
            skeleton=Skeleton(load_service(provider.service), delegate),
            lifecycle_hook=service_lifecycle_hook,
        )
        service._provider = provider

        return service


class ServiceState(Enum):
    UNKNOWN = "unknown"
    AVAILABLE = "available"
    DISABLED = "disabled"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


# time (in seconds) callers wait for an in-progress start/stop before giving up
SERVICE_START_WAIT_TIMEOUT = 30.0
# time (in seconds) a stop waits for in-flight requests to finish before force-stopping the service
SERVICE_STOP_DRAIN_TIMEOUT = 5.0


# Legal transitions of the service lifecycle state machine. This table is the single source of truth
# shared by the service containers, the service manager, the request handler chain and the shutdown
# hooks. Any transition that is not listed here is rejected as a whole (no partial state change).
SERVICE_STATE_TRANSITIONS: dict[ServiceState, frozenset[ServiceState]] = {
    ServiceState.UNKNOWN: frozenset({ServiceState.AVAILABLE, ServiceState.DISABLED}),
    ServiceState.AVAILABLE: frozenset({ServiceState.STARTING, ServiceState.DISABLED}),
    ServiceState.DISABLED: frozenset(),
    ServiceState.STARTING: frozenset(
        {ServiceState.RUNNING, ServiceState.AVAILABLE, ServiceState.ERROR}
    ),
    ServiceState.RUNNING: frozenset({ServiceState.STOPPING, ServiceState.ERROR}),
    ServiceState.STOPPING: frozenset({ServiceState.STOPPED, ServiceState.ERROR}),
    ServiceState.STOPPED: frozenset({ServiceState.STARTING, ServiceState.DISABLED}),
    ServiceState.ERROR: frozenset({ServiceState.AVAILABLE, ServiceState.DISABLED}),
}


def can_transition(current: ServiceState, target: ServiceState) -> bool:
    return target in SERVICE_STATE_TRANSITIONS.get(current, frozenset())


def is_startable(state: ServiceState) -> bool:
    return can_transition(state, ServiceState.STARTING)


def is_stoppable(state: ServiceState) -> bool:
    return can_transition(state, ServiceState.STOPPING)


def is_terminal(state: ServiceState) -> bool:
    return not SERVICE_STATE_TRANSITIONS.get(state, frozenset())


def is_servable(state: ServiceState) -> bool:
    return state == ServiceState.RUNNING


@dataclass
class LifecyclePhase:
    """
    A declared assembly phase. ``action`` executes the phase; ``rollback`` (if declared) reverses the
    effects of a completed phase when a later phase fails.
    """

    name: str
    action: Callable[[], Any]
    rollback: Callable[[], Any] | None = None


@dataclass
class ServiceFailure:
    """
    Explains an ``ERROR`` state: the phase during which assembly or shutdown failed, the captured error,
    and the names of the phases that have been rolled back (in rollback execution order).
    """

    phase: str
    error: Exception
    rolled_back_phases: list[str]
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class ServiceStopResult:
    """Result of a stop attempt."""

    service: str
    state: ServiceState
    interrupted_requests: int = 0
    drained: bool = True
    error: Exception | None = None

    @property
    def stopped(self) -> bool:
        return self.state == ServiceState.STOPPED


class _PhaseFailure(Exception):
    """Internal signal that an assembly phase raised. Carries index, phase name and original error."""

    def __init__(self, index: int, phase: str, error: Exception) -> None:
        super().__init__(str(error))
        self.index = index
        self.phase = phase
        self.error = error


class ServiceContainer:
    """
    Holds a service, its state, and exposes lifecycle methods of the service.

    The container implements a reentrant, rollback-capable lifecycle state machine:

      * State transitions are validated against ``SERVICE_STATE_TRANSITIONS``. Illegal transitions raise
        ``IllegalServiceStateTransition`` and leave the container completely untouched.
      * Assembly is single-flight: concurrent ``assemble()`` calls for the same service trigger exactly
        one assembly; all other callers block on the container condition and share the outcome (either
        the running service or the very same captured error).
      * If an assembly phase fails, all completed phases are rolled back in reverse declaration order
        and the container settles in an explainable ``ERROR`` state (see ``ServiceFailure``). The only
        ways out are ``retry()`` (re-assemble, bringing the service back to ``RUNNING``) or
        ``deactivate()`` (permanently ``DISABLED``).
      * ``stop()`` drains in-flight requests, invokes the service stop function, and reports the number
        of requests that were still in flight when the drain timeout was reached.
    """

    service: Service
    state: ServiceState
    lock: threading.RLock
    errors: list[Exception]

    def __init__(self, service: Service, state=ServiceState.UNKNOWN):
        self.service = service
        self.state = state
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.errors = []
        self.failure: ServiceFailure | None = None
        self.inflight_requests = 0
        self._request_generation = 0
        self._phases: list[LifecyclePhase] | None = None

    def get(self) -> Service:
        return self.service

    # ------------------------------------------------------------------
    # state machine primitives
    # ------------------------------------------------------------------

    def _transition(self, target: ServiceState) -> None:
        """Transition to ``target``. Must be called while holding ``self.condition``."""
        current = self.state
        if current == target:
            return
        if not can_transition(current, target):
            raise IllegalServiceStateTransition(self.service.name(), current, target)
        LOG.debug(
            "service %s state transition: %s -> %s",
            self.service.name(),
            current.value,
            target.value,
        )
        self.state = target

    def _capture_failure(
        self, phase: str, error: Exception, rolled_back_phases: list[str]
    ) -> ServiceFailure:
        """Must be called while holding ``self.condition``."""
        failure = ServiceFailure(
            phase=phase,
            error=error,
            rolled_back_phases=rolled_back_phases,
            message=(
                f"service {self.service.name()} failed in phase '{phase}': {error}; "
                f"rolled back phases: {rolled_back_phases or '<none>'}"
            ),
        )
        self.errors.append(error)
        self.failure = failure
        return failure

    def _failure_exception(self) -> Exception:
        if self.failure is not None:
            return self.failure.error
        if self.errors:
            return self.errors[-1]
        return ServiceStateException(
            f"service {self.service.name()} is in error state without a captured error"
        )

    # ------------------------------------------------------------------
    # assembly phases
    # ------------------------------------------------------------------

    def _build_phases(self) -> list[LifecyclePhase]:
        """
        Declares the assembly phases in declaration order. These invoke the exact same callables, in the
        exact same order and with the exact same keyword arguments as the previous (linear) startup path
        ``Service.start(asynchronous=True)`` followed by ``Service.check(print_error=True)``.
        """
        phases = [
            LifecyclePhase(
                name="before_start",
                action=lambda: call_safe(self.service.lifecycle_hook.on_before_start),
            )
        ]

        start_function = self.service.start_function
        if start_function is not None and start_function is not _default:
            skeleton = self.service.skeleton

            def _start():
                kwargs = {"asynchronous": True}
                if skeleton:
                    kwargs["update_listener"] = skeleton
                return start_function(**kwargs)

            # a completed start phase is reclaimed by the service's regular stop procedure
            phases.append(LifecyclePhase(name="start", action=_start, rollback=self._stop_service))

        phases.append(
            LifecyclePhase(
                name="check",
                action=lambda: self.service.check(print_error=True),
            )
        )
        return phases

    @property
    def phases(self) -> list[LifecyclePhase]:
        if self._phases is None:
            self._phases = self._build_phases()
        return self._phases

    def _run_phases(self) -> None:
        for index, phase in enumerate(self.phases):
            try:
                phase.action()
            except Exception as e:
                raise _PhaseFailure(index=index, phase=phase.name, error=e) from e

    def _rollback_completed_phases(self, failed_index: int) -> list[str]:
        """Rolls back completed phases (indices before the failing one) in reverse declaration order."""
        rolled_back: list[str] = []
        for phase in reversed(self.phases[:failed_index]):
            if phase.rollback is None:
                continue
            try:
                phase.rollback()
            except Exception as e:
                LOG.error(
                    "error while rolling back phase '%s' of service %s: %s",
                    phase.name,
                    self.service.name(),
                    e,
                )
            rolled_back.append(phase.name)
        return rolled_back

    def _stop_service(self) -> Any:
        """Invokes the legacy stop procedure (``Service.stop``) used as start-phase compensation."""
        return self.service.stop()

    # ------------------------------------------------------------------
    # lifecycle operations
    # ------------------------------------------------------------------

    def assemble(self, timeout: float = SERVICE_START_WAIT_TIMEOUT) -> Service:
        """
        Returns the running service, or raises the error that caused assembly to fail. Assembly is
        single-flight: only the first eligible caller executes the assembly phases, all concurrent
        callers wait on the container and receive the same result.
        """
        is_leader = False

        with self.condition:
            if self.state == ServiceState.STARTING:
                if not self.condition.wait_for(
                    lambda: self.state != ServiceState.STARTING, timeout
                ):
                    raise TimeoutError(
                        f"gave up waiting for service {self.service.name()} to start"
                    )

            if self.state == ServiceState.STOPPING:
                if not self.condition.wait_for(
                    lambda: self.state != ServiceState.STOPPING, timeout
                ):
                    raise TimeoutError(f"gave up waiting for service {self.service.name()} to stop")
                # another caller may have already restarted the service after the stop completed
                if self.state == ServiceState.STARTING:
                    if not self.condition.wait_for(
                        lambda: self.state != ServiceState.STARTING, timeout
                    ):
                        raise TimeoutError(
                            f"gave up waiting for service {self.service.name()} to start"
                        )

            if self.state == ServiceState.RUNNING:
                return self.service
            if self.state == ServiceState.DISABLED:
                raise ServiceDisabled(f"service {self.service.name()} is disabled")
            if self.state == ServiceState.ERROR:
                raise self._failure_exception()
            if self.state in (ServiceState.AVAILABLE, ServiceState.STOPPED):
                self._transition(ServiceState.STARTING)
                # a fresh generation: in-flight tickets from a previous (possibly force-stopped)
                # generation are invalid and the counter starts at zero
                self._request_generation += 1
                self.inflight_requests = 0
                is_leader = True
            else:
                raise ServiceStateException(
                    f"service {self.service.name()} is not ready ({self.state.value}) and could not "
                    f"be started"
                )

        if is_leader:
            try:
                self._run_phases()
            except _PhaseFailure as pf:
                # phases are executed outside the lock; roll back everything that completed, in reverse
                # declaration order, then settle in the explainable ERROR state.
                rolled_back = self._rollback_completed_phases(pf.index)
                call_safe(self.service.lifecycle_hook.on_exception)
                LOG.error(
                    "error while starting service %s in phase '%s': %s",
                    self.service.name(),
                    pf.phase,
                    pf.error,
                )
                with self.condition:
                    self._capture_failure(pf.phase, pf.error, rolled_back)
                    self._transition(ServiceState.ERROR)
                    self.condition.notify_all()
                raise pf.error
            else:
                with self.condition:
                    self._transition(ServiceState.RUNNING)
                    self.condition.notify_all()
                return self.service

        # concurrent callers re-evaluate the state published by the leader and share its outcome
        with self.condition:
            if self.state == ServiceState.RUNNING:
                return self.service
            if self.state == ServiceState.DISABLED:
                raise ServiceDisabled(f"service {self.service.name()} is disabled")
            if self.state == ServiceState.ERROR:
                raise self._failure_exception()

        raise ServiceStateException(
            f"service {self.service.name()} is not ready ({self.state.value}) and could not be started"
        )

    def retry(self, timeout: float = SERVICE_START_WAIT_TIMEOUT) -> Service:
        """
        Explicit retry entry. Only legal from the ``ERROR`` state: clears the captured failure, moves
        the service back to ``AVAILABLE`` and runs a fresh single-flight assembly. On success the
        service is ``RUNNING`` again; on failure it settles in a new explainable ``ERROR`` state.
        """
        with self.condition:
            if self.state == ServiceState.RUNNING:
                return self.service
            if self.state != ServiceState.ERROR:
                raise IllegalServiceStateTransition(
                    self.service.name(), self.state, ServiceState.AVAILABLE
                )
            self.errors.clear()
            self.failure = None
            self._transition(ServiceState.AVAILABLE)
            self.condition.notify_all()

        return self.assemble(timeout)

    def deactivate(self) -> None:
        """
        Permanently decommissions the service (``DISABLED``). Only legal from a quiescent state
        (``AVAILABLE``, ``STOPPED`` or ``ERROR``); services that are starting, running or stopping must
        be stopped first.
        """
        with self.condition:
            if self.state == ServiceState.DISABLED:
                return
            if not can_transition(self.state, ServiceState.DISABLED):
                raise IllegalServiceStateTransition(
                    self.service.name(), self.state, ServiceState.DISABLED
                )
            self._transition(ServiceState.DISABLED)
            self.condition.notify_all()

    def check(self) -> bool:
        """
        Health-check path (not part of lazy assembly). Preserves the previous behavior: a failed check
        puts the service in ``ERROR`` without tearing the backend down; a successful check of an errored
        service restores ``RUNNING``.
        """
        try:
            self.service.check(print_error=True)
        except Exception as e:
            LOG.error("error while checking service %s: %s", self.service.name(), e)
            with self.condition:
                if self.state == ServiceState.RUNNING:
                    self._capture_failure("check", e, [])
                    self._transition(ServiceState.ERROR)
                elif self.state == ServiceState.ERROR:
                    # already failed: refresh the captured error but keep the state
                    rolled_back = self.failure.rolled_back_phases if self.failure else []
                    self._capture_failure("check", e, rolled_back)
                self.condition.notify_all()
            return False

        with self.condition:
            if self.state == ServiceState.ERROR:
                self.errors.clear()
                self.failure = None
                self._transition(ServiceState.AVAILABLE)
                self._transition(ServiceState.RUNNING)
                self.condition.notify_all()
        return True

    def stop(self, drain_timeout: float = SERVICE_STOP_DRAIN_TIMEOUT) -> ServiceStopResult:
        """
        Stops the service after draining in-flight requests. If in-flight requests do not finish within
        ``drain_timeout``, the service is force-stopped anyway and the number of interrupted requests is
        recorded in the result.
        """
        name = self.service.name()

        with self.condition:
            # never cut across an in-progress assembly - wait for it and stop the result
            if self.state == ServiceState.STARTING:
                if not self.condition.wait_for(
                    lambda: self.state != ServiceState.STARTING, drain_timeout
                ):
                    raise TimeoutError(f"gave up waiting for service {name} to start")

            # serialize concurrent stops: everybody observes the same terminal stop result
            if self.state == ServiceState.STOPPING:
                self.condition.wait_for(lambda: self.state != ServiceState.STOPPING, drain_timeout)

            if self.state == ServiceState.STOPPED:
                return ServiceStopResult(name, ServiceState.STOPPED)

            # nothing was ever started (or a previous attempt already failed/was disabled)
            if self.state in (
                ServiceState.AVAILABLE,
                ServiceState.DISABLED,
                ServiceState.ERROR,
            ):
                return ServiceStopResult(name, self.state)

            if self.state != ServiceState.RUNNING:
                raise IllegalServiceStateTransition(name, self.state, ServiceState.STOPPING)

            self._transition(ServiceState.STOPPING)
            # wait for in-flight requests of the current generation to finish
            drained = self.condition.wait_for(lambda: self.inflight_requests == 0, drain_timeout)
            interrupted_requests = 0 if drained else self.inflight_requests

        # invoke the (potentially blocking) stop function without holding the lock
        error = None
        try:
            self.service.stop()
        except Exception as e:
            error = e
            LOG.error("error while stopping service %s: %s", name, e)

        with self.condition:
            # invalidate tickets of requests still in flight after (possibly forced) shutdown, so that
            # late end_request calls do not leak into the next running generation
            self._request_generation += 1
            if error is not None:
                self._capture_failure("stop", error, [])
                self._transition(ServiceState.ERROR)
                result = ServiceStopResult(
                    name,
                    ServiceState.ERROR,
                    interrupted_requests=interrupted_requests,
                    drained=drained,
                    error=error,
                )
            else:
                if interrupted_requests:
                    LOG.warning(
                        "force-stopped service %s with %d in-flight request(s) still running after "
                        "%.1fs drain timeout",
                        name,
                        interrupted_requests,
                        drain_timeout,
                    )
                self._transition(ServiceState.STOPPED)
                result = ServiceStopResult(
                    name,
                    ServiceState.STOPPED,
                    interrupted_requests=interrupted_requests,
                    drained=drained,
                )
            self.condition.notify_all()

        return result

    def start(self) -> bool:
        """Compatibility wrapper around the single-flight ``assemble``."""
        try:
            self.assemble()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # in-flight request tracking (used by the request handler chain)
    # ------------------------------------------------------------------

    def begin_request(self) -> int | None:
        """
        Registers an in-flight request. Returns an opaque request-generation ticket, or ``None`` if the
        service is not currently serving requests (e.g. already stopping/stopped/disabled).
        """
        with self.condition:
            if self.state != ServiceState.RUNNING:
                return None
            self.inflight_requests += 1
            return self._request_generation

    def end_request(self, ticket: int | None) -> None:
        with self.condition:
            if ticket is None or ticket != self._request_generation:
                return
            if self.inflight_requests > 0:
                self.inflight_requests -= 1
            if self.inflight_requests == 0:
                self.condition.notify_all()


class ServiceManager:
    def __init__(self) -> None:
        super().__init__()
        self._services: dict[str, ServiceContainer] = {}
        self._mutex = threading.RLock()

    def get_service_container(self, name: str) -> ServiceContainer | None:
        return self._services.get(name)

    def get_service(self, name: str) -> Service | None:
        container = self.get_service_container(name)
        return container.service if container else None

    def add_service(self, service: Service) -> bool:
        state = ServiceState.AVAILABLE if service.is_enabled() else ServiceState.DISABLED
        self._services[service.name()] = ServiceContainer(service, state)

        return True

    def list_available(self) -> list[str]:
        return list(self._services.keys())

    def exists(self, name: str) -> bool:
        return name in self._services

    def is_running(self, name: str) -> bool:
        return self.get_state(name) == ServiceState.RUNNING

    def check(self, name: str) -> bool:
        container = self.get_service_container(name)
        if container and container.state in [ServiceState.RUNNING, ServiceState.ERROR]:
            return container.check()

    def check_all(self):
        return any(self.check(service_name) for service_name in self.list_available())

    def get_state(self, name: str) -> ServiceState | None:
        container = self.get_service_container(name)
        return container.state if container else None

    def get_failure(self, name: str) -> ServiceFailure | None:
        container = self.get_service_container(name)
        return container.failure if container else None

    def get_states(self) -> dict[str, ServiceState]:
        return {name: self.get_state(name) for name in self.list_available()}

    @log_duration()
    def require(self, name: str) -> Service:
        """
        High level function that always returns a running service, or raises an error. If the service is in a state
        that it could be transitioned into a running state, then invoking this function will attempt that transition,
        e.g., by starting the service if it is available. Concurrent calls for the same service trigger exactly one
        assembly and share its result. A service in ``ERROR`` raises the captured failure; use ``retry`` to re-enter
        the assembly, or ``deactivate`` to disable it permanently.
        """
        container = self.get_service_container(name)

        if not container:
            raise ValueError(f"no such service {name}")

        return container.assemble()

    def retry(self, name: str) -> Service:
        """
        Explicit retry entry for a failed service. Resets the explainable failure state and re-runs the
        single-flight assembly. Returns the running service or raises the new captured error.
        """
        container = self.get_service_container(name)

        if not container:
            raise ValueError(f"no such service {name}")

        return container.retry()

    def deactivate(self, name: str) -> None:
        """
        Permanently decommissions the service. Legal from ``AVAILABLE``, ``STOPPED`` or ``ERROR``; the
        service then rejects every ``require`` with ``ServiceDisabled``.
        """
        container = self.get_service_container(name)

        if not container:
            raise ValueError(f"no such service {name}")

        container.deactivate()

    def stop_service(
        self, name: str, drain_timeout: float = SERVICE_STOP_DRAIN_TIMEOUT
    ) -> ServiceStopResult:
        """Stops a single service after draining its in-flight requests."""
        container = self.get_service_container(name)

        if not container:
            raise ValueError(f"no such service {name}")

        return container.stop(drain_timeout)

    def begin_request(self, name: str) -> int | None:
        container = self.get_service_container(name)
        return container.begin_request() if container else None

    def end_request(self, name: str, ticket: int | None) -> None:
        container = self.get_service_container(name)
        if container:
            container.end_request(ticket)

    # legacy map compatibility

    def items(self):
        return {
            container.service.name(): container.service for container in self._services.values()
        }.items()

    def keys(self):
        return self._services.keys()

    def values(self):
        return [container.service for container in self._services.values()]

    def get(self, key):
        return self.get_service(key)

    def __iter__(self):
        return self._services


class ServicePlugin(Plugin):
    service: Service
    api: str

    @abc.abstractmethod
    def create_service(self) -> Service:
        raise NotImplementedError

    def load(self):
        self.service = self.create_service()
        return self.service


class ServicePluginAdapter(ServicePlugin):
    def __init__(
        self,
        api: str,
        create_service: Callable[[], Service],
        should_load: Callable[[], bool] = None,
    ) -> None:
        super().__init__()
        self.api = api
        self._create_service = create_service
        self._should_load = should_load

    def should_load(self) -> bool:
        if self._should_load:
            return self._should_load()
        return True

    def create_service(self) -> Service:
        return self._create_service()


def aws_provider(api: str = None, name="default", should_load: Callable[[], bool] = None):
    """
    Decorator for marking methods that create a Service instance as a ServicePlugin. Methods marked with this
    decorator are discoverable as a PluginSpec within the namespace "localstack.aws.provider", with the name
    "<api>:<name>". If api is not explicitly specified, then the method name is used as api name.
    """

    def wrapper(fn):
        # sugar for being able to name the function like the api
        _api = api or fn.__name__

        # this causes the plugin framework into pointing the entrypoint to the original function rather than the
        # nested factory function
        @functools.wraps(fn)
        def factory() -> ServicePluginAdapter:
            return ServicePluginAdapter(api=_api, should_load=should_load, create_service=fn)

        return PluginSpec(PLUGIN_NAMESPACE, f"{_api}:{name}", factory=factory)

    return wrapper


class ServicePluginErrorCollector(PluginLifecycleListener):
    """
    A PluginLifecycleListener that collects errors related to service plugins.
    """

    errors: dict[tuple[str, str], Exception]  # keys are: (api, provider)

    def __init__(self, errors: dict[str, Exception] = None) -> None:
        super().__init__()
        self.errors = errors or {}

    def get_key(self, plugin_name) -> tuple[str, str]:
        # the convention is <api>:<provider>, currently we don't really expose the provider
        # TODO: faulty plugin names would break this
        return tuple(plugin_name.split(":", maxsplit=1))

    def on_resolve_exception(self, namespace: str, entrypoint, exception: Exception):
        self.errors[self.get_key(entrypoint.name)] = exception

    def on_init_exception(self, plugin_spec: PluginSpec, exception: Exception):
        self.errors[self.get_key(plugin_spec.name)] = exception

    def on_load_exception(self, plugin_spec: PluginSpec, plugin: Plugin, exception: Exception):
        self.errors[self.get_key(plugin_spec.name)] = exception

    def has_errors(self, api: str, provider: str = None) -> bool:
        for e_api, e_provider in self.errors.keys():
            if api == e_api:
                if not provider:
                    return True
                else:
                    return e_provider == provider

        return False


class ServicePluginManager(ServiceManager):
    plugin_manager: PluginManager[ServicePlugin]
    plugin_errors: ServicePluginErrorCollector

    def __init__(
        self,
        plugin_manager: PluginManager[ServicePlugin] = None,
        provider_config: ServiceProviderConfig = None,
    ) -> None:
        super().__init__()
        self.plugin_errors = ServicePluginErrorCollector()
        self.plugin_manager = plugin_manager or PluginManager(
            PLUGIN_NAMESPACE, listener=self.plugin_errors
        )
        self._api_provider_specs = None
        self.provider_config = provider_config or config.SERVICE_PROVIDER_CONFIG

        # locks used to make sure plugin loading is thread safe - will be cleared after single use
        self._plugin_load_locks: dict[str, threading.RLock] = SynchronizedDefaultDict(
            threading.RLock
        )

    def get_active_provider(self, service: str) -> str:
        """
        Get configured provider for a given service

        :param service: Service name
        :return: configured provider
        """
        return self.provider_config.get_provider(service)

    def get_default_provider(self) -> str:
        """
        Get the default provider

        :return: default provider
        """
        return self.provider_config.default_value

    # TODO make the abstraction clearer, to provide better information if service is available versus discoverable
    # especially important when considering pro services
    def list_available(self) -> list[str]:
        """
        List all available services, which have an available, configured provider

        :return: List of service names
        """
        return [
            service
            for service, providers in self.api_provider_specs.items()
            if self.get_active_provider(service) in providers
        ]

    def _get_loaded_service_containers(
        self, services: list[str] | None = None
    ) -> list[ServiceContainer]:
        """
        Returns all the available service containers.
        :param services: the list of services to restrict the search to. If empty or NULL then service containers for
                         all available services are queried.
        :return: a list of all the available service containers.
        """
        services = services or self.list_available()
        return [
            c for s in services if (c := super(ServicePluginManager, self).get_service_container(s))
        ]

    def list_loaded_services(self) -> list[str]:
        """
        Lists all the services which have a provider that has been initialized

        :return: a list of service names
        """
        return [
            service_container.service.name()
            for service_container in self._get_loaded_service_containers()
        ]

    def list_active_services(self) -> list[str]:
        """
        Lists all services that have an initialised provider and are currently running.

        :return: the list of active service names.
        """
        return [
            service_container.service.name()
            for service_container in self._get_loaded_service_containers()
            if service_container.state == ServiceState.RUNNING
        ]

    def exists(self, name: str) -> bool:
        return name in self.list_available()

    def get_state(self, name: str) -> ServiceState | None:
        if name in self._services:
            # ServiceContainer exists, which means the plugin has been loaded
            return super().get_state(name)

        if not self.exists(name):
            # there's definitely no service with this name
            return None

        # if a PluginSpec exists, then we can get the container and check whether there was an error loading the plugin
        provider = self.get_active_provider(name)
        if self.plugin_errors.has_errors(name, provider):
            return ServiceState.ERROR

        return ServiceState.AVAILABLE if is_api_enabled(name) else ServiceState.DISABLED

    def get_service_container(self, name: str) -> ServiceContainer | None:
        if container := self._services.get(name):
            return container

        if not self.exists(name):
            return None

        load_lock = self._plugin_load_locks[name]
        with load_lock:
            # check once again to avoid race conditions
            if container := self._services.get(name):
                return container

            # this is where we start lazy loading. we now know the PluginSpec for the API exists,
            # but the ServiceContainer has not been created.
            # this control path will be executed once per service
            plugin = self._load_service_plugin(name)
            if not plugin or not plugin.service:
                return None

            with self._mutex:
                super().add_service(plugin.service)

            del self._plugin_load_locks[name]  # we only needed the service lock once

            return self._services.get(name)

    @property
    def api_provider_specs(self) -> dict[str, list[str]]:
        """
        Returns all provider names within the service plugin namespace and parses their name according to the convention,
        that is "<api>:<provider>". The result is a dictionary that maps api => List[str (name of a provider)].
        """
        if self._api_provider_specs is not None:
            return self._api_provider_specs

        with self._mutex:
            if self._api_provider_specs is None:
                self._api_provider_specs = self._resolve_api_provider_specs()
            return self._api_provider_specs

    @log_duration()
    def _load_service_plugin(self, name: str) -> ServicePlugin | None:
        providers = self.api_provider_specs.get(name)
        if not providers:
            # no providers for this api
            return None

        preferred_provider = self.get_active_provider(name)
        if preferred_provider in providers:
            provider = preferred_provider
        else:
            default = self.get_default_provider()
            LOG.warning(
                "Configured provider (%s) does not exist for service (%s). Available options are: %s. "
                "Falling back to default provider '%s'. This can impact the availability of Pro functionality, "
                "please fix this configuration issue as soon as possible.",
                preferred_provider,
                name,
                providers,
                default,
            )
            provider = default

        plugin_name = f"{name}:{provider}"
        plugin = self.plugin_manager.load(plugin_name)
        plugin.name = plugin_name

        return plugin

    @log_duration()
    def _resolve_api_provider_specs(self) -> dict[str, list[str]]:
        result = defaultdict(list)

        for spec in self.plugin_manager.list_plugin_specs():
            api, provider = spec.name.split(
                ":"
            )  # TODO: error handling, faulty plugins could break the runtime
            result[api].append(provider)

        return result

    def apis_with_provider(self, provider: str) -> list[str]:
        """
        Lists all apis where a given provider exists for.
        :param provider: Name of the provider
        :return: List of apis the given provider provides
        """
        apis = []
        for api, providers in self.api_provider_specs.items():
            if provider in providers:
                apis.append(api)
        return apis

    def _stop_services(
        self,
        service_containers: list[ServiceContainer],
        drain_timeout: float = SERVICE_STOP_DRAIN_TIMEOUT,
    ) -> list[ServiceStopResult]:
        """
        Stops all given service containers through the common lifecycle state machine. Each stop drains
        in-flight requests; containers that are not running are skipped by the state machine itself.

        :param service_containers: the list of service containers to be stopped.
        :param drain_timeout: per-service timeout to wait for in-flight requests to finish.
        :return: the stop results, including the number of interrupted requests per service.
        """
        results: list[ServiceStopResult] = []
        with self._mutex:
            containers = list(service_containers)
        for service_container in containers:
            results.append(service_container.stop(drain_timeout))
        return results

    def stop_services(
        self,
        services: list[str] = None,
        drain_timeout: float = SERVICE_STOP_DRAIN_TIMEOUT,
    ) -> list[ServiceStopResult]:
        """
        Stops services for this service manager, if they are currently active.
        Will not stop services not already started or in and error state.

        :param services: Service names to stop. If not provided, all services for this manager will be stopped.
        :param drain_timeout: per-service timeout to wait for in-flight requests to finish.
        :return: the stop results, including the number of interrupted requests per service.
        """
        target_service_containers = self._get_loaded_service_containers(services=services)
        return self._stop_services(target_service_containers, drain_timeout)

    def stop_all_services(
        self, drain_timeout: float = SERVICE_STOP_DRAIN_TIMEOUT
    ) -> list[ServiceStopResult]:
        """
        Stops all services for this service manager, if they are currently active.
        Will not stop services not already started or in and error state.

        :param drain_timeout: per-service timeout to wait for in-flight requests to finish.
        :return: the stop results, including the number of interrupted requests per service.
        """
        target_service_containers = self._get_loaded_service_containers()
        return self._stop_services(target_service_containers, drain_timeout)


# map of service plugins, mapping from service name to plugin details
SERVICE_PLUGINS: ServicePluginManager = ServicePluginManager()


# -----------------------------
# INFRASTRUCTURE HEALTH CHECKS
# -----------------------------


def wait_for_infra_shutdown():
    apis = get_enabled_apis()

    names = [name for name, plugin in SERVICE_PLUGINS.items() if name in apis]

    def check(name):
        check_service_health(api=name, expect_shutdown=True)
        LOG.debug("[shutdown] api %s has shut down", name)

    # no special significance to 10 workers, seems like a reasonable number given the number of services we have
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(check, names)


def check_service_health(api, expect_shutdown=False):
    status = SERVICE_PLUGINS.check(api)
    if status == expect_shutdown:
        if not expect_shutdown:
            LOG.warning('Service "%s" not yet available, retrying...', api)
        else:
            LOG.warning('Service "%s" still shutting down, retrying...', api)
        raise Exception(f"Service check failed for api: {api}")


@hooks.on_infra_start(should_load=lambda: config.EAGER_SERVICE_LOADING)
def eager_load_services():
    from localstack.utils.bootstrap import get_preloaded_services

    preloaded_apis = get_preloaded_services()
    LOG.debug("Eager loading services: %s", sorted(preloaded_apis))

    for api in preloaded_apis:
        try:
            SERVICE_PLUGINS.require(api)
        except ServiceDisabled as e:
            LOG.debug("%s", e)
        except Exception:
            LOG.error(
                "could not load service plugin %s",
                api,
                exc_info=LOG.isEnabledFor(logging.DEBUG),
            )
