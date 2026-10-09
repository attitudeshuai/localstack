import contextlib
import logging
import threading
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor

from localstack.services.lambda_.invocation.execution_environment import (
    EnvironmentStartupTimeoutException,
    ExecutionEnvironment,
    InvalidStatusException,
    RuntimeStatus,
)
from localstack.services.lambda_.invocation.executor_endpoint import StatusErrorException
from localstack.services.lambda_.invocation.lambda_models import (
    FunctionVersion,
    InitializationType,
    OtherServiceEndpoint,
)

LOG = logging.getLogger(__name__)


class AssignmentException(Exception):
    pass


class AssignmentService(OtherServiceEndpoint):
    """
    scope: LocalStack global
    """

    # function_version manager id => runtime_environment_id => runtime_environment
    environments: dict[str, dict[str, ExecutionEnvironment]]

    # Global pool for spawning and killing provisioned Lambda runtime environments
    provisioning_pool: ThreadPoolExecutor

    # Semaphore limiting the number of on-demand containers starting simultaneously.
    # Concurrent container starts are I/O-heavy (Docker API calls, copying runtime files)
    # and can exhaust OS file descriptor limits on machines with low ulimits.
    on_demand_start_semaphore: threading.Semaphore

    def __init__(self):
        self.environments = defaultdict(dict)
        self.provisioning_pool = ThreadPoolExecutor(thread_name_prefix="lambda-provisioning-pool")
        # TODO: make this value configurable; 16 is a conservative default
        self.on_demand_start_semaphore = threading.Semaphore(16)

    @contextlib.contextmanager
    def get_environment(
        self,
        version_manager_id: str,
        function_version: FunctionVersion,
        provisioning_type: InitializationType,
    ) -> Iterator[ExecutionEnvironment]:
        # Snapshot the values list before iterating to avoid skipped entries
        # that can be caused by concurrent invocations
        applicable_envs = [
            env
            for env in list(self.environments[version_manager_id].values())
            if env.initialization_type == provisioning_type
        ]
        execution_environment = None
        for environment in applicable_envs:
            try:
                environment.reserve()
                execution_environment = environment
                break
            except InvalidStatusException:
                pass

        if execution_environment is None:
            if provisioning_type == InitializationType.provisioned_concurrency:
                raise AssignmentException(
                    "No provisioned concurrency environment available despite lease."
                )
            elif provisioning_type == InitializationType.on_demand:
                with self.on_demand_start_semaphore:
                    execution_environment = self.start_environment(
                        version_manager_id, function_version
                    )
                self.environments[version_manager_id][execution_environment.id] = (
                    execution_environment
                )
                execution_environment.reserve()
            else:
                raise ValueError(f"Invalid provisioning type {provisioning_type}")

        try:
            yield execution_environment
            execution_environment.release()
        except InvalidStatusException as invalid_e:
            LOG.error("InvalidStatusException: %s", invalid_e)
        except Exception as e:
            LOG.error(
                "Failed invocation <%s>: %s", type(e), e, exc_info=LOG.isEnabledFor(logging.DEBUG)
            )
            if execution_environment.initialization_type == InitializationType.on_demand:
                self.stop_environment(execution_environment)
            else:
                # Try to restore to READY rather than stopping.
                # Transient errors (e.g., OS-level connection failures) should not
                # permanently remove healthy provisioned containers from the pool.
                try:
                    execution_environment.release()
                except InvalidStatusException:
                    self.stop_environment(execution_environment)
            raise e

    def start_environment(
        self, version_manager_id: str, function_version: FunctionVersion
    ) -> ExecutionEnvironment:
        LOG.debug("Starting new environment")
        initialization_type = InitializationType.on_demand
        if function_version.config.capacity_provider_config:
            initialization_type = InitializationType.lambda_managed_instances
        execution_environment = ExecutionEnvironment(
            function_version=function_version,
            initialization_type=initialization_type,
            on_timeout=self.on_timeout,
            version_manager_id=version_manager_id,
        )
        try:
            execution_environment.start()
        except StatusErrorException:
            raise
        except EnvironmentStartupTimeoutException:
            raise
        except Exception as e:
            message = f"Could not start new environment: {type(e).__name__}:{e}"
            raise AssignmentException(message) from e
        return execution_environment

    def on_timeout(self, version_manager_id: str, environment_id: str) -> None:
        """Callback for deleting environment after function times out"""
        del self.environments[version_manager_id][environment_id]

    def stop_environment(self, environment: ExecutionEnvironment) -> None:
        version_manager_id = environment.version_manager_id
        try:
            environment.stop()
            self.environments.get(version_manager_id).pop(environment.id)
        except Exception as e:
            LOG.debug(
                "Error while stopping environment for lambda %s, manager id %s, environment: %s, error: %s",
                environment.function_version.qualified_arn,
                version_manager_id,
                environment.id,
                e,
            )

    def stop_environments_for_version(self, version_manager_id: str):
        # We have to materialize the list before iterating due to concurrency
        environments_to_stop = list(self.environments.get(version_manager_id, {}).values())
        for env in environments_to_stop:
            self.stop_environment(env)

    # == Incremental provisioned concurrency primitives ==
    #
    # The provisioned concurrency coordinator reconciles the pool towards a
    # target using only deltas: already-serving environments are never
    # recreated. Environments are registered in ``self.environments`` in
    # creation order, which is the declared FIFO order used when reclaiming.

    def provisioned_environments(self, version_manager_id: str) -> list[ExecutionEnvironment]:
        """Return provisioned environments in declaration (creation) order.

        The per-manager mapping preserves insertion order, and provisioned
        environments are never reordered, so the filtered mapping order is the
        FIFO order used by scale-down.
        """
        # Materialize before filtering due to concurrent pool modifications.
        return [
            env
            for env in list(self.environments.get(version_manager_id, {}).values())
            if env.initialization_type == InitializationType.provisioned_concurrency
        ]

    def count_provisioned_environments(self, version_manager_id: str) -> tuple[int, int, int]:
        """Count provisioned environments from physical state.

        :return: ``(total, serviceable, in_flight)`` where serviceable
            environments are READY or INVOKING and in_flight environments are
            INVOKING. Used as the ground truth to reconcile the quota ledger.
        """
        total = 0
        serviceable = 0
        in_flight = 0
        for env in self.provisioned_environments(version_manager_id):
            total += 1
            if env.status in (RuntimeStatus.READY, RuntimeStatus.INVOKING):
                serviceable += 1
            if env.status == RuntimeStatus.INVOKING:
                in_flight += 1
        return total, serviceable, in_flight

    def create_provisioned_environments(
        self,
        version_manager_id: str,
        function_version: FunctionVersion,
        count: int,
    ) -> list[tuple[ExecutionEnvironment, Future[None]]]:
        """Register and start ``count`` new provisioned environments.

        Only the delta is created; existing environments are untouched. The
        environments are registered in the pool (i.e. allocation slots exist)
        before the start tasks are submitted, and are returned in FIFO order.
        """
        created: list[tuple[ExecutionEnvironment, Future[None]]] = []
        for _ in range(count):
            execution_environment = ExecutionEnvironment(
                function_version=function_version,
                initialization_type=InitializationType.provisioned_concurrency,
                on_timeout=self.on_timeout,
                version_manager_id=version_manager_id,
            )
            self.environments[version_manager_id][execution_environment.id] = execution_environment
            future = self.provisioning_pool.submit(execution_environment.start)
            created.append((execution_environment, future))
        return created

    def reclaim_provisioned_environment(self, environment: ExecutionEnvironment) -> str:
        """Reclaim exactly one provisioned environment, FIFO-safe.

        A READY environment is flipped to STOPPED while holding its status
        lock (so a concurrent ``reserve`` cannot pick it up) and its runtime is
        stopped afterwards. Environments that already died during startup are
        discarded from the pool without a stop call.

        :raises InvalidStatusException: if the environment is INVOKING or still
            STARTING; it must not be reclaimed and the caller waits/retries.
        :return: ``"stopped"`` when a serviceable environment was stopped, or
            ``"discarded"`` when a dead slot was removed.
        """
        with environment.status_lock:
            status = environment.status
            if status == RuntimeStatus.READY:
                # Flip atomically so concurrent reserve() calls fail to claim.
                environment.status = RuntimeStatus.STOPPED
                stop_runtime = True
            elif status in (
                RuntimeStatus.STARTUP_FAILED,
                RuntimeStatus.STARTUP_TIMED_OUT,
                RuntimeStatus.STOPPED,
            ):
                stop_runtime = False
            else:
                # INVOKING (in-flight call) or STARTING (concurrent scale-up):
                # never reclaim; the coordinator waits and retries in order.
                raise InvalidStatusException(
                    f"Provisioned environment {environment.id} cannot be reclaimed while"
                    f" {status}. Current status: {status}"
                )
        # Blocking I/O without holding the environment status lock.
        if stop_runtime:
            environment.runtime_executor.stop()
            if environment.keepalive_timer is not None:
                environment.keepalive_timer.cancel()
        self.environments.get(environment.version_manager_id, {}).pop(environment.id, None)
        return "stopped" if stop_runtime else "discarded"

    def discard_provisioned_environment(self, environment: ExecutionEnvironment) -> None:
        """Remove a provisioned environment from the pool without stopping it.

        Used during scale-up rollback for environments whose startup failed.
        """
        self.environments.get(environment.version_manager_id, {}).pop(environment.id, None)

    def stop(self):
        self.provisioning_pool.shutdown(cancel_futures=True)
