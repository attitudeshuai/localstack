import logging
import threading
import time
from concurrent.futures import Future
from concurrent.futures._base import CancelledError

from localstack import config
from localstack.aws.api.lambda_ import (
    ServiceException,
    State,
    StateReasonCode,
)
from localstack.services.lambda_ import hooks as lambda_hooks
from localstack.services.lambda_.invocation.assignment import AssignmentService
from localstack.services.lambda_.invocation.counting_service import CountingService
from localstack.services.lambda_.invocation.execution_environment import ExecutionEnvironment
from localstack.services.lambda_.invocation.executor_endpoint import StatusErrorException
from localstack.services.lambda_.invocation.lambda_models import (
    Function,
    FunctionVersion,
    Invocation,
    InvocationResult,
    VersionState,
)
from localstack.services.lambda_.invocation.logs import LogHandler, LogItem
from localstack.services.lambda_.invocation.metrics import (
    record_cw_metric_error,
    record_cw_metric_invocation,
)
from localstack.services.lambda_.invocation.provisioned_concurrency import (
    ProvisionedConcurrencyCoordinator,
    ProvisionedConcurrencyLedger,
    ProvisionedConcurrencySnapshot,
    ProvisionedConcurrencyUpdatePolicy,
)
from localstack.services.lambda_.invocation.runtime_executor import get_runtime_executor
from localstack.services.lambda_.ldm import LDMProvisioner
from localstack.utils.strings import long_uid, to_bytes, truncate
from localstack.utils.threads import start_thread

LOG = logging.getLogger(__name__)


class LambdaVersionManager:
    # arn this Lambda Version manager manages
    function_arn: str
    function_version: FunctionVersion
    function: Function

    # Additional guard to prevent scheduling invocation on version during shutdown
    shutdown_event: threading.Event

    state: VersionState | None
    # Incremental provisioned concurrency scaling: coordination, progress and
    # unified quota accounting (allocated/ready/in-flight) live in the coordinator
    provisioned_coordinator: ProvisionedConcurrencyCoordinator
    log_handler: LogHandler
    counting_service: CountingService
    assignment_service: AssignmentService

    ldm_provisioner: LDMProvisioner | None

    def __init__(
        self,
        function_arn: str,
        function_version: FunctionVersion,
        # HACK allowing None for Lambda@Edge; only used in invoke for get_invocation_lease
        function: Function | None,
        counting_service: CountingService,
        assignment_service: AssignmentService,
        provisioned_ledger: ProvisionedConcurrencyLedger,
    ):
        self.id = long_uid()
        self.function_arn = function_arn
        self.function_version = function_version
        self.function = function
        self.counting_service = counting_service
        self.assignment_service = assignment_service
        self.log_handler = LogHandler(function_version.config.role, function_version.id.region)

        # async
        self.shutdown_event = threading.Event()

        self.provisioned_coordinator = ProvisionedConcurrencyCoordinator(
            qualified_arn=function_arn,
            version_manager_id=self.id,
            function_version=function_version,
            assignment_service=assignment_service,
            ledger=provisioned_ledger,
        )
        # https://aws.amazon.com/blogs/compute/coming-soon-expansion-of-aws-lambda-states-to-all-functions/
        self.state: VersionState = VersionState(state=State.Pending)

        self.ldm_provisioner = None
        lambda_hooks.inject_ldm_provisioner.run(self)

    def start(self) -> VersionState:
        try:
            self.log_handler.start_subscriber()
            time_before = time.perf_counter()
            get_runtime_executor().prepare_version(self.function_version)  # TODO: make pluggable?
            LOG.debug(
                "Version preparation of function %s took %0.2fms",
                self.function_version.qualified_arn,
                (time.perf_counter() - time_before) * 1000,
            )

            # code and reason not set for success scenario because only failed states provide this field:
            # https://docs.aws.amazon.com/lambda/latest/dg/API_GetFunctionConfiguration.html#SSS-GetFunctionConfiguration-response-LastUpdateStatusReasonCode
            new_state = State.Active
            if (
                self.function_version.config.capacity_provider_config
                and self.function_version.id.qualifier == "$LATEST"
            ):
                new_state = State.ActiveNonInvocable
                # HACK: trying to match the AWS timing behavior of Lambda Managed Instances for the operation
                # update_function_configuration followed by get_function because transitioning LastUpdateStatus from
                # InProgress to Successful happens too fast on LocalStack (thanks to caching in prepare_version).
                # Without this hack, test_latest_published_update_config fails at get_function_response_postupdate_latest
                # TODO: this sleep has side-effects and we should be looking into alternatives
                # Increasing this sleep too much (e.g., 3s) could cause the side effect that a created function is not
                # ready for updates (i.e., rejected with a ResourceConflictException) and failing other tests
                # time.sleep(0.1)
            self.state = VersionState(state=new_state)
            LOG.debug(
                "Changing Lambda %s (id %s) to %s",
                self.function_arn,
                self.function_version.config.internal_revision,
                new_state,
            )
        except Exception as e:
            self.state = VersionState(
                state=State.Failed,
                code=StateReasonCode.InternalError,
                reason=f"Error while creating lambda: {e}",
            )
            LOG.debug(
                "Changing Lambda %s (id %s) to Failed. Reason: %s",
                self.function_arn,
                self.function_version.config.internal_revision,
                e,
                exc_info=LOG.isEnabledFor(logging.DEBUG),
            )
        return self.state

    def stop(self) -> None:
        LOG.debug("Stopping lambda version '%s'", self.function_arn)
        self.state = VersionState(
            state=State.Inactive, code=StateReasonCode.Idle, reason="Shutting down"
        )
        self.shutdown_event.set()
        # Stop incremental provisioned concurrency scaling before tearing the
        # environment pool down, so the scaling worker does not race the
        # wholesale shutdown (in-flight calls on provisioned environments are
        # interrupted by stop_environments_for_version as before).
        self.provisioned_coordinator.shutdown()
        self.provisioned_coordinator.join(timeout=2)
        self.log_handler.stop()
        self.assignment_service.stop_environments_for_version(self.id)
        get_runtime_executor().cleanup_version(self.function_version)  # TODO: make pluggable?

    def update_provisioned_concurrency_config(
        self,
        provisioned_concurrent_executions: int,
        *,
        qualifier: str | None = None,
        policy: ProvisionedConcurrencyUpdatePolicy = ProvisionedConcurrencyUpdatePolicy.merge,
    ) -> Future[None]:
        """Declare a new provisioned concurrency target.

        Only the delta is created or reclaimed; environments already in service
        are not rebuilt. Declarations arriving while an adjustment is running are
        merged (default, final target = last declaration) or queued FIFO per
        ``policy``. The returned Future completes when this declaration settles.

        :param provisioned_concurrent_executions: target count; 0 deprovisions
        :param qualifier: declared qualifier (version number or alias name)
        :param policy: merge or queue declarations arriving mid-adjustment
        """
        return self.provisioned_coordinator.declare(
            provisioned_concurrent_executions,
            qualifier=qualifier or self.function_version.id.qualifier,
            policy=policy,
        )

    def provisioned_snapshot(self) -> ProvisionedConcurrencySnapshot | None:
        """Queryable progress/result of the current provisioned concurrency
        adjustment, or None if no target has ever been declared."""
        return self.provisioned_coordinator.snapshot()

    # Extract environment handling

    def invoke(self, *, invocation: Invocation) -> InvocationResult:
        """
        synchronous invoke entrypoint

        0. check counter, get lease
        1. try to get an inactive (no active invoke) environment
        2.(allgood) send invoke to environment
        3. wait for invocation result
        4. return invocation result & release lease

        2.(nogood) fail fast fail hard

        """
        LOG.debug(
            "Got an invocation for function %s with request_id %s",
            self.function_arn,
            invocation.request_id,
        )
        if self.shutdown_event.is_set():
            message = f"Got an invocation with request_id {invocation.request_id} for a version shutting down"
            LOG.warning(message)
            raise ServiceException(message)

        # If the environment has debugging enabled, route the invocation there;
        # debug environments bypass Lambda service quotas.
        if self.ldm_provisioner and (
            ldm_execution_environment := self.ldm_provisioner.get_execution_environment(
                qualified_lambda_arn=self.function_version.qualified_arn,
                user_agent=invocation.user_agent,
            )
        ):
            try:
                invocation_result = ldm_execution_environment.invoke(invocation)
                invocation_result.executed_version = self.function_version.id.qualifier
                self.store_logs(
                    invocation_result=invocation_result, execution_env=ldm_execution_environment
                )
            except CancelledError as e:
                # Timeouts for invocation futures are managed by LDM, a cancelled error here is
                # aligned with the debug container terminating whilst debugging/invocation.
                LOG.debug("LDM invocation future encountered a cancelled error: '%s'", e)
                invocation_result = InvocationResult(
                    request_id="",
                    payload=to_bytes(
                        "The invocation was canceled because the debug configuration "
                        "was removed or the operation timed out"
                    ),
                    is_error=True,
                    logs="",
                    executed_version=self.function_version.id.qualifier,
                )
            except StatusErrorException as e:
                invocation_result = InvocationResult(
                    request_id="",
                    payload=e.payload,
                    is_error=True,
                    logs="",
                    executed_version=self.function_version.id.qualifier,
                )
            finally:
                ldm_execution_environment.release()
            return invocation_result

        with self.counting_service.get_invocation_lease(
            self.function, self.function_version, self.function_arn
        ) as provisioning_type:
            # TODO: potential race condition when changing provisioned concurrency after getting the lease but before
            #   getting an environment
            try:
                # Blocks and potentially creates a new execution environment for this invocation
                with self.assignment_service.get_environment(
                    self.id, self.function_version, provisioning_type
                ) as execution_env:
                    invocation_result = execution_env.invoke(invocation)
                    invocation_result.executed_version = self.function_version.id.qualifier
                    self.store_logs(
                        invocation_result=invocation_result, execution_env=execution_env
                    )
            except StatusErrorException as e:
                invocation_result = InvocationResult(
                    request_id="",
                    payload=e.payload,
                    is_error=True,
                    logs="",
                    executed_version=self.function_version.id.qualifier,
                )

        function_id = self.function_version.id
        # Record CloudWatch metrics in separate threads
        # MAYBE reuse threads rather than starting new threads upon every invocation
        if invocation_result.is_error:
            start_thread(
                lambda *args, **kwargs: record_cw_metric_error(
                    function_name=function_id.function_name,
                    account_id=function_id.account,
                    region_name=function_id.region,
                ),
                name=f"record-cloudwatch-metric-error-{function_id.function_name}:{function_id.qualifier}",
            )
        else:
            start_thread(
                lambda *args, **kwargs: record_cw_metric_invocation(
                    function_name=function_id.function_name,
                    account_id=function_id.account,
                    region_name=function_id.region,
                ),
                name=f"record-cloudwatch-metric-{function_id.function_name}:{function_id.qualifier}",
            )
        # TODO: consider using the same prefix logging as in error case for execution environment.
        #   possibly as separate named logger.
        if invocation_result.logs is not None:
            LOG.debug("Got logs for invocation '%s'", invocation.request_id)
            for log_line in invocation_result.logs.splitlines():
                LOG.debug(
                    "[%s-%s] %s",
                    function_id.function_name,
                    invocation.request_id,
                    truncate(log_line, config.LAMBDA_TRUNCATE_STDOUT),
                )
        else:
            LOG.warning(
                "[%s] Error while printing logs for function '%s': Received no logs from environment.",
                invocation.request_id,
                function_id.function_name,
            )
        return invocation_result

    def store_logs(
        self, invocation_result: InvocationResult, execution_env: ExecutionEnvironment
    ) -> None:
        if invocation_result.logs:
            log_item = LogItem(
                execution_env.get_log_group_name(),
                execution_env.get_log_stream_name(),
                invocation_result.logs,
            )
            self.log_handler.add_logs(log_item)
        else:
            LOG.warning(
                "Received no logs from invocation with id %s for lambda %s. Execution environment logs: \n%s",
                invocation_result.request_id,
                self.function_arn,
                execution_env.get_prefixed_logs(),
            )
