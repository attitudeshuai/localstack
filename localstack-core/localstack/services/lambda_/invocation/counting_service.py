import contextlib
import logging
from collections import defaultdict
from collections.abc import Iterator
from threading import RLock
from typing import TYPE_CHECKING

from localstack import config
from localstack.aws.api.lambda_ import TooManyRequestsException
from localstack.services.lambda_.invocation.lambda_models import (
    Function,
    FunctionVersion,
    InitializationType,
)
from localstack.services.lambda_.invocation.models import lambda_stores

if TYPE_CHECKING:
    from localstack.services.lambda_.invocation.provisioned_concurrency import (
        ProvisionedConcurrencyLedger,
    )

LOG = logging.getLogger(__name__)


class ConcurrencyTracker:
    """Keeps track of the number of concurrent executions per lock scope (e.g., per function or function version).
    The lock scope depends on the provisioning type (i.e., on-demand or provisioned):
    * on-demand concurrency per function: unqualified arn ending with my-function
    * provisioned concurrency per function version: qualified arn ending with my-function:1
    """

    # Lock scope => concurrent executions counter
    concurrent_executions: dict[str, int]
    # Lock for safely updating the concurrent executions counter
    lock: RLock

    def __init__(self):
        self.concurrent_executions = defaultdict(int)
        self.lock = RLock()

    def increment(self, scope: str) -> None:
        self.concurrent_executions[scope] += 1

    def atomic_decrement(self, scope: str):
        with self.lock:
            self.decrement(scope)

    def decrement(self, scope: str) -> None:
        self.concurrent_executions[scope] -= 1


def calculate_provisioned_concurrency_sum(function: Function) -> int:
    """Returns the total provisioned concurrency for a given function, including all versions."""
    provisioned_concurrency_sum_for_fn = sum(
        [
            provisioned_configs.provisioned_concurrent_executions
            for provisioned_configs in function.provisioned_concurrency_configs.values()
        ]
    )
    return provisioned_concurrency_sum_for_fn


class CountingService:
    """
    The CountingService enforces quota limits per region and account in get_invocation_lease()
    for every Lambda invocation. It uses separate ConcurrencyTrackers for on-demand and provisioned concurrency
    to keep track of the number of concurrent invocations.

    Concurrency limits are per region and account:
    https://repost.aws/knowledge-center/lambda-concurrency-limit-increase
    https://docs.aws.amazon.com/lambda/latest/dg/lambda-concurrency.htm
    https://docs.aws.amazon.com/lambda/latest/dg/monitoring-concurrency.html
    """

    # (account, region) => ConcurrencyTracker (unqualified arn) => concurrent executions
    on_demand_concurrency_trackers: dict[tuple[str, str], ConcurrencyTracker]
    # Lock for safely initializing new on-demand concurrency trackers
    on_demand_init_lock: RLock

    # Unified ledger holding allocated/ready/in-flight provisioned concurrency
    # accounts per function, alias and physical version.
    provisioned_ledger: "ProvisionedConcurrencyLedger | None"

    def __init__(
        self,
        provisioned_ledger: "ProvisionedConcurrencyLedger | None" = None,
    ):
        self.on_demand_concurrency_trackers = {}
        self.on_demand_init_lock = RLock()
        self.provisioned_ledger = provisioned_ledger

    @contextlib.contextmanager
    def get_invocation_lease(
        self,
        function: Function | None,
        function_version: FunctionVersion,
        provisioned_qualified_arn: str | None = None,
    ) -> Iterator[InitializationType]:
        """An invocation lease reserves the right to schedule an invocation.
        The returned lease type can either be on-demand or provisioned.
        Scheduling preference:
        1) Check for free provisioned concurrency => provisioned
        2) Check for reserved concurrency => on-demand
        3) Check for unreserved concurrency => on-demand

        HACK: We allow the function to be None for Lambda@Edge to skip provisioned and reserved concurrency.
        """
        account = function_version.id.account
        region = function_version.id.region
        scope_tuple = (account, region)
        on_demand_tracker = self.on_demand_concurrency_trackers.get(scope_tuple)
        # Double-checked locking pattern to initialize an on-demand concurrency tracker if it does not exist
        if not on_demand_tracker:
            with self.on_demand_init_lock:
                on_demand_tracker = self.on_demand_concurrency_trackers.get(scope_tuple)
                if not on_demand_tracker:
                    on_demand_tracker = self.on_demand_concurrency_trackers[scope_tuple] = (
                        ConcurrencyTracker()
                    )

        unqualified_function_arn = function_version.id.unqualified_arn()
        qualified_arn = provisioned_qualified_arn or function_version.id.qualified_arn()

        lease_type = None
        # Keep the account object returned by the lease grant: the finally block
        # must decrement on THIS object even if scale-to-zero detached it from
        # the ledger while the invocation was still running.
        provisioned_account = None
        # HACK: skip reserved and provisioned concurrency if function not available (e.g., in Lambda@Edge)
        if function is not None and self.provisioned_ledger is not None:
            # 1) Check for free provisioned concurrency
            provisioned_concurrency_config = function.provisioned_concurrency_configs.get(
                function_version.id.qualifier
            )
            if not provisioned_concurrency_config:
                # check if any aliases point to the current version, and check the provisioned concurrency config
                # for them. There can be only one config for a version, not matter if defined on the alias or version itself.
                for alias in function.aliases.values():
                    if alias.function_version == function_version.id.qualifier:
                        provisioned_concurrency_config = (
                            function.provisioned_concurrency_configs.get(alias.name)
                        )
                        break
            # Favor provisioned concurrency if configured and ready. The unified
            # ledger is the single source of truth for the allocated/ready/
            # in-flight accounts; the atomic lease grant together with the
            # scale-down reclaim gate guarantees that scale-down never takes
            # away the environment of an outstanding (even not-yet-reserved)
            # provisioned lease.
            # TODO: test updating provisioned concurrency? Does AWS serve on-demand during updates?
            if provisioned_concurrency_config:
                candidate_account = self.provisioned_ledger.get(qualified_arn)
                if candidate_account is not None and candidate_account.try_acquire_in_flight():
                    provisioned_account = candidate_account
                    lease_type = InitializationType.provisioned_concurrency

        if not lease_type:
            with on_demand_tracker.lock:
                # 2) If reserved concurrency is set AND no provisioned concurrency available:
                # => Check if enough reserved concurrency is available for the specific function.
                # HACK: skip reserved if function not available (e.g., in Lambda@Edge)
                if function and function.reserved_concurrent_executions is not None:
                    on_demand_running_invocation_count = on_demand_tracker.concurrent_executions[
                        unqualified_function_arn
                    ]
                    available_reserved_concurrency = (
                        function.reserved_concurrent_executions
                        - calculate_provisioned_concurrency_sum(function)
                        - on_demand_running_invocation_count
                    )
                    if available_reserved_concurrency > 0:
                        on_demand_tracker.increment(unqualified_function_arn)
                        lease_type = InitializationType.on_demand
                    else:
                        extras = {
                            "available_reserved_concurrency": available_reserved_concurrency,
                            "reserved_concurrent_executions": function.reserved_concurrent_executions,
                            "provisioned_concurrency_sum": calculate_provisioned_concurrency_sum(
                                function
                            ),
                            "on_demand_running_invocation_count": on_demand_running_invocation_count,
                        }
                        LOG.debug("Insufficient reserved concurrency available: %s", extras)
                        raise TooManyRequestsException(
                            "Rate Exceeded.",
                            Reason="ReservedFunctionConcurrentInvocationLimitExceeded",
                            Type="User",
                        )
                # 3) If no reserved concurrency is set AND no provisioned concurrency available.
                # => Check the entire state within the scope of account and region.
                else:
                    # TODO: Consider a dedicated counter for unavailable concurrency with locks for updates on
                    #  reserved and provisioned concurrency if this is too slow
                    # The total concurrency allocated or used (i.e., unavailable concurrency) per account and region
                    total_used_concurrency = 0
                    store = lambda_stores[account][region]
                    for fn in store.functions.values():
                        if fn.reserved_concurrent_executions is not None:
                            total_used_concurrency += fn.reserved_concurrent_executions
                        else:
                            fn_provisioned_concurrency = calculate_provisioned_concurrency_sum(fn)
                            total_used_concurrency += fn_provisioned_concurrency
                            fn_on_demand_concurrent_executions = (
                                on_demand_tracker.concurrent_executions[
                                    fn.latest().id.unqualified_arn()
                                ]
                            )
                            total_used_concurrency += fn_on_demand_concurrent_executions

                    available_unreserved_concurrency = (
                        config.LAMBDA_LIMITS_CONCURRENT_EXECUTIONS - total_used_concurrency
                    )
                    if available_unreserved_concurrency > 0:
                        on_demand_tracker.increment(unqualified_function_arn)
                        lease_type = InitializationType.on_demand
                    else:
                        if available_unreserved_concurrency < 0:
                            LOG.error(
                                "Invalid function concurrency state detected for function: %s | available unreserved concurrency: %d",
                                unqualified_function_arn,
                                available_unreserved_concurrency,
                            )
                        extras = {
                            "available_unreserved_concurrency": available_unreserved_concurrency,
                            "lambda_limits_concurrent_executions": config.LAMBDA_LIMITS_CONCURRENT_EXECUTIONS,
                            "total_used_concurrency": total_used_concurrency,
                        }
                        LOG.debug("Insufficient unreserved concurrency available: %s", extras)
                        raise TooManyRequestsException(
                            "Rate Exceeded.",
                            Reason="ReservedFunctionConcurrentInvocationLimitExceeded",
                            Type="User",
                        )
        try:
            yield lease_type
        finally:
            if lease_type == InitializationType.provisioned_concurrency:
                # Decrement on the granted account object itself: it may have
                # been detached from the ledger by a concurrent shutdown, but
                # the outstanding lease still owns one of its in-flight slots.
                assert provisioned_account is not None
                provisioned_account.decrement_in_flight()
            elif lease_type == InitializationType.on_demand:
                on_demand_tracker.atomic_decrement(unqualified_function_arn)
            else:
                LOG.error(
                    "Invalid lease type detected for function: %s: %s",
                    unqualified_function_arn,
                    lease_type,
                )
