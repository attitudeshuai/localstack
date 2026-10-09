"""Unit tests for the unified provisioned concurrency ledger (allocated / ready /
in-flight accounts per function, alias and version) and for the counting
service integration, including byte-for-byte on-demand behavior when no
provisioned concurrency is configured."""

import threading

import pytest

from localstack import config
from localstack.aws.api.lambda_ import (
    ProvisionedConcurrencyStatusEnum,
    TooManyRequestsException,
)
from localstack.services.lambda_.invocation.counting_service import CountingService
from localstack.services.lambda_.invocation.lambda_models import (
    Function,
    InitializationType,
    ProvisionedConcurrencyConfiguration,
    VersionIdentifier,
)
from localstack.services.lambda_.invocation.provisioned_concurrency import (
    ProvisionedConcurrencyLedger,
)

ACCOUNT = "123456789012"
REGION = "us-east-1"
FUNCTION = "fn-ledger"


def _version_id(qualifier: str = "1") -> VersionIdentifier:
    return VersionIdentifier(
        function_name=FUNCTION, qualifier=qualifier, region=REGION, account=ACCOUNT
    )


class TestProvisionedConcurrencyLedger:
    def test_accounts_per_alias_and_version_and_function_totals(self):
        ledger = ProvisionedConcurrencyLedger()
        v1 = _version_id("1")
        v2 = _version_id("2")

        # alias "live" points to version 1 and provisions 2; version 2 provisions 1
        live = ledger.open(
            account_id=ACCOUNT,
            region=REGION,
            function_name=FUNCTION,
            qualifier="live",
            qualified_arn=v1.qualified_arn(),
        )
        version2 = ledger.open(
            account_id=ACCOUNT,
            region=REGION,
            function_name=FUNCTION,
            qualifier="2",
            qualified_arn=v2.qualified_arn(),
        )

        # version 1: both slots ready, one invocation in flight
        live.adjust_allocated(2)
        live.adjust_ready(2)
        live.increment_in_flight()
        # version 2: one slot allocated but still starting
        version2.adjust_allocated(1)

        # read by physical arn, by declared qualifier and via function listing
        assert ledger.get(v1.qualified_arn()) is live
        assert (
            ledger.get_by_qualifier(
                account_id=ACCOUNT, region=REGION, function_name=FUNCTION, qualifier="live"
            )
            is live
        )
        assert {
            a.qualifier
            for a in ledger.list_for_function(
                account_id=ACCOUNT, region=REGION, function_name=FUNCTION
            )
        } == {"live", "2"}

        live_snapshot = live.snapshot()
        assert (live_snapshot.allocated, live_snapshot.ready, live_snapshot.in_flight) == (
            2,
            2,
            1,
        )
        assert live_snapshot.provisioning == 0
        assert live_snapshot.available == 1
        version2_snapshot = version2.snapshot()
        assert (
            version2_snapshot.allocated,
            version2_snapshot.ready,
            version2_snapshot.in_flight,
        ) == (1, 0, 0)
        assert version2_snapshot.provisioning == 1
        assert version2_snapshot.available == 0

        totals = ledger.function_totals(account_id=ACCOUNT, region=REGION, function_name=FUNCTION)
        assert (totals.allocated, totals.ready, totals.in_flight) == (3, 2, 1)
        assert totals.provisioning == 1
        assert totals.available == 1

    def test_close_removes_account_and_indexes(self):
        ledger = ProvisionedConcurrencyLedger()
        v1 = _version_id("1")
        account = ledger.open(
            account_id=ACCOUNT,
            region=REGION,
            function_name=FUNCTION,
            qualifier="1",
            qualified_arn=v1.qualified_arn(),
        )
        account.adjust_allocated(1)
        ledger.close(
            account_id=ACCOUNT,
            region=REGION,
            function_name=FUNCTION,
            qualifier="1",
        )
        assert ledger.get(v1.qualified_arn()) is None
        assert (
            ledger.list_for_function(account_id=ACCOUNT, region=REGION, function_name=FUNCTION)
            == []
        )
        totals = ledger.function_totals(account_id=ACCOUNT, region=REGION, function_name=FUNCTION)
        assert (totals.allocated, totals.ready, totals.in_flight) == (0, 0, 0)

    def test_invariants_hold_under_concurrent_paired_mutations(self):
        ledger = ProvisionedConcurrencyLedger()
        v1 = _version_id("1")
        account = ledger.open(
            account_id=ACCOUNT,
            region=REGION,
            function_name=FUNCTION,
            qualifier="1",
            qualified_arn=v1.qualified_arn(),
        )
        account.adjust_allocated(10)
        account.adjust_ready(10)
        account.set_status(ProvisionedConcurrencyStatusEnum.READY)

        stop = threading.Event()

        def worker():
            for _ in range(2000):
                if account.try_acquire_in_flight():
                    account.decrement_in_flight()
                account.adjust_allocated(0)  # paired no-op, exercises validation

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        stop.set()

        snapshot = account.snapshot()
        assert snapshot.in_flight == 0
        assert 0 <= snapshot.ready <= snapshot.allocated
        assert 0 <= snapshot.in_flight <= snapshot.ready


class TestCountingServiceLeaseCompatibility:
    @pytest.fixture
    def function_and_version(self):
        from types import SimpleNamespace

        version_id = _version_id("1")
        version = SimpleNamespace(id=version_id)
        function = Function(function_name=FUNCTION)
        return function, version

    def test_no_provisioned_config_serves_on_demand_without_ledger_writes(
        self, function_and_version
    ):
        function, version = function_and_version
        ledger = ProvisionedConcurrencyLedger()
        counting = CountingService(provisioned_ledger=ledger)

        with counting.get_invocation_lease(
            function, version, version.id.qualified_arn()
        ) as lease_type:
            assert lease_type == InitializationType.on_demand

        assert ledger.get(version.id.qualified_arn()) is None

    def test_works_without_ledger_for_standalone_instantiation(self, function_and_version):
        function, version = function_and_version
        counting = CountingService()
        with counting.get_invocation_lease(function, version) as lease_type:
            assert lease_type == InitializationType.on_demand

    def test_ready_account_grants_provisioned_lease_and_counts_in_flight(
        self, function_and_version
    ):
        function, version = function_and_version
        function.provisioned_concurrency_configs["1"] = ProvisionedConcurrencyConfiguration(
            2, "now"
        )
        ledger = ProvisionedConcurrencyLedger()
        account = ledger.open(
            account_id=ACCOUNT,
            region=REGION,
            function_name=FUNCTION,
            qualifier="1",
            qualified_arn=version.id.qualified_arn(),
        )
        account.reset(allocated=2, ready=2)
        account.set_status(ProvisionedConcurrencyStatusEnum.READY)
        counting = CountingService(provisioned_ledger=ledger)

        with counting.get_invocation_lease(
            function, version, version.id.qualified_arn()
        ) as lease_type:
            assert lease_type == InitializationType.provisioned_concurrency
            assert account.snapshot().in_flight == 1
        assert account.snapshot().in_flight == 0

    def test_in_progress_and_full_capacity_fall_back_to_on_demand(self, function_and_version):
        function, version = function_and_version
        function.provisioned_concurrency_configs["1"] = ProvisionedConcurrencyConfiguration(
            1, "now"
        )
        ledger = ProvisionedConcurrencyLedger()
        account = ledger.open(
            account_id=ACCOUNT,
            region=REGION,
            function_name=FUNCTION,
            qualifier="1",
            qualified_arn=version.id.qualified_arn(),
        )
        account.reset(allocated=1, ready=1, in_flight=1)
        account.set_status(ProvisionedConcurrencyStatusEnum.READY)
        counting = CountingService(provisioned_ledger=ledger)

        # capacity exhausted (ready == in_flight) => on-demand
        with counting.get_invocation_lease(
            function, version, version.id.qualified_arn()
        ) as lease_type:
            assert lease_type == InitializationType.on_demand

        # not READY (IN_PROGRESS) => on-demand
        account.reset(allocated=1, ready=1, in_flight=0)
        account.set_status(ProvisionedConcurrencyStatusEnum.IN_PROGRESS)
        with counting.get_invocation_lease(
            function, version, version.id.qualified_arn()
        ) as lease_type:
            assert lease_type == InitializationType.on_demand

    def test_alias_config_grants_provisioned_lease_for_version(self, function_and_version):
        function, version = function_and_version
        function.aliases["live"] = type(
            "Alias",
            (),
            {"function_version": "1", "name": "live", "routing_configuration": None},
        )()
        function.provisioned_concurrency_configs["live"] = ProvisionedConcurrencyConfiguration(
            1, "now"
        )
        ledger = ProvisionedConcurrencyLedger()
        account = ledger.open(
            account_id=ACCOUNT,
            region=REGION,
            function_name=FUNCTION,
            qualifier="live",
            qualified_arn=version.id.qualified_arn(),
        )
        account.reset(allocated=1, ready=1)
        account.set_status(ProvisionedConcurrencyStatusEnum.READY)
        counting = CountingService(provisioned_ledger=ledger)

        with counting.get_invocation_lease(
            function, version, version.id.qualified_arn()
        ) as lease_type:
            assert lease_type == InitializationType.provisioned_concurrency

    def test_reserved_concurrency_zero_rejects_as_before(self, function_and_version):
        function, version = function_and_version
        function.reserved_concurrent_executions = 0
        counting = CountingService(provisioned_ledger=ProvisionedConcurrencyLedger())
        with pytest.raises(TooManyRequestsException):
            with counting.get_invocation_lease(function, version, version.id.qualified_arn()):
                pass

    def test_unreserved_concurrency_zero_rejects_as_before(self, function_and_version, monkeypatch):
        function, version = function_and_version
        monkeypatch.setattr(config, "LAMBDA_LIMITS_CONCURRENT_EXECUTIONS", 0)
        counting = CountingService(provisioned_ledger=ProvisionedConcurrencyLedger())
        with pytest.raises(TooManyRequestsException):
            with counting.get_invocation_lease(function, version, version.id.qualified_arn()):
                pass
