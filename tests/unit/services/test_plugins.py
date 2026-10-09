import threading
from queue import Queue
from types import SimpleNamespace

import pytest

from localstack.aws.api import RequestContext
from localstack.aws.handlers.service_plugin import (
    SERVICE_REQUEST_FINALIZERS,
    ServiceLoader,
    ServiceRequestFinalizer,
)
from localstack.http import Request, Response
from localstack.services import plugins
from localstack.services.plugins import (
    IllegalServiceStateTransition,
    Service,
    ServiceDisabled,
    ServiceManager,
    ServicePluginManager,
    ServiceState,
)
from localstack.services.sqs.provider import SqsProvider


class TestServicePluginManager:
    def test_get_service_calls_init_hook_once(self, monkeypatch):
        manager = ServicePluginManager()

        calls_to_on_after_init = []

        def _on_after_init(_self):
            calls_to_on_after_init.append(_self)

        monkeypatch.setattr(SqsProvider, "on_after_init", _on_after_init)

        s1 = manager.get_service("sqs")
        s2 = manager.get_service("sqs")

        assert s1 is s2, "instantiated two different services"
        assert len(calls_to_on_after_init) == 1, "on_after_init should be called once"

    def test_concurrent_get_service_calls_init_hook_once(self, monkeypatch):
        manager = ServicePluginManager()

        calls_to_get_service = Queue()
        calls_to_on_after_init = []

        def _call_get_service():
            service = manager.get_service("sqs")
            calls_to_get_service.put(service)

        def _on_after_init(_self):
            calls_to_on_after_init.append(_self)

        monkeypatch.setattr(SqsProvider, "on_after_init", _on_after_init)

        threading.Thread(target=_call_get_service).start()
        threading.Thread(target=_call_get_service).start()

        s1 = calls_to_get_service.get()
        s2 = calls_to_get_service.get()

        assert s1 is s2, "instantiated two different services"
        assert len(calls_to_on_after_init) == 1, "on_after_init should be called once"

    def test_nested_concurrent_get_service_calls_init_hook_once(self, monkeypatch):
        manager = ServicePluginManager()

        calls_to_get_service = Queue()
        calls_to_on_after_init = []

        def _call_get_service():
            service = manager.get_service("sqs")
            calls_to_get_service.put(service)

        def _on_after_init(_self):
            calls_to_on_after_init.append(_self)
            threading.Thread(target=_call_get_service).start()

        monkeypatch.setattr(SqsProvider, "on_after_init", _on_after_init)

        threading.Thread(target=_call_get_service).start()

        s1 = calls_to_get_service.get()
        s2 = calls_to_get_service.get()

        assert s1 is s2, "instantiated two different services"
        assert len(calls_to_on_after_init) == 1, "on_after_init should be called once"


class TestServiceLifecycleStateMachine:
    @pytest.fixture(autouse=True)
    def _enable_dummy_apis(self, monkeypatch):
        # dummy service names used below are all considered enabled
        monkeypatch.setattr(plugins, "is_api_enabled", lambda api: True)

    @staticmethod
    def _manager_with_service(name, start=None, check=None, stop=None) -> ServiceManager:
        service = Service(name=name, start=start, check=check, stop=stop)
        manager = ServiceManager()
        manager.add_service(service)
        return manager

    def test_assemble_runs_phases_and_runs(self):
        calls = []
        manager = self._manager_with_service(
            "svc",
            start=lambda asynchronous: calls.append("start"),
            check=lambda **kwargs: calls.append("check"),
            stop=lambda: calls.append("stop"),
        )

        assert manager.get_state("svc") == ServiceState.AVAILABLE
        assert manager.require("svc").name() == "svc"
        assert manager.get_state("svc") == ServiceState.RUNNING
        assert calls == ["start", "check"]

        result = manager.stop_service("svc")
        assert result.stopped
        assert result.interrupted_requests == 0
        assert result.drained
        assert calls == ["start", "check", "stop"]

    def test_failed_check_rolls_back_start_phase(self):
        checks = {"fail": True}
        stop_calls = []

        def _check(**kwargs):
            if checks["fail"]:
                raise RuntimeError("check boom")

        manager = self._manager_with_service(
            "svc",
            start=lambda asynchronous: None,
            check=_check,
            stop=lambda: stop_calls.append(1),
        )

        with pytest.raises(RuntimeError, match="check boom"):
            manager.require("svc")

        assert manager.get_state("svc") == ServiceState.ERROR
        failure = manager.get_failure("svc")
        assert failure.phase == "check"
        assert failure.rolled_back_phases == ["start"]
        assert "check" in str(failure)
        assert stop_calls == [1]  # the completed start phase was reclaimed exactly once

        # subsequent requires replay the same captured error, there is no implicit retry
        with pytest.raises(RuntimeError, match="check boom"):
            manager.require("svc")
        assert stop_calls == [1]

        # explicit retry brings the service back to RUNNING
        checks["fail"] = False
        assert manager.retry("svc").name() == "svc"
        assert manager.get_state("svc") == ServiceState.RUNNING
        assert manager.get_failure("svc") is None

    def test_failed_start_phase_records_explainable_failure(self):
        attempts = {"n": 0}

        def _start(asynchronous):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("start boom")

        manager = self._manager_with_service("svc", start=_start, stop=lambda: None)

        with pytest.raises(RuntimeError, match="start boom"):
            manager.require("svc")

        failure = manager.get_failure("svc")
        assert failure is not None
        assert failure.phase == "start"
        assert failure.rolled_back_phases == []
        assert manager.get_state("svc") == ServiceState.ERROR

        # retrying after the start was fixed works
        assert manager.retry("svc") is manager.get_service("svc")
        assert attempts["n"] == 2
        assert manager.get_state("svc") == ServiceState.RUNNING

    def test_concurrent_require_triggers_single_assembly(self):
        assembly_started = threading.Event()
        release_assembly = threading.Event()
        start_count = {"n": 0}

        def _start(asynchronous):
            start_count["n"] += 1
            assembly_started.set()
            assert release_assembly.wait(timeout=5)

        manager = self._manager_with_service("svc", start=_start)

        results: list[tuple[str, object]] = []

        def _require():
            try:
                results.append(("ok", manager.require("svc")))
            except Exception as e:
                results.append(("err", e))

        threads = [threading.Thread(target=_require) for _ in range(5)]
        for t in threads:
            t.start()

        assert assembly_started.wait(timeout=2)
        assert manager.get_state("svc") == ServiceState.STARTING
        # give the followers time to block on the in-progress assembly
        threading.Event().wait(0.2)
        assert start_count["n"] == 1

        release_assembly.set()
        for t in threads:
            t.join(timeout=5)

        assert start_count["n"] == 1
        assert len(results) == 5
        assert all(kind == "ok" for kind, _ in results)
        assert len({id(service) for _, service in results}) == 1
        assert manager.get_state("svc") == ServiceState.RUNNING

    def test_concurrent_require_shares_the_same_failure(self):
        release_check = threading.Event()
        check_started = threading.Event()
        check_count = {"n": 0}

        def _check(**kwargs):
            check_count["n"] += 1
            check_started.set()
            assert release_check.wait(timeout=5)
            raise RuntimeError("nope")

        manager = self._manager_with_service("svc", start=lambda asynchronous: None, check=_check)

        errors = []

        def _require():
            try:
                manager.require("svc")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_require) for _ in range(4)]
        for t in threads:
            t.start()
        assert check_started.wait(timeout=2)
        threading.Event().wait(0.2)
        assert check_count["n"] == 1
        release_check.set()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 4
        assert {type(e) for e in errors} == {RuntimeError}
        assert len({id(e) for e in errors}) == 1  # one captured failure, shared by all

    def test_retry_only_legal_from_error(self):
        manager = self._manager_with_service("svc", start=lambda asynchronous: None)
        # AVAILABLE services cannot be "retried"
        with pytest.raises(IllegalServiceStateTransition):
            manager.retry("svc")
        manager.require("svc")
        # RUNNING services cannot be "retried" either, but retry is a no-op-returning success
        assert manager.retry("svc") is manager.get_service("svc")

    def test_deactivate_is_terminal(self):
        def _check(**kwargs):
            raise RuntimeError("dead")

        manager = self._manager_with_service("svc", start=lambda asynchronous: None, check=_check)
        with pytest.raises(RuntimeError):
            manager.require("svc")
        assert manager.get_state("svc") == ServiceState.ERROR

        manager.deactivate("svc")
        assert manager.get_state("svc") == ServiceState.DISABLED

        with pytest.raises(ServiceDisabled):
            manager.require("svc")
        with pytest.raises(IllegalServiceStateTransition):
            manager.retry("svc")
        # deactivate is idempotent
        manager.deactivate("svc")

    def test_illegal_transition_rejects_entire_operation(self):
        manager = self._manager_with_service("svc", start=lambda asynchronous: None)
        manager.require("svc")
        assert manager.get_state("svc") == ServiceState.RUNNING

        # a running service cannot be deactivated, and the rejection leaves state untouched
        with pytest.raises(IllegalServiceStateTransition):
            manager.deactivate("svc")
        assert manager.get_state("svc") == ServiceState.RUNNING
        assert manager.require("svc") is manager.get_service("svc")

    def test_stop_drains_in_flight_requests(self):
        manager = self._manager_with_service("svc", start=lambda asynchronous: None)
        manager.require("svc")

        t1 = manager.begin_request("svc")
        t2 = manager.begin_request("svc")

        result_box = {}
        done = threading.Event()

        def _stop():
            result_box["result"] = manager.stop_service("svc", drain_timeout=5)
            done.set()

        thread = threading.Thread(target=_stop)
        thread.start()
        threading.Event().wait(0.3)
        assert not done.is_set(), "stop must wait for in-flight requests"

        manager.end_request("svc", t1)
        threading.Event().wait(0.2)
        assert not done.is_set()

        manager.end_request("svc", t2)
        assert done.wait(timeout=2)
        assert result_box["result"].stopped
        assert result_box["result"].drained
        assert result_box["result"].interrupted_requests == 0

    def test_stop_force_interrupts_after_timeout(self):
        stop_calls = []
        manager = self._manager_with_service(
            "svc", start=lambda asynchronous: None, stop=lambda: stop_calls.append(1)
        )
        manager.require("svc")

        manager.begin_request("svc")
        manager.begin_request("svc")

        result = manager.stop_service("svc", drain_timeout=0.2)

        assert result.stopped
        assert not result.drained
        assert result.interrupted_requests == 2
        assert stop_calls == [1]
        assert manager.get_state("svc") == ServiceState.STOPPED

        # a later end of an interrupted request is a no-op, and a fresh start resets the counter
        manager.end_request("svc", 0)
        assert manager.require("svc") is manager.get_service("svc")
        clean = manager.stop_service("svc", drain_timeout=0.2)
        assert clean.stopped
        assert clean.interrupted_requests == 0

    def test_stop_waits_for_concurrent_assembly(self):
        release = threading.Event()
        manager = self._manager_with_service("svc", start=lambda asynchronous: release.wait(5))

        def _require():
            manager.require("svc")

        thread = threading.Thread(target=_require)
        thread.start()
        threading.Event().wait(0.2)
        assert manager.get_state("svc") == ServiceState.STARTING

        result = manager.stop_service("svc", drain_timeout=5)
        release.set()
        thread.join(timeout=5)

        assert result.stopped
        assert manager.get_state("svc") == ServiceState.STOPPED

    def test_stop_is_idempotent_for_non_running_states(self):
        manager = self._manager_with_service("svc")
        # AVAILABLE: nothing to stop, no stop function called
        result = manager.stop_service("svc")
        assert result.state == ServiceState.AVAILABLE
        assert result.interrupted_requests == 0
        # stopping twice yields the same terminal result
        manager.require("svc")
        first = manager.stop_service("svc")
        second = manager.stop_service("svc")
        assert first.stopped and second.stopped


class TestServiceLoaderInFlightTracking:
    @pytest.fixture(autouse=True)
    def _enable_dummy_apis(self, monkeypatch):
        # the service manager and the handler chain each hold their own import binding
        monkeypatch.setattr(plugins, "is_api_enabled", lambda api: True)
        monkeypatch.setattr(
            "localstack.aws.handlers.service_plugin.is_api_enabled", lambda api: True
        )

    def test_request_is_tracked_for_entire_chain_lifetime(self):
        service = Service(name="svc", start=lambda asynchronous: None)
        manager = ServiceManager()
        manager.add_service(service)
        loader = ServiceLoader(manager, SimpleNamespace(add_skeleton=lambda s: None))

        context = RequestContext(Request("GET", "/"))
        context.service = SimpleNamespace(service_name="svc")
        response = Response()

        loader.require_service(None, context, response)

        container = manager.get_service_container("svc")
        assert container.state == ServiceState.RUNNING
        assert container.inflight_requests == 1

        # the data-plane loader calling require_service again for the same request must not double-count
        loader.require_service(None, context, response)
        assert container.inflight_requests == 1

        # the request remains in flight until the chain finalizer runs
        ServiceRequestFinalizer()(None, context, response)
        assert container.inflight_requests == 0

        finalizers = context.get(SERVICE_REQUEST_FINALIZERS)
        assert finalizers  # registered on the context for the chain finalizer to drain
