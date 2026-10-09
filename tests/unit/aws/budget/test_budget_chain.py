import json
import threading
import time

import pytest

from localstack.aws.api import RequestContext
from localstack.aws.budget import (
    BUDGET_ERROR_HEADER,
    BUDGET_ERROR_HEADER_VALUE,
    BudgetConfig,
    BudgetExemptionHandler,
    ExecutionBudget,
    current_budget,
    inject_budget_into_dto,
)
from localstack.aws.budget.handler import BudgetHandlerChain
from localstack.aws.chain import HandlerChain
from localstack.aws.connect import INTERNAL_REQUEST_PARAMS_HEADER
from localstack.aws.gateway import Gateway
from localstack.http import Request, Response
from localstack.utils.threads import start_thread


@pytest.fixture
def response():
    return Response()


def _sleep_handler(seconds):
    def _handle(chain: HandlerChain, context: RequestContext, response: Response):
        time.sleep(seconds)
        response.status_code = 200
        response.set_json({"ok": True})
        chain.stop()

    return _handle


def _cooperative_handler(seconds, step=0.02):
    """Handler that processes in interruptible chunks, as providers doing I/O would."""

    def _handle(chain: HandlerChain, context: RequestContext, response: Response):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            context.budget.check()
            time.sleep(step)
        response.status_code = 200
        response.set_json({"ok": True})
        chain.stop()

    return _handle


def _gateway(config: BudgetConfig, *handlers, exception_handlers=None) -> Gateway:
    gateway = Gateway(
        request_handlers=list(handlers),
        exception_handlers=list(exception_handlers or []),
        budget_config=config,
    )
    return gateway


class TestGatewayWithoutBudget:
    def test_unconfigured_gateway_verbatim(self, response):
        config = BudgetConfig.from_values()
        gateway = _gateway(config, _sleep_handler(0.05))
        gateway.process(Request("POST", "/"), response)

        assert response.status_code == 200
        assert response.get_json() == {"ok": True}
        assert BUDGET_ERROR_HEADER not in response.headers

    def test_budget_attribute_is_none_without_configuration(self, response):
        config = BudgetConfig.from_values()
        seen = {}

        def recorder(chain, context, response):
            seen["budget"] = context.budget
            seen["current"] = current_budget()
            chain.respond(200, {"seen": True})

        gateway = _gateway(config, recorder)
        gateway.process(Request("GET", "/"), response)
        assert seen["budget"] is None
        assert seen["current"] is None


class TestGatewayBudgetEnforcement:
    def test_request_within_budget_succeeds(self, response):
        config = BudgetConfig.from_values(default_timeout="5")
        gateway = _gateway(config, _sleep_handler(0.02))
        gateway.process(Request("POST", "/"), response)
        assert response.status_code == 200

    def test_exhausted_budget_returns_distinguishable_failure(self, response):
        from localstack.aws.budget import BudgetExhaustedHandler

        config = BudgetConfig.from_values(default_timeout="0.2")
        gateway = _gateway(
            config,
            _cooperative_handler(1),
            exception_handlers=[BudgetExhaustedHandler(config)],
        )
        started = time.monotonic()
        gateway.process(Request("POST", "/"), response)
        elapsed = time.monotonic() - started

        assert response.status_code == 503
        assert response.headers[BUDGET_ERROR_HEADER] == BUDGET_ERROR_HEADER_VALUE
        body = response.get_json()
        assert body["error"] == "RequestBudgetExhausted"
        assert "budget" in body["message"]
        # the server aborted well before the handler would have returned
        assert elapsed < 1

    def test_uninterruptible_handler_is_aborted_at_boundary(self, response):
        # a handler blocked in a non-interruptible call is caught as soon as it returns and
        # can never produce a success response for an exhausted request
        from localstack.aws.budget import BudgetExhaustedHandler

        config = BudgetConfig.from_values(default_timeout="0.05")
        gateway = _gateway(
            config,
            _sleep_handler(0.2),
            exception_handlers=[BudgetExhaustedHandler(config)],
        )
        gateway.process(Request("POST", "/"), response)
        assert response.status_code == 503
        assert response.headers[BUDGET_ERROR_HEADER] == BUDGET_ERROR_HEADER_VALUE

    def test_watchdog_cancels_downstream_thread(self, response):
        from localstack.aws.budget import BudgetExhaustedHandler

        config = BudgetConfig.from_values(default_timeout="0.2")
        cancelled = threading.Event()

        def spawning_handler(chain, context, response):
            holder = {}

            def background(params):
                # downstream work cooperatively bound to the request budget (via start_thread)
                while "thread" not in params:
                    time.sleep(0.001)
                while not params["thread"]._stop_event.wait(0.01):
                    pass
                cancelled.set()

            holder["thread"] = start_thread(background, params=holder, name="bg")
            # emulate an interruptible resource registered directly on the budget
            context.budget.on_cancel(holder["thread"].stop)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                context.budget.check()
                time.sleep(0.02)
            chain.respond(200)

        gateway = _gateway(
            config, spawning_handler, exception_handlers=[BudgetExhaustedHandler(config)]
        )
        gateway.process(Request("POST", "/"), response)
        assert response.status_code == 503
        assert cancelled.wait(timeout=2)

    def test_start_thread_inherits_budget_and_is_stopped(self, response):
        from localstack.aws.budget import BudgetExhaustedHandler

        config = BudgetConfig.from_values(default_timeout="0.3")
        observed = threading.Event()

        def handler(chain, context, response):
            holder = {}

            def background(params):
                budget = current_budget()
                assert budget is context.budget
                observed.set()
                while "thread" not in params:
                    time.sleep(0.001)
                while not params["thread"]._stop_event.wait(0.005):
                    pass

            holder["thread"] = start_thread(background, params=holder, name="bg")
            observed.wait(timeout=2)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                context.budget.check()
                time.sleep(0.02)
            chain.respond(200)

        gateway = _gateway(config, handler, exception_handlers=[BudgetExhaustedHandler(config)])
        gateway.process(Request("POST", "/"), response)
        assert response.status_code == 503
        assert observed.is_set()


class TestGatewayBudgetOverrides:
    def test_valid_override_applied(self, response):
        from localstack.aws.budget import BudgetExhaustedHandler

        config = BudgetConfig.from_values(
            default_timeout="30", max_timeout="30", allow_override=True
        )
        gateway = _gateway(
            config,
            _cooperative_handler(1),
            exception_handlers=[BudgetExhaustedHandler(config)],
        )
        request = Request("POST", "/", headers={"x-localstack-budget-timeout": "0.1"})
        gateway.process(request, response)
        assert response.status_code == 503

    def test_override_above_max_rejected_before_chain(self, response):
        ran = threading.Event()

        def handler(chain, context, response):
            ran.set()
            chain.respond(200)

        config = BudgetConfig.from_values(default_timeout="1", max_timeout="5", allow_override=True)
        gateway = _gateway(config, handler)
        request = Request("POST", "/", headers={"x-localstack-budget-timeout": "10"})
        gateway.process(request, response)

        assert response.status_code == 400
        assert response.get_json()["error"] == "InvalidBudgetRequest"
        assert not ran.is_set()

    def test_override_disallowed_rejected(self, response):
        config = BudgetConfig.from_values(default_timeout="30", allow_override=False)
        gateway = _gateway(config, _sleep_handler(0.01))
        request = Request("POST", "/", headers={"x-localstack-budget-timeout": "1"})
        gateway.process(request, response)
        assert response.status_code == 403
        assert response.get_json()["error"] == "BudgetOverrideNotAllowed"

    @pytest.mark.parametrize("bad", ["abc", "-5"])
    def test_malformed_override_rejected(self, response, bad):
        config = BudgetConfig.from_values(default_timeout="30", allow_override=True)
        gateway = _gateway(config, _sleep_handler(0.01))
        request = Request("POST", "/", headers={"x-localstack-budget-timeout": bad})
        gateway.process(request, response)
        assert response.status_code == 400
        assert response.get_json()["error"] == "InvalidBudgetRequest"


class TestAwsProtocolBudgetFailure:
    def test_aws_request_gets_serialized_budget_error(self, response):
        from localstack.aws.budget import BudgetExhaustedHandler
        from localstack.aws.forwarder import create_aws_request_context

        config = BudgetConfig.from_values(default_timeout="0.1")
        context = create_aws_request_context(
            "sqs",
            "ReceiveMessage",
            parameters={"QueueUrl": "http://localhost/0"},
            protocol="query",
        )
        context.budget = ExecutionBudget(timeout=0.1)

        def slow_handler(chain, ctx, resp):
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                ctx.budget.check()
                time.sleep(0.02)

        chain = BudgetHandlerChain(
            request_handlers=[slow_handler],
            exception_handlers=[BudgetExhaustedHandler(config)],
        )
        chain.handle(context, response)

        assert response.status_code == 503
        assert response.headers[BUDGET_ERROR_HEADER] == BUDGET_ERROR_HEADER_VALUE
        assert b"RequestBudgetExhausted" in response.data
        assert context.service_exception.code == "RequestBudgetExhausted"


class TestPropagatedInternalBudget:
    def test_propagated_header_inherited_and_enforced(self, response):
        from localstack.aws.budget import BudgetExhaustedHandler

        config = BudgetConfig.from_values()  # locally disabled, but propagation honored
        observed = {}

        def handler(chain, context, response):
            observed["budget"] = context.budget
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                context.budget.check()
                time.sleep(0.02)
            chain.respond(200)

        gateway = _gateway(config, handler, exception_handlers=[BudgetExhaustedHandler(config)])
        request = Request(
            "POST",
            "/",
            headers={INTERNAL_REQUEST_PARAMS_HEADER: json.dumps({"request_timeout_ms": 100})},
        )
        gateway.process(request, response)

        assert response.status_code == 503
        assert observed["budget"].propagated is True
        assert observed["budget"].remaining() <= 0.1

    def test_child_cannot_extend_budget(self, response):
        root = ExecutionBudget(timeout=10)
        observed = {}

        def handler(chain, context, response):
            child = context.budget
            observed["propagated"] = child.propagated
            observed["deadline"] = child.deadline
            observed["same_deadline"] = child.deadline == root.deadline
            chain.respond(200, {"ok": True})

        chain = BudgetHandlerChain(request_handlers=[handler])
        request = Request("POST", "/")
        context = RequestContext(request)
        context.budget = root.child_for_internal_call()
        chain.handle(context, response)

        assert response.status_code == 200
        assert observed == {
            "propagated": True,
            "deadline": root.deadline,
            "same_deadline": True,
        }
        root.close()

    def test_dto_injection_only_with_live_budget(self):
        dto = {}
        assert inject_budget_into_dto(dto) == {}

        budget = ExecutionBudget(timeout=10)
        from localstack.aws.budget.core import _current_budget

        token = _current_budget.set(budget)
        try:
            enriched = inject_budget_into_dto({})
        finally:
            _current_budget.reset(token)
        assert 0 < enriched["request_timeout_ms"] <= 10_000

        # exempt budgets propagate no timeout
        budget.exempt("test")
        token = _current_budget.set(budget)
        try:
            assert inject_budget_into_dto({}) == {}
        finally:
            _current_budget.reset(token)


class TestBudgetExemptionRules:
    class _Operation:
        def __init__(
            self, name="TestOp", has_streaming_output=False, has_event_stream_output=False
        ):
            self.name = name
            self.has_streaming_output = has_streaming_output
            self.has_event_stream_output = has_event_stream_output

    class _Service:
        def __init__(self, name):
            self.service_name = name

    def _context(self, service, operation_name, request, **operation_flags):
        context = RequestContext(Request("POST", "/"))
        context.service = self._Service(service)
        context.operation = self._Operation(name=operation_name, **operation_flags)
        context.service_request = request
        return context

    def test_streaming_output_exempt(self):
        config = BudgetConfig.from_values(default_timeout="1", exempt_streaming=True)
        handler = BudgetExemptionHandler(config)
        budget = ExecutionBudget(timeout=1)
        context = self._context("s3", "GetObject", None, has_streaming_output=True)
        context.budget = budget
        handler(HandlerChain(), context, Response())
        assert budget.is_exempt
        assert budget.exemption_reason.startswith("streaming-output")

    def test_declared_long_poll_exempt(self):
        config = BudgetConfig.from_values(default_timeout="1", exempt_long_polling=True)
        handler = BudgetExemptionHandler(config)
        budget = ExecutionBudget(timeout=1)
        context = self._context("sqs", "ReceiveMessage", {"WaitTimeSeconds": 20})
        context.budget = budget
        handler(HandlerChain(), context, Response())
        assert budget.is_exempt
        assert "long-polling" in budget.exemption_reason

    def test_long_poll_without_wait_parameter_not_exempt(self):
        config = BudgetConfig.from_values(default_timeout="1", exempt_long_polling=True)
        handler = BudgetExemptionHandler(config)
        budget = ExecutionBudget(timeout=1)
        context = self._context("sqs", "ReceiveMessage", {"WaitTimeSeconds": 0})
        context.budget = budget
        handler(HandlerChain(), context, Response())
        assert not budget.is_exempt

    def test_declared_operation_exempt(self):
        config = BudgetConfig.from_values(
            default_timeout="1", exempt_operations="kinesis:GetRecords"
        )
        handler = BudgetExemptionHandler(config)
        budget = ExecutionBudget(timeout=1)
        context = self._context("kinesis", "GetRecords", None)
        context.budget = budget
        handler(HandlerChain(), context, Response())
        assert budget.is_exempt

    def test_propagated_budget_never_exempt(self):
        config = BudgetConfig.from_values(default_timeout="1", exempt_streaming=True)
        handler = BudgetExemptionHandler(config)
        root = ExecutionBudget(timeout=10)
        child = root.child_for_internal_call()
        context = self._context("s3", "GetObject", None, has_streaming_output=True)
        context.budget = child
        handler(HandlerChain(), context, Response())
        assert not child.is_exempt
        assert child.is_live
