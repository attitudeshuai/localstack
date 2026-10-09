"""
Handler chain integration of the request execution budget.

* :class:`BudgetHandlerChain` binds the request budget to the execution context, starts the
  deadline watchdog, enforces the deadline before every request handler, and releases the budget
  once the request is finalized. Requests without a budget are processed byte-for-byte like with
  the plain handler chain.
* :class:`BudgetExemptionHandler` applies the declarative long-polling/streaming exemptions once
  the AWS operation has been parsed.
* :class:`BudgetExhaustedHandler` serializes exhausted budgets (and rejected budget overrides)
  into distinguishable failure responses.
"""

from __future__ import annotations

import logging

from rolo.gateway import HandlerChain as RoloHandlerChain

from localstack.http import Response

from ..api import CommonServiceException, RequestContext
from ..chain import ExceptionHandler, Handler, HandlerChain
from .config import (
    BUDGET_ERROR_HEADER,
    BUDGET_ERROR_HEADER_VALUE,
    BudgetConfig,
    BudgetRequestError,
    is_exempt_operation,
)
from .core import BudgetExhaustedError, ExecutionBudget, _current_budget

LOG = logging.getLogger(__name__)


class _BudgetCheckpointHandler:
    """
    Wraps a request handler and enforces the budget both before and after its invocation: a
    deadline firing while the handler is blocking is caught at the boundary as soon as the
    handler returns, so an exhausted request cannot populate a success response.
    """

    def __init__(self, delegate: Handler):
        self.delegate = delegate

    def __call__(self, chain: HandlerChain, context: RequestContext, response: Response):
        budget: ExecutionBudget = context.budget
        budget.check()
        try:
            result = self.delegate(chain, context, response)
        except BudgetExhaustedError:
            raise
        budget.check()
        return result


class BudgetHandlerChain(RoloHandlerChain[RequestContext]):
    """
    Handler chain that enforces a request-level execution budget.

    The chain is a strict superset of the plain rolo ``HandlerChain``: when the request context
    carries no budget, control is delegated unchanged.
    """

    def handle(self, context: RequestContext, response: Response) -> None:
        budget = getattr(context, "budget", None)
        if budget is None:
            # fast path: budgeting disabled -> behavior identical to the plain chain
            return super().handle(context, response)

        request_handlers = self.request_handlers
        self.request_handlers = [_BudgetCheckpointHandler(handler) for handler in request_handlers]
        token = _current_budget.set(budget)
        budget.arm()
        try:
            return super().handle(context, response)
        finally:
            self.request_handlers = request_handlers
            budget.close()
            _current_budget.reset(token)


class BudgetExemptionHandler(Handler):
    """
    Applies the declarative exemption rules (streaming responses and long polling) once the AWS
    operation has been parsed. Nested internal calls always keep their inherited deadline and can
    never exempt themselves.
    """

    def __init__(self, config: BudgetConfig = None):
        self.config = config or BudgetConfig.load()

    def __call__(self, chain: HandlerChain, context: RequestContext, response: Response):
        budget = context.budget
        if budget is None or budget.propagated or not budget.is_live:
            return

        service_name = context.service.service_name if context.service else None
        operation_name = context.operation.name if context.operation else None
        reason = is_exempt_operation(
            service_name=service_name,
            operation_name=operation_name,
            operation_model=context.operation,
            service_request=context.service_request,
            config=self.config,
        )
        if reason:
            LOG.debug(
                "exempting request %s (%s) from execution budget enforcement: %s",
                context.request_id,
                f"{service_name}.{operation_name}",
                reason,
            )
            budget.exempt(reason)


class BudgetExhaustedHandler(ExceptionHandler):
    """
    Serializes budget-related failures into a distinguishable response. For parsed AWS requests
    the failure is serialized through the service serializer; all other requests receive a JSON
    envelope. Both carry the ``x-localstack-error: request-budget-exhausted`` header.
    """

    def __init__(self, config: BudgetConfig = None):
        self.config = config or BudgetConfig.load()

    def __call__(
        self,
        chain: HandlerChain,
        exception: Exception,
        context: RequestContext,
        response: Response,
    ):
        if isinstance(exception, BudgetExhaustedError):
            LOG.warning(
                "request %s budget exhausted (%s): %s",
                context.request_id,
                exception.budget,
                exception.reason,
            )
            populate_budget_error_response(
                response,
                context,
                code=self.config.error_code,
                status_code=self.config.error_status,
                message=str(exception),
            )
        elif isinstance(exception, BudgetRequestError):
            LOG.info("rejecting request %s: %s", context.request_id, exception)
            populate_budget_error_response(
                response,
                context,
                code=exception.code,
                status_code=exception.status_code,
                message=str(exception),
            )


def populate_budget_error_response(
    response: Response,
    context: RequestContext | None,
    *,
    code: str,
    status_code: int,
    message: str,
) -> None:
    """
    Populate a response with a distinguishable budget failure. Shared by the exception handler
    and the gateway entry point (which can reject invalid overrides before the chain runs).
    """
    response.headers[BUDGET_ERROR_HEADER] = BUDGET_ERROR_HEADER_VALUE

    if context is not None and context.service and context.operation:
        service_exception = CommonServiceException(
            code=code,
            message=message,
            status_code=status_code,
            sender_fault=False,
        )
        context.service_exception = service_exception
        # imported lazily: the serializer package pulls in the protocol stack
        from localstack.aws.protocol.serializer import create_serializer

        serialized = create_serializer(
            context.service, context.protocol
        ).serialize_error_to_response(
            service_exception,
            context.operation,
            context.request.headers if context.request else None,
            context.request_id,
        )
        response.update_from(serialized)
    else:
        response.status_code = status_code
        response.set_json({"error": code, "message": message})
