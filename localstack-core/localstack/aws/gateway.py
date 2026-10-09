from rolo.gateway import Gateway as RoloGateway
from rolo.response import Response
from rolo.websocket.request import WebSocketRequest

from .budget import (
    BudgetConfig,
    BudgetHandlerChain,
    populate_budget_error_response,
    read_override_header,
    read_propagated_timeout_ms,
    resolve_request_budget,
)
from .chain import ExceptionHandler, Handler, RequestContext

__all__ = [
    "Gateway",
]


class Gateway(RoloGateway):
    def __init__(
        self,
        request_handlers: list[Handler] = None,
        response_handlers: list[Handler] = None,
        finalizers: list[Handler] = None,
        exception_handlers: list[ExceptionHandler] = None,
        context_class: type[RequestContext] = None,
        budget_config: BudgetConfig = None,
    ) -> None:
        super().__init__(
            request_handlers,
            response_handlers,
            finalizers,
            exception_handlers,
            context_class or RequestContext,
        )
        self.budget_config = budget_config

    def new_chain(self) -> BudgetHandlerChain:
        return BudgetHandlerChain(
            self.request_handlers,
            self.response_handlers,
            self.finalizers,
            self.exception_handlers,
        )

    def handle(self, context: RequestContext, response: Response) -> None:
        """Exposes the same interface as ``HandlerChain.handle``."""
        return self.new_chain().handle(context, response)

    def process(self, request, response: Response):
        """
        Resolves the request-level execution budget at gateway entry and attaches it to the
        request context before the handler chain runs. Requests without a budget, as well as
        websocket requests, are processed exactly as by the plain gateway.
        """
        chain = self.new_chain()
        context = self.context_class(request)

        if not isinstance(request, WebSocketRequest):
            budget_config = self.budget_config or BudgetConfig.load()
            if budget_config.enabled or budget_config.propagate:
                resolution = resolve_request_budget(
                    budget_config,
                    propagated_timeout_ms=read_propagated_timeout_ms(request.headers),
                    override_timeout=read_override_header(request.headers),
                )
                if resolution.rejected:
                    populate_budget_error_response(
                        response,
                        context,
                        code=resolution.error.code,
                        status_code=resolution.error.status_code,
                        message=str(resolution.error),
                    )
                    return
                context.budget = resolution.budget

        chain.handle(context, response)
