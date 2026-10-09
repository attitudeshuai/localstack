"""
Request-level execution budget for the LocalStack gateway.

The budget bounds the processing time of a single request through service assembly, parameter
parsing, provider execution and internal forwarding. It propagates with the request context
through the entire (possibly nested) call tree: nested internal calls inherit the remaining
quota and cannot extend it. When the budget is exhausted, processing is aborted with a
distinguishable failure and all tasks spawned downstream are cancelled.

When no budget is configured, the gateway behaves exactly as before.
"""

from .config import (
    BUDGET_ERROR_HEADER,
    BUDGET_ERROR_HEADER_VALUE,
    BUDGET_OVERRIDE_HEADER,
    BudgetConfig,
    BudgetConfigurationError,
    BudgetRequestError,
    BudgetResolution,
    ExhaustionPolicy,
    is_exempt_operation,
    resolve_request_budget,
)
from .core import (
    BudgetExhaustedError,
    ExecutionBudget,
    current_budget,
)
from .handler import (
    BudgetExemptionHandler,
    BudgetExhaustedHandler,
    BudgetHandlerChain,
    populate_budget_error_response,
)
from .propagation import (
    BUDGET_TIMEOUT_DTO_KEY,
    BudgetedRequestsClient,
    inject_budget_into_dto,
    read_override_header,
    read_propagated_timeout_ms,
    rewrite_dto_budget_header,
)

__all__ = [
    "BUDGET_ERROR_HEADER",
    "BUDGET_ERROR_HEADER_VALUE",
    "BUDGET_OVERRIDE_HEADER",
    "BUDGET_TIMEOUT_DTO_KEY",
    "BudgetConfig",
    "BudgetConfigurationError",
    "BudgetExhaustedError",
    "BudgetExhaustedHandler",
    "BudgetExemptionHandler",
    "BudgetHandlerChain",
    "BudgetedRequestsClient",
    "BudgetRequestError",
    "BudgetResolution",
    "ExecutionBudget",
    "ExhaustionPolicy",
    "current_budget",
    "inject_budget_into_dto",
    "is_exempt_operation",
    "populate_budget_error_response",
    "read_override_header",
    "read_propagated_timeout_ms",
    "resolve_request_budget",
    "rewrite_dto_budget_header",
]
