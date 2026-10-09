"""
Propagation of request budgets across internal calls.

Two propagation paths exist:

* **HTTP internal calls** -- the remaining budget is serialized into the internal-call data
  transfer object (the ``x-localstack-data`` header), following the same mechanism used for
  ``source_arn`` and ``service_principal``. The value is the remaining quota in milliseconds at
  the time of dispatch, so each hop inherits at most the remaining quota (gRPC-deadline style).
* **In-memory internal calls** (``GatewayShortCircuit``) -- the nested request context receives
  a child budget sharing the root budget's deadline and cancellation state.

This module must not import ``localstack.aws.connect`` at import time (the connection module
imports this module).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests
from rolo.client import SimpleRequestsClient
from werkzeug.datastructures import Headers

from .core import BudgetExhaustedError, ExecutionBudget, current_budget

LOG = logging.getLogger(__name__)

# Header carrying the internal data transfer object (defined in localstack.aws.connect)
INTERNAL_REQUEST_PARAMS_HEADER = "x-localstack-data"

# Key inside the internal data transfer object carrying the remaining budget in milliseconds
BUDGET_TIMEOUT_DTO_KEY = "request_timeout_ms"


def inject_budget_into_dto(dto: dict[str, Any]) -> dict[str, Any]:
    """
    Add the remaining budget (if any) to an outgoing internal-call data transfer object.

    Called at boto ``before-call`` time for internal clients. When no live budget is bound to the
    current context, the DTO is returned untouched, so unconfigured requests remain byte-identical.
    """
    budget = current_budget()
    if budget is None or budget.is_exhausted:
        return dto
    remaining_ms = budget.remaining_ms()
    if remaining_ms is not None:
        dto[BUDGET_TIMEOUT_DTO_KEY] = remaining_ms
    return dto


def read_propagated_timeout_ms(headers: Headers | dict[str, Any] | None) -> int | None:
    """
    Read the remaining budget propagated through the internal-call DTO header. Malformed headers
    are ignored defensively (they are reported by the regular internal-params handler).
    """
    raw = _get_header(headers, INTERNAL_REQUEST_PARAMS_HEADER)
    if not raw:
        return None
    try:
        dto = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(dto, dict):
        return None
    value = dto.get(BUDGET_TIMEOUT_DTO_KEY)
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def read_override_header(headers: Headers | dict[str, Any] | None) -> str | None:
    """Read the raw per-request budget override header."""
    return _get_header(headers, "x-localstack-budget-timeout")


def rewrite_dto_budget_header(headers: Headers, budget: ExecutionBudget | None) -> None:
    """
    Rewrite the propagated remaining budget on a request that is forwarded to another backend.

    Forwarded requests copy the original request headers, including a possibly stale remaining
    budget. It is refreshed against the current budget right before forwarding so a downstream
    LocalStack cannot inherit more than the caller currently has left.
    """
    if budget is None or budget.is_exhausted:
        return
    raw = headers.get(INTERNAL_REQUEST_PARAMS_HEADER)
    if not raw:
        return
    try:
        dto = json.loads(raw)
    except (ValueError, TypeError):
        return
    if not isinstance(dto, dict):
        return
    remaining_ms = budget.remaining_ms()
    if remaining_ms is None:
        # exempt budget: downstream must not be bounded by a stale value
        dto.pop(BUDGET_TIMEOUT_DTO_KEY, None)
    else:
        dto[BUDGET_TIMEOUT_DTO_KEY] = remaining_ms
    headers[INTERNAL_REQUEST_PARAMS_HEADER] = json.dumps(dto, separators=(",", ":"))


class BudgetedRequestsClient(SimpleRequestsClient):
    """
    A ``SimpleRequestsClient`` that applies the remaining request budget as socket timeout on
    every call, so blocking forwards to external backends (e.g. DynamoDBLocal) are interrupted
    when the budget is exhausted instead of holding the worker thread indefinitely.
    """

    def request(self, request, server: str | None = None):
        budget = current_budget()
        if budget is None or not budget.is_live:
            return super().request(request, server)

        remaining = budget.remaining()
        timeout = max(0.0, remaining or 0.0)

        original_session_request = self.session.request

        def _timed_request(*args, **kwargs):
            kwargs["timeout"] = timeout
            return original_session_request(*args, **kwargs)

        self.session.request = _timed_request
        try:
            return super().request(request, server)
        except requests.exceptions.Timeout as e:
            # the socket timeout is derived from the remaining budget: report the
            # distinguishable budget failure instead of the raw transport error
            budget.exhaust("backend-timeout")
            raise BudgetExhaustedError(budget, "backend-timeout") from e
        finally:
            self.session.request = original_session_request


def _get_header(headers: Headers | dict[str, Any] | None, name: str) -> str | None:
    if headers is None:
        return None
    if isinstance(headers, Headers):
        return headers.get(name)
    if isinstance(headers, dict):
        lowered = name.lower()
        for key, value in headers.items():
            if key.lower() == lowered:
                return value
    return None
