"""
Unified configuration, parsing and validation for the request execution budget.

This module is the single entry point for:

* the default budget (read from the environment/configuration),
* the way individual requests are allowed to override it,
* the rejection conditions for invalid configuration or invalid overrides,
* the declarative exemption rules for long polling and streaming responses.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .core import ExecutionBudget

LOG = logging.getLogger(__name__)

#
# Environment variable names
#
ENV_BUDGET = "GATEWAY_REQUEST_BUDGET"
ENV_BUDGET_MAX = "GATEWAY_REQUEST_BUDGET_MAX"
ENV_BUDGET_ALLOW_OVERRIDE = "GATEWAY_REQUEST_BUDGET_ALLOW_OVERRIDE"
ENV_BUDGET_POLICY = "GATEWAY_REQUEST_BUDGET_POLICY"
ENV_BUDGET_EXEMPT_STREAMING = "GATEWAY_REQUEST_BUDGET_EXEMPT_STREAMING"
ENV_BUDGET_EXEMPT_LONG_POLLING = "GATEWAY_REQUEST_BUDGET_EXEMPT_LONG_POLLING"
ENV_BUDGET_EXEMPT_OPERATIONS = "GATEWAY_REQUEST_BUDGET_EXEMPT_OPERATIONS"
ENV_BUDGET_PROPAGATE = "GATEWAY_REQUEST_BUDGET_PROPAGATE"

# Per-request override header. Value is the requested budget in seconds (float).
BUDGET_OVERRIDE_HEADER = "x-localstack-budget-timeout"

# Response header that identifies a budget failure.
BUDGET_ERROR_HEADER = "x-localstack-error"
BUDGET_ERROR_HEADER_VALUE = "request-budget-exhausted"

# Default identifiable failure codes and statuses
DEFAULT_ERROR_CODE = "RequestBudgetExhausted"
DEFAULT_ERROR_STATUS = 503
OVERRIDE_REJECTED_CODE = "BudgetOverrideNotAllowed"
OVERRIDE_INVALID_CODE = "InvalidBudgetRequest"

# Minimum budget (seconds) a client can request via the override header
MIN_OVERRIDE_TIMEOUT = 0.001


class ExhaustionPolicy(StrEnum):
    """Declared behavior when a budget is exhausted."""

    ABORT = "abort"
    """Abort processing and return an identifiable failure response."""


# Declarative long-polling rules: the request parameter that expresses the client-side wait time.
# Additional operations can be declared through the GATEWAY_REQUEST_BUDGET_EXEMPT_OPERATIONS setting.
LONG_POLL_REQUEST_PARAMETERS: Mapping[tuple[str, str], tuple[str, ...]] = {
    ("sqs", "ReceiveMessage"): ("WaitTimeSeconds",),
}


class BudgetConfigurationError(ValueError):
    """Raised at startup when the budget configuration is invalid."""


@dataclass(frozen=True)
class BudgetConfig:
    """Validated budget configuration."""

    default_timeout: float | None
    """Default per-request budget in seconds, or ``None`` if budgeting is disabled."""

    max_timeout: float | None
    """Upper bound for per-request overrides (and default), in seconds."""

    allow_override: bool = False
    """Whether external callers may override the budget through the request header."""

    exhaustion_policy: ExhaustionPolicy = ExhaustionPolicy.ABORT
    """Behavior when the budget is exhausted."""

    error_code: str = DEFAULT_ERROR_CODE
    """Error code returned when a budget is exhausted on an AWS-protocol request."""

    error_status: int = DEFAULT_ERROR_STATUS
    """HTTP status returned when a budget is exhausted."""

    exempt_streaming: bool = True
    """Whether operations with streaming/event-stream output are exempt by default."""

    exempt_long_polling: bool = True
    """Whether declared long-polling operations are exempt while a wait parameter is set."""

    exempt_operations: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    """Explicitly declared exempt operations as ``(service, operation)`` tuples."""

    propagate: bool = True
    """Whether budgets are propagated to internal cross-service calls."""

    @property
    def enabled(self) -> bool:
        return self.default_timeout is not None

    @classmethod
    def from_values(
        cls,
        *,
        default_timeout: str | float | None = None,
        max_timeout: str | float | None = None,
        allow_override: bool | str = False,
        exhaustion_policy: str = ExhaustionPolicy.ABORT.value,
        exempt_streaming: bool | str = True,
        exempt_long_polling: bool | str = True,
        exempt_operations: str | tuple[str, str] | None = None,
        propagate: bool | str = True,
    ) -> BudgetConfig:
        """
        Parse and validate raw configuration values. This is the single validation entry for
        configuration values; invalid values raise :class:`BudgetConfigurationError`.
        """
        parsed_default = _parse_timeout(default_timeout, name=ENV_BUDGET)
        parsed_max = _parse_timeout(max_timeout, name=ENV_BUDGET_MAX)

        if parsed_default is not None and parsed_default <= 0:
            raise BudgetConfigurationError(
                f"{ENV_BUDGET} must be a positive number of seconds, got '{default_timeout}'"
            )
        if parsed_max is not None and parsed_max <= 0:
            raise BudgetConfigurationError(
                f"{ENV_BUDGET_MAX} must be a positive number of seconds, got '{max_timeout}'"
            )
        if parsed_max is not None and parsed_default is not None and parsed_max < parsed_default:
            raise BudgetConfigurationError(
                f"{ENV_BUDGET_MAX} ({parsed_max}s) must not be smaller than {ENV_BUDGET} "
                f"({parsed_default}s)"
            )

        try:
            policy = ExhaustionPolicy(str(exhaustion_policy).strip().lower())
        except ValueError:
            allowed = ", ".join(p.value for p in ExhaustionPolicy)
            raise BudgetConfigurationError(
                f"{ENV_BUDGET_POLICY} must be one of [{allowed}], got '{exhaustion_policy}'"
            )

        operations = _parse_exempt_operations(exempt_operations)

        # without an explicit ceiling, overrides cannot exceed the default budget
        if parsed_max is None and parsed_default is not None:
            parsed_max = parsed_default

        return cls(
            default_timeout=parsed_default,
            max_timeout=parsed_max,
            allow_override=_as_bool(allow_override, ENV_BUDGET_ALLOW_OVERRIDE),
            exhaustion_policy=policy,
            exempt_streaming=_as_bool(exempt_streaming, ENV_BUDGET_EXEMPT_STREAMING),
            exempt_long_polling=_as_bool(exempt_long_polling, ENV_BUDGET_EXEMPT_LONG_POLLING),
            exempt_operations=operations,
            propagate=_as_bool(propagate, ENV_BUDGET_PROPAGATE),
        )

    @classmethod
    @functools.lru_cache(maxsize=1)
    def load(cls) -> BudgetConfig:
        """Load and validate the budget configuration from the process environment/config."""
        from localstack import config as localstack_config

        return cls.from_values(
            default_timeout=getattr(localstack_config, "GATEWAY_REQUEST_BUDGET", "") or "",
            max_timeout=getattr(localstack_config, "GATEWAY_REQUEST_BUDGET_MAX", "") or "",
            allow_override=getattr(
                localstack_config, "GATEWAY_REQUEST_BUDGET_ALLOW_OVERRIDE", False
            ),
            exhaustion_policy=getattr(
                localstack_config,
                "GATEWAY_REQUEST_BUDGET_POLICY",
                ExhaustionPolicy.ABORT.value,
            ),
            exempt_streaming=getattr(
                localstack_config, "GATEWAY_REQUEST_BUDGET_EXEMPT_STREAMING", True
            ),
            exempt_long_polling=getattr(
                localstack_config, "GATEWAY_REQUEST_BUDGET_EXEMPT_LONG_POLLING", True
            ),
            exempt_operations=getattr(
                localstack_config, "GATEWAY_REQUEST_BUDGET_EXEMPT_OPERATIONS", ""
            ),
            propagate=getattr(localstack_config, "GATEWAY_REQUEST_BUDGET_PROPAGATE", True),
        )


@dataclass
class BudgetResolution:
    """Result of resolving the budget of an incoming request."""

    budget: ExecutionBudget | None
    """The budget to enforce, or ``None`` if the request is not budgeted."""

    error: BudgetRequestError | None = None
    """A rejection error if the request could not be accepted as configured."""

    @property
    def rejected(self) -> bool:
        return self.error is not None


class BudgetRequestError(Exception):
    """A request had to be rejected because of an invalid budget declaration."""

    def __init__(self, code: str, message: str, status_code: int = 400):
        self.code = code
        self.status_code = status_code
        super().__init__(message)


def resolve_request_budget(
    config: BudgetConfig,
    *,
    propagated_timeout_ms: int | None = None,
    override_timeout: str | float | None = None,
) -> BudgetResolution:
    """
    Resolve the budget of an incoming request.

    Inbound internal calls (``propagated_timeout_ms``) always inherit the remaining quota of the
    upstream request; they can neither declare a new budget nor extend the inherited one. External
    requests use the configured default, optionally overridden through the request header when
    overrides are allowed. Invalid overrides are rejected with a distinguishable error.

    :param config: the validated budget configuration
    :param propagated_timeout_ms: remaining budget propagated by an internal caller
    :param override_timeout: raw per-request override value (header value)
    :return: the resolution containing either a budget or a rejection error
    """
    # internal calls inherit the remaining quota (trusted server-to-server propagation)
    if propagated_timeout_ms is not None and config.propagate:
        timeout = max(0.0, propagated_timeout_ms / 1000.0)
        budget = ExecutionBudget(timeout=timeout, propagated=True)
        return BudgetResolution(budget=budget)

    # external per-request overrides
    if override_timeout is not None and str(override_timeout).strip() != "":
        if not config.allow_override:
            return BudgetResolution(
                budget=None,
                error=BudgetRequestError(
                    OVERRIDE_REJECTED_CODE,
                    "request budget overrides are not allowed on this gateway",
                    status_code=403,
                ),
            )
        try:
            requested = float(str(override_timeout).strip())
        except (TypeError, ValueError):
            return BudgetResolution(
                budget=None,
                error=BudgetRequestError(
                    OVERRIDE_INVALID_CODE,
                    f"invalid request budget override '{override_timeout}': "
                    "expected a number of seconds",
                ),
            )
        if requested < MIN_OVERRIDE_TIMEOUT:
            return BudgetResolution(
                budget=None,
                error=BudgetRequestError(
                    OVERRIDE_INVALID_CODE,
                    f"request budget override must be >= {MIN_OVERRIDE_TIMEOUT} seconds, "
                    f"got {requested}",
                ),
            )
        ceiling = config.max_timeout
        if ceiling is not None and requested > ceiling:
            return BudgetResolution(
                budget=None,
                error=BudgetRequestError(
                    OVERRIDE_INVALID_CODE,
                    f"request budget override {requested}s exceeds the allowed maximum of "
                    f"{ceiling}s",
                ),
            )
        return BudgetResolution(budget=ExecutionBudget(timeout=requested))

    # default budget (may be disabled -> None, preserving the historical behavior verbatim)
    if config.default_timeout is not None:
        return BudgetResolution(budget=ExecutionBudget(timeout=config.default_timeout))

    return BudgetResolution(budget=None)


def is_exempt_operation(
    *,
    service_name: str | None,
    operation_name: str | None,
    operation_model: Any | None,
    service_request: Mapping[str, Any] | None,
    config: BudgetConfig,
) -> str | None:
    """
    Evaluate the declarative exemption rules for a parsed request.

    :return: the exemption reason, or ``None`` if the operation is not exempt
    """
    if not service_name or not operation_name:
        return None

    key = (service_name, operation_name)

    if key in config.exempt_operations:
        return f"declared-exempt:{service_name}.{operation_name}"

    if config.exempt_streaming and operation_model is not None:
        if getattr(operation_model, "has_streaming_output", False) or getattr(
            operation_model, "has_event_stream_output", False
        ):
            return f"streaming-output:{service_name}.{operation_name}"

    if config.exempt_long_polling:
        if wait_params := LONG_POLL_REQUEST_PARAMETERS.get(key):
            service_request = service_request or {}
            for param in wait_params:
                if service_request.get(param):
                    return f"long-polling:{service_name}.{operation_name}.{param}"

    return None


#
# helpers
#


def _parse_timeout(value: str | float | None, *, name: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        raise BudgetConfigurationError(f"{name} must be a number of seconds, got '{value}'")


def _as_bool(value: bool | str, name: str = "") -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("", "true", "1", "yes", "on"):
        return True
    if normalized in ("false", "0", "no", "off"):
        return False
    raise BudgetConfigurationError(f"{name} must be 'true' or 'false', got '{value}'")


def _parse_exempt_operations(
    value: str | tuple[str, str] | None,
) -> frozenset[tuple[str, str]]:
    if value is None or value == "":
        return frozenset()
    if isinstance(value, frozenset):
        return value
    if isinstance(value, (set, tuple, list)) and value and isinstance(next(iter(value)), tuple):
        return frozenset(value)  # type: ignore[arg-type]

    operations: set[tuple[str, str]] = set()
    for entry in str(value).split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise BudgetConfigurationError(
                f"{ENV_BUDGET_EXEMPT_OPERATIONS} entries must have the form 'service:Operation', "
                f"got '{entry}'"
            )
        service, operation = entry.split(":", 1)
        service, operation = service.strip(), operation.strip()
        if not service or not operation:
            raise BudgetConfigurationError(
                f"{ENV_BUDGET_EXEMPT_OPERATIONS} entries must have the form 'service:Operation', "
                f"got '{entry}'"
            )
        operations.add((service.lower(), operation))
    return frozenset(operations)
