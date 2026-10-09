import pytest

from localstack.aws.budget.config import (
    BudgetConfig,
    BudgetConfigurationError,
    BudgetRequestError,
    ExhaustionPolicy,
    resolve_request_budget,
)


class TestBudgetConfigParsing:
    def test_disabled_by_default(self):
        config = BudgetConfig.from_values()
        assert config.enabled is False
        assert config.default_timeout is None
        assert config.propagate is True
        assert config.exempt_streaming is True
        assert config.exempt_long_polling is True

    def test_enable_default_timeout(self):
        config = BudgetConfig.from_values(default_timeout="5.5")
        assert config.enabled is True
        assert config.default_timeout == 5.5
        assert config.max_timeout == 5.5

    def test_invalid_timeout_rejected(self):
        with pytest.raises(BudgetConfigurationError):
            BudgetConfig.from_values(default_timeout="not-a-number")
        with pytest.raises(BudgetConfigurationError):
            BudgetConfig.from_values(default_timeout="-1")
        with pytest.raises(BudgetConfigurationError):
            BudgetConfig.from_values(default_timeout="0")

    def test_max_must_not_be_smaller_than_default(self):
        with pytest.raises(BudgetConfigurationError):
            BudgetConfig.from_values(default_timeout="10", max_timeout="2")

    def test_invalid_policy_rejected(self):
        with pytest.raises(BudgetConfigurationError) as e:
            BudgetConfig.from_values(default_timeout="1", exhaustion_policy="nuke")
        assert "GATEWAY_REQUEST_BUDGET_POLICY" in str(e.value)
        # the only declared policy parses
        config = BudgetConfig.from_values(default_timeout="1", exhaustion_policy="abort")
        assert config.exhaustion_policy is ExhaustionPolicy.ABORT

    def test_exempt_operations_parsed(self):
        config = BudgetConfig.from_values(
            default_timeout="1", exempt_operations="sqs:ReceiveMessage, kinesis:GetRecords"
        )
        assert ("sqs", "ReceiveMessage") in config.exempt_operations
        assert ("kinesis", "GetRecords") in config.exempt_operations

    @pytest.mark.parametrize("bad", ["Sqs", "sqs:", ":Op", "just-a-service"])
    def test_invalid_exempt_operations_rejected(self, bad):
        with pytest.raises(BudgetConfigurationError):
            BudgetConfig.from_values(default_timeout="1", exempt_operations=bad)


class TestRequestBudgetResolution:
    def test_no_budget_when_disabled(self):
        config = BudgetConfig.from_values()
        result = resolve_request_budget(config)
        assert result.budget is None
        assert result.rejected is False

    def test_default_budget(self):
        config = BudgetConfig.from_values(default_timeout="3")
        result = resolve_request_budget(config)
        assert result.budget is not None
        assert result.budget.timeout == 3
        assert result.budget.propagated is False

    def test_internal_call_inherits_remaining_only(self):
        config = BudgetConfig.from_values(default_timeout="30")
        result = resolve_request_budget(config, propagated_timeout_ms=250)
        assert result.budget is not None
        assert result.budget.timeout == pytest.approx(0.25)
        assert result.budget.propagated is True
        # the inherited value must not be capped/extended by the local default
        assert result.budget.remaining() <= 0.25

    def test_internal_propagation_honored_when_locally_disabled(self):
        # a downstream LocalStack must enforce an upstream propagated budget even if its own
        # default budgeting is switched off
        config = BudgetConfig.from_values()
        result = resolve_request_budget(config, propagated_timeout_ms=100)
        assert result.budget is not None
        assert result.budget.propagated is True

    def test_internal_propagation_can_be_switched_off(self):
        config = BudgetConfig.from_values(propagate=False)
        result = resolve_request_budget(config, propagated_timeout_ms=100)
        assert result.budget is None

    def test_override_accepted(self):
        config = BudgetConfig.from_values(
            default_timeout="10", max_timeout="30", allow_override=True
        )
        result = resolve_request_budget(config, override_timeout="5")
        assert result.budget is not None
        assert result.budget.timeout == 5

    def test_override_not_allowed(self):
        config = BudgetConfig.from_values(default_timeout="10", allow_override=False)
        result = resolve_request_budget(config, override_timeout="5")
        assert result.budget is None
        assert result.rejected
        assert result.error.code == "BudgetOverrideNotAllowed"
        assert result.error.status_code == 403

    def test_override_above_max_rejected(self):
        config = BudgetConfig.from_values(
            default_timeout="10", max_timeout="20", allow_override=True
        )
        result = resolve_request_budget(config, override_timeout="21")
        assert result.rejected
        assert result.error.code == "InvalidBudgetRequest"
        assert result.error.status_code == 400

    @pytest.mark.parametrize("bad", ["abc", "-1", "0"])
    def test_override_invalid_values_rejected(self, bad):
        config = BudgetConfig.from_values(default_timeout="10", allow_override=True)
        result = resolve_request_budget(config, override_timeout=bad)
        assert result.rejected
        assert isinstance(result.error, BudgetRequestError)
