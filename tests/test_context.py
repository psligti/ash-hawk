"""Tests for Ash-Hawk logging context and secret redaction."""

import asyncio

from ash_hawk.context import (
    REDACTED,
    EvalContext,
    EvalContextFormatter,
    clear_eval_context,
    get_eval_context,
    redact_secrets,
    set_eval_context,
    setup_eval_logging,
)


class TestEvalContext:
    """Tests for EvalContext dataclass."""

    def test_eval_context_defaults(self):
        """EvalContext should have all None defaults."""
        ctx = EvalContext()
        assert ctx.run_id is None
        assert ctx.suite_id is None
        assert ctx.trial_id is None

    def test_eval_context_with_values(self):
        """EvalContext should accept values on construction."""
        ctx = EvalContext(run_id="run-1", suite_id="suite-1", trial_id="trial-1")
        assert ctx.run_id == "run-1"
        assert ctx.suite_id == "suite-1"
        assert ctx.trial_id == "trial-1"


class TestContextManagement:
    """Tests for context management functions."""

    def test_set_and_get_eval_context(self):
        """set_eval_context should update the context."""
        clear_eval_context()

        set_eval_context(run_id="run-123", suite_id="suite-456", trial_id="trial-789")
        ctx = get_eval_context()

        assert ctx.run_id == "run-123"
        assert ctx.suite_id == "suite-456"
        assert ctx.trial_id == "trial-789"

    def test_set_eval_context_partial_update(self):
        """set_eval_context should update only specified fields."""
        clear_eval_context()

        set_eval_context(run_id="run-1")
        ctx = get_eval_context()
        assert ctx.run_id == "run-1"
        assert ctx.suite_id is None
        assert ctx.trial_id is None

        set_eval_context(suite_id="suite-1")
        ctx = get_eval_context()
        assert ctx.run_id == "run-1"
        assert ctx.suite_id == "suite-1"
        assert ctx.trial_id is None

    def test_clear_eval_context(self):
        """clear_eval_context should reset all fields to None."""
        set_eval_context(run_id="run-1", suite_id="suite-1", trial_id="trial-1")
        clear_eval_context()

        ctx = get_eval_context()
        assert ctx.run_id is None
        assert ctx.suite_id is None
        assert ctx.trial_id is None

    def test_context_is_thread_local(self):
        """Context should be isolated between async tasks."""
        results = {}

        async def task_a():
            set_eval_context(run_id="task-a")
            await asyncio.sleep(0.01)
            ctx = get_eval_context()
            results["a"] = ctx.run_id

        async def task_b():
            set_eval_context(run_id="task-b")
            await asyncio.sleep(0.01)
            ctx = get_eval_context()
            results["b"] = ctx.run_id

        async def run():
            clear_eval_context()
            await asyncio.gather(task_a(), task_b())

        asyncio.run(run())

        assert results["a"] == "task-a"
        assert results["b"] == "task-b"


class TestSecretRedaction:
    """Tests for secret redaction functions."""

    def test_redact_secrets_api_key(self):
        """Should redact api_key."""
        data = {"api_key": "secret-key-123"}
        result = redact_secrets(data)
        assert result["api_key"] == REDACTED

    def test_redact_secrets_variations(self):
        """Should redact various key patterns."""
        data = {
            "api_key": "val1",
            "API_KEY": "val2",
            "api-key": "val3",
            "secret": "val4",
            "my_secret": "val5",
            "password": "val6",
            "user_password": "val7",
            "token": "val8",
            "access_token": "val9",
            "credential": "val10",
            "credentials": "val11",
            "auth_key": "val12",
            "private_key": "val13",
            "access_key": "val14",
        }
        result = redact_secrets(data)

        for key in data:
            assert result[key] == REDACTED, f"Key {key} should be redacted"

    def test_redact_secrets_preserves_safe_keys(self):
        """Should preserve non-sensitive values."""
        data = {
            "name": "test",
            "count": 42,
            "enabled": True,
            "items": [1, 2, 3],
        }
        result = redact_secrets(data)

        assert result["name"] == "test"
        assert result["count"] == 42
        assert result["enabled"] is True
        assert result["items"] == [1, 2, 3]

    def test_redact_secrets_nested_dict(self):
        """Should redact secrets in nested dictionaries."""
        data = {
            "config": {
                "api_key": "nested-secret",
                "name": "config-name",
            },
            "user_settings": {
                "password": "nested-password",
            },
        }
        result = redact_secrets(data)

        assert result["config"]["api_key"] == REDACTED
        assert result["config"]["name"] == "config-name"
        assert result["user_settings"]["password"] == REDACTED

    def test_redact_secrets_nested_list(self):
        """Should redact secrets in nested lists."""
        data = {
            "items": [
                {"name": "item1", "token": "token1"},
                {"name": "item2", "secret": "secret2"},
            ],
        }
        result = redact_secrets(data)

        assert result["items"][0]["name"] == "item1"
        assert result["items"][0]["token"] == REDACTED
        assert result["items"][1]["name"] == "item2"
        assert result["items"][1]["secret"] == REDACTED

    def test_redact_secrets_deeply_nested(self):
        """Should redact secrets in deeply nested structures."""
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "api_key": "deep-secret",
                    },
                },
            },
        }
        result = redact_secrets(data)

        assert result["level1"]["level2"]["level3"]["api_key"] == REDACTED

    def test_redact_secrets_returns_new_dict(self):
        """Should not modify the original dictionary."""
        data = {"api_key": "secret"}
        result = redact_secrets(data)

        assert data["api_key"] == "secret"
        assert result["api_key"] == REDACTED

    def test_redact_secrets_empty_dict(self):
        """Should handle empty dictionary."""
        result = redact_secrets({})
        assert result == {}

    def test_redact_secrets_case_insensitive(self):
        """Should redact regardless of case."""
        data = {
            "API_KEY": "val1",
            "Secret": "val2",
            "PASSWORD": "val3",
            "Token": "val4",
        }
        result = redact_secrets(data)

        assert result["API_KEY"] == REDACTED
        assert result["Secret"] == REDACTED
        assert result["PASSWORD"] == REDACTED
        assert result["Token"] == REDACTED


class TestEvalContextFormatter:
    """Tests for EvalContextFormatter."""

    def test_formatter_includes_context(self):
        """Formatter should include eval context in output."""
        import logging

        set_eval_context(run_id="run-1", suite_id="suite-2", trial_id="trial-3")

        formatter = EvalContextFormatter(
            "[run=%(run_id)s suite=%(suite_id)s trial=%(trial_id)s] %(message)s"
        )

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        assert "run=run-1" in output
        assert "suite=suite-2" in output
        assert "trial=trial-3" in output
        assert "test message" in output

    def test_formatter_handles_missing_context(self):
        """Formatter should handle missing context gracefully."""
        import logging

        clear_eval_context()

        formatter = EvalContextFormatter(
            "[run=%(run_id)s suite=%(suite_id)s trial=%(trial_id)s] %(message)s"
        )

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        assert "run=-" in output
        assert "suite=-" in output
        assert "trial=-" in output


class TestSetupEvalLogging:
    """Tests for setup_eval_logging function."""

    def test_setup_eval_logging_default(self):
        """Should setup logging with default settings."""
        import logging

        setup_eval_logging()

        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) > 0

    def test_setup_eval_logging_custom_level(self):
        """Should accept custom logging level."""
        import logging

        setup_eval_logging(level=logging.DEBUG)

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_setup_eval_logging_json_format(self):
        """Should use JSON format when specified."""
        import logging

        setup_eval_logging(json_format=True)

        root_logger = logging.getLogger()
        handler = root_logger.handlers[-1]
        assert isinstance(handler.formatter, EvalContextFormatter)
