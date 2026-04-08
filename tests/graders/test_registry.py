"""Tests for ash_hawk.graders module."""

import pytest

from ash_hawk.graders import (
    ENTRY_POINT_GROUP,
    Grader,
    GraderRegistry,
    PassThroughGrader,
    get_default_registry,
)
from ash_hawk.types import (
    EvalOutcome,
    EvalStatus,
    EvalTranscript,
    EvalTrial,
    GraderResult,
    GraderSpec,
    TrialResult,
)


class CustomGrader(Grader):
    """Test grader for testing."""

    def __init__(self, name: str = "custom", score: float = 1.0, passed: bool = True):
        self._name = name
        self._score = score
        self._passed = passed

    @property
    def name(self) -> str:
        return self._name

    async def grade(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> GraderResult:
        return GraderResult(
            grader_type=self.name,
            score=self._score,
            passed=self._passed,
            details={"custom": True},
        )


class TestGraderABC:
    """Test Grader abstract base class."""

    def test_grader_is_abstract(self):
        """Cannot instantiate Grader directly."""
        with pytest.raises(TypeError):
            Grader()

    def test_grader_requires_name_property(self):
        """Grader subclass must implement name property."""

        class IncompleteGrader(Grader):
            async def grade(self, trial, transcript, spec):
                return GraderResult(grader_type="test", score=1.0, passed=True)

        with pytest.raises(TypeError):
            IncompleteGrader()

    def test_grader_requires_grade_method(self):
        """Grader subclass must implement grade method."""

        class IncompleteGrader(Grader):
            @property
            def name(self):
                return "incomplete"

        with pytest.raises(TypeError):
            IncompleteGrader()

    def test_grader_repr(self):
        """Grader repr includes class name and grader name."""
        grader = CustomGrader(name="test_grader")
        assert "CustomGrader" in repr(grader)
        assert "test_grader" in repr(grader)


class TestPassThroughGrader:
    """Test PassThroughGrader implementation."""

    def test_name(self):
        """PassThroughGrader has correct name."""
        grader = PassThroughGrader()
        assert grader.name == "pass_through"

    @pytest.mark.asyncio
    async def test_grade_always_passes(self):
        """PassThroughGrader always returns score 1.0 and passed=True."""
        grader = PassThroughGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript()
        spec = GraderSpec(grader_type="pass_through")

        result = await grader.grade(trial, transcript, spec)

        assert result.score == 1.0
        assert result.passed is True
        assert result.grader_type == "pass_through"
        assert "message" in result.details


class TestGraderRegistry:
    """Test GraderRegistry class."""

    def test_empty_registry(self):
        """New registry is empty."""
        registry = GraderRegistry()
        assert len(registry) == 0
        assert registry.list_graders() == []

    def test_register_grader(self):
        """Can register a grader."""
        registry = GraderRegistry()
        grader = PassThroughGrader()

        registry.register(grader)

        assert len(registry) == 1
        assert "pass_through" in registry
        assert registry.get("pass_through") is grader

    def test_register_multiple_graders(self):
        """Can register multiple graders."""
        registry = GraderRegistry()
        grader1 = PassThroughGrader()
        grader2 = CustomGrader(name="custom1")
        grader3 = CustomGrader(name="custom2")

        registry.register(grader1)
        registry.register(grader2)
        registry.register(grader3)

        assert len(registry) == 3
        assert registry.list_graders() == ["custom1", "custom2", "pass_through"]

    def test_default_registry_includes_trace_content(self):
        import ash_hawk.graders.registry as reg_module

        reg_module._default_registry = None
        registry = get_default_registry()
        assert "trace_content" in registry.list_graders()

    def test_default_registry_includes_trace_quality(self):
        import ash_hawk.graders.registry as reg_module

        reg_module._default_registry = None
        registry = get_default_registry()
        assert "trace_quality" in registry.list_graders()

    def test_default_registry_includes_scenario_contract_graders(self):
        import ash_hawk.graders.registry as reg_module

        reg_module._default_registry = None
        registry = get_default_registry()
        grader_names = registry.list_graders()
        assert "todo_state" in grader_names
        assert "repo_diff" in grader_names
        assert "completion_honesty" in grader_names
        assert "summary_truthfulness" in grader_names

    def test_register_overwrites(self):
        """Registering with same name overwrites previous grader."""
        registry = GraderRegistry()
        grader1 = CustomGrader(name="test", score=0.5, passed=False)
        grader2 = CustomGrader(name="test", score=1.0, passed=True)

        registry.register(grader1)
        registry.register(grader2)

        assert len(registry) == 1
        assert registry.get("test") is grader2

    def test_get_nonexistent_grader(self):
        """Getting nonexistent grader returns None."""
        registry = GraderRegistry()
        assert registry.get("nonexistent") is None

    def test_list_graders_sorted(self):
        """list_graders returns sorted list."""
        registry = GraderRegistry()
        registry.register(CustomGrader(name="zebra"))
        registry.register(CustomGrader(name="alpha"))
        registry.register(CustomGrader(name="middle"))

        assert registry.list_graders() == ["alpha", "middle", "zebra"]

    def test_contains(self):
        """Can check if grader is registered with 'in'."""
        registry = GraderRegistry()
        registry.register(PassThroughGrader())

        assert "pass_through" in registry
        assert "nonexistent" not in registry

    def test_load_from_entry_points_empty(self, monkeypatch):
        """load_from_entry_points handles no entry points gracefully."""
        registry = GraderRegistry()

        def mock_entry_points(*args, **kwargs):
            return []

        monkeypatch.setattr(
            "ash_hawk.graders.registry.entry_points",
            mock_entry_points,
        )

        registry.load_from_entry_points()
        assert len(registry) == 0


class TestGetDefaultRegistry:
    """Test get_default_registry function."""

    def test_returns_singleton(self, monkeypatch):
        """get_default_registry returns same instance each time."""
        import ash_hawk.graders.registry as reg_module

        reg_module._default_registry = None

        def mock_entry_points(*args, **kwargs):
            return []

        monkeypatch.setattr(
            "ash_hawk.graders.registry.entry_points",
            mock_entry_points,
        )

        registry1 = get_default_registry()
        registry2 = get_default_registry()

        assert registry1 is registry2


class TestEntryPointGroup:
    """Test entry point group constant."""

    def test_entry_point_group_value(self):
        """ENTRY_POINT_GROUP has correct value."""
        assert ENTRY_POINT_GROUP == "ash_hawk.graders"


class TestGraderWithRealTypes:
    """Test grader integration with real EvalTrial and EvalTranscript."""

    @pytest.mark.asyncio
    async def test_grader_with_full_trial(self):
        """Grader works with fully populated trial."""
        grader = PassThroughGrader()

        trial = EvalTrial(
            id="trial-123",
            task_id="task-456",
            status=EvalStatus.COMPLETED,
            result=TrialResult(
                trial_id="trial-123",
                outcome=EvalOutcome.success(),
                aggregate_passed=True,
            ),
        )
        transcript = EvalTranscript(
            messages=[{"role": "user", "content": "test"}],
            tool_calls=[{"tool": "read", "input": {"path": "/tmp"}}],
        )
        spec = GraderSpec(
            grader_type="pass_through",
            config={"key": "value"},
            weight=2.0,
            required=True,
        )

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0
