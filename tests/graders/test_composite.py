"""Tests for CompositeGrader."""

import asyncio

import pytest

from ash_hawk.graders.base import Grader
from ash_hawk.graders.composite import CompositeGrader, CompositeMode
from ash_hawk.types import (
    EvalTranscript,
    EvalTrial,
    GraderResult,
    GraderSpec,
)


class MockGrader(Grader):
    """Mock grader for testing."""

    def __init__(
        self,
        name: str = "mock",
        score: float = 1.0,
        passed: bool = True,
        delay: float = 0.0,
    ):
        self._name = name
        self._score = score
        self._passed = passed
        self._delay = delay

    @property
    def name(self) -> str:
        return self._name

    async def grade(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> GraderResult:
        if self._delay:
            await asyncio.sleep(self._delay)
        return GraderResult(
            grader_type=self._name,
            score=self._score,
            passed=self._passed,
            details={"mock": True},
        )


class FailingGrader(Grader):
    """Grader that raises an exception."""

    def __init__(self, name: str = "failing"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def grade(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> GraderResult:
        raise RuntimeError("Grader failed!")


@pytest.fixture
def trial():
    return EvalTrial(id="trial-1", task_id="task-1")


@pytest.fixture
def transcript():
    return EvalTranscript()


@pytest.fixture
def spec():
    return GraderSpec(grader_type="composite")


class TestCompositeMode:
    """Test CompositeMode enum."""

    def test_mode_values(self):
        assert CompositeMode.WEIGHTED == "weighted"
        assert CompositeMode.ALL_OR_NOTHING == "all_or_nothing"
        assert CompositeMode.THRESHOLD == "threshold"


class TestCompositeGraderInit:
    """Test CompositeGrader initialization."""

    def test_init_with_graders(self):
        graders = [MockGrader("g1"), MockGrader("g2")]
        composite = CompositeGrader(graders)
        assert len(composite.graders) == 2
        assert composite.mode == CompositeMode.WEIGHTED
        assert composite.threshold == 0.7

    def test_init_with_mode(self):
        graders = [MockGrader("g1")]
        composite = CompositeGrader(graders, mode=CompositeMode.ALL_OR_NOTHING)
        assert composite.mode == CompositeMode.ALL_OR_NOTHING

    def test_init_with_threshold(self):
        graders = [MockGrader("g1")]
        composite = CompositeGrader(graders, threshold=0.8)
        assert composite.threshold == 0.8

    def test_init_with_weights(self):
        graders = [MockGrader("g1"), MockGrader("g2")]
        composite = CompositeGrader(graders, weights=[0.3, 0.7])
        assert composite._weights == [0.3, 0.7]

    def test_init_empty_graders_raises(self):
        with pytest.raises(ValueError, match="at least one grader"):
            CompositeGrader([])

    def test_init_weights_mismatch_raises(self):
        graders = [MockGrader("g1"), MockGrader("g2")]
        with pytest.raises(ValueError, match="weights"):
            CompositeGrader(graders, weights=[1.0])

    def test_name_property(self):
        composite = CompositeGrader([MockGrader()])
        assert composite.name == "composite"


class TestWeightedMode:
    """Test weighted scoring mode."""

    @pytest.mark.asyncio
    async def test_weighted_equal_weights(self, trial, transcript, spec):
        graders = [
            MockGrader("g1", score=0.5, passed=True),
            MockGrader("g2", score=1.0, passed=True),
        ]
        composite = CompositeGrader(graders, mode=CompositeMode.WEIGHTED)
        result = await composite.grade(trial, transcript, spec)

        assert result.score == 0.75
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_weighted_custom_weights(self, trial, transcript, spec):
        graders = [
            MockGrader("g1", score=0.5, passed=True),
            MockGrader("g2", score=1.0, passed=True),
        ]
        composite = CompositeGrader(
            graders,
            mode=CompositeMode.WEIGHTED,
            weights=[0.25, 0.75],
        )
        result = await composite.grade(trial, transcript, spec)

        expected = (0.5 * 0.25 + 1.0 * 0.75) / 1.0
        assert result.score == expected

    @pytest.mark.asyncio
    async def test_weighted_fails_below_threshold(self, trial, transcript, spec):
        graders = [
            MockGrader("g1", score=0.0, passed=False),
            MockGrader("g2", score=0.0, passed=False),
        ]
        composite = CompositeGrader(graders, mode=CompositeMode.WEIGHTED)
        result = await composite.grade(trial, transcript, spec)

        assert result.score == 0.0
        assert result.passed is False


class TestAllOrNothingMode:
    """Test all-or-nothing scoring mode."""

    @pytest.mark.asyncio
    async def test_all_pass(self, trial, transcript, spec):
        graders = [
            MockGrader("g1", score=1.0, passed=True),
            MockGrader("g2", score=1.0, passed=True),
        ]
        composite = CompositeGrader(graders, mode=CompositeMode.ALL_OR_NOTHING)
        result = await composite.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_one_fails(self, trial, transcript, spec):
        graders = [
            MockGrader("g1", score=1.0, passed=True),
            MockGrader("g2", score=0.0, passed=False),
        ]
        composite = CompositeGrader(graders, mode=CompositeMode.ALL_OR_NOTHING)
        result = await composite.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.5

    @pytest.mark.asyncio
    async def test_all_fail(self, trial, transcript, spec):
        graders = [
            MockGrader("g1", score=0.0, passed=False),
            MockGrader("g2", score=0.0, passed=False),
        ]
        composite = CompositeGrader(graders, mode=CompositeMode.ALL_OR_NOTHING)
        result = await composite.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.0


class TestThresholdMode:
    """Test threshold scoring mode."""

    @pytest.mark.asyncio
    async def test_above_threshold(self, trial, transcript, spec):
        graders = [
            MockGrader("g1", score=0.6, passed=True),
            MockGrader("g2", score=0.8, passed=True),
        ]
        composite = CompositeGrader(
            graders,
            mode=CompositeMode.THRESHOLD,
            threshold=0.65,
        )
        result = await composite.grade(trial, transcript, spec)

        assert result.score == 0.7
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_below_threshold(self, trial, transcript, spec):
        graders = [
            MockGrader("g1", score=0.5, passed=True),
            MockGrader("g2", score=0.6, passed=True),
        ]
        composite = CompositeGrader(
            graders,
            mode=CompositeMode.THRESHOLD,
            threshold=0.7,
        )
        result = await composite.grade(trial, transcript, spec)

        assert result.score == 0.55
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_exactly_at_threshold(self, trial, transcript, spec):
        graders = [
            MockGrader("g1", score=0.6, passed=True),
            MockGrader("g2", score=0.8, passed=True),
        ]
        composite = CompositeGrader(
            graders,
            mode=CompositeMode.THRESHOLD,
            threshold=0.7,
        )
        result = await composite.grade(trial, transcript, spec)

        assert result.score == 0.7
        assert result.passed is True


class TestParallelExecution:
    """Test parallel grader execution."""

    @pytest.mark.asyncio
    async def test_parallel_execution(self, trial, transcript, spec):
        graders = [
            MockGrader("g1", delay=0.1),
            MockGrader("g2", delay=0.1),
        ]
        composite = CompositeGrader(graders, run_parallel=True)

        import time

        start = time.time()
        result = await composite.grade(trial, transcript, spec)
        elapsed = time.time() - start

        assert result.score == 1.0
        assert elapsed < 0.25

    @pytest.mark.asyncio
    async def test_sequential_execution(self, trial, transcript, spec):
        graders = [
            MockGrader("g1", delay=0.05),
            MockGrader("g2", delay=0.05),
        ]
        composite = CompositeGrader(graders, run_parallel=False)

        import time

        start = time.time()
        result = await composite.grade(trial, transcript, spec)
        elapsed = time.time() - start

        assert result.score == 1.0
        assert elapsed >= 0.1


class TestErrorHandling:
    """Test error handling in composite grader."""

    @pytest.mark.asyncio
    async def test_grader_exception_in_parallel(self, trial, transcript, spec):
        graders = [
            MockGrader("g1", score=1.0, passed=True),
            FailingGrader("failing"),
        ]
        composite = CompositeGrader(graders, run_parallel=True)
        result = await composite.grade(trial, transcript, spec)

        assert len(result.details["grader_results"]) == 2
        failing_result = result.details["grader_results"][1]
        assert failing_result["passed"] is False
        assert failing_result["score"] == 0.0

    @pytest.mark.asyncio
    async def test_grader_exception_in_sequential(self, trial, transcript, spec):
        graders = [
            MockGrader("g1", score=1.0, passed=True),
            FailingGrader("failing"),
        ]
        composite = CompositeGrader(graders, run_parallel=False)
        result = await composite.grade(trial, transcript, spec)

        assert len(result.details["grader_results"]) == 2


class TestConfigOverride:
    """Test config overrides in spec."""

    @pytest.mark.asyncio
    async def test_config_override_mode(self, trial, transcript):
        graders = [
            MockGrader("g1", score=1.0, passed=True),
            MockGrader("g2", score=0.0, passed=False),
        ]
        composite = CompositeGrader(graders, mode=CompositeMode.WEIGHTED)

        spec = GraderSpec(
            grader_type="composite",
            config={"mode": "all_or_nothing"},
        )
        result = await composite.grade(trial, transcript, spec)

        assert result.passed is False

    @pytest.mark.asyncio
    async def test_config_override_threshold(self, trial, transcript):
        graders = [
            MockGrader("g1", score=0.6, passed=True),
            MockGrader("g2", score=0.6, passed=True),
        ]
        composite = CompositeGrader(
            graders,
            mode=CompositeMode.THRESHOLD,
            threshold=0.5,
        )

        spec = GraderSpec(
            grader_type="composite",
            config={"threshold": 0.7},
        )
        result = await composite.grade(trial, transcript, spec)

        assert result.score == 0.6
        assert result.passed is False


class TestResultDetails:
    """Test result details structure."""

    @pytest.mark.asyncio
    async def test_result_includes_mode(self, trial, transcript, spec):
        graders = [MockGrader("g1")]
        composite = CompositeGrader(graders, mode=CompositeMode.ALL_OR_NOTHING)
        result = await composite.grade(trial, transcript, spec)

        assert result.details["mode"] == "all_or_nothing"

    @pytest.mark.asyncio
    async def test_result_includes_grader_results(self, trial, transcript, spec):
        graders = [
            MockGrader("g1", score=0.5, passed=True),
            MockGrader("g2", score=1.0, passed=True),
        ]
        composite = CompositeGrader(graders)
        result = await composite.grade(trial, transcript, spec)

        assert len(result.details["grader_results"]) == 2
        assert result.details["grader_results"][0]["score"] == 0.5
        assert result.details["grader_results"][1]["score"] == 1.0
