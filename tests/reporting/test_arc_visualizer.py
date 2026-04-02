# type-hygiene: skip-file  # test file — mock/factory types are intentionally loose

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pydantic as pd
import pytest

from ash_hawk.reporting.arc_visualizer import ArcVisualizationConfig, ArcVisualizer


class FakeScatter:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class FakeFigure:
    def __init__(self) -> None:
        self.traces: list[object] = []
        self.vlines: list[dict[str, object]] = []
        self.hlines: list[dict[str, object]] = []
        self.annotations: list[dict[str, object]] = []
        self.layouts: list[dict[str, object]] = []

    def add_trace(self, trace: object) -> None:
        self.traces.append(trace)

    def add_vline(self, **kwargs: object) -> None:
        self.vlines.append(kwargs)

    def add_hline(self, **kwargs: object) -> None:
        self.hlines.append(kwargs)

    def add_annotation(self, **kwargs: object) -> None:
        self.annotations.append(kwargs)

    def update_layout(self, **kwargs: object) -> None:
        self.layouts.append(kwargs)

    def to_html(self, **kwargs: object) -> str:
        if kwargs.get("full_html"):
            return "<html>Plot</html>"
        return "<div>Plot</div>"


def make_fake_plotly() -> SimpleNamespace:
    return SimpleNamespace(Figure=FakeFigure, Scatter=FakeScatter)


class TestArcVisualizationConfig:
    def test_defaults(self) -> None:
        config = ArcVisualizationConfig()

        assert config.title == "Emotional Arc"
        assert config.width == 1200
        assert config.height == 600
        assert config.include_plotlyjs == "cdn"

    def test_extra_forbid(self) -> None:
        with pytest.raises(pd.ValidationError):
            ArcVisualizationConfig.model_validate({"extra_field": "nope"})


class TestArcVisualizer:
    def test_generate_inline_html_no_data(self) -> None:
        visualizer = ArcVisualizer()

        html = visualizer.generate_inline_html([], [], [])

        assert html == "<div>No data</div>"

    def test_generate_inline_html_with_data(self) -> None:
        visualizer = ArcVisualizer()
        step_scores = [{"step_index": 0, "scores": {"joy": 0.2}}]

        with patch("importlib.import_module", return_value=make_fake_plotly()):
            html = visualizer.generate_inline_html(step_scores, [], ["joy"])

        assert "<div>Plot</div>" in html

    def test_generate_no_data_writes_file(self, tmp_path: Path) -> None:
        visualizer = ArcVisualizer()
        path = tmp_path / "arc.html"

        visualizer.generate([], [], [], path)

        assert path.read_text(encoding="utf-8") == "<div>No data</div>"

    def test_generate_with_data_writes_file(self, tmp_path: Path) -> None:
        visualizer = ArcVisualizer()
        path = tmp_path / "arc.html"
        step_scores = [{"step_index": 0, "scores": {"joy": 0.2}}]

        with patch("importlib.import_module", return_value=make_fake_plotly()):
            visualizer.generate(step_scores, [], ["joy"], path)

        assert "<html>Plot</html>" in path.read_text(encoding="utf-8")

    def test_extract_step_indices_uses_step_index(self) -> None:
        visualizer = ArcVisualizer()
        extractor = getattr(visualizer, "_extract_step_indices")
        step_scores = [{"step_index": 5}, {"step_index": 8}]

        assert extractor(step_scores) == [5, 8]

    def test_extract_step_indices_fallback(self) -> None:
        visualizer = ArcVisualizer()
        extractor = getattr(visualizer, "_extract_step_indices")
        step_scores = [{"scores": {"joy": 0.1}}, {"scores": {"joy": 0.2}}]

        assert extractor(step_scores) == [0, 1]

    def test_extract_score_from_scores_dict(self) -> None:
        visualizer = ArcVisualizer()
        extractor = getattr(visualizer, "_extract_score")
        step = {"scores": {"joy": 0.3}}

        assert extractor(step, "joy") == 0.3

    def test_extract_score_from_flat_dimension(self) -> None:
        visualizer = ArcVisualizer()
        extractor = getattr(visualizer, "_extract_score")
        step = {"joy": 0.4}

        assert extractor(step, "joy") == 0.4

    def test_extract_score_handles_invalid(self) -> None:
        visualizer = ArcVisualizer()
        extractor = getattr(visualizer, "_extract_score")

        assert extractor({"scores": {"joy": None}}, "joy") is None
        assert extractor({"joy": "nope"}, "joy") is None

    def test_extract_confidence_from_dict(self) -> None:
        visualizer = ArcVisualizer()
        extractor = getattr(visualizer, "_extract_confidence")
        step = {"confidence": {"joy": 0.7}}

        assert extractor(step, "joy") == 0.7

    def test_safe_int_handles_types(self) -> None:
        visualizer = ArcVisualizer()
        safe_int = getattr(visualizer, "_safe_int")

        assert safe_int(3) == 3
        assert safe_int(3.5) == 3
        assert safe_int("nope") is None
