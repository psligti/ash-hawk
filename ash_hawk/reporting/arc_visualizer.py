# type-hygiene: skip-file  # dynamic trace data payloads are intentionally Any
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pydantic as pd


class ArcVisualizationConfig(pd.BaseModel):
    title: str = pd.Field(default="Emotional Arc", description="Chart title.")
    width: int = pd.Field(default=1200, description="Figure width in pixels.")
    height: int = pd.Field(default=600, description="Figure height in pixels.")
    color_scheme: dict[str, str] = pd.Field(
        default_factory=lambda: {
            "positive": "rgba(46, 204, 113, 0.25)",
            "negative": "rgba(231, 76, 60, 0.25)",
            "neutral": "#34495e",
            "zero_line": "#95a5a6",
        },
        description="Color scheme for the arc visualization.",
    )
    show_confidence: bool = pd.Field(default=False, description="Show confidence bands.")
    show_inflections: bool = pd.Field(default=True, description="Show inflection markers.")
    show_events: bool = pd.Field(default=True, description="Show event markers.")
    include_plotlyjs: str = pd.Field(default="cdn", description="Plotly JS include mode.")

    model_config = pd.ConfigDict(extra="forbid")


class ArcVisualizer:
    def __init__(self, config: ArcVisualizationConfig | None = None) -> None:
        self._config = config or ArcVisualizationConfig()

    def generate(
        self,
        step_scores: list[dict[str, Any]],
        inflection_points: list[dict[str, Any]],
        dimensions: list[str],
        output_path: str | Path,
        *,
        run_id: str | None = None,
        trial_id: str | None = None,
    ) -> Path:
        path = Path(output_path)
        if not step_scores or not dimensions:
            path.write_text("<div>No data</div>", encoding="utf-8")
            return path

        figure = self._build_figure(
            step_scores,
            inflection_points,
            dimensions,
            run_id=run_id,
            trial_id=trial_id,
        )
        html = figure.to_html(
            full_html=True,
            include_plotlyjs=self._config.include_plotlyjs,
            config={"responsive": True},
        )
        path.write_text(html, encoding="utf-8")
        return path

    def generate_inline_html(
        self,
        step_scores: list[dict[str, Any]],
        inflection_points: list[dict[str, Any]],
        dimensions: list[str],
    ) -> str:
        if not step_scores or not dimensions:
            return "<div>No data</div>"

        figure = self._build_figure(step_scores, inflection_points, dimensions)
        html: str = figure.to_html(
            full_html=False,
            include_plotlyjs=self._config.include_plotlyjs,
            config={"responsive": True},
        )
        return html

    def _build_figure(
        self,
        step_scores: list[dict[str, Any]],
        inflection_points: list[dict[str, Any]],
        dimensions: list[str],
        *,
        run_id: str | None = None,
        trial_id: str | None = None,
    ) -> Any:
        go = self._require_plotly()

        x_values = self._extract_step_indices(step_scores)
        if not x_values:
            return self._empty_figure(go)

        figure = go.Figure()
        line_colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
        ]
        dash_patterns = ["solid", "dash", "dot", "dashdot"]

        for idx, dimension in enumerate(dimensions):
            scores = [self._extract_score(step, dimension) for step in step_scores]
            if all(score is None for score in scores):
                continue

            positive_scores = [
                score if score is not None and score > 0 else None for score in scores
            ]
            negative_scores = [
                score if score is not None and score < 0 else None for score in scores
            ]

            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=positive_scores,
                    mode="lines",
                    line={"color": "rgba(0,0,0,0)"},
                    fill="tozeroy",
                    fillcolor=self._config.color_scheme["positive"],
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=negative_scores,
                    mode="lines",
                    line={"color": "rgba(0,0,0,0)"},
                    fill="tozeroy",
                    fillcolor=self._config.color_scheme["negative"],
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

            error_y = None
            if self._config.show_confidence:
                confidence = [self._extract_confidence(step, dimension) for step in step_scores]
                if any(value is not None for value in confidence):
                    error_y = {
                        "type": "data",
                        "array": [value or 0 for value in confidence],
                        "visible": True,
                    }

            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=[score if score is not None else None for score in scores],
                    mode="lines+markers",
                    name=dimension,
                    line={
                        "color": line_colors[idx % len(line_colors)],
                        "dash": dash_patterns[idx % len(dash_patterns)],
                    },
                    marker={"size": 6},
                    error_y=error_y,
                )
            )

        figure.add_hline(
            y=0,
            line_color=self._config.color_scheme["zero_line"],
            line_dash="dot",
        )

        if self._config.show_inflections:
            for inflection in inflection_points:
                step_index = self._safe_int(inflection.get("step_index"))
                if step_index is None:
                    continue
                label = inflection.get("label") or "inflection"
                figure.add_vline(
                    x=step_index,
                    line_dash="dash",
                    line_color="#7f8c8d",
                    annotation_text=str(label),
                    annotation_position="top",
                )

        if self._config.show_events:
            event_markers = self._extract_event_markers(step_scores, dimensions)
            for event_type, marker in event_markers.items():
                figure.add_trace(
                    go.Scatter(
                        x=marker["x"],
                        y=marker["y"],
                        mode="markers",
                        name=event_type,
                        marker={
                            "symbol": marker["symbol"],
                            "size": 10,
                            "color": self._config.color_scheme["neutral"],
                        },
                        text=marker["text"],
                        hoverinfo="text",
                    )
                )

        title_bits = [self._config.title]
        if run_id:
            title_bits.append(f"Run {run_id}")
        if trial_id:
            title_bits.append(f"Trial {trial_id}")

        figure.update_layout(
            title=" - ".join(title_bits),
            width=self._config.width,
            height=self._config.height,
            xaxis_title="Step",
            yaxis_title="Score",
            template="plotly_white",
            legend={"orientation": "h", "y": -0.2},
        )

        return figure

    @staticmethod
    def _require_plotly() -> Any:
        try:
            import importlib

            return importlib.import_module("plotly.graph_objects")
        except ImportError as exc:
            raise ImportError(
                "Plotly is required for arc visualization. Install plotly to render charts."
            ) from exc

    @staticmethod
    def _extract_step_indices(step_scores: list[dict[str, Any]]) -> list[int]:
        indices: list[int] = []
        for idx, step in enumerate(step_scores):
            value = step.get("step_index")
            if isinstance(value, int):
                indices.append(value)
            else:
                indices.append(idx)
        return indices

    @staticmethod
    def _extract_score(step: dict[str, Any], dimension: str) -> float | None:
        if isinstance(step.get("scores"), dict):
            raw = step["scores"].get(dimension)
        else:
            raw = step.get(dimension)
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_confidence(step: dict[str, Any], dimension: str) -> float | None:
        confidence = step.get("confidence")
        if isinstance(confidence, dict):
            confidence_map = cast(dict[str, Any], confidence)
            raw = confidence_map.get(dimension)
        else:
            raw = None
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        return None

    def _empty_figure(self, go: Any) -> Any:
        figure = go.Figure()
        figure.add_annotation(text="No data", showarrow=False, x=0.5, y=0.5)
        figure.update_layout(
            width=self._config.width,
            height=self._config.height,
            template="plotly_white",
        )
        return figure

    def _extract_event_markers(
        self,
        step_scores: list[dict[str, Any]],
        dimensions: list[str],
    ) -> dict[str, dict[str, Any]]:
        symbols = {
            "tool_call": "circle",
            "verification": "diamond",
            "tool_result": "square",
            "model_message": "star",
        }
        markers: dict[str, dict[str, Any]] = {}

        for idx, step in enumerate(step_scores):
            step_events: list[dict[str, Any]] = []
            if isinstance(step.get("events"), list):
                for item in step["events"]:
                    if isinstance(item, dict):
                        step_events.append(cast(dict[str, Any], item))
            if step.get("event_type"):
                step_events.append(step)

            for event in step_events:
                event_type = str(event.get("event_type") or event.get("type") or "event")
                symbol_key = event_type if event_type in symbols else "model_message"
                summary = event.get("event_summary") or event.get("summary") or event_type
                dimension = event.get("dimension")
                if not isinstance(dimension, str) and dimensions:
                    dimension = dimensions[0]
                y_value = 0.0
                if isinstance(dimension, str):
                    score = self._extract_score(step, dimension)
                    if score is not None:
                        y_value = score

                if event_type not in markers:
                    markers[event_type] = {
                        "x": [],
                        "y": [],
                        "text": [],
                        "symbol": symbols[symbol_key],
                    }
                markers[event_type]["x"].append(step.get("step_index", idx))
                markers[event_type]["y"].append(y_value)
                markers[event_type]["text"].append(str(summary))

        return markers
