from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from rich.console import Console
from rich.status import Status

from ash_hawk.thin_runtime.hooks import HookDispatcher
from ash_hawk.thin_runtime.models import HookEvent, ThinRuntimeExecutionResult
from ash_hawk.thin_runtime.persistence import ThinRuntimePersistence


@dataclass(frozen=True)
class _MutedHandlerState:
    handler: logging.Handler
    level: int


@contextmanager
def mute_console_logging() -> Iterator[None]:
    muted_handlers: list[_MutedHandlerState] = []
    for logger in _all_loggers():
        for handler in logger.handlers:
            if not _is_console_handler(handler):
                continue
            muted_handlers.append(_MutedHandlerState(handler=handler, level=handler.level))
            handler.setLevel(logging.CRITICAL + 1)
    try:
        yield
    finally:
        for state in muted_handlers:
            state.handler.setLevel(state.level)


def _all_loggers() -> list[logging.Logger]:
    loggers = [logging.getLogger()]
    for logger in logging.root.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            loggers.append(logger)
    return loggers


def _is_console_handler(handler: logging.Handler) -> bool:
    if isinstance(handler, logging.FileHandler):
        return False
    if isinstance(handler, logging.StreamHandler):
        return True
    stream = getattr(handler, "stream", None)
    if stream in {sys.stdout, sys.stderr}:
        return True
    return handler.__class__.__name__ == "RichHandler"


class ThinRuntimeConsoleReporter:
    def __init__(self, hooks: HookDispatcher) -> None:
        self._step_index = 0
        self._console = Console(highlight=False, soft_wrap=True)
        self._active_status: Status | None = None
        self._active_status_text: str | None = None
        self._active_tool_signature = "unknown-tool()"
        self._streamed_event_count = 0
        hooks.register("before_run", self._before_run)
        hooks.register("before_skill", self._before_skill)
        hooks.register("on_policy_decision", self._on_policy_decision)
        hooks.register("before_tool", self._before_tool)
        hooks.register("after_tool", self._after_tool)
        hooks.register("on_observed_event", self._on_observed_event)
        hooks.register("before_delegation", self._before_delegation)
        hooks.register("on_stop_condition", self._on_stop_condition)
        hooks.register("after_dream_state", self._after_dream_state)

    def _before_run(self, event: HookEvent) -> None:
        self._stop_spinner()
        self._step_index = 0
        goal_id = self._text(event.payload.get("goal_id"), fallback="unknown-goal")
        description = self._text(
            event.payload.get("description"), fallback="No description provided"
        )
        agent = self._text(event.payload.get("agent"), fallback="unknown-agent")
        skills = self._list_text(event.payload.get("skills"))

        self._console.rule("[bold]Thin Runtime Run[/bold]")
        self._console.print(f"[bold cyan]Goal:[/bold cyan] {goal_id}")
        self._console.print(f"[bold cyan]Description:[/bold cyan] {description}")
        self._console.print(f"[bold cyan]Agent:[/bold cyan] {agent}")
        if skills:
            self._console.print(f"[bold cyan]Skills:[/bold cyan] {skills}")
        max_iterations = event.payload.get("max_iterations")
        if isinstance(max_iterations, int):
            self._console.print(f"[bold cyan]Iteration cap:[/bold cyan] {max_iterations}")
        self._console.print(
            "[dim]Console: human-readable steps only. Logger output is muted for this run.[/dim]"
        )
        self._console.print()

    def _before_skill(self, event: HookEvent) -> None:
        skill = self._text(event.payload.get("skill"), fallback="unknown-skill") or "unknown-skill"
        source = self._text(event.payload.get("source"))
        message = self._text(event.payload.get("message"))
        available = event.payload.get("available")
        current_status = self._active_status_text
        self._stop_spinner()
        status_label = "Skill activated"
        if isinstance(available, bool) and not available:
            status_label = "Skill activation failed"
        line = f"[bold green]{status_label}:[/bold green] {skill}"
        if source:
            line = f"{line} [dim]({source})[/dim]"
        self._console.print(line)
        if message and message not in {"skill activated", "skill activation failed"}:
            self._console.print(f"  [dim]{self._truncate(message, limit=180)}[/dim]")
        if current_status is not None:
            self._start_spinner(current_status)

    def _before_tool(self, event: HookEvent) -> None:
        self._stop_spinner()
        self._step_index += 1
        self._streamed_event_count = 0
        tool_name = self._text(event.payload.get("tool"), fallback="unknown-tool") or "unknown-tool"
        tool_args = self._value(event.payload, "tool_args")
        self._active_tool_signature = self._tool_signature(tool_name, tool_args)
        step_total = self._value(event.payload, "max_iterations")
        step_label = f"Step {self._step_index}"
        if isinstance(step_total, int):
            step_label = f"{step_label}/{step_total}"
        self._console.print(f"[bold cyan]{step_label}:[/bold cyan] {self._active_tool_signature}")
        status_text = f"[bold yellow]Running:[/bold yellow] {self._active_tool_signature}"
        self._start_spinner(status_text)

    def _on_policy_decision(self, event: HookEvent) -> None:
        reason = self._text(event.payload.get("reason"))
        if reason is None:
            return
        selected_tool = self._text(event.payload.get("tool"), fallback="none")
        source = self._text(event.payload.get("source"), fallback="unknown")
        confidence = self._value(event.payload, "confidence")
        is_model_reasoning = bool(self._value(event.payload, "reason_model_authored"))
        considered_tools = event.payload.get("considered_tools")
        considered_text = self._list_text(considered_tools)
        if is_model_reasoning:
            confidence_suffix = ""
            if isinstance(confidence, int | float):
                confidence_suffix = f" [dim](confidence {float(confidence):.2f})[/dim]"
            self._console.print(
                f"[bold magenta]Thinking:[/bold magenta] {self._truncate(reason, limit=220)}"
                f"{confidence_suffix}"
            )
            return
        self._console.print(f"[bold magenta]Decision:[/bold magenta] {selected_tool} via {source}.")
        self._console.print(f"  [dim]{self._truncate(reason, limit=220)}[/dim]")
        if considered_text:
            self._console.print(f"  [dim]Considered: {considered_text}[/dim]")

    def _after_tool(self, event: HookEvent) -> None:
        self._stop_spinner()
        success = bool(event.payload.get("success"))
        summary = self._text(event.payload.get("message"))
        error = self._text(event.payload.get("error"))
        status_color = "green" if success else "red"
        status_icon = "✓" if success else "✗"
        preview = summary if success else error
        self._console.print(
            f"[bold cyan]Step {self._step_index} result:[/bold cyan] "
            f"[{status_color}]{status_icon}[/{status_color}] {self._active_tool_signature}"
        )
        if preview is not None:
            self._console.print(f"  [dim]→ {self._truncate(preview, limit=180)}[/dim]")
        self._print_live_tool_result(event.payload)
        self._console.print()
        self._active_status_text = None

    def _on_observed_event(self, event: HookEvent) -> None:
        event_type = self._value_text(event.payload, "event_type")
        if event_type == "skill_activation":
            self._before_skill(event)
            return
        formatted = self._format_event_line(event.payload)
        if formatted is None:
            return
        current_status = self._active_status_text
        self._stop_spinner()
        if self._streamed_event_count == 0:
            self._console.print("  [bold cyan]Observed events:[/bold cyan]")
        self._streamed_event_count += 1
        self._console.print(f"    - {formatted}")
        if current_status is not None:
            self._start_spinner(current_status)

    def _before_delegation(self, event: HookEvent) -> None:
        self._stop_spinner()
        from_agent = self._text(event.payload.get("agent"), fallback="unknown-agent")
        delegated_agent = self._text(event.payload.get("delegated_agent"), fallback="unknown-agent")
        goal_id = self._text(event.payload.get("goal_id"), fallback="unknown-goal")
        self._console.print(
            f"[bold yellow]Delegation:[/bold yellow] {from_agent} -> {delegated_agent} for {goal_id}."
        )

    def _on_stop_condition(self, event: HookEvent) -> None:
        self._stop_spinner()
        reason = self._text(event.payload.get("reason"))
        if reason is None:
            return
        self._console.print(f"[bold yellow]Stopping:[/bold yellow] {reason}")

    def _after_dream_state(self, event: HookEvent) -> None:
        self._stop_spinner()
        applied = event.payload.get("applied", 0)
        if isinstance(applied, int):
            self._console.print(
                f"[bold magenta]Memory consolidation:[/bold magenta] applied {applied} updates."
            )
            self._console.print()

    def print_run_summary(
        self,
        execution: ThinRuntimeExecutionResult,
        persistence: ThinRuntimePersistence,
    ) -> None:
        self._stop_spinner()
        status = "completed" if execution.success else "failed"
        status_color = "green" if execution.success else "red"
        self._console.rule("[bold]Run Summary[/bold]")
        self._console.print(
            f"[bold cyan]Status:[/bold cyan] [{status_color}]{status}[/{status_color}]"
        )
        self._console.print(f"[bold cyan]Run ID:[/bold cyan] {execution.run_id}")
        self._console.print(f"[bold cyan]Agent:[/bold cyan] {execution.agent.name}")
        if execution.selected_tool_names:
            self._console.print(
                f"[bold cyan]Tools used:[/bold cyan] {', '.join(execution.selected_tool_names)}"
            )
        if execution.error:
            self._console.print(f"[bold red]Error:[/bold red] {execution.error}")
        self._print_decision_trace(execution)
        self._print_recorded_tool_activity(execution)
        self._console.print("[bold cyan]Copyable paths:[/bold cyan]")
        self._console.print(f"  [green]Run directory:[/green] {execution.artifact_dir}")
        self._console.print(
            f"  [green]Transcript JSON:[/green] {persistence.execution_file(execution.run_id)}"
        )
        self._console.print(
            f"  [green]Run summary JSON:[/green] {persistence.summary_file(execution.run_id)}"
        )
        self._console.print(
            f"  [green]Session memory snapshot (global):[/green] {persistence.session_file()}"
        )
        self._console.print(
            f"  [green]Durable memory snapshot (global):[/green] {persistence.memory_file()}"
        )
        if persistence.dream_queue_file().exists():
            self._console.print(
                f"  [green]Pending dream queue:[/green] {persistence.dream_queue_file()}"
            )

    def _text(self, value: object, *, fallback: str | None = None) -> str | None:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
        return fallback

    def _list_text(self, value: object) -> str:
        if not isinstance(value, list):
            return ""
        items = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        return ", ".join(items)

    def _print_decision_trace(self, execution: ThinRuntimeExecutionResult) -> None:
        audit = execution.context.audit
        raw_trace = audit.get("decision_trace", [])
        if not isinstance(raw_trace, list):
            return
        decision_trace = [
            item.strip() for item in raw_trace if isinstance(item, str) and item.strip()
        ]
        if not decision_trace:
            return
        self._console.print("[bold magenta]Decision trace:[/bold magenta]")
        for item in decision_trace[:5]:
            self._console.print(f"  - {self._truncate(item, limit=220)}")
        remaining = len(decision_trace) - 5
        if remaining > 0:
            self._console.print(f"  [dim]... +{remaining} more decision entries[/dim]")

    def _print_recorded_tool_activity(self, execution: ThinRuntimeExecutionResult) -> None:
        if not execution.tool_results:
            return
        self._console.print("[bold magenta]Recorded tool activity:[/bold magenta]")
        for result in execution.tool_results:
            status = "ok" if result.success else "failed"
            status_color = "green" if result.success else "red"
            self._console.print(
                f"  - [cyan]{result.tool_name}[/cyan]: [{status_color}]{status}[/{status_color}]"
            )
            self._print_result_details(self._result_view_from_tool_result(result), indent="    ")

    def _print_validation_result(self, view: dict[str, object], *, indent: str) -> None:
        run_result = view.get("run_result")
        if not isinstance(run_result, dict):
            return
        details: list[str] = []
        run_id = run_result.get("run_id")
        if isinstance(run_id, str) and run_id.strip():
            details.append(f"run {run_id}")
        aggregate_score = run_result.get("aggregate_score")
        if isinstance(aggregate_score, int | float):
            details.append(f"score {aggregate_score:.2f}")
        aggregate_passed = run_result.get("aggregate_passed")
        if isinstance(aggregate_passed, bool):
            details.append("passed" if aggregate_passed else "failed")
        if details:
            self._console.print(f"{indent}[bold cyan]Validation:[/bold cyan] {', '.join(details)}")

    def _print_failure_signals(self, view: dict[str, object], *, indent: str) -> None:
        raw_explanations = view.get("failure_signals", [])
        explanations = (
            [item.strip() for item in raw_explanations if isinstance(item, str) and item.strip()]
            if isinstance(raw_explanations, list)
            else []
        )
        if not explanations:
            return
        self._console.print(f"{indent}[bold yellow]Failure signals:[/bold yellow]")
        for item in explanations[:3]:
            self._console.print(f"{indent}  - {self._truncate(item, limit=180)}")
        remaining = len(explanations) - 3
        if remaining > 0:
            self._console.print(f"{indent}  [dim]... +{remaining} more failure signals[/dim]")

    def _print_tool_usage(self, view: dict[str, object], *, indent: str) -> None:
        tool_usage = view.get("tool_usage")
        llm_calls = view.get("llm_calls")
        deterministic_tool_calls = view.get("deterministic_tool_calls")
        agent_invocations = view.get("agent_invocations")

        if isinstance(tool_usage, list):
            tool_items = [item for item in tool_usage if isinstance(item, str) and item.strip()]
            if tool_items:
                self._print_capped_items(
                    f"{indent}Tool usage",
                    tool_items,
                    limit=6,
                )
        if isinstance(llm_calls, list):
            llm_items = [item for item in llm_calls if isinstance(item, str) and item.strip()]
            if llm_items:
                self._print_capped_items(
                    f"{indent}LLM calls",
                    llm_items,
                    limit=4,
                )
        if isinstance(deterministic_tool_calls, list):
            deterministic_items = [
                item for item in deterministic_tool_calls if isinstance(item, str) and item.strip()
            ]
            if deterministic_items:
                self._print_capped_items(
                    f"{indent}Deterministic tool calls",
                    deterministic_items,
                    limit=4,
                )
        if isinstance(agent_invocations, list):
            invocation_items = [
                item for item in agent_invocations if isinstance(item, str) and item.strip()
            ]
            if invocation_items:
                self._print_capped_items(
                    f"{indent}Delegated agents",
                    invocation_items,
                    limit=4,
                )

    def _print_observed_events(
        self,
        view: dict[str, object],
        *,
        indent: str,
        enabled: bool = True,
    ) -> None:
        if not enabled:
            return
        raw_events = view.get("events")
        if not isinstance(raw_events, list) or not raw_events:
            return
        formatted = [self._format_event_line(event) for event in raw_events]
        formatted_events = [item for item in formatted if item is not None]
        if not formatted_events:
            return
        self._console.print(f"{indent}[bold cyan]Observed events:[/bold cyan]")
        for item in formatted_events:
            self._console.print(f"{indent}  - {item}")

    def _format_event_line(self, event: object) -> str | None:
        tool = self._value_text(event, "tool")
        skill = self._value_text(event, "skill")
        event_type = self._value_text(event, "event_type")
        phase = self._value_text(event, "phase")
        rationale = self._value_text(event, "rationale")
        error = self._value_text(event, "error")
        preview = self._value_text(event, "preview")
        success = self._value(event, "success")

        details: list[str] = []
        if tool:
            details.append(tool)
        if skill:
            details.append(f"skill {skill}")
        if event_type and event_type != "tool_result":
            details.append(event_type)
        if phase:
            details.append(f"phase {phase}")
        if isinstance(success, bool):
            details.append("ok" if success else "failed")
        if rationale:
            details.append(rationale)
        if preview:
            details.append(self._truncate(preview, limit=120))
        if error:
            details.append(error)
        if not details:
            return "event recorded"
        return " | ".join(details)

    def _print_capped_items(self, label: str, items: list[str], *, limit: int) -> None:
        visible = items[:limit]
        self._console.print(f"{label}: {', '.join(visible)}")
        remaining = len(items) - len(visible)
        if remaining > 0:
            self._console.print(f"{' ' * len(label)}  [dim]... +{remaining} more[/dim]")

    def _truncate(self, value: str, *, limit: int) -> str:
        if len(value) <= limit:
            return value
        return f"{value[: limit - 3].rstrip()}..."

    def _tool_signature(self, tool_name: str, raw_args: object) -> str:
        if not isinstance(raw_args, dict) or not raw_args:
            return f"{tool_name}()"
        parts: list[str] = []
        for key, value in raw_args.items():
            if not isinstance(key, str):
                continue
            if self._is_sensitive_arg_key(key):
                parts.append(f"{key}=***")
                continue
            formatted = self._format_arg_value(value)
            parts.append(f"{key}={formatted}")
        joined = ", ".join(parts)
        return self._truncate(f"{tool_name}({joined})", limit=180)

    def _format_arg_value(self, value: object) -> str:
        if isinstance(value, str):
            return self._truncate(repr(value), limit=80)
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, int | float):
            return str(value)
        if isinstance(value, list):
            listed = ", ".join(self._format_arg_value(item) for item in value[:3])
            suffix = "" if len(value) <= 3 else ", ..."
            return f"[{listed}{suffix}]"
        if isinstance(value, dict):
            keys = [item for item in value.keys() if isinstance(item, str)]
            preview = ", ".join(keys[:3])
            suffix = "" if len(keys) <= 3 else ", ..."
            return f"dict({preview}{suffix})"
        return self._truncate(repr(value), limit=40)

    def _is_sensitive_arg_key(self, key: str) -> bool:
        lowered = key.lower()
        sensitive_markers = ("token", "secret", "password", "api_key", "apikey", "auth")
        return any(marker in lowered for marker in sensitive_markers)

    def _start_spinner(self, status_text: str) -> None:
        self._active_status_text = status_text
        self._active_status = self._console.status(status_text, spinner="dots")
        self._active_status.start()

    def _stop_spinner(self) -> None:
        if self._active_status is None:
            return
        self._active_status.stop()
        self._active_status = None

    def _print_live_tool_result(self, payload: dict[str, object]) -> None:
        view = self._result_view_from_hook_payload(payload)
        if not self._has_result_details(view):
            return
        self._console.print("  [bold magenta]Result snippet:[/bold magenta]")
        self._print_result_details(
            view,
            indent="    ",
            suppress_observed_events=self._streamed_event_count > 0,
        )

    def _print_result_details(
        self,
        view: dict[str, object],
        *,
        indent: str,
        suppress_observed_events: bool = False,
    ) -> None:
        message = view.get("message")
        if isinstance(message, str):
            stripped = message.strip()
            if stripped:
                self._console.print(
                    f"{indent}[bold cyan]Summary:[/bold cyan] {self._truncate(stripped, limit=160)}"
                )
        error = view.get("error")
        if isinstance(error, str):
            stripped_error = error.strip()
            if stripped_error:
                self._console.print(
                    f"{indent}[bold red]Error:[/bold red] {self._truncate(stripped_error, limit=180)}"
                )
        self._print_validation_result(view, indent=indent)
        self._print_run_summary_fields(view, indent=indent)
        self._print_diff_report(view, indent=indent)
        self._print_failure_signals(view, indent=indent)
        self._print_tool_usage(view, indent=indent)
        self._print_observed_events(
            view,
            indent=indent,
            enabled=not suppress_observed_events,
        )

    def _has_result_details(self, view: dict[str, object]) -> bool:
        return any(
            view.get(key)
            for key in (
                "message",
                "run_result",
                "run_summary",
                "diff_report",
                "failure_signals",
                "tool_usage",
                "llm_calls",
                "deterministic_tool_calls",
                "agent_invocations",
                "events",
            )
        )

    def _result_view_from_tool_result(self, result: object) -> dict[str, object]:
        payload = getattr(result, "payload")
        return {
            "message": payload.message,
            "error": getattr(result, "error", None),
            "run_result": payload.audit_updates.run_result.model_dump(exclude_none=True),
            "run_summary": dict(payload.audit_updates.run_summary),
            "diff_report": dict(payload.audit_updates.diff_report),
            "failure_signals": list(payload.failure_updates.explanations),
            "tool_usage": list(payload.audit_updates.tool_usage),
            "llm_calls": list(payload.audit_updates.llm_calls),
            "deterministic_tool_calls": list(payload.audit_updates.deterministic_tool_calls),
            "agent_invocations": list(payload.audit_updates.agent_invocations),
            "events": [
                event.model_dump(exclude_none=True) for event in payload.audit_updates.events
            ],
        }

    def _result_view_from_hook_payload(self, payload: dict[str, object]) -> dict[str, object]:
        return {
            "message": payload.get("message"),
            "error": payload.get("error"),
            "run_result": payload.get("run_result"),
            "run_summary": payload.get("run_summary"),
            "diff_report": payload.get("diff_report"),
            "failure_signals": payload.get("failure_signals"),
            "tool_usage": payload.get("tool_usage"),
            "llm_calls": payload.get("llm_calls"),
            "deterministic_tool_calls": payload.get("deterministic_tool_calls"),
            "agent_invocations": payload.get("agent_invocations"),
            "events": payload.get("events"),
        }

    def _value(self, source: object, key: str) -> object:
        if isinstance(source, dict):
            return source.get(key)
        return getattr(source, key, None)

    def _value_text(self, source: object, key: str) -> str | None:
        return self._text(self._value(source, key))

    def _print_run_summary_fields(self, view: dict[str, object], *, indent: str) -> None:
        raw_summary = view.get("run_summary")
        if not isinstance(raw_summary, dict):
            return
        rows = [
            (str(key), value)
            for key, value in raw_summary.items()
            if isinstance(key, str) and isinstance(value, str) and value.strip()
        ]
        if not rows:
            return
        self._console.print(f"{indent}[bold cyan]Output details:[/bold cyan]")
        for key, value in rows:
            label = key.replace("_", " ").capitalize()
            rendered = self._truncate(value, limit=220)
            self._console.print(f"{indent}  [cyan]{label}:[/cyan] {rendered}")

    def _print_diff_report(self, view: dict[str, object], *, indent: str) -> None:
        raw_diff_report = view.get("diff_report")
        if not isinstance(raw_diff_report, dict):
            return
        rows = [
            (str(key), value)
            for key, value in raw_diff_report.items()
            if isinstance(key, str) and isinstance(value, int)
        ]
        if not rows:
            return
        self._console.print(f"{indent}[bold cyan]Diff summary:[/bold cyan]")
        for key, value in rows:
            label = key.replace("_", " ").capitalize()
            self._console.print(f"{indent}  [cyan]{label}:[/cyan] {value}")
