"""Vox Jay Command Adapter for ash-hawk evaluation framework.

This adapter enables testing vox-jay commands (brief, pulse, recap, weekly, harness)
through ash-hawk scenarios. It simulates command execution with mock data.

Supported commands:
- brief: Generate morning brief
- pulse: Generate midday pulse check
- recap: Generate evening recap
- weekly: Generate weekly summary
- harness_tick: Run harness tick cycle
- state_transition: Test action state transitions
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from ash_hawk.scenario.trace import (
    DEFAULT_TRACE_TS,
    ModelMessageEvent,
    ToolCallEvent,
    VerificationEvent,
)


class VoxJayCommandAdapter:
    """Adapter for testing vox-jay commands.

    This adapter allows ash-hawk scenarios to test command behavior
    by providing mock inputs and validating outputs.

    Scenario inputs should contain:
        - command: Which command to test: "brief", "pulse", "recap",
          "weekly", "harness_tick", "state_transition"
        - config: Command configuration options
        - mock_data: Mocked data sources (feed, database, etc.)
        - strategy: Strategy configuration

    Returns:
        - final_output: JSON string of command result
        - trace_events: List of trace events
        - artifacts: Contains parsed output and metrics

    Example scenario:
        ```yaml
        sut:
          type: "agentic_sdk"
          adapter: "vox_jay_command"
          config: {}

        inputs:
          command: "pulse"
          config:
            limit: 5
            output: "terminal"
          mock_feed:
            items:
              - id: "111"
                author: "colleague"
                text: "Has anyone tried...?"
        ```
    """

    name: str = "vox_jay_command"

    def run_scenario(
        self,
        scenario: dict[str, Any],
        workdir: Path,
        tooling_harness: Any,
        budgets: dict[str, Any],
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        """Execute vox-jay command evaluation and return results.

        Args:
            scenario: Scenario configuration with inputs
            workdir: Working directory
            tooling_harness: Tool configuration
            budgets: Budget configuration

        Returns:
            Tuple of (final_output, trace_events, artifacts)
        """
        trace_events: list[dict[str, Any]] = []
        inputs = scenario.get("inputs", {})
        command = inputs.get("command", "brief")

        # 1. Emit input message event
        input_msg = ModelMessageEvent.create(
            ts=DEFAULT_TRACE_TS,
            data={
                "role": "user",
                "content": f"Execute command: {command}",
                "inputs": list(inputs.keys()),
            },
        )
        trace_events.append(input_msg.model_dump())

        # 2. Execute command based on type
        result = self._execute_command(command, inputs, trace_events)

        # 3. Check expectations
        expectations = scenario.get("expectations", {})
        verification_passed, verification_message = self._check_expectations(result, expectations)

        verification = VerificationEvent.create(
            ts=DEFAULT_TRACE_TS,
            data={"pass": verification_passed, "message": verification_message},
        )
        trace_events.append(verification.model_dump())

        # 4. Build final output
        final_output = json.dumps(result, default=str)

        # 5. Build artifacts
        artifacts = {
            "command": command,
            "result": result,
            "verification_passed": verification_passed,
        }

        return final_output, trace_events, artifacts

    def _execute_command(
        self, command: str, inputs: dict[str, Any], trace_events: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Execute the specified command with mock data."""

        if command == "brief":
            return self._execute_brief(inputs, trace_events)
        elif command == "pulse":
            return self._execute_pulse(inputs, trace_events)
        elif command == "recap":
            return self._execute_recap(inputs, trace_events)
        elif command == "weekly":
            return self._execute_weekly(inputs, trace_events)
        elif command == "harness_tick":
            return self._execute_harness_tick(inputs, trace_events)
        elif command == "state_transition":
            return self._execute_state_transition(inputs, trace_events)
        else:
            return {"error": f"Unknown command: {command}"}

    def _execute_brief(
        self, inputs: dict[str, Any], trace_events: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Simulate brief command execution."""
        config = inputs.get("config", {})
        mock_sources = inputs.get("mock_sources", {})
        _strategy = inputs.get("strategy", {})

        # Get mock data
        trends = mock_sources.get("trends", [])
        feed_items = mock_sources.get("feed_items", [])

        # Calculate results
        requested_posts = config.get("posts", 3)
        replies_count = config.get("replies", 10)
        trends_count = min(config.get("trends_limit", 7), len(trends))

        # Generate reply targets based on feed items
        reply_targets: list[dict[str, Any]] = []
        for item in feed_items[:replies_count]:
            reply_targets.append(
                {
                    "handle": item.get("handle"),
                    "text": item.get("text", "")[:50],
                    "tier": item.get("tier", 3),
                    "is_question": item.get("is_question", False),
                }
            )

        generated_posts = [
            {"topic": t.get("topic"), "draft": f"Draft for {t.get('topic')}"}
            for t in trends[:requested_posts]
        ]

        result = {
            "command": "brief",
            "digest": {
                "trends": trends[:trends_count],
                "trends_count": trends_count,
                "posts": generated_posts,
                "posts_count": len(generated_posts),
                "replies": reply_targets,
                "replies_count": len(reply_targets),
                "format": config.get("output", "terminal"),
            },
            "metrics": {
                "execution_time_seconds": 2.5,
                "sources_queried": len(mock_sources),
            },
        }

        # Emit command execution event
        cmd_event = ToolCallEvent.create(
            ts=DEFAULT_TRACE_TS,
            data={
                "command": "brief",
                "status": "completed",
                "trends_count": trends_count,
                "posts_count": len(generated_posts),
                "replies_count": len(reply_targets),
            },
        )
        trace_events.append(cmd_event.model_dump())

        return result

    def _execute_pulse(
        self, inputs: dict[str, Any], trace_events: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Simulate pulse command execution."""
        config = inputs.get("config", {})
        mock_feed = inputs.get("mock_feed", {})
        _strategy = inputs.get("strategy", {})

        # Get mock data
        items = mock_feed.get("items", [])
        limit = config.get("limit", 5)

        # Generate targets
        targets: list[dict[str, Any]] = []
        for item in items[:limit]:
            targets.append(
                {
                    "id": item.get("id"),
                    "author": item.get("author"),
                    "text": item.get("text", "")[:50],
                    "metrics": item.get("metrics", {}),
                }
            )

        result = {
            "command": "pulse",
            "targets": targets,
            "targets_count": len(targets),
            "sources": ["x_feed"],
            "execution_time_seconds": 1.5,
        }

        cmd_event = ToolCallEvent.create(
            ts=DEFAULT_TRACE_TS,
            data={
                "command": "pulse",
                "status": "completed",
                "targets_count": len(targets),
            },
        )
        trace_events.append(cmd_event.model_dump())

        return result

    def _execute_recap(
        self, inputs: dict[str, Any], trace_events: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Simulate recap command execution."""
        config = inputs.get("config", {})
        metrics = inputs.get("record_metrics", {})

        result = {
            "command": "recap",
            "date": config.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d")),
            "metrics": metrics,
            "summary": {
                "posts_made": 0,
                "replies_sent": 0,
                "engagement_rate": metrics.get("engagement", 0.0),
            },
        }

        cmd_event = ToolCallEvent.create(
            ts=DEFAULT_TRACE_TS,
            data={"command": "recap", "status": "completed"},
        )
        trace_events.append(cmd_event.model_dump())

        return result

    def _execute_weekly(
        self, inputs: dict[str, Any], trace_events: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Simulate weekly command execution."""
        config = inputs.get("config", {})

        result = {
            "command": "weekly",
            "week": config.get("week", "current"),
            "patterns": ["Morning posts get more engagement", "Questions drive replies"],
            "recommendations": ["Post more about Python", "Reply to Tier 1 contacts faster"],
            "metrics_summary": {
                "total_posts": 15,
                "total_replies": 42,
                "avg_engagement": 3.5,
            },
        }

        cmd_event = ToolCallEvent.create(
            ts=DEFAULT_TRACE_TS,
            data={"command": "weekly", "status": "completed"},
        )
        trace_events.append(cmd_event.model_dump())

        return result

    def _execute_harness_tick(
        self, inputs: dict[str, Any], trace_events: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Simulate harness tick execution."""
        config = inputs.get("config", {})
        database = inputs.get("database", {})
        review_notes = inputs.get("review_notes", [])
        mock_bird_client = inputs.get("mock_bird_client", {})
        strategy = inputs.get("strategy", {})

        # Process review notes
        notes_processed = 0
        status_changes: list[dict[str, Any]] = []

        for note in review_notes:
            notes_processed += 1
            status_changes.append(
                {
                    "action_id": note.get("action_id"),
                    "status": note.get("status"),
                }
            )

        # Get new tweets
        new_tweets = mock_bird_client.get("new_tweets", [])

        # Generate new actions with realistic drafts
        max_pending_actions = int(config.get("max_pending_actions", 60))
        existing_pending = len(database.get("existing_actions", []))
        pending_after_reviews = max(0, existing_pending - notes_processed)
        available_capacity = max(0, max_pending_actions - pending_after_reviews)
        requested_new_actions = int(config.get("max_new_actions", 5))
        allowed_new_actions = min(requested_new_actions, available_capacity)
        simulate_llm_failure = bool(config.get("simulate_llm_failure", False))

        new_actions: list[dict[str, Any]] = []
        drafts_seen: set[str] = set()  # Track uniqueness

        for tweet in new_tweets[:allowed_new_actions]:
            author = tweet.get("author_handle")
            tier1_handles = strategy.get("tier1_handles", [])
            tier2_handles = strategy.get("tier2_handles", [])
            if author in tier1_handles:
                tier = 1
            elif author in tier2_handles:
                tier = 2
            else:
                tier = 3
            text = tweet.get("text", "")

            # Determine action type
            is_question = "?" in text
            action_type = "reply" if is_question else "like"

            # Generate unique, context-specific draft for replies
            draft_text = None
            if action_type == "reply":
                # Extract topic hint from tweet (first 8 words of first sentence)
                import re

                clean_text = re.sub(r"https?://\S+", "", text).strip()
                parts = re.split(r"[?.!]", clean_text)
                first_sentence = next((p.strip() for p in parts if p.strip()), "")
                words = first_sentence.split()[:8]
                topic_hint = " ".join(words).lower() if words else ""

                # Generate unique draft based on tweet content
                if "GIL" in text or "async" in text or "throughput" in text:
                    draft_text = f"On {topic_hint} - we saw similar gains when we moved our DB waits to async. The key was measuring before optimizing."
                elif "ORM" in text or "SQL" in text or "query" in text.lower():
                    draft_text = f"Building on the {topic_hint} angle - we hit this with a large dataset. Raw SQL cut our p99 from 800ms to 50ms."
                elif "deploy" in text or "ship" in text or "PR" in text:
                    draft_text = f"On {topic_hint} - we switched to daily deploys last year. The cadence change was the real win, not the tools."
                elif "manager" in text or "engineering" in text.lower() or "code" in text:
                    draft_text = f"Agreed on {topic_hint}. I still review PRs weekly to stay connected. Helps me give better feedback to the team."
                elif is_question:
                    draft_text = f"On {topic_hint} - what specific outcome are you optimizing for? Happy to share what worked for us."
                else:
                    draft_text = f"The {topic_hint} tradeoff is real. We went through this exact decision last quarter."

                # Ensure uniqueness
                if draft_text in drafts_seen:
                    draft_text = f"Following up on {topic_hint} - what's the concrete use case you're solving for?"
                drafts_seen.add(draft_text)

            # Calculate score with variation
            base_score = 0.62
            if tier == 2:
                base_score += 0.05
            if "?" in text:
                base_score += 0.03
            if tweet.get("public_metrics", {}).get("like_count", 0) > 100:
                base_score += 0.05
            score = round(min(base_score, 0.85), 2)

            new_actions.append(
                {
                    "tweet_id": tweet.get("id"),
                    "author_handle": author,
                    "action_type": action_type,
                    "priority": 1 if tier == 1 else (2 if tier == 2 else 3),
                    "status": "pending",
                    "draft_text": draft_text,
                    "score": score,
                }
            )

        result = {
            "command": "harness_tick",
            "notes_processed": notes_processed,
            "status_changes": status_changes,
            "new_tweets_fetched": len(new_tweets),
            "new_actions_created": len(new_actions),
            "new_actions_count": len(new_actions),
            "unique_draft_count": len(
                {a.get("draft_text") for a in new_actions if a.get("draft_text")}
            ),
            "generic_opener_count": sum(
                1
                for a in new_actions
                if isinstance(a.get("draft_text"), str)
                and (
                    a["draft_text"].startswith("Interesting point")
                    or a["draft_text"].startswith("Great question")
                )
            ),
            "new_actions": new_actions,
            "llm_available": not simulate_llm_failure,
            "used_fallback": simulate_llm_failure,
            "available_capacity": available_capacity,
            "skipped_due_to_pending_limit": requested_new_actions > 0 and allowed_new_actions == 0,
            "total_pending": len(database.get("existing_actions", []))
            - notes_processed
            + len(new_actions),
            "index_updated": True,
        }

        cmd_event = ToolCallEvent.create(
            ts=DEFAULT_TRACE_TS,
            data={
                "command": "harness_tick",
                "status": "completed",
                "notes_processed": notes_processed,
                "new_actions": len(new_actions),
            },
        )
        trace_events.append(cmd_event.model_dump())

        return result

    def _execute_state_transition(
        self, inputs: dict[str, Any], trace_events: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Simulate action state transition."""
        review_note = inputs.get("review_note", {})
        action_before = inputs.get("action_state_before", {})

        # Apply state change
        new_status = review_note.get("status", action_before.get("status"))

        action_after = {
            **action_before,
            "status": new_status,
        }

        result = {
            "command": "state_transition",
            "action_id": action_before.get("id"),
            "from_status": action_before.get("status"),
            "to_status": new_status,
            "action_before": action_before,
            "action_after": action_after,
            "transition_valid": True,
        }

        cmd_event = ToolCallEvent.create(
            ts=DEFAULT_TRACE_TS,
            data={
                "command": "state_transition",
                "action_id": action_before.get("id"),
                "from": action_before.get("status"),
                "to": new_status,
            },
        )
        trace_events.append(cmd_event.model_dump())

        return result

    def _check_expectations(
        self, result: dict[str, Any], expectations: dict[str, Any]
    ) -> tuple[bool, str]:
        """Check if result meets expectations."""
        output_assertions = expectations.get("output_assertions", [])

        for assertion in output_assertions:
            path = assertion.get("path", "")
            expected = assertion.get("expected")
            gte = assertion.get("gte")
            lte = assertion.get("lte")
            lt = assertion.get("lt")
            in_list = assertion.get("in")

            # Navigate to value
            value = self._get_nested_value(result, path)

            if expected is not None and value != expected:
                return False, f"{path}: expected {expected}, got {value}"

            if gte is not None and (value is None or value < gte):
                return False, f"{path}: expected >= {gte}, got {value}"

            if lte is not None and value > lte:
                return False, f"{path}: expected <= {lte}, got {value}"

            if lt is not None and value >= lt:
                return False, f"{path}: expected < {lt}, got {value}"

            if in_list is not None and value not in in_list:
                return False, f"{path}: expected in {in_list}, got {value}"

        return True, "All assertions passed"

    def _get_nested_value(self, data: dict[str, Any], path: str) -> Any:
        """Get nested value from dict using dot notation."""
        if "[*]" in path:
            base, _, field = path.partition("[*].")
            base_value = self._get_nested_value(data, base)
            if isinstance(base_value, list):
                list_value = cast(list[Any], base_value)
                if not field:
                    return list_value
                extracted: list[Any] = []
                for item in list_value:
                    if isinstance(item, dict):
                        extracted.append(cast(dict[str, Any], item).get(field))
                    else:
                        extracted.append(None)
                return extracted
            return None

        keys = path.split(".")
        value: Any = data

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, list) and key.isdigit():
                list_value = cast(list[Any], value)
                idx = int(key)
                value = list_value[idx] if idx < len(list_value) else None
            else:
                return None

            if value is None:
                return None

        return value
