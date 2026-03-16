"""Vox Jay Policy Adapter for ash-hawk evaluation framework.

This adapter enables testing vox-jay policy agents through ash-hawk scenarios.
It supports testing EngagementPolicy, RankingPolicy, StrategyPolicy, and MasterPolicy.
"""

from __future__ import annotations

from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

from ash_hawk.scenario.trace import (
    DEFAULT_TRACE_TS,
    ModelMessageEvent,
    PolicyDecisionEvent,
    VerificationEvent,
)

# Default policy to use if not specified
DEFAULT_POLICY = "master"


class VoxJayPolicyAdapter:
    """Adapter for testing vox-jay policy agents.

    This adapter allows ash-hawk scenarios to test policy decisions
    by providing structured inputs (tweets, relationships, strategy, budget)
    and validating the policy outputs.

    Scenario inputs should contain:
        - policy: (optional) Which policy to test: "engagement", "ranking",
          "strategy", or "master" (default: "master")
        - tweets: List of tweet objects with id, author_handle, text, etc.
        - relationships: Dict mapping handles to relationship info
        - strategy: Strategy configuration (tiers, topics, etc.)
        - budget: Budget constraints

    Returns:
        - final_output: JSON string of PolicyOutput
        - trace_events: List of trace events including policy decisions
        - artifacts: Contains the parsed PolicyOutput

    Example scenario:
        ```yaml
        sut:
          type: "agentic_sdk"
          adapter: "vox_jay_policy"
          config: {}

        inputs:
          policy: "engagement"
          tweets:
            - id: "123"
              author_handle: "test_user"
              text: "What do you think?"
          relationships:
            test_user:
              tier: 1
          budget:
            max_pending_actions: 10
        ```
    """

    name: str = "vox_jay_policy"

    def run_scenario(
        self,
        scenario: dict[str, Any],
        workdir: Path,
        tooling_harness: Any,
        budgets: dict[str, Any],
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        """Execute vox-jay policy evaluation and return results.

        Args:
            scenario: Scenario configuration with inputs
            workdir: Working directory (unused for policy tests)
            tooling_harness: Tool configuration (unused for policy tests)
            budgets: Budget configuration (passed to policy)

        Returns:
            Tuple of (final_output, trace_events, artifacts)
        """
        import json

        from vox_jay.policies import (
            ActionType,
            AuthorRelationship,
            BudgetExhaustedError,
            BudgetInfo,
            EngagementPolicy,
            MasterPolicy,
            PolicyInput,
            PolicyOutput,
            RankingPolicy,
            StrategyContext,
            StrategyPolicy,
            TierConfiguration,
            TopicConfiguration,
            TweetContext,
            TweetMetrics,
        )

        trace_events: list[dict[str, Any]] = []
        inputs = scenario.get("inputs", {})

        # 1. Emit input message event
        input_msg = ModelMessageEvent.create(
            ts=DEFAULT_TRACE_TS,
            data={
                "role": "user",
                "content": f"Evaluate policy: {inputs.get('policy', DEFAULT_POLICY)}",
                "tweets_count": len(inputs.get("tweets", [])),
            },
        )
        trace_events.append(input_msg.model_dump())

        # 2. Build TweetContext objects
        tweet_contexts = []
        for tweet_data in inputs.get("tweets", []):
            metrics = TweetMetrics(
                like_count=tweet_data.get("public_metrics", {}).get("like_count", 0),
                retweet_count=tweet_data.get("public_metrics", {}).get("retweet_count", 0),
                reply_count=tweet_data.get("public_metrics", {}).get("reply_count", 0),
                quote_count=tweet_data.get("public_metrics", {}).get("quote_count", 0),
                impression_count=tweet_data.get("public_metrics", {}).get("impression_count", 0),
            )

            created_at_str = tweet_data.get("created_at", "2026-01-01T00:00:00Z")
            try:
                created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            except ValueError:
                created_at = datetime.now(UTC)

            tweet_context = TweetContext(
                id=tweet_data.get("id", ""),
                author_handle=tweet_data.get("author_handle", ""),
                author_name=tweet_data.get("author_name", tweet_data.get("author_handle", "")),
                text=tweet_data.get("text", ""),
                created_at=created_at,
                public_metrics=metrics,
                is_question="?" in tweet_data.get("text", ""),
                detected_topics=tweet_data.get("detected_topics", []),
            )
            tweet_contexts.append(tweet_context)

        # 3. Build relationships
        relationships: dict[str, AuthorRelationship] = {}
        for handle, rel_data in inputs.get("relationships", {}).items():
            relationships[handle] = AuthorRelationship(
                handle=handle,
                tier=rel_data.get("tier", 3),
                interaction_count=rel_data.get("interaction_count", 0),
                mutual_follow=rel_data.get("mutual_follow", False),
            )

        # 4. Build strategy context
        strategy_data = inputs.get("strategy", {})
        tiers_data = strategy_data.get("tiers", {})
        topics_data = strategy_data.get("topics", {})

        strategy_context = StrategyContext(
            tiers=TierConfiguration(
                tier1_handles=tiers_data.get("tier1_handles", []),
                tier2_handles=tiers_data.get("tier2_handles", []),
            ),
            topics=TopicConfiguration(
                primary_topics=topics_data.get("primary_topics", []),
                secondary_topics=topics_data.get("secondary_topics", []),
                expertise_areas=topics_data.get("expertise_areas", []),
                avoid_topics=topics_data.get("avoid_topics", []),
            ),
            principles=strategy_data.get("principles", []),
            boundaries=strategy_data.get("boundaries", []),
        )

        # 5. Build budget
        budget_data = inputs.get("budget", {})
        budget_info = BudgetInfo(
            max_pending_actions=budget_data.get("max_pending_actions", 10),
            current_pending_actions=budget_data.get("current_pending_actions", 0),
            max_api_calls=budget_data.get("max_api_calls", 100),
            api_calls_consumed=budget_data.get("api_calls_consumed", 0),
        )

        # 6. Create policy input
        policy_input = PolicyInput(
            tweets=tweet_contexts,
            relationships=relationships,
            strategy=strategy_context,
            budget=budget_info,
        )

        # 7. Get the appropriate policy
        policy_name = inputs.get("policy", DEFAULT_POLICY)
        policies = {
            "engagement": EngagementPolicy,
            "ranking": RankingPolicy,
            "strategy": StrategyPolicy,
            "master": MasterPolicy,
        }
        policy_class = policies.get(policy_name, MasterPolicy)
        policy = policy_class()

        # 8. Run policy
        try:
            output = policy.propose(policy_input)
        except BudgetExhaustedError as e:
            # Budget exhausted - return empty output gracefully
            output = PolicyOutput(
                actions=[],
                rationale=f"Budget exhausted: {e}", 
                confidence=0.0,
                strategy_alignment_score=0.0,
                constraints_checked=["budget"],
            )

        # 9. Emit policy decision event
        policy_event = PolicyDecisionEvent.create(
            ts=DEFAULT_TRACE_TS,
            data={
                "policy": policy_name,
                "actions_count": len(output.actions),
                "confidence": output.confidence,
                "strategy_alignment_score": output.strategy_alignment_score,
                "rationale": output.rationale,
                "actions": [
                    {
                        "tweet_id": a.tweet_id,
                        "author_handle": a.author_handle,
                        "action_type": a.action_type.value
                        if isinstance(a.action_type, ActionType)
                        else str(a.action_type),
                        "priority": a.priority if isinstance(a.priority, int) else int(a.priority),
                        "score": a.score,
                        "rationale": a.rationale,
                    }
                    for a in output.actions
                ],
            },
        )
        trace_events.append(policy_event.model_dump())

        # 10. Check expectations and emit verification event
        expectations = scenario.get("expectations", {})
        verification_passed = True
        verification_message = "All assertions passed"

        # Check output assertions
        for assertion in expectations.get("output_assertions", []):
            path = assertion.get("path", "")
            expected = assertion.get("expected")

            if path.startswith("actions["):
                # Parse array index
                import re

                match = re.match(r"actions\[(\d+)\]\.(.+)", path)
                if match:
                    idx = int(match.group(1))
                    field = match.group(2)

                    if idx < len(output.actions):
                        action = output.actions[idx]
                        actual = getattr(action, field, None)

                        if expected is not None and actual != expected:
                            # Check for 'in' constraint
                            if "in" in assertion and actual not in assertion["in"]:
                                verification_passed = False
                                verification_message = (
                                    f"Assertion failed: {path} = {actual}, "
                                    f"expected in {assertion['in']}"
                                )
                            elif "in" not in assertion:
                                verification_passed = False
                                verification_message = (
                                    f"Assertion failed: {path} = {actual}, expected {expected}"
                                )

        verification = VerificationEvent.create(
            ts=DEFAULT_TRACE_TS,
            data={"pass": verification_passed, "message": verification_message},
        )
        trace_events.append(verification.model_dump())

        # 11. Build final output
        final_output = output.model_dump_json()

        # 12. Build artifacts
        artifacts = {
            "policy_output": output.model_dump(),
            "policy_name": policy_name,
            "tweets_evaluated": len(tweet_contexts),
            "actions_proposed": len(output.actions),
        }

        return final_output, trace_events, artifacts
