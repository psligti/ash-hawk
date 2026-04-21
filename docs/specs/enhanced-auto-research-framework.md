---
title: Enhanced Auto-Research Framework
status: draft
owner: ash-hawk
last_updated: 2026-03-25
version: 0.1.0
---

# Enhanced Auto-Research Framework

> **Status note (2026-04-13):** This is a draft design document, not a description of the
> currently shipped implementation. Several module paths referenced here under
> `ash_hawk/auto_research/` do not exist in the live repo today. Use it as a future design
> spec only.

## Abstract

This specification defines an enhanced auto-research framework for Ash Hawk that enables **multi-target parallel improvement**, **lever matrix search**, **intent discovery**, and **knowledge promotion**. The framework transforms the current single-target, sequential improvement cycle into a comprehensive agent optimization system that can:

1. Improve multiple skills/tools simultaneously
2. Experiment with all available levers (prompts, context, tools, policy)
3. Discover agent intent from behavioral patterns
4. Promote validated learnings to persistent knowledge bases

---

## 1. Problem Statement

### 1.1 Current Limitations

The existing auto-research infrastructure (`ash_hawk/auto_research/`) has several critical limitations:

| Limitation | Current Behavior | Impact |
|------------|------------------|--------|
| **Single-target** | Discovers ONE skill/tool per cycle | Cannot improve interconnected capabilities |
| **Limited levers** | Only modifies skill content | Cannot optimize tools, context strategy, prompts, policy |
| **No intent discovery** | No analysis of agent behavior | Cannot extract "what agent is trying to do" |
| **Sequential execution** | One iteration at a time | Slow exploration of search space |
| **Convergence disabled** | `_check_convergence()` returns `False` | Wastes iterations after convergence |
| **No knowledge promotion** | Lessons stay in experiment scope | Learnings lost between sessions |
| **Genetic optimizer separate** | Not integrated with auto-research | Cannot combine population + hill-climbing |

### 1.2 User Requirements

Users need an auto-research system that can:

1. **Fine-tune agents** through process, prompt, and context engineering
2. **Discover intent** from scenario behavior patterns
3. **Experiment with all levers** (skills, tools, prompts, context strategy)
4. **Build findings back** into the agent for future sessions
5. **Run across multiple targets** simultaneously

---

## 2. Goals

### 2.1 Primary Goals

1. **Multi-target improvement**: Run improvement cycles across multiple skills/tools in parallel
2. **Lever matrix search**: Explore combinations of agent, skills, tools, prompts, and context strategies
3. **Intent discovery**: Extract behavioral patterns and decision sequences from transcripts
4. **Knowledge promotion**: Automatically promote validated lessons to persistent knowledge bases (note-lark)
5. **Efficient search**: Use genetic optimization + hill-climbing for faster convergence

### 2.2 Non-Goals

- Model weight fine-tuning (RLHF, LoRA, etc.)
- Replacement of human oversight for production promotion
- Blind score maximization without understanding
- Treating all failures as agent problems (vs infrastructure/eval issues)

---

## 3. Architecture Overview

### 3.1 System Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ENHANCED AUTO-RESEARCH FRAMEWORK                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Target         │     │  Lever Matrix   │     │  Intent         │
│  Discovery      │────▶│  Search         │────▶│  Analyzer       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Multi-Target   │     │  Genetic +      │     │  Knowledge      │
│  Cycle Runner   │     │  Hill-Climbing  │     │  Promotion      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
                    ┌─────────────────┐
                    │  Convergence    │
                    │  Detection      │
                    └─────────────────┘
                                │
                                ▼
                    ┌─────────────────┐
                    │  Note-Lark      │
                    │  Integration    │
                    └─────────────────┘
```

### 3.2 Data Flow

```
Scenarios ──▶ Target Discovery ──▶ Multi-Target Cycle
                    │                      │
                    │                      ▼
                    │              Intent Analysis
                    │                      │
                    │                      ▼
                    │              Lever Matrix Search
                    │                      │
                    │                      ▼
                    │              Genetic Optimization
                    │                      │
                    │                      ▼
                    └────────────────▶ Convergence Check
                                           │
                                           ▼
                                    Knowledge Promotion
                                           │
                                           ▼
                                    Note-Lark Persistence
```

---

## 4. Component Specifications

### 4.1 Multi-Target Discovery

**Location**: `ash_hawk/auto_research/target_discovery.py`

**Purpose**: Discover all improvement targets (skills, tools, agents) from project structure.

**Interface**:

```python
@dataclass
class ImprovementTarget:
    target_type: TargetType  # AGENT, SKILL, TOOL
    name: str
    discovered_path: Path
    structured_path: Path
    injector: DawnKestrelInjector
    dependencies: list[str]  # Other targets this depends on
    priority: int  # Higher = improve first


class TargetDiscovery:
    def discover_all_targets(
        self,
        project_root: Path,
        scenarios: list[Path],
    ) -> list[ImprovementTarget]:
        """Discover all skills, tools, and agents that could be improved.

        Priority order:
        1. Agents (highest - affect everything downstream)
        2. Skills (medium - affect specific capabilities)
        3. Tools (lowest - focused scope)

        Dependencies tracked for ordered improvement.
        """
        ...

    def rank_targets_by_impact(
        self,
        targets: list[ImprovementTarget],
        scenario_results: list[ScenarioResult],
    ) -> list[ImprovementTarget]:
        """Rank targets by potential impact on scenario scores.

        Uses:
        - Frequency of target usage in failed scenarios
        - Correlation between target and failure patterns
        - Dependency graph analysis
        """
        ...
```

**Search Locations**:

```python
TARGET_SEARCH_DIRS = {
    TargetType.SKILL: [
        Path("dawn-kestrel/skills"),
        Path(".opencode/skills"),
        Path(".claude/skills"),
    ],
    TargetType.TOOL: [
        Path("dawn-kestrel/tools"),
        Path(".opencode/tools"),
        Path(".claude/tools"),
    ],
    TargetType.AGENT: [
        Path("dawn-kestrel/agents"),
        Path(".opencode/agents"),
    ],
}
```

---

### 4.2 Multi-Target Cycle Runner

**Location**: `ash_hawk/auto_research/multi_target_runner.py`

**Purpose**: Run improvement cycles for multiple targets in parallel.

**Interface**:

```python
@dataclass
class MultiTargetResult:
    agent_name: str
    target_results: dict[str, CycleResult]  # target_name -> result
    overall_improvement: float
    best_target: str
    converged: bool
    convergence_reason: str | None
    promoted_lessons: list[str]


class MultiTargetCycleRunner:
    def __init__(
        self,
        parallelism: int = 4,
        min_improvement_threshold: float = 0.02,
        convergence_window: int = 5,
        convergence_variance_threshold: float = 0.001,
    ): ...

    async def run_multi_target_cycle(
        self,
        scenarios: list[Path],
        targets: list[ImprovementTarget],
        iterations_per_target: int = 50,
        storage_path: Path | None = None,
        llm_client: Any = None,
        strategy_name: str | None = None,
    ) -> MultiTargetResult:
        """Run improvement cycles for multiple targets in parallel.

        Process:
        1. Rank targets by impact
        2. Run top N targets in parallel (N = parallelism)
        3. Aggregate results
        4. Detect overall convergence
        5. Promote validated lessons
        """
        ...

    async def _run_single_target_cycle(
        self,
        target: ImprovementTarget,
        scenarios: list[Path],
        iterations: int,
        shared_llm_client: Any,
    ) -> CycleResult:
        """Run improvement cycle for a single target.

        Uses existing cycle_runner.run_cycle() internally.
        """
        ...
```

**Concurrency Model**:

```python
# Parallel execution with semaphore
semaphore = asyncio.Semaphore(self.parallelism)

async def run_with_limit(target: ImprovementTarget) -> CycleResult:
    async with semaphore:
        return await self._run_single_target_cycle(target, ...)

results = await asyncio.gather(*[
    run_with_limit(target)
    for target in ranked_targets
])
```

---

### 4.3 Lever Matrix Search

**Location**: `ash_hawk/auto_research/lever_matrix.py`

**Purpose**: Define and search the space of all configurable levers.

**Lever Definitions**:

```python
@dataclass
class LeverDimension:
    name: str
    values: list[Any]
    weight: float  # Importance for optimization
    mutation_rate: float  # How often to change this lever


DEFAULT_LEVER_SPACE: dict[str, LeverDimension] = {
    "agent": LeverDimension(
        name="agent",
        values=["orchestrator", "master_orchestrator", "explore", "consult"],
        weight=0.3,
        mutation_rate=0.1,
    ),
    "skills": LeverDimension(
        name="skills",
        values=[
            [],
            ["council"],
            ["frontend-ui-ux"],
            ["playwright"],
            ["git-master"],
            ["council", "git-master"],
        ],
        weight=0.2,
        mutation_rate=0.3,
    ),
    "tools": LeverDimension(
        name="tools",
        values=[
            ["read", "edit", "write"],
            ["read", "edit", "write", "bash"],
            ["read", "edit", "write", "bash", "grep", "glob"],
            # ... more combinations
        ],
        weight=0.2,
        mutation_rate=0.2,
    ),
    "context_strategy": LeverDimension(
        name="context_strategy",
        values=["file-based", "dynamic", "composite"],
        weight=0.15,
        mutation_rate=0.15,
    ),
    "prompt_preset": LeverDimension(
        name="prompt_preset",
        values=["balanced", "delegation_heavy", "precision", "throughput"],
        weight=0.15,
        mutation_rate=0.25,
    ),
    "timeout_multiplier": LeverDimension(
        name="timeout_multiplier",
        values=[0.75, 1.0, 1.25, 1.5, 2.0],
        weight=0.0,  # Not optimized, just tested
        mutation_rate=0.1,
    ),
}
```

**Interface**:

```python
@dataclass
class LeverConfiguration:
    agent: str
    skills: tuple[str, ...]
    tools: tuple[str, ...]
    context_strategy: str
    prompt_preset: str
    timeout_multiplier: float

    def to_genome(self) -> Genome:
        """Convert to genetic optimizer genome format."""
        ...

    @classmethod
    def from_genome(cls, genome: Genome) -> "LeverConfiguration":
        """Convert from genetic optimizer genome format."""
        ...


class LeverMatrixSearch:
    def __init__(
        self,
        lever_space: dict[str, LeverDimension] | None = None,
        genetic_config: dict[str, Any] | None = None,
    ): ...

    def sample_random(self) -> LeverConfiguration:
        """Sample a random configuration from the lever space."""
        ...

    def sample_neighbors(
        self,
        config: LeverConfiguration,
        n: int = 5,
    ) -> list[LeverConfiguration]:
        """Sample n neighbors by mutating 1-2 levers."""
        ...

    def crossover(
        self,
        a: LeverConfiguration,
        b: LeverConfiguration,
    ) -> LeverConfiguration:
        """Combine two configurations via crossover."""
        ...

    async def evaluate(
        self,
        config: LeverConfiguration,
        scenarios: list[Path],
        storage_path: Path,
    ) -> float:
        """Evaluate a configuration against scenarios.

        Returns fitness score (0.0 - 1.0).
        """
        ...
```

---

### 4.4 Intent Discovery

**Location**: `ash_hawk/auto_research/intent_analyzer.py`

**Purpose**: Extract behavioral patterns and intent from agent transcripts.

**Interface**:

```python
@dataclass
class ToolUsagePattern:
    tool_name: str
    call_count: int
    success_rate: float
    avg_duration_seconds: float
    common_sequences: list[tuple[str, ...]]  # e.g., [("read", "grep", "edit")]


@dataclass
class DecisionPattern:
    pattern_type: str  # "parallel_explore", "sequential_fix", "delegate_then_synthesize"
    frequency: int
    success_rate: float
    example_sequences: list[list[str]]


@dataclass
class FailurePattern:
    failure_type: str  # "timeout", "tool_not_allowed", "infinite_loop", "wrong_output"
    frequency: int
    affected_tools: list[str]
    recovery_attempts: int
    recovery_success_rate: float


@dataclass
class IntentPatterns:
    dominant_tools: list[str]
    tool_usage_patterns: list[ToolUsagePattern]
    decision_patterns: list[DecisionPattern]
    failure_patterns: list[FailurePattern]
    inferred_intent: str  # Natural language description of what agent is trying to do
    confidence: float


class IntentAnalyzer:
    def __init__(self, llm_client: Any | None = None): ...

    async def analyze_transcripts(
        self,
        transcripts: list[EvalTranscript],
    ) -> IntentPatterns:
        """Extract intent patterns from agent transcripts.

        Process:
        1. Extract tool usage statistics
        2. Identify tool sequences
        3. Cluster decision patterns
        4. Classify failure modes
        5. Generate intent hypothesis via LLM
        """
        ...

    def _extract_tool_patterns(
        self,
        transcripts: list[EvalTranscript],
    ) -> list[ToolUsagePattern]:
        """Extract tool usage patterns from transcripts."""
        ...

    def _identify_sequences(
        self,
        transcripts: list[EvalTranscript],
        min_length: int = 2,
        max_length: int = 5,
    ) -> dict[tuple[str, ...], int]:
        """Identify common tool sequences and their frequencies."""
        ...

    def _cluster_decision_patterns(
        self,
        transcripts: list[EvalTranscript],
    ) -> list[DecisionPattern]:
        """Cluster decision patterns using sequence analysis."""
        ...

    def _classify_failures(
        self,
        transcripts: list[EvalTranscript],
    ) -> list[FailurePattern]:
        """Classify failure patterns from error traces."""
        ...

    async def _generate_intent_hypothesis(
        self,
        patterns: IntentPatterns,
    ) -> str:
        """Use LLM to generate natural language intent description."""
        ...
```

**Intent Hypothesis Prompt**:

```python
INTENT_HYPOTHESIS_PROMPT = """
Analyze the following agent behavior patterns and infer the agent's intent.

## Tool Usage
{tool_usage}

## Decision Patterns
{decision_patterns}

## Failure Patterns
{failure_patterns}

## Task
Describe in 2-3 sentences what this agent appears to be trying to accomplish.
Focus on:
1. The primary goal the agent is pursuing
2. The strategy/approach the agent prefers
3. What's preventing the agent from succeeding

Be specific and avoid generic descriptions.
"""
```

---

### 4.5 Convergence Detection

**Location**: `ash_hawk/auto_research/convergence.py`

**Purpose**: Detect when improvement has converged.

**Interface**:

```python
@dataclass
class ConvergenceResult:
    converged: bool
    reason: str
    iterations_since_improvement: int
    score_variance: float
    confidence: float


class ConvergenceDetector:
    def __init__(
        self,
        window_size: int = 5,
        variance_threshold: float = 0.001,
        min_improvement: float = 0.005,
        max_iterations_without_improvement: int = 10,
    ): ...

    def check(
        self,
        iterations: list[IterationResult],
    ) -> ConvergenceResult:
        """Check if improvement has converged.

        Convergence criteria:
        1. Score variance < variance_threshold for window_size iterations
        2. No improvement > min_improvement for max_iterations_without_improvement
        3. Score has decreased for 3+ consecutive iterations (regression)

        Returns convergence status with reason and confidence.
        """
        ...

    def _compute_variance(
        self,
        scores: list[float],
    ) -> float:
        """Compute variance of recent scores."""
        ...

    def _iterations_since_improvement(
        self,
        iterations: list[IterationResult],
        threshold: float,
    ) -> int:
        """Count iterations since last improvement above threshold."""
        ...
```

**Convergence Criteria**:

```python
# 1. Plateau detection: variance < threshold for N iterations
if score_variance < self.variance_threshold:
    return ConvergenceResult(
        converged=True,
        reason="Score plateau detected",
        confidence=1.0 - score_variance / self.variance_threshold,
        ...
    )

# 2. No improvement for N iterations
if iterations_since_improvement >= self.max_iterations_without_improvement:
    return ConvergenceResult(
        converged=True,
        reason=f"No improvement for {iterations_since_improvement} iterations",
        confidence=0.8,
        ...
    )

# 3. Regression detection: 3+ consecutive decreases
if self._detect_regression(iterations):
    return ConvergenceResult(
        converged=True,
        reason="Score regression detected - stopping to prevent overfitting",
        confidence=0.9,
        ...
    )
```

---

### 4.6 Knowledge Promotion

**Location**: `ash_hawk/auto_research/knowledge_promotion.py`

**Purpose**: Promote validated lessons to persistent knowledge bases.

**Interface**:

```python
@dataclass
class PromotionCriteria:
    min_improvement: float = 0.05  # 5% improvement required
    min_consecutive_successes: int = 3  # Must succeed 3 times in a row
    max_regression: float = 0.02  # No more than 2% regression on other scenarios
    require_stability: bool = True  # Must not regress on re-run


@dataclass
class PromotedLesson:
    lesson_id: str
    source_experiment: str
    target_name: str
    target_type: TargetType
    improvement_text: str
    score_delta: float
    promoted_at: datetime
    promotion_confidence: float


class KnowledgePromoter:
    def __init__(
        self,
        criteria: PromotionCriteria | None = None,
        note_lark_enabled: bool = True,
    ): ...

    async def should_promote(
        self,
        iteration: IterationResult,
        all_iterations: list[IterationResult],
        scenario_results: list[ScenarioResult],
    ) -> tuple[bool, str]:
        """Determine if an improvement should be promoted.

        Returns (should_promote, reason).
        """
        ...

    async def promote_lesson(
        self,
        lesson: PromotedLesson,
        targets: list[str] | None = None,  # None = global
    ) -> bool:
        """Promote a lesson to persistent storage.

        1. Write to local .ash-hawk/lessons/
        2. If note_lark_enabled, also write to note-lark knowledge base
        """
        ...

    async def promote_to_note_lark(
        self,
        lesson: PromotedLesson,
        intent_patterns: IntentPatterns | None = None,
    ) -> str | None:
        """Promote lesson to note-lark knowledge base.

        Returns note_id if successful.
        """
        ...
```

**Note-Lark Integration**:

```python
async def promote_to_note_lark(
    self,
    lesson: PromotedLesson,
    intent_patterns: IntentPatterns | None = None,
) -> str | None:
    """Promote lesson to note-lark knowledge base."""

    # Build tags from lesson metadata and intent patterns
    tags = ["auto-research", "improvement", lesson.target_type.value]
    if intent_patterns:
        tags.extend(intent_patterns.dominant_tools[:3])

    # Determine memory type based on target type
    memory_type = {
        TargetType.AGENT: "procedural",
        TargetType.SKILL: "procedural",
        TargetType.TOOL: "reference",
    }[lesson.target_type]

    # Calculate confidence based on score delta and stability
    confidence = min(0.95, lesson.score_delta / 0.2)

    # Call note-lark memory_structured
    result = await note_lark_memory_structured(payload={
        "title": f"Auto-discovered: {lesson.improvement_text[:80]}",
        "memory_type": memory_type,
        "scope": "project",
        "project": "bolt-merlin",  # TODO: Make configurable
        "status": "validated",
        "confidence": confidence,
        "evidence_count": 1,
        "tags": tags,
        "body": f"""# Improvement

{lesson.improvement_text}

## Impact

- Score improvement: +{lesson.score_delta:.3f}
- Target: {lesson.target_name} ({lesson.target_type.value})
- Experiment: {lesson.source_experiment}

## Context

{self._format_intent_context(intent_patterns) if intent_patterns else "No intent analysis available."}
""",
    })

    return result.get("note_id")
```

---

### 4.7 Genetic + Hill-Climbing Hybrid

**Location**: `ash_hawk/auto_research/hybrid_optimizer.py`

**Purpose**: Combine genetic optimization with hill-climbing for efficient search.

**Interface**:

```python
@dataclass
class HybridConfig:
    # Population settings
    population_size: int = 8
    elite_count: int = 2

    # Genetic operators
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7

    # Hill-climbing settings
    hill_climb_iterations: int = 5
    hill_climb_neighbors: int = 3

    # Convergence
    max_generations: int = 10
    convergence_patience: int = 3


class HybridOptimizer:
    def __init__(
        self,
        lever_space: dict[str, LeverDimension],
        config: HybridConfig | None = None,
    ): ...

    async def optimize(
        self,
        scenarios: list[Path],
        storage_path: Path,
        initial_population: list[LeverConfiguration] | None = None,
    ) -> OptimizationResult:
        """Run hybrid genetic + hill-climbing optimization.

        Process:
        1. Initialize population (random or provided)
        2. For each generation:
           a. Evaluate all configurations
           b. Select elites
           c. Generate offspring via crossover
           d. Apply mutations
           e. Hill-climb top performers
           f. Check convergence
        3. Return best configuration and history
        """
        ...

    async def _evaluate_population(
        self,
        population: list[LeverConfiguration],
        scenarios: list[Path],
        storage_path: Path,
    ) -> list[tuple[LeverConfiguration, float]]:
        """Evaluate all configurations in parallel."""
        ...

    def _select_elites(
        self,
        evaluated: list[tuple[LeverConfiguration, float]],
    ) -> list[LeverConfiguration]:
        """Select top performers as elites."""
        ...

    def _crossover(
        self,
        population: list[LeverConfiguration],
    ) -> list[LeverConfiguration]:
        """Generate offspring via crossover."""
        ...

    def _mutate(
        self,
        population: list[LeverConfiguration],
    ) -> list[LeverConfiguration]:
        """Apply mutations to population."""
        ...

    async def _hill_climb(
        self,
        config: LeverConfiguration,
        scenarios: list[Path],
        storage_path: Path,
    ) -> tuple[LeverConfiguration, float]:
        """Hill-climb from a configuration to find local optimum."""
        ...
```

---

## 5. Integration Specification

### 5.1 Enhanced Cycle Runner

**Location**: `ash_hawk/auto_research/enhanced_cycle_runner.py`

**Purpose**: Main entry point for enhanced auto-research.

**Interface**:

```python
@dataclass
class EnhancedCycleConfig:
    # Multi-target settings
    enable_multi_target: bool = True
    max_parallel_targets: int = 4

    # Lever search settings
    enable_lever_search: bool = True
    lever_space: dict[str, LeverDimension] | None = None

    # Intent discovery settings
    enable_intent_analysis: bool = True

    # Knowledge promotion settings
    enable_knowledge_promotion: bool = True
    promotion_criteria: PromotionCriteria | None = None
    note_lark_enabled: bool = True

    # Convergence settings
    convergence_window: int = 5
    convergence_variance_threshold: float = 0.001
    max_iterations_without_improvement: int = 10

    # Hybrid optimization settings
    enable_hybrid_optimization: bool = True
    hybrid_config: HybridConfig | None = None


@dataclass
class EnhancedCycleResult:
    agent_name: str
    target_results: dict[str, CycleResult]
    lever_result: OptimizationResult | None
    intent_patterns: IntentPatterns | None
    promoted_lessons: list[PromotedLesson]
    overall_improvement: float
    converged: bool
    convergence_reason: str | None
    total_iterations: int
    total_duration_seconds: float


async def run_enhanced_cycle(
    scenarios: list[Path],
    config: EnhancedCycleConfig | None = None,
    storage_path: Path | None = None,
    llm_client: Any = None,
    project_root: Path | None = None,
) -> EnhancedCycleResult:
    """Run enhanced auto-research cycle.

    Process:
    1. Discover all improvement targets
    2. Run intent analysis on baseline transcripts
    3. Execute multi-target improvement cycles in parallel
    4. Run lever matrix search (if enabled)
    5. Run hybrid optimization (if enabled)
    6. Check convergence
    7. Promote validated lessons to knowledge base
    8. Return aggregated results
    """
    ...
```

### 5.2 CLI Integration

**Location**: `ash_hawk/auto_research/enhanced_cli.py`

**Interface**:

```python
@cli.command("enhanced-run")
@click.option("-s", "--scenario", multiple=True, type=click.Path())
@click.option("-t", "--target", multiple=True, type=click.Path())
@click.option("-i", "--iterations", default=100)
@click.option("--threshold", default=0.02)
@click.option("--parallel/--sequential", default=True)
@click.option("--multi-target/--single-target", default=True)
@click.option("--lever-search/--no-lever-search", default=True)
@click.option("--intent-analysis/--no-intent-analysis", default=True)
@click.option("--knowledge-promotion/--no-knowledge-promotion", default=True)
@click.option("--note-lark/--no-note-lark", default=True)
@click.option("--hybrid-optimization/--no-hybrid-optimization", default=True)
def enhanced_run(
    scenario: tuple[str, ...],
    target: tuple[str, ...],
    iterations: int,
    threshold: float,
    parallel: bool,
    multi_target: bool,
    lever_search: bool,
    intent_analysis: bool,
    knowledge_promotion: bool,
    note_lark: bool,
    hybrid_optimization: bool,
) -> None:
    """Run enhanced auto-research improvement cycle.

    Examples:
        # Basic enhanced run
        ash-hawk auto-research enhanced-run -s evals/scenarios/*.yaml

        # Multi-target with lever search
        ash-hawk auto-research enhanced-run -s evals/scenarios/*.yaml --multi-target --lever-search

        # Full featured with knowledge promotion
        ash-hawk auto-research enhanced-run -s evals/scenarios/*.yaml \\
            --multi-target --lever-search --intent-analysis \\
            --knowledge-promotion --note-lark --hybrid-optimization
    """
    ...
```

---

## 6. Data Models

### 6.1 Core Types

```python
# ash_hawk/auto_research/types.py (extended)

from enum import StrEnum
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class TargetType(StrEnum):
    AGENT = "agent"
    SKILL = "skill"
    TOOL = "tool"


class CycleStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CONVERGED = "converged"
    ERROR = "error"


@dataclass
class IterationResult:
    iteration_num: int
    score_before: float
    score_after: float
    improvement_text: str | None = None
    applied: bool = False
    delta: float = 0.0
    artifact_path: Path | None = None
    timestamp: datetime | None = None


@dataclass
class CycleResult:
    agent_name: str
    target_path: str
    target_type: TargetType
    scenario_paths: list[str]
    status: CycleStatus
    initial_score: float = 0.0
    final_score: float = 0.0
    iterations: list[IterationResult] = field(default_factory=list)
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def total_iterations(self) -> int:
        return len(self.iterations)

    @property
    def improvement_delta(self) -> float:
        return self.final_score - self.initial_score

    @property
    def applied_iterations(self) -> list[IterationResult]:
        return [i for i in self.iterations if i.applied]


@dataclass
class MultiTargetResult:
    agent_name: str
    target_results: dict[str, CycleResult]
    overall_improvement: float
    best_target: str
    converged: bool
    convergence_reason: str | None
    promoted_lessons: list[str]
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class OptimizationResult:
    best_configuration: LeverConfiguration
    best_fitness: float
    best_metrics: dict[str, float]
    history: list[dict[str, Any]]
    generations: int
    converged: bool
    convergence_reason: str | None
```

---

## 7. Configuration

### 7.1 Environment Variables

```bash
# Multi-target settings
ASH_HAWK_MULTI_TARGET_ENABLED=true
ASH_HAWK_MAX_PARALLEL_TARGETS=4

# Lever search settings
ASH_HAWK_LEVER_SEARCH_ENABLED=true
ASH_HAWK_LEVER_SEARCH_SPACE=default  # default, minimal, extended

# Intent analysis settings
ASH_HAWK_INTENT_ANALYSIS_ENABLED=true

# Knowledge promotion settings
ASH_HAWK_KNOWLEDGE_PROMOTION_ENABLED=true
ASH_HAWK_NOTE_LARK_ENABLED=true
ASH_HAWK_PROMOTION_MIN_IMPROVEMENT=0.05
ASH_HAWK_PROMOTION_MIN_SUCCESSES=3

# Convergence settings
ASH_HAWK_CONVERGENCE_WINDOW=5
ASH_HAWK_CONVERGENCE_VARIANCE_THRESHOLD=0.001
ASH_HAWK_MAX_ITERATIONS_WITHOUT_IMPROVEMENT=10

# Hybrid optimization settings
ASH_HAWK_HYBRID_OPTIMIZATION_ENABLED=true
ASH_HAWK_POPULATION_SIZE=8
ASH_HAWK_ELITE_COUNT=2
ASH_HAWK_MUTATION_RATE=0.2
ASH_HAWK_MAX_GENERATIONS=10
```

### 7.2 Configuration File

```yaml
# .ash-hawk/auto-research.yaml

multi_target:
  enabled: true
  max_parallel_targets: 4

lever_search:
  enabled: true
  space: default  # default, minimal, extended
  custom_dimensions:
    - name: custom_prompt_variant
      values: ["variant_a", "variant_b", "variant_c"]
      weight: 0.1
      mutation_rate: 0.2

intent_analysis:
  enabled: true
  llm_temperature: 0.3

knowledge_promotion:
  enabled: true
  note_lark_enabled: true
  criteria:
    min_improvement: 0.05
    min_consecutive_successes: 3
    max_regression: 0.02
    require_stability: true

convergence:
  window: 5
  variance_threshold: 0.001
  max_iterations_without_improvement: 10

hybrid_optimization:
  enabled: true
  population_size: 8
  elite_count: 2
  mutation_rate: 0.2
  crossover_rate: 0.7
  hill_climb_iterations: 5
  hill_climb_neighbors: 3
  max_generations: 10
  convergence_patience: 3
```

---

## 8. Implementation Phases

### Phase 1: Multi-Target Parallel Improvement (Week 1-2)

**Deliverables**:
1. `target_discovery.py` - Target discovery and ranking
2. `multi_target_runner.py` - Parallel cycle execution
3. `enhanced_cycle_runner.py` - Main entry point
4. Updated CLI commands

**Success Criteria**:
- Can discover all skills/tools/agents in a project
- Can run improvement cycles for multiple targets in parallel
- Results aggregated with overall improvement tracking

### Phase 2: Lever Matrix Search (Week 3-4)

**Deliverables**:
1. `lever_matrix.py` - Lever space definition and search
2. Integration with existing `genetic_optimizer.py`
3. Scenario configuration generation from lever combinations

**Success Criteria**:
- Can sample configurations from lever space
- Can evaluate configurations against scenarios
- Can find optimal lever combinations

### Phase 3: Intent Discovery (Week 5)

**Deliverables**:
1. `intent_analyzer.py` - Pattern extraction and intent inference
2. Integration with cycle results for context

**Success Criteria**:
- Can extract tool usage patterns from transcripts
- Can identify decision patterns
- Can generate intent hypothesis

### Phase 4: Convergence Detection (Week 5)

**Deliverables**:
1. `convergence.py` - Convergence detection
2. Integration with all cycle runners

**Success Criteria**:
- Detects score plateaus
- Detects lack of improvement
- Detects regression

### Phase 5: Knowledge Promotion (Week 6)

**Deliverables**:
1. `knowledge_promotion.py` - Promotion logic
2. Note-lark integration
3. Promotion criteria configuration

**Success Criteria**:
- Can determine if lesson should be promoted
- Can write to local lesson store
- Can write to note-lark knowledge base

### Phase 6: Hybrid Optimization (Week 7-8)

**Deliverables**:
1. `hybrid_optimizer.py` - Genetic + hill-climbing
2. Full integration with all components

**Success Criteria**:
- Combines population search with local optimization
- Outperforms either approach alone
- Converges to better solutions faster

---

## 9. Testing Strategy

### 9.1 Unit Tests

```python
# tests/auto_research/test_target_discovery.py
def test_discover_skills():
    """Test skill discovery from project structure."""

def test_discover_tools():
    """Test tool discovery from project structure."""

def test_rank_targets_by_impact():
    """Test target ranking based on scenario results."""


# tests/auto_research/test_multi_target_runner.py
@pytest.mark.asyncio
async def test_parallel_cycle_execution():
    """Test running multiple cycles in parallel."""

@pytest.mark.asyncio
async def test_semaphore_limiting():
    """Test that parallelism is limited correctly."""


# tests/auto_research/test_lever_matrix.py
def test_sample_random():
    """Test random sampling from lever space."""

def test_sample_neighbors():
    """Test neighbor sampling via mutation."""

def test_crossover():
    """Test configuration crossover."""


# tests/auto_research/test_intent_analyzer.py
def test_extract_tool_patterns():
    """Test tool pattern extraction."""

def test_identify_sequences():
    """Test sequence identification."""

@pytest.mark.asyncio
async def test_generate_intent_hypothesis():
    """Test LLM-based intent generation."""


# tests/auto_research/test_convergence.py
def test_plateau_detection():
    """Test score plateau detection."""

def test_no_improvement_detection():
    """Test detection of no improvement."""

def test_regression_detection():
    """Test regression detection."""


# tests/auto_research/test_knowledge_promotion.py
@pytest.mark.asyncio
async def test_should_promote_criteria():
    """Test promotion criteria evaluation."""

@pytest.mark.asyncio
async def test_promote_to_note_lark():
    """Test note-lark integration."""
```

### 9.2 Integration Tests

```python
# tests/integration/test_enhanced_cycle.py
@pytest.mark.asyncio
async def test_full_enhanced_cycle():
    """Test full enhanced cycle with all components."""

@pytest.mark.asyncio
async def test_enhanced_cycle_with_mock_adapter():
    """Test enhanced cycle using mock adapter."""
```

### 9.3 E2E Tests

```python
# tests/e2e/test_enhanced_auto_research.py
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_enhanced_cycle_bolt_merlin():
    """Test enhanced cycle on bolt-merlin scenarios."""

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_knowledge_promotion_e2e():
    """Test knowledge promotion to note-lark."""
```

---

## 10. Migration Plan

### 10.1 Backward Compatibility

The enhanced framework must maintain backward compatibility with existing auto-research:

```python
# Existing CLI continues to work
ash-hawk auto-research run -s scenarios/*.yaml

# New enhanced CLI
ash-hawk auto-research enhanced-run -s scenarios/*.yaml
```

### 10.2 Feature Flags

All new features are opt-in via configuration:

```yaml
# Minimal config (backward compatible)
multi_target:
  enabled: false
lever_search:
  enabled: false
intent_analysis:
  enabled: false
knowledge_promotion:
  enabled: false
hybrid_optimization:
  enabled: false
```

### 10.3 Gradual Rollout

1. **Week 1-2**: Deploy multi-target as opt-in
2. **Week 3-4**: Deploy lever search as opt-in
3. **Week 5**: Deploy intent analysis + convergence as opt-in
4. **Week 6**: Deploy knowledge promotion as opt-in
5. **Week 7-8**: Deploy hybrid optimization as opt-in
6. **Week 9+**: Consider making enhanced mode the default

---

## 11. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Parallel execution increases API costs | High | Configurable parallelism, cost budgets |
| Genetic optimization explores too many variants | Medium | Population size limits, early stopping |
| Knowledge promotion promotes bad lessons | High | Strict promotion criteria, human review option |
| Intent analysis inaccurate | Medium | Confidence thresholds, fallback to heuristics |
| Convergence detection too aggressive | Medium | Configurable thresholds, patience parameters |
| Note-lark integration failures | Low | Local storage fallback, retry logic |

---

## 12. Success Metrics

### 12.1 Performance Metrics

- **Improvement rate**: % of runs that show improvement > 0
- **Average improvement**: Mean score delta across runs
- **Convergence time**: Iterations until convergence
- **Cost efficiency**: Improvement per dollar spent

### 12.2 Quality Metrics

- **Lesson quality**: % of promoted lessons that remain valid after 30 days
- **Intent accuracy**: Correlation between inferred intent and actual agent goals
- **Lever effectiveness**: Which levers have highest impact

### 12.3 Operational Metrics

- **Success rate**: % of runs that complete without error
- **Promotion rate**: % of improvements that get promoted
- **Revert rate**: % of improvements that get reverted

---

## 13. Open Questions

1. **Should we support cross-project knowledge promotion?**
   - Currently scoped to single project
   - Could enable sharing lessons across projects

2. **How to handle conflicting lessons from different targets?**
   - Current conflict resolver is basic
   - May need more sophisticated merge strategies

3. **Should intent analysis be cached?**
   - Could speed up repeated runs
   - But may miss behavioral changes

4. **What's the right balance between exploration and exploitation?**
   - Genetic optimization favors exploration
   - Hill-climbing favors exploitation
   - Need to tune the balance

---

## 14. References

- [Ash Hawk Auto-Research Pipeline Spec](../spec.md)
- [Bolt Merlin Genetic Optimizer](../../bolt-merlin/evals/scripts/genetic_optimizer.py)
- [Note-Lark MCP Integration](/.opencode/skills/note-lark-memory/SKILL.md)
- [Auto-Research Cycle Skill](/.opencode/skills/auto-research-cycle/SKILL.md)

---

## Appendix A: Existing Infrastructure

### A.1 Current Auto-Research Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `cycle_runner.py` | `ash_hawk/auto_research/` | Main improvement loop |
| `llm.py` | `ash_hawk/auto_research/` | LLM-based improvement generation |
| `types.py` | `ash_hawk/auto_research/` | Core data types |
| `cli.py` | `ash_hawk/auto_research/` | CLI commands |
| `prompt_stack_optimizer.py` | `ash_hawk/graders/` | PSO grader |
| `genetic_optimizer.py` | `bolt-merlin/evals/scripts/` | Genetic optimization |
| `experiment_store.py` | `ash_hawk/curation/` | Lesson storage |
| `lesson_injector.py` | `ash_hawk/services/` | Runtime lesson injection |

### A.2 Key Gaps in Current Infrastructure

| Gap | Current State | Needed State |
|-----|---------------|--------------|
| Target discovery | Single target | Multi-target with dependencies |
| Search space | Skill content only | All levers (agent, tools, context, prompts) |
| Intent understanding | None | Pattern extraction + LLM inference |
| Convergence | Disabled | Plateau + no-improvement + regression |
| Knowledge persistence | Local only | Local + note-lark |
| Optimization | Hill-climbing only | Genetic + hill-climbing hybrid |

---

## Appendix B: Example Usage

### B.1 Basic Enhanced Run

```bash
# Run enhanced auto-research on bolt-merlin scenarios
ash-hawk auto-research enhanced-run \
  -s ../bolt-merlin/evals/scenarios/*.yaml \
  --multi-target \
  --intent-analysis \
  --knowledge-promotion \
  --note-lark
```

### B.2 Full Featured Run

```bash
# Run with all features enabled
ash-hawk auto-research enhanced-run \
  -s ../bolt-merlin/evals/scenarios/*.yaml \
  -i 100 \
  --threshold 0.02 \
  --multi-target \
  --lever-search \
  --intent-analysis \
  --knowledge-promotion \
  --note-lark \
  --hybrid-optimization
```

### B.3 Configuration-Driven Run

```bash
# Run with configuration file
ash-hawk auto-research enhanced-run \
  -s ../bolt-merlin/evals/scenarios/*.yaml \
  --config .ash-hawk/auto-research.yaml
```

### B.4 Python API Usage

```python
from ash_hawk.auto_research import run_enhanced_cycle
from ash_hawk.auto_research.enhanced_cycle_runner import EnhancedCycleConfig
from pathlib import Path

config = EnhancedCycleConfig(
    enable_multi_target=True,
    enable_lever_search=True,
    enable_intent_analysis=True,
    enable_knowledge_promotion=True,
    note_lark_enabled=True,
    enable_hybrid_optimization=True,
)

result = await run_enhanced_cycle(
    scenarios=[Path("../bolt-merlin/evals/scenarios/mvp_01.yaml")],
    config=config,
    project_root=Path("../bolt-merlin"),
)

print(f"Overall improvement: {result.overall_improvement:.3f}")
print(f"Promoted lessons: {len(result.promoted_lessons)}")
print(f"Converged: {result.converged}")
```
