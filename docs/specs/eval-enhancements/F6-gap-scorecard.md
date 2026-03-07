# F6: Gap Scorecard System

**Status**: IMPLEMENTED  
**Priority**: P2 (High Value)  
**Source**: iron-rook `gap_scorecard.py` (851 lines)

## Problem Statement

Eval suites grow organically without visibility into:
- Which capability areas are under-tested
- Whether coverage is improving over time
- What tests to prioritize adding next

## Solution

A requirements-driven scorecard system that tracks:
1. Coverage by dimension (security, architecture, etc.)
2. Agent depth (tasks per agent type)
3. Priority-weighted scoring
4. Baseline comparison for regression detection

## API

```python
from ash_hawk.reporting import GapScorecardGenerator, GapScorecard

# Generate scorecard
generator = GapScorecardGenerator()
scorecard: GapScorecard = generator.analyze_suite(eval_suite)

# Key metrics
scorecard.overall_score       # Priority-weighted coverage (0.0-1.0)
scorecard.dimension_scores    # Per-dimension coverage
scorecard.agent_depth         # Tasks per agent vs targets
scorecard.requirement_coverage  # Which requirements are covered
scorecard.blueprint_recommendations  # What to add next

# Output formats
markdown = GapScorecardGenerator.to_markdown(scorecard)
json_dict = GapScorecardGenerator.to_json(scorecard)

# Baseline comparison
baseline = load_scorecard("./baseline-scorecard.json")
diff = generator.compare_baseline(scorecard, baseline)
diff.overall_score_delta     # Change in overall score
dim.dimension_deltas         # Per-dimension changes
dim.new_covered_requirements  # Newly covered requirements
dim.regression_requirements   # Requirements that regressed
```

## Default Requirements

| ID | Dimension | Priority | Description |
|----|-----------|----------|-------------|
| SEC-001 | security_depth | critical | SQL injection detection |
| SEC-002 | security_depth | critical | XSS vulnerability detection |
| SEC-003 | security_depth | critical | Hardcoded secrets detection |
| ARCH-001 | architecture_depth | high | Boundary violation detection |
| ARCH-002 | architecture_depth | high | Circular dependency detection |
| DELEG-001 | delegation_quality | high | Subagent delegation tests |
| TOOL-001 | tool_behavior | medium | Tool call verification |
| PROMPT-001 | prompt_robustness | medium | Vague prompt handling |
| RUNTIME-001 | runtime_reliability | medium | Timeout handling |

## Default Agent Targets

| Agent | Target Tasks |
|-------|-------------|
| security | 10 |
| architecture | 8 |
| documentation | 6 |
| unit_tests | 6 |
| linting | 6 |
| performance | 6 |
| general | 8 |
| explore | 5 |

## CLI Usage

```bash
# Generate scorecard
uv run ash-hawk gap-scorecard ./evals/suites/main.yaml \
    --output ./reports/gap-scorecard.json

# Generate markdown report
uv run ash-hawk gap-scorecard ./evals/suites/main.yaml \
    --format markdown \
    --output ./reports/gap-scorecard.md

# Compare to baseline
uv run ash-hawk gap-scorecard ./evals/suites/main.yaml \
    --baseline ./reports/baseline-scorecard.json \
    --fail-on-regression
```

## CI Integration

```bash
# In CI pipeline
uv run ash-hawk check-gap-thresholds ./reports/gap-scorecard.json \
    --threshold overall_score=0.70 \
    --threshold security_depth=0.50 \
    --baseline ./reports/main-scorecard.json \
    --output-format github
```

## Files

- `ash_hawk/reporting/gap_scorecard.py` - Main generator
- `ash_hawk/reporting/scorecard_types.py` - Data types
- `ash_hawk/reporting/ci_thresholds.py` - CLI for CI
- `tests/reporting/test_gap_scorecard.py` - Tests

## Output Example

```markdown
# Gap Scorecard: security-suite

**Generated**: 2026-03-06T14:30:00Z
**Overall Score**: 75.00%
**Requirements**: 6/9 covered

## Dimension Scores

- ✅ **security_depth**: 85.00%
- ⚠️ **architecture_depth**: 60.00%
- ❌ **delegation_quality**: 40.00%

## Agent Depth

- ✅ **security**: 12/10 tasks
- ⚠️ **architecture**: 5/8 tasks
- ❌ **general**: 2/8 tasks

## Recommendations

- [HIGH] Add tests for: Subagent delegation tests
- [GAP] Add 3 more tasks for agent 'architecture'
- [GAP] Add 6 more tasks for agent 'general'
```

## Testing

```bash
uv run pytest tests/reporting/test_gap_scorecard.py -v
```
