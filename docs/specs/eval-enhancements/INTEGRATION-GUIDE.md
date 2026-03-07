# Integration Guide: Eval Enhancements

**Target Audience**: Teams integrating ash-hawk eval enhancements into their repos

## Quick Start

### 1. Install Dependencies

```bash
# Add to pyproject.toml
[project.dependencies]
ash-hawk = { path = "../ash-hawk", editable = true }

# Or from git
ash-hawk = { git = "https://github.com/org/ash-hawk.git", branch = "main" }
```

### 2. Use Judge Normalizer

Replace manual score extraction with the normalizer:

```python
# Before
def grade(self, response: str) -> dict:
    data = json.loads(response)
    score = data.get("score", 0.0)
    passed = data.get("passed", score >= 0.7)
    return {"score": score, "passed": passed}

# After
from ash_hawk.graders.judge_normalizer import normalize_judge_output

def grade(self, response: str) -> dict:
    result = normalize_judge_output(response, pass_threshold=0.7)
    return {
        "score": result.score,
        "passed": result.passed,
        "reasoning": result.reasoning,
        "issues": result.issues,
        "strengths": result.strengths,
    }
```

### 3. Enforce Rubric-Based Evaluation

```python
from ash_hawk.graders.rubric_guard import enforce_rubric_based_evaluation

# In your suite validation
def validate_suite(suite: EvalSuite) -> None:
    enforce_rubric_based_evaluation(suite)
    # Raises ValueError if any task lacks rubric-based LLM judge
```

### 4. Run Calibration

```python
from ash_hawk.calibration import CalibrationRunner, GroundTruth

# Load your ground truth
ground_truths = [
    GroundTruth(task_id=t["id"], expected_passed=t["expected_pass"])
    for t in load_test_cases()
]

# Add your eval results
runner = CalibrationRunner()
for task_id, result in eval_results.items():
    runner.add_result(
        task_id=task_id,
        predicted_score=result["score"],
        predicted_passed=result["passed"],
        actual_passed=next(gt.expected_passed for gt in ground_truths if gt.task_id == task_id),
    )

# Compute calibration
calibration = runner.compute()
print(f"ECE: {calibration.ece:.3f}")
print(f"Brier Score: {calibration.brier_score:.3f}")
print(f"Well Calibrated: {calibration.is_well_calibrated}")
```

### 5. Generate Gap Scorecard

```python
from ash_hawk.reporting import GapScorecardGenerator

# Analyze your suite
generator = GapScorecardGenerator()
scorecard = generator.analyze_suite(your_suite)

# Output
print(f"Overall Coverage: {scorecard.overall_score:.1%}")
print(f"Requirements Covered: {scorecard.covered_requirements}/{scorecard.total_requirements}")

# Get recommendations
for rec in scorecard.blueprint_recommendations:
    print(f"  - {rec}")

# Save for CI
with open("gap-scorecard.json", "w") as f:
    json.dump(scorecard.to_dict(), f, indent=2)
```

## CI Integration

### GitHub Actions

```yaml
# .github/workflows/eval-coverage.yml
name: Eval Coverage Check

on:
  push:
    paths:
      - 'evals/**'

jobs:
  check-coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install dependencies
        run: uv sync
        
      - name: Generate gap scorecard
        run: |
          uv run ash-hawk gap-scorecard ./evals/suites/main.yaml \
            --output ./reports/gap-scorecard.json
      
      - name: Check thresholds
        run: |
          uv run ash-hawk check-gap-thresholds ./reports/gap-scorecard.json \
            --threshold overall_score=0.70 \
            --threshold security_depth=0.50 \
            --baseline ./reports/baseline-scorecard.json \
            --output-format github
      
      - name: Upload scorecard
        uses: actions/upload-artifact@v4
        with:
          name: gap-scorecard
          path: reports/gap-scorecard.json
```

### GitLab CI

```yaml
# .gitlab-ci.yml
eval-coverage:
  script:
    - uv run ash-hawk gap-scorecard ./evals/suites/main.yaml --output scorecard.json
    - uv run ash-hawk check-gap-thresholds scorecard.json --threshold overall_score=0.70
  artifacts:
    paths:
      - scorecard.json
  rules:
    - changes:
        - evals/**/*
```

## Custom Requirements

Extend the default requirements for your domain:

```python
from ash_hawk.reporting.scorecard_types import Requirement
from ash_hawk.reporting import GapScorecardGenerator

custom_requirements = [
    # Default requirements
    *GapScorecardGenerator.DEFAULT_REQUIREMENTS,
    # Your custom requirements
    Requirement(
        req_id="CUSTOM-001",
        dimension="domain_specific",
        priority="high",
        description="Tests for your specific domain",
        agents=("your_agent",),
        keyword_any=("domain", "specific"),
        minimum_matches=2,
    ),
]

generator = GapScorecardGenerator(
    requirements=custom_requirements,
    agent_targets={"your_agent": 10},
)
```

## Custom Grader Presets

Create reusable grader configurations:

```python
from ash_hawk.graders.grader_presets import expand_preset, GraderPresetConfig

# Use preset
config = GraderPresetConfig(
    preset="security_review",
    pass_threshold=0.65,  # Override default
    expected_tools=["bash", "grep", "rg"],
)
grader_specs = expand_preset(config)

# Use in your suite
task = EvalTask(
    id="security-scan",
    grader_specs=grader_specs,
)
```

## Migration Checklist

- [ ] Add ash-hawk as dependency
- [ ] Replace manual JSON parsing with `normalize_judge_output()`
- [ ] Add `enforce_rubric_based_evaluation()` to suite validation
- [ ] Create ground truth labels for calibration
- [ ] Generate baseline gap scorecard
- [ ] Add CI workflow for coverage checks
- [ ] Set threshold values based on baseline
- [ ] Document custom requirements for your domain

## Support

- Spec docs: `docs/specs/eval-enhancements/`
- Tests: `tests/graders/`, `tests/calibration/`, `tests/reporting/`
- Examples: `examples/eval-enhancements/`
